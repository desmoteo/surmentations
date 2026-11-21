"""A road's own coordinates, its lanes, and the wheel tracks drawn along them.

``surmentations.utils.road`` turns a mask of where the road is into a
position across the road for each pixel of it. The masks here are drawn
from roads whose sides are known, so that what comes back can be checked
against the lines it should follow.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from surmentations.atmosphere import RandomRain, RandomSnow
from surmentations.depth.estimator import DepthEstimator
from surmentations.depth.provider import SharedDepthProvider
from surmentations.diagnostics import recording
from surmentations.perception import SceneClass, SceneLabelEstimator
from surmentations.utils.road import (
    GroundView,
    Lanes,
    RoadFrame,
    RoadVehicle,
    ground_view,
    painted_lines,
    read_road,
    road_frame,
    road_lanes,
    road_mist,
    road_vehicles,
    spaced_speeds,
    spray_amount,
    straighten,
    traffic_directions,
    vehicle_facing,
    vehicle_lamps,
    vehicle_spray,
    wake_lies,
    wheel_tracks,
)

HEIGHT, WIDTH = 240, 320
VANISHING = (160.0, 60.0)


def _fan(left_foot: float, right_foot: float, top: int = 72) -> np.ndarray:
    """A straight road running away from the camera: a mask between two lines through one point.

    The lines run from the vanishing point to ``left_foot`` and
    ``right_foot`` on the frame's bottom row, which may lie outside it.
    """
    rows, columns = np.mgrid[0:HEIGHT, 0:WIDTH]
    share = (rows - VANISHING[1]) / (HEIGHT - 1 - VANISHING[1])
    left = VANISHING[0] + (left_foot - VANISHING[0]) * share
    right = VANISHING[0] + (right_foot - VANISHING[0]) * share
    mask: np.ndarray = (columns >= left) & (columns <= right) & (rows >= top)
    return mask


def _on_line(foot: float, rows: np.ndarray) -> np.ndarray:
    """The columns of the line from the vanishing point to ``foot`` on the bottom row."""
    share = (rows - VANISHING[1]) / (HEIGHT - 1 - VANISHING[1])
    return np.rint(VANISHING[0] + (foot - VANISHING[0]) * share).astype(int)


# ------------------------------------------------------------------------- the frame


def test_lines_of_constant_position_run_through_the_vanishing_point() -> None:
    """On a straight road, a lane line is a line through the vanishing point,
    and the position across the road is the same all the way down it."""
    frame = road_frame(_fan(30.0, 290.0))
    assert frame is not None and int(frame.piece.max()) == 1
    assert frame.across.shape == (HEIGHT, WIDTH) and frame.across.dtype == np.float32
    rows = np.arange(100, 230)
    for foot, expected in ((95.0, 0.25), (160.0, 0.5), (225.0, 0.75)):
        along_line = frame.across[rows, _on_line(foot, rows)]
        assert float(np.abs(along_line - expected).max()) < 0.04
    # The road's width through a pixel is its width along the row, and the
    # way along it at the middle is straight down the frame.
    assert float(frame.width_px[200, 160]) == pytest.approx(260.0 * (200 - 60) / 179, rel=0.06)
    assert abs(float(frame.along[200, 160, 1])) > 0.98
    assert float(frame.weight[200, 160]) == 1.0 and float(frame.weight[30, 160]) == 0.0


def test_a_side_that_is_out_of_the_frame_is_carried_on_to_the_corner() -> None:
    """Most roads leave the picture by its bottom corners. The side that is
    not seen is drawn from the far end of the road to the corner, which is a
    line through the vanishing point too: the lanes still fan from it."""
    frame = road_frame(_fan(-200.0, 240.0))
    assert frame is not None
    rows = np.arange(120, 230)
    line = frame.across[rows, _on_line(150.0, rows)]
    assert float(line.max() - line.min()) < 0.08
    # And further across the road is further across the frame, everywhere.
    assert (np.diff(frame.across[200, 10:215]) >= -1e-4).all()


def test_a_bend_bends_the_lanes_with_it() -> None:
    rows, columns = np.mgrid[0:HEIGHT, 0:WIDTH]
    middle = 100.0 + 0.004 * (rows - 60.0) ** 2
    half = 8.0 + 0.3 * (rows - 60.0)
    mask = (np.abs(columns - middle) <= half) & (rows >= 70)
    frame = road_frame(mask)
    assert frame is not None
    down = np.arange(90, 230)
    centre_line = frame.across[down, np.rint(middle[down, 0]).astype(int)]
    assert float(np.abs(centre_line - 0.5).max()) < 0.06
    # The way along the road leans as the bend does, more at the bottom.
    assert float(frame.along[220, int(middle[220, 0]), 0]) > 0.5
    assert abs(float(frame.along[90, int(middle[90, 0]), 0])) < 0.4


def test_a_road_across_the_picture_is_read_by_columns() -> None:
    """Its rows run along it, and neither side is met along a row."""
    mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
    mask[110:170] = True
    frame = road_frame(mask)
    assert frame is not None
    assert float(frame.across[125, 60]) == pytest.approx(0.25, abs=0.05)
    assert float(frame.across[155, 250]) == pytest.approx(0.75, abs=0.05)
    assert abs(float(frame.along[140, 160, 0])) > 0.98


def test_two_carriageways_are_two_roads_and_holes_in_one_are_filled() -> None:
    mask = _fan(-60.0, 120.0) | _fan(200.0, 420.0)
    # A van standing on the first, cut out of the mask.
    mask[170:190, 40:70] = False
    frame = road_frame(mask)
    assert frame is not None and int(frame.piece.max()) == 2
    assert frame.piece[180, 55] == frame.piece[220, 30] != 0, "the hole was left in the road"
    left, right = frame.piece[220, 30], frame.piece[220, 290]
    assert left != right
    # Each has its own sides: the middle of each is the middle of itself.
    rows = np.arange(120, 230)
    assert float(np.abs(frame.across[rows, _on_line(30.0, rows)] - 0.5).max()) < 0.1


def test_a_region_with_no_smooth_sides_is_not_a_road() -> None:
    rows, columns = np.mgrid[0:HEIGHT, 0:WIDTH]
    teeth = 40.0 * np.sin(rows / 3.0) + 25.0 * np.sin(rows / 1.3)
    ragged = (columns > 90 + teeth) & (columns < 230 - teeth) & (rows > 60)
    assert road_frame(ragged) is None
    assert road_frame(np.zeros((HEIGHT, WIDTH), dtype=bool)) is None
    speck = np.zeros((HEIGHT, WIDTH), dtype=bool)
    speck[100:104, 100:104] = True
    assert road_frame(speck) is None


def test_one_ragged_side_is_drawn_as_a_side_not_seen() -> None:
    """Lamp posts and a bridge in front of one kerb break it up. The road is
    still a road: that side is carried from its far end to the corner."""
    rows, columns = np.mgrid[0:HEIGHT, 0:WIDTH]
    mask = _fan(40.0, 300.0)
    mask &= columns < 300 - 60.0 * (np.sin(rows / 3.0) > 0) * (rows > 150)
    frame = road_frame(mask)
    assert frame is not None
    line = frame.across[np.arange(100, 150), _on_line(100.0, np.arange(100, 150))]
    assert float(line.max() - line.min()) < 0.1


# ------------------------------------------------------------------------- the lanes


def _vehicles(frame_mask: np.ndarray, feet: list[tuple[float, int]], size: float) -> np.ndarray:
    """Boxes standing on the fan's road: each at a bottom-row position and a row."""
    vehicles = np.zeros(frame_mask.shape, dtype=bool)
    for foot, row in feet:
        column = int(_on_line(foot, np.array([row]))[0])
        wide = max(3, round(size * (row - VANISHING[1]) / 179.0))
        vehicles[row - wide : row, column - wide // 2 : column + wide // 2 + 1] = True
    return vehicles


def _lane_of(lanes: Lanes, position: float) -> tuple[float, float]:
    """How far ``position`` is from the middle of its lane, in lanes, and the lane's width."""
    lines = lanes.edges[1]
    index = int(np.clip(np.searchsorted(lines, position) - 1, 0, len(lines) - 2))
    wide = float(lines[index + 1] - lines[index])
    return abs(position - 0.5 * float(lines[index] + lines[index + 1])) / wide, wide


def test_the_lanes_are_where_the_vehicles_are() -> None:
    """Three lanes of 80 pixels at the bottom row, a car in each a good way
    up the road, and no paint: the comb of lanes is slid to where the cars
    sit in the middle of its teeth, and a lane is as wide as 1.7 cars."""
    mask = _fan(40.0, 280.0)
    frame = road_frame(mask)
    assert frame is not None
    feet = [(80.0, 150), (160.0, 180), (240.0, 210), (160.0, 120), (80.0, 220)]
    lanes = road_lanes(frame, _vehicles(mask, feet, 47.0), np.random.default_rng(0))
    assert lanes.painted[1] == 0
    for foot in (80.0, 160.0, 240.0):
        off_centre, wide = _lane_of(lanes, (foot - 40.0) / 240.0)
        assert off_centre < 0.15, f"the car at {foot} is not in the middle of a lane"
        assert wide == pytest.approx(80.0 / 240.0, rel=0.2)


def _painted_road(lines: tuple[float, ...] = (40.0, 120.0, 200.0, 280.0)) -> np.ndarray:
    """A grey fan of road with white lines painted from the vanishing point to those feet."""
    scene = np.full((HEIGHT, WIDTH, 3), 0.12, dtype=np.float32)
    rows = np.arange(74, HEIGHT)
    for foot in lines:
        columns = _on_line(foot, rows)
        for offset in (0, 1):
            scene[rows, np.clip(columns + offset, 0, WIDTH - 1)] = 0.8
    return scene


def test_painted_lines_are_thin_bright_long_and_lie_along_the_road() -> None:
    """Which is what tells a lane line from a crossing, an arrow or a patch
    of light, all of which are bright."""
    mask = _fan(30.0, 290.0)
    frame = road_frame(mask)
    assert frame is not None
    scene = _painted_road()
    scene[150:156, 60:250][:, ::14] = 0.8  # studs across the road
    scene[180:186, 100:220] = 0.8  # a stop line, across it
    scene[100:140, 152:172] = np.maximum(scene[100:140, 152:172], 0.5)  # a pool of light
    paint = painted_lines(scene, frame, mask, None)
    rows = np.arange(90, 145)
    for foot in (120.0, 200.0):
        assert paint[rows, _on_line(foot, rows)].mean() > 0.8, "a lane line was not found"
    assert not paint[181:185, 110:210].any(), "a line across the road was taken for a lane line"
    assert paint[105:135, 156:168].mean() < 0.05, "a pool of light was taken for paint"
    # And not near what stands on the road.
    on_road = np.zeros_like(mask)
    on_road[200:230, 100:140] = True
    assert not painted_lines(scene, frame, mask, on_road)[205:225, 105:135].any()


def test_lanes_are_read_off_the_paint_before_anything_else() -> None:
    """Three lanes between four lines. The cars say a lane is half as wide
    as it is, and sit off-centre: the paint is believed."""
    mask = _fan(30.0, 290.0)
    frame = road_frame(mask)
    assert frame is not None
    scene = _painted_road()
    paint = painted_lines(scene, frame, mask, None)
    feet = [(70.0, 150), (150.0, 200)]
    lanes = road_lanes(frame, _vehicles(mask, feet, 30.0), np.random.default_rng(0), paint=paint)
    assert lanes.painted[1] >= 3
    lines = lanes.edges[1]
    for foot in (120.0, 200.0):
        position = (foot - 30.0) / 260.0
        assert float(np.abs(lines - position).min()) < 0.02, f"no lane line on the paint at {foot}"
    assert float(np.median(np.diff(lines))) == pytest.approx(80.0 / 260.0, rel=0.1)


def test_a_line_worn_away_leaves_a_gap_of_two_lanes() -> None:
    mask = _fan(30.0, 290.0)
    frame = road_frame(mask)
    assert frame is not None
    scene = _painted_road((40.0, 120.0, 280.0))
    paint = painted_lines(scene, frame, mask, None)
    lanes = road_lanes(frame, None, np.random.default_rng(0), paint=paint)
    assert float(np.abs(lanes.edges[1] - (200.0 - 30.0) / 260.0).min()) < 0.03


def test_lanes_are_one_width_whatever_the_paint_says() -> None:
    mask = _fan(30.0, 290.0)
    frame = road_frame(mask)
    assert frame is not None
    scene = _painted_road((40.0, 95.0, 118.0, 201.0, 280.0))
    paint = painted_lines(scene, frame, mask, None)
    lanes = road_lanes(frame, None, np.random.default_rng(0), paint=paint)
    # Whatever it makes of them, the lanes it lays are of one width.
    gaps = np.diff(lanes.edges[1][1:-1])
    assert len(lanes.edges[1]) >= 3
    assert len(gaps) == 0 or float(np.ptp(gaps)) < 0.02


def test_painted_lines_put_a_road_right_where_its_mask_is_wrong() -> None:
    """A mask that takes in a hard shoulder on one side, low down, has a
    side that is no line through the vanishing point, and the lanes read
    off it lean. The paint runs to the vanishing point whatever the mask
    does, and the road is redrawn through that."""
    mask = _fan(30.0, 290.0)
    rows, columns = np.mgrid[0:HEIGHT, 0:WIDTH]
    mask |= (rows > 150) & (columns > 150) & (columns < 300 + 0.4 * (rows - 150))
    scene = _painted_road()
    crooked = road_frame(mask)
    assert crooked is not None
    paint = painted_lines(scene, crooked, mask, None)
    true = straighten(crooked, paint)
    down = np.arange(100, 230)
    line = true.across[down, _on_line(120.0, down)]
    assert float(line.max() - line.min()) < 0.02
    was = crooked.across[down, _on_line(120.0, down)]
    assert float(was.max() - was.min()) > 2.0 * float(line.max() - line.min())
    assert abs(float(true.along[200, 250, 0])) > 0.3, "the way along it does not fan"
    # Nothing to go by: left as it was.
    assert np.array_equal(straighten(crooked, np.zeros_like(mask)).across, crooked.across)


def test_reading_a_road_does_all_of_it_in_one_call() -> None:
    mask = _fan(30.0, 290.0)
    nothing = np.zeros_like(mask)
    read = read_road(_painted_road(), mask, nothing, None, np.random.default_rng(0))
    assert read is not None
    frame, lanes = read
    assert lanes.painted[1] >= 3 and float(frame.weight[200, 160]) == 1.0
    assert read_road(_painted_road(), nothing, nothing, None, np.random.default_rng(0)) is None


def test_a_road_with_no_vehicles_and_no_paint_gets_a_drawn_number_of_lanes() -> None:
    frame = road_frame(_fan(40.0, 280.0))
    assert frame is not None
    counts = set()
    for seed in range(12):
        lanes = road_lanes(frame, None, np.random.default_rng(seed), lane_count_range=(2, 4))
        counts.add(len(lanes.edges[1]) - 1)
    assert counts == {2, 3, 4}


def test_vehicles_lying_across_a_road_say_it_was_read_the_wrong_way() -> None:
    """A junction seen from above, read as a road running up the frame, has
    its traffic lying across it. Such a road gets no lanes."""
    mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
    mask[:, 100:220] = True
    frame = road_frame(mask)
    assert frame is not None
    lying = np.zeros_like(mask)
    lying[60:72, 120:160] = True
    lying[150:162, 150:190] = True
    lanes = road_lanes(frame, lying, np.random.default_rng(0))
    assert len(lanes.edges[1]) == 0
    tracks = wheel_tracks(frame, lanes, 0.1, np.random.default_rng(0))
    assert float(tracks.max()) == 0.0
    standing = np.zeros_like(mask)
    standing[60:100, 120:136] = True
    assert len(road_lanes(frame, standing, np.random.default_rng(0)).edges[1]) >= 2


# ------------------------------------------------------------------------ the tracks


def _straight_lanes() -> tuple[np.ndarray, Lanes]:
    """A road up the middle of the frame, 200 pixels wide, in two lanes."""
    mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
    mask[:, 60:260] = True
    return mask, Lanes(edges=[np.zeros(0), np.array([0.0, 0.5, 1.0])], painted=[0, 0])


def test_each_lane_has_two_tracks_where_its_wheels_run() -> None:
    mask, lanes = _straight_lanes()
    frame = road_frame(mask)
    assert frame is not None
    tracks = wheel_tracks(frame, lanes, 0.06, np.random.default_rng(1))
    assert tracks.shape == (HEIGHT, WIDTH) and tracks.dtype == np.float32
    assert float(tracks.min()) >= 0.0 and float(tracks.max()) <= 1.0
    row = tracks[120]
    # Lane centres at 110 and 210; wheels 22 pixels either side of each.
    for wheel in (88, 132, 188, 232):
        assert row[wheel] > 0.3, f"no track at column {wheel}"
    for bare in (110, 160, 210):
        assert row[bare] < 0.02, f"a track where no wheel runs, at column {bare}"
    assert float(tracks[:, :55].max()) == 0.0, "tracks off the road"
    # Even along its length, with nothing to make it otherwise.
    assert float(np.ptp(tracks[20:220, 88])) < 0.05


def test_no_two_lanes_and_no_two_stretches_are_worn_alike() -> None:
    """Ruled lines of one darkness are a pedestrian crossing."""
    mask, lanes = _straight_lanes()
    frame = road_frame(mask)
    assert frame is not None
    rng = np.random.default_rng(4)
    rows = np.linspace(0.0, 1.0, HEIGHT, dtype=np.float32)[:, None] * np.ones((1, WIDTH))
    worn = (0.5 + 0.5 * np.sin(rows * 20.0)).astype(np.float32)
    lightly = wheel_tracks(frame, lanes, 0.06, rng, worn=worn, wear_floor=0.2)
    column = lightly[:, 86:91].max(axis=1)
    assert float(column.min()) < 0.4 * float(column.max())
    heavily = wheel_tracks(frame, lanes, 0.06, rng, worn=worn, wear_floor=0.9)
    column = heavily[:, 86:91].max(axis=1)
    assert float(column.min()) > 0.8 * float(column.max())
    # Lanes carry different shares of the traffic.
    peaks = {round(float(lightly[60, around - 4 : around + 5].max()), 3) for around in (88, 188)}
    assert len(peaks) == 2

    wander = np.broadcast_to(rows, (HEIGHT, WIDTH)).astype(np.float32)
    strayed = wheel_tracks(frame, lanes, 0.06, np.random.default_rng(1), ragged=wander)
    top, bottom = int(np.argmax(strayed[5, 60:120])), int(np.argmax(strayed[234, 60:120]))
    assert abs(top - bottom) > 8, "a track that strays nowhere"


def test_far_down_the_road_the_tracks_run_together() -> None:
    """Where a lane is a few pixels wide there are no two tracks to see: the
    road is greyer all over, and no more of it than the tracks were."""
    frame = road_frame(_fan(40.0, 280.0))
    assert frame is not None
    lanes = Lanes(edges=[np.zeros(0), np.linspace(0.0, 1.0, 6)], painted=[0, 0])
    tracks = wheel_tracks(frame, lanes, 0.06, np.random.default_rng(1))
    near, far = tracks[225, 60:260], tracks[78, 155:166]
    assert float(near.max()) > 0.3 and float(near.min()) < 0.02
    assert float(far.max()) < 0.75 * float(near.max())
    assert float(far.mean()) < 2.0 * float(near.mean())


# --------------------------------------------------------------------- in the snow


class _Level(DepthEstimator):
    """Level ground under a camera pitched well down, filling the frame."""

    def estimate(self, image: np.ndarray) -> np.ndarray:
        """Depth falling towards the bottom of the frame, as a ground plane's does."""
        height, width = image.shape[:2]
        rows = np.arange(height, dtype=np.float32).reshape(height, 1)
        depth = 1.0 / (rows + 0.6 * height)
        depth = (depth - depth.min()) / (depth.max() - depth.min())
        return np.broadcast_to(depth, (height, width)).astype(np.float32)


class _Labels(SceneLabelEstimator):
    """A label map that is always the same one."""

    def __init__(self, labels: np.ndarray) -> None:
        """Keep the map."""
        self.labels = labels

    def estimate(self, image: np.ndarray) -> np.ndarray:
        """The map, whatever the frame."""
        return self.labels


def _snowed(traffic: float, labelled: bool = True, seed: int = 0) -> tuple[np.ndarray, dict]:
    """A dark road up the frame under deep snow, worn by that much traffic."""
    labels = np.full((HEIGHT, WIDTH), int(SceneClass.GRASS), dtype=np.uint8)
    labels[:, 60:260] = int(SceneClass.ROAD)
    labels[150:180, 96:124] = int(SceneClass.VEHICLE)
    labels[90:120, 196:224] = int(SceneClass.VEHICLE)
    image = np.full((HEIGHT, WIDTH, 3), 60, dtype=np.uint8)
    extra = {"labels": _Labels(labels)} if labelled else {}
    provider = SharedDepthProvider(_Level(), depth_range_m=(4.0, 40.0), **extra)  # type: ignore[arg-type]
    snow = RandomSnow(
        depth_provider=provider,
        p=1.0,
        max_flakes=0,
        accumulation_probability=1.0,
        accumulation_strength_range=(1.0, 1.0),
        accumulation_patchiness_range=(0.0, 0.0),
        overcast_range=(0.0, 0.0),
        camera_pitch_range_deg=(35.0, 35.0),
        vehicle_cover_range=(0.0, 0.0),
        wheel_track_range=(traffic, traffic),
    )
    snow.set_random_seed(seed)
    with recording() as log:
        out = snow(image=image)["image"].astype(np.float32)
    return out, log[0]["drew"]


def test_traffic_wears_dark_tracks_into_a_snowed_road() -> None:
    untouched, drew = _snowed(0.0)
    assert not drew["wheel_tracks_drawn"]
    worn, drew = _snowed(0.7)
    assert drew["wheel_tracks_drawn"] and drew["road_share"] == pytest.approx(0.625, abs=0.02)
    road = (slice(20, 80), slice(70, 250))
    assert float(untouched[road].std()) < 10.0, "the fixture's snow is not an even coat"
    assert float(worn[road].min()) < 0.7 * float(untouched[road].mean())
    assert float(worn[road].max()) > 0.9 * float(untouched[road].mean()), "no snow left between"
    # Tracks run along the road: a column down it is one thing or the other
    # for most of its length, a row across it is both.
    across = worn[50, 70:250].mean(axis=1)
    assert float(across.max() - across.min()) > 60.0
    # The verge is as it was.
    assert np.array_equal(worn[:, :50], untouched[:, :50])

    heavier, _ = _snowed(1.0)
    assert float(heavier[road].mean()) < float(worn[road].mean()) < float(untouched[road].mean())


def test_without_labels_there_is_no_road_to_follow_and_nothing_changes() -> None:
    """The draws for the tracks come after every other, so a frame without
    them is the frame it was before there were any."""
    plain, drew = _snowed(0.9, labelled=False)
    assert not drew["wheel_tracks_drawn"] and drew["road_share"] == 0.0
    assert np.array_equal(plain, _snowed(0.0, labelled=False)[0])
    with pytest.raises(ValueError):
        RandomSnow(depth_provider=SharedDepthProvider(_Level()), wheel_track_range=(0.5, 1.5))


# -------------------------------------------------------------------- on a wet road


def test_water_lies_in_the_gutters_and_the_wheel_ruts() -> None:
    from surmentations.atmosphere.rain import water_lie

    mask, lanes = _straight_lanes()
    frame = road_frame(mask)
    assert frame is not None
    lie = water_lie(frame, lanes, np.random.default_rng(0))
    assert lie.shape == (HEIGHT, WIDTH) and lie.dtype == np.float32
    row = lie[120]
    assert row[61] > 0.9 and row[258] > 0.9, "no water against the kerbs"
    assert row[88] > 0.2 and row[232] > 0.2, "none in the ruts"
    assert row[110] < 0.02 and row[160] < 0.02, "the crown of the lane is as low as its ruts"
    assert float(lie[:, :55].max()) == 0.0


def test_puddles_stand_along_the_road_where_it_lies_low() -> None:
    """With a little water it is all in the ruts and gutters, in strips; with
    a lot it is everywhere. And a puddle has an edge."""
    from surmentations.atmosphere.rain import _puddle_map

    mask, lanes = _straight_lanes()
    frame = road_frame(mask)
    assert frame is not None
    lie = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    lie[:, 86:92] = lie[:, 130:136] = 1.0
    depth = _Level().estimate(np.zeros((HEIGHT, WIDTH, 3), np.uint8)) * 36.0 + 4.0
    wet = mask.astype(np.float32)

    def water(coverage: float, low: np.ndarray | None) -> np.ndarray:
        """The puddles for that much water, with or without a lie of the land."""
        return _puddle_map(depth, 300.0, 35.0, coverage, np.random.default_rng(5), low, wet)

    little = water(0.05, lie)
    in_ruts = float(little[:, 86:92].mean() + little[:, 130:136].mean()) / 2.0
    assert in_ruts > 6.0 * float(little[:, 100:125].mean() + 1e-4)
    assert float(little[mask].mean()) == pytest.approx(0.05, abs=0.03), "a share of the wet road"
    assert float(water(0.05, None)[:, 86:92].mean()) < 0.3 * in_ruts
    plenty = water(0.6, lie)
    assert float(plenty[:, 100:125].mean()) > 0.3
    edge = (little > 0.05) & (little < 0.95)
    assert float(edge.mean()) < 0.5 * float((little > 0.5).mean()) + 0.01, "puddles with no edge"


def _rained(labelled: bool = True, **kwargs: object) -> tuple[np.ndarray, dict]:
    """The snow fixture's road in rain, soaked, with whatever else is asked."""
    labels = np.full((HEIGHT, WIDTH), int(SceneClass.GRASS), dtype=np.uint8)
    labels[:60] = int(SceneClass.SKY)
    labels[60:, 60:260] = int(SceneClass.ROAD)
    labels[150:180, 96:124] = int(SceneClass.VEHICLE)
    labels[90:120, 196:224] = int(SceneClass.VEHICLE)
    image = np.full((HEIGHT, WIDTH, 3), 90, dtype=np.uint8)
    image[:60] = (150, 180, 230)
    extra = {"labels": _Labels(labels)} if labelled else {}
    provider = SharedDepthProvider(_Level(), depth_range_m=(4.0, 40.0), **extra)  # type: ignore[arg-type]
    settings: dict[str, object] = {
        "max_streaks": 0,
        "wetness_probability": 1.0,
        "wetness_darkening_range": (0.4, 0.4),
        "puddle_coverage_range": (0.1, 0.1),
        "camera_pitch_range_deg": (35.0, 35.0),
        "rain_rate_range_mm_h": (2.0, 2.0),
    }
    seed = int(kwargs.pop("seed", 0))  # type: ignore[call-overload]
    rain = RandomRain(depth_provider=provider, p=1.0, **{**settings, **kwargs})  # type: ignore[arg-type]
    rain.set_random_seed(seed)
    with recording() as log:
        out = rain(image=image)["image"].astype(np.float32)
    return out, {**log[0]["drew"], **log[0]["out"]}


def test_a_soaked_road_is_under_cloud_whatever_is_falling() -> None:
    """A road does not get soaked under a clear sky. Light rain on a wet
    road used to be lit by the sun, the light a wet road is least seen in."""
    _out, soaked = _rained()
    _out, dry = _rained(wetness_probability=0.0)
    _out, damp = _rained(wetness_darkening_range=(0.9, 0.9))
    assert soaked["overcast"] > 0.7
    assert dry["overcast"] < 0.2 and damp["overcast"] < 0.2
    _out, told = _rained(overcast_range=(0.0, 0.0))
    assert told["overcast"] == 0.0, "a caller who says there is no cloud is believed"


def test_spray_trails_the_vehicles_on_a_wet_road_and_only_when_asked() -> None:
    plain, drew = _rained()
    assert not drew["spray_drawn"] and "spray" not in drew or drew["spray"] == 0.0
    # A vehicle may be standing, and have none: the first draw in which
    # one of the two is moving.
    for seed in range(8):
        plain, _drew = _rained(seed=seed)
        sprayed, drew = _rained(spray_range=(1.0, 1.0), seed=seed)
        if drew["spray_drawn"]:
            break
    assert drew["spray_drawn"]
    changed = np.abs(sprayed - plain).max(axis=2) > 6.0
    assert 0.002 < float(changed.mean()) < 0.4
    # Round the vehicles, and not off in the grass.
    assert float(changed[:, :40].mean()) < 0.01
    assert float(changed[60:230, 70:250].mean()) > 3.0 * float(changed.mean()) * 0.3

    # Nothing without labels, without a film on the road, or without a road.
    for kwargs in ({"wetness_probability": 0.0}, {"wetness_darkening_range": (0.9, 0.9)}):
        _out, drew = _rained(spray_range=(1.0, 1.0), **kwargs)
        assert not drew["spray_drawn"]
    _out, drew = _rained(labelled=False, spray_range=(1.0, 1.0))
    assert not drew["spray_drawn"]


def test_a_plume_lies_behind_its_vehicle_along_the_road() -> None:
    """Densest at the tail, a couple of lengths long, and on one side of the
    vehicle only: the side it has come from."""
    mask, _lanes = _straight_lanes()
    frame = road_frame(mask)
    assert frame is not None
    vehicles = np.zeros((HEIGHT, WIDTH), dtype=bool)
    vehicles[100:130, 146:174] = True
    seen = set()
    for seed in range(40):
        spray = vehicle_spray(frame, vehicles, np.random.default_rng(seed))
        assert spray.shape == (HEIGHT, WIDTH) and spray.dtype == np.float32
        above, below = float(spray[40:95, 130:190].sum()), float(spray[135:200, 130:190].sum())
        seen.add("towards" if below > above else "away")
        assert min(above, below) < 0.05 * max(above, below), "a plume on both sides of a car"
        # Not out to the sides, and not on the far side of the frame.
        assert float(spray[:, :100].max()) < 0.02 and float(spray[:, 220:].max()) < 0.02
        if below > above:
            assert float(spray[131:140, 150:170].mean()) > float(spray[180:190, 150:170].mean())
        else:
            # Hidden by the vehicle that threw it.
            assert float(spray[105:125, 150:170].max()) < 0.05
    assert seen == {"towards", "away"}, "the direction is not drawn"


def test_no_dry_rim_is_left_round_a_labelled_vehicle() -> None:
    """A network's outline runs a little outside the car, and softening it
    left a ring of dry, pale road round every vehicle on a dark wet one."""
    from surmentations.atmosphere.rain import _read_wet_ground

    depth = _Level().estimate(np.zeros((HEIGHT, WIDTH, 3), np.uint8)) * 36.0 + 4.0
    scene = np.full((HEIGHT, WIDTH, 3), 0.1, dtype=np.float32)
    labels = np.full((HEIGHT, WIDTH), int(SceneClass.ROAD), dtype=np.uint8)
    labels[100:140, 130:190] = int(SceneClass.VEHICLE)
    wet, *_ = _read_wet_ground(depth, scene, 60.0, 35.0, 6.0, labels)
    assert float(wet[115:125, 150:170].mean()) < 0.05, "the car is wet"
    for rim in (wet[97:99, 140:180], wet[141:143, 140:180], wet[105:135, 127:129]):
        assert float(rim.mean()) > 0.85, "a dry rim round the car"


def test_a_plume_follows_its_vehicle_s_own_heading() -> None:
    """A car crossing the lanes at an angle trails its spray along its own
    line, not the road's; and it is as wide as the car is across that line."""
    mask, _lanes = _straight_lanes()
    frame = road_frame(mask)
    assert frame is not None
    slanted = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    # 44 long and 18 wide, leaning 30 degrees from the road's direction.
    box = cv2.boxPoints(((160.0, 110.0), (18.0, 44.0), 30.0))
    cv2.fillConvexPoly(slanted, box.astype(np.int32), 1)
    leans = []
    for seed in range(30):
        spray = vehicle_spray(frame, slanted.astype(bool), np.random.default_rng(seed))
        below = spray[140:210]
        if float(below.sum()) < 1.0 or float(spray[:80].sum()) > float(below.sum()):
            continue
        rows, columns = np.mgrid[140:210, 0:WIDTH]
        centre = float((columns * below).sum() / below.sum())
        leans.append(centre - 160.0)
    assert leans, "no plume was thrown towards the camera in thirty draws"
    # The tail of a box leaning that way is to the left of its middle, and
    # the plume runs on down and to the left from it.
    assert max(leans) < -8.0


def test_tail_lamps_say_which_way_the_traffic_runs() -> None:
    """Red lamps are seen from behind: those vehicles are driving away, and
    their spray trails towards the camera, every time and not half of it."""
    mask, _lanes = _straight_lanes()
    frame = road_frame(mask)
    assert frame is not None
    vehicles = np.zeros((HEIGHT, WIDTH), dtype=bool)
    scene = np.full((HEIGHT, WIDTH, 3), 0.1, dtype=np.float32)
    for left, top in ((100, 60), (190, 120)):
        vehicles[top : top + 30, left : left + 26] = True
        scene[top + 24 : top + 27, left + 2 : left + 6] = (0.9, 0.05, 0.05)
        scene[top + 24 : top + 27, left + 20 : left + 24] = (0.9, 0.05, 0.05)
    for seed in range(12):
        spray = vehicle_spray(frame, vehicles, np.random.default_rng(seed), scene)
        above = float(spray[20:58].sum() + spray[95:118, 180:230].sum())
        below = float(spray[92:118, 90:140].sum() + spray[152:200].sum())
        assert above <= 0.05 * below + 1e-6, "a plume trailing ahead of a car seen from behind"


def test_a_vehicle_lying_across_the_road_and_one_on_a_scrap_of_road_get_no_plume() -> None:
    """Crossing, turning or parked; or on a road whose direction is a guess.
    Both used to shoot a jet of mist out sideways."""
    mask, _lanes = _straight_lanes()
    frame = road_frame(mask)
    assert frame is not None
    across = np.zeros((HEIGHT, WIDTH), dtype=bool)
    across[110:124, 130:190] = True
    for seed in range(10):
        assert float(vehicle_spray(frame, across, np.random.default_rng(seed)).max()) == 0.0
    scrap = np.zeros((HEIGHT, WIDTH), dtype=bool)
    scrap[60:200, 150:168] = True
    small = road_frame(scrap)
    assert small is not None and float((small.piece > 0).mean()) < 0.05
    car = np.zeros((HEIGHT, WIDTH), dtype=bool)
    car[100:124, 152:166] = True
    for seed in range(10):
        assert float(vehicle_spray(small, car, np.random.default_rng(seed)).max()) == 0.0


def test_a_plume_is_a_low_cloud_as_wide_as_its_vehicle() -> None:
    """Not a jet: at its tail it is about the vehicle's width, and at town
    speed it does not reach far beyond a length and a half."""
    mask, _lanes = _straight_lanes()
    frame = road_frame(mask)
    assert frame is not None
    vehicles = np.zeros((HEIGHT, WIDTH), dtype=bool)
    vehicles[90:130, 148:172] = True
    for seed in range(40):
        rng = np.random.default_rng(seed)
        slow = road_vehicles(frame, vehicles, rng, speed_kmh=45.0)
        spray = vehicle_spray(frame, vehicles, rng, traffic=slow)
        if float(spray[131:].sum()) < 0.2:
            continue
        row = spray[136]
        wide = int((row > 0.5 * row.max()).sum())
        assert 14 <= wide <= 44, f"a plume {wide} pixels wide behind a car of 24"
        assert float(spray[215:].max()) < 0.1 * float(spray.max())
        return
    raise AssertionError("no plume was thrown towards the camera in forty draws")


def test_a_car_seen_from_the_front_trails_its_spray_away_from_the_camera() -> None:
    """A rear shows red lenses and a front shows none. Cars with no red on
    them are coming on, and their spray is behind them, up the frame: never
    off the bonnet. And a red car is not a tail lamp."""
    mask, _lanes = _straight_lanes()
    frame = road_frame(mask)
    assert frame is not None
    vehicles = np.zeros((HEIGHT, WIDTH), dtype=bool)
    scene = np.full((HEIGHT, WIDTH, 3), 0.1, dtype=np.float32)
    for left, top in ((100, 60), (190, 120)):
        vehicles[top : top + 30, left : left + 26] = True
        scene[top : top + 30, left : left + 26] = (0.3, 0.32, 0.35)
    # A red car between them, which used to be taken for one big tail lamp.
    vehicles[150:180, 140:166] = True
    scene[150:180, 140:166] = (0.5, 0.04, 0.04)
    thrown = 0.0
    for seed in range(12):
        spray = vehicle_spray(frame, vehicles, np.random.default_rng(seed), scene)
        towards = float(spray[92:118, 90:124].sum() + spray[152:200, 184:230].sum())
        towards += float(spray[182:230, 130:176].sum())
        away = float(spray[20:58, 90:140].sum() + spray[80:118, 180:230].sum())
        assert towards <= 0.05 * away + 1e-6, "spray off the front of a car that is coming on"
        thrown += away
    assert thrown > 0.0


def test_spray_hangs_over_a_busy_road_and_thickens_with_distance() -> None:
    """What one vehicle throws up the next drives through: a layer of mist
    over the carriageway, which a sight line runs through more of the
    further off it meets the road. Near cars are clear and far traffic is
    gone from the wheels up, its roofs last."""
    mask, _lanes = _straight_lanes()
    frame = road_frame(mask)
    assert frame is not None
    rows = np.arange(HEIGHT, dtype=np.float32)[:, None]
    depth = np.broadcast_to(20.0 + 2.0 * (HEIGHT - 1 - rows), (HEIGHT, WIDTH)).astype(np.float32)
    vehicles = np.zeros((HEIGHT, WIDTH), dtype=bool)
    for top, left in ((30, 80), (60, 180), (100, 120), (150, 200), (190, 90), (120, 220)):
        vehicles[top : top + 24, left : left + 20] = True
    mist = road_mist(frame, vehicles, depth)
    assert mist.shape == (HEIGHT, WIDTH) and mist.dtype == np.float32
    assert float(mist[:, :50].max()) < 0.01, "mist off the road"
    far, near = float(mist[20, 150]), float(mist[230, 150])
    assert far > 3.0 * near > 0.0
    # On a vehicle, thick at its wheels and thinner at its roof, which is
    # most of the way up the layer and not clear of it.
    assert float(mist[82, 190]) > 2.0 * float(mist[61, 190]) > 0.0
    # An empty road has none, and a road with one car on it little.
    assert float(road_mist(frame, np.zeros_like(vehicles), depth).max()) == 0.0
    alone = np.zeros_like(vehicles)
    alone[100:124, 120:140] = True
    assert float(road_mist(frame, alone, depth)[20, 150]) < 0.25 * far


def test_a_wake_is_long_and_bigger_behind_a_bigger_vehicle() -> None:
    """Some lengths long, thinning all the way, and several cars' worth
    behind a lorry."""
    mask, _lanes = _straight_lanes()
    frame = road_frame(mask)
    assert frame is not None
    traffic = np.zeros((HEIGHT, WIDTH), dtype=bool)
    for left in (80, 150, 215):
        traffic[60:84, left : left + 16] = True
    lorry = traffic.copy()
    lorry[52:84, 144:172] = True
    behind_car, behind_lorry, lengths = 0.0, 0.0, []
    for seed in range(30):
        cars = vehicle_spray(frame, traffic, np.random.default_rng(seed))
        if float(cars[90:].sum()) < 1.0:
            continue
        column = cars[90:, 150:166].mean(axis=1)
        lengths.append(int((column > 0.1 * column.max()).sum()))
        assert float(column[:10].mean()) > float(column[40:60].mean()), "a wake that does not thin"
        behind_car += float(cars[90:, 140:176].sum())
        behind_lorry += float(
            vehicle_spray(frame, lorry, np.random.default_rng(seed))[90:, 140:176].sum()
        )
    assert lengths and float(np.median(lengths)) > 2.0 * 24, "a wake shorter than two car lengths"
    assert behind_lorry > 1.5 * behind_car


def _car(colour: tuple[float, float, float], lenses: tuple[float, float, float] | None) -> tuple:
    """A car of 30 by 26 pixels, as its colours and where they are, with a pair of lamps or none."""
    rows, columns = np.mgrid[0:30, 0:26]
    picture = np.empty((30, 26, 3), dtype=np.float32)
    picture[:] = colour
    if lenses is not None:
        picture[22:25, 2:6] = lenses
        picture[22:25, 20:24] = lenses
    return picture.reshape(-1, 3), rows.ravel(), columns.ravel()


def test_a_rear_is_known_by_its_lenses_and_not_by_red_paint_or_an_indicator() -> None:
    """Small patches of red are tail lamps. A red car is red all over and
    says nothing; an amber indicator is on the front of a car too; and a
    car with no red at all, large enough to show it, is seen from the front."""
    grey = (0.3, 0.32, 0.35)
    assert vehicle_facing(*_car(grey, (0.6, 0.04, 0.04)), night=False) == 1.0
    assert vehicle_facing(*_car(grey, None), night=False) == -1.0
    assert vehicle_facing(*_car((0.5, 0.04, 0.04), None), night=False) is None
    assert vehicle_facing(*_car(grey, (0.9, 0.4, 0.02)), night=False) != 1.0
    # One lens is enough: the other is as often hidden or in shade.
    colour, rows, columns = _car(grey, (0.6, 0.04, 0.04))
    colour = colour.reshape(30, 26, 3).copy()
    colour[22:25, 20:24] = grey
    assert vehicle_facing(colour.reshape(-1, 3), rows, columns, night=False) == 1.0
    # Too small to show a lens, a car with none says nothing.
    few = rows < 6
    assert vehicle_facing(_car(grey, None)[0][few], rows[few], columns[few], night=False) is None
    # At night, white lamps burning are headlamps.
    dark = _car((0.01, 0.01, 0.01), (0.95, 0.95, 0.9))
    assert vehicle_facing(*dark, night=True) == -1.0


def test_a_road_s_traffic_is_heard_together() -> None:
    """One way, or two ways divided at the widest empty strip across the
    road: a few vehicles' evidence decides for all of them, and one misread
    among a carriageway's lanes does not turn its neighbours round."""
    rng = np.random.default_rng(0)
    at = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    even = np.ones(6)
    # One rear seen, on a road with no divide to speak of: all drive away.
    ways = traffic_directions(at, np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]), even, rng)
    assert (ways == 1.0).all()
    # Fronts on one side of the gap and rears on the other: two ways.
    ways = traffic_directions(at, np.array([-1.0, -1.0, 0.0, 0.0, 1.0, 1.0]), even, rng)
    assert ways.tolist() == [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]
    # A front misread among the rears is outvoted, not obeyed.
    ways = traffic_directions(at, np.array([-1.0, -1.0, 0.0, -1.0, 1.0, 1.0]), even, rng)
    assert ways.tolist() == [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]
    # And the divide is the widest gap, not wherever suits a misreading:
    # this used to cut the right-hand carriageway in two.
    ways = traffic_directions(
        at, np.array([0.0, 0.0, 0.0, -1.0, 1.0, 1.0]), np.array([1.0, 1.0, 1.0, 0.3, 1.0, 1.0]), rng
    )
    assert (ways[3:] == 1.0).all()
    # With nothing said the road's one direction is drawn, either way.
    drawn = {
        float(traffic_directions(at, np.zeros(6), even, np.random.default_rng(seed))[0])
        for seed in range(20)
    }
    assert drawn == {-1.0, 1.0}
    for seed in range(5):
        ways = traffic_directions(at, np.zeros(6), even, np.random.default_rng(seed))
        assert len(set(ways.tolist())) == 1


def _two_carriageways() -> tuple:
    """Fronts on the left of a road and rears on its right, with the picture of them."""
    mask, _lanes = _straight_lanes()
    frame = road_frame(mask)
    assert frame is not None
    vehicles = np.zeros((HEIGHT, WIDTH), dtype=bool)
    scene = np.full((HEIGHT, WIDTH, 3), 0.1, dtype=np.float32)
    for left, top, rear in (
        (80, 100, False),
        (100, 170, False),
        (200, 100, True),
        (215, 170, True),
    ):
        vehicles[top : top + 30, left : left + 26] = True
        scene[top : top + 30, left : left + 26] = (0.3, 0.32, 0.35)
        if rear:
            scene[top + 22 : top + 25, left + 2 : left + 6] = (0.6, 0.04, 0.04)
            scene[top + 22 : top + 25, left + 20 : left + 24] = (0.6, 0.04, 0.04)
    return frame, vehicles, scene


def test_each_side_of_a_road_trails_its_spray_its_own_way() -> None:
    """Oncoming traffic on one side and departing on the other, as a divided
    road has: each vehicle's spray is behind it."""
    frame, vehicles, scene = _two_carriageways()
    for seed in range(8):
        traffic = road_vehicles(frame, vehicles, np.random.default_rng(seed), scene)
        assert len(traffic) == 4
        for vehicle in traffic:
            assert vehicle.away == (1.0 if vehicle.foot[1] > 160 else -1.0)
        spray = vehicle_spray(frame, vehicles, np.random.default_rng(seed), scene, traffic)
        # Coming on, on the left: nothing off the bonnet of the nearer car,
        # below it, and its wake up the road behind it.
        assert float(spray[203:232, 96:130].sum()) < 0.05 * float(spray[135:168, 96:130].sum())
        # Driving away, on the right: nothing ahead of it, above the car.
        assert float(spray[60:98, 198:228].sum()) < 0.05 * float(spray[132:165, 198:228].sum())


def test_spray_goes_with_speed() -> None:
    """With nearly its cube: half the speed is a seventh of the spray, a
    queue throws up none, and a wake is as long as its vehicle goes while
    the cloud hangs."""
    assert spray_amount(0.0) == 0.0
    assert spray_amount(110.0) == pytest.approx(1.0)
    assert spray_amount(55.0) == pytest.approx(0.5**2.8)
    assert spray_amount(40.0) < spray_amount(80.0) < spray_amount(120.0)
    mask, _lanes = _straight_lanes()
    frame = road_frame(mask)
    assert frame is not None
    vehicles = np.zeros((HEIGHT, WIDTH), dtype=bool)
    scene = np.full((HEIGHT, WIDTH, 3), 0.1, dtype=np.float32)
    vehicles[60:84, 150:166] = True
    scene[78:81, 151:154] = scene[78:81, 162:165] = (0.6, 0.04, 0.04)
    thrown, reach = [], []
    for speed in (0.0, 40.0, 80.0, 120.0):
        total, far = 0.0, 0.0
        for seed in range(6):
            rng = np.random.default_rng(seed)
            traffic = road_vehicles(frame, vehicles, rng, scene, speed)
            spray = vehicle_spray(frame, vehicles, rng, scene, traffic)
            total += float(spray.sum())
            far += float(spray[150:].sum())
        thrown.append(total)
        reach.append(far)
    assert thrown[0] == 0.0, "spray behind a standing car"
    assert thrown[1] < 0.3 * thrown[2] < 0.3 * thrown[3]
    assert reach[1] < 0.05 * reach[3], "a wake as long at town speed as on a motorway"


def test_speeds_are_spread_about_each_road_s_mean_and_a_lorry_keeps_to_its_limit() -> None:
    """Normal about the road's mean, with a spread of a ninth to a fifth of
    it, and the road's mean a little either side of the frame's."""
    mask, _lanes = _straight_lanes()
    frame = road_frame(mask)
    assert frame is not None
    vehicles = np.zeros((HEIGHT, WIDTH), dtype=bool)
    for top in (70, 100, 130, 160, 190):
        for left in (75, 110, 145, 180, 215):
            vehicles[top : top + 18, left : left + 14] = True
    means, spreads = [], []
    for seed in range(40):
        traffic = road_vehicles(frame, vehicles, np.random.default_rng(seed), speed_kmh=80.0)
        speeds = np.array([vehicle.speed_kmh for vehicle in traffic])
        assert len(speeds) == 25 and (speeds >= 0.0).all()
        means.append(float(speeds.mean()))
        spreads.append(float(speeds.std() / speeds.mean()))
    assert 74.0 < float(np.mean(means)) < 86.0
    assert 2.0 < float(np.std(means)) < 12.0, "every road at the frame's own speed"
    assert 0.09 < float(np.median(spreads)) < 0.2
    # A lorry, twice as wide as the cars about it, on a fast road.
    vehicles[100:118, 138:166] = True
    for seed in range(10):
        traffic = road_vehicles(frame, vehicles, np.random.default_rng(seed), speed_kmh=130.0)
        widest = max(traffic, key=lambda vehicle: vehicle.girth)
        assert widest.speed_kmh <= 90.0
        assert max(vehicle.speed_kmh for vehicle in traffic) > 100.0


def test_a_wake_covers_what_follows_and_is_hidden_by_what_stands_nearer() -> None:
    """Spray is a thing in the scene. The wake of a car coming on lies up the
    road behind it, over the wheels and the bonnet of the car that follows,
    and that car is not a clear hole in it; the car that threw it stands in
    front of it. And a branch much nearer the camera than the cars hides it."""
    mask, _lanes = _straight_lanes()
    frame = road_frame(mask)
    assert frame is not None
    vehicles = np.zeros((HEIGHT, WIDTH), dtype=bool)
    scene = np.full((HEIGHT, WIDTH, 3), 0.1, dtype=np.float32)
    for top in (95, 160):
        vehicles[top : top + 30, 147:173] = True
        scene[top : top + 30, 147:173] = (0.3, 0.32, 0.35)
    rows = np.arange(HEIGHT, dtype=np.float32)[:, None]
    depth = np.broadcast_to(20.0 + 2.0 * (HEIGHT - 1 - rows), (HEIGHT, WIDTH)).astype(np.float32)
    branch = depth.copy()
    branch[130:150, 100:220] = 5.0
    covered = 0
    for seed in range(8):
        rng = np.random.default_rng(seed)
        traffic = road_vehicles(frame, vehicles, rng, scene, 110.0)
        assert all(vehicle.away == -1.0 for vehicle in traffic)
        spray = vehicle_spray(frame, vehicles, np.random.default_rng(seed), scene, traffic, depth)
        own = float(spray[170:185, 152:168].mean())
        follower = float(spray[114:124, 152:168].mean())
        between = float(spray[132:150, 152:168].mean())
        assert own < 0.1 * between, "a car hidden by its own wake, which is behind it"
        covered += follower > 0.25 * between
        hidden = vehicle_spray(frame, vehicles, np.random.default_rng(seed), scene, traffic, branch)
        assert float(hidden[130:150, 100:220].max()) == 0.0
        assert float(hidden[60:90].sum()) > 0.0
    assert covered >= 6, "the car that follows stands in a clear hole in the wake it drives through"


def test_no_mist_hangs_over_a_queue() -> None:
    """The mist over a road is what its traffic throws up, and standing
    traffic throws up nothing."""
    mask, _lanes = _straight_lanes()
    frame = road_frame(mask)
    assert frame is not None
    rows = np.arange(HEIGHT, dtype=np.float32)[:, None]
    depth = np.broadcast_to(20.0 + 2.0 * (HEIGHT - 1 - rows), (HEIGHT, WIDTH)).astype(np.float32)
    vehicles = np.zeros((HEIGHT, WIDTH), dtype=bool)
    for top, left in ((30, 80), (60, 180), (100, 120), (150, 200), (190, 90), (120, 220)):
        vehicles[top : top + 24, left : left + 20] = True
    rng = np.random.default_rng(0)
    queue = road_vehicles(frame, vehicles, rng, speed_kmh=0.0)
    slow = road_vehicles(frame, vehicles, rng, speed_kmh=50.0)
    fast = road_vehicles(frame, vehicles, rng, speed_kmh=120.0)
    assert float(road_mist(frame, vehicles, depth, queue).max()) == 0.0
    assert (
        0.0
        < float(road_mist(frame, vehicles, depth, slow).sum())
        < 0.3 * float(road_mist(frame, vehicles, depth, fast).sum())
    )


def test_lamps_are_lit_by_which_end_of_a_vehicle_is_seen() -> None:
    """Two headlamps on a vehicle coming on and two tail lamps on one driving
    away, on the lower part of its outline, either side of its middle, far
    over full scale; with a glow wider than the lamp, and a weight of where
    they are for the road to reflect."""
    frame, vehicles, scene = _two_carriageways()
    lit_fronts = lit_rears = 0
    for seed in range(40):
        rng = np.random.default_rng(seed)
        traffic = road_vehicles(frame, vehicles, rng, scene)
        light, glow, cores = vehicle_lamps(frame, traffic, rng, 1.0, scene)
        assert light.shape == glow.shape == (HEIGHT, WIDTH, 3) and cores.shape == (HEIGHT, WIDTH)
        assert float(cores.max()) <= 1.0 and float(light.min()) >= 0.0
        assert float(light[~vehicles].max(initial=0.0)) < 0.5 * float(light.max()), (
            "lamps off the cars"
        )
        for left, top, rear in (
            (80, 100, False),
            (100, 170, False),
            (200, 100, True),
            (215, 170, True),
        ):
            car = light[top : top + 30, left : left + 26]
            if float(car.max()) == 0.0:
                continue
            # On the lower part of it, and one either side.
            assert float(car[:12].max()) < 0.05 * float(car.max())
            assert float(car[:, :10].max()) > 1.0 and float(car[:, 16:].max()) > 1.0
            assert float(car[:, 11:15].max()) < 0.5 * float(car.max())
            red, green, _blue = (float(car[..., channel].sum()) for channel in range(3))
            if rear:
                lit_rears += 1
                assert green < 0.1 * red, "a tail lamp that is not red"
            else:
                lit_fronts += 1
                assert green > 0.8 * red, "a headlamp that is not white"
            halo = glow[top : top + 30, left - 6 : left + 32]
            assert float(halo.max()) < 0.2 * float(car.max())
            assert int((halo.max(axis=2) > 0.2 * halo.max()).sum()) > int(
                (car.max(axis=2) > 0.2 * car.max()).sum()
            )
    # Most are lit by day, not all, and fewer rears than fronts.
    assert 48 <= lit_rears < lit_fronts <= 80


def test_a_lit_rear_shows_its_lamps_where_its_lenses_are() -> None:
    """Lenses that can be seen are lit where they are, wherever on the
    outline that is."""
    mask, _lanes = _straight_lanes()
    frame = road_frame(mask)
    assert frame is not None
    vehicles = np.zeros((HEIGHT, WIDTH), dtype=bool)
    scene = np.full((HEIGHT, WIDTH, 3), 0.1, dtype=np.float32)
    vehicles[120:150, 147:173] = True
    scene[120:150, 147:173] = (0.3, 0.32, 0.35)
    scene[133:136, 148:152] = scene[133:136, 168:172] = (0.6, 0.04, 0.04)
    for seed in range(20):
        rng = np.random.default_rng(seed)
        traffic = road_vehicles(frame, vehicles, rng, scene)
        light, _glow, _cores = vehicle_lamps(frame, traffic, rng, 1.0, scene)
        if float(light.max()) == 0.0:
            continue
        rows, columns = np.nonzero(light[..., 0] > 0.9 * light[..., 0].max())
        assert 132 <= int(rows.min()) and int(rows.max()) <= 136
        return
    raise AssertionError("no lamp lit in twenty draws")


def test_rain_lights_the_traffic_s_lamps_only_when_asked_and_the_road_reflects_them() -> None:
    plain, drew = _rained()
    assert drew["vehicle_lamps"] == 0.0 and not drew["lamps_lit"]
    lit, drew = _rained(vehicle_lamp_range=(1.0, 1.0))
    assert drew["vehicle_lamps"] == 1.0 and drew["lamps_lit"]
    change = lit - plain
    on_cars = np.zeros((HEIGHT, WIDTH), dtype=bool)
    on_cars[150:180, 96:124] = on_cars[90:120, 196:224] = True
    assert float(change[on_cars].max()) > 100.0, "no lamp was lit on either car"
    assert float(np.abs(change[:60]).max()) < 2.0, "lamps in the sky"
    # Down the wet road below a lit car there is more light than there was.
    below = change[181:215, 96:124].mean() + change[121:150, 196:224].mean()
    assert float(below) > 0.3
    # Without labels there is no traffic to light.
    _out, drew = _rained(labelled=False, vehicle_lamp_range=(1.0, 1.0))
    assert not drew["lamps_lit"]


def test_the_traffic_s_speed_is_drawn_from_its_range() -> None:
    """Per call, and what it is drawn as is what the spray is made with."""
    speeds = []
    for seed in range(8):
        _out, drew = _rained(
            seed=seed, spray_range=(0.8, 0.8), traffic_speed_range_kmh=(30.0, 60.0)
        )
        speeds.append(drew["traffic_speed_kmh"])
        assert 30.0 <= drew["traffic_speed_kmh"] <= 60.0
    assert len({round(speed, 3) for speed in speeds}) > 4
    fast, _ = _rained(spray_range=(0.8, 0.8), traffic_speed_range_kmh=(120.0, 120.0))
    slow, _ = _rained(spray_range=(0.8, 0.8), traffic_speed_range_kmh=(30.0, 30.0))
    still, _ = _rained(spray_range=(0.8, 0.8), traffic_speed_range_kmh=(0.0, 0.0))
    none, _ = _rained()
    assert float(np.abs(still - none).max()) < 1.0, "spray over a standing queue"
    assert float(np.abs(fast - none).mean()) > 3.0 * float(np.abs(slow - none).mean()) > 0.0


def test_rain_is_drawn_in_front_of_the_wet_road_and_not_wetted_with_it() -> None:
    """The streaks are the drops near the lens, and the road is behind them.
    Against a road the rain has darkened they stand out the more. Drawn
    before the road was wetted, as they used to be, they were darkened and
    mirrored with it, and showed less on the road than on the grass beside
    it."""

    def linear(image: np.ndarray) -> np.ndarray:
        """An 8-bit frame's luminance in linear light, near enough."""
        return ((image / 255.0) ** 2.2).mean(axis=2)

    on_road = on_grass = 0.0
    for seed in range(4):
        settings = {
            "seed": seed,
            "rain_rate_range_mm_h": (20.0, 20.0),
            "puddle_coverage_range": (0.0, 0.0),
        }
        streaked, _ = _rained(max_streaks=16000, **settings)
        bare, _ = _rained(**settings)
        lift = linear(streaked) - linear(bare)
        on_road += float(np.percentile(lift[130:230, 80:240], 99.0))
        beside = np.concatenate([lift[130:230, :50].ravel(), lift[130:230, 270:].ravel()])
        on_grass += float(np.percentile(beside, 99.0))
    assert on_grass > 0.0, "no rain was drawn"
    assert on_road > 1.2 * on_grass


def test_rain_is_drawn_in_front_of_spray() -> None:
    """Spray is in the scene and the rain is nearer than the scene. A mist
    thick enough to hide everything behind it leaves the streaks in front
    of it showing; laid over them, as it used to be, it painted them out and
    left a blank frame."""
    labels = np.full((HEIGHT, WIDTH), int(SceneClass.GRASS), dtype=np.uint8)
    labels[:60] = int(SceneClass.SKY)
    labels[60:, 60:260] = int(SceneClass.ROAD)
    image = np.full((HEIGHT, WIDTH, 3), 90, dtype=np.uint8)
    image[:60] = (150, 180, 230)
    provider = SharedDepthProvider(_Level(), depth_range_m=(4.0, 40.0), labels=_Labels(labels))
    rain = RandomRain(
        depth_provider=provider,
        p=1.0,
        wetness_probability=1.0,
        rain_rate_range_mm_h=(20.0, 20.0),
        camera_pitch_range_deg=(35.0, 35.0),
    )
    rain.set_random_seed(0)
    params = rain.get_params_dependent_on_data({"shape": image.shape}, {"image": image})
    everywhere = np.ones((HEIGHT, WIDTH), dtype=np.float32)
    hidden = rain.apply(image, **{**params, "spray": everywhere}).astype(np.float32)
    no_rain = {name: values[:0] for name, values in params["streaks"].items()}
    blank = rain.apply(image, **{**params, "spray": everywhere, "streaks": no_rain})
    flat = blank.astype(np.float32).std(axis=(0, 1))
    assert float(flat.max()) < 0.5, "the mist does not hide the scene"
    spread = hidden.max(axis=(0, 1)) - hidden.min(axis=(0, 1))
    assert float(spread.max()) > 4.0, "the mist hid the rain in front of it"


# ----------------------------------------------------------- wakes laid on the ground

# A camera eight metres up, looking twenty degrees down, through a lens of
# 300 pixels: its ground in its own frame, x to the right, y down the
# picture, z along the lens.
_FOCAL = 300.0
_MAST_M = 8.0
_TILT = np.deg2rad(20.0)
_UP = np.array([0.0, -np.cos(_TILT), -np.sin(_TILT)])
_AHEAD = np.array([0.0, -np.sin(_TILT), np.cos(_TILT)])
_RIGHT = np.array([1.0, 0.0, 0.0])
_GROUND = GroundView(-_MAST_M * _UP, _UP, _FOCAL)


def _seen_at(point: np.ndarray) -> tuple[float, float]:
    """The column and row the mast's camera sees a point at."""
    return (
        _FOCAL * float(point[0] / point[2]) + 0.5 * WIDTH,
        _FOCAL * float(point[1] / point[2]) + 0.5 * HEIGHT,
    )


def _on_the_ground(aside_m: float, ahead_m: float) -> np.ndarray:
    """A point of the ground, so far to the right of the mast's foot and so far ahead of it."""
    return np.asarray(-_MAST_M * _UP + aside_m * _RIGHT + ahead_m * _AHEAD)


def _boxed_car(
    aside_m: float,
    ahead_m: float,
    turned_deg: float = 0.0,
    coming: bool = False,
    lamps: bool = True,
    speed_kmh: float = 100.0,
) -> tuple[RoadVehicle, np.ndarray, np.ndarray]:
    """A car on the mast's ground as a network would give it, the place of its rear, its heading.

    A box 4.5 m long, 1.8 m wide and 1.5 m tall whose near end is at the
    place given, turned so many degrees to the right of straight ahead
    and driving that way, or towards the camera when ``coming``. Its
    outline is the box as the camera sees it, and its lamps are on the end
    that is seen.
    """
    angle = np.deg2rad(turned_deg)
    along = np.cos(angle) * _AHEAD + np.sin(angle) * _RIGHT
    across = np.cross(_UP, along)
    near = _on_the_ground(aside_m, ahead_m)
    corners = [
        near + far * along + side * across + tall * _UP
        for far in (0.0, 4.5)
        for side in (-0.9, 0.9)
        for tall in (0.0, 1.5)
    ]
    hull = cv2.convexHull(np.array([_seen_at(corner) for corner in corners], dtype=np.float32))
    outline = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    cv2.fillConvexPoly(outline, np.rint(hull).astype(np.int32), 1)
    rows, columns = np.nonzero(outline)
    found = None
    if lamps:
        places = [_seen_at(near + side * across + 0.8 * _UP) for side in (-0.7, 0.7)]
        found = np.array([(x, y, 1.0, 1.0, 0.9) for x, y in places], dtype=np.float32)
    foot_column, foot_row = _seen_at(near)
    foot = (min(int(round(foot_row)), HEIGHT - 1), int(round(foot_column)))
    away = -1.0 if coming else 1.0
    heading = -along if coming else along
    rear = near + 4.5 * along if coming else near
    vehicle = RoadVehicle(1, rows, columns, foot, 0.1, away, away, speed_kmh, found, True)
    return vehicle, rear, heading


def _sideways_road() -> RoadFrame:
    """A road over the whole frame whose own direction is said to run across the picture."""
    frame = road_frame(np.ones((HEIGHT, WIDTH), dtype=bool))
    assert frame is not None
    across = np.zeros((HEIGHT, WIDTH, 2), dtype=np.float32)
    across[..., 0] = 1.0
    return frame._replace(along=across)


def test_the_ground_of_a_scene_is_the_plane_its_depth_lies_on() -> None:
    rows, columns = np.mgrid[0:HEIGHT, 0:WIDTH]
    rays = np.stack(
        [(columns - 0.5 * WIDTH) / _FOCAL, (rows - 0.5 * HEIGHT) / _FOCAL, np.ones_like(rows)],
        axis=2,
    )
    falling = rays @ _UP
    depth = np.where(falling < -0.02, -_MAST_M / np.minimum(falling, -0.02), 400.0)
    ground = ground_view(depth.astype(np.float32), falling < -0.05, _FOCAL)
    assert ground is not None
    assert float(ground.up @ _UP) > 0.999
    assert float(ground.middle @ ground.up) == pytest.approx(-_MAST_M, abs=0.2)
    assert ground_view(depth.astype(np.float32), np.zeros((HEIGHT, WIDTH), bool), _FOCAL) is None


def test_a_vehicle_points_the_way_its_own_lamps_show_and_not_the_way_of_the_road() -> None:
    """A car off to one side that drives straight ahead is seen from behind
    and from its near side. Its heading is read off that, in metres on the
    ground, whatever the road's direction in the picture is said to be."""
    frame = _sideways_road()
    for aside, ahead, turned in ((6.0, 25.0, 0.0), (-7.0, 30.0, 0.0), (5.0, 28.0, 20.0)):
        for coming in (False, True):
            vehicle, rear, heading = _boxed_car(aside, ahead, turned, coming=coming)
            (lie,) = wake_lies(frame, [vehicle], _GROUND, (HEIGHT, WIDTH))
            assert lie is not None
            assert float(lie[1] @ heading) > np.cos(np.deg2rad(10.0)), (aside, turned, coming)
            assert abs(float(lie[1] @ _UP)) < 1e-6, "a heading off the ground"
            assert float(np.linalg.norm(lie[0] - rear)) < 1.5
            assert 0.8 < lie[2] < 1.25


def test_a_vehicle_with_no_lamps_found_takes_its_neighbour_s_heading_or_the_line_of_sight() -> None:
    frame = _sideways_road()
    read, _rear, heading = _boxed_car(6.0, 25.0)
    unread, _rear, _heading = _boxed_car(9.5, 31.0, lamps=False)
    # Beside one that was read: parallel to it, though it is further off
    # the camera's axis and its own line of sight is turned further.
    _first, second = wake_lies(frame, [read, unread], _GROUND, (HEIGHT, WIDTH))
    assert second is not None and float(second[1] @ heading) > np.cos(np.deg2rad(10.0))
    # Going the other way, it is turned round.
    oncoming = unread._replace(shows=-1.0, away=-1.0)
    _first, second = wake_lies(frame, [read, oncoming], _GROUND, (HEIGHT, WIDTH))
    assert second is not None and float(second[1] @ heading) < -np.cos(np.deg2rad(10.0))
    # Alone, with its end seen: along the line from the camera, and not
    # the road's direction, which here runs across the picture.
    (alone,) = wake_lies(frame, [unread], _GROUND, (HEIGHT, WIDTH))
    assert alone is not None
    sight = _on_the_ground(9.5, 31.0) - _on_the_ground(0.0, 0.0)
    assert float(alone[1] @ sight) / float(np.linalg.norm(sight)) > 0.999
    # Showing neither end, it has only its road to go by.
    (by_road,) = wake_lies(frame, [unread._replace(shows=0.0)], _GROUND, (HEIGHT, WIDTH))
    assert by_road is not None and abs(float(by_road[1] @ _RIGHT)) > 0.9


def _wake(vehicle: RoadVehicle, frame: RoadFrame, seed: int, ground: bool = True) -> np.ndarray:
    """One vehicle's spray, laid on the mast's ground or in the picture."""
    outline = np.zeros((HEIGHT, WIDTH), dtype=bool)
    outline[vehicle.rows, vehicle.columns] = True
    return vehicle_spray(
        frame,
        outline,
        np.random.default_rng(seed),
        traffic=[vehicle],
        ground=_GROUND if ground else None,
    )


def test_a_wake_lies_behind_its_vehicle_on_the_ground_and_not_along_the_picture_s_road() -> None:
    """The fault this is for: a car off to one side, seen from behind, with
    its spray laid sideways because the road's direction in the picture
    was wrong there."""
    frame = _sideways_road()
    for aside in (6.0, -6.0):
        vehicle, rear, heading = _boxed_car(aside, 25.0)
        behind = _seen_at(rear - 7.0 * heading + 0.5 * _UP)
        foot_row, foot_column = vehicle.foot
        for seed in range(6):
            spray = _wake(vehicle, frame, seed)
            assert float(spray.max()) > 0.3
            rows, columns = np.nonzero(spray > 0.25 * float(spray.max()))
            # Below its vehicle in the picture, which is nearer the camera,
            # and over the point of the road seven metres behind it.
            assert float(np.mean(rows > foot_row - 4)) > 0.9
            assert spray[int(behind[1]), int(behind[0])] > 0.2 * float(spray.max())
            # Not out along the picture's row, where the road was said to run.
            level = np.abs(rows - foot_row) < 6
            assert float(np.mean(level & (np.abs(columns - foot_column) > 30))) < 0.02
            # In the picture alone it goes where the road is said to go.
            flat = _wake(vehicle, frame, seed, ground=False)
            _rows, flat_columns = np.nonzero(flat > 0.25 * float(flat.max()))
            assert float(np.mean(np.abs(flat_columns - foot_column) > 30)) > 0.3


def test_a_wake_is_as_long_in_the_picture_as_its_metres_are_at_its_distance() -> None:
    """Twenty metres of wake is half the frame behind a car near by and a
    few rows behind one far off."""
    frame = _sideways_road()
    lengths = []
    for ahead in (36.0, 72.0, 144.0):
        vehicle, rear, heading = _boxed_car(0.0, ahead)
        spray = np.mean([_wake(vehicle, frame, seed) for seed in range(6)], axis=0)
        rows = np.nonzero((spray > 0.1 * float(spray.max())).any(axis=1))[0]
        lengths.append(float(np.ptp(rows)))
        # No further than its metres: 0.75 s at 100 km/h, and the reach of
        # its last puff.
        end = _seen_at(rear - 28.0 * heading)
        assert float(rows.max()) <= min(end[1], HEIGHT - 1) + 2
    assert lengths[0] > 2.0 * lengths[1] > 4.0 * lengths[2]


def test_a_vehicle_coming_on_trails_its_wake_up_the_road_behind_it() -> None:
    frame = _sideways_road()
    vehicle, rear, heading = _boxed_car(3.0, 22.0, coming=True)
    beyond = _seen_at(rear - 6.0 * heading + 0.8 * _UP)
    foot_row = vehicle.foot[0]
    for seed in range(6):
        spray = _wake(vehicle, frame, seed)
        assert float(spray[foot_row + 3 :].max(initial=0.0)) < 0.05 * float(spray.max())
        assert spray[int(beyond[1]), int(beyond[0])] > 0.0
        # Hidden by the vehicle that threw it, but for its feathered edge.
        within = cv2.erode(
            np.isin(np.arange(HEIGHT * WIDTH), vehicle.rows * WIDTH + vehicle.columns)
            .reshape(HEIGHT, WIDTH)
            .astype(np.uint8),
            np.ones((9, 9), np.uint8),
        )
        assert float(spray[within > 0].max()) < 0.2 * float(spray.max())


def test_traffic_goes_no_faster_than_the_gap_ahead_of_it_allows() -> None:
    """A frame does not say how fast its traffic goes, but it says how
    close it is: a column six metres apart is not doing a hundred."""
    frame = _sideways_road()
    lead, _rear, _heading = _boxed_car(2.0, 40.5)
    follower, _rear, _heading = _boxed_car(2.0, 30.0)
    beside, _rear, _heading = _boxed_car(-2.0, 31.0)
    oncoming, _rear, _heading = _boxed_car(-6.0, 24.0, coming=True)
    spaced = spaced_speeds(frame, [lead, follower, beside, oncoming], _GROUND, (HEIGHT, WIDTH))
    # Six metres behind the car ahead: 21.6 km/h at a second's headway.
    assert 14.0 < spaced[1].speed_kmh < 30.0
    # The car ahead moves with it, and the cars in other lanes as they did.
    assert spaced[0].speed_kmh < 1.3 * spaced[1].speed_kmh
    assert spaced[2].speed_kmh == beside.speed_kmh
    assert spaced[3].speed_kmh == oncoming.speed_kmh
    # With room ahead, the speed that was drawn.
    far_apart = spaced_speeds(
        frame, [_boxed_car(2.0, 90.0)[0], follower], _GROUND, (HEIGHT, WIDTH)
    )
    assert far_apart[1].speed_kmh == follower.speed_kmh
    # And a column throws up next to none of the spray it would at speed.
    outline = np.zeros((HEIGHT, WIDTH), dtype=bool)
    for vehicle in (lead, follower):
        outline[vehicle.rows, vehicle.columns] = True
    kwargs = {"ground": _GROUND}
    column = [lead, follower]
    fast = vehicle_spray(frame, outline, np.random.default_rng(0), traffic=column, **kwargs)
    slow = vehicle_spray(frame, outline, np.random.default_rng(0), traffic=spaced[:2], **kwargs)
    assert float(slow.sum()) < 0.1 * float(fast.sum())
