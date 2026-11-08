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
    Lanes,
    painted_lines,
    read_road,
    road_frame,
    road_lanes,
    straighten,
    vehicle_spray,
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


def test_paint_that_fits_no_even_spacing_is_not_lane_lines() -> None:
    mask = _fan(30.0, 290.0)
    frame = road_frame(mask)
    assert frame is not None
    scene = _painted_road((40.0, 95.0, 118.0, 201.0, 280.0))
    paint = painted_lines(scene, frame, mask, None)
    lanes = road_lanes(frame, None, np.random.default_rng(0), paint=paint)
    assert lanes.painted[1] == 0 and len(lanes.edges[1]) >= 3


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
        if above + below == 0.0:
            seen.add("none")
            continue
        seen.add("towards" if below > above else "away")
        assert min(above, below) < 0.05 * max(above, below), "a plume on both sides of a car"
        # Not out to the sides, and not on the far side of the frame.
        assert float(spray[:, :100].max()) < 0.02 and float(spray[:, 220:].max()) < 0.02
        if below > above:
            assert float(spray[131:140, 150:170].mean()) > float(spray[180:190, 150:170].mean())
        else:
            # Hidden by the vehicle that threw it.
            assert float(spray[105:125, 150:170].max()) < 0.05
    assert seen == {"towards", "away", "none"}, "the direction and the speed are not drawn"


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
    """Not a jet: at its tail it is about the vehicle's width, and it does
    not reach far beyond a length and a half."""
    mask, _lanes = _straight_lanes()
    frame = road_frame(mask)
    assert frame is not None
    vehicles = np.zeros((HEIGHT, WIDTH), dtype=bool)
    vehicles[90:130, 148:172] = True
    for seed in range(40):
        spray = vehicle_spray(frame, vehicles, np.random.default_rng(seed))
        if float(spray[131:].sum()) < 1.0:
            continue
        row = spray[136]
        wide = int((row > 0.5 * row.max()).sum())
        assert 14 <= wide <= 44, f"a plume {wide} pixels wide behind a car of 24"
        assert float(spray[215:].max()) < 0.1 * float(spray.max())
        return
    raise AssertionError("no plume was thrown towards the camera in forty draws")
