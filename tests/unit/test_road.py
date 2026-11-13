"""A road's own coordinates, its lanes, and the wheel tracks drawn along them.

``surmentations.utils.road`` turns a mask of where the road is into a
position across the road for each pixel of it. The masks here are drawn
from roads whose sides are known, so that what comes back can be checked
against the lines it should follow.
"""

from __future__ import annotations

import numpy as np
import pytest

from surmentations.atmosphere import RandomSnow
from surmentations.depth.estimator import DepthEstimator
from surmentations.depth.provider import SharedDepthProvider
from surmentations.diagnostics import recording
from surmentations.perception import SceneClass, SceneLabelEstimator
from surmentations.utils.road import Lanes, road_frame, road_lanes, wheel_tracks

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


def test_the_lanes_are_where_the_vehicles_are() -> None:
    """Three lanes of 80 pixels at the bottom row, a car in each a good way
    up the road: the comb of lanes lands on them, and is as wide as 1.7 cars."""
    mask = _fan(40.0, 280.0)
    frame = road_frame(mask)
    assert frame is not None
    feet = [(80.0, 150), (160.0, 180), (240.0, 210), (160.0, 120), (80.0, 220)]
    lanes = road_lanes(frame, _vehicles(mask, feet, 47.0), np.random.default_rng(0))
    assert float(lanes.width[1]) == pytest.approx(80.0 / 240.0, rel=0.15)
    for foot in (80.0, 160.0, 240.0):
        position = (foot - 40.0) / 240.0
        turns = (position - lanes.centre[1]) / lanes.width[1]
        assert abs(turns - round(turns)) < 0.12, f"no lane under the car at {foot}"
    assert lanes.first[1] < 0.15 and lanes.last[1] > 0.85


def test_a_road_with_no_vehicles_gets_a_drawn_number_of_lanes() -> None:
    frame = road_frame(_fan(40.0, 280.0))
    assert frame is not None
    counts = set()
    for seed in range(12):
        lanes = road_lanes(frame, None, np.random.default_rng(seed), lane_count_range=(2, 4))
        counts.add(round(0.9 / float(lanes.width[1])))
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
    assert lanes.first[1] > lanes.last[1]
    tracks = wheel_tracks(frame, lanes, 0.1, np.random.default_rng(0))
    assert float(tracks.max()) == 0.0
    standing = np.zeros_like(mask)
    standing[60:100, 120:136] = True
    assert road_lanes(frame, standing, np.random.default_rng(0)).first[1] < 0.5


# ------------------------------------------------------------------------ the tracks


def _straight_lanes() -> tuple[np.ndarray, Lanes]:
    """A road up the middle of the frame, 200 pixels wide, in two lanes."""
    mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
    mask[:, 60:260] = True
    lanes = Lanes(
        centre=np.array([0.0, 0.25]),
        width=np.array([1.0, 0.5]),
        first=np.array([0.0, 0.02]),
        last=np.array([1.0, 0.98]),
    )
    return mask, lanes


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
    lanes = Lanes(np.array([0.0, 0.1]), np.array([1.0, 0.2]), np.zeros(2), np.ones(2))
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
    assert float(worn[road].min()) < 0.6 * float(untouched[road].mean())
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
