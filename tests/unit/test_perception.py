"""Scene labels and installation geometry, on scenes whose answers are known.

The estimators in ``surmentations.perception`` turn a depth map and a label
map into statements about the world: where the horizon is, how high the
camera hangs, how many metres a disparity is. Each is checked here against a
scene built from a camera whose geometry was chosen, so that what comes back
can be compared with what went in.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from surmentations.perception import (
    Camera,
    GroundPlaneGeometryEstimator,
    InstallationGeometry,
    SceneClass,
    class_weight,
    classes_of,
)
from surmentations.perception.geometry import (
    camera_height_from_objects,
    fit_ground_disparity,
    focal_px,
    metric_depth,
    metric_scale,
    top_of_ground,
)
from surmentations.perception.labels import (
    CLASS_OF_NAME,
    GROUND,
    NOT_GROUND,
    PAVED,
    check_labels,
)

WIDTH, HEIGHT = 320, 240


# ------------------------------------------------------------------ a scene with known answers


def _project(camera: Camera, across: float, up: float, ahead: float) -> tuple[float, float, float]:
    """Column, row and depth of a point in the world, for a camera of known geometry.

    ``across`` is metres to the right of the camera, ``up`` metres above the
    ground and ``ahead`` metres along the ground in front of it. Written out
    from the rotation, and independent of the code under test.
    """
    assert camera.pitch_deg is not None and camera.height_m is not None
    pitch = math.radians(camera.pitch_deg)
    below = camera.height_m - up
    along_axis = ahead * math.cos(pitch) + below * math.sin(pitch)
    down = -ahead * math.sin(pitch) + below * math.cos(pitch)
    focal = camera.focal_px
    column = 0.5 * camera.width + focal * across / along_axis
    row = 0.5 * camera.height + focal * down / along_axis
    return column, row, along_axis


def _ground_scene(
    camera: Camera, scale: float = 0.4, offset: float = 0.05
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Depth, disparity and labels of level road under ``camera`` with sky above it.

    The disparity is ``scale / depth + offset``: a depth network's output,
    which is the reciprocal of distance up to two numbers nobody is told.
    """
    rows, columns = np.mgrid[0 : camera.height, 0 : camera.width]
    assert camera.height_m is not None
    depth = camera.height_m * camera.ground_depth_per_metre(rows + 0.5, columns + 0.5)
    ground = np.isfinite(depth)
    disparity = np.where(ground, scale / np.where(ground, depth, 1.0), 0.0) + offset
    labels = np.where(ground, int(SceneClass.ROAD), int(SceneClass.SKY)).astype(np.uint8)
    return depth, disparity.astype(np.float32), labels


def _frame(seed: int = 0) -> np.ndarray:
    """A frame of noise: what the estimator is shown matters only to its choice of lens."""
    return np.random.default_rng(seed).integers(0, 256, (HEIGHT, WIDTH, 3), dtype=np.uint8)


# --------------------------------------------------------------------------------- labels


def test_a_dataset_s_class_names_fold_into_the_scene_classes() -> None:
    """By name, so that a checkpoint needs no table of its own: ADE20K's and
    Cityscapes' names, synonyms after a comma and letter case ignored."""
    names = ["road, route", "Sky", "sidewalk, pavement", "car, auto", "tree", "armchair", "terrain"]
    assert classes_of(names).tolist() == [
        SceneClass.ROAD,
        SceneClass.SKY,
        SceneClass.PAVEMENT,
        SceneClass.VEHICLE,
        SceneClass.FOLIAGE,
        SceneClass.OTHER,
        SceneClass.TERRAIN,
    ]
    assert classes_of(names).dtype == np.uint8


def test_a_mountainside_is_not_ground_a_camera_stands_over() -> None:
    """The datasets name mountains, hills and rock. A plane fitted to one put
    the horizon in the sky, so they are left without an opinion."""
    for name in ("mountain", "hill", "rock"):
        assert name not in CLASS_OF_NAME
    assert classes_of(["mountain, mount"]).tolist() == [SceneClass.OTHER]


def test_the_class_groups_say_what_the_effects_need() -> None:
    assert set(PAVED) < set(GROUND)
    assert not set(GROUND) & set(NOT_GROUND)
    # A lawn is ground, and holds its water in the grass: ground, not paved.
    assert SceneClass.GRASS in GROUND and SceneClass.GRASS not in PAVED
    # A wall is not vetoed as ground by name: a network that calls a road
    # seen from above "building" must not be able to dry it.
    assert SceneClass.STRUCTURE not in NOT_GROUND and SceneClass.OTHER not in NOT_GROUND


def test_class_weight_is_one_on_the_classes_and_soft_at_their_edge() -> None:
    labels = np.zeros((40, 60), dtype=np.uint8)
    labels[:, 30:] = int(SceneClass.VEHICLE)
    hard = class_weight(labels, (SceneClass.VEHICLE,))
    assert hard.dtype == np.float32 and set(np.unique(hard)) == {0.0, 1.0}
    soft = class_weight(labels, (SceneClass.VEHICLE, SceneClass.PERSON), soften_px=2.0)
    assert soft[20, 5] < 0.01 and soft[20, 55] > 0.99
    assert 0.3 < soft[20, 30] < 0.7, "the edge is a step and not a ramp"


def test_labels_of_the_wrong_shape_or_range_are_refused_by_name() -> None:
    good = np.zeros((4, 6), dtype=np.int64)
    assert check_labels(good, (4, 6), source="x").dtype == np.uint8
    with pytest.raises(ValueError, match="shape"):
        check_labels(good, (6, 4), source="MyLabels")
    with pytest.raises(ValueError, match="SceneClass"):
        check_labels(good + 150, (4, 6), source="MyLabels")


# ------------------------------------------------------------------------------- geometry


def test_a_geometry_no_camera_has_is_refused() -> None:
    for wrong in (
        {"horizontal_fov_deg": 0.0},
        {"horizontal_fov_deg": 200.0},
        {"pitch_deg": 90.0},
        {"roll_deg": -120.0},
        {"height_m": 0.0},
    ):
        with pytest.raises(ValueError):
            InstallationGeometry(**wrong)


def test_the_ground_is_where_the_camera_s_geometry_puts_it() -> None:
    """Depth of level ground at a pixel, against the same point projected the
    long way round: out along the ground, up to the camera, into the picture."""
    camera = Camera(WIDTH, HEIGHT, 62.0, 14.0, 0.0, 5.5)
    for across, ahead in ((0.0, 8.0), (-3.0, 15.0), (6.0, 40.0), (1.0, 150.0)):
        column, row, along_axis = _project(camera, across, 0.0, ahead)
        assert camera.height_m is not None
        depth = camera.height_m * camera.ground_depth_per_metre(np.array(row), np.array(column))
        assert float(depth) == pytest.approx(along_axis, rel=1e-6)
    horizon = camera.horizon_row
    assert horizon == pytest.approx(0.5 * HEIGHT - camera.focal_px * math.tan(math.radians(14.0)))
    above = camera.ground_depth_per_metre(np.array(horizon - 3.0), np.array(WIDTH / 2.0))
    assert np.isinf(above), "there is no ground above the horizon"


def test_a_rolled_camera_tilts_its_horizon() -> None:
    camera = Camera(WIDTH, HEIGHT, 60.0, 5.0, 4.0, 3.0)
    left, right = camera.horizon_rows(np.array([0.0, float(WIDTH)]))
    assert right - left == pytest.approx(WIDTH * math.tan(math.radians(4.0)))
    # Downhill to the right is positive.
    assert right > left


def test_a_view_takes_its_pitch_from_wherever_it_is_best_known() -> None:
    # Given outright.
    given = InstallationGeometry(horizontal_fov_deg=70.0, pitch_deg=20.0, horizon=0.9)
    camera = given.view(WIDTH, HEIGHT, 50.0, 5.0)
    assert camera is not None
    assert (camera.horizontal_fov_deg, camera.pitch_deg) == (70.0, 20.0)

    # From the horizon, through whatever field of view is taken: the tilt
    # that puts the horizon where the picture has it.
    seen = InstallationGeometry(horizon=0.25)
    for fov in (45.0, 60.0, 80.0):
        camera = seen.view(WIDTH, HEIGHT, fov, pitch_deg=-30.0)
        assert camera is not None and camera.horizon_row == pytest.approx(0.25 * HEIGHT)
    assert seen.fixes_tilt and given.fixes_tilt

    # A horizon in the middle of the frame is a level camera.
    level = InstallationGeometry(horizon=0.5).view(WIDTH, HEIGHT, 60.0)
    assert level is not None and level.pitch_deg == pytest.approx(0.0, abs=1e-9)

    # Nothing known: the caller's own draws.
    camera = InstallationGeometry().view(WIDTH, HEIGHT, 55.0, 12.0)
    assert camera is not None and (camera.horizontal_fov_deg, camera.pitch_deg) == (55.0, 12.0)
    assert InstallationGeometry().view(WIDTH, HEIGHT) is None, "no lens known and none proposed"


def test_a_horizon_that_is_only_a_bound_lets_a_draw_stand_unless_it_breaks_it() -> None:
    """With no sky in the frame the horizon is at a row *or above it*: the
    camera is tilted down at least that far."""
    bound = InstallationGeometry(horizon=0.2, horizon_is_bound=True)
    assert not bound.fixes_tilt
    least = math.degrees(math.atan(0.3 * HEIGHT / focal_px(WIDTH, 60.0)))
    steeper = bound.view(WIDTH, HEIGHT, 60.0, pitch_deg=55.0)
    shallower = bound.view(WIDTH, HEIGHT, 60.0, pitch_deg=2.0)
    nothing_drawn = bound.view(WIDTH, HEIGHT, 60.0)
    assert steeper is not None and steeper.pitch_deg == 55.0
    assert shallower is not None and shallower.pitch_deg == pytest.approx(least)
    assert nothing_drawn is not None and nothing_drawn.pitch_deg is None


def test_what_is_given_outranks_what_is_estimated() -> None:
    estimated = InstallationGeometry(
        horizontal_fov_deg=61.0, horizon=0.3, roll_deg=2.0, height_m=9.0
    )
    given = InstallationGeometry(pitch_deg=25.0, height_m=6.5)
    merged = given.filled_from(estimated)
    assert merged.pitch_deg == 25.0 and merged.height_m == 6.5
    assert merged.horizontal_fov_deg == 61.0, "what was left open is filled in"
    assert merged.horizon is None and merged.roll_deg == 0.0, "a given tilt keeps its own roll"
    open_tilt = InstallationGeometry(horizontal_fov_deg=90.0).filled_from(estimated)
    assert (open_tilt.horizon, open_tilt.roll_deg, open_tilt.horizontal_fov_deg) == (0.3, 2.0, 90.0)


# ------------------------------------------------------------------------- the estimator


@pytest.mark.parametrize("pitch", [-4.0, 6.0, 15.0])
@pytest.mark.parametrize("roll", [0.0, 3.0])
def test_the_horizon_and_the_roll_come_back_from_the_ground_s_disparity(
    pitch: float, roll: float
) -> None:
    """On level ground the reciprocal of distance is linear across the
    picture, and reaches the sky's on the horizon. Whatever scale and offset
    the network put on it."""
    camera = Camera(WIDTH, HEIGHT, 60.0, pitch, roll, 6.0)
    estimator = GroundPlaneGeometryEstimator(horizontal_fov_deg=60.0, estimate_height=False)
    for scale, offset in ((0.4, 0.05), (3.0, -0.2)):
        _depth, disparity, labels = _ground_scene(camera, scale, offset)
        found = estimator.estimate(_frame(), disparity, labels)
        assert found.horizon is not None and not found.horizon_is_bound
        assert found.horizon * HEIGHT == pytest.approx(camera.horizon_row, abs=1.5)
        assert found.roll_deg == pytest.approx(roll, abs=0.3)
        view = found.view(WIDTH, HEIGHT)
        assert view is not None and view.pitch_deg == pytest.approx(pitch, abs=0.5)


def test_the_ground_is_never_above_the_horizon() -> None:
    """A monocular depth map places far ground too near, and a plane fitted
    to it meets the sky's disparity below where the road visibly still is.
    The road wins: the horizon is no lower than the top of the ground."""
    camera = Camera(WIDTH, HEIGHT, 60.0, 8.0, 0.0, 6.0)
    _depth, disparity, labels = _ground_scene(camera)
    # The whole road a little further off than it is against the sky: its
    # disparity then reaches the sky's while there is still road above.
    squeezed = np.where(labels == SceneClass.ROAD, disparity - 0.03, disparity).astype(np.float32)
    plane = fit_ground_disparity(squeezed, labels == SceneClass.ROAD)
    assert plane is not None
    across, down, constant = plane
    fitted = (0.05 - constant - 0.5 * across) / down
    top = top_of_ground(labels == SceneClass.ROAD)
    assert fitted > top + 0.02, "the fixture's fit is not biased"
    found = GroundPlaneGeometryEstimator(horizontal_fov_deg=60.0).estimate(
        _frame(), squeezed, labels
    )
    assert found.horizon == pytest.approx(top, abs=1e-6)
    assert found.horizon * HEIGHT == pytest.approx(camera.horizon_row, abs=2.0)


def test_with_no_sky_the_horizon_is_a_bound_and_no_more() -> None:
    """Nothing marks the disparity of infinity, so the farthest thing in view
    stands in for it, and the horizon is at that row or above."""
    camera = Camera(WIDTH, HEIGHT, 60.0, 30.0, 0.0, 6.0)
    _depth, disparity, labels = _ground_scene(camera)
    assert (labels == SceneClass.SKY).sum() == 0, "the fixture has sky in it"
    found = GroundPlaneGeometryEstimator(horizontal_fov_deg=60.0).estimate(
        _frame(), disparity, labels
    )
    assert found.horizon_is_bound and not found.fixes_tilt
    assert found.horizon is not None and found.horizon * HEIGHT >= camera.horizon_row - 1.0


def test_an_estimate_says_nothing_it_cannot_know() -> None:
    estimator = GroundPlaneGeometryEstimator(height_m=4.0)
    camera = Camera(WIDTH, HEIGHT, 60.0, 8.0, 0.0, 6.0)
    _depth, disparity, labels = _ground_scene(camera)
    for missing in ((None, labels), (disparity, None)):
        found = estimator.estimate(_frame(), *missing)
        assert found.horizon is None and found.pitch_deg is None
        assert found.height_m == 4.0 and found.horizontal_fov_deg is not None
    # No ground in view: a wall of sky.
    sky = np.full((HEIGHT, WIDTH), int(SceneClass.SKY), dtype=np.uint8)
    assert estimator.estimate(_frame(), disparity, sky).horizon is None
    # A slope that is not a tilted camera is taken as no roll.
    tilted = Camera(WIDTH, HEIGHT, 60.0, 8.0, 20.0, 6.0)
    _depth, disparity, labels = _ground_scene(tilted)
    assert estimator.estimate(_frame(), disparity, labels).roll_deg == 0.0


def test_an_unknown_lens_is_one_lens_for_one_frame() -> None:
    """Drawn from the range by the frame's own content: the same frame gets
    the same field of view in every run, and frames differ."""
    estimator = GroundPlaneGeometryEstimator(fov_range_deg=(40.0, 90.0))
    drawn = [estimator.estimate(_frame(seed), None, None).horizontal_fov_deg for seed in range(8)]
    assert all(fov is not None and 40.0 <= fov <= 90.0 for fov in drawn)
    assert len(set(drawn)) == 8
    again = GroundPlaneGeometryEstimator(fov_range_deg=(40.0, 90.0))
    assert again.estimate(_frame(3), None, None).horizontal_fov_deg == drawn[3]
    known = GroundPlaneGeometryEstimator(horizontal_fov_deg=71.0)
    assert known.estimate(_frame(3), None, None).horizontal_fov_deg == 71.0
    with pytest.raises(ValueError):
        GroundPlaneGeometryEstimator(fov_range_deg=(90.0, 40.0))


# ---------------------------------------------------------------------------- the height


def _draw_box(
    labels: np.ndarray,
    camera: Camera,
    scene_class: SceneClass,
    across: float,
    ahead: float,
    size: tuple[float, float, float],
) -> None:
    """Paint a box standing on the ground into ``labels``: its eight corners, projected."""
    tall, long, wide = size
    corners = [
        _project(camera, across + dx, dy, ahead + dz)[:2]
        for dx in (-0.5 * wide, 0.5 * wide)
        for dy in (0.0, tall)
        for dz in (0.0, long)
    ]
    hull = cv2.convexHull(np.array(corners, dtype=np.float32)).astype(np.int32)
    cv2.fillConvexPoly(labels, hull, int(scene_class))


@pytest.mark.parametrize("height_m", [3.0, 7.0, 12.0])
def test_people_of_known_height_give_the_camera_s(height_m: float) -> None:
    """A person reaches a known share of the way from their feet to the horizon."""
    camera = Camera(640, 480, 60.0, 12.0, 0.0, height_m)
    labels = np.full((480, 640), int(SceneClass.ROAD), dtype=np.uint8)
    for across, ahead in ((-4.0, 14.0), (0.0, 20.0), (5.0, 26.0), (-7.0, 32.0), (2.0, 40.0)):
        _draw_box(labels, camera, SceneClass.PERSON, across, ahead, (1.7, 0.4, 0.5))
    found = camera_height_from_objects(labels, camera)
    assert found is not None and found == pytest.approx(height_m, rel=0.15)


def test_cars_seen_from_above_give_the_height_too() -> None:
    """A car shows its roof as well as its side, so the top of its outline is
    the far edge of its roof. Read as a thing of a car's height alone it puts
    the camera far too low; tried at every heading until its width fits too,
    it does not."""
    camera = Camera(640, 480, 60.0, 16.0, 0.0, 9.0)
    labels = np.full((480, 640), int(SceneClass.ROAD), dtype=np.uint8)
    for across, ahead in ((-6.0, 16.0), (0.0, 24.0), (6.0, 30.0), (-3.0, 38.0), (4.0, 46.0)):
        _draw_box(labels, camera, SceneClass.VEHICLE, across, ahead, (1.55, 4.4, 1.8))
    found = camera_height_from_objects(labels, camera)
    assert found is not None and found == pytest.approx(9.0, rel=0.25)


def test_too_few_things_or_things_run_together_give_no_height() -> None:
    camera = Camera(640, 480, 60.0, 12.0, 0.0, 6.0)
    labels = np.full((480, 640), int(SceneClass.ROAD), dtype=np.uint8)
    _draw_box(labels, camera, SceneClass.PERSON, 0.0, 20.0, (1.7, 0.4, 0.5))
    _draw_box(labels, camera, SceneClass.PERSON, 3.0, 25.0, (1.7, 0.4, 0.5))
    assert camera_height_from_objects(labels, camera) is None
    # A queue of traffic is one long region and no car.
    labels[300:330, 20:620] = int(SceneClass.VEHICLE)
    assert camera_height_from_objects(labels, camera) is None
    unknown_pitch = Camera(640, 480, 60.0, None, 0.0, None)
    assert camera_height_from_objects(labels, unknown_pitch) is None


def test_the_estimator_finds_the_height_from_what_stands_on_the_road() -> None:
    camera = Camera(640, 480, 60.0, 12.0, 0.0, 6.0)
    _depth, disparity, labels = _ground_scene(camera)
    for across, ahead in ((-4.0, 14.0), (0.0, 20.0), (5.0, 26.0), (-7.0, 32.0)):
        _draw_box(labels, camera, SceneClass.PERSON, across, ahead, (1.7, 0.4, 0.5))
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    found = GroundPlaneGeometryEstimator(horizontal_fov_deg=60.0).estimate(frame, disparity, labels)
    assert found.height_m == pytest.approx(6.0, rel=0.15)
    told = GroundPlaneGeometryEstimator(horizontal_fov_deg=60.0, height_m=4.2)
    assert told.estimate(frame, disparity, labels).height_m == 4.2
    silent = GroundPlaneGeometryEstimator(horizontal_fov_deg=60.0, estimate_height=False)
    assert silent.estimate(frame, disparity, labels).height_m is None


# ----------------------------------------------------------------------------- the metres


def test_a_known_camera_puts_a_disparity_map_in_metres() -> None:
    """Level ground is at a known distance at every row, so the ground's
    disparity regressed on the reciprocal of that distance gives the scale,
    and the disparity of infinity with it: no sky is needed."""
    camera = Camera(WIDTH, HEIGHT, 60.0, 25.0, 0.0, 6.0)
    depth, disparity, labels = _ground_scene(camera, scale=0.7, offset=0.11)
    ground = labels == SceneClass.ROAD
    found = metric_scale(disparity, ground, camera)
    assert found is not None
    scale, offset = found
    assert scale == pytest.approx(0.7, rel=1e-3) and offset == pytest.approx(0.11, abs=1e-4)
    metres = metric_depth(disparity, scale, offset)
    near = depth < 60.0
    assert np.abs(metres[near] / depth[near] - 1.0).max() < 0.02

    # Half the height, half the distance to everything.
    lower = camera._replace(height_m=3.0)
    halved = metric_scale(disparity, ground, lower)
    assert halved is not None and halved[0] == pytest.approx(0.35, rel=1e-3)


def test_metres_are_capped_where_the_depth_map_stops_meaning_anything() -> None:
    disparity = np.array([[1.0, 0.5, 0.0101, 0.01]], dtype=np.float32)
    metres = metric_depth(disparity, 2.0, 0.01)
    assert metres[0, 0] == pytest.approx(2.0 / 0.99)
    assert metres[0, 2] == metres[0, 3] == pytest.approx(100.0 * 2.0 / 0.99, rel=1e-4)
    assert np.isfinite(metres).all()


def test_there_are_no_metres_without_a_height_a_pitch_and_some_ground() -> None:
    camera = Camera(WIDTH, HEIGHT, 60.0, 25.0, 0.0, 6.0)
    _depth, disparity, labels = _ground_scene(camera)
    ground = labels == SceneClass.ROAD
    assert metric_scale(disparity, ground, camera._replace(height_m=None)) is None
    assert metric_scale(disparity, ground, camera._replace(pitch_deg=None)) is None
    assert metric_scale(disparity, np.zeros_like(ground), camera) is None
    # Ground that gets *further* down the frame is not ground under this camera.
    assert metric_scale(disparity.max() - disparity, ground, camera) is None


# -------------------------------------------------------------------- the real backends


def test_depth_anything_hands_back_the_disparity_its_depth_was_made_from() -> None:
    """The estimator normalises the network's disparity, reciprocates it and
    rescales; `disparity` is those three steps backwards. Checked on the
    arithmetic alone, with no network loaded."""
    from surmentations.depth.estimator import DepthAnythingV2Estimator

    estimator = object.__new__(DepthAnythingV2Estimator)
    estimator.disparity_offset = 0.03
    normalised = np.random.default_rng(0).uniform(0.0, 1.0, (30, 40))
    normalised.flat[0], normalised.flat[1] = 0.0, 1.0
    depth = 1.0 / (normalised + 0.03)
    relative = ((depth - depth.min()) / (depth.max() - depth.min())).astype(np.float32)
    back = estimator.disparity(relative)
    assert back is not None and back.dtype == np.float32
    assert np.abs(back - normalised).max() < 1e-4


def test_a_depth_backend_that_cannot_give_its_disparity_says_so() -> None:
    from surmentations.depth.estimator import DepthEstimator

    class _Opaque(DepthEstimator):
        """A backend that returns depth and nothing else."""

        def estimate(self, image: np.ndarray) -> np.ndarray:
            """Flat depth."""
            return np.zeros(image.shape[:2], dtype=np.float32)

    assert _Opaque().disparity(np.zeros((4, 4), dtype=np.float32)) is None


requires_cuda = pytest.mark.skipif(
    not __import__("torch").cuda.is_available(), reason="the segmentation network needs a GPU"
)


@requires_cuda
def test_the_bundled_network_names_the_reference_scene() -> None:
    """A motorway under an open sky, with traffic and verges. Not a test of
    the network's accuracy, which is its authors' to report: a test that its
    classes arrive folded, at the frame's size, and say what is plainly there."""

    from docs.generate_examples import _base_scene
    from surmentations import mode
    from surmentations.perception import SegmentationLabelEstimator

    image = _base_scene()
    estimator = SegmentationLabelEstimator()
    labels = estimator.estimate(image)
    assert labels.shape == image.shape[:2] and labels.dtype == np.uint8
    assert int(labels.max()) <= max(SceneClass)
    share = {c: float((labels == int(c)).mean()) for c in SceneClass}
    assert share[SceneClass.SKY] > 0.25 and share[SceneClass.ROAD] > 0.2
    assert share[SceneClass.VEHICLE] > 0.03 and share[SceneClass.GRASS] > 0.05
    # The sky is the top of this frame and the road the bottom.
    assert (labels[:60] == SceneClass.SKY).mean() > 0.95
    assert (labels[-40:, 250:500] == SceneClass.ROAD).mean() > 0.8

    # A float frame is the same frame, and half precision the same answer
    # to within a sliver of the outlines.
    as_float = estimator.estimate(image.astype(np.float32) / 255.0)
    assert (as_float == labels).mean() > 0.995
    was_fast = mode.fast()
    try:
        mode.set_fast(False)
        exact = estimator.estimate(image)
        mode.set_fast(True)
        fast = estimator.estimate(image)
    finally:
        mode.set_fast(was_fast)
    assert (exact == fast).mean() > 0.99

    # A network that is unsure of everything has no opinion.
    unsure = SegmentationLabelEstimator(min_confidence=0.999)
    assert (unsure.estimate(image) == SceneClass.OTHER).mean() > 0.3


@requires_cuda
def test_a_confidence_that_is_not_a_share_is_refused() -> None:
    from surmentations.perception import SegmentationLabelEstimator

    with pytest.raises(ValueError, match="min_confidence"):
        SegmentationLabelEstimator(min_confidence=1.5)
