"""Depth: the estimator's contract, metric scaling, horizon recovery, and the shared provider.

Tests that need the real network need CUDA and skip without it.
"""

from unittest.mock import patch

import numpy as np
import pytest
import torch

from surmentations.depth.estimator import DepthAnythingV2Estimator, DepthEstimator
from surmentations.depth.provider import SharedDepthProvider
from surmentations.depth.scale import (
    DEFAULT_MAX_DISTANCE_METERS,
    DEFAULT_MIN_DISTANCE_METERS,
    to_pseudo_metric,
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="depth-conditioned effects require CUDA"
)


def test_no_cuda_raises_actionable_error() -> None:
    with patch("torch.cuda.is_available", return_value=False):
        with pytest.raises(RuntimeError, match="requires a CUDA GPU"):
            DepthAnythingV2Estimator()


@pytest.mark.parametrize(
    ("estimator", "loader"),
    [
        ("surmentations.depth.estimator.DepthAnythingV2Estimator", "AutoImageProcessor"),
        ("surmentations.perception.SegmentationLabelEstimator", "AutoImageProcessor"),
        ("surmentations.perception.OpenVocabularyVehicleEstimator", "AutoProcessor"),
    ],
)
def test_a_cpu_asked_for_by_name_is_honoured(estimator: str, loader: str) -> None:
    """The documentation's gallery is rendered where there is no GPU. Asked
    for by name, the CPU gets past the guard and on to loading the network,
    which is stopped here; the default still refuses."""
    import importlib

    module, name = estimator.rsplit(".", 1)
    cls = getattr(importlib.import_module(module), name)
    with (
        patch("torch.cuda.is_available", return_value=False),
        patch(f"transformers.{loader}.from_pretrained", side_effect=LookupError("loading")),
    ):
        with pytest.raises(RuntimeError, match="requires a CUDA GPU"):
            cls()
        with pytest.raises(LookupError, match="loading"):
            cls(device="cpu")


class _FakeEstimator(DepthEstimator):
    """Counts calls so the provider's caching behavior can be verified without a GPU."""

    def __init__(self) -> None:
        """Start counting calls at zero."""
        self.call_count = 0

    def estimate(self, image: np.ndarray) -> np.ndarray:
        """A flat depth equal to the number of calls so far: a new value each time it runs."""
        self.call_count += 1
        return np.full(image.shape[:2], float(self.call_count), dtype=np.float32)


def test_shared_depth_provider_caches_by_array_identity() -> None:
    estimator = _FakeEstimator()
    provider = SharedDepthProvider(estimator)

    img = np.zeros((8, 8, 3), dtype=np.uint8)
    first = provider.get_depth(img)
    second = provider.get_depth(img)  # same object -> cache hit, no recompute
    assert estimator.call_count == 1
    np.testing.assert_array_equal(first, second)

    other_img = img.copy()  # different object -> cache miss, recompute
    third = provider.get_depth(other_img)
    assert estimator.call_count == 2
    assert not np.array_equal(first, third)


def test_to_pseudo_metric_maps_between_a_near_and_a_far_plane() -> None:
    """Relative depth spans a *range* of distances, not [0, far].

    The nearest scene point must land on the near plane. Mapping it to zero
    instead is a singularity for any inverse-square quantity — it gave the
    night-mode illuminator a 27 778x near:far irradiance ratio on a scene
    whose real span is closer to 300x.
    """
    relative = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)

    default = to_pseudo_metric(relative)
    assert default[0, 0] == pytest.approx(DEFAULT_MIN_DISTANCE_METERS)
    assert default[0, -1] == pytest.approx(DEFAULT_MAX_DISTANCE_METERS)
    # Linear in between.
    assert default[0, 1] == pytest.approx(
        0.5 * (DEFAULT_MIN_DISTANCE_METERS + DEFAULT_MAX_DISTANCE_METERS)
    )

    explicit = to_pseudo_metric(relative, depth_scale=10.0, near_distance=2.0)
    np.testing.assert_allclose(explicit, np.array([[2.0, 6.0, 10.0]]), rtol=1e-6)


def test_to_pseudo_metric_is_monotonic_and_never_degenerate() -> None:
    """Whatever the calibration, farther must mean farther and no distance may
    reach zero — every downstream formula divides by it."""
    relative = np.linspace(0.0, 1.0, 32, dtype=np.float32)
    for far, near in ((50.0, 3.0), (10.0, 0.5), (5.0, 20.0)):
        depth = to_pseudo_metric(relative, depth_scale=far, near_distance=near)
        assert np.all(np.diff(depth) >= 0.0)
        assert depth.min() > 0.0


@requires_cuda
def test_depth_anything_v2_estimator_output_contract() -> None:
    estimator = DepthAnythingV2Estimator()
    img = np.random.default_rng(0).integers(0, 256, size=(120, 160, 3), dtype=np.uint8)
    depth = estimator.estimate(img)
    assert depth.shape == (120, 160)
    assert depth.dtype == np.float32
    assert depth.min() >= 0.0
    assert depth.max() <= 1.0


@requires_cuda
def test_depth_anything_v2_gives_the_same_depth_for_float_and_uint8_input() -> None:
    """The image processor rescales by 1/255 assuming 8-bit input. A float
    image in [0, 1] went through that as-is and the network was asked for the
    depth of a black frame: its answer correlated at -0.16 with the depth of
    the same picture supplied as uint8."""
    estimator = DepthAnythingV2Estimator()
    rng = np.random.default_rng(0)
    img = np.zeros((160, 224, 3), dtype=np.uint8)
    img[:60] = (190, 205, 225)
    img[60:] = (80, 84, 76)
    img[90:130, 60:140] = (150, 40, 35)
    img = np.clip(img.astype(np.int16) + rng.integers(-10, 11, img.shape), 0, 255).astype(np.uint8)

    from_uint8 = estimator.estimate(img)
    from_float = estimator.estimate(img.astype(np.float32) / 255.0)
    np.testing.assert_allclose(from_float, from_uint8, atol=1e-4)


def test_two_callers_sharing_one_provider_see_consistent_depth() -> None:
    """Simulates two depth-conditioned transforms chained in one `Compose` call,
    sharing a provider (mirrors the sequence-diagram contract in plan.md §2.3).
    A full integration test with real RandomFog/RandomRain transforms is added
    once those land in Phase 3/4."""
    estimator = _FakeEstimator()
    provider = SharedDepthProvider(estimator)

    img = np.zeros((8, 8, 3), dtype=np.uint8)
    depth_seen_by_first = provider.get_depth(img)
    depth_seen_by_second = provider.get_depth(img)

    assert estimator.call_count == 1
    np.testing.assert_array_equal(depth_seen_by_first, depth_seen_by_second)


def test_horizon_row_recovers_a_known_ground_plane() -> None:
    """Inverse depth is linear in the row index on a ground plane and crosses
    zero at the horizon, so the horizon is recoverable without knowing the
    camera height, the focal length or the pitch.

    Exactness matters here: the wet-road reflection mirrors the scene about
    this row, so an error of a few rows puts the reflection of a car
    somewhere the car is not.
    """
    from surmentations.depth.normals import horizon_row

    height = width = 240
    rows = np.arange(height, dtype=np.float64).reshape(height, 1)
    for focal, camera_height in ((300.0, 1.5), (300.0, 6.0), (420.0, 4.0)):
        for true_horizon in (-200.0, -60.0, 20.0, 90.0):
            far = np.where(
                rows > true_horizon + 1.0,
                camera_height * focal / np.maximum(rows - true_horizon, 1e-3),
                1e6,
            )
            depth = np.broadcast_to(np.minimum(far, 1e6), (height, width)).astype(np.float32)
            ground = (depth < 500.0).astype(np.float32)
            assert horizon_row(depth, ground) == pytest.approx(true_horizon, abs=0.05)


def test_horizon_row_falls_back_rather_than_returning_nonsense() -> None:
    """A mask that is not a ground plane must not produce a confident answer."""
    from surmentations.depth.normals import horizon_row

    height = width = 64
    flat = np.full((height, width), 10.0, dtype=np.float32)
    # No slope in inverse depth: a wall, not a floor.
    assert horizon_row(flat, np.ones_like(flat)) <= 0.0
    # Almost nothing masked as ground.
    assert horizon_row(flat, np.zeros_like(flat)) == -float(height)


def test_horizon_row_is_stable_against_a_distant_flat_plane() -> None:
    """A sky is a flat far plane whose normal points straight at the camera.

    ``n . up`` there is ``sin(pitch)``, so a lenient ground threshold starts
    admitting the sky at around 18 degrees of downward tilt. On the reference
    frame that moved the fitted horizon from row 120 to row 49 — which would
    mirror the wet-road reflection about the wrong line — so the fit has to be
    given the same falloff-weighted mask the renderer uses, not a raw cosine.
    """
    from surmentations.depth.normals import horizon_row, upward_facing

    height = width = 200
    focal, camera_height, true_horizon = 250.0, 4.0, 40.0
    rows = np.arange(height, dtype=np.float64).reshape(height, 1)
    ground = camera_height * focal / np.maximum(rows - true_horizon, 1e-3)
    depth = np.where(rows > true_horizon + 2, np.minimum(ground, 400.0), 400.0)
    depth = np.broadcast_to(depth, (height, width)).astype(np.float32)
    fov = float(np.rad2deg(2 * np.arctan(width / (2 * focal))))

    estimates = []
    for pitch in (0.0, 10.0, 18.0, 26.0):
        cosine = upward_facing(depth, horizontal_fov_deg=fov, camera_pitch_deg=pitch)
        weighted = cosine**6.5
        estimates.append(horizon_row(depth, weighted, min_weight=0.05))

    spread = max(estimates) - min(estimates)
    assert spread < 8.0, f"horizon moved {spread:.1f} rows across the pitch range: {estimates}"
    assert all(abs(e - true_horizon) < 12.0 for e in estimates), estimates


def _box_on_a_road(softness_px: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """A level camera over a ground plane with an upright box standing on it.

    Returns the depth, a mask of the road just *outside* the box's top and
    sides, a mask of the box's face, and the field of view. ``softness_px``
    blurs the depth the way a monocular estimator's upsampled output is soft.
    """
    import cv2

    height = width = 240
    fov = 60.0
    focal = width / (2.0 * np.tan(np.deg2rad(fov) / 2.0))
    cy = height / 2.0
    rows = np.arange(height, dtype=np.float32).reshape(height, 1)
    ground = np.minimum(1.5 * focal / np.maximum(rows - cy, 1e-3), 50.0)
    depth = np.broadcast_to(np.where(rows > cy + 4, ground, 50.0), (height, width)).copy()
    top, bottom, left, right = 150, 200, 90, 150
    depth[top:bottom, left:right] = depth[bottom, 0]
    if softness_px > 0:
        depth = cv2.GaussianBlur(depth, (0, 0), sigmaX=softness_px)

    beside = np.zeros((height, width), dtype=bool)
    beside[top : bottom - 9, left - 9 : left - 2] = True
    beside[top : bottom - 9, right + 2 : right + 9] = True
    beside[top - 9 : top - 2, left:right] = True
    face = np.zeros((height, width), dtype=bool)
    face[top + 8 : bottom - 8, left + 8 : right - 8] = True
    return depth.astype(np.float32), beside, face, fov


def test_ground_reaches_up_to_an_object_instead_of_stopping_short_of_it() -> None:
    """The road beside a car is road.

    Differencing a blurred depth map across a silhouette invents a sloping
    surface joining the object to what is behind it, several pixels wide and
    not horizontal — so the wet-road darkening stopped short of every car and
    left it in a dry halo. On this scene the weight in the ring beside the box
    averaged 0.71 with a fifth percentile of 0.01; it should be 1.
    """
    from surmentations.depth.normals import upward_facing

    depth, beside, face, fov = _box_on_a_road(softness_px=1.0)
    # The exponent the wet-road pass applies; it is what turns a slightly
    # wrong normal into a visibly dry pixel.
    wet = upward_facing(depth, horizontal_fov_deg=fov) ** 6.5

    # Not 1: the depth edge is soft, and the pixels on it belong to neither
    # surface. But the weakest part of the ring went from dry to mostly wet.
    assert wet[beside].mean() > 0.76, f"road beside the box reads {wet[beside].mean():.2f}"
    assert np.percentile(wet[beside], 5) > 0.2
    assert wet[face].mean() < 0.05, "an upright face is not ground"
    assert wet[210:230, 10:60].mean() > 0.97, "open road must be untouched"


@pytest.mark.parametrize("softness_px", [0.0, 1.0, 2.0])
def test_a_silhouette_against_the_sky_is_not_a_surface(softness_px: float) -> None:
    """A treeline is an edge, not a slope.

    Where the far side of a depth jump is *above* the near one, the invented
    surface recedes upward exactly as ground does, so it collected snow: a
    bright rim 16-18 rows deep along the top of every tree and roof in frame.
    """
    import cv2

    from surmentations.depth.normals import upward_facing

    height = width = 200
    depth = np.full((height, width), 50.0, dtype=np.float32)
    depth[90:] = 12.0  # a wall of foliage below open sky; no ground anywhere
    if softness_px > 0:
        depth = cv2.GaussianBlur(depth, (0, 0), sigmaX=softness_px)

    # Snow's own exponent, with the camera pitched down as it usually is.
    snow = upward_facing(depth, horizontal_fov_deg=60.0, camera_pitch_deg=8.0) ** 3.5
    rim_rows = int((snow.mean(axis=1) > 0.5).sum())
    assert rim_rows <= 4, f"{rim_rows} rows of the silhouette read as upward-facing"


def test_surface_normals_of_planes_hold_up_to_the_frame_border() -> None:
    """One-sided differences must not cost accuracy where there is no edge,
    and the frame border — where only one neighbour exists — is where a
    difference scheme usually goes wrong."""
    from surmentations.depth.normals import surface_normals, upward_facing

    height, width, fov = 120, 160, 60.0
    focal = width / (2.0 * np.tan(np.deg2rad(fov) / 2.0))
    rows = np.arange(height, dtype=np.float32).reshape(height, 1)

    wall = np.full((height, width), 7.0, dtype=np.float32)
    normals = surface_normals(wall, fov)
    np.testing.assert_allclose(normals[..., 2], -1.0, atol=1e-4)
    assert upward_facing(wall, fov).max() < 1e-3

    # Ground seen by a camera pitched down far enough that its horizon is 80
    # rows above the frame, so the whole image is ground. On such a plane
    # depth goes as 1 / (row - horizon), and the pitch is the angle the
    # horizon subtends from the principal point.
    horizon = -80.0
    ground = np.broadcast_to(2.0 * focal / (rows - horizon), (height, width)).astype(np.float32)
    pitch = float(np.rad2deg(np.arctan((height / 2.0 - horizon) / focal)))
    facing = upward_facing(ground, fov, camera_pitch_deg=pitch)
    assert facing[3:-3, 3:-3].min() > 0.99, f"ground reads {facing[3:-3, 3:-3].min():.3f} inside"
    # The outermost rows see the depth through a reflected border, which the
    # one-pixel smoothing turns into a slight tilt: a few percent, not a seam.
    assert facing.min() > 0.97, f"ground plane reads {facing.min():.3f} at the frame border"

    for tiny in ((1, 9), (9, 1), (2, 2), (3, 5)):
        assert upward_facing(np.full(tiny, 5.0, dtype=np.float32)).shape == tiny
