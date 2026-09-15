from unittest.mock import patch

import numpy as np
import pytest
import torch

from weathermentations.depth.estimator import DepthAnythingV2Estimator, DepthEstimator
from weathermentations.depth.provider import SharedDepthProvider
from weathermentations.depth.scale import (
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


class _FakeEstimator(DepthEstimator):
    """Counts calls so the provider's caching behavior can be verified without a GPU."""

    def __init__(self) -> None:
        self.call_count = 0

    def estimate(self, image: np.ndarray) -> np.ndarray:
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
    from weathermentations.depth.normals import horizon_row

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
    from weathermentations.depth.normals import horizon_row

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
    from weathermentations.depth.normals import horizon_row, upward_facing

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
