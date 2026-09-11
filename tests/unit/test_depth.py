from unittest.mock import patch

import numpy as np
import pytest
import torch

from weathermentations.depth.estimator import DepthAnythingV2Estimator, DepthEstimator
from weathermentations.depth.provider import SharedDepthProvider
from weathermentations.depth.scale import DEFAULT_MAX_DISTANCE_METERS, to_pseudo_metric

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


def test_to_pseudo_metric_default_and_explicit_scale() -> None:
    relative = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
    default = to_pseudo_metric(relative)
    np.testing.assert_allclose(default, relative * DEFAULT_MAX_DISTANCE_METERS)

    explicit = to_pseudo_metric(relative, depth_scale=10.0)
    np.testing.assert_allclose(explicit, relative * 10.0)


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
