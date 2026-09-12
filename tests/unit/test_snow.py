import numpy as np
import pytest

from weathermentations.atmosphere.snow import RandomSnow
from weathermentations.depth.estimator import DepthEstimator
from weathermentations.depth.provider import SharedDepthProvider


class _LinearDepthEstimator(DepthEstimator):
    """Depth increases linearly from 0 (top row) to 1 (bottom row)."""

    def estimate(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        return np.tile(np.linspace(0.0, 1.0, height)[:, None], (1, width)).astype(np.float32)


@pytest.fixture
def provider() -> SharedDepthProvider:
    return SharedDepthProvider(_LinearDepthEstimator())


@pytest.fixture
def scene() -> np.ndarray:
    img = np.zeros((80, 60, 3), dtype=np.uint8)
    img[:, :] = (40, 70, 60)
    return img


def test_output_shape_dtype_and_range(provider: SharedDepthProvider, scene: np.ndarray) -> None:
    t = RandomSnow(depth_provider=provider, p=1.0)
    out = t(image=scene)["image"]
    assert out.shape == scene.shape
    assert out.dtype == scene.dtype
    assert out.min() >= 0
    assert out.max() <= 255


def test_p_zero_is_identity(provider: SharedDepthProvider, scene: np.ndarray) -> None:
    t = RandomSnow(depth_provider=provider, p=0.0)
    out = t(image=scene)["image"]
    np.testing.assert_array_equal(out, scene)


def test_zero_density_no_accumulation_is_identity(
    provider: SharedDepthProvider, scene: np.ndarray
) -> None:
    t = RandomSnow(
        depth_provider=provider,
        p=1.0,
        flake_density_range=(0, 0),
        accumulation_probability=0.0,
    )
    out = t(image=scene)["image"]
    np.testing.assert_allclose(out.astype(np.float32), scene.astype(np.float32), atol=1.0)


def test_flake_size_scales_with_depth(provider: SharedDepthProvider, scene: np.ndarray) -> None:
    """Nearer flakes (small relative_depth, i.e. top of a _LinearDepthEstimator's
    output... wait: here row 0 has depth=0 (nearest) so flakes placed there
    should render *larger* than flakes placed near the bottom (depth=1,
    farthest)."""
    t = RandomSnow(depth_provider=provider, p=1.0, flake_size_range=(10.0, 10.0))
    params = t.get_params_dependent_on_data({"shape": scene.shape}, {"image": scene})
    flakes = params["flakes"]
    near_sizes = [size for x, y, size in flakes if y < scene.shape[0] * 0.1]
    far_sizes = [size for x, y, size in flakes if y > scene.shape[0] * 0.9]
    assert near_sizes and far_sizes
    assert np.mean(near_sizes) > np.mean(far_sizes)


def test_accumulation_brightens_ground_band(
    provider: SharedDepthProvider, scene: np.ndarray
) -> None:
    t = RandomSnow(
        depth_provider=provider,
        p=1.0,
        flake_density_range=(0, 0),
        accumulation_probability=1.0,
        accumulation_strength_range=(0.5, 0.5),
        accumulation_band_fraction=0.2,
    )
    out = t(image=scene)["image"].astype(np.float32)
    original_mean = scene.astype(np.float32).mean()
    bottom_row_mean = out[-1].mean()
    assert bottom_row_mean > original_mean


def test_accumulation_never_fires_when_probability_zero(
    provider: SharedDepthProvider, scene: np.ndarray
) -> None:
    t = RandomSnow(
        depth_provider=provider,
        p=1.0,
        flake_density_range=(0, 0),
        accumulation_probability=0.0,
    )
    for seed in range(10):
        t.set_random_seed(seed)
        out = t(image=scene)["image"]
        np.testing.assert_array_equal(out, scene)
