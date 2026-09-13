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
        veil_beta_range=(0.0, 0.0),
    )
    out = t(image=scene)["image"]
    np.testing.assert_allclose(out.astype(np.float32), scene.astype(np.float32), atol=1.0)


def test_flake_sizes_follow_a_volume_distribution(
    provider: SharedDepthProvider, scene: np.ndarray
) -> None:
    """Flakes hang in the air in front of the scene, so their sizes come from
    a volume-uniform distance distribution: mostly small/distant with a few
    large near ones. The median must sit well below the maximum."""
    t = RandomSnow(depth_provider=provider, p=1.0, flake_density_range=(2000, 2000))
    params = t.get_params_dependent_on_data({"shape": scene.shape}, {"image": scene})
    sizes = np.array([size for _x, _y, size, _near in params["flakes"]])
    assert np.median(sizes) < 0.35 * sizes.max()


def test_flake_size_is_independent_of_scene_depth(
    provider: SharedDepthProvider, scene: np.ndarray
) -> None:
    """A flake's size must not track the depth of whatever is *behind* it —
    tying the two together makes flakes look pasted onto surfaces. With a
    depth map that ramps top-to-bottom, flake size must show no such ramp."""
    t = RandomSnow(depth_provider=provider, p=1.0, flake_density_range=(3000, 3000))
    t.set_random_seed(0)
    params = t.get_params_dependent_on_data({"shape": scene.shape}, {"image": scene})
    ys = np.array([y for _x, y, _size, _near in params["flakes"]])
    sizes = np.array([size for _x, _y, size, _near in params["flakes"]])

    top = sizes[ys < scene.shape[0] * 0.25]
    bottom = sizes[ys > scene.shape[0] * 0.75]
    assert len(top) > 50 and len(bottom) > 50
    # Same underlying distribution, so the medians must be close.
    assert abs(np.median(top) - np.median(bottom)) < 0.25 * np.median(sizes)


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
        veil_beta_range=(0.0, 0.0),
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
        veil_beta_range=(0.0, 0.0),
    )
    for seed in range(10):
        t.set_random_seed(seed)
        out = t(image=scene)["image"]
        # Not bit-exact: compositing happens in linear light, and the
        # sRGB -> linear -> sRGB round trip costs up to one 8-bit level.
        np.testing.assert_allclose(out.astype(np.float32), scene.astype(np.float32), atol=1.0)
