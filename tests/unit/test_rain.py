import numpy as np
import pytest

from weathermentations.atmosphere.rain import RandomRain, _apply_ground_wetness
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
    img[:, :] = (40, 70, 130)
    return img


def test_output_shape_dtype_and_range(provider: SharedDepthProvider, scene: np.ndarray) -> None:
    t = RandomRain(depth_provider=provider, p=1.0)
    out = t(image=scene)["image"]
    assert out.shape == scene.shape
    assert out.dtype == scene.dtype
    assert out.min() >= 0
    assert out.max() <= 255


def test_p_zero_is_identity(provider: SharedDepthProvider, scene: np.ndarray) -> None:
    t = RandomRain(depth_provider=provider, p=0.0)
    out = t(image=scene)["image"]
    np.testing.assert_array_equal(out, scene)


def test_zero_density_and_zero_veil_beta_is_near_identity(
    provider: SharedDepthProvider, scene: np.ndarray
) -> None:
    t = RandomRain(
        depth_provider=provider,
        p=1.0,
        streak_density_range=(0, 0),
        veil_beta_range=(0.0, 0.0),
        wetness_probability=0.0,
    )
    out = t(image=scene)["image"]
    np.testing.assert_allclose(out.astype(np.float32), scene.astype(np.float32), atol=1.0)


def test_veil_transmission_monotonic_with_depth(
    provider: SharedDepthProvider, scene: np.ndarray
) -> None:
    """Same Koschmieder invariant as fog: distance to the veil color must be
    non-increasing as depth (row index here) increases."""
    t = RandomRain(
        depth_provider=provider,
        p=1.0,
        streak_density_range=(0, 0),
        veil_beta_range=(0.05, 0.05),
        wetness_probability=0.0,
    )
    out = t(image=scene)["image"].astype(np.float32)
    veil_color = np.array([185.0, 190.0, 200.0])
    dist_to_veil = np.linalg.norm(out - veil_color, axis=-1).mean(axis=1)
    assert np.all(np.diff(dist_to_veil) <= 1e-3)


def test_streak_alpha_layer_stays_in_bounds(
    provider: SharedDepthProvider, scene: np.ndarray
) -> None:
    t = RandomRain(depth_provider=provider, p=1.0, streak_density_range=(50, 50))
    params = t.get_params_dependent_on_data({"shape": scene.shape}, {"image": scene})
    assert params["veil_transmission"].min() >= 0.0
    assert params["veil_transmission"].max() <= 1.0
    assert len(params["streaks"]) == 50


def test_ground_wetness_feathers_without_hard_seam() -> None:
    # A smooth image (not noise) so row-to-row differences reflect the effect
    # itself, not pre-existing pixel variance.
    height, width = 100, 60
    img = np.tile(np.linspace(50.0, 200.0, height)[:, None, None], (1, width, 3)).astype(np.float32)
    out = _apply_ground_wetness(img.copy(), band_fraction=0.3)
    band_start = int(height * 0.7)
    # The row just above the band and the first row inside it should not
    # differ by much more than the smooth image's own natural row-to-row step
    # (no abrupt seam from the wetness effect switching on).
    natural_step = np.abs(img[band_start] - img[band_start - 1]).mean()
    seam_diff = np.abs(out[band_start] - out[band_start - 1]).mean()
    assert seam_diff <= natural_step + 5.0
