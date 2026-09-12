import numpy as np
import pytest

from weathermentations.atmosphere.fog import RandomFog
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
    img = np.zeros((64, 48, 3), dtype=np.uint8)
    img[:, :] = (30, 60, 120)
    return img


def test_output_shape_dtype_and_range(provider: SharedDepthProvider, scene: np.ndarray) -> None:
    t = RandomFog(depth_provider=provider, p=1.0)
    out = t(image=scene)["image"]
    assert out.shape == scene.shape
    assert out.dtype == scene.dtype
    assert out.min() >= 0
    assert out.max() <= 255


def test_p_zero_is_identity(provider: SharedDepthProvider, scene: np.ndarray) -> None:
    t = RandomFog(depth_provider=provider, p=0.0)
    out = t(image=scene)["image"]
    np.testing.assert_array_equal(out, scene)


def test_monotonic_approach_to_airlight_with_depth(
    provider: SharedDepthProvider, scene: np.ndarray
) -> None:
    """Koschmieder's law invariant: as depth increases, pixel color must move
    monotonically closer to the airlight color (transmission t=exp(-beta*d)
    is itself monotonically decreasing in d)."""
    airlight = (220, 220, 220)
    t = RandomFog(
        depth_provider=provider, p=1.0, beta_range=(0.05, 0.05), atmospheric_light=airlight
    )
    out = t(image=scene)["image"].astype(np.float32)

    airlight_arr = np.array(airlight, dtype=np.float32)
    dist_to_airlight = np.linalg.norm(out - airlight_arr, axis=-1).mean(axis=1)  # per row

    # Depth increases with row index (top->bottom), so distance to airlight
    # must be non-increasing as row index increases.
    assert np.all(np.diff(dist_to_airlight) <= 1e-3)


def test_zero_beta_is_near_identity(provider: SharedDepthProvider, scene: np.ndarray) -> None:
    """beta=0 => transmission=exp(0)=1 everywhere => output should equal input."""
    t = RandomFog(depth_provider=provider, p=1.0, beta_range=(0.0, 0.0))
    out = t(image=scene)["image"]
    np.testing.assert_allclose(out.astype(np.float32), scene.astype(np.float32), atol=1.0)


def test_fixed_atmospheric_light_is_respected(
    provider: SharedDepthProvider, scene: np.ndarray
) -> None:
    """At very high beta, transmission -> 0 everywhere, so output -> atmospheric_light."""
    airlight = (200, 210, 190)
    t = RandomFog(depth_provider=provider, p=1.0, beta_range=(2.0, 2.0), atmospheric_light=airlight)
    out = t(image=scene)["image"].astype(np.float32)
    # Row 0 has depth=0 so transmission=1 regardless of beta; check a deep row instead.
    np.testing.assert_allclose(out[-1].mean(axis=0), np.array(airlight, dtype=np.float32), atol=2.0)
