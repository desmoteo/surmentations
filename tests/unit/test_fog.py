"""``RandomFog``: the output contract, the approach to airlight with depth, and the sky."""

import numpy as np
import pytest

from surmentations.atmosphere.fog import RandomFog
from surmentations.depth.estimator import DepthEstimator
from surmentations.depth.provider import SharedDepthProvider


class _LinearDepthEstimator(DepthEstimator):
    """Depth increases linearly from 0 (top row) to 1 (bottom row)."""

    def estimate(self, image: np.ndarray) -> np.ndarray:
        """Depth rising linearly from 0 at the top row to 1 at the bottom."""
        height, width = image.shape[:2]
        return np.tile(np.linspace(0.0, 1.0, height)[:, None], (1, width)).astype(np.float32)


@pytest.fixture
def provider() -> SharedDepthProvider:
    """A depth provider over the linear depth, as a fixture."""
    return SharedDepthProvider(_LinearDepthEstimator())


@pytest.fixture
def scene() -> np.ndarray:
    """A flat blue frame, as a fixture."""
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
    is itself monotonically decreasing in d).

    The sky cap and the forward-scatter blur are both disabled here: they are
    deliberate departures from the bare equation (a capped sky keeps its
    gradient; the blur mixes neighbouring rows), so leaving them on would be
    testing them rather than the scattering law.
    """
    airlight = (220, 220, 220)
    t = RandomFog(
        depth_provider=provider,
        p=1.0,
        beta_range=(0.05, 0.05),
        scattering_gamma_range=(0.0, 0.0),
        atmospheric_light=airlight,
        forward_scatter_range=(0.0, 0.0),
        sky_optical_depth_cap=10.0,
    )
    out = t(image=scene)["image"].astype(np.float32)

    airlight_arr = np.array(airlight, dtype=np.float32)
    dist_to_airlight = np.linalg.norm(out - airlight_arr, axis=-1).mean(axis=1)  # per row

    # Depth increases with row index (top->bottom), so distance to airlight
    # must be non-increasing as row index increases.
    assert np.all(np.diff(dist_to_airlight) <= 1e-3)


def test_thin_haze_is_bluer_than_dense_fog(
    provider: SharedDepthProvider, scene: np.ndarray
) -> None:
    """Wavelength-dependent extinction: beta ~ lambda^-gamma, so with a large
    gamma the blue channel is scattered hardest and the distance goes cool,
    while gamma=0 (droplets >> wavelength, i.e. dense fog) stays neutral."""
    common = dict(
        depth_provider=provider,
        p=1.0,
        beta_range=(0.03, 0.03),
        atmospheric_light=(210, 210, 210),
        forward_scatter_range=(0.0, 0.0),
    )
    neutral = RandomFog(scattering_gamma_range=(0.0, 0.0), **common)(image=scene)["image"]
    selective = RandomFog(scattering_gamma_range=(3.0, 3.0), **common)(image=scene)["image"]

    far_neutral = neutral[-1].astype(np.float32).mean(axis=0)
    far_selective = selective[-1].astype(np.float32).mean(axis=0)

    # Blue relative to red must rise once scattering is wavelength-selective.
    assert (far_selective[2] - far_selective[0]) > (far_neutral[2] - far_neutral[0])


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
    t = RandomFog(
        depth_provider=provider,
        p=1.0,
        beta_range=(2.0, 2.0),
        scattering_gamma_range=(0.0, 0.0),
        atmospheric_light=airlight,
        forward_scatter_range=(0.0, 0.0),
        sky_optical_depth_cap=10.0,
    )
    out = t(image=scene)["image"].astype(np.float32)
    # Row 0 has depth=0 so transmission=1 regardless of beta; check a deep row instead.
    np.testing.assert_allclose(out[-1].mean(axis=0), np.array(airlight, dtype=np.float32), atol=2.0)


def test_sky_keeps_its_gradient() -> None:
    """Uncapped, the sky collapses to exactly the airlight colour and goes
    flat. The cap must leave measurable variation up there.

    The sky is found by what it looks like — featureless, at one depth,
    reaching the top of the frame — and not by being far, so the scene here
    has one: a smooth gradient above textured ground, at the arbitrary flat
    depth a network gives it."""

    class _SkyOverGround(DepthEstimator):
        """A sky at an arbitrary depth over ground receding from the camera."""
        def estimate(self, image: np.ndarray) -> np.ndarray:
            height, width = image.shape[:2]
            depth = np.full((height, width), 0.4, dtype=np.float32)
            ground = height - height // 2
            depth[height // 2 :] = np.linspace(1.0, 0.0, ground, dtype=np.float32)[:, None]
            return depth

    rng = np.random.default_rng(0)
    scene = np.empty((96, 96, 3), dtype=np.uint8)
    scene[:48] = np.linspace(200, 150, 48).astype(np.uint8)[:, None, None]
    scene[48:] = rng.integers(30, 120, size=(48, 96, 1), dtype=np.uint8)

    common = dict(
        depth_provider=SharedDepthProvider(_SkyOverGround()),
        p=1.0,
        beta_range=(0.5, 0.5),
        scattering_gamma_range=(0.0, 0.0),
        atmospheric_light=(220, 220, 220),
        forward_scatter_range=(0.0, 0.0),
    )
    flat = RandomFog(sky_optical_depth_cap=10.0, **common)(image=scene)["image"]
    capped = RandomFog(sky_optical_depth_cap=1.0, **common)(image=scene)["image"]

    sky_rows = slice(4, 40)
    assert flat[sky_rows].astype(np.float32).std() < 0.5, "uncapped, the sky is all airlight"
    assert capped[sky_rows].astype(np.float32).std() > 2.0
