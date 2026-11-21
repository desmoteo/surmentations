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

    # A haze: a visibility of two kilometres, over a scene three deep.
    common = dict(
        depth_provider=SharedDepthProvider(_SkyOverGround(), depth_range_m=(5.0, 3000.0)),
        p=1.0,
        scattering_gamma_range=(0.0, 0.0),
        atmospheric_light=(220, 220, 220),
        forward_scatter_range=(0.0, 0.0),
    )
    haze = {"beta_range": (0.002, 0.002)}
    flat = RandomFog(sky_optical_depth_cap=10.0, **haze, **common)(image=scene)["image"]
    capped = RandomFog(sky_optical_depth_cap=1.0, **haze, **common)(image=scene)["image"]

    sky_rows = slice(4, 40)
    assert flat[sky_rows].astype(np.float32).std() < 0.5, "uncapped, the sky is all airlight"
    assert capped[sky_rows].astype(np.float32).std() > 2.0
    # Overhead, that is. A line of sight just over the horizon runs through
    # kilometres of the same haze, and the sky there is the haze's colour.
    low, high = capped[44:47].astype(np.float32), capped[4:8].astype(np.float32)
    assert abs(float(low.mean()) - 220.0) < 3.0 < abs(float(high.mean()) - 220.0)

    # A fog that hides what stands 150 m off hides the sky behind it too,
    # cap or no cap: held back, the sky stayed darker than the buildings
    # the fog had already taken, and their outline showed against it.
    fog = RandomFog(sky_optical_depth_cap=1.0, beta_range=(0.04, 0.04), **common)
    thick = fog(image=scene)["image"]
    assert thick[sky_rows].astype(np.float32).std() < 0.5
    assert abs(float(thick[sky_rows].mean()) - 220.0) < 1.5


def test_traffic_in_fog_shows_its_lamps_through_it_with_a_halo_round_each() -> None:
    """In thick fog a car is two haloes before it is a car. The lamps are
    lit where a reader found them, dimmed by the fog in front of them, and
    what that fog scatters out of their light glows round them."""
    from surmentations.perception import (
        SceneClass,
        SceneLabelEstimator,
        VehicleView,
        VehicleViewEstimator,
    )

    height, width = 120, 160
    labels = np.full((height, width), int(SceneClass.ROAD), dtype=np.uint8)
    labels[:30] = int(SceneClass.SKY)
    labels[60:90, 60:100] = int(SceneClass.VEHICLE)
    places = np.array([(66.0, 80.0, 2.0, 2.0, 0.9), (94.0, 80.0, 2.0, 2.0, 0.9)], np.float32)

    class _Named(SceneLabelEstimator):
        """The labels, whatever the frame."""

        def estimate(self, image: np.ndarray) -> np.ndarray:
            """The labels."""
            return labels

    class _Reader(VehicleViewEstimator):
        """One car, seen from the front, with its pair of headlamps."""

        def estimate(self, image: np.ndarray, vehicles: np.ndarray) -> list[VehicleView]:
            """The one car."""
            return [VehicleView((60, 60, 40, 30), 0.9, 0.0, places)]

    image = np.full((height, width, 3), 70, dtype=np.uint8)
    image[:30] = 200

    def fogged(lamps: float, read: bool = True) -> np.ndarray:
        """The frame in one fog, with the traffic's lamps at a strength."""
        extra = {"vehicles": _Reader()} if read else {}
        provider = SharedDepthProvider(
            _LinearDepthEstimator(),
            depth_range_m=(5.0, 80.0),
            labels=_Named(),
            **extra,  # type: ignore[arg-type]
        )
        fog = RandomFog(
            depth_provider=provider,
            p=1.0,
            beta_range=(0.03, 0.03),
            scattering_gamma_range=(0.0, 0.0),
            forward_scatter_range=(0.0, 0.0),
            vehicle_lamp_range=(lamps, lamps),
        )
        fog.set_random_seed(3)
        return fog(image=image)["image"].astype(np.float32)

    plain, lit = fogged(0.0), fogged(1.0)
    gained = (lit - plain).mean(axis=2)
    # Both lamps, far over what the fog left of the car.
    for column, row in ((66, 80), (94, 80)):
        assert gained[row, column] > 60.0
    # A halo round each that fades with distance from it, and nothing far off.
    assert gained[80, 70] > gained[80, 72] > 0.5 and gained[80, 80] < 0.5
    assert float(gained[:40].max()) < 1.0 and float(gained[:, :30].max()) < 1.0
    # Without a reader there are no lamps to light.
    assert np.array_equal(fogged(1.0, read=False), fogged(0.0, read=False))
