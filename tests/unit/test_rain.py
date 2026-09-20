import numpy as np
import pytest

from weathermentations.atmosphere.rain import (
    RandomRain,
    _apply_ground_wetness,
    _perspective_angles,
    render_streak_layer,
    streak_alpha,
)
from weathermentations.depth.estimator import DepthEstimator
from weathermentations.depth.provider import SharedDepthProvider
from weathermentations.utils.precipitation import raindrop_terminal_velocity


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


@pytest.fixture
def scene_hd() -> np.ndarray:
    """Large enough that drop widths are not all quantised to one pixel.

    At 60px wide the focal length is ~52px, so every drop in the simulated
    column projects to well under a pixel and every streak is drawn one pixel
    thick — which silently flattens any test of how width varies.
    """
    img = np.zeros((480, 854, 3), dtype=np.uint8)
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
        max_streaks=0,
        veil_beta_range=(0.0, 0.0),
        wetness_probability=0.0,
    )
    out = t(image=scene)["image"]
    np.testing.assert_allclose(out.astype(np.float32), scene.astype(np.float32), atol=1.0)


def test_veil_transmission_monotonic_with_depth(
    provider: SharedDepthProvider, scene: np.ndarray
) -> None:
    """Same Koschmieder invariant as fog: the veil must build monotonically
    with depth, so each row moves closer to the veil's own radiance than the
    row above it."""
    t = RandomRain(
        depth_provider=provider,
        p=1.0,
        max_streaks=0,
        veil_beta_range=(0.05, 0.05),
        wetness_probability=0.0,
    )
    out = t(image=scene)["image"].astype(np.float32)
    # Rain streaks and veil both composite toward the scene's own sky
    # radiance, which on this flat scene is just its brightest value.
    veil_color = out[-1].reshape(-1, 3).mean(axis=0)
    dist_to_veil = np.linalg.norm(out - veil_color, axis=-1).mean(axis=1)
    assert np.all(np.diff(dist_to_veil) <= 1e-2)


def test_streak_alpha_follows_diameter_over_length() -> None:
    """Garg & Nayar photometry: a drop occupies a pixel for
    diameter/length of the exposure, so a longer streak from the same drop
    must be proportionally fainter."""
    assert streak_alpha(3.0, 60.0) == pytest.approx(0.05)
    assert streak_alpha(3.0, 20.0) == pytest.approx(0.15)
    # Physically sane magnitude: streaks are a few percent opaque, not 50%.
    assert streak_alpha(2.0, 50.0) < 0.05


def test_streak_alpha_layer_stays_in_bounds(
    provider: SharedDepthProvider, scene_hd: np.ndarray
) -> None:
    # HD: streaks narrower than `min_streak_width_px` go to the veil rather
    # than being drawn, and on a 60px-wide frame nothing passes that cut.
    t = RandomRain(depth_provider=provider, p=1.0, max_streaks=50)
    params = t.get_params_dependent_on_data({"shape": scene_hd.shape}, {"image": scene_hd})
    assert params["veil_transmission"].min() >= 0.0
    assert params["veil_transmission"].max() <= 1.0

    # Streaks are a struct-of-arrays: one parallel array per property, each
    # holding one entry per rendered streak.
    streaks = params["streaks"]
    lengths = {field.shape for field in streaks.values()}
    assert len(lengths) == 1, "streak fields must be parallel arrays"
    assert streaks["alpha"].min() >= 0.0
    assert streaks["alpha"].max() <= 1.0


def test_streak_opacity_is_physically_small(
    provider: SharedDepthProvider, scene_hd: np.ndarray
) -> None:
    """Garg & Nayar: a drop occupies any one pixel for only diameter/length
    of the exposure, so streaks are a *few percent* opaque. Painting them at
    30-60% is the single thing that makes synthetic rain read as white
    scratches drawn over a photo."""
    t = RandomRain(depth_provider=provider, p=1.0, rain_rate_range_mm_h=(20.0, 20.0))
    t.set_random_seed(0)
    params = t.get_params_dependent_on_data({"shape": scene_hd.shape}, {"image": scene_hd})
    alpha = params["streaks"]["alpha"]
    assert np.median(alpha) < 0.25
    assert alpha.max() <= 1.0


def test_terminal_velocity_matches_gunn_and_kinzer() -> None:
    """The fall speed law is load-bearing: it sets streak length, and through
    ``alpha = D / (v T)`` the streak's opacity too. Pin it against the
    measured Gunn & Kinzer (1949) table.

    An earlier fit in this module was saturated — it returned 9.40 m/s for
    every drop from 0.5 mm up, i.e. more than 4x too fast for a small drop —
    and went unnoticed because nothing called it.
    """
    measured = {0.5: 2.06, 1.0: 4.03, 2.0: 6.49, 3.0: 8.06, 4.0: 8.83, 5.0: 9.09}
    diameters = np.array(list(measured), dtype=np.float32)
    predicted = raindrop_terminal_velocity(diameters)
    expected = np.array(list(measured.values()), dtype=np.float32)

    relative_error = np.abs(predicted - expected) / expected
    assert relative_error.max() < 0.05, dict(zip(measured, predicted, strict=True))
    # Strictly increasing, and emphatically not a constant.
    assert np.all(np.diff(predicted) > 0)


def test_bigger_drops_draw_longer_streaks(
    provider: SharedDepthProvider, scene_hd: np.ndarray
) -> None:
    """Length, width and opacity are not independent: they all follow from
    one drop diameter through the terminal-velocity law. A longer streak must
    therefore also be a wider one."""
    t = RandomRain(depth_provider=provider, p=1.0, rain_rate_range_mm_h=(40.0, 40.0))
    t.set_random_seed(0)
    params = t.get_params_dependent_on_data({"shape": scene_hd.shape}, {"image": scene_hd})
    streaks = params["streaks"]
    # Thickness is quantised to whole pixels, so compare group means rather
    # than correlating the raw arrays.
    order = np.argsort(streaks["length"])
    quartile = len(order) // 4
    short = streaks["thickness"][order[:quartile]].mean()
    long = streaks["thickness"][order[-quartile:]].mean()
    assert long > short


def test_streak_angles_vary_within_jitter_bound(
    provider: SharedDepthProvider, scene_hd: np.ndarray
) -> None:
    """Streaks must not all share the exact same angle (real rain isn't a
    perfectly parallel line-hatch pattern), but with a level camera — whose
    vanishing point is at infinity — each one's deviation from the base
    angle must stay within angle_jitter_range."""
    t = RandomRain(
        depth_provider=provider,
        p=1.0,
        rain_rate_range_mm_h=(20.0, 20.0),
        camera_pitch_range_deg=(0.0, 0.0),
        wind_angle_range=(10.0, 10.0),
        angle_jitter_range=(5.0, 5.0),
    )
    t.set_random_seed(0)
    params = t.get_params_dependent_on_data({"shape": scene_hd.shape}, {"image": scene_hd})
    streak_angles = {round(float(a), 6) for a in params["streaks"]["angle"]}
    assert len(streak_angles) > 1, "all streaks share the exact same angle"
    for angle in streak_angles:
        assert 10.0 - 5.0 - 1e-6 <= angle <= 10.0 + 5.0 + 1e-6


def test_level_camera_gives_parallel_streaks() -> None:
    """Rain falls along parallel world lines. With the camera level that
    direction is perpendicular to the optical axis, its vanishing point is at
    infinity, and the projected streaks are exactly parallel."""
    x = np.array([50.0, 320.0, 600.0], dtype=np.float32)
    y = np.full(3, 240.0, dtype=np.float32)
    angles = _perspective_angles(x, y, 640, 480, 554.0, camera_pitch_deg=0.0, wind_angle_deg=12.0)
    np.testing.assert_allclose(angles, 12.0, atol=1e-4)


def test_tilted_camera_makes_streaks_converge() -> None:
    """Parallel 3-D lines project through a common vanishing point, so a
    camera tilted down makes the streaks splay symmetrically about it —
    which is what says the rain is falling *through* the scene rather than
    being drawn on top of it."""
    # Symmetric about the frame centre (320), so the flanking angles must
    # come out equal and opposite.
    x = np.array([50.0, 320.0, 590.0], dtype=np.float32)
    y = np.full(3, 240.0, dtype=np.float32)
    angles = _perspective_angles(x, y, 640, 480, 554.0, camera_pitch_deg=30.0, wind_angle_deg=0.0)

    # Centre column falls straight down; the flanks lean toward it, by equal
    # and opposite amounts for a symmetric pair.
    assert angles[1] == pytest.approx(0.0, abs=1e-4)
    assert angles[0] > 2.0
    assert angles[2] < -2.0
    assert angles[0] == pytest.approx(-angles[2], abs=1e-3)


def test_ground_wetness_only_touches_the_ground_mask() -> None:
    """Wetness is masked by the surface, not by image rows.

    The screen-space band this replaced wet the bottom 12-32% of rows, which
    covered only 49% of the reference frame's road and left the rest dry.
    """
    height, width = 100, 60
    img = np.full((height, width, 3), 0.4, dtype=np.float32)
    wetness = np.zeros((height, width), dtype=np.float32)
    wetness[:, :30] = 1.0  # a vertical split: no relationship to rows at all

    out = _apply_ground_wetness(img.copy(), wetness, 0.6, horizon_px=10.0, focal_px=80.0)

    np.testing.assert_allclose(out[:, 30:], img[:, 30:], rtol=1e-6)
    assert np.all(out[:, :30] != img[:, :30])


def test_ground_wetness_mirrors_the_scene_about_the_horizon() -> None:
    """The reflection at row y must be the scene at row 2*horizon - y.

    Mirroring about anything else - the band's own top edge, as this used to -
    puts the reflection of a car somewhere the car is not.
    """
    height, width, horizon = 120, 40, 20.0
    # A single bright bar; everything else black, so the reflection's position
    # is unambiguous.
    img = np.zeros((height, width, 3), dtype=np.float32)
    img[8:12] = 1.0
    wetness = np.ones((height, width), dtype=np.float32)

    out = _apply_ground_wetness(img.copy(), wetness, 0.0, horizon_px=horizon, focal_px=60.0)

    # Source rows 8-12 mirror to 2*20 - [8, 12] = [32, 28].
    profile = out.mean(axis=(1, 2))
    below = profile[int(horizon) + 1 :]
    peak = int(np.argmax(below)) + int(horizon) + 1
    assert 27 <= peak <= 33, f"reflection peaked at row {peak}, expected near 30"


def test_ground_wetness_reflectance_is_strongest_at_grazing_incidence() -> None:
    """A wet road is mirror-bright toward the horizon and matte at your feet.

    That is Fresnel, and it is the part a single blend weight cannot express.
    """
    height, width = 200, 40
    img = np.full((height, width, 3), 0.5, dtype=np.float32)
    img[:60] = 1.0  # bright sky to be reflected
    wetness = np.ones((height, width), dtype=np.float32)

    out = _apply_ground_wetness(img.copy(), wetness, 0.5, horizon_px=60.0, focal_px=100.0)

    near_horizon = float(out[65:85].mean())
    far_below = float(out[175:195].mean())
    assert near_horizon > far_below, (
        f"reflection should fade away from the horizon: {near_horizon:.3f} "
        f"near vs {far_below:.3f} far"
    )
    # Far below the horizon the surface is close to Lambertian, so the result
    # should approach the darkened diffuse value rather than the reflection.
    assert far_below == pytest.approx(0.5 * 0.5, abs=0.06)


def test_rain_rate_drives_visible_streak_coverage(
    provider: SharedDepthProvider, scene_hd: np.ndarray
) -> None:
    """Heavier rain must put visibly more streak coverage in frame, at every
    step of the ladder.

    This is the end-to-end guard on the budget logic. When the render cap
    binds, the *width* threshold is raised rather than the field truncated;
    truncating instead makes apparent intensity depend on the render budget
    rather than on the weather, which showed up as coverage peaking mid-ladder
    and then falling as the rate kept climbing.
    """

    def coverage(rate: float) -> float:
        t = RandomRain(depth_provider=provider, p=1.0, rain_rate_range_mm_h=(rate, rate))
        t.set_random_seed(0)
        params = t.get_params_dependent_on_data({"shape": scene_hd.shape}, {"image": scene_hd})
        layer = render_streak_layer(scene_hd.shape[0], scene_hd.shape[1], params["streaks"])
        return float((-np.expm1(-layer) > 0.02).mean())

    coverages = [coverage(rate) for rate in (5.0, 20.0, 45.0, 70.0, 95.0)]
    assert all(a < b for a, b in zip(coverages, coverages[1:], strict=False)), coverages


def test_streak_width_and_length_track_distance(
    provider: SharedDepthProvider, scene_hd: np.ndarray
) -> None:
    """Both scale as 1/z, so near drops must draw longer *and* wider streaks.

    Distance is compared directly rather than via a rendered dimension: width
    and length each depend on the drop's diameter as well as its distance, so
    neither is a usable proxy for the other.
    """
    t = RandomRain(depth_provider=provider, p=1.0, rain_rate_range_mm_h=(40.0, 40.0))
    t.set_random_seed(0)
    params = t.get_params_dependent_on_data({"shape": scene_hd.shape}, {"image": scene_hd})
    streaks = params["streaks"]

    order = np.argsort(streaks["distance"])
    near, far = order[: len(order) // 4], order[-len(order) // 4 :]
    assert streaks["length"][near].mean() > streaks["length"][far].mean()
    assert streaks["thickness"][near].mean() >= streaks["thickness"][far].mean()


def test_gamma_drop_size_distribution_reduces_to_marshall_palmer() -> None:
    """`dsd_shape_mu=0` must be exactly the exponential, not approximately.

    Every published number in this library was measured against Marshall &
    Palmer, so the gamma generalisation has to leave that case untouched — a
    shape parameter that quietly shifts the default would invalidate the lot.
    """
    from weathermentations.utils.precipitation import (
        gamma_dsd_slope,
        marshall_palmer_slope,
        rain_number_density,
        sample_gamma_diameters,
    )

    for rate in (5.0, 25.0, 100.0):
        assert gamma_dsd_slope(rate, 0.0) == pytest.approx(marshall_palmer_slope(rate))
        assert rain_number_density(rate, 0.8, 0.0) == pytest.approx(
            rain_number_density(rate, 0.8, 1e-9), rel=1e-6
        ), "the gamma calibration puts a step at mu=0"

    rng = np.random.default_rng(0)
    exponential = sample_gamma_diameters(rng, 60000, marshall_palmer_slope(50.0), 0.8, 6.0, 0.0)
    analytic_mean = 0.8 + 1.0 / marshall_palmer_slope(50.0)
    assert exponential.mean() == pytest.approx(analytic_mean, rel=0.05)


def test_gamma_shape_moves_mass_toward_larger_drops() -> None:
    """The point of the gamma is the large-drop end: Ulbrich (1983) fits
    observed spectra better than an exponential because a real distribution
    has a mode at a finite diameter rather than piling up at zero."""
    from weathermentations.utils.precipitation import gamma_dsd_slope, sample_gamma_diameters

    rng = np.random.default_rng(1)
    fractions = []
    for mu in (0.0, 3.0):
        drops = sample_gamma_diameters(rng, 80000, gamma_dsd_slope(50.0, mu), 0.8, 6.0, mu)
        fractions.append(float((drops > 2.0).mean()))
    assert fractions[1] > fractions[0] * 1.2, (
        f"mu=3 should carry noticeably more large drops: {fractions[0]:.3f} -> {fractions[1]:.3f}"
    )


def test_drop_size_distribution_is_not_the_visibility_lever() -> None:
    """Pins the measurement behind the documentation.

    The summed projected area of the drops is what decides whether rain can be
    seen, and at a fixed rain rate it barely moves with the distribution's
    shape. Recording it as a test because it is the claim most likely to be
    re-litigated, and because it is cheap to check and expensive to re-derive.
    """
    from weathermentations.utils.precipitation import (
        gamma_dsd_slope,
        raindrop_terminal_velocity,
    )

    diameters = np.linspace(0.2, 6.0, 2000)
    step = float(diameters[1] - diameters[0])
    speeds = raindrop_terminal_velocity(diameters)
    areas = []
    for mu in (0.0, 2.0, 4.0):
        shape = diameters**mu * np.exp(-gamma_dsd_slope(50.0, mu) * diameters)
        rate = float((shape * diameters**3 * speeds).sum() * step)
        shape = shape / rate  # normalise to a common rain rate
        areas.append(float((shape[diameters >= 0.8] * diameters[diameters >= 0.8] ** 2).sum()))
    spread = (max(areas) - min(areas)) / min(areas)
    assert spread < 0.10, (
        f"projected area varies by {spread:.1%} across mu=0..4; the documentation "
        "claims a few percent, so one of the two is wrong"
    )
