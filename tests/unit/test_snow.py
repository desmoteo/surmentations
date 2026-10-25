import numpy as np
import pytest

from weathermentations.atmosphere.snow import RandomSnow, snow_fall_speed
from weathermentations.depth.estimator import DepthEstimator
from weathermentations.depth.provider import SharedDepthProvider
from weathermentations.utils.precipitation import snow_physical_diameter

_FOV_DEG = 60.0
_MAX_DEPTH_M = 50.0


class _GroundPlaneDepthEstimator(DepthEstimator):
    """A level camera 1.5 m above a horizontal ground plane.

    Deliberately not a top-to-bottom ramp. Depth on real ground *recedes
    upward* toward the horizon, so a linear ramp with the far end at the
    bottom of the frame describes a ceiling — which correctly collects no
    snow, and silently made the accumulation tests assert nothing.
    """

    def estimate(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        focal = width / (2.0 * np.tan(np.deg2rad(_FOV_DEG) / 2.0))
        cy = height / 2.0
        rows = np.arange(height, dtype=np.float32).reshape(height, 1)
        below_horizon = rows > cy + 2
        depth = np.where(below_horizon, 1.5 * focal / np.maximum(rows - cy, 1e-3), _MAX_DEPTH_M)
        depth = np.minimum(depth, _MAX_DEPTH_M)
        return np.broadcast_to(depth / _MAX_DEPTH_M, (height, width)).astype(np.float32)


class _WallDepthEstimator(DepthEstimator):
    """A flat wall square-on to the camera: vertical everywhere."""

    def estimate(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        return np.full((height, width), 0.2, dtype=np.float32)


@pytest.fixture
def provider() -> SharedDepthProvider:
    return SharedDepthProvider(_GroundPlaneDepthEstimator())


@pytest.fixture
def scene() -> np.ndarray:
    img = np.zeros((160, 220, 3), dtype=np.uint8)
    img[:, :] = (40, 70, 60)
    return img


@pytest.fixture
def scene_hd() -> np.ndarray:
    """Large enough that a useful number of flakes actually resolve.

    Flakes narrower than ``min_flake_width_px`` are left to the veil rather
    than drawn, and that threshold is in pixels — so on a 220px-wide frame
    (focal ~190px) almost nothing passes it and the rendered population
    collapses to a handful, which silently flattens any test of its
    distribution.
    """
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
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
        flake_count_scale=0.0,
        accumulation_probability=0.0,
        veil_beta_range=(0.0, 0.0),
    )
    out = t(image=scene)["image"]
    np.testing.assert_allclose(out.astype(np.float32), scene.astype(np.float32), atol=1.0)


def test_snow_falls_far_slower_than_rain_and_barely_varies_with_size() -> None:
    """Locatelli & Hobbs: a snow aggregate's drag and mass scale together, so
    its fall speed is nearly size-independent — unlike a raindrop's, which
    more than triples over the same size range."""
    speeds = snow_fall_speed(np.array([1.0, 3.0, 10.0]))
    assert np.all(speeds < 2.0), speeds
    # Barely varies: a 10x size increase changes speed by well under 50%.
    assert speeds[2] / speeds[0] < 1.5


def test_flake_sizes_follow_the_gunn_marshall_distribution(
    provider: SharedDepthProvider, scene_hd: np.ndarray
) -> None:
    """Flake sizes are exponentially distributed, not uniform: many small
    flakes with a thin tail of large aggregates. So the median must sit well
    below the midpoint of the rendered size range."""
    t = RandomSnow(depth_provider=provider, p=1.0, flake_count_scale=4.0)
    t.set_random_seed(0)
    params = t.get_params_dependent_on_data({"shape": scene_hd.shape}, {"image": scene_hd})
    radii = params["flakes"]["radius"]
    assert len(radii) > 200
    assert np.median(radii) < 0.35 * radii.max()


def test_flake_size_is_independent_of_scene_depth(
    provider: SharedDepthProvider, scene_hd: np.ndarray
) -> None:
    """A flake's size must not track the depth of whatever is *behind* it —
    tying the two together makes flakes look pasted onto surfaces. The scene
    depth ramps strongly with height, so flake size must show no such ramp."""
    t = RandomSnow(depth_provider=provider, p=1.0, flake_count_scale=6.0)
    t.set_random_seed(0)
    params = t.get_params_dependent_on_data({"shape": scene_hd.shape}, {"image": scene_hd})
    ys = params["flakes"]["y"]
    radii = params["flakes"]["radius"]

    top = radii[ys < scene_hd.shape[0] * 0.25]
    bottom = radii[ys > scene_hd.shape[0] * 0.75]
    assert len(top) > 50 and len(bottom) > 50
    assert abs(np.median(top) - np.median(bottom)) < 0.3 * np.median(radii)


def test_motion_smear_scales_with_proximity(
    provider: SharedDepthProvider, scene_hd: np.ndarray
) -> None:
    """Smear length is f*v*T/z, the same 1/z as the size — so a near flake
    must smear over more pixels than a far one. A single global motion blur
    (the previous model) gives every flake the same smear regardless of
    distance, which is the one thing the geometry rules out."""
    t = RandomSnow(depth_provider=provider, p=1.0, flake_count_scale=4.0)
    t.set_random_seed(0)
    params = t.get_params_dependent_on_data({"shape": scene_hd.shape}, {"image": scene_hd})
    flakes = params["flakes"]

    # Sort by distance itself. Rendered radius is not a usable proxy: it
    # depends on the flake's diameter as well as its distance, so a large far
    # flake and a small near one can render the same size.
    order = np.argsort(flakes["distance"])
    nearest = flakes["smear"][order[:50]]
    farthest = flakes["smear"][order[-50:]]
    assert nearest.mean() > farthest.mean() * 3.0


def test_flakes_do_not_all_share_one_direction(
    provider: SharedDepthProvider, scene: np.ndarray
) -> None:
    """Flakes tumble, so each one's path deviates from the prevailing slant.
    Identical directions across the field read as a combed texture."""
    t = RandomSnow(depth_provider=provider, p=1.0, flutter_range=(20.0, 20.0))
    t.set_random_seed(0)
    params = t.get_params_dependent_on_data({"shape": scene.shape}, {"image": scene})
    assert len(np.unique(np.round(params["flakes"]["dir_x"], 6))) > 10


def test_accumulation_follows_surface_orientation(scene: np.ndarray) -> None:
    """Snow settles by Lambert's cosine law against "up", so a horizontal
    ground plane must collect it and a vertical wall must not — regardless of
    where either sits in the frame."""
    kwargs = dict(
        p=1.0,
        flake_count_scale=0.0,
        veil_beta_range=(0.0, 0.0),
        accumulation_probability=1.0,
        accumulation_strength_range=(0.8, 0.8),
        accumulation_patchiness_range=(0.0, 0.0),
    )
    ground = RandomSnow(
        depth_provider=SharedDepthProvider(_GroundPlaneDepthEstimator()), **kwargs
    )
    wall = RandomSnow(depth_provider=SharedDepthProvider(_WallDepthEstimator()), **kwargs)

    ground.set_random_seed(0)
    wall.set_random_seed(0)
    ground_out = ground(image=scene)["image"].astype(np.float32)
    wall_out = wall(image=scene)["image"].astype(np.float32)

    base = scene.astype(np.float32).mean()
    # The lower half of the frame is ground in one case and wall in the other.
    assert ground_out[100:].mean() > base + 20.0
    np.testing.assert_allclose(wall_out, scene.astype(np.float32), atol=1.5)


def test_accumulation_never_fires_when_probability_zero(
    provider: SharedDepthProvider, scene: np.ndarray
) -> None:
    t = RandomSnow(
        depth_provider=provider,
        p=1.0,
        flake_count_scale=0.0,
        accumulation_probability=0.0,
        veil_beta_range=(0.0, 0.0),
    )
    for seed in range(10):
        t.set_random_seed(seed)
        out = t(image=scene)["image"]
        # Not bit-exact: compositing happens in linear light, and the
        # sRGB -> linear -> sRGB round trip costs up to one 8-bit level.
        np.testing.assert_allclose(out.astype(np.float32), scene.astype(np.float32), atol=1.0)


def test_melted_equivalent_diameter_is_converted_to_physical_size() -> None:
    """Gauge-derived distributions (Gunn-Marshall included) are stated in
    *melted-equivalent* diameters. A snowflake is mostly air, so the flake the
    camera sees is far bigger, and using the melted diameter as the optical
    size is what made flakes too small to resolve at all."""
    melted = np.array([1.0, 2.0, 5.0], dtype=np.float32)

    fresh = snow_physical_diameter(melted, 0.05)
    wet = snow_physical_diameter(melted, 0.2)

    # (rho_water / rho_snow)^(1/3): 2.71x for fresh dendrites, 1.71x for wet.
    np.testing.assert_allclose(fresh / melted, 2.714, rtol=1e-3)
    np.testing.assert_allclose(wet / melted, 1.710, rtol=1e-3)
    # Lower density always means a physically larger flake.
    assert np.all(fresh > wet)


def test_lower_density_snow_renders_larger_flakes(
    provider: SharedDepthProvider, scene_hd: np.ndarray
) -> None:
    """The density conversion has to reach the render, not just the helper."""

    def median_radius(density: float) -> float:
        t = RandomSnow(
            depth_provider=provider, p=1.0, snow_density_range_g_cm3=(density, density)
        )
        t.set_random_seed(0)
        params = t.get_params_dependent_on_data({"shape": scene_hd.shape}, {"image": scene_hd})
        return float(np.median(params["flakes"]["radius"]))

    assert median_radius(0.04) > median_radius(0.15)


def test_snowfall_rate_drives_visible_flake_coverage(
    provider: SharedDepthProvider, scene_hd: np.ndarray
) -> None:
    """Heavier snowfall must put visibly more flake area in frame. This is the
    end-to-end check that the rate actually reaches the picture — the count,
    the size distribution and the veil all derive from it."""

    def flake_area(rate: float) -> float:
        t = RandomSnow(
            depth_provider=provider, p=1.0, snowfall_rate_range_mm_h=(rate, rate)
        )
        t.set_random_seed(0)
        params = t.get_params_dependent_on_data({"shape": scene_hd.shape}, {"image": scene_hd})
        flakes = params["flakes"]
        rendered = np.hypot(flakes["radius"], flakes["blur_radius"])
        alpha = np.minimum(1.0, (flakes["radius"] / np.maximum(rendered, 1e-6)) ** 2)
        return float(np.sum(np.pi * rendered**2 * alpha))

    light, moderate, heavy = flake_area(0.5), flake_area(3.0), flake_area(15.0)
    assert light < moderate < heavy


def test_a_binding_flake_budget_keeps_the_widest_flakes_not_a_random_subset() -> None:
    """``max_flakes`` is a render budget, and a budget must not decide what
    the weather looks like.

    Truncating the count draws an arbitrary subset of the field, so once the
    cap binds a heavier snowfall stops looking heavier. Rain converts the
    budget into a projected-width threshold instead — the flakes drawn are the
    most visible ones, and the rest are what the veil already stands for — and
    snow did not: at 1920 px and severity 5 it was over budget in five draws of
    six and dropped a fifth of its flakes at random.
    """
    from weathermentations.diagnostics import recording

    image = np.full((720, 960, 3), 110, dtype=np.uint8)
    provider = SharedDepthProvider(_GroundPlaneDepthEstimator())

    def field(max_flakes: int) -> tuple[np.ndarray, dict[str, float]]:
        transform = RandomSnow(
            depth_provider=provider,
            p=1.0,
            snowfall_rate_range_mm_h=(8.0, 8.0),
            snow_density_range_g_cm3=(0.05, 0.05),
            horizontal_fov_range_deg=(60.0, 60.0),
            max_flakes=max_flakes,
        )
        transform.set_random_seed(0)
        with recording() as log:
            params = transform.get_params_dependent_on_data(
                {"shape": image.shape}, {"image": image}
            )
        return 2.0 * params["flakes"]["radius"], log[0]["drew"]

    everything, drew_all = field(200000)
    budget = len(everything) // 4
    assert budget > 100, "scene too sparse for the budget to mean anything"
    kept, drew_kept = field(budget)

    assert len(kept) == budget
    assert drew_all["width_cut_px"] == pytest.approx(1.5), "no budget pressure, no raised cut"
    assert drew_kept["width_cut_px"] > 1.5

    # The narrowest flake drawn under the budget is about as wide as the
    # budget-th widest flake of the whole field: the cut fell where it should.
    threshold = np.sort(everything)[-budget]
    assert kept.min() == pytest.approx(threshold, rel=0.2)
    # A random quarter of the field would have the field's own median.
    assert np.median(kept) > 1.3 * np.median(everything)
