"""Snow lit by the camera's illuminator.

The transform exists because of one measured failure: ``RandomSnow`` placed
before ``NightIrMode`` sends its flakes through the reflectance-to-infrared
conversion, which lights every pixel by the depth of the *scene* there. A
flake half a metre from the lens, drawn over a sky fifty metres away, is lit
as though it were fifty metres away. Night snow came out nearly invisible
exactly where real infrared footage shows it brightest.
"""

from __future__ import annotations

import albumentations as A
import numpy as np
import pytest

from surmentations.atmosphere import NightSnow, RandomSnow
from surmentations.camera import NightIrMode
from surmentations.depth.estimator import DepthEstimator
from surmentations.depth.provider import SharedDepthProvider


class _SkyOverGround(DepthEstimator):
    """Far sky in the upper half, a receding ground plane below."""

    def estimate(self, image: np.ndarray) -> np.ndarray:
        """Depth of a level road under the lower half of the frame, and a far wall above it."""
        height, width = image.shape[:2]
        rows = np.arange(height, dtype=np.float32).reshape(height, 1)
        depth = np.clip(1.5 * 300.0 / np.maximum(rows - height / 2.0, 1e-3), 0.0, 50.0)
        return np.broadcast_to(depth / 50.0, (height, width)).astype(np.float32)


def _daylit_scene() -> np.ndarray:
    """A sky over grey ground, by day, with a little noise."""
    rng = np.random.default_rng(0)
    image = np.zeros((320, 320, 3), dtype=np.uint8)
    image[:160] = (150, 170, 200)
    image[160:] = (90, 92, 88)
    noise = rng.integers(-6, 7, image.shape, dtype=np.int16)
    return np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def _night_frame() -> np.ndarray:
    """A dark noisy frame."""
    rng = np.random.default_rng(1)
    return rng.integers(8, 40, (320, 320, 3), dtype=np.uint8)


def _snow(**overrides: object) -> NightSnow:
    """``NightSnow`` with a fixed snowfall, density and field of view, and ``overrides``."""
    settings: dict[str, object] = {
        "snowfall_rate_range_mm_h": (6.0, 6.0),
        "snow_density_range_g_cm3": (0.05, 0.05),
        "horizontal_fov_range_deg": (60.0, 60.0),
        "backscatter_range": (0.0, 0.0),
    }
    settings.update(overrides)
    return NightSnow(p=1.0, **settings)  # type: ignore[arg-type]


def test_flakes_are_visible_against_a_far_dark_sky_where_the_daylight_model_loses_them() -> None:
    """The reason the transform exists, asserted on the region that exposed it."""
    provider = SharedDepthProvider(_SkyOverGround())
    quiet_ir = {"read_noise_range": (0.0, 0.0), "gain_range": (0.1, 0.1)}
    scene = _daylit_scene()

    daylight_model = A.Compose(
        [
            RandomSnow(
                depth_provider=provider,
                p=1.0,
                snowfall_rate_range_mm_h=(6.0, 6.0),
                veil_beta_range=(0.0, 0.0),
                accumulation_probability=0.0,
            ),
            NightIrMode(depth_provider=provider, p=1.0, **quiet_ir),  # type: ignore[arg-type]
        ],
        seed=0,
    )(image=scene)["image"]
    lamp_lit = A.Compose(
        [NightIrMode(depth_provider=provider, p=1.0, **quiet_ir), _snow()],  # type: ignore[arg-type]
        seed=0,
    )(image=scene)["image"]

    def lit_share_of_sky(frame: np.ndarray) -> float:
        """Share of the sky standing well clear of the sky's own level."""
        sky = frame[:150, :, 1].astype(np.float32)
        return float((sky > np.median(sky) + 60.0).mean())

    lost, lit = lit_share_of_sky(daylight_model), lit_share_of_sky(lamp_lit)
    # Measured over three seeds: 0.01-0.11% against 2.7-6.5%.
    assert lost < 0.004, (
        f"premise changed: RandomSnow before NightIrMode now lights {lost:.2%} of the sky"
    )
    assert lit > 0.015, f"only {lit:.2%} of the sky carries a lit flake"
    assert lit > 10.0 * lost


def test_a_flake_is_lit_by_its_own_distance_from_the_lamp() -> None:
    """Inverse-square from the lens: the nearest flakes are the effect."""
    transform = _snow(illuminator_range_m=(60.0, 60.0), illuminator_gain_range=(1.0, 1.0))
    transform.set_random_seed(0)
    frame = _night_frame()
    flakes = transform.get_params_dependent_on_data({"shape": frame.shape}, {"image": frame})[
        "flakes"
    ]
    assert len(flakes["distance"]) > 50

    # With the reach far away the roll-off is negligible and the law is bare.
    expected = 1.0 / flakes["distance"] ** 2
    np.testing.assert_allclose(flakes["brightness"], expected, rtol=0.02)


def test_the_lamps_reach_limits_how_deep_the_lit_snowfall_goes() -> None:
    def far_brightness(reach: float) -> float:
        transform = _snow(illuminator_range_m=(reach, reach), illuminator_gain_range=(1.0, 1.0))
        transform.set_random_seed(0)
        frame = _night_frame()
        flakes = transform.get_params_dependent_on_data(
            {"shape": frame.shape}, {"image": frame}
        )["flakes"]
        far = flakes["distance"] > np.percentile(flakes["distance"], 75)
        return float(flakes["brightness"][far].mean())

    assert far_brightness(1.0) < 0.2 * far_brightness(20.0)


def test_it_only_adds_light_and_needs_no_depth() -> None:
    frame = _night_frame()
    transform = _snow(backscatter_range=(0.05, 0.05))
    transform.set_random_seed(2)
    out = transform(image=frame)["image"]

    assert out.shape == frame.shape and out.dtype == frame.dtype
    assert (out.astype(np.int16) - frame.astype(np.int16)).min() >= -1
    assert out.astype(np.float32).mean() > frame.astype(np.float32).mean() + 1.0


def test_backscatter_is_a_separate_glow_that_follows_the_beam() -> None:
    """The unresolved flakes: no texture, brightest on the optical axis."""
    frame = np.full((240, 240, 3), 20, dtype=np.uint8)

    def render(backscatter: float) -> np.ndarray:
        transform = _snow(
            backscatter_range=(backscatter, backscatter),
            beam_falloff_range=(3.0, 3.0),
            snowfall_rate_range_mm_h=(0.02, 0.02),
            max_flakes=1,
        )
        transform.set_random_seed(4)
        result: np.ndarray = transform(image=frame)["image"]
        return result

    glow = render(0.3).astype(np.float32) - render(0.0).astype(np.float32)
    centre = glow[100:140, 100:140].mean()
    corner = glow[:30, :30].mean()
    assert centre > 20.0
    assert corner < 0.5 * centre


def test_more_snowfall_puts_more_flakes_in_the_beam() -> None:
    def count(rate: float) -> int:
        transform = _snow(snowfall_rate_range_mm_h=(rate, rate))
        transform.set_random_seed(0)
        frame = _night_frame()
        params = transform.get_params_dependent_on_data({"shape": frame.shape}, {"image": frame})
        return len(params["flakes"]["x"])

    light, heavy = count(0.5), count(6.0)
    assert heavy > 2 * light > 0


def test_a_binding_budget_keeps_the_widest_flakes() -> None:
    def radii(max_flakes: int) -> np.ndarray:
        transform = _snow(max_flakes=max_flakes)
        transform.set_random_seed(0)
        frame = _night_frame()
        params = transform.get_params_dependent_on_data({"shape": frame.shape}, {"image": frame})
        radius: np.ndarray = params["flakes"]["radius"]
        return radius

    everything = radii(200000)
    budget = len(everything) // 4
    assert budget > 20
    kept = radii(budget)
    assert len(kept) == budget
    assert np.median(kept) > 1.3 * np.median(everything)


def test_seeded_runs_are_reproducible() -> None:
    frame = _night_frame()
    first, second = _snow(), _snow()
    first.set_random_seed(9)
    second.set_random_seed(9)
    np.testing.assert_array_equal(first(image=frame)["image"], second(image=frame)["image"])


def test_impossible_ranges_are_rejected_at_construction() -> None:
    with pytest.raises(ValueError):
        NightSnow(illuminator_gain_range=(3.0, 1.0))
    with pytest.raises(ValueError):
        NightSnow(max_flakes=0)
