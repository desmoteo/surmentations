"""HeatShimmer: displacement and blur from the theory of seeing; nothing over a street's length."""

from __future__ import annotations

import numpy as np
import pytest

from weathermentations.atmosphere import HeatShimmer
from weathermentations.atmosphere.heat_shimmer import (
    blur_by_sigma,
    blur_fwhm_rad,
    smooth_field,
    tilt_sigma_rad,
)
from weathermentations.depth.estimator import DepthEstimator
from weathermentations.depth.provider import SharedDepthProvider

SHAPE = (160, 200)


class _NearBelowFarAbove(DepthEstimator):
    """Depth rising from near at the bottom of the frame to far at the top."""

    def estimate(self, image: np.ndarray) -> np.ndarray:
        """Relative depth 1 on the top row to 0 on the bottom one."""
        height, width = image.shape[:2]
        return np.tile(np.linspace(1.0, 0.0, height, dtype=np.float32)[:, None], (1, width))


def _shimmer(far_m: float, **settings: object) -> HeatShimmer:
    """Hot ground seen through a 4 degree lens, over a scene that reaches ``far_m``."""
    arguments: dict[str, object] = {
        "log10_cn2_range": (-12.0, -12.0),
        "horizontal_fov_range_deg": (4.0, 4.0),
        "aperture_diameter_range_mm": (20.0, 20.0),
        **settings,
    }
    provider = SharedDepthProvider(_NearBelowFarAbove(), depth_range_m=(3.0, far_m))
    return HeatShimmer(depth_provider=provider, p=1.0, **arguments)


def _params(shimmer: HeatShimmer, seed: int = 0) -> dict:
    """The displacement and blur a shimmer draws for a frame."""
    image = np.zeros((*SHAPE, 3), dtype=np.uint8)
    shimmer.set_random_seed(seed)
    return shimmer.get_params_dependent_on_data({"shape": image.shape}, {"image": image})


def test_the_tilt_is_the_formula_s() -> None:
    # 2.91 D^(-1/3) Cn2 L, for 20 mm, 1e-12 and 1 km: 1.07e-8 rad^2.
    sigma = float(tilt_sigma_rad(1000.0, 1e-12, 0.02))
    assert sigma**2 == pytest.approx(2.91 * 0.02 ** (-1 / 3) * 1e-12 * 1000.0)
    assert sigma == pytest.approx(1.035e-4, rel=0.01)
    assert float(tilt_sigma_rad(4000.0, 1e-12, 0.02)) == pytest.approx(2.0 * sigma)
    assert float(tilt_sigma_rad(1000.0, 1e-14, 0.02)) == pytest.approx(0.1 * sigma)


def test_blur_starts_where_the_air_is_worse_than_the_lens() -> None:
    # A 2 mm lens is limited by its own diffraction over any street.
    assert float(blur_fwhm_rad(50.0, 1e-13, 0.002)) == 0.0
    near, far = float(blur_fwhm_rad(300.0, 1e-12, 0.02)), float(blur_fwhm_rad(3000.0, 1e-12, 0.02))
    assert far > near > 0.0
    # Well past that point the blur goes as the path to the power 3/5.
    assert far / near == pytest.approx(10.0**0.6, rel=0.1)


def test_over_a_street_s_length_it_does_nothing() -> None:
    image = np.random.default_rng(0).integers(0, 256, (*SHAPE, 3), dtype=np.uint8)
    street = HeatShimmer(
        depth_provider=SharedDepthProvider(_NearBelowFarAbove(), depth_range_m=(3.0, 50.0)),
        p=1.0,
        horizontal_fov_range_deg=(60.0, 60.0),
        aperture_diameter_range_mm=(2.0, 2.0),
        log10_cn2_range=(-13.0, -13.0),
    )
    street.set_random_seed(0)
    params = _params(street)
    assert float(np.abs(params["shift"]).max()) < 0.05
    assert np.array_equal(street(image=image)["image"], image)


def test_displacement_grows_with_distance_and_is_nothing_on_average() -> None:
    shifts = np.stack([_params(_shimmer(4000.0), seed)["shift"] for seed in range(12)])
    spread_by_row = shifts.std(axis=(0, 2, 3))
    assert spread_by_row[:20].mean() > 3.0 * spread_by_row[-20:].mean()
    assert spread_by_row[:20].mean() > 0.5
    # Shimmer does not move the picture as a whole.
    assert abs(float(shifts.mean())) < 0.1 * float(shifts.std())


def test_the_far_part_of_the_frame_moves_by_what_the_formula_gives() -> None:
    shifts = np.stack([_params(_shimmer(4000.0), seed)["shift"] for seed in range(24)])
    focal = 200 / (2.0 * np.tan(np.radians(2.0)))
    expected = float(tilt_sigma_rad(4000.0, 1e-12, 0.02)) * focal
    assert float(shifts[:, :3].std()) == pytest.approx(expected, rel=0.2)


def test_hotter_ground_ripples_more() -> None:
    mild = float(np.abs(_params(_shimmer(4000.0, log10_cn2_range=(-14.0, -14.0)))["shift"]).mean())
    hot = float(np.abs(_params(_shimmer(4000.0))["shift"]).mean())
    assert hot == pytest.approx(10.0 * mild, rel=0.05)


def test_the_field_is_smooth_over_its_cells_with_no_mean() -> None:
    field = smooth_field(np.random.default_rng(0), 200, 300, 30.0)
    assert float(field.mean()) == pytest.approx(0.0, abs=1e-5)
    assert float(field.std()) == pytest.approx(1.0, rel=1e-4)
    step = float(np.abs(np.diff(field, axis=1)).mean())
    assert step < 0.1


def test_blur_follows_each_pixel_s_own_sigma() -> None:
    rng = np.random.default_rng(0)
    linear = rng.random((80, 120, 3)).astype(np.float32)
    sigma = np.zeros((80, 120), dtype=np.float32)
    sigma[:, 60:] = 3.0
    out = blur_by_sigma(linear, sigma)
    assert np.array_equal(out[:, :50], linear[:, :50])
    assert float(out[:, 70:].std()) < 0.4 * float(linear[:, 70:].std())
    assert np.array_equal(blur_by_sigma(linear, np.full((80, 120), 0.05, np.float32)), linear)


def test_the_frame_keeps_its_size_and_its_level() -> None:
    import cv2

    small = np.random.default_rng(0).integers(40, 220, (16, 20, 3), dtype=np.uint8)
    image = cv2.resize(small, (200, 160), interpolation=cv2.INTER_CUBIC)
    shimmer = _shimmer(4000.0)
    shimmer.set_random_seed(3)
    out = shimmer(image=image)["image"]
    assert out.shape == image.shape and out.dtype == image.dtype
    assert float(out.mean()) == pytest.approx(float(image.mean()), rel=0.03)
