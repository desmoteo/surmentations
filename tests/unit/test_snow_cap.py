"""SnowOnHousing: a cap that hangs from the top of the frame and hides what is behind it."""

from __future__ import annotations

import numpy as np
import pytest

from weathermentations.lens import SnowOnHousing
from weathermentations.lens.snow_cap import cap_edge

SHAPE = (240, 320)


def _params(seed: int = 0, **settings: object) -> dict:
    """What a cap with no patches, unless asked for, would do to a frame."""
    arguments: dict[str, object] = {"patch_count_range": (0, 0), **settings}
    transform = SnowOnHousing(p=1.0, **arguments)
    transform.set_random_seed(seed)
    return transform.get_params_dependent_on_data({"shape": (*SHAPE, 3)}, {})


def _scene() -> np.ndarray:
    """A daylit frame: sky over ground, with noise."""
    rng = np.random.default_rng(0)
    image = np.zeros((*SHAPE, 3), dtype=np.uint8)
    image[:100] = (170, 190, 220)
    image[100:] = (70, 80, 60)
    return np.clip(image + rng.integers(-20, 21, image.shape), 0, 255).astype(np.uint8)


def test_the_cap_hangs_from_the_top_as_deep_as_asked() -> None:
    for depth in (0.1, 0.3):
        covered = [
            float((_params(seed, cap_depth_range=(depth, depth))["cover"] > 0.5).mean())
            for seed in range(6)
        ]
        assert np.mean(covered) == pytest.approx(depth, rel=0.2)
    cover = _params(1, cap_depth_range=(0.25, 0.25))["cover"]
    assert cover[:10].mean() > 0.9
    assert cover[-60:].max() < 0.02


def test_a_lumpy_edge_is_uneven_and_a_smooth_one_is_not() -> None:
    def unevenness(lumpiness: float) -> float:
        """Spread of how far the cap hangs, across the frame."""
        return float(
            np.mean([cap_edge(np.random.default_rng(s), 320, lumpiness).std() for s in range(8)])
        )

    assert unevenness(0.0) < 1e-6
    assert unevenness(0.9) > 3.0 * unevenness(0.2) > 0.0
    assert cap_edge(np.random.default_rng(0), 320, 0.7).mean() == pytest.approx(1.0, abs=1e-5)


def test_the_edge_is_a_gradient_not_a_line() -> None:
    cover = _params(
        2, cap_depth_range=(0.3, 0.3), lumpiness_range=(0.0, 0.0), defocus_range=(0.03, 0.03)
    )["cover"]
    column = cover[:, 160]
    partly = int(((column > 0.1) & (column < 0.9)).sum())
    assert partly > 15


def test_snow_hides_the_scene_and_shows_no_detail() -> None:
    image = _scene()
    transform = SnowOnHousing(p=1.0, cap_depth_range=(0.3, 0.3), patch_count_range=(0, 0))
    transform.set_random_seed(0)
    out = transform(image=image)["image"].astype(np.float32)
    top = out[:30]
    assert float(np.abs(np.diff(top, axis=1)).mean()) < 1.0
    assert np.array_equal(out[-40:], image[-40:].astype(np.float32))


def test_by_day_the_underside_is_darker_where_the_cap_is_thicker() -> None:
    # Dim enough that the snow, lit by the same light, does not clip.
    image = np.full((*SHAPE, 3), 70, dtype=np.uint8)
    transform = SnowOnHousing(
        p=1.0, cap_depth_range=(0.5, 0.5), lumpiness_range=(0.0, 0.0), patch_count_range=(0, 0)
    )
    transform.set_random_seed(0)
    out = transform(image=image)["image"]
    assert out[2, 160, 0] < out[100, 160, 0]


def test_by_day_snow_is_pale_and_a_little_blue_and_its_surface_uneven() -> None:
    image = np.full((*SHAPE, 3), 70, dtype=np.uint8)
    transform = SnowOnHousing(
        p=1.0, cap_depth_range=(0.5, 0.5), lumpiness_range=(0.0, 0.0), patch_count_range=(0, 0)
    )
    transform.set_random_seed(0)
    out = transform(image=image)["image"].astype(np.float32)
    snow = out[:60]
    assert snow.mean() > 1.3 * 70.0
    assert snow[..., 2].mean() > snow[..., 0].mean()
    assert float(snow[..., 1].std()) > 1.0


def test_under_the_lamp_snow_clips() -> None:
    night = np.full((*SHAPE, 3), 20, dtype=np.uint8)
    transform = SnowOnHousing(
        p=1.0, cap_depth_range=(0.3, 0.3), patch_count_range=(0, 0), glow_range=(2.0, 2.0)
    )
    transform.set_random_seed(0)
    out = transform(image=night)["image"]
    assert (out[:20] == 255).all()
    assert np.array_equal(out[-40:], night[-40:])


def test_patches_cover_more_than_the_cap_alone() -> None:
    alone = _params(3, cap_depth_range=(0.1, 0.1))["cover"]
    with_patches = _params(3, cap_depth_range=(0.1, 0.1), patch_count_range=(5, 5))["cover"]
    assert with_patches.sum() > 1.2 * alone.sum()


def test_no_snow_leaves_the_frame_alone() -> None:
    image = _scene()
    transform = SnowOnHousing(p=1.0, cap_depth_range=(0.0, 0.0), patch_count_range=(0, 0))
    transform.set_random_seed(0)
    assert np.array_equal(transform(image=image)["image"], image)
