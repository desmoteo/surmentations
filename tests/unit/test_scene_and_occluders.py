"""Cast shadows and housing occluders (plan.md 6.2 / Phase 7)."""

from __future__ import annotations

import numpy as np
import pytest

from weathermentations.depth.estimator import DepthEstimator
from weathermentations.depth.provider import SharedDepthProvider
from weathermentations.lens import HousingOccluders
from weathermentations.scene import CastShadows

_FOV_DEG = 60.0
_MAX_DEPTH_M = 50.0


class _GroundPlaneDepth(DepthEstimator):
    """Level camera above a horizontal plane, sky above the horizon."""

    def estimate(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        focal = width / (2.0 * np.tan(np.deg2rad(_FOV_DEG) / 2.0))
        cy = height / 2.0
        rows = np.arange(height, dtype=np.float32).reshape(height, 1)
        depth = np.where(
            rows > cy + 2, 1.5 * focal / np.maximum(rows - cy, 1e-3), _MAX_DEPTH_M
        )
        return np.broadcast_to(
            np.minimum(depth, _MAX_DEPTH_M) / _MAX_DEPTH_M, (height, width)
        ).astype(np.float32)


@pytest.fixture
def scene() -> np.ndarray:
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    img[:120] = (150, 170, 200)  # sky
    img[120:] = (120, 118, 110)  # ground
    return img


def test_shadow_darkens_without_flattening_texture(scene: np.ndarray) -> None:
    """A shadow removes the direct beam and leaves the surface's albedo and
    texture intact. Multiplying radiance does that; blending toward grey
    destroys it, which is the tell-tale of a drawn-on shadow.

    So a textured patch must keep its *relative* contrast when shadowed, even
    though its absolute level drops.
    """
    rng = np.random.default_rng(0)
    textured = scene.copy()
    noise = rng.integers(-25, 26, textured.shape, dtype=np.int16)
    textured = np.clip(textured.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    t = CastShadows(
        depth_provider=SharedDepthProvider(_GroundPlaneDepth()),
        p=1.0,
        shadow_count_range=(6, 6),
        shadow_scale_range=(0.3, 0.3),
        direct_fraction_range=(0.7, 0.7),
    )
    t.set_random_seed(0)
    out = t(image=textured)["image"].astype(np.float32)
    before = textured.astype(np.float32)

    ground_before, ground_after = before[170:230], out[170:230]
    assert ground_after.mean() < ground_before.mean() - 5.0

    # Relative contrast survives: the ratio of spread to level is preserved.
    ratio_before = ground_before.std() / ground_before.mean()
    ratio_after = ground_after.std() / ground_after.mean()
    assert ratio_after > 0.75 * ratio_before


def test_shadows_do_not_land_on_sky(scene: np.ndarray) -> None:
    """Shadows fall on surfaces. The sky is not a surface, and a shadow drawn
    across it is the most obvious possible giveaway."""
    t = CastShadows(
        depth_provider=SharedDepthProvider(_GroundPlaneDepth()),
        p=1.0,
        shadow_count_range=(8, 8),
        shadow_scale_range=(0.35, 0.35),
        direct_fraction_range=(0.9, 0.9),
    )
    t.set_random_seed(0)
    out = t(image=scene)["image"].astype(np.float32)

    sky_change = np.abs(out[:100] - scene[:100].astype(np.float32)).mean()
    ground_change = np.abs(out[160:] - scene[160:].astype(np.float32)).mean()
    assert sky_change < 1.0
    assert ground_change > 5.0


def test_shadow_is_bluer_than_the_light_it_replaces(scene: np.ndarray) -> None:
    """What survives in shadow is skylight, which is blue. A daylight shadow is
    therefore measurably bluer than its surroundings, and rendering it as
    neutral grey misses that."""
    t = CastShadows(
        depth_provider=SharedDepthProvider(_GroundPlaneDepth()),
        p=1.0,
        shadow_count_range=(8, 8),
        shadow_scale_range=(0.4, 0.4),
        direct_fraction_range=(0.85, 0.85),
    )
    t.set_random_seed(0)
    out = t(image=scene)["image"].astype(np.float32)
    before = scene.astype(np.float32)

    ground_before, ground_after = before[160:], out[160:]
    # Blue survives better than red, so the red/blue ratio falls.
    ratio_before = ground_before[..., 0].mean() / max(ground_before[..., 2].mean(), 1e-6)
    ratio_after = ground_after[..., 0].mean() / max(ground_after[..., 2].mean(), 1e-6)
    assert ratio_after < ratio_before


def test_occluders_are_soft_edged(scene: np.ndarray) -> None:
    """An object on the dome is centimetres from a lens focused metres away, so
    it cannot have a sharp edge. Anything crisp here would be wrong."""
    t = HousingOccluders(
        p=1.0,
        insect_count_range=(3, 3),
        web_count_range=(0, 0),
        defocus_range=(0.03, 0.03),
        opacity_range=(0.9, 0.9),
        glow_range=(0.0, 0.0),
    )
    t.set_random_seed(0)
    params = t.get_params_dependent_on_data({"shape": scene.shape}, {})
    mask = params["occluder"]

    assert mask.max() > 0.05, "no occluder was drawn"
    # A hard-edged mask would have gradients approaching its own peak value;
    # a heavily defocused one cannot.
    gradient = np.abs(np.diff(mask, axis=1)).max()
    assert gradient < 0.1 * mask.max()


def test_occluders_darken_by_day_and_glow_under_ir(scene: np.ndarray) -> None:
    """Same geometry, opposite appearance: backlit by the scene they
    silhouette, lit from behind by the IR ring they glow. One parameter covers
    both because one mechanism does."""

    def mean_change(glow: float) -> float:
        t = HousingOccluders(
            p=1.0,
            insect_count_range=(4, 4),
            web_count_range=(1, 1),
            opacity_range=(0.85, 0.85),
            glow_range=(glow, glow),
        )
        t.set_random_seed(1)
        out = t(image=scene)["image"].astype(np.float32)
        return float(out.mean() - scene.astype(np.float32).mean())

    assert mean_change(0.0) < -1.0  # daylight: darker
    assert mean_change(2.0) > 1.0  # IR: brighter
