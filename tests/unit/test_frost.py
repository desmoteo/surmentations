"""FrostOnGlass: ice that grows in from the rim and scatters what it covers."""

from __future__ import annotations

import numpy as np
import pytest

from weathermentations.lens import FrostOnGlass
from weathermentations.lens.frost import frost_depth

SHAPE = (240, 320)


def _depth(seed: int = 0, **settings: float) -> np.ndarray:
    """Frost's optical depth for one seed: half-grown, of middling depth and feather unless said."""
    values = {"coverage": 0.5, "thickness": 2.0, "feather": 0.5, **settings}
    return frost_depth(np.random.default_rng(seed), *SHAPE, **values)


def _scene() -> np.ndarray:
    """A daylit frame with detail in it: bright above, darker and textured below."""
    rng = np.random.default_rng(0)
    image = np.zeros((*SHAPE, 3), dtype=np.uint8)
    image[:100] = (170, 190, 220)
    image[100:] = (70, 80, 60)
    return np.clip(image + rng.integers(-25, 26, image.shape), 0, 255).astype(np.uint8)


def test_frost_grows_in_from_the_rim_and_the_centre_is_last() -> None:
    for seed in range(4):
        depth = _depth(seed)
        rim = np.concatenate([depth[:12].ravel(), depth[-12:].ravel(), depth[:, :12].ravel()])
        centre = depth[100:140, 140:180]
        assert float(centre.max()) == 0.0
        assert float(rim.mean()) > 0.5


def test_more_coverage_frosts_more_of_the_frame() -> None:
    frosted = [float((_depth(1, coverage=c) > 0.2).mean()) for c in (0.2, 0.5, 0.8, 1.1)]
    assert frosted == sorted(frosted)
    assert frosted[0] < 0.5 < frosted[-1]


def test_feathering_makes_plumes_that_point_at_the_centre() -> None:
    def roughness(feather: float) -> tuple[float, float]:
        """How much the frost varies along the top edge, and down the frame under its middle."""
        depth = _depth(2, coverage=1.0, feather=feather)
        across = depth[20, 60:260]
        down = depth[5:45, 160]
        return float(across.std() / across.mean()), float(down.std() / down.mean())

    even_across, _ = roughness(0.0)
    across, down = roughness(1.0)
    assert across > 3.0 * max(even_across, 0.02)
    # A plume runs inwards, so the frost changes faster across them than along.
    assert across > 1.5 * down


def test_frost_scatters_and_does_not_darken() -> None:
    image = _scene()
    transform = FrostOnGlass(p=1.0, coverage_range=(1.2, 1.2), thickness_range=(4.0, 4.0))
    transform.set_random_seed(0)
    out = transform(image=image)["image"].astype(np.float32)
    detail_before = float(np.abs(np.diff(image.astype(np.float32), axis=1)).mean())
    detail_after = float(np.abs(np.diff(out, axis=1)).mean())
    assert detail_after < 0.35 * detail_before
    assert out.mean() > 0.9 * image.mean()


def test_the_light_it_picks_up_makes_frost_paler() -> None:
    image = _scene()
    means = []
    for ambient in (0.0, 0.3):
        transform = FrostOnGlass(
            p=1.0, coverage_range=(1.0, 1.0), ambient_light_range=(ambient, ambient)
        )
        transform.set_random_seed(3)
        means.append(float(transform(image=image)["image"].mean()))
    assert means[1] > means[0] + 10.0


def test_under_the_lamp_frost_glows_and_the_clear_patch_does_not() -> None:
    night = np.full((*SHAPE, 3), 20, dtype=np.uint8)
    transform = FrostOnGlass(
        p=1.0,
        coverage_range=(0.5, 0.5),
        thickness_range=(3.0, 3.0),
        feather_range=(0.0, 0.0),
        illuminator_glow_range=(1.5, 1.5),
    )
    transform.set_random_seed(0)
    out = transform(image=night)["image"]
    assert out[:8].mean() > 200
    assert np.array_equal(out[110:130, 150:170], night[110:130, 150:170])


def test_no_frost_leaves_the_frame_alone() -> None:
    image = _scene()
    transform = FrostOnGlass(p=1.0, thickness_range=(0.0, 0.0))
    transform.set_random_seed(0)
    assert np.array_equal(transform(image=image)["image"], image)


def test_depth_is_never_negative_and_scales_with_thickness() -> None:
    thin, thick = _depth(5, thickness=1.0), _depth(5, thickness=3.0)
    assert float(thin.min()) >= 0.0
    assert np.allclose(thick, 3.0 * thin, rtol=1e-5, atol=1e-6)
    assert float(thick.max()) == pytest.approx(3.0, rel=0.5)
