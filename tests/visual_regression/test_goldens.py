"""Pixel-diff regression tests against fixed-seed golden images.

These catch unintended rendering changes (stability), not realism — realism
is checked separately via qualitative comparison and by eye during
development. Regenerate goldens deliberately with `generate_goldens.py` when
an effect's rendering intentionally changes.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from weathermentations.atmosphere.fog import RandomFog
from weathermentations.atmosphere.rain import RandomRain
from weathermentations.atmosphere.snow import RandomSnow
from weathermentations.depth.estimator import DepthEstimator
from weathermentations.depth.provider import SharedDepthProvider
from weathermentations.lens import DirtSoil, SunFlare, WaterDroplets

GOLDEN_DIR = Path(__file__).parent / "golden"


class _LinearDepthEstimator(DepthEstimator):
    """Deterministic, GPU-free depth stand-in: 0 (top row) to 1 (bottom row).

    Used only so RandomFog's *compositing* logic has a golden-image test that
    runs on any CI runner. The real DepthAnythingV2Estimator is covered
    separately by CUDA-gated tests in test_depth.py.
    """

    def estimate(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        return np.tile(np.linspace(0.0, 1.0, height)[:, None], (1, width)).astype(np.float32)


TRANSFORMS = {
    "water_droplets": WaterDroplets(p=1.0),
    "dirt_soil": DirtSoil(p=1.0),
    "sun_flare": SunFlare(p=1.0, flare_position=(0.8, 0.15)),
    "fog": RandomFog(
        depth_provider=SharedDepthProvider(_LinearDepthEstimator()),
        p=1.0,
        beta_range=(0.05, 0.05),
    ),
    "rain": RandomRain(
        depth_provider=SharedDepthProvider(_LinearDepthEstimator()),
        p=1.0,
        streak_density_range=(100, 100),
        wetness_probability=1.0,
    ),
    "snow": RandomSnow(
        depth_provider=SharedDepthProvider(_LinearDepthEstimator()),
        p=1.0,
        flake_density_range=(150, 150),
        accumulation_probability=1.0,
    ),
}

# Mean absolute per-pixel difference (0-255 scale) tolerated before a golden
# test fails. Not zero, since float rounding can differ trivially across
# OpenCV/numpy versions; large enough to catch an actually-different render.
MAX_MEAN_ABS_DIFF = 1.5


def test_goldens_exist() -> None:
    missing = [name for name in TRANSFORMS if not (GOLDEN_DIR / f"{name}.png").exists()]
    assert not missing, (
        f"Missing golden images for {missing}; run "
        "`python tests/visual_regression/generate_goldens.py` once to create them."
    )


def test_matches_golden(scene: np.ndarray) -> None:
    for name, transform in TRANSFORMS.items():
        golden_path = GOLDEN_DIR / f"{name}.png"
        if not golden_path.exists():
            continue  # covered by test_goldens_exist
        golden = cv2.imread(str(golden_path))

        transform.set_random_seed(0)
        out = transform(image=scene)["image"]

        diff = np.abs(out.astype(np.float32) - golden.astype(np.float32)).mean()
        assert diff <= MAX_MEAN_ABS_DIFF, f"{name}: golden mismatch (mean abs diff={diff:.3f})"
