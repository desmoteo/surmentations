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

from surmentations.atmosphere.fog import RandomFog
from surmentations.atmosphere.rain import RandomRain
from surmentations.atmosphere.snow import RandomSnow
from surmentations.depth.estimator import DepthEstimator
from surmentations.depth.provider import SharedDepthProvider
from surmentations.lens import DirtSoil, SunFlare, WaterDroplets

GOLDEN_DIR = Path(__file__).parent / "golden"


class _LinearDepthEstimator(DepthEstimator):
    """Deterministic, GPU-free depth stand-in: 0 (top row) to 1 (bottom row).

    Used only so RandomFog's *compositing* logic has a golden-image test that
    runs on any CI runner. The real DepthAnythingV2Estimator is covered
    separately by CUDA-gated tests in test_depth.py.
    """

    def estimate(self, image: np.ndarray) -> np.ndarray:
        """Depth rising linearly from 0 at the top row to 1 at the bottom."""
        height, width = image.shape[:2]
        return np.tile(np.linspace(0.0, 1.0, height)[:, None], (1, width)).astype(np.float32)


class _GroundPlaneDepthEstimator(DepthEstimator):
    """A level camera above a horizontal ground plane, receding to a horizon.

    Snow's accumulation is derived from surface *orientation*, so it needs a
    depth map describing a plausible surface. The linear ramp above has its
    far end at the bottom of the frame, which back-projects to a ceiling and
    correctly collects no snow — leaving the golden with nothing to pin.
    """

    def estimate(self, image: np.ndarray) -> np.ndarray:
        """Depth of level ground seen from 1.5 m up, horizon at mid-frame, 50 m above it.

        Each row's distance follows from the camera height and the focal
        length of a 60-degree field of view, as a real road's would.
        """
        height, width = image.shape[:2]
        focal = width / (2.0 * np.tan(np.deg2rad(60.0) / 2.0))
        cy = height / 2.0
        rows = np.arange(height, dtype=np.float32).reshape(height, 1)
        depth = np.where(rows > cy + 2, 1.5 * focal / np.maximum(rows - cy, 1e-3), 50.0)
        return np.broadcast_to(np.minimum(depth, 50.0) / 50.0, (height, width)).astype(np.float32)


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
        max_streaks=400,
        wetness_probability=1.0,
    ),
    "snow": RandomSnow(
        depth_provider=SharedDepthProvider(_GroundPlaneDepthEstimator()),
        p=1.0,
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
