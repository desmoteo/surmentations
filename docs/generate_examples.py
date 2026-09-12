"""Render a spread of parameter values per transform into docs/examples/.

Run with: .venv/bin/python docs/generate_examples.py
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from weathermentations.atmosphere.fog import RandomFog
from weathermentations.atmosphere.rain import RandomRain
from weathermentations.atmosphere.snow import RandomSnow
from weathermentations.depth.estimator import DepthAnythingV2Estimator
from weathermentations.depth.provider import SharedDepthProvider
from weathermentations.lens import DirtSoil, SunFlare, WaterDroplets

OUT_DIR = Path(__file__).parent / "examples"


def _base_scene(height: int = 300, width: int = 400) -> np.ndarray:
    """A simple synthetic scene (sky gradient + ground + a building) for consistent previews."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        t = y / height
        img[y, :] = (
            np.array([135, 180, 235]) * (1 - t) + np.array([230, 220, 200]) * t
        ).astype(np.uint8)
    ground_y = int(height * 0.65)
    img[ground_y:, :] = np.array([90, 110, 70], dtype=np.uint8)
    cv2.rectangle(img, (60, int(height * 0.35)), (180, ground_y), (150, 140, 130), -1)
    cv2.rectangle(img, (220, int(height * 0.45)), (320, ground_y), (120, 110, 100), -1)
    return img


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scene = _base_scene()
    cv2.imwrite(str(OUT_DIR / "00_base_scene.png"), scene)

    depth_provider = SharedDepthProvider(DepthAnythingV2Estimator())
    transforms = {
        "water_droplets": WaterDroplets(p=1.0),
        "dirt_soil": DirtSoil(p=1.0),
        "sun_flare_fixed": SunFlare(p=1.0, flare_position=(0.8, 0.15)),
        "fog": RandomFog(depth_provider=depth_provider, p=1.0, depth_scale=10.0),
        "rain": RandomRain(
            depth_provider=depth_provider, p=1.0, streak_density_range=(250, 250)
        ),
        "snow": RandomSnow(
            depth_provider=depth_provider,
            p=1.0,
            flake_density_range=(400, 400),
            accumulation_probability=1.0,
        ),
    }
    rng = np.random.default_rng(42)
    for name, transform in transforms.items():
        for i in range(3):
            transform.set_random_seed(int(rng.integers(0, 2**31 - 1)))
            out = transform(image=scene)["image"]
            cv2.imwrite(str(OUT_DIR / f"{name}_{i}.png"), out)

    print(f"Wrote examples to {OUT_DIR}")


if __name__ == "__main__":
    main()
