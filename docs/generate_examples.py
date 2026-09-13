"""Render a spread of parameter values per transform into docs/examples/.

Run with: .venv/bin/python docs/generate_examples.py
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from weathermentations.atmosphere.fog import RandomFog
from weathermentations.atmosphere.rain import RandomRain
from weathermentations.atmosphere.snow import RandomSnow
from weathermentations.depth.estimator import DepthAnythingV2Estimator
from weathermentations.depth.provider import SharedDepthProvider
from weathermentations.lens import DirtSoil, SunFlare, WaterDroplets

OUT_DIR = Path(__file__).parent / "examples"
REFERENCE_IMAGE = OUT_DIR / "reference.webp"


def _base_scene(max_dim: int = 640) -> np.ndarray:
    """Real reference photo (docs/examples/reference.webp), downscaled and in RGB.

    A real highway/traffic scene with genuine depth structure (near road ->
    mid-ground cars/guardrail -> distant buildings/skyline), which is what the
    depth-conditioned effects (fog/rain/snow) actually need to look
    convincing, unlike a flat synthetic scene.
    """
    img_bgr = cv2.imread(str(REFERENCE_IMAGE))
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read reference image at {REFERENCE_IMAGE}")
    height, width = img_bgr.shape[:2]
    scale = max_dim / max(height, width)
    img_bgr = cv2.resize(img_bgr, (int(width * scale), int(height * scale)))
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scene = _base_scene()
    cv2.imwrite(str(OUT_DIR / "00_base_scene.png"), cv2.cvtColor(scene, cv2.COLOR_RGB2BGR))

    depth_provider = SharedDepthProvider(DepthAnythingV2Estimator())
    transforms = {
        "water_droplets": WaterDroplets(p=1.0),
        "dirt_soil": DirtSoil(p=1.0),
        "sun_flare_fixed": SunFlare(p=1.0, flare_position=None),
        "fog": RandomFog(depth_provider=depth_provider, p=1.0, depth_scale=60.0),
        "rain": RandomRain(depth_provider=depth_provider, p=1.0),
        "snow": RandomSnow(
            depth_provider=depth_provider, p=1.0, accumulation_probability=1.0
        ),
    }
    rng = np.random.default_rng(42)
    # Most of the time here goes into depth estimation, so show progress
    # rather than appearing to hang.
    for name, transform in tqdm(transforms.items(), total=len(transforms), desc="effects"):
        for i in range(5):
            transform.set_random_seed(int(rng.integers(0, 2**31 - 1)))
            out = transform(image=scene)["image"]
            cv2.imwrite(str(OUT_DIR / f"{name}_{i}.png"), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))

    print(f"Wrote examples to {OUT_DIR}")


if __name__ == "__main__":
    main()
