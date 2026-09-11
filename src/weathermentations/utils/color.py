"""Color-estimation helpers shared by atmosphere effects."""

from __future__ import annotations

import numpy as np


def estimate_airlight(image: np.ndarray, top_percent: float = 0.1) -> np.ndarray:
    """Estimate atmospheric light (airlight) color from an image's brightest pixels.

    Not a full dark-channel-prior implementation (He, Sun & Tang 2009,
    "Single Image Haze Removal Using Dark Channel Prior") — a lightweight
    proxy that's good enough for a training-time augmentation: averages the
    color of the brightest ``top_percent``% of pixels by per-pixel intensity
    sum, which in most outdoor scenes is sky or existing haze.

    Returns an ``(C,)`` float32 color.
    """
    flat = image.reshape(-1, image.shape[-1]).astype(np.float32)
    intensity = flat.sum(axis=1)
    n = max(1, int(len(intensity) * top_percent / 100.0))
    brightest_idx = np.argpartition(intensity, -n)[-n:]
    return flat[brightest_idx].mean(axis=0)
