import numpy as np
import pytest


def build_scene() -> np.ndarray:
    """Deterministic synthetic scene shared by every golden-image test."""
    height, width = 128, 160
    img = np.zeros((height, width, 3), dtype=np.uint8)
    rng = np.random.default_rng(1234)
    for y in range(height):
        t = y / height
        img[y, :] = (np.array([135, 180, 235]) * (1 - t) + np.array([230, 220, 200]) * t).astype(
            np.uint8
        )
    img[int(height * 0.65) :, :] = np.array([90, 110, 70], dtype=np.uint8)
    noise = rng.integers(-3, 4, size=img.shape)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


@pytest.fixture
def scene() -> np.ndarray:
    return build_scene()
