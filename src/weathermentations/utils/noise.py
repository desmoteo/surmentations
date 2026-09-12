"""Procedural noise helpers for irregular blob/cluster shapes (dirt, snow accumulation, ...).

Uses value noise (bilinearly-upsampled random grids, summed octave-by-octave)
rather than a Perlin/simplex library dependency — it's a handful of lines of
numpy and is visually indistinguishable from Perlin noise for the soft,
irregular cluster masks this library needs.
"""

from __future__ import annotations

import cv2
import numpy as np


def value_noise(
    height: int,
    width: int,
    *,
    rng: np.random.Generator,
    octaves: int = 4,
    base_cells: int = 4,
    persistence: float = 0.5,
) -> np.ndarray:
    """Fractal value noise in ``[0, 1]``, shape ``(height, width)``, float32.

    Each octave draws an independent random grid at ``base_cells * 2**i``
    resolution, upsampled to full size with bicubic interpolation, and sums
    them with amplitude falling off by ``persistence`` per octave.
    """
    result = np.zeros((height, width), dtype=np.float32)
    amplitude = 1.0
    total_amplitude = 0.0
    for octave in range(octaves):
        cells = max(2, base_cells * (2**octave))
        grid = rng.random((cells, cells)).astype(np.float32)
        layer = cv2.resize(grid, (width, height), interpolation=cv2.INTER_CUBIC)
        result += amplitude * layer
        total_amplitude += amplitude
        amplitude *= persistence
    result /= total_amplitude
    return np.clip(result, 0.0, 1.0)


def blob_cluster_mask(
    height: int,
    width: int,
    *,
    rng: np.random.Generator,
    count: int,
    size_range: tuple[float, float],
    irregularity: float = 1.5,
) -> np.ndarray:
    """Soft alpha mask (float32, ``[0, 1]``) with ``count`` irregular noise-based blobs.

    Each blob is a localized patch of value noise modulated by a radial
    falloff, normalized so it always peaks at 1.0 regardless of the random
    noise draw, then raised to ``irregularity`` (>1 shrinks the blob toward
    an irregular core; 1 keeps the raw soft-edged shape). Blobs are combined
    with a max so overlapping blobs don't double-darken.
    """
    mask = np.zeros((height, width), dtype=np.float32)
    for _ in range(count):
        radius = float(rng.uniform(*size_range))
        cx = rng.uniform(0, width)
        cy = rng.uniform(0, height)

        patch_size = max(4, int(round(radius * 2)))
        noise = value_noise(patch_size, patch_size, rng=rng, octaves=3, base_cells=3)

        yy, xx = np.mgrid[0:patch_size, 0:patch_size].astype(np.float32)
        center = patch_size / 2.0
        dist = np.sqrt((xx - center) ** 2 + (yy - center) ** 2) / center
        falloff = np.clip(1.0 - dist, 0.0, 1.0) ** 0.5

        raw = noise * falloff
        peak = raw.max()
        blob = raw / peak if peak > 1e-6 else raw
        blob = np.clip(blob, 0.0, 1.0) ** irregularity

        x0, y0 = int(round(cx - center)), int(round(cy - center))
        x1, y1 = x0 + patch_size, y0 + patch_size
        src_x0, src_y0 = max(0, -x0), max(0, -y0)
        dst_x0, dst_y0 = max(0, x0), max(0, y0)
        dst_x1, dst_y1 = min(width, x1), min(height, y1)
        if dst_x1 <= dst_x0 or dst_y1 <= dst_y0:
            continue
        src_x1 = src_x0 + (dst_x1 - dst_x0)
        src_y1 = src_y0 + (dst_y1 - dst_y0)

        region = mask[dst_y0:dst_y1, dst_x0:dst_x1]
        np.maximum(region, blob[src_y0:src_y1, src_x0:src_x1], out=region)

    return mask
