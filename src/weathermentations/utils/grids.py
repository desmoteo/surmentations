"""Coordinate grids for procedurally drawn patches and full-frame fields.

Every sprite, aperture image, halo and falloff in this library is a function
of pixel coordinates, so these grids are built thousands of times per frame.
``np.mgrid[0:h, 0:w].astype(np.float32)`` — the obvious spelling —
materialises two full ``int64`` arrays and then converts them, which is four
times the memory traffic of what is needed and showed up as a measurable
share of the sun-flare and snow render cost.

The grids here are returned *unbroadcast*, as a ``(h, 1)`` column and a
``(1, w)`` row. Numpy broadcasts them together in the expression that uses
them, so an ``xx - cx`` style term never allocates a full 2-D copy of the
coordinate itself, only of the result.
"""

from __future__ import annotations

import numpy as np


def pixel_grid(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    """``(yy, xx)`` float32 coordinate axes, shaped ``(height, 1)`` and ``(1, width)``."""
    yy = np.arange(height, dtype=np.float32).reshape(height, 1)
    xx = np.arange(width, dtype=np.float32).reshape(1, width)
    return yy, xx


def centered_grid(size: int, scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """``(dy, dx)`` axes for a square patch, centred and divided by ``scale``.

    ``scale`` is normally the patch's own radius, which makes the resulting
    coordinates scale-invariant — the reason a sprite's shape can be built
    once at a canonical size and reused for every size it is drawn at.
    """
    centre = size / 2.0
    axis = (np.arange(size, dtype=np.float32) - centre) / scale
    return axis.reshape(size, 1), axis.reshape(1, size)


def radial_distance(height: int, width: int, cx: float, cy: float) -> np.ndarray:
    """``(height, width)`` float32 Euclidean distance from ``(cx, cy)``."""
    yy, xx = pixel_grid(height, width)
    return np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
