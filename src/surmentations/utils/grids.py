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

from collections.abc import Callable

import cv2
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


# Samples a coarse grid extends past the frame on every side, so that the
# enlargement interpolates at the frame's edge as it does inside it.
_COARSE_MARGIN = 2


def on_coarse_grid(
    height: int,
    width: int,
    field: Callable[[np.ndarray, np.ndarray], np.ndarray],
    reduction: int = 4,
) -> np.ndarray:
    """``field(yy, xx)`` for every pixel of the frame, worked out on a coarse grid.

    For a field that changes over tens of pixels: a lamp's beam, a veiling
    glare. ``field`` is given a column of row coordinates and a row of
    column coordinates, as :func:`pixel_grid` gives them, and is evaluated
    at the centre of each block of ``reduction`` by ``reduction`` pixels;
    the result is enlarged bilinearly to ``(height, width)`` float32. That
    is ``reduction`` squared times fewer evaluations, and as close to the
    field at every pixel as bilinear interpolation over a block allows: for
    the night illuminator's beam, within 0.3% of its peak, and that only
    where the beam's profile is clipped at the corners of the frame. Each
    field it is used for has its bound tested in ``tests/unit/test_modes.py``.
    """
    if reduction <= 1:
        yy, xx = pixel_grid(height, width)
        return np.asarray(field(yy, xx), dtype=np.float32) * np.ones((height, width), np.float32)
    margin = _COARSE_MARGIN
    work_h = -(-height // reduction) + 2 * margin
    work_w = -(-width // reduction) + 2 * margin
    # Where each coarse sample sits on the frame, by the convention the
    # enlargement below reads them back with.
    ys = ((np.arange(work_h, dtype=np.float32) - margin + 0.5) * reduction - 0.5).reshape(-1, 1)
    xs = ((np.arange(work_w, dtype=np.float32) - margin + 0.5) * reduction - 0.5).reshape(1, -1)
    small = np.ascontiguousarray(np.broadcast_to(field(ys, xs), (work_h, work_w)), np.float32)
    large = cv2.resize(
        small, (work_w * reduction, work_h * reduction), interpolation=cv2.INTER_LINEAR
    )
    edge = margin * reduction
    return np.ascontiguousarray(large[edge : edge + height, edge : edge + width])
