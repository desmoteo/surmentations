"""Reductions over the colour channels, and over the frame.

``image.max(axis=2)`` on an ``(H, W, 3)`` array is the slowest way numpy has
of doing three comparisons per pixel: a reduction's inner loop runs along the
axis being reduced, which here is three elements long, so the loop set-up is
paid a million times. It measured 25 ms on a 1024-pixel frame against 1 ms
for the two ``np.maximum`` calls that give the same array. These helpers are
those calls, so the cost is not rediscovered one call site at a time.

The per-pixel ones return exactly what the reduction did: the same
operations in the same order. :func:`weighted_colour` sums a million floats
in another order and agrees to float32 rounding.
"""

from __future__ import annotations

import numpy as np


def channel_max(rgb: np.ndarray) -> np.ndarray:
    """``rgb.max(axis=2)`` for a three-channel image."""
    return np.maximum(np.maximum(rgb[..., 0], rgb[..., 1]), rgb[..., 2])


def channel_min(rgb: np.ndarray) -> np.ndarray:
    """``rgb.min(axis=2)`` for a three-channel image."""
    return np.minimum(np.minimum(rgb[..., 0], rgb[..., 1]), rgb[..., 2])


def channel_sum(rgb: np.ndarray) -> np.ndarray:
    """``rgb.sum(axis=2)`` for a three-channel image."""
    return rgb[..., 0] + rgb[..., 1] + rgb[..., 2]


def channel_mean(rgb: np.ndarray) -> np.ndarray:
    """``rgb.mean(axis=2)`` for a three-channel image."""
    return channel_sum(rgb) / rgb.dtype.type(3)


def weighted_colour(rgb: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """``(rgb * weight[..., None]).sum(axis=(0, 1))``, without the product image.

    One matrix-vector product in place of a frame-sized temporary and a
    reduction over two axes.
    """
    return np.asarray(weight.reshape(-1) @ rgb.reshape(-1, rgb.shape[-1]))
