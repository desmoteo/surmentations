"""Reductions over the colour channels, and over the frame.

``image.max(axis=2)`` on an ``(H, W, 3)`` array is the slowest way numpy has
of doing three comparisons per pixel: a reduction's inner loop runs along the
axis being reduced, which here is three elements long, so the loop set-up is
paid a million times. It measured 25 ms on a 1024-pixel frame against 1 ms
for the two ``np.maximum`` calls that give the same array. These helpers are
those calls, so the cost is not rediscovered one call site at a time.

The per-pixel ones return exactly what the reduction did: the same
operations in the same order. :func:`weighted_colour` has a fast form that
sums in another order and agrees to float32 rounding, and keeps the
reduction for exact mode (:mod:`surmentations.mode`).
"""

from __future__ import annotations

import cv2
import numpy as np

from surmentations import mode


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


def frame_mean(image: np.ndarray) -> np.ndarray:
    """``image.reshape(-1, C).mean(axis=0)``: the mean colour of a frame.

    The reduction runs down a column of a million three-element rows, which
    numpy does without its pairwise blocks or its vector loops: 13 ms on a
    1024-pixel frame, and a million float32 added one after another, good
    to five digits. Fast mode asks OpenCV, which takes 0.6 ms and adds them
    up in double. The two agree to those five digits, and where they differ
    it is the reference that is off. Exact mode is the expression above.
    """
    channels = image.shape[-1]
    if mode.fast() and image.dtype == np.float32 and image.ndim == 3 and channels <= 4:
        return np.asarray(cv2.mean(image)[:channels], dtype=np.float32)
    return np.asarray(image.reshape(-1, channels).mean(axis=0))


def weighted_colour(rgb: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """``(rgb * weight[..., None]).sum(axis=(0, 1))``: a weighted sum per channel.

    Fast mode takes it as one matrix-vector product, without the frame-sized
    product image or a reduction over two axes: a twelfth of the time, and a
    sum of a million floats in another order, so the last digits differ.
    Exact mode is the expression above.
    """
    if not mode.fast():
        return np.asarray((rgb * weight[..., None]).sum(axis=(0, 1)))
    return np.asarray(weight.reshape(-1) @ rgb.reshape(-1, rgb.shape[-1]))


# Bins of the histogram :func:`percentile` reads a percentile from in fast mode.
_PERCENTILE_BINS = 16384


def percentile(values: np.ndarray, q: float) -> float:
    """``np.percentile(values, q)`` of a float32 field, by counting in fast mode.

    numpy finds a percentile with a partial sort, 6 ms for a million values.
    Fast mode counts them into 16384 bins over their range, finds the bin
    the percentile falls in, and interpolates within it by rank: one pass,
    and the value to within a 16384th of the field's range, which for the
    exposure scales it is used for is a few parts in a hundred thousand.
    Exact mode, and any other dtype, is ``np.percentile``.
    """
    if not mode.fast() or values.dtype != np.float32 or values.size < 2:
        return float(np.percentile(values, q))
    field = (
        np.ascontiguousarray(values).reshape(values.shape[0], -1)
        if values.ndim > 1
        else (np.ascontiguousarray(values).reshape(1, -1))
    )
    low, high, _, _ = cv2.minMaxLoc(field)
    if not high > low:
        return float(low)
    # The upper end is past the largest value, which would otherwise fall
    # outside the last bin and not be counted.
    width = (high - low) * (1.0 + 1e-6) / _PERCENTILE_BINS
    counts = cv2.calcHist(
        [field], [0], None, [_PERCENTILE_BINS], [low, low + width * _PERCENTILE_BINS]
    ).ravel()
    # numpy's rank of the percentile, counted from 0, and the bin it is in.
    rank = q / 100.0 * (field.size - 1)
    reached = np.cumsum(counts)
    found = int(np.searchsorted(reached, rank, side="right"))
    found = min(found, _PERCENTILE_BINS - 1)
    before = float(reached[found] - counts[found])
    within = (rank - before + 0.5) / max(float(counts[found]), 1.0)
    return float(low + (found + min(max(within, 0.0), 1.0)) * width)
