"""Shared alpha-compositing helpers used by every effect in this library."""

from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np

from surmentations.utils.grids import radial_distance


def alpha_composite(
    base: np.ndarray,
    overlay_color: np.ndarray | tuple[float, float, float],
    alpha: np.ndarray,
) -> np.ndarray:
    """Standard "over" alpha composite: ``out = base * (1 - alpha) + overlay * alpha``.

    Args:
        base: ``(H, W, C)`` image, any numeric dtype.
        overlay_color: either a single ``(C,)`` color broadcast over the whole
            image, or a full ``(H, W, C)`` array (e.g. a per-pixel refracted
            sample for water droplets).
        alpha: ``(H, W)`` or ``(H, W, 1)`` array in ``[0, 1]``.

    Returns:
        Composited image in the same dtype as ``base``, values clipped to the
        dtype's valid range.
    """
    dtype = base.dtype
    base_f = base.astype(np.float32)
    overlay_f = np.asarray(overlay_color, dtype=np.float32)
    if overlay_f.ndim == 1:
        overlay_f = np.broadcast_to(overlay_f, base_f.shape)
    alpha_f = np.asarray(alpha, dtype=np.float32)
    if alpha_f.ndim == 2:
        alpha_f = alpha_f[..., None]

    out = base_f * (1.0 - alpha_f) + overlay_f * alpha_f

    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        out = np.clip(out, info.min, info.max)
    else:
        out = np.clip(out, 0.0, 1.0)
    return out.astype(dtype)


def normalized_radial_falloff(height: int, width: int, cx: float, cy: float) -> np.ndarray:
    """``(H, W)`` float32 field in ``[0, 1]``: 1 at ``(cx, cy)``, 0 at the farthest corner."""
    dist = radial_distance(height, width, cx, cy)
    corners = [(0, 0), (width, 0), (0, height), (width, height)]
    max_dist = max(((cx - x) ** 2 + (cy - y) ** 2) ** 0.5 for x, y in corners)
    max_dist = max(max_dist, 1e-6)
    return np.clip(1.0 - dist / max_dist, 0.0, 1.0)


def directional_blur_kernel(length: int, angle_deg: float) -> np.ndarray:
    """``(length, length)`` normalized kernel blurring along ``angle_deg`` from vertical.

    Approximates exposure-time motion blur for a falling particle (rain
    streak, snowflake) moving at a slant from straight down.
    """
    length = max(3, length | 1)  # force odd
    kernel = np.zeros((length, length), dtype=np.float32)
    kernel[length // 2, :] = 1.0
    rot = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), 90 - angle_deg, 1.0)
    rotated: np.ndarray = cv2.warpAffine(kernel, rot, (length, length))
    total = rotated.sum()
    return rotated / total if total > 1e-6 else rotated


def feathered_band_blend(
    img_f: np.ndarray, band_fraction: float, blend: Callable[[np.ndarray], np.ndarray]
) -> np.ndarray:
    """Blend the bottom ``band_fraction`` of an image toward ``blend(band)``, feathered in.

    Used for effects that only apply near the "ground" (rain's wet-ground
    reflection, snow's accumulation overlay): a hard cutoff at the band
    boundary reads as a visible seam, so the blend fades in via a
    ``sqrt``-shaped ramp instead of switching on abruptly.
    """
    height = img_f.shape[0]
    band_start = int(height * (1.0 - band_fraction))
    if band_start >= height:
        return img_f

    band_height = height - band_start
    band = img_f[band_start:height]
    blended = blend(band)

    feather = np.linspace(0.0, 1.0, band_height, dtype=np.float32) ** 0.5
    result = band * (1.0 - feather[:, None, None]) + blended * feather[:, None, None]

    out = img_f.copy()
    out[band_start:height] = result
    return out
