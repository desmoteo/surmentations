"""Filters whose reach is many pixels, computed on a reduced copy of the frame.

OpenCV's Gaussian blur costs in proportion to its sigma, and its dilation by
a disc in proportion to the disc's area. Several effects here blur or grow a
field by a fixed share of the frame's diagonal, tens of pixels on a large
frame, and the result holds no detail a smaller image could not. These do
the work at the size the result needs.

They are fast-mode forms (:mod:`weathermentations.mode`): in exact mode each
is the plain OpenCV call, and how far the two differ is stated per function
and tested in ``tests/unit/test_wide_filters.py``.
"""

from __future__ import annotations

import cv2
import numpy as np

from weathermentations import mode

# The sigma, in pixels, a wide blur is run at on its reduced copy. At 8 the
# result is within 0.2% of the full-size blur's peak; at 4 it was 0.7%.
_WIDE_BLUR_WORK_SIGMA = 8.0


def wide_blur(
    image: np.ndarray,
    sigma: float,
    work_sigma: float = _WIDE_BLUR_WORK_SIGMA,
    sigma_y: float | None = None,
) -> np.ndarray:
    """Gaussian blur for a sigma of many pixels, computed on a smaller copy.

    A Gaussian filter's cost grows with its sigma, and a result blurred that
    far holds no detail a smaller image could not: the frame is reduced until
    the sigma is ``work_sigma`` pixels, blurred there, and enlarged. The two
    blurs in ``sky_to_depth_edge`` have a sigma of 4% of the diagonal, 58
    pixels on a 1024-pixel frame. On one core, which is what a ``DataLoader``
    worker has, they took 400 ms of every fog, rain and snow frame; this way
    they take 5.

    ``work_sigma`` trades accuracy for time. At the default of 8 the result is
    within 0.2% of the full-size blur's peak, and only a sigma of 16 pixels
    or more is reduced at all. At 4 it is within 0.7%, and a sigma of 8 is
    already halved: for a field that is soft by nature, a defocused deposit
    on the glass, say. Below twice the working sigma, and in exact mode
    (:mod:`weathermentations.mode`), it is the plain blur.

    ``sigma_y`` is the sigma down the frame where that differs from the one
    across it, and each axis is then reduced by its own factor: a reflection
    smeared 36 pixels down a road and 11 across is reduced four times in
    height and not at all in width.
    """
    down = sigma if sigma_y is None else sigma_y
    across_factor = max(1, int(sigma / work_sigma)) if mode.fast() else 1
    down_factor = max(1, int(down / work_sigma)) if mode.fast() else 1
    if across_factor < 2 and down_factor < 2:
        return cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=down)
    height, width = image.shape[:2]
    size = (max(1, round(width / across_factor)), max(1, round(height / down_factor)))
    small = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(
        small, (0, 0), sigmaX=sigma * size[0] / width, sigmaY=down * size[1] / height
    )
    return cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)


# The diameter, in pixels, a wide dilation's disc has on its reduced copy.
_WIDE_DILATE_WORK_PX = 9


def wide_dilate(mask: np.ndarray, diameter: int) -> np.ndarray:
    """Grey dilation of an ``(H, W)`` mask by a disc many pixels across.

    A disc is not a separable structuring element, so OpenCV visits every
    pixel of it for every pixel of the image: 59 pixels across on a
    1024-pixel frame is 80 ms on one core. The mask is first reduced by
    taking the maximum of each block, which loses nothing a dilation would
    keep, then dilated by a disc of a few pixels and enlarged: 2 ms. It
    agrees with the exact dilation except in a band about one block wide
    where the reach ends, which it draws soft. Good for a bound on how far
    something may grow, which is its one use here; not for a shape whose
    outline matters. Below twice the working diameter, and in exact mode,
    it is the plain dilation.
    """
    factor = diameter // _WIDE_DILATE_WORK_PX if mode.fast() else 1
    if factor < 2:
        return cv2.dilate(
            mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
        )
    height, width = mask.shape[:2]
    # A square dilation is separable and cheap; one sample from the middle of
    # each block of it is that block's maximum.
    pooled = cv2.dilate(mask, np.ones((factor, factor), np.uint8))
    small = np.ascontiguousarray(pooled[factor // 2 :: factor, factor // 2 :: factor])
    disc = max(3, round(diameter / factor) | 1)
    grown = cv2.dilate(small, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (disc, disc)))
    return cv2.resize(grown, (width, height), interpolation=cv2.INTER_LINEAR)
