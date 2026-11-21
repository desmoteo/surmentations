"""Filters whose reach is many pixels, computed on a reduced copy of the frame.

OpenCV's Gaussian blur costs in proportion to its sigma, and its dilation by
a disc in proportion to the disc's area. Several effects here blur or grow a
field by a fixed share of the frame's diagonal, tens of pixels on a large
frame, and the result holds no detail a smaller image could not. These do
the work at the size the result needs.

They are fast-mode forms (:mod:`surmentations.mode`): in exact mode each
is the plain OpenCV call, and how far the two differ is stated per function
and tested in ``tests/unit/test_wide_filters.py``.
"""

from __future__ import annotations

import cv2
import numpy as np

from surmentations import mode

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
    (:mod:`surmentations.mode`), it is the plain blur.

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


# A glow round lamps is blurred on a copy reduced no more than this many
# times, in fast mode, and only from this sigma up. Lamps are points: at a
# half they are within 2% of the halo's own peak, at a third within 3%.
_GLOW_REDUCED = 2
_GLOW_LEAST_SIGMA = 8.0
# The share of a frame past which the lamps' box is taken to be the frame.
_GLOW_MOST_OF_FRAME = 0.7


def blur_of_lights(field: np.ndarray, sigma: float) -> np.ndarray:
    """A Gaussian blur of a field that is nothing but a few lights, taken only where they are.

    ``field`` is ``(H, W)`` or ``(H, W, C)`` float32, nought but for its
    lamps, and none of it negative. The blur of nothing is nothing, so the
    filter is run on the box that holds the lamps and as far round it as
    the filter reaches, and the rest is left at nought: the same values as
    ``cv2.GaussianBlur`` of the whole frame to the bit, in the time the
    lamps' share of the frame takes. A halo round two tail lamps is a
    hundredth of a frame.

    In fast mode (:mod:`surmentations.mode`) a halo of sigma 8 or more is
    blurred on a copy of the box at half the size, to within 2% of the
    halo's peak: traffic across a whole road fills the box, and the
    smaller copy is what saves that case. Where the box meets the frame's
    edge it is first carried on past it as the filter would mirror it,
    since a smaller copy mirrors about another place and a lamp at the
    edge came out a quarter wrong.
    """
    height, width = field.shape[:2]
    radius = (round(sigma * 8.0 + 1.0) | 1) // 2
    reduced = _GLOW_REDUCED if mode.fast() and sigma >= _GLOW_LEAST_SIGMA else 1
    if min(height, width) <= radius + 1 or field.dtype != np.float32:
        return np.asarray(cv2.GaussianBlur(field, (0, 0), sigmaX=sigma), dtype=np.float32)
    # Which rows and which columns hold a lamp: the largest value of each.
    flat = np.ascontiguousarray(field).reshape(height, -1)
    rows = np.flatnonzero(flat.max(axis=1) > 0.0)
    if not rows.size:
        return np.zeros(field.shape, dtype=np.float32)
    lit_columns = flat[rows[0] : rows[-1] + 1].max(axis=0).reshape(width, -1)
    columns = np.flatnonzero(lit_columns.max(axis=1) > 0.0)
    # Twice the filter's own reach: what it writes, and what that reads.
    reach = 2 * radius + 2
    top, bottom = int(rows[0]) - reach, int(rows[-1]) + 1 + reach
    left, right = int(columns[0]) - reach, int(columns[-1]) + 1 + reach
    inside = (slice(max(top, 0), min(bottom, height)), slice(max(left, 0), min(right, width)))
    whole = (inside[0].stop - inside[0].start) * (inside[1].stop - inside[1].start) > (
        _GLOW_MOST_OF_FRAME * height * width
    )
    if whole:
        # The lamps are all over the frame: the box would be the frame.
        if reduced == 1:
            return np.asarray(cv2.GaussianBlur(field, (0, 0), sigmaX=sigma), dtype=np.float32)
        top, bottom, left, right = -radius, height + radius, -radius, width + radius
        inside = (slice(0, height), slice(0, width))
    box = field[inside]
    # Past the frame's edge, the frame mirrored, as far as the filter reads.
    past = [
        min(radius, max(0, -top)),
        min(radius, max(0, bottom - height)),
        min(radius, max(0, -left)),
        min(radius, max(0, right - width)),
    ]
    if any(past):
        box = cv2.copyMakeBorder(box, past[0], past[1], past[2], past[3], cv2.BORDER_REFLECT_101)
    if reduced > 1:
        size = (max(1, round(box.shape[1] / reduced)), max(1, round(box.shape[0] / reduced)))
        small = cv2.resize(box, size, interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(
            small,
            (0, 0),
            sigmaX=sigma * size[0] / box.shape[1],
            sigmaY=sigma * size[1] / box.shape[0],
        )
        blurred = cv2.resize(small, box.shape[1::-1], interpolation=cv2.INTER_LINEAR)
    else:
        blurred = cv2.GaussianBlur(np.ascontiguousarray(box), (0, 0), sigmaX=sigma)
    kept = blurred[past[0] : blurred.shape[0] - past[1], past[2] : blurred.shape[1] - past[3]]
    if whole:
        return np.ascontiguousarray(kept, dtype=np.float32)
    out = np.zeros(field.shape, dtype=np.float32)
    out[inside] = kept
    return out
