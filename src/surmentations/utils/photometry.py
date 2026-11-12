"""Linear-light helpers.

Photographs are stored gamma-encoded (sRGB), but light does not add up that
way. Atmospheric scattering, flare and bloom are all *additive radiance*
processes, and compositing them directly onto sRGB values is one of the most
common reasons synthetic weather looks flat and washed out rather than
luminous: mid-tones lift far too much and highlights compress far too
little. Every effect that adds or mixes light in this library converts to
linear light first, works there, and re-encodes at the end.

Why this is table-driven
------------------------
Every transform pays for one decode and one encode, so this module is on the
critical path of the whole library — it was ~40 ms per 1024px image, more
than the rendering of several effects. Two things made the textbook
implementation slow: ``np.power`` over three million elements, and
``np.where`` evaluating *both* branches of the piecewise curve everywhere.

Both conversions are pure scalar functions, so they table perfectly:

* **uint8 input** (the overwhelmingly common case) needs only a 256-entry
  table and is then **exact** — no approximation at all.
* **float input** is quantised to a 16-bit index into a 65536-entry table.
  The worst-case error is 0.05/255 of an 8-bit code value, i.e. twenty times
  finer than the quantisation of the uint8 images this library mostly
  processes, and well below the noise floor of any augmented training image.

Small arrays (colour constants, airlight, a droplet's tint) skip the table
and use the exact expression, because below a few thousand elements the
gather costs more than the arithmetic and there is no reason to accept even
that much error on a value that will be broadcast over a whole frame.

Why the encode has a table of its own
-------------------------------------
Encoding to uint8 was still five passes over the image: clip, scale, gather
a float, scale and round it, clip and cast. The last three depend only on
the table entry, so they are folded into a second table, of code values. It
gives the same bytes, in three passes, or in one where the compiled kernels
are installed (:mod:`surmentations._native`).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from surmentations import _native

# Size of the float->float lookup tables. 16 bits keeps the worst-case error
# at 0.05/255 while the table itself (256 KB) still sits in L2.
_LUT_SIZE = 1 << 16
# Below this many elements the exact expression beats a table gather.
_EXACT_BELOW = 1 << 13


def _srgb_to_linear_exact(image: np.ndarray) -> np.ndarray:
    """The sRGB decoding curve, evaluated, for arrays too small for the table."""
    image = np.clip(image, 0.0, 1.0)
    return np.where(
        image <= 0.04045,
        image / 12.92,
        np.power((image + 0.055) / 1.055, 2.4),
    ).astype(np.float32)


def _linear_to_srgb_exact(image: np.ndarray) -> np.ndarray:
    """The sRGB encoding curve, evaluated, for arrays too small for the table."""
    image = np.clip(image, 0.0, 1.0)
    return np.where(
        image <= 0.0031308,
        image * 12.92,
        1.055 * np.power(image, 1.0 / 2.4) - 0.055,
    ).astype(np.float32)


_RAMP = np.linspace(0.0, 1.0, _LUT_SIZE, dtype=np.float32)
_SRGB_TO_LINEAR_LUT = _srgb_to_linear_exact(_RAMP)
_LINEAR_TO_SRGB_LUT = _linear_to_srgb_exact(_RAMP)
# uint8 sRGB codes decode exactly through a 256-entry table.
_U8_TO_LINEAR_LUT = _srgb_to_linear_exact(np.arange(256, dtype=np.float32) / 255.0)
# Linear light straight to a uint8 code: the encode table, already scaled,
# rounded and clipped as `from_linear_unit` does it.
_LINEAR_TO_SRGB_U8_LUT = np.clip(np.rint(_LINEAR_TO_SRGB_LUT * 255.0), 0, 255).astype(np.uint8)


def _lut_index(image: np.ndarray) -> np.ndarray:
    """Table index of each value: ``int(clip(image, 0, 1) * (size - 1))``.

    Scaled first and clipped in place, which is the same number and one
    full-size temporary fewer.

    A NaN is refused with a ``ValueError``: it is a renderer's bug, and a
    frame with one must not come out with a black pixel where it was. That
    used to be left to the conversion to an integer, which on x86-64 makes
    a NaN an index far out of range that the lookup rejects. On arm64 it
    makes it 0, a valid index, and the NaN went through as black. So it is
    looked for, at the cost of one maximum over the frame.
    """
    scaled = np.multiply(image, _LUT_SIZE - 1)
    np.clip(scaled, 0.0, _LUT_SIZE - 1, out=scaled)
    if scaled.size and np.isnan(scaled.max()):
        raise ValueError("a NaN reached the conversion out of linear light: a renderer's bug")
    return scaled.astype(np.int32)


def _through_lut(image: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """``image`` through a float32 table of :data:`TABLE_SIZE` entries over ``[0, 1]``.

    By the colour kernel where it is built, by numpy where it is not; a NaN
    is refused either way.
    """
    out = _native.encode_f32(image, lut)
    if out is None:
        out = np.take(lut, _lut_index(image))
    return out


# The length a table for :func:`through_table` must have.
TABLE_SIZE = _LUT_SIZE


def through_table(image: np.ndarray, table: np.ndarray) -> np.ndarray:
    """``table[int(clip(image, 0, 1) * (TABLE_SIZE - 1))]`` for a float32 table.

    Any function of one value in ``[0, 1]`` that costs more than a lookup,
    a power for one, taken from a table of :data:`TABLE_SIZE` entries as
    the transfer functions here are. Entry ``k`` serves the values from
    ``k / (TABLE_SIZE - 1)`` up to the next entry's.
    """
    return _through_lut(image, table)


def _decode_u8(image: np.ndarray) -> np.ndarray:
    """An 8-bit frame in linear light through the 256-entry table, by the kernel or numpy."""
    out = _native.decode_u8(image, _U8_TO_LINEAR_LUT)
    if out is None:
        # `take` is the same gather as `lut[image]`, a quarter faster.
        out = np.take(_U8_TO_LINEAR_LUT, image)
    return out


def srgb_to_linear(image: np.ndarray) -> np.ndarray:
    """Decode sRGB-encoded values in ``[0, 1]`` to linear light.

    ``uint8`` input is decoded exactly through a 256-entry table and needs no
    prior division by 255.
    """
    if image.dtype == np.uint8:
        return _decode_u8(image)
    if image.size < _EXACT_BELOW:
        return _srgb_to_linear_exact(image)
    return _through_lut(image, _SRGB_TO_LINEAR_LUT)


def linear_to_srgb(image: np.ndarray) -> np.ndarray:
    """Encode linear-light values in ``[0, 1]`` back to sRGB."""
    if image.size < _EXACT_BELOW:
        return _linear_to_srgb_exact(image)
    return _through_lut(image, _LINEAR_TO_SRGB_LUT)


def to_linear_unit(img: np.ndarray) -> tuple[np.ndarray, float, bool]:
    """Convert an image of any supported dtype to linear-light floats in ``[0, 1]``.

    Returns ``(linear, scale, is_integer)``; pass all three to
    :func:`from_linear_unit` to invert.
    """
    is_integer = np.issubdtype(img.dtype, np.integer)
    scale = 255.0 if is_integer else 1.0
    if img.dtype == np.uint8:
        # The table already folds in the /255, so this skips a full-size
        # float conversion of the source image.
        return _decode_u8(img), scale, True
    return srgb_to_linear(img.astype(np.float32) / scale), scale, is_integer


def from_linear_unit(linear: np.ndarray, scale: float, is_integer: bool, dtype: Any) -> np.ndarray:
    """Invert :func:`to_linear_unit`, clipping into the original dtype's range.

    Integer output is **rounded**, not truncated. ``astype`` truncates toward
    zero, which sent 189 of the 256 possible 8-bit values back one level low —
    a systematic bias of about -0.25 levels every time an effect ran, in the
    same direction every time. One transform is invisible; a ``Compose`` of
    five applies it five times, and an effect whose whole job is a few percent
    of contrast (rain streaks at 2% opacity) is competing with it.
    """
    if is_integer and np.dtype(dtype) == np.uint8 and linear.size >= _EXACT_BELOW:
        # The table holds `clip(rint(srgb * 255), 0, 255)` per entry, so this
        # is the general expression below, bit for bit.
        codes = _native.encode_u8(linear, _LINEAR_TO_SRGB_U8_LUT)
        if codes is None:
            codes = np.take(_LINEAR_TO_SRGB_U8_LUT, _lut_index(linear))
        return codes
    encoded = linear_to_srgb(linear) * scale
    if is_integer:
        return np.clip(np.rint(encoded), 0, 255).astype(dtype)
    return np.clip(encoded, 0.0, 1.0).astype(dtype)


def luminance(linear_rgb: np.ndarray) -> np.ndarray:
    """Rec.709 relative luminance of a linear-light RGB image."""
    return linear_rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
