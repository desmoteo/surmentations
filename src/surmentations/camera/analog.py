"""What an analog video link does to a picture.

Physical model
--------------
A great many cameras in service do not send a digital stream. They send a
composite signal down a coaxial cable, as PAL or NTSC did and as the analog
high-definition formats that replaced them (AHD, TVI, CVI) still do: one
voltage, a line at a time, carrying brightness directly and colour on a
subcarrier added to it. Everything characteristic of that picture follows
from the signal being one-dimensional and from the cable and the decoder
being imperfect. None of it is in a codec's repertoire, so
``VideoCodecArtifacts`` does not produce it.

**Detail is lost along the line and not down the frame.** A cable is a
low-pass filter, more so the longer it is, and it acts on the signal, which
runs along the scan lines. Vertical edges soften; horizontal ones do not.
Receivers boost what is left, which rings on either side of an edge.

**Colour has a fraction of the brightness's resolution, and is late.** The
two colour-difference signals are band-limited to a few percent of the
picture's width before they are modulated, and the filters that recover
them delay them: colour is smeared along the line and sits a little to the
right of the edge it belongs to.

**Brightness and colour leak into each other.** They share one signal, and
a decoder can only separate them by frequency. Fine brightness detail near
the subcarrier's frequency is taken for colour, and shimmers in rainbow
(cross-colour); a sharp change of colour leaves subcarrier in the
brightness, a line of dots along the edge (dot crawl). This is done here
by doing it: the frame is encoded onto a subcarrier whose phase alternates
from line to line, and decoded by multiplying by the carrier and
low-passing. What a better decoder would have removed is the share of that
result which is mixed back in.

**The cable adds its own.** A mismatched termination reflects the signal,
and the reflection arrives late: a faint copy of the picture displaced to
the right. A ground loop adds mains hum, a broad dark and light bar or two
across the frame. Noise is added to the signal, so it too is smeared along
the line; and where the receiver's line sync wavers, lines start a little
early or late.

**The frame is two fields.** Interlaced video scans odd and even lines a
fiftieth or sixtieth of a second apart. Anything that moved in between is
drawn twice, a comb of alternate lines. A single frame has no motion to
show, so the comb is drawn for a camera that panned, by a shift of
alternate lines against each other.

The signal is gamma-encoded, so all of this is done on code values and not
in linear light. Lines are displaced by a few pixels at most; annotations
are not moved.
"""

from __future__ import annotations

import math
from typing import Annotated, Any

import cv2
import numpy as np
from albumentations.core.pydantic import check_range_bounds, nondecreasing
from albumentations.core.transforms_interface import BaseTransformInitSchema
from pydantic import AfterValidator

from surmentations.base import CameraTransform
from surmentations.diagnostics import record_draw

# Code values to brightness and two colour differences, and back (BT.601).
_TO_YUV = np.array(
    [[0.299, 0.587, 0.114], [-0.14713, -0.28886, 0.436], [0.615, -0.51499, -0.10001]],
    dtype=np.float32,
)
_TO_RGB = np.linalg.inv(_TO_YUV).astype(np.float32)
# Sigma, in pixels, of the Gaussian whose response has fallen by half at
# the frame's own limit of one cycle in two pixels; a filter with a
# fraction b of that bandwidth has this over b.
_SIGMA_AT_FULL_BAND = 0.265


def along_lines(plane: np.ndarray, sigma_px: float) -> np.ndarray:
    """A Gaussian low-pass along the scan lines only, ``sigma_px`` wide."""
    if sigma_px < 0.05:
        return plane
    blurred: np.ndarray = cv2.GaussianBlur(
        plane, (0, 1), sigmaX=sigma_px, sigmaY=0.0, borderType=cv2.BORDER_REPLICATE
    )
    return blurred


def band_limit(bandwidth: float) -> float:
    """Sigma in pixels of the low-pass with ``bandwidth`` of the frame's own, 0 to 1."""
    return _SIGMA_AT_FULL_BAND / max(bandwidth, 1e-3)


def composite_round_trip(
    yuv: np.ndarray, subcarrier: float, chroma_sigma: float, phase: float
) -> np.ndarray:
    """Encode brightness and colour onto one signal and decode them again.

    The two colour differences, band-limited by ``chroma_sigma``, modulate
    a subcarrier of ``subcarrier`` cycles a pixel in quadrature, its phase
    turned by half a cycle from each line to the next, and are added to
    the brightness. The decoder multiplies the signal by the carrier and
    low-passes to recover the colour, then takes that colour's carrier back
    off the signal to leave the brightness. What it cannot tell apart stays
    where it should not be: that is cross-colour and dot crawl.
    """
    height, width = yuv.shape[:2]
    lines = np.arange(height, dtype=np.float32)[:, None]
    across = np.arange(width, dtype=np.float32)[None, :]
    angle = 2.0 * np.pi * subcarrier * across + np.pi * lines + phase
    sin, cos = np.sin(angle), np.cos(angle)
    u = along_lines(np.ascontiguousarray(yuv[..., 1]), chroma_sigma)
    v = along_lines(np.ascontiguousarray(yuv[..., 2]), chroma_sigma)
    signal = yuv[..., 0] + u * sin + v * cos

    decoded = np.empty_like(yuv)
    decoded[..., 1] = 2.0 * along_lines(signal * sin, chroma_sigma)
    decoded[..., 2] = 2.0 * along_lines(signal * cos, chroma_sigma)
    decoded[..., 0] = signal - (decoded[..., 1] * sin + decoded[..., 2] * cos)
    return decoded


def displace_lines(image: np.ndarray, shift_px: np.ndarray) -> np.ndarray:
    """Each line moved along itself by its entry of ``shift_px``, ``(H,)``, to the right."""
    height, width = image.shape[:2]
    map_x = np.arange(width, dtype=np.float32)[None, :] - shift_px.astype(np.float32)[:, None]
    map_y = np.broadcast_to(np.arange(height, dtype=np.float32)[:, None], (height, width))
    moved: np.ndarray = cv2.remap(
        image, map_x, np.ascontiguousarray(map_y), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )
    return moved


class AnalogVideoArtifacts(CameraTransform):
    """A frame as it arrives over a composite or analog-HD link.

    Needs no depth. Transmission stage: in place of a codec, for a camera
    that has none.

    Args:
        luma_bandwidth_range: how much of the frame's horizontal resolution
            the link passes, 0 to 1. A long cable is at the low end.
        peaking_range: how hard the receiver boosts what is left, which
            rings beside edges. Zero is none.
        chroma_bandwidth_range: the same for colour, which has far less.
        chroma_delay_range: how far colour sits to the right of where it
            belongs, as a fraction of the frame's width.
        subcarrier_range: the colour subcarrier's frequency, in cycles a
            pixel. Detail at about that pitch is what turns to rainbow.
        crosstalk_range: how much of the leakage between brightness and
            colour the decoder fails to remove, 0 to 1. One is the simplest
            decoder there is; a comb filter is near zero.
        ghost_strength_range: strength of the cable's reflection, a faint
            copy of the picture to the right.
        ghost_delay_range: how far to the right, as a fraction of the width.
        hum_depth_range: depth of the mains hum bars, in units of full
            scale.
        hum_cycles_range: how many of them fit down the frame.
        noise_range: noise on the signal, as a standard deviation in units
            of full scale.
        line_jitter_range: how far lines start early or late, as a standard
            deviation in fractions of the width.
        comb_shift_range: how far the two fields are apart, as a fraction
            of the width: what a pan between them leaves.

    Targets:
        image

    Image types:
        uint8, float32
    """

    class InitSchema(BaseTransformInitSchema):
        """The arguments of ``AnalogVideoArtifacts`` as Albumentations validates them:
        each range ordered low to high and within the bounds given here.
        """

        luma_bandwidth_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.05, 1.0)),
        ]
        peaking_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 2.0)),
        ]
        chroma_bandwidth_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.01, 1.0)),
        ]
        chroma_delay_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 0.05)),
        ]
        subcarrier_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.1, 0.45)),
        ]
        crosstalk_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 1.0)),
        ]
        ghost_strength_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 1.0)),
        ]
        ghost_delay_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 0.2)),
        ]
        hum_depth_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 0.5)),
        ]
        hum_cycles_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.1, 10.0)),
        ]
        noise_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 0.5)),
        ]
        line_jitter_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 0.02)),
        ]
        comb_shift_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 0.05)),
        ]

    def __init__(
        self,
        luma_bandwidth_range: tuple[float, float] = (0.3, 0.9),
        peaking_range: tuple[float, float] = (0.0, 0.6),
        chroma_bandwidth_range: tuple[float, float] = (0.05, 0.2),
        chroma_delay_range: tuple[float, float] = (0.0, 0.004),
        subcarrier_range: tuple[float, float] = (0.2, 0.3),
        crosstalk_range: tuple[float, float] = (0.2, 1.0),
        ghost_strength_range: tuple[float, float] = (0.0, 0.2),
        ghost_delay_range: tuple[float, float] = (0.005, 0.03),
        hum_depth_range: tuple[float, float] = (0.0, 0.06),
        hum_cycles_range: tuple[float, float] = (0.5, 2.5),
        noise_range: tuple[float, float] = (0.005, 0.035),
        line_jitter_range: tuple[float, float] = (0.0, 0.0012),
        comb_shift_range: tuple[float, float] = (0.0, 0.004),
        p: float = 0.5,
    ):
        """Store the ranges; each argument is described in the class docstring."""
        super().__init__(p=p)
        self.luma_bandwidth_range = luma_bandwidth_range
        self.peaking_range = peaking_range
        self.chroma_bandwidth_range = chroma_bandwidth_range
        self.chroma_delay_range = chroma_delay_range
        self.subcarrier_range = subcarrier_range
        self.crosstalk_range = crosstalk_range
        self.ghost_strength_range = ghost_strength_range
        self.ghost_delay_range = ghost_delay_range
        self.hum_depth_range = hum_depth_range
        self.hum_cycles_range = hum_cycles_range
        self.noise_range = noise_range
        self.line_jitter_range = line_jitter_range
        self.comb_shift_range = comb_shift_range

    def get_params_dependent_on_data(
        self, params: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, Any]:
        """Draw the link: its bandwidths, its decoder, its cable's faults and its noise.

        Returns every scalar the renderer needs, the per-line ``shift`` in
        pixels (the comb of the two fields and the sync's jitter together),
        and the ``noise`` to add to the signal.
        """
        del data
        height, width = params["shape"][:2]
        rng = self.random_generator
        draw = self.py_random.uniform

        luma_sigma = band_limit(draw(*self.luma_bandwidth_range))
        comb = draw(*self.comb_shift_range) * width * self.py_random.choice([-1.0, 1.0])
        jitter = draw(*self.line_jitter_range) * width
        shift = rng.normal(0.0, 1.0, height) * jitter
        shift[1::2] += comb
        noise_level = draw(*self.noise_range)
        # Noise is on the signal, so the link's low-pass smears it along the
        # line with everything else; scaled back up to the level asked for.
        noise = rng.normal(0.0, 1.0, (height, width)).astype(np.float32)
        noise = along_lines(noise, luma_sigma)
        noise *= np.float32(noise_level / max(float(noise.std()), 1e-6))

        values = {
            "luma_sigma": luma_sigma,
            "peaking": draw(*self.peaking_range),
            "chroma_sigma": band_limit(draw(*self.chroma_bandwidth_range)),
            "chroma_delay": draw(*self.chroma_delay_range) * width,
            "subcarrier": draw(*self.subcarrier_range),
            "carrier_phase": draw(0.0, 2.0 * math.pi),
            "crosstalk": draw(*self.crosstalk_range),
            "ghost_strength": draw(*self.ghost_strength_range),
            "ghost_delay": draw(*self.ghost_delay_range) * width,
            "hum_depth": draw(*self.hum_depth_range),
            "hum_cycles": draw(*self.hum_cycles_range),
            "hum_phase": draw(0.0, 2.0 * math.pi),
        }
        record_draw(**values, noise=noise_level, comb_shift_px=comb, line_jitter_px=jitter)
        return {**values, "shift": shift.astype(np.float32), "noise": noise}

    def apply(
        self,
        img: np.ndarray,
        luma_sigma: float,
        peaking: float,
        chroma_sigma: float,
        chroma_delay: float,
        subcarrier: float,
        carrier_phase: float,
        crosstalk: float,
        ghost_strength: float,
        ghost_delay: float,
        hum_depth: float,
        hum_cycles: float,
        hum_phase: float,
        shift: np.ndarray,
        noise: np.ndarray,
        **params: Any,
    ) -> np.ndarray:
        """Send the frame down the link and decode what arrives.

        In the order the signal meets them: the fields' offset, the
        subcarrier, the cable's low-pass and its reflection, hum and noise,
        the decoder, the receiver's peaking, and the lines' jitter.
        """
        is_integer = np.issubdtype(img.dtype, np.integer)
        height, width = img.shape[:2]
        rgb = img.astype(np.float32) / (255.0 if is_integer else 1.0)
        if float(np.abs(shift).max()) > 0.02:
            rgb = displace_lines(rgb, shift)
        yuv = rgb @ _TO_YUV.T

        clean = yuv.copy()
        clean[..., 1] = along_lines(np.ascontiguousarray(yuv[..., 1]), chroma_sigma)
        clean[..., 2] = along_lines(np.ascontiguousarray(yuv[..., 2]), chroma_sigma)
        if crosstalk > 0.0:
            decoded = composite_round_trip(yuv, subcarrier, chroma_sigma, carrier_phase)
            yuv = clean + np.float32(crosstalk) * (decoded - clean)
        else:
            yuv = clean

        luma = along_lines(np.ascontiguousarray(yuv[..., 0]), luma_sigma)
        if ghost_strength > 0.0:
            ghost = displace_lines(luma, np.full(height, ghost_delay, dtype=np.float32))
            luma = luma + np.float32(ghost_strength) * (ghost - luma.mean())
        if hum_depth > 0.0:
            lines = np.arange(height, dtype=np.float32)[:, None] / max(height, 1)
            luma = luma + np.float32(hum_depth) * np.sin(
                2.0 * np.pi * hum_cycles * lines + hum_phase
            )
        luma = luma + noise
        if peaking > 0.0:
            luma = luma + np.float32(peaking) * (luma - along_lines(luma, 2.0 * luma_sigma))
        yuv[..., 0] = luma
        if chroma_delay > 0.02:
            late = np.full(height, chroma_delay, dtype=np.float32)
            yuv[..., 1:] = displace_lines(np.ascontiguousarray(yuv[..., 1:]), late)

        out = np.clip(yuv @ _TO_RGB.T, 0.0, 1.0)
        if is_integer:
            return np.rint(out * 255.0).astype(img.dtype)
        return out.astype(img.dtype)
