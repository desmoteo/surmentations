"""Wide-dynamic-range local tone mapping, and the halos it leaves.

Physical model
--------------
A surveillance camera routinely faces scenes no sensor can hold in one
exposure — a doorway against bright sky, headlights against an unlit street.
Its ISP therefore applies **local** tone mapping: it splits the image into a
slowly-varying *base* layer and a *detail* layer, compresses only the base,
and puts the detail back. Shadows lift and highlights survive while local
texture keeps its contrast.

The halos are not a bug bolted on afterwards; they fall out of how the split
is done. Separating base from detail needs an edge-preserving filter, and
edge-preserving filters are expensive, so embedded ISPs use cheap ones with a
finite range. At a strong edge the base layer *leaks* across it, so the
detail layer picks up a compensating overshoot — and once the base is
compressed and the two are recombined, that overshoot appears as a bright
band on the dark side of the edge and a dark band on the bright side. The
worse the filter, the bigger the halo.

That gives a knob with a real meaning rather than an arbitrary one:
``halo_strength`` interpolates the base layer between a properly
edge-preserving filter and a plain Gaussian, i.e. between an expensive ISP
and a cheap one.

Working in log luminance is also not a stylistic choice. Tone mapping
compresses *ratios* — a lamp is not "200 units" brighter than a wall, it is
some factor brighter — and a ratio is a difference in log space, so the
base/detail split has to happen there to be scale-invariant.
"""

from __future__ import annotations

import math
from typing import Annotated, Any

import cv2
import numpy as np
from albumentations.core.pydantic import check_range_bounds, nondecreasing
from albumentations.core.transforms_interface import BaseTransformInitSchema
from pydantic import AfterValidator

from weathermentations.base import WeatherTransform
from weathermentations.utils.photometry import from_linear_unit, luminance, to_linear_unit

# Radius above which the base layer is computed on a downsampled copy. The
# bilateral filter is O(radius^2) per pixel, so this caps the work per frame.
_BASE_FILTER_MAX_RADIUS_PX = 12.0


def _base_filter_scale(radius_px: float) -> tuple[int, float]:
    """``(downsampling factor, radius at that scale)`` for the base-layer filter.

    The factor is rounded **up**, so the radius actually filtered never
    exceeds the cap. Rounding it down — as this did — only starts downsampling
    at twice the cap, and every radius between one and two caps was filtered
    at full resolution: measured at 512 px, a 24 px radius cost 2.0 s where
    the same frame takes 0.12 s downsampled by two. Half of all draws at that
    size landed in the gap.
    """
    factor = max(1, math.ceil(radius_px / _BASE_FILTER_MAX_RADIUS_PX))
    return factor, max(1.0, radius_px / factor)


class WdrToneMapping(WeatherTransform):
    """Local tone mapping with the halos a cheap edge-preserving filter leaves.

    Args:
        compression_range: how hard the base layer's dynamic range is
            squeezed. 1 leaves it alone; 0.4 compresses the scene's range to
            40% of its log extent, lifting shadows and pulling highlights
            down hard.
        detail_gain_range: how much the detail layer is boosted on the way
            back in. Above 1 is the over-crisp look of an ISP trying to hide
            the flattening the compression caused.
        halo_strength_range: how far the base layer leaks across strong edges,
            i.e. how cheap the filter is. 0 is a properly edge-preserving
            filter and almost no halo; 1 is a plain Gaussian and a pronounced
            one.
        filter_radius_range: base-layer filter radius as a fraction of the
            image diagonal. This sets the halo's *width* — the scale over
            which the base leaks.

    Targets:
        image

    Image types:
        uint8, float32
    """

    class InitSchema(BaseTransformInitSchema):
        compression_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.1, 1.0)),
        ]
        detail_gain_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.5, 3.0)),
        ]
        halo_strength_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 1.0)),
        ]
        filter_radius_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.005, 0.2)),
        ]

    def __init__(
        self,
        compression_range: tuple[float, float] = (0.45, 0.85),
        detail_gain_range: tuple[float, float] = (1.0, 1.6),
        halo_strength_range: tuple[float, float] = (0.2, 0.8),
        filter_radius_range: tuple[float, float] = (0.02, 0.07),
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.compression_range = compression_range
        self.detail_gain_range = detail_gain_range
        self.halo_strength_range = halo_strength_range
        self.filter_radius_range = filter_radius_range

    def get_params_dependent_on_data(
        self, params: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, Any]:
        del data
        return {
            "compression": self.py_random.uniform(*self.compression_range),
            "detail_gain": self.py_random.uniform(*self.detail_gain_range),
            "halo_strength": self.py_random.uniform(*self.halo_strength_range),
            "filter_radius": self.py_random.uniform(*self.filter_radius_range),
        }

    def apply(
        self,
        img: np.ndarray,
        compression: float,
        detail_gain: float,
        halo_strength: float,
        filter_radius: float,
        **params: Any,
    ) -> np.ndarray:
        linear, scale, is_integer = to_linear_unit(img)
        height, width = linear.shape[:2]
        radius_px = max(2.0, filter_radius * float(np.hypot(width, height)))

        # Tone mapping compresses ratios, so the split happens in log space.
        luma = np.maximum(luminance(linear), 1e-4)
        log_luma = np.log(luma)

        # The two ends of the "how good is the ISP's filter" axis: a bilateral
        # filter that respects edges, and a Gaussian that does not.
        #
        # The base layer is low-frequency by construction — that is what makes
        # it the base layer — so it is computed on a downsampled copy and
        # scaled back up. A bilateral filter costs O(radius^2) per pixel, so
        # at full resolution a 60px radius took over ten seconds a frame;
        # at quarter resolution it is ~0.1 s and, since the layer cannot carry
        # detail finer than its own radius anyway, visually identical. Same
        # reasoning as the scatter PSF in ``lens/dirt.py``.
        factor, small_radius = _base_filter_scale(radius_px)
        if factor > 1:
            small = cv2.resize(
                log_luma,
                (max(8, width // factor), max(8, height // factor)),
                interpolation=cv2.INTER_AREA,
            )
        else:
            small = log_luma

        edge_preserving = cv2.bilateralFilter(
            small, d=0, sigmaColor=0.35, sigmaSpace=small_radius
        )
        leaking = cv2.GaussianBlur(small, (0, 0), sigmaX=small_radius)
        base = edge_preserving * (1.0 - halo_strength) + leaking * halo_strength
        if factor > 1:
            base = cv2.resize(base, (width, height), interpolation=cv2.INTER_LINEAR)

        detail = log_luma - base

        # Compress the base about its own mean; keep (or boost) the detail.
        base_mean = float(base.mean())
        mapped = base_mean + (base - base_mean) * compression + detail * detail_gain

        # Back to a gain on the original colour, so hue and saturation are
        # carried through unchanged — the ISP is remapping luminance, not
        # repainting the scene.
        gain = np.exp(mapped - log_luma)
        return from_linear_unit(linear * gain[..., None], scale, is_integer, img.dtype)
