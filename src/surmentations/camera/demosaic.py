"""False colour from the sensor's mosaic.

Physical model
--------------
A sensor has one colour filter over each pixel, red, green or blue in a
Bayer pattern, and the other two colours at that pixel are interpolated
from its neighbours. Where the picture changes slowly that is invisible.
Where it has detail as fine as the pixels, a fence, a grille, a striped
shirt, lettering, the three colours have been sampled at different places
and no longer agree: the detail comes out fringed with colour that is not
in the scene, and edges show a one-pixel zip of alternating shades.

A photograph that is already a full-colour image has had this done to it
once, by the camera that took it, usually with a good interpolator on a
sensor with more pixels than the picture. A cheap camera at its native
resolution shows a great deal more, and this transform is that camera:

* **the lens and its low-pass filter** blur the scene slightly before the
  sensor, which is what limits the effect, and why a sharp lens on a
  sensor with no anti-alias filter is the worst case;
* **the sensor** samples one colour a pixel. It may have fewer pixels than
  the picture, the stream being enlarged afterwards, which makes the
  fringes as many pixels wide as the enlargement;
* **the processor** interpolates, bilinearly here as the cheapest do, and
  then smooths colour differences to hide the fringes, at the cost of
  colour resolution.

In linear light, on the sensor's own data: this happens before the tone
curve.
"""

from __future__ import annotations

from typing import Annotated, Any

import cv2
import numpy as np
from albumentations.core.pydantic import check_range_bounds, nondecreasing
from albumentations.core.transforms_interface import BaseTransformInitSchema
from pydantic import AfterValidator

from surmentations.base import CameraTransform
from surmentations.camera.sensor_noise import bayer_sites, demosaic
from surmentations.diagnostics import record_draw
from surmentations.utils.photometry import from_linear_unit, to_linear_unit


def through_mosaic(linear_rgb: np.ndarray, phase: tuple[int, int]) -> np.ndarray:
    """A frame as a Bayer sensor with a bilinear interpolator records it.

    Each pixel keeps the one channel its filter passes and the other two
    are interpolated from the neighbours that have them. ``phase`` is which
    corner of the 2 by 2 pattern the frame's first pixel falls on, ``(row,
    column)`` each 0 or 1: where the red filters lie.
    """
    height, width = linear_rgb.shape[:2]
    dy, dx = phase
    padded = cv2.copyMakeBorder(linear_rgb, 2 + dy, 2, 2 + dx, 2, cv2.BORDER_REFLECT_101)
    sites = bayer_sites(*padded.shape[:2])
    mosaic = np.take_along_axis(padded, sites[..., None].astype(np.intp), axis=2)[..., 0]
    full = demosaic(mosaic)
    recorded: np.ndarray = full[2 + dy : 2 + dy + height, 2 + dx : 2 + dx + width]
    return recorded


def smooth_chroma(linear_rgb: np.ndarray, radius_px: int) -> np.ndarray:
    """Median-filter the colour differences ``R - G`` and ``B - G``, leaving green alone.

    What a processor does to hide false colour: a fringe is a colour
    difference that changes sign from pixel to pixel, which a median over a
    few pixels removes, along with any real colour detail that fine.
    """
    if radius_px < 1:
        return linear_rgb
    size = 2 * radius_px + 1
    green = linear_rgb[..., 1]
    out = linear_rgb.copy()
    for channel in (0, 2):
        difference = np.ascontiguousarray(linear_rgb[..., channel] - green)
        out[..., channel] = green + cv2.medianBlur(difference, size)
    return out


class DemosaicArtifacts(CameraTransform):
    """False colour and zipped edges on fine detail, from a cheap camera's mosaic.

    Needs no depth. Sensor stage, before the processor's tone curve.

    Args:
        optical_blur_range_px: blur of the lens and its low-pass filter
            before the sensor, as a Gaussian sigma in sensor pixels. Near
            zero is a sharp lens with no anti-alias filter, and the most
            false colour.
        sensor_scale_range: output pixels for each sensor pixel. One is a
            sensor at the picture's resolution; two is one with half as
            many pixels each way, enlarged, with fringes twice as wide.
        chroma_filter_range_px: radius of the processor's median filter on
            the colour differences, in sensor pixels. Zero leaves the
            fringes as interpolated; two removes most of them.

    Targets:
        image

    Image types:
        uint8, float32
    """

    class InitSchema(BaseTransformInitSchema):
        """The arguments of ``DemosaicArtifacts`` as Albumentations validates them: each
        range ordered low to high and within the bounds given here.
        """

        optical_blur_range_px: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 3.0)),
        ]
        sensor_scale_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(1.0, 4.0)),
        ]
        chroma_filter_range_px: Annotated[
            tuple[int, int],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0, 2)),
        ]

    def __init__(
        self,
        optical_blur_range_px: tuple[float, float] = (0.0, 0.6),
        sensor_scale_range: tuple[float, float] = (1.0, 2.0),
        chroma_filter_range_px: tuple[int, int] = (0, 1),
        p: float = 0.5,
    ):
        """Store the ranges; each argument is described in the class docstring."""
        super().__init__(p=p)
        self.optical_blur_range_px = optical_blur_range_px
        self.sensor_scale_range = sensor_scale_range
        self.chroma_filter_range_px = chroma_filter_range_px

    def get_params_dependent_on_data(
        self, params: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, Any]:
        """Draw the lens's blur, the sensor's size and pattern, and the processor's filter."""
        del params, data
        blur = self.py_random.uniform(*self.optical_blur_range_px)
        sensor_scale = self.py_random.uniform(*self.sensor_scale_range)
        chroma = self.py_random.randint(*self.chroma_filter_range_px)
        phase = (self.py_random.randint(0, 1), self.py_random.randint(0, 1))
        record_draw(
            optical_blur_px=blur, sensor_scale=sensor_scale, chroma_filter_px=chroma, phase=phase
        )
        return {
            "optical_blur": blur,
            "sensor_scale": sensor_scale,
            "chroma_filter": chroma,
            "phase": phase,
        }

    def apply(
        self,
        img: np.ndarray,
        optical_blur: float,
        sensor_scale: float,
        chroma_filter: int,
        phase: tuple[int, int],
        **params: Any,
    ) -> np.ndarray:
        """Put the frame through the lens's blur, the mosaic, and the processor."""
        linear, scale, is_integer = to_linear_unit(img)
        height, width = img.shape[:2]
        sensor = (max(4, round(width / sensor_scale)), max(4, round(height / sensor_scale)))
        scene = linear
        if sensor != (width, height):
            scene = cv2.resize(linear, sensor, interpolation=cv2.INTER_AREA)
        if optical_blur > 0.05:
            scene = cv2.GaussianBlur(scene, (0, 0), sigmaX=optical_blur)
        recorded = smooth_chroma(through_mosaic(scene, phase), chroma_filter)
        if sensor != (width, height):
            recorded = cv2.resize(recorded, (width, height), interpolation=cv2.INTER_LINEAR)
        return from_linear_unit(recorded, scale, is_integer, img.dtype)
