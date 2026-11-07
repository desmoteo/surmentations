"""A day/night filter that has stuck open.

Physical model
--------------
A silicon sensor sees to about 1000 nm, well past the red end of the
visible spectrum, and the dyes of its colour filters are all nearly clear
there. So a colour camera has an infrared-cut filter in front of the
sensor, and a day/night camera has one on a small motor: in place by day,
swung away at dusk so that the sensor can use the illuminator's light.

The motor is the camera's one moving part and it sticks. Stuck in place,
the night picture is black. Stuck open, the camera goes on producing a
colour picture by day with the near infrared added to all three channels,
and that is the fault this models, because it does not look like a fault
until one knows it:

**Every channel gains the same infrared signal, more or less.** Past
800 nm the red, green and blue filters pass much the same, so the infrared
image is added almost equally to all three: colours wash out. Red passes a
little more, so what is left is pink or purple.

**The infrared image is not the visible one.** Foliage reflects about half
of the near infrared and a twentieth of red, so trees and grass, dark
green by eye, gain most and turn pale, between pink and lavender. Black synthetic cloth and many
black plastics are bright in the infrared and turn purple-brown. Sky and
water are dark in it and keep their colour. The estimate of infrared
reflectance is ``NightIrMode``'s.

**The camera then corrects what it can.** Auto-exposure brings the level
back down, and auto white balance takes out the cast on average. What
neither can undo is that the added signal follows a different picture: the
foliage stays pink against a scene that is otherwise nearly right.

**A filter stuck half-way covers half the sensor.** The edge of the filter
lies across the picture, blurred because it is just in front of the sensor
and not on it: one side normal, the other washed pink.
"""

from __future__ import annotations

import math
from typing import Annotated, Any

import numpy as np
from albumentations.core.pydantic import check_range_bounds, nondecreasing
from albumentations.core.transforms_interface import BaseTransformInitSchema
from pydantic import AfterValidator

from surmentations.base import CameraTransform
from surmentations.camera.night_ir import near_infrared_reflectance
from surmentations.diagnostics import record_draw
from surmentations.utils.channels import frame_mean
from surmentations.utils.photometry import from_linear_unit, to_linear_unit

# How much of the near infrared each colour filter passes, relative to red:
# nearly the same, red a little ahead.
NIR_THROUGH_DYES = np.array([1.0, 0.86, 0.80], dtype=np.float32)
# Weights of the three channels in the level auto-exposure holds.
_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def unfiltered_side(
    height: int, width: int, share: float, angle: float, softness_px: float
) -> np.ndarray:
    """How much of each pixel is behind no filter: ``(H, W)``, 0 to 1.

    The filter's edge is a straight line across the frame, at ``angle`` and
    placed so that ``share`` of the frame is uncovered, blurred over
    ``softness_px``. A share of one is the whole frame.
    """
    if share >= 1.0:
        return np.ones((height, width), dtype=np.float32)
    yy = np.arange(height, dtype=np.float32)[:, None] - (height - 1) / 2.0
    xx = np.arange(width, dtype=np.float32)[None, :] - (width - 1) / 2.0
    along = xx * math.cos(angle) + yy * math.sin(angle)
    # Where along the frame the edge must lie for `share` of it to be on
    # the open side: a quantile, so it holds at any angle.
    edge = float(np.quantile(along, share))
    side: np.ndarray = np.clip((edge - along) / max(softness_px, 1e-3) + 0.5, 0.0, 1.0)
    return side


class IrCutFilterFault(CameraTransform):
    """A colour picture taken with the infrared-cut filter stuck open.

    A daytime fault; needs no depth. Sensor stage.

    Args:
        leak_range: how much infrared signal reaches the sensor, relative
            to the visible signal of a white surface. About 1 for a filter
            fully out of the way in sunlight.
        coverage_range: how much of the sensor is uncovered. One is a
            filter fully open; less is one stuck part-way, with its edge
            across the picture.
        white_balance_adaptation_range: how far the camera's auto white
            balance takes the cast back out, 0 to 1. At zero the picture
            keeps a pink cast overall; at one it is neutral on average and
            only what is bright in the infrared stays pink.
        vegetation_gain_range: how strongly foliage stands out in the
            infrared, as in ``NightIrMode``.
        edge_softness_range: width of the blur of a half-open filter's
            edge, as a fraction of the diagonal.

    Targets:
        image

    Image types:
        uint8, float32
    """

    class InitSchema(BaseTransformInitSchema):
        """The arguments of ``IrCutFilterFault`` as Albumentations validates them: each
        range ordered low to high and within the bounds given here.
        """

        leak_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 4.0)),
        ]
        coverage_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.05, 1.0)),
        ]
        white_balance_adaptation_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 1.0)),
        ]
        vegetation_gain_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 3.0)),
        ]
        edge_softness_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 0.5)),
        ]

    def __init__(
        self,
        leak_range: tuple[float, float] = (0.3, 1.0),
        coverage_range: tuple[float, float] = (0.5, 1.0),
        white_balance_adaptation_range: tuple[float, float] = (0.0, 0.8),
        vegetation_gain_range: tuple[float, float] = (0.25, 0.7),
        edge_softness_range: tuple[float, float] = (0.02, 0.08),
        p: float = 0.5,
    ):
        """Store the ranges; each argument is described in the class docstring."""
        super().__init__(p=p)
        self.leak_range = leak_range
        self.coverage_range = coverage_range
        self.white_balance_adaptation_range = white_balance_adaptation_range
        self.vegetation_gain_range = vegetation_gain_range
        self.edge_softness_range = edge_softness_range

    def get_params_dependent_on_data(
        self, params: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, Any]:
        """Draw how far open the filter is and what the camera does about it.

        Returns the ``open_side`` of the sensor, the ``leak``, the white
        balance ``adaptation`` and the ``vegetation_gain``.
        """
        del data
        height, width = params["shape"][:2]
        leak = self.py_random.uniform(*self.leak_range)
        coverage = self.py_random.uniform(*self.coverage_range)
        softness = self.py_random.uniform(*self.edge_softness_range) * float(
            np.hypot(width, height)
        )
        # The filter slides across the sensor, usually sideways.
        angle = float(self.random_generator.choice([0.0, math.pi]))
        angle += self.py_random.uniform(-0.3, 0.3)
        record_draw(leak=leak, coverage=coverage, edge_angle_rad=angle, edge_softness_px=softness)
        return {
            "open_side": unfiltered_side(height, width, coverage, angle, softness),
            "leak": leak,
            "adaptation": self.py_random.uniform(*self.white_balance_adaptation_range),
            "vegetation_gain": self.py_random.uniform(*self.vegetation_gain_range),
        }

    def apply(
        self,
        img: np.ndarray,
        open_side: np.ndarray,
        leak: float,
        adaptation: float,
        vegetation_gain: float,
        **params: Any,
    ) -> np.ndarray:
        """Add the infrared picture to all three channels, then expose and balance as a camera does.

        ``raw = rgb + leak * nir * dyes`` where the filter is open. White
        balance scales each channel towards the mean it had, by
        ``adaptation``; exposure then scales all three to the level the
        frame had.
        """
        if leak <= 0.0:
            return img
        linear, scale, is_integer = to_linear_unit(img)
        infrared = near_infrared_reflectance(linear, vegetation_gain) * (leak * open_side)
        raw = linear + infrared[..., None] * NIR_THROUGH_DYES[None, None, :]

        before, after = frame_mean(linear), frame_mean(raw)
        # What a grey-world balance would do, taken part of the way.
        balance = (np.maximum(before, 1e-6) / np.maximum(after, 1e-6)) ** adaptation
        raw *= balance[None, None, :].astype(np.float32)
        # Auto-exposure: the level the frame had before.
        level = float(frame_mean(raw) @ _LUMA)
        raw *= np.float32(float(before @ _LUMA) / max(level, 1e-6))
        return from_linear_unit(raw, scale, is_integer, img.dtype)
