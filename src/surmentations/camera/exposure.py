"""Auto-exposure and auto-white-balance caught mid-correction.

Physical model
--------------
A fixed camera's automatic exposure and white balance are closed loops with
lag. They are *continuously* wrong by some amount and continuously correcting:
a headlight enters frame and the whole image darkens for a second before
recovering; a cloud crosses and the colour cast swings. The settled state a
training set is usually built from is only one of the states the camera
spends its time in, and the transient ones are just as common.

Two errors, modelled separately because they have different physics:

* **Exposure error** is a gain on radiance, so it is expressed in **stops**
  (powers of two) and applied in linear light. A stop is a factor of two in
  the light reaching the sensor; applying a "brightness" offset to
  gamma-encoded values is a different operation that no camera performs.
  Overshoot is signed, because a hunting loop passes through the right answer
  from both sides.

* **White-balance error** is *not* an arbitrary per-channel gain. AWB
  estimates the illuminant's colour temperature and divides it out; when it
  guesses wrong, what is left is the ratio between the blackbody colour it
  assumed and the one actually lighting the scene. That ratio lies along the
  blue-orange axis, because that is the axis real illuminants vary on.
  Independent random channel gains wander off it and leave casts — strong
  green, strong magenta — that a white-balance algorithm would never produce.
  See :func:`~surmentations.utils.spectral.blackbody_rgb`.

A smaller green-magenta error is applied on top, which *is* the residual AWB
leaves on non-blackbody illuminants such as fluorescent and LED lighting —
the "tint" axis, orthogonal to temperature.
"""

from __future__ import annotations

from typing import Annotated, Any

import numpy as np
from albumentations.core.pydantic import check_range_bounds, nondecreasing
from albumentations.core.transforms_interface import BaseTransformInitSchema
from pydantic import AfterValidator

from surmentations.base import CameraTransform
from surmentations.diagnostics import record_draw
from surmentations.utils.photometry import from_linear_unit, to_linear_unit
from surmentations.utils.spectral import blackbody_rgb


class ExposureTransient(CameraTransform):
    """Auto-exposure / auto-white-balance error, as a camera mid-hunt shows it.

    Args:
        exposure_error_stops_range: signed exposure error in stops. +1 is one
            stop over (twice the light), -1 one stop under.
        assumed_temperature_range_k: the colour temperature AWB *thinks* it is
            correcting for.
        actual_temperature_range_k: the temperature actually lighting the
            scene. The residual cast is the ratio of the two blackbody
            colours, so equal ranges give a correct white balance on average
            and a realistic spread of errors around it.
        tint_error_range: residual green-magenta error, the axis orthogonal to
            colour temperature. Positive is green, negative magenta; non-
            blackbody sources (fluorescent, LED) are what leave it.

    Targets:
        image

    Image types:
        uint8, float32
    """

    class InitSchema(BaseTransformInitSchema):
        """The arguments of ``ExposureTransient`` as Albumentations validates them: each
        range ordered low to high and within the bounds given here.
        """
        exposure_error_stops_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(-6.0, 6.0)),
        ]
        assumed_temperature_range_k: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(1500.0, 20000.0)),
        ]
        actual_temperature_range_k: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(1500.0, 20000.0)),
        ]
        tint_error_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(-0.5, 0.5)),
        ]

    def __init__(
        self,
        exposure_error_stops_range: tuple[float, float] = (-1.2, 1.2),
        assumed_temperature_range_k: tuple[float, float] = (4000.0, 7000.0),
        actual_temperature_range_k: tuple[float, float] = (3000.0, 8000.0),
        tint_error_range: tuple[float, float] = (-0.08, 0.08),
        p: float = 0.5,
    ):
        """Store the ranges; each argument is described in the class docstring."""
        super().__init__(p=p)
        self.exposure_error_stops_range = exposure_error_stops_range
        self.assumed_temperature_range_k = assumed_temperature_range_k
        self.actual_temperature_range_k = actual_temperature_range_k
        self.tint_error_range = tint_error_range

    def get_params_dependent_on_data(
        self, params: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, Any]:
        """Draw the exposure error and the white-balance error the camera is caught with.

        Returns the overall ``exposure_gain`` and the ``channel_gain``: the
        ratio between the illuminant the white balance assumed and the one
        that was really there.
        """
        del data
        assumed = self.py_random.uniform(*self.assumed_temperature_range_k)
        actual = self.py_random.uniform(*self.actual_temperature_range_k)

        # What AWB failed to remove: the ratio between the illuminant it
        # assumed and the one that was really there.
        cast = blackbody_rgb(actual) / np.maximum(blackbody_rgb(assumed), 1e-6)

        tint = self.py_random.uniform(*self.tint_error_range)
        cast = cast * np.array([1.0 - tint, 1.0 + tint, 1.0 - tint], dtype=np.float32)

        stops = self.py_random.uniform(*self.exposure_error_stops_range)
        record_draw(
            assumed_temperature_k=assumed,
            actual_temperature_k=actual,
            tint=tint,
            exposure_error_stops=stops,
        )
        return {
            "exposure_gain": float(2.0**stops),
            "channel_gain": (cast / max(float(cast[1]), 1e-6)).astype(np.float32),
        }

    def apply(
        self,
        img: np.ndarray,
        exposure_gain: float,
        channel_gain: np.ndarray,
        **params: Any,
    ) -> np.ndarray:
        """Scale the frame in linear light by both gains; encoding it back clips past white."""
        linear, scale, is_integer = to_linear_unit(img)
        linear = linear * exposure_gain * channel_gain[None, None, :]
        return from_linear_unit(linear, scale, is_integer, img.dtype)
