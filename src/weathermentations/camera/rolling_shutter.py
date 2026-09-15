"""Rolling-shutter distortion from camera motion during readout.

Physical model
--------------
A CMOS sensor exposes and reads its rows sequentially rather than all at
once. Row ``r`` is therefore sampled at a different instant from row 0, later
by ``r/H * readout_time``. If the camera moves during that interval, each row
records the scene from a slightly different pose, and the frame is distorted
rather than merely blurred.

Two motions matter for a fixed surveillance camera, and both are whole-frame:

* **Pole sway and pan.** A steady angular rate ``omega`` displaces row ``r``
  by ``f * omega * (r/H) * readout_time`` — a displacement growing linearly
  down the frame, i.e. a **shear**. Vertical structures lean.
* **Vibration.** Wind, traffic and nearby machinery excite the mount at a few
  Hz to a few tens of Hz. The displacement is then oscillatory in row index
  and the frame **wobbles**: verticals take on an S-bend rather than a lean.
  The number of wiggles across the frame is the vibration frequency times the
  readout time.

Scope, stated plainly: this models motion of the **camera**, which shifts
every row of the frame together. Rolling shutter also skews *individual
moving objects* relative to a static background — a passing car leans while
the building behind it does not — and that needs per-object motion, which a
single image does not carry. This transform does not attempt it; nothing here
silently stands in for it.
"""

from __future__ import annotations

from typing import Annotated, Any

import cv2
import numpy as np
from albumentations.core.pydantic import check_range_bounds, nondecreasing
from albumentations.core.transforms_interface import BaseTransformInitSchema
from pydantic import AfterValidator

from weathermentations.base import WeatherTransform


class RollingShutter(WeatherTransform):
    """Shear and wobble from line-by-line readout under camera motion.

    Args:
        pan_shift_range_px: horizontal displacement accumulated between the
            first row and the last, from a steady pan or sway. Signed: the
            camera can drift either way.
        wobble_amplitude_range_px: amplitude of the oscillatory component from
            mount vibration.
        wobble_cycles_range: how many vibration cycles elapse during readout,
            i.e. how many wiggles appear down the frame. Below 1 the wobble is
            indistinguishable from a lean; a few cycles gives the
            characteristic S-bend.
        vertical_fraction: how much of the motion is vertical rather than
            horizontal. A vertical component compresses and stretches rows
            instead of shifting them, which is the less common case for a pole
            mount but not a negligible one.

    Targets:
        image

    Image types:
        uint8, float32
    """

    class InitSchema(BaseTransformInitSchema):
        pan_shift_range_px: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(-200.0, 200.0)),
        ]
        wobble_amplitude_range_px: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 100.0)),
        ]
        wobble_cycles_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 20.0)),
        ]
        vertical_fraction_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 1.0)),
        ]

    def __init__(
        self,
        pan_shift_range_px: tuple[float, float] = (-12.0, 12.0),
        wobble_amplitude_range_px: tuple[float, float] = (0.0, 5.0),
        wobble_cycles_range: tuple[float, float] = (0.5, 4.0),
        vertical_fraction_range: tuple[float, float] = (0.0, 0.35),
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.pan_shift_range_px = pan_shift_range_px
        self.wobble_amplitude_range_px = wobble_amplitude_range_px
        self.wobble_cycles_range = wobble_cycles_range
        self.vertical_fraction_range = vertical_fraction_range

    def get_params_dependent_on_data(
        self, params: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, Any]:
        del data
        return {
            "pan_shift": self.py_random.uniform(*self.pan_shift_range_px),
            "wobble_amplitude": self.py_random.uniform(*self.wobble_amplitude_range_px),
            "wobble_cycles": self.py_random.uniform(*self.wobble_cycles_range),
            "wobble_phase": self.py_random.uniform(0.0, 2.0 * np.pi),
            "vertical_fraction": self.py_random.uniform(*self.vertical_fraction_range),
        }

    def apply(
        self,
        img: np.ndarray,
        pan_shift: float,
        wobble_amplitude: float,
        wobble_cycles: float,
        wobble_phase: float,
        vertical_fraction: float,
        **params: Any,
    ) -> np.ndarray:
        height, width = img.shape[:2]
        if abs(pan_shift) < 0.05 and wobble_amplitude < 0.05:
            return img

        # Row index normalised to the readout interval: row 0 is read first.
        rows = np.arange(height, dtype=np.float32) / max(height - 1, 1)

        displacement = pan_shift * rows + wobble_amplitude * np.sin(
            2.0 * np.pi * wobble_cycles * rows + wobble_phase
        )

        horizontal = displacement * (1.0 - vertical_fraction)
        vertical = displacement * vertical_fraction

        # Sampling *backwards*: to place row r's content where the moved
        # camera would have put it, read from where it came from.
        map_x = np.arange(width, dtype=np.float32)[None, :] - horizontal[:, None]
        map_y = np.arange(height, dtype=np.float32)[:, None] - vertical[:, None]

        return cv2.remap(
            img,
            np.ascontiguousarray(np.broadcast_to(map_x, (height, width)), dtype=np.float32),
            np.ascontiguousarray(np.broadcast_to(map_y, (height, width)), dtype=np.float32),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
