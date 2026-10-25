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

from collections.abc import Callable
from typing import Annotated, Any

import cv2
import numpy as np
from albumentations.core.pydantic import check_range_bounds, nondecreasing
from albumentations.core.transforms_interface import BaseTransformInitSchema
from albumentations.core.type_definitions import Targets
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
        image, mask, bboxes, keypoints

    Image types:
        any channel count and bit depth OpenCV can resample
    """

    # The only transform in the library that moves pixels, so the only one
    # whose annotations have to move with them. Left image-only — as it was —
    # a 30 px shear moved the picture 29 px and left the mask and the boxes
    # where they were.
    _targets = (Targets.IMAGE, Targets.MASK, Targets.BBOXES, Targets.KEYPOINTS)

    # Rows are displaced, so depth estimated before this no longer lines up
    # with the pixels after it.
    preserves_scene_geometry = False
    # A resampling of pixel positions: it does not care what the pixels are,
    # so greyscale (an IR camera's native output) and 16-bit frames are fine.
    requires_rgb = False

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

    @property
    def targets(self) -> dict[str, Callable[..., Any]]:
        return {
            **super().targets,
            "mask": self.apply_to_mask,
            "masks": self.apply_to_masks,
            "bboxes": self.apply_to_bboxes,
            "keypoints": self.apply_to_keypoints,
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
        if _is_negligible(pan_shift, wobble_amplitude):
            return img
        horizontal, vertical = _row_displacement(
            img.shape[0], pan_shift, wobble_amplitude, wobble_cycles, wobble_phase,
            vertical_fraction,
        )
        return _resample(img, horizontal, vertical, cv2.INTER_LINEAR)

    def apply_to_mask(
        self,
        mask: np.ndarray,
        pan_shift: float,
        wobble_amplitude: float,
        wobble_cycles: float,
        wobble_phase: float,
        vertical_fraction: float,
        **params: Any,
    ) -> np.ndarray:
        if _is_negligible(pan_shift, wobble_amplitude):
            return mask
        horizontal, vertical = _row_displacement(
            mask.shape[0], pan_shift, wobble_amplitude, wobble_cycles, wobble_phase,
            vertical_fraction,
        )
        # Nearest neighbour: a mask holds labels, and interpolating between
        # class 3 and class 7 invents a class 5 along every boundary.
        return _resample(mask, horizontal, vertical, cv2.INTER_NEAREST)

    def apply_to_masks(self, masks: np.ndarray, *args: Any, **params: Any) -> np.ndarray:
        return np.stack([self.apply_to_mask(mask, **params) for mask in masks])

    def apply_to_keypoints(
        self,
        keypoints: np.ndarray,
        pan_shift: float,
        wobble_amplitude: float,
        wobble_cycles: float,
        wobble_phase: float,
        vertical_fraction: float,
        **params: Any,
    ) -> np.ndarray:
        if _is_negligible(pan_shift, wobble_amplitude) or len(keypoints) == 0:
            return keypoints
        horizontal, vertical = _row_displacement(
            params["shape"][0], pan_shift, wobble_amplitude, wobble_cycles, wobble_phase,
            vertical_fraction,
        )
        moved = keypoints.copy()
        moved[:, 0], moved[:, 1] = _move_points(
            keypoints[:, 0], keypoints[:, 1], horizontal, vertical
        )
        return moved

    def apply_to_bboxes(
        self,
        bboxes: np.ndarray,
        pan_shift: float,
        wobble_amplitude: float,
        wobble_cycles: float,
        wobble_phase: float,
        vertical_fraction: float,
        **params: Any,
    ) -> np.ndarray:
        """Boxes arrive normalised, ``[x_min, y_min, x_max, y_max, ...]``.

        A sheared rectangle is not a rectangle, so the result is the box that
        encloses where the original's outline lands. The two vertical sides
        are sampled along their length, because the displacement varies down
        them: a wobble can push the middle of a side out further than either
        corner.
        """
        if _is_negligible(pan_shift, wobble_amplitude) or len(bboxes) == 0:
            return bboxes
        height, width = params["shape"][:2]
        horizontal, vertical = _row_displacement(
            height, pan_shift, wobble_amplitude, wobble_cycles, wobble_phase, vertical_fraction
        )
        along = np.linspace(0.0, 1.0, _BOX_SIDE_SAMPLES, dtype=np.float64)[None, :]
        y_min, y_max = bboxes[:, 1:2] * height, bboxes[:, 3:4] * height
        side_y = y_min + (y_max - y_min) * along
        moved = bboxes.copy()
        left_x, left_y = _move_points(
            np.broadcast_to(bboxes[:, 0:1] * width, side_y.shape), side_y, horizontal, vertical
        )
        right_x, right_y = _move_points(
            np.broadcast_to(bboxes[:, 2:3] * width, side_y.shape), side_y, horizontal, vertical
        )
        moved[:, 0] = np.minimum(left_x, right_x).min(axis=1) / width
        moved[:, 2] = np.maximum(left_x, right_x).max(axis=1) / width
        moved[:, 1] = np.minimum(left_y, right_y).min(axis=1) / height
        moved[:, 3] = np.maximum(left_y, right_y).max(axis=1) / height
        return moved


# Points sampled along each vertical side of a box when mapping it.
_BOX_SIDE_SAMPLES = 17


def _is_negligible(pan_shift: float, wobble_amplitude: float) -> bool:
    return abs(pan_shift) < 0.05 and wobble_amplitude < 0.05


def _row_displacement(
    height: int,
    pan_shift: float,
    wobble_amplitude: float,
    wobble_cycles: float,
    wobble_phase: float,
    vertical_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    """``(horizontal, vertical)`` displacement in pixels of each output row.

    One function for the image, the mask, the boxes and the keypoints, so
    there is exactly one description of where things went.
    """
    # Row index normalised to the readout interval: row 0 is read first.
    rows = np.arange(height, dtype=np.float32) / max(height - 1, 1)
    displacement = pan_shift * rows + wobble_amplitude * np.sin(
        2.0 * np.pi * wobble_cycles * rows + wobble_phase
    )
    return displacement * (1.0 - vertical_fraction), displacement * vertical_fraction


def _resample(
    array: np.ndarray, horizontal: np.ndarray, vertical: np.ndarray, interpolation: int
) -> np.ndarray:
    """Output row ``r`` shows the input displaced by that row's motion."""
    height, width = array.shape[:2]
    # Sampling *backwards*: to place row r's content where the moved camera
    # would have put it, read from where it came from.
    map_x = np.arange(width, dtype=np.float32)[None, :] - horizontal[:, None]
    map_y = np.arange(height, dtype=np.float32)[:, None] - vertical[:, None]
    warped: np.ndarray = cv2.remap(
        array,
        np.ascontiguousarray(np.broadcast_to(map_x, (height, width)), dtype=np.float32),
        np.ascontiguousarray(np.broadcast_to(map_y, (height, width)), dtype=np.float32),
        interpolation=interpolation,
        borderMode=cv2.BORDER_REPLICATE,
    )
    # OpenCV drops a trailing channel axis of one, turning (H, W, 1) into
    # (H, W).
    return warped.reshape(array.shape)


def _move_points(
    x: np.ndarray, y: np.ndarray, horizontal: np.ndarray, vertical: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Where source points ``(x, y)`` land in the output.

    The image is resampled backwards — output row ``r`` reads source row
    ``r - vertical[r]`` — so a source point at ``y`` lands on the row that
    satisfies ``r = y + vertical[r]``. That is implicit in ``r``, and is
    solved by iterating from ``r = y``; it converges in a few steps because
    the displacement changes by far less than a pixel per row.
    """
    rows = np.arange(len(vertical), dtype=np.float64)
    source_y = np.asarray(y, dtype=np.float64)
    landed = source_y.copy()
    for _ in range(12):
        landed = source_y + np.interp(landed, rows, vertical)
    return np.asarray(x, dtype=np.float64) + np.interp(landed, rows, horizontal), landed
