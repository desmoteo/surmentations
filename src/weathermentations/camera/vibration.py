"""A camera that shakes on its mount.

Physical model
--------------
A camera on a pole sways in wind at a few hertz; one on a bracket by a road
or a railway shakes with every lorry at tens of hertz. Either way the
camera's pointing swings about its mean, and a swing of a twentieth of a
degree is a pixel or more.

Two things reach the picture, and both follow from where the camera was
pointing while the shutter was open:

**The frame is displaced.** At the moment of the exposure the camera is
somewhere along its swing, so the whole picture sits a few pixels off
where it would be at rest: by the mean of the pointing over the exposure.

**It is smeared along the swing, by as much as it moved meanwhile.** The
exposure adds up the picture at every pointing it passed through. With a
short shutter, 1/2000 s by day, the camera has not moved and there is no
blur, only the displacement. With a long one, 1/25 s at night, it has
travelled a good part of the swing, or all of it and back. The smear is
that path: not a box along a line, but the time spent at each point of it,
which piles up at the ends of a swing where the motion turns round.

So the blur is drawn from the path, a sinusoid of the swing's amplitude
sampled over the exposure, and its length goes with the shutter time until
the exposure holds a whole half-cycle, where it stops growing at the full
swing.

The exposure is taken as the same instant for every row. A rolling
shutter reads rows at different times and turns the same motion into a
wobble, which ``RollingShutter`` models.

The displacement moves pixels, so masks, boxes and keypoints are moved
with them, and depth estimated before this transform is not carried past
it.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Annotated, Any

import cv2
import numpy as np
from albumentations.core.pydantic import check_range_bounds, nondecreasing
from albumentations.core.transforms_interface import BaseTransformInitSchema
from albumentations.core.type_definitions import Targets
from pydantic import AfterValidator

from weathermentations.base import WeatherTransform
from weathermentations.diagnostics import record_draw
from weathermentations.utils.photometry import from_linear_unit, to_linear_unit

# Moments of the exposure the path is sampled at.
_PATH_SAMPLES = 96
# Motion under this many pixels changes nothing worth resampling for.
_NEGLIGIBLE_PX = 0.05


def swing_path(
    amplitude_px: float, cycles: float, phase: float, direction_rad: float
) -> tuple[np.ndarray, tuple[float, float]]:
    """Where the picture is over one exposure, as a blur kernel and a displacement.

    The pointing follows ``amplitude sin(2 pi t + phase)`` along
    ``direction`` while ``t`` runs over ``cycles`` of the swing. Returns
    the kernel, the share of the exposure spent at each offset from the
    mean position, summing to one; and the mean position itself, ``(dx,
    dy)`` in pixels, by which the whole frame is displaced.
    """
    moments = (np.arange(_PATH_SAMPLES) + 0.5) / _PATH_SAMPLES
    along = amplitude_px * np.sin(2.0 * np.pi * cycles * moments + phase)
    mean = float(along.mean())
    offset = along - mean
    dx, dy = offset * math.cos(direction_rad), offset * math.sin(direction_rad)
    half = int(math.floor(float(np.abs(offset).max()))) + 1
    size = 2 * half + 1
    kernel = np.zeros((size, size), dtype=np.float64)
    # Each moment's share goes to the four pixels around where it falls.
    x, y = dx + half, dy + half
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - x0, y - y0
    for row, column, weight in (
        (y0, x0, (1 - fx) * (1 - fy)),
        (y0, x0 + 1, fx * (1 - fy)),
        (y0 + 1, x0, (1 - fx) * fy),
        (y0 + 1, x0 + 1, fx * fy),
    ):
        np.add.at(kernel, (row, column), weight)
    kernel /= kernel.sum()
    return kernel.astype(np.float32), (
        mean * math.cos(direction_rad),
        mean * math.sin(direction_rad),
    )


def _translate(image: np.ndarray, shift: tuple[float, float], interpolation: int) -> np.ndarray:
    """A frame moved by ``(dx, dy)`` pixels, its edge repeated where it is uncovered."""
    matrix = np.array([[1.0, 0.0, shift[0]], [0.0, 1.0, shift[1]]], dtype=np.float32)
    moved: np.ndarray = cv2.warpAffine(
        image,
        matrix,
        (image.shape[1], image.shape[0]),
        flags=interpolation,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return moved


class CameraVibration(WeatherTransform):
    """The frame displaced and smeared by a camera swinging on its mount.

    Needs no depth. Moves pixels: annotations move with the frame, and
    depth is not carried through.

    Args:
        amplitude_range: how far the picture swings either side of rest,
            as a fraction of the diagonal. A twentieth of a degree on a
            wide lens is about a thousandth.
        frequency_range_hz: how fast it swings: a few hertz for a pole in
            wind, tens for a bracket beside traffic.
        exposure_time_range: shutter time. The smear is what the camera
            moved in that time, so 1/2000 s leaves the displacement alone
            and 1/25 s smears by most of the swing.
        direction_range_deg: direction of the swing, from horizontal.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    _targets = (Targets.IMAGE, Targets.MASK, Targets.BBOXES, Targets.KEYPOINTS)

    # The frame is displaced, so depth estimated before this no longer
    # lines up with the pixels after it.
    preserves_scene_geometry = False

    class InitSchema(BaseTransformInitSchema):
        """The arguments of ``CameraVibration`` as Albumentations validates them: each
        range ordered low to high and within the bounds given here.
        """

        amplitude_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 0.05)),
        ]
        frequency_range_hz: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.1, 500.0)),
        ]
        exposure_time_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(1e-5, 0.5)),
        ]
        direction_range_deg: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(-180.0, 180.0)),
        ]

    def __init__(
        self,
        amplitude_range: tuple[float, float] = (0.0008, 0.006),
        frequency_range_hz: tuple[float, float] = (2.0, 40.0),
        exposure_time_range: tuple[float, float] = (0.0005, 0.04),
        direction_range_deg: tuple[float, float] = (-90.0, 90.0),
        p: float = 0.5,
    ):
        """Store the ranges; each argument is described in the class docstring."""
        super().__init__(p=p)
        self.amplitude_range = amplitude_range
        self.frequency_range_hz = frequency_range_hz
        self.exposure_time_range = exposure_time_range
        self.direction_range_deg = direction_range_deg

    def get_params_dependent_on_data(
        self, params: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, Any]:
        """Draw the swing and the exposure, and where in the swing the exposure falls.

        Returns the blur ``kernel`` and the frame's ``shift``, ``(dx, dy)``
        in pixels.
        """
        del data
        height, width = params["shape"][:2]
        amplitude = self.py_random.uniform(*self.amplitude_range) * float(np.hypot(width, height))
        frequency = self.py_random.uniform(*self.frequency_range_hz)
        exposure = self.py_random.uniform(*self.exposure_time_range)
        direction = math.radians(self.py_random.uniform(*self.direction_range_deg))
        phase = self.py_random.uniform(0.0, 2.0 * math.pi)
        kernel, shift = swing_path(amplitude, frequency * exposure, phase, direction)
        record_draw(
            amplitude_px=amplitude,
            frequency_hz=frequency,
            exposure_s=exposure,
            cycles_in_exposure=frequency * exposure,
            direction_deg=math.degrees(direction),
            shift_px=np.array(shift, dtype=np.float32),
            blur_extent_px=kernel.shape[0] - 3,
        )
        return {"kernel": kernel, "shift": shift}

    @property
    def targets(self) -> dict[str, Callable[..., Any]]:
        """The image is smeared and displaced; masks, boxes and keypoints are displaced."""
        return {
            **super().targets,
            "mask": self.apply_to_mask,
            "masks": self.apply_to_masks,
            "bboxes": self.apply_to_bboxes,
            "keypoints": self.apply_to_keypoints,
        }

    def apply(
        self, img: np.ndarray, kernel: np.ndarray, shift: tuple[float, float], **params: Any
    ) -> np.ndarray:
        """Add the picture up over the swing's path, in linear light, and displace it."""
        still = kernel.shape[0] <= 3 and float(kernel[1, 1]) > 0.999
        if still and max(abs(shift[0]), abs(shift[1])) < _NEGLIGIBLE_PX:
            return img
        linear, scale, is_integer = to_linear_unit(img)
        if not still:
            linear = cv2.filter2D(linear, -1, kernel, borderType=cv2.BORDER_REPLICATE)
        linear = _translate(linear, shift, cv2.INTER_LINEAR)
        return from_linear_unit(linear, scale, is_integer, img.dtype)

    def apply_to_mask(
        self, mask: np.ndarray, shift: tuple[float, float], **params: Any
    ) -> np.ndarray:
        """Displace a mask with the frame, by nearest neighbour: it holds labels."""
        if max(abs(shift[0]), abs(shift[1])) < 0.5:
            return mask
        return _translate(mask, shift, cv2.INTER_NEAREST)

    def apply_to_masks(self, masks: np.ndarray, *args: Any, **params: Any) -> np.ndarray:
        """Displace each mask of a stack as :meth:`apply_to_mask` does."""
        return np.stack([self.apply_to_mask(mask, **params) for mask in masks])

    def apply_to_keypoints(
        self, keypoints: np.ndarray, shift: tuple[float, float], **params: Any
    ) -> np.ndarray:
        """Move each keypoint with the frame."""
        if len(keypoints) == 0:
            return keypoints
        moved = keypoints.copy()
        moved[:, 0] += shift[0]
        moved[:, 1] += shift[1]
        return moved

    def apply_to_bboxes(
        self, bboxes: np.ndarray, shift: tuple[float, float], **params: Any
    ) -> np.ndarray:
        """Move each box with the frame. Boxes arrive normalised."""
        if len(bboxes) == 0:
            return bboxes
        height, width = params["shape"][:2]
        moved = bboxes.copy()
        moved[:, [0, 2]] += shift[0] / width
        moved[:, [1, 3]] += shift[1] / height
        return moved
