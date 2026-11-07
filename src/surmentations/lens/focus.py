"""A lens that is not focused where it should be.

Physical model
--------------
A fixed-focus surveillance lens is set once, by eye, by day. Three things
then move the plane of focus without anyone touching it:

* **Infrared.** Glass bends near-infrared light less than visible light,
  so a lens that is not IR-corrected focuses the illuminator's light
  behind the sensor. The picture that was sharp at noon is soft every
  night. This is the commonest of the three, and the reason "IR-corrected"
  is printed on lens boxes.
* **Temperature.** A plastic lens barrel grows and shrinks by tens of
  micrometres between a winter night and a summer afternoon.
* **A ring that was never locked,** and turns with vibration.

All three displace the sensor from the plane where distant objects focus,
which is the same thing as the lens being focused at some near distance
``s`` instead of far away. From that everything follows, by the thin-lens
formula:

**A point is drawn as the aperture.** At distance ``z`` its blur disc has
diameter ``A f |1/s - 1/z|`` pixels, for an aperture ``A`` and a focal
length of ``f`` pixels. It is a disc, with an edge, not a Gaussian.

**Far things are blurred most, and all alike.** Beyond a few times ``s``
the disc is ``A f / s`` whatever the distance, which is why a mis-focused
wide lens looks evenly soft: a 4 mm lens with a 2 mm stop, focused at half
a metre, blurs everything past a few metres by about 3 px in 1000. With a
depth map the blur follows the formula per pixel, and whatever happens to
stand at ``s`` is sharp; without one the whole frame gets the far value.

The layers of blur are blended by depth with no account of what hides
what, so a sharp object in front of a blurred background does not bleed
its outline as a real lens would make it. At the few pixels involved that
is below what the picture resolves.
"""

from __future__ import annotations

from typing import Annotated, Any

import cv2
import numpy as np
from albumentations.core.pydantic import check_range_bounds, nondecreasing
from albumentations.core.transforms_interface import BaseTransformInitSchema
from pydantic import AfterValidator, ConfigDict

from surmentations.base import CameraTransform, scene_of
from surmentations.depth.provider import SharedDepthProvider
from surmentations.diagnostics import record_draw
from surmentations.lens.water_droplets import _aperture_blur, _disc_kernel
from surmentations.utils.photometry import from_linear_unit, to_linear_unit
from surmentations.utils.precipitation import focal_length_px

# How many blurs the range of disc sizes in a frame is drawn with.
_LAYERS = 6
# A frame whose discs differ by less than this, in pixels, gets one blur.
_SAME_BLUR_PX = 0.4


def blur_diameter(
    depth_m: np.ndarray | float, focus_m: float, aperture_m: float, focal_px: float
) -> np.ndarray:
    """Diameter in pixels of the blur disc of a point at ``depth_m``: ``A f |1/s - 1/z|``."""
    depth = np.maximum(np.asarray(depth_m, dtype=np.float32), 1e-3)
    diameter: np.ndarray = aperture_m * focal_px * np.abs(1.0 / focus_m - 1.0 / depth)
    return diameter


def disc_blur(linear: np.ndarray, diameter_px: float) -> np.ndarray:
    """Convolve with a disc ``diameter_px`` across; one a pixel across or less changes nothing.

    ``water_droplets``' aperture blur, which works on a reduced copy for a
    large disc, and below the size it starts at the same disc at full size:
    a focus error is often a pixel or two.
    """
    if diameter_px <= 1.0:
        return linear
    if diameter_px >= 1.5:
        return _aperture_blur(linear, diameter_px)
    blurred: np.ndarray = cv2.filter2D(
        linear, -1, _disc_kernel(diameter_px), borderType=cv2.BORDER_REFLECT_101
    )
    return blurred


def blur_by_depth(linear: np.ndarray, diameter_px: np.ndarray) -> np.ndarray:
    """Blur each pixel by a disc of its own diameter, ``diameter_px`` being ``(H, W)``.

    The frame is blurred whole at a few diameters spanning those asked for,
    and each pixel takes its value from the two nearest, in proportion.
    """
    low, high = float(diameter_px.min()), float(diameter_px.max())
    if high - low < _SAME_BLUR_PX:
        return disc_blur(linear, 0.5 * (low + high))
    levels = np.linspace(low, high, _LAYERS)
    position = (diameter_px - low) / (high - low) * (_LAYERS - 1)
    out = np.zeros_like(linear)
    for index, level in enumerate(levels):
        weight = np.clip(1.0 - np.abs(position - index), 0.0, 1.0)
        if float(weight.max()) > 0.0:
            out += disc_blur(linear, float(level)) * weight[..., None]
    return out


class FocusShift(CameraTransform):
    """A lens focused too near: by infrared, by temperature, or by a loose ring.

    Works with or without depth. Given a ``depth_provider`` the blur follows
    each pixel's distance; without one the frame is blurred as a distant
    scene is, evenly.

    Args:
        depth_provider: one shared provider, or ``None`` for an even blur.
        focus_distance_range_m: where the lens is in fact focused. The
            nearer, the worse: everything far is blurred by a disc
            ``A f / s`` across. Drawn evenly in ``1 / s``, which is what
            the blur is proportional to, so that the range is covered
            evenly in blur and not mostly at its sharp end.
        aperture_diameter_range_mm: the lens's stop. The blur disc is its
            image, so an open lens at night is blurred most.
        horizontal_fov_range_deg: field of view, which gives the focal
            length in pixels.

    Targets:
        image

    Image types:
        uint8, float32
    """

    class InitSchema(BaseTransformInitSchema):
        """The arguments of ``FocusShift`` as Albumentations validates them: each
        range ordered low to high and within the bounds given here.
        """

        model_config = ConfigDict(arbitrary_types_allowed=True)

        depth_provider: SharedDepthProvider | None
        focus_distance_range_m: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.01, 1000.0)),
        ]
        aperture_diameter_range_mm: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.2, 20.0)),
        ]
        horizontal_fov_range_deg: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(2.0, 150.0)),
        ]

    def __init__(
        self,
        depth_provider: SharedDepthProvider | None = None,
        focus_distance_range_m: tuple[float, float] = (0.08, 1.5),
        aperture_diameter_range_mm: tuple[float, float] = (0.9, 2.6),
        horizontal_fov_range_deg: tuple[float, float] = (48.0, 78.0),
        p: float = 0.5,
    ):
        """Store the provider and the ranges; each is described in the class docstring."""
        super().__init__(p=p)
        self.depth_provider = depth_provider
        self.focus_distance_range_m = focus_distance_range_m
        self.aperture_diameter_range_mm = aperture_diameter_range_mm
        self.horizontal_fov_range_deg = horizontal_fov_range_deg

    def get_params_dependent_on_data(
        self, params: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, Any]:
        """Draw where the lens is focused and work out each pixel's blur.

        Returns ``diameter``: the blur disc's diameter in pixels, per pixel
        where there is depth and a single number where there is not.
        """
        height, width = params["shape"][:2]
        near, far_focus = self.focus_distance_range_m
        focus = 1.0 / self.py_random.uniform(1.0 / far_focus, 1.0 / near)
        aperture_m = self.py_random.uniform(*self.aperture_diameter_range_mm) * 1e-3
        focal = focal_length_px(
            width, height, self.py_random.uniform(*self.horizontal_fov_range_deg)
        )
        far = float(blur_diameter(np.inf, focus, aperture_m, focal))
        diameter: np.ndarray | float = far
        if self.depth_provider is not None:
            depth_m = self.depth_provider.get_metric_depth(scene_of(data))
            diameter = blur_diameter(depth_m, focus, aperture_m, focal)
        record_draw(
            focus_distance_m=focus,
            aperture_mm=aperture_m * 1e3,
            focal_px=focal,
            blur_diameter_far_px=far,
            by_depth=self.depth_provider is not None,
        )
        return {"diameter": diameter}

    def apply(self, img: np.ndarray, diameter: np.ndarray | float, **params: Any) -> np.ndarray:
        """Blur the frame by the aperture's disc, at each pixel's own size."""
        linear, scale, is_integer = to_linear_unit(img)
        if isinstance(diameter, np.ndarray):
            out = blur_by_depth(linear, diameter)
        else:
            out = disc_blur(linear, float(diameter))
        return from_linear_unit(out, scale, is_integer, img.dtype)
