"""Heat shimmer: the air itself as a bad lens.

Physical model
--------------
Over ground warmer than the air, the air is a mixture of parcels at
different temperatures and so of different refractive index, churning. A
ray through it is bent a little at every parcel. Over a short path that is
nothing; over hundreds of metres, seen through a long lens, distant detail
ripples and loses its edges. It is what limits a long-range camera on a
warm afternoon, and it is the standard theory of seeing, Kolmogorov
turbulence, with the strength of the turbulence given by the
refractive-index structure constant ``Cn2``: about ``1e-14 m^-2/3`` on a
mild day near the ground and ``1e-12`` over hot tarmac.

Two things happen to a point at distance ``L``, both growing with ``L``:

**It is displaced.** The wavefront reaching the lens is tilted, by an
angle whose variance over an aperture ``D`` is::

    sigma^2 = 2.91 D^(-1/3) Cn2 L      (radians squared)

and the tilt differs from place to place across the picture, coherent
over patches of some tens of pixels. So the picture is warped by a smooth
random field, scaled at each pixel by the square root of its distance.

**It is blurred.** Scales of turbulence smaller than the aperture smear
the image instead of moving it. The blur is that of an aperture the size
of the coherence length ``r0 = (0.423 k^2 Cn2 L)^(-3/5)`` (``k`` the
wavenumber of the light): an angle of about ``lambda / r0``, where that is
worse than the lens's own diffraction, ``lambda / D``, and nothing where
it is not.

What follows from the numbers is the test of the model: over the 3 to
50 m of an ordinary street view it does nothing at any turbulence, and it
takes a path of a kilometre and a lens of a few degrees to move a point by
a pixel. Give the transform a depth range and a field of view that say so.

The warp moves pixels by a pixel or two at most and by nothing on
average, so annotations are left where they are. A single frame shows
shimmer only as distortion; its motion from frame to frame is not
modelled.
"""

from __future__ import annotations

import math
from typing import Annotated, Any

import cv2
import numpy as np
from albumentations.core.pydantic import check_range_bounds, nondecreasing
from albumentations.core.transforms_interface import BaseTransformInitSchema
from pydantic import AfterValidator, ConfigDict, Field

from weathermentations.base import WeatherTransform, scene_of
from weathermentations.depth.provider import SharedDepthProvider
from weathermentations.diagnostics import record_draw
from weathermentations.utils.photometry import from_linear_unit, to_linear_unit
from weathermentations.utils.precipitation import focal_length_px

# Wavelength the blur is worked out for: the middle of the visible band.
_WAVELENGTH_M = 550e-9
# Blurs the frame is drawn with across its range of distances.
_LAYERS = 5
# A Gaussian's full width at half its height, in sigmas.
_FWHM = 2.355


def tilt_sigma_rad(distance_m: np.ndarray | float, cn2: float, aperture_m: float) -> np.ndarray:
    """Standard deviation of the angle a point at a distance is displaced by, in radians."""
    distance = np.maximum(np.asarray(distance_m, dtype=np.float64), 0.0)
    sigma: np.ndarray = np.sqrt(2.91 * aperture_m ** (-1.0 / 3.0) * cn2 * distance)
    return sigma


def blur_fwhm_rad(distance_m: np.ndarray | float, cn2: float, aperture_m: float) -> np.ndarray:
    """Angular width of the blur turbulence adds beyond the lens's own diffraction, in radians.

    ``lambda / r0`` and ``lambda / D`` in quadrature, the second taken off
    the first: nothing while the coherence length is larger than the
    aperture.
    """
    distance = np.maximum(np.asarray(distance_m, dtype=np.float64), 1e-9)
    wavenumber = 2.0 * math.pi / _WAVELENGTH_M
    coherence = (0.423 * wavenumber**2 * cn2 * distance) ** (-3.0 / 5.0)
    seeing = 0.98 * _WAVELENGTH_M / coherence
    own = 0.98 * _WAVELENGTH_M / aperture_m
    extra: np.ndarray = np.sqrt(np.maximum(seeing**2 - own**2, 0.0))
    return extra


def smooth_field(rng: np.random.Generator, height: int, width: int, cell_px: float) -> np.ndarray:
    """A smooth random field with zero mean and unit variance, coherent over ``cell_px``."""
    rows = max(2, round(height / cell_px) + 2)
    columns = max(2, round(width / cell_px) + 2)
    coarse = rng.normal(0.0, 1.0, (rows, columns)).astype(np.float32)
    enlarged: np.ndarray = cv2.resize(coarse, (width, height), interpolation=cv2.INTER_CUBIC)
    field = (enlarged - float(enlarged.mean())) / max(float(enlarged.std()), 1e-6)
    return field.astype(np.float32)


def blur_by_sigma(linear: np.ndarray, sigma_px: np.ndarray) -> np.ndarray:
    """Gaussian blur with a sigma of its own at each pixel, ``sigma_px`` being ``(H, W)``.

    The frame is blurred whole at a few sigmas spanning those asked for,
    and each pixel takes its value from the two nearest, in proportion.
    Under a fifth of a pixel everywhere, nothing is done.
    """
    low, high = float(sigma_px.min()), float(sigma_px.max())
    if high < 0.2:
        return linear
    levels = np.linspace(low, high, _LAYERS)
    position = (sigma_px - low) / max(high - low, 1e-6) * (_LAYERS - 1)
    out = np.zeros_like(linear)
    for index, level in enumerate(levels):
        weight = np.clip(1.0 - np.abs(position - index), 0.0, 1.0)
        if float(weight.max()) <= 0.0:
            continue
        layer = linear if level < 0.2 else cv2.GaussianBlur(linear, (0, 0), sigmaX=float(level))
        out += layer * weight[..., None]
    return out


class HeatShimmer(WeatherTransform):
    """Distant detail rippled and softened by turbulent air over a long path.

    Depth-conditioned, and it needs that depth to be long: over a street's
    length it rightly does nothing. Moves pixels by a pixel or two;
    annotations are not moved.

    Args:
        depth_provider: one shared provider.
        log10_cn2_range: strength of the turbulence, as the base-10
            logarithm of ``Cn2`` in ``m^-2/3``: -14 on a mild day, -12 over
            hot ground.
        horizontal_fov_range_deg: field of view. Shimmer is an angle, so a
            long lens shows it and a wide one does not.
        aperture_diameter_range_mm: the lens's aperture. A larger one
            averages the tilt over more air and is displaced less, and
            resolves finer, so it is blurred more.
        cell_size_range: size of the patches the picture moves in as one,
            as a fraction of the diagonal.
        depth_scale: pseudo-metres at relative depth 1, for this transform
            alone; ``None`` uses the provider's own far plane.

    Targets:
        image

    Image types:
        uint8, float32
    """

    class InitSchema(BaseTransformInitSchema):
        """The arguments of ``HeatShimmer`` as Albumentations validates them: each
        range ordered low to high and within the bounds given here.
        """

        model_config = ConfigDict(arbitrary_types_allowed=True)

        depth_provider: SharedDepthProvider
        log10_cn2_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(-17.0, -10.0)),
        ]
        horizontal_fov_range_deg: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.2, 150.0)),
        ]
        aperture_diameter_range_mm: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.2, 500.0)),
        ]
        cell_size_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.002, 0.5)),
        ]
        depth_scale: float | None = Field(default=None, gt=0.0)

    def __init__(
        self,
        depth_provider: SharedDepthProvider,
        log10_cn2_range: tuple[float, float] = (-13.5, -12.0),
        horizontal_fov_range_deg: tuple[float, float] = (2.0, 12.0),
        aperture_diameter_range_mm: tuple[float, float] = (10.0, 40.0),
        cell_size_range: tuple[float, float] = (0.01, 0.04),
        depth_scale: float | None = None,
        p: float = 0.5,
    ):
        """Store the provider and the ranges; each is described in the class docstring."""
        super().__init__(p=p)
        self.depth_provider = depth_provider
        self.log10_cn2_range = log10_cn2_range
        self.horizontal_fov_range_deg = horizontal_fov_range_deg
        self.aperture_diameter_range_mm = aperture_diameter_range_mm
        self.cell_size_range = cell_size_range
        self.depth_scale = depth_scale

    def get_params_dependent_on_data(
        self, params: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, Any]:
        """Draw the air and the lens, and each pixel's displacement and blur from its distance.

        Returns the ``shift``, ``(H, W, 2)`` of pixels across and down, and
        the blur's ``sigma`` per pixel.
        """
        scene = scene_of(data)
        height, width = params["shape"][:2]
        rng = self.random_generator
        cn2 = 10.0 ** self.py_random.uniform(*self.log10_cn2_range)
        focal = focal_length_px(
            width, height, self.py_random.uniform(*self.horizontal_fov_range_deg)
        )
        aperture_m = self.py_random.uniform(*self.aperture_diameter_range_mm) * 1e-3
        cell = self.py_random.uniform(*self.cell_size_range) * float(np.hypot(width, height))

        depth_m = self.depth_provider.to_metric(
            self.depth_provider.get_depth(scene), far_m=self.depth_scale
        )
        tilt_px = (tilt_sigma_rad(depth_m, cn2, aperture_m) * focal).astype(np.float32)
        shift = np.stack(
            [smooth_field(rng, height, width, cell) * tilt_px for _ in range(2)], axis=2
        )
        sigma = (blur_fwhm_rad(depth_m, cn2, aperture_m) * focal / _FWHM).astype(np.float32)

        record_draw(
            cn2=cn2,
            focal_px=focal,
            aperture_mm=aperture_m * 1e3,
            cell_px=cell,
            depth_max_m=lambda: float(depth_m.max()),
            tilt_sigma_max_px=lambda: float(tilt_px.max()),
            blur_sigma_max_px=lambda: float(sigma.max()),
        )
        return {"shift": shift, "sigma": sigma}

    def apply(
        self, img: np.ndarray, shift: np.ndarray, sigma: np.ndarray, **params: Any
    ) -> np.ndarray:
        """Blur each pixel by the air in front of it, then move the picture by the air's tilt."""
        if float(np.abs(shift).max()) < 0.02 and float(sigma.max()) < 0.2:
            return img
        linear, scale, is_integer = to_linear_unit(img)
        linear = blur_by_sigma(linear, sigma)
        height, width = img.shape[:2]
        map_x = np.arange(width, dtype=np.float32)[None, :] + shift[..., 0]
        map_y = np.arange(height, dtype=np.float32)[:, None] + shift[..., 1]
        linear = cv2.remap(
            linear, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
        )
        return from_linear_unit(linear, scale, is_integer, img.dtype)
