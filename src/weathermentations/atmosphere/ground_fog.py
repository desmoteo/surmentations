"""Fog that lies on the ground.

Physical model
--------------
``RandomFog`` fills the world evenly: the same extinction at every height,
which is advection fog or cloud at ground level. The fog a camera on a pole
meets most often is not that. Radiation fog forms on still clear nights in
the air chilled by the ground, and it is a layer: thick at the surface,
thinning with height, often shallower than the pole. Seen from above it,
the road is gone and the roofs and lamp heads stand clear of it.

**Density falls with height.** The layer is taken as exponential,
``beta(h) = beta0 exp(-h / H)``: extinction ``beta0`` at the ground, given
through the visibility there by Koschmieder's relation as in ``RandomFog``,
and a scale height ``H`` of a few metres.

**What a pixel sees is the integral along its ray.** From a camera at
height ``hc`` to a point at range ``r`` and height ``hp``, the optical
depth has a closed form::

    tau = beta0 r exp(-hc / H) (1 - exp(-u)) / u,   u = (hp - hc) / H

which is ``beta0 r exp(-hc / H)`` for a level ray and grows as the ray
dips into the layer. So the same distance is clear along a line of roofs
and opaque along the road, and a camera looking down from above the layer
sees the ground through less fog than one standing in it would. The
heights come from the depth map and the camera's height, pitch and field
of view; nothing is taken from image position alone.

**It is patchy.** A layer this shallow follows the ground and drifts in
banks, with wisps a few times smaller trailing from them. The extinction
is modulated by two noises laid out on the ground plane, in metres, and
read part-way along each ray, so that the banks have the size they are
given at any distance and foreshorten as the ground does.

**Above the layer the sky is clear, and the horizon is not.** A ray into
the sky leaves the layer and meets nothing more, so its optical depth is
finite however far it goes: ``beta0 exp(-hc / H) H / sin(elevation)``.
Straight up that is very little. Towards the horizon it grows without
limit, so the fog closes the horizon in a white band and thins upwards
into clear sky, which is the look of ground fog from above it. Sky pixels
are given that value from the ray's direction alone, since their depth is
not a distance.

**It softens what it veils.** Fog scatters mostly forwards, so what is
seen through it is blurred as well as faded, by more the more fog is in
the way, as in ``RandomFog``.

The veil's colour is ``RandomFog``'s: the horizon sky's radiance, read
where sky is in frame and inferred where it is not.
"""

from __future__ import annotations

import math
from typing import Annotated, Any

import cv2
import numpy as np
from albumentations.core.pydantic import check_range_bounds, nondecreasing
from albumentations.core.transforms_interface import BaseTransformInitSchema
from pydantic import AfterValidator, ConfigDict, Field

from weathermentations.base import WeatherTransform, first_frame, scene_of
from weathermentations.depth.provider import SharedDepthProvider
from weathermentations.diagnostics import record_draw
from weathermentations.utils.color import veil_radiance_linear
from weathermentations.utils.filters import wide_blur
from weathermentations.utils.noise import value_noise
from weathermentations.utils.photometry import from_linear_unit, to_linear_unit
from weathermentations.utils.precipitation import focal_length_px
from weathermentations.utils.sky import lift_sky, sky_to_depth_edge, sky_weight

# Koschmieder: visibility is where contrast has fallen to 2%.
_KOSCHMIEDER = 3.912
# Side of the tile of noise the banks are read from, and how many banks it
# holds across: it repeats beyond that, far enough apart not to show.
_BANK_TILE, _BANKS_ACROSS = 256, 12
# Wisps: how many times smaller than the banks, and their share of the
# unevenness.
_WISP_RATIO, _WISP_SHARE = 4.0, 0.35
# The most optical depth a ray into the sky is given: at the horizon the
# formula has no limit, and beyond this nothing is seen anyway.
_OPEN_SKY_CEILING = 12.0


def scene_points(
    depth_m: np.ndarray, focal_px: float, pitch_rad: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Where each pixel's point is, from a camera pitched down by ``pitch_rad``.

    ``depth_m`` is distance along the optical axis. Returns the point's
    ``range`` from the camera along its ray, its ``height`` relative to the
    camera (negative below it), and its position on the ground plane,
    ``across`` and ``forward``, all in metres and all ``(H, W)``.
    """
    height, width = depth_m.shape
    y = (np.arange(height, dtype=np.float32) - (height - 1) / 2.0)[:, None] / focal_px
    x = (np.arange(width, dtype=np.float32) - (width - 1) / 2.0)[None, :] / focal_px
    cos, sin = math.cos(pitch_rad), math.sin(pitch_rad)
    # Camera axes: x right, y down, z along the view. The world's up is
    # (0, -cos, -sin) in them.
    relative_height = -depth_m * (y * cos + sin)
    forward = depth_m * (cos - y * sin)
    across = depth_m * x
    distance = depth_m * np.sqrt(1.0 + x * x + y * y)
    return distance, relative_height, across, forward


def open_sky_optical_depth(
    height: int,
    width: int,
    focal_px: float,
    pitch_rad: float,
    *,
    beta_ground: float,
    scale_height_m: float,
    camera_height_m: float,
) -> np.ndarray:
    """Optical depth of a ray that leaves the layer and meets nothing: ``(H, W)``.

    ``beta0 exp(-hc / H) H / sin(elevation)`` for a ray above the horizon,
    from its direction alone; capped at ``_OPEN_SKY_CEILING``, which a ray
    at or below the horizon is given outright.
    """
    y = (np.arange(height, dtype=np.float32) - (height - 1) / 2.0)[:, None] / focal_px
    x = (np.arange(width, dtype=np.float32) - (width - 1) / 2.0)[None, :] / focal_px
    rising = -(y * math.cos(pitch_rad) + math.sin(pitch_rad)) / np.sqrt(1.0 + x * x + y * y)
    column = beta_ground * math.exp(-camera_height_m / scale_height_m) * scale_height_m
    depth: np.ndarray = np.where(
        rising > 1e-4, column / np.maximum(rising, 1e-4), _OPEN_SKY_CEILING
    )
    return np.minimum(depth, _OPEN_SKY_CEILING).astype(np.float32)


def layer_optical_depth(
    distance_m: np.ndarray,
    relative_height_m: np.ndarray,
    *,
    beta_ground: float,
    scale_height_m: float,
    camera_height_m: float,
) -> np.ndarray:
    """Optical depth along a ray through fog whose extinction falls exponentially with height.

    From a camera ``camera_height_m`` above the ground to a point
    ``distance_m`` away and ``relative_height_m`` above the camera. The
    point is not taken below the ground. The formula is the module
    docstring's.
    """
    rise = np.maximum(relative_height_m, -camera_height_m) / scale_height_m
    # (1 - exp(-u)) / u, which is 1 at u = 0.
    safe = np.where(np.abs(rise) < 1e-4, 1.0, rise)
    along = np.where(np.abs(rise) < 1e-4, 1.0, -np.expm1(-np.clip(safe, -60.0, 60.0)) / safe)
    depth: np.ndarray = (
        beta_ground * math.exp(-camera_height_m / scale_height_m) * distance_m * along
    )
    return depth.astype(np.float32)


class GroundFog(WeatherTransform):
    """A shallow layer of fog over the ground, thinning with height and lying in banks.

    Depth-conditioned: the heights the layer depends on come from the depth
    map, so it needs a ``depth_provider``.

    Args:
        depth_provider: one shared provider.
        visibility_range_m: visibility at ground level, inside the layer.
        layer_height_range_m: the layer's scale height: how far up its
            density falls to 37%. A metre or two is mist on a field; ten is
            fog to the rooftops.
        camera_height_range_m: how high the camera is above the ground. A
            camera above the layer looks down through less of it.
        camera_pitch_range_deg: how far the camera looks down from
            horizontal.
        horizontal_fov_range_deg: field of view, which with the pitch turns
            depth into height.
        patchiness_range: how uneven the layer is, 0 to 1. Zero is an even
            sheet.
        patch_scale_range_m: the size of its banks.
        forward_scatter_range: blur of what is seen through the fog, as a
            fraction of the diagonal, where the fog is thick.
        depth_scale: pseudo-metres at relative depth 1, for this transform
            alone; ``None`` uses the provider's own far plane.

    Targets:
        image

    Image types:
        uint8, float32
    """

    class InitSchema(BaseTransformInitSchema):
        """The arguments of ``GroundFog`` as Albumentations validates them: each
        range ordered low to high and within the bounds given here.
        """

        model_config = ConfigDict(arbitrary_types_allowed=True)

        depth_provider: SharedDepthProvider
        visibility_range_m: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(1.0, 50_000.0)),
        ]
        layer_height_range_m: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.1, 500.0)),
        ]
        camera_height_range_m: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.1, 500.0)),
        ]
        camera_pitch_range_deg: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(-30.0, 90.0)),
        ]
        horizontal_fov_range_deg: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(2.0, 150.0)),
        ]
        patchiness_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 1.0)),
        ]
        patch_scale_range_m: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.5, 5000.0)),
        ]
        forward_scatter_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 0.05)),
        ]
        depth_scale: float | None = Field(default=None, gt=0.0)

    def __init__(
        self,
        depth_provider: SharedDepthProvider,
        visibility_range_m: tuple[float, float] = (10.0, 80.0),
        layer_height_range_m: tuple[float, float] = (1.0, 8.0),
        camera_height_range_m: tuple[float, float] = (3.0, 8.0),
        camera_pitch_range_deg: tuple[float, float] = (5.0, 30.0),
        horizontal_fov_range_deg: tuple[float, float] = (48.0, 78.0),
        patchiness_range: tuple[float, float] = (0.2, 0.9),
        patch_scale_range_m: tuple[float, float] = (8.0, 60.0),
        forward_scatter_range: tuple[float, float] = (0.001, 0.004),
        depth_scale: float | None = None,
        p: float = 0.5,
    ):
        """Store the provider and the ranges; each is described in the class docstring."""
        super().__init__(p=p)
        self.depth_provider = depth_provider
        self.visibility_range_m = visibility_range_m
        self.layer_height_range_m = layer_height_range_m
        self.camera_height_range_m = camera_height_range_m
        self.camera_pitch_range_deg = camera_pitch_range_deg
        self.horizontal_fov_range_deg = horizontal_fov_range_deg
        self.patchiness_range = patchiness_range
        self.patch_scale_range_m = patch_scale_range_m
        self.forward_scatter_range = forward_scatter_range
        self.depth_scale = depth_scale

    def get_params_dependent_on_data(
        self, params: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, Any]:
        """Lay the layer over the scene's geometry and integrate it along each pixel's ray.

        Returns the ``transmission`` ``exp(-tau)`` per pixel and the
        ``airlight`` the fog scatters, in linear light.
        """
        scene = scene_of(data)
        image = first_frame(scene)
        height, width = params["shape"][:2]

        visibility = self.py_random.uniform(*self.visibility_range_m)
        scale_height = self.py_random.uniform(*self.layer_height_range_m)
        camera_height = self.py_random.uniform(*self.camera_height_range_m)
        pitch = math.radians(self.py_random.uniform(*self.camera_pitch_range_deg))
        focal = focal_length_px(
            width, height, self.py_random.uniform(*self.horizontal_fov_range_deg)
        )
        patchiness = self.py_random.uniform(*self.patchiness_range)
        patch_scale = self.py_random.uniform(*self.patch_scale_range_m)

        relative_depth = self.depth_provider.get_depth(scene)
        depth_m = self.depth_provider.to_metric(relative_depth, far_m=self.depth_scale)
        linear, _scale, _is_integer = to_linear_unit(image)
        sky = sky_to_depth_edge(sky_weight(linear, relative_depth), relative_depth, linear)
        depth_m = lift_sky(depth_m, sky)

        distance, relative_height, across, forward = scene_points(depth_m, focal, pitch)
        beta = _KOSCHMIEDER / visibility
        tau = layer_optical_depth(
            distance,
            relative_height,
            beta_ground=beta,
            scale_height_m=scale_height,
            camera_height_m=camera_height,
        )
        if patchiness > 0.0:
            rng = self.random_generator
            banks = value_noise(_BANK_TILE, _BANK_TILE, rng=rng, base_cells=_BANKS_ACROSS)
            wisps = value_noise(_BANK_TILE, _BANK_TILE, rng=rng, base_cells=_BANKS_ACROSS)
            per_metre = _BANK_TILE / (_BANKS_ACROSS * patch_scale)

            def read(tile: np.ndarray, share: float, zoom: float) -> np.ndarray:
                """A tile of noise at ``share`` of the way along each ray, about a mean of one."""
                found = cv2.remap(
                    tile,
                    (share * zoom * per_metre * across).astype(np.float32),
                    (share * zoom * per_metre * forward).astype(np.float32),
                    cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_WRAP,
                )
                return found / max(float(tile.mean()), 1e-3)

            # Read between the camera and the point: the fog a pixel looks
            # through lies along the ray, not at its end. About a mean of
            # one, so that banks move the fog about and add none.
            uneven = (1.0 - _WISP_SHARE) * read(banks, 0.5, 1.0) + _WISP_SHARE * read(
                wisps, 0.7, _WISP_RATIO
            )
            tau *= (1.0 - patchiness) + patchiness * uneven

        # Sky: from the ray's direction, not from a depth it does not have.
        open_sky = open_sky_optical_depth(
            height,
            width,
            focal,
            pitch,
            beta_ground=beta,
            scale_height_m=scale_height,
            camera_height_m=camera_height,
        )
        tau = tau * (1.0 - sky) + open_sky * sky

        record_draw(
            visibility_m=visibility,
            beta_ground=beta,
            layer_height_m=scale_height,
            camera_height_m=camera_height,
            camera_pitch_deg=math.degrees(pitch),
            focal_px=focal,
            patchiness=patchiness,
            patch_scale_m=patch_scale,
            optical_depth_p50=lambda: float(np.median(tau)),
        )
        diagonal = float(np.hypot(width, height))
        return {
            "transmission": np.exp(-tau).astype(np.float32),
            "airlight": veil_radiance_linear(linear, sky).astype(np.float32),
            "scatter_sigma": self.py_random.uniform(*self.forward_scatter_range) * diagonal,
        }

    def apply(
        self,
        img: np.ndarray,
        transmission: np.ndarray,
        airlight: np.ndarray,
        scatter_sigma: float,
        **params: Any,
    ) -> np.ndarray:
        """Soften what the fog veils, then veil it: ``linear t + airlight (1 - t)``."""
        linear, scale, is_integer = to_linear_unit(img)
        clear = transmission[..., None]
        if scatter_sigma > 0.3:
            blurred = wide_blur(linear, scatter_sigma)
            linear = linear * clear + blurred * (1.0 - clear)
        out = linear * clear + airlight[None, None, :] * (1.0 - clear)
        return from_linear_unit(out, scale, is_integer, img.dtype)
