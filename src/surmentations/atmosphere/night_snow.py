"""Snow lit by the camera's own infrared illuminator.

The night counterpart of :class:`~surmentations.atmosphere.snow.RandomSnow`,
for the same reason ``NightRain`` is the counterpart of ``RandomRain``: at
night the thing lighting a flake is no longer the sky.

``RandomSnow`` composites each flake toward the scene's illuminant. Run before
``NightIrMode`` — which is where the pipelines put it — the flakes then go
through the reflectance-to-infrared conversion with everything else, and that
conversion lights each pixel by the depth of the *scene* at that pixel. A
flake half a metre from the lens, drawn over a sky fifty metres away, is lit
as if it were fifty metres away: it goes dark. The result is night snow that
is nearly invisible exactly where real footage shows it best.

In real infrared footage snow is the opposite of subtle. A flake is a bright
diffuse scatterer — snow reflects most of the near infrared that reaches it —
sitting a metre from a lamp whose light falls off as ``1/z^2``, in front of a
dark scene. It is the brightest thing in the frame. So:

* **A flake's brightness is set by its own distance from the lamp**, not by
  what is behind it: ``(1/z)^2``, rolled off at the lamp's reach.
* **Near flakes are large, soft and streaked.** A night camera runs a long
  shutter (1/50-1/25 s), so a flake a metre away travels several times its
  own width during the exposure, and it is far inside the lens's near focus
  limit. The sprite model is ``RandomSnow``'s — an image of the aperture swept
  along the fall direction, flux-conserving — with only the lighting changed.
* **The unresolved flakes glow back.** Snow scatters roughly seven times as
  hard as rain at the same precipitation rate, so the backscatter that makes
  an IR camera struggle in rain makes it nearly blind in a snowstorm.

Composition: after ``NightIrMode``, like ``NightRain``. The veil and the
settled snow are properties of the scene and still belong to ``RandomSnow``
*before* the conversion — build it with ``max_flakes=0`` so the two do not
both draw the falling flakes. The ready-made pipelines do exactly that.
"""

from __future__ import annotations

from typing import Annotated, Any

import cv2
import numpy as np
from albumentations.core.pydantic import check_range_bounds, nondecreasing
from albumentations.core.transforms_interface import BaseTransformInitSchema
from pydantic import AfterValidator, Field

from surmentations.atmosphere.night_rain import (
    _REFERENCE_DISTANCE_M,
    _beam_profile,
    _frustum_volume,
)
from surmentations.atmosphere.snow import render_flake_field, snow_fall_speed
from surmentations.base import CameraTransform, scene_fov
from surmentations.diagnostics import record_draw
from surmentations.utils.photometry import from_linear_unit, to_linear_unit
from surmentations.utils.precipitation import (
    focal_length_px,
    gunn_marshall_slope,
    resolvable_fraction,
    sample_resolvable_particles,
    snow_number_density,
    snow_physical_diameter,
    width_cut_for_budget,
)


class NightSnow(CameraTransform):
    """Falling snow lit by the camera's infrared illuminator.

    Needs no depth provider: the flakes are in the air in front of the scene
    and are lit by the lamp, so nothing about them depends on the geometry
    behind. Runs on CPU.

    Args:
        snowfall_rate_range_mm_h: precipitation rate in mm/h of melted water
            equivalent. Sets the size distribution and how many flakes are in
            the beam, exactly as in ``RandomSnow``.
        snow_density_range_g_cm3: bulk density of the falling snow, which
            converts the melted-equivalent diameters the distribution is
            stated in to the size the camera sees.
        flake_diameter_range_mm: truncation window for the size distribution,
            in melted-equivalent millimetres.
        distance_range_m: the column of air in which flakes are drawn. The
            near bound matters most: ``1/z^2`` makes the nearest flakes the
            whole effect.
        illuminator_range_m: how far the lamp throws. Flakes beyond it are
            unlit, which is what limits the depth of the visible snowfall.
        illuminator_gain_range: radiance a flake one metre away returns, in
            units of full scale. Snow is a bright diffuse scatterer, so this
            sits higher for a given lamp than a raindrop's does.
        beam_falloff_range: how sharply the beam dims off-axis.
        backscatter_range: glow from the flakes the beam lights but the lens
            cannot resolve. Heavy snow in a strong beam approaches the upper
            end and whites the frame out, as it does in reality.
        exposure_time_range: shutter time. The default is the 1/50-1/25 s of
            a night camera, long enough to draw near flakes out into streaks.
        fall_speed_scale_range: multiplier on the still-air fall speed, for
            wind-driven snow.
        fall_angle_range: prevailing slant from vertical, in degrees.
        flutter_range: how far each flake's path deviates from that slant.
        horizontal_fov_range_deg: field of view, used to project sizes and
            distances into pixels.
        aperture_diameter_range_mm: lens aperture, which sets the bokeh of a
            flake inside the near focus limit.
        focus_distance_range_m: where the lens is focused.
        glare_range: fraction of a flake's light that the dome and the lens
            scatter into a soft glow around it. A flake a hand's width from
            the lamp is tens of times brighter than full scale, and that much
            light does not stay inside its own outline: without this every
            near flake is a hard-edged white cut-out.
        max_flakes: render budget. When it binds, the flakes drawn are the
            widest ones.
        min_flake_width_px: flakes projecting narrower than this are left to
            the backscatter glow.

    Targets:
        image

    Image types:
        uint8, float32
    """

    class InitSchema(BaseTransformInitSchema):
        """The arguments of ``NightSnow`` as Albumentations validates them: each
        range ordered low to high and within the bounds given here.
        """
        snowfall_rate_range_mm_h: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.02, 100.0)),
        ]
        snow_density_range_g_cm3: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.005, 0.6)),
        ]
        flake_diameter_range_mm: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.1, 30.0)),
        ]
        distance_range_m: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.05, 60.0)),
        ]
        illuminator_range_m: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.5, 80.0)),
        ]
        illuminator_gain_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.05, 20.0)),
        ]
        beam_falloff_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 6.0)),
        ]
        backscatter_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 1.0)),
        ]
        exposure_time_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(1e-4, 1.0)),
        ]
        fall_speed_scale_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.05, 20.0)),
        ]
        fall_angle_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(-60.0, 60.0)),
        ]
        flutter_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 90.0)),
        ]
        horizontal_fov_range_deg: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(10.0, 140.0)),
        ]
        aperture_diameter_range_mm: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.1, 30.0)),
        ]
        focus_distance_range_m: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.5, 200.0)),
        ]
        glare_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 1.0)),
        ]
        max_flakes: int = Field(ge=1, le=200_000)
        min_flake_width_px: float = Field(gt=0.0, le=20.0)

    def __init__(
        self,
        snowfall_rate_range_mm_h: tuple[float, float] = (0.8, 8.0),
        snow_density_range_g_cm3: tuple[float, float] = (0.035, 0.11),
        flake_diameter_range_mm: tuple[float, float] = (1.0, 15.0),
        distance_range_m: tuple[float, float] = (0.15, 8.0),
        illuminator_range_m: tuple[float, float] = (4.0, 14.0),
        illuminator_gain_range: tuple[float, float] = (0.6, 2.5),
        beam_falloff_range: tuple[float, float] = (0.6, 2.2),
        backscatter_range: tuple[float, float] = (0.03, 0.2),
        exposure_time_range: tuple[float, float] = (0.020, 0.040),
        fall_speed_scale_range: tuple[float, float] = (0.7, 1.8),
        fall_angle_range: tuple[float, float] = (-12.0, 12.0),
        flutter_range: tuple[float, float] = (8.0, 30.0),
        horizontal_fov_range_deg: tuple[float, float] = (48.0, 78.0),
        aperture_diameter_range_mm: tuple[float, float] = (0.9, 2.6),
        focus_distance_range_m: tuple[float, float] = (8.0, 40.0),
        glare_range: tuple[float, float] = (0.05, 0.25),
        max_flakes: int = 9000,
        min_flake_width_px: float = 1.0,
        p: float = 0.5,
    ):
        """Store the ranges; each argument is described in the class docstring."""
        super().__init__(p=p)
        self.snowfall_rate_range_mm_h = snowfall_rate_range_mm_h
        self.snow_density_range_g_cm3 = snow_density_range_g_cm3
        self.flake_diameter_range_mm = flake_diameter_range_mm
        self.distance_range_m = distance_range_m
        self.illuminator_range_m = illuminator_range_m
        self.illuminator_gain_range = illuminator_gain_range
        self.beam_falloff_range = beam_falloff_range
        self.backscatter_range = backscatter_range
        self.exposure_time_range = exposure_time_range
        self.fall_speed_scale_range = fall_speed_scale_range
        self.fall_angle_range = fall_angle_range
        self.flutter_range = flutter_range
        self.horizontal_fov_range_deg = horizontal_fov_range_deg
        self.aperture_diameter_range_mm = aperture_diameter_range_mm
        self.focus_distance_range_m = focus_distance_range_m
        self.glare_range = glare_range
        self.max_flakes = max_flakes
        self.min_flake_width_px = min_flake_width_px

    def get_params_dependent_on_data(
        self, params: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, Any]:
        """Draw the flakes in the illuminator's beam and how brightly each is lit.

        The same derived chain as ``RandomSnow`` decides how many flakes are
        in the column and which resolve. Returns the ``flakes`` with each
        one's ``brightness`` by its distance from the lamp, the veiling
        ``glare``, the ``beam`` across the frame and the ``backscatter``.
        """
        height, width = params["shape"][:2]
        rng = self.random_generator

        snowfall_rate = self.py_random.uniform(*self.snowfall_rate_range_mm_h)
        snow_density = self.py_random.uniform(*self.snow_density_range_g_cm3)
        horizontal_fov = scene_fov(
            data, self.py_random.uniform(*self.horizontal_fov_range_deg)
        )
        focus_distance = self.py_random.uniform(*self.focus_distance_range_m)
        exposure = self.py_random.uniform(*self.exposure_time_range)
        aperture_m = self.py_random.uniform(*self.aperture_diameter_range_mm) * 1e-3
        fall_speed_scale = self.py_random.uniform(*self.fall_speed_scale_range)
        fall_angle = self.py_random.uniform(*self.fall_angle_range)
        flutter = self.py_random.uniform(*self.flutter_range)
        reach = self.py_random.uniform(*self.illuminator_range_m)
        gain = self.py_random.uniform(*self.illuminator_gain_range)
        falloff = self.py_random.uniform(*self.beam_falloff_range)
        backscatter = self.py_random.uniform(*self.backscatter_range)

        focal = focal_length_px(width, height, horizontal_fov)
        near, far = self.distance_range_m
        d_min, d_max = self.flake_diameter_range_mm

        # The same derived chain as RandomSnow: the weather and the camera
        # decide how many flakes are in the column and which of them resolve.
        slope = gunn_marshall_slope(snowfall_rate)
        size_scale = float(snow_physical_diameter(1.0, snow_density))
        density = snow_number_density(snowfall_rate, d_min)
        volume = _frustum_volume(width, height, focal, near, far)
        visible = resolvable_fraction(
            slope, d_min, d_max, near, far, focal, self.min_flake_width_px, size_scale=size_scale
        )
        population = density * volume * visible
        width_cut = width_cut_for_budget(
            population, self.max_flakes, slope, d_min, d_max, near, far, focal,
            self.min_flake_width_px, size_scale=size_scale,
        )
        count = int(min(self.max_flakes, round(population)))

        melted_mm, distance_m = sample_resolvable_particles(
            rng, count, slope, d_min, d_max, near, far, focal, width_cut, size_scale=size_scale
        )
        count = len(melted_mm)
        diameter_mm = snow_physical_diameter(melted_mm, snow_density)

        radius_px = 0.5 * focal * (diameter_mm * 1e-3) / distance_m
        blur_radius_px = (
            0.5 * focal * aperture_m * np.abs(1.0 / distance_m - 1.0 / focus_distance)
        )
        speed = snow_fall_speed(diameter_mm) * fall_speed_scale
        smear_px = focal * speed * exposure / distance_m

        # The whole point of this transform: a flake is lit by how far *it* is
        # from the lamp. Inverse-square from the lens, rolled off smoothly at
        # the lamp's reach rather than cut, so the far end of the lit column
        # does not end on a visible shell.
        illumination = (_REFERENCE_DISTANCE_M / np.maximum(distance_m, 1e-3)) ** 2
        illumination = illumination * np.exp(-((distance_m / max(reach, 1e-3)) ** 2))

        angle_rad = np.deg2rad(fall_angle + rng.uniform(-flutter, flutter, count))

        record_draw(
            snowfall_rate_mm_h=snowfall_rate,
            snow_density_g_cm3=snow_density,
            exposure_s=exposure,
            illuminator_reach_m=reach,
            illuminator_gain=gain,
            beam_falloff=falloff,
            backscatter=backscatter,
            focal_px=focal,
            volume_m3=volume,
            resolvable_fraction=visible,
            population=population,
            width_cut_px=width_cut,
            count_requested=count,
            illumination_p50=float(np.median(illumination)) if count else 0.0,
        )
        return {
            "flakes": {
                "x": rng.uniform(0.0, width, count).astype(np.float32),
                "y": rng.uniform(0.0, height, count).astype(np.float32),
                "distance": distance_m.astype(np.float32),
                "radius": radius_px.astype(np.float32),
                "blur_radius": blur_radius_px.astype(np.float32),
                "smear": smear_px.astype(np.float32),
                "dir_x": np.sin(angle_rad).astype(np.float32),
                "dir_y": np.cos(angle_rad).astype(np.float32),
                "brightness": (gain * illumination).astype(np.float32),
            },
            "glare": self.py_random.uniform(*self.glare_range),
            "beam": _beam_profile(height, width, falloff),
            "backscatter": backscatter,
        }

    def apply(
        self,
        img: np.ndarray,
        flakes: dict[str, np.ndarray],
        glare: float,
        beam: np.ndarray,
        backscatter: float,
        **params: Any,
    ) -> np.ndarray:
        """Render the lit flakes through a round stop and add them, with glare and backscatter.

        A surveillance lens has a fixed circular stop, so a flake swept across
        it is a capsule. The flakes add to the dark scene; the glare is a share
        of what each returned before the sensor clipped it.
        """
        height, width = img.shape[:2]
        frame_centre = (width / 2.0, height / 2.0)

        lit = np.zeros((height, width), dtype=np.float32)
        render_flake_field(
            lit,
            flakes,
            # A round aperture. RandomSnow draws polygonal bokeh because a
            # photographic lens has an iris; a surveillance board lens has
            # a fixed circular stop, and a flake swept across it is a
            # capsule, not a parallelogram.
            blades=0,
            blade_rotation=0.0,
            rim_gain=1.0,
            cats_eye=0.0,
            frame_centre=frame_centre,
            gain=flakes["brightness"],
        )

        if glare > 0.0 and len(flakes["x"]):
            # Veiling glare, on the unclipped light: what spills is a share of
            # what the flake actually returned, not of what the sensor could
            # record of it.
            sigma = max(1.0, 0.006 * float(np.hypot(width, height)))
            lit = lit * (1.0 - glare) + cv2.GaussianBlur(lit, (0, 0), sigmaX=sigma) * glare

        linear, scale, is_integer = to_linear_unit(img)
        # Emissive against a dark scene, like the streaks in NightRain: a
        # flake scattering the lamp's light back adds to whatever is behind
        # it, and at night there is very little behind it to hide.
        out = linear + (lit * beam)[..., None]
        if backscatter > 0.0:
            # The flakes the beam lights but the lens cannot resolve. Flat in
            # depth, because the 1/z^2 irradiance and the z^2 growth of the
            # lit volume cancel.
            out = out + (backscatter * beam)[..., None]
        return from_linear_unit(np.clip(out, 0.0, None), scale, is_integer, img.dtype)
