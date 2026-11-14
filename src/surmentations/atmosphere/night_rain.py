"""Rain lit by the camera's own infrared illuminator.

This is the case everyone pictures when they picture rain on CCTV, and it is
the only one in which rain reliably reads as *streaks* rather than as haze and
a wet road. ``docs/rain_visibility.md`` measures why: in daylight a streak
takes the sky's radiance and lifts the pixels it crosses by about 8 sRGB
levels, against 17 levels of the asphalt's own texture, so a physically
correct streak is half a sigma into the background and cannot be seen.

At night nothing about the drop changes. What changes is both terms of the
contrast:

* **The drop is lit by the camera**, not by the sky. The illuminator's
  irradiance falls as ``1/z^2``, so a drop at 0.5 m receives a hundred times
  what one at 5 m does. Near drops become far brighter than anything in the
  scene instead of matching it.
* **The background is dark.** A night frame's road sits near the bottom of
  the range, so the same absolute lift is a much larger relative one.

Together those turn the 8-level lift into a saturated streak, which is why
the effect is dramatic in IR footage and absent in daylight. Nothing here
raises the amount of light the physics gives a drop; it puts the light
somewhere it can be seen.

Two further consequences worth having, both visible in real footage:

* **Long streaks.** A night camera runs a long shutter — 1/25 to 1/50 s is
  normal where daylight would use 1/500 — so streaks are tens of pixels long
  rather than ten. The opacity drops in proportion (``alpha = w^2/L``), but
  against a dark background there is contrast to spare.
* **Backscatter.** The illuminator lights the whole near column, not only the
  drops that resolve, and that unresolved rain scatters light straight back
  into the lens. It is the same veil daylight rain has, except lit from the
  camera, so it is brightest at the centre of the beam and falls off with it —
  the "wall of light" that makes an IR camera nearly useless in heavy rain.

Composition: this belongs **after** ``NightIrMode``, not before it. The
streaks are already in infrared terms — the illuminator lit them and the
sensor sees them directly — so running them through the scene's
reflectance-to-IR conversion would convert them twice.
"""

from __future__ import annotations

from typing import Annotated, Any

import cv2
import numpy as np
from albumentations.core.pydantic import check_range_bounds, nondecreasing
from albumentations.core.transforms_interface import BaseTransformInitSchema
from pydantic import AfterValidator, Field

from surmentations import mode
from surmentations.atmosphere.rain import render_streak_layer
from surmentations.base import CameraTransform, scene_fov
from surmentations.diagnostics import record_draw
from surmentations.utils.grids import pixel_grid
from surmentations.utils.photometry import from_linear_unit, to_linear_unit
from surmentations.utils.precipitation import (
    far_field_texture,
    focal_length_px,
    gamma_dsd_slope,
    marshall_palmer_slope,
    rain_number_density,
    raindrop_terminal_velocity,
    resolvable_fraction,
    sample_resolvable_particles,
    width_cut_for_budget,
)

#: Distance at which the illuminator is taken to deliver unit irradiance, so
#: the ``1/z^2`` falloff is expressed relative to something. Drops nearer than
#: this saturate, which is exactly what they do in real footage.
_REFERENCE_DISTANCE_M = 1.0


class NightRain(CameraTransform):
    """Rain streaks lit by the camera's infrared illuminator.

    Needs no depth provider: the drops are in the air *in front of* the scene,
    and the backscatter is a property of the beam rather than of the geometry
    behind it. So this runs on CPU with no GPU requirement, unlike
    :class:`~surmentations.atmosphere.rain.RandomRain`.

    Args:
        rain_rate_range_mm_h: precipitation rate, which fixes the drop-size
            distribution and hence how many drops are in the beam.
        drop_diameter_range_mm: truncation window for the distribution.
        dsd_shape_mu: ``0`` for Marshall & Palmer's exponential, 2-4 for
            Ulbrich's gamma. See ``RandomRain``.
        distance_range_m: the column of air the illuminator lights. Its near
            bound matters more here than in daylight, because ``1/z^2`` makes
            the nearest drops the whole effect.
        illuminator_range_m: how far the lamp throws. Drops beyond it are
            unlit and effectively invisible, which is what limits the depth of
            the visible rain.
        beam_falloff_range: how sharply the beam dims off-axis. An IR ring
            around the lens is fairly even; a single lamp is not.
        exposure_time_range: shutter time. Defaults to the 1/50-1/25 s a night
            camera actually uses, which is what makes the streaks long.
        illuminator_gain_range: radiance a drop at 1 m returns to the sensor,
            in units of full scale. This is what makes night rain visible and
            daylight rain not: a drop lit by the lamp is *brighter than
            anything in the scene* rather than matching the sky, so with the
            ``1/z^2`` term a drop at 0.4 m saturates while one at 4 m is
            invisible. Values near 1 give a restrained drizzle; the upper end
            gives the hard white streaks of heavy rain in a strong beam.
        backscatter_grain_range: how much of its grain that glow keeps, drawn
            per call. At 0 it is an even wall of light, as it was. At 1 it
            varies as a count of drops does: what comes back from the air
            just beyond the last streak drawn is half of it, and there a
            pixel looks through a handful of them, so the glow is a
            fine streaked grain that thins into evenness with distance. See
            :func:`~surmentations.utils.precipitation.far_field_texture`.
        backscatter_range: strength of the glow from the unresolved drops the
            beam lights. 0 disables it; heavy rain in a strong beam approaches
            the upper end and washes the frame out, as it does in reality.
        horizontal_fov_range_deg: field of view, used to project drop sizes
            and distances into pixels.
        aperture_diameter_range_mm: aperture, which sets the circle of
            confusion for drops far from the focal plane.
        focus_distance_range_m: where the lens is focused.
        wind_angle_range: fall direction from vertical, in degrees.
        max_streaks: render budget.
        min_streak_width_px: resolvability cut, in projected width.

    Targets:
        image

    Image types:
        uint8, float32
    """

    class InitSchema(BaseTransformInitSchema):
        """The arguments of ``NightRain`` as Albumentations validates them: each
        range ordered low to high and within the bounds given here.
        """
        rain_rate_range_mm_h: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.1, 300.0)),
        ]
        drop_diameter_range_mm: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.1, 8.0)),
        ]
        dsd_shape_mu: float = Field(ge=0.0, le=10.0)
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
        beam_falloff_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 6.0)),
        ]
        exposure_time_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(1e-4, 1.0)),
        ]
        illuminator_gain_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 20.0)),
        ]
        backscatter_grain_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 2.0)),
        ]
        backscatter_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 1.0)),
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
        wind_angle_range: Annotated[tuple[float, float], AfterValidator(nondecreasing)]
        max_streaks: int = Field(ge=1, le=200_000)
        min_streak_width_px: float = Field(gt=0.0, le=8.0)

    def __init__(
        self,
        rain_rate_range_mm_h: tuple[float, float] = (5.0, 90.0),
        drop_diameter_range_mm: tuple[float, float] = (0.8, 6.0),
        dsd_shape_mu: float = 0.0,
        distance_range_m: tuple[float, float] = (0.25, 12.0),
        illuminator_range_m: tuple[float, float] = (4.0, 14.0),
        beam_falloff_range: tuple[float, float] = (0.6, 2.2),
        exposure_time_range: tuple[float, float] = (0.020, 0.040),
        illuminator_gain_range: tuple[float, float] = (1.5, 6.0),
        backscatter_range: tuple[float, float] = (0.02, 0.14),
        backscatter_grain_range: tuple[float, float] = (0.7, 1.2),
        horizontal_fov_range_deg: tuple[float, float] = (48.0, 78.0),
        aperture_diameter_range_mm: tuple[float, float] = (0.9, 2.6),
        focus_distance_range_m: tuple[float, float] = (8.0, 40.0),
        wind_angle_range: tuple[float, float] = (-14.0, 14.0),
        max_streaks: int = 16000,
        min_streak_width_px: float = 0.35,
        p: float = 0.5,
    ):
        """Store the ranges; each argument is described in the class docstring."""
        super().__init__(p=p)
        self.rain_rate_range_mm_h = rain_rate_range_mm_h
        self.drop_diameter_range_mm = drop_diameter_range_mm
        self.dsd_shape_mu = dsd_shape_mu
        self.distance_range_m = distance_range_m
        self.illuminator_range_m = illuminator_range_m
        self.beam_falloff_range = beam_falloff_range
        self.exposure_time_range = exposure_time_range
        self.illuminator_gain_range = illuminator_gain_range
        self.backscatter_range = backscatter_range
        self.backscatter_grain_range = backscatter_grain_range
        self.horizontal_fov_range_deg = horizontal_fov_range_deg
        self.aperture_diameter_range_mm = aperture_diameter_range_mm
        self.focus_distance_range_m = focus_distance_range_m
        self.wind_angle_range = wind_angle_range
        self.max_streaks = max_streaks
        self.min_streak_width_px = min_streak_width_px

    def get_params_dependent_on_data(
        self, params: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, Any]:
        """Draw the drops in the illuminator's beam and how brightly each is lit.

        The drop population follows from the rain rate and the lit column's
        volume, as in ``RandomRain``. Returns the ``streaks`` with each one's
        ``gain``, set by the inverse square of its own distance from the lamp
        and rolled off smoothly at the lamp's reach, the ``beam`` across the
        frame, and the ``backscatter`` of the drops too small to resolve.
        """
        height, width = params["shape"][:2]
        rng = self.random_generator

        rain_rate = self.py_random.uniform(*self.rain_rate_range_mm_h)
        horizontal_fov = scene_fov(
            data, self.py_random.uniform(*self.horizontal_fov_range_deg)
        )
        focus_distance = self.py_random.uniform(*self.focus_distance_range_m)
        exposure = self.py_random.uniform(*self.exposure_time_range)
        aperture_m = self.py_random.uniform(*self.aperture_diameter_range_mm) * 1e-3
        wind_angle = self.py_random.uniform(*self.wind_angle_range)
        reach = self.py_random.uniform(*self.illuminator_range_m)
        falloff = self.py_random.uniform(*self.beam_falloff_range)
        gain = self.py_random.uniform(*self.illuminator_gain_range)
        backscatter = self.py_random.uniform(*self.backscatter_range)

        focal = focal_length_px(width, height, horizontal_fov)
        slope = gamma_dsd_slope(rain_rate, self.dsd_shape_mu)
        near, far = self.distance_range_m
        d_min, d_max = self.drop_diameter_range_mm

        visible = resolvable_fraction(
            slope, d_min, d_max, near, far, focal, self.min_streak_width_px,
            shape_mu=self.dsd_shape_mu,
        )
        density = rain_number_density(rain_rate, d_min, self.dsd_shape_mu)
        # The lit column is a frustum, but only its near part matters once the
        # 1/z^2 falloff is applied, so the volume is computed over the whole
        # range and the illumination term does the weighting.
        volume = _frustum_volume(width, height, focal, near, far)
        population = density * volume * visible
        width_cut = width_cut_for_budget(
            population, self.max_streaks, slope, d_min, d_max, near, far, focal,
            self.min_streak_width_px, shape_mu=self.dsd_shape_mu,
        )
        count = int(min(population, self.max_streaks))

        diameter_mm, distance_m = sample_resolvable_particles(
            rng, count, slope, d_min, d_max, near, far, focal, width_cut,
            shape_mu=self.dsd_shape_mu,
        )
        count = len(diameter_mm)
        speed = raindrop_terminal_velocity(diameter_mm)

        drop_width = focal * (diameter_mm * 1e-3) / distance_m
        length = focal * (speed * exposure) / distance_m
        blur_radius = (
            0.5 * focal * aperture_m * np.abs(1.0 / distance_m - 1.0 / focus_distance)
        )
        thickness = drop_width + 2.0 * blur_radius

        # The whole point of this transform. Irradiance on the drop falls as
        # 1/z^2 from the lamp, and the lamp only throws so far, so the
        # brightness of a streak is set by where the drop is rather than by
        # what is behind it. Squaring the reach term rather than clipping it
        # keeps the far end smooth instead of ending on a visible shell.
        illumination = (_REFERENCE_DISTANCE_M / np.maximum(distance_m, 1e-3)) ** 2
        illumination = illumination * np.exp(-((distance_m / max(reach, 1e-3)) ** 2))

        geometric_alpha = np.clip(drop_width / np.maximum(length, 1e-3), 0.0, 1.0) * (
            drop_width / np.maximum(thickness, 1e-3)
        )
        alpha = np.clip(geometric_alpha * illumination, 0.0, 1.0)

        angle_rad = np.deg2rad(wind_angle)
        record_draw(
            rain_rate_mm_h=rain_rate,
            exposure_s=exposure,
            illuminator_reach_m=reach,
            beam_falloff=falloff,
            illuminator_gain=gain,
            backscatter=backscatter,
            focal_px=focal,
            volume_m3=volume,
            resolvable_fraction=visible,
            population=population,
            count_requested=count,
            illumination_p50=float(np.median(illumination)) if count else 0.0,
        )
        drawn = {
            "streaks": {
                "x": rng.uniform(0.0, width, count).astype(np.float32),
                "y": rng.uniform(0.0, height, count).astype(np.float32),
                "distance": distance_m.astype(np.float32),
                "length": length.astype(np.float32),
                "thickness": np.maximum(1, np.round(thickness)).astype(np.int32),
                "angle": np.full(count, wind_angle, dtype=np.float32),
                "bend": rng.uniform(-0.05, 0.05, count).astype(np.float32),
                "alpha": alpha.astype(np.float32),
                "phase": rng.uniform(0.0, 2 * np.pi, count).astype(np.float32),
                "blur_sigma": (0.45 * blur_radius).astype(np.float32),
            },
            "beam": _beam_profile(height, width, falloff),
            "gain": gain,
            "backscatter": backscatter,
            "angle": float(np.rad2deg(angle_rad)),
        }
        # The grain of the glow, last of all the draws.
        grain = self.py_random.uniform(*self.backscatter_grain_range)
        grain_seed = self.py_random.randint(0, 2**31 - 1)
        if grain > 0.0 and backscatter > 0.0:
            falls = float(raindrop_terminal_velocity(3.0 / marshall_palmer_slope(rain_rate)))
            drawn["backscatter_grain"] = far_field_texture(
                np.zeros((height, width), dtype=np.float32),
                np.random.default_rng(grain_seed),
                rain_number_density(rain_rate, 0.0),
                focal,
                far,
                focal * falls * exposure,
                wind_angle,
                grain,
                lit_from_lens=True,
            )
        return drawn

    def apply(
        self,
        img: np.ndarray,
        streaks: dict[str, np.ndarray],
        beam: np.ndarray,
        gain: float,
        backscatter: float,
        backscatter_grain: np.ndarray | None = None,
        **params: Any,
    ) -> np.ndarray:
        """Add the lit streaks and the beam's backscatter to the frame.

        The streaks are emissive against a dark scene, so they add rather than
        cover; the unresolved drops add a veil that is flat in depth.
        """
        linear, scale, is_integer = to_linear_unit(img)

        occupancy = render_streak_layer(img.shape[0], img.shape[1], streaks)
        occupancy = cv2.GaussianBlur(occupancy, (0, 0), sigmaX=0.35)
        lit = -np.expm1(-occupancy) * beam * gain

        # The streaks are emissive against a dark scene rather than occluding
        # something brighter, so they add. A drop scattering the lamp's light
        # back does not hide what is behind it; at these opacities there is
        # nothing behind it to hide.
        out = linear + lit[..., None]

        if backscatter > 0.0:
            # The unresolved drops the beam lights, scattering straight back.
            # Flat in depth because the 1/z^2 gain and the z^2 growth of the
            # illuminated volume cancel -- the same cancellation that makes a
            # fog bank in headlights an even wall rather than a gradient.
            # Not even across the frame, though: it is a count of drops,
            # and has a count's grain.
            glow = backscatter * beam
            if backscatter_grain is not None:
                glow = glow * backscatter_grain
            out = out + glow[..., None]

        return from_linear_unit(np.clip(out, 0.0, None), scale, is_integer, img.dtype)


def _beam_profile(height: int, width: int, falloff: float) -> np.ndarray:
    """Relative irradiance across the frame from a lamp on the optical axis.

    A Gaussian of the distance from the frame's centre, ``(H, W)`` float32.
    The coordinates are a row and a column that numpy broadcasts, the same
    floats as two frame-sized grids cast from int64 and a third of the time.

    In fast mode (:mod:`surmentations.mode`) it is the product of a
    profile across the frame and one down it, which a Gaussian of a sum of
    squares is: two exponentials of a line where the reference takes a
    square root, a square and an exponential of a frame. It differs in the
    last bit of a float32, 2e-7 of the peak.
    """
    yy, xx = pixel_grid(height, width)
    across = (xx - width / 2.0) / max(width / 2.0, 1.0)
    down = (yy - height / 2.0) / max(height / 2.0, 1.0)
    if mode.fast():
        profile: np.ndarray = np.exp(-falloff * (across * across)) * np.exp(
            -falloff * (down * down)
        )
        return profile
    radius = np.hypot(across, down)
    return np.exp(-falloff * radius**2).astype(np.float32)


def _frustum_volume(
    width: int, height: int, focal_px: float, near_m: float, far_m: float
) -> float:
    """Volume of the viewing frustum between two distances, in cubic metres."""
    solid = (width / focal_px) * (height / focal_px)
    return float(solid * (far_m**3 - near_m**3) / 3.0)
