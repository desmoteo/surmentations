"""Depth-conditioned rain: streaks, a distant rain veil, and optional ground wetness.

Photometric model (Garg & Nayar, "Vision and Rain" IJCV 2007;
"Photorealistic Rendering of Rain Streaks" SIGGRAPH 2006)
---------------------------------------------------------------------------
A raindrop crossing a pixel during exposure only occupies it for a fraction
of the exposure time, so the pixel is a *time average*::

    I = (1 - alpha) * I_background + alpha * L_streak
    alpha = tau / T_exposure = 2 * r0 / (v * T_exposure)

with terminal velocity ``v(r0) = 9.40 * (1 - exp(-3.45e6 * r0^1.31))`` m/s.
For a 1 mm drop, ``v ~ 6.9 m/s``, and at 1/60 s that gives **alpha ~ 0.017**
— a streak is only a *few percent* opaque. Painting streaks at 30-60% alpha,
as a naive implementation does, is the main reason synthetic rain reads as
white scratches drawn on top of a photo instead of rain.

Getting alpha right also buys contrast-dependent visibility for free: since
``dI = alpha * (L_streak - L_background)``, a streak is obvious against dark
foliage and nearly invisible against bright sky, exactly as in real photos —
no special-casing needed.

``L_streak`` is *not* white. A drop refracts light from a ~165 degree field
of view and reflects about 6% of its whole environment, so it is essentially
an image of the sky. We take it from the scene's own bright/sky radiance,
which is why streaks tint with the weather rather than staying neutral.

Real streaks are also not uniform bars of light: the drop oscillates as it
falls, so the streak is beaded, with several smeared highlights along its
length. We modulate each streak's intensity with a quasi-periodic profile
(two harmonics and a random phase) to approximate that without shipping
Garg & Nayar's 400 MB measured streak database. That profile is sampled
*per pixel* through a lookup table rather than per drawn segment: evaluating
a three-cycle sine at six segment midpoints aliases it into a strict
two-level alternation, which turns every streak in the frame into the same
three-bright/three-dark dashed line — precisely the hatch pattern the
per-streak variation exists to avoid.

Finally, near drops are *defocused*: closer means wider **and dimmer**,
since the same light spreads over a larger circle of confusion — the
opposite of the intuitive "closer is brighter".

Everything composites in linear light, and the frame's mean level is
renormalised afterwards to stand in for the camera's auto-exposure.

The camera's exposure decides what rain looks like
-------------------------------------------------
The same shower is three different pictures depending on how the camera is
exposing, and none of the three is the weather's doing:

* **Shutter time** sets how far a drop travels while the shutter is open. At
  1/2000 s a 2 mm drop moves 3 mm, barely more than its own width, and is
  recorded as a near-opaque fleck; at 1/30 s it is drawn 22 cm long and 1%
  opaque. The light it deposits is the same either way.
* **Aperture** sets how defocused the near drops are — the ones wide enough
  to resolve at all. A small aperture keeps them thin and therefore opaque;
  a wide one spreads each over its circle of confusion.
* **The two are not independent.** For a given light level the camera needs
  a given exposure, ``T = K N^2 / E``: a lens stopped down four stops needs
  a shutter sixteen times longer. So a stopped-down iris comes with long,
  thin, faint streaks and an open one with short soft flecks, and drawing
  the two from separate ranges produces the pairs no camera would choose —
  a wide-open lens at 1/30 s in daylight, six or seven stops over-exposed.

So the shutter time is not drawn. The light level is, along with the
aperture, and the shutter follows from them through the metering equation,
held within the range the camera can actually run.

Accumulating many streaks
-------------------------
Streaks are accumulated as **occupancy**, not as repeated "over" composites:
each streak deposits its ``alpha`` additively into a layer, and the final
coverage is ``1 - exp(-layer)``. For independent drops crossing the same
pixel during one exposure that is the right composition law — the pixel is
uncovered only if *no* drop covered it, so the transmissions multiply — and
unlike sequential ``over`` blending it does not depend on the order the
streaks happen to be drawn in. That order-independence is what lets the whole
field be rasterised in one batch (and, on a GPU, with a plain scatter-add).
"""

from __future__ import annotations

from typing import Annotated, Any

import cv2
import numpy as np
from albumentations.core.pydantic import check_range_bounds, nondecreasing
from albumentations.core.transforms_interface import BaseTransformInitSchema
from pydantic import AfterValidator, Field

from weathermentations.base import WeatherTransform, require_depth_provider, scene_of
from weathermentations.depth.normals import horizon_row, upward_facing
from weathermentations.depth.provider import SharedDepthProvider
from weathermentations.diagnostics import record_draw
from weathermentations.utils.color import illuminant_radiance_linear, sky_radiance_linear
from weathermentations.utils.photometry import from_linear_unit, to_linear_unit
from weathermentations.utils.precipitation import (
    focal_length_px,
    frustum_volume,
    gamma_dsd_slope,
    rain_extinction_coefficient,
    rain_number_density,
    raindrop_terminal_velocity,
    resolvable_fraction,
    sample_resolvable_particles,
    width_cut_for_budget,
)

# Width of the sensor the field of view is taken across, for turning an
# aperture diameter into an f-number: a 1/2.8" sensor, the commonest in
# surveillance cameras, on which a 70 degree field of view is a 4 mm lens.
_SENSOR_WIDTH_MM = 5.6

# K in the metering equation T = K N^2 / E: the incident-light exposure
# constant (250 lux s at ISO 100) over a base sensitivity of ISO 200. At
# 10 000 lux this gives 1/3100 s at f/1.6 and 1/125 s at f/8.
_METERING_CONSTANT_LUX_S = 1.25


def metered_exposure_s(f_number: float, illuminance_lux: float) -> float:
    """Shutter time that exposes a scene under ``illuminance_lux`` at ``f_number``.

    The standard incident-light exposure equation. It is what ties aperture
    to shutter: the light a lens passes goes as ``1 / N^2``, so every stop
    the iris closes doubles the time the shutter must stay open.
    """
    return _METERING_CONSTANT_LUX_S * f_number**2 / max(illuminance_lux, 1e-6)


# Lowest rain rate the transform will accept. Named so presets and the docs
# generator can clamp to it instead of hard-coding a number that silently
# drifts out of step with the validation.
MIN_RAIN_RATE_MM_H = 2.0

# Cross-section softening applied to the accumulated streak layer, in pixels.
#
# Kept small deliberately. Streaks are already rasterised anti-aliased, so this
# only takes the edge off the accumulation — and a typical streak is one or two
# pixels wide, so blurring it is not a cosmetic touch but a direct division of
# its peak opacity. Measured at sigma 0.7 (the previous value) against 0.3: the
# peak alpha fell from 0.100 to 0.058 and the fraction of the frame carrying a
# visible streak fell by two thirds. Most of the rain was being erased by its
# own anti-aliasing pass.
_CROSS_SECTION_SOFTEN_PX = 0.35


def _perspective_angles(
    x: np.ndarray,
    y: np.ndarray,
    width: int,
    height: int,
    focal: float,
    camera_pitch_deg: float,
    wind_angle_deg: float,
) -> np.ndarray:
    """Per-streak slant angle (degrees from vertical) under perspective.

    Rain falls along a single direction in the world. Parallel 3-D lines
    project to lines through one vanishing point, so a streak's direction in
    the image is the direction from that point — which means streaks
    **splay** rather than staying parallel whenever the camera is tilted.
    That splay is one of the strongest cues that rain is falling through a
    scene instead of being drawn over it, and it is free: it follows from
    the projection, with no extra parameter.

    With the camera level the fall direction is perpendicular to the optical
    axis, its vanishing point is at infinity, and the streaks come out
    exactly parallel — so the parallel case is recovered rather than
    special-cased.
    """
    pitch = np.deg2rad(camera_pitch_deg)
    wind = np.deg2rad(wind_angle_deg)

    # World "down" in camera coordinates (+x right, +y down, +z forward),
    # tilted by the wind slant in the image plane.
    direction = np.array(
        [np.sin(wind), np.cos(pitch) * np.cos(wind), np.sin(pitch)], dtype=np.float64
    )

    # A direction with no forward component has its vanishing point at
    # infinity: every streak is parallel to the projected direction.
    if abs(direction[2]) < 1e-4:
        return np.full_like(x, np.rad2deg(np.arctan2(direction[0], direction[1])), dtype=np.float32)

    vanishing_x = width / 2.0 + focal * direction[0] / direction[2]
    vanishing_y = height / 2.0 + focal * direction[1] / direction[2]

    # Drops travel toward the vanishing point of their direction of motion.
    delta_x = vanishing_x - x
    delta_y = vanishing_y - y
    return np.rad2deg(np.arctan2(delta_x, delta_y)).astype(np.float32)


def streak_alpha(drop_diameter_px: float, streak_length_px: float) -> float:
    """Fraction of the exposure a drop occupies one pixel of its own streak.

    Garg & Nayar write this as ``tau/T = 2*r0 / (v * T_exposure)`` in world
    units, but that ratio is scale-invariant and equals
    ``drop_diameter / streak_length`` measured in *pixels* — which is the
    form to use when rendering, because it stays consistent with however long
    the streak is actually drawn. A 3 px drop smeared over a 60 px streak is
    5% opaque; draw the same drop over a 20 px streak (shorter exposure) and
    it is 15% opaque but covers a third as many pixels. Total deposited light
    is conserved either way, which is exactly right.
    """
    return float(np.clip(drop_diameter_px / max(streak_length_px, 1e-3), 0.0, 1.0))


class RandomRain(WeatherTransform):
    """Physically-grounded rain: photometrically correct streaks, veil, wet ground.

    Args:
        depth_provider: shared :class:`~weathermentations.depth.provider.SharedDepthProvider`.
        rain_rate_range_mm_h: rainfall rate, drawn per call. This is the
            *physical* intensity knob:
            it sets the Marshall-Palmer slope, so heavier rain means a longer
            tail of large, fast, conspicuous drops rather than simply more of
            the same drop. It also sets how many drops are in frame, since
            the count is derived from the drop concentration and the volume
            of air being simulated.
        drop_diameter_range_mm: truncation window for the drop-size
            distribution, in millimetres.
        dsd_shape_mu: shape parameter of the drop-size distribution.
            ``0`` is Marshall & Palmer's exponential, the default and what
            every published number in this library was measured against.
            Ulbrich (1983) showed observed spectra are better fitted by a
            gamma, ``N(D) = N0 D^mu exp(-Lambda D)``, with ``mu`` near 2-4: it
            removes the pile-up of very small drops an exponential predicts
            and puts the mode at a finite diameter. At 50 mm/h, ``mu = 3``
            raises the share of drops above 2 mm from 11.5% to 14.8%.

            It will not make rain more visible, and that is worth stating
            plainly: what decides visibility is the summed projected area of
            the drops, and at a fixed rain rate that moves by 2.5% across the
            whole useful range of ``mu``. See ``docs/rain_visibility.md``.
        distance_range_m: near and far bounds of the column of air in front
            of the lens whose drops are drawn individually. Drops beyond it
            are sub-pixel in width and are represented by the veil instead.
        exposure_time_range: the shortest and longest shutter time the
            camera can run, in seconds. With the drop's terminal velocity the
            shutter time fixes the streak length, and therefore the opacity
            ``alpha = D / (v T)``: a longer exposure draws a longer *and
            proportionally fainter* streak, conserving deposited light. The
            time itself is not drawn from this range but metered — see
            ``scene_illuminance_range_lux`` — and held inside it; give the
            two ends the same value to pin the shutter.
        scene_illuminance_range_lux: how much light the scene is under,
            drawn per call on a log scale. Together with the aperture this
            sets the shutter time, ``T = K N^2 / E``, which is why the two
            come out correlated the way a camera's are: stopped down, a long
            shutter and long thin streaks; wide open, a short one and soft
            flecks. Rainy daylight runs from about 800 lux under a storm
            cloud to 25 000 in a sunlit shower; a street at night is 5-150.
            ``None`` draws the shutter time from ``exposure_time_range``
            directly, on a log scale, independent of the aperture.
        horizontal_fov_range_deg: camera field of view, drawn per call. Used
            to project physical
            drop sizes and distances into pixels.
        aperture_diameter_range_mm: lens aperture diameter, drawn per call on
            a log scale (evenly in stops), which sets the
            circle of confusion for drops away from the focal plane. Small
            values are both realistic for a surveillance or dash camera (f/1.6
            on a 4 mm lens is a 2.5 mm aperture) and what makes rain *visible*
            in such footage: the deep depth of field keeps streaks narrow, and
            since defocus spreads a streak's fixed light over a wider band, a
            narrower streak is a proportionally more opaque one. It is the
            only lever that raises streak contrast without also shortening the
            streaks, which shutter time does. The default spans a lens
            stopped down in daylight (0.4 mm, about f/10 at 4 mm) to one wide
            open (3 mm).
        focus_distance_range_m: what the camera is focused on, drawn per call.
        camera_pitch_range_deg: how far the camera is tilted **down** from
            level, drawn per call.
            Rain falls along parallel lines in the world, and parallel lines
            project to lines through a common vanishing point, so a tilted
            camera makes the streaks visibly *splay*. A level camera puts
            that point at infinity and the streaks come out parallel.
        wind_angle_range: slant of the fall direction in degrees, from wind.
        angle_jitter_range: per-streak random deviation from the perspective
            direction, so streaks are not perfectly regular (gust
            turbulence).
        wobble_range: per-streak mid-point bend, as a fraction of its length.
        definition_softness: how soft a defocused streak's edges get, as a
            multiple of its own defocus spread.
        max_streaks: hard cap on individually-rendered streaks, so an extreme
            rain rate cannot make one call unboundedly slow.
        min_streak_width_px: drops projecting narrower than this are not drawn
            individually — they are what the veil represents.
        veil_beta_range: extinction coefficient for the distant rain veil.
            ``None`` (the default) derives it from ``rain_rate_mm_h`` via the
            Atlas (1953) visibility-in-rain relation, which is the physically
            consistent choice: the veil *is* the drops too small to draw
            individually, so setting it independently permits a frame with a
            downpour's haze and a drizzle's streaks. Pass an explicit range
            only to override that deliberately.
        wetness_probability: chance of rendering the ground-wetness pass.
            Deliberately not tied to the drawn rain rate: a road stays wet
            long after the rain stops and is dry for the first seconds of a
            downpour, so wetness follows the rain's *history*, which a
            single-frame transform cannot know.
        wetness_darkening_range: diffuse albedo of wet ground relative to
            dry. Water fills the surface pores and traps light in the film by
            total internal reflection, so wet asphalt scatters back much less
            than dry: measured albedos run about 0.10-0.15 dry against
            0.05-0.08 wet, i.e. roughly a halving, which is what this
            brackets.
        wetness_falloff_range: exponent on the upward-facing cosine. Water
            sheets on horizontal surfaces and runs off anything steeper, more
            sharply than snow settles, so this sits above snow's equivalent.

    Targets:
        image

    Image types:
        uint8, float32
    """

    class InitSchema(BaseTransformInitSchema):
        depth_provider: SharedDepthProvider
        rain_rate_range_mm_h: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(MIN_RAIN_RATE_MM_H, 300.0)),
        ]
        dsd_shape_mu: float = Field(ge=0.0, le=10.0)
        drop_diameter_range_mm: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.1, 8.0)),
        ]
        distance_range_m: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.05, 500.0)),
        ]
        exposure_time_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(1e-5, 1.0)),
        ]
        scene_illuminance_range_lux: (
            Annotated[
                tuple[float, float],
                AfterValidator(nondecreasing),
                AfterValidator(check_range_bounds(0.01, 200000.0)),
            ]
            | None
        )
        horizontal_fov_range_deg: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(5.0, 175.0)),
        ]
        aperture_diameter_range_mm: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.1, 100.0)),
        ]
        focus_distance_range_m: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.2, 1000.0)),
        ]
        camera_pitch_range_deg: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(-89.0, 89.0)),
        ]
        max_streaks: int = Field(ge=0, le=200000)
        min_streak_width_px: float = Field(ge=0.0, le=20.0)
        wind_angle_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(-45.0, 45.0)),
        ]
        angle_jitter_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 30.0)),
        ]
        wobble_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 0.5)),
        ]
        definition_softness: float = Field(ge=0.0, le=2.0)
        veil_beta_range: (
            Annotated[
                tuple[float, float],
                AfterValidator(nondecreasing),
                AfterValidator(check_range_bounds(0.0, 2.0)),
            ]
            | None
        )
        wetness_probability: float = Field(ge=0.0, le=1.0)
        wetness_darkening_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 1.0)),
        ]
        wetness_falloff_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(1.0, 16.0)),
        ]

    def __init__(
        self,
        depth_provider: SharedDepthProvider,
        rain_rate_range_mm_h: tuple[float, float] = (5.0, 200.0),
        drop_diameter_range_mm: tuple[float, float] = (0.8, 6.0),
        dsd_shape_mu: float = 0.0,
        distance_range_m: tuple[float, float] = (0.4, 6.0),
        exposure_time_range: tuple[float, float] = (1.0 / 8000.0, 1.0 / 25.0),
        horizontal_fov_range_deg: tuple[float, float] = (48.0, 78.0),
        aperture_diameter_range_mm: tuple[float, float] = (0.4, 3.0),
        scene_illuminance_range_lux: tuple[float, float] | None = (800.0, 25000.0),
        focus_distance_range_m: tuple[float, float] = (8.0, 40.0),
        camera_pitch_range_deg: tuple[float, float] = (0.0, 22.0),
        max_streaks: int = 16000,
        min_streak_width_px: float = 0.40,
        wind_angle_range: tuple[float, float] = (-10.0, 10.0),
        angle_jitter_range: tuple[float, float] = (2.0, 8.0),
        wobble_range: tuple[float, float] = (0.05, 0.2),
        definition_softness: float = 0.45,
        veil_beta_range: tuple[float, float] | None = None,
        wetness_probability: float = 0.4,
        wetness_darkening_range: tuple[float, float] = (0.4, 0.7),
        wetness_falloff_range: tuple[float, float] = (4.0, 9.0),
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.depth_provider = require_depth_provider(depth_provider, cls_name=type(self).__name__)
        self.rain_rate_range_mm_h = rain_rate_range_mm_h
        self.drop_diameter_range_mm = drop_diameter_range_mm
        self.dsd_shape_mu = dsd_shape_mu
        self.distance_range_m = distance_range_m
        self.exposure_time_range = exposure_time_range
        self.scene_illuminance_range_lux = scene_illuminance_range_lux
        self.horizontal_fov_range_deg = horizontal_fov_range_deg
        self.aperture_diameter_range_mm = aperture_diameter_range_mm
        self.focus_distance_range_m = focus_distance_range_m
        self.camera_pitch_range_deg = camera_pitch_range_deg
        self.max_streaks = max_streaks
        self.min_streak_width_px = min_streak_width_px
        self.wind_angle_range = wind_angle_range
        self.angle_jitter_range = angle_jitter_range
        self.wobble_range = wobble_range
        self.definition_softness = definition_softness
        self.veil_beta_range = veil_beta_range
        self.wetness_probability = wetness_probability
        self.wetness_darkening_range = wetness_darkening_range
        self.wetness_falloff_range = wetness_falloff_range

    def _veil_beta_bounds(self, rain_rate_mm_h: float) -> tuple[float, float]:
        """Veil extinction range, derived from the rain rate unless overridden."""
        if self.veil_beta_range is not None:
            return self.veil_beta_range
        beta = rain_extinction_coefficient(rain_rate_mm_h)
        return (beta * 0.75, beta * 1.3)

    def _exposure(self, horizontal_fov_deg: float) -> tuple[float, float, float, float]:
        """Draw the camera's operating point: shutter, aperture, f-number, light.

        Aperture and light are drawn; the shutter is what the camera would
        then have to use. Both draws are log-uniform — stops and light levels
        are ratios, and a uniform draw over 1/8000-1/25 s would put the
        shutter slower than 1/50 s half the time.
        """

        def log_uniform(low: float, high: float) -> float:
            return float(np.exp(self.py_random.uniform(np.log(low), np.log(high))))

        light = log_uniform(*(self.scene_illuminance_range_lux or self.exposure_time_range))
        aperture_mm = log_uniform(*self.aperture_diameter_range_mm)
        focal_mm = 0.5 * _SENSOR_WIDTH_MM / np.tan(np.deg2rad(horizontal_fov_deg) / 2.0)
        f_number = float(focal_mm / aperture_mm)
        if self.scene_illuminance_range_lux is None:
            # `light` was drawn from the shutter range: there is no meter.
            return light, aperture_mm * 1e-3, f_number, float("nan")
        shortest, longest = self.exposure_time_range
        # Outside what the shutter can do, the camera makes up the rest with
        # gain or runs dark; neither changes what a streak looks like.
        exposure = float(np.clip(metered_exposure_s(f_number, light), shortest, longest))
        return exposure, aperture_mm * 1e-3, f_number, light

    def get_params_dependent_on_data(
        self, params: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, Any]:
        scene = scene_of(data)
        height, width = params["shape"][:2]
        rng = self.random_generator

        depth_pm = self.depth_provider.get_metric_depth(scene)

        # Every physical quantity is drawn per call. A scalar here would make
        # a whole dataset share one rain rate, one camera geometry and one
        # shutter speed, which is the opposite of what an augmentation is for.
        rain_rate = self.py_random.uniform(*self.rain_rate_range_mm_h)
        horizontal_fov = self.py_random.uniform(*self.horizontal_fov_range_deg)
        focus_distance = self.py_random.uniform(*self.focus_distance_range_m)
        camera_pitch = self.py_random.uniform(*self.camera_pitch_range_deg)

        veil_beta = self.py_random.uniform(*self._veil_beta_bounds(rain_rate))
        exposure, aperture_m, f_number, illuminance = self._exposure(horizontal_fov)
        wind_angle = self.py_random.uniform(*self.wind_angle_range)
        angle_jitter = self.py_random.uniform(*self.angle_jitter_range)
        wobble_strength = self.py_random.uniform(*self.wobble_range)
        focal = focal_length_px(width, height, horizontal_fov)

        # How many drops belong in frame follows from the rain rate and the
        # volume of air being simulated — it is not an independent knob that
        # could contradict them. Of those, only the ones projecting wider than
        # `min_streak_width_px` are drawn individually; the rest are what the
        # veil represents.
        #
        # The cut is on *projected width*, not on diameter. Diameter alone
        # cannot tell a resolvable drop from an unresolvable one, because a
        # large distant drop and a small near one subtend the same angle — and
        # cutting by diameter discards exactly the near drops that carry the
        # visible streaks while keeping far ones that cannot be seen at all.
        volume = frustum_volume(
            width, height, horizontal_fov, *self.distance_range_m
        )
        slope = gamma_dsd_slope(rain_rate, self.dsd_shape_mu)
        visible = resolvable_fraction(
            slope,
            *self.drop_diameter_range_mm,
            *self.distance_range_m,
            focal,
            self.min_streak_width_px,
            shape_mu=self.dsd_shape_mu,
        )
        density = rain_number_density(
            rain_rate, self.drop_diameter_range_mm[0], self.dsd_shape_mu
        )
        population = density * volume * visible

        # When the budget binds, raise the *width* threshold rather than
        # truncating the field, so the streaks drawn stay the most visible
        # ones and intensity keeps tracking the rain rate instead of the
        # render budget.
        width_cut = width_cut_for_budget(
            population,
            self.max_streaks,
            slope,
            *self.drop_diameter_range_mm,
            *self.distance_range_m,
            focal,
            self.min_streak_width_px,
            shape_mu=self.dsd_shape_mu,
        )
        count = int(min(self.max_streaks, round(population)))

        record_draw(
            rain_rate_mm_h=rain_rate,
            horizontal_fov_deg=horizontal_fov,
            focus_distance_m=focus_distance,
            camera_pitch_deg=camera_pitch,
            exposure_s=exposure,
            aperture_mm=aperture_m * 1e3,
            f_number=f_number,
            scene_illuminance_lux=illuminance,
            wind_angle_deg=wind_angle,
            veil_beta=veil_beta,
            mp_slope=slope,
            focal_px=focal,
            volume_m3=volume,
            resolvable_fraction=visible,
            population=population,
            width_cut_px=width_cut,
            count_requested=count,
        )

        # Marshall-Palmer sizes, Gunn-Kinzer speeds. Everything a streak looks
        # like follows from these two plus the drop's distance: nothing
        # downstream picks length, width and opacity separately, because in a
        # real photograph they are not separate.
        diameter_mm, distance_m = sample_resolvable_particles(
            rng,
            count,
            slope,
            *self.drop_diameter_range_mm,
            *self.distance_range_m,
            focal,
            width_cut,
            # The slope above is the *gamma* slope. Sampling it as an
            # exponential — which leaving this out did — is a distribution
            # that belongs to neither model: at 50 mm/h and mu = 3 it put 2%
            # of drops above 2 mm where the gamma has 14.8% and Marshall &
            # Palmer 11.5%, and cut the streak count by 62%.
            shape_mu=self.dsd_shape_mu,
        )
        count = len(diameter_mm)
        speed = raindrop_terminal_velocity(diameter_mm)

        # A falling drop is in the air *in front of* the scene, so its
        # distance has nothing to do with the depth map's value behind it.
        # Reading the drop's scale off the scene depth — the previous model —
        # made streaks shrink over distant sky and swell over near road, i.e.
        # pasted onto surfaces rather than falling through the air.
        drop_width = focal * (diameter_mm * 1e-3) / distance_m
        # What the drop sweeps is its travel plus its own width. The second
        # term is nothing at 1/100 s and most of the streak at 1/4000 s,
        # where the drop moves less than its diameter: without it a frozen
        # drop had no length and its opacity ran past one.
        length = focal * (speed * exposure) / distance_m + drop_width

        # Defocus spreads a streak across the circle of confusion without
        # adding light, so it gets wider *and proportionally fainter*. This is
        # the opposite of the intuitive "closer is brighter".
        blur_radius = (
            0.5 * focal * aperture_m * np.abs(1.0 / distance_m - 1.0 / focus_distance)
        )
        thickness = drop_width + 2.0 * blur_radius

        x = rng.uniform(0.0, width, count).astype(np.float32)
        y = rng.uniform(0.0, height, count).astype(np.float32)

        # Rain falls along one direction in the world, and parallel 3-D lines
        # project to lines through a single vanishing point. So the streaks in
        # a tilted camera's frame are not parallel — they splay out from that
        # point, which is one of the strongest cues that rain is falling
        # *through* a scene rather than being drawn on top of it. A level
        # camera puts the vanishing point at infinity and recovers the
        # parallel case exactly.
        angle = _perspective_angles(
            x, y, width, height, focal, camera_pitch, wind_angle
        )
        angle = angle + rng.uniform(-angle_jitter, angle_jitter, count)

        # Wet ground is *ground*, not "the bottom of the frame". The screen-
        # space band this replaced covered only 49% of the reference frame's
        # road surface and left a seam where it ended; on a camera pitched
        # further down — the normal CCTV mounting — the ground fills the
        # frame while a bottom band stays pinned to the bottom third.
        draw_wetness = self.py_random.random() < self.wetness_probability
        wetness_darkening = self.py_random.uniform(*self.wetness_darkening_range)
        wetness_falloff = self.py_random.uniform(*self.wetness_falloff_range)
        wetness = np.zeros((height, width), dtype=np.float32)
        horizon = 0.0
        if draw_wetness:
            cosine = upward_facing(
                depth_pm,
                horizontal_fov_deg=horizontal_fov,
                camera_pitch_deg=camera_pitch,
            )
            # Water runs off a slope rather than lying on it, and does so more
            # decisively than snow, which is why this exponent is higher than
            # snow's accumulation falloff.
            wetness = (cosine**wetness_falloff).astype(np.float32)
            # Fit the horizon on the surface actually being wet, not on the
            # raw cosine. A distant flat plane's normal points at the camera,
            # so `n . up` there is sin(pitch) and crosses a lenient threshold
            # at about 18 degrees of downward pitch -- past which the raw mask
            # swallows the sky and the fitted horizon jumped from row 120 to
            # row 49 on the reference frame. The falloff exponent already
            # excludes those pixels from the render (they carry 1-3% weight
            # against the road's 41-73%), so fitting on the same mask the
            # renderer uses keeps the two consistent and holds the estimate at
            # 123 +- 4 rows across the whole pitch range.
            horizon = horizon_row(depth_pm, wetness, min_weight=0.05)

        record_draw(
            draw_wetness=draw_wetness,
            wetness_darkening=wetness_darkening,
            wetness_falloff=wetness_falloff,
            horizon_row=horizon,
            ground_fraction=float((wetness > 0.05).mean()),
        )

        return {
            "streaks": {
                "x": x,
                "y": y,
                # Carried through for introspection and testing: length,
                # width and opacity all derive from it together.
                "distance": distance_m.astype(np.float32),
                "length": length.astype(np.float32),
                "thickness": np.maximum(1, np.round(thickness)).astype(np.int32),
                "angle": angle.astype(np.float32),
                "bend": rng.uniform(-wobble_strength, wobble_strength, count).astype(np.float32),
                # Garg & Nayar: a drop occupies a pixel for diameter/length of
                # the exposure. Widening it by defocus divides that back down,
                # conserving the light the drop actually deposits.
                "alpha": (
                    np.clip(drop_width / np.maximum(length, 1e-3), 0.0, 1.0)
                    * (drop_width / np.maximum(thickness, 1e-3))
                ).astype(np.float32),
                "phase": rng.uniform(0.0, 2 * np.pi, count).astype(np.float32),
                "blur_sigma": (self.definition_softness * blur_radius).astype(np.float32),
            },
            "angle": wind_angle,
            "veil_transmission": np.exp(-veil_beta * depth_pm).astype(np.float32),
            "draw_wetness": draw_wetness,
            "wetness": wetness,
            "wetness_darkening": wetness_darkening,
            "horizon_px": horizon,
            "focal_px": focal,
        }

    def apply(
        self,
        img: np.ndarray,
        streaks: dict[str, np.ndarray],
        angle: float,
        veil_transmission: np.ndarray,
        draw_wetness: bool,
        wetness: np.ndarray,
        wetness_darkening: float,
        horizon_px: float,
        focal_px: float,
        **params: Any,
    ) -> np.ndarray:
        height, width = img.shape[:2]
        linear, scale, is_integer = to_linear_unit(img)
        original_mean = float(linear.mean())

        # A drop images a ~165 degree field of view, so a streak is basically
        # a picture of the sky. Take its radiance from the scene itself.
        # A streak is a lens imaging the sky, which may be out of frame, so
        # it takes the illuminant. The veil is scattered skylight and
        # saturates at the sky's own recorded radiance instead.
        streak_radiance = illuminant_radiance_linear(linear)
        airlight = sky_radiance_linear(linear)

        occupancy = render_streak_layer(height, width, streaks)

        # Only a light cross-section softening here: the streak *is* the
        # motion blur, so smearing it again along its own axis would divide
        # its (already only a few percent) alpha by the kernel length and
        # erase the rain entirely.
        softened: np.ndarray = cv2.GaussianBlur(
            occupancy, (0, 0), sigmaX=_CROSS_SECTION_SOFTEN_PX
        )
        # Occupancy -> coverage. See the module docstring: drops crossing one
        # pixel during the exposure occlude independently, so the fractions of
        # the exposure they are *absent* for multiply.
        alpha_layer = -np.expm1(-softened)

        # Distant rain that no longer resolves into individual streaks
        # scatters into a veil, the same Koschmieder compositing as fog but
        # with the rain's own (sky-coloured) radiance.
        t = veil_transmission[..., None]
        out = linear * t + airlight[None, None, :] * (1.0 - t)

        # Time-averaged streaks: I = (1-a)*I_bg + a*L_streak.
        a = alpha_layer[..., None]
        out = out * (1.0 - a) + streak_radiance[None, None, :] * a

        if draw_wetness:
            out = _apply_ground_wetness(out, wetness, wetness_darkening, horizon_px, focal_px)

        # Stand-in for the camera's auto-exposure pulling the frame back to a
        # normal level after the scene got brighter/hazier.
        new_mean = float(out.mean())
        if new_mean > 1e-5:
            out *= np.clip(original_mean / new_mean, 0.75, 1.25)

        return from_linear_unit(out, scale, is_integer, img.dtype)


# The beading profile is a function of one phase-like argument, so it tables
# once for the whole library instead of being evaluated per pixel. `s` runs
# 0..1 along a streak, a little beyond at the anti-aliased ends, and the
# random per-streak phase adds up to 2*pi on top.
_BEADING_ARGUMENT_MAX = 6.0 * np.pi * 1.3 + 2.0 * np.pi
_BEADING_LUT_SIZE = 2048
_BEADING_SCALE = _BEADING_LUT_SIZE / _BEADING_ARGUMENT_MAX


def _build_beading_lut() -> np.ndarray:
    t = np.linspace(0.0, _BEADING_ARGUMENT_MAX, _BEADING_LUT_SIZE, dtype=np.float32)
    return ((0.55 + 0.45 * np.sin(t)) * (0.75 + 0.25 * np.sin(2.0 * t))).astype(np.float32)


_BEADING_LUT = _build_beading_lut()


def streak_geometry(streaks: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Per-streak polyline vertices and patch rectangles, for the whole field at once.

    Returns integer patch origins/sizes and ``(N, 7, 2)`` int32 polyline
    vertices in patch-local coordinates, so the rasterising loop does no
    arithmetic of its own.
    """
    x, y = streaks["x"], streaks["y"]
    length = streaks["length"]
    thickness = streaks["thickness"]
    bend = streaks["bend"]
    blur_sigma = streaks["blur_sigma"]

    angle_rad = np.deg2rad(streaks["angle"])
    dir_x, dir_y = np.sin(angle_rad), np.cos(angle_rad)
    perp_x, perp_y = -dir_y, dir_x

    # Tight bounding box. The mid-streak bend peaks at |bend|*length along the
    # perpendicular, so project that onto each axis rather than reserving the
    # streak's whole length in both directions — a near-vertical streak needs
    # only a few pixels of width, and over-reserving it made every patch
    # several times larger than the streak actually drawn into it.
    margin = thickness + 2
    bend_extent = np.abs(bend) * length
    end_x, end_y = x + dir_x * length, y + dir_y * length
    x_lo = np.minimum(x, end_x) - np.abs(dir_y) * bend_extent - margin
    x_hi = np.maximum(x, end_x) + np.abs(dir_y) * bend_extent + margin
    y_lo = np.minimum(y, end_y) - np.abs(dir_x) * bend_extent - margin
    y_hi = np.maximum(y, end_y) + np.abs(dir_x) * bend_extent + margin

    # A defocused streak spreads past its own bounding box, so the patch is
    # grown by the blur's reach up front; blurring a tight patch in place
    # would clip the spread into a hard rectangle.
    pad = np.where(blur_sigma > 0.3, np.ceil(3.0 * blur_sigma), 0.0)

    x0 = np.floor(x_lo - pad).astype(np.int32)
    y0 = np.floor(y_lo - pad).astype(np.int32)
    patch_w = np.ceil(x_hi - x_lo + 2.0 * pad).astype(np.int32)
    patch_h = np.ceil(y_hi - y_lo + 2.0 * pad).astype(np.int32)

    # Seven vertices = the six segments the streak is drawn as. The bend peaks
    # at the middle of the streak.
    s = np.linspace(0.0, 1.0, 7, dtype=np.float32)[None, :]
    offset = bend[:, None] * length[:, None] * 4.0 * s * (1.0 - s)
    along = length[:, None] * s
    vx = x[:, None] + dir_x[:, None] * along + perp_x[:, None] * offset - x0[:, None]
    vy = y[:, None] + dir_y[:, None] * along + perp_y[:, None] * offset - y0[:, None]

    return {
        "x0": x0,
        "y0": y0,
        "patch_w": patch_w,
        "patch_h": patch_h,
        "vertices": np.round(np.stack([vx, vy], axis=-1)).astype(np.int32),
        "dir_x": dir_x.astype(np.float32),
        "dir_y": dir_y.astype(np.float32),
    }


def render_streak_layer(
    height: int, width: int, streaks: dict[str, np.ndarray]
) -> np.ndarray:
    """Rasterise the whole streak field into an ``(H, W)`` occupancy layer.

    A falling drop oscillates (an oblate-prolate mode plus a transverse mode
    at roughly twice the frequency), so its streak is beaded — several smeared
    highlights along its length rather than one even bar. The beading is
    applied as a per-pixel modulation of the drawn polyline, read out of
    :data:`_BEADING_LUT` by the pixel's position *along* the streak, which is
    an affine function of its row and column and so costs one broadcast add
    and one gather per patch.

    Streaks accumulate additively (see the module docstring); they are not
    composited one over the next, so the result does not depend on drawing
    order.
    """
    layer = np.zeros((height, width), dtype=np.float32)
    geometry = streak_geometry(streaks)
    x0s, y0s = geometry["x0"], geometry["y0"]
    patch_ws, patch_hs = geometry["patch_w"], geometry["patch_h"]
    vertices = geometry["vertices"]
    dir_xs, dir_ys = geometry["dir_x"], geometry["dir_y"]
    alphas = streaks["alpha"]
    thicknesses = streaks["thickness"]
    blur_sigmas = streaks["blur_sigma"]
    lengths = streaks["length"]
    phases = streaks["phase"]
    xs, ys = streaks["x"], streaks["y"]

    for i in range(len(alphas)):
        patch_w = int(patch_ws[i])
        patch_h = int(patch_hs[i])
        if patch_w <= 0 or patch_h <= 0:
            continue
        x0, y0 = int(x0s[i]), int(y0s[i])
        # Skip streaks whose patch misses the frame entirely before doing any
        # rasterisation work for them.
        if x0 + patch_w <= 0 or y0 + patch_h <= 0 or x0 >= width or y0 >= height:
            continue

        patch: np.ndarray = np.zeros((patch_h, patch_w), dtype=np.float32)
        cv2.polylines(
            patch,
            [vertices[i]],
            isClosed=False,
            color=float(alphas[i]),
            thickness=int(thicknesses[i]),
            lineType=cv2.LINE_AA,
        )

        # Beading. `s` is the distance along the streak normalised by its
        # length, which is affine in (row, col): s = a*col + b*row + c.
        rate = 6.0 * np.pi / max(float(lengths[i]), 1e-3)
        col_term = (np.arange(patch_w, dtype=np.float32) + (x0 - xs[i])) * (
            rate * dir_xs[i] * _BEADING_SCALE
        )
        row_term = (np.arange(patch_h, dtype=np.float32) + (y0 - ys[i])) * (
            rate * dir_ys[i] * _BEADING_SCALE
        ) + phases[i] * _BEADING_SCALE
        index = (row_term[:, None] + col_term[None, :]).astype(np.int32)
        patch *= np.take(_BEADING_LUT, index, mode="clip")

        if blur_sigmas[i] > 0.3:
            patch = cv2.GaussianBlur(patch, (0, 0), sigmaX=float(blur_sigmas[i]))

        dst_x0, dst_y0 = max(0, x0), max(0, y0)
        dst_x1, dst_y1 = min(width, x0 + patch_w), min(height, y0 + patch_h)
        if dst_x1 <= dst_x0 or dst_y1 <= dst_y0:
            continue
        layer[dst_y0:dst_y1, dst_x0:dst_x1] += patch[
            dst_y0 - y0 : dst_y1 - y0, dst_x0 - x0 : dst_x1 - x0
        ]

    return layer


# Water's normal-incidence reflectance, ((n - 1)/(n + 1))^2 for n = 1.33.
_WATER_R0 = 0.02


def _apply_ground_wetness(
    linear: np.ndarray,
    wetness: np.ndarray,
    darkening: float,
    horizon_px: float,
    focal_px: float,
) -> np.ndarray:
    """Wet ground: a darker diffuse base plus a Fresnel-weighted mirror image.

    Two things here that the previous screen-space band could not express.

    **Where the reflection comes from.** A horizontal mirror flips a ray's
    vertical angle about the horizontal, so the reflection seen at row ``y``
    is the scene at row ``2 * horizon - y``. Mirroring about anything else —
    the band's own top edge, as this used to — puts the reflection of a car
    somewhere the car is not.

    **How strong it is.** Reflectance is not a constant: at grazing incidence
    water approaches a perfect mirror and at normal incidence it reflects 2%.
    Schlick's approximation over the depression angle below the horizon gives
    the familiar look of a wet road — mirror-bright toward the horizon,
    merely dark and matte at your feet — which a single blend weight cannot.
    """
    height, width = linear.shape[:2]
    rows = np.arange(height, dtype=np.float32)

    # Mirror about the horizon, sampling with replication for the rows whose
    # reflection looks above the top of the frame (they see open sky).
    source_y = np.clip(2.0 * horizon_px - rows, 0.0, height - 1.0)
    map_y = np.broadcast_to(source_y.reshape(height, 1), (height, width))
    map_x = np.broadcast_to(np.arange(width, dtype=np.float32), (height, width))
    reflection = cv2.remap(
        linear,
        map_x.copy(),
        map_y.copy(),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    # A road is a rough mirror: it smears the reflection along the direction
    # of travel far more than across it.
    reflection = cv2.GaussianBlur(reflection, (0, 0), sigmaX=1.0, sigmaY=4.0)

    # Schlick: R = R0 + (1 - R0)(1 - cos t)^5, where t is measured from the
    # surface normal, i.e. from vertical. cos t is the *sine* of the ray's
    # depression below the horizon.
    drop = np.maximum(rows - horizon_px, 0.0)
    cos_theta = drop / np.sqrt(drop * drop + focal_px * focal_px)
    fresnel = _WATER_R0 + (1.0 - _WATER_R0) * (1.0 - cos_theta) ** 5
    fresnel_map = np.broadcast_to(fresnel.reshape(height, 1, 1), linear.shape)

    wet = linear * darkening * (1.0 - fresnel_map) + reflection * fresnel_map
    w = wetness[..., None]
    return (linear * (1.0 - w) + wet * w).astype(linear.dtype)
