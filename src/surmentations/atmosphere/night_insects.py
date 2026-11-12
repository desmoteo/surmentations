"""Insects and dust in the illuminator's beam.

Physical model
--------------
An infrared ring is a lamp, and a lamp at night collects insects. They fly
through the first metre or two in front of the lens, lit from centimetres
away against a scene lit from metres, so each is far brighter than anything
behind it. Recorders that trigger on motion fill up with them, and the
streaks they leave are common enough to have a folk name, "rods".

Three things make the picture, all of them consequences of where the insect
is and how long the shutter is open:

**It is a streak, because it moves.** A night camera exposes for 1/50 to
1/25 s. A moth at 1.5 m/s half a metre away crosses 60 to 120 px of a
640 px frame in that time, several times its own image, so what is recorded
is its path. The light it returns in the exposure is fixed, so the longer
the streak the dimmer each part of it: flux is conserved along the path.

**The streak is beaded, because the wings beat.** A wing broadside to the
lamp returns far more light than one edge-on, so the light along the path
rises and falls once a beat. A moth at 40 Hz makes one to two beats in an
exposure; a midge at 250 Hz makes ten. That modulation, on a streak, is the
"rod" with its row of lobes.

**It is out of focus, and lit by its own distance.** A point of it is drawn
as the aperture's disc, ``A f / 2z`` pixels in radius, so a near insect is a
broad soft band and a far one a fine line; and its brightness follows the
lamp's inverse square at *its* distance, not the depth of what is behind
it, which is the same correction ``NightRain`` and ``NightSnow`` make.

**Dust is the same thing, standing still.** A mote a few centimetres from
the lens is an unresolved point: it is drawn as the aperture's disc and
nothing else, a faint round "orb" that drifts too slowly to streak.

Composition: after ``NightIrMode``, with the other things the camera lights
itself.
"""

from __future__ import annotations

import math
from typing import Annotated, Any

import numpy as np
from albumentations.core.pydantic import check_range_bounds, nondecreasing
from albumentations.core.transforms_interface import BaseTransformInitSchema
from pydantic import AfterValidator

from surmentations.atmosphere.night_rain import _beam_profile
from surmentations.base import CameraTransform
from surmentations.diagnostics import record_draw
from surmentations.utils.illuminator import lamp_falloff, splat_discs, veiling_glare
from surmentations.utils.photometry import from_linear_unit, to_linear_unit
from surmentations.utils.precipitation import focal_length_px

# Where ``illuminator_gain_range`` is stated, as for the night rain and snow.
_GAIN_DISTANCE_M = 1.0
# An insect's image is not drawn smaller than this, in pixels of radius.
_MIN_RADIUS_PX = 0.5
# Radius of the disc with the area an insect of unit length presents: body and
# wings together are much narrower than they are long.
_AREA_RADIUS = 0.3
# Discs a streak is drawn with at most, and how far apart along it, in radii.
_MAX_STREAK_DISCS = 600
_STREAK_STEP = 0.5
# Where dust is close enough to show: from the glass to a hand's width out.
_DUST_RANGE_M = (0.02, 0.25)


def _log_uniform(rng: np.random.Generator, low: float, high: float, count: int) -> np.ndarray:
    """Values between ``low`` and ``high``, uniform in their logarithm."""
    values: np.ndarray = np.exp(rng.uniform(math.log(low), math.log(high), count))
    return values


def insect_streaks(
    rng: np.random.Generator,
    *,
    width: int,
    height: int,
    focal: float,
    aperture_m: float,
    focus_m: float,
    exposure_s: float,
    gain: float,
    distance_m: np.ndarray,
    length_m: np.ndarray,
    speed_m_s: np.ndarray,
    wing_beat_hz: np.ndarray,
    wing_share: np.ndarray,
) -> dict[str, np.ndarray]:
    """The discs a set of flying insects is drawn as over one exposure.

    Each insect is a disc, its own image blurred by the aperture's, moved
    along a gently curved path as long as it flies in the exposure. The
    light it returns in that time, its radiance at its distance from the
    lamp times the pixels it covers, is shared out along the path, more
    where the wings are broadside. Returns ``x``, ``y``, ``radius`` and
    ``weight`` as :func:`~surmentations.utils.illuminator.splat_discs`
    takes them, and ``insect``, which insect each disc belongs to.
    """
    count = len(distance_m)
    seen = _AREA_RADIUS * length_m * focal / distance_m
    blur = 0.5 * aperture_m * focal * np.abs(1.0 / distance_m - 1.0 / focus_m)
    radius = np.maximum(_MIN_RADIUS_PX, np.hypot(seen, blur))
    radiance = gain * lamp_falloff(distance_m) / lamp_falloff(_GAIN_DISTANCE_M)
    flux = radiance * np.pi * seen * seen

    path = focal * speed_m_s * exposure_s / distance_m
    heading = rng.uniform(0.0, 2.0 * np.pi, count)
    bow = rng.uniform(-0.25, 0.25, count) * path
    start_x = rng.uniform(-0.1, 1.1, count) * width
    start_y = rng.uniform(-0.1, 1.1, count) * height
    phase = rng.uniform(0.0, 2.0 * np.pi, count)

    discs = np.clip(np.ceil(path / (_STREAK_STEP * radius)), 1, _MAX_STREAK_DISCS).astype(np.int64)
    owner = np.repeat(np.arange(count), discs)
    first = np.cumsum(discs) - discs
    along = (np.arange(int(discs.sum())) - first[owner] + 0.5) / discs[owner]
    centred = along - 0.5
    bend = 4.0 * along * (1.0 - along) * bow[owner]
    cos, sin = np.cos(heading[owner]), np.sin(heading[owner])
    # One beat is one rise and fall of the light, about a mean of one.
    beat = 1.0 + np.cos(2.0 * np.pi * wing_beat_hz[owner] * exposure_s * along + phase[owner])
    share = (1.0 - wing_share[owner]) + wing_share[owner] * beat
    return {
        "x": (start_x[owner] + centred * path[owner] * cos - bend * sin).astype(np.float32),
        "y": (start_y[owner] + centred * path[owner] * sin + bend * cos).astype(np.float32),
        "radius": radius[owner].astype(np.float32),
        "weight": (flux[owner] / discs[owner] * share).astype(np.float32),
        "insect": owner,
    }


class NightInsects(CameraTransform):
    """Insects flying through the illuminator's beam, and dust hanging in it.

    Needs no depth provider: the insects are in front of the scene and lit
    by the lamp. Runs on CPU. Apply it after ``NightIrMode``.

    Args:
        insect_count_range: how many insects cross the view in one exposure.
        body_length_range_mm: their size, midge to large moth.
        distance_range_m: how far from the lens they fly. Drawn evenly in
            the logarithm, so near ones are as common as far ones, as they
            are around a lamp.
        flight_speed_range_m_s: how fast, which with the shutter time and
            the distance gives the streak's length.
        wing_beat_range_hz: wing-beat frequency: about 30 for a moth,
            hundreds for a midge. With the shutter time it gives the number
            of lobes along the streak.
        wing_share_range: how much of an insect's light comes from its
            wings and so rises and falls with the beat, 0 to 1.
        dust_count_range: motes of dust near the lens, each one faint disc.
        dust_brightness_range: brightness of a mote's disc 10 cm from the
            lens, in units of full scale.
        illuminator_gain_range: radiance of an insect one metre away, in
            units of full scale. An insect at 10 cm is some eighty times
            that, and clips.
        beam_falloff_range: how sharply the beam dims off-axis.
        exposure_time_range: shutter time.
        horizontal_fov_range_deg: field of view, which projects metres to
            pixels.
        aperture_diameter_range_mm: the lens's stop, which a point out of
            focus is drawn as.
        focus_distance_range_m: where the lens is focused.
        glare_range: fraction of the light that the dome and the lens
            scatter into a glow around it.

    Targets:
        image

    Image types:
        uint8, float32
    """

    class InitSchema(BaseTransformInitSchema):
        """The arguments of ``NightInsects`` as Albumentations validates them: each
        range ordered low to high and within the bounds given here.
        """

        insect_count_range: Annotated[
            tuple[int, int],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0, 200)),
        ]
        body_length_range_mm: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.5, 80.0)),
        ]
        distance_range_m: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.02, 20.0)),
        ]
        flight_speed_range_m_s: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 10.0)),
        ]
        wing_beat_range_hz: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(1.0, 1000.0)),
        ]
        wing_share_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 1.0)),
        ]
        dust_count_range: Annotated[
            tuple[int, int],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0, 500)),
        ]
        dust_brightness_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 5.0)),
        ]
        illuminator_gain_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 50.0)),
        ]
        beam_falloff_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 10.0)),
        ]
        exposure_time_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(1e-4, 0.5)),
        ]
        horizontal_fov_range_deg: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(10.0, 150.0)),
        ]
        aperture_diameter_range_mm: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.2, 20.0)),
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

    def __init__(
        self,
        insect_count_range: tuple[int, int] = (1, 10),
        body_length_range_mm: tuple[float, float] = (2.0, 25.0),
        distance_range_m: tuple[float, float] = (0.15, 4.0),
        flight_speed_range_m_s: tuple[float, float] = (0.3, 3.0),
        wing_beat_range_hz: tuple[float, float] = (20.0, 250.0),
        wing_share_range: tuple[float, float] = (0.3, 0.9),
        dust_count_range: tuple[int, int] = (0, 25),
        dust_brightness_range: tuple[float, float] = (0.02, 0.2),
        illuminator_gain_range: tuple[float, float] = (0.6, 3.0),
        beam_falloff_range: tuple[float, float] = (0.6, 2.2),
        exposure_time_range: tuple[float, float] = (0.020, 0.040),
        horizontal_fov_range_deg: tuple[float, float] = (48.0, 78.0),
        aperture_diameter_range_mm: tuple[float, float] = (0.9, 2.6),
        focus_distance_range_m: tuple[float, float] = (8.0, 40.0),
        glare_range: tuple[float, float] = (0.05, 0.25),
        p: float = 0.5,
    ):
        """Store the ranges; each argument is described in the class docstring."""
        super().__init__(p=p)
        self.insect_count_range = insect_count_range
        self.body_length_range_mm = body_length_range_mm
        self.distance_range_m = distance_range_m
        self.flight_speed_range_m_s = flight_speed_range_m_s
        self.wing_beat_range_hz = wing_beat_range_hz
        self.wing_share_range = wing_share_range
        self.dust_count_range = dust_count_range
        self.dust_brightness_range = dust_brightness_range
        self.illuminator_gain_range = illuminator_gain_range
        self.beam_falloff_range = beam_falloff_range
        self.exposure_time_range = exposure_time_range
        self.horizontal_fov_range_deg = horizontal_fov_range_deg
        self.aperture_diameter_range_mm = aperture_diameter_range_mm
        self.focus_distance_range_m = focus_distance_range_m
        self.glare_range = glare_range

    def get_params_dependent_on_data(
        self, params: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, Any]:
        """Draw the insects and the dust and reduce both to discs of light.

        Returns the ``discs`` (centres, radii, and the light each holds),
        the ``glare`` and the ``beam`` across the frame.
        """
        del data
        height, width = params["shape"][:2]
        rng = self.random_generator

        insects = self.py_random.randint(*self.insect_count_range)
        motes = self.py_random.randint(*self.dust_count_range)
        gain = self.py_random.uniform(*self.illuminator_gain_range)
        dust_brightness = self.py_random.uniform(*self.dust_brightness_range)
        falloff = self.py_random.uniform(*self.beam_falloff_range)
        exposure = self.py_random.uniform(*self.exposure_time_range)
        horizontal_fov = self.py_random.uniform(*self.horizontal_fov_range_deg)
        aperture_m = self.py_random.uniform(*self.aperture_diameter_range_mm) * 1e-3
        focus = self.py_random.uniform(*self.focus_distance_range_m)
        glare = self.py_random.uniform(*self.glare_range)
        focal = focal_length_px(width, height, horizontal_fov)

        distance = _log_uniform(rng, *self.distance_range_m, insects)
        speed = rng.uniform(*self.flight_speed_range_m_s, insects)
        streaks = insect_streaks(
            rng,
            width=width,
            height=height,
            focal=focal,
            aperture_m=aperture_m,
            focus_m=focus,
            exposure_s=exposure,
            gain=gain,
            distance_m=distance,
            length_m=rng.uniform(*self.body_length_range_mm, insects) * 1e-3,
            speed_m_s=speed,
            wing_beat_hz=_log_uniform(rng, *self.wing_beat_range_hz, insects),
            wing_share=rng.uniform(*self.wing_share_range, insects),
        )

        # Dust: an unresolved point, so the aperture's disc and nothing
        # else. Its brightness follows the lamp alone: the light a point
        # returns and the area its disc covers both go as 1/z^2.
        dust_distance = _log_uniform(rng, *_DUST_RANGE_M, motes)
        dust_radius = np.maximum(
            _MIN_RADIUS_PX, 0.5 * aperture_m * focal * np.abs(1.0 / dust_distance - 1.0 / focus)
        )
        brightness = dust_brightness * lamp_falloff(dust_distance) * rng.uniform(0.3, 1.0, motes)
        discs = {
            "x": np.concatenate([streaks["x"], rng.uniform(0.0, width, motes).astype(np.float32)]),
            "y": np.concatenate([streaks["y"], rng.uniform(0.0, height, motes).astype(np.float32)]),
            "radius": np.concatenate([streaks["radius"], dust_radius.astype(np.float32)]),
            "weight": np.concatenate(
                [
                    streaks["weight"],
                    (brightness * np.pi * (dust_radius**2 + 1.0 / 12.0)).astype(np.float32),
                ]
            ),
        }

        record_draw(
            insect_count=insects,
            dust_count=motes,
            exposure_s=exposure,
            illuminator_gain=gain,
            beam_falloff=falloff,
            focal_px=focal,
            aperture_mm=aperture_m * 1e3,
            insect_distance_m=distance.astype(np.float32),
            streak_length_px=(focal * speed * exposure / distance).astype(np.float32),
            disc_count=len(discs["x"]),
        )
        return {"discs": discs, "glare": glare, "beam": _beam_profile(height, width, falloff)}

    def apply(
        self,
        img: np.ndarray,
        discs: dict[str, np.ndarray],
        glare: float,
        beam: np.ndarray,
        **params: Any,
    ) -> np.ndarray:
        """Add the insects' and the dust's light to the frame, with its glare.

        Emissive against a dark scene, like the streaks of ``NightRain``: an
        insect covers too little of any pixel, for too little of the
        exposure, to hide what is behind it.
        """
        height, width = img.shape[:2]
        lit = np.zeros((height, width), dtype=np.float32)
        splat_discs(lit, discs["x"], discs["y"], discs["radius"], discs["weight"])
        lit = veiling_glare(lit, glare)
        lit *= beam
        linear, scale, is_integer = to_linear_unit(img)
        return from_linear_unit(linear + lit[..., None], scale, is_integer, img.dtype)
