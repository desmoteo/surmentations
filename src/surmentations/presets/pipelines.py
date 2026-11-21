"""Ready-made pipelines whose effects cannot contradict each other.

Listing every effect with an independent probability produces frames that
could not exist: fog and snow at once, a sun flare in a scene the camera is
lighting with its own infrared illuminator, hard midday shadows under an
IR-mode night sky. These builders sample a **lighting state** and a **weather
state** as exclusive choices, so the impossible combinations are impossible
structurally rather than merely unlikely.

Two things are deliberately *not* exclusive. Water on the front element is
correlated with rain, not implied by it — a shielded lens stays dry through a
downpour, and dew, residual drops and wind-blown spray put water on the glass
under a clear sky — so each weather state carries its own probability of
droplets rather than forcing them on or off. And everything downstream of the
optics (dirt, sensor, ISP, codec) is drawn independently, because a dirty dome
tells you nothing about whether the encoder dropped a frame.

The defaults encode how often each condition occurs in a temperate maritime
climate on a mixed camera fleet. **They will not match your footage**, so every
one is a keyword argument; see ``docs/pipelines.md`` for what each encodes and
which to change first.
"""

from __future__ import annotations

from typing import Any

import albumentations as A

from surmentations.atmosphere.fog import RandomFog
from surmentations.atmosphere.night_fog import NightFog
from surmentations.atmosphere.night_rain import NightRain
from surmentations.atmosphere.night_snow import NightSnow
from surmentations.atmosphere.rain import MIN_RAIN_RATE_MM_H, RandomRain
from surmentations.atmosphere.snow import RandomSnow
from surmentations.camera import (
    AcFlicker,
    ExposureTransient,
    NightColorMode,
    NightIrMode,
    RollingShutter,
    SensorNoise,
    TemporalDenoiseArtifacts,
    TransmissionFaults,
    VideoCodecArtifacts,
    WdrToneMapping,
)
from surmentations.depth.provider import SharedDepthProvider
from surmentations.lens import (
    DirtSoil,
    HousingOccluders,
    SnowOnHousing,
    SourceGlare,
    SunFlare,
    WaterDroplets,
)
from surmentations.presets.severity import GROUND_SEVERITY, SEVERITY
from surmentations.presets.source import BySource
from surmentations.scene import CastShadows

#: Severity the pipelines build at by default. Level 1 is "a slightly bad
#: day" — present in every frame it touches, never the subject of the frame.
#: A pipeline stacks two or three effects, so per-effect severity compounds:
#: level 2 on each of three effects is not an ordinary frame, it is a bad one.
#: Raise it deliberately if your deployment really is that harsh. The transforms' own constructor
#: defaults are deliberately *not* used here: they span the whole physically
#: plausible range, so ``RandomRain`` reaches 200 mm/h, a tropical cloudburst,
#: where severity 1 is 2-8 mm/h. Those wide defaults exist so a single
#: transform can explore its parameter space; they are the wrong thing for a
#: pipeline that stacks three or four effects at once, where each needs to be
#: typical rather than extreme.
DEFAULT_SEVERITY = 1

#: The level the weather was once built at for every frame, kept for callers
#: who ask for it by name (``weather_severity=DEFAULT_WEATHER_SEVERITY``).
#: Unset, each state of the weather is now drawn at its own strength: most
#: rain is light and most fog is thin, and a pipeline that drew rain as often
#: as it rains and always at this level drew a downpour of an hour a year on
#: six frames in a hundred.
DEFAULT_WEATHER_SEVERITY = 2

#: Probability of packet-loss damage on a frame, where the pipeline has a
#: transmission stage at all. Rare: a link that loses packets on more than a
#: few frames in a hundred gets fixed or replaced.
DEFAULT_FAULTS = 0.03


#: How strongly the infrared ring lights something sitting on the dome,
#: relative to full scale. It is centimetres from the lamp, so it is lit far
#: harder than anything in the scene.
_IR_RING_GLOW = (0.6, 1.6)

#: Entrance pupil, in millimetres, of a camera with its iris wide open — every
#: camera at night, when there is no light to spare. It decides how sharp
#: water on the glass looks: drops narrower than this dissolve into a loss of
#: contrast. By day the transform's own default applies, which also covers
#: lenses that stop down in sunlight and so render the same drops crisply.
_OPEN_IRIS_MM = (1.4, 3.0)

#: Brightness of the wall of light the infrared ring raises in fog, relative
#: to full scale. Enough to cost the frame most of its contrast, which is
#: what fog does to an infrared camera.
_IR_FOG_BACKSCATTER = (0.08, 0.3)

#: Illuminance of a street at night under its own lighting, in lux. Rain's
#: shutter time is metered from the light level, and at these levels it sits
#: at the slow end of what the camera can run: long, faint streaks.
_NIGHT_STREET_LUX = (5.0, 150.0)

# How many times full scale a street lamp is taken to be when it lights rain
# and reflects off a wet road. A clipped pixel is a lower bound and nothing
# in the frame says how far above it the lamp was.
_LAMP_OVER_RANGE = (4.0, 12.0)


def _at(factory: Any, name: str, probability: float, severity: int, **extra: Any) -> Any:
    """One transform at a severity level, built directly.

    Deliberately **not** wrapped in a container to pick the severity per
    frame. Albumentations seeds every child of a ``Compose`` from the same
    stream, and a ``OneOf`` that declines to fire consumes exactly one number,
    so a pipeline of same-shaped ``OneOf`` gates never desynchronises: they
    all fire together whenever the shared draw falls below the smallest ``p``.
    Measured, that turned an expected 2.4 effects per frame into 64% of frames
    with one effect or none and 12% with nine or more — the all-or-nothing
    distribution that makes a pipeline look far too destructive on the frames
    where it does fire. Adding a ``NoOp`` arm does not fix it either.

    So severity is fixed per pipeline rather than drawn per frame. Build two
    pipelines and alternate if you want a mixture; that composes correctly
    because the choice is then outside Albumentations' RNG.
    """
    # `extra` overrides the preset rather than colliding with it, so a caller
    # can keep a level's physics and still pin one knob — the rain branch
    # raises `wetness_probability` that way.
    return factory(p=probability, **{**SEVERITY[severity][name], **extra})


def _ground(name: str, ground_severity: int | None) -> dict[str, Any]:
    """Ground-state arguments for the rain or snow transform.

    ``None`` leaves how wet or how deep to the transform, which draws it
    per frame over its whole range, and says only how often there is
    anything on the ground at all: rain mostly falls on a road that is wet
    already, and snow on snow that is lying.
    """
    if ground_severity is None:
        if name == "RandomRain":
            return {"wetness_probability": WET_UNDER_RAIN}
        return {"accumulation_probability": SNOW_UNDER_SNOW}
    if ground_severity not in GROUND_SEVERITY:
        raise ValueError(
            f"ground_severity must be one of {sorted(GROUND_SEVERITY)} or None, "
            f"got {ground_severity!r}"
        )
    return dict(GROUND_SEVERITY[ground_severity][name])


# How often each stage of the camera's own chain shows, by the light the
# frame is in: by day, under street lighting, under the camera's own
# infrared. See ``docs/pipelines.md`` for what each is an estimate of.
#
# Insects and webs gather at a lit lens: a share of ``occluders`` by day and
# under street lamps, more than it under the infrared ring.
_OCCLUDERS_BY_LIGHT = {"day": 0.25, "street": 0.4, "infrared": 1.7}
# Mains banding needs mains light and a short shutter. The camera's own
# infrared LEDs are not on the mains, and band only what street lamps light.
_FLICKER_BY_LIGHT = {"day": 0.02, "street": 0.15, "infrared": 0.05}
# The noise reducer is a setting, and at night it is on: the share of
# ``denoise`` at which its artefacts show in each light. By day there is
# little noise for it to work on, and it shows a quarter as often.
_DENOISE_BY_LIGHT = {"day": 0.25, "street": 1.0, "infrared": 1.0}
#: How often the noise reducer's artefacts show at night, and wide dynamic
#: range's in any light: what ``denoise`` and ``wdr`` are unless given.
DEFAULT_DENOISE = 0.8
DEFAULT_WDR = 0.30
# A whole-frame shear needs the mount to move: wind, not traffic.
_ROLLING_SHUTTER = 0.10
# An exposure caught mid-correction is an event of a second or two.
_EXPOSURE_TRANSIENT = 0.05


def _sensor_noise(*, light: str, own_noise: bool, severity: int) -> list[Any]:
    """Colour sensor noise, where the transform that made the frame's light has none of its own.

    Gain is how a camera copes with less light, so it follows the lighting
    state and not a draw: by day it is low and the noise occasional, under
    artificial light it is two severity levels up and always there. That is
    the one place a pipeline moves a transform off its own severity, and it
    does so because severity *is* gain for this transform. A frame made
    into a night by `NightIrMode` or `NightColorMode` has that transform's
    own noise, and a second source would count it twice.
    """
    if own_noise:
        return []
    if light == "day":
        return [_at(SensorNoise, "SensorNoise", 0.15, severity)]
    return [_at(SensorNoise, "SensorNoise", 1.0, min(severity + 2, 5))]


def _stream(
    *,
    lit_by_mains: bool,
    infrared: bool,
    dirt: float,
    occluders: float,
    codec: float,
    faults: float,
    severity: int,
    own_noise: bool | None = None,
    wdr: float = DEFAULT_WDR,
    denoise: float = DEFAULT_DENOISE,
) -> list[Any]:
    """Glass, sensor, ISP and transmission: independent draws, physical order.

    ``lit_by_mains`` and ``infrared`` say what light the frame is in: day,
    street lighting, or the camera's own illuminator. ``own_noise`` says
    the frame was made into a night by a transform that has its own sensor
    noise; unset, that is taken to be so under infrared and nowhere else.
    ``wdr`` and ``denoise`` are how often the two settings of the camera's
    own processing show: wide dynamic range in any light, the noise
    reducer at night and a quarter as often by day.

    Written in order by hand rather than through
    :func:`~surmentations.presets.pipeline.physical_order`, which sorts a
    flat list and cannot see inside the containers these pipelines are built
    from. ``tests/unit/test_pipeline_presets.py`` checks the order still holds.
    """
    light = "infrared" if infrared else "street" if lit_by_mains else "day"
    if own_noise is None:
        own_noise = infrared
    stream = [
        # Optics: on the glass, in front of everything the sensor does.
        _at(DirtSoil, "DirtSoil", dirt, severity),
        # An insect on the dome is a dark silhouette unless the camera's own
        # infrared ring is lighting it from a few centimetres away, which only
        # happens in the IR-night state; and it is there far more often then,
        # the ring being what draws it.
        _at(
            HousingOccluders,
            "HousingOccluders",
            min(1.0, occluders * _OCCLUDERS_BY_LIGHT[light]),
            severity,
            glow_range=_IR_RING_GLOW if infrared else (0.0, 0.0),
        ),
        # Glare from headlights and lamps: only where lights are what the
        # frame is exposed against.
        *([_at(SourceGlare, "SourceGlare", 0.35, severity)] if lit_by_mains else []),
        # Sensor.
        _at(AcFlicker, "AcFlicker", _FLICKER_BY_LIGHT[light], severity),
        _at(RollingShutter, "RollingShutter", _ROLLING_SHUTTER, severity),
        *_sensor_noise(light=light, own_noise=own_noise, severity=severity),
        # ISP.
        _at(ExposureTransient, "ExposureTransient", _EXPOSURE_TRANSIENT, severity),
        _at(WdrToneMapping, "WdrToneMapping", wdr, severity),
        _at(
            TemporalDenoiseArtifacts,
            "TemporalDenoiseArtifacts",
            min(1.0, denoise * _DENOISE_BY_LIGHT[light]),
            severity,
        ),
    ]
    # Transmission, last: it acts on the finished picture. Left out entirely
    # at probability zero rather than constructed and never fired, because
    # constructing it requires the ffmpeg binary — and `codec=0.0` is both the
    # documented setting for frames decoded from video and the only way to run
    # these pipelines on a machine without ffmpeg.
    if codec > 0.0:
        stream.append(_at(VideoCodecArtifacts, "VideoCodecArtifacts", codec, severity))
    # What the link loses of the stream, after the camera has encoded it.
    # The same rule: it needs ffmpeg to construct, so at zero it is absent.
    if faults > 0.0:
        stream.append(_at(TransmissionFaults, "TransmissionFaults", faults, severity))
    return stream


def _fault_rate(faults: float | None, codec: float) -> float:
    """Packet-loss probability, off by default wherever the codec stage is.

    ``codec=0`` is how a caller says either that their frames already came
    out of a video stream or that the machine has no ffmpeg. The second must
    keep working, so an unset ``faults`` follows ``codec`` to zero.
    """
    if faults is None:
        return DEFAULT_FAULTS if codec > 0.0 else 0.0
    return faults


# ------------------------------------------------------------------ the weather
#
# The states a frame's air and ground can be in, and what each is drawn
# with. The shares of time are the caller's (``rain``, ``snow``, ``fog``,
# ``wet_road``, ``snow_lying``); these say how each divides. They are
# estimates for a lowland site in central Europe, and
# ``docs/aware_effects_and_pipelines_review.md`` says where each is from.

#: Rain by strength: the share of rain time, and the rate in mm/h. Most
#: rain is under the first rate `RandomRain` draws: its streaks cannot be
#: seen, and the frame is a wet road under a grey sky.
RAIN_STRENGTHS: dict[str, tuple[float, tuple[float, float]]] = {
    "light": (0.72, (MIN_RAIN_RATE_MM_H, 3.0)),
    "moderate": (0.255, (MIN_RAIN_RATE_MM_H, 10.0)),
    "heavy": (0.025, (10.0, 50.0)),
}
#: The share of fog that has rain or drizzle falling through it.
FOG_WITH_RAIN = 0.2
#: Fog by thickness: the share of the fog without rain that is dense, and
#: the visibility of thin fog and of dense, in metres.
DENSE_FOG = 0.25
FOG_VISIBILITY_M = {"thin": (200.0, 1000.0), "dense": (50.0, 200.0)}
#: How much likelier fog is by night than over the whole day, and by day:
#: it forms after dark and clears after sunrise.
FOG_BY_HOUR = {"day": 0.5, "night": 1.5}
#: The share of rain that falls on a road already wet, and of falling snow
#: that falls on snow already lying.
WET_UNDER_RAIN = 0.9
SNOW_UNDER_SNOW = 0.8
#: How likely snow is still on the housing once it has stopped falling,
#: against how likely it is while it falls: it slides off or melts.
SNOW_STAYS_ON_HOUSING = 0.5
#: Snowfall, in mm/h of water: a flurry to a moderate fall. Heavier is rare
#: enough to be asked for by name (``weather_severity``).
SNOWFALL_MM_H = (0.2, 2.5)
#: A road still wet after the rain: from damp to wet, with a little water
#: standing; and the cloud that may be left over it.
_AFTER_RAIN = {
    "wetness_probability": 1.0,
    "wetness_darkening_range": (0.45, 0.9),
    "puddle_coverage_range": (0.0, 0.2),
    "overcast_range": (0.0, 0.7),
}
#: Snow lying after the fall: a thin coat to a deep one.
_AFTER_SNOW = {"accumulation_probability": 1.0, "accumulation_strength_range": (0.3, 1.0)}
#: The traffic on a wet road by day: its lamps, and its spray. Both need a
#: provider that reads the vehicles, and draw nothing without one.
_LAMPS_BY_DAY = (0.6, 1.2)
_SPRAY = (0.2, 0.8)
#: The power the shares are raised to by default: the square root, which
#: over-samples what is rare without making it common.
DEFAULT_PRIOR_POWER = 0.5
#: The states of the air and the ground, in the order :func:`weather_shares`
#: gives them and :func:`weather_state` holds their arms.
WEATHER_STATES = (
    "dry",
    "wet_road",
    "rain_light",
    "rain_moderate",
    "rain_heavy",
    "fog_thin",
    "fog_dense",
    "fog_rain",
    "snow",
    "snow_lying",
)


def weather_shares(
    *,
    rain: float = 0.065,
    snow: float = 0.006,
    fog: float = 0.02,
    wet_road: float = 0.10,
    snow_lying: float = 0.03,
    hour: str | None = None,
) -> dict[str, float]:
    """The share of time each state of the air and the ground holds, summing to 1.

    ``rain``, ``snow`` and ``fog`` are the shares of time each is there at
    all; ``wet_road`` and ``snow_lying`` the shares in which nothing falls
    and the ground still shows what fell. ``hour`` is ``"day"`` or
    ``"night"`` to weigh fog by the hour, or ``None`` for the whole day.

    The states are exclusive, and they are not rain, snow and fog. Rain
    falls through fog; most rain is too light to see; a road is wet for
    longer than it rains, and snow lies for days after an hour's fall.

    Raises:
        ValueError: if the shares leave no room for dry air over dry
            ground.
    """
    if hour is not None:
        # No further than half of what the caller's shares leave dry: fog
        # asked for on most frames is not pushed past all of them at night.
        room = max(0.0, 1.0 - (rain + snow + fog + wet_road + snow_lying))
        fog = min(fog * FOG_BY_HOUR[hour], fog + 0.5 * room)
    both = FOG_WITH_RAIN * min(fog, rain)
    shares = {
        f"rain_{name}": share * (rain - both) for name, (share, _mm) in RAIN_STRENGTHS.items()
    }
    shares["fog_thin"] = (1.0 - DENSE_FOG) * (fog - both)
    shares["fog_dense"] = DENSE_FOG * (fog - both)
    shares["fog_rain"] = both
    shares["snow"] = snow
    shares["wet_road"] = wet_road
    shares["snow_lying"] = snow_lying
    total = sum(shares.values())
    if total >= 1.0:
        raise ValueError(
            f"rain, snow, fog, wet_road and snow_lying must leave room for clear weather, "
            f"got {total:.3f}. These are shares of time, not relative weights."
        )
    shares["dry"] = 1.0 - total
    return {name: shares[name] for name in WEATHER_STATES}


def rebalanced(shares: dict[str, float], prior_power: float) -> dict[str, float]:
    """Shares raised to a power and summed to 1 again.

    At 1 they are what they were: frames in the proportions a camera sees
    them, nearly all dry. At 0 every state is as likely as every other.
    Between, what is rare is drawn more often than it happens and still
    less often than what is common; 0.5 is the square root, the usual
    choice for classes with a long tail.
    """
    if not 0.0 <= prior_power <= 1.0:
        raise ValueError(f"prior_power must be between 0 and 1, got {prior_power!r}")
    raised = {name: share**prior_power if share > 0.0 else 0.0 for name, share in shares.items()}
    total = sum(raised.values())
    return {name: value / total for name, value in raised.items()}


def weather_state(
    depth_provider: SharedDepthProvider,
    *,
    rain: float = 0.065,
    snow: float = 0.006,
    fog: float = 0.02,
    wet_road: float = 0.10,
    snow_lying: float = 0.03,
    wet_rain: float = 0.35,
    wet_snow: float = 0.10,
    wet_fog: float = 0.18,
    wet_clear: float = 0.06,
    wet_after: float = 0.15,
    snow_on_housing: float = 0.30,
    prior_power: float = DEFAULT_PRIOR_POWER,
    severity: int | None = None,
    droplet_severity: int = DEFAULT_SEVERITY,
    ground_severity: int | None = None,
    night: bool = False,
    relight: Any = None,
    infrared: bool = False,
    sunlit: tuple[Any, ...] = (),
    sunlit_after: tuple[Any, ...] = (),
    traffic: bool = True,
) -> A.OneOf:
    """Exactly one state of the air and the ground per frame, including "dry".

    Args:
        depth_provider: one shared provider, passed to every depth-conditioned
            transform so depth is estimated once per image.
        rain, snow, fog: the share of time each is there, over the whole
            day. Rain falls through fog a fifth of the time there is fog.
        wet_road, snow_lying: the share of time nothing falls and the road
            is still wet, or the snow still lies. Together with ``rain``,
            ``snow`` and ``fog`` they must sum to less than 1.
        wet_rain, wet_snow, wet_fog, wet_clear, wet_after:
            ``P(water on the glass | state)``. Rain splashing straight onto
            an exposed dome is the commonest source, which is why
            ``wet_rain`` leads; fog has to condense on glass that is colder
            than the air. ``wet_after`` is the drops still there on a road
            that is still wet; ``wet_clear`` covers dew and wind-blown
            spray, which is why it is not zero.
        snow_on_housing: ``P(snow on the housing | snow falling)``: a cap
            hanging into the top of the view, and patches on the glass
            (`SnowOnHousing`). Half as likely over snow that only lies.
        prior_power: how far the states are drawn as often as they happen:
            1 for that, 0 for every state alike, and between them for the
            rare ones drawn more often than they happen
            (:func:`rebalanced`). The default is the square root.
        severity: ``None``, the default, draws each state at its own
            strength (:data:`RAIN_STRENGTHS`, :data:`FOG_VISIBILITY_M`,
            :data:`SNOWFALL_MM_H`). A level 1-5 draws all falling weather
            and all fog at that level of
            :data:`~surmentations.presets.severity.SEVERITY` instead: a
            pipeline of storms, for a model that has to be shown them.
        droplet_severity: level for the water on the glass, which follows the
            **camera chain** rather than the weather: a drop on the front
            element is centimetres from the sensor and dominates any frame
            it is on.
        ground_severity: level 0-5 from
            :data:`~surmentations.presets.severity.GROUND_SEVERITY` for
            how wet the road is under rain and how much snow has settled
            under snow, or ``None`` — the default — to draw it per frame.
        night: the frame is lit artificially. With no ``relight`` it is
            taken as it is, a frame that was a night already: the lens is
            wide open, rain is metered for street lighting and lit by the
            lamps in the picture, and fog is `NightFog`.
        relight: the transform that makes a night of a daylight frame
            (`NightIrMode` or `NightColorMode`, at ``p=1``). Each state
            then comes in two parts with it between them: what is in the
            scene, the wet road and the settled snow, before it, so that
            the night's lamps light them; and what is in the air after it,
            lit by those lamps (`NightRain`, `NightSnow`, `NightFog`).
        infrared: ``relight`` is the camera's own illuminator. What is in
            the air and on the glass is lit by it from centimetres away;
            in a night in colour there is no such lamp.
        sunlit, sunlit_after: transforms that need the sun (cast shadows,
            a flare), put before and after the state's own in the states
            that can have it: dry air over a dry road or over lying snow.
            Never where anything falls or hangs in the air.
        traffic: by day, light the traffic's lamps in rain and fog and lay
            its spray on a wet road. Both need a provider that reads the
            vehicles and draw nothing without one.

    Raises:
        ValueError: if the shares sum to 1 or more.
    """
    hour = "night" if night else "day"
    shares = rebalanced(
        weather_shares(
            rain=rain, snow=snow, fog=fog, wet_road=wet_road, snow_lying=snow_lying, hour=hour
        ),
        prior_power,
    )
    relit = relight is not None
    glass: dict[str, Any] = {"aperture_diameter_range_mm": _OPEN_IRIS_MM} if night else {}
    if infrared:
        # Water on the glass sits in the infrared ring's beam, centimetres
        # from the lamp: under its own illuminator a wet dome glows.
        glass["illuminator_glow_range"] = _IR_RING_GLOW
    # Light that comes from a pole is not changed by cloud overhead.
    lamplit: dict[str, Any] = {"overcast_range": (0.0, 0.0)} if night else {}
    # A night frame taken as it is: the same camera is looking at the rain,
    # with a shutter metered for street lighting, and the rain is lit by
    # the lamps in the picture.
    as_it_is: dict[str, Any] = (
        {
            "aperture_diameter_range_mm": _OPEN_IRIS_MM,
            "scene_illuminance_range_lux": _NIGHT_STREET_LUX,
            "lamp_glint_range": _LAMP_OVER_RANGE,
        }
        if night and not relit
        else {}
    )
    by_day = traffic and not night

    def droplets(probability: float) -> Any:
        """``WaterDroplets`` at ``probability``, at this pipeline's severity and glass."""
        return _at(WaterDroplets, "WaterDroplets", probability, droplet_severity, **glass)

    def capped(probability: float) -> Any:
        """``SnowOnHousing`` at ``probability``: lit by the infrared ring where there is one."""
        return _at(
            SnowOnHousing,
            "SnowOnHousing",
            probability,
            droplet_severity,
            glow_range=_IR_RING_GLOW if infrared else (0.0, 0.0),
        )

    def of(name: str, level: int, **own: Any) -> dict[str, Any]:
        """A transform's ranges: the state's own, or the level the caller fixed."""
        if severity is not None:
            return dict(SEVERITY[severity][name])
        return {**SEVERITY[level][name], **own}

    def wet(strength: str | None, streaks: bool) -> Any:
        """`RandomRain` for one strength of rain, or for a road left wet with none falling."""
        if strength is None:
            settings: dict[str, Any] = {
                "rain_rate_range_mm_h": (MIN_RAIN_RATE_MM_H, MIN_RAIN_RATE_MM_H),
                **_AFTER_RAIN,
            }
        else:
            settings = of("RandomRain", 2, rain_rate_range_mm_h=RAIN_STRENGTHS[strength][1])
            settings.update(_ground("RandomRain", ground_severity))
        if not streaks:
            # No streaks, and no grain of far rain to go with them: the
            # grain is that of a count of drops, a tenth of the transform's
            # time, and with nothing falling or with the night's own
            # transform drawing the rain there is nothing for it to match.
            settings.update(max_streaks=0, distant_rain_range=(0.0, 0.0))
        if by_day:
            settings["spray_range"] = _SPRAY
            if strength is not None:
                settings["vehicle_lamp_range"] = _LAMPS_BY_DAY
        return RandomRain(
            depth_provider=depth_provider, p=1.0, **{**settings, **lamplit, **as_it_is}
        )

    def snowing(flakes: bool, lying: bool) -> Any:
        """`RandomSnow` for snow that is falling, or for snow that only lies."""
        if lying:
            settings = {
                **SEVERITY[1]["RandomSnow"],
                "snowfall_rate_range_mm_h": (SNOWFALL_MM_H[0], SNOWFALL_MM_H[0]),
                **_AFTER_SNOW,
            }
        else:
            settings = of("RandomSnow", 1, snowfall_rate_range_mm_h=SNOWFALL_MM_H)
            settings.update(_ground("RandomSnow", ground_severity))
        if not flakes:
            # As for rain without streaks: no speckle of far snow either.
            settings.update(max_flakes=0, distant_snow_range=(0.0, 0.0))
        return RandomSnow(depth_provider=depth_provider, p=1.0, **{**settings, **lamplit})

    def fogged(thickness: str) -> Any:
        """Fog of one thickness: lit by the sky by day, by the lamps in it at night."""
        visibility = FOG_VISIBILITY_M[thickness]
        if night:
            own = {"visibility_range_m": visibility}
            if infrared:
                # The wall of light it raises is the illuminator's own.
                own["illuminator_backscatter_range"] = _IR_FOG_BACKSCATTER
            return NightFog(depth_provider=depth_provider, p=1.0, **of("NightFog", 1, **own))
        level = 1 if thickness == "thin" else 3
        settings = of("RandomFog", level, beta_range=(3.912 / visibility[1], 3.912 / visibility[0]))
        if by_day:
            settings["vehicle_lamp_range"] = _LAMPS_BY_DAY
        return RandomFog(depth_provider=depth_provider, p=1.0, **settings)

    def lit_rain(strength: str) -> Any:
        """Rain in the light of a night that was made: `NightRain`."""
        settings = of(
            "NightRain",
            2 if strength == "heavy" else 1,
            rain_rate_range_mm_h=RAIN_STRENGTHS[strength][1],
        )
        if not infrared:
            # No lamp at the lens: the rain shows in the street's light.
            settings.update(illuminator_gain_range=(0.0, 0.0), backscatter_range=(0.0, 0.0))
        return NightRain(p=1.0, **settings)

    def lit_snow() -> Any:
        """Falling snow in the light of a night that was made: `NightSnow`."""
        settings = of("NightSnow", 1, snowfall_rate_range_mm_h=SNOWFALL_MM_H)
        if not infrared:
            settings.update(illuminator_gain_range=(0.05, 0.05), backscatter_range=(0.0, 0.0))
        return NightSnow(p=1.0, **settings)

    both = max(wet_rain, wet_fog)
    falling, lain = capped(snow_on_housing), capped(SNOW_STAYS_ON_HOUSING * snow_on_housing)
    if relit:
        # A daylight frame made into a night: the scene first, then the
        # night's light, then what is in the air in that light.
        arms: dict[str, list[Any]] = {
            "dry": [relight, droplets(wet_clear)],
            "wet_road": [wet(None, streaks=False), relight, droplets(wet_after)],
            **{
                f"rain_{strength}": [
                    wet(strength, streaks=False),
                    relight,
                    lit_rain(strength),
                    droplets(wet_rain),
                ]
                for strength in RAIN_STRENGTHS
            },
            "fog_thin": [relight, fogged("thin"), droplets(wet_fog)],
            "fog_dense": [relight, fogged("dense"), droplets(wet_fog)],
            "fog_rain": [wet("light", streaks=False), relight, fogged("thin"), droplets(both)],
            "snow": [
                snowing(flakes=False, lying=False),
                relight,
                lit_snow(),
                droplets(wet_snow),
                falling,
            ],
            "snow_lying": [snowing(flakes=False, lying=True), relight, droplets(wet_clear), lain],
        }
    else:
        arms = {
            "dry": [*sunlit, droplets(wet_clear), *sunlit_after],
            "wet_road": [wet(None, streaks=False), droplets(wet_after)],
            **{
                f"rain_{strength}": [wet(strength, streaks=True), droplets(wet_rain)]
                for strength in RAIN_STRENGTHS
            },
            "fog_thin": [fogged("thin"), droplets(wet_fog)],
            "fog_dense": [fogged("dense"), droplets(wet_fog)],
            # The road is the rain's and the air is the fog's: the fog is
            # drawn over the wet road, and rain this light has no streaks
            # that fog would leave to be seen.
            "fog_rain": [wet("light", streaks=False), fogged("thin"), droplets(both)],
            "snow": [snowing(flakes=True, lying=False), droplets(wet_snow), falling],
            "snow_lying": [
                snowing(flakes=False, lying=True),
                *sunlit,
                droplets(wet_clear),
                lain,
                *sunlit_after,
            ],
        }
    # In the order of `WEATHER_STATES`, less the states with no share.
    return A.OneOf(
        [
            A.Sequential(arms[name], p=shares[name])
            for name in WEATHER_STATES
            if shares[name] > 0.0
        ],
        p=1.0,
    )


def surveillance(
    depth_provider: SharedDepthProvider,
    *,
    day: float = 0.55,
    lit_night: float = 0.15,
    ir_night: float = 0.30,
    codec: float = 0.15,
    faults: float | None = None,
    dirt: float = 0.25,
    occluders: float = 0.12,
    shadows: float = 0.30,
    flare: float = 0.12,
    wdr: float = DEFAULT_WDR,
    denoise: float = DEFAULT_DENOISE,
    severity: int = DEFAULT_SEVERITY,
    weather_severity: int | None = None,
    ground_severity: int | None = None,
    prior_power: float = DEFAULT_PRIOR_POWER,
    read_source: bool = True,
    **weather: Any,
) -> A.Compose:
    """The full pipeline: weather, optics and the camera chain. **Needs CUDA.**

    One lighting state and one state of the air and the ground per frame,
    then the camera's own degradations. See ``docs/pipelines.md`` for what
    every number is an estimate of.

    Args:
        depth_provider: construct **one**
            :class:`~surmentations.depth.provider.SharedDepthProvider` and
            pass it here, so depth is estimated once per image rather than once
            per transform. What it can read of the scene (labels, the
            camera's mounting, the vehicles) every branch uses.
        day, lit_night, ir_night: relative weights of the three lighting
            states. ``lit_night`` is a night in colour under street
            lighting (`NightColorMode`); ``ir_night`` is the camera cut to
            monochrome with its illuminator on (`NightIrMode`). Which of
            the two a camera does is a matter of where it stands, and for
            one camera one of them should be 0.
        codec: probability of re-encoding. Usually **0** for frames already
            decoded from video, where the artefacts are already in the pixels.
        faults: probability of packet loss damaging the frame — stale,
            smeared or green bands. Unset, it is 0.03, or 0 when ``codec``
            is 0: like the codec stage it needs ffmpeg, and ``codec=0.0``
            has to go on working without one.
        dirt, occluders: probability of grime, and of insects and webs on
            the housing under the infrared ring, which is where they
            gather; by day and under street lamps a share of it.
        wdr: probability of wide dynamic range's artefacts, in any light.
            It is a setting of some cameras: 0 for a fleet that has it
            off, which is also a few milliseconds a frame less.
        denoise: probability of the noise reducer's artefacts at night;
            by day it is a quarter of this. A setting too, and at night
            the costliest stage of the camera's chain: 0 for cameras that
            have it off.
        shadows, flare: probability of hard cast shadows and of a sun
            flare, in the day frames that can have a sun: dry air, over a
            dry road or over lying snow.
        severity: level 1-5, from
            :data:`~surmentations.presets.severity.SEVERITY`. **This is
            the knob that decides how degraded a typical frame is**, and the
            first one to change if the output looks too heavy. 1 is barely
            visible, 2 an ordinary bad day, 5 a frame you would file a fault
            report about.
        weather_severity: ``None``, the default, draws each state of the
            weather at its own strength: most rain light, most fog thin.
            A level 1-5 draws all of it at that level instead.
        ground_severity: level 0-5 for how wet the road is under rain and
            how much snow has settled under snow, or ``None`` to draw it per
            frame, which is the default.
        prior_power: 1 draws the weather as often as it happens, 0.5, the
            default, the rare states more often than that
            (:func:`rebalanced`).
        read_source: look at each frame before anything is done to it
            (:func:`~surmentations.presets.source.read_source`). A frame
            taken at night is not made into a night again, and gets the
            weather of a night and no sun; a frame taken under a roof, which
            only a provider with scene labels can tell, gets no weather at
            all. ``False`` takes every frame for a dry daylight one.
        **weather: forwarded to :func:`weather_state`: ``rain``, ``snow``,
            ``fog``, ``wet_road`` and ``snow_lying`` as shares of time, and
            the odds of water on the glass in each.

    Example:
        >>> from surmentations.depth import (
        ...     DepthAnythingV2Estimator, SharedDepthProvider)
        >>> from surmentations.presets import surveillance
        >>> provider = SharedDepthProvider(DepthAnythingV2Estimator())
        >>> transform = surveillance(provider, codec=0.0)
        >>> augmented = transform(image=image)["image"]     # doctest: +SKIP
    """
    tail = {
        "dirt": dirt,
        "occluders": occluders,
        "codec": codec,
        "faults": _fault_rate(faults, codec),
        "wdr": wdr,
        "denoise": denoise,
    }
    shared: dict[str, Any] = {
        "severity": weather_severity,
        "droplet_severity": severity,
        "ground_severity": ground_severity,
        "prior_power": prior_power,
        **weather,
    }
    in_colour = _at(NightColorMode, "NightColorMode", 1.0, severity, depth_provider=depth_provider)
    in_infrared = _at(NightIrMode, "NightIrMode", 1.0, severity, depth_provider=depth_provider)
    by_day = A.OneOf(
        [
            A.Sequential(
                [
                    weather_state(
                        depth_provider,
                        night=False,
                        # Shadows fall on the scene and a flare is in the
                        # lens: either side of whatever lies on the ground.
                        sunlit=(
                            _at(
                                CastShadows, "CastShadows", shadows, severity,
                                depth_provider=depth_provider,
                            ),
                        ),
                        sunlit_after=(_at(SunFlare, "SunFlare", flare, severity),),
                        **shared,
                    ),
                    *_stream(
                        lit_by_mains=False,
                        infrared=False,
                        own_noise=False,
                        severity=severity,
                        **tail,
                    ),
                ],
                p=day,
            ),
            A.Sequential(
                [
                    # A night in colour, under street lighting: lit from
                    # its lamps, where the provider knows where they are.
                    weather_state(depth_provider, night=True, relight=in_colour, **shared),
                    *_stream(
                        lit_by_mains=True, infrared=False, own_noise=True, severity=severity, **tail
                    ),
                ],
                p=lit_night,
            ),
            A.Sequential(
                [
                    weather_state(
                        depth_provider, night=True, relight=in_infrared, infrared=True, **shared
                    ),
                    *_stream(
                        lit_by_mains=True,
                        infrared=True,
                        own_noise=True,
                        severity=severity,
                        **tail,
                    ),
                ],
                p=ir_night,
            ),
        ],
        p=1.0,
    )
    if not read_source:
        return A.Compose([by_day])
    # A frame that is a night already is not lit again: it gets a night's
    # weather, in the light it has. One under a roof gets the camera alone.
    at_night = A.Sequential(
        [
            weather_state(depth_provider, night=True, **shared),
            *_stream(
                lit_by_mains=True,
                infrared=False,
                own_noise=False,
                severity=severity,
                **tail,
            ),
        ],
        p=1.0,
    )
    indoors = A.Sequential(
        _stream(
            lit_by_mains=True, infrared=False, own_noise=False, severity=severity, **tail
        ),
        p=1.0,
    )
    return A.Compose([BySource(by_day, at_night, indoors, depth_provider)])


def surveillance_lite(
    *,
    night: float = 0.45,
    codec: float = 0.15,
    faults: float | None = None,
    dirt: float = 0.25,
    occluders: float = 0.12,
    droplets: float = 0.10,
    flare: float = 0.05,
    wdr: float = DEFAULT_WDR,
    denoise: float = DEFAULT_DENOISE,
    severity: int = DEFAULT_SEVERITY,
) -> A.Compose:
    """Everything that does not need depth. **CPU only, no GPU required.**

    The weather effects, cast shadows and both night modes condition on a
    monocular depth estimate, which costs a fixed ~46 ms and a CUDA device.
    Dropping them leaves the lens and the camera chain — which is most of what
    actually degrades a surveillance stream, and none of it needs a GPU.

    Use it when you have no GPU to spare in the input pipeline, when your
    data already contains real weather, or as the cheap majority of a mixed
    pipeline.

    What it cannot do: no fog, rain or snow, no cast shadows, and **no night**
    at all. The ``night`` fraction only turns the mains-lit artefacts on — the
    flicker banding and the denoiser — because nothing here can relight a
    daylit frame, and tinting one does not approximate night. Use
    :func:`surveillance` if you need night as a domain.

    Args:
        night: fraction of frames treated as artificially lit, which raises
            flicker banding and temporal-denoise artefacts.
        codec: probability of re-encoding; usually 0 for frames already
            decoded from video.
        faults: probability of packet-loss damage, as for
            :func:`surveillance`: 0.03 unless ``codec`` is 0.
        dirt, occluders: grime and insects on the housing.
        droplets: water on the front element from dew, spray or rain this
            pipeline is not rendering.
        flare: probability of a sun flare by day. Lower than
            :func:`surveillance`'s, which draws it only on the frames it
            knows to be dry: here nothing says whether the sun is out.
        wdr, denoise: probability of wide dynamic range's artefacts, and of
            the noise reducer's at night (a quarter of it by day), as for
            :func:`surveillance`. Both are settings of a camera, and the
            two stages of this pipeline that cost the most: 0 for cameras
            that have them off.
        severity: level 1-5, as for :func:`surveillance`.
    """
    tail = {
        "dirt": dirt,
        "occluders": occluders,
        "codec": codec,
        "faults": _fault_rate(faults, codec),
        "wdr": wdr,
        "denoise": denoise,
    }
    return A.Compose(
        [
            A.OneOf(
                [
                    A.Sequential(
                        [
                            _at(WaterDroplets, "WaterDroplets", droplets, severity),
                            _at(SunFlare, "SunFlare", flare, severity),
                            *_stream(
                                lit_by_mains=False,
                                infrared=False,
                                own_noise=False,
                                severity=severity,
                                **tail,
                            ),
                        ],
                        p=1.0 - night,
                    ),
                    # Deliberately *not* an under-exposed, warm-cast copy of
                    # the day branch. Nothing here can render night, and a
                    # colour shift on a daylit frame does not approximate
                    # one. All "night" means here is that the lighting is
                    # artificial, which raises the mains flicker and the
                    # denoiser's workload. Use `surveillance` if you need
                    # night as a domain.
                    A.Sequential(
                        [
                            _at(
                                WaterDroplets, "WaterDroplets", droplets, severity,
                                aperture_diameter_range_mm=_OPEN_IRIS_MM,
                            ),
                            *_stream(
                                lit_by_mains=True,
                                infrared=False,
                                own_noise=False,
                                severity=severity,
                                **tail,
                            ),
                        ],
                        p=night,
                    ),
                ],
                p=1.0,
            )
        ]
    )
