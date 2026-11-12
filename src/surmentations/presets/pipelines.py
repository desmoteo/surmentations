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
from surmentations.atmosphere.rain import RandomRain
from surmentations.atmosphere.snow import RandomSnow
from surmentations.camera import (
    AcFlicker,
    ExposureTransient,
    NightIrMode,
    RollingShutter,
    SensorNoise,
    TemporalDenoiseArtifacts,
    TransmissionFaults,
    VideoCodecArtifacts,
    WdrToneMapping,
)
from surmentations.depth.provider import SharedDepthProvider
from surmentations.lens import DirtSoil, HousingOccluders, SourceGlare, SunFlare, WaterDroplets
from surmentations.presets.severity import GROUND_SEVERITY, SEVERITY
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

#: Severity the *weather* is built at, separately and higher.
#:
#: Weather is the only thing in these pipelines a viewer is meant to notice.
#: It also fires rarely — 12% of frames by default, because that is roughly
#: how often it rains or fogs in a temperate maritime climate — so a frame
#: that draws it and then renders an invisible drizzle has wasted the draw.
#: Measured at 512 px, severity 1 snow puts **19 flakes** in the whole frame
#: and almost all of its effect comes from the distance veil rather than from
#: anything the eye reads as snow; severity 2 puts 179 there, and rain goes
#: from 269 streaks to 1144.
#:
#: The camera chain is the opposite case: it is present in nearly every frame,
#: so it should sit at the bottom of its ladder and compound gently.
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


# Share of the pipelines' rain frames that get a wet road when the ground is
# left to chance. It was 0.85; see `_ground`.
_WET_ROAD_SHARE = 0.5


def _ground(name: str, ground_severity: int | None) -> dict[str, Any]:
    """Ground-state arguments for the rain or snow transform.

    ``None`` leaves the ground to the transform, which draws it per frame
    over its whole range: dry to soaked, bare to buried, whatever is falling.
    Rain's chance of a wet road is raised a little on the way, since by day
    the road is most of what shows that it is raining at all. Only a little:
    a wet road is also two to three times the cost of the frame.
    """
    if ground_severity is None:
        return {"wetness_probability": _WET_ROAD_SHARE} if name == "RandomRain" else {}
    if ground_severity not in GROUND_SEVERITY:
        raise ValueError(
            f"ground_severity must be one of {sorted(GROUND_SEVERITY)} or None, "
            f"got {ground_severity!r}"
        )
    return dict(GROUND_SEVERITY[ground_severity][name])


def _sensor_noise(*, lit_by_mains: bool, infrared: bool, severity: int) -> list[Any]:
    """Colour sensor noise, where the camera is in colour.

    Gain is how a camera copes with less light, so it follows the lighting
    state and not a draw: by day it is low and the noise occasional, under
    artificial light it is two severity levels up and usual. That is the one
    place a pipeline moves a transform off its own severity, and it does so
    because severity *is* gain for this transform. Under the infrared ring
    there is none of this: `NightIrMode` has its own noise, for a monochrome
    frame, and a second source would count it twice.
    """
    if infrared:
        return []
    if lit_by_mains:
        return [_at(SensorNoise, "SensorNoise", 0.70, min(severity + 2, 5))]
    return [_at(SensorNoise, "SensorNoise", 0.15, severity)]


def _stream(
    *,
    lit_by_mains: bool,
    infrared: bool,
    dirt: float,
    occluders: float,
    codec: float,
    faults: float,
    severity: int,
) -> list[Any]:
    """Glass, sensor, ISP and transmission: independent draws, physical order.

    Written in order by hand rather than through
    :func:`~surmentations.presets.pipeline.physical_order`, which sorts a
    flat list and cannot see inside the containers these pipelines are built
    from. ``tests/unit/test_pipeline_presets.py`` checks the order still holds.
    """
    stream = [
        # Optics: on the glass, in front of everything the sensor does.
        _at(DirtSoil, "DirtSoil", dirt, severity),
        # An insect on the dome is a dark silhouette unless the camera's own
        # infrared ring is lighting it from a few centimetres away, which only
        # happens in the IR-night state. The transform's default draws the
        # glow from the whole 0-1.6 range whatever the lighting — right for
        # exploring it, and here it put glowing white shapes on four daylit
        # frames in five, a combination these pipelines exist to rule out.
        _at(
            HousingOccluders,
            "HousingOccluders",
            occluders,
            severity,
            glow_range=_IR_RING_GLOW if infrared else (0.0, 0.0),
        ),
        # Glare from headlights and lamps: only where lights are what the
        # frame is exposed against.
        *([_at(SourceGlare, "SourceGlare", 0.35, severity)] if lit_by_mains else []),
        # Sensor. Mains banding needs artificial light, so outdoors it belongs
        # to the night branches; indoors it would apply all day.
        _at(AcFlicker, "AcFlicker", 0.45 if lit_by_mains else 0.05, severity),
        _at(RollingShutter, "RollingShutter", 0.50, severity),
        *_sensor_noise(lit_by_mains=lit_by_mains, infrared=infrared, severity=severity),
        # ISP.
        _at(ExposureTransient, "ExposureTransient", 0.10, severity),
        _at(WdrToneMapping, "WdrToneMapping", 0.12, severity),
        _at(
            TemporalDenoiseArtifacts,
            "TemporalDenoiseArtifacts",
            0.30 if lit_by_mains else 0.12,
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


def weather_state(
    depth_provider: SharedDepthProvider,
    *,
    rain: float = 0.06,
    snow: float = 0.03,
    fog: float = 0.03,
    wet_rain: float = 0.35,
    wet_snow: float = 0.10,
    wet_fog: float = 0.18,
    wet_clear: float = 0.06,
    severity: int = DEFAULT_WEATHER_SEVERITY,
    droplet_severity: int = DEFAULT_SEVERITY,
    ground_severity: int | None = None,
    night: bool = False,
) -> A.OneOf:
    """Exactly one atmospheric state per frame, including "clear".

    Args:
        depth_provider: one shared provider, passed to every depth-conditioned
            transform so depth is estimated once per image.
        rain, snow, fog: absolute probability of each state. The remainder is
            clear weather; they must sum to less than 1.
        wet_rain, wet_snow, wet_fog, wet_clear: ``P(water on the glass | state)``.
            Rain splashing straight onto an exposed dome is the commonest
            source, which is why ``wet_rain`` leads; fog has to condense on
            glass that is colder than the air, which happens but less often.
            ``wet_clear`` covers dew, residual drops after the rain stops and
            wind-blown spray, which is why it is not zero.
        severity: level 1-5 for the weather itself. Defaults to
            :data:`DEFAULT_WEATHER_SEVERITY`, which is higher than the rest of
            the pipeline on purpose — see that constant.
        droplet_severity: level for the water on the glass, which follows the
            **camera chain** rather than the weather. Droplets are an optics
            -stage effect that rain merely makes likely; at the weather's
            severity they dominate every frame they appear on, because a drop
            on the front element is centimetres from the sensor and a raindrop
            in the air is metres away. Measured at 512 px, severity 2 droplets
            cost 24% of the frame's high-frequency detail against 15% at
            severity 1, and they were the most conspicuous thing in the
            output while the rain they signalled was invisible.
        ground_severity: level 0-5 from
            :data:`~surmentations.presets.severity.GROUND_SEVERITY` for
            how wet the road is under rain and how much snow has settled
            under snow, or ``None`` — the default — to draw it per frame
            over the whole range. Separate from ``severity`` because it is:
            the ground follows how long the weather has lasted, so a
            cloudburst on a dry road and a flurry over deep snow both occur.
        night: the frame is lit artificially, so the lens is wide open. That
            matters to the water on the glass, whose sharpness is set by the
            aperture — at night most of it blurs away into soft patches of
            lost contrast — and to the rain, whose shutter time is metered
            for street lighting and so runs long.

    Raises:
        ValueError: if the three weather probabilities sum to 1 or more.
    """
    total = rain + snow + fog
    if total >= 1.0:
        raise ValueError(
            f"rain + snow + fog must leave room for clear weather, got {total:.3f}. "
            "These are absolute probabilities, not relative weights."
        )
    glass = {"aperture_diameter_range_mm": _OPEN_IRIS_MM} if night else {}
    # The same camera is looking at the rain: open iris, and a shutter
    # metered for street lighting rather than for daylight.
    # And the rain is lit by the lamps, not by a sky: streaks show near a
    # lamp and nowhere else, a wet road reflects the lamps, and cloud
    # overhead changes nothing about light that comes from a pole.
    exposure = (
        {
            **glass,
            "scene_illuminance_range_lux": _NIGHT_STREET_LUX,
            "lamp_glint_range": _LAMP_OVER_RANGE,
            "overcast_range": (0.0, 0.0),
        }
        if night
        else {}
    )

    # Cloud changes nothing about light that comes from a pole, for snow as
    # for rain.
    lamplit: dict[str, Any] = {"overcast_range": (0.0, 0.0)} if night else {}

    def _droplets(probability: float) -> Any:
        """``WaterDroplets`` at ``probability``, at this pipeline's severity and glass."""
        return _at(WaterDroplets, "WaterDroplets", probability, droplet_severity, **glass)

    return A.OneOf(
        [
            A.Sequential(
                [
                    # In daylight a streak carries the sky's radiance, so
                    # against a bright sky it has almost no contrast --
                    # measured, lengthening the exposure 5x leaves the
                    # high-frequency energy it adds flat, because alpha =
                    # D/(vT) spreads the same light further. What actually
                    # says "rain" in a daylit frame is the wet road, which
                    # `_ground` makes likely.
                    _at(
                        RandomRain, "RandomRain", 1.0, severity,
                        depth_provider=depth_provider,
                        **_ground("RandomRain", ground_severity),
                        **exposure,
                    ),
                    _droplets(wet_rain),
                ],
                p=rain,
            ),
            A.Sequential(
                [
                    _at(
                        RandomSnow, "RandomSnow", 1.0, severity,
                        depth_provider=depth_provider,
                        **_ground("RandomSnow", ground_severity),
                        **lamplit,
                    ),
                    _droplets(wet_snow),
                ],
                p=snow,
            ),
            A.Sequential(
                [
                    # At night fog is lit by the lamps in it, not by a sky.
                    _at(NightFog, "NightFog", 1.0, severity, depth_provider=depth_provider)
                    if night
                    else _at(
                        RandomFog, "RandomFog", 1.0, severity, depth_provider=depth_provider
                    ),
                    _droplets(wet_fog),
                ],
                p=fog,
            ),
            A.Sequential(
                [_droplets(wet_clear)],
                p=1.0 - total,
            ),
        ],
        p=1.0,
    )


def _infrared_night(
    depth_provider: SharedDepthProvider,
    *,
    severity: int,
    weather_severity: int,
    tail: dict[str, float],
    rain: float = 0.06,
    snow: float = 0.03,
    fog: float = 0.03,
    wet_rain: float = 0.35,
    wet_snow: float = 0.10,
    wet_fog: float = 0.18,
    wet_clear: float = 0.06,
    ground_severity: int | None = None,
) -> A.OneOf:
    """The IR-night branch, where rain is a different transform.

    Everywhere else the weather is chosen once and applied before the sensor.
    Here it cannot be, because rain lit by the camera's own illuminator has to
    follow the reflectance-to-infrared conversion rather than precede it — the
    streaks are already in infrared terms. Fog still goes before it: it veils
    the *scene*, and the sensor then sees the veiled scene.

    Snow is split in two for the same reason. Its veil and whatever has
    settled belong to the scene and go before the conversion; the falling
    flakes are lit by the lamp and come after it, as `NightSnow`. Left to
    `RandomSnow` alone they went through the conversion with everything else
    and were lit by the depth of whatever was *behind* them, which made a
    flake in front of the sky as dark as the sky.

    So this duplicates the shared tail across the four weather states rather
    than sharing one copy. That costs four sets of constructed objects and no
    runtime, and it keeps the exclusivity structural: a frame still gets
    exactly one weather state, and `NightRain` can only fire on the arm that
    chose rain.
    """
    ir = _at(NightIrMode, "NightIrMode", 1.0, severity, depth_provider=depth_provider)

    def _with(*before: Any) -> list[Any]:
        """The stages given, followed by the camera stages of an infrared night."""
        return [*before, *_stream(lit_by_mains=True, infrared=True, severity=severity, **tail)]

    def _droplets(probability: float) -> Any:
        # Water on the glass sits in the infrared ring's beam, centimetres
        # from the lamp, and the iris is wide open. Both are properties of
        # this lighting state, not draws: a wet dome under its own
        # illuminator glows, and by day it cannot.
        """``WaterDroplets`` lit by the infrared ring, with the iris wide open, as at night."""
        return _at(
            WaterDroplets, "WaterDroplets", probability, severity,
            illuminator_glow_range=_IR_RING_GLOW,
            aperture_diameter_range_mm=_OPEN_IRIS_MM,
        )

    return A.OneOf(
        [
            A.Sequential(
                _with(
                    ir,
                    _at(NightRain, "NightRain", 1.0, weather_severity),
                    _droplets(wet_rain),
                ),
                p=rain,
            ),
            A.Sequential(
                _with(
                    _at(
                        RandomSnow, "RandomSnow", 1.0, weather_severity,
                        # Veil and settled snow only; the flakes are NightSnow's.
                        depth_provider=depth_provider, max_flakes=0,
                        overcast_range=(0.0, 0.0),
                        **_ground("RandomSnow", ground_severity),
                    ),
                    ir,
                    _at(NightSnow, "NightSnow", 1.0, weather_severity),
                    _droplets(wet_snow),
                ),
                p=snow,
            ),
            A.Sequential(
                _with(
                    ir,
                    # After the conversion, unlike daylight fog: the wall of
                    # light it raises is the illuminator's own.
                    _at(
                        NightFog, "NightFog", 1.0, weather_severity,
                        depth_provider=depth_provider,
                        illuminator_backscatter_range=_IR_FOG_BACKSCATTER,
                    ),
                    _droplets(wet_fog),
                ),
                p=fog,
            ),
            A.Sequential(
                _with(ir, _droplets(wet_clear)),
                p=max(1.0 - rain - snow - fog, 1e-6),
            ),
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
    severity: int = DEFAULT_SEVERITY,
    weather_severity: int = DEFAULT_WEATHER_SEVERITY,
    ground_severity: int | None = None,
    **weather: float,
) -> A.Compose:
    """The full pipeline: weather, optics and the camera chain. **Needs CUDA.**

    One lighting state and one weather state per frame, then the camera's own
    degradations. Costs roughly 112 ms per 512 px frame on a current laptop
    GPU; see ``docs/pipelines.md`` for the breakdown and
    ``optimizations.md`` for where the time goes.

    Args:
        depth_provider: construct **one**
            :class:`~surmentations.depth.provider.SharedDepthProvider` and
            pass it here, so depth is estimated once per image rather than once
            per transform.
        day, lit_night, ir_night: relative weights of the three lighting
            states. ``lit_night`` is a colour frame under street lighting;
            ``ir_night`` is the camera cut to monochrome with its illuminator
            on. Neither *relights* the scene — see the caveat in the docs.
        codec: probability of re-encoding. Usually **0** for frames already
            decoded from video, where the artefacts are already in the pixels.
        faults: probability of packet loss damaging the frame — stale,
            smeared or green bands. Unset, it is 0.03, or 0 when ``codec``
            is 0: like the codec stage it needs ffmpeg, and ``codec=0.0``
            has to go on working without one.
        dirt, occluders: probability of grime and of insects/webs on the
            housing. Both rise steeply with maintenance interval.
        severity: level 1-5, from
            :data:`~surmentations.presets.severity.SEVERITY`. **This is
            the knob that decides how degraded a typical frame is**, and the
            first one to change if the output looks too heavy. 1 is barely
            visible, 2 an ordinary bad day, 5 a frame you would file a fault
            report about.
        weather_severity: level for rain, snow and fog, kept separate and
            higher — see :data:`DEFAULT_WEATHER_SEVERITY`.
        ground_severity: level 0-5 for how wet the road is under rain and
            how much snow has settled under snow, or ``None`` to draw it per
            frame over the whole range, which is the default. It is not
            tied to ``weather_severity``: see :func:`weather_state`.
        **weather: forwarded to :func:`weather_state`. The defaults put
            weather on 12% of frames, which is climatological rather than
            convenient: over 20 sample frames you will typically see one rain
            and no snow. Raise ``rain``/``snow``/``fog`` if you want weather
            coverage in training rather than a matched climate.

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
    }
    return A.Compose(
        [
            A.OneOf(
                [
                    A.Sequential(
                        [
                            _at(
                                CastShadows, "CastShadows", 0.30, severity,
                                depth_provider=depth_provider,
                            ),
                            weather_state(
                                depth_provider,
                                severity=weather_severity,
                                droplet_severity=severity,
                                night=False,
                                ground_severity=ground_severity,
                                **weather,
                            ),
                            _at(SunFlare, "SunFlare", 0.12, severity),
                            *_stream(lit_by_mains=False, infrared=False, severity=severity, **tail),
                        ],
                        p=day,
                    ),
                    A.Sequential(
                        [
                            weather_state(
                                depth_provider,
                                severity=weather_severity,
                                droplet_severity=severity,
                                night=True,
                                ground_severity=ground_severity,
                                **weather,
                            ),
                            # Night under street or shop lighting, camera still
                            # in colour: under-exposed, with the warm cast of
                            # sodium or low-CRI LED against a daylight AWB.
                            # Kept mild deliberately. This branch cannot
                            # relight the scene, so a strong version does not
                            # read as night — it reads as a sepia filter over
                            # a daylit frame, sky and all. Measured at
                            # -1.5 stops / 2900 K it put a 65-level red-blue
                            # cast on the frame, which dominated everything
                            # else the pipeline did. Under-exposure carries
                            # the impression; the tint only needs to hint.
                            ExposureTransient(
                                p=1.0,
                                exposure_error_stops_range=(-0.9, -0.35),
                                assumed_temperature_range_k=(4900.0, 5400.0),
                                actual_temperature_range_k=(3900.0, 4600.0),
                            ),
                            *_stream(lit_by_mains=True, infrared=False, severity=severity, **tail),
                        ],
                        p=lit_night,
                    ),
                    A.Sequential(
                        [
                            # Rain at night is a different transform and a
                            # different place in the chain, so this branch
                            # builds its own weather state rather than
                            # reusing `weather_state`. See `_infrared_night`.
                            _infrared_night(
                                depth_provider,
                                severity=severity,
                                weather_severity=weather_severity,
                                tail=tail,
                                ground_severity=ground_severity,
                                **weather,
                            ),
                        ],
                        p=ir_night,
                    ),
                ],
                p=1.0,
            )
        ]
    )


def surveillance_lite(
    *,
    night: float = 0.45,
    codec: float = 0.15,
    faults: float | None = None,
    dirt: float = 0.25,
    occluders: float = 0.12,
    droplets: float = 0.10,
    severity: int = DEFAULT_SEVERITY,
) -> A.Compose:
    """Everything that does not need depth. **CPU only, no GPU required.**

    The weather effects, cast shadows and IR night mode all condition on a
    monocular depth estimate, which costs a fixed ~46 ms and a CUDA device.
    Dropping them leaves the lens and the camera chain — which is most of what
    actually degrades a surveillance stream, and none of it needs a GPU.

    Measured at 512 px it is about **1.9x cheaper** than :func:`surveillance`
    (60 ms median against 112 ms) — a useful saving but not a transformative
    one, because the expensive lens and ISP effects are in both. Its real
    advantage is that it needs **no GPU at all**, so it runs in a
    ``DataLoader`` worker on any machine. Use it when you have no GPU to spare
    in the input pipeline, when your data already contains real weather, or as
    the cheap majority of a mixed pipeline.

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
        severity: level 1-5, as for :func:`surveillance`.
    """
    tail = {
        "dirt": dirt,
        "occluders": occluders,
        "codec": codec,
        "faults": _fault_rate(faults, codec),
    }
    return A.Compose(
        [
            A.OneOf(
                [
                    A.Sequential(
                        [
                            _at(WaterDroplets, "WaterDroplets", droplets, severity),
                            _at(SunFlare, "SunFlare", 0.12, severity),
                            *_stream(lit_by_mains=False, infrared=False, severity=severity, **tail),
                        ],
                        p=1.0 - night,
                    ),
                    # Deliberately *not* an under-exposed, warm-cast copy of
                    # the day branch. Without `NightIrMode` there is nothing
                    # here that can render night, and a colour shift on a
                    # daylit frame does not approximate one — measured, it put
                    # a 29-level median red-blue cast across the pipeline and
                    # dominated every other effect. All "night" means here is
                    # that the lighting is artificial, which raises the mains
                    # flicker and the denoiser's workload. Use `surveillance`
                    # if you need night as a domain.
                    A.Sequential(
                        [
                            _at(
                                WaterDroplets, "WaterDroplets", droplets, severity,
                                aperture_diameter_range_mm=_OPEN_IRIS_MM,
                            ),
                            *_stream(lit_by_mains=True, infrared=False, severity=severity, **tail),
                        ],
                        p=night,
                    ),
                ],
                p=1.0,
            )
        ]
    )
