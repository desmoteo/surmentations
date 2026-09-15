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

from weathermentations.atmosphere.fog import RandomFog
from weathermentations.atmosphere.rain import RandomRain
from weathermentations.atmosphere.snow import RandomSnow
from weathermentations.camera import (
    AcFlicker,
    ExposureTransient,
    NightIrMode,
    RollingShutter,
    TemporalDenoiseArtifacts,
    VideoCodecArtifacts,
    WdrToneMapping,
)
from weathermentations.depth.provider import SharedDepthProvider
from weathermentations.lens import DirtSoil, HousingOccluders, SunFlare, WaterDroplets
from weathermentations.presets.severity import SEVERITY
from weathermentations.scene import CastShadows

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


def _stream(
    *,
    lit_by_mains: bool,
    dirt: float,
    occluders: float,
    codec: float,
    severity: int,
) -> list[Any]:
    """Glass, sensor, ISP and transmission: independent draws, physical order.

    Written in order by hand rather than through
    :func:`~weathermentations.presets.pipeline.physical_order`, which sorts a
    flat list and cannot see inside the containers these pipelines are built
    from. ``tests/unit/test_pipeline_presets.py`` checks the order still holds.
    """
    return [
        # Optics: on the glass, in front of everything the sensor does.
        _at(DirtSoil, "DirtSoil", dirt, severity),
        _at(HousingOccluders, "HousingOccluders", occluders, severity),
        # Sensor. Mains banding needs artificial light, so outdoors it belongs
        # to the night branches; indoors it would apply all day.
        _at(AcFlicker, "AcFlicker", 0.45 if lit_by_mains else 0.05, severity),
        _at(RollingShutter, "RollingShutter", 0.50, severity),
        # ISP.
        _at(ExposureTransient, "ExposureTransient", 0.10, severity),
        _at(WdrToneMapping, "WdrToneMapping", 0.12, severity),
        _at(
            TemporalDenoiseArtifacts,
            "TemporalDenoiseArtifacts",
            0.30 if lit_by_mains else 0.12,
            severity,
        ),
        # Transmission, last: it acts on the finished picture.
        _at(VideoCodecArtifacts, "VideoCodecArtifacts", codec, severity),
    ]


def weather_state(
    depth_provider: SharedDepthProvider,
    *,
    rain: float = 0.06,
    snow: float = 0.03,
    fog: float = 0.03,
    wet_rain: float = 0.35,
    wet_snow: float = 0.10,
    wet_fog: float = 0.30,
    wet_clear: float = 0.06,
    severity: int = DEFAULT_WEATHER_SEVERITY,
) -> A.OneOf:
    """Exactly one atmospheric state per frame, including "clear".

    Args:
        depth_provider: one shared provider, passed to every depth-conditioned
            transform so depth is estimated once per image.
        rain, snow, fog: absolute probability of each state. The remainder is
            clear weather; they must sum to less than 1.
        wet_rain, wet_snow, wet_fog, wet_clear: ``P(water on the glass | state)``.
            ``wet_clear`` covers dew, residual drops after the rain stops and
            wind-blown spray, which is why it is not zero.
        severity: level 1-5 for the weather itself. Defaults to
            :data:`DEFAULT_WEATHER_SEVERITY`, which is higher than the rest of
            the pipeline on purpose — see that constant.

    Raises:
        ValueError: if the three weather probabilities sum to 1 or more.
    """
    total = rain + snow + fog
    if total >= 1.0:
        raise ValueError(
            f"rain + snow + fog must leave room for clear weather, got {total:.3f}. "
            "These are absolute probabilities, not relative weights."
        )
    return A.OneOf(
        [
            A.Sequential(
                [
                    # Wetness raised above the severity preset's default.
                    # In daylight a streak carries the sky's radiance, so
                    # against a bright sky it has almost no contrast --
                    # measured, lengthening the exposure 5x leaves the
                    # high-frequency energy it adds flat, because alpha =
                    # D/(vT) spreads the same light further. What actually
                    # says "rain" in a daylit frame is the wet road.
                    _at(
                        RandomRain, "RandomRain", 1.0, severity,
                        depth_provider=depth_provider, wetness_probability=0.85,
                    ),
                    _at(WaterDroplets, "WaterDroplets", wet_rain, severity),
                ],
                p=rain,
            ),
            A.Sequential(
                [
                    _at(RandomSnow, "RandomSnow", 1.0, severity, depth_provider=depth_provider),
                    _at(WaterDroplets, "WaterDroplets", wet_snow, severity),
                ],
                p=snow,
            ),
            A.Sequential(
                [
                    _at(RandomFog, "RandomFog", 1.0, severity, depth_provider=depth_provider),
                    _at(WaterDroplets, "WaterDroplets", wet_fog, severity),
                ],
                p=fog,
            ),
            A.Sequential(
                [_at(WaterDroplets, "WaterDroplets", wet_clear, severity)],
                p=1.0 - total,
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
    dirt: float = 0.25,
    occluders: float = 0.12,
    severity: int = DEFAULT_SEVERITY,
    weather_severity: int = DEFAULT_WEATHER_SEVERITY,
    **weather: float,
) -> A.Compose:
    """The full pipeline: weather, optics and the camera chain. **Needs CUDA.**

    One lighting state and one weather state per frame, then the camera's own
    degradations. Costs roughly 112 ms per 512 px frame on a current laptop
    GPU; see ``docs/pipelines.md`` for the breakdown and
    ``optimizations.md`` for where the time goes.

    Args:
        depth_provider: construct **one**
            :class:`~weathermentations.depth.provider.SharedDepthProvider` and
            pass it here, so depth is estimated once per image rather than once
            per transform.
        day, lit_night, ir_night: relative weights of the three lighting
            states. ``lit_night`` is a colour frame under street lighting;
            ``ir_night`` is the camera cut to monochrome with its illuminator
            on. Neither *relights* the scene — see the caveat in the docs.
        codec: probability of re-encoding. Usually **0** for frames already
            decoded from video, where the artefacts are already in the pixels.
        dirt, occluders: probability of grime and of insects/webs on the
            housing. Both rise steeply with maintenance interval.
        severity: level 1-5, from
            :data:`~weathermentations.presets.severity.SEVERITY`. **This is
            the knob that decides how degraded a typical frame is**, and the
            first one to change if the output looks too heavy. 1 is barely
            visible, 2 an ordinary bad day, 5 a frame you would file a fault
            report about.
        weather_severity: level for rain, snow and fog, kept separate and
            higher — see :data:`DEFAULT_WEATHER_SEVERITY`.
        **weather: forwarded to :func:`weather_state`. The defaults put
            weather on 12% of frames, which is climatological rather than
            convenient: over 20 sample frames you will typically see one rain
            and no snow. Raise ``rain``/``snow``/``fog`` if you want weather
            coverage in training rather than a matched climate.

    Example:
        >>> from weathermentations.depth import (
        ...     DepthAnythingV2Estimator, SharedDepthProvider)
        >>> from weathermentations.presets import surveillance
        >>> provider = SharedDepthProvider(DepthAnythingV2Estimator())
        >>> transform = surveillance(provider, codec=0.0)
        >>> augmented = transform(image=image)["image"]     # doctest: +SKIP
    """
    tail = {"dirt": dirt, "occluders": occluders, "codec": codec}
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
                            weather_state(depth_provider, severity=weather_severity, **weather),
                            _at(SunFlare, "SunFlare", 0.12, severity),
                            *_stream(lit_by_mains=False, severity=severity, **tail),
                        ],
                        p=day,
                    ),
                    A.Sequential(
                        [
                            weather_state(depth_provider, severity=weather_severity, **weather),
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
                            *_stream(lit_by_mains=True, severity=severity, **tail),
                        ],
                        p=lit_night,
                    ),
                    A.Sequential(
                        [
                            weather_state(depth_provider, severity=weather_severity, **weather),
                            # p=1.0 because the branch *is* the IR state, and
                            # its weight is what says how often that occurs. A
                            # camera that has cut to monochrome has done so
                            # completely; there is no 30% of an IR frame.
                            _at(
                                NightIrMode, "NightIrMode", 1.0, severity,
                                depth_provider=depth_provider,
                            ),
                            *_stream(lit_by_mains=True, severity=severity, **tail),
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
        dirt, occluders: grime and insects on the housing.
        droplets: water on the front element from dew, spray or rain this
            pipeline is not rendering.
        severity: level 1-5, as for :func:`surveillance`.
    """
    tail = {"dirt": dirt, "occluders": occluders, "codec": codec}
    return A.Compose(
        [
            A.OneOf(
                [
                    A.Sequential(
                        [
                            _at(WaterDroplets, "WaterDroplets", droplets, severity),
                            _at(SunFlare, "SunFlare", 0.12, severity),
                            *_stream(lit_by_mains=False, severity=severity, **tail),
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
                            _at(WaterDroplets, "WaterDroplets", droplets, severity),
                            *_stream(lit_by_mains=True, severity=severity, **tail),
                        ],
                        p=night,
                    ),
                ],
                p=1.0,
            )
        ]
    )
