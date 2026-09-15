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
from weathermentations.scene import CastShadows


def _stream(
    *,
    lit_by_mains: bool,
    dirt: float,
    occluders: float,
    codec: float,
) -> list[Any]:
    """Glass, sensor, ISP and transmission: independent draws, physical order.

    Written in order by hand rather than through
    :func:`~weathermentations.presets.pipeline.physical_order`, which sorts a
    flat list and cannot see inside the containers these pipelines are built
    from. ``tests/unit/test_pipeline_recipe.py`` checks the order still holds.
    """
    return [
        # Optics: on the glass, in front of everything the sensor does.
        DirtSoil(p=dirt),
        HousingOccluders(p=occluders),
        # Sensor. Mains banding needs artificial light, so outdoors it belongs
        # to the night branches; indoors it would apply all day.
        AcFlicker(p=0.45 if lit_by_mains else 0.05),
        RollingShutter(p=0.50),
        # ISP.
        ExposureTransient(p=0.10),
        WdrToneMapping(p=0.12),
        TemporalDenoiseArtifacts(p=0.30 if lit_by_mains else 0.12),
        # Transmission, last: it acts on the finished picture.
        VideoCodecArtifacts(p=codec),
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
                [RandomRain(depth_provider=depth_provider, p=1.0), WaterDroplets(p=wet_rain)],
                p=rain,
            ),
            A.Sequential(
                [RandomSnow(depth_provider=depth_provider, p=1.0), WaterDroplets(p=wet_snow)],
                p=snow,
            ),
            A.Sequential(
                [RandomFog(depth_provider=depth_provider, p=1.0), WaterDroplets(p=wet_fog)],
                p=fog,
            ),
            A.Sequential([WaterDroplets(p=wet_clear)], p=1.0 - total),
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
        **weather: forwarded to :func:`weather_state`.

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
                            CastShadows(depth_provider=depth_provider, p=0.30),
                            weather_state(depth_provider, **weather),
                            SunFlare(p=0.12),
                            *_stream(lit_by_mains=False, **tail),
                        ],
                        p=day,
                    ),
                    A.Sequential(
                        [
                            weather_state(depth_provider, **weather),
                            # Night under street or shop lighting, camera still
                            # in colour: under-exposed, with the warm cast of
                            # sodium or low-CRI LED against a daylight AWB.
                            ExposureTransient(
                                p=1.0,
                                exposure_error_stops_range=(-2.6, -1.0),
                                assumed_temperature_range_k=(5000.0, 6500.0),
                                actual_temperature_range_k=(2000.0, 3200.0),
                            ),
                            *_stream(lit_by_mains=True, **tail),
                        ],
                        p=lit_night,
                    ),
                    A.Sequential(
                        [
                            weather_state(depth_provider, **weather),
                            # p=1.0 because the branch *is* the IR state, and
                            # its weight is what says how often that occurs. A
                            # camera that has cut to monochrome has done so
                            # completely; there is no 30% of an IR frame.
                            NightIrMode(depth_provider=depth_provider, p=1.0),
                            *_stream(lit_by_mains=True, **tail),
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

    What it cannot do: no fog, rain or snow, no cast shadows, and no infrared
    night. The ``night`` fraction here only turns the mains-lit artefacts on;
    it does not darken the frame.

    Args:
        night: fraction of frames treated as artificially lit, which raises
            flicker banding and temporal-denoise artefacts.
        codec: probability of re-encoding; usually 0 for frames already
            decoded from video.
        dirt, occluders: grime and insects on the housing.
        droplets: water on the front element from dew, spray or rain this
            pipeline is not rendering.
    """
    tail = {"dirt": dirt, "occluders": occluders, "codec": codec}
    return A.Compose(
        [
            A.OneOf(
                [
                    A.Sequential(
                        [
                            WaterDroplets(p=droplets),
                            SunFlare(p=0.12),
                            *_stream(lit_by_mains=False, **tail),
                        ],
                        p=1.0 - night,
                    ),
                    A.Sequential(
                        [
                            WaterDroplets(p=droplets),
                            ExposureTransient(
                                p=1.0,
                                exposure_error_stops_range=(-2.6, -1.0),
                                assumed_temperature_range_k=(5000.0, 6500.0),
                                actual_temperature_range_k=(2000.0, 3200.0),
                            ),
                            *_stream(lit_by_mains=True, **tail),
                        ],
                        p=night,
                    ),
                ],
                p=1.0,
            )
        ]
    )
