"""Composition helper that keeps effects in physical order.

Why this exists
---------------
Augmentation stacks usually apply their effects in whatever order they were
written down. For effects that model *physical* processes that is a bug, not a
style choice: each stage of the imaging chain acts on the output of the one
before it, so applying them out of order produces images that could not exist.

Concretely — codec blocking underneath lens blur means the lens blurred an
image that had already been compressed, which is backwards; sensor noise
applied before an atmospheric veil means the veil attenuated noise it should
have sat on top of; a lens flare added after tone mapping was never subject to
the tone curve that shaped the rest of the frame.

The chain is:

    scene -> atmosphere -> optics -> sensor -> ISP -> codec

so this module sorts a set of transforms into it. A caller supplies whichever
effects they want and gets them back ordered, rather than having to remember
the chain themselves.
"""

from __future__ import annotations

from typing import Any

# Stage index per transform. Lower runs first.
_STAGE_ORDER: dict[str, int] = {
    # Scene: things that are in the world, in front of everything else.
    "CastShadows": 0,
    # Atmosphere: between the scene and the lens.
    "RandomFog": 1,
    "RandomRain": 1,
    "RandomSnow": 1,
    # Optics: on or in the lens.
    "WaterDroplets": 2,
    "DirtSoil": 2,
    "HousingOccluders": 2,
    # More that sits on the housing or hangs in front of it.
    "FrostOnGlass": 2,
    "SnowOnHousing": 2,
    "LensTamper": 2,
    "FoliageIntrusion": 2,
    "SunFlare": 3,
    # The same optics, on the lights that are in the frame.
    "SourceGlare": 3,
    # Sensor: exposure, readout and their artefacts.
    "NightIrMode": 4,
    # The same filter, stuck open by day.
    "IrCutFilterFault": 4,
    # Rain lit by the camera's own illuminator, so it has to follow the IR
    # conversion rather than precede it: the streaks are already in infrared
    # terms and running them through the scene's reflectance-to-IR conversion
    # would convert them twice.
    "NightRain": 5,
    # Likewise lit by the illuminator, and for the same reason.
    "NightSnow": 5,
    # Fog lit by lamps and by the illuminator. After the infrared conversion
    # for the same reason as the two above: its backscatter is the camera's
    # own light.
    "NightFog": 5,
    # A web across the housing, which shows only by the illuminator's light:
    # in front of everything, and still after the infrared conversion, which
    # would otherwise light it by the depth of the scene behind it.
    "SpiderWeb": 5,
    # Insects and dust in the beam, and the beam's own reflection in the
    # dome: the camera's light again, so after the conversion.
    "NightInsects": 5,
    "IrDomeGlare": 5,
    "AcFlicker": 6,
    "RollingShutter": 6,
    "SensorNoise": 6,
    # The mosaic, and the interpolation that follows it.
    "DemosaicArtifacts": 6,
    # ISP: everything the processor does to the raw frame.
    "ExposureTransient": 7,
    "WdrToneMapping": 8,
    "TemporalDenoiseArtifacts": 9,
    # Transmission: last, because it acts on the finished picture.
    "VideoCodecArtifacts": 10,
    # An analog link, for a camera that has no codec.
    "AnalogVideoArtifacts": 10,
    # What the link then loses of the encoded stream.
    "TransmissionFaults": 11,
}

# Anything unknown sorts here — after the optics, before the ISP. A neutral
# guess, and `physical_order` is stable, so unknown transforms keep the
# relative order the caller gave them.
_DEFAULT_STAGE = 3


def stage_of(transform: Any) -> int:
    """Pipeline stage index for a transform instance or class."""
    name = transform.__name__ if isinstance(transform, type) else type(transform).__name__
    return _STAGE_ORDER.get(name, _DEFAULT_STAGE)


def physical_order(transforms: list[Any]) -> list[Any]:
    """Sort ``transforms`` into the order the imaging chain applies them.

    Stable, so transforms sharing a stage — two lens effects, say — keep the
    order they were given in, which is the right default when physics does not
    prefer one over the other.

    Note this orders a *list*; it does not build the ``Compose``. Callers keep
    control of probabilities, seeding and everything else Albumentations
    offers::

        A.Compose(physical_order([
            VideoCodecArtifacts(p=0.4),
            RandomRain(depth_provider=provider, p=0.3),
            DirtSoil(p=0.2),
        ]))
    """
    return sorted(transforms, key=stage_of)
