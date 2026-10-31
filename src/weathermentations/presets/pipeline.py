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
    "SunFlare": 3,
    # Sensor: exposure, readout and their artefacts.
    "NightIrMode": 4,
    # Rain lit by the camera's own illuminator, so it has to follow the IR
    # conversion rather than precede it: the streaks are already in infrared
    # terms and running them through the scene's reflectance-to-IR conversion
    # would convert them twice.
    "NightRain": 5,
    # Likewise lit by the illuminator, and for the same reason.
    "NightSnow": 5,
    "AcFlicker": 6,
    "RollingShutter": 6,
    "SensorNoise": 6,
    # ISP: everything the processor does to the raw frame.
    "ExposureTransient": 7,
    "WdrToneMapping": 8,
    "TemporalDenoiseArtifacts": 9,
    # Transmission: last, because it acts on the finished picture.
    "VideoCodecArtifacts": 10,
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
