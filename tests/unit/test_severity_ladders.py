"""Every severity ladder must actually order its effect.

A ladder whose rungs do not increase is worse than no ladder: callers select
a severity expecting "more", and silently get noise. Two ladders had already
broken this way before the property was tested — rain's coverage peaked at
level 3 and fell, and SunFlare's level 1 moved the frame by a third of an
8-bit level, i.e. not at all.

Two design notes, both learned by getting them wrong first:

**Enough seeds.** Several of these effects draw a *signed* or conditional
quantity — exposure error is symmetric about zero, flicker banding cancels
outright when the exposure lands on a whole number of flicker periods. Over
three seeds they look non-monotonic; over a dozen they are clean. The count
here is chosen so the ladders are measured rather than sampled.

**The right metric.** `NightIrMode` is normalised by its own auto-exposure, so
no whole-frame difference can see its severity at all — measured, it runs
*backwards* (72, 69, 66, 64, 66). Its severity means "less light to work
with", which shows up as the noise floor, so that is what is measured. Using
the generic metric there would have pinned a number that means nothing.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import pytest

from weathermentations.atmosphere.fog import RandomFog
from weathermentations.atmosphere.night_fog import NightFog
from weathermentations.atmosphere.night_rain import NightRain
from weathermentations.atmosphere.night_snow import NightSnow
from weathermentations.atmosphere.rain import RandomRain
from weathermentations.atmosphere.snow import RandomSnow
from weathermentations.camera import (
    AcFlicker,
    ExposureTransient,
    NightIrMode,
    RollingShutter,
    SensorNoise,
    TemporalDenoiseArtifacts,
    WdrToneMapping,
)
from weathermentations.depth.estimator import DepthEstimator
from weathermentations.depth.provider import SharedDepthProvider
from weathermentations.lens import DirtSoil, HousingOccluders, SpiderWeb, SunFlare, WaterDroplets
from weathermentations.presets.severity import SEVERITY
from weathermentations.scene import CastShadows

SEEDS = 12
SIZE = 320


class _GroundPlaneDepth(DepthEstimator):
    """Level camera above a horizontal plane, sky above the horizon."""

    def estimate(self, image: np.ndarray) -> np.ndarray:
        """Depth of level ground seen from 1.5 m up, horizon at mid-frame, 50 m above it.

        Each row's distance follows from the camera height and the focal
        length of a 60-degree field of view, as a real road's would.
        """
        height, width = image.shape[:2]
        focal = width / (2.0 * np.tan(np.deg2rad(60.0) / 2.0))
        cy = height / 2.0
        rows = np.arange(height, dtype=np.float32).reshape(height, 1)
        depth = np.where(rows > cy + 2, 1.5 * focal / np.maximum(rows - cy, 1e-3), 50.0)
        return np.broadcast_to(np.minimum(depth, 50.0) / 50.0, (height, width)).astype(np.float32)


REFERENCE_IMAGE = Path(__file__).resolve().parents[2] / "docs" / "examples" / "reference.webp"


def _scene() -> np.ndarray:
    """The real reference photograph, downscaled.

    Deliberately a photograph rather than the synthetic scene the golden tests
    use. Severity is being calibrated here, and several of these effects have
    a strength that genuinely depends on scene content — `DirtSoil` scatters
    light rather than tinting it, so it is near-invisible in flat light by
    design (its own unit test pins exactly that). Measured at severity 1 it
    moves a synthetic sky-over-ground scene by 0.75 levels and a real
    photograph by 1.79, without either being wrong.

    Calibrating against a synthetic scene would therefore tune the presets for
    an image nobody applies them to. The golden tests keep their synthetic
    scene, where the requirement is reproducibility rather than realism.
    """
    image = cv2.imread(str(REFERENCE_IMAGE))
    if image is None:  # pragma: no cover - the asset ships with the repo
        pytest.skip(f"reference image not found at {REFERENCE_IMAGE}")
    image = cv2.resize(image, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _provider() -> SharedDepthProvider:
    """A depth provider over the ground plane."""
    return SharedDepthProvider(_GroundPlaneDepth())


# name -> builder taking the preset kwargs.
BUILDERS: dict[str, Callable[[dict], object]] = {
    "WaterDroplets": lambda k: WaterDroplets(p=1.0, **k),
    "DirtSoil": lambda k: DirtSoil(p=1.0, **k),
    # Fixed position: a randomly placed source is a large extra variance that
    # severity does not control.
    "SunFlare": lambda k: SunFlare(p=1.0, flare_position=(0.75, 0.15), **k),
    "HousingOccluders": lambda k: HousingOccluders(p=1.0, **k),
    # Lit by the camera like the night rain and snow below, and measured on
    # the daylit reference for their reason: its light adds whatever is
    # behind it.
    "SpiderWeb": lambda k: SpiderWeb(p=1.0, **k),
    "AcFlicker": lambda k: AcFlicker(p=1.0, **k),
    "RollingShutter": lambda k: RollingShutter(p=1.0, **k),
    "SensorNoise": lambda k: SensorNoise(p=1.0, **k),
    "ExposureTransient": lambda k: ExposureTransient(p=1.0, **k),
    "WdrToneMapping": lambda k: WdrToneMapping(p=1.0, **k),
    "TemporalDenoiseArtifacts": lambda k: TemporalDenoiseArtifacts(p=1.0, **k),
    "RandomFog": lambda k: RandomFog(depth_provider=_provider(), p=1.0, **k),
    # SourceGlare is absent on purpose: it acts only on lights in the frame,
    # and this photograph has none. Its ladder is in test_night_lights.py.
    "NightFog": lambda k: NightFog(depth_provider=_provider(), p=1.0, **k),
    "RandomRain": lambda k: RandomRain(depth_provider=_provider(), p=1.0, **k),
    "RandomSnow": lambda k: RandomSnow(
        depth_provider=_provider(), p=1.0, accumulation_probability=1.0, **k
    ),
    "NightIrMode": lambda k: NightIrMode(depth_provider=_provider(), p=1.0, **k),
    # Needs no depth: the drops are lit by the camera, not the scene. Measured
    # on the daylit reference like everything else here — its streaks add
    # light whatever the background, so the ladder is meaningful even though
    # the effect only *looks* right over a night frame.
    "NightRain": lambda k: NightRain(p=1.0, **k),
    "NightSnow": lambda k: NightSnow(p=1.0, **k),
    "CastShadows": lambda k: CastShadows(depth_provider=_provider(), p=1.0, **k),
}


def _mean_change(out: np.ndarray, clean: np.ndarray) -> float:
    """Generic strength: how far the frame moved."""
    return float(np.abs(out - clean).mean())


def _noise_floor(out: np.ndarray, clean: np.ndarray) -> float:
    """Night mode's severity is 'less light', which reads as noise, not level.

    Its auto-exposure deliberately normalises the level, so `_mean_change`
    cannot see severity at all and in fact runs backwards.
    """
    del clean
    return float(np.abs(cv2.Laplacian(out.mean(axis=2), cv2.CV_32F)).mean())


METRICS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "NightIrMode": _noise_floor,
}

# VideoCodecArtifacts is covered by its own CRF-monotonicity test in
# test_camera.py; two ffmpeg subprocesses per render would make this suite
# minutes long for no extra coverage.


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_severity_is_monotonic(name: str) -> None:
    scene = _scene()
    clean = scene.astype(np.float32)
    metric = METRICS.get(name, _mean_change)

    strengths = []
    for level in sorted(SEVERITY):
        transform = BUILDERS[name](SEVERITY[level][name])
        samples = []
        for seed in range(SEEDS):
            transform.set_random_seed(seed)
            out = transform(image=scene)["image"].astype(np.float32)
            samples.append(metric(out, clean))
        strengths.append(float(np.mean(samples)))

    assert all(a < b for a, b in zip(strengths, strengths[1:], strict=False)), (
        f"{name} severity is not monotonic: {[round(s, 2) for s in strengths]}"
    )


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_severity_one_is_mild_not_absent(name: str) -> None:
    """The bottom rung must still do something.

    Three effects have already shipped a severity 1 below one 8-bit level —
    SunFlare at 0.33, DirtSoil at 0.94, TemporalDenoiseArtifacts at 0.71. A
    rung that changes nothing is indistinguishable from a bug, and wastes a
    fifth of the ladder.
    """
    scene = _scene()
    clean = scene.astype(np.float32)

    transform = BUILDERS[name](SEVERITY[1][name])
    samples = []
    for seed in range(SEEDS):
        transform.set_random_seed(seed)
        out = transform(image=scene)["image"].astype(np.float32)
        samples.append(_mean_change(out, clean))

    assert np.mean(samples) > 1.0, (
        f"{name} at severity 1 moves the frame by {np.mean(samples):.2f} 8-bit levels, "
        f"which is at or below the quantisation it is rendered into"
    )
