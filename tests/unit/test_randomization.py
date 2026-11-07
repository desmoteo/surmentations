"""Every transform must actually randomize per call.

An augmentation whose physical parameters are fixed scalars applies the *same*
weather to every image in a dataset. That is not a visual defect you would
notice in one render — each frame looks fine — it is a silent loss of the
diversity the augmentation exists to provide, and it only shows up as a model
that failed to generalise.

So these are guards on the property rather than on any one value: two seeds
must give materially different parameters and materially different pixels, and
the physical knobs must be ranges rather than scalars.
"""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np
import pytest

from weathermentations.atmosphere.fog import RandomFog
from weathermentations.atmosphere.night_fog import NightFog
from weathermentations.atmosphere.night_insects import NightInsects
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
from weathermentations.camera.analog import AnalogVideoArtifacts
from weathermentations.camera.demosaic import DemosaicArtifacts
from weathermentations.camera.ir_cut import IrCutFilterFault
from weathermentations.depth.estimator import DepthEstimator
from weathermentations.depth.provider import SharedDepthProvider
from weathermentations.lens import DirtSoil, HousingOccluders, SpiderWeb, SunFlare, WaterDroplets
from weathermentations.lens.dome_glare import IrDomeGlare
from weathermentations.lens.foliage import FoliageIntrusion
from weathermentations.lens.frost import FrostOnGlass
from weathermentations.lens.snow_cap import SnowOnHousing
from weathermentations.lens.tamper import LensTamper
from weathermentations.scene import CastShadows

# Scalars that are legitimately *not* ranges, with the reason each one is
# exempt. Anything else scalar is a bug this test exists to catch.
ALLOWED_SCALARS = {
    # Probabilities are sampled against once per call; they are the knob, not
    # a quantity that itself needs a range.
    "trail_probability",
    "streak_probability",
    "wetness_probability",
    "accumulation_probability",
    "dispersion_ring_probability",
    "sensor_smear_probability",
    "ir_leak_probability",
    "window_ghost_probability",
    "spider_probability",
    # Render budget / quality, not physics. Varying these would make cost and
    # fidelity random rather than the weather.
    "max_streaks",
    "max_flakes",
    "max_discs",
    "min_streak_width_px",
    "min_flake_width_px",
    "flake_count_scale",
    # Selects which drop-size distribution family is used (0 = Marshall &
    # Palmer's exponential, 2-4 = Ulbrich's gamma), not a quantity drawn per
    # frame. Real spectra do vary in shape between convective and stratiform
    # rain, so a range would not be meaningless — but mixing distribution
    # families inside one dataset is a modelling decision a caller should make
    # deliberately rather than have sampled for them.
    "dsd_shape_mu",
    # Fixed model constants documented as such in their modules.
    "min_gap_ratio",
    "definition_softness",
    "sky_optical_depth_cap",
    "residual_tint",
    "surface_sharpness",
    # Properties of the installation rather than quantities that vary shot to
    # shot: a given camera has one mains frequency, one mount angle, one lens.
    "mains_frequency_hz",
    "camera_pitch_deg",
    "horizontal_fov_deg",
}


class _GroundPlaneDepth(DepthEstimator):
    """A level camera above a horizontal ground plane receding to a horizon.

    Deliberately not a top-to-bottom ramp. Real ground recedes *upward* toward
    the horizon, so a ramp with its far end at the bottom of the frame
    back-projects to a ceiling — which correctly collects no snow and catches
    no shadow, and would make those transforms silently do nothing here.
    """

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


def _scene() -> np.ndarray:
    """Structured, non-uniform scene: a flat one has no illuminant to read."""
    rng = np.random.default_rng(0)
    image = np.zeros((360, 480, 3), dtype=np.uint8)
    image[:130] = (200, 210, 225)  # sky
    image[130:] = (60, 62, 58)  # ground
    image[200:260, 100:300] = (120, 40, 40)  # something with contrast
    noise = rng.integers(-8, 9, image.shape, dtype=np.int16)
    return np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def _transforms() -> dict[str, Any]:
    """Every transform at its defaults, sharing one depth provider."""
    provider = SharedDepthProvider(_GroundPlaneDepth())
    return {
        "WaterDroplets": WaterDroplets(p=1.0),
        "DirtSoil": DirtSoil(p=1.0),
        "SunFlare": SunFlare(p=1.0),
        "RandomFog": RandomFog(depth_provider=provider, p=1.0),
        "NightFog": NightFog(depth_provider=provider, p=1.0),
        "RandomRain": RandomRain(depth_provider=provider, p=1.0),
        "RandomSnow": RandomSnow(depth_provider=provider, p=1.0, accumulation_probability=1.0),
        # Surveillance-stream effects. VideoCodecArtifacts is excluded: it
        # shells out to ffmpeg, so running it eight times per test for three
        # tests is far too slow, and it has no sampled physical quantity of
        # its own beyond the CRF.
        "AcFlicker": AcFlicker(p=1.0),
        "RollingShutter": RollingShutter(p=1.0),
        "SensorNoise": SensorNoise(p=1.0),
        "ExposureTransient": ExposureTransient(p=1.0),
        "WdrToneMapping": WdrToneMapping(p=1.0),
        "TemporalDenoiseArtifacts": TemporalDenoiseArtifacts(p=1.0),
        "NightIrMode": NightIrMode(depth_provider=provider, p=1.0),
        "HousingOccluders": HousingOccluders(p=1.0, insect_count_range=(2, 5)),
        "AnalogVideoArtifacts": AnalogVideoArtifacts(p=1.0),
        "DemosaicArtifacts": DemosaicArtifacts(p=1.0),
        "IrCutFilterFault": IrCutFilterFault(p=1.0),
        "FoliageIntrusion": FoliageIntrusion(p=1.0),
        "LensTamper": LensTamper(p=1.0),
        "SnowOnHousing": SnowOnHousing(p=1.0),
        "FrostOnGlass": FrostOnGlass(p=1.0),
        "IrDomeGlare": IrDomeGlare(p=1.0),
        "NightInsects": NightInsects(p=1.0),
        "SpiderWeb": SpiderWeb(p=1.0),
        "CastShadows": CastShadows(depth_provider=provider, p=1.0),
    }


@pytest.mark.parametrize("name", sorted(_transforms()))
def test_physical_parameters_are_ranges_not_scalars(name: str) -> None:
    """A physical quantity exposed as a bare number cannot vary per call."""
    transform = _transforms()[name]
    signature = inspect.signature(type(transform).__init__)

    offenders = []
    for param, spec in signature.parameters.items():
        if param in {"self", "p", "depth_provider"} or param in ALLOWED_SCALARS:
            continue
        default = spec.default
        if isinstance(default, bool) or default is inspect.Parameter.empty:
            continue
        if isinstance(default, (int, float)):
            offenders.append(f"{param}={default}")

    assert not offenders, (
        f"{name} exposes physical quantities as fixed scalars, so they cannot "
        f"vary per call: {offenders}. Make them (min, max) ranges, or add them "
        f"to ALLOWED_SCALARS with a reason."
    )


@pytest.mark.parametrize("name", sorted(_transforms()))
def test_output_differs_across_seeds(name: str) -> None:
    """Two seeds must produce visibly different images, not the same render."""
    transform = _transforms()[name]
    scene = _scene()

    outputs = []
    for seed in range(4):
        transform.set_random_seed(seed)
        outputs.append(transform(image=scene)["image"].astype(np.float32))

    for index, other in enumerate(outputs[1:], start=1):
        difference = np.abs(outputs[0] - other).mean()
        assert difference > 0.25, (
            f"{name}: seeds 0 and {index} produced near-identical output "
            f"(mean abs diff {difference:.4f})"
        )


@pytest.mark.parametrize("name", sorted(_transforms()))
def test_effect_strength_varies_across_seeds(name: str) -> None:
    """The *amount* of effect must vary, not just its placement.

    Two calls could differ pixel-wise while always applying exactly the same
    strength — e.g. droplets in different spots but always the same number of
    them. Comparing each render against the clean scene catches that: the
    spread of those distances is what says the intensity itself is being drawn.
    """
    transform = _transforms()[name]
    scene = _scene()
    clean = scene.astype(np.float32)

    strengths = []
    for seed in range(8):
        transform.set_random_seed(seed)
        out = transform(image=scene)["image"].astype(np.float32)
        strengths.append(float(np.abs(out - clean).mean()))

    spread = np.std(strengths) / max(np.mean(strengths), 1e-6)
    assert spread > 0.05, (
        f"{name}: effect strength is nearly constant across seeds "
        f"(relative spread {spread:.4f}, values {[round(s, 2) for s in strengths]})"
    )
