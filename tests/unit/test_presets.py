"""The severity presets: every transform accepts its preset at every level, and levels order."""

import warnings

import numpy as np
import pytest

from surmentations.atmosphere.fog import RandomFog
from surmentations.atmosphere.ground_fog import GroundFog
from surmentations.atmosphere.heat_shimmer import HeatShimmer
from surmentations.atmosphere.night_fog import NightFog
from surmentations.atmosphere.night_insects import NightInsects
from surmentations.atmosphere.night_rain import NightRain
from surmentations.atmosphere.night_snow import NightSnow
from surmentations.atmosphere.plumes import LocalPlumes
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
from surmentations.camera.analog import AnalogVideoArtifacts
from surmentations.camera.codec import ffmpeg_path
from surmentations.camera.demosaic import DemosaicArtifacts
from surmentations.camera.ir_cut import IrCutFilterFault
from surmentations.camera.overlay import OnScreenOverlay
from surmentations.camera.vibration import CameraVibration
from surmentations.depth.estimator import DepthEstimator
from surmentations.depth.provider import SharedDepthProvider
from surmentations.lens import DirtSoil, HousingOccluders, SpiderWeb, SunFlare, WaterDroplets
from surmentations.lens.dome_glare import IrDomeGlare
from surmentations.lens.focus import FocusShift
from surmentations.lens.foliage import FoliageIntrusion
from surmentations.lens.frost import FrostOnGlass
from surmentations.lens.snow_cap import SnowOnHousing
from surmentations.lens.source_glare import SourceGlare
from surmentations.lens.tamper import LensTamper
from surmentations.presets.severity import SEVERITY
from surmentations.scene import CastShadows

_DEPTH_FREE = {
    "WaterDroplets": WaterDroplets,
    # Lit by the camera's own illuminator rather than by the scene, so unlike
    # every other atmospheric effect here it needs no depth.
    "NightRain": NightRain,
    "NightSnow": NightSnow,
    "DirtSoil": DirtSoil,
    "SunFlare": SunFlare,
    "SourceGlare": SourceGlare,
    "HousingOccluders": HousingOccluders,
    "OnScreenOverlay": OnScreenOverlay,
    "CameraVibration": CameraVibration,
    "FocusShift": FocusShift,
    "AnalogVideoArtifacts": AnalogVideoArtifacts,
    "DemosaicArtifacts": DemosaicArtifacts,
    "IrCutFilterFault": IrCutFilterFault,
    "FoliageIntrusion": FoliageIntrusion,
    "LensTamper": LensTamper,
    "SnowOnHousing": SnowOnHousing,
    "FrostOnGlass": FrostOnGlass,
    "IrDomeGlare": IrDomeGlare,
    "NightInsects": NightInsects,
    "SpiderWeb": SpiderWeb,
    "AcFlicker": AcFlicker,
    "RollingShutter": RollingShutter,
    "SensorNoise": SensorNoise,
    "ExposureTransient": ExposureTransient,
    "WdrToneMapping": WdrToneMapping,
    "TemporalDenoiseArtifacts": TemporalDenoiseArtifacts,
}
_DEPTH_CONDITIONED = {
    "RandomFog": RandomFog,
    "NightFog": NightFog,
    "RandomRain": RandomRain,
    "RandomSnow": RandomSnow,
    "NightIrMode": NightIrMode,
    "CastShadows": CastShadows,
    "GroundFog": GroundFog,
    "LocalPlumes": LocalPlumes,
    "HeatShimmer": HeatShimmer,
}
# Needs the ffmpeg binary, so it is covered separately and skipped without it.
_EXTERNAL_TOOL = {
    "VideoCodecArtifacts": VideoCodecArtifacts,
    "TransmissionFaults": TransmissionFaults,
}


class _LinearDepthEstimator(DepthEstimator):
    """Depth rising linearly down the frame."""
    def estimate(self, image: np.ndarray) -> np.ndarray:
        """Depth rising linearly from 0 at the top row to 1 at the bottom."""
        height, width = image.shape[:2]
        return np.tile(np.linspace(0.0, 1.0, height)[:, None], (1, width)).astype(np.float32)


@pytest.fixture
def scene() -> np.ndarray:
    """A flat blue-grey frame, as a fixture."""
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:, :] = (60, 90, 120)
    return img


@pytest.mark.parametrize("severity", [1, 2, 3, 4, 5])
def test_every_transform_accepts_its_preset_at_every_severity(
    severity: int, scene: np.ndarray
) -> None:
    """Albumentations only *warns* about unknown constructor arguments, so a
    preset referring to a renamed parameter would silently fall back to
    defaults. Promote that warning to an error here, or this test passes
    while the presets quietly do nothing."""
    warnings.simplefilter("error", UserWarning)
    kwargs_by_name = SEVERITY[severity]
    depth_provider = SharedDepthProvider(_LinearDepthEstimator())

    for name, cls in _DEPTH_FREE.items():
        transform = cls(p=1.0, **kwargs_by_name[name])
        out = transform(image=scene)["image"]
        assert out.shape == scene.shape

    for name, cls in _DEPTH_CONDITIONED.items():
        transform = cls(depth_provider=depth_provider, p=1.0, **kwargs_by_name[name])
        out = transform(image=scene)["image"]
        assert out.shape == scene.shape


def test_every_transform_present_at_every_severity() -> None:
    expected = set(_DEPTH_FREE) | set(_DEPTH_CONDITIONED) | set(_EXTERNAL_TOOL)
    for severity, kwargs_by_name in SEVERITY.items():
        assert set(kwargs_by_name) == expected, f"severity {severity} missing/extra transforms"


@pytest.mark.skipif(ffmpeg_path() is None, reason="VideoCodecArtifacts needs the ffmpeg binary")
@pytest.mark.parametrize("severity", [1, 2, 3, 4, 5])
def test_codec_preset_runs_at_every_severity(severity: int, scene: np.ndarray) -> None:
    warnings.simplefilter("error", UserWarning)
    for name, cls in _EXTERNAL_TOOL.items():
        transform = cls(p=1.0, **SEVERITY[severity][name])
        assert transform(image=scene)["image"].shape == scene.shape


def test_fog_severity_is_monotonically_denser() -> None:
    """Higher severity beta => lower transmission => output further from the
    original scene color on average, for a fixed depth field.

    Needs a non-uniform scene: with a flat single-color image, the
    auto-estimated airlight color equals the scene color itself, so fog
    would produce ~zero deviation at *any* beta (nothing to distinguish).
    """
    scene = np.zeros((64, 64, 3), dtype=np.uint8)
    scene[:, :] = (30, 60, 120)  # a distinct, non-sky-like color
    depth_provider = SharedDepthProvider(_LinearDepthEstimator())
    fixed_airlight = (220, 220, 220)
    deviations = []
    for severity in range(1, 6):
        beta_range = SEVERITY[severity]["RandomFog"]["beta_range"]
        t = RandomFog(
            depth_provider=depth_provider,
            p=1.0,
            beta_range=beta_range,
            atmospheric_light=fixed_airlight,
        )
        t.set_random_seed(0)
        out = t(image=scene)["image"].astype(np.float32)
        deviations.append(np.abs(out - scene.astype(np.float32)).mean())

    assert all(a <= b + 1e-6 for a, b in zip(deviations, deviations[1:], strict=False))
