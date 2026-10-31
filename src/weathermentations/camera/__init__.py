"""Sensor, ISP and transmission effects — the camera rather than the weather.

These sit downstream of the optical effects in ``lens/`` and the atmospheric
ones in ``atmosphere/``: the physical chain is scene -> atmosphere -> optics
-> sensor -> ISP -> codec, and applying them out of that order produces
images that could not exist (codec blocking underneath lens blur, say).
"""

from weathermentations.camera.codec import VideoCodecArtifacts
from weathermentations.camera.denoise import TemporalDenoiseArtifacts
from weathermentations.camera.exposure import ExposureTransient
from weathermentations.camera.flicker import AcFlicker
from weathermentations.camera.night_ir import NightIrMode
from weathermentations.camera.rolling_shutter import RollingShutter
from weathermentations.camera.sensor_noise import SensorNoise
from weathermentations.camera.tone_mapping import WdrToneMapping

__all__ = [
    "AcFlicker",
    "ExposureTransient",
    "NightIrMode",
    "RollingShutter",
    "SensorNoise",
    "TemporalDenoiseArtifacts",
    "VideoCodecArtifacts",
    "WdrToneMapping",
]
