"""Sensor, ISP and transmission effects — the camera rather than the weather.

These sit downstream of the optical effects in ``lens/`` and the atmospheric
ones in ``atmosphere/``: the physical chain is scene -> atmosphere -> optics
-> sensor -> ISP -> codec, and applying them out of that order produces
images that could not exist (codec blocking underneath lens blur, say).

By stage: the sensor (``NightIrMode`` and its filter stuck open,
``IrCutFilterFault``; ``AcFlicker``, ``RollingShutter``,
``CameraVibration``, ``SensorNoise``, ``DemosaicArtifacts``); the processor
(``ExposureTransient``, ``WdrToneMapping``, ``TemporalDenoiseArtifacts``);
the link (``VideoCodecArtifacts`` or ``AnalogVideoArtifacts``, then
``TransmissionFaults``); and ``OnScreenOverlay``, drawn by the camera
before the link or by the recorder after it.
"""

from weathermentations.camera.analog import AnalogVideoArtifacts
from weathermentations.camera.codec import VideoCodecArtifacts
from weathermentations.camera.demosaic import DemosaicArtifacts
from weathermentations.camera.denoise import TemporalDenoiseArtifacts
from weathermentations.camera.exposure import ExposureTransient
from weathermentations.camera.flicker import AcFlicker
from weathermentations.camera.ir_cut import IrCutFilterFault
from weathermentations.camera.night_ir import NightIrMode
from weathermentations.camera.overlay import OnScreenOverlay
from weathermentations.camera.rolling_shutter import RollingShutter
from weathermentations.camera.sensor_noise import SensorNoise
from weathermentations.camera.tone_mapping import WdrToneMapping
from weathermentations.camera.transmission import TransmissionFaults
from weathermentations.camera.vibration import CameraVibration

__all__ = [
    "AcFlicker",
    "AnalogVideoArtifacts",
    "CameraVibration",
    "DemosaicArtifacts",
    "ExposureTransient",
    "IrCutFilterFault",
    "NightIrMode",
    "OnScreenOverlay",
    "RollingShutter",
    "SensorNoise",
    "TemporalDenoiseArtifacts",
    "TransmissionFaults",
    "VideoCodecArtifacts",
    "WdrToneMapping",
]
