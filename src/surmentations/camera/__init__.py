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

from surmentations.camera.analog import AnalogVideoArtifacts
from surmentations.camera.codec import VideoCodecArtifacts
from surmentations.camera.demosaic import DemosaicArtifacts
from surmentations.camera.denoise import TemporalDenoiseArtifacts
from surmentations.camera.exposure import ExposureTransient
from surmentations.camera.flicker import AcFlicker
from surmentations.camera.ir_cut import IrCutFilterFault
from surmentations.camera.night_ir import NightIrMode
from surmentations.camera.overlay import OnScreenOverlay
from surmentations.camera.rolling_shutter import RollingShutter
from surmentations.camera.sensor_noise import SensorNoise
from surmentations.camera.tone_mapping import WdrToneMapping
from surmentations.camera.transmission import TransmissionFaults
from surmentations.camera.vibration import CameraVibration

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
