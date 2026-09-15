"""Mains-frequency banding from artificial lighting.

Physical model
--------------
A lamp on an AC supply does not emit steadily. Incandescent and (especially)
gas-discharge lamps dim twice per mains cycle as the current passes through
zero, so the illumination pulses at **twice** the mains frequency: 100 Hz on
a 50 Hz supply, 120 Hz on 60 Hz.

A rolling-shutter sensor reads its rows out one after another over the
readout interval, so each row integrates a different part of that pulse.
The frame therefore carries a *spatial* record of a temporal oscillation —
horizontal bands, with

    bands across the frame = 2 * f_mains * readout_time

Two consequences are worth getting right rather than drawing bands by eye:

* **The exposure time sets the band depth.** A row integrates the light over
  its exposure, so the modulation it records is the pulse averaged over that
  window — a ``sinc`` in the ratio of exposure to flicker period. When the
  exposure is an exact multiple of the flicker period the sinc is zero and
  the banding **vanishes completely**. That is not a curiosity; it is why
  cameras have a "50 Hz / 60 Hz" anti-flicker setting, and it means the
  effect must disappear at those exposures rather than merely getting
  weaker.
* **It is a change in illumination**, so it multiplies radiance and has to be
  applied in linear light. Applied to gamma-encoded values it lifts the
  mid-tones wrongly, exactly as with every other light-adding effect here.

Single-frame scope: on video the bands *drift* vertically frame to frame, as
the flicker phase advances against the frame period. Per frame that drift is
just the phase, which is drawn randomly here — so this effect loses nothing
by being single-frame, unlike the temporal artefacts in ``denoise.py``.
"""

from __future__ import annotations

from typing import Annotated, Any

import numpy as np
from albumentations.core.pydantic import check_range_bounds, nondecreasing
from albumentations.core.transforms_interface import BaseTransformInitSchema
from pydantic import AfterValidator, Field

from weathermentations.base import WeatherTransform
from weathermentations.diagnostics import record_draw
from weathermentations.utils.photometry import from_linear_unit, to_linear_unit


def flicker_modulation_depth(
    nominal_depth: float, exposure_time_s: float, flicker_frequency_hz: float
) -> float:
    """Banding depth actually recorded, after the exposure averages the pulse.

    A row integrates the illumination over its exposure time, which is a box
    filter over the flicker waveform; in the frequency domain that is a
    ``sinc``. At an exposure of exactly one flicker period (or any multiple)
    the sinc is zero and the banding disappears — the physical basis of a
    camera's anti-flicker shutter setting.
    """
    cycles = exposure_time_s * flicker_frequency_hz
    if cycles < 1e-9:
        return nominal_depth
    return float(nominal_depth * abs(np.sinc(cycles)))


class AcFlicker(WeatherTransform):
    """Horizontal banding from AC-powered lighting beating against the shutter.

    Args:
        mains_frequency_hz: supply frequency. 50 Hz across most of the world,
            60 Hz in the Americas and parts of Asia. The lighting pulses at
            twice this.
        readout_time_range_s: time the sensor takes to read all rows out.
            Together with the flicker frequency this fixes how many bands
            cross the frame.
        exposure_time_range_s: per-row exposure. Sets the band *depth*
            through the sinc term — an exposure that is a whole number of
            flicker periods cancels the banding entirely, which is exactly
            what a camera's "anti-flicker" setting arranges. **Sampled
            log-uniformly**, because exposure is a logarithmic quantity:
            photographers count it in stops, and the interval 1/1000-1/500 s
            is as large a change as 1/8-1/4 s. Drawing it uniformly puts most
            of the mass at the long end, which here is the end the sinc
            cancels — median |sinc| 0.22 uniform against 0.76 log-uniform
            over the default range, i.e. most frames came out with no visible
            banding at all.
        modulation_depth_range: peak-to-mean modulation of the lamp itself,
            before the exposure averages it. Gas-discharge lamps without
            smoothing ballast approach 1; an incandescent filament has thermal
            inertia and modulates far less.

    Targets:
        image

    Image types:
        uint8, float32
    """

    class InitSchema(BaseTransformInitSchema):
        mains_frequency_hz: float = Field(gt=0.0, le=400.0)
        readout_time_range_s: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(1e-4, 1.0)),
        ]
        exposure_time_range_s: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(1e-6, 1.0)),
        ]
        modulation_depth_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 1.0)),
        ]

    def __init__(
        self,
        mains_frequency_hz: float = 50.0,
        readout_time_range_s: tuple[float, float] = (0.012, 0.033),
        exposure_time_range_s: tuple[float, float] = (0.001, 0.016),
        modulation_depth_range: tuple[float, float] = (0.05, 0.35),
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.mains_frequency_hz = mains_frequency_hz
        self.readout_time_range_s = readout_time_range_s
        self.exposure_time_range_s = exposure_time_range_s
        self.modulation_depth_range = modulation_depth_range

    def get_params_dependent_on_data(
        self, params: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, Any]:
        del data
        readout = self.py_random.uniform(*self.readout_time_range_s)
        # Log-uniform: see exposure_time_range_s. Readout time stays uniform -
        # it is a fixed property of the sensor over a narrow range, not an
        # exposure setting that moves in stops.
        low, high = self.exposure_time_range_s
        exposure = float(
            np.exp(self.py_random.uniform(np.log(max(low, 1e-9)), np.log(max(high, 1e-9))))
        )
        nominal = self.py_random.uniform(*self.modulation_depth_range)

        flicker_hz = 2.0 * self.mains_frequency_hz
        record_draw(
            readout_s=readout,
            exposure_s=exposure,
            nominal_depth=nominal,
            flicker_hz=flicker_hz,
            exposure_in_flicker_periods=exposure * flicker_hz,
        )
        return {
            # Bands across the frame: how many flicker cycles elapse while the
            # sensor is reading rows out.
            "cycles_per_frame": flicker_hz * readout,
            "depth": flicker_modulation_depth(nominal, exposure, flicker_hz),
            # On video this advances frame to frame; per frame it is a draw.
            "phase": self.py_random.uniform(0.0, 2.0 * np.pi),
        }

    def apply(
        self,
        img: np.ndarray,
        cycles_per_frame: float,
        depth: float,
        phase: float,
        **params: Any,
    ) -> np.ndarray:
        if depth < 1e-4:
            return img

        height = img.shape[0]
        rows = np.arange(height, dtype=np.float32) / max(height, 1)
        gain = 1.0 + depth * np.sin(2.0 * np.pi * cycles_per_frame * rows + phase)

        linear, scale, is_integer = to_linear_unit(img)
        linear = linear * gain[:, None, None]
        return from_linear_unit(linear, scale, is_integer, img.dtype)
