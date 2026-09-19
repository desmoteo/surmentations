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

Three consequences are worth getting right rather than drawing bands by eye:

* **The band count is what makes it legible.** Two broad bands across a frame
  is a smooth luminance gradient, and the eye discounts those as an exposure
  difference no matter how deep they are — a 27% peak-to-peak modulation over
  two cycles is genuinely hard to see. It is the *number* of bands, not their
  depth, that makes the artefact read as an artefact. Readout time is
  therefore worth getting right rather than assuming a fast sensor: a
  rolling-shutter camera reads out over roughly its frame period, so 25-30 fps
  gives 33-40 ms and a bandwidth-limited 17 fps stream gives 60 ms, for 2-7
  bands rather than 1-3.

* **The lamp is not a sinusoid.** Only an incandescent filament is, and only
  because its thermal mass low-passes everything. A discharge lamp follows
  ``|sin|`` and a cheap LED driver conducts in narrow pulses near the mains
  peaks, both of which carry harmonics of the flicker frequency. Each
  harmonic is box-filtered by the exposure *independently*, so each gets its
  own ``sinc(n * T * f)``. Anti-flicker cancellation survives this exactly:
  when ``T * f`` is a whole number so is ``n * T * f``, so every harmonic
  vanishes together.

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


#: Harmonics of the flicker fundamental kept when building the band profile.
#: Beyond this they are below a quantisation step for any plausible depth.
_HARMONICS = 6


def lamp_waveform_harmonics(pulse_exponent: float, count: int = _HARMONICS) -> np.ndarray:
    """Fourier amplitudes of one flicker period of the lamp's light output.

    The lamp is modelled as ``|sin(pi * u)| ** pulse_exponent`` over the
    flicker period, which spans the range of real sources with one knob:

    * ``1`` — a gas-discharge lamp, whose output follows the rectified current.
    * ``2`` — ``sin^2``, which *is* a pure sinusoid at the flicker frequency.
      An incandescent filament sits here or smoother, its thermal mass having
      low-passed the supply.
    * ``4`` and above — a cheap LED driver conducting in narrow pulses near
      the mains peaks.

    Returned normalised so the amplitudes sum to 1: with no exposure averaging
    the peak modulation is then exactly the caller's nominal depth.
    """
    phase = np.linspace(0.0, 1.0, 512, endpoint=False)
    output = np.abs(np.sin(np.pi * phase)) ** float(pulse_exponent)
    modulation = output / max(float(output.mean()), 1e-9) - 1.0
    spectrum = np.fft.rfft(modulation)[1 : count + 1] / len(modulation) * 2.0
    magnitude = np.abs(spectrum)
    total = float(magnitude.sum())
    if total < 1e-9:
        return np.zeros(count, dtype=np.float64)
    return np.asarray(magnitude / total, dtype=np.float64)


class AcFlicker(WeatherTransform):
    """Horizontal banding from AC-powered lighting beating against the shutter.

    Args:
        mains_frequency_hz: supply frequency. 50 Hz across most of the world,
            60 Hz in the Americas and parts of Asia. The lighting pulses at
            twice this.
        readout_time_range_s: time the sensor takes to read all rows out.
            Together with the flicker frequency this fixes how many bands
            cross the frame, and that count is what makes the artefact
            legible — see the module docstring. A rolling shutter reads out
            over roughly its frame period, so the default spans a 50 fps
            sensor (20 ms) down to a bandwidth-limited 17 fps stream (60 ms),
            giving 2-6 bands on 50 Hz mains and 2.4-7.2 on 60 Hz.
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
        pulse_exponent_range: shape of the lamp's output over one flicker
            period, as ``|sin| ** exponent``. 1 is a gas-discharge lamp
            following the rectified current, 2 is exactly a sinusoid (an
            incandescent filament, thermally smoothed), 4 and above a cheap
            LED driver conducting in narrow pulses. Higher exponents put more
            energy in harmonics, which gives the bands visible edges instead
            of a smooth roll.
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
        pulse_exponent_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.5, 16.0)),
        ]
        modulation_depth_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 1.0)),
        ]

    def __init__(
        self,
        mains_frequency_hz: float = 50.0,
        readout_time_range_s: tuple[float, float] = (0.020, 0.060),
        exposure_time_range_s: tuple[float, float] = (0.001, 0.016),
        pulse_exponent_range: tuple[float, float] = (1.0, 5.0),
        modulation_depth_range: tuple[float, float] = (0.05, 0.35),
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.mains_frequency_hz = mains_frequency_hz
        self.readout_time_range_s = readout_time_range_s
        self.exposure_time_range_s = exposure_time_range_s
        self.pulse_exponent_range = pulse_exponent_range
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
        pulse_exponent = self.py_random.uniform(*self.pulse_exponent_range)
        nominal = self.py_random.uniform(*self.modulation_depth_range)

        flicker_hz = 2.0 * self.mains_frequency_hz
        record_draw(
            readout_s=readout,
            exposure_s=exposure,
            nominal_depth=nominal,
            flicker_hz=flicker_hz,
            exposure_in_flicker_periods=exposure * flicker_hz,
            pulse_exponent=pulse_exponent,
        )
        # Each harmonic is box-filtered by the exposure on its own, so each
        # carries its own sinc. Cancellation is preserved exactly: when
        # `exposure * flicker_hz` is a whole number so is every multiple of
        # it, and the whole profile vanishes together.
        orders = np.arange(1, _HARMONICS + 1, dtype=np.float64)
        harmonics = lamp_waveform_harmonics(pulse_exponent) * np.abs(
            np.sinc(orders * exposure * flicker_hz)
        )
        return {
            # Bands across the frame: how many flicker cycles elapse while the
            # sensor is reading rows out.
            "cycles_per_frame": flicker_hz * readout,
            "depth": nominal,
            "harmonics": (nominal * harmonics).astype(np.float32),
            # On video this advances frame to frame; per frame it is a draw.
            "phase": self.py_random.uniform(0.0, 2.0 * np.pi),
        }

    def apply(
        self,
        img: np.ndarray,
        cycles_per_frame: float,
        depth: float,
        harmonics: np.ndarray,
        phase: float,
        **params: Any,
    ) -> np.ndarray:
        if float(np.abs(harmonics).sum()) < 1e-4:
            return img

        height = img.shape[0]
        rows = np.arange(height, dtype=np.float64) / max(height, 1)
        modulation = np.zeros(height, dtype=np.float64)
        for order, amplitude in enumerate(harmonics, start=1):
            modulation += amplitude * np.sin(
                2.0 * np.pi * order * cycles_per_frame * rows + order * phase
            )
        gain = (1.0 + modulation).astype(np.float32)

        linear, scale, is_integer = to_linear_unit(img)
        linear = linear * gain[:, None, None]
        return from_linear_unit(linear, scale, is_integer, img.dtype)
