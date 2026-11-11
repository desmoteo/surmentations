"""Sensor noise for a camera still in colour.

Physical model
--------------
A pixel counts photoelectrons, and counting is noisy in two ways that behave
differently:

* **Shot noise.** Photon arrivals are Poisson, so a pixel that collected
  ``N`` electrons carries a noise of ``sqrt(N)``. It grows with the signal
  but more slowly, so it is worst *relative to the signal* in the shadows.
* **Read noise.** The readout chain adds a few electrons of its own whatever
  the signal was. It is what is left in the dark.

Neither depends on gain. What gain does is decide how few electrons count as
a full-scale picture: at 0 dB a white pixel is a full well, several thousand
electrons, and at 30 dB it is a few hundred, with the same read noise and
proportionally more shot noise. That is the whole reason a night frame is
noisy, and why ``gain_range_db`` is the parameter that matters here.

So, with the image in linear light and ``g`` the linear gain::

    electrons = image * full_well / g
    noisy     = electrons + Normal(0, sqrt(electrons + read_noise^2))
    image'    = noisy * g / full_well

(The Gaussian stands in for the Poisson, which it matches closely above ten
electrons or so. Below that real shot noise is skewed; this is not.)

**The noise is drawn on the colour filter mosaic, not per channel.** A sensor
has one colour sample per pixel, in a Bayer pattern — half green, a quarter
each red and blue — and the other two values at each pixel are interpolated
from neighbours. Interpolation is linear, so the noise goes through it too:
what reaches the picture is not three independent grains but one, spread by
the demosaicing kernel, wider in red and blue than in green because those
samples are further apart. That is the coarse coloured blotching of a real
low-light frame, and independent per-channel Gaussian noise — fine, sharp and
evenly coloured — does not look like it.

Where it sits
-------------
At the sensor, before everything the image processor does. It is what
``TemporalDenoiseArtifacts`` exists to clean up: without a noise source ahead
of it, that transform smears a frame that had nothing to remove.

``NightIrMode`` carries its own noise for the monochrome case and should not
be followed by this.
"""

from __future__ import annotations

from typing import Annotated, Any

import cv2
import numpy as np
from albumentations.core.pydantic import check_range_bounds, nondecreasing
from albumentations.core.transforms_interface import BaseTransformInitSchema
from pydantic import AfterValidator

from weathermentations.base import WeatherTransform
from weathermentations.diagnostics import record_draw
from weathermentations.utils.photometry import from_linear_unit, to_linear_unit

# Bilinear demosaicing as a pair of convolutions over a mosaic in which the
# sites of other colours are zero: one kernel for the two sparse colours and
# one for green.
_SPARSE_KERNEL = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 4.0
_GREEN_KERNEL = np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]], dtype=np.float32) / 4.0


def bayer_sites(height: int, width: int) -> np.ndarray:
    """Which channel each pixel of an RGGB mosaic samples: ``(H, W)`` of 0, 1, 2."""
    sites = np.ones((height, width), dtype=np.uint8)
    sites[0::2, 0::2] = 0
    sites[1::2, 1::2] = 2
    return sites


def _one_colour(mosaic: np.ndarray, channel: int) -> np.ndarray:
    """An RGGB ``mosaic`` at the sites of ``channel``, zero elsewhere.

    Copies along the mosaic's own stride. Selecting through a frame-sized
    map of site labels gave the same plane three times slower.
    """
    if channel == 1:
        plane = mosaic.astype(np.float32, copy=True)
        plane[0::2, 0::2] = 0.0
        plane[1::2, 1::2] = 0.0
        return plane
    start = 0 if channel == 0 else 1
    plane = np.zeros(mosaic.shape, dtype=np.float32)
    plane[start::2, start::2] = mosaic[start::2, start::2]
    return plane


def demosaic(mosaic: np.ndarray, sites: np.ndarray | None = None) -> np.ndarray:
    """Bilinear interpolation of a single-plane RGGB mosaic to ``(H, W, 3)``.

    ``sites`` is accepted for callers that pass :func:`bayer_sites` and is
    not read: the layout is RGGB, which is what the kernels assume too.
    """
    planes = []
    for channel, kernel in ((0, _SPARSE_KERNEL), (1, _GREEN_KERNEL), (2, _SPARSE_KERNEL)):
        sampled = _one_colour(mosaic, channel)
        planes.append(cv2.filter2D(sampled, -1, kernel, borderType=cv2.BORDER_REFLECT_101))
    return np.stack(planes, axis=2)


def photon_transfer_noise(
    linear_rgb: np.ndarray,
    rng: np.random.Generator,
    gain: float,
    full_well_e: float,
    read_noise_e: float,
) -> np.ndarray:
    """Noise to add to a linear-light image, in the image's own units.

    Drawn per mosaic site from the electrons that site collected, then
    demosaiced. See the module docstring.
    """
    height, width = linear_rgb.shape[:2]
    # The mosaic: `linear_rgb[rows, cols, bayer_sites]`, as three strided
    # copies. Green everywhere, then red and blue on their own sites.
    sampled = linear_rgb[..., 1].copy()
    sampled[0::2, 0::2] = linear_rgb[0::2, 0::2, 0]
    sampled[1::2, 1::2] = linear_rgb[1::2, 1::2, 2]

    electrons_per_unit = full_well_e / gain
    electrons = np.maximum(sampled, 0.0) * electrons_per_unit
    sigma = np.sqrt(electrons + read_noise_e * read_noise_e) / electrons_per_unit
    mosaic = rng.standard_normal((height, width)).astype(np.float32) * sigma.astype(np.float32)
    return demosaic(mosaic)


class SensorNoise(WeatherTransform):
    """Shot and read noise on the colour filter mosaic, set by sensor gain.

    Args:
        gain_range_db: analogue gain, drawn per call. **This is the knob.**
            Gain does not create noise; it decides how few electrons make a
            full-scale picture, and so how large the fixed noise is beside
            it. 0 dB is a camera in daylight; 18-24 dB a street at night;
            36 dB is the top of most sensors' range, a frame that is mostly
            grain.
        full_well_range_e: electrons a pixel holds at saturation with no
            gain. Small surveillance pixels (2-3 micrometres) hold a few
            thousand; a larger pixel holds more and is cleaner at any gain.
        read_noise_range_e: electrons of signal-independent noise from the
            readout chain. Modern CMOS sensors are at 1-3; older or hotter
            ones several times that. It sets the floor in the shadows.

    Targets:
        image

    Image types:
        uint8, float32
    """

    class InitSchema(BaseTransformInitSchema):
        gain_range_db: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 48.0)),
        ]
        full_well_range_e: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(500.0, 200000.0)),
        ]
        read_noise_range_e: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 50.0)),
        ]

    def __init__(
        self,
        gain_range_db: tuple[float, float] = (0.0, 30.0),
        full_well_range_e: tuple[float, float] = (4000.0, 12000.0),
        read_noise_range_e: tuple[float, float] = (1.5, 5.0),
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.gain_range_db = gain_range_db
        self.full_well_range_e = full_well_range_e
        self.read_noise_range_e = read_noise_range_e

    def get_params_dependent_on_data(
        self, params: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, Any]:
        gain_db = self.py_random.uniform(*self.gain_range_db)
        full_well = self.py_random.uniform(*self.full_well_range_e)
        read_noise = self.py_random.uniform(*self.read_noise_range_e)
        gain = 10.0 ** (gain_db / 20.0)
        record_draw(
            gain_db=gain_db,
            electrons_at_full_scale=full_well / gain,
            # Signal-to-noise of a mid-grey (18%) pixel: the one number that
            # says how noisy the frame is.
            snr_mid_grey=0.18 * full_well / gain
            / float(np.sqrt(0.18 * full_well / gain + read_noise**2)),
        )
        return {
            "gain": gain,
            "full_well_e": full_well,
            "read_noise_e": read_noise,
            "noise_seed": self.py_random.randint(0, 2**31 - 1),
        }

    def apply(
        self,
        img: np.ndarray,
        gain: float,
        full_well_e: float,
        read_noise_e: float,
        noise_seed: int,
        **params: Any,
    ) -> np.ndarray:
        linear, scale, is_integer = to_linear_unit(img)
        rng = np.random.default_rng(noise_seed)
        noise = photon_transfer_noise(linear, rng, gain, full_well_e, read_noise_e)
        return from_linear_unit(linear + noise, scale, is_integer, img.dtype)
