"""The illuminator's own light, reflected in the dome.

Physical model
--------------
A dome camera's LEDs and its lens sit under the same bubble of
polycarbonate. Most of the LEDs' light leaves through it; a few percent is
reflected by its inner surface, and more is scattered by whatever is on it:
dust, a film of dried rain, scratches. Some of that comes back into the
lens. Installers know it as IR bleed or IR reflection, and the foam gasket
around the lens of a dome camera is there to prevent it. When the gasket is
missing, perished or not seated, or the dome is dirty, the night picture
has a pale ring or arc of haze over it and the blacks are grey.

**It is a ring, because the lamp is.** The LEDs stand in a circle around
the lens, and the dome's inner surface mirrors them back as a circle. That
image is centimetres away, so it is far out of focus: a soft band, not a
row of points, though with few LEDs and a small stop the individual lamps
still show as lobes along it.

**It is usually an arc, because the lens is rarely centred.** A dome's
camera is tilted to look where it is needed, which puts the lens off the
dome's axis; the reflection moves across the frame and part of it leaves.
A crescent on one side is the usual picture.

**It is mottled, because the dome is dirty.** A clean surface reflects
evenly. Deposits scatter where they lie, so the band is patchy, and a film
over the whole dome adds an even veil that lifts the blacks everywhere.

It only adds light, in the sensor's terms, so it belongs after
``NightIrMode`` with the other things the camera lights itself.
"""

from __future__ import annotations

import math
from typing import Annotated, Any

import cv2
import numpy as np
from albumentations.core.pydantic import check_range_bounds, nondecreasing
from albumentations.core.transforms_interface import BaseTransformInitSchema
from pydantic import AfterValidator

from weathermentations import mode
from weathermentations.base import WeatherTransform
from weathermentations.diagnostics import record_draw
from weathermentations.utils.grids import on_coarse_grid
from weathermentations.utils.noise import value_noise
from weathermentations.utils.photometry import from_linear_unit, to_linear_unit

# The glow is smooth over tens of pixels, so in fast mode it is worked out
# on a grid this many times coarser and enlarged.
_REDUCTION = 4
# Side of the patch of noise the dome's deposits are drawn on; they are
# large and soft, so it is enlarged to the frame.
_DIRT_CELLS = 96


def ring_glow(
    yy: np.ndarray,
    xx: np.ndarray,
    *,
    centre: tuple[float, float],
    scale: float,
    radius: float,
    width: float,
    arc_towards: float,
    arc_share: float,
    led_count: int,
    led_contrast: float,
    led_phase: float,
) -> np.ndarray:
    """Relative brightness of the reflected ring of LEDs at each pixel, 0 to 1.

    ``radius`` and ``width`` are in units of ``scale`` pixels. Across the
    band it is a Gaussian about ``radius``. Around it, the part within
    ``arc_share`` of a full turn about the direction ``arc_towards`` is
    seen and the rest fades out, and ``led_contrast`` of it is gathered
    into ``led_count`` lobes.
    """
    dx = (xx - centre[0]) / scale
    dy = (yy - centre[1]) / scale
    out = np.sqrt(dx * dx + dy * dy)
    band = np.exp(-0.5 * ((out - radius) / width) ** 2)
    around = np.arctan2(dy, dx)
    # How far round from the middle of the arc, as a share of half a turn.
    # The arc fades out over an eighth of its own length at either end, and
    # is a little longer than its share by that much, so that a share of
    # one is the whole ring at full strength.
    off = np.abs(np.angle(np.exp(1j * (around - arc_towards)))) / np.pi
    fade = max(0.25 * arc_share, 1e-3)
    seen = np.clip((1.125 * arc_share - off) / fade + 0.5, 0.0, 1.0)
    seen = seen * seen * (3.0 - 2.0 * seen)
    lobes = 1.0 - led_contrast * (0.5 - 0.5 * np.cos(led_count * around + led_phase))
    glow: np.ndarray = (band * seen * lobes).astype(np.float32)
    return glow


class IrDomeGlare(WeatherTransform):
    """The infrared ring reflected in a dome that is dirty or badly sealed.

    Adds light only; needs no depth. Apply it after ``NightIrMode``.

    Args:
        intensity_range: brightness of the reflected ring at its brightest,
            in units of full scale.
        ring_radius_range: radius of the ring, as a fraction of the frame's
            half-diagonal.
        ring_width_range: width of its band, the same way. Wide for a lens
            wide open, which blurs the reflection most.
        offset_range: how far the ring's centre is from the frame's, as a
            fraction of the half-diagonal: how far the lens is tilted off
            the dome's axis.
        arc_range: about how much of the ring is seen, 0 to 1. One is the
            whole ring; a quarter is a crescent.
        led_count_range: how many LEDs the ring has.
        led_contrast_range: how distinct the individual LEDs are along the
            band, 0 to 1. Zero is an even band.
        veil_range: an even haze over the whole frame, from a film on the
            dome, in units of full scale.
        dirt_range: how patchy the band is, 0 to 1, from deposits that
            scatter where they lie.

    Targets:
        image

    Image types:
        uint8, float32
    """

    class InitSchema(BaseTransformInitSchema):
        """The arguments of ``IrDomeGlare`` as Albumentations validates them: each
        range ordered low to high and within the bounds given here.
        """

        intensity_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 5.0)),
        ]
        ring_radius_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.05, 2.0)),
        ]
        ring_width_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.01, 1.0)),
        ]
        offset_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 1.5)),
        ]
        arc_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.05, 1.0)),
        ]
        led_count_range: Annotated[
            tuple[int, int],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(1, 64)),
        ]
        led_contrast_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 1.0)),
        ]
        veil_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 1.0)),
        ]
        dirt_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 1.0)),
        ]

    def __init__(
        self,
        intensity_range: tuple[float, float] = (0.08, 0.6),
        ring_radius_range: tuple[float, float] = (0.3, 0.9),
        ring_width_range: tuple[float, float] = (0.06, 0.25),
        offset_range: tuple[float, float] = (0.0, 0.5),
        arc_range: tuple[float, float] = (0.2, 1.0),
        led_count_range: tuple[int, int] = (4, 24),
        led_contrast_range: tuple[float, float] = (0.0, 0.7),
        veil_range: tuple[float, float] = (0.0, 0.06),
        dirt_range: tuple[float, float] = (0.0, 0.8),
        p: float = 0.5,
    ):
        """Store the ranges; each argument is described in the class docstring."""
        super().__init__(p=p)
        self.intensity_range = intensity_range
        self.ring_radius_range = ring_radius_range
        self.ring_width_range = ring_width_range
        self.offset_range = offset_range
        self.arc_range = arc_range
        self.led_count_range = led_count_range
        self.led_contrast_range = led_contrast_range
        self.veil_range = veil_range
        self.dirt_range = dirt_range

    def get_params_dependent_on_data(
        self, params: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, Any]:
        """Place the reflected ring and draw the light it adds at each pixel.

        Returns ``glow``, ``(H, W)`` in units of full scale: the ring, as
        patchy as the dome is dirty, and the even veil under it.
        """
        del data
        height, width = params["shape"][:2]
        rng = self.random_generator
        half_diagonal = 0.5 * float(np.hypot(width, height))

        intensity = self.py_random.uniform(*self.intensity_range)
        radius = self.py_random.uniform(*self.ring_radius_range)
        band = self.py_random.uniform(*self.ring_width_range)
        offset = self.py_random.uniform(*self.offset_range)
        arc = self.py_random.uniform(*self.arc_range)
        leds = self.py_random.randint(*self.led_count_range)
        contrast = self.py_random.uniform(*self.led_contrast_range)
        veil = self.py_random.uniform(*self.veil_range)
        dirt = self.py_random.uniform(*self.dirt_range)

        # The ring moves off-centre the way the lens is tilted, and the part
        # of it that stays in the frame is the part on the frame's side.
        tilt = float(rng.uniform(0.0, 2.0 * np.pi))
        centre = (
            (width - 1) / 2.0 + offset * half_diagonal * math.cos(tilt),
            (height - 1) / 2.0 + offset * half_diagonal * math.sin(tilt),
        )
        shape = {
            "centre": centre,
            "scale": half_diagonal,
            "radius": radius,
            "width": band,
            "arc_towards": tilt + math.pi + float(rng.uniform(-0.6, 0.6)),
            "arc_share": arc,
            "led_count": leds,
            "led_contrast": contrast,
            "led_phase": float(rng.uniform(0.0, 2.0 * np.pi)),
        }
        glow = on_coarse_grid(
            height,
            width,
            lambda yy, xx: ring_glow(yy, xx, **shape),
            reduction=_REDUCTION if mode.fast() else 1,
        )
        if dirt > 0.0:
            patches = value_noise(_DIRT_CELLS, _DIRT_CELLS, rng=rng, octaves=4, base_cells=3)
            patches = cv2.resize(patches, (width, height), interpolation=cv2.INTER_CUBIC)
            # About a mean of one, so dirt moves the light and adds none.
            glow *= (1.0 - dirt) + dirt * patches / max(float(patches.mean()), 1e-3)
        glow *= np.float32(intensity)
        glow += np.float32(veil)

        record_draw(
            intensity=intensity,
            ring_radius=radius,
            ring_width=band,
            offset=offset,
            arc=arc,
            led_count=leds,
            led_contrast=contrast,
            veil=veil,
            dirt=dirt,
            lit_fraction=lambda: float((glow > veil + 0.25 * intensity).mean()),
        )
        return {"glow": glow}

    def apply(self, img: np.ndarray, glow: np.ndarray, **params: Any) -> np.ndarray:
        """Add the reflection's light to the frame."""
        linear, scale, is_integer = to_linear_unit(img)
        return from_linear_unit(linear + glow[..., None], scale, is_integer, img.dtype)
