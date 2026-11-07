"""Frost on the glass.

Physical model
--------------
On a clear cold night the dome radiates its heat to the sky and falls below
the frost point, and ice grows on it. By morning the feed is a clear patch
in the middle of a white vignette, and it stays that way until the sun or
the camera's own warmth clears it.

**It grows in from the rim.** The centre of the glass is over the lens and
the electronics, which are warm; the rim is over nothing. Ice nucleates at
the edge and on whatever dirt is there and spreads inwards, so the frost is
a border whose inner edge is ragged, with the middle of the view the last
to go.

**It is feathered.** Ice on glass grows as fern-like plumes along the
direction it is spreading, so the border is not an even grey but a field of
streaks pointing at the centre. They are on the glass, centimetres inside
the lens's near focus limit, so they are seen blurred: what survives of a
crystal a tenth of a millimetre wide is the texture of the plumes, not
their edges.

**It scatters; it does not absorb.** Ice is clear. A layer of small
crystals passes nearly all the light that reaches it, in the wrong
direction. So behind frost the scene is not darkened but replaced, by a
wide blur of itself and by light the frost picks up from everywhere else
in front of the glass, the sky above the frame included: frost is pale by
day, and under the infrared ring at night, which lights it from a few
centimetres away, it is white. The same model as ``DirtSoil``'s, with a
deposit that has no colour of its own.
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
from weathermentations.utils.color import illuminant_radiance_linear
from weathermentations.utils.filters import wide_blur
from weathermentations.utils.noise import value_noise
from weathermentations.utils.photometry import from_linear_unit, to_linear_unit

# How wide a blur of the scene the frost passes on, as a fraction of the
# diagonal: light through it has lost its direction almost entirely. And
# how much of it: the rest is scattered back out of the glass.
_DIFFUSE_SIGMA = 0.08
_DIFFUSE_SHARE = 0.7
# The plumes are drawn in polar coordinates, this many samples round the
# frame by this many from rim to centre, and wound back onto the frame.
_PLUME_TURN, _PLUME_DEPTH = 720, 96


def frost_depth(
    rng: np.random.Generator,
    height: int,
    width: int,
    *,
    coverage: float,
    thickness: float,
    feather: float,
) -> np.ndarray:
    """Optical depth of a border of frost, ``(H, W)`` float32, before defocus.

    ``coverage`` is how far the frost has grown in from the rim, as a share
    of the way to the centre; ``thickness`` its optical depth where it is
    established; ``feather`` how much of it is in plumes pointing inwards,
    0 to 1, the rest being an even layer.
    """
    rows = (np.arange(height, dtype=np.float32) - (height - 1) / 2.0) / (height / 2.0)
    columns = (np.arange(width, dtype=np.float32) - (width - 1) / 2.0) / (width / 2.0)
    # 0 at the rim, 1 at the centre. The power makes the clear patch a
    # rounded rectangle, which is what a rectangular view of a round patch
    # on a dome is.
    inward = 1.0 - (np.abs(rows[:, None]) ** 4 + np.abs(columns[None, :]) ** 4) ** 0.25
    angle = np.arctan2(rows[:, None], columns[None, :])

    # Plumes: noise that changes quickly round the frame and slowly along
    # the way in, which drawn on the frame are streaks towards the centre.
    polar = value_noise(_PLUME_DEPTH, _PLUME_TURN, rng=rng, octaves=3, base_cells=6)
    fine = rng.random((6, _PLUME_TURN // 2)).astype(np.float32)
    polar = 0.5 * polar + 0.5 * cv2.resize(
        fine, (_PLUME_TURN, _PLUME_DEPTH), interpolation=cv2.INTER_CUBIC
    )
    # Stretched to its full range: a plume is ice or it is clear glass.
    low, high = np.percentile(polar, (5.0, 95.0))
    polar = np.clip((polar - low) / max(float(high - low), 1e-3), 0.0, 1.0)
    # The front: how far in the ice has reached, ragged by a slow noise and
    # by the plumes, which run ahead of the layer between them. The same
    # noise bends the plumes, which do not grow dead straight.
    ragged = value_noise(height, width, rng=rng, octaves=3, base_cells=3)
    map_x = ((angle / (2.0 * np.pi) + 0.5) * (_PLUME_TURN - 1)).astype(np.float32)
    map_x += (0.06 * _PLUME_TURN) * (ragged - 0.5)
    map_y = (np.clip(inward, 0.0, 1.0) * (_PLUME_DEPTH - 1)).astype(np.float32)
    plumes = cv2.remap(polar, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    reach = coverage * (0.7 + 0.6 * ragged) + 0.25 * feather * coverage * (plumes - 0.5)
    grown = np.clip((reach - inward) / max(0.12 * coverage, 1e-3), 0.0, 1.0)
    grown = grown * grown * (3.0 - 2.0 * grown)
    depth: np.ndarray = thickness * grown * ((1.0 - feather) + feather * 2.0 * plumes * plumes)
    return depth.astype(np.float32)


class FrostOnGlass(WeatherTransform):
    """A border of frost on the dome, grown in from the rim.

    Needs no depth. Under the infrared ring, set ``illuminator_glow_range``
    and apply it after ``NightIrMode``, as with ``WaterDroplets``.

    Args:
        coverage_range: how far the frost has grown in from the rim, as a
            share of the way to the centre of the frame.
        thickness_range: optical depth of established frost. At 1 it passes
            a third of the scene directly; at 3, a twentieth.
        feather_range: how much of the frost is in plumes pointing inwards,
            0 to 1.
        defocus_range: blur of the frost's pattern, as a fraction of the
            diagonal. It is on the glass, far inside the near focus limit.
        ambient_light_range: light the frost picks up from in front of the
            glass and scatters into the lens, as a share of the scene's
            illuminant. What makes frost pale and not just blurred.
        illuminator_glow_range: light the frost returns from the infrared
            ring, in units of full scale. Zero by day.

    Targets:
        image

    Image types:
        uint8, float32
    """

    class InitSchema(BaseTransformInitSchema):
        """The arguments of ``FrostOnGlass`` as Albumentations validates them: each
        range ordered low to high and within the bounds given here.
        """

        coverage_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 1.5)),
        ]
        thickness_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 10.0)),
        ]
        feather_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 1.0)),
        ]
        defocus_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 0.1)),
        ]
        ambient_light_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 2.0)),
        ]
        illuminator_glow_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 5.0)),
        ]

    def __init__(
        self,
        coverage_range: tuple[float, float] = (0.25, 0.9),
        thickness_range: tuple[float, float] = (0.4, 2.5),
        feather_range: tuple[float, float] = (0.3, 0.9),
        defocus_range: tuple[float, float] = (0.002, 0.012),
        ambient_light_range: tuple[float, float] = (0.05, 0.3),
        illuminator_glow_range: tuple[float, float] = (0.0, 0.0),
        p: float = 0.5,
    ):
        """Store the ranges; each argument is described in the class docstring."""
        super().__init__(p=p)
        self.coverage_range = coverage_range
        self.thickness_range = thickness_range
        self.feather_range = feather_range
        self.defocus_range = defocus_range
        self.ambient_light_range = ambient_light_range
        self.illuminator_glow_range = illuminator_glow_range

    def get_params_dependent_on_data(
        self, params: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, Any]:
        """Grow the frost and blur it as the lens sees it.

        Returns ``scattered``, the share of each pixel's light the frost has
        scattered, ``1 - exp(-depth)``; the ``ambient`` light it picks up,
        as a share of the illuminant; and its ``glow`` under the lamp.
        """
        del data
        height, width = params["shape"][:2]
        diagonal = float(np.hypot(width, height))
        coverage = self.py_random.uniform(*self.coverage_range)
        thickness = self.py_random.uniform(*self.thickness_range)
        feather = self.py_random.uniform(*self.feather_range)
        sigma = self.py_random.uniform(*self.defocus_range) * diagonal

        depth = frost_depth(
            self.random_generator,
            height,
            width,
            coverage=coverage,
            thickness=thickness,
            feather=feather,
        )
        scattered = -np.expm1(-depth)
        if sigma > 0.3:
            scattered = wide_blur(scattered, sigma)
        record_draw(
            coverage=coverage,
            thickness=thickness,
            feather=feather,
            defocus_sigma_px=sigma,
            frosted_fraction=lambda: float((scattered > 0.5).mean()),
        )
        return {
            "scattered": np.clip(scattered, 0.0, 1.0),
            "ambient": self.py_random.uniform(*self.ambient_light_range),
            "glow": self.py_random.uniform(*self.illuminator_glow_range),
        }

    def apply(
        self,
        img: np.ndarray,
        scattered: np.ndarray,
        ambient: float,
        glow: float,
        **params: Any,
    ) -> np.ndarray:
        """Replace what the frost covers by the scene's wide blur and the light it picks up.

        ``linear (1 - s) + s (0.7 diffuse + ambient illuminant + glow)``:
        the frost passes most of the scene's light on with its direction
        lost, and adds what reaches it from elsewhere.
        """
        if float(scattered.max()) < 1e-3:
            return img
        linear, scale, is_integer = to_linear_unit(img)
        height, width = img.shape[:2]
        diffuse = wide_blur(linear, _DIFFUSE_SIGMA * float(np.hypot(width, height)))
        light = ambient * illuminant_radiance_linear(linear) + glow
        share = scattered[..., None]
        out = linear * (1.0 - share) + share * (_DIFFUSE_SHARE * diffuse + light[None, None, :])
        return from_linear_unit(out, scale, is_integer, img.dtype)
