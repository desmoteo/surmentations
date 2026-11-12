"""A lens that has been tampered with: sprayed, taped or covered.

Physical model
--------------
Before a camera is stolen, or something is done that it should not see, it
is often blinded. Three ways account for nearly all of it, and they are the
cases a tamper detector is built to recognise:

* **Spray paint.** A burst from a can a hand's width away: an opaque core
  where the cone hit, thinning to a speckle of separate droplets around it,
  and runs where it went on thick enough to flow.
* **Tape or a sticker.** One or a few straight strips across the glass,
  opaque or nearly so, with clean parallel edges.
* **A cover.** A bag, a cloth or a cap over the whole housing. A little
  light comes through fabric or polythene, with no image in it, and the
  cover's folds show as slow changes in how much.

None of this is weather or optics, and only one thing about it is physics:
**it is on the housing**, so it is far out of focus. The edge of a strip of
tape is a gradient, the speckle of a spray merges into a haze, and what is
left is a soft shape of one colour, lit from behind the camera's point of
view: by day it is in its own shade and shows as its own colour, dimly; at
night the infrared ring lights it from centimetres away and anything that
is not black in the infrared glows.

This transform hides part or all of the scene by design. What a covered
object's annotation should then be is the caller's decision; the transform
leaves annotations alone.
"""

from __future__ import annotations

import math
from typing import Annotated, Any

import cv2
import numpy as np
from albumentations.core.pydantic import check_range_bounds, nondecreasing
from albumentations.core.transforms_interface import BaseTransformInitSchema
from pydantic import AfterValidator, field_validator

from weathermentations.base import WeatherTransform
from weathermentations.diagnostics import record_draw
from weathermentations.utils.color import illuminant_radiance_linear
from weathermentations.utils.filters import wide_blur
from weathermentations.utils.noise import value_noise
from weathermentations.utils.photometry import from_linear_unit, to_linear_unit

KINDS = ("spray", "tape", "cover")

# Reflectances the paint, tape or cover may have, in linear light: black
# most often, then white and grey, then the colours cans and tape come in.
_ALBEDOS = np.array(
    [
        (0.02, 0.02, 0.02),
        (0.02, 0.02, 0.02),
        (0.03, 0.03, 0.035),
        (0.75, 0.75, 0.72),
        (0.35, 0.35, 0.35),
        (0.55, 0.05, 0.04),
        (0.04, 0.10, 0.45),
        (0.05, 0.30, 0.08),
        (0.60, 0.45, 0.25),
        (0.45, 0.42, 0.38),
    ],
    dtype=np.float32,
)
# How wide a blur of the scene a cover lets through, as a fraction of the
# diagonal: light through cloth carries no image.
_COVER_SIGMA = 0.12


def spray_mask(rng: np.random.Generator, height: int, width: int, coverage: float) -> np.ndarray:
    """Paint from a spray can: how much of each pixel it covers, before defocus.

    One to three bursts, each an opaque core thinning outwards into
    speckle, sized so that together they cover about ``coverage`` of the
    frame, with runs under the thickest parts.
    """
    bursts = int(rng.integers(1, 4))
    yy = np.arange(height, dtype=np.float32)[:, None]
    xx = np.arange(width, dtype=np.float32)[None, :]
    reach = math.sqrt(coverage * height * width / (math.pi * bursts))
    amount = np.zeros((height, width), dtype=np.float32)
    for _ in range(bursts):
        cx, cy = rng.uniform(0.2, 0.8) * width, rng.uniform(0.2, 0.8) * height
        rx = reach * rng.uniform(0.8, 1.3)
        ry = reach * rng.uniform(0.8, 1.3)
        amount += np.exp(-0.5 * (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2) ** 1.5) * 2.2
    # Speckle: where the paint is thin it lands as separate droplets, so
    # coverage there is the chance that a pixel was hit.
    grain = rng.random((height, width)).astype(np.float32)
    mask = (amount > grain * 1.4 + 0.25).astype(np.float32)
    # Runs: from the thickest paint, straight down, tapering.
    thick = np.argwhere(amount > 1.6)
    for _ in range(int(rng.integers(0, 5)) if len(thick) else 0):
        y, x = thick[rng.integers(len(thick))]
        length = int(rng.uniform(0.05, 0.3) * height)
        cv2.line(
            mask, (int(x), int(y)), (int(x), int(y) + length), 1.0,
            max(2, int(0.006 * width * rng.uniform(0.6, 1.6))),
        )  # fmt: skip
    return mask


def tape_mask(rng: np.random.Generator, height: int, width: int, coverage: float) -> np.ndarray:
    """Strips of tape: how much of each pixel they cover, before defocus.

    One to three straight strips right across the frame, their widths
    adding up to about ``coverage`` of it.
    """
    strips = int(rng.integers(1, 4))
    diagonal = math.hypot(width, height)
    mask = np.zeros((height, width), dtype=np.float32)
    for _ in range(strips):
        angle = rng.uniform(0.0, math.pi)
        across = np.array([math.cos(angle), math.sin(angle)])
        along = np.array([-across[1], across[0]])
        centre = np.array([rng.uniform(0.25, 0.75) * width, rng.uniform(0.25, 0.75) * height])
        half = 0.5 * coverage * min(height, width) / strips * rng.uniform(0.7, 1.3)
        corners = [
            centre + sign_a * diagonal * along + sign_b * half * across
            for sign_a, sign_b in ((-1, -1), (1, -1), (1, 1), (-1, 1))
        ]
        cv2.fillConvexPoly(mask, np.rint(corners).astype(np.int32), 1.0, cv2.LINE_AA)
    return mask


class LensTamper(WeatherTransform):
    """Paint sprayed on the lens, tape across it, or a cover over the housing.

    Needs no depth. Hides the scene; annotations are left as they are.

    Args:
        kinds: which kinds of tampering to draw from, one a call: any of
            ``"spray"``, ``"tape"`` and ``"cover"``.
        coverage_range: how much of the frame paint or tape covers. A cover
            takes all of it whatever this says.
        opacity_range: how much light what is on the lens blocks. Paint and
            cloth are rarely quite opaque.
        defocus_range: blur of its outline, as a fraction of the diagonal.
        shade_range: how brightly it shows by day, as a share of what its
            colour would return in the open: it is seen from behind, in its
            own shade.
        glow_range: light it returns from the infrared ring, in units of
            full scale, for something white in the infrared; dark paint
            returns its share of that. Zero by day.

    Targets:
        image

    Image types:
        uint8, float32
    """

    class InitSchema(BaseTransformInitSchema):
        """The arguments of ``LensTamper`` as Albumentations validates them: each
        range ordered low to high and within the bounds given here.
        """

        kinds: tuple[str, ...]
        coverage_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.01, 1.0)),
        ]
        opacity_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 1.0)),
        ]
        defocus_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.001, 0.1)),
        ]
        shade_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 1.0)),
        ]
        glow_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 5.0)),
        ]

        @field_validator("kinds")
        @classmethod
        def _known_kinds(cls, kinds: tuple[str, ...]) -> tuple[str, ...]:
            """At least one kind, and none that is not in ``KINDS``."""
            unknown = [kind for kind in kinds if kind not in KINDS]
            if unknown or not kinds:
                raise ValueError(f"kinds must be one or more of {KINDS}, got {kinds!r}")
            return kinds

    def __init__(
        self,
        kinds: tuple[str, ...] = KINDS,
        coverage_range: tuple[float, float] = (0.15, 0.8),
        opacity_range: tuple[float, float] = (0.8, 1.0),
        defocus_range: tuple[float, float] = (0.006, 0.03),
        shade_range: tuple[float, float] = (0.1, 0.5),
        glow_range: tuple[float, float] = (0.0, 0.0),
        p: float = 0.5,
    ):
        """Store the ranges; each argument is described in the class docstring."""
        super().__init__(p=p)
        self.kinds = tuple(kinds)
        self.coverage_range = coverage_range
        self.opacity_range = opacity_range
        self.defocus_range = defocus_range
        self.shade_range = shade_range
        self.glow_range = glow_range

    def get_params_dependent_on_data(
        self, params: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, Any]:
        """Choose what was done to the lens and draw it as the lens sees it.

        Returns ``cover``, how much of each pixel is hidden; the ``albedo``
        of what hides it; whether it ``passes_light`` as a cover does; and
        its ``shade`` by day and ``glow`` under the lamp.
        """
        del data
        height, width = params["shape"][:2]
        diagonal = float(np.hypot(width, height))
        rng = self.random_generator

        kind = self.kinds[self.py_random.randrange(len(self.kinds))]
        coverage = self.py_random.uniform(*self.coverage_range)
        opacity = self.py_random.uniform(*self.opacity_range)
        sigma = max(1.0, self.py_random.uniform(*self.defocus_range) * diagonal)
        albedo = _ALBEDOS[int(rng.integers(len(_ALBEDOS)))]

        if kind == "spray":
            cover = wide_blur(spray_mask(rng, height, width, coverage), sigma)
        elif kind == "tape":
            cover = wide_blur(tape_mask(rng, height, width, coverage), sigma)
        else:
            # Everything, a little less where the cover is one layer thick:
            # folds show as slow changes in how much light comes through.
            folds = value_noise(height, width, rng=rng, octaves=3, base_cells=3)
            cover = 1.0 - 0.3 * (1.0 - opacity + 0.1) * folds
        cover = np.clip(cover * opacity, 0.0, 1.0).astype(np.float32)

        record_draw(
            kind=kind,
            coverage=coverage,
            opacity=opacity,
            defocus_sigma_px=sigma,
            albedo=albedo,
            covered_fraction=lambda: float((cover > 0.5).mean()),
        )
        return {
            "cover": cover,
            "albedo": albedo,
            "passes_light": kind == "cover",
            "shade": self.py_random.uniform(*self.shade_range),
            "glow": self.py_random.uniform(*self.glow_range),
        }

    def apply(
        self,
        img: np.ndarray,
        cover: np.ndarray,
        albedo: np.ndarray,
        passes_light: bool,
        shade: float,
        glow: float,
        **params: Any,
    ) -> np.ndarray:
        """Hide the scene behind what is on the lens, and show that in its place.

        What is seen where it covers is its own colour, in its own shade by
        day and lit by the ring at night. What shows where it does not
        quite cover is the scene, or under a cover the scene with no image
        left in it.
        """
        linear, scale, is_integer = to_linear_unit(img)
        surface = albedo * (shade * illuminant_radiance_linear(linear) + glow)
        behind = linear
        if passes_light:
            height, width = img.shape[:2]
            behind = wide_blur(linear, _COVER_SIGMA * float(np.hypot(width, height)))
        alpha = cover[..., None]
        out = behind * (1.0 - alpha) + alpha * surface[None, None, :]
        return from_linear_unit(out, scale, is_integer, img.dtype)
