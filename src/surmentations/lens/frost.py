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

**It is feathered.** Ice on glass grows as ferns: a stem along the
direction it is spreading, side branches off it at sixty degrees, the
angle of the ice crystal's own lattice, and smaller branches off those.
So the border is not an even grey but a field of fronds pointing inwards,
longest where they started first. They are on the glass, centimetres
inside the lens's near focus limit, which does two things at once: it
magnifies them, a frond a centimetre long being a fifth of the frame, and
it blurs them by the aperture's disc. What is seen is the frond's shape,
soft, without the crystal's edges.

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

import math
from typing import Annotated, Any

import cv2
import numpy as np
from albumentations.core.pydantic import check_range_bounds, nondecreasing
from albumentations.core.transforms_interface import BaseTransformInitSchema
from pydantic import AfterValidator

from surmentations.base import CameraTransform
from surmentations.diagnostics import record_draw
from surmentations.utils.color import illuminant_radiance_linear
from surmentations.utils.filters import wide_blur
from surmentations.utils.noise import value_noise
from surmentations.utils.photometry import from_linear_unit, to_linear_unit

# How wide a blur of the scene the frost passes on, as a fraction of the
# diagonal: light through it has lost its direction almost entirely. And
# how much of it: the rest is scattered back out of the glass.
_DIFFUSE_SIGMA = 0.08
_DIFFUSE_SHARE = 0.7
# The angle between a frond's stem and its branches: the ice lattice's.
_BRANCH_ANGLE = math.radians(60.0)
# Fronds are drawn on a canvas this many times smaller than the frame and
# enlarged: they are blurred by the lens afterwards, by more than that.
_FROND_REDUCTION = 2


def _frond(
    canvas: np.ndarray,
    rng: np.random.Generator,
    start: np.ndarray,
    direction: float,
    length: float,
    thickness: float,
    depth: int,
) -> None:
    """Draw one frond into ``canvas``, in place: a stem with branches either side, and theirs.

    The stem runs ``length`` from ``start`` along ``direction``, bending a
    little, and thins towards its tip. Branches leave it at sixty degrees,
    alternately, shorter the further along they start, and each is a
    frond of its own, ``depth`` levels down.
    """
    steps = max(2, int(length / max(2.8 * thickness, 3.0)))
    stride = length / steps
    point = start.astype(np.float64)
    for step in range(steps):
        direction += rng.uniform(-0.06, 0.06)
        following = point + stride * np.array([math.cos(direction), math.sin(direction)])
        along = step / steps
        width = max(1, int(round(thickness * (1.0 - 0.6 * along))))
        cv2.line(
            canvas,
            (int(round(point[0])), int(round(point[1]))),
            (int(round(following[0])), int(round(following[1]))),
            1.0,
            width,
            cv2.LINE_AA,
        )
        if depth > 0 and step > 0:
            side = 1.0 if step % 2 else -1.0
            reach = length * (0.45 - 0.4 * along) * rng.uniform(0.6, 1.1)
            if reach > 3.0 * thickness:
                _frond(
                    canvas,
                    rng,
                    following,
                    direction + side * _BRANCH_ANGLE,
                    reach,
                    max(1.0, 0.6 * thickness),
                    depth - 1,
                )
        point = following


def fronds(
    rng: np.random.Generator, height: int, width: int, *, reach: float, scale: float
) -> np.ndarray:
    """Ice ferns grown in from the frame's edge: where there is crystal, ``(H, W)`` 0 to 1.

    Stems start at points round the rim and head for the centre, each as
    long as ``reach`` of the way there, more or less. ``scale`` is the
    thickness of a stem at its root as a fraction of the frame's diagonal,
    which sets the size of everything else about a frond.
    """
    small_h, small_w = max(8, height // _FROND_REDUCTION), max(8, width // _FROND_REDUCTION)
    diagonal = math.hypot(small_h, small_w)
    thickness = max(1.0, scale * diagonal)
    canvas = np.zeros((small_h, small_w), dtype=np.float32)
    centre = np.array([small_w / 2.0, small_h / 2.0])
    # One stem every twenty-odd thicknesses of rim, on average: far enough
    # apart that the glass shows between their branches.
    perimeter = 2.0 * (small_h + small_w)
    for _ in range(max(6, int(perimeter / (24.0 * thickness)))):
        along = rng.uniform(0.0, perimeter)
        if along < small_w:
            start = np.array([along, 0.0])
        elif along < small_w + small_h:
            start = np.array([small_w - 1.0, along - small_w])
        elif along < 2 * small_w + small_h:
            start = np.array([along - small_w - small_h, small_h - 1.0])
        else:
            start = np.array([0.0, along - 2 * small_w - small_h])
        inward = centre - start
        length = float(np.linalg.norm(inward)) * reach * rng.uniform(0.4, 1.15)
        direction = math.atan2(inward[1], inward[0]) + rng.uniform(-0.45, 0.45)
        _frond(canvas, rng, start, direction, length, thickness, depth=2)
    crystal: np.ndarray = cv2.resize(canvas, (width, height), interpolation=cv2.INTER_LINEAR)
    return np.clip(crystal, 0.0, 1.0)


def frost_depth(
    rng: np.random.Generator,
    height: int,
    width: int,
    *,
    coverage: float,
    thickness: float,
    feather: float,
    crystal_scale: float = 0.006,
) -> np.ndarray:
    """Optical depth of a border of frost, ``(H, W)`` float32, before defocus.

    ``coverage`` is how far the frost has grown in from the rim, as a share
    of the way to the centre; ``thickness`` its optical depth where it is
    established; ``feather`` how much of it is in ferns, 0 to 1, the rest
    being an even layer behind their front; ``crystal_scale`` the
    thickness of a fern's stem as a fraction of the diagonal.
    """
    rows = (np.arange(height, dtype=np.float32) - (height - 1) / 2.0) / (height / 2.0)
    columns = (np.arange(width, dtype=np.float32) - (width - 1) / 2.0) / (width / 2.0)
    # 0 at the rim, 1 at the centre. The power makes the clear patch a
    # rounded rectangle, which is what a rectangular view of a round patch
    # on a dome is.
    inward = 1.0 - (np.abs(rows[:, None]) ** 4 + np.abs(columns[None, :]) ** 4) ** 0.25

    # The even layer: established frost, behind a ragged front.
    ragged = value_noise(height, width, rng=rng, octaves=3, base_cells=3)
    front = coverage * (0.2 + 0.4 * ragged)
    layer = np.clip((front - inward) / max(0.15 * coverage, 1e-3), 0.0, 1.0)
    layer = layer * layer * (3.0 - 2.0 * layer)
    if feather <= 0.0:
        return (thickness * layer).astype(np.float32)

    # The ferns run ahead of it, and where they crowd together the fine
    # crystals between them, too small to draw, thicken the ice as well.
    crystal = fronds(rng, height, width, reach=coverage, scale=crystal_scale)
    crowd_sigma = max(1.0, 3.0 * crystal_scale * math.hypot(height, width))
    crowd = cv2.GaussianBlur(crystal, (0, 0), sigmaX=crowd_sigma)
    ferns = np.clip(1.4 * crystal + 0.5 * crowd, 0.0, 1.6)
    depth: np.ndarray = thickness * np.maximum((1.0 - 0.6 * feather) * layer, feather * ferns)
    return depth.astype(np.float32)


class FrostOnGlass(CameraTransform):
    """A border of frost on the dome, grown in from the rim.

    Needs no depth. Under the infrared ring, set ``illuminator_glow_range``
    and apply it after ``NightIrMode``, as with ``WaterDroplets``.

    Args:
        coverage_range: how far the frost has grown in from the rim, as a
            share of the way to the centre of the frame.
        thickness_range: optical depth of established frost. At 1 it passes
            a third of the scene directly; at 3, a twentieth.
        feather_range: how much of the frost is in ferns growing inwards,
            0 to 1. Zero is an even layer with a ragged edge.
        crystal_scale_range: thickness of a fern's stem where it starts, as
            a fraction of the diagonal. The glass is centimetres from the
            lens, so a stem a millimetre thick is several pixels.
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
        crystal_scale_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.001, 0.05)),
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
        thickness_range: tuple[float, float] = (0.8, 3.0),
        feather_range: tuple[float, float] = (0.4, 1.0),
        crystal_scale_range: tuple[float, float] = (0.006, 0.014),
        defocus_range: tuple[float, float] = (0.002, 0.006),
        ambient_light_range: tuple[float, float] = (0.05, 0.3),
        illuminator_glow_range: tuple[float, float] = (0.0, 0.0),
        p: float = 0.5,
    ):
        """Store the ranges; each argument is described in the class docstring."""
        super().__init__(p=p)
        self.coverage_range = coverage_range
        self.thickness_range = thickness_range
        self.feather_range = feather_range
        self.crystal_scale_range = crystal_scale_range
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
        crystal_scale = self.py_random.uniform(*self.crystal_scale_range)
        sigma = self.py_random.uniform(*self.defocus_range) * diagonal

        depth = frost_depth(
            self.random_generator,
            height,
            width,
            coverage=coverage,
            thickness=thickness,
            feather=feather,
            crystal_scale=crystal_scale,
        )
        scattered = -np.expm1(-depth)
        if sigma > 0.3:
            scattered = wide_blur(scattered, sigma)
        record_draw(
            coverage=coverage,
            thickness=thickness,
            feather=feather,
            crystal_scale=crystal_scale,
            defocus_sigma_px=sigma,
            frosted_fraction=lambda: float((scattered > 0.5).mean()),
        )
        return {
            "scattered": np.clip(scattered, 0.0, 1.0),
            "ambient": self.py_random.uniform(*self.ambient_light_range),
            "glow": self.py_random.uniform(*self.illuminator_glow_range),
        }

    def legibility_planes(
        self, params: dict[str, Any], before: np.ndarray, after: np.ndarray
    ) -> dict[str, Any] | None:
        """What the frost replaces by a wide blur: the share of the light it scatters."""
        del before, after
        return {"hidden": params["scattered"]}

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
