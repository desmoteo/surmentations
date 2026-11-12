"""A branch that has grown in front of the camera.

Physical model
--------------
Cameras are installed in winter and by July there is a branch across the
view. It hangs a few tens of centimetres from the lens, moves in any wind,
and is the commonest cause of both lost coverage and false motion alarms on
an outdoor camera that is otherwise working.

**It is near, so it is large and soft.** A 6 cm leaf half a metre from a
4 mm lens is a fifth of the frame wide. It is inside the near focus limit,
so its edge is blurred by the aperture's disc, ``A f / 2z`` pixels in
radius: a few pixels by day, when the picture is exposed for daylight
through a small stop, and several times that at night with the lens open.

**It hides what is behind it.** A leaf is opaque enough: where it is, the
scene is not. Where its blurred edge is, the two mix.

**By day it is dark; by night it is white.** By day the camera sees the
underside of leaves against a brighter scene: green, at a fraction of the
light the scene is exposed for, more where the sun comes through a leaf
and less in the shade of the others. At night the infrared ring lights it
from a hand's width away, and leaves reflect about half of the near
infrared that reaches them, where they reflect a twentieth of red light.
So a branch that is a dark shape by day is the brightest thing in the
night picture, by the lamp's inverse square at its own distance.

**It moves.** Wind sways it during the exposure, which at night is long,
so the leaves are smeared along the sway by its speed times the shutter
time, as seen from where they are.
"""

from __future__ import annotations

import math
from typing import Annotated, Any

import cv2
import numpy as np
from albumentations.core.pydantic import check_range_bounds, nondecreasing
from albumentations.core.transforms_interface import BaseTransformInitSchema
from pydantic import AfterValidator

from weathermentations.base import WeatherTransform
from weathermentations.diagnostics import record_draw
from weathermentations.lens.water_droplets import _aperture_blur
from weathermentations.utils.color import illuminant_radiance_linear
from weathermentations.utils.compositing import directional_blur_kernel
from weathermentations.utils.illuminator import lamp_falloff
from weathermentations.utils.photometry import from_linear_unit, to_linear_unit
from weathermentations.utils.precipitation import focal_length_px

# A leaf in visible light, linear: green, and dark.
_LEAF_ALBEDO = np.array([0.055, 0.13, 0.035], dtype=np.float32)
# Bark.
_TWIG_ALBEDO = np.array([0.06, 0.05, 0.04], dtype=np.float32)
# What leaves and bark return of the near infrared.
_LEAF_NIR, _TWIG_NIR = 0.5, 0.25
# Where ``illuminator_gain_range`` is stated.
_GAIN_DISTANCE_M = 1.0


def _leaf(centre: np.ndarray, along: np.ndarray, length: float) -> np.ndarray:
    """The outline of a leaf: pointed at both ends, a third as wide as it is long."""
    across = np.array([-along[1], along[0]])
    t = np.linspace(0.0, 1.0, 9)
    # Widest a little before the middle, as most leaves are.
    half_width = 0.33 * length * np.sin(np.pi * t**0.8) ** 0.8
    spine = centre[None, :] + (t[:, None] - 0.5) * length * along[None, :]
    outline: np.ndarray = np.concatenate(
        [spine + half_width[:, None] * across, (spine - half_width[:, None] * across)[::-1]]
    )
    return outline


def draw_branch(
    rng: np.random.Generator,
    layer: np.ndarray,
    cover: np.ndarray,
    *,
    leaf_px: float,
    reach_px: float,
    light: tuple[float, float],
    leaf_colour: np.ndarray,
    twig_colour: np.ndarray,
) -> int:
    """Draw one branch into ``layer`` (colour) and ``cover`` (how much it hides).

    It enters from the top or a side and runs ``reach_px`` into the frame,
    bending, with twigs off it to either side and leaves along all of it,
    ``leaf_px`` long. Each leaf's colour is ``leaf_colour`` times a draw
    from ``light``, a ``(low, high)`` pair: leaves are not evenly lit.
    Returns how many leaves were drawn.
    """
    height, width = cover.shape
    side = rng.choice(4, p=[0.5, 0.2, 0.2, 0.1])
    if side == 0:
        start, inward = np.array([rng.uniform(0, width), -leaf_px]), 0.5 * math.pi
    elif side == 1:
        start, inward = np.array([-leaf_px, rng.uniform(0, 0.7 * height)]), 0.0
    elif side == 2:
        start, inward = np.array([width + leaf_px, rng.uniform(0, 0.7 * height)]), math.pi
    else:
        start, inward = np.array([rng.uniform(0, width), height + leaf_px]), -0.5 * math.pi
    heading = inward + rng.uniform(-0.6, 0.6)

    def twig(origin: np.ndarray, direction: float, length: float, thickness: float) -> list:
        """A bending twig as a list of points, drawn into both layers."""
        points = [origin]
        steps = max(2, int(length / (0.8 * leaf_px)))
        for _ in range(steps):
            direction += rng.uniform(-0.25, 0.25)
            step = (length / steps) * np.array([math.cos(direction), math.sin(direction)])
            points.append(points[-1] + step)
        line = np.rint(points).astype(np.int32)
        width_px = max(1, int(round(thickness)))
        cv2.polylines(cover, [line], False, 1.0, width_px, cv2.LINE_AA)
        cv2.polylines(layer, [line], False, twig_colour.tolist(), width_px, cv2.LINE_AA)
        return points

    stems = [twig(start, heading, reach_px, 0.09 * leaf_px)]
    for point in stems[0][1:-1]:
        if rng.uniform() < 0.7:
            turn = rng.choice([-1.0, 1.0]) * rng.uniform(0.5, 1.1)
            stems.append(
                twig(point, heading + turn, reach_px * rng.uniform(0.2, 0.5), 0.05 * leaf_px)
            )

    leaves = 0
    for stem in stems:
        for index, point in enumerate(stem[1:], start=1):
            for sign in (-1.0, 1.0):
                if rng.uniform() < 0.25:
                    continue
                before = stem[index - 1]
                run = math.atan2(point[1] - before[1], point[0] - before[0])
                angle = run + sign * rng.uniform(0.5, 1.2)
                direction = np.array([math.cos(angle), math.sin(angle)])
                length = leaf_px * rng.uniform(0.7, 1.2)
                outline = np.rint(_leaf(point + 0.55 * length * direction, direction, length))
                colour = leaf_colour * rng.uniform(*light)
                points = outline.astype(np.int32)
                cv2.fillPoly(cover, [points], 1.0, cv2.LINE_AA)
                cv2.fillPoly(layer, [points], colour.tolist(), cv2.LINE_AA)
                # One half of the blade a little darker than the other, and
                # a paler midrib: a leaf is folded along it, not flat.
                half = len(points) // 2
                fold = np.concatenate([points[:half], points[[half - 1, 0]]])
                cv2.fillPoly(layer, [fold], (colour * rng.uniform(0.6, 0.9)).tolist(), cv2.LINE_AA)
                cv2.line(
                    layer,
                    tuple(int(v) for v in points[0]),
                    tuple(int(v) for v in points[half - 1]),
                    (colour * 1.5).tolist(),
                    max(1, int(0.03 * length)),
                    cv2.LINE_AA,
                )
                leaves += 1
    return leaves


class FoliageIntrusion(WeatherTransform):
    """A leafy branch hanging in front of the lens, out of focus and swaying.

    Needs no depth: the branch is in front of everything. Under the
    infrared ring, set ``illuminator_gain_range``, set ``leaf_light_range``
    to zero, since there is no daylight for the leaves to be green in, and
    apply it after ``NightIrMode``.

    Args:
        branch_count_range: how many branches reach into the view.
        distance_range_m: how far from the lens they hang. Sets how large
            the leaves are, how blurred, and how brightly the lamp lights
            them.
        leaf_length_range_mm: the length of a leaf.
        reach_range: how far into the frame a branch runs, as a fraction of
            the frame's smaller side.
        leaf_light_range: how brightly a leaf's underside shows by day, as
            a share of what its colour would return in the open: low in the
            shade of the leaves above it, higher where the sun comes
            through. Each leaf draws its own.
        sway_speed_range_m_s: how fast the wind moves the branch, which
            with the shutter time smears it.
        exposure_time_range: shutter time.
        illuminator_gain_range: radiance of a white surface one metre from
            the infrared ring, in units of full scale; leaves return half
            of it. Zero by day.
        horizontal_fov_range_deg: field of view, which projects metres to
            pixels.
        aperture_diameter_range_mm: the lens's stop, which sets the blur.
        focus_distance_range_m: where the lens is focused.

    Targets:
        image

    Image types:
        uint8, float32
    """

    class InitSchema(BaseTransformInitSchema):
        """The arguments of ``FoliageIntrusion`` as Albumentations validates them: each
        range ordered low to high and within the bounds given here.
        """

        branch_count_range: Annotated[
            tuple[int, int],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0, 12)),
        ]
        distance_range_m: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.05, 20.0)),
        ]
        leaf_length_range_mm: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(5.0, 400.0)),
        ]
        reach_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.02, 2.0)),
        ]
        leaf_light_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 2.0)),
        ]
        sway_speed_range_m_s: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 5.0)),
        ]
        exposure_time_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(1e-5, 0.5)),
        ]
        illuminator_gain_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 50.0)),
        ]
        horizontal_fov_range_deg: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(10.0, 150.0)),
        ]
        aperture_diameter_range_mm: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.2, 20.0)),
        ]
        focus_distance_range_m: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.5, 200.0)),
        ]

    def __init__(
        self,
        branch_count_range: tuple[int, int] = (1, 3),
        distance_range_m: tuple[float, float] = (0.2, 1.2),
        leaf_length_range_mm: tuple[float, float] = (30.0, 90.0),
        reach_range: tuple[float, float] = (0.25, 0.8),
        leaf_light_range: tuple[float, float] = (0.15, 0.9),
        sway_speed_range_m_s: tuple[float, float] = (0.0, 0.4),
        exposure_time_range: tuple[float, float] = (0.001, 0.04),
        illuminator_gain_range: tuple[float, float] = (0.0, 0.0),
        horizontal_fov_range_deg: tuple[float, float] = (48.0, 78.0),
        aperture_diameter_range_mm: tuple[float, float] = (0.9, 2.6),
        focus_distance_range_m: tuple[float, float] = (8.0, 40.0),
        p: float = 0.5,
    ):
        """Store the ranges; each argument is described in the class docstring."""
        super().__init__(p=p)
        self.branch_count_range = branch_count_range
        self.distance_range_m = distance_range_m
        self.leaf_length_range_mm = leaf_length_range_mm
        self.reach_range = reach_range
        self.leaf_light_range = leaf_light_range
        self.sway_speed_range_m_s = sway_speed_range_m_s
        self.exposure_time_range = exposure_time_range
        self.illuminator_gain_range = illuminator_gain_range
        self.horizontal_fov_range_deg = horizontal_fov_range_deg
        self.aperture_diameter_range_mm = aperture_diameter_range_mm
        self.focus_distance_range_m = focus_distance_range_m

    def get_params_dependent_on_data(
        self, params: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, Any]:
        """Draw the branches, each at its distance, blurred and smeared as it is seen.

        Returns ``cover``, how much of each pixel the foliage hides;
        ``reflectance``, its colour by day as a share of the illuminant,
        already times ``cover``; and ``lamp``, the infrared ring's light it
        returns, already times ``cover``.
        """
        del data
        height, width = params["shape"][:2]
        rng = self.random_generator

        branches = self.py_random.randint(*self.branch_count_range)
        gain = self.py_random.uniform(*self.illuminator_gain_range)
        exposure = self.py_random.uniform(*self.exposure_time_range)
        sway = self.py_random.uniform(*self.sway_speed_range_m_s)
        horizontal_fov = self.py_random.uniform(*self.horizontal_fov_range_deg)
        aperture_m = self.py_random.uniform(*self.aperture_diameter_range_mm) * 1e-3
        focus = self.py_random.uniform(*self.focus_distance_range_m)
        focal = focal_length_px(width, height, horizontal_fov)
        sway_angle = float(rng.uniform(60.0, 120.0))

        cover = np.zeros((height, width), dtype=np.float32)
        reflectance = np.zeros((height, width, 3), dtype=np.float32)
        lamp = np.zeros((height, width), dtype=np.float32)
        distances, leaves = [], 0
        # Far to near, so that a nearer branch hides a further one.
        for distance in sorted(
            (self.py_random.uniform(*self.distance_range_m) for _ in range(branches)), reverse=True
        ):
            one = np.zeros((height, width), dtype=np.float32)
            colour = np.zeros((height, width, 3), dtype=np.float32)
            leaf_px = self.py_random.uniform(*self.leaf_length_range_mm) * 1e-3 * focal / distance
            leaves += draw_branch(
                rng,
                colour,
                one,
                leaf_px=leaf_px,
                reach_px=self.py_random.uniform(*self.reach_range) * min(height, width),
                light=self.leaf_light_range,
                leaf_colour=_LEAF_ALBEDO,
                twig_colour=_TWIG_ALBEDO * float(np.mean(self.leaf_light_range)),
            )
            # Premultiplied, so that blurring mixes colour by coverage.
            colour *= one[..., None]
            blur = aperture_m * focal * abs(1.0 / distance - 1.0 / focus)
            smear = int(round(focal * sway * exposure / distance))
            stack = np.dstack([colour, one])
            stack = _aperture_blur(stack, blur)
            if smear >= 3:
                stack = cv2.filter2D(stack, -1, directional_blur_kernel(smear, sway_angle))
            one = np.clip(stack[..., 3], 0.0, 1.0)
            lit = gain * _LEAF_NIR * float(lamp_falloff(distance) / lamp_falloff(_GAIN_DISTANCE_M))
            reflectance = reflectance * (1.0 - one[..., None]) + stack[..., :3]
            lamp = lamp * (1.0 - one) + one * lit
            cover = cover + one * (1.0 - cover)
            distances.append(distance)

        record_draw(
            branch_count=branches,
            distance_m=np.array(distances, dtype=np.float32),
            leaf_count=leaves,
            exposure_s=exposure,
            sway_speed_m_s=sway,
            illuminator_gain=gain,
            focal_px=focal,
            aperture_mm=aperture_m * 1e3,
            covered_fraction=lambda: float((cover > 0.5).mean()),
        )
        return {"cover": cover, "reflectance": reflectance, "lamp": lamp}

    def apply(
        self,
        img: np.ndarray,
        cover: np.ndarray,
        reflectance: np.ndarray,
        lamp: np.ndarray,
        **params: Any,
    ) -> np.ndarray:
        """Hide the scene behind the foliage and show the foliage, lit, in its place."""
        if float(cover.max()) < 1e-3:
            return img
        linear, scale, is_integer = to_linear_unit(img)
        illuminant = illuminant_radiance_linear(linear)
        out = linear * (1.0 - cover[..., None]) + reflectance * illuminant[None, None, :]
        out += lamp[..., None]
        return from_linear_unit(out, scale, is_integer, img.dtype)
