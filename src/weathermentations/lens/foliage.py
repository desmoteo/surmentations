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

**A leaf is not a green shape.** What makes foliage read as foliage, at
the few pixels of blur it has by day, is that no two leaves look alike.
Each is turned its own way, so it is foreshortened along or across its
midrib; each is either catching light on the side the camera sees or
letting it through from behind, which are different colours, a dull blue
green and a bright yellow green; it has a midrib and veins, and one half
of the blade a little darker than the other; and the leaves in front
shade the ones behind. Under the lamp the same turning decides how bright
a leaf is, by the cosine of how far it faces away, and the nearer leaves
of a branch are brighter than the further ones.
"""

from __future__ import annotations

import math
from typing import Annotated, Any

import cv2
import numpy as np
from albumentations.core.pydantic import check_range_bounds, nondecreasing
from albumentations.core.transforms_interface import BaseTransformInitSchema
from pydantic import AfterValidator

from weathermentations.atmosphere.night_rain import _beam_profile
from weathermentations.base import WeatherTransform
from weathermentations.diagnostics import record_draw
from weathermentations.lens.water_droplets import _aperture_blur
from weathermentations.utils.color import illuminant_radiance_linear
from weathermentations.utils.compositing import directional_blur_kernel
from weathermentations.utils.illuminator import lamp_falloff, veiling_glare
from weathermentations.utils.photometry import from_linear_unit, to_linear_unit
from weathermentations.utils.precipitation import focal_length_px

# A leaf seen by the light it reflects, and by the light that comes through
# it, in linear light: a dull blue green, and a bright yellow green.
_LEAF_REFLECTS = np.array([0.045, 0.105, 0.035], dtype=np.float32)
_LEAF_PASSES = np.array([0.24, 0.42, 0.05], dtype=np.float32)
# Bark.
_TWIG_ALBEDO = np.array([0.06, 0.05, 0.04], dtype=np.float32)
# What leaves and bark return of the near infrared.
_LEAF_NIR, _TWIG_NIR = 0.5, 0.25
# Where ``illuminator_gain_range`` is stated.
_GAIN_DISTANCE_M = 1.0
# How much a leaf darkens the foliage just behind and below it.
_SHADE = 0.35
# The leaf every leaf is drawn from: this long, in pixels, and half as wide.
_LEAF_PX = 128


def _leaf_template() -> tuple[np.ndarray, np.ndarray]:
    """One leaf, tip to the right: how much of each pixel it covers, and its markings.

    ``(H, W)`` each. The outline is ovate, widest before the middle and
    pointed at the tip. The markings are a factor about one on its colour:
    a pale midrib, fainter veins running forward from it, one half of the
    blade a little darker than the other, and the edge a little darker
    than the middle.
    """
    width, height = _LEAF_PX, _LEAF_PX // 2
    u = (np.arange(width, dtype=np.float32)[None, :] + 0.5) / width
    v = ((np.arange(height, dtype=np.float32)[:, None] + 0.5) / height - 0.5) * 2.0
    half_width = np.sin(np.pi * u**0.72) ** 0.85 * (1.0 - 0.25 * u)
    across = np.abs(v) / np.maximum(half_width, 1e-3)
    cover = np.clip((1.0 - across) * half_width * height * 0.25, 0.0, 1.0)

    midrib = np.exp(-((v / 0.05) ** 2))
    # Veins leave the midrib every eighth of the leaf and run forwards.
    along = (u * 8.0 - np.abs(v) * 2.2) % 1.0
    veins = np.exp(-((np.minimum(along, 1.0 - along) / 0.09) ** 2)) * (1.0 - midrib)
    fold = np.where(v > 0.0, 0.86, 1.0)
    markings = fold * (1.0 - 0.18 * across**2) * (1.0 + 0.3 * midrib + 0.1 * veins)
    return cover.astype(np.float32), markings.astype(np.float32)


_LEAF_COVER, _LEAF_MARKINGS = _leaf_template()


def paste_leaf(
    layers: dict[str, np.ndarray],
    base: np.ndarray,
    direction: float,
    length_px: float,
    turned: tuple[float, float],
    colour: np.ndarray,
    lamp: float,
) -> None:
    """Draw one leaf over what is already in ``layers``, in place.

    Its stalk end is at ``base`` (x, y) and it points along ``direction``,
    ``length_px`` long before it is ``turned``: foreshortened to that share
    of its length and of its width, each 0 to 1. ``layers`` holds
    ``cover``, ``colour`` (premultiplied by cover) and ``lamp`` (likewise).
    The leaf first shades whatever foliage is just below it, then covers
    what is behind it.
    """
    height, width = layers["cover"].shape
    long, wide = length_px * turned[0], 0.5 * length_px * turned[1]
    cos, sin = math.cos(direction), math.sin(direction)
    reach = int(math.ceil(max(long, wide))) + 3
    centre = base + 0.5 * long * np.array([cos, sin])
    left, top = int(centre[0]) - reach, int(centre[1]) - reach
    x0, y0, x1, y1 = (
        max(0, left),
        max(0, top),
        min(width, left + 2 * reach),
        min(height, top + 2 * reach),
    )
    if x1 <= x0 or y1 <= y0 or long < 1.0 or wide < 0.5:
        return
    # Template pixels to the patch: scale, turn, and put the stalk at `base`.
    scale_x, scale_y = long / _LEAF_PX, wide / (_LEAF_PX // 2)
    matrix = np.array(
        [
            [cos * scale_x, -sin * scale_y, base[0] - x0 + sin * scale_y * (_LEAF_PX // 4)],
            [sin * scale_x, cos * scale_y, base[1] - y0 - cos * scale_y * (_LEAF_PX // 4)],
        ],
        dtype=np.float32,
    )
    size = (x1 - x0, y1 - y0)
    cover = cv2.warpAffine(_LEAF_COVER, matrix, size, flags=cv2.INTER_LINEAR)
    if float(cover.max()) <= 0.0:
        return
    markings = cv2.warpAffine(_LEAF_MARKINGS, matrix, size, flags=cv2.INTER_LINEAR, borderValue=1.0)
    region = (slice(y0, y1), slice(x0, x1))
    # The shadow it casts on the foliage behind: its own outline, a little
    # way down. Empty pixels hold no colour, so only foliage is darkened.
    drop = max(1, int(0.1 * length_px))
    shadow = np.zeros_like(cover)
    shadow[drop:] = cover[:-drop]
    dim = 1.0 - _SHADE * shadow
    layers["colour"][region] *= dim[..., None]
    layers["lamp"][region] *= dim

    clear = 1.0 - cover
    layers["colour"][region] = layers["colour"][region] * clear[..., None] + (
        (cover * markings)[..., None] * colour[None, None, :]
    )
    layers["lamp"][region] = layers["lamp"][region] * clear + cover * markings * lamp
    layers["cover"][region] = layers["cover"][region] * clear + cover


def draw_branch(
    rng: np.random.Generator,
    layers: dict[str, np.ndarray],
    *,
    leaf_px: float,
    reach_px: float,
    light: tuple[float, float],
    lamp: float,
) -> int:
    """Draw one branch into ``layers``, as :func:`paste_leaf` takes them, in place.

    It enters from the top or a side and runs ``reach_px`` into the frame,
    drooping, with twigs off it to either side and leaves on stalks along
    all of them, ``leaf_px`` long. ``light`` is the range a leaf's daylight
    is drawn from and ``lamp`` what a leaf facing the infrared ring
    returns of it. Returns how many leaves were drawn.
    """
    height, width = layers["cover"].shape
    side = rng.choice(4, p=[0.5, 0.2, 0.2, 0.1])
    if side == 0:
        start, inward = np.array([rng.uniform(0, width), -0.5 * leaf_px]), 0.5 * math.pi
    elif side == 1:
        start, inward = np.array([-0.5 * leaf_px, rng.uniform(0, 0.7 * height)]), 0.0
    elif side == 2:
        start, inward = np.array([width + 0.5 * leaf_px, rng.uniform(0, 0.7 * height)]), math.pi
    else:
        start, inward = np.array([rng.uniform(0, width), height + 0.5 * leaf_px]), -0.5 * math.pi
    heading = inward + rng.uniform(-0.6, 0.6)
    # Whether this branch is seen against the light, its leaves lit through,
    # or with the light on the side the camera sees.
    against_light = rng.uniform(0.0, 1.0)

    def twig(origin: np.ndarray, direction: float, length: float, thickness: float) -> list:
        """A drooping twig as a list of points, drawn into the layers, thinner along its length."""
        points = [origin]
        steps = max(3, int(length / (0.6 * leaf_px)))
        for step in range(steps):
            # Bends at random, and a little further downwards at every step.
            direction += rng.uniform(-0.2, 0.2) + 0.12 * math.cos(direction)
            stride = (length / steps) * np.array([math.cos(direction), math.sin(direction)])
            points.append(points[-1] + stride)
            a = tuple(int(round(v)) for v in points[-2])
            b = tuple(int(round(v)) for v in points[-1])
            width_px = max(1, int(round(thickness * (1.0 - 0.7 * step / steps))))
            bark = _TWIG_ALBEDO * float(np.mean(light))
            for name, value in (
                ("cover", 1.0),
                ("lamp", lamp * _TWIG_NIR / _LEAF_NIR),
            ):
                cv2.line(layers[name], a, b, value, width_px, cv2.LINE_AA)
            cv2.line(layers["colour"], a, b, bark.tolist(), width_px, cv2.LINE_AA)
        return points

    stems = [twig(start, heading, reach_px, 0.1 * leaf_px)]
    for index, point in enumerate(stems[0][1:-1], start=1):
        if rng.uniform() < 0.75:
            turn = (1.0 if index % 2 else -1.0) * rng.uniform(0.5, 1.0)
            stems.append(
                twig(point, heading + turn, reach_px * rng.uniform(0.2, 0.5), 0.05 * leaf_px)
            )

    leaves = []
    for stem in stems:
        for index, point in enumerate(stem[1:], start=1):
            before = stem[index - 1]
            run = math.atan2(point[1] - before[1], point[0] - before[0])
            # One leaf to either side of each joint, and one at the tip.
            sides = [-1.0, 1.0] + ([0.0] if index == len(stem) - 1 else [])
            for sign in sides:
                if sign and rng.uniform() < 0.15:
                    continue
                leaves.append(
                    (point, run + sign * rng.uniform(0.5, 1.1) + rng.uniform(-0.15, 0.15))
                )
    for index in rng.permutation(len(leaves)):
        point, angle = leaves[index]
        # A leaf hangs: its direction is pulled towards straight down.
        angle += 0.25 * math.cos(angle)
        turned = (float(rng.uniform(0.55, 1.0)), float(rng.uniform(0.3, 1.0)))
        lit_through = float(np.clip(against_light + rng.normal(0.0, 0.3), 0.0, 1.0))
        colour = (1.0 - lit_through) * _LEAF_REFLECTS + lit_through * _LEAF_PASSES
        colour = colour * rng.uniform(*light) * rng.uniform(0.9, 1.1, 3).astype(np.float32)
        stalk = 0.12 * leaf_px * np.array([math.cos(angle), math.sin(angle)])
        paste_leaf(
            layers,
            point + stalk,
            angle,
            leaf_px * float(rng.uniform(0.7, 1.2)),
            turned,
            colour.astype(np.float32),
            # Facing the lamp squarely it returns the most; turned away, less.
            lamp * turned[0] * turned[1] * float(rng.uniform(0.8, 1.25)),
        )
    return len(leaves)


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
        beam_falloff_range: how sharply the ring's beam dims off-axis.
        glare_range: fraction of the lamp-lit foliage's light that the dome
            and the lens scatter into a glow around it.
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
        beam_falloff_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 10.0)),
        ]
        glare_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 1.0)),
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
        beam_falloff_range: tuple[float, float] = (0.6, 2.2),
        glare_range: tuple[float, float] = (0.05, 0.25),
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
        self.beam_falloff_range = beam_falloff_range
        self.glare_range = glare_range
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
            layers = {
                "cover": np.zeros((height, width), dtype=np.float32),
                "colour": np.zeros((height, width, 3), dtype=np.float32),
                "lamp": np.zeros((height, width), dtype=np.float32),
            }
            leaf_px = self.py_random.uniform(*self.leaf_length_range_mm) * 1e-3 * focal / distance
            lit = gain * _LEAF_NIR * float(lamp_falloff(distance) / lamp_falloff(_GAIN_DISTANCE_M))
            leaves += draw_branch(
                rng,
                layers,
                leaf_px=leaf_px,
                reach_px=self.py_random.uniform(*self.reach_range) * min(height, width),
                light=self.leaf_light_range,
                lamp=lit,
            )
            # Colour and lamp are premultiplied by cover, so that blurring
            # mixes them by how much of each pixel is leaf.
            blur = aperture_m * focal * abs(1.0 / distance - 1.0 / focus)
            smear = int(round(focal * sway * exposure / distance))
            stack = np.dstack([layers["colour"], layers["cover"]])
            stack = _aperture_blur(stack, blur)
            lit_layer = _aperture_blur(layers["lamp"], blur)
            if smear >= 3:
                kernel = directional_blur_kernel(smear, sway_angle)
                stack = cv2.filter2D(stack, -1, kernel)
                lit_layer = cv2.filter2D(lit_layer, -1, kernel)
            one = np.clip(stack[..., 3], 0.0, 1.0)
            reflectance = reflectance * (1.0 - one[..., None]) + stack[..., :3]
            lamp = lamp * (1.0 - one) + lit_layer
            cover = cover + one * (1.0 - cover)
            distances.append(distance)

        if gain > 0.0:
            lamp = veiling_glare(lamp, self.py_random.uniform(*self.glare_range))
            lamp *= _beam_profile(height, width, self.py_random.uniform(*self.beam_falloff_range))

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
