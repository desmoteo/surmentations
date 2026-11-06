"""Hard cast shadows — the classic surveillance false positive.

Physical model
--------------
A moving cast shadow is the single most reliable way to make a detector or a
background-subtraction model produce a false alarm. It is a sharply bounded,
person- or vehicle-shaped, *moving* dark region, which is nearly the
definition of what such a system is looking for. Shadows from clouds, trees
and passing vehicles sweep across a fixed camera's view all day.

Two things must be right or a shadow reads as a grey decal laid over the
image:

**A shadow is a loss of the direct beam, not a coat of grey paint.** Outdoor
daylight has two components: the direct sun, and the diffuse light of the
whole sky dome. A shadow removes the first and leaves the second, so the
shadowed region keeps its texture and its albedo and merely receives less
light. Multiplying the radiance by the surviving fraction preserves all that,
whereas blending toward grey flattens texture and washes colour out — the
tell-tale of a drawn-on shadow.

It also changes the *colour*. What remains is skylight, which is blue, so real
daylight shadows are measurably bluer than the sunlit areas around them. That
is why the shadow colour here is a per-channel fraction rather than a scalar.

**A shadow falls on surfaces, not on the frame.** It belongs on the ground and
on upward-facing surfaces, and it cannot land on the sky. The upward-facing
term already recovered from the depth map for snow accumulation is exactly the
right quantity, so it is reused — the two effects ask the same geometric
question and only differ in what they do with the answer.

Penumbra: the sun is a disc of about half a degree, not a point, so a shadow
edge is not sharp. Its softness grows with the distance between the occluder
and the surface — a shadow at someone's feet is crisp, the same shadow's head
is noticeably soft. That gradient is modelled, since a uniformly hard or
uniformly soft shadow both read as fake.
"""

from __future__ import annotations

import math
from typing import Annotated, Any

import cv2
import numpy as np
from albumentations.core.pydantic import check_range_bounds, nondecreasing
from albumentations.core.transforms_interface import BaseTransformInitSchema
from pydantic import AfterValidator, Field

from weathermentations.base import WeatherTransform, require_depth_provider, scene_of
from weathermentations.depth.normals import upward_facing
from weathermentations.depth.provider import SharedDepthProvider
from weathermentations.diagnostics import record_draw
from weathermentations.utils.photometry import from_linear_unit, to_linear_unit

# Relative spectrum of skylight against direct sun, normalised on green. What
# survives in shadow is the sky, so a daylight shadow is bluer than its
# surroundings — a shadow rendered as neutral grey is missing this.
_SKYLIGHT_RATIO = np.array([0.82, 0.94, 1.18], dtype=np.float32)


#: What casts the shadows. A fixed camera's false alarms come from all three.
SHADOW_CASTERS = ("person", "vehicle", "foliage")


def _ellipse(cx: float, cy: float, rx: float, ry: float, points: int = 20) -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * np.pi, points, endpoint=False)
    return np.stack([cx + rx * np.cos(theta), cy + ry * np.sin(theta)], axis=1)


def _limb(start: tuple[float, float], end: tuple[float, float], half_width: float) -> np.ndarray:
    (x0, y0), (x1, y1) = start, end
    return np.array(
        [
            (x0 - half_width, y0),
            (x0 + half_width, y0),
            (x1 + half_width, y1),
            (x1 - half_width, y1),
        ]
    )


def _person_outline(rng: np.random.Generator) -> list[np.ndarray]:
    """A standing figure as convex parts, feet at ``y = 0`` and head at ``y = 1``.

    ``x`` is in units of the figure's height. The parts overlap, so filled
    together they read as one silhouette with a head, a narrower neck and two
    legs — the features that make a detector treat a shadow as a person, and
    that an ellipse does not have.
    """
    stride = float(rng.uniform(0.02, 0.11))
    swing = float(rng.uniform(-0.05, 0.05))
    return [
        _ellipse(0.0, 0.925, 0.062, 0.075),  # head
        _ellipse(0.0, 0.68, 0.125, 0.19),  # torso
        _ellipse(0.0, 0.80, 0.15, 0.045),  # shoulders, which carry the arms
        _ellipse(0.0, 0.50, 0.105, 0.07),  # hips
        _limb((-0.055, 0.50), (-stride, 0.0), 0.042),  # legs
        _limb((0.055, 0.50), (stride, 0.0), 0.042),
        _limb((-0.125, 0.80), (-0.16 + swing, 0.46), 0.03),  # arms
        _limb((0.125, 0.80), (0.16 - swing, 0.46), 0.03),
    ]


def _rounded_box(half_width: float, half_length: float, points: int = 32) -> np.ndarray:
    """A superellipse: a rectangle with rounded corners, as a vehicle's plan."""
    theta = np.linspace(0.0, 2.0 * np.pi, points, endpoint=False)
    cos, sin = np.cos(theta), np.sin(theta)
    return np.stack(
        [
            half_width * np.sign(cos) * np.abs(cos) ** 0.4,
            half_length * np.sign(sin) * np.abs(sin) ** 0.4,
        ],
        axis=1,
    )


def _fill(mask: np.ndarray, polygon: np.ndarray) -> None:
    cv2.fillPoly(mask, [np.round(polygon).astype(np.int32)], 1.0, lineType=cv2.LINE_AA)


def _draw_cast_shape(
    mask: np.ndarray,
    rng: np.random.Generator,
    width: int,
    height: int,
    scale: float,
    kind: str = "foliage",
    sun_azimuth_deg: float | None = None,
    elongation: float = 1.0,
) -> None:
    """One cast shadow, thrown along the sun's azimuth.

    ``scale`` sets the size of the thing casting it, in pixels: a blob of
    foliage is a few ``scale`` long, a vehicle a little over one wide, a
    figure two tall. Sized that way a vehicle's shadow covers about what a
    blob's does and a figure's a fifth of it, which keeps most of the
    effect's strength while two shadows in three have an outline.
    ``elongation``
    is ``1 / tan(sun elevation)``: how many times its own height an upright
    object's shadow is long. Both it and the azimuth belong to the *sun*, so
    every shadow in a frame shares them — which is the difference between
    shadows and dark shapes scattered at random angles.
    """
    azimuth = float(rng.uniform(0.0, 360.0)) if sun_azimuth_deg is None else sun_azimuth_deg
    along = np.array([math.cos(math.radians(azimuth)), math.sin(math.radians(azimuth))])
    across = np.array([-along[1], along[0]])
    origin = np.array(
        [float(rng.uniform(-0.1, 1.1)) * width, float(rng.uniform(0.25, 1.05)) * height]
    )

    if kind == "person":
        # An upright figure: its width is its own, its length is its height
        # stretched by the sun.
        figure_height = 2.0 * scale
        shadow_length = figure_height * elongation
        for part in _person_outline(rng):
            placed = (
                origin
                + part[:, 0:1] * figure_height * across
                + part[:, 1:2] * shadow_length * along
            )
            _fill(mask, placed)
        return

    if kind == "vehicle":
        # The plan of the vehicle, on whatever heading the road gives it,
        # swept along the sun's direction by its height. A sweep of a convex
        # outline is the convex hull of its two ends.
        half_width = scale * float(rng.uniform(0.5, 0.7))
        half_length = half_width * float(rng.uniform(1.9, 2.5))
        heading = math.radians(float(rng.uniform(0.0, 180.0)))
        rotation = np.array(
            [[math.cos(heading), -math.sin(heading)], [math.sin(heading), math.cos(heading)]]
        )
        plan = _rounded_box(half_width, half_length) @ rotation.T + origin
        thrown = plan + along * (1.5 * half_width * elongation)
        hull = cv2.convexHull(np.vstack([plan, thrown]).astype(np.float32))
        _fill(mask, hull.reshape(-1, 2))
        return

    # Foliage, or anything else without a recognisable outline: an elongated
    # blob with a few lobes, since real cast shadows come from limbs and
    # crowns rather than from ellipses.
    length = scale * float(rng.uniform(1.0, 2.6))
    breadth = scale * float(rng.uniform(0.25, 0.6))
    angle = azimuth + float(rng.uniform(-15.0, 15.0))
    cv2.ellipse(
        mask,
        (int(origin[0]), int(origin[1])),
        (max(1, int(length)), max(1, int(breadth))),
        angle,
        0,
        360,
        1.0,
        thickness=-1,
    )
    for _ in range(int(rng.integers(1, 4))):
        theta = math.radians(angle) + float(rng.uniform(-0.9, 0.9))
        offset = length * float(rng.uniform(0.3, 0.9))
        cv2.circle(
            mask,
            (int(origin[0] + offset * math.cos(theta)), int(origin[1] + offset * math.sin(theta))),
            max(1, int(breadth * rng.uniform(0.5, 1.1))),
            1.0,
            -1,
        )


class CastShadows(WeatherTransform):
    """Hard-edged cast shadows on upward-facing surfaces.

    Args:
        depth_provider: shared :class:`~weathermentations.depth.provider.SharedDepthProvider`.
        shadow_count_range: how many cast shadows to place.
        shadow_scale_range: size of the things casting them, as a fraction of
            the image diagonal.
        casters: what casts the shadows, any of ``"person"``, ``"vehicle"``
            and ``"foliage"``; each shadow picks one. A figure and a vehicle
            have outlines a detector recognises, which is what makes their
            shadows false alarms rather than merely dark patches.
        sun_elevation_range_deg: height of the sun, drawn per frame and shared
            by every shadow in it together with one azimuth. A low sun throws
            long shadows: an upright object's is ``1 / tan(elevation)`` times
            its height.
        direct_fraction_range: how much of the total illumination is the
            direct beam, which is what the shadow removes. 0.85 is hard
            sunlight and gives a deep shadow; 0.3 is hazy light and a faint
            one. Overcast approaches 0, where cast shadows disappear entirely
            — which is correct, and the reason this is a physical parameter
            rather than a "shadow darkness" dial.
        penumbra_range: softness of the shadow edge as a fraction of the
            diagonal, from the sun's angular diameter.
        surface_sharpness: exponent on the upward-facing term. The cosine
            alone is how much light a surface *would* lose, but it does not
            fall off fast enough to keep shadows off things that are not
            really ground: with the camera pitched down even a fronto-parallel
            sky plane picks up a small upward component (0.14 at 8 degrees of
            pitch), which a linear term leaves as a faint shadow drawn across
            the sky. Raising it to a power suppresses that — the same
            threshold argument as snow accumulation's ``accumulation_falloff``,
            for the same reason.
        camera_pitch_deg: how far the camera is tilted down, which orients
            "up" for the surface test.
        horizontal_fov_deg: field of view, for the same back-projection.

    Targets:
        image

    Image types:
        uint8, float32
    """

    class InitSchema(BaseTransformInitSchema):
        depth_provider: SharedDepthProvider
        shadow_count_range: Annotated[
            tuple[int, int],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0, 20)),
        ]
        shadow_scale_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.01, 0.6)),
        ]
        direct_fraction_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 0.95)),
        ]
        penumbra_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 0.05)),
        ]
        casters: tuple[str, ...]
        sun_elevation_range_deg: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(5.0, 85.0)),
        ]
        surface_sharpness: float = Field(ge=1.0, le=12.0)
        camera_pitch_deg: float = Field(ge=-89.0, le=89.0)
        horizontal_fov_deg: float = Field(gt=1.0, lt=179.0)

    def __init__(
        self,
        depth_provider: SharedDepthProvider,
        shadow_count_range: tuple[int, int] = (1, 5),
        shadow_scale_range: tuple[float, float] = (0.06, 0.3),
        direct_fraction_range: tuple[float, float] = (0.45, 0.88),
        penumbra_range: tuple[float, float] = (0.002, 0.02),
        casters: tuple[str, ...] = SHADOW_CASTERS,
        sun_elevation_range_deg: tuple[float, float] = (20.0, 65.0),
        surface_sharpness: float = 4.0,
        camera_pitch_deg: float = 8.0,
        horizontal_fov_deg: float = 60.0,
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.depth_provider = require_depth_provider(depth_provider, cls_name=type(self).__name__)
        self.shadow_count_range = shadow_count_range
        self.shadow_scale_range = shadow_scale_range
        self.direct_fraction_range = direct_fraction_range
        self.penumbra_range = penumbra_range
        unknown = sorted(set(casters) - set(SHADOW_CASTERS))
        if unknown or not casters:
            raise ValueError(
                f"casters must be a non-empty selection of {SHADOW_CASTERS}, got {casters!r}"
            )
        self.casters = tuple(casters)
        self.sun_elevation_range_deg = sun_elevation_range_deg
        self.surface_sharpness = surface_sharpness
        self.camera_pitch_deg = camera_pitch_deg
        self.horizontal_fov_deg = horizontal_fov_deg

    def get_params_dependent_on_data(
        self, params: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, Any]:
        scene = scene_of(data)
        height, width = params["shape"][:2]
        diagonal = float(np.hypot(width, height))
        rng = self.random_generator

        # One sun per frame. Every shadow is thrown the same way and stretched
        # by the same amount; drawing each at its own angle, as this did,
        # describes a scene lit from several directions at once.
        sun_azimuth = self.py_random.uniform(0.0, 360.0)
        sun_elevation = self.py_random.uniform(*self.sun_elevation_range_deg)
        elongation = 1.0 / math.tan(math.radians(sun_elevation))

        shapes = np.zeros((height, width), dtype=np.float32)
        shadow_count = self.py_random.randint(*self.shadow_count_range)
        for _ in range(shadow_count):
            _draw_cast_shape(
                shapes,
                rng,
                width,
                height,
                self.py_random.uniform(*self.shadow_scale_range) * diagonal,
                kind=self.py_random.choice(self.casters),
                sun_azimuth_deg=sun_azimuth,
                elongation=elongation,
            )

        penumbra = self.py_random.uniform(*self.penumbra_range) * diagonal
        if penumbra > 0.5:
            shapes = cv2.GaussianBlur(shapes, (0, 0), sigmaX=penumbra)

        # A shadow lands on surfaces, not on the frame — and never on sky.
        surface = upward_facing(
            self.depth_provider.get_metric_depth(scene),
            horizontal_fov_deg=self.horizontal_fov_deg,
            camera_pitch_deg=self.camera_pitch_deg,
        )
        occlusion = np.clip(shapes, 0.0, 1.0) * surface**self.surface_sharpness

        record_draw(
            sun_azimuth_deg=sun_azimuth,
            sun_elevation_deg=sun_elevation,
            shadow_count=shadow_count,
            penumbra_px=penumbra,
            surface_fraction=float((surface > 0.5).mean()),
            shadowed_fraction=float((occlusion > 0.1).mean()),
        )

        return {
            "occlusion": occlusion.astype(np.float32),
            "direct_fraction": self.py_random.uniform(*self.direct_fraction_range),
        }

    def apply(
        self,
        img: np.ndarray,
        occlusion: np.ndarray,
        direct_fraction: float,
        **params: Any,
    ) -> np.ndarray:
        if float(occlusion.max()) < 1e-3:
            return img

        linear, scale, is_integer = to_linear_unit(img)

        # Remove the direct component where the shadow falls; the diffuse
        # skylight survives. Multiplying keeps albedo and texture intact,
        # which blending toward grey would destroy.
        blocked = occlusion[..., None] * direct_fraction
        surviving = 1.0 - blocked + blocked * _SKYLIGHT_RATIO[None, None, :]
        # `surviving` carries the skylight's own colour, so what is left is
        # both dimmer and bluer, as a real daylight shadow is.
        out = linear * (1.0 - blocked) * surviving

        return from_linear_unit(np.clip(out, 0.0, 1.0), scale, is_integer, img.dtype)
