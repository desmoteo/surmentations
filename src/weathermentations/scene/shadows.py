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


def _draw_cast_shape(
    mask: np.ndarray, rng: np.random.Generator, width: int, height: int, scale: float
) -> None:
    """One cast shadow: an elongated blob stretched along the sun's azimuth."""
    cx = float(rng.uniform(-0.1, 1.1)) * width
    cy = float(rng.uniform(0.25, 1.05)) * height
    length = scale * float(rng.uniform(1.0, 2.6))
    breadth = scale * float(rng.uniform(0.25, 0.6))
    angle = float(rng.uniform(0.0, 180.0))

    cv2.ellipse(
        mask,
        (int(cx), int(cy)),
        (max(1, int(length)), max(1, int(breadth))),
        angle,
        0,
        360,
        1.0,
        thickness=-1,
    )
    # A couple of lobes so the outline is not a bare ellipse: real cast
    # shadows come from limbs, wing mirrors, foliage.
    for _ in range(int(rng.integers(1, 4))):
        theta = math.radians(angle) + float(rng.uniform(-0.9, 0.9))
        offset = length * float(rng.uniform(0.3, 0.9))
        cv2.circle(
            mask,
            (int(cx + offset * math.cos(theta)), int(cy + offset * math.sin(theta))),
            max(1, int(breadth * rng.uniform(0.5, 1.1))),
            1.0,
            -1,
        )


class CastShadows(WeatherTransform):
    """Hard-edged cast shadows on upward-facing surfaces.

    Args:
        depth_provider: shared :class:`~weathermentations.depth.provider.SharedDepthProvider`.
        shadow_count_range: how many cast shadows to place.
        shadow_scale_range: their size as a fraction of the image diagonal.
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

        shapes = np.zeros((height, width), dtype=np.float32)
        shadow_count = self.py_random.randint(*self.shadow_count_range)
        for _ in range(shadow_count):
            _draw_cast_shape(
                shapes,
                rng,
                width,
                height,
                self.py_random.uniform(*self.shadow_scale_range) * diagonal,
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
