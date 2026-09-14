"""Depth-conditioned snow: falling flakes plus a heuristic ground-accumulation overlay.

Physical model for the falling flakes mirrors ``atmosphere/rain.py``: each
flake's apparent size scales with distance (nearer flakes are bigger,
farther ones shrink toward a diffuse haze), and flakes drift at a slant
angle (wind) with a much smaller motion-blur elongation than rain streaks,
since snow falls roughly 3-5x slower than rain.

A near flake isn't just "bigger", though — a camera focused on a scene
several meters out renders anything within a few centimeters of the lens as
a large, very soft, translucent disc (defocus/"bokeh"), well beyond the
flake's true size, while a flake far enough to be near the focal plane stays
a small, comparatively crisp point. We render each flake as its own soft
radial sprite whose *softness and translucency*, not just its radius, scale
with how close it is (``depth_factor``): near flakes get a larger, gentler
falloff and lower peak opacity (the same total "light" spread over more
pixels), far flakes stay tight and crisp.

Ground accumulation is **not** modeled from first principles the way fog,
rain-veil, and the flakes themselves are: how much snow settles on a surface
actually depends on that surface's orientation and material (needs real
surface normals, which this library doesn't estimate), not just depth. We
render it as an explicitly-heuristic brightness/desaturation overlay on the
image's lower band — documented here as a heuristic, not a physical claim,
per the plan's validation strategy.
"""

from __future__ import annotations

from typing import Annotated, Any

import cv2
import numpy as np
from albumentations.core.pydantic import check_range_bounds, nondecreasing
from albumentations.core.transforms_interface import BaseTransformInitSchema
from pydantic import AfterValidator, Field

from weathermentations.base import WeatherTransform, require_depth_provider
from weathermentations.depth.provider import SharedDepthProvider
from weathermentations.depth.scale import to_pseudo_metric
from weathermentations.utils.color import sky_radiance_linear
from weathermentations.utils.compositing import directional_blur_kernel, feathered_band_blend
from weathermentations.utils.photometry import from_linear_unit, to_linear_unit

_SNOW_COLOR = np.array([245.0, 248.0, 252.0])
_ACCUMULATION_COLOR = np.array([235.0, 240.0, 245.0])


def _paste_flake_sprite(
    canvas: np.ndarray,
    x: float,
    y: float,
    size: float,
    depth_factor: float,
    bokeh_strength: float,
    blades: int,
    blade_rotation: float,
    rim_gain: float,
    cats_eye: float,
    frame_centre: tuple[float, float],
) -> None:
    """Max-blend one flake's sprite into ``canvas``.

    A defocused flake is not a Gaussian blob — it is an **image of the
    aperture**. That means a polygon with as many sides as the lens has
    blades, with an intensity profile set by spherical aberration (often
    bright-rimmed rather than flat), and, away from the optical axis, clipped
    into a "cat's eye" lemon by optical vignetting. Those three cues are what
    make out-of-focus points read as a photograph rather than as airbrushed
    dots.

    ``depth_factor`` (0=far, 1=near) drives the circle of confusion: near
    flakes become large, soft and *faint* (flux is conserved as the disc
    grows), far flakes stay small, crisp and closer to opaque.
    """
    base_radius = max(0.5, size / 2.0)
    coc = 1.0 + bokeh_strength * depth_factor
    render_radius = base_radius * coc
    patch_size = max(3, int(round(render_radius * 2.6)) | 1)

    yy, xx = np.mgrid[0:patch_size, 0:patch_size].astype(np.float32)
    centre = patch_size / 2.0
    dx = (xx - centre) / render_radius
    dy = (yy - centre) / render_radius

    if blades >= 3:
        # Signed distance to a regular polygon: max over the half-planes of
        # its edges. This is the aperture's own shape.
        angles = blade_rotation + 2.0 * np.pi * np.arange(blades) / blades
        dist = np.max(
            np.stack([dx * np.cos(a) + dy * np.sin(a) for a in angles], axis=0), axis=0
        ) / np.cos(np.pi / blades)
    else:
        dist = np.sqrt(dx**2 + dy**2)

    # Cat's-eye: optical vignetting clips off-axis bokeh with a second
    # aperture displaced towards the frame centre, and the further off-axis
    # the flake sits the more of it is cut away.
    if cats_eye > 0.0:
        to_centre = np.array([frame_centre[0] - x, frame_centre[1] - y], dtype=np.float32)
        norm = float(np.hypot(*to_centre))
        if norm > 1e-3:
            direction = to_centre / norm
            offset = cats_eye * min(1.0, norm / max(frame_centre[0], frame_centre[1]))
            clip = np.sqrt((dx - direction[0] * offset) ** 2 + (dy - direction[1] * offset) ** 2)
            dist = np.maximum(dist, clip)

    inside = np.clip(1.0 - dist, 0.0, 1.0)
    # Hard-ish edge (a defocus disc has a defined rim), then the aberration
    # profile: rim_gain > 1 brightens the edge, < 1 the centre.
    edge = np.clip(inside / 0.12, 0.0, 1.0)
    profile = 1.0 + (rim_gain - 1.0) * np.clip(1.0 - inside / 0.35, 0.0, 1.0)
    sprite = edge * profile

    # Spreading the disc dims it, but a nearer flake also subtends a larger
    # solid angle, so the two effects partly cancel: a pure 1/coc^2 point-
    # source falloff over-dims an extended object like a flake.
    sprite = sprite / (coc**1.2)

    ksize = max(1, int(render_radius * 0.12)) | 1
    if ksize > 1:
        sprite = cv2.GaussianBlur(sprite, (ksize, ksize), 0)

    h, w = canvas.shape[:2]
    x0 = int(round(x - centre))
    y0 = int(round(y - centre))
    dst_x0, dst_y0 = max(0, x0), max(0, y0)
    dst_x1, dst_y1 = min(w, x0 + patch_size), min(h, y0 + patch_size)
    if dst_x1 <= dst_x0 or dst_y1 <= dst_y0:
        return
    src_x0, src_y0 = dst_x0 - x0, dst_y0 - y0
    src_x1, src_y1 = src_x0 + (dst_x1 - dst_x0), src_y0 + (dst_y1 - dst_y0)

    region = canvas[dst_y0:dst_y1, dst_x0:dst_x1]
    np.maximum(region, sprite[src_y0:src_y1, src_x0:src_x1], out=region)


class RandomSnow(WeatherTransform):
    """Depth-scaled falling snowflakes with an optional, explicitly-heuristic
    ground-accumulation overlay.

    Args:
        depth_provider: shared :class:`~weathermentations.depth.provider.SharedDepthProvider`.
        flake_density_range: number of flakes to render.
        flake_size_range: apparent flake diameter for a flake right at the
            near plane, as a fraction of the image diagonal. Flakes further
            back in the volume are proportionally smaller.
        fall_angle_range: slant angle in degrees from vertical (wind).
        motion_blur_range: motion-blur kernel length as a fraction of a
            flake's own size (0 = crisp round flake, higher = more streaked).
        bokeh_strength_range: how much extra softness/size the *nearest*
            flakes get beyond their nominal ``flake_size_range`` radius, to
            mimic real defocus/bokeh discs (0 = no extra bokeh, flakes are
            just scaled circles; higher = near flakes become large, soft,
            translucent discs while far flakes stay crisp points).
        veil_beta_range: extinction coefficient for the snow veil. Snow
            scatters roughly 7x more than rain at the same precipitation
            rate, so heavy snowfall drops visibility hard — this is the
            single biggest cue that it is *snowing* rather than that white
            dots have been pasted on.
        depth_ratio: ratio between the farthest and nearest flake distance
            in the simulated volume. Higher means a deeper snow column, hence
            a larger spread between the tiny background flakes and the big
            near ones.
        aperture_blades: number of lens aperture blades. Defocused flakes are
            images of the aperture, so they take its polygonal shape; 0 gives
            perfectly circular bokeh.
        rim_gain_range: brightness of a bokeh disc's rim relative to its
            centre (>1 = bright-rimmed, as most modern fast lenses render).
        cats_eye_strength: how strongly off-axis bokeh is clipped into
            lemon shapes by optical vignetting.
        accumulation_probability: chance of rendering the ground-accumulation
            overlay.
        accumulation_strength_range: how strongly the ground band is pushed
            toward the accumulation color.
        accumulation_band_fraction: fraction of the image height (from the
            bottom) treated as "ground" for the accumulation pass.

    Targets:
        image

    Image types:
        uint8, float32
    """

    class InitSchema(BaseTransformInitSchema):
        depth_provider: SharedDepthProvider
        flake_density_range: Annotated[
            tuple[int, int],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0, 20000)),
        ]
        flake_size_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.001, 0.15)),
        ]
        fall_angle_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(-45.0, 45.0)),
        ]
        motion_blur_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 2.0)),
        ]
        bokeh_strength_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 4.0)),
        ]
        veil_beta_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 2.0)),
        ]
        depth_ratio: float = Field(gt=1.0, le=200.0)
        aperture_blades: int = Field(ge=0, le=12)
        rim_gain_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.5, 2.0)),
        ]
        cats_eye_strength: float = Field(ge=0.0, le=1.0)
        accumulation_probability: float = Field(ge=0.0, le=1.0)
        accumulation_strength_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 1.0)),
        ]
        accumulation_band_fraction: float = Field(ge=0.0, le=1.0)

    def __init__(
        self,
        depth_provider: SharedDepthProvider,
        flake_density_range: tuple[int, int] = (900, 2600),
        flake_size_range: tuple[float, float] = (0.03, 0.07),
        fall_angle_range: tuple[float, float] = (-5.0, 5.0),
        motion_blur_range: tuple[float, float] = (0.1, 0.4),
        bokeh_strength_range: tuple[float, float] = (0.8, 2.0),
        veil_beta_range: tuple[float, float] = (0.006, 0.045),
        depth_ratio: float = 18.0,
        aperture_blades: int = 7,
        rim_gain_range: tuple[float, float] = (0.95, 1.45),
        cats_eye_strength: float = 0.45,
        accumulation_probability: float = 0.4,
        accumulation_strength_range: tuple[float, float] = (0.2, 0.5),
        accumulation_band_fraction: float = 0.15,
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.depth_provider = require_depth_provider(depth_provider, cls_name=type(self).__name__)
        self.flake_density_range = flake_density_range
        self.flake_size_range = flake_size_range
        self.fall_angle_range = fall_angle_range
        self.motion_blur_range = motion_blur_range
        self.bokeh_strength_range = bokeh_strength_range
        self.veil_beta_range = veil_beta_range
        self.depth_ratio = depth_ratio
        self.aperture_blades = aperture_blades
        self.rim_gain_range = rim_gain_range
        self.cats_eye_strength = cats_eye_strength
        self.accumulation_probability = accumulation_probability
        self.accumulation_strength_range = accumulation_strength_range
        self.accumulation_band_fraction = accumulation_band_fraction

    def get_params_dependent_on_data(
        self, params: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, Any]:
        image = data["image"]
        height, width = params["shape"][:2]

        relative_depth = self.depth_provider.get_depth(image)
        depth_pm = to_pseudo_metric(relative_depth, None)

        count = self.py_random.randint(*self.flake_density_range)
        angle = self.py_random.uniform(*self.fall_angle_range)
        motion_blur_amount = self.py_random.uniform(*self.motion_blur_range)
        bokeh_strength = self.py_random.uniform(*self.bokeh_strength_range)
        depth_ratio = self.depth_ratio
        base_size = self.py_random.uniform(*self.flake_size_range) * float(np.hypot(width, height))

        # A falling flake hangs in the air *in front of* the scene, so its
        # distance has nothing to do with the scene depth behind it. Sizing
        # flakes from the depth map (the previous approach) makes them look
        # pasted onto surfaces — big over the near road, tiny over the distant
        # sky — which is not how snowfall looks. Instead each flake gets its
        # own distance, sampled so the flakes fill the *volume* in front of
        # the lens: shells at distance d hold ~d^2 more flakes while
        # projecting ~1/d smaller, which is what produces a field of many
        # tiny distant flakes with a few big near ones instead of confetti.
        flakes = []
        for _ in range(count):
            x = self.py_random.uniform(0, width)
            y = self.py_random.uniform(0, height)
            # Volume-uniform distance: the number of flakes closer than d
            # grows as d^3, so sampling d = (1 + u*(ratio^3 - 1))^(1/3) puts
            # most of them far away. Apparent size then goes as 1/d, which is
            # what gives a dense haze of tiny distant flakes punctuated by a
            # handful of big near ones.
            u = self.py_random.random()
            distance = (1.0 + u * (depth_ratio**3 - 1.0)) ** (1.0 / 3.0)
            nearness = 1.0 / distance
            size = base_size * nearness
            flakes.append((x, y, size, nearness))

        draw_accumulation = self.py_random.random() < self.accumulation_probability
        accumulation_strength = (
            self.py_random.uniform(*self.accumulation_strength_range) if draw_accumulation else 0.0
        )

        veil_beta = self.py_random.uniform(*self.veil_beta_range)

        return {
            "flakes": flakes,
            "angle": angle,
            "motion_blur_amount": motion_blur_amount,
            "bokeh_strength": bokeh_strength,
            "blade_rotation": self.py_random.uniform(0, 2 * np.pi),
            "rim_gain": self.py_random.uniform(*self.rim_gain_range),
            "veil_transmission": np.exp(-veil_beta * depth_pm).astype(np.float32),
            "draw_accumulation": draw_accumulation,
            "accumulation_strength": accumulation_strength,
        }

    def apply(
        self,
        img: np.ndarray,
        flakes: list[tuple[float, float, float, float]],
        angle: float,
        motion_blur_amount: float,
        bokeh_strength: float,
        blade_rotation: float,
        rim_gain: float,
        veil_transmission: np.ndarray,
        draw_accumulation: bool,
        accumulation_strength: float,
        **params: Any,
    ) -> np.ndarray:
        height, width = img.shape[:2]
        frame_centre = (width / 2.0, height / 2.0)

        flake_layer = np.zeros((height, width), dtype=np.float32)
        for x, y, size, depth_factor in flakes:
            _paste_flake_sprite(
                flake_layer,
                x,
                y,
                size,
                depth_factor,
                bokeh_strength,
                self.aperture_blades,
                blade_rotation,
                rim_gain,
                self.cats_eye_strength,
                frame_centre,
            )

        if flakes:
            avg_size = float(np.mean([f[2] for f in flakes]))
            blur_length = max(3, int(avg_size * (1.0 + motion_blur_amount * 2)))
            kernel = directional_blur_kernel(blur_length, angle)
            blurred: np.ndarray = cv2.filter2D(flake_layer, -1, kernel)
            streaked = blurred * (0.6 + motion_blur_amount)
            flake_layer = np.clip(np.maximum(flake_layer, streaked), 0.0, 1.0)

        linear, scale, is_integer = to_linear_unit(img)

        # Flakes are ice lit by, and imaging, the same sky as the scene — so
        # they are composited toward the scene's own sky radiance, not toward
        # white. That makes them disappear into a bright overcast sky and
        # stand out against dark trees, exactly as real snow does, with no
        # special-casing.
        flake_radiance = sky_radiance_linear(linear)

        # Heavy snow scatters hard: the veil is most of what says "snowing".
        t_veil = veil_transmission[..., None]
        out = linear * t_veil + flake_radiance[None, None, :] * (1.0 - t_veil)

        alpha = np.clip(flake_layer, 0.0, 1.0)[..., None]
        out = out * (1.0 - alpha) + flake_radiance[None, None, :] * alpha

        if draw_accumulation:
            accumulation = _srgb_to_linear_color(_ACCUMULATION_COLOR / 255.0)

            def blend(band: np.ndarray) -> np.ndarray:
                keep = 1.0 - accumulation_strength
                return band * keep + accumulation * accumulation_strength

            out = feathered_band_blend(out, self.accumulation_band_fraction, blend)

        return from_linear_unit(out, scale, is_integer, img.dtype)


def _srgb_to_linear_color(color: np.ndarray) -> np.ndarray:
    from weathermentations.utils.photometry import srgb_to_linear

    return srgb_to_linear(color.reshape(1, 1, -1)).reshape(-1)
