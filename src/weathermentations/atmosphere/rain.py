"""Depth-conditioned rain: streaks, a distant rain veil, and optional ground wetness.

Photometric model (Garg & Nayar, "Vision and Rain" IJCV 2007;
"Photorealistic Rendering of Rain Streaks" SIGGRAPH 2006)
---------------------------------------------------------------------------
A raindrop crossing a pixel during exposure only occupies it for a fraction
of the exposure time, so the pixel is a *time average*::

    I = (1 - alpha) * I_background + alpha * L_streak
    alpha = tau / T_exposure = 2 * r0 / (v * T_exposure)

with terminal velocity ``v(r0) = 9.40 * (1 - exp(-3.45e6 * r0^1.31))`` m/s.
For a 1 mm drop, ``v ~ 6.9 m/s``, and at 1/60 s that gives **alpha ~ 0.017**
— a streak is only a *few percent* opaque. Painting streaks at 30-60% alpha,
as a naive implementation does, is the main reason synthetic rain reads as
white scratches drawn on top of a photo instead of rain.

Getting alpha right also buys contrast-dependent visibility for free: since
``dI = alpha * (L_streak - L_background)``, a streak is obvious against dark
foliage and nearly invisible against bright sky, exactly as in real photos —
no special-casing needed.

``L_streak`` is *not* white. A drop refracts light from a ~165 degree field
of view and reflects about 6% of its whole environment, so it is essentially
an image of the sky. We take it from the scene's own bright/sky radiance,
which is why streaks tint with the weather rather than staying neutral.

Real streaks are also not uniform bars of light: the drop oscillates as it
falls, so the streak is beaded, with several smeared highlights along its
length. We modulate each streak's intensity with a quasi-periodic profile
(two harmonics and a random phase) to approximate that without shipping
Garg & Nayar's 400 MB measured streak database. That profile is sampled
*per pixel* through a lookup table rather than per drawn segment: evaluating
a three-cycle sine at six segment midpoints aliases it into a strict
two-level alternation, which turns every streak in the frame into the same
three-bright/three-dark dashed line — precisely the hatch pattern the
per-streak variation exists to avoid.

Finally, near drops are *defocused*: closer means wider **and dimmer**,
since the same light spreads over a larger circle of confusion — the
opposite of the intuitive "closer is brighter".

Everything composites in linear light, and the frame's mean level is
renormalised afterwards to stand in for the camera's auto-exposure.

Accumulating many streaks
-------------------------
Streaks are accumulated as **occupancy**, not as repeated "over" composites:
each streak deposits its ``alpha`` additively into a layer, and the final
coverage is ``1 - exp(-layer)``. For independent drops crossing the same
pixel during one exposure that is the right composition law — the pixel is
uncovered only if *no* drop covered it, so the transmissions multiply — and
unlike sequential ``over`` blending it does not depend on the order the
streaks happen to be drawn in. That order-independence is what lets the whole
field be rasterised in one batch (and, on a GPU, with a plain scatter-add).
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
from weathermentations.utils.compositing import feathered_band_blend
from weathermentations.utils.photometry import from_linear_unit, to_linear_unit


def terminal_velocity(radius_mm: float) -> float:
    """Raindrop terminal velocity in m/s (Garg & Nayar's fit to Gunn & Kinzer)."""
    radius_m = radius_mm * 1e-3
    return 9.40 * (1.0 - float(np.exp(-3.45e6 * radius_m**1.31)))


def streak_alpha(drop_diameter_px: float, streak_length_px: float) -> float:
    """Fraction of the exposure a drop occupies one pixel of its own streak.

    Garg & Nayar write this as ``tau/T = 2*r0 / (v * T_exposure)`` in world
    units, but that ratio is scale-invariant and equals
    ``drop_diameter / streak_length`` measured in *pixels* — which is the
    form to use when rendering, because it stays consistent with however long
    the streak is actually drawn. A 3 px drop smeared over a 60 px streak is
    5% opaque; draw the same drop over a 20 px streak (shorter exposure) and
    it is 15% opaque but covers a third as many pixels. Total deposited light
    is conserved either way, which is exactly right.
    """
    return float(np.clip(drop_diameter_px / max(streak_length_px, 1e-3), 0.0, 1.0))


class RandomRain(WeatherTransform):
    """Physically-grounded rain: photometrically correct streaks, veil, wet ground.

    Args:
        depth_provider: shared :class:`~weathermentations.depth.provider.SharedDepthProvider`.
        streak_density_range: number of streaks to render.
        drop_diameter_range: apparent drop diameter at the nearest depth, as
            a fraction of the image diagonal. Together with the streak length
            this sets streak opacity: ``alpha = diameter / length``.
        streak_length_range: streak length at the nearest depth, as a
            fraction of the image diagonal. This stands in for exposure
            time — a longer streak is a longer exposure, and is
            correspondingly *fainter* per pixel.
        angle_range: slant angle in degrees from vertical (wind).
        angle_jitter_range: per-streak random deviation from ``angle``, so
            streaks are not perfectly parallel (gust turbulence).
        wobble_range: per-streak mid-point bend, as a fraction of its length.
        defocus_range: how much nearer streaks spread and dim through
            defocus. 0 keeps every streak equally crisp.
        definition_softness: how soft a fully-defocused streak's edges get,
            as a multiple of its own defocus spread. Drops sit at a range of
            distances from the focal plane, so some land as crisp strokes and
            others as soft smears; a single sharpness for all of them reads
            as a hatch pattern.
        veil_beta_range: extinction coefficient for the distant rain veil.
        wetness_probability: chance of rendering the ground-wetness pass.
        wetness_band_fraction: fraction of image height treated as ground.

    Targets:
        image

    Image types:
        uint8, float32
    """

    class InitSchema(BaseTransformInitSchema):
        depth_provider: SharedDepthProvider
        streak_density_range: Annotated[
            tuple[int, int],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0, 20000)),
        ]
        drop_diameter_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0005, 0.02)),
        ]
        streak_length_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.002, 0.4)),
        ]
        angle_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(-45.0, 45.0)),
        ]
        angle_jitter_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 30.0)),
        ]
        wobble_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 0.5)),
        ]
        definition_softness: float = Field(ge=0.0, le=2.0)
        defocus_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 4.0)),
        ]
        veil_beta_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 2.0)),
        ]
        wetness_probability: float = Field(ge=0.0, le=1.0)
        wetness_band_fraction: float = Field(ge=0.0, le=1.0)

    def __init__(
        self,
        depth_provider: SharedDepthProvider,
        streak_density_range: tuple[int, int] = (1500, 4000),
        drop_diameter_range: tuple[float, float] = (0.005, 0.009),
        streak_length_range: tuple[float, float] = (0.04, 0.08),
        angle_range: tuple[float, float] = (-10.0, 10.0),
        angle_jitter_range: tuple[float, float] = (2.0, 8.0),
        wobble_range: tuple[float, float] = (0.05, 0.2),
        defocus_range: tuple[float, float] = (0.0, 0.9),
        definition_softness: float = 0.45,
        veil_beta_range: tuple[float, float] = (0.008, 0.03),
        wetness_probability: float = 0.4,
        wetness_band_fraction: float = 0.2,
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.depth_provider = require_depth_provider(depth_provider, cls_name=type(self).__name__)
        self.streak_density_range = streak_density_range
        self.drop_diameter_range = drop_diameter_range
        self.streak_length_range = streak_length_range
        self.angle_range = angle_range
        self.angle_jitter_range = angle_jitter_range
        self.wobble_range = wobble_range
        self.defocus_range = defocus_range
        self.definition_softness = definition_softness
        self.veil_beta_range = veil_beta_range
        self.wetness_probability = wetness_probability
        self.wetness_band_fraction = wetness_band_fraction

    def get_params_dependent_on_data(
        self, params: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, Any]:
        image = data["image"]
        height, width = params["shape"][:2]
        diagonal = float(np.hypot(width, height))

        relative_depth = self.depth_provider.get_depth(image)
        depth_pm = to_pseudo_metric(relative_depth, None)

        count = self.py_random.randint(*self.streak_density_range)
        angle = self.py_random.uniform(*self.angle_range)
        angle_jitter = self.py_random.uniform(*self.angle_jitter_range)
        wobble_strength = self.py_random.uniform(*self.wobble_range)
        veil_beta = self.py_random.uniform(*self.veil_beta_range)
        base_length = self.py_random.uniform(*self.streak_length_range) * diagonal
        drop_diameter_px = self.py_random.uniform(*self.drop_diameter_range) * diagonal

        # Every per-streak quantity is drawn as one array rather than in a
        # Python loop. With a few thousand streaks per frame, the scalar
        # `np.sin`/`round`/`uniform` calls that a loop needs cost several
        # times more than the rasterisation they feed.
        rng = self.random_generator
        x = rng.uniform(0.0, width, count).astype(np.float32)
        y = rng.uniform(0.0, height, count).astype(np.float32)
        px = np.clip(x.astype(np.int32), 0, width - 1)
        py = np.clip(y.astype(np.int32), 0, height - 1)
        # nearer (small relative_depth) projects larger
        depth_factor = np.maximum(0.15, 1.0 - relative_depth[py, px])

        # Both the streak's length and the drop's apparent width scale with
        # proximity, so alpha = width/length stays depth-invariant — but near
        # streaks cover far more pixels, which is why rain reads as rain at
        # all. Real rain is not one drop size repeated either: the
        # Marshall-Palmer distribution is broad, drops sit at a range of
        # distances from the focal plane, and each crosses a different part of
        # the exposure. So length, width and sharpness all vary streak to
        # streak — without that, a field of identical strokes reads as a hatch
        # pattern rather than as rain.
        length = base_length * depth_factor * rng.uniform(0.55, 1.7, count)
        drop_width = drop_diameter_px * depth_factor * rng.uniform(0.7, 1.5, count)
        # Defocus varies *per drop*, because each sits at its own distance
        # from the focal plane. One circle of confusion then drives all three
        # of width, brightness and edge softness together, so a near-focus
        # drop is a thin bright crisp stroke and an off-focus one is a wide,
        # dim, soft smear. Blurring on top of a separately chosen width would
        # double-count the same defocus and wash the whole field out.
        coc = 1.0 + rng.uniform(*self.defocus_range, count) * depth_factor
        blur_sigma = self.definition_softness * (coc - 1.0) * np.maximum(1.0, drop_width)

        return {
            "streaks": {
                "x": x,
                "y": y,
                "length": length.astype(np.float32),
                "thickness": np.maximum(1, np.round(drop_width * coc)).astype(np.int32),
                "angle": (angle + rng.uniform(-angle_jitter, angle_jitter, count)).astype(
                    np.float32
                ),
                "bend": rng.uniform(-wobble_strength, wobble_strength, count).astype(np.float32),
                "alpha": (
                    np.clip(drop_width / np.maximum(length, 1e-3), 0.0, 1.0) / coc
                ).astype(np.float32),
                "phase": rng.uniform(0.0, 2 * np.pi, count).astype(np.float32),
                "blur_sigma": blur_sigma.astype(np.float32),
            },
            "angle": angle,
            "veil_transmission": np.exp(-veil_beta * depth_pm).astype(np.float32),
            "draw_wetness": self.py_random.random() < self.wetness_probability,
        }

    def apply(
        self,
        img: np.ndarray,
        streaks: dict[str, np.ndarray],
        angle: float,
        veil_transmission: np.ndarray,
        draw_wetness: bool,
        **params: Any,
    ) -> np.ndarray:
        height, width = img.shape[:2]
        linear, scale, is_integer = to_linear_unit(img)
        original_mean = float(linear.mean())

        # A drop images a ~165 degree field of view, so a streak is basically
        # a picture of the sky. Take its radiance from the scene itself.
        streak_radiance = sky_radiance_linear(linear)

        occupancy = render_streak_layer(height, width, streaks)

        # Only a light cross-section softening here: the streak *is* the
        # motion blur, so smearing it again along its own axis would divide
        # its (already only a few percent) alpha by the kernel length and
        # erase the rain entirely.
        softened: np.ndarray = cv2.GaussianBlur(occupancy, (0, 0), sigmaX=0.7)
        # Occupancy -> coverage. See the module docstring: drops crossing one
        # pixel during the exposure occlude independently, so the fractions of
        # the exposure they are *absent* for multiply.
        alpha_layer = -np.expm1(-softened)

        # Distant rain that no longer resolves into individual streaks
        # scatters into a veil, the same Koschmieder compositing as fog but
        # with the rain's own (sky-coloured) radiance.
        t = veil_transmission[..., None]
        out = linear * t + streak_radiance[None, None, :] * (1.0 - t)

        # Time-averaged streaks: I = (1-a)*I_bg + a*L_streak.
        a = alpha_layer[..., None]
        out = out * (1.0 - a) + streak_radiance[None, None, :] * a

        if draw_wetness:
            out = _apply_ground_wetness(out, self.wetness_band_fraction)

        # Stand-in for the camera's auto-exposure pulling the frame back to a
        # normal level after the scene got brighter/hazier.
        new_mean = float(out.mean())
        if new_mean > 1e-5:
            out *= np.clip(original_mean / new_mean, 0.75, 1.25)

        return from_linear_unit(out, scale, is_integer, img.dtype)


# The beading profile is a function of one phase-like argument, so it tables
# once for the whole library instead of being evaluated per pixel. `s` runs
# 0..1 along a streak, a little beyond at the anti-aliased ends, and the
# random per-streak phase adds up to 2*pi on top.
_BEADING_ARGUMENT_MAX = 6.0 * np.pi * 1.3 + 2.0 * np.pi
_BEADING_LUT_SIZE = 2048
_BEADING_SCALE = _BEADING_LUT_SIZE / _BEADING_ARGUMENT_MAX


def _build_beading_lut() -> np.ndarray:
    t = np.linspace(0.0, _BEADING_ARGUMENT_MAX, _BEADING_LUT_SIZE, dtype=np.float32)
    return ((0.55 + 0.45 * np.sin(t)) * (0.75 + 0.25 * np.sin(2.0 * t))).astype(np.float32)


_BEADING_LUT = _build_beading_lut()


def streak_geometry(streaks: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Per-streak polyline vertices and patch rectangles, for the whole field at once.

    Returns integer patch origins/sizes and ``(N, 7, 2)`` int32 polyline
    vertices in patch-local coordinates, so the rasterising loop does no
    arithmetic of its own.
    """
    x, y = streaks["x"], streaks["y"]
    length = streaks["length"]
    thickness = streaks["thickness"]
    bend = streaks["bend"]
    blur_sigma = streaks["blur_sigma"]

    angle_rad = np.deg2rad(streaks["angle"])
    dir_x, dir_y = np.sin(angle_rad), np.cos(angle_rad)
    perp_x, perp_y = -dir_y, dir_x

    # Tight bounding box. The mid-streak bend peaks at |bend|*length along the
    # perpendicular, so project that onto each axis rather than reserving the
    # streak's whole length in both directions — a near-vertical streak needs
    # only a few pixels of width, and over-reserving it made every patch
    # several times larger than the streak actually drawn into it.
    margin = thickness + 2
    bend_extent = np.abs(bend) * length
    end_x, end_y = x + dir_x * length, y + dir_y * length
    x_lo = np.minimum(x, end_x) - np.abs(dir_y) * bend_extent - margin
    x_hi = np.maximum(x, end_x) + np.abs(dir_y) * bend_extent + margin
    y_lo = np.minimum(y, end_y) - np.abs(dir_x) * bend_extent - margin
    y_hi = np.maximum(y, end_y) + np.abs(dir_x) * bend_extent + margin

    # A defocused streak spreads past its own bounding box, so the patch is
    # grown by the blur's reach up front; blurring a tight patch in place
    # would clip the spread into a hard rectangle.
    pad = np.where(blur_sigma > 0.3, np.ceil(3.0 * blur_sigma), 0.0)

    x0 = np.floor(x_lo - pad).astype(np.int32)
    y0 = np.floor(y_lo - pad).astype(np.int32)
    patch_w = np.ceil(x_hi - x_lo + 2.0 * pad).astype(np.int32)
    patch_h = np.ceil(y_hi - y_lo + 2.0 * pad).astype(np.int32)

    # Seven vertices = the six segments the streak is drawn as. The bend peaks
    # at the middle of the streak.
    s = np.linspace(0.0, 1.0, 7, dtype=np.float32)[None, :]
    offset = bend[:, None] * length[:, None] * 4.0 * s * (1.0 - s)
    along = length[:, None] * s
    vx = x[:, None] + dir_x[:, None] * along + perp_x[:, None] * offset - x0[:, None]
    vy = y[:, None] + dir_y[:, None] * along + perp_y[:, None] * offset - y0[:, None]

    return {
        "x0": x0,
        "y0": y0,
        "patch_w": patch_w,
        "patch_h": patch_h,
        "vertices": np.round(np.stack([vx, vy], axis=-1)).astype(np.int32),
        "dir_x": dir_x.astype(np.float32),
        "dir_y": dir_y.astype(np.float32),
    }


def render_streak_layer(
    height: int, width: int, streaks: dict[str, np.ndarray]
) -> np.ndarray:
    """Rasterise the whole streak field into an ``(H, W)`` occupancy layer.

    A falling drop oscillates (an oblate-prolate mode plus a transverse mode
    at roughly twice the frequency), so its streak is beaded — several smeared
    highlights along its length rather than one even bar. The beading is
    applied as a per-pixel modulation of the drawn polyline, read out of
    :data:`_BEADING_LUT` by the pixel's position *along* the streak, which is
    an affine function of its row and column and so costs one broadcast add
    and one gather per patch.

    Streaks accumulate additively (see the module docstring); they are not
    composited one over the next, so the result does not depend on drawing
    order.
    """
    layer = np.zeros((height, width), dtype=np.float32)
    geometry = streak_geometry(streaks)
    x0s, y0s = geometry["x0"], geometry["y0"]
    patch_ws, patch_hs = geometry["patch_w"], geometry["patch_h"]
    vertices = geometry["vertices"]
    dir_xs, dir_ys = geometry["dir_x"], geometry["dir_y"]
    alphas = streaks["alpha"]
    thicknesses = streaks["thickness"]
    blur_sigmas = streaks["blur_sigma"]
    lengths = streaks["length"]
    phases = streaks["phase"]
    xs, ys = streaks["x"], streaks["y"]

    for i in range(len(alphas)):
        patch_w = int(patch_ws[i])
        patch_h = int(patch_hs[i])
        if patch_w <= 0 or patch_h <= 0:
            continue
        x0, y0 = int(x0s[i]), int(y0s[i])
        # Skip streaks whose patch misses the frame entirely before doing any
        # rasterisation work for them.
        if x0 + patch_w <= 0 or y0 + patch_h <= 0 or x0 >= width or y0 >= height:
            continue

        patch: np.ndarray = np.zeros((patch_h, patch_w), dtype=np.float32)
        cv2.polylines(
            patch,
            [vertices[i]],
            isClosed=False,
            color=float(alphas[i]),
            thickness=int(thicknesses[i]),
            lineType=cv2.LINE_AA,
        )

        # Beading. `s` is the distance along the streak normalised by its
        # length, which is affine in (row, col): s = a*col + b*row + c.
        rate = 6.0 * np.pi / max(float(lengths[i]), 1e-3)
        col_term = (np.arange(patch_w, dtype=np.float32) + (x0 - xs[i])) * (
            rate * dir_xs[i] * _BEADING_SCALE
        )
        row_term = (np.arange(patch_h, dtype=np.float32) + (y0 - ys[i])) * (
            rate * dir_ys[i] * _BEADING_SCALE
        ) + phases[i] * _BEADING_SCALE
        index = (row_term[:, None] + col_term[None, :]).astype(np.int32)
        patch *= np.take(_BEADING_LUT, index, mode="clip")

        if blur_sigmas[i] > 0.3:
            patch = cv2.GaussianBlur(patch, (0, 0), sigmaX=float(blur_sigmas[i]))

        dst_x0, dst_y0 = max(0, x0), max(0, y0)
        dst_x1, dst_y1 = min(width, x0 + patch_w), min(height, y0 + patch_h)
        if dst_x1 <= dst_x0 or dst_y1 <= dst_y0:
            continue
        layer[dst_y0:dst_y1, dst_x0:dst_x1] += patch[
            dst_y0 - y0 : dst_y1 - y0, dst_x0 - x0 : dst_x1 - x0
        ]

    return layer


def _apply_ground_wetness(linear: np.ndarray, band_fraction: float) -> np.ndarray:
    """Wet ground: a darker diffuse base plus a blurred specular reflection."""
    height, width = linear.shape[:2]
    band_start = int(height * (1.0 - band_fraction))
    if band_start >= height:
        return linear

    band_height = height - band_start
    reflect_start = max(0, band_start - band_height)
    reflection_source = linear[reflect_start:band_start]
    if reflection_source.shape[0] == 0:
        return linear
    reflection: np.ndarray = reflection_source[::-1]
    if reflection.shape[0] != band_height:
        reflection = cv2.resize(reflection, (width, band_height), interpolation=cv2.INTER_LINEAR)
    reflection = cv2.GaussianBlur(reflection, (0, 0), sigmaX=3.0, sigmaY=1.0)

    def blend(ground: np.ndarray) -> np.ndarray:
        return ground * 0.85 * 0.6 + reflection * 0.4

    return feathered_band_blend(linear, band_fraction, blend)
