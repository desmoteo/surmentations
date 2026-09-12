"""Water droplets adhering to the lens/glass.

Physical model: a droplet resting on glass behaves like a small plano-convex
lens. We approximate its surface as a spherical cap of radius ``r`` (droplet
radius) and height ``h = r * cap_height_ratio``; the surface normal at a
point a horizontal distance ``d`` from the droplet's center follows the cap's
local slope. Rays entering the camera are refracted by that normal per
Snell's law (air, n=1.0 -> water, n=1.33), which bends the sampling ray
enough to invert/magnify the miniature scene visible inside the droplet
(the well-known "looking through a droplet" look). We render this as a
backward mapping: for each pixel inside a droplet's footprint, compute the
refracted ray direction and sample the source image at the resulting
offset position, then add a Blinn-Phong specular highlight from a fixed
overhead "sky" light and darken the contact-line edge (surface tension
pooling makes the rim optically denser).

Reference: droplet-as-refractive-lens models used in raindrop detection /
removal literature, e.g. Qian et al. 2018 ("Attentive Generative Adversarial
Network for Raindrop Removal from a Single Image") and adherent-raindrop
photometric models building on Roser & Geiger 2009.
"""

from __future__ import annotations

from typing import Annotated, Any

import cv2
import numpy as np
from albumentations.core.pydantic import check_range_bounds, nondecreasing
from albumentations.core.transforms_interface import BaseTransformInitSchema
from pydantic import AfterValidator, Field

from weathermentations.base import WeatherTransform

_WATER_IOR = 1.33  # refractive index of water relative to air


def _droplet_refraction_offset(
    dist_norm: np.ndarray, radius_px: float, cap_height_ratio: float
) -> np.ndarray:
    """Radial pixel offset for the refracted sampling ray, for points inside a droplet.

    ``dist_norm`` is distance from the droplet center normalized by
    ``radius_px`` (0 at center, 1 at the rim). Returns the *inward* radial
    offset (pixels) to add to the sampling position, derived from Snell's
    law applied at the spherical-cap surface normal.
    """
    dist_norm = np.clip(dist_norm, 0.0, 0.999)
    # Spherical-cap surface height profile z(d) = h * sqrt(1 - d^2), whose slope
    # gives the local surface-normal tilt angle theta_n from vertical.
    slope = dist_norm / np.sqrt(1.0 - dist_norm**2)
    theta_n = np.arctan(slope * cap_height_ratio)

    # Incident ray is vertical (straight into the lens); angle of incidence at
    # the tilted surface equals theta_n. Snell's law gives the refracted angle.
    sin_refracted = np.clip(np.sin(theta_n) / _WATER_IOR, -1.0, 1.0)
    theta_refracted = np.arcsin(sin_refracted)
    bend = theta_n - theta_refracted

    # Convert the angular bend into a radial pixel offset, scaled by droplet
    # radius so larger droplets show proportionally more distortion, and flip
    # sign so the offset samples *toward* the center (the characteristic
    # inverted/magnified look through a convex droplet).
    return -np.tan(bend) * radius_px * 2.0


def _render_droplet(
    source: np.ndarray, cx: float, cy: float, radius_px: float, cap_height_ratio: float
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Render one droplet's refracted patch and alpha mask.

    Returns ``(patch, alpha, x0, y0)`` where ``patch`` is an ``(S, S, C)``
    refracted image sample and ``alpha`` an ``(S, S)`` mask, positioned with
    top-left corner at ``(x0, y0)`` in source-image coordinates.
    """
    height, width = source.shape[:2]
    size = max(4, int(round(radius_px * 2.2)))
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    center = size / 2.0
    dx = xx - center
    dy = yy - center
    dist_px = np.sqrt(dx**2 + dy**2)
    dist_norm = dist_px / radius_px

    inside = dist_norm <= 1.0
    offset = _droplet_refraction_offset(dist_norm, radius_px, cap_height_ratio)
    safe_dist = np.where(dist_px > 1e-3, dist_px, 1.0)
    map_x = cx + dx + (dx / safe_dist) * offset
    map_y = cy + dy + (dy / safe_dist) * offset

    patch = cv2.remap(
        source,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )

    # Blinn-Phong specular highlight from a fixed overhead light, using the cap's
    # normal field (z-component derived from the same height profile as above).
    normal_xy = np.stack([dx, dy], axis=-1) / (radius_px + 1e-6)
    normal_z = np.sqrt(np.clip(1.0 - np.sum(normal_xy**2, axis=-1), 0.0, 1.0))
    light_dir = np.array([-0.3, -0.6, 0.75])
    light_dir /= np.linalg.norm(light_dir)
    normal = np.concatenate([normal_xy, normal_z[..., None]], axis=-1)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-6
    view_dir = np.array([0.0, 0.0, 1.0])
    half_vec = light_dir + view_dir
    half_vec /= np.linalg.norm(half_vec)
    specular = np.clip(np.sum(normal * half_vec, axis=-1), 0.0, 1.0) ** 40

    # Contact-line darkening: surface tension pools water at the rim, increasing
    # optical density there.
    rim = np.clip((dist_norm - 0.8) / 0.2, 0.0, 1.0)

    alpha = inside.astype(np.float32)
    alpha *= 1.0 - 0.25 * rim  # slight rim darkening handled via alpha+color below

    patch_f = patch.astype(np.float32)
    patch_f += specular[..., None] * 200.0
    patch_f -= rim[..., None] * 40.0
    patch = np.clip(patch_f, 0, 255).astype(source.dtype)

    x0, y0 = int(round(cx - center)), int(round(cy - center))
    return patch, alpha, x0, y0


class WaterDroplets(WeatherTransform):
    """Refractive water droplets adhering to the lens.

    Args:
        droplet_count_range: number of droplets to render.
        droplet_radius_range: droplet radius in pixels.
        cap_height_ratio_range: droplet "bulge" as a fraction of its radius
            (spherical-cap height / radius) — higher values refract more
            strongly. Typical sessile water droplets: 0.3-0.6.
        streak_probability: chance that a given droplet is instead rendered
            as a thin, vertically-elongated sliding streak (gravity trail)
            rather than a round bead.
        streak_length_range: vertical elongation multiplier applied to a
            droplet's own rendered size when it's drawn as a gravity streak
            (e.g. 2.0 means the streak is twice as tall as a round droplet
            of the same radius).

    Targets:
        image

    Image types:
        uint8, float32
    """

    class InitSchema(BaseTransformInitSchema):
        droplet_count_range: Annotated[
            tuple[int, int],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(1, 60)),
        ]
        droplet_radius_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(2.0, 150.0)),
        ]
        cap_height_ratio_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.05, 1.0)),
        ]
        streak_probability: float = Field(ge=0.0, le=1.0)
        streak_length_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(1.0, 6.0)),
        ]

    def __init__(
        self,
        droplet_count_range: tuple[int, int] = (5, 20),
        droplet_radius_range: tuple[float, float] = (4.0, 22.0),
        cap_height_ratio_range: tuple[float, float] = (0.3, 0.6),
        streak_probability: float = 0.25,
        streak_length_range: tuple[float, float] = (1.5, 3.0),
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.droplet_count_range = droplet_count_range
        self.droplet_radius_range = droplet_radius_range
        self.cap_height_ratio_range = cap_height_ratio_range
        self.streak_probability = streak_probability
        self.streak_length_range = streak_length_range

    def get_params_dependent_on_data(
        self, params: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, Any]:
        height, width = params["shape"][:2]
        count = self.py_random.randint(*self.droplet_count_range)

        droplets = []
        for _ in range(count):
            radius = self.py_random.uniform(*self.droplet_radius_range)
            cx = self.py_random.uniform(0, width)
            cy = self.py_random.uniform(0, height)
            cap_ratio = self.py_random.uniform(*self.cap_height_ratio_range)
            is_streak = self.py_random.random() < self.streak_probability
            stretch = self.py_random.uniform(*self.streak_length_range) if is_streak else 1.0
            droplets.append((cx, cy, radius, cap_ratio, stretch))

        return {"droplets": droplets}

    def apply(
        self,
        img: np.ndarray,
        droplets: list[tuple[float, float, float, float, float]],
        **params: Any,
    ) -> np.ndarray:
        if not droplets:
            return img

        height, width = img.shape[:2]
        out = img.copy()

        for cx, cy, radius, cap_ratio, stretch in droplets:
            if stretch > 1.0:
                # Gravity streak: render a droplet into a tall canvas, then squash
                # it vertically via resize to elongate the refracted patch.
                tall_source = img
                patch, alpha, x0, y0 = _render_droplet(tall_source, cx, cy, radius, cap_ratio)
                new_h = int(round(patch.shape[0] * stretch))
                patch = cv2.resize(patch, (patch.shape[1], new_h), interpolation=cv2.INTER_LINEAR)
                alpha = cv2.resize(alpha, (alpha.shape[1], new_h), interpolation=cv2.INTER_LINEAR)
                y0 = int(round(cy - new_h / 2.0))
            else:
                patch, alpha, x0, y0 = _render_droplet(img, cx, cy, radius, cap_ratio)

            ph, pw = patch.shape[:2]
            dst_x0, dst_y0 = max(0, x0), max(0, y0)
            dst_x1, dst_y1 = min(width, x0 + pw), min(height, y0 + ph)
            if dst_x1 <= dst_x0 or dst_y1 <= dst_y0:
                continue
            src_x0, src_y0 = dst_x0 - x0, dst_y0 - y0
            src_x1, src_y1 = src_x0 + (dst_x1 - dst_x0), src_y0 + (dst_y1 - dst_y0)

            region = out[dst_y0:dst_y1, dst_x0:dst_x1].astype(np.float32)
            a = alpha[src_y0:src_y1, src_x0:src_x1][..., None]
            region = region * (1 - a) + patch[src_y0:src_y1, src_x0:src_x1].astype(np.float32) * a
            out[dst_y0:dst_y1, dst_x0:dst_x1] = np.clip(region, 0, 255).astype(img.dtype)

        return out
