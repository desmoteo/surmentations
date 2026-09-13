"""Dirt, dust and smudges on the lens.

Physical model
--------------
Dirt on a front element does **not** work like paint. It sits far outside the
depth of field, so it never resolves as a shape; what it actually does is
*scatter* light. Two consequences drive this implementation, and both are
what make a dirty lens recognisable:

1. **It redistributes light rather than adding it.** The whole effect is a
   lerp toward a wide, heavy-tailed blur of the scene, weighted by how much
   grime covers each pixel. Because that blur is energy-conserving, the
   result cannot blow out — which an unbounded additive "bloom" very much
   can, and did: gain-driven bloom turned bright sky into flat white discs
   that read as lens flare rather than dirt.

   Highlights are weighted up *inside* the scattered term instead, because
   forward scatter off dust is strongly intensity-dependent. That keeps the
   defining behaviour — grime is near invisible in flat light and awful the
   moment a bright source is in or near frame — without the blowout.

2. **It costs local contrast through forward scatter.** Mie scattering off
   dust/grease is strongly forward-peaked with heavy tails, so a fraction of
   every ray is redistributed over a wide angle. We approximate the
   Lorentzian ``1/(1 + (r/r0)^2)`` point-spread function with a sum of
   Gaussians (cheap, and keeps the heavy tail) and lerp toward it under the
   mask — which washes out local contrast the way real haze on glass does.

Only a small absorption tint is applied on top, for genuinely opaque muck.
Note that front-element dust specks are *not* drawn as visible specks: they
are optically invisible in their own right. Sharp dark spots come from
sensor dust, which is a different (and per-camera fixed) phenomenon.

The mask itself is deliberately multi-scale and anisotropic: a low-frequency
smudge field, plus curved *wipe arcs* (someone smeared it with a cloth or a
wiper blade, which leaves directional streaks rather than isotropic blobs),
plus dried water-spot rings — droplets dry via the coffee-ring effect,
leaving a high-scatter annulus with a cleaner centre.

All size-like parameters are fractions of the image diagonal, so the effect
looks the same at any resolution.
"""

from __future__ import annotations

import math
from typing import Annotated, Any

import cv2
import numpy as np
from albumentations.core.pydantic import check_range_bounds, nondecreasing
from albumentations.core.transforms_interface import BaseTransformInitSchema
from pydantic import AfterValidator, Field

from weathermentations.base import WeatherTransform
from weathermentations.utils.noise import blob_cluster_mask
from weathermentations.utils.photometry import (
    from_linear_unit,
    luminance,
    srgb_to_linear,
    to_linear_unit,
)


def scatter_psf(image: np.ndarray, radius_px: float, max_sigma_px: float = 12.0) -> np.ndarray:
    """Approximate a heavy-tailed Lorentzian scatter PSF with summed Gaussians.

    A true ``1/(1 + (r/r0)^2)`` convolution is expensive and its tail is what
    matters most (it is why a single bright light fogs a wide area of a dirty
    lens), so we sum three Gaussians spanning an order of magnitude in sigma.

    The widest of those sigmas can reach several hundred pixels on a
    high-resolution frame, and OpenCV would then build a kernel a couple of
    thousand pixels across — which cost close to a second per image. But a
    blur that wide cannot represent any detail finer than its own scale, so
    it is computed on a downsampled copy and scaled back up: visually
    indistinguishable, and orders of magnitude cheaper. ``max_sigma_px``
    caps the sigma actually used at the reduced resolution.
    """
    sigmas = (radius_px * 0.35, radius_px, radius_px * 3.0)
    weights = (0.5, 0.32, 0.18)

    height, width = image.shape[:2]
    factor = max(1, int(np.ceil(max(sigmas) / max_sigma_px)))
    if factor > 1:
        small = cv2.resize(
            image, (max(1, width // factor), max(1, height // factor)), interpolation=cv2.INTER_AREA
        )
    else:
        small = image

    accumulated = np.zeros_like(small, dtype=np.float32)
    for sigma, weight in zip(sigmas, weights, strict=True):
        scaled_sigma = max(0.8, sigma / factor)
        ksize = int(scaled_sigma * 6) | 1
        accumulated += weight * cv2.GaussianBlur(small, (ksize, ksize), sigmaX=scaled_sigma)

    if factor > 1:
        return cv2.resize(accumulated, (width, height), interpolation=cv2.INTER_LINEAR)
    return accumulated


def _wipe_arcs(
    height: int, width: int, rng: np.random.Generator, count: int, thickness_px: float
) -> np.ndarray:
    """Curved, directional smear streaks left by a cloth or a wiper blade."""
    mask = np.zeros((height, width), dtype=np.float32)
    diagonal = math.hypot(width, height)
    for _ in range(count):
        # A wide, shallow arc crossing the frame.
        cx = rng.uniform(-0.3, 1.3) * width
        cy = rng.uniform(-0.3, 1.3) * height
        radius = rng.uniform(0.4, 1.4) * diagonal
        start = rng.uniform(0, 2 * math.pi)
        sweep = rng.uniform(0.15, 0.6)
        steps = 48
        pts = []
        for i in range(steps):
            angle = start + sweep * i / (steps - 1)
            pts.append([cx + radius * math.cos(angle), cy + radius * math.sin(angle)])
        polyline = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(
            mask,
            [polyline],
            isClosed=False,
            color=float(rng.uniform(0.5, 1.0)),
            thickness=max(1, int(thickness_px * rng.uniform(0.5, 1.6))),
            lineType=cv2.LINE_AA,
        )
    return mask


def _water_spot_rings(
    height: int,
    width: int,
    rng: np.random.Generator,
    count: int,
    radius_px_range: tuple[float, float],
) -> np.ndarray:
    """Dried droplet residue: coffee-ring annuli with a cleaner centre."""
    mask = np.zeros((height, width), dtype=np.float32)
    for _ in range(count):
        radius = rng.uniform(*radius_px_range)
        cx = int(rng.uniform(0, width))
        cy = int(rng.uniform(0, height))
        ring_thickness = max(1, int(radius * rng.uniform(0.12, 0.25)))
        cv2.circle(
            mask,
            (cx, cy),
            max(1, int(radius)),
            float(rng.uniform(0.6, 1.0)),
            thickness=ring_thickness,
            lineType=cv2.LINE_AA,
        )
        # Faint residue inside the ring, much weaker than the rim.
        cv2.circle(mask, (cx, cy), max(1, int(radius - ring_thickness)), 0.18, thickness=-1)
    return mask


class DirtSoil(WeatherTransform):
    """Scattering dirt/grime on the lens: contrast-killing veil of scattered light.

    Args:
        smudge_count_range: number of low-frequency grime patches.
        smudge_size_range: grime patch radius, as a fraction of the image
            diagonal.
        wipe_arc_count_range: number of directional smear arcs (cloth or
            wiper marks). Set to ``(0, 0)`` for untouched grime.
        water_spot_count_range: number of dried droplet rings.
        water_spot_radius_range: dried droplet radius, as a fraction of the
            image diagonal.
        coverage_range: fraction of the frame the mask is normalised to
            cover, roughly. Real lens grime is sparse.
        veil_strength_range: how much of each covered pixel is replaced by
            scattered light. This is the whole effect: contrast loss plus
            smeared highlights, energy-conserving by construction.
        highlight_boost_range: how much brighter-than-average scene content
            dominates the scattered light. Higher makes the grime near
            invisible in flat light and glaring when backlit.
        scatter_radius_range: scatter PSF radius, as a fraction of the image
            diagonal — how far a bright source fogs out across the grime.
        absorption_range: how much genuinely opaque muck darkens/tints, on
            top of the scattering. Keep small; dirt mostly scatters.

    Targets:
        image

    Image types:
        uint8, float32
    """

    class InitSchema(BaseTransformInitSchema):
        smudge_count_range: Annotated[
            tuple[int, int],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0, 40)),
        ]
        smudge_size_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.005, 0.5)),
        ]
        wipe_arc_count_range: Annotated[
            tuple[int, int],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0, 20)),
        ]
        water_spot_count_range: Annotated[
            tuple[int, int],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0, 60)),
        ]
        water_spot_radius_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.002, 0.1)),
        ]
        coverage_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 1.0)),
        ]
        highlight_boost_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 8.0)),
        ]
        veil_strength_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 1.0)),
        ]
        scatter_radius_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.005, 0.3)),
        ]
        absorption_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 0.6)),
        ]
        tint: tuple[int, int, int] = Field(default=(120, 112, 100))

    def __init__(
        self,
        smudge_count_range: tuple[int, int] = (4, 12),
        smudge_size_range: tuple[float, float] = (0.04, 0.22),
        wipe_arc_count_range: tuple[int, int] = (2, 6),
        water_spot_count_range: tuple[int, int] = (0, 14),
        water_spot_radius_range: tuple[float, float] = (0.004, 0.022),
        coverage_range: tuple[float, float] = (0.10, 0.32),
        highlight_boost_range: tuple[float, float] = (1.5, 5.0),
        veil_strength_range: tuple[float, float] = (0.35, 0.8),
        scatter_radius_range: tuple[float, float] = (0.02, 0.07),
        absorption_range: tuple[float, float] = (0.02, 0.12),
        tint: tuple[int, int, int] = (120, 112, 100),
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.smudge_count_range = smudge_count_range
        self.smudge_size_range = smudge_size_range
        self.wipe_arc_count_range = wipe_arc_count_range
        self.water_spot_count_range = water_spot_count_range
        self.water_spot_radius_range = water_spot_radius_range
        self.coverage_range = coverage_range
        self.highlight_boost_range = highlight_boost_range
        self.veil_strength_range = veil_strength_range
        self.scatter_radius_range = scatter_radius_range
        self.absorption_range = absorption_range
        self.tint = tint

    def get_params_dependent_on_data(
        self, params: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, Any]:
        height, width = params["shape"][:2]
        diagonal = float(np.hypot(width, height))
        rng = self.random_generator

        mask = blob_cluster_mask(
            height,
            width,
            rng=rng,
            count=self.py_random.randint(*self.smudge_count_range),
            size_range=(
                self.smudge_size_range[0] * diagonal,
                self.smudge_size_range[1] * diagonal,
            ),
        )

        arc_count = self.py_random.randint(*self.wipe_arc_count_range)
        if arc_count:
            arcs = _wipe_arcs(height, width, rng, arc_count, thickness_px=diagonal * 0.02)
            arcs = cv2.GaussianBlur(arcs, (0, 0), sigmaX=diagonal * 0.012)
            mask = np.maximum(mask, arcs * 1.4)

        spot_count = self.py_random.randint(*self.water_spot_count_range)
        if spot_count:
            spots = _water_spot_rings(
                height,
                width,
                rng,
                spot_count,
                (
                    self.water_spot_radius_range[0] * diagonal,
                    self.water_spot_radius_range[1] * diagonal,
                ),
            )
            spots = cv2.GaussianBlur(spots, (0, 0), sigmaX=max(1.0, diagonal * 0.003))
            mask = np.maximum(mask, spots)

        # Grime is never sharp — it is far outside the depth of field.
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(1.0, diagonal * 0.006))

        # Normalise to the requested (sparse) coverage so the knobs stay
        # predictable regardless of how the blobs happened to land.
        target_coverage = self.py_random.uniform(*self.coverage_range)
        current = float(mask.mean())
        if current > 1e-5:
            mask = np.clip(mask * (target_coverage / current), 0.0, 1.0)

        return {
            "dirt_mask": mask.astype(np.float32),
            "veil_strength": self.py_random.uniform(*self.veil_strength_range),
            "highlight_boost": self.py_random.uniform(*self.highlight_boost_range),
            "scatter_radius": self.py_random.uniform(*self.scatter_radius_range) * diagonal,
            "absorption": self.py_random.uniform(*self.absorption_range),
        }

    def apply(
        self,
        img: np.ndarray,
        dirt_mask: np.ndarray,
        veil_strength: float,
        highlight_boost: float,
        scatter_radius: float,
        absorption: float,
        **params: Any,
    ) -> np.ndarray:
        linear, scale, is_integer = to_linear_unit(img)
        mask = dirt_mask[..., None]

        # Grime *redistributes* light, it does not create it. So the whole
        # effect is a lerp toward a wide, heavy-tailed blur of the scene,
        # weighted by how much dirt covers each pixel. Because the PSF is
        # normalised, this conserves total energy: it can wash contrast out
        # and smear a bright source across the muck, but it can never blow
        # the frame out the way an unbounded additive bloom does.
        #
        # Highlights are weighted up inside the scattered term because
        # forward scatter off dust is strongly intensity-dependent — which is
        # what makes a dirty lens invisible in flat light and awful when a
        # bright source is in or near frame.
        luma = luminance(linear)
        highlight = np.clip((luma - 0.35) / 0.65, 0.0, 1.0)[..., None]
        emphasised = linear * (1.0 + highlight_boost * highlight)
        scattered = scatter_psf(emphasised, scatter_radius)

        veil = np.clip(veil_strength * mask, 0.0, 0.95)
        out = linear * (1.0 - veil) + scattered * veil

        # A little genuinely opaque muck on top.
        tint = srgb_to_linear(np.array(self.tint, dtype=np.float32).reshape(1, 1, 3) / 255.0)
        out = out * (1.0 - absorption * mask) + tint * (absorption * mask)

        return from_linear_unit(out, scale, is_integer, img.dtype)
