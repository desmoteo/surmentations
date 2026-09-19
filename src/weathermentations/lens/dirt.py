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

3. **It is not one substance, and not one colour.** This is the third thing
   that decides whether grime reads as real, and the easiest to get wrong.
   A pane collects several deposits at once, each with its own *single-
   scattering albedo* — the fraction of light it returns, per wavelength:

   * **Mineral dust and sand** (silica with iron-oxide staining) absorb
     strongly toward the blue, so they scatter a warm tan. This is the
     Ångström-exponent behaviour that makes dust-hazed skies yellow.
   * **Clay and soil splash** absorb harder still across the blue-green,
     reading light brown.
   * **Soot and urban road film** (black carbon) absorb almost flat across
     the visible band — the canonical *grey* absorber, dark but neutral.
   * **Limescale, sea salt and dried spray residue** barely absorb at all,
     so they scatter nearly white, and being highly scattering they are the
     most visible deposit per unit of thickness.
   * **Pollen and organic films** absorb blue and reflect yellow-green.

   The albedo multiplies the scattered term rather than being painted over
   the top, so each deposit both tints *and* dims what scatters through it,
   and ``1 - albedo`` is the light it absorbs. A single global brown tint —
   the previous model — collapses all of that into a decal.

Front-element dust specks are *not* drawn as visible specks: they are
optically invisible in their own right. Sharp dark spots come from sensor
dust, which is a different (and per-camera fixed) phenomenon.

The mask itself is deliberately multi-scale and anisotropic: a low-frequency
smudge field, plus curved *wipe arcs* (someone smeared it with a cloth or a
wiper blade, which leaves directional streaks rather than isotropic blobs),
plus dried water-spot rings — droplets dry via the coffee-ring effect,
leaving a high-scatter annulus with a cleaner centre. Each of those carries
its own deposit colour: a wipe arc is smeared road film, a dried spot is
mineral residue left by evaporating water, and they do not look alike.

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
from pydantic import AfterValidator

from weathermentations.base import WeatherTransform
from weathermentations.diagnostics import record_draw
from weathermentations.utils.noise import colored_blob_cluster
from weathermentations.utils.photometry import (
    from_linear_unit,
    luminance,
    to_linear_unit,
)

# Single-scattering albedo of each deposit as **linear-light** RGB: the
# fraction of light the substance returns in each channel, so 1 - albedo is
# what it absorbs. These are representative values for the material classes,
# not measurements of a specific sample — the point is the relative spectral
# shape (which channel is suppressed), because that is what the eye reads as
# "dust" versus "soot" versus "salt".
DEPOSIT_ALBEDOS: dict[str, tuple[float, float, float]] = {
    # Salt, limescale, dried spray residue: essentially non-absorbing, so it
    # scatters near-white and is the most conspicuous deposit per unit of
    # thickness.
    "limescale": (0.92, 0.93, 0.94),
    # Silica sand stained with iron oxide. Absorbs toward the blue (the
    # Angstrom-exponent behaviour that yellows dust-hazed skies), so it
    # scatters a warm tan.
    "mineral_dust": (0.80, 0.70, 0.48),
    # Clay/soil splash: absorbs harder across blue and green, reading light
    # brown.
    "clay_soil": (0.52, 0.40, 0.27),
    # Black carbon is the canonical *grey* absorber: near-flat absorption
    # across the visible band, so it darkens without tinting.
    "soot": (0.22, 0.22, 0.23),
    # The usual windscreen mixture of diesel exhaust and road dust.
    "road_film": (0.45, 0.44, 0.42),
    # Pollen and organic films absorb blue and reflect yellow-green.
    "pollen": (0.72, 0.66, 0.30),
}

# What a wiper or a cloth leaves: every deposit present, mechanically mixed,
# so it is duller and more neutral than any single one of them.
_WIPE_DEPOSIT = "road_film"
# Evaporating water leaves behind its dissolved minerals, not whatever else
# was on the glass.
_WATER_SPOT_DEPOSIT = "limescale"

# A caked, opaque crust is the same substance seen in its own shadow rather
# than lit through: darker than the deposit's albedo, but the same hue.
_OPAQUE_BODY_FRACTION = 0.35


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
        absorption_range: how much genuinely *opaque* caked muck blocks the
            scene outright, on top of the scattering. Keep small; dirt
            mostly scatters. What it blocks toward is the deposit's own dark
            body colour, so this stays consistent with the mix on the glass.
        deposits: which deposit types may appear, as keys of
            :data:`DEPOSIT_ALBEDOS` (``None`` uses all of them). Each smudge
            patch independently draws one, so a single frame normally shows
            several. Restrict it to model a specific environment — e.g.
            ``("mineral_dust", "clay_soil")`` for a dirt road, ``("soot",
            "road_film")`` for urban traffic, ``("limescale",)`` for dried
            sea spray.

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
        deposits: tuple[str, ...] | None

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
        deposits: tuple[str, ...] | None = None,
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
        if deposits is not None:
            unknown = sorted(set(deposits) - set(DEPOSIT_ALBEDOS))
            if unknown:
                raise ValueError(
                    f"Unknown deposit type(s) {unknown}. "
                    f"Available: {sorted(DEPOSIT_ALBEDOS)}"
                )
            if not deposits:
                raise ValueError("`deposits` must name at least one deposit type.")
        self.deposits = deposits

    def get_params_dependent_on_data(
        self, params: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, Any]:
        height, width = params["shape"][:2]
        diagonal = float(np.hypot(width, height))
        rng = self.random_generator

        names = tuple(self.deposits) if self.deposits is not None else tuple(DEPOSIT_ALBEDOS)

        # Each smudge patch is its own substance, drawn independently, so one
        # frame normally carries several deposits at once — which is what a
        # real pane looks like and what a single global tint cannot express.
        smudge_count = self.py_random.randint(*self.smudge_count_range)
        chosen = [names[self.py_random.randrange(len(names))] for _ in range(smudge_count)]
        smudge_albedos = np.array(
            [DEPOSIT_ALBEDOS[name] for name in chosen] or [(1.0, 1.0, 1.0)], dtype=np.float32
        )
        mask, albedo = colored_blob_cluster(
            height,
            width,
            rng=rng,
            colors=smudge_albedos,
            size_range=(
                self.smudge_size_range[0] * diagonal,
                self.smudge_size_range[1] * diagonal,
            ),
        )

        def merge(
            layer: np.ndarray, color: tuple[float, float, float]
        ) -> None:
            """Fold one extra mask layer in, carrying its own deposit colour.

            Masks combine with a max, so the colour has to follow the same
            rule: wherever the new layer is the thicker deposit, it is also
            the one whose albedo is seen.
            """
            nonlocal mask, albedo
            takes_over = layer > mask
            albedo = np.where(
                takes_over[..., None], np.array(color, dtype=np.float32), albedo
            )
            mask = np.maximum(mask, layer)

        # Which substances were deposited, not just how much: the whole point
        # of the palette is that a pane carries several at once, and that is
        # invisible in the composited albedo field.
        record_draw(
            smudge_count=smudge_count,
            deposits=",".join(sorted(set(chosen))),
            deposit_albedo_mean=smudge_albedos.mean(axis=0),
        )

        arc_count = self.py_random.randint(*self.wipe_arc_count_range)
        if arc_count:
            arcs = _wipe_arcs(height, width, rng, arc_count, thickness_px=diagonal * 0.02)
            arcs = cv2.GaussianBlur(arcs, (0, 0), sigmaX=diagonal * 0.012)
            merge(arcs * 1.4, DEPOSIT_ALBEDOS[_WIPE_DEPOSIT])

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
            merge(spots, DEPOSIT_ALBEDOS[_WATER_SPOT_DEPOSIT])

        # Grime is never sharp — it is far outside the depth of field. The
        # albedo field is softened with it, or deposits would meet along hard
        # seams that the blurred mask no longer has.
        blur_sigma = max(1.0, diagonal * 0.006)
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=blur_sigma)
        albedo = cv2.GaussianBlur(albedo, (0, 0), sigmaX=blur_sigma)

        # Normalise to the requested (sparse) coverage so the knobs stay
        # predictable regardless of how the blobs happened to land.
        target_coverage = self.py_random.uniform(*self.coverage_range)
        current = float(mask.mean())
        if current > 1e-5:
            mask = np.clip(mask * (target_coverage / current), 0.0, 1.0)

        return {
            "dirt_mask": mask.astype(np.float32),
            "deposit_albedo": albedo.astype(np.float32),
            "veil_strength": self.py_random.uniform(*self.veil_strength_range),
            "highlight_boost": self.py_random.uniform(*self.highlight_boost_range),
            "scatter_radius": self.py_random.uniform(*self.scatter_radius_range) * diagonal,
            "absorption": self.py_random.uniform(*self.absorption_range),
        }

    def apply(
        self,
        img: np.ndarray,
        dirt_mask: np.ndarray,
        deposit_albedo: np.ndarray,
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

        # The deposit's single-scattering albedo multiplies the scattered
        # term. Light that scatters off a tan dust grain comes back tan, and
        # the 1 - albedo it did not return is simply absorbed — so tint and
        # absorption are the same physical quantity here rather than two
        # separately-tuned layers. It also stays energy-conserving: albedo is
        # at most 1, so this can only ever remove light from the scattered
        # term, never manufacture it.
        scattered = scatter_psf(emphasised, scatter_radius) * deposit_albedo

        veil = np.clip(veil_strength * mask, 0.0, 0.95)
        out = linear * (1.0 - veil) + scattered * veil

        # A little genuinely opaque, caked muck on top — thick enough to block
        # rather than scatter. It blocks toward a dark version of whatever
        # deposit is actually there, so a soot patch goes near-black while a
        # limescale crust stays pale.
        body_color = deposit_albedo * _OPAQUE_BODY_FRACTION
        out = out * (1.0 - absorption * mask) + body_color * (absorption * mask)

        return from_linear_unit(out, scale, is_integer, img.dtype)
