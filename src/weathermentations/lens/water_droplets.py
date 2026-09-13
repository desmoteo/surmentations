"""Water droplets adhering to the lens/glass.

Physical model
--------------
A droplet resting on glass is a plano-convex lens with a very short focal
length. For a water sphere of radius ``R`` and index ``n = 1.33`` the focal
distance is ``f = nR / (2(n-1)) ~ 2R``, so the scene — effectively at
infinity compared to ``R`` — is imaged *inside/just behind the droplet*,
**inverted and strongly demagnified**. That is why a real droplet contains a
tiny upside-down picture of the whole scene rather than a slightly warped
copy of whatever sits directly behind it. We render that directly: for a
point at offset ``(dx, dy)`` from the droplet centre we sample the source at
``centre - (dx, dy) * scene_compression`` — the negative sign inverts, the
``scene_compression > 1`` factor squeezes a wide field of view into the
droplet's footprint. A Snell's-law term (spherical-cap surface normal, air
-> water) is added on top, which bends rays progressively harder towards the
rim, reproducing the crowded, smeared ring of scene detail seen at a real
droplet's edge.

The second thing that makes real rain-on-glass photographs read as real is
*focus*: a camera that resolves the droplets is focused on the glass, so the
scene behind it falls out of focus. The droplets stay sharp and the world
goes soft — not the other way round. ``background_blur_range`` controls that
defocus; set it to ``(0, 0)`` for a deep-focus look where the scene stays
sharp (a camera focused past the glass, the way a dashcam usually is).

Droplets never overlap: surface tension keeps sessile droplets physically
separate until they coalesce (which this transform does not model), so
positions are rejection-sampled against every already-placed droplet with a
mandatory gap. Sizes follow a small-biased power distribution — real rain on
glass is a great many tiny droplets with a few large ones, not a uniform
scatter of mid-size blobs.

References: Garg & Nayar's adherent/raindrop appearance work; Qian et al.
2018 ("Attentive GAN for Raindrop Removal"); the drop-as-lens refraction and
focus split used in Lucas Bebber's WebGL rain experiments
(tympanus.net/codrops/2015/11/04/rain-water-effect-experiments/), which
renders a blurred background layer and a sharp, flipped, distorted
refraction inside each drop; and the raindrop-on-lens generator in ROLE
(github.com/ricky40403/ROLE).
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
from weathermentations.utils.noise import value_noise

_WATER_IOR = 1.33  # refractive index of water relative to air


def _droplet_rim_bend(
    dist_norm: np.ndarray, radius_px: float, cap_height_ratio: float
) -> np.ndarray:
    """Extra radial sampling offset from Snell refraction at the cap surface.

    ``dist_norm`` is distance from the droplet centre normalised by the
    droplet's (possibly angle-varying) local radius: 0 at the centre, 1 at
    the rim. Returns a radial pixel offset that is ~0 at the centre and grows
    steeply towards the rim, where the surface is steepest and bends rays
    most — the reason a real droplet's edge shows a crowded, smeared band of
    scene detail rather than a clean continuation of its middle.
    """
    dist_norm = np.clip(dist_norm, 0.0, 0.995)
    # Spherical-cap height profile z(d) = h*sqrt(1 - d^2); its slope gives the
    # surface-normal tilt from vertical.
    slope = dist_norm / np.sqrt(1.0 - dist_norm**2)
    theta_n = np.arctan(slope * cap_height_ratio)

    # Vertical incident ray: angle of incidence at the tilted surface is
    # theta_n, and Snell's law gives the refracted angle inside the water.
    sin_refracted = np.clip(np.sin(theta_n) / _WATER_IOR, -1.0, 1.0)
    bend = theta_n - np.arcsin(sin_refracted)
    return np.tan(bend) * radius_px * 3.0


def _local_radius(
    angle: np.ndarray, radius_px: float, irregularity: float, phase: tuple[float, float]
) -> np.ndarray:
    """Angle-varying droplet radius: a circle plus two sine harmonics.

    Real sessile droplets are never perfectly circular — surface tension
    against a slightly uneven or tilted surface pulls them into a mildly
    irregular blob. ``irregularity`` is the harmonics' combined amplitude as
    a fraction of ``radius_px`` (0 = perfect circle).
    """
    if irregularity <= 0.0:
        return np.full_like(angle, radius_px)
    wobble = 0.6 * np.sin(2 * angle + phase[0]) + 0.4 * np.sin(3 * angle + phase[1])
    return radius_px * (1.0 + irregularity * wobble)


def _render_droplet(
    sharp_source: np.ndarray,
    cx: float,
    cy: float,
    radius_px: float,
    cap_height_ratio: float,
    irregularity: float,
    phase: tuple[float, float],
    scene_compression: float,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Render one droplet's refracted patch and alpha mask.

    Returns ``(patch, alpha, x0, y0)``: an ``(S, S, C)`` refracted sample and
    an ``(S, S)`` alpha mask whose top-left corner sits at ``(x0, y0)`` in
    source-image coordinates.
    """
    size = max(4, int(round(radius_px * 2.4)))
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    centre = size / 2.0
    dx = xx - centre
    dy = yy - centre
    dist_px = np.sqrt(dx**2 + dy**2)
    angle = np.arctan2(dy, dx)
    local_radius = _local_radius(angle, radius_px, irregularity, phase)
    dist_norm = dist_px / local_radius

    inside = dist_norm <= 1.0

    # Inverted, demagnified image of the scene (the droplet is a short-focal
    # -length lens), plus the rim's progressively harder Snell bending.
    rim_bend = _droplet_rim_bend(dist_norm, radius_px, cap_height_ratio)
    safe_dist = np.where(dist_px > 1e-3, dist_px, 1.0)
    map_x = cx - dx * scene_compression - (dx / safe_dist) * rim_bend
    map_y = cy - dy * scene_compression - (dy / safe_dist) * rim_bend

    patch = cv2.remap(
        sharp_source,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    ).astype(np.float32)

    # Shading from the cap's surface normal: a tight Blinn-Phong specular
    # highlight where the "sky" light reflects off the bulge, and a broad
    # bright crescent on the opposite side where light refracts through the
    # meniscus — both are what make a droplet read as a 3D bead rather than a
    # flat distorted disc.
    normal_xy = np.stack([dx, dy], axis=-1) / (radius_px + 1e-6)
    normal_z = np.sqrt(np.clip(1.0 - np.sum(normal_xy**2, axis=-1), 0.0, 1.0))
    normal = np.concatenate([normal_xy, normal_z[..., None]], axis=-1)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-6

    light_dir = np.array([-0.35, -0.55, 0.76])
    light_dir /= np.linalg.norm(light_dir)
    half_vec = light_dir + np.array([0.0, 0.0, 1.0])
    half_vec /= np.linalg.norm(half_vec)
    ndoth = np.clip(np.sum(normal * half_vec, axis=-1), 0.0, 1.0)
    specular = ndoth**60
    crescent = np.clip(-np.sum(normal * light_dir, axis=-1), 0.0, 1.0) ** 3

    # Contact line: surface tension pools water at the rim, and rays there
    # exceed the critical angle and are lost to total internal reflection, so
    # the outline reads as a dark ring.
    rim = np.clip((dist_norm - 0.62) / 0.38, 0.0, 1.0) ** 1.4

    patch += specular[..., None] * 210.0
    patch += crescent[..., None] * 45.0
    patch -= rim[..., None] * 85.0
    patch = np.clip(patch, 0, 255)

    # Feather only the outermost pixel ring, so the droplet keeps a crisp
    # silhouette (it is in focus) without an aliased staircase edge.
    alpha = inside.astype(np.float32) * np.clip((1.0 - dist_norm) / 0.04, 0.0, 1.0)

    x0, y0 = int(round(cx - centre)), int(round(cy - centre))
    return patch.astype(sharp_source.dtype), alpha, x0, y0


def _sample_positions(
    rng: np.random.Generator,
    count: int,
    width: int,
    height: int,
    radius_range: tuple[float, float],
    size_bias: float,
    streak_probability: float,
    streak_length_range: tuple[float, float],
    min_gap_ratio: float,
    footprint_inflation: float = 1.0,
) -> list[tuple[float, float, float, float]]:
    """Greedy, vectorised rejection sampling of non-overlapping droplets.

    Returns ``(cx, cy, radius, stretch)`` tuples. Candidates are drawn in one
    batch and accepted greedily: a candidate is kept only if its footprint
    clears every already-accepted footprint by ``min_gap_ratio`` of their
    combined radii, so no two droplets ever touch. Radii follow
    ``r = r_min + (r_max - r_min) * u**size_bias`` with ``size_bias > 1``,
    biasing towards many small droplets and few large ones the way real rain
    on glass does.
    """
    oversample = 6
    n_candidates = max(count * oversample, count + 32)

    u = rng.random(n_candidates)
    radii = radius_range[0] + (radius_range[1] - radius_range[0]) * u**size_bias
    xs = rng.uniform(0, width, n_candidates)
    ys = rng.uniform(0, height, n_candidates)
    is_streak = rng.random(n_candidates) < streak_probability
    stretches = np.where(
        is_streak, rng.uniform(streak_length_range[0], streak_length_range[1], n_candidates), 1.0
    )
    # Gravity elongates every drop after placement, so reserve room for it
    # here or drops that only just fitted will overlap once stretched.
    footprints = radii * np.maximum(1.0, stretches) * footprint_inflation

    # Try the large droplets first: they are the ones that most need room, and
    # placing them early keeps them from being crowded out by small ones.
    order = np.argsort(-footprints)

    accepted_x = np.empty(count, dtype=np.float64)
    accepted_y = np.empty(count, dtype=np.float64)
    accepted_r = np.empty(count, dtype=np.float64)
    droplets: list[tuple[float, float, float, float]] = []

    n_accepted = 0
    for idx in order:
        if n_accepted >= count:
            break
        fx, fy, fr = xs[idx], ys[idx], footprints[idx]
        if n_accepted:
            min_dist = (accepted_r[:n_accepted] + fr) * (1.0 + min_gap_ratio)
            dx = accepted_x[:n_accepted] - fx
            dy = accepted_y[:n_accepted] - fy
            if np.any(dx * dx + dy * dy < min_dist * min_dist):
                continue
        accepted_x[n_accepted] = fx
        accepted_y[n_accepted] = fy
        accepted_r[n_accepted] = fr
        n_accepted += 1
        droplets.append((float(fx), float(fy), float(radii[idx]), float(stretches[idx])))

    return droplets


def _condensation_field(
    height: int,
    width: int,
    rng: np.random.Generator,
    patch_scale: float,
    coverage: float,
    edge_softness: float,
) -> np.ndarray:
    """Patchy condensation coverage in ``[0, 1]``.

    Condensation does not form as an even wash over a pane: it nucleates
    where the glass is coldest and least disturbed, so it builds into
    irregular patches with clear glass between them, and the patch edges are
    ragged rather than smooth. We build that from low-frequency value noise
    (the patches) modulated by a finer octave (the ragged edges), then
    threshold so that roughly ``coverage`` of the pane is affected.

    Crucially this returns *coverage*, not individual droplets. For a camera
    focused metres away, the droplets of a condensation film are orders of
    magnitude smaller than its circle of confusion, so they can never resolve
    as discrete specks — only as a milky veil that varies in density. Drawing
    them individually is what makes synthetic mist look like scattered dots
    stuck to the picture.

    The border of a stain is a *nucleation gradient*, not a boundary: droplet
    density simply thins out until the glass reads as clear, so the edge
    fades gradually over a wide band. ``edge_softness`` is the width of that
    fade as a fraction of the noise range, and it is smoothstepped so the
    transition eases in and out rather than ramping linearly into a visible
    crease. (A wipe edge is the opposite case and is handled separately —
    see where the wipe mask is applied.)
    """
    base_cells = max(2, int(round(2.0 / max(patch_scale, 1e-3))))
    patches = value_noise(height, width, rng=rng, octaves=3, base_cells=base_cells)
    detail = value_noise(height, width, rng=rng, octaves=4, base_cells=base_cells * 4)
    field = np.clip(patches * 0.75 + detail * 0.25, 0.0, 1.0)

    # Pick the threshold from the field's own distribution so `coverage`
    # means what it says regardless of how the noise happened to land.
    threshold = float(np.quantile(field, 1.0 - np.clip(coverage, 0.0, 1.0)))
    softness = max(edge_softness, 1e-3)
    # Centre the fade on the threshold so `coverage` still means what it says
    # once the band is wide.
    ramp = np.clip((field - (threshold - 0.5 * softness)) / softness, 0.0, 1.0)
    smooth = ramp * ramp * (3.0 - 2.0 * ramp)  # smoothstep
    return smooth.astype(np.float32)


def _trail_droplets(
    rng: np.random.Generator,
    cx: float,
    cy: float,
    radius: float,
    length_px: float,
    scale_range: tuple[float, float],
) -> list[tuple[float, float, float]]:
    """Shrinking droplets left in the wake of a drop that has slid down.

    A sliding drop sheds part of itself continuously, so behind (above) it
    sits a tapering chain of smaller droplets rather than a smooth smear, and
    the drop itself arrives at the bottom smaller than it started.
    """
    droplets: list[tuple[float, float, float]] = []
    y = cy - radius
    x = cx
    travelled = 0.0
    while travelled < length_px:
        scale = float(rng.uniform(*scale_range))
        child_r = radius * scale
        if child_r < 0.7:
            break
        # Spacing grows with the child's size so trail droplets never merge.
        step = child_r * float(rng.uniform(2.2, 4.5))
        y -= step
        travelled += step
        x += float(rng.uniform(-radius, radius)) * 0.12
        # Taper: the further back up the trail, the smaller the remnant.
        taper = 1.0 - 0.65 * (travelled / max(length_px, 1e-3))
        droplets.append((x, y, max(0.7, child_r * taper)))
    return droplets


class WaterDroplets(WeatherTransform):
    """Refractive, non-overlapping water droplets adhering to the lens.

    All size-like parameters are expressed as a **fraction of the image
    diagonal**, not in pixels: a droplet of a given physical size subtends a
    fixed fraction of the camera's field of view, so its pixel size scales
    with resolution. This keeps the effect looking identical whether it is
    applied to a 256px crop or a 4K frame.

    Args:
        droplet_count_range: number of droplets to *attempt* to place. Fewer
            may render if the canvas cannot fit them without overlap.
        droplet_radius_range: droplet radius, as a fraction of the image
            diagonal (0.01 on a 1000px-diagonal image is a 10px radius).
        size_bias: exponent biasing the radius distribution towards the small
            end (1 = uniform, higher = many small droplets and few large
            ones, as real rain on glass looks).
        cap_height_ratio_range: droplet "bulge" as a fraction of its radius
            (spherical-cap height / radius) — higher bends rays harder at the
            rim. Typical sessile water droplets: 0.3-0.6.
        scene_compression_range: how much of the scene each droplet's lens
            squeezes into its footprint. 1 would be a plain inverted copy;
            real droplets compress a wide field of view into a tiny inverted
            image, so values well above 1 are what read as correct.
        background_blur_range: defocus applied to the scene *outside* the
            droplets, as a fraction of the image diagonal. Leave at 0 for the
            usual camera case — focused on the scene, not on the glass — and
            raise it only for the "focused on the window" look, where the
            world behind the drops goes soft instead.
        droplet_defocus_range: how far out of focus the droplets themselves
            are, as a fraction of their own radius. A camera focused on the
            distance renders anything sitting on its front element wildly
            defocused, so the drops soften and their refracted miniature
            scene smears — this is the normal case for a dashcam or a
            surveillance camera.
        condensation_coverage_range: fraction of the pane that has fogged
            over. Condensation forms in patches, so well below 1 leaves clear
            glass between them.
        condensation_patch_scale_range: size of those patches, as a fraction
            of the frame. Small values give many small patches.
        condensation_edge_softness_range: width of the gradual fade at a
            stain's border. A stain thins out through a nucleation gradient,
            so this wants to be wide; the crisp boundaries in the result come
            from drops picking the film up, not from the stain edges.
        condensation_milkiness_range: how far a fogged patch washes toward
            white. This is the whole look: a far-focused camera cannot
            resolve the droplets, only the milky veil they scatter into.
        condensation_blur_range: how much a fogged patch scatters/softens
            what is behind it, as a fraction of the image diagonal.
        trail_probability: chance that a drop has slid down and left a trail.
        trail_length_range: trail length, as a fraction of image height.
        streak_probability: chance a droplet renders as a vertically
            elongated sliding streak (gravity trail) instead of a round bead.
        streak_length_range: vertical elongation multiplier for such streaks.
        vertical_stretch_range: how much gravity elongates a resting drop.
        shape_irregularity_range: how far a droplet's outline deviates from a
            perfect circle, as a fraction of its radius.
        min_gap_ratio: minimum gap between two droplet footprints, as a
            fraction of their combined radii (0 would let droplets touch).

    Targets:
        image

    Image types:
        uint8, float32
    """

    class InitSchema(BaseTransformInitSchema):
        droplet_count_range: Annotated[
            tuple[int, int],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(1, 2000)),
        ]
        droplet_radius_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0005, 0.25)),
        ]
        size_bias: float = Field(ge=0.5, le=8.0)
        cap_height_ratio_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.05, 1.0)),
        ]
        scene_compression_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(1.0, 20.0)),
        ]
        background_blur_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 0.05)),
        ]
        droplet_defocus_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 1.5)),
        ]
        condensation_coverage_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 1.0)),
        ]
        condensation_patch_scale_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.02, 1.5)),
        ]
        condensation_edge_softness_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.01, 1.0)),
        ]
        condensation_milkiness_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 1.0)),
        ]
        condensation_blur_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 0.08)),
        ]
        trail_probability: float = Field(ge=0.0, le=1.0)
        trail_length_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 1.0)),
        ]
        streak_probability: float = Field(ge=0.0, le=1.0)
        streak_length_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(1.0, 6.0)),
        ]
        vertical_stretch_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(1.0, 2.5)),
        ]
        shape_irregularity_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 0.4)),
        ]
        min_gap_ratio: float = Field(ge=0.0, le=2.0)

    def __init__(
        self,
        droplet_count_range: tuple[int, int] = (30, 90),
        droplet_radius_range: tuple[float, float] = (0.003, 0.028),
        size_bias: float = 2.5,
        cap_height_ratio_range: tuple[float, float] = (0.35, 0.65),
        scene_compression_range: tuple[float, float] = (3.0, 7.0),
        background_blur_range: tuple[float, float] = (0.0, 0.0),
        droplet_defocus_range: tuple[float, float] = (0.12, 0.35),
        condensation_coverage_range: tuple[float, float] = (0.15, 0.5),
        condensation_patch_scale_range: tuple[float, float] = (0.15, 0.5),
        condensation_edge_softness_range: tuple[float, float] = (0.3, 0.6),
        condensation_milkiness_range: tuple[float, float] = (0.25, 0.6),
        condensation_blur_range: tuple[float, float] = (0.006, 0.02),
        trail_probability: float = 0.35,
        trail_length_range: tuple[float, float] = (0.08, 0.35),
        streak_probability: float = 0.15,
        streak_length_range: tuple[float, float] = (1.5, 3.0),
        vertical_stretch_range: tuple[float, float] = (1.15, 1.5),
        shape_irregularity_range: tuple[float, float] = (0.03, 0.12),
        min_gap_ratio: float = 0.12,
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.droplet_count_range = droplet_count_range
        self.droplet_radius_range = droplet_radius_range
        self.size_bias = size_bias
        self.cap_height_ratio_range = cap_height_ratio_range
        self.scene_compression_range = scene_compression_range
        self.background_blur_range = background_blur_range
        self.droplet_defocus_range = droplet_defocus_range
        self.condensation_coverage_range = condensation_coverage_range
        self.condensation_patch_scale_range = condensation_patch_scale_range
        self.condensation_edge_softness_range = condensation_edge_softness_range
        self.condensation_milkiness_range = condensation_milkiness_range
        self.condensation_blur_range = condensation_blur_range
        self.trail_probability = trail_probability
        self.trail_length_range = trail_length_range
        self.streak_probability = streak_probability
        self.streak_length_range = streak_length_range
        self.vertical_stretch_range = vertical_stretch_range
        self.shape_irregularity_range = shape_irregularity_range
        self.min_gap_ratio = min_gap_ratio

    def get_params_dependent_on_data(
        self, params: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, Any]:
        height, width = params["shape"][:2]
        diagonal = float(np.hypot(width, height))
        count = self.py_random.randint(*self.droplet_count_range)

        radius_px_range = (
            self.droplet_radius_range[0] * diagonal,
            self.droplet_radius_range[1] * diagonal,
        )
        placements = _sample_positions(
            self.random_generator,
            count,
            width,
            height,
            radius_px_range,
            self.size_bias,
            self.streak_probability,
            self.streak_length_range,
            self.min_gap_ratio,
            footprint_inflation=self.vertical_stretch_range[1],
        )

        rng = self.random_generator
        droplets = []
        # Every droplet that lands on the pane, trail remnants included, is
        # checked against everything already placed: the no-touching rule has
        # to hold for the whole field, not just the drops placed first.
        occupied_x = [p[0] for p in placements]
        occupied_y = [p[1] for p in placements]
        occupied_r = [p[2] * max(1.0, p[3]) * self.vertical_stretch_range[1] for p in placements]

        def _fits(x: float, y: float, r: float) -> bool:
            for ox, oy, orad in zip(occupied_x, occupied_y, occupied_r, strict=True):
                min_dist = (r + orad) * (1.0 + self.min_gap_ratio)
                if (x - ox) ** 2 + (y - oy) ** 2 < min_dist**2:
                    return False
            return True

        # Where a drop has slid down it squeegees the mist away, leaving a
        # visibly clearer track behind it. That wiped track is as much of the
        # "rainy window" look as the drops themselves.
        wipe = np.zeros((height, width), dtype=np.float32)

        for cx, cy, radius, stretch in placements:
            trail = []
            if self.py_random.random() < self.trail_probability:
                trail_length = self.py_random.uniform(*self.trail_length_range) * height
                trail = _trail_droplets(rng, cx, cy, radius, trail_length, (0.25, 0.5))
                cv2.line(
                    wipe,
                    (int(round(cx)), int(round(cy))),
                    (int(round(cx)), int(round(cy - trail_length))),
                    1.0,
                    thickness=max(1, int(round(radius * 0.86))),
                    lineType=cv2.LINE_AA,
                )

            shared = (
                self.py_random.uniform(*self.cap_height_ratio_range),
                self.py_random.uniform(*self.shape_irregularity_range),
                (
                    self.py_random.uniform(0, 2 * math.pi),
                    self.py_random.uniform(0, 2 * math.pi),
                ),
            )
            gravity_stretch = self.py_random.uniform(*self.vertical_stretch_range)
            droplets.append((cx, cy, radius, shared[0], stretch * gravity_stretch, *shared[1:]))
            for tx, ty, tr in trail:
                if not _fits(tx, ty, tr * gravity_stretch):
                    continue
                occupied_x.append(tx)
                occupied_y.append(ty)
                occupied_r.append(tr * gravity_stretch)
                droplets.append((tx, ty, tr, shared[0], gravity_stretch, shared[1], shared[2]))

        condensation = _condensation_field(
            height,
            width,
            rng,
            patch_scale=self.py_random.uniform(*self.condensation_patch_scale_range),
            coverage=self.py_random.uniform(*self.condensation_coverage_range),
            edge_softness=self.py_random.uniform(*self.condensation_edge_softness_range),
        )

        # A drop sliding down does not *fade* the film out, it picks it up:
        # its contact line sweeps the glass mechanically, so the track it
        # leaves has a crisp, well-defined boundary — the opposite of the
        # stain border's gradual nucleation falloff above. So the wipe mask
        # keeps its hard edge here (only the anti-aliasing already baked in
        # by the line drawing), and is applied after the soft-edged field.
        condensation *= 1.0 - np.clip(wipe, 0.0, 1.0)

        return {
            "droplets": droplets,
            "condensation": condensation,
            "scene_compression": self.py_random.uniform(*self.scene_compression_range),
            "background_blur": self.py_random.uniform(*self.background_blur_range) * diagonal,
            "droplet_defocus": self.py_random.uniform(*self.droplet_defocus_range),
            "milkiness": self.py_random.uniform(*self.condensation_milkiness_range),
            "condensation_blur": self.py_random.uniform(*self.condensation_blur_range) * diagonal,
        }

    def apply(
        self,
        img: np.ndarray,
        droplets: list[tuple[float, float, float, float, float, float, tuple[float, float]]],
        condensation: np.ndarray,
        scene_compression: float,
        background_blur: float,
        droplet_defocus: float,
        milkiness: float,
        condensation_blur: float,
        **params: Any,
    ) -> np.ndarray:
        if not droplets:
            return img

        height, width = img.shape[:2]

        # The camera is focused on the glass, so the scene behind the droplets
        # is out of focus while the droplets themselves — and the miniature
        # scene each one images — stay sharp.
        if background_blur > 0.1:
            ksize = int(background_blur * 6) | 1
            out = cv2.GaussianBlur(img, (ksize, ksize), sigmaX=background_blur)
        else:
            out = img.copy()

        # Condensation. A camera focused metres away sees the film on its own
        # front element far outside the depth of field, so the individual
        # droplets never resolve — they only scatter. The visible result is a
        # milky, patchy veil that softens and lifts whatever is behind it,
        # heaviest where the film is thickest and absent where a sliding drop
        # has wiped the glass clear.
        if condensation is not None and float(condensation.max()) > 1e-3:
            base = out.astype(np.float32)
            fogged: np.ndarray = cv2.GaussianBlur(
                base, (0, 0), sigmaX=max(0.6, condensation_blur)
            )
            # Scattering is spectrally flat here (droplets >> wavelength), so
            # the veil desaturates toward its own luminance as it brightens.
            grey = fogged.mean(axis=2, keepdims=True)
            milky = fogged * (1.0 - milkiness) + (grey * 0.45 + 255.0 * 0.55) * milkiness
            c = condensation[..., None]
            out = np.clip(base * (1.0 - c) + milky * c, 0, 255).astype(img.dtype)

        for cx, cy, radius, cap_ratio, stretch, irregularity, phase in droplets:
            patch, alpha, x0, y0 = _render_droplet(
                img, cx, cy, radius, cap_ratio, irregularity, phase, scene_compression
            )
            if stretch > 1.0:
                new_h = int(round(patch.shape[0] * stretch))
                patch = cv2.resize(patch, (patch.shape[1], new_h), interpolation=cv2.INTER_LINEAR)
                alpha = cv2.resize(alpha, (alpha.shape[1], new_h), interpolation=cv2.INTER_LINEAR)
                y0 = int(round(cy - new_h / 2.0))

            if droplet_defocus > 0.01:
                # Pad before blurring. Defocus spreads the droplet *beyond*
                # its own footprint, so blurring the tight patch in place
                # truncates the spread at the patch rectangle and leaves a
                # hard rectangular edge around every drop.
                sigma = max(0.4, droplet_defocus * radius)
                pad = int(np.ceil(3.0 * sigma))
                patch = cv2.copyMakeBorder(
                    patch, pad, pad, pad, pad, cv2.BORDER_REPLICATE
                )
                alpha = cv2.copyMakeBorder(
                    alpha, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0.0
                )
                patch = cv2.GaussianBlur(patch, (0, 0), sigmaX=sigma)
                alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=sigma)
                x0 -= pad
                y0 -= pad

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
