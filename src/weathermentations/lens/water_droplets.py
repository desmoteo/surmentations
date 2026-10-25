"""Water on the camera's front glass: drops, runners and condensation.

What the camera is looking at
-----------------------------
A surveillance camera is focused metres away, and the water is on a pane a
few centimetres in front of its lens. Everything this module draws follows
from taking that seriously, and most of what synthetic rain-on-glass gets
wrong follows from not doing so — from drawing the window as a macro
photographer sees it, crisp beads each holding a tiny sharp picture.

**Defocus is set by the aperture, not chosen.** With the lens focused at
infinity, a point on the glass is imaged as a disc whose diameter, measured
on the glass, *is the diameter of the entrance pupil* — whatever the distance
between them. So a drop looks sharp only if it is much wider than the
aperture. Against the 2.5 mm pupil of a 4 mm f/1.6 lens, a 0.6 mm bead is
spread over seventeen times its own area and survives as a 6% dip in local
contrast; only drops of several millimetres remain as soft blobs. Stop the
iris down to 0.5 mm in daylight and the same drops are nearly crisp. That is
why ``aperture_diameter_range_mm`` replaces a free "defocus" knob, and why
the same wet glass looks so different by day and by night.

The image is formed accordingly. Every ray through the pupil that reaches a
given pixel crosses the glass somewhere inside that disc; where it crosses
clear glass it sees the scene in the pixel's own direction, undisturbed, and
where it crosses water it is bent and sees somewhere else. So::

    out = (1 - blur(alpha)) * scene + blur(alpha * refracted)

with ``blur`` the aperture disc. Small drops do not get blurry *pictures* in
them; they fade into the scene behind.

**All the water is one surface.** Drops are not drawn one at a time. Each is
a lobe of a single height field over the pane, and lobes that touch merge
through a smooth union, so two drops caught mid-coalescence form a peanut,
and a runner is an outline drawn row by row — a slug, a pear, or a head
trailing a thread that swells and breaks into beads — so no two are alike.
How far any drop departs from a round cap follows its size against the
capillary length: fine beads are round because surface tension makes them
so, and large drops are lopsided and bottom-heavy. Refraction is then one remap of the
whole frame through the field's slope: Snell's law at the water-air surface
gives the deflection, the focal length in pixels turns it into an offset, and
because a convex surface deflects towards its own centre the picture inside
every drop comes out inverted and squeezed, with no "compression" parameter.
The dark outline is not painted either. Past 48.8 degrees of surface tilt the
ray cannot leave the water at all — total internal reflection — and looks
back into the housing.

**Nothing is lit by a light that is not there.** A camera behind the glass
never sees a specular highlight off the outside of a drop; that reflection
leaves in the other direction. What it sees as a glint is the *refracted
image of a bright source*, so glints here are refracted, their brightness is
relative to the scene's own brightest light, and at night there are none
unless there are lamps. The one light that does come from the camera is its
infrared ring, which is handled separately: see ``illuminator_glow_range``.

Condensation
------------
The droplets of a condensation film are tens of micrometres across — orders
of magnitude below the aperture disc — so they can only ever appear as
scattering. Scattering *redistributes* the light arriving at the glass: most
of it into a narrow forward lobe, which softens, and the rest into a long
tail, which puts a glow round every light and a veil over the frame. All of
it is computed from the scene, in linear light. That is
the whole difference between a fogged lens by day (a milky loss of contrast,
because the wide lobe averages a bright scene) and at night (a frame that
stays dark, with every lamp carrying a large glow). Blending towards a fixed
white instead adds light from nowhere: on a night street it raised the
fogged areas from level 35 to level 102.

Where the film sits is decided by the temperature of the glass, which varies
smoothly: it clears outward from wherever the glass is warmest, so the film
is one region with one ragged front, not scattered patches.

Shedding
--------
Whether a drop runs is still derived rather than drawn: it slides once its
weight beats contact-line pinning, at a radius set by the capillary length.
A runner's track meanders, because the contact line jumps sideways when it
meets a pinned drop or a dirty patch; it wipes the film off the glass as it
goes, swallows the drops in its path, and leaves a few beads behind.

References: Garg & Nayar on adherent drops; You et al. 2013 ("Adherent
Raindrop Detection and Removal in Video") for the aperture-disc image
formation; Halimeh & Roser 2009 for refraction through a drop on a
windscreen; Qian et al. 2018 ("Attentive GAN for Raindrop Removal"); and the
height-field treatment of merging drops in Lucas Bebber's WebGL rain
(tympanus.net/codrops/2015/11/04/rain-water-effect-experiments/).
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
from weathermentations.diagnostics import record_draw
from weathermentations.utils.color import sky_radiance_linear
from weathermentations.utils.noise import value_noise
from weathermentations.utils.photometry import from_linear_unit, to_linear_unit

_WATER_IOR = 1.33  # refractive index of water relative to air

# Water against air at room temperature.
_SURFACE_TENSION_N_PER_M = 0.072
_WATER_DENSITY_KG_PER_M3 = 1000.0
_GRAVITY_M_PER_S2 = 9.81

# Exponent of the dropwise-condensation size distribution. Coalescence-driven
# growth on a surface reaches a self-similar regime whose drop-size
# distribution is a power law N(r) ~ r^-tau with tau close to 2 (Family &
# Meakin; Rose's dropwise-condensation work). That is a far heavier tail than
# any reshaped uniform: it is what puts hundreds of fine drops on the glass
# for every large one.
_CONDENSATION_TAU = 2.0

# Level at which the summed lobe field is cut to give the water's outline.
# Each lobe is 1 at its centre and exactly this at its own rim, and reaches a
# little further before it hits zero — that reach is what lets two lobes that
# nearly touch pull a neck of water between them. Beyond its rim a lobe falls
# to zero along a parabola that meets both ends without a corner, which fixes
# the reach at a third of the radius: short enough that drops held apart by
# the default gap stay separate.
_OUTLINE_LEVEL = 0.25
_LOBE_REACH = 1.0 + _OUTLINE_LEVEL / (1.0 - _OUTLINE_LEVEL)

# Lobes smaller than this are drawn round: the outline wobble is a fraction
# of a pixel there and costs an arctangent per pixel to compute.
_MIN_RADIUS_FOR_WOBBLE_PX = 4.0

# The aperture disc is convolved at reduced resolution once it is wider than
# this. A disc that large leaves nothing fine to lose, and the cost of the
# convolution grows with the square of its width.
_MAX_DISC_KERNEL_PX = 12.0

# Where the light a film scatters goes. Scattering by droplets is strongly
# forward-peaked with a long tail, which no single blur reproduces: most of
# the light lands close by (the softening, whose width is the transform's
# `condensation_blur`), some of it several lamp-widths out (the glow round a
# light), and the rest across the whole frame (the veil). Each of the two
# outer lobes is ``(share of the scattered light, sigma as a fraction of the
# image diagonal)``; the narrow one takes what is left.
_GLOW_LOBE = (0.25, 0.04)
_VEIL_LOBE = (0.30, 0.16)
# Not all of it comes back: some is scattered out of the lens altogether.
_FILM_ALBEDO = 0.9

# A clipped pixel is a light whose real radiance is unknown but well above
# full scale. The scattered term treats it as this many times brighter, which
# is what gives a lamp behind fogged glass its glow.
_CLIPPED_SOURCE_GAIN = 8.0

# What a ray sees after total internal reflection: the inside of the housing,
# lit by whatever stray light is about — a fraction of the scene's mean.
_HOUSING_ALBEDO = 0.25

# A source outside the frame, relative to the scene's brightest light.
_GLINT_SOURCE_GAIN = 4.0
_GLINT_ANGULAR_RADIUS_RAD = math.radians(3.0)

# Width of the block the wet pixels' sampling coordinates are folded into.
_REMAP_BLOCK_WIDTH = 1024

# How much of the infrared ring's light comes back off the water. A film
# scatters it diffusely; a drop returns it where the ring is internally
# reflected, which is mostly its rim.
_IR_FILM_RETURN = 0.22
_IR_DROP_RETURN = 0.3

# Harmonics of a pinned drop's outline, from the second (an ellipse at any
# angle) to the fifth. Higher ones are below what a contact line holds
# against surface tension, which straightens anything that fine.
_OUTLINE_HARMONICS = (2, 3, 4, 5)

# How far a drop departs from a round cap grows with its size against the
# capillary length: surface tension keeps a small drop a near-perfect cap,
# and loses to gravity and to contact-line pinning as the drop grows. This
# is the departure at one capillary length, relative to the drawn amplitude.
_DEFORMATION_AT_CAPILLARY_LENGTH = 1.8

# A thread of water breaks into beads about nine thread-radii apart
# (Rayleigh-Plateau); real tails scatter around that.
_BEADING_WAVELENGTH_RADII = (6.0, 12.0)


def capillary_length_mm() -> float:
    """``sqrt(sigma / (rho g))`` for water, in millimetres: about 2.7 mm.

    The length at which surface tension and gravity balance, and therefore
    the scale that decides whether a drop on a vertical pane stays put or
    runs. Every "does this drop slide?" question reduces to a comparison
    against it.
    """
    length_m = np.sqrt(
        _SURFACE_TENSION_N_PER_M / (_WATER_DENSITY_KG_PER_M3 * _GRAVITY_M_PER_S2)
    )
    return float(length_m * 1e3)


def shedding_radius_mm(contact_angle_hysteresis: float = 0.35) -> float:
    """Base radius above which a drop on a vertical surface starts to slide.

    A sessile drop is pinned by contact-angle hysteresis: the contact line
    resists motion with a force proportional to the surface tension, the
    drop's width and ``cos(theta_receding) - cos(theta_advancing)``. Gravity
    pulls with the drop's weight, which grows as the cube of its size. Volume
    beating perimeter is the whole story, and setting the two equal leaves
    the capillary length scaled by the hysteresis term.

    This is why a window in rain shows a great many small drops sitting
    still and only a handful of large ones running: sliding is not a
    property some drops randomly have, it is what happens once a drop grows
    past this size. Deriving it removes ``streak_probability`` as a free
    knob — the fraction of drops that slide now follows from the size
    distribution.
    """
    return capillary_length_mm() * float(np.sqrt(max(contact_angle_hysteresis, 1e-3)))


def power_law_radii(
    rng: np.random.Generator, count: int, r_min: float, r_max: float, tau: float
) -> np.ndarray:
    """Sample ``count`` radii from ``N(r) ~ r^-tau`` truncated to ``[r_min, r_max]``.

    Inverted analytically; for ``tau == 1`` the inverse is logarithmic and is
    handled separately.
    """
    if count <= 0:
        return np.zeros(0, dtype=np.float64)
    u = rng.random(count)
    low = max(r_min, 1e-6)
    high = max(r_max, low * (1.0 + 1e-9))
    if abs(tau - 1.0) < 1e-6:
        return low * (high / low) ** u
    exponent = 1.0 - tau
    return (low**exponent + u * (high**exponent - low**exponent)) ** (1.0 / exponent)


def _sample_positions(
    rng: np.random.Generator,
    count: int,
    width: int,
    height: int,
    radius_range: tuple[float, float],
    tau: float,
    shedding_radius_px: float,
    streak_length_range: tuple[float, float],
    min_gap_ratio: float,
    footprint_inflation: float = 1.0,
) -> list[tuple[float, float, float, float]]:
    """Greedy, vectorised rejection sampling of non-overlapping droplets.

    Returns ``(cx, cy, radius, stretch)`` tuples. Candidates are drawn in one
    batch and accepted greedily: a candidate is kept only if its footprint
    clears every already-accepted footprint by ``min_gap_ratio`` of their
    combined radii, so no two droplets ever touch.

    Radii follow the ``r^-tau`` power law of dropwise condensation, and
    whether a drop is drawn as a sliding streak is decided by whether it
    exceeds ``shedding_radius_px`` — not by a coin flip. Both together are
    what produce the real pattern: a dense spatter of fine stationary beads
    with a few large runners through it.
    """
    oversample = 6
    n_candidates = max(count * oversample, count + 32)

    radii = power_law_radii(rng, n_candidates, radius_range[0], radius_range[1], tau)
    xs = rng.uniform(0, width, n_candidates)
    ys = rng.uniform(0, height, n_candidates)

    # Sliding is not random: a drop runs once it outgrows the size at which
    # gravity beats contact-line pinning. How elongated it becomes still is,
    # since that depends on how far past the threshold it has grown and on
    # how long it has been running.
    is_streak = radii > shedding_radius_px
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


def _runner_path(
    rng: np.random.Generator,
    x: float,
    y: float,
    radius: float,
    length_px: float,
    straight_px: float,
) -> np.ndarray:
    """The track a runner has left, as ``(K, 2)`` points from its head upward.

    A drop sliding down glass does not fall in a straight line. Its contact
    line is pinned and released over and over, so the track is a run of
    near-vertical stretches joined by short sideways jogs, each about a drop
    width, where it met a pinned bead or a patch that wets differently. No
    jog falls within the first ``straight_px`` — that stretch is the drop's
    own tail, still one body of water — but it wanders there like the rest.
    """
    step = max(1.0, radius * 0.25)
    count = max(2, int(math.ceil(length_px / step)) + 1)
    travelled = np.arange(count, dtype=np.float64) * step

    # Slow wander: a random walk a few percent of the radius per step.
    drift = np.cumsum(rng.normal(0.0, 0.02 * radius, count))
    drift -= drift[0]

    # Jogs. A run between two of them is several drop widths long.
    jog = np.zeros(count, dtype=np.float64)
    position = straight_px + float(rng.uniform(2.0, 9.0)) * radius
    while position < length_px:
        shift = float(rng.uniform(0.5, 1.8)) * radius * float(rng.choice((-1.0, 1.0)))
        span = max(step, float(rng.uniform(1.2, 3.5)) * radius)
        ramp = np.clip((travelled - position) / span, 0.0, 1.0)
        jog += shift * ramp * ramp * (3.0 - 2.0 * ramp)
        position += span + float(rng.uniform(4.0, 14.0)) * radius

    xs = x + drift + jog
    return np.stack([xs, y - travelled], axis=1)


def _draw_track(wipe: np.ndarray, path: np.ndarray, radius: float) -> None:
    """Rasterise the strip of glass a runner has swept clear.

    The strip is as wide as the drop was when it passed, and the drop was
    smaller then — it grows by collecting what is in its way — so the track
    narrows towards its top. It is stamped as the drop's own round footprint
    at each position rather than as thick line segments, whose square ends
    turn every sideways jog into a staircase.
    """
    last = max(len(path) - 1, 1)
    for index, (x, y) in enumerate(path):
        swept = radius * (0.85 - 0.4 * index / last)
        cv2.circle(
            wipe,
            (int(round(x * 16.0)), int(round(y * 16.0))),
            max(16, int(round(swept * 16.0))),
            1.0,
            thickness=-1,
            lineType=cv2.LINE_AA,
            shift=4,
        )


def _trail_beads(
    rng: np.random.Generator, path: np.ndarray, radius: float, start_px: float
) -> list[tuple[float, float, float]]:
    """Beads a runner has left behind it, as ``(x, y, radius)``.

    The receding contact line sheds a little water wherever it is briefly
    pinned. What stays is small — a fraction of the track's width — and
    smaller the further back it was left, because the drop was smaller then.
    """
    beads: list[tuple[float, float, float]] = []
    length = float(path[0, 1] - path[-1, 1])
    travelled = start_px
    while travelled < length:
        age = travelled / max(length, 1e-3)
        bead = radius * float(rng.uniform(0.1, 0.3)) * (1.0 - 0.6 * age)
        if bead < 0.7:
            break
        y = float(path[0, 1]) - travelled
        # `path` runs upward, so its y decreases; np.interp wants increasing.
        x = float(np.interp(y, path[::-1, 1], path[::-1, 0]))
        x += float(rng.uniform(-0.35, 0.35)) * radius * (0.85 - 0.4 * age)
        beads.append((x, y, bead))
        travelled += bead * float(rng.uniform(3.0, 9.0))
    return beads


def _lobe_profile(distance_sq: np.ndarray) -> np.ndarray:
    """A lobe's contribution to the level field, from its normalised distance.

    Inside its own rim (distance below 1) this is ``1 - (1 - cut) d^2``,
    which is what makes a lone lobe an exact ellipsoidal cap. Outside it
    continues as the parabola that reaches zero with zero slope.

    The smooth landing matters wherever one lobe ends inside another. A lobe
    simply clipped at zero has a corner there; the summed surface inherits
    it as a crease, and a crease in a refracting surface is a line — every
    merged drop showed the outline of the other lobe drawn across it.
    """
    falloff = 1.0 - _OUTLINE_LEVEL
    distance = np.sqrt(np.minimum(distance_sq, _LOBE_REACH**2))
    skirt = (falloff * falloff / _OUTLINE_LEVEL) * (_LOBE_REACH - distance) ** 2
    return np.where(distance_sq <= 1.0, 1.0 - falloff * distance_sq, skirt)


class _RunnerBody:
    """A running drop's outline, row by row.

    ``top`` is the image row of the first entry. For each row the outline is
    ``centre - left`` to ``centre + right``, so a body can lean, curve, bulge
    on one side and pinch off into separate beads — none of which a circle
    with a tail can do. ``gain`` fades the body in over the rows where it
    emerges from the head, so that it has no end inside the head to show.
    """

    __slots__ = ("cap", "centre", "gain", "left", "right", "top")

    def __init__(
        self,
        top: int,
        centre: np.ndarray,
        left: np.ndarray,
        right: np.ndarray,
        gain: np.ndarray,
        cap: float,
    ) -> None:
        self.top = top
        self.centre = centre
        self.left = left
        self.right = right
        self.gain = gain
        self.cap = cap


def _edge_noise(rng: np.random.Generator, along_px: np.ndarray, radius: float) -> np.ndarray:
    """Smooth unevenness of one side of a running drop, about zero.

    A contact line dragged over glass catches and releases, so neither side
    of a runner is a clean curve and the two sides do not match. Features are
    about a drop width long: finer ones surface tension pulls flat.
    """
    noise = np.zeros(along_px.shape, dtype=np.float64)
    for share in (0.65, 0.35):
        wavelength = radius * float(rng.uniform(1.0, 4.5))
        phase = float(rng.uniform(0.0, 2.0 * math.pi))
        noise += share * np.sin(2.0 * math.pi * along_px / wavelength + phase)
    return noise * float(rng.uniform(0.06, 0.24))


def _runner_body(
    rng: np.random.Generator,
    head_x: float,
    head_y: float,
    radius: float,
    tail_px: float,
    cap: float,
    path: np.ndarray | None,
) -> _RunnerBody:
    """Draw what a running drop trails behind its head.

    No two runners on a window look alike, and the differences are not
    noise: they are where each one is in the same sequence. A drop that has
    just let go is a blunt slug. As it speeds up its receding contact line
    is dragged out, first into a pear and then into a head trailing a thread
    of water; the thread is unstable, swells into a row of bulges, and
    finally breaks into beads from its far end. Which stage a runner shows,
    how long each part is, and which way it leans are drawn here.

    The head itself is an ordinary drop — the caller adds it as a lobe with
    its own irregular outline — and this is the rest, joined to it through
    the same smooth union as any two drops that touch:

    * a **body**, rooted inside the head and falling away above it as
      ``(1 - (s/a)^2)^q``: ``q`` near a half is an ellipse (the slug), higher
      values pinch it in towards a shoulder;
    * a **thread**, a fraction of the head's width, with a ripple that grows
      along it and cuts it into beads where it exceeds the thread itself;
    * two **sides** that are uneven and do not mirror each other;
    * a **centreline** that follows the track the drop has run down, bent a
      little further by its own drift.
    """
    length = radius + tail_px
    top = int(math.floor(head_y - length))
    rows = np.arange(top, int(math.ceil(head_y)) + 1, dtype=np.float64)
    above = head_y - rows  # positive up the pane
    along = np.clip(above / length, 0.0, 1.0)

    # Body. A slug has no thread and keeps its whole length as body.
    slug = rng.random() < 0.25
    thread = 0.0 if slug else float(rng.uniform(0.07, 0.28))
    root = float(rng.uniform(0.6, 0.95))
    reach = length * float(rng.uniform(0.7, 1.0)) if slug else min(
        length, radius * float(rng.uniform(1.3, 4.2))
    )
    pinch = float(rng.uniform(0.5, 1.1)) if slug else float(np.exp(rng.uniform(-0.5, 1.0)))
    body = np.clip(1.0 - (np.clip(above, 0.0, None) / reach) ** 2, 0.0, 1.0) ** pinch

    # Thread: a tapering envelope cut up by the beading instability, which
    # has had longest to grow at the far end.
    taper = float(rng.uniform(1.5, 4.0))
    wavelength = max(4.0, float(rng.uniform(*_BEADING_WAVELENGTH_RADII)) * thread * radius)
    phase_a, phase_b = rng.uniform(0.0, 2.0 * math.pi, 2)
    ripple = np.sin(2.0 * math.pi * above / wavelength + phase_a)
    ripple += 0.5 * np.sin(2.0 * math.pi * above / (0.41 * wavelength) + phase_b)
    growth = float(rng.uniform(0.2, 1.8)) * along**1.5
    strand = np.sqrt(np.clip((1.0 - along**taper) * (1.0 + growth * ripple / 1.5), 0.0, None))

    half_width = radius * ((root - thread) * body + thread * strand)
    half_width[above > length] = 0.0
    # Rooted at the head's centre and at full strength by its upper edge.
    emerge = np.clip(above / (0.9 * radius), 0.0, 1.0)
    gain = emerge * emerge * (3.0 - 2.0 * emerge)

    left = half_width * np.clip(1.0 + _edge_noise(rng, above, radius), 0.4, 1.6)
    right = half_width * np.clip(1.0 + _edge_noise(rng, above, radius), 0.4, 1.6)

    centre = np.full(rows.shape, head_x, dtype=np.float64)
    if path is not None:
        centre = np.interp(rows, path[::-1, 1], path[::-1, 0])
    bend = float(np.clip(rng.normal(0.0, 0.06), -0.14, 0.14)) * length
    sway = 0.12 * radius * math.sin(float(rng.uniform(0.0, 2.0 * math.pi)))
    turns = float(rng.uniform(1.0, 2.5))
    centre = centre + bend * along**2 + sway * np.sin(math.pi * along * turns)

    return _RunnerBody(top, centre, left, right, gain, cap)


def _drop_outline(
    rng: np.random.Generator, irregularity: float, deformation: float
) -> tuple[float, float, np.ndarray]:
    """One pinned drop's departure from a round cap: ``(egg, sag, harmonics)``.

    A drop's contact line stays wherever it was last pinned, so its outline
    is whatever its history left: the union of the drops that merged into
    it, pulled smooth by surface tension. That is a low-order shape — an
    ellipse at some angle, a rounded triangle, a lopsided lozenge — and a
    different one every time, which is what the random amplitudes and phases
    here give. Gravity adds two things that are not random: the outline is
    wider below than above (``egg``), and the water pools towards the lower
    edge, so the drop is thickest below its middle (``sag``).

    All of it scales with ``deformation``, the drop's size against the
    capillary length. Small drops come out round because they are.
    """
    orders = np.asarray(_OUTLINE_HARMONICS, dtype=np.float64)
    weights = rng.uniform(0.3, 1.0, orders.size) / orders**0.8
    weights /= weights.sum()
    amplitude = min(irregularity * deformation, 0.34)
    harmonics = np.stack(
        [amplitude * weights, rng.uniform(0.0, 2.0 * math.pi, orders.size)], axis=1
    )
    egg = min(float(rng.uniform(0.0, 0.1)) * deformation, 0.2)
    sag = min(float(rng.uniform(0.1, 0.35)) * deformation, 0.6)
    return egg, sag, harmonics


def _water_surface(
    lobes: np.ndarray,
    height: int,
    width: int,
    harmonics: np.ndarray | None = None,
    bodies: tuple[_RunnerBody, ...] | list[_RunnerBody] = (),
) -> tuple[np.ndarray, np.ndarray]:
    """Accumulate every drop on the pane into one water surface.

    ``lobes`` is ``(N, 7)``: centre x, centre y, radius, vertical elongation,
    cap height as a fraction of the radius, and the two gravity terms of
    :func:`_drop_outline`. ``harmonics`` is ``(N, K, 2)``, the amplitude and
    phase of each outline harmonic; omitted, the lobes are plain ellipses.
    ``bodies`` are the running drops, which are outlines rather than lobes.

    Returns ``(level, height_squared)``. ``level`` is the summed lobe field,
    whose contour at ``_OUTLINE_LEVEL`` is the water's outline; the height is
    returned squared because that is the smooth quantity — the height itself
    meets the glass with a vertical tangent, and differencing it directly
    would put the largest error exactly where the slope matters most.

    For one round lobe this is an ellipsoidal cap, ``h = c * sqrt(r^2 -
    d^2)``. For several it is the same expression with every contribution
    summed, which is what a merged drop needs: one outline, one continuous
    surface. A running drop keeps a cap as its cross-section all the way
    along, scaled to its local width.
    """
    level = np.zeros((height, width), dtype=np.float32)
    weighted = np.zeros((height, width), dtype=np.float32)
    falloff = 1.0 - _OUTLINE_LEVEL

    # As Python floats: unpacking numpy scalars row by row costs more than the
    # arithmetic on a lobe a few pixels across.
    for index, (cx, cy, radius, elongation, cap, egg, sag) in enumerate(lobes.tolist()):
        outline = None
        if harmonics is not None and radius >= _MIN_RADIUS_FOR_WOBBLE_PX:
            outline = harmonics[index]
        stretch = 1.0 + egg
        if outline is not None:
            stretch += float(outline[:, 0].sum())
        reach_x = radius * stretch * _LOBE_REACH
        reach_y = reach_x * elongation
        x0, x1 = max(0, int(cx - reach_x)), min(width, int(cx + reach_x) + 2)
        y0, y1 = max(0, int(cy - reach_y)), min(height, int(cy + reach_y) + 2)
        if x1 <= x0 or y1 <= y0:
            continue
        dx = (np.arange(x0, x1, dtype=np.float32) - np.float32(cx)).reshape(1, -1)
        dy = (np.arange(y0, y1, dtype=np.float32) - np.float32(cy)).reshape(-1, 1)
        dy = dy / np.float32(elongation)
        distance_sq = dx * dx + dy * dy
        thickness = np.float32((cap * radius) ** 2)
        if outline is not None:
            angle = np.arctan2(dy, dx)
            # Image rows run downward, so sin(angle) is positive below the
            # centre: the outline is wider there, and the water deeper.
            down = np.sin(angle)
            local = 1.0 + np.float32(egg) * down
            for order, (amplitude, phase) in zip(_OUTLINE_HARMONICS, outline, strict=True):
                local = local + np.float32(amplitude) * np.sin(order * angle + np.float32(phase))
            local = np.float32(radius) * local
            lobe = _lobe_profile(distance_sq / (local * local))
            pooled = 1.0 + np.float32(sag) * down * np.minimum(np.sqrt(distance_sq) / local, 1.0)
            level[y0:y1, x0:x1] += lobe
            weighted[y0:y1, x0:x1] += lobe * thickness * pooled * pooled
            continue
        lobe = _lobe_profile(distance_sq / np.float32(radius * radius))
        level[y0:y1, x0:x1] += lobe
        weighted[y0:y1, x0:x1] += lobe * thickness

    for body in bodies:
        first, last = max(0, body.top), min(height, body.top + len(body.centre))
        if last <= first:
            continue
        rows = slice(first - body.top, last - body.top)
        centre, left, right = body.centre[rows], body.left[rows], body.right[rows]
        widest = float(max(left.max(), right.max())) * _LOBE_REACH
        x0 = max(0, int(centre.min() - widest))
        x1 = min(width, int(centre.max() + widest) + 2)
        if x1 <= x0:
            continue
        dx = np.arange(x0, x1, dtype=np.float32).reshape(1, -1) - centre.astype(
            np.float32
        ).reshape(-1, 1)
        side = np.where(dx < 0.0, left.reshape(-1, 1), right.reshape(-1, 1)).astype(np.float32)
        # Narrower than a third of a pixel is where the thread has broken.
        lobe = _lobe_profile((dx / np.maximum(side, 0.3)) ** 2)
        lobe *= (side > 0.3) * body.gain[rows].astype(np.float32).reshape(-1, 1)
        mean_width = (0.5 * (left + right)).astype(np.float32).reshape(-1, 1)
        level[first:last, x0:x1] += lobe
        weighted[first:last, x0:x1] += lobe * (np.float32(body.cap) * mean_width) ** 2

    inside = np.maximum(level, _OUTLINE_LEVEL)
    height_sq = weighted * (1.0 - _OUTLINE_LEVEL / inside) / falloff
    return level, height_sq


class _DropOptics:
    """What the water does to a ray, for each pixel of the pane it covers.

    Depends only on the water's geometry and the lens, never on the picture,
    so it is worked out once per draw and reused for every frame of a clip.
    Water covers a small part of the pane, so every array here is indexed by
    ``wet`` — the flat indices of the covered pixels — rather than by frame.
    """

    __slots__ = ("alpha", "glint", "map_x", "map_y", "reflect", "wet")

    def __init__(
        self,
        wet: np.ndarray,
        alpha: np.ndarray,
        map_x: np.ndarray,
        map_y: np.ndarray,
        reflect: np.ndarray,
        glint: np.ndarray,
    ) -> None:
        self.wet = wet
        self.alpha = alpha
        self.map_x = map_x
        self.map_y = map_y
        self.reflect = reflect
        self.glint = glint


def _drop_optics(
    level: np.ndarray,
    height_sq: np.ndarray,
    focal_px: float,
    light_direction: tuple[float, float, float],
) -> _DropOptics | None:
    """Trace the camera's rays out through the water surface.

    A ray leaving the lens crosses flat glass, then water, and exits through
    the drop's curved face into air. Snell's law there gives its deflection;
    Fresnel's equations give how much of it gets out. Beyond the critical
    angle none does, which is the dark outline of every drop on a window.
    """
    wet = np.flatnonzero(level > 0.0)
    if wet.size == 0:
        return None
    cols = level.shape[1]

    def _at_wet(field: np.ndarray, dx: int, dy: int) -> np.ndarray:
        gradient = cv2.Sobel(field, cv2.CV_32F, dx, dy, ksize=3, scale=0.125)
        return np.asarray(gradient.ravel()[wet])

    # Coverage from the level field's own gradient: (level - cut) / |grad| is
    # the distance to the outline in pixels, so this is a one-pixel
    # anti-aliased edge whatever the drop's size.
    grad_x, grad_y = _at_wet(level, 1, 0), _at_wet(level, 0, 1)
    steepness = np.sqrt(grad_x * grad_x + grad_y * grad_y) + 1e-6
    alpha = np.clip((level.ravel()[wet] - _OUTLINE_LEVEL) / steepness + 0.5, 0.0, 1.0)

    # Slope of h from the gradient of h^2, which stays finite at the rim.
    scale = 0.5 / (np.sqrt(height_sq.ravel()[wet]) + 0.25)
    slope_x, slope_y = _at_wet(height_sq, 1, 0) * scale, _at_wet(height_sq, 0, 1) * scale
    slope = np.minimum(np.sqrt(slope_x * slope_x + slope_y * slope_y), 12.0)

    cos_in = 1.0 / np.sqrt(1.0 + slope * slope)
    sin_out = _WATER_IOR * slope * cos_in
    escapes = sin_out < 1.0
    cos_out = np.sqrt(np.clip(1.0 - sin_out * sin_out, 1e-6, 1.0))
    tan_out = sin_out / cos_out
    # tan(theta_out - theta_in): how far the ray is turned.
    tan_turn = np.where(escapes, (tan_out - slope) / (1.0 + slope * tan_out), 0.0)

    # Unpolarised Fresnel reflectance, water to air.
    r_s = (_WATER_IOR * cos_in - cos_out) / (_WATER_IOR * cos_in + cos_out)
    r_p = (_WATER_IOR * cos_out - cos_in) / (_WATER_IOR * cos_out + cos_in)
    reflect = np.where(escapes, 0.5 * (r_s * r_s + r_p * r_p), 1.0)

    # A convex surface turns the ray towards its own centre — up the slope —
    # which is what inverts the picture inside a drop.
    safe = np.maximum(slope, 1e-6)
    unit_x, unit_y = slope_x / safe, slope_y / safe
    offset = np.float32(focal_px) * tan_turn
    map_x = (wet % cols).astype(np.float32) + offset * unit_x
    map_y = (wet // cols).astype(np.float32) + offset * unit_y

    # Where the refracted ray points at the off-frame source.
    cos_turn = 1.0 / np.sqrt(1.0 + tan_turn * tan_turn)
    sin_turn = tan_turn * cos_turn
    light_x, light_y, light_z = light_direction
    aligned = sin_turn * (unit_x * light_x + unit_y * light_y) + cos_turn * light_z
    glint = np.exp((aligned - 1.0) * (2.0 / _GLINT_ANGULAR_RADIUS_RAD**2)) * (1.0 - reflect)

    def _column(values: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(values, dtype=np.float32).reshape(-1, 1)

    def _lookup(values: np.ndarray) -> np.ndarray:
        # cv2.remap takes its coordinates as an image and refuses one with
        # more than 32767 rows, so the list is folded into a block.
        padded = np.zeros(-(-wet.size // _REMAP_BLOCK_WIDTH) * _REMAP_BLOCK_WIDTH, np.float32)
        padded[: wet.size] = values
        return padded.reshape(-1, _REMAP_BLOCK_WIDTH)

    return _DropOptics(
        wet, _column(alpha), _lookup(map_x), _lookup(map_y), _column(reflect), _column(glint)
    )


def _disc_kernel(diameter_px: float) -> np.ndarray:
    radius = diameter_px / 2.0
    half = int(math.ceil(radius))
    axis = np.arange(-half, half + 1, dtype=np.float32)
    distance = np.sqrt(axis.reshape(-1, 1) ** 2 + axis.reshape(1, -1) ** 2)
    kernel = np.clip(radius + 0.5 - distance, 0.0, 1.0)
    return (kernel / kernel.sum()).astype(np.float32)


def _aperture_blur(image: np.ndarray, diameter_px: float) -> np.ndarray:
    """Convolve with the image of the aperture: a uniform disc.

    A disc and not a Gaussian, because that is what an out-of-focus point
    is. The difference is visible wherever something small and bright sits on
    the glass: it comes out as a flat-topped circle with an edge.
    """
    if diameter_px < 1.5:
        return image
    rows, cols = image.shape[:2]
    factor = 1
    while diameter_px / factor > _MAX_DISC_KERNEL_PX:
        factor *= 2
    if factor == 1:
        return cv2.filter2D(image, -1, _disc_kernel(diameter_px), borderType=cv2.BORDER_REFLECT_101)
    small = cv2.resize(
        image,
        (max(1, math.ceil(cols / factor)), max(1, math.ceil(rows / factor))),
        interpolation=cv2.INTER_AREA,
    )
    small = cv2.filter2D(
        small, -1, _disc_kernel(diameter_px / factor), borderType=cv2.BORDER_REFLECT_101
    )
    return cv2.resize(small, (cols, rows), interpolation=cv2.INTER_LINEAR)


def _soft_blur(image: np.ndarray, sigma_px: float) -> np.ndarray:
    """A Gaussian blur, computed at reduced size once it is wide enough.

    A blur of several pixels leaves nothing fine in its output, so it loses
    nothing by being computed small, and a full-size one was most of this
    transform's cost: 43 ms a call at 1280 px.
    """
    rows, cols = image.shape[:2]
    factor = max(1, int(sigma_px // 2))
    if factor == 1:
        return cv2.GaussianBlur(image, (0, 0), sigmaX=max(sigma_px, 0.5))
    small = cv2.resize(
        image,
        (max(2, cols // factor), max(2, rows // factor)),
        interpolation=cv2.INTER_AREA,
    )
    small = cv2.GaussianBlur(
        small, (0, 0), sigmaX=sigma_px / factor, borderType=cv2.BORDER_REFLECT_101
    )
    return cv2.resize(small, (cols, rows), interpolation=cv2.INTER_LINEAR)


def _condensation_field(
    height: int,
    width: int,
    rng: np.random.Generator,
    coverage: float,
    edge_softness: float,
    irregularity: float,
) -> np.ndarray:
    """Where the glass has fogged, and how heavily, in ``[0, 1]``.

    Condensation forms where the glass is below the dew point, and the
    temperature of a pane varies smoothly across it: it is warmest in front
    of the electronics, or on the side the sun or a heater reaches, and
    coldest at the far rim. So a fogged pane is not a scatter of patches. It
    is one film with one clear area, and the boundary between them is a
    front that moves across the glass as it warms or cools.

    The field is therefore a distance from the warmest point — drawn anywhere
    from the middle of the pane to well outside it, which covers a clear
    centre with a fogged rim, and a film advancing from one side — bent by a
    little low-frequency noise so the front is ragged rather than an arc, and
    cut so that ``coverage`` of the pane is affected. Occasionally the sign is
    reversed: a cold spot, fogged in the middle and clear around it.

    This returns *coverage*, never droplets. The droplets of a film are far
    below the aperture disc and cannot resolve as anything but scattering.

    The border is a nucleation gradient: droplet density thins until the
    glass reads clear, so it fades over a wide band. ``edge_softness`` is the
    width of that fade as a fraction of the field's range. (A wipe edge is
    the opposite case and is applied separately.)
    """
    # The field has no detail finer than its front, so it is built at a
    # quarter of the frame's size and enlarged.
    full_height, full_width = height, width
    reduce = max(1, min(height, width) // 128)
    height, width = max(2, height // reduce), max(2, width // reduce)

    half_diagonal = 0.5 * math.hypot(width, height)
    bearing = float(rng.uniform(0.0, 2.0 * math.pi))
    offset = float(rng.uniform(0.0, 1.5)) * half_diagonal
    warm_x = width / 2.0 + offset * math.cos(bearing)
    warm_y = height / 2.0 + offset * math.sin(bearing)

    ys = np.arange(height, dtype=np.float32).reshape(-1, 1) - np.float32(warm_y)
    xs = np.arange(width, dtype=np.float32).reshape(1, -1) - np.float32(warm_x)
    field = np.sqrt(xs * xs + ys * ys) / np.float32(2.0 * half_diagonal)
    ragged = value_noise(height, width, rng=rng, octaves=3, base_cells=2)
    field = field + np.float32(irregularity) * (ragged - 0.5)
    if rng.random() < 0.2:
        field = -field
    field -= field.min()
    field /= max(float(field.max()), 1e-6)

    # Pick the threshold from the field's own distribution so `coverage`
    # means what it says wherever the warm point landed.
    threshold = float(np.quantile(field, 1.0 - np.clip(coverage, 0.0, 1.0)))
    softness = max(edge_softness, 1e-3)
    ramp = np.clip((field - (threshold - 0.5 * softness)) / softness, 0.0, 1.0)
    film = (ramp * ramp * (3.0 - 2.0 * ramp)).astype(np.float32)
    if reduce == 1:
        return film
    return np.clip(
        cv2.resize(film, (full_width, full_height), interpolation=cv2.INTER_CUBIC), 0.0, 1.0
    )


class WaterDroplets(WeatherTransform):
    """Drops, runners and condensation on the glass of a far-focused camera.

    All size-like parameters are expressed as a **fraction of the image
    diagonal**, not in pixels: a droplet of a given physical size subtends a
    fixed fraction of the camera's field of view, so its pixel size scales
    with resolution.

    Args:
        droplet_count_range: number of droplets to *attempt* to place. Fewer
            may render if the canvas cannot fit them without overlap, or if a
            runner has swept them up.
        droplet_radius_range: droplet radius, as a fraction of the image
            diagonal (0.01 on a 1000px-diagonal image is a 10px radius).
        size_exponent_range: exponent ``tau`` of the ``N(r) ~ r^-tau``
            drop-size power law, drawn per call. Coalescence-driven growth
            settles near ``tau = 2``; larger values push the distribution
            toward fine spatter, smaller ones toward a few big beads.
        pane_width_range_mm: physical width of the piece of glass the frame
            covers, drawn per call. This gives droplet sizes a real scale,
            which two things depend on: whether a drop is heavy enough to run
            (its size in millimetres against the capillary length), and how
            defocused it is (its size against the aperture).
        aperture_diameter_range_mm: diameter of the lens's entrance pupil,
            drawn per call. **This is what sets how sharp the water looks.**
            Focused far away, the lens images every point on the glass as a
            disc this wide, so drops narrower than the aperture dissolve into
            a loss of contrast and only wider ones keep an outline. A 4 mm
            f/1.6 lens is 2.5 mm; the same lens with its iris stopped to f/8
            in sunlight is 0.5 mm. Fixed-iris cameras, and all cameras at
            night, sit at the wide end.
        horizontal_fov_range_deg: the camera's field of view, drawn per call.
            A drop bends rays through a fixed range of angles, so this decides
            how much of the frame each drop's inverted picture is taken from.
        contact_angle_hysteresis_range: ``cos(theta_receding) -
            cos(theta_advancing)`` for the glass, drawn per call. Higher means
            a stickier surface that pins larger drops in place; lower (a
            freshly coated dome) lets smaller drops run.
        cap_height_ratio_range: how far a drop bulges, as a fraction of its
            radius. Steeper drops bend rays harder and have a wider dark rim,
            because more of the surface is past the critical angle. Typical
            sessile water droplets: 0.3-0.6.
        coalescence_range: fraction of pinned drops caught merged with a
            neighbour. Two drops that touch become one, but on glass the
            contact line stays pinned where it was, so the merged drop keeps
            a lobed outline instead of relaxing to a circle.
        glint_range: brightness of the refracted image of a light source
            outside the frame, relative to the brightest light in it. Zero
            leaves the drops with only what the frame itself contains.
        illuminator_glow_range: how strongly the camera's own infrared ring
            lights the water, relative to full scale. Zero — the default —
            means the illuminator is off. On, the film scatters the ring's
            light back as a veil and every drop returns it from its rim,
            which is why a wet or fogged dome whites out at night. Set it
            only for frames that are in infrared mode; the shipped pipelines
            do.
        background_blur_range: defocus applied to the scene *outside* the
            droplets, as a fraction of the image diagonal. Above zero the
            camera is taken to be focused on the glass instead of the scene:
            the world goes soft and the water is drawn sharp, with no
            aperture disc. Leave at 0 for a surveillance camera.
        condensation_coverage_range: fraction of the pane that has fogged
            over.
        condensation_irregularity_range: how ragged the film's front is. Zero
            is a clean arc around the warmest point of the glass.
        condensation_edge_softness_range: width of the gradual fade at the
            film's front.
        condensation_milkiness_range: fraction of the light a full film
            scatters. What that does to the picture depends on the scene: it
            veils a bright frame and puts a glow round the lamps in a dark
            one.
        condensation_blur_range: width of the film's narrow forward
            scattering — how much it softens what is behind it — as a
            fraction of the image diagonal.
        trail_probability: chance that a drop past the shedding threshold has
            already run some way, and so left a track. Pinned drops never do.
        trail_length_range: track length, as a fraction of image height.
        streak_length_range: how long a running drop is for its width. Which
            drops run is *derived*, not sampled — see
            ``contact_angle_hysteresis_range``. What each one looks like
            within that length is drawn per drop: a blunt slug, a pear, or a
            head trailing a thread that swells and breaks into beads.
        vertical_stretch_range: how much gravity elongates a resting drop
            that is on the point of running. Smaller drops show less of it,
            in proportion to the square of their size, and small ones none.
        shape_irregularity_range: how far the outline of a drop one
            capillary length across (2.7 mm) departs from a circle, as a
            fraction of its radius. The departure scales with size, so fine
            beads stay round and large drops come out as the lopsided,
            bottom-heavy shapes their pinned contact lines leave them in.
        min_gap_ratio: minimum gap between two separate droplets, as a
            fraction of their combined radii.

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
        size_exponent_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.5, 4.0)),
        ]
        pane_width_range_mm: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(1.0, 2000.0)),
        ]
        aperture_diameter_range_mm: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 50.0)),
        ]
        horizontal_fov_range_deg: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(10.0, 150.0)),
        ]
        contact_angle_hysteresis_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.01, 2.0)),
        ]
        cap_height_ratio_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.05, 1.0)),
        ]
        coalescence_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 1.0)),
        ]
        glint_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 4.0)),
        ]
        illuminator_glow_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 4.0)),
        ]
        background_blur_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 0.05)),
        ]
        condensation_coverage_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 1.0)),
        ]
        condensation_irregularity_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 1.0)),
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
        droplet_count_range: tuple[int, int] = (180, 520),
        droplet_radius_range: tuple[float, float] = (0.0016, 0.030),
        size_exponent_range: tuple[float, float] = (1.7, 2.4),
        pane_width_range_mm: tuple[float, float] = (30.0, 190.0),
        aperture_diameter_range_mm: tuple[float, float] = (0.5, 3.0),
        horizontal_fov_range_deg: tuple[float, float] = (60.0, 100.0),
        contact_angle_hysteresis_range: tuple[float, float] = (0.15, 0.65),
        cap_height_ratio_range: tuple[float, float] = (0.35, 0.65),
        coalescence_range: tuple[float, float] = (0.1, 0.35),
        glint_range: tuple[float, float] = (0.0, 1.0),
        illuminator_glow_range: tuple[float, float] = (0.0, 0.0),
        background_blur_range: tuple[float, float] = (0.0, 0.0),
        condensation_coverage_range: tuple[float, float] = (0.15, 0.5),
        condensation_irregularity_range: tuple[float, float] = (0.1, 0.4),
        condensation_edge_softness_range: tuple[float, float] = (0.3, 0.6),
        condensation_milkiness_range: tuple[float, float] = (0.25, 0.6),
        condensation_blur_range: tuple[float, float] = (0.006, 0.02),
        trail_probability: float = 0.35,
        trail_length_range: tuple[float, float] = (0.08, 0.35),
        streak_length_range: tuple[float, float] = (1.5, 3.0),
        vertical_stretch_range: tuple[float, float] = (1.15, 1.5),
        shape_irregularity_range: tuple[float, float] = (0.03, 0.12),
        min_gap_ratio: float = 0.12,
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.droplet_count_range = droplet_count_range
        self.droplet_radius_range = droplet_radius_range
        self.size_exponent_range = size_exponent_range
        self.pane_width_range_mm = pane_width_range_mm
        self.aperture_diameter_range_mm = aperture_diameter_range_mm
        self.horizontal_fov_range_deg = horizontal_fov_range_deg
        self.contact_angle_hysteresis_range = contact_angle_hysteresis_range
        self.cap_height_ratio_range = cap_height_ratio_range
        self.coalescence_range = coalescence_range
        self.glint_range = glint_range
        self.illuminator_glow_range = illuminator_glow_range
        self.background_blur_range = background_blur_range
        self.condensation_coverage_range = condensation_coverage_range
        self.condensation_irregularity_range = condensation_irregularity_range
        self.condensation_edge_softness_range = condensation_edge_softness_range
        self.condensation_milkiness_range = condensation_milkiness_range
        self.condensation_blur_range = condensation_blur_range
        self.trail_probability = trail_probability
        self.trail_length_range = trail_length_range
        self.streak_length_range = streak_length_range
        self.vertical_stretch_range = vertical_stretch_range
        self.shape_irregularity_range = shape_irregularity_range
        self.min_gap_ratio = min_gap_ratio

    def get_params_dependent_on_data(
        self, params: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, Any]:
        height, width = params["shape"][:2]
        diagonal = float(np.hypot(width, height))
        rng = self.random_generator
        count = self.py_random.randint(*self.droplet_count_range)

        radius_px_range = (
            self.droplet_radius_range[0] * diagonal,
            self.droplet_radius_range[1] * diagonal,
        )
        # Both physical thresholds — the size at which a drop runs, and the
        # size below which the aperture dissolves it — are in millimetres, so
        # they are converted into pixels through how much glass the frame
        # spans. Drawn per call: the pane, its coating and the iris all vary
        # from one camera and one hour to the next.
        pane_width_mm = self.py_random.uniform(*self.pane_width_range_mm)
        hysteresis = self.py_random.uniform(*self.contact_angle_hysteresis_range)
        mm_per_px = pane_width_mm / max(width, 1)
        shedding_px = shedding_radius_mm(hysteresis) / mm_per_px
        aperture_mm = self.py_random.uniform(*self.aperture_diameter_range_mm)
        fov_deg = self.py_random.uniform(*self.horizontal_fov_range_deg)
        focal_px = (width / 2.0) / math.tan(math.radians(fov_deg) / 2.0)

        placements = _sample_positions(
            rng,
            count,
            width,
            height,
            radius_px_range,
            self.py_random.uniform(*self.size_exponent_range),
            shedding_px,
            self.streak_length_range,
            self.min_gap_ratio,
            footprint_inflation=self.vertical_stretch_range[1],
        )

        # Runners first: each one's track decides what else is still on the
        # glass. Where a drop has slid it has squeegeed the film away and
        # collected every bead in its path.
        wipe = np.zeros((height, width), dtype=np.float32)
        runners = []
        pinned = []
        for cx, cy, radius, stretch in placements:
            sag = self.py_random.uniform(*self.vertical_stretch_range)
            if stretch <= 1.0:
                # Gravity's pull on a drop against surface tension's hold goes
                # as the square of its size, so the stretch drawn is what a
                # drop on the point of running shows, and a small one shows
                # almost none of it.
                weight = min(1.0, (radius / max(shedding_px, 1e-6)) ** 2)
                pinned.append((cx, cy, radius, 1.0 + (sag - 1.0) * weight))
                continue
            elongation = stretch * sag
            # The placement reserved a circle around (cx, cy); the head sits
            # at the bottom of it and the tail fills the rest.
            head_y = cy + (elongation - 1.0) * radius
            tail_px = 2.0 * radius * (elongation - 1.0)
            path = None
            if self.py_random.random() < self.trail_probability:
                length = self.py_random.uniform(*self.trail_length_range) * height
                if length > tail_px + radius:
                    path = _runner_path(rng, cx, head_y, radius, length, tail_px)
                    _draw_track(wipe, path, radius)
            runners.append((cx, cy, radius, elongation, head_y, tail_px, path))

        def _swept(x: float, y: float) -> bool:
            column, row = int(round(x)), int(round(y))
            return 0 <= row < height and 0 <= column < width and wipe[row, column] > 0.5

        pinned = [drop for drop in pinned if not _swept(drop[0], drop[1])]

        # Everything on the pane, for the rule that separate drops never
        # touch. Kept as growing numpy buffers: the check runs once per bead
        # against everything already placed.
        capacity = max((len(pinned) + len(runners)) * 6, 64)
        occupied = np.empty((capacity, 3), dtype=np.float64)
        n_occupied = 0

        def _occupy(x: float, y: float, r: float) -> None:
            nonlocal occupied, n_occupied
            if n_occupied == len(occupied):
                occupied = np.concatenate([occupied, np.empty_like(occupied)])
            occupied[n_occupied] = (x, y, r)
            n_occupied += 1

        def _fits(x: float, y: float, r: float, gap: float, skip: int = -1) -> bool:
            placed = occupied[:n_occupied]
            reach = (placed[:, 2] + r) * (1.0 + gap)
            clash = (placed[:, 0] - x) ** 2 + (placed[:, 1] - y) ** 2 < reach * reach
            if skip >= 0:
                clash[skip] = False
            return not bool(clash.any())

        for cx, cy, radius, elongation in pinned:
            _occupy(cx, cy, radius * elongation)
        for cx, cy, radius, elongation, *_ in runners:
            _occupy(cx, cy, radius * elongation)

        droplets: list[tuple[float, float, float, float, float, float, tuple[float, float], bool]]
        droplets = []
        lobes: list[tuple[float, ...]] = []
        outlines: list[np.ndarray] = []
        bodies: list[_RunnerBody] = []
        round_outline = np.zeros((len(_OUTLINE_HARMONICS), 2), dtype=np.float64)
        capillary_px = capillary_length_mm() / mm_per_px

        def _shape() -> tuple[float, float, float, float]:
            return (
                self.py_random.uniform(*self.cap_height_ratio_range),
                self.py_random.uniform(*self.shape_irregularity_range),
                self.py_random.uniform(0, 2 * math.pi),
                self.py_random.uniform(0, 2 * math.pi),
            )

        def _outline(
            radius: float, irregularity: float, deformation: float
        ) -> tuple[float, float, np.ndarray]:
            # Most of the drops on a pane are a few pixels across, where an
            # outline has nothing to show; drawing one for each of them was a
            # fifth of this method's time.
            if radius < _MIN_RADIUS_FOR_WOBBLE_PX:
                return 0.0, 0.0, round_outline
            return _drop_outline(rng, irregularity, deformation)

        coalescence = self.py_random.uniform(*self.coalescence_range)
        merged = 0
        for index, (cx, cy, radius, elongation) in enumerate(pinned):
            cap, irregularity, phase_a, phase_b = _shape()
            droplets.append(
                (cx, cy, radius, cap, elongation, irregularity, (phase_a, phase_b), False)
            )
            deformation = float(
                np.clip(_DEFORMATION_AT_CAPILLARY_LENGTH * radius / capillary_px, 0.25, 2.4)
            )
            egg, sag, outline = _outline(radius, irregularity, deformation)
            lobes.append((cx, cy, radius, elongation, cap, egg, sag))
            outlines.append(outline)
            if self.py_random.random() >= coalescence:
                continue
            # Caught mid-merge: a second, smaller lobe close enough that the
            # two outlines join, as long as it does not run into a third drop.
            for _ in range(self.py_random.randint(1, 2)):
                partner = radius * self.py_random.uniform(0.45, 0.9)
                bearing = self.py_random.uniform(0, 2 * math.pi)
                reach = (radius + partner) * self.py_random.uniform(0.72, 1.02)
                px, py = cx + reach * math.cos(bearing), cy + reach * math.sin(bearing)
                if not _fits(px, py, partner * elongation, self.min_gap_ratio, skip=index):
                    continue
                _occupy(px, py, partner * elongation)
                egg, sag, outline = _outline(partner, irregularity, deformation)
                lobes.append((px, py, partner, elongation, cap, egg, sag))
                outlines.append(outline)
                merged += 1

        for cx, cy, radius, elongation, head_y, tail_px, path in runners:
            cap, irregularity, phase_a, phase_b = _shape()
            droplets.append(
                (cx, cy, radius, cap, elongation, irregularity, (phase_a, phase_b), True)
            )
            # The head is a drop like any other, with the outline its size
            # gives it; what it trails is joined on behind.
            deformation = float(
                np.clip(_DEFORMATION_AT_CAPILLARY_LENGTH * radius / capillary_px, 0.25, 2.4)
            )
            egg, sag, outline = _drop_outline(rng, irregularity, deformation)
            lobes.append((cx, head_y, radius, self.py_random.uniform(0.85, 1.2), cap, egg, sag))
            outlines.append(outline)
            bodies.append(_runner_body(rng, cx, head_y, radius, tail_px, cap, path))
            if path is None:
                continue
            for bx, by, br in _trail_beads(rng, path, radius, tail_px + 1.5 * radius):
                if not _fits(bx, by, br, self.min_gap_ratio):
                    continue
                _occupy(bx, by, br)
                # Shed behind the runner and pinned again: round beads.
                droplets.append((bx, by, br, cap, 1.0, 0.0, (0.0, 0.0), False))
                lobes.append((bx, by, br, 1.0, cap, 0.0, 0.0))
                outlines.append(round_outline)

        # The source a drop may hold a glint of: somewhere above the frame,
        # off the optical axis by less than a drop can bend a ray.
        off_axis = math.radians(self.py_random.uniform(12.0, 38.0))
        bearing = math.radians(self.py_random.uniform(-70.0, 70.0))
        light_direction = (
            math.sin(off_axis) * math.sin(bearing),
            -math.sin(off_axis) * math.cos(bearing),
            math.cos(off_axis),
        )

        optics = None
        if lobes or bodies:
            level, height_sq = _water_surface(
                np.asarray(lobes, dtype=np.float64).reshape(-1, 7),
                height,
                width,
                harmonics=np.asarray(outlines, dtype=np.float64).reshape(
                    -1, len(_OUTLINE_HARMONICS), 2
                ),
                bodies=bodies,
            )
            optics = _drop_optics(level, height_sq, focal_px, light_direction)
        coverage = 0.0 if optics is None else float(optics.alpha.sum()) / (height * width)

        condensation = _condensation_field(
            height,
            width,
            rng,
            coverage=self.py_random.uniform(*self.condensation_coverage_range),
            edge_softness=self.py_random.uniform(*self.condensation_edge_softness_range),
            irregularity=self.py_random.uniform(*self.condensation_irregularity_range),
        )
        # A drop sliding down does not *fade* the film out, it picks it up:
        # its contact line sweeps the glass mechanically, so on the glass the
        # track has a crisp boundary — the opposite of the film's own gradual
        # front. How crisp it looks to the camera is then up to the aperture.
        condensation *= 1.0 - np.clip(wipe, 0.0, 1.0)

        record_draw(
            pane_width_mm=pane_width_mm,
            contact_angle_hysteresis=hysteresis,
            mm_per_px=mm_per_px,
            shedding_radius_px=shedding_px,
            shedding_radius_mm=shedding_radius_mm(hysteresis),
            aperture_mm=aperture_mm,
            horizontal_fov_deg=fov_deg,
            count_requested=count,
            runners=len(runners),
            merged_lobes=merged,
            water_coverage=coverage,
            radius_px_min=radius_px_range[0],
            radius_px_max=radius_px_range[1],
        )
        return {
            "droplets": droplets,
            "optics": optics,
            "condensation": condensation,
            "aperture_px": aperture_mm / mm_per_px,
            "background_blur": self.py_random.uniform(*self.background_blur_range) * diagonal,
            "milkiness": self.py_random.uniform(*self.condensation_milkiness_range),
            "condensation_blur": self.py_random.uniform(*self.condensation_blur_range) * diagonal,
            "glint": self.py_random.uniform(*self.glint_range),
            "glow": self.py_random.uniform(*self.illuminator_glow_range),
        }

    def apply(
        self,
        img: np.ndarray,
        optics: _DropOptics | None,
        condensation: np.ndarray,
        aperture_px: float,
        background_blur: float,
        milkiness: float,
        condensation_blur: float,
        glint: float,
        glow: float,
        **params: Any,
    ) -> np.ndarray:
        has_film = condensation is not None and float(condensation.max()) > 1e-3
        if optics is None and not has_film:
            return img

        # Scattering and defocus both add up light, so all of it happens in
        # linear light; that is also what makes a lamp spread further than
        # the wall it is mounted on.
        scene, scale, is_integer = to_linear_unit(img)
        rows, cols = scene.shape[:2]
        diagonal = float(np.hypot(cols, rows))

        # Focused on the glass instead of the scene: the world behind goes
        # soft and the water is sharp.
        if background_blur > 0.1:
            out = _soft_blur(scene, background_blur)
            aperture_px = 0.0
        else:
            out = scene

        if has_film:
            # The film is on the glass, so the camera sees its pattern through
            # the same aperture disc as everything else there.
            film = _aperture_blur(condensation, aperture_px)
            scattered_fraction = (film * np.float32(milkiness))[..., None]

            # Both scattering lobes are blurs several pixels wide, so the
            # light they redistribute is gathered from a reduced copy.
            narrow_sigma = max(0.6, condensation_blur)
            reduce = max(1, int(narrow_sigma // 2))
            small = scene
            if reduce > 1:
                small = cv2.resize(
                    scene,
                    (max(2, cols // reduce), max(2, rows // reduce)),
                    interpolation=cv2.INTER_AREA,
                )
            brightest = np.maximum(np.maximum(small[..., 0], small[..., 1]), small[..., 2])
            clipped = np.clip((brightest - 0.9) / 0.1, 0.0, 1.0)[..., None]
            source = small * (1.0 + (_CLIPPED_SOURCE_GAIN - 1.0) * clipped * clipped)
            narrow = cv2.GaussianBlur(source, (0, 0), sigmaX=narrow_sigma / reduce)
            scattered = narrow * ((1.0 - _GLOW_LOBE[0] - _VEIL_LOBE[0]) * _FILM_ALBEDO)
            for share, sigma in (_GLOW_LOBE, _VEIL_LOBE):
                lobe = _soft_blur(source, sigma * diagonal / reduce)
                scattered += lobe * (share * _FILM_ALBEDO)
            if glow > 0.0:
                # The infrared ring surrounds the lens, so its light reaches
                # the rim of the pane more directly than the middle.
                ys = np.linspace(-1.0, 1.0, small.shape[0], dtype=np.float32).reshape(-1, 1)
                xs = np.linspace(-1.0, 1.0, small.shape[1], dtype=np.float32).reshape(1, -1)
                ring = 0.6 + 0.4 * np.clip(0.5 * (xs * xs + ys * ys), 0.0, 1.0)
                scattered += (glow * _IR_FILM_RETURN) * ring[..., None]
            if reduce > 1:
                scattered = cv2.resize(scattered, (cols, rows), interpolation=cv2.INTER_LINEAR)
            # out + f * (scattered - out), without the temporaries.
            scattered -= out
            scattered *= scattered_fraction
            out = out + scattered

        if optics is not None:
            # The picture inside a drop is squeezed several-fold, so it is
            # taken from a softened copy of the scene; sampling the sharp one
            # that sparsely would alias into speckle.
            soft = _soft_blur(scene, max(0.6, 0.004 * diagonal))
            sampled = cv2.remap(
                soft,
                optics.map_x,
                optics.map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )
            inside: np.ndarray = sampled.reshape(-1, 3)[: optics.wet.size]
            ambient = np.asarray(cv2.mean(scene)[:3], dtype=np.float32)
            inside *= 1.0 - optics.reflect
            inside += optics.reflect * (ambient * _HOUSING_ALBEDO + glow * _IR_DROP_RETURN)
            if glint > 0.0:
                # Read from a thinned copy: this is one colour for the frame.
                source_colour = sky_radiance_linear(scene[::4, ::4], 99.0)
                inside += optics.glint * (source_colour * (glint * _GLINT_SOURCE_GAIN))

            layers = np.zeros((rows * cols, 4), dtype=np.float32)
            layers[optics.wet, :3] = inside * optics.alpha
            layers[optics.wet, 3] = optics.alpha[:, 0]
            layers = _aperture_blur(layers.reshape(rows, cols, 4), aperture_px)
            out = out * (1.0 - layers[..., 3:]) + layers[..., :3]

        return from_linear_unit(out, scale, is_integer, img.dtype)
