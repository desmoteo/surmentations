"""A spider's web across the housing, lit by the camera's own illuminator.

Physical model
--------------
Infrared illuminators attract insects and spiders follow them, so an outdoor
camera that is not cleaned ends up with an orb web anchored to its housing,
its hub a hand's width in front of the lens. By day the feed shows nothing.
At night the web is the brightest thing in the frame: rows of soft glowing
discs strung along curves, across a scene that was already dark.

``HousingOccluders`` draws a web as blurred lines and leaves it at that. This
transform is the web itself, as geometry in front of a lens, because the three
things that make the night picture are all consequences of where each thread
is:

**The silk is far too thin to see, and far too close to be dim.** A thread is
a few micrometres across. Ten centimetres from a 4 mm lens it projects 0.04 px
wide, and it is so far inside the near focus limit that this is spread over a
blur disc some 15 px across: it covers a four-hundredth of each pixel it
touches. Lit by the sky it adds a quarter of a percent of full scale, which
is why a web is invisible by day. Lit by a lamp 10 cm away, against a scene
lit by the same lamp from 10 m, it receives ten thousand times the scene's
irradiance, and a four-hundredth of that is 25 times a white wall. So what is
drawn here is added light, never a shadow, and it is driven by the lamp.

**Each point of a thread is an image of the aperture.** Out of focus, a point
is drawn as the lens's stop: a disc of radius ``A f / 2z`` pixels for an
aperture ``A``, a focal length of ``f`` pixels and a distance ``z``. A thread
is a row of them. The light it returns falls as the lamp's does, ``1/z^2``,
so the near side of a tilted web is both brighter and blurred wider than the
far side. Only down to a point: the LEDs stand in a ring around the lens and
shine forwards, so the last centimetres in front of the glass are beside
their beams, and a thread across the dome itself is in the dark.

**The beads are what catch the light.** The capture spiral carries droplets
of glue, and by night every thread carries dew. Each is a point glint, so
each is one disc, and a thread reads as a string of pearls. Whether the
pearls touch is decided by one ratio: their spacing along the thread over
the aperture's diameter, at any distance, because the spacing and the disc
both project as ``1/z``. Beads a millimetre apart in front of a 2 mm stop
merge into a band; 4 mm apart they are separate discs, each brighter than
the band by that same ratio; and on a thread far enough to be nearly in
focus they are a dotted line.

The web is an orb web as it is built: a frame moored to the housing, radial
threads from a hub to the frame, and a capture spiral across them from the
edge of a free zone outwards, straight from radius to radius and a little
slack. Holes are torn in it, as they are in any web a day old. It stands in a
plane tilted to the lens, with the hub wherever it happens to be, usually
outside the frame. Loose threads may cross the view on their own, and the
spider may be at home: a body centimetres from the lamp, which no exposure
holds, so it is a white shape with a glow around it.

Composition: after ``NightIrMode``, like ``NightRain`` and ``NightSnow`` and
for their reason. The web's light is the camera's own, already in the
sensor's terms; converted with the scene it would be lit by the depth of
whatever is behind it and go dark.
"""

from __future__ import annotations

import math
from typing import Annotated, Any

import cv2
import numpy as np
from albumentations.core.pydantic import check_range_bounds, nondecreasing
from albumentations.core.transforms_interface import BaseTransformInitSchema
from pydantic import AfterValidator, Field

from surmentations.atmosphere.night_rain import _beam_profile
from surmentations.base import CameraTransform
from surmentations.diagnostics import record_draw
from surmentations.utils.illuminator import lamp_falloff, splat_discs, veiling_glare
from surmentations.utils.photometry import from_linear_unit, to_linear_unit
from surmentations.utils.precipitation import focal_length_px

# The dome's glass: nothing is nearer the lens than this.
_NEAREST_M = 0.012
# A disc is not drawn smaller than this: a pixel is the least a point covers.
_MIN_BLUR_PX = 0.5
# Threads are cut into straight pieces no longer than this before they are
# projected, so that a piece has one distance, and one blur, to within a few
# percent.
_PIECE_M = 0.008
# The continuous light of a thread is drawn as discs this fraction of their
# own radius apart, close enough that they read as a band.
_BAND_STEP = 0.5
# A diffuse body returns light from all of its surface, a thread from a
# four-hundredth of the pixels its blur covers: at the same distance a
# spider is this many times as bright as a thread's band.
_BODY_TO_THREAD = 40.0
# What each kind of thread returns: (its light, relative to a radius; how
# beaded it is, relative to the capture spiral). The spiral is thin and
# carries the glue; radii and frame are thicker dry silk, doubled in the
# frame, and hold only dew; the hub is a fine dry mesh.
_KINDS = {
    "spiral": (0.8, 1.0),
    "radius": (1.0, 0.6),
    "frame": (1.6, 0.6),
    "hub": (0.6, 0.25),
    "stray": (1.0, 1.0),
}

Threads = dict[str, np.ndarray]


def _about_one(rng: np.random.Generator, spread: float, count: int) -> np.ndarray:
    """Log-normal factors with a mean of one: unequal shares that add nothing overall."""
    factors: np.ndarray = np.exp(rng.normal(-0.5 * spread * spread, spread, count))
    return factors


def _threads(
    start: np.ndarray, end: np.ndarray, sag: np.ndarray, kind: str, rng: np.random.Generator
) -> Threads:
    """Threads of one kind from ``start`` to ``end``, ``(n, d)`` each, bowed by ``sag``.

    A thread is the curve ``start + t (end - start) + 4 t (1 - t) sag``. Each
    gets its own brightness, log-normal with the kind's as its mean: silk
    scatters according to how it lies to the light, and no two threads lie
    alike.
    """
    count = len(start)
    band, beads = _KINDS[kind]
    lie = _about_one(rng, 0.4, count)
    return {
        "start": np.asarray(start, dtype=np.float64).reshape(count, -1),
        "end": np.asarray(end, dtype=np.float64).reshape(count, -1),
        "sag": np.asarray(sag, dtype=np.float64).reshape(count, -1),
        "band": band * lie,
        "beads": np.full(count, beads),
    }


def _join(parts: list[Threads]) -> Threads:
    """Several sets of threads as one."""
    return {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}


def _orb_web(
    rng: np.random.Generator, radius: float, radials: int, mesh: float, damage: float
) -> Threads:
    """An orb web in its own plane, the hub at the origin, in metres.

    Built in the spider's order. A frame of a few straight threads between
    mooring points; radii from the hub to the frame; a fine hub; and the
    capture spiral, which starts outside a free zone and winds outwards,
    ``mesh`` apart, straight from each radius to the next and a little slack.
    ``damage`` (0 to 1) tears holes in the spiral and cuts some of the radii
    that cross them.
    """
    angles = np.sort(
        2.0 * np.pi * (np.arange(radials) + rng.uniform(-0.3, 0.3, radials)) / radials
    )
    # Webs are not round: the side below the hub is the longer one.
    long_side = rng.uniform(0.0, 2.0 * np.pi)
    stretch = 1.0 + rng.uniform(0.0, 0.35) * np.cos(angles - long_side)
    directions = np.stack([np.cos(angles), np.sin(angles)], axis=1)

    # The frame: straight threads between every few radii's ends.
    moorings = np.sort(
        rng.choice(radials, size=int(rng.integers(4, 8)), replace=False)
    )
    corner = directions[moorings] * (radius * stretch[moorings] * 1.08)[:, None]
    after = np.roll(corner, -1, axis=0)
    reach = np.empty(radials)
    for index in range(radials):
        # Which side of the frame this radius meets, and how far out.
        side = int(np.searchsorted(moorings, index, side="right") - 1) % len(moorings)
        a, b = corner[side], after[side]
        edge = b - a
        ray = directions[index]
        cross = ray[0] * edge[1] - ray[1] * edge[0]
        along = (a[0] * edge[1] - a[1] * edge[0]) / cross if abs(cross) > 1e-9 else radius
        reach[index] = along if 0.0 < along < 2.0 * radius else radius * stretch[index]
    reach[moorings] = np.linalg.norm(corner, axis=1)

    gravity_angle = rng.uniform(0.0, 2.0 * np.pi)
    gravity = np.array([np.cos(gravity_angle), np.sin(gravity_angle)])

    # Holes: discs torn out of the web, and a share of the spiral lost anyway.
    hole_count = int(rng.poisson(3.0 * damage))
    hole_at = rng.uniform(-0.8, 0.8, (hole_count, 2)) * radius
    hole_radius = rng.uniform(0.12, 0.4, hole_count) * radius * math.sqrt(max(damage, 1e-6))

    def torn(points: np.ndarray) -> np.ndarray:
        """Whether each point lies in a hole."""
        if hole_count == 0:
            return np.zeros(len(points), dtype=bool)
        apart = np.linalg.norm(points[:, None, :] - hole_at[None, :, :], axis=2)
        gone: np.ndarray = (apart < hole_radius[None, :]).any(axis=1)
        return gone

    parts = [
        _threads(corner, after, np.zeros_like(corner), "frame", rng),
        # Mooring lines, from the frame's corners out to the housing.
        _threads(
            corner,
            corner * rng.uniform(1.6, 3.5, (len(corner), 1))
            + rng.normal(0.0, 0.3 * radius, corner.shape),
            np.zeros_like(corner),
            "frame",
            rng,
        ),
    ]

    # Radii. One that crosses a hole may be cut there.
    ends = directions * reach[:, None]
    cut = torn(ends * 0.6) & (rng.uniform(size=radials) < 0.5)
    hub = np.zeros_like(ends)
    parts.append(_threads(hub[~cut], ends[~cut], hub[~cut], "radius", rng))
    if cut.any():
        stub = ends[cut] * rng.uniform(0.15, 0.45, (int(cut.sum()), 1))
        parts.append(_threads(np.zeros_like(stub), stub, np.zeros_like(stub), "radius", rng))

    def loops(first: float, last: float, pitch: float, kind: str, tear: float) -> None:
        """A spiral from ``first`` to ``last`` of the web's radius, ``pitch`` apart."""
        turns = max(1, int((last - first) * radius / pitch))
        # Where turn k meets radius i: a true spiral, a pitch further out
        # each time round, and each attachment a little off.
        step = np.arange(turns)[:, None] + np.arange(radials)[None, :] / radials
        out = first * radius + step * pitch * stretch[None, :]
        out = out + rng.normal(0.0, 0.16 * pitch, out.shape)
        attached = out < 0.95 * reach[None, :]
        nodes = directions[None, :, :] * out[:, :, None]
        # Each node joins the same turn's node on the next radius; the last
        # radius joins the next turn's node on the first.
        following = np.roll(nodes, -1, axis=1)
        following[:-1, -1] = nodes[1:, 0]
        joined = attached & np.roll(attached, -1, axis=1)
        joined[-1, -1] = False
        joined[:-1, -1] = attached[:-1, -1] & attached[1:, 0]
        start, end = nodes[joined], following[joined]
        if tear > 0.0:
            keep = ~torn(0.5 * (start + end)) & (rng.uniform(size=len(start)) > 0.2 * tear)
            start, end = start[keep], end[keep]
        chord = np.linalg.norm(end - start, axis=1, keepdims=True)
        slack = chord * rng.uniform(0.0, 0.08, (len(start), 1)) * gravity[None, :]
        parts.append(_threads(start, end, slack, kind, rng))

    free_zone = rng.uniform(0.14, 0.24)
    loops(0.015, 0.07, 0.4 * mesh, "hub", 0.0)
    loops(free_zone, 1.3, mesh, "spiral", damage)
    return _join(parts)


def _pieces(threads: Threads) -> Threads:
    """Every thread cut into straight pieces of at most ``_PIECE_M``."""
    start, end, sag = threads["start"], threads["end"], threads["sag"]
    length = np.linalg.norm(end - start, axis=1)
    count = np.maximum(1, np.ceil(length / _PIECE_M)).astype(np.int64)
    owner = np.repeat(np.arange(len(start)), count)
    first = np.cumsum(count) - count
    within = np.arange(int(count.sum())) - first[owner]
    t0 = (within / count[owner])[:, None]
    t1 = ((within + 1) / count[owner])[:, None]

    def at(t: np.ndarray) -> np.ndarray:
        """The point of each piece's thread at ``t``."""
        point: np.ndarray = (
            start[owner] + t * (end[owner] - start[owner]) + 4.0 * t * (1.0 - t) * sag[owner]
        )
        return point

    return {
        "start": at(t0),
        "end": at(t1),
        "band": threads["band"][owner],
        "beads": threads["beads"][owner],
    }


def _glints(
    rng: np.random.Generator,
    pieces: Threads,
    *,
    width: int,
    height: int,
    focal: float,
    aperture_m: float,
    focus_m: float,
    gain: float,
    bead_spacing_m: float,
    bead_share: float,
    budget: int,
) -> dict[str, np.ndarray]:
    """The discs a set of straight pieces of thread is drawn as.

    Each piece, in camera coordinates (metres, z along the axis), gives
    discs for its continuous light, close enough together to read as a band,
    and one for each of its beads. Returns their centres ``x``, ``y`` and
    ``radius`` in pixels and the light each holds, ``weight``, in units of
    full scale times pixels: a disc's brightness is its weight over its area.

    The light is set so that a bare thread's band, at the reference distance
    and across the line of sight, would peak at ``gain`` times its own
    ``band``. A line of light of density ``q`` per pixel, blurred by a disc
    of radius ``r``, peaks at ``2 q / (pi r)``, which gives the density; the
    lamp's falloff and the aperture's solid angle do the rest, so a thread
    seen along its length, which puts more silk in each pixel, is brighter.
    ``bead_share`` of that light is then gathered into the beads, so beading
    changes a thread's texture and not how much light it returns.
    """
    start, end = pieces["start"], pieces["end"]
    middle = 0.5 * (start + end)
    z = middle[:, 2]
    length_m = np.linalg.norm(end - start, axis=1)
    ahead = (z > _NEAREST_M) & (start[:, 2] > _NEAREST_M) & (end[:, 2] > _NEAREST_M)
    z = np.where(ahead, z, 1.0)

    half_stop = 0.5 * aperture_m * focal
    radius = np.maximum(_MIN_BLUR_PX, half_stop * np.abs(1.0 / z - 1.0 / focus_m))
    centre_x, centre_y = (width - 1) / 2.0, (height - 1) / 2.0

    def project(point: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Pixel coordinates of camera-space points."""
        depth = np.maximum(point[:, 2], _NEAREST_M)
        return centre_x + focal * point[:, 0] / depth, centre_y + focal * point[:, 1] / depth

    x0, y0 = project(start)
    x1, y1 = project(end)
    # Only what can reach the frame.
    reach = radius + 1.0
    seen = (
        ahead
        & (np.maximum(x0, x1) > -reach)
        & (np.minimum(x0, x1) < width + reach)
        & (np.maximum(y0, y1) > -reach)
        & (np.minimum(y0, y1) < height + reach)
    )
    x0, y0, x1, y1 = x0[seen], y0[seen], x1[seen], y1[seen]
    z, radius, length_m = z[seen], radius[seen], length_m[seen]
    band, beads = pieces["band"][seen], pieces["beads"][seen]
    length_px = np.hypot(x1 - x0, y1 - y0)

    # The light a metre of thread puts in the image, in full scale times
    # pixels: brightness at the peak of its band, times the band's width,
    # times the pixels a metre of it spans.
    per_metre = (
        gain * band * lamp_falloff(z) * (np.pi / 2.0) * (half_stop / z) * (focal / z)
    )

    def along(count: np.ndarray, jitter: float) -> tuple[np.ndarray, np.ndarray]:
        """Which piece each of ``count`` discs a piece belongs to, and where along it."""
        owner = np.repeat(np.arange(len(count)), count)
        first = np.cumsum(count) - count
        within = np.arange(int(count.sum())) - first[owner]
        offset = 0.5 + (rng.uniform(-jitter, jitter, len(owner)) if jitter else 0.0)
        return owner, (within + offset) / count[owner]

    step = _BAND_STEP
    band_count = np.maximum(1, np.ceil(length_px / np.maximum(0.75, step * radius)))
    bead_count = np.floor(length_m / bead_spacing_m + rng.uniform(size=len(z))).astype(np.int64)
    # A kind of thread that is less beaded keeps more of its light in the band.
    share = np.clip(bead_share * beads, 0.0, 1.0)
    bead_count[share <= 0.0] = 0
    share = np.where(bead_count > 0, share, 0.0)
    spare = budget - int(bead_count.sum())
    if band_count.sum() > max(spare, budget // 4):
        # Over budget: fewer, wider-spaced discs for the band, holding the
        # same light.
        band_count = np.maximum(
            1, np.floor(band_count * max(spare, budget // 4) / band_count.sum())
        )
    band_count = band_count.astype(np.int64)

    owner, t = along(band_count, 0.0)
    band_weight = ((1.0 - share) * per_metre * length_m / band_count)[owner]
    bead_owner, bead_t = along(bead_count, 0.5)
    # A piece's beads share its beads' light, more or less evenly: droplets
    # are not all one size.
    bead_weight = (share * per_metre * length_m / np.maximum(bead_count, 1))[
        bead_owner
    ] * _about_one(rng, 0.5, len(bead_owner))

    owner = np.concatenate([owner, bead_owner])
    t = np.concatenate([t, bead_t])
    return {
        "x": (x0[owner] + t * (x1[owner] - x0[owner])).astype(np.float32),
        "y": (y0[owner] + t * (y1[owner] - y0[owner])).astype(np.float32),
        "radius": radius[owner].astype(np.float32),
        "weight": np.concatenate([band_weight, bead_weight]).astype(np.float32),
    }


def _disc_kernel(radius_px: float) -> np.ndarray:
    """A disc of the given radius with an edge a pixel wide, summing to one."""
    half = int(math.ceil(radius_px + 0.5))
    axis = np.arange(-half, half + 1, dtype=np.float32)
    distance = np.hypot(axis[:, None], axis[None, :])
    kernel = np.clip(radius_px + 0.5 - distance, 0.0, 1.0)
    normalised: np.ndarray = kernel / kernel.sum()
    return normalised


def _spider_patch(
    rng: np.random.Generator, body_px: float, heading: float, blur_px: float
) -> np.ndarray:
    """How much of each pixel a spider covers, seen out of focus: a square patch.

    Two ellipses, the abdomen behind and the smaller front part ahead along
    ``heading``, and eight legs from the front part, each bent at a knee: two
    pairs reaching forwards and two back, as a spider sits at its hub.
    ``body_px`` is the length of the body; the legs reach a little further
    than that again. The shape is then blurred by the aperture's disc,
    ``blur_px`` in radius, like everything else this close.
    """
    reach = 2.2 * body_px
    half = int(math.ceil(reach + blur_px + 2.0))
    size = 2 * half + 1
    # Drawn oversize and reduced: OpenCV's shapes snap to whole pixels, and a
    # spider far enough to be small is a few pixels long.
    over = 4 if body_px < 24.0 else 2
    canvas: np.ndarray = np.zeros((size * over, size * over), dtype=np.float32)
    centre = half * over + over // 2
    scale = body_px * over
    forward = np.array([math.cos(heading), math.sin(heading)])
    side = np.array([-forward[1], forward[0]])

    def pixel(point: np.ndarray) -> tuple[int, int]:
        """A point in body lengths from the centre, as a pixel of the canvas."""
        return int(round(centre + scale * point[0])), int(round(centre + scale * point[1]))

    def ellipse(at: np.ndarray, long: float, short: float) -> None:
        """A filled ellipse along the body's axis, its half-axes in body lengths."""
        axes = (max(1, int(round(long * scale))), max(1, int(round(short * scale))))
        cv2.ellipse(canvas, pixel(at), axes, math.degrees(heading), 0, 360, 1.0, -1, cv2.LINE_AA)

    ellipse(-0.22 * forward, 0.36, 0.29)
    ellipse(0.26 * forward, 0.22, 0.19)
    for sign in (-1.0, 1.0):
        for fan in (28.0, 58.0, 118.0, 148.0):
            angle = math.radians(fan + rng.uniform(-9.0, 9.0))
            out = math.cos(angle) * forward + sign * math.sin(angle) * side
            span = rng.uniform(1.1, 1.6)
            hip = 0.26 * forward + 0.1 * sign * side
            foot = hip + span * out
            # The knee stands off the line from hip to foot: ahead of it on
            # the front legs, behind it on the hind ones.
            lift = (1.0 if fan < 90.0 else -1.0) * rng.uniform(0.18, 0.34) * span
            knee = hip + 0.45 * span * out + lift * forward
            cv2.line(
                canvas, pixel(hip), pixel(knee), 1.0, max(1, int(round(0.11 * scale))), cv2.LINE_AA
            )
            cv2.line(
                canvas, pixel(knee), pixel(foot), 1.0, max(1, int(round(0.07 * scale))), cv2.LINE_AA
            )
    if over > 1:
        canvas = cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)
    if blur_px > _MIN_BLUR_PX:
        canvas = cv2.filter2D(
            canvas, -1, _disc_kernel(blur_px), borderType=cv2.BORDER_CONSTANT
        )
    return np.clip(canvas, 0.0, 1.0)


class SpiderWeb(CameraTransform):
    """An orb web in front of the lens, lit by the camera's infrared illuminator.

    Needs no depth provider: the web is centimetres from the lens, in front
    of everything, and lit by the lamp. Runs on CPU. Apply it after
    ``NightIrMode``; see the module docstring.

    Args:
        distance_range_m: how far the web's hub is from the lens.
        web_radius_range_m: the web's radius, hub to frame.
        tilt_range_deg: angle between the web's plane and the image plane.
            At zero the web faces the lens and is evenly blurred; tilted, one
            side is near, bright and blurred wide, the other far, dim and
            fine.
        radial_count_range: how many radii the web has.
        mesh_spacing_range_mm: distance between turns of the capture spiral.
        bead_spacing_range_mm: distance between the droplets along a thread.
            Against the aperture's diameter this decides the look: closer
            than the aperture is wide and the discs merge into a band;
            further apart and they are a string of pearls.
        bead_share_range: how much of a thread's light comes from its beads,
            0 to 1. Zero is dry silk, a smooth band; near one, the thread
            between the beads is hardly there.
        damage_range: how much of the capture spiral is torn away, 0 to 1.
        stray_thread_count_range: loose threads crossing the view on their
            own, besides the web.
        spider_probability: chance the spider is at home, at the hub. It is
            in the picture when the hub is.
        spider_size_range_mm: its body length.
        illuminator_gain_range: brightness of a bare thread's band 10 cm
            from the lens, on the beam's axis, in units of full scale. The
            near side of a web is several times this and separate beads are
            brighter again, so well below 1 the near threads already clip,
            which a web against the housing does.
        beam_falloff_range: how sharply the beam dims off-axis.
        horizontal_fov_range_deg: field of view, which projects metres to
            pixels.
        aperture_diameter_range_mm: the lens's stop. The blur disc of a
            thread is its image.
        focus_distance_range_m: where the lens is focused.
        glare_range: fraction of the web's light the dome and the lens
            scatter into a soft glow around it.
        max_discs: render budget. When it binds, a thread's continuous light
            is drawn with fewer discs, wider apart, holding the same light;
            the beads are never thinned.

    Targets:
        image

    Image types:
        uint8, float32
    """

    class InitSchema(BaseTransformInitSchema):
        """The arguments of ``SpiderWeb`` as Albumentations validates them: each
        range ordered low to high and within the bounds given here.
        """

        distance_range_m: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.02, 2.0)),
        ]
        web_radius_range_m: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.02, 0.6)),
        ]
        tilt_range_deg: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 80.0)),
        ]
        radial_count_range: Annotated[
            tuple[int, int],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(8, 60)),
        ]
        mesh_spacing_range_mm: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(1.0, 40.0)),
        ]
        bead_spacing_range_mm: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.3, 30.0)),
        ]
        bead_share_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 1.0)),
        ]
        damage_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 1.0)),
        ]
        stray_thread_count_range: Annotated[
            tuple[int, int],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0, 20)),
        ]
        spider_probability: float = Field(ge=0.0, le=1.0)
        spider_size_range_mm: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(1.0, 40.0)),
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
        glare_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 1.0)),
        ]
        max_discs: int = Field(ge=100, le=1_000_000)

    def __init__(
        self,
        distance_range_m: tuple[float, float] = (0.04, 0.25),
        web_radius_range_m: tuple[float, float] = (0.06, 0.2),
        tilt_range_deg: tuple[float, float] = (0.0, 65.0),
        radial_count_range: tuple[int, int] = (14, 30),
        mesh_spacing_range_mm: tuple[float, float] = (3.0, 9.0),
        bead_spacing_range_mm: tuple[float, float] = (1.0, 4.0),
        bead_share_range: tuple[float, float] = (0.5, 0.95),
        damage_range: tuple[float, float] = (0.0, 0.6),
        stray_thread_count_range: tuple[int, int] = (0, 3),
        spider_probability: float = 0.25,
        spider_size_range_mm: tuple[float, float] = (4.0, 14.0),
        illuminator_gain_range: tuple[float, float] = (0.15, 1.0),
        beam_falloff_range: tuple[float, float] = (0.6, 2.2),
        horizontal_fov_range_deg: tuple[float, float] = (48.0, 78.0),
        aperture_diameter_range_mm: tuple[float, float] = (0.9, 2.6),
        focus_distance_range_m: tuple[float, float] = (8.0, 40.0),
        glare_range: tuple[float, float] = (0.05, 0.25),
        max_discs: int = 60_000,
        p: float = 0.5,
    ):
        """Store the ranges; each argument is described in the class docstring."""
        super().__init__(p=p)
        self.distance_range_m = distance_range_m
        self.web_radius_range_m = web_radius_range_m
        self.tilt_range_deg = tilt_range_deg
        self.radial_count_range = radial_count_range
        self.mesh_spacing_range_mm = mesh_spacing_range_mm
        self.bead_spacing_range_mm = bead_spacing_range_mm
        self.bead_share_range = bead_share_range
        self.damage_range = damage_range
        self.stray_thread_count_range = stray_thread_count_range
        self.spider_probability = spider_probability
        self.spider_size_range_mm = spider_size_range_mm
        self.illuminator_gain_range = illuminator_gain_range
        self.beam_falloff_range = beam_falloff_range
        self.horizontal_fov_range_deg = horizontal_fov_range_deg
        self.aperture_diameter_range_mm = aperture_diameter_range_mm
        self.focus_distance_range_m = focus_distance_range_m
        self.glare_range = glare_range
        self.max_discs = max_discs

    def get_params_dependent_on_data(
        self, params: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, Any]:
        """Build a web in front of the lens and reduce it to the discs it is drawn as.

        Returns the ``discs`` (centres, radii and the light each holds), the
        ``spider`` if one is home (its patch of coverage, where it goes and
        how bright it is), the ``glare`` and the ``beam`` across the frame.
        """
        del data
        height, width = params["shape"][:2]
        rng = self.random_generator

        distance = self.py_random.uniform(*self.distance_range_m)
        radius = self.py_random.uniform(*self.web_radius_range_m)
        tilt = math.radians(self.py_random.uniform(*self.tilt_range_deg))
        radials = self.py_random.randint(*self.radial_count_range)
        mesh = self.py_random.uniform(*self.mesh_spacing_range_mm) * 1e-3
        bead_spacing = self.py_random.uniform(*self.bead_spacing_range_mm) * 1e-3
        bead_share = self.py_random.uniform(*self.bead_share_range)
        damage = self.py_random.uniform(*self.damage_range)
        strays = self.py_random.randint(*self.stray_thread_count_range)
        has_spider = self.py_random.random() < self.spider_probability
        spider_mm = self.py_random.uniform(*self.spider_size_range_mm)
        gain = self.py_random.uniform(*self.illuminator_gain_range)
        falloff = self.py_random.uniform(*self.beam_falloff_range)
        horizontal_fov = self.py_random.uniform(*self.horizontal_fov_range_deg)
        aperture_m = self.py_random.uniform(*self.aperture_diameter_range_mm) * 1e-3
        focus = self.py_random.uniform(*self.focus_distance_range_m)
        glare = self.py_random.uniform(*self.glare_range)

        focal = focal_length_px(width, height, horizontal_fov)
        centre_x, centre_y = (width - 1) / 2.0, (height - 1) / 2.0

        def towards(fraction_x: float, fraction_y: float, depth: float) -> np.ndarray:
            """The point at ``depth`` seen at a position given in fractions of the frame."""
            return np.array(
                [
                    (fraction_x * (width - 1) - centre_x) / focal * depth,
                    (fraction_y * (height - 1) - centre_y) / focal * depth,
                    depth,
                ]
            )

        # The hub is wherever the spider found anchors: usually out of the
        # picture, but never so far out that none of the web is in it.
        seen_radius = radius * math.cos(tilt) * focal / distance
        margin = min(0.3, 0.3 * seen_radius / max(width, height))
        hub = towards(
            float(rng.uniform(-margin, 1.0 + margin)),
            float(rng.uniform(-margin, 1.0 + margin)),
            distance,
        )
        # The web's plane: the image plane, tipped by `tilt` about an axis
        # in a random direction.
        turn = rng.uniform(0.0, 2.0 * np.pi)
        axis_u = np.array([math.cos(turn), math.sin(turn), 0.0])
        axis_v = np.array(
            [-math.sin(turn) * math.cos(tilt), math.cos(turn) * math.cos(tilt), math.sin(tilt)]
        )
        normal = np.cross(axis_u, axis_v)
        # A web is never quite flat: the wind bellies it.
        belly = rng.uniform(-0.12, 0.12) * radius

        def in_space(flat: np.ndarray) -> np.ndarray:
            """Points of the web's plane, as camera coordinates."""
            out_of_plane = belly * (1.0 - (flat**2).sum(axis=1) / radius**2)
            placed: np.ndarray = (
                hub[None, :]
                + flat[:, :1] * axis_u[None, :]
                + flat[:, 1:2] * axis_v[None, :]
                + out_of_plane[:, None] * normal[None, :]
            )
            return placed

        web = _pieces(_orb_web(rng, radius, radials, mesh, damage))
        parts = [
            {
                "start": in_space(web["start"]),
                "end": in_space(web["end"]),
                "band": web["band"],
                "beads": web["beads"],
            }
        ]
        if strays:
            # A loose thread: between two points near the housing, anywhere
            # around the view, hanging a little.
            near, far = _NEAREST_M * 2.0, max(distance * 1.5, 0.1)
            ends = np.array(
                [
                    [
                        towards(
                            float(rng.uniform(-0.4, 1.4)),
                            float(rng.uniform(-0.4, 1.4)),
                            float(rng.uniform(near, far)),
                        )
                        for _ in range(2)
                    ]
                    for _ in range(strays)
                ]
            )
            span = np.linalg.norm(ends[:, 1] - ends[:, 0], axis=1)
            hang = np.zeros((strays, 3))
            hang[:, 1] = span * rng.uniform(0.0, 0.06, strays)
            parts.append(_pieces(_threads(ends[:, 0], ends[:, 1], hang, "stray", rng)))

        discs = _glints(
            rng,
            _join(parts),
            width=width,
            height=height,
            focal=focal,
            aperture_m=aperture_m,
            focus_m=focus,
            gain=gain,
            bead_spacing_m=bead_spacing,
            bead_share=bead_share,
            budget=self.max_discs,
        )

        spider = None
        if has_spider and hub[2] > 2.0 * _NEAREST_M:
            body_px = spider_mm * 1e-3 * focal / hub[2]
            blur_px = 0.5 * aperture_m * focal * abs(1.0 / hub[2] - 1.0 / focus)
            at_x = centre_x + focal * hub[0] / hub[2]
            at_y = centre_y + focal * hub[1] / hub[2]
            reach = 2.2 * body_px + blur_px + 2.0
            # Drawn only if any of it can be seen.
            if -reach < at_x < width + reach and -reach < at_y < height + reach:
                patch = _spider_patch(
                    rng, body_px, float(rng.uniform(0.0, 2.0 * np.pi)), blur_px
                )
                spider = {
                    "patch": patch,
                    "x": int(round(at_x)) - patch.shape[1] // 2,
                    "y": int(round(at_y)) - patch.shape[0] // 2,
                    "brightness": float(gain * _BODY_TO_THREAD * lamp_falloff(hub[2])),
                }

        record_draw(
            hub_distance_m=distance,
            web_radius_m=radius,
            tilt_deg=math.degrees(tilt),
            radial_count=radials,
            mesh_spacing_mm=mesh * 1e3,
            bead_spacing_mm=bead_spacing * 1e3,
            bead_share=bead_share,
            damage=damage,
            stray_thread_count=strays,
            spider=spider is not None,
            illuminator_gain=gain,
            beam_falloff=falloff,
            focal_px=focal,
            aperture_mm=aperture_m * 1e3,
            blur_radius_at_hub_px=0.5 * aperture_m * focal * abs(1.0 / distance - 1.0 / focus),
            disc_count=len(discs["x"]),
        )
        return {
            "discs": discs,
            "spider": spider,
            "glare": glare,
            "beam": _beam_profile(height, width, falloff),
        }

    def apply(
        self,
        img: np.ndarray,
        discs: dict[str, np.ndarray],
        spider: dict[str, Any] | None,
        glare: float,
        beam: np.ndarray,
        **params: Any,
    ) -> np.ndarray:
        """Add the web's light to the frame, and the spider in front of both.

        The threads only add light: they are too thin to hide anything. The
        spider is a body, so it hides what is behind it and returns the
        lamp's light from its whole surface. Glare is a share of all of that
        light, taken before the sensor clips it.
        """
        height, width = img.shape[:2]
        lit = np.zeros((height, width), dtype=np.float32)
        splat_discs(lit, discs["x"], discs["y"], discs["radius"], discs["weight"])

        linear, scale, is_integer = to_linear_unit(img)
        if spider is not None:
            patch = spider["patch"]
            top, left = spider["y"], spider["x"]
            rows = slice(max(0, top), min(height, top + patch.shape[0]))
            columns = slice(max(0, left), min(width, left + patch.shape[1]))
            if rows.stop > rows.start and columns.stop > columns.start:
                cover = patch[
                    rows.start - top : rows.stop - top, columns.start - left : columns.stop - left
                ]
                linear[rows, columns] *= (1.0 - cover)[..., None]
                # The web behind the body is hidden with the scene.
                lit[rows, columns] *= 1.0 - cover
                lit[rows, columns] += cover * np.float32(spider["brightness"])

        lit = veiling_glare(lit, glare)

        lit *= beam
        # Nothing here is negative, so there is nothing to clip from below;
        # what is above full scale is clipped on the way out of linear light.
        out = linear + lit[..., None]
        return from_linear_unit(out, scale, is_integer, img.dtype)
