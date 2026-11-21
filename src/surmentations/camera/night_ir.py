"""Night mode: IR-cut filter removed, scene lit by the camera's own illuminator.

Physical model
--------------
A day/night camera swings its IR-cut filter out of the optical path after
dark and switches on a ring of 850 nm LEDs around the lens. Three things
change at once, and none of them is "the image gets darker":

**The image goes monochrome, and not by luminance.** Silicon is sensitive
well past 700 nm, and all three Bayer dyes transmit there, so every channel
sees mostly the same infrared signal and the colour collapses. But what it
collapses *to* is near-infrared **reflectance**, which is not the visible
brightness of a material. The largest discrepancy is vegetation: chlorophyll
absorbs red strongly and reflects NIR strongly — the "red edge" at ~700 nm —
so foliage that reads dark green by day is among the *brightest* things in
frame at night. Converting to luminance gets this exactly backwards, which is
the single most visible error in a naive night-mode augmentation.

We have no NIR channel to read, so reflectance is estimated from RGB: a broad
red-weighted response (silicon's sensitivity rises toward the red end) plus
an explicit boost where the pixel is green relative to red, which is where
chlorophyll is. It is an approximation, and it is calibrated to reproduce the
one large, well-known material difference rather than to be spectrally exact.

**The scene is lit by the camera, not by the sun.** The illuminator sits at
the lens, so irradiance falls as ``1/z^2`` and is shaped by the LED beam
profile. This is why night CCTV looks the way it does: a subject at two
metres is blown to featureless white while the background five metres behind
is nearly black, and no exposure setting fixes it, because the ratio is 6:1
in the illumination itself. Getting the inverse-square right is what produces
that automatically instead of having to paint it on.

**Gain goes up, so noise does.** Shot noise scales with the square root of
the signal and read noise does not scale at all, so a gained-up dark frame is
noisiest in the shadows — not uniformly noisy.

**Traffic has its lamps on.** A photograph taken by day shows cars with their
lamps off, and by night they are the brightest things in the frame: a pair of
points far over full scale, each with the halo a saturated source has on
these sensors. Which lamps a vehicle shows, and where, is not something
colour says in a picture that is about to lose its colour, so they are lit
only where the provider has a vehicle reader
(:mod:`surmentations.perception.vehicles`): headlamps on a vehicle seen from
the front, tail lamps on one seen from behind, where the reader found them.
With the filter out the sensor sees a red tail lamp as it sees a white
headlamp, by how bright it is, and both come out white.
"""

from __future__ import annotations

from typing import Annotated, Any

import cv2
import numpy as np
from albumentations.core.pydantic import check_range_bounds, nondecreasing
from albumentations.core.transforms_interface import BaseTransformInitSchema
from pydantic import AfterValidator, Field

from surmentations import _native, mode
from surmentations.base import CameraTransform, first_frame, require_depth_provider, scene_of
from surmentations.camera.night_lighting import (
    SceneLighting,
    light_scene,
    lit_air_of,
    note_lit_air,
)
from surmentations.depth.provider import SharedDepthProvider
from surmentations.diagnostics import record_draw
from surmentations.perception.labels import VEGETATION, SceneClass, class_weight
from surmentations.perception.vehicles import VehicleView
from surmentations.utils.channels import percentile
from surmentations.utils.grids import on_coarse_grid, pixel_grid
from surmentations.utils.night_light import even_road
from surmentations.utils.photometry import from_linear_unit, to_linear_unit
from surmentations.utils.road import paint_on
from surmentations.utils.sky import sky_to_depth_edge, sky_weight

# Broad, red-weighted stand-in for silicon's response with the IR-cut filter
# removed. Not a luminance vector: the eye's green-heavy weighting is exactly
# what does not apply once the sensor is seeing past 700 nm.
_NIR_RESPONSE = np.array([0.50, 0.30, 0.20], dtype=np.float32)


# What is left of the sky at night, as a share of what the ambient light
# would make of a surface as pale. The ambient term is light *reflected*:
# moonlight and spill off the ground and the walls. The sky reflects
# nothing, and in a frame the illuminator cannot reach, which is every frame
# whose distances are real and long, it came out as bright as by day.
_NIGHT_SKY_LEVEL = 0.12

# How green a pixel labelled as vegetation is taken to be at the least, on
# the scale the chlorophyll term works in: about what a leaf in full light
# measures. A leaf in shade, in autumn or under a sodium lamp measures
# nothing, and reflects the near infrared all the same.
_LABELLED_GREENNESS = 0.3

# How much of the glow round a lamp the sensor shows: a source far over
# full scale blooms on it whatever the air is like.
_LAMP_BLOOM = 0.3
# What a road in a beam sends back at the least, as a share of the middle
# value of the frame's road.
_BEAM_ROAD = 0.7
# The auto-exposure, once lamps light the scene, holds the brightest
# hundredth of its surfaces to this level: a camera that exposed a lit road
# for its dark verges would show the road as a white sheet.
_HIGHLIGHT_LEVEL = 0.95
_HIGHLIGHT_PERCENTILE = 99.0
# How many times what a white matt surface sends back to a lamp at the lens
# each of these does: the sheeting of a road sign and of a number plate,
# and the glass beads of road paint.
_SHEETING_GAIN = 500.0
_PAINT_GAIN = 3.0
_PAINT_SOFT_PX = 0.8
# What a plate returns of that, one plate against the next: road film on a
# plate takes most of its return, and a plate seen aslant returns less.
_PLATE_RETURNS = (0.25, 1.0)
# A plate's or a sign's sheeting is looked for in its box and this far
# round it, as a share of the box's shorter side and in pixels at the least.
_SHEETING_MARGIN = (0.3, 2.0)
# How far from the colour of a plate's ground a pixel is still ground, and
# where it is none of it, as shares of that colour's own size. Between the
# two is the edge of a character, which a pixel shares with the ground.
_GROUND_APART = (0.2, 0.6)
# The height of a plate or a sign, in pixels, under which its characters
# are lost in the ground's return, and the height from which they show as
# the picture has them: a stroke an eighth of the height, two pixels wide
# and four.
_CHARACTERS_SHOW = (16.0, 32.0)
# The share of its box a shape found in a picture has to fill for the
# picture to be believed, and the share of its own convex outline, with
# its holes filled and as it is: a triangle fills half its box, a thing
# that is one piece nearly all of its outline, and a plate's ground is the
# plate less its characters.
_SHEETING_LEAST = 0.35
_OUTLINE_FILLED = (0.7, 0.5)
# The share of the ring round a box a group of pixels may take in and
# still be the thing in the box, and not what the thing stands against.
_RING_TOO = 0.5
# The shorter side of a box, in pixels, from which a group of pixels is
# rid of what is thinner than three pixels before its shape is taken.
_OPENED_FROM = 16.0
# The longer side of a window, in pixels, up to which a shape's outline is
# drawn four times finer than the frame, for the shares of its edge pixels.
_DRAWN_FINE_TO = 96
# A sign's legend: darker than this share of the sign's pale parts, it is
# black film, which sends nothing back.
_LEGEND_BELOW = 0.3
# The most a sheeting's return is carried at, in full scales of the sensor:
# what a lens spreads of a source is in proportion to it, and a source a
# hundred times over full scale is not drawn a hundred times as wide.
_SHEETING_MOST = 24.0
# How a lens spreads a source far over full scale: the widths, in pixels of
# a frame a thousand across its diagonal, of the Gaussians that stand for a
# glare falling off as the cube of the distance, and the share of the light
# in each. The first is the lens in focus.
_GLARE = ((0.6, 0.906), (1.5, 0.05), (3.0, 0.025), (6.0, 0.0125), (12.0, 0.0065))


def _box_window(
    box: tuple[float, float, float, float], shape: tuple[int, int], margin: float
) -> tuple[slice, slice, np.ndarray] | None:
    """A box's place in a frame: the rows and columns round it, and each pixel's share in it.

    The window reaches ``margin`` pixels beyond the box on every side, cut
    to the frame. The share is the part of the pixel's area the box covers,
    so a box whose edges fall between pixels has soft edges and not ones
    moved to the nearest pixel. ``None`` for a box outside the frame.
    """
    height, width = shape
    x0, y0, x1, y1 = box
    left, top = max(0, int(np.floor(x0 - margin))), max(0, int(np.floor(y0 - margin)))
    right = min(width, int(np.ceil(x1 + margin)))
    bottom = min(height, int(np.ceil(y1 + margin)))
    if right <= left or bottom <= top:
        return None
    columns = np.arange(left, right, dtype=np.float32)
    rows = np.arange(top, bottom, dtype=np.float32)
    across = np.clip(np.minimum(x1, columns + 1.0) - np.maximum(x0, columns), 0.0, 1.0)
    down = np.clip(np.minimum(y1, rows + 1.0) - np.maximum(y0, rows), 0.0, 1.0)
    return slice(top, bottom), slice(left, right), np.outer(down, across).astype(np.float32)


def _share_in(
    box: tuple[float, float, float, float],
    window: tuple[slice, slice, np.ndarray],
    shape: tuple[int, int],
) -> np.ndarray:
    """Each pixel's share in another box, over a window already cut round the first.

    ``window`` is what :func:`_box_window` gave for the first box, and the
    result has its size. For a box that is the first one grown, or shrunk,
    by less than the window's margin.
    """
    rows, columns, inside = window
    share = np.zeros_like(inside)
    other = _box_window(box, shape, 0.0)
    if other is None:
        return share
    top, left = other[0].start - rows.start, other[1].start - columns.start
    bottom, right = top + other[2].shape[0], left + other[2].shape[1]
    if top < 0 or left < 0 or bottom > share.shape[0] or right > share.shape[1]:
        return inside
    share[top:bottom, left:right] = other[2]
    return share


def _split_in_two(values: np.ndarray, within: np.ndarray) -> np.ndarray | None:
    """The upper of the two groups a plane of values falls into, by Otsu's threshold.

    ``values`` is ``(h, w)`` in ``[0, 1]`` and the threshold is taken over
    the pixels of ``within`` alone. ``None`` where they are all but one
    value, which no threshold splits.
    """
    levels = np.clip(np.round(values * 255.0), 0.0, 255.0).astype(np.uint8)
    chosen = levels[within]
    if chosen.size < 4 or int(chosen.max()) - int(chosen.min()) < 8:
        return None
    threshold, _ = cv2.threshold(chosen.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    upper: np.ndarray = levels > threshold
    return upper


def _outline_of(
    found: np.ndarray,
    reach: np.ndarray,
    box: tuple[float, float, float, float],
    corner: tuple[int, int],
) -> tuple[float, np.ndarray] | None:
    """One guess at where a thing is in its box, made into a shape and marked for how good it is.

    ``found`` is a mask over the window round the box, ``reach`` each
    pixel's share in the box grown by what the thing may pass it by, and
    ``corner`` the window's first row and column in the frame. Returns the
    mark and the shape, each pixel's share in it; ``None`` for a guess
    that is no shape at all.

    A sign, a plate and a hoarding are all convex: so the shape is the
    convex outline of the guess's largest piece, filled, with edges that
    fall between pixels drawn as shares. Small pieces and holes (a legend,
    a plate's characters, a twig across it) are thereby dropped and
    filled. The mark is how much of its outline the piece itself fills,
    times how well the outline's own upright box agrees with the
    reader's: a reader's box is loose, but it is round the thing, so a
    guess that found half of it, or its legend, or the sky in a corner,
    scores low. A guess is no shape when its piece, holes filled, fills
    less than seven tenths of its outline, or as it is less than half
    (which is what is *round* a thing: the bodywork round a plate, the
    sky in the corners of a round sign's box, whose outline is the box),
    or when the outline is less than ``_SHEETING_LEAST`` of the box.
    """
    mask: np.ndarray = (found & (reach > 0.5)).astype(np.uint8)
    if min(box[2] - box[0], box[3] - box[1]) >= _OPENED_FROM:
        # Specks, and a hair's breadth joining the thing to its surround:
        # on a thing large enough that three pixels are not a third of it.
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, parts, sizes, _middles = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count < 2:
        return None
    largest = 1 + int(np.argmax(sizes[1:, cv2.CC_STAT_AREA]))
    piece = (parts == largest).astype(np.uint8)
    # Each pixel is a square, and the outline goes round the squares:
    # round their centres it would be half a pixel short on every side.
    # Drawn four times finer, a pixel is the four by four from its first
    # fine pixel to its last. A large thing is drawn as it is: a share
    # of a pixel on its edge is nothing to it.
    fine = 4 if max(mask.shape) <= _DRAWN_FINE_TO else 1
    rows, columns = np.nonzero(piece)
    corners = np.concatenate(
        [
            np.stack([columns * fine + dx, rows * fine + dy], axis=1)
            for dx, dy in ((0, 0), (fine - 1, 0), (0, fine - 1), (fine - 1, fine - 1))
        ]
    ).astype(np.int32)
    drawn = np.zeros((mask.shape[0] * fine, mask.shape[1] * fine), dtype=np.uint8)
    cv2.fillConvexPoly(drawn, cv2.convexHull(corners), 255)
    shape = cv2.resize(
        drawn, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_AREA
    ).astype(np.float32) / np.float32(255.0)
    # The piece with its holes filled, against its outline.
    edges, _ = cv2.findContours(piece, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    solid = np.zeros_like(piece)
    cv2.drawContours(solid, edges, -1, 1, thickness=cv2.FILLED)
    outline = float(shape.sum())
    x0, y0, x1, y1 = box
    area = max((x1 - x0) * (y1 - y0), 1e-6)
    filled, holed = _OUTLINE_FILLED
    if (
        outline < _SHEETING_LEAST * area
        or float(solid.sum()) < filled * outline
        or float(piece.sum()) < holed * outline
    ):
        return None
    left, top = corner[1] + float(columns.min()), corner[0] + float(rows.min())
    right, bottom = corner[1] + float(columns.max()) + 1.0, corner[0] + float(rows.max()) + 1.0
    shared = max(0.0, min(x1, right) - max(x0, left)) * max(0.0, min(y1, bottom) - max(y0, top))
    agreed = shared / (area + (right - left) * (bottom - top) - shared)
    return min(float(solid.sum()) / outline, 1.0) * agreed, shape


def sheeting_within(
    linear_rgb: np.ndarray, box: tuple[float, float, float, float], plate: bool
) -> tuple[slice, slice, np.ndarray] | None:
    """Where a number plate's or a sign's sheeting is, in and round the box a reader put on it.

    Returns the rows and the columns of a window round the box and, over
    that window, the share of each pixel that is sheeting: ``(h, w)``
    float32 in ``[0, 1]``. ``None`` for a box outside the frame.

    A reader's box is upright and a little loose. What is in it is
    neither: a plate seen from a mast is a slanted four-sided thing that
    fills two thirds of its box, a hoarding seen along the road is a
    wedge, and a sign is as often round, three-cornered or on its point as
    square. Lit as the box, any of them is a white rectangle laid on the
    picture. So the box says only where to look, and the shape is taken
    from the picture, by the plain means that suffice in a window of a
    few hundred pixels with one thing in the middle of it.

    Several guesses are made at which pixels are the thing:

    * those **unlike what is round the box**: the colour of the ring of
      pixels beyond the box, and how much that ring varies, say what the
      thing stands against, a sky, a wall or bodywork;
    * the **paler** and the **darker** of the two groups the window's
      brightness falls into, and the **more coloured** of the two its
      colourfulness falls into, each by Otsu's threshold;
    * for a plate, those **of its ground's colour**: the middle colour of
      the paler half of the box's middle, which is the ground and not the
      characters, nor the bodywork the box's edges take in.

    A guess that takes in half the ring round the box as well is what
    the thing stands against and is dropped. Each of the others is made
    into a convex shape and marked (:func:`_outline_of`), the mark
    lowered by the share of the ring the guess takes in, and the best is
    kept. Where none is a shape
    (a white plate on a white car, a plate in deep shade, a sign against
    a wall of its own colour) the box is taken whole.

    Within the shape not everything is sheeting. A plate's characters and
    a sign's black legend are paint or film over it, and send back what a
    matt black does: they are left out where the picture resolves them,
    which is from ``_CHARACTERS_SHOW`` pixels of height. Below that their
    strokes are under two pixels wide, and no lens keeps a stroke dark
    beside a ground at tens of full scales: the shape is filled. A sign's
    colours are all sheeting, and dyes that are red or blue to the eye
    pass the near infrared, so they return alike.
    """
    x0, y0, x1, y1 = box
    share, least = _SHEETING_MARGIN
    margin = max(least, share * min(x1 - x0, y1 - y0))
    frame = (linear_rgb.shape[0], linear_rgb.shape[1])
    window = _box_window(box, frame, margin)
    if window is None:
        return None
    rows, columns, inside = window
    patch = linear_rgb[rows, columns]
    core, ring = inside > 0.5, inside == 0.0
    if int(core.sum()) < 4 or int(ring.sum()) < 4:
        return rows, columns, inside

    light = np.sqrt(np.clip(patch.mean(axis=-1), 0.0, 1.0))
    most, fewest = patch.max(axis=-1), patch.min(axis=-1)
    coloured = (most - fewest) / (most + np.float32(0.05))
    behind = np.median(patch[ring], axis=0)
    spread = float(np.median(np.linalg.norm(patch[ring] - behind, axis=-1))) + 0.02
    unlike = np.linalg.norm(patch - behind, axis=-1) > 3.0 * spread
    guesses: list[np.ndarray] = [unlike]
    everywhere = np.ones_like(core)
    paler = _split_in_two(light, everywhere)
    if paler is not None:
        guesses += [paler, ~paler]
    vivid = _split_in_two(np.clip(coloured, 0.0, 1.0), everywhere)
    if vivid is not None:
        guesses.append(vivid)
    ground = np.ones_like(inside)
    if plate:
        # The middle of the box is plate, whatever its edges take in of
        # the car: the half of it across and the half of it down.
        across, down = 0.25 * (x1 - x0), 0.25 * (y1 - y0)
        middle = _share_in((x0 + across, y0 + down, x1 - across, y1 - down), window, frame) > 0.0
        centre = middle if int(middle.sum()) >= 4 else core
        brightness = patch.sum(axis=-1)
        pale = centre & (brightness >= float(np.median(brightness[centre])))
        colour = np.median(patch[pale], axis=0)
        apart = np.linalg.norm(patch - colour, axis=-1) / (float(np.linalg.norm(colour)) + 0.02)
        near, far = _GROUND_APART
        ground = np.clip((far - apart) / (far - near), 0.0, 1.0).astype(np.float32)
        guesses.insert(0, ground > 0.5)

    # A thing may pass its box by a pixel, where the picture shows it to.
    reach = _share_in((x0 - 1.0, y0 - 1.0, x1 + 1.0, y1 + 1.0), window, frame)
    best: tuple[float, np.ndarray] | None = None
    for guess in guesses:
        beyond = float(guess[ring].mean())
        if beyond >= _RING_TOO:
            # What is round the box is of this group too: it is what the
            # thing stands against, and not the thing.
            continue
        outlined = _outline_of(guess, reach, box, (rows.start, columns.start))
        if outlined is None:
            continue
        # And the less of the ring a group takes in, the more its edge is
        # one the picture drew and not one the box cut.
        mark = outlined[0] * (1.0 - beyond)
        if best is None or mark > best[0]:
            best = (mark, outlined[1])
    shape = inside if best is None else best[1]

    # What in the shape is not sheeting, where the picture resolves it.
    low, high = _CHARACTERS_SHOW
    shown = float(np.clip(((y1 - y0) - low) / (high - low), 0.0, 1.0))
    if shown > 0.0:
        if not plate:
            sure = shape > 0.5
            palest = float(np.percentile(most[sure], 90.0)) if bool(sure.any()) else 0.0
            if palest > 0.0:
                ground = np.clip(most / (_LEGEND_BELOW * palest), 0.0, 1.0).astype(np.float32)
        shape = shape * (np.float32(1.0 - shown) + np.float32(shown) * ground)
    return rows, columns, shape


def retroreflectors(
    linear_rgb: np.ndarray,
    labels: np.ndarray,
    views: list[VehicleView] | None,
    signs: np.ndarray | None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """What in a frame sends a lamp at the lens its light back: the road's paint, and sheeting.

    Two ``(H, W)`` float32 planes, either ``None`` where the frame has
    nothing of the kind:

    * **paint**: how many times its own matt return each pixel of road
      paint sends back. The scene's labels' road and the thin bright
      things on it (:func:`~surmentations.utils.road.paint_on`): the glass
      beads in the paint, a few times a matt white's return. It multiplies
      what the surface's own brightness makes of the illuminator's light.
    * **sheeting**: how many times a matt *white* surface's return each
      pixel of a sign's or a number plate's sheeting sends back, whatever
      it photographed as: a plate in the shade of its car is as much
      sheeting as one in the sun. Signs (``signs``) and plates (``views``)
      are the provider's reader's, and their shape in the box the reader
      drew is the picture's (:func:`sheeting_within`).

    Some things on a road are made to send light back where it came from.
    To a driver that is what makes them show in headlamps. To a camera
    whose illuminator sits round its lens it is the same thing, and
    stronger: the lens is on the lamp's own axis. Under anyone else's
    lamp, a street lamp's or a car's, none of this happens: the light goes
    back to that lamp.

    Sheeting returns some five hundred times a matt white. A matt white
    returns about a third of a candela per lux and square metre; sheeting
    is quoted at some seventy for the plainest grade to several hundred
    for the prismatic ones, so the figure is the middle of a range of ten
    to one, and not one that was read off a standard. A plate returns
    between a quarter of it and all, each its own share drawn from
    ``rng``: road film takes most of a plate's return, and so does being
    seen aslant. Without ``rng`` every plate returns the middle of that.
    """
    height, width = labels.shape
    paint: np.ndarray | None = None
    road = labels == int(SceneClass.ROAD)
    if int(road.sum()) > 64:
        painted = paint_on(linear_rgb, road)
        if bool(painted.any()):
            sharp = np.zeros((height, width), dtype=np.float32)
            sharp[painted] = _PAINT_GAIN
            # A lens does not draw an edge a pixel sharp.
            paint = cv2.GaussianBlur(sharp, (0, 0), sigmaX=_PAINT_SOFT_PX)
    boxes = [] if signs is None else [(tuple(row[:4]), False) for row in signs.tolist()]
    boxes += [(view.plate, True) for view in views or () if view.plate is not None]
    sheeting: np.ndarray | None = None
    for box, plate in boxes:
        returns = 1.0
        if plate:
            low, high = _PLATE_RETURNS
            returns = 0.5 * (low + high) if rng is None else float(rng.uniform(low, high))
        found = sheeting_within(linear_rgb, box, plate)
        if found is None:
            continue
        rows, columns, share = found
        if sheeting is None:
            sheeting = np.zeros((height, width), dtype=np.float32)
        np.maximum(
            sheeting[rows, columns],
            np.float32(_SHEETING_GAIN * returns) * share,
            out=sheeting[rows, columns],
        )
    return paint, sheeting


def lens_glare(light: np.ndarray) -> np.ndarray:
    """A plane of light far over full scale, as a lens spreads it over the sensor.

    ``light`` is ``(H, W)`` in full scales of the sensor, mostly 0. No
    lens draws a point as a point: most of its light lands within a pixel,
    and a few hundredths are scattered round it, falling off roughly as
    the cube of the distance. On anything within the sensor's range those
    hundredths are lost in the picture. On a number plate returning tens
    of full scales they are the picture: the plate grows by a pixel or two,
    its corners round, its characters drown, and it sits in a haze that
    fades over a dozen pixels. Drawn without that spread, the same plate is
    a hard white rectangle with a pixel-sharp edge, which no camera makes.

    The fall is stood for by a few Gaussians of doubling width, each with
    half the light of the one before. Only the part of the frame within
    reach of some light is worked on, and the two widest are taken on a
    grid four times coarser.
    """
    lit = np.flatnonzero(light.max(axis=1) > 0.0), np.flatnonzero(light.max(axis=0) > 0.0)
    if lit[0].size == 0:
        return light
    height, width = light.shape
    scale = float(np.hypot(height, width)) / 1000.0
    reach = int(np.ceil(3.0 * _GLARE[-1][0] * scale)) + 4
    top, bottom = max(0, int(lit[0][0]) - reach), min(height, int(lit[0][-1]) + reach + 1)
    left, right = max(0, int(lit[1][0]) - reach), min(width, int(lit[1][-1]) + reach + 1)
    source = np.ascontiguousarray(light[top:bottom, left:right])
    spread = np.zeros_like(source)
    coarse: np.ndarray | None = None
    for index, (sigma, share) in enumerate(_GLARE):
        sigma_px = sigma * scale
        if index >= len(_GLARE) - 2 and min(source.shape) >= 16:
            if coarse is None:
                coarse = cv2.resize(
                    source,
                    (max(1, source.shape[1] // 4), max(1, source.shape[0] // 4)),
                    interpolation=cv2.INTER_AREA,
                )
            wide = cv2.GaussianBlur(coarse, (0, 0), sigmaX=max(sigma_px / 4.0, 0.3))
            wide = cv2.resize(
                wide, (source.shape[1], source.shape[0]), interpolation=cv2.INTER_LINEAR
            )
        else:
            wide = cv2.GaussianBlur(source, (0, 0), sigmaX=max(sigma_px, 0.3))
        spread += np.float32(share) * wide
    out = np.zeros_like(light)
    out[top:bottom, left:right] = spread
    return out


def near_infrared_reflectance(
    linear_rgb: np.ndarray, vegetation_gain: float, vegetation: np.ndarray | None = None
) -> np.ndarray:
    """Estimate NIR reflectance from a visible-light image.

    Broad red-weighted response plus a chlorophyll term. ``vegetation_gain``
    controls how strongly foliage lifts; 0 disables the correction and leaves
    a plain broad-band conversion.

    ``vegetation`` is the weight of what the scene's labels call vegetation,
    ``(H, W)`` in ``[0, 1]``, where there are labels
    (:mod:`surmentations.perception.labels`). The chlorophyll term is
    read off colour, and colour says "leaf" only for a green leaf in good
    light; a labelled one is lifted whatever colour it photographed as.
    """
    if mode.fast():
        # In one pass where the compiled kernels are there. The frame's
        # mean response, which the labelled leaves are lit against, is the
        # response of its mean colour.
        usual = float(np.dot(cv2.mean(linear_rgb)[:3], _NIR_RESPONSE.astype(np.float64)))
        done = _native.nir_reflectance(
            linear_rgb, vegetation, _NIR_RESPONSE, vegetation_gain, _LABELLED_GREENNESS, usual
        )
        if done is not None:
            return done
    broadband = linear_rgb @ _NIR_RESPONSE

    # Greenness relative to red is where chlorophyll is, and chlorophyll is
    # what reflects hard in the near infrared.
    red, green, blue = linear_rgb[..., 0], linear_rgb[..., 1], linear_rgb[..., 2]
    greenness = np.clip(
        (green - np.maximum(red, blue)) / np.maximum(green + red + blue, 1e-4), 0.0, 1.0
    )
    if vegetation is not None:
        # By the pixel's own brightness against the frame's, so that the
        # shade inside a crown stays shade and a tree keeps its shape.
        lit = np.clip(broadband / max(float(broadband.mean()), 1e-4), 0.0, 1.0)
        greenness = np.maximum(greenness, np.float32(_LABELLED_GREENNESS) * vegetation * lit)
    return np.clip(broadband + vegetation_gain * greenness, 0.0, 4.0)


class NightIrMode(CameraTransform):
    """Monochrome IR night mode with the camera's own illuminator.

    With a provider that carries scene labels
    (:mod:`surmentations.perception.labels`): vegetation is lifted by
    name as well as by colour, so a tree is bright in the near infrared in
    shade, in autumn and under a sodium lamp too; and the sky, which
    reflects neither the illuminator nor the ambient light, is dark. Where
    the camera's height is known the illuminator's inverse-square law runs
    over real metres, and a scene beyond its reach is lit by the ambient
    term alone.

    With a provider that also carries a vehicle reader
    (:mod:`surmentations.perception.vehicles`) the traffic's lamps are
    lit: see ``vehicle_lamp_range``.

    Args:
        depth_provider: shared :class:`~surmentations.depth.provider.SharedDepthProvider`.
        illuminator_range_m: distance at which the illuminator delivers unit
            exposure. Objects nearer than this blow out, objects further fade
            into the noise — and the ``1/z^2`` between them is why night CCTV
            has such a narrow band of usable distance.
        beam_falloff_range: how tightly the LED beam is focused. Higher
            concentrates light on the optical axis and darkens the corners
            further; the exponent of a ``cos^k`` beam profile.
        ambient_range: residual scene lighting (moon, streetlights, spill) as
            a fraction of unit exposure, so the far field is not pure black.
        vegetation_gain_range: how strongly foliage lifts relative to a plain
            broad-band conversion — the chlorophyll red-edge term.
        target_level_range: level the camera's auto-exposure drives the scene
            to, as a fraction of full scale measured at the 80th percentile.
            This is not a cosmetic normalisation — a night camera really does
            gain up until the scene is usable, and without modelling it the
            result depends entirely on how the depth map happened to be
            scaled: pick an illuminator range slightly too short and the whole
            lower frame saturates. Auto-exposure fixes the *level* while
            leaving the ``1/z^2`` **ratio** between near and far intact, and
            that ratio is the part that makes night footage recognisable.
        gain_range: sensor gain, which sets how far the noise floor is
            lifted. It does not scale the signal: the auto-exposure has
            already placed that, and gain is one of the levers it used to get
            there. Higher gain therefore means a noisier frame at the same
            brightness, which is what raising ISO actually buys you.
        read_noise_range: signal-independent noise, as a fraction of unit
            exposure. Unlike shot noise this does not shrink in the shadows,
            which is why a gained-up night frame is noisiest where it is
            darkest.
        street_lamp_range: how much light a street lamp throws on the road
            straight under it, in unit exposures, drawn per call; 0 is
            none, and the scene is lit by the ambient term alone, as
            before. Above it the scene is lit as a road is at night, from
            a few points: each street lamp throws a pool under itself that
            falls off with the square of the distance, the lamps
            themselves burn, and what lies between is as dark as
            ``ambient_range`` makes it. Where the lamps are comes from the
            provider: from its vehicle reader, which finds them all along
            a road, or from its scene labels, which find the nearest. The
            scene's ground has to be labelled, since the light is worked
            out in space. With neither lamps nor labels this does nothing.
            The labelled road also has the day's light and shade taken off
            it: the sun's shadows have no business under street lamps.
        retroreflector_range: how strongly the things made to send light
            back where it came from return the illuminator's, drawn per
            call, as a multiple of what they usually do; 0 is none. Road
            paint, from the scene's labels, comes back a few times as
            bright as a white surface would. Road signs and number plates,
            which the provider's vehicle reader finds, come back some
            five hundred times as bright: within the illuminator's
            reach a sign is a white shape and a plate a white bar in a
            haze of its own, its characters gone, as they are on a real
            camera that meters for the scene and not for the plate. The
            shape is the plate's or the sign's in the picture, not the
            reader's box, and its edge is the lens's
            (:func:`sheeting_within`, :func:`lens_glare`). It is the
            illuminator's light alone that comes back this way, since the
            lens is beside the lamp; street lamps and headlamps get theirs
            back themselves. See :func:`retroreflectors`.
        vehicle_lamp_range: how brightly the lamps of the vehicles in the
            frame are lit, drawn per call; 1 is dipped beams, 0 none. It
            does something only on a provider built with ``vehicles=``:
            headlamps on each vehicle its reader saw from the front and
            tail lamps on each it saw from behind, where it found them,
            far over full scale and white, as an infrared camera shows
            both, with the bloom of a saturated source. Most vehicles are
            lit and some are not, as parked ones are not. Lit here and not
            before, the lamps are light and not paint: the illuminator's
            falloff does not dim them, and whatever follows in a pipeline,
            fog, rain or snow at night, finds them as it finds any lamp in
            a frame. A lit vehicle also throws its beam on the road ahead
            of it. Without a vehicle reader no lamp is lit, whatever this
            is.

    Targets:
        image

    Image types:
        uint8, float32
    """

    class InitSchema(BaseTransformInitSchema):
        """The arguments of ``NightIrMode`` as Albumentations validates them: each
        range ordered low to high and within the bounds given here.
        """
        depth_provider: SharedDepthProvider
        illuminator_range_m: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.5, 200.0)),
        ]
        beam_falloff_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 12.0)),
        ]
        ambient_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 1.0)),
        ]
        vegetation_gain_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 2.0)),
        ]
        target_level_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.05, 0.95)),
        ]
        gain_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.1, 20.0)),
        ]
        read_noise_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 0.2)),
        ]
        residual_tint: float = Field(ge=0.0, le=0.5)
        vehicle_lamp_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 2.0)),
        ]
        street_lamp_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 5.0)),
        ]
        retroreflector_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 3.0)),
        ]

    def __init__(
        self,
        depth_provider: SharedDepthProvider,
        illuminator_range_m: tuple[float, float] = (4.0, 14.0),
        beam_falloff_range: tuple[float, float] = (1.0, 4.0),
        ambient_range: tuple[float, float] = (0.01, 0.12),
        vegetation_gain_range: tuple[float, float] = (0.25, 0.7),
        target_level_range: tuple[float, float] = (0.35, 0.6),
        gain_range: tuple[float, float] = (0.8, 1.6),
        read_noise_range: tuple[float, float] = (0.004, 0.03),
        residual_tint: float = 0.04,
        vehicle_lamp_range: tuple[float, float] = (0.7, 1.3),
        street_lamp_range: tuple[float, float] = (0.6, 1.4),
        retroreflector_range: tuple[float, float] = (0.6, 1.4),
        p: float = 0.5,
    ):
        """Store the ranges; each argument is described in the class docstring.

        Raises ``ValueError`` without a ``depth_provider``: the one
        :class:`~surmentations.depth.provider.SharedDepthProvider` of the
        pipeline must be passed in, so that depth is estimated once per frame.
        """
        super().__init__(p=p)
        self.depth_provider = require_depth_provider(depth_provider, cls_name=type(self).__name__)
        self.illuminator_range_m = illuminator_range_m
        self.beam_falloff_range = beam_falloff_range
        self.ambient_range = ambient_range
        self.vegetation_gain_range = vegetation_gain_range
        self.target_level_range = target_level_range
        self.gain_range = gain_range
        self.read_noise_range = read_noise_range
        self.residual_tint = residual_tint
        self.vehicle_lamp_range = vehicle_lamp_range
        self.street_lamp_range = street_lamp_range
        self.retroreflector_range = retroreflector_range

    def get_params_dependent_on_data(
        self, params: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, Any]:
        """Draw the illuminator and the camera, and how much light reaches each pixel.

        Returns the ``exposure``: the inverse square of the distance from the
        illuminator at the lens, times its beam profile. With it, the ambient
        light, the auto-exposure target, the vegetation gain, the sensor gain
        and read noise, and a seed for the noise.
        """
        scene = scene_of(data)
        height, width = params["shape"][:2]

        depth_m = np.maximum(self.depth_provider.get_metric_depth(scene), 0.3)

        reference = self.py_random.uniform(*self.illuminator_range_m)
        falloff = self.py_random.uniform(*self.beam_falloff_range)

        # Inverse-square from an illuminator sitting at the lens.
        irradiance = (reference / depth_m) ** 2

        # LED beam profile: brightest on the optical axis, falling as cos^k of
        # the field angle. Approximated from normalised image radius, which is
        # monotonic in field angle.
        cx, cy = width / 2.0, height / 2.0

        def beam_at(yy: np.ndarray, xx: np.ndarray) -> np.ndarray:
            """The illuminator's beam at the given coordinates: ``cos(field angle) ** falloff``.

            The field angle is taken from the normalised distance from the
            frame's centre, at most 60 degrees at its edge.
            """
            radius = np.sqrt(((xx - cx) / max(cx, 1)) ** 2 + ((yy - cy) / max(cy, 1)) ** 2)
            profile: np.ndarray = (
                np.clip(np.cos(np.clip(radius, 0.0, 1.0) * (np.pi / 3.0)), 0.0, 1.0) ** falloff
            )
            return profile

        # The beam changes over a third of the frame. In fast mode its cosine
        # and power are taken on a grid four times coarser and enlarged.
        beam = (
            on_coarse_grid(height, width, beam_at)
            if mode.fast()
            else beam_at(*pixel_grid(height, width))
        )

        # What the scene's labels call vegetation, where there are labels:
        # the near infrared's brightest material, whatever colour it is.
        labels = self.depth_provider.get_labels(scene)
        vegetation = night_sky = None
        if labels is not None:
            soften = max(1.0, 0.003 * float(np.hypot(height, width)))
            vegetation = class_weight(labels, VEGETATION, soften_px=soften)
            # And the sky, which no lamp lights and nothing is reflected by.
            # Drawn up to the skyline's own edge: a network's outline is a
            # few pixels out, and a band of sky left as bright as by day
            # is a white line round every roof.
            relative_depth = self.depth_provider.get_depth(scene)
            linear, _scale, _is_integer = to_linear_unit(first_frame(scene))
            night_sky = sky_to_depth_edge(
                sky_weight(linear, relative_depth, labels), relative_depth, linear
            )

        record_draw(
            illuminator_range_m=reference,
            beam_falloff=falloff,
            depth_min_m=lambda: float(depth_m.min()),
            depth_p50_m=lambda: float(np.median(depth_m)),
            depth_max_m=lambda: float(depth_m.max()),
        )
        drawn = {
            "exposure": (irradiance * beam).astype(np.float32),
            "ambient": self.py_random.uniform(*self.ambient_range),
            "target_level": self.py_random.uniform(*self.target_level_range),
            "vegetation_gain": self.py_random.uniform(*self.vegetation_gain_range),
            "gain": self.py_random.uniform(*self.gain_range),
            "read_noise": self.py_random.uniform(*self.read_noise_range),
            "noise_seed": self.py_random.randint(0, 2**31 - 1),
            "vegetation": vegetation,
            "night_sky": night_sky,
        }

        # The lamps, drawn after everything else is: a provider that knows
        # of none draws what it always drew.
        lamps = self.py_random.uniform(*self.vehicle_lamp_range)
        lamp_seed = self.py_random.randint(0, 2**31 - 1)
        street = self.py_random.uniform(*self.street_lamp_range)
        lighting = light_scene(
            self.depth_provider, scene, labels, depth_m, lamp_seed, lamps, street
        )
        drawn["lighting"] = lighting
        if lighting.in_beam is not None and labels is not None:
            # What a road in a beam sends back at the least: the road's own
            # middle value, where there is a road to take it from.
            paved = labels == int(SceneClass.ROAD)
            if int(paved.sum()) > 50:
                linear = to_linear_unit(first_frame(scene))[0]
                drawn["road_level"] = float(np.median(linear[paved] @ _NIR_RESPONSE))
        # The light in the air, for whatever puts something in that air
        # afterwards. Colourless: this camera sees no colour.
        white = np.ones(3, dtype=np.float32)
        drawn["air"] = lit_air_of(lighting, white, white)
        drawn["exposed"] = [1.0]

        # What is made to send light back where it came from, under a lamp
        # that is at the lens: the last of the draws.
        retro = self.py_random.uniform(*self.retroreflector_range)
        returning = False
        if retro > 0.0 and labels is not None:
            linear = to_linear_unit(first_frame(scene))[0]
            paint, sheeting = retroreflectors(
                linear,
                labels,
                self.depth_provider.get_vehicles(scene),
                self.depth_provider.get_signs(scene),
                np.random.default_rng(self.py_random.randint(0, 2**31 - 1)),
            )
            if paint is not None:
                drawn["retro"] = np.float32(retro) * paint
            if sheeting is not None:
                drawn["sheeting"] = np.float32(retro) * sheeting
            returning = paint is not None or sheeting is not None
        record_draw(
            vehicle_lamps=lamps,
            vehicles_read=lighting.vehicles_read,
            vehicles_lit=lighting.vehicles_lit,
            street_lamps=lighting.street_lamps if lighting.lit else 0,
            street_lamp_level=street,
            retroreflectors=retro,
            retroreflectors_drawn=returning,
        )
        return drawn

    def apply_with_params(
        self, params: dict[str, Any], *args: Any, **kwargs: Any
    ) -> dict[str, Any]:
        """Apply, then note for the frame returned what light hangs in its air.

        For the fog, rain or snow a pipeline puts in that air afterwards
        (:func:`~surmentations.camera.night_lighting.note_lit_air`).
        """
        result = super().apply_with_params(params, *args, **kwargs)
        note_lit_air(self.depth_provider, params, result)
        return result

    def apply(
        self,
        img: np.ndarray,
        exposure: np.ndarray,
        ambient: float,
        target_level: float,
        vegetation_gain: float,
        gain: float,
        read_noise: float,
        noise_seed: int,
        vegetation: np.ndarray | None = None,
        night_sky: np.ndarray | None = None,
        lighting: SceneLighting | None = None,
        road_level: float | None = None,
        retro: np.ndarray | None = None,
        exposed: list[float] | None = None,
        sheeting: np.ndarray | None = None,
        **params: Any,
    ) -> np.ndarray:
        """Render the frame as the near-infrared camera sees it at night.

        Near-infrared reflectance times the light that arrived, scaled by the
        auto-exposure to put the 80th percentile at the target, plus the
        traffic's lamps where there are any to light, plus shot and read
        noise, in a grey with the residual tint of the colour filters.
        """
        linear, scale, is_integer = to_linear_unit(img)

        reflectance = near_infrared_reflectance(linear, vegetation_gain, vegetation)
        lighting = SceneLighting() if lighting is None else lighting
        if lighting.lit and lighting.road is not None:
            # Lit by lamps: the road loses the day's light and shade, which
            # no lamp explains, and takes the pools and the beams.
            reflectance = even_road(reflectance, lighting.road)
        beamed = lighting.lit and lighting.in_beam is not None and road_level is not None
        sky_scale = 1.0 - _NIGHT_SKY_LEVEL
        lit = None
        if mode.fast():
            # The light on every surface in one pass, where the compiled
            # kernels are there; the same arithmetic below.
            lit = _native.night_ir_light(
                reflectance,
                exposure,
                lighting.in_beam if beamed else None,
                lighting.street if lighting.lit else None,
                lighting.beams if lighting.lit else None,
                night_sky,
                ambient,
                _BEAM_ROAD * road_level if beamed and road_level is not None else 0.0,
                sky_scale,
            )
        if lit is not None:
            signal, reflectance = lit
        else:
            arriving = exposure + ambient
            if lighting.lit:
                if lighting.in_beam is not None and road_level is not None:
                    # A beam falls on a road, and a road in the day's deep
                    # shade photographed as black: lit, it would stay black.
                    # The shade was the light and not the road, so where a
                    # beam falls the road sends back most of what a road does.
                    floor = np.float32(_BEAM_ROAD * road_level) * lighting.in_beam
                    reflectance = np.maximum(reflectance, floor)
                for thrown in (lighting.street, lighting.beams):
                    if thrown is not None:
                        arriving = arriving + thrown
            signal = reflectance * arriving
            if night_sky is not None:
                signal = signal * (1.0 - np.float32(sky_scale) * night_sky)

        # Auto-exposure. The camera gains up until the scene is usable, so the
        # absolute level is set by the AE loop rather than by the illuminator's
        # nominal range — while the near/far *ratio* the inverse-square law
        # produced is untouched, which is the part that reads as night footage.
        reference_level = percentile(signal, 80.0)
        level = 1.0
        returned = glare = None
        if reference_level > 1e-6:
            level = target_level / reference_level
            if lighting.lit:
                # A scene lit from a few points is mostly dark, and its
                # 80th percentile is in the dark. Exposed for that, the
                # lit road is a white sheet; a camera holds its highlights.
                brightest = percentile(signal, _HIGHLIGHT_PERCENTILE)
                level = min(level, _HIGHLIGHT_LEVEL / max(brightest, 1e-6))
            if exposed is not None:
                exposed[0] = float(level)
            if retro is not None:
                # The glass beads in road paint send the illuminator's
                # light back down its own beam, to a lens that is beside
                # it: a few times what the paint's own brightness would.
                returned = (retro, reflectance, exposure)
            if sheeting is not None:
                # And sheeting hundreds of times what a white surface
                # would, whatever the plate or the sign photographed as.
                # Added once the camera has exposed for the scene, which is
                # not for these; and spread as a lens spreads what is far
                # over full scale, which is all that shows of it.
                glare = lens_glare(
                    np.minimum(np.float32(level) * sheeting * exposure, _SHEETING_MOST)
                )

        # Gain deliberately does *not* rescale the signal here. On a real
        # camera gain is one of the things the AE loop uses to *reach* its
        # target, not something applied on top of a level it already hit.
        # Multiplying here as well drove the 80th percentile to
        # target x gain -- up to 0.6 x 1.6 = 0.96 -- so a fifth of the frame
        # sat against the clipping point and the near road washed out to a
        # featureless white plate covering 24% of the image. What gain does
        # control is the noise, below.

        # Shot noise scales as sqrt(signal); read noise does not scale at all,
        # which is why the shadows of a gained-up frame are the noisy part.
        rng = np.random.default_rng(noise_seed)
        noise = rng.standard_normal(signal.shape).astype(np.float32)
        # Near-monochrome, with the slight residual cast these sensors keep
        # rather than a mathematically perfect grey.
        tint = np.array([1.0 + self.residual_tint, 1.0, 1.0 - self.residual_tint * 0.5])
        vehicle_lamps = head_lamps = None
        if lighting.vehicle_light is not None and lighting.vehicle_glow is not None:
            vehicle_lamps = (lighting.vehicle_light, lighting.vehicle_glow)
        if lighting.head_light is not None and lighting.head_glow is not None:
            head_lamps = (lighting.head_light, lighting.head_glow)
        done = None
        if vehicle_lamps is None or vehicle_lamps[0].ndim == 3:
            done = _native.night_ir_expose(
                signal,
                noise,
                [level, _LAMP_BLOOM, 0.02, read_noise, gain, *tint],
                returned,
                glare,
                vehicle_lamps,
                head_lamps,
            )
        if done is not None:
            return from_linear_unit(done, scale, is_integer, img.dtype)

        if reference_level > 1e-6:
            signal = signal * level
        if returned is not None:
            signal = signal + np.float32(level) * returned[0] * returned[1] * returned[2]
        if glare is not None:
            signal = signal + glare
        # The lamps. They are sources and not surfaces: what they send the
        # lens is theirs, whatever the illuminator delivers where they are
        # and wherever the auto-exposure has put the scene, which is
        # exposed for the road and not for them. So they are added here, at
        # their own level, after both. White whatever their colour: with
        # the filter out a red lamp is a bright one and no more. (Channel
        # against channel: a maximum along an axis of three is far slower.)
        for lamps in (vehicle_lamps, head_lamps):
            if lamps is None:
                continue
            light, glow = lamps
            if light.ndim == 3:
                light = np.maximum(np.maximum(light[..., 0], light[..., 1]), light[..., 2])
                glow = np.maximum(np.maximum(glow[..., 0], glow[..., 1]), glow[..., 2])
            signal = signal + light + np.float32(_LAMP_BLOOM) * glow

        signal = signal + noise * (np.sqrt(np.maximum(signal, 0.0)) * 0.02 + read_noise) * gain

        signal = np.clip(signal, 0.0, 1.0)

        out = signal[..., None] * tint.astype(np.float32)[None, None, :]
        np.clip(out, 0.0, 1.0, out=out)
        return from_linear_unit(out, scale, is_integer, img.dtype)
