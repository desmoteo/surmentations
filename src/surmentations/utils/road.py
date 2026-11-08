"""Which way a road runs, and where across it each pixel lies.

Some of what happens on a road is laid out *along* it. Wheels wear two
tracks down every lane of a snowy road. Water stands in the same wheel ruts
and along the kerb. A car on a wet road trails its spray behind it. None of
those can be drawn from a mask of where the road is: they need the road's
own coordinates.

One of the two is enough, and it can be had without finding lanes or a
vanishing point. A road in a picture is a region with two long sides. For
each pixel of it, its position **across** the road is how far it is from
one side as a share of the way to the other, 0 on one kerb and 1 on the
other (:func:`road_frame`). Lines of constant position run along the road.
On a straight road they are the lines through its vanishing point, exactly:
a line of the picture that crosses three lines through one point is cut by
them in the same proportion wherever it crosses. On a bend they bend with
it.

Finding the sides
-----------------
A mask of a road is a ragged thing: cut by the lamp posts in front of it,
joined to the slip road beside it, with a bite out of it wherever the
network was unsure. So the sides are not its outline. The mask is read row
by row. In each row the road is a run of pixels; the run is followed up and
down the frame from the row where it is widest, taking in each row the run
that continues the last, which keeps to one carriageway where two run side
by side. Its two ends, row by row, are the sides, and a smooth curve is
fitted through each, robustly, so that a bite or a bulge of a few rows
moves nothing. A side that runs out of the frame is not seen and is not
fitted: it is drawn from the far end of the road to the frame's corner,
which on a road running away from the camera is a line through its
vanishing point as well.

What is left of the mask once a road has been taken out of it is read
again, for the slip road and the other carriageway, a few times over.

A road that runs *across* the picture has rows that run along it, and
neither of its sides is met by reading along a row: both are out of the
frame. A piece of which that is true is read by columns instead. (Which
way a road runs was first read off the way its outline faces. Seen from a
low camera the sides of a road that runs straight away from it lie nearly
flat across the picture, and such roads were read sideways.)

When there is no road
---------------------
A piece of mask whose ends lie far from the curves fitted to them, by the
measure of its own width, on both sides, gets no frame, and whatever asked
for one draws nothing along it. That is what a junction or a car park
looks like when it is ragged, and it is all that tells one from a road: a
tidy rectangle of paving is a road seen from above as far as this can
know. And a mask is only as right as whatever made it. The network that
names a crossing seen from straight above "building" leaves a piece of it
with one straight edge, which is read as a road running the wrong way.
:func:`road_lanes` catches the case where the vehicles on it lie across
the way it was read; nothing catches the rest.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import cv2
import numpy as np

from surmentations.perception.vehicles import HEAVY_KINDS, VehicleView

# The longer side, in pixels, of the copy a road's frame is worked out on:
# a road's sides are long, and their curves are smooth.
_WORK_SIDE = 320
# Most roads read out of one mask, and the least share of the frame a piece
# of mask may cover to be read at all.
_MOST_ROADS = 4
_LEAST_SHARE = 0.012
# Holes in the road smaller than this share of the frame are things standing
# on it, and are filled; the mask is first closed over gaps this fraction of
# the diagonal wide.
_HOLE_SHARE = 0.03
_CLOSE = 0.012
# Fewest rows a road is followed over, and fewest rows in which a side is
# in the frame for a curve to be fitted to it: a parabola from that many, a
# straight line from a third as many.
_FEWEST_ROWS = 12
_ROWS_FOR_A_CURVE = 24
# The share of a road's rows in which a side must be in the frame for it to
# be fitted as a bend, and how much nearer its points a parabola must pass
# than a straight line does.
_SEEN_FOR_A_CURVE = 0.7
_BEND_FITS_BETTER = 0.6
# How far, as a share of the road's width there, the middle one of a side's
# points may lie from the curve fitted to it. Past this it is not a side.
_SIDE_TOLERANCE = 0.1
# The narrowest a road is taken to be, in pixels of the working copy.
_LEAST_WIDTH = 3.0


class RoadFrame(NamedTuple):
    """A road's own coordinates, per pixel of the frame.

    ``weight`` is 1 on road that has a frame and 0 elsewhere, and the other
    fields mean something only where it is set.

    The fields:

    * ``weight``: ``(H, W)`` float32 in ``[0, 1]``.
    * ``across``: ``(H, W)`` float32 in ``[0, 1]``: position across the road,
      from one side to the other.
    * ``width_px``: ``(H, W)`` float32: the road's width through the pixel,
      in pixels, which is what turns a width in the world into one in
      the picture.
    * ``along``: ``(H, W, 2)`` float32: the unit vector along the road,
      ``(dx, dy)``, pointing down the frame where it has a choice.
    * ``piece``: ``(H, W)`` int32: which connected piece of road the pixel
      belongs to, from 1; 0 off the road.
    * ``heights_wide``: for each road, at the index ``piece`` gives it, how
      wide it is in heights of the camera above it, where the road was read
      as a camera sees it (:class:`RoadModel`); ``None`` otherwise. With
      the camera's height in metres, that is the road's width in metres.
    """

    weight: np.ndarray
    across: np.ndarray
    width_px: np.ndarray
    along: np.ndarray
    piece: np.ndarray
    heights_wide: np.ndarray | None = None


def _tidy(road: np.ndarray) -> np.ndarray:
    """A road mask with the gaps closed and the holes things stand in filled, as uint8."""
    height, width = road.shape
    reach = max(3, round(_CLOSE * float(np.hypot(height, width))) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (reach, reach))
    mask = cv2.morphologyEx(road.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    count, holes, stats, _centroids = cv2.connectedComponentsWithStats(1 - mask, connectivity=4)
    for index in range(1, count):
        left, top, wide, tall, area = (int(value) for value in stats[index])
        inside = left > 0 and top > 0 and left + wide < width and top + tall < height
        if inside and area < _HOLE_SHARE * height * width:
            mask[holes == index] = 1
    return mask


def _widest_run(row: np.ndarray, near: tuple[int, int] | None) -> tuple[int, int] | None:
    """The run of set pixels in ``row`` that continues ``near``, as ``(first, last)``.

    The one that overlaps ``near`` most; with no ``near``, the longest.
    ``None`` when the row holds none, or none that overlaps.
    """
    padded = np.concatenate([[0], row, [0]]).astype(np.int8)
    starts = np.flatnonzero(np.diff(padded) == 1)
    ends = np.flatnonzero(np.diff(padded) == -1) - 1
    if starts.size == 0:
        return None
    if near is None:
        best = int(np.argmax(ends - starts))
        return int(starts[best]), int(ends[best])
    overlap = np.minimum(ends, near[1]) - np.maximum(starts, near[0]) + 1
    best = int(np.argmax(overlap))
    if overlap[best] <= 0:
        return None
    return int(starts[best]), int(ends[best])


def _fit_side(rows: np.ndarray, ends: np.ndarray, seen: np.ndarray) -> np.ndarray | None:
    """A smooth curve through one side of a road: polynomial coefficients, or ``None``.

    ``ends`` are the side's column in each of ``rows``, and ``seen`` says in
    which rows the side is inside the frame. Fitted by reweighting, so that
    a few rows that stray do not move the curve. ``None`` with too few rows
    to fit anything.
    """
    if int(seen.sum()) < _ROWS_FOR_A_CURVE // 3:
        return None
    x, y = rows[seen].astype(np.float64), ends[seen].astype(np.float64)

    def fit(degree: int) -> tuple[np.ndarray, float]:
        """The curve of that degree, and how far the middle point lies from it."""
        weights = np.ones_like(x)
        curve = np.zeros(degree + 1)
        spread = 0.0
        for _ in range(6):
            curve = np.polyfit(x, y, degree, w=weights)
            residual = y - np.polyval(curve, x)
            spread = float(np.median(np.abs(residual)))
            weights = 1.0 / (1.0 + (residual / (2.5 * (1.4826 * spread + 0.5))) ** 2)
        return curve, spread

    line, off_line = fit(1)
    # A bend, if the side is in the frame for most of the road's length and
    # a parabola fits it plainly better. A side seen for a short way is
    # carried on as a straight line: a parabola fitted to a little of a
    # road's edge and followed out of the frame goes anywhere.
    whole = int(seen.sum()) >= max(_ROWS_FOR_A_CURVE, _SEEN_FOR_A_CURVE * len(rows))
    if whole:
        bend, off_bend = fit(2)
        if off_bend < _BEND_FITS_BETTER * off_line - 0.25:
            return bend
    return line


def _road_in(piece: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int] | None:
    """One road read out of a mask by rows: its body, and its two sides per row.

    Returns the mask of the road's body, ``(H, W)`` bool; the column of its
    first and second side in every row, ``(H,)`` each, ``nan`` in rows the
    road does not reach; and how many of the two sides were seen in the
    frame and fitted. ``None`` when the mask holds no road with two sides:
    too few rows, or ends that fit no smooth curve.
    """
    height, width = piece.shape
    counts = piece.sum(axis=1)
    if int(counts.max()) < _LEAST_WIDTH:
        return None
    start = int(np.argmax(counts))
    runs: dict[int, tuple[int, int]] = {}
    first = _widest_run(piece[start], None)
    if first is None:
        return None
    runs[start] = first
    for step in (-1, 1):
        near, row = first, start + step
        while 0 <= row < height:
            run = _widest_run(piece[row], near)
            if run is None:
                break
            runs[row] = near = run
            row += step
    if len(runs) < _FEWEST_ROWS:
        return None
    rows = np.array(sorted(runs))
    lefts = np.array([runs[row][0] for row in rows], dtype=np.float64)
    rights = np.array([runs[row][1] for row in rows], dtype=np.float64)

    far = int(rows[0])
    fitted = []
    seen_sides = ragged_sides = 0
    for ends, border, corner in ((lefts, 0.0, 0.0), (rights, width - 1.0, width - 1.0)):
        seen = np.abs(ends - border) > 1.0
        curve = _fit_side(rows, ends, seen)
        side = None
        if curve is not None:
            side = np.polyval(curve, rows)
            stray = np.abs(ends[seen] - side[seen]) / np.maximum(rights[seen] - lefts[seen], 1.0)
            if float(np.median(stray)) > _SIDE_TOLERANCE:
                # In the frame, and no smooth curve: a side broken up by
                # what stands in front of it. Drawn as one not seen is, and
                # a road both of whose sides are like that is not a road.
                ragged_sides += 1
                side = None
            else:
                seen_sides += 1
        if side is None:
            # Out of the frame. From where the road ends, far off, to the
            # frame's corner: through the vanishing point of a road that
            # runs away from the camera, and down the border of one seen
            # from above.
            slope = (corner - ends[0]) / max(float(rows[-1] - far), 1.0)
            side = ends[0] + slope * (rows - far)
        fitted.append(side)
    if ragged_sides == 2 or float(np.median(fitted[1] - fitted[0])) < _LEAST_WIDTH:
        return None

    body = np.zeros(piece.shape, dtype=bool)
    first_side = np.full(height, np.nan)
    second_side = np.full(height, np.nan)
    for index, row in enumerate(rows):
        body[row, runs[row][0] : runs[row][1] + 1] = True
        first_side[row], second_side[row] = fitted[0][index], fitted[1][index]
    return body, first_side, second_side, seen_sides


def road_frame(road: np.ndarray) -> RoadFrame | None:
    """The road's own coordinates for a mask of where the road is, or ``None``.

    ``road`` is ``(H, W)``, true or above a half where there is road, and
    should include what stands on the road: a road with its vehicles cut
    out of it has a hole wherever one is. Holes that are left are filled if
    they are small.

    ``None`` when nothing in the mask is a road with two sides; see the
    module docstring.
    """
    height, width = road.shape
    scale = min(1.0, _WORK_SIDE / max(height, width))
    size = (max(8, round(width * scale)), max(8, round(height * scale)))
    small = cv2.resize((road > 0.5).astype(np.uint8), size, interpolation=cv2.INTER_AREA)
    left_over = _tidy(small)

    across = np.zeros(left_over.shape, dtype=np.float32)
    span = np.zeros(left_over.shape, dtype=np.float32)
    road_of = np.zeros(left_over.shape, dtype=np.int32)
    columns = np.arange(left_over.shape[1], dtype=np.float32)[None, :]
    lines = np.arange(left_over.shape[0], dtype=np.float32)[:, None]
    found = 0
    for _ in range(2 * _MOST_ROADS):
        if found == _MOST_ROADS:
            break
        count, pieces, stats, _centroids = cv2.connectedComponentsWithStats(
            left_over, connectivity=8
        )
        if count <= 1:
            break
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        if stats[largest, cv2.CC_STAT_AREA] < _LEAST_SHARE * left_over.size:
            break
        piece = (pieces == largest).astype(np.uint8)
        # By rows; and by columns if no side of it is met along a row.
        sideways = False
        read = _road_in(piece)
        if read is None or read[3] == 0:
            turned = _road_in(np.ascontiguousarray(piece.T))
            if turned is not None and turned[3] > 0:
                sideways, read = True, turned
        if read is None:
            # Not a road; and not to be looked at again.
            left_over[piece > 0] = 0
            continue
        body, first_side, second_side, _seen = read
        if sideways:
            body = body.T
            first, second, position = first_side[None, :], second_side[None, :], lines
        else:
            first, second, position = first_side[:, None], second_side[:, None], columns
        with np.errstate(invalid="ignore"):
            wide = np.maximum(second - first, _LEAST_WIDTH)
            share = (position - first) / wide
        found += 1
        across[body] = np.clip(share, 0.0, 1.0)[body]
        span[body] = np.broadcast_to(wide, body.shape)[body]
        road_of[body] = found
        left_over[body] = 0
        # And a rim round it, so that the next road is not the last one's kerb.
        left_over[cv2.dilate(body.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0] = 0
    if found == 0:
        return None

    # Along the road is square to the way its position across it changes.
    smooth = cv2.GaussianBlur(across, (0, 0), sigmaX=2.0)
    change_x = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    change_y = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    length = np.maximum(np.hypot(change_x, change_y), 1e-6)
    along = np.stack([-change_y / length, change_x / length], axis=2)
    along[along[..., 1] < 0.0] *= -1.0

    full = (width, height)
    weight = cv2.resize((road_of > 0).astype(np.float32), full, interpolation=cv2.INTER_LINEAR)
    return RoadFrame(
        weight=np.clip(weight, 0.0, 1.0).astype(np.float32),
        across=cv2.resize(across, full, interpolation=cv2.INTER_LINEAR).astype(np.float32),
        width_px=(cv2.resize(span, full, interpolation=cv2.INTER_LINEAR) / scale).astype(
            np.float32
        ),
        along=cv2.resize(along, full, interpolation=cv2.INTER_LINEAR).astype(np.float32),
        piece=cv2.resize(road_of, full, interpolation=cv2.INTER_NEAREST),
    )


# A lane is this many times as wide as the vehicles in it, and a vehicle's
# wheels are this share of a lane apart: 1.6 m of track in 3.5 m of lane.
_LANE_OVER_VEHICLE = 1.7
_WHEELS_APART = 0.44
# The narrowest and widest a lane may be, as a share of the road's width,
# and how much of the road's width on either side is verge and not lane.
_LANE_SHARE = (0.08, 0.9)
_VERGE = 0.05
# How far a track strays from its line, in lanes, and the least share of
# the traffic a lane carries.
_TRACK_WANDER = 0.1
_LEAST_LANE_USE = 0.35
# A vehicle this many times longer across the road than along it is being
# driven across it, and a road most of whose vehicles are is not running
# the way its frame says: a junction seen from above.
_CROSSWISE = 1.8
# Fewest pixels a vehicle covers to be counted, and how much wider than the
# rest one may be before it is two that touch.
_FEWEST_VEHICLE_PIXELS = 30
_WIDEST_VEHICLE = 0.6
# From this many vehicles on a road, the lanes are taken to end a lane's
# width past the outermost of them; with fewer they fill the road.
_ENOUGH_VEHICLES = 4
# Paint, counted by position across the road: the least of it that is a
# line, in pixels of length weighted by the road's width there; and the
# least share of the tallest peak another peak may be.
_LEAST_PAINT = 0.25
_PEAK_SHARE = 0.07
# How much of a lane has to fit between the last line and the verge for a
# lane to be put there.
_PART_OF_A_LANE = 0.6
# How much the vehicles' say counts against the paint's in placing lanes.
_VEHICLES_AGAINST_PAINT = 0.5
# The width of a lane, in metres.
_LANE_M = 3.4
# What each lane line costs a comb, against the share of the paint it explains.
_TOOTH_COST = 0.05
# How far from a whole number of lanes the gaps between painted lines may
# be, on average, as a share of a lane; and the narrowest a lane read off
# paint may be against the one the vehicles suggest.
_GAPS_FIT = 0.17
_NARROWEST_LANE = 0.6
# And the narrowest in any case, as a share of the road: eight lanes abreast.
_NARROWEST_PAINTED_LANE = 0.12


class Lanes(NamedTuple):
    """Where the lanes of each road in a :class:`RoadFrame` lie across it.

    The fields:

    * ``edges``: for each road, at the index :attr:`RoadFrame.piece` gives it
      (index 0 is unused), the positions across the road of its lane
      lines in order, in the units of :attr:`RoadFrame.across`: lane
      ``i`` runs from ``edges[road][i]`` to ``edges[road][i + 1]``. An
      empty array for a road with no lanes.
    * ``painted``: for each road, how many of those lines were read off paint
      on the road, the rest being placed between and beyond them.
    """

    edges: list[np.ndarray]
    painted: list[int]


def _comb(
    paint_at: np.ndarray,
    paint_weight: np.ndarray,
    vehicles_at: np.ndarray,
    lane_by_vehicles: float | None,
    lane_by_height: float | None = None,
) -> tuple[np.ndarray, int] | None:
    """The lane lines of one road as an even comb, and how many of them lie on paint.

    A road's lanes are one width, so its lane lines are a comb: a pitch and
    an offset. Every pitch and offset in reach is tried, and scored on
    both things that say where lanes are, at once:

    * **paint lies on the comb's teeth.** The score is the share of the
      road's paint that lies on a tooth, lowered by up to half for teeth
      with no paint on them. The first alone is best for a comb twice too
      fine, which has a tooth under everything; the second alone for one
      twice too coarse, all of whose teeth are painted.
    * **vehicles sit between the teeth**, in the middle of a lane.

    ``paint_at`` and ``vehicles_at`` are positions across the road,
    ``paint_weight`` the length of line each painted pixel counts for, and
    ``lane_by_vehicles`` the lane's width as the vehicles' own widths
    suggest it, which bounds the pitches tried. ``lane_by_height`` is the
    width of a lane of ordinary size where the camera's height is known,
    and bounds them more surely: a lane is three to four metres whatever
    is driving in it. ``None`` when there is neither paint nor a vehicle
    to go by.
    """
    bins = 400
    piled = np.bincount(
        np.clip((paint_at * bins).astype(np.int64), 0, bins - 1),
        weights=paint_weight,
        minlength=bins,
    ).astype(np.float32)
    has_paint = float(piled.sum()) >= _LEAST_PAINT
    if not has_paint and lane_by_vehicles is None and lane_by_height is None:
        return None
    # The paint within a hundredth of the road's width of each position.
    reach = 5
    summed = np.concatenate([[0.0], np.cumsum(piled, dtype=np.float64)])
    upper = np.minimum(np.arange(bins) + reach + 1, bins)
    lower = np.maximum(np.arange(bins) - reach, 0)
    nearby = summed[upper] - summed[lower]
    total = max(float(piled.sum()), 1e-9)
    low, high = 0.12, 0.6
    if lane_by_vehicles is not None:
        low, high = 0.8 * lane_by_vehicles, 1.25 * lane_by_vehicles
        if has_paint:
            # The paint says how wide a lane is, within what the vehicles
            # allow: a lane is neither narrower than what drives in it nor
            # more than three times as wide.
            low, high = 0.7 * lane_by_vehicles, 2.0 * lane_by_vehicles
    if lane_by_height is not None:
        low, high = 0.8 * lane_by_height, 1.25 * lane_by_height
    low, high = float(np.clip(low, *_LANE_SHARE)), float(np.clip(high, *_LANE_SHARE))
    best: tuple[float, float, float, int] | None = None
    for pitch in np.geomspace(low, max(high, low * 1.01), 40):
        for offset in np.linspace(0.0, pitch, 48, endpoint=False):
            teeth = np.arange(offset - pitch, 1.0 + pitch, pitch)
            teeth = teeth[(teeth > -0.25 * pitch) & (teeth < 1.0 + 0.25 * pitch)]
            score, on_paint = 0.0, 0
            if has_paint:
                at = np.clip(np.rint(teeth * bins).astype(np.int64), 0, bins - 1)
                mass = nearby[at]
                inside = (teeth > 0.04) & (teeth < 0.96)
                painted = mass > 0.2 * max(float(mass.max()), 1e-9)
                on_paint = int(painted.sum())
                covered = float(painted[inside].mean()) if inside.any() else 0.0
                explained = min(float(mass.sum()) / total, 1.0)
                # And every tooth has to earn its place: a comb fine enough
                # has a tooth under any paint there is.
                score += explained * (0.5 + 0.5 * covered) - _TOOTH_COST * float(inside.sum())
            if len(vehicles_at):
                in_lane = np.cos(2.0 * np.pi * (vehicles_at - offset) / pitch - np.pi)
                score += _VEHICLES_AGAINST_PAINT * float(in_lane.mean())
                if lane_by_vehicles is not None:
                    score -= 0.05 * abs(float(np.log(pitch / lane_by_vehicles)))
            if best is None or score > best[0]:
                best = (score, float(pitch), float(offset), on_paint)
    assert best is not None
    _score, lane, start, supported = best
    # Out to both verges: the last lane on either side is cut short by the
    # road's edge, and is a lane if more than half of one is left.
    lines = np.clip(np.arange(start - 2.0 * lane, 1.0 + 2.0 * lane, lane), 0.0, 1.0)
    lines = np.unique(lines)
    if len(lines) > 2 and lines[1] - lines[0] < 0.5 * lane:
        lines = lines[1:]
    if len(lines) > 2 and lines[-1] - lines[-2] < 0.5 * lane:
        lines = lines[:-1]
    return lines, supported if has_paint else 0


def _even_lanes(count: int) -> np.ndarray:
    """Lane lines for ``count`` lanes spread evenly between a road's verges."""
    return np.linspace(_VERGE, 1.0 - _VERGE, count + 1)


def road_lanes(
    frame: RoadFrame,
    vehicles: np.ndarray | None,
    rng: np.random.Generator,
    lane_count_range: tuple[int, int] = (2, 3),
    paint: np.ndarray | None = None,
    camera_height_m: float | None = None,
    guess: bool = True,
) -> Lanes:
    """Where the lanes are: one comb per road, set by its paint and its vehicles together.

    ``paint`` is the mask of painted lines (:func:`painted_lines`) and
    ``vehicles`` of the vehicles. Each vehicle's foot gives a position
    across the road and its width a size: a lane is some 1.7 times as wide
    as what drives in it, which bounds the lane widths tried. The lane
    lines are then the comb that the paint lies on and the vehicles sit
    between (:func:`_comb`). Where the camera's height is known and the
    road was read as a camera sees it (:class:`RoadModel`), the road's
    width is known in metres, and a lane is 3.4 of them, give or take a
    quarter: that bounds the comb before anything else does. A road with
    nothing to go by gets a number of lanes drawn from ``lane_count_range``,
    or none at all with ``guess`` off.

    A road whose vehicles lie across the way it was read gets none: it is
    a junction seen from above.
    """
    roads = int(frame.piece.max())
    edges: list[np.ndarray] = [np.zeros(0) for _ in range(roads + 1)]
    painted = [0] * (roads + 1)
    seen: dict[int, list[tuple[float, float]]] = {road: [] for road in range(1, roads + 1)}
    crosswise = np.zeros(roads + 1)
    if vehicles is not None:
        mask = (vehicles > 0.5).astype(np.uint8)
        count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        height, width = mask.shape
        for index in range(1, count):
            left, top, wide, tall, area = (int(value) for value in stats[index])
            if area < _FEWEST_VEHICLE_PIXELS:
                continue
            # Its foot: the middle of the bottom of its box, where it meets
            # the road. Its top leans with the perspective.
            row = min(top + tall - 1, height - 1)
            column = min(left + wide // 2, width - 1)
            road = int(frame.piece[row, column])
            if road == 0:
                continue
            # As wide as it is at its foot, and across the road, not across
            # the picture: the box round a vehicle seen at an angle takes in
            # its length, and its roof leans away from its wheels.
            foot_rows = slice(max(top, row - max(1, tall // 4)), row + 1)
            foot = mask[foot_rows, left : left + wide] > 0
            reach = frame.across[foot_rows, left : left + wide][foot]
            share = float(np.percentile(reach, 95.0) - np.percentile(reach, 5.0))
            run_x, run_y = (abs(float(value)) for value in frame.along[row, column])
            lies_across = (wide > _CROSSWISE * tall and run_y > run_x) or (
                tall > _CROSSWISE * wide and run_x > run_y
            )
            crosswise[road] += 1.0 if lies_across else -1.0
            if share < _WIDEST_VEHICLE:
                seen[road].append((float(frame.across[row, column]), share))
    for road in range(1, roads + 1):
        if crosswise[road] > 0.0:
            continue
        here = np.zeros(frame.piece.shape, dtype=bool)
        if paint is not None:
            here = paint & (frame.piece == road)
        where = np.array([position for position, _share in seen[road]])
        by_vehicles = None
        if len(where):
            # The narrower of them: two that touch, and one seen side on,
            # are both wider than a vehicle and never narrower.
            size = float(np.percentile([share for _position, share in seen[road]], 40.0))
            by_vehicles = float(np.clip(_LANE_OVER_VEHICLE * size, *_LANE_SHARE))
        # Paint counted by length along the road and not by pixels, or the
        # paint under the camera would be all of it.
        by_height = None
        heights_wide = frame.heights_wide
        if camera_height_m is not None and heights_wide is not None and heights_wide[road] > 0.0:
            by_height = _LANE_M / (camera_height_m * float(heights_wide[road]))
        comb = _comb(frame.across[here], 1.0 / frame.width_px[here], where, by_vehicles, by_height)
        if comb is None or len(comb[0]) < 2:
            count = int(rng.integers(lane_count_range[0], lane_count_range[1] + 1))
            if guess:
                edges[road] = _even_lanes(count)
            continue
        edges[road], painted[road] = comb
    return Lanes(edges, painted)


def wheel_tracks(
    frame: RoadFrame,
    lanes: Lanes,
    band: float,
    rng: np.random.Generator,
    ragged: np.ndarray | None = None,
    worn: np.ndarray | None = None,
    wear_floor: float = 0.25,
) -> np.ndarray:
    """``[0, 1]`` weight of the two wheel tracks down each lane, ``(H, W)`` float32.

    ``band`` is half the width of one track as a share of the lane: 0.05 is
    a rut a tyre wide, 0.2 two tracks that have all but met.

    Ruled lines of one width and one darkness are a pedestrian crossing and
    not the marks of traffic, so nothing about a track is even. Each lane
    carries its own share of the traffic, drawn from ``rng``, and some lanes
    are hardly driven in. ``ragged``, a field in ``[0, 1]`` laid on the
    ground, moves every track a little from side to side and varies its
    width; ``worn``, another, is how much has been driven over each stretch
    of road, so that a track is deep in one place and nearly gone in the
    next: down to ``wear_floor`` of itself, which is low for a road a few
    cars have been down and near one for a road worn bare. Without them the
    tracks are even along their length.

    A track's edge is soft, the snow thinning towards its middle, and as
    soft as the picture can show: where a lane is a few pixels wide, far
    down the road, the tracks run together into a road that is greyer all
    over, as they do to the eye.
    """
    shape = frame.across.shape
    turns = np.zeros(shape, dtype=np.float32)
    lane = np.ones(shape, dtype=np.float32)
    which = np.zeros(shape, dtype=np.int64)
    inside = np.zeros(shape, dtype=bool)
    for road in range(1, len(lanes.edges)):
        lines = lanes.edges[road]
        if len(lines) < 2:
            continue
        here = frame.piece == road
        position = frame.across[here]
        index = np.clip(np.searchsorted(lines, position) - 1, 0, len(lines) - 2)
        wide = (lines[index + 1] - lines[index]).astype(np.float32)
        turns[here] = (position - 0.5 * (lines[index] + lines[index + 1])) / wide
        lane[here] = wide
        which[here] = index + 17 * road
        inside[here] = (position >= lines[0]) & (position <= lines[-1])
    if ragged is not None:
        # A tenth of a lane to either side: nobody drives in a ruled line.
        turns = (turns + _TRACK_WANDER * (2.0 * ragged - 1.0)).astype(np.float32)
    from_wheel = np.abs(np.abs(turns) - 0.5 * _WHEELS_APART)
    half: np.ndarray = np.full(shape, band, dtype=np.float32)
    if ragged is not None:
        half = (half * (0.6 + 0.8 * ragged)).astype(np.float32)
    # A pixel's width, in lanes: nothing is drawn sharper than that.
    pixel = 1.0 / np.maximum(lane * frame.width_px, 1.0)
    reach = np.maximum(1.6 * half, 1.5 * pixel)
    edge = np.clip(1.0 - from_wheel / reach, 0.0, 1.0)
    track = edge * edge * (3.0 - 2.0 * edge)
    # Never more of a lane than the tracks are of it, however soft.
    track = track * np.clip(1.6 * half / reach, 0.0, 1.0)
    # How much each lane is driven in.
    use = rng.uniform(_LEAST_LANE_USE, 1.0, size=64).astype(np.float32)
    track = track * use[which % 64]
    if worn is not None:
        track = track * np.clip(wear_floor + (1.0 - wear_floor) * 2.0 * worn, 0.0, 1.0)
    return (track * inside * frame.weight).astype(np.float32)


# Most vehicles given spray in one frame, largest first, and the fewest
# pixels one covers to be given any.
_MOST_SPRAYING = 48
_FEWEST_SPRAY_PIXELS = 40
# The wake behind a vehicle: how many puffs it is drawn as; how long the
# cloud hangs, in seconds, which with the speed is its length (some twenty
# metres at motorway speed); the length of a car, by which that is turned
# into lengths of the outline; the longest it is drawn, in those; how fast
# it thins along its length, as the exponent over the whole of it; and how
# dense the tufts at the wheels are against the wake at its thickest.
_WAKE_PUFFS = 14
_WAKE_SECONDS = 0.75
_CAR_LENGTH_M = 4.5
_LONGEST_WAKE = 6.0
_WAKE_DECAY = 2.6
_TUFT = 0.9
# The tallest a vehicle is taken to be, in its own widths.
_TALLEST = 1.6
# Spray against speed: the speed at which the amount is 1, the power it
# goes by, the most it is taken to, and the least that is drawn at all.
_FULL_SPRAY_KMH = 110.0
_SPRAY_POWER = 2.8
_MOST_AMOUNT = 1.6
_LEAST_AMOUNT = 0.01
# Speeds. How far one road's mean lies from the frame's, as a standard
# deviation; the spread of the vehicles of one road about its mean, as a
# share of it, drawn between these; and what a lorry keeps to, a lorry
# being a vehicle this many times wider than the rest.
_ROADS_DIFFER = 0.08
_SPEED_SPREAD = (0.11, 0.18)
_LORRY_KMH = 90.0
_LORRY_GIRTH = 1.35
# What is nearer the camera than this share of the distance to the vehicle
# a puff trails from stands in front of the puff.
_STANDS_NEARER = 0.7
# An outline this many times wider than the rest is several vehicles, and
# its showing no red is not heard. And what a front seen counts for against
# a rear seen: a front is only red not seen, and a rear in shade shows none
# either.
_SEVERAL_VEHICLES = 1.5
_FRONT_COUNTS = 0.3
# A network's reading of a vehicle leans one way when it is this much surer
# of one kind of lamp than the other, and is believed over the vehicle's
# road when it is this much surer.
_VIEW_LEANS = 0.03
_VIEW_SURE = 0.15
# The most spray counted in front of one pixel, in plumes at their densest:
# ten cars abreast do not make ten times the mist of one.
_MOST_SPRAY = 2.5
# A vehicle's own heading is read off its outline when the outline is this
# many times longer than wide, and lies within this cosine of the road's.
_HEADING_ELONGATION = 1.3
_HEADING_NEAR_ROAD = 0.82
# A vehicle whose long way lies further than this cosine from the road's
# lies across it, and trails nothing. And the least share of the frame a
# road covers for its vehicles to be given spray at all.
_HEADING_ACROSS = 0.7
_LEAST_ROAD_FOR_SPRAY = 0.05
# A tail lamp's lens: red at this level, in linear light, and this many
# times its other channels, which is what keeps an amber indicator out: the
# front of a car has those, and they were read as its tail lamps.
_LENS_LEVEL = 0.05
_LENS_RED = 2.8
# A rear's lenses are between these shares of the vehicle's outline; below
# the third, on a vehicle of at least so many pixels, there are none and it
# is a front.
_LENS_SHARE = (0.004, 0.09)
_NO_LENS = 0.001
_SHOWS_LENSES = 250
# One lens is a patch of red of at most this share of the vehicle.
_LENS_PATCH = 0.04
# How much more of the evidence two-way traffic has to explain than
# one-way, as a share of all of it, to be believed.
_TWO_WAYS_EARNS = 0.05
# A headlamp: every channel at this level, in a frame darker than the second.
_HEAD_LAMP_LEVEL = 0.8
_NIGHT_FRAME = 0.08


def vehicle_facing(
    colour: np.ndarray,
    rows: np.ndarray,
    columns: np.ndarray,
    night: bool,
) -> float | None:
    """Which end of a vehicle its own picture shows: 1 its rear, -1 its front, ``None`` unknown.

    ``colour`` holds the vehicle's pixels in linear light and ``rows`` and
    ``columns`` where they are.

    * **A rear has red lenses**, lit or not: a few small patches of red,
      each a small part of the vehicle. One is enough, because the other
      is as often hidden, in shade or lost to the outline.
    * **A front has none.** A vehicle large enough to show lenses and
      showing no red at all is seen from the front.
    * **A red vehicle says nothing.** Its red is its bodywork, in one large
      patch or in the many the glass and the trim cut it into.
    * At night, white lamps burning are headlamps.

    This is weak evidence, a handful of pixels on each vehicle, and it is
    meant to be pooled over a road (:func:`traffic_directions`) and not
    believed one vehicle at a time. Unlit headlamps are not looked for:
    pale and colourless at the corners of the near end is also every white
    car seen from behind.
    """
    top, left = int(rows.min()), int(columns.min())
    size = len(rows)
    if size < 6:
        return None
    red = (colour[:, 0] > _LENS_LEVEL) & (colour[:, 0] > _LENS_RED * colour[:, 1:].max(axis=1))
    if night and int((colour.min(axis=1) > _HEAD_LAMP_LEVEL).sum()) >= 2 and not red.any():
        return -1.0
    if float(red.mean()) > _LENS_SHARE[1]:
        return None
    box = np.zeros((int(rows.max()) - top + 1, int(columns.max()) - left + 1), dtype=np.uint8)
    box[rows[red] - top, columns[red] - left] = 1
    count, _parts, stats, _centres = cv2.connectedComponentsWithStats(box, connectivity=8)
    areas = stats[1:count, cv2.CC_STAT_AREA]
    lenses = int(areas[(areas >= 2) & (areas <= max(4.0, _LENS_PATCH * size))].sum())
    if lenses >= max(3.0, _LENS_SHARE[0] * size):
        return 1.0
    if int(red.sum()) <= _NO_LENS * size and size >= _SHOWS_LENSES:
        return -1.0
    return None


def traffic_directions(
    at: np.ndarray, faces: np.ndarray, weight: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Which way each vehicle of one road is going, from what a few of them show.

    ``at`` is each vehicle's position across the road, ``faces`` what its
    own picture says of it (:func:`vehicle_facing`; 0 for nothing) and
    ``weight`` how much its word counts. The result is 1 for driving away
    from the camera and -1 for coming on.

    One vehicle's picture is poor evidence, and a road's traffic is not a
    set of independent choices. It all goes one way; or it goes one way on
    one side of the road and the other way on the other, and then what
    divides it is the widest strip across the road with no vehicle in it,
    which is where a reserve or a barrier would be. Those are the only
    answers considered, and the one chosen is the one the most evidence
    agrees with, two ways only when it explains more than one way does by
    a margin. With no evidence at all the road's one direction is drawn.

    The divide used to be put wherever suited the evidence best, between
    any two vehicles. Two rears in shade, showing no red, then cut a
    carriageway in two between its lanes.
    """
    said = faces != 0.0
    if not said.any():
        return np.full(len(at), float(rng.choice([-1.0, 1.0])))
    total = float(weight[said].sum())

    def explained(ways: np.ndarray) -> float:
        """The share of the evidence that agrees with these directions."""
        return float(weight[said & (faces == ways)].sum()) / total

    one_way = max((np.full(len(at), sign) for sign in (1.0, -1.0)), key=explained)
    order = np.sort(np.unique(at))
    if len(order) < 2:
        return one_way
    widest = int(np.argmax(np.diff(order)))
    divide = 0.5 * float(order[widest] + order[widest + 1])
    two_ways = max((np.where(at < divide, sign, -sign) for sign in (1.0, -1.0)), key=explained)
    if explained(two_ways) > explained(one_way) + _TWO_WAYS_EARNS:
        return two_ways
    return one_way


class RoadVehicle(NamedTuple):
    """One vehicle on a road, as far as its outline and its picture tell.

    The fields:

    * ``road``: which of the frame's roads it is on, counted from 1.
    * ``rows``, ``columns``: its pixels.
    * ``foot``: the row and the column where it stands on its road.
    * ``girth``: its width across the road, as a share of the road's.
    * ``shows``: what its own picture says of it
      (:func:`vehicle_facing`): 1 its rear, -1 its front, 0 nothing.
    * ``away``: which way it is taken to be going, once its road's traffic
      has been heard together: 1 away from the camera, its rear to it, and
      -1 coming on.
    * ``speed_kmh``: how fast it is taken to be going.
    * ``lamps``: where a network found the lamps of the end it shows
      (:class:`~surmentations.perception.vehicles.VehicleView`), one to a
      row as column, row, half width, half height and score; ``None`` where
      none was asked or it found the lamps of the other end.
    * ``alone``: whether its outline is one vehicle: told apart from its
      neighbours by a network, or no wider than the rest. An outline that
      is several vehicles run together has no one end to put lamps on.
    """

    road: int
    rows: np.ndarray
    columns: np.ndarray
    foot: tuple[int, int]
    girth: float
    shows: float
    away: float
    speed_kmh: float
    lamps: np.ndarray | None = None
    alone: bool = True


def spray_amount(speed_kmh: float) -> float:
    """How much spray a vehicle throws up at a speed, against what it does on a motorway.

    As the 2.8th power of the speed: the fit of the Road Research
    Laboratory's measurements of spray density behind cars and lorries
    (1965, as reported at the time; met in a summary and not checked at
    source). Half the speed is a seventh of the spray, and a queue throws
    up none.
    """
    return float(min((max(speed_kmh, 0.0) / _FULL_SPRAY_KMH) ** _SPRAY_POWER, _MOST_AMOUNT))


def _outlines(
    vehicles: np.ndarray, views: Sequence[VehicleView] | None
) -> list[tuple[np.ndarray, np.ndarray, VehicleView | None]]:
    """The vehicles of a mask, largest first: each one's rows, columns and view, if it has one.

    Where the views carry their own masks the vehicles are theirs, and
    what of the mask no view covers is its connected pieces, as all of it
    is with no views. A view without a mask is the connected piece that has
    its box.
    """
    mask = (vehicles > 0.5).astype(np.uint8)
    height, width = mask.shape
    found: list[tuple[np.ndarray, np.ndarray, VehicleView | None]] = []
    boxed: dict[tuple[int, ...], VehicleView] = {}
    for view in views or ():
        if view.mask is None:
            boxed[tuple(view.box)] = view
            continue
        left, top, _wide, _tall = view.box
        rows, columns = np.nonzero(view.mask)
        rows, columns = rows + top, columns + left
        inside = (rows >= 0) & (rows < height) & (columns >= 0) & (columns < width)
        rows, columns = rows[inside], columns[inside]
        inside = mask[rows, columns] > 0
        rows, columns = rows[inside], columns[inside]
        mask[rows, columns] = 0
        if len(rows) >= _FEWEST_SPRAY_PIXELS:
            found.append((rows, columns, view))
    count, parts, stats, _centres = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for index in range(1, count):
        left, top, wide, tall, area = (int(value) for value in stats[index])
        if area < _FEWEST_SPRAY_PIXELS:
            continue
        rows, columns = np.nonzero(parts[top : top + tall, left : left + wide] == index)
        found.append((rows + top, columns + left, boxed.get((left, top, wide, tall))))
    found.sort(key=lambda one: -len(one[0]))
    return found


def road_vehicles(
    frame: RoadFrame,
    vehicles: np.ndarray,
    rng: np.random.Generator,
    linear_rgb: np.ndarray | None = None,
    speed_kmh: float = 100.0,
    views: Sequence[VehicleView] | None = None,
) -> list[RoadVehicle]:
    """The vehicles on a frame's roads: where each stands, which way it goes and how fast.

    **Which road** a vehicle is on is the road most of the ground about its
    foot belongs to.

    **Which way** is a vote. Each vehicle's own picture says a little
    (:func:`vehicle_facing`, from ``linear_rgb``), and a road's traffic is
    heard together (:func:`traffic_directions`). A rear seen counts for
    more than a front, which is only red not seen; and an outline much
    wider than the rest is several vehicles run together or a lorry far
    off, whose lenses are under a pixel each, so its showing no red is not
    heard at all.

    With ``views`` (:mod:`surmentations.perception.vehicles`) a network has
    looked at each vehicle for its lamps, and what it found is heard in
    place of the colour: a front and a rear alike, each by how much surer
    the network is of one than of the other. A vehicle it is sure of keeps
    its own reading whatever its road says, which is what a junction needs,
    where the traffic of one road goes three ways; the rest go with their
    road. A vehicle the network did not read is read by colour as before.
    And where the views say which pixels are which vehicle, the vehicles
    are theirs and not the connected pieces of ``vehicles``: three cars
    nose to tail are three, each with its own wake and its own lamps, where
    a semantic mask makes one long vehicle of them. A vehicle known for one
    is never taken for several, and a lorry is known by its name and not
    by its width.

    **How fast** is drawn about ``speed_kmh``, the speed of the frame's
    traffic. Each road's own mean is a little either side of it, as two
    carriageways never quite agree; and each vehicle's speed is normal
    about its road's mean with a spread of 11 to 18 % of it, which is what
    free-flowing traffic is usually taken to be (Leong, 1968, as cited in
    reviews of speed distributions; not checked at source). A lorry keeps
    to its limit.
    """
    height, width = vehicles.shape
    night = linear_rgb is not None and float(linear_rgb.mean()) < _NIGHT_FRAME
    found: list[tuple[int, np.ndarray, np.ndarray, tuple[int, int], float, float]] = []
    read: list[VehicleView | None] = []
    for rows, columns, view in _outlines(vehicles, views)[:_MOST_SPRAYING]:
        left, top = int(columns.min()), int(rows.min())
        wide, tall = int(columns.max()) - left + 1, int(rows.max()) - top + 1
        # Its foot is the lowest of it, and its road is the one most of
        # the ground about its foot belongs to: the one pixel under the
        # middle of its box is as often a kerb, a shadow or another car.
        low = rows >= top + 0.6 * tall
        row = min(top + tall - 1, height - 1)
        column = min(int(np.median(columns[low])), width - 1)
        reach = max(2, tall // 4)
        first_row, first_column = max(0, row - reach), max(0, left - reach)
        about = frame.piece[first_row : row + reach + 1, first_column : left + wide + reach + 1]
        counts = np.bincount(about.ravel(), minlength=2)[1:]
        if int(counts.sum()) == 0:
            continue
        road = int(np.argmax(counts)) + 1
        if int(frame.piece[row, column]) != road:
            near_rows, near_columns = np.nonzero(about == road)
            off = np.hypot(near_rows + first_row - row, near_columns + first_column - column)
            row = first_row + int(near_rows[int(np.argmin(off))])
            column = first_column + int(near_columns[int(np.argmin(off))])
        offsets = np.stack([columns - columns.mean(), rows - rows.mean()], axis=1)
        way = frame.along[row, column].astype(np.float64)
        across = float(np.percentile(np.abs(offsets @ np.array([-way[1], way[0]])), 90.0))
        girth = across / max(float(frame.width_px[row, column]), 1.0)
        shows = None
        if (view is None or not view.read) and linear_rgb is not None:
            shows = vehicle_facing(linear_rgb[rows, columns], rows, columns, night)
        found.append((road, rows, columns, (row, column), girth, 0.0 if shows is None else shows))
        read.append(view)
    if not found:
        return []
    typical = float(np.median([girth for _road, _rows, _columns, _foot, girth, _shows in found]))
    traffic: list[RoadVehicle] = []
    for road in sorted({one[0] for one in found}):
        members = [index for index, one in enumerate(found) if one[0] == road]
        here = [found[index] for index in members]
        looked = [read[index] for index in members]
        at = np.array([float(frame.across[one[3]]) for one in here])
        girths = np.array([one[4] for one in here])
        said = np.array([one[5] for one in here])
        # Told apart by a network, a vehicle is one vehicle however wide.
        one_vehicle = np.array([view is not None and view.kind != "" for view in looked])
        several = (girths > _SEVERAL_VEHICLES * typical) & ~one_vehicle
        said[(said < 0.0) & several] = 0.0
        weight = np.array([float(np.sqrt(len(one[1]))) for one in here])
        weight = weight * np.where(said < 0.0, _FRONT_COUNTS, 1.0)
        sure = np.zeros(len(here), dtype=bool)
        for index, view in enumerate(looked):
            if view is None or not view.read:
                continue
            # A network's reading: which lamp it is surer of, and by how much.
            lean = abs(view.facing)
            said[index] = 0.0 if lean < _VIEW_LEANS else float(np.sign(view.facing))
            weight[index] = float(np.sqrt(len(here[index][1]))) * min(1.0, lean / _VIEW_SURE)
            sure[index] = lean >= _VIEW_SURE and not several[index]
        ways = traffic_directions(at, said, weight, rng)
        ways = np.where(sure, said, ways)
        mean = speed_kmh * float(np.clip(rng.normal(1.0, _ROADS_DIFFER), 0.7, 1.3))
        spread = float(rng.uniform(*_SPEED_SPREAD)) * mean
        speeds = np.clip(rng.normal(mean, spread, size=len(here)), 0.0, None)
        # A lorry by its name, where a network has named the vehicle, and
        # by being wider than the rest where none has.
        named = np.array([view.kind if view is not None else "" for view in looked])
        heavy = np.where(
            named != "", np.isin(named, sorted(HEAVY_KINDS)), girths > _LORRY_GIRTH * typical
        )
        speeds = np.where(heavy, np.minimum(speeds, _LORRY_KMH), speeds)
        for one, view, shown, away, speed in zip(here, looked, said, ways, speeds, strict=True):
            # Its lamps as found, where the end they are on is the end it
            # is taken to show.
            lamps = None
            if view is not None and len(view.lamps) and view.facing * away > 0.0:
                lamps = view.lamps
            alone = bool(view is not None and view.kind != "") or one[4] <= (
                _SEVERAL_VEHICLES * typical
            )
            traffic.append(
                RoadVehicle(
                    one[0],
                    one[1],
                    one[2],
                    one[3],
                    one[4],
                    float(shown),
                    float(away),
                    speed,
                    lamps,
                    alone,
                )
            )
    return traffic


def _vehicle_lie(
    frame: RoadFrame, vehicle: RoadVehicle
) -> tuple[tuple[float, float], np.ndarray, np.ndarray] | None:
    """How a vehicle lies on its road: its centre, its pixels about it, and its heading.

    The heading is the road's direction at its foot, turned to the long way
    of its own outline where that is long and close to the road's: a car
    changing lanes or taking a bend points along its own line. It is given
    in the road's sense, towards the camera, whichever way the vehicle is
    going. ``None`` for a vehicle whose long way lies across the road: it
    is crossing, turning or parked, or the road was read wrong there, and
    it is no car at speed.
    """
    row, column = vehicle.foot
    along_road = frame.along[row, column].astype(np.float64)
    centre = (float(vehicle.columns.mean()), float(vehicle.rows.mean()))
    offsets = np.stack([vehicle.columns - centre[0], vehicle.rows - centre[1]], axis=1)
    values, vectors = np.linalg.eigh(np.cov(offsets.T) + 1e-6 * np.eye(2))
    own = vectors[:, 1] * (1.0 if float(vectors[:, 1] @ along_road) >= 0.0 else -1.0)
    long_way = float(np.sqrt(values[1] / max(values[0], 1e-6)))
    agrees = float(own @ along_road)
    if long_way > _HEADING_ELONGATION and agrees < _HEADING_ACROSS:
        return None
    turned = long_way > _HEADING_ELONGATION and agrees > _HEADING_NEAR_ROAD
    return centre, offsets, own if turned else along_road


def vehicle_spray(
    frame: RoadFrame,
    vehicles: np.ndarray,
    rng: np.random.Generator,
    linear_rgb: np.ndarray | None = None,
    traffic: list[RoadVehicle] | None = None,
    depth_m: np.ndarray | None = None,
) -> np.ndarray:
    """How much spray hangs in front of each pixel, ``(H, W)`` float32, 0 and up.

    A vehicle at speed on a wet road lifts the water off it and trails it
    as a low cloud that lies on the road behind it: two dense tufts at the
    rear wheels, and a wake of fine spray as wide as the vehicle, then a
    little wider, as tall as the vehicle a length back, and thinning out
    over some lengths.

    **Where the wake lies is the vehicle's own heading.** Its outline in
    the picture is longer one way than the other, and where that way is
    near the road's it is the vehicle's heading: a car changing lanes or
    taking a bend trails its spray along its own line and not the road's.
    A vehicle too small or too square to say is given the road's direction.

    **Which end is the tail, and how fast it is going,** come with
    ``traffic`` (:func:`road_vehicles`, which is called with ``linear_rgb``
    when it is not given). The wake's amount goes with the speed
    (:func:`spray_amount`) and so does its length: the cloud hangs for
    under a second, and is as long as the vehicle goes in that time.

    **Some vehicles get none.** One whose long way lies across the road is
    crossing it, turning or parked, or the road was misread there. One on a
    scrap of road, a slip road cut by the frame's edge, is on a road whose
    direction is a guess. Both used to shoot a jet of mist out sideways.

    **It is a thing in the scene**, and what stands nearer the camera than
    a part of it hides that part. The wake of a vehicle coming on is behind
    the vehicle. The wake of one driving away lies between it and the
    camera, over its wheels and its bumper, and over whatever follows it
    further up the road than the wake reaches; a follower nearer than that
    stands in front of it. Every part of a wake is cut by the outlines of
    the vehicles that stand below it in the picture and by no others. It
    used to be cut by every outline there was, which left each car in a
    cloud standing in a clear hole with a hard edge. With ``depth_m`` the
    rest of the scene hides it too: a branch, a post or a parapet much
    nearer the camera than the vehicle the wake trails from.

    The result is a density, to be turned into an opacity by whatever knows
    how wet the road is: ``1 - exp(-strength * density)``.
    """
    height, width = vehicles.shape
    if traffic is None:
        traffic = road_vehicles(frame, vehicles, rng, linear_rgb)
    spray = np.zeros((height, width), dtype=np.float32)
    if not traffic:
        return spray
    mask = (vehicles > 0.5).astype(np.uint8)
    _count, parts, stats, _centres = cv2.connectedComponentsWithStats(mask, connectivity=8)
    # Where each outline stands: the row of its foot, on every pixel of it
    # and a little way round it, so that the feathered edge knows whose it
    # is. A puff is hidden by an outline only where that row is below its
    # own place on the road.
    feather = max(1.5, 0.006 * float(np.hypot(height, width)))
    hidden = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigmaX=feather)
    bottoms = (stats[:, cv2.CC_STAT_TOP] + stats[:, cv2.CC_STAT_HEIGHT] - 1).astype(np.float32)
    bottoms[0] = -1.0
    standing = bottoms[parts]
    # Each vehicle of the traffic by its own foot, where a network has told
    # it from the ones its outline runs into.
    for vehicle in traffic:
        standing[vehicle.rows, vehicle.columns] = float(vehicle.rows.max())
    round_it = 2 * int(2.0 * feather) + 1
    stands = cv2.dilate(standing, np.ones((round_it, round_it), np.uint8))
    roads = int(frame.piece.max())
    share = np.bincount(frame.piece.ravel(), minlength=roads + 1) / frame.piece.size

    def puff(
        at: tuple[float, float],
        axes: tuple[np.ndarray, np.ndarray],
        radii: tuple[float, float],
        strength: float,
        ground: float,
        gate: tuple[float, float, np.ndarray, float] | None = None,
        reach_m: float = 0.0,
    ) -> None:
        """Add one soft puff: a Gaussian with its own two axes and radii.

        ``ground`` is the row of the road it hangs over, which decides what
        vehicles hide it, and ``reach_m`` how far off its vehicle is, which
        decides what else does: where the road under a puff is itself
        hidden, the depth there is that of what hides it. ``gate`` cuts it
        off behind a line, softly: a point, the direction past it that is
        kept, and how far the cut is feathered.
        """
        reach = int(3.0 * max(radii)) + 1
        x0, x1 = max(0, int(at[0]) - reach), min(width, int(at[0]) + reach + 1)
        y0, y1 = max(0, int(at[1]) - reach), min(height, int(at[1]) + reach + 1)
        if x0 >= x1 or y0 >= y1:
            return
        dx = np.arange(x0, x1, dtype=np.float32)[None, :] - np.float32(at[0])
        dy = np.arange(y0, y1, dtype=np.float32)[:, None] - np.float32(at[1])
        first = (dx * axes[0][0] + dy * axes[0][1]) / max(radii[0], 1.0)
        second = (dx * axes[1][0] + dy * axes[1][1]) / max(radii[1], 1.0)
        blob = np.exp(-0.5 * (first**2 + second**2))
        if gate is not None:
            gate_x, gate_y, kept, feathered = gate
            past = (np.arange(y0, y1, dtype=np.float32)[:, None] - gate_y) * kept[1] + (
                np.arange(x0, x1, dtype=np.float32)[None, :] - gate_x
            ) * kept[0]
            start = np.clip(past / max(feathered, 1.0), 0.0, 1.0)
            blob = blob * start * start * (3.0 - 2.0 * start)
        in_front = stands[y0:y1, x0:x1] > ground
        blob = blob * (1.0 - hidden[y0:y1, x0:x1] * in_front)
        if depth_m is not None:
            clear = (depth_m[y0:y1, x0:x1] > _STANDS_NEARER * reach_m) | (mask[y0:y1, x0:x1] > 0)
            blob = blob * clear
        spray[y0:y1, x0:x1] += np.float32(strength) * blob

    # A lorry is known by being wider than the rest.
    typical = float(np.median([vehicle.girth for vehicle in traffic]))

    for vehicle in traffic:
        # Not on a scrap of road, whose direction is a guess: a slip road
        # cut by the frame's edge, the far carriageway behind a barrier.
        if share[vehicle.road] < _LEAST_ROAD_FOR_SPRAY:
            continue
        amount = spray_amount(vehicle.speed_kmh)
        if amount < _LEAST_AMOUNT:
            continue
        rows = vehicle.rows
        row, column = vehicle.foot
        tall = int(rows.max()) - int(rows.min()) + 1
        foot = float(rows.max())
        far = 0.0 if depth_m is None else float(depth_m[row, column])
        lie = _vehicle_lie(frame, vehicle)
        if lie is None:
            continue
        centre, offsets, heading = lie
        way = heading * vehicle.away
        side = np.array([-way[1], way[0]])
        half_wide = max(1.5, float(np.percentile(np.abs(offsets @ side), 90.0)))
        long_px = max(3.0, 2.0 * float(np.percentile(np.abs(offsets @ way), 90.0)))
        towards = way[1] > 0.0
        # What it throws up goes with its size: a lorry's wake is several
        # cars' worth, and as tall as the lorry.
        size = float(np.clip(vehicle.girth / max(typical, 1e-6), 0.7, 2.0)) ** 2
        # Where it all begins: at the end of the outline it trails from.
        # That is the vehicle's foot when it is driving away, and behind its
        # own roof, part of the way to the far end, when it is coming on.
        reach_along = float((offsets @ way).max())
        tail_x = float(centre[0] + way[0] * reach_along * (1.0 if towards else 0.6))
        tail_y = float(centre[1] + way[1] * reach_along * (1.0 if towards else 0.6))
        gate = (tail_x, tail_y, way, 0.35 * long_px)
        wide_here = max(float(frame.width_px[row, column]), 1.0)
        # As tall as its vehicle, which an outline of several run together
        # is not: no taller than a lorry is against its width.
        high = min(float(tall), _TALLEST * 2.0 * half_wide)

        # The wake. Fine droplets, off the tread and out of the film
        # between the tyre and the road, are carried along in the air the
        # vehicle drags behind it: a cloud as long as the vehicle goes in
        # the time it hangs, as wide as the vehicle and widening, as tall
        # as the vehicle a length back, and thinning all the way.
        # Turbulent, so no two puffs of it are alike.
        lengths = vehicle.speed_kmh / 3.6 * _WAKE_SECONDS / _CAR_LENGTH_M
        length = float(np.clip(lengths, 0.3, _LONGEST_WAKE)) * long_px
        for step in range(_WAKE_PUFFS):
            run = (step + 0.5) / _WAKE_PUFFS
            x = tail_x + way[0] * length * run
            y = tail_y + way[1] * length * run
            if not (0 <= int(y) < height and 0 <= int(x) < width):
                continue
            # Further up the road everything is smaller: by the road's own
            # width there, which is the perspective.
            scale = float(np.clip(frame.width_px[int(y), int(x)] / wide_here, 0.15, 2.5))
            if frame.piece[int(y), int(x)] != vehicle.road:
                # Off its road, and not over a vehicle on it. Up the frame
                # that is the road's far end or its horizon, and a wake
                # that ran on past it stood in the sky; down it, the road
                # has turned away or is hidden, and a wake that ran on lay
                # across the verge.
                if mask[int(y), int(x)] == 0:
                    break
                scale = 1.0 if towards else max(0.15, 1.0 - 0.8 * run)
            across = (0.9 + 0.9 * run) * half_wide * scale
            lengthways = max(1.5, 0.9 * length / _WAKE_PUFFS * scale)
            rise = min(1.0, 0.35 + 1.3 * run) * high * scale * min(size, 1.6)
            sway = float(rng.normal(0.0, 0.25)) * half_wide * scale
            gust = float(rng.uniform(0.55, 1.45))
            thin = np.exp(-_WAKE_DECAY * run) * gust * amount * size
            # Trailing towards the camera it is this side of its own
            # vehicle, whatever the outline's lowest row says.
            ground = max(y, foot + 1.0) if towards else y
            for lift, share_up in ((0.0, 1.0), (0.45, 0.6), (0.9, 0.3)):
                at = (x + side[0] * sway, y + side[1] * sway - lift * rise)
                puff(at, (way, side), (lengthways, across), thin * share_up, ground, gate, far)

        # The tufts. Coarse drops are flung off the back of each tyre, and
        # the air stops them within a metre or two and below half a metre:
        # two small dense knots at the rear wheels, which is all the spray
        # there is at low speed and the brightest of it at any.
        if towards:
            for wheel in (-0.7, 0.7):
                at = (
                    tail_x + side[0] * wheel * half_wide + way[0] * 0.12 * long_px,
                    tail_y + side[1] * wheel * half_wide + way[1] * 0.12 * long_px - 0.08 * tall,
                )
                radii = (0.2 * long_px, 0.42 * half_wide)
                puff(at, (way, side), radii, _TUFT * min(1.0, amount**0.5), foot + 1.0, None, far)
    return np.minimum(spray, np.float32(_MOST_SPRAY))


def _placed_lamps(
    vehicle: RoadVehicle,
    colour: np.ndarray | None,
    shape: tuple[float, float],
    alone: bool,
) -> list[tuple[float, float, tuple[float, float]]]:
    """Where a vehicle's two lamps are put when no network has found them.

    Each as its column, its row and its size across and up the picture.
    The end that is seen is the lower part of the outline, and its two
    lamps are at one height on that, either side: a line across a road is
    level in the picture, near enough. So they are placed by the picture's
    rows and columns, and not along the long way of the outline, which for
    a box seen from above runs corner to corner. Each is at its own height
    above the outline's lower edge, which is the slant a vehicle off to one
    side is seen with.

    ``colour`` is the vehicle's pixels in linear light where it shows its
    rear: red lenses that can be seen are lit where they are. A vehicle
    seen from the side, and one that is not ``alone`` in its outline, has
    no near end to put a pair on, and only lenses that are seen are lit on
    it. ``shape`` is a lamp's width and height against its size.
    """
    rows, columns = vehicle.rows, vehicle.columns
    top, bottom = int(rows.min()), int(rows.max())
    tall = bottom - top + 1
    level = bottom - _LAMP_UP * tall
    band = np.abs(rows - level) <= max(1.0, 0.06 * tall)
    if int(band.sum()) < 4:
        return []
    left, right = (float(value) for value in np.percentile(columns[band], [4.0, 96.0]))
    half_wide = max(1.5, 0.5 * (right - left))
    size = max(0.6, _LAMP_SIZE * half_wide)
    radii = (shape[0] * size, shape[1] * size)
    lenses = None
    if colour is not None:
        lenses = (colour[:, 0] > _LENS_LEVEL) & (
            colour[:, 0] > _LENS_RED * colour[:, 1:].max(axis=1)
        )
        if float(lenses.mean()) > _LENS_SHARE[1]:
            lenses = None
    broad = float(np.ptp(np.percentile(columns, [2.0, 98.0]))) / max(tall, 1)
    placed = alone and broad < _END_ON
    middle = 0.5 * (left + right)
    places = []
    for which in (-1.0, 1.0):
        x = middle + which * (1.0 - 2.0 * _LAMP_INSET) * half_wide
        below = np.abs(columns - x) <= max(1.0, 0.16 * half_wide)
        y = (float(rows[below].max()) if below.any() else float(bottom)) - _LAMP_UP * tall
        found = placed
        if lenses is not None:
            seen = (
                lenses & ((columns - middle) * which > 0.3 * half_wide) & (rows > top + 0.3 * tall)
            )
            if int(seen.sum()) >= 2:
                x, y, found = float(columns[seen].mean()), float(rows[seen].mean()), True
        if found:
            places.append((x, y, radii))
    return places


# Lamps. The smallest vehicle given any, in pixels. Where a lamp is on the
# end of the outline nearer the camera: this share of the outline's height
# up from its lower edge, and this share of its width in from either side.
# And how many times wider than tall an outline may be and still be a
# vehicle seen end on: a car seen from the side is three times as long as
# it is tall.
_FEWEST_LAMP_PIXELS = 60
_LAMP_UP = 0.27
_LAMP_INSET = 0.14
_END_ON = 1.6
# A lamp's size, as a share of the vehicle's half width, and the glow round
# it in the wet air as a multiple of that and as a share of its light.
_LAMP_SIZE = 0.06
# A lamp a network found is lit over this share of the box it was found in.
_FOUND_SIZE = 0.5
# And no larger than this share of the smaller side of its vehicle's outline.
_LARGEST_LAMP = 0.06
# One lamp found on a vehicle seen end on has its fellow put across from it
# when the two would be at least this share of the vehicle's width apart.
_PAIR_APART = 0.35
_GLOW_SIZE = 5.0
_GLOW_SHARE = 0.025
# How far over full scale a dipped headlamp and a tail lamp are at their
# middles, at a strength of 1; how much brighter a brake light is and how
# often one is on; and the share of fronts and of rears lit by day.
_HEAD_PEAK = 6.0
_TAIL_PEAK = 2.2
_BRAKE = (2.5, 0.12)
_LIT_BY_DAY = (0.9, 0.75)
# Lamps the picture already shows burning are at most this share of their
# vehicle, which is otherwise darker than this at its median.
_BURNING_SHARE = 0.03
_DARK = 0.08
# The colours: a halogen headlamp, an LED one, and a tail lamp.
_HALOGEN = (1.0, 0.92, 0.78)
_LED = (0.9, 0.95, 1.0)
_TAIL = (1.0, 0.05, 0.03)


def vehicle_lamps(
    frame: RoadFrame,
    traffic: list[RoadVehicle],
    rng: np.random.Generator,
    strength: float,
    linear_rgb: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The lamps of a road's traffic, lit: their light, their glow, and where they are.

    Traffic in rain drives with its lamps on, and a photograph taken in the
    dry has them off. Each vehicle of ``traffic`` (:func:`road_vehicles`)
    is given the pair that the end of it nearer the camera carries: two
    headlamps where it is coming on, two tail lamps where it is driving
    away, which is the same reading that lays its spray.

    **Where.** Where a network found them, for a vehicle that comes with
    its ``lamps`` (:mod:`surmentations.perception.vehicles`): as many as it
    found, each as large as it found it; and where it found one of a pair
    on a vehicle seen end on, the other across from it. Otherwise they are
    placed. The
    end that is seen is the lower part of the outline, and the pair is at
    one height on it, a little in from either side and a quarter of the way
    up. A rear whose red lenses can be seen in the picture
    (``linear_rgb``) has its lamps lit where they are. A vehicle seen from
    the side, and an outline that is several vehicles, has no such end, and
    only lenses that are seen are lit on it.

    **How bright.** Far over full scale, as a lamp is: a dipped headlamp
    several times it and a tail lamp about twice, at a ``strength`` of 1,
    each a little more or less. One rear in eight is braking. By day not
    every vehicle is lit, and fewer rears than fronts, daytime running
    lamps being at the front only; by night every one is, except those
    whose lamps the picture already shows burning.

    Three things come back, each the size of the frame: the lamps' own
    light, linear RGB over the scene's; the glow each throws into wet air,
    to be added in proportion to how much of that there is in front of it;
    and a ``[0, 1]`` weight of where the lamps are, which is what a wet
    road needs to reflect them as lamps and not as paint.
    """
    height, width = frame.piece.shape
    light = np.zeros((height, width, 3), dtype=np.float32)
    glow = np.zeros((height, width, 3), dtype=np.float32)
    cores = np.zeros((height, width), dtype=np.float32)
    night = linear_rgb is not None and float(linear_rgb.mean()) < _NIGHT_FRAME
    whose = np.zeros((height, width), dtype=np.int32)
    for number, vehicle in enumerate(traffic, start=1):
        whose[vehicle.rows, vehicle.columns] = number

    def spot(
        target: np.ndarray,
        at: tuple[float, float],
        radii: tuple[float, float],
        value: np.ndarray | float,
    ) -> None:
        """Add one lamp-shaped spot of ``value`` to an image, or raise a weight to it.

        ``radii`` are its size across the picture and up it.
        """
        reach = int(3.0 * max(radii)) + 1
        x0, x1 = max(0, int(at[0]) - reach), min(width, int(at[0]) + reach + 1)
        y0, y1 = max(0, int(at[1]) - reach), min(height, int(at[1]) + reach + 1)
        if x0 >= x1 or y0 >= y1:
            return
        dx = np.arange(x0, x1, dtype=np.float32)[None, :] - np.float32(at[0])
        dy = np.arange(y0, y1, dtype=np.float32)[:, None] - np.float32(at[1])
        blob = np.exp(-0.5 * ((dx / radii[0]) ** 2 + (dy / radii[1]) ** 2))
        if target.ndim == 3:
            target[y0:y1, x0:x1] += blob[..., None] * np.asarray(value, dtype=np.float32)
        else:
            np.maximum(target[y0:y1, x0:x1], blob * np.float32(value), out=target[y0:y1, x0:x1])

    for number, vehicle in enumerate(traffic, start=1):
        rows, columns = vehicle.rows, vehicle.columns
        if len(rows) < _FEWEST_LAMP_PIXELS:
            continue
        lie = _vehicle_lie(frame, vehicle)
        if lie is None:
            continue
        rear = vehicle.away > 0.0
        colour = None if linear_rgb is None else linear_rgb[rows, columns]
        if night and colour is not None:
            # Burning already: a few pixels at full scale on a vehicle that
            # is otherwise dark. A pale car under a dull sky has many, and
            # is not dark.
            level = colour.max(axis=1)
            burning = int((level > _HEAD_LAMP_LEVEL).sum())
            if 2 <= burning <= _BURNING_SHARE * len(rows) and float(np.median(level)) < _DARK:
                continue
        if not night and float(rng.uniform()) > _LIT_BY_DAY[1 if rear else 0]:
            continue
        if rear:
            tint = np.array(_TAIL, dtype=np.float32)
            peak = _TAIL_PEAK * (_BRAKE[0] if float(rng.uniform()) < _BRAKE[1] else 1.0)
            shape = (1.3, 0.7)
        else:
            tint = np.array(_LED if float(rng.uniform()) < 0.5 else _HALOGEN, dtype=np.float32)
            peak = _HEAD_PEAK
            shape = (1.0, 1.0)
        peak = strength * peak * float(rng.uniform(0.7, 1.3))
        places: list[tuple[float, float, tuple[float, float]]] = []
        if vehicle.lamps is not None:
            # Found by a network that looked for them: lit where they are
            # and as large as they are, one where only one is seen.
            # No larger than a lamp is against its vehicle: in a far
            # outline of several cars the box found is half the outline.
            most = _LARGEST_LAMP * float(min(np.ptp(vehicle.columns), np.ptp(vehicle.rows)) + 1)
            for x, y, half_wide, half_tall, _score in vehicle.lamps.tolist():
                radii = (
                    float(np.clip(_FOUND_SIZE * half_wide, 0.6, max(0.6, most))),
                    float(np.clip(_FOUND_SIZE * half_tall, 0.6, max(0.6, most))),
                )
                places.append((x, y, radii))
            if len(places) == 1 and vehicle.alone:
                # One of a pair, on a vehicle seen end on: the other is at
                # the same height the same way in from the other side, and
                # is lit there where that is still the vehicle.
                x, y, radii = places[0]
                level = np.abs(rows - y) <= max(1.0, 0.06 * float(np.ptp(rows) + 1))
                broad = float(np.ptp(np.percentile(columns, [2.0, 98.0]))) / float(np.ptp(rows) + 1)
                if int(level.sum()) >= 4 and broad < _END_ON:
                    left, right = (float(v) for v in np.percentile(columns[level], [4.0, 96.0]))
                    other = left + right - x
                    apart = abs(other - x) >= _PAIR_APART * (right - left)
                    if apart and 0 <= int(other) < width and whose[int(y), int(other)] == number:
                        places.append((other, y, radii))
        else:
            places = _placed_lamps(vehicle, colour if rear else None, shape, vehicle.alone)
            places = [
                (x, y, radii)
                for x, y, radii in places
                if 0 <= int(y) < height and 0 <= int(x) < width and whose[int(y), int(x)] == number
            ]
        for x, y, radii in places:
            spot(light, (x, y), radii, tint * peak)
            wide = (_GLOW_SIZE * radii[0], _GLOW_SIZE * radii[1])
            spot(glow, (x, y), wide, tint * peak * _GLOW_SHARE)
            spot(cores, (x, y), radii, min(strength, 1.0) * (0.5 if rear else 1.0))
    return light, glow, cores


# Painted lines. A line is a thin thing: it does not survive an opening by a
# disc this fraction of the diagonal across, and stands at least this much
# above what the opening leaves, as a ratio of gamma-encoded luminance; it
# is brighter than the middle of the road by the second ratio; and it is
# long and lies along the road, at least this many times longer than wide,
# within this cosine of the road's own direction, and at least this many
# pixels in all, and this fraction of the diagonal long.
_LINE_WIDTH = 0.012
_LINE_STANDS_OUT = 0.22
_LINE_OVER_ROAD = 1.12
_LINE_ELONGATION = 3.0
_LINE_ALONG = 0.75
_LINE_FEWEST_PIXELS = 10
# The most of a road that may be thin and bright and still be paint.
_MOST_PAINT = 0.15
_LINE_LENGTH = 0.012


def painted_lines(
    linear_rgb: np.ndarray, frame: RoadFrame, road: np.ndarray, on_road: np.ndarray | None
) -> np.ndarray:
    """Which pixels of a road are paint that runs along it, ``(H, W)`` bool.

    Lane lines and edge lines: thin, brighter than the road, long, and lying
    the way the road runs. The last is what the road's own direction is
    for, and it is what tells a lane line from a zebra crossing, a painted
    arrow, a word, a kerb seen end on or a patch of light, all of which are
    bright and some of which are thin.

    ``road`` is the mask of the carriageway itself and ``on_road`` of what
    stands on it; paint is looked for on the first and not near the second,
    whose pale roofs are thin and bright as well.
    """
    height, width = road.shape
    where = (road > 0.5) & (frame.weight > 0.5)
    if on_road is not None:
        near = cv2.dilate((on_road > 0.5).astype(np.uint8), np.ones((5, 5), np.uint8))
        where &= near == 0
    parts, count = _thin_bright(linear_rgb, where)
    thin = parts > 0
    stats = np.zeros((count, 5), dtype=np.int64)
    stats[:, cv2.CC_STAT_AREA] = np.bincount(parts.ravel(), minlength=count)
    keep = np.zeros(count, dtype=bool)
    rows, columns = np.nonzero(thin)
    part = parts[rows, columns]
    order = np.argsort(part, kind="stable")
    rows, columns, part = rows[order], columns[order], part[order]
    starts = np.searchsorted(part, np.arange(1, count + 1))
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] < _LINE_FEWEST_PIXELS:
            continue
        ys = rows[starts[index - 1] : starts[index]].astype(np.float64)
        xs = columns[starts[index - 1] : starts[index]].astype(np.float64)
        spread = np.cov(np.stack([xs, ys]))
        values, vectors = np.linalg.eigh(spread)
        long_enough = 4.0 * float(np.sqrt(values[1])) >= _LINE_LENGTH * float(
            np.hypot(height, width)
        )
        if not long_enough or values[1] < _LINE_ELONGATION**2 * max(values[0], 0.08):
            continue
        way = frame.along[int(ys.mean()), int(xs.mean())]
        if abs(float(vectors[0, 1] * way[0] + vectors[1, 1] * way[1])) >= _LINE_ALONG:
            keep[index] = True
    lines: np.ndarray = keep[parts]
    return lines


# The road as a camera sees it. How much a pixel of paint, of the road's
# edge and of a vehicle's outline counts for as a cue to its direction; the
# fewest cues a road is fitted to; the angle, in degrees, at which a cue
# agrees with a road, the share of the cues, by weight, that has to agree
# for there to be a road at all, and how many pairs of cues a road is
# looked for through; how far below the horizon, as a share of the frame's
# height, a pixel has to be to be given a place on the road; the rows above
# the top of the road, as shares of the frame's height, at which the
# horizon is looked for; and how much of a position's length has to be road
# for it to be on a carriageway.
_CUE_PAINT = 1.0
_CUE_EDGE = 0.5
_CUE_VEHICLE = 0.6
_FEWEST_CUES = 6
_CUE_TOLERANCE_DEG = 5.0
_CUES_AGREE = 0.3
_CUES_AGREE_KNOWN_HORIZON = 0.2
_PAIRS_TRIED = 200
_BEND_EARNS = 0.1
_BELOW_HORIZON = 0.02
_HORIZON_LADDER = (0.01, 0.03, 0.06, 0.1, 0.16, 0.25, 0.4, 0.6, 0.9, 1.4, 2.2)
_CARRIAGEWAY_SOLID = 0.5

# Fewest painted lines a vanishing point is found from; how far, in degrees,
# the middle one of them may point from it; and how unlike in direction
# they have to be for their meeting point to be worth having, as the
# smaller share of the scatter of their normals.
_FEWEST_LINES = 3
_POINTS_AT_IT = 2.0
_LEAST_SPREAD = 0.004
# How far below the point its lines run to a pixel of road has to be to be
# given a line through it, as a share of the road's height in the frame.
_BELOW_THE_POINT = 0.03


def straighten(frame: RoadFrame, paint: np.ndarray) -> RoadFrame:
    """A road's frame, redrawn through the point its painted lines run to.

    :func:`road_frame` reads a road's sides off the outline of its mask,
    which is the outline of a network's opinion: a kerb hidden by a van, a
    hard shoulder taken in on one side. The lines painted on the road are
    better evidence of where it goes than that. On a straight road they all
    run to one point, the vanishing point; it is found here as the point
    nearest all of them, and the road's position across is then counted
    along lines through it, between the outermost such lines that still
    hold road. A lane line then lies at one position along its whole
    length, to within a pixel, and can be found by counting
    (:func:`road_lanes`).

    A road is left as it was when its paint does not run to one point: too
    few lines, lines too nearly parallel to say where they meet, or a bend.
    """
    across, width_px = frame.across.copy(), frame.width_px.copy()
    along = frame.along.copy()
    for road in range(1, int(frame.piece.max()) + 1):
        body = frame.piece == road
        count, parts, stats, centres = cv2.connectedComponentsWithStats(
            (paint & body).astype(np.uint8), connectivity=8
        )
        points, normals, lengths = [], [], []
        for index in range(1, count):
            if stats[index, cv2.CC_STAT_AREA] < _LINE_FEWEST_PIXELS:
                continue
            rows, columns = np.nonzero(parts == index)
            values, vectors = np.linalg.eigh(np.cov(np.stack([columns, rows]).astype(np.float64)))
            points.append(centres[index])
            normals.append(vectors[:, 0])
            lengths.append(float(np.sqrt(values[1])))
        if len(points) < _FEWEST_LINES:
            continue
        at, facing, long = np.array(points), np.array(normals), np.array(lengths)
        weights = long.copy()
        meet = np.zeros(2)
        well_posed = True
        for _ in range(8):
            scatter = (facing * weights[:, None]).T @ facing
            values = np.linalg.eigvalsh(scatter)
            if values[0] < _LEAST_SPREAD * values[1]:
                well_posed = False
                break
            meet = np.linalg.solve(
                scatter, (facing * weights[:, None]).T @ (facing * at).sum(axis=1)
            )
            away = np.maximum(np.linalg.norm(at - meet, axis=1), 1.0)
            off = np.degrees(
                np.arcsin(np.clip(np.abs(((meet - at) * facing).sum(axis=1)) / away, 0, 1))
            )
            weights = long / (1.0 + (off / _POINTS_AT_IT) ** 2)
        if not well_posed or float(np.median(off)) > _POINTS_AT_IT:
            continue
        rows, columns = np.nonzero(body)
        # The road below the point its lines run to: all of it, on a road
        # running away up the frame, but for the last few rows where a
        # network's mask runs on into the horizon. A road much of which is
        # level with that point or above it is not such a road.
        below = rows > meet[1] + _BELOW_THE_POINT * (float(rows.max()) - meet[1])
        if float(below.mean()) < 0.9:
            continue
        rows, columns = rows[below], columns[below]
        # Each pixel's line through the meeting point, by where that line
        # crosses the road's nearest row.
        foot = float(rows.max())
        stretch = (foot - meet[1]) / (rows - meet[1])
        crossing = meet[0] + (columns - meet[0]) * stretch
        first, last = np.percentile(crossing, [0.5, 99.5])
        if last - first < _LEAST_WIDTH:
            continue
        across[rows, columns] = np.clip((crossing - first) / (last - first), 0.0, 1.0)
        width_px[rows, columns] = (last - first) / stretch
        towards = np.stack([columns - meet[0], rows - meet[1]], axis=1)
        along[rows, columns] = towards / np.maximum(
            np.linalg.norm(towards, axis=1, keepdims=True), 1e-6
        )
    return frame._replace(across=across, width_px=width_px, along=along)


class RoadModel(NamedTuple):
    """A road as a camera sees it: every lane line of it at once.

    On level ground, a line along a road of constant curvature is, in the
    picture,

        u = centre + offset * (v - horizon) + bend / (v - horizon)

    for a column ``u`` and a row ``v``. ``horizon`` is the row of the
    horizon, ``centre`` the column the road runs to on it, and ``bend`` the
    road's curvature; those three are the road's, and the same for every
    line along it. ``offset`` is the line's own: how far to the side it
    lies, in camera heights. So ``offset`` is a position across the road
    for every pixel, constant along every lane line, kerb and wheel track,
    on a bend as on a straight, and across two carriageways as across one.
    """

    centre: float
    horizon: float
    bend: float

    def offset(self, columns: np.ndarray, rows: np.ndarray) -> np.ndarray:
        """The position across the road of each pixel; ``nan`` at the horizon and above."""
        below = np.asarray(rows, dtype=np.float64) - self.horizon
        with np.errstate(divide="ignore", invalid="ignore"):
            across = (
                np.asarray(columns, dtype=np.float64) - self.centre - self.bend / below
            ) / below
        return np.where(below > 0.0, across, np.nan)

    def slope(self, columns: np.ndarray, rows: np.ndarray) -> np.ndarray:
        """How many columns the road runs per row down the frame, at each pixel."""
        below = np.maximum(np.asarray(rows, dtype=np.float64) - self.horizon, 1e-6)
        return (np.asarray(columns, dtype=np.float64) - self.centre) / below - 2.0 * self.bend / (
            below * below
        )


def _thin_bright(linear_rgb: np.ndarray, where: np.ndarray) -> tuple[np.ndarray, int]:
    """Thin things brighter than the road round them: a label for each, and how many labels."""
    height, width = where.shape
    luma = np.sqrt(
        np.clip(linear_rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32), 0.0, 1.0)
    ).astype(np.float32)
    reach = max(3, round(_LINE_WIDTH * float(np.hypot(height, width))) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (reach, reach))
    without = cv2.morphologyEx(luma, cv2.MORPH_OPEN, kernel)
    if int(where.sum()) < 64:
        return np.zeros(where.shape, dtype=np.int32), 1
    typical = float(np.median(luma[where]))
    thin = (
        where
        & (luma - without > _LINE_STANDS_OUT * (without + 0.05))
        & (luma > _LINE_OVER_ROAD * typical)
    )
    if float(thin.sum()) > _MOST_PAINT * float(where.sum()):
        # Paint is a little of a road. This much of it "paint" is the grain
        # of the asphalt under a lamp, or the noise of a night frame.
        return np.zeros(where.shape, dtype=np.int32), 1
    count, parts = cv2.connectedComponents(thin.astype(np.uint8), connectivity=8)
    return parts, count


def road_cues(
    linear_rgb: np.ndarray, road: np.ndarray, on_road: np.ndarray, vehicles: np.ndarray | None
) -> np.ndarray:
    """Everything in a picture that says which way its road runs, ``(N, 5)``.

    Each row is a short piece of line: its column and row, its direction as
    a unit vector pointing down the frame, and how much it counts for.
    Three kinds of thing give them:

    * **paint**: every thin bright streak on the road, a pixel of weight
      for a pixel of length. A lane line runs along the road by definition;
    * **the road's edges**: the outline of its mask, where that is not the
      frame's own border. A kerb runs along the road, the far end of a road
      does not, and the fit has to tell them apart;
    * **the vehicles**: the long way of each one's outline, for what it is
      worth, which from most cameras is roughly the way it is driving.

    None of them is trusted. They are handed to a robust fit
    (:func:`fit_road_model`), which keeps the ones that agree.
    """
    height, width = road.shape
    cues: list[tuple[float, float, float, float, float]] = []

    def add(columns: np.ndarray, rows: np.ndarray, weight: float, least: float) -> None:
        """One cue from a set of pixels: their middle, their long axis, if long enough."""
        if len(rows) < 3:
            return
        values, vectors = np.linalg.eigh(np.cov(np.stack([columns, rows]).astype(np.float64)))
        if values[1] < least * least * max(values[0], 0.08):
            return
        axis = vectors[:, 1] * (1.0 if vectors[1, 1] >= 0.0 else -1.0)
        cues.append((float(columns.mean()), float(rows.mean()), axis[0], axis[1], weight))

    carriageway = (road > 0.5) & (
        cv2.dilate((on_road > 0.5).astype(np.uint8), np.ones((5, 5), np.uint8)) == 0
    )
    parts, count = _thin_bright(linear_rgb, carriageway)
    rows, columns = np.nonzero(parts)
    order = np.argsort(parts[rows, columns], kind="stable")
    rows, columns = rows[order], columns[order]
    starts = np.searchsorted(parts[rows, columns], np.arange(1, count + 1))
    for index in range(1, count):
        ys, xs = rows[starts[index - 1] : starts[index]], columns[starts[index - 1] : starts[index]]
        if len(ys) >= _LINE_FEWEST_PIXELS:
            add(xs, ys, _CUE_PAINT * float(np.ptp(ys) + np.ptp(xs)), _LINE_ELONGATION)

    # The outline, in stretches a fortieth of the diagonal long.
    whole = ((road > 0.5) | (on_road > 0.5)).astype(np.uint8)
    stretch = max(6, round(0.025 * float(np.hypot(height, width))))
    contours, _hierarchy = cv2.findContours(whole, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        points = contour.reshape(-1, 2)
        if len(points) < 4 * stretch:
            continue
        for start in range(0, len(points) - stretch, stretch):
            xs, ys = points[start : start + stretch, 0], points[start : start + stretch, 1]
            on_border = (xs <= 1) | (ys <= 1) | (xs >= width - 2) | (ys >= height - 2)
            if not on_border.any():
                add(xs, ys, _CUE_EDGE * float(stretch), 2.5)

    if vehicles is not None:
        count, parts = cv2.connectedComponents((vehicles > 0.5).astype(np.uint8), connectivity=8)
        for index in range(1, count):
            ys, xs = np.nonzero(parts == index)
            if len(ys) >= _FEWEST_VEHICLE_PIXELS:
                add(xs, ys, _CUE_VEHICLE * float(np.sqrt(len(ys))), 1.3)
    return np.array(cues, dtype=np.float64).reshape(-1, 5)


def fit_road_model(
    cues: np.ndarray, height: int, top_of_road: float, horizon_row: float | None = None
) -> RoadModel | None:
    """The :class:`RoadModel` most of a picture's cues agree on, or ``None``.

    For a horizon, the way the road runs at a pixel is a linear function of
    the other two numbers of the model, so two cues give a road. Many pairs
    are tried, the road kept is the one the largest share of the cues lies
    along to within a few degrees, and those then settle it between them by
    least squares. The horizon is the one known (``horizon_row``, from the
    camera's geometry), or the one of a ladder of rows above the top of the
    road at which the most cues agree.

    ``None`` when the cues are too few, or fewer than two in five of them
    agree on any road. That is a road running across the picture, a
    junction, or a road that is not on level ground.
    """
    if len(cues) < _FEWEST_CUES:
        return None
    tolerance = np.sin(np.radians(_CUE_TOLERANCE_DEG))
    # A known horizon is used, give or take a little. Left to find its own,
    # the fit has a number too many: a road can be bent to pass through
    # almost anything by moving its horizon as well.
    if horizon_row is not None:
        tried = [horizon_row + height * step for step in (0.0, -0.02, 0.02, -0.04, 0.04)]
    else:
        tried = [top_of_road - height * step for step in _HORIZON_LADDER]
    # The same pairs for the same picture, whatever else has been drawn.
    rng = np.random.default_rng(0)
    best: tuple[float, RoadModel] | None = None
    for horizon in tried:
        assert horizon is not None
        below = cues[:, 1] - horizon
        use = below > _BELOW_HORIZON * height
        if int(use.sum()) < _FEWEST_CUES:
            continue
        u, t, dx, dy, weight = cues[use, 0], below[use], cues[use, 2], cues[use, 3], cues[use, 4]
        # dx = dy * slope, with slope = (u - centre) / t - 2 bend / t^2:
        # linear in the centre and the bend.
        design = np.stack([dy / t, 2.0 * dy / (t * t)], axis=1)
        scale = np.maximum(np.abs(design).max(axis=0), 1e-12)
        design = design / scale
        target = dy * u / t - dx

        def off(found: np.ndarray) -> np.ndarray:
            """The sine of the angle between each cue and the road there, for a fit."""
            slope = (u - found[0]) / t - 2.0 * found[1] / (t * t)  # noqa: B023
            return np.asarray((dx - dy * slope) / np.sqrt(1.0 + slope * slope))  # noqa: B023

        # Two cues make a road. Many pairs are tried, and the road kept is
        # the one most of the others lie along: a slip road, a kerb's end
        # and a crossing are each a few cues, and never most of them.
        chosen = rng.choice(len(u), size=(_PAIRS_TRIED, 2), p=weight / weight.sum())
        agreed, start = 0.0, None
        straight, start_straight = 0.0, None
        for first, second in chosen:
            # One cue makes a straight road, which is tried first.
            if abs(float(design[first, 0])) > 1e-9:
                found = np.array([target[first] / design[first, 0] / scale[0], 0.0])
                share = float(weight[np.abs(off(found)) < tolerance].sum() / weight.sum())
                if share > straight:
                    straight, start_straight = share, found
            pair = design[[first, second]]
            if first == second or abs(float(np.linalg.det(pair))) < 1e-9:
                continue
            found = np.linalg.solve(pair, target[[first, second]]) / scale
            share = float(weight[np.abs(off(found)) < tolerance].sum() / weight.sum())
            if share > agreed:
                agreed, start = share, found
        # A bend has to earn itself: with one more number to choose, some
        # curve always passes through a few more cues than a straight road.
        if start is None or (start_straight is not None and agreed < straight + _BEND_EARNS):
            start = start_straight
        if start is None:
            continue
        # And then all that agree with it settle it between them.
        found = start
        for _ in range(6):
            keep = weight / (1.0 + (off(found) / tolerance) ** 4)
            found, *_ = np.linalg.lstsq(design * keep[:, None], target * keep, rcond=None)
            found = found / scale
        agreed = float(weight[np.abs(off(found)) < tolerance].sum() / weight.sum())
        enough = _CUES_AGREE_KNOWN_HORIZON if horizon_row is not None else _CUES_AGREE
        if agreed >= enough and (best is None or agreed > best[0]):
            best = (agreed, RoadModel(float(found[0]), float(horizon), float(found[1])))
    return None if best is None else best[1]


def model_frame(model: RoadModel, road: np.ndarray, on_road: np.ndarray) -> RoadFrame | None:
    """A :class:`RoadFrame` from a :class:`RoadModel`: its carriageways, and each one's coordinates.

    The model gives every pixel a position across the road. Counted by that
    position, the road's mask is solid where there is a carriageway and
    empty where there is a reserve or a barrier between two, however the
    two masks run together at the bottom of the frame: so the carriageways
    are the stretches of position most of whose length is road, each is a
    road of its own, and no lane of one crosses to the other.
    """
    height, width = road.shape
    rows, columns = np.mgrid[0:height, 0:width]
    below = rows - model.horizon
    seen = below > _BELOW_HORIZON * height
    offset = np.where(seen, model.offset(columns, rows), 0.0)
    whole = ((road > 0.5) | (on_road > 0.5)) & seen
    if int(whole.sum()) < _LEAST_SHARE * height * width:
        return None
    low, high = np.percentile(offset[whole], [0.5, 99.5])
    if not high - low > 1e-9:
        return None
    bins = 240
    at = np.clip(((offset - low) / (high - low) * bins).astype(np.int64), -1, bins)
    inside = seen & (at >= 0) & (at < bins)
    # A row is a row: each counts the same for every position it reaches.
    per_row = 1.0 / np.maximum(below, 1.0)
    of_road = np.bincount(at[inside & whole], weights=per_row[inside & whole], minlength=bins)
    of_frame = np.bincount(at[inside], weights=per_row[inside], minlength=bins)
    solid = cv2.GaussianBlur(
        (of_road / np.maximum(of_frame, 1e-9)).astype(np.float32).reshape(1, -1), (0, 0), 1.2
    ).ravel()
    is_road = (solid > _CARRIAGEWAY_SOLID) & (of_frame > 0.1 * float(of_frame.max()))
    padded = np.concatenate([[0], is_road.astype(np.int8), [0]])
    starts = np.flatnonzero(np.diff(padded) == 1)
    ends = np.flatnonzero(np.diff(padded) == -1)
    found = []
    for start, end in zip(starts, ends, strict=True):
        body = inside & whole & (at >= start) & (at < end)
        if end - start >= 4 and int(body.sum()) >= _LEAST_SHARE * height * width:
            found.append((int(body.sum()), body, start, end))
    if not found:
        return None
    found.sort(key=lambda entry: -entry[0])
    across = np.zeros((height, width), dtype=np.float32)
    span = np.ones((height, width), dtype=np.float32)
    piece = np.zeros((height, width), dtype=np.int32)
    heights_wide = np.zeros(min(len(found), _MOST_ROADS) + 1)
    for index, (_area, body, start, end) in enumerate(found[:_MOST_ROADS], start=1):
        first = low + (high - low) * start / bins
        last = low + (high - low) * end / bins
        heights_wide[index] = last - first
        across[body] = np.clip((offset[body] - first) / (last - first), 0.0, 1.0)
        span[body] = ((last - first) * below[body]).astype(np.float32)
        piece[body] = index
    slope = model.slope(columns, rows)
    length = np.sqrt(1.0 + slope * slope)
    along = np.stack([slope / length, 1.0 / length], axis=2).astype(np.float32)
    return RoadFrame(
        weight=(piece > 0).astype(np.float32),
        across=across,
        width_px=np.maximum(span, 1.0),
        along=along,
        piece=piece,
        heights_wide=heights_wide,
    )


def read_road(
    linear_rgb: np.ndarray,
    road: np.ndarray,
    on_road: np.ndarray,
    vehicles: np.ndarray | None,
    rng: np.random.Generator,
    lane_count_range: tuple[int, int] = (2, 3),
    horizon_row: float | None = None,
    camera_height_m: float | None = None,
) -> tuple[RoadFrame, Lanes] | None:
    """A picture's roads and their lanes, from everything that says where they are.

    First as a camera's view of a road on level ground
    (:class:`RoadModel`): every cue to the road's direction is gathered,
    paint, kerbs and vehicles (:func:`road_cues`), the model they agree on
    is fitted (:func:`fit_road_model`), and it gives the carriageways and
    their coordinates (:func:`model_frame`). ``horizon_row`` is the horizon
    where the camera's mounting is known, which is one number fewer to fit,
    and ``camera_height_m`` its height, which says how wide a lane is.

    Where the cues agree on no such road, one running across the picture
    or seen from straight above, the road is read off its mask alone
    (:func:`road_frame`), and put right by its paint where that runs to one
    point (:func:`straighten`).

    Either way the lanes are then the comb the paint lies on and the
    vehicles sit between (:func:`road_lanes`).

    ``road`` is the carriageway itself, ``on_road`` what stands on it, and
    ``vehicles`` the vehicles among that. ``None`` when the picture holds
    no road at all.
    """
    height, width = road.shape
    whole = (road > 0.5) | (on_road > 0.5)
    held = np.flatnonzero(whole.sum(axis=1) >= max(3.0, 0.01 * width))
    frame = None
    if held.size:
        cues = road_cues(linear_rgb, road, on_road, vehicles)
        model = fit_road_model(cues, height, float(held[0]), horizon_row)
        if model is not None:
            frame = model_frame(model, road, on_road)
    if frame is not None:
        paint = painted_lines(linear_rgb, frame, road, on_road)
        # The road's width is in camera heights only when the model's
        # horizon is the camera's: found by the fit alone, it is whatever
        # made the cues agree, and the widths that follow from it are not
        # metres of anything.
        height_m = camera_height_m if horizon_row is not None else None
        return frame, road_lanes(frame, vehicles, rng, lane_count_range, paint, height_m)
    frame = road_frame(whole)
    if frame is None:
        return None
    # Read off the mask alone, a road is only as good as the mask, and its
    # paint is counted only where the paint itself put the road right.
    paint = painted_lines(linear_rgb, frame, road, on_road)
    trusted = np.zeros(road.shape, dtype=bool)
    if paint.any():
        truer = straighten(frame, paint)
        moved = np.abs(truer.across - frame.across) > 1e-6
        for index in range(1, int(frame.piece.max()) + 1):
            if moved[frame.piece == index].any():
                trusted |= frame.piece == index
        frame = truer
        paint = painted_lines(linear_rgb, frame, road, on_road) & trusted
    # And nothing is guessed: a road nothing vouches for, neither a model
    # nor its paint nor a vehicle on it, is as likely a junction seen from
    # above, and tracks ruled across a junction are worse than none.
    return frame, road_lanes(frame, vehicles, rng, lane_count_range, paint, guess=False)


# The mist that hangs over a carriageway in traffic: the distance, in
# metres, through which it has an optical depth of one at its thickest, and
# the number of vehicles on a road at which it is that thick.
_MIST_REACH_M = 110.0
_MIST_TRAFFIC = 10
# How far up the layer of mist a vehicle's roof is.
_ROOF_IN_MIST = 0.75


def road_mist(
    frame: RoadFrame,
    vehicles: np.ndarray,
    depth_m: np.ndarray,
    traffic: list[RoadVehicle] | None = None,
) -> np.ndarray:
    """The optical depth of the spray that hangs over a road in traffic, ``(H, W)`` float32.

    What one vehicle throws up, the next drives through: over a busy wet
    road there is a layer of fine spray a couple of metres deep, on the
    carriageway and nowhere else. Seen from above and along the road, a
    sight line runs through more of it the further away it meets the road,
    so the mist thickens with distance: the nearest cars are clear, and
    the far traffic is gone from the wheels up, with its lamps and its
    roofs the last of it to go. That, more than any one vehicle's wake, is
    what a motorway in rain looks like.

    The depth is in proportion to the distance to each pixel of road, and
    to how much traffic the road carries. With ``traffic``
    (:func:`road_vehicles`) each vehicle counts for what it throws up at
    its speed (:func:`spray_amount`), so a queue hangs no mist over its
    road; without, each counts for one.

    On a vehicle it is that of the road at its foot, less towards its roof:
    a sight line that ends on a vehicle leaves the layer where it meets the
    vehicle, and has run through the less of it the higher up that is. A
    car's roof is three quarters of the way up the layer. It used to be
    taken as clear of it, and each car stood out of the mist behind it with
    a hard edge all the way round.
    """
    height, width = vehicles.shape
    mask = (vehicles > 0.5).astype(np.uint8)
    count, parts, stats, _centres = cv2.connectedComponentsWithStats(mask, connectivity=8)
    roads = int(frame.piece.max())
    carried = np.zeros(roads + 1)
    low = np.zeros((height, width), dtype=np.float32)
    foot_depth = depth_m.astype(np.float32).copy()
    for index in range(1, count):
        left, top, wide, tall, area = (int(value) for value in stats[index])
        if area < _FEWEST_VEHICLE_PIXELS:
            continue
        row = min(top + tall - 1, height - 1)
        column = min(left + wide // 2, width - 1)
        road = int(frame.piece[row, column])
        if road == 0:
            continue
        carried[road] += 1.0
        body = parts[top : top + tall, left : left + wide] == index
        # How far up the vehicle each pixel of it is, from its foot.
        up = (np.arange(tall, dtype=np.float32)[::-1] / max(tall - 1, 1))[:, None]
        low[top : top + tall, left : left + wide][body] = np.broadcast_to(
            1.0 - _ROOF_IN_MIST * up, body.shape
        )[body]
        foot_depth[top : top + tall, left : left + wide][body] = float(depth_m[row, column])
    if traffic is not None:
        carried = np.zeros(roads + 1)
        for vehicle in traffic:
            carried[vehicle.road] += spray_amount(vehicle.speed_kmh)
            # Each by its own height and its own foot, where a network has
            # told it from the ones its outline runs into: a column of
            # cars is not one tall vehicle with its roof at the far end.
            rows, columns = vehicle.rows, vehicle.columns
            up = (float(rows.max()) - rows) / max(float(np.ptp(rows)), 1.0)
            low[rows, columns] = 1.0 - _ROOF_IN_MIST * up
            foot_depth[rows, columns] = float(depth_m[vehicle.foot])
    on_road = np.where(mask > 0, low, frame.weight)
    busy = np.clip(carried / _MIST_TRAFFIC, 0.0, 1.0)[frame.piece]
    # A vehicle's pixels take the traffic of the road at its foot.
    busy = cv2.dilate(busy.astype(np.float32), np.ones((3, 3), np.uint8))
    by_piece = np.clip(carried / _MIST_TRAFFIC, 0.0, 1.0)
    for index in range(1, count):
        left, top, wide, tall, area = (int(value) for value in stats[index])
        row, column = min(top + tall - 1, height - 1), min(left + wide // 2, width - 1)
        if area >= _FEWEST_VEHICLE_PIXELS and frame.piece[row, column] > 0:
            body = parts[top : top + tall, left : left + wide] == index
            busy[top : top + tall, left : left + wide][body] = by_piece[frame.piece[row, column]]
    for vehicle in traffic or ():
        busy[vehicle.rows, vehicle.columns] = by_piece[vehicle.road]
    mist = on_road * busy * foot_depth / _MIST_REACH_M
    soften = max(1.2, 0.003 * float(np.hypot(height, width)))
    return cv2.GaussianBlur(mist.astype(np.float32), (0, 0), sigmaX=soften)
