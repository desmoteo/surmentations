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

from typing import NamedTuple

import cv2
import numpy as np

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

    Attributes:
        weight: ``(H, W)`` float32 in ``[0, 1]``.
        across: ``(H, W)`` float32 in ``[0, 1]``: position across the road,
            from one side to the other.
        width_px: ``(H, W)`` float32: the road's width through the pixel,
            in pixels, which is what turns a width in the world into one in
            the picture.
        along: ``(H, W, 2)`` float32: the unit vector along the road,
            ``(dx, dy)``, pointing down the frame where it has a choice.
        piece: ``(H, W)`` int32: which connected piece of road the pixel
            belongs to, from 1; 0 off the road.
    """

    weight: np.ndarray
    across: np.ndarray
    width_px: np.ndarray
    along: np.ndarray
    piece: np.ndarray


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
_LEAST_PAINT = 0.08
_PEAK_SHARE = 0.07
# How much of a lane has to fit between the last line and the verge for a
# lane to be put there.
_PART_OF_A_LANE = 0.6
# How near two painted lines may be, as a share of a lane, and still be two
# lanes' edges.
_NEAREST_LINES = 0.6
# How far from a whole number of lanes the gaps between painted lines may
# be, on average, as a share of a lane; and the narrowest a lane read off
# paint may be against the one the vehicles suggest.
_GAPS_FIT = 0.17
_NARROWEST_LANE = 0.6
# And the narrowest in any case, as a share of the road: eight lanes abreast.
_NARROWEST_PAINTED_LANE = 0.12


class Lanes(NamedTuple):
    """Where the lanes of each road in a :class:`RoadFrame` lie across it.

    Attributes:
        edges: for each road, at the index :attr:`RoadFrame.piece` gives it
            (index 0 is unused), the positions across the road of its lane
            lines in order, in the units of :attr:`RoadFrame.across`: lane
            ``i`` runs from ``edges[road][i]`` to ``edges[road][i + 1]``. An
            empty array for a road with no lanes.
        painted: for each road, how many of those lines were read off paint
            on the road, the rest being placed between and beyond them.
    """

    edges: list[np.ndarray]
    painted: list[int]


def _line_positions(across: np.ndarray, weight: np.ndarray, least_apart: float) -> np.ndarray:
    """The positions across a road at which its paint piles up: its lane lines.

    A lane line lies at one position across the road for its whole length,
    so the paint on a road, counted by position, stands in narrow peaks.
    ``across`` and ``weight`` are the position and the length each painted
    pixel counts for. Returns the peaks in order, at least ``least_apart``
    from each other; empty when there is no paint to speak of.
    """
    bins = 400
    piled = np.bincount(
        np.clip((across * bins).astype(np.int64), 0, bins - 1), weights=weight, minlength=bins
    ).astype(np.float32)
    counts: np.ndarray = cv2.GaussianBlur(piled.reshape(1, -1), (0, 0), sigmaX=1.6).ravel()
    if float(counts.max()) < _LEAST_PAINT:
        return np.zeros(0)
    apart = max(2, round(least_apart * bins))
    found: list[int] = []
    for index in np.argsort(-counts):
        if counts[index] < max(_PEAK_SHARE * float(counts.max()), _LEAST_PAINT):
            break
        # The strongest first, so of two lines too close to be two lanes'
        # edges, a kerb and the line painted beside it, the bolder is kept.
        if all(abs(int(index) - other) >= apart for other in found):
            found.append(int(index))
    return np.sort(np.array(found, dtype=np.float64) + 0.5) / bins


def _lane_width(gaps: np.ndarray, by_vehicles: float | None) -> float | None:
    """One lane's width, from the gaps between a road's painted lines, or ``None``.

    Every gap is a whole number of lanes: one, or two where a line is worn
    away or was never painted. So the width is the largest that every gap
    is a whole multiple of, to within a sixth of a lane; tried are the gaps
    themselves and the width the vehicles suggest. ``None`` when nothing
    fits, or when what fits is a lane narrower than an eighth of the road or
    much narrower than what drives on it, which is paint that is not lane
    lines.
    """
    tried = sorted([*gaps.tolist(), *([] if by_vehicles is None else [by_vehicles])], reverse=True)
    for width in tried:
        lanes = np.maximum(np.rint(gaps / width), 1.0)
        if float(np.mean(np.abs(gaps / width - lanes) / lanes)) > _GAPS_FIT:
            continue
        lane = float(np.median(gaps / lanes))
        if lane < _NARROWEST_PAINTED_LANE:
            # Narrow enough, a lane divides anything.
            return None
        if by_vehicles is not None and lane < _NARROWEST_LANE * by_vehicles:
            return None
        return float(np.clip(lane, *_LANE_SHARE))
    return None


def _lanes_between(lines: np.ndarray, lane: float, first: float, last: float) -> np.ndarray:
    """Lane lines every ``lane`` across a road, through the ones that were found.

    Between two found lines there is a whole number of lanes, as near
    ``lane`` wide as divides the gap. Beyond the outermost, lanes of that
    width for as long as a good part of one fits before the verge.
    """
    edges = [float(lines[0])]
    for line in lines[1:]:
        gap = float(line) - edges[-1]
        count = max(1, round(gap / lane))
        edges.extend(edges[-1] + gap * (step + 1) / count for step in range(count))
    while edges[0] - first > _PART_OF_A_LANE * lane:
        edges.insert(0, max(first, edges[0] - lane))
    while last - edges[-1] > _PART_OF_A_LANE * lane:
        edges.append(min(last, edges[-1] + lane))
    return np.array(edges)


def road_lanes(
    frame: RoadFrame,
    vehicles: np.ndarray | None,
    rng: np.random.Generator,
    lane_count_range: tuple[int, int] = (2, 3),
    paint: np.ndarray | None = None,
) -> Lanes:
    """Where the lanes are: read off the paint, and off the vehicles in them.

    Three things say where a road's lanes are, and they are used in that
    order.

    **The paint.** ``paint`` is the mask of painted lines
    (:func:`painted_lines`). Counted by position across the road it stands
    in peaks, one to a line, and those are lane lines to the pixel. Two of
    them give the width of a lane; the gaps between them are divided into
    whole lanes of about that width, a dashed line too worn to be found
    being a gap of two; and lanes are carried on beyond the outermost for
    as long as there is road.

    **The vehicles.** Each one's foot gives a position across the road and
    its width a size: a lane is some 1.7 times as wide as what drives in
    it. With one painted line, or none, that is the width of a lane; with
    none, the lanes are a comb of that pitch slid across the road to where
    the vehicles sit closest to the middle of its teeth, which is the mean
    of their positions taken round the circle of one lane's width.

    **A guess.** A road with neither gets a number of lanes drawn from
    ``lane_count_range``, spread between its verges.

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
            # As wide as it is at its foot: the box round a vehicle seen at
            # an angle takes in its length as well.
            foot = mask[max(top, row - max(1, tall // 4)) : row + 1, left : left + wide]
            wide_at_foot = float(foot.sum(axis=1).max())
            share = wide_at_foot / max(float(frame.width_px[row, column]), 1.0)
            # Which way it lies against the way the road is said to run.
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
        first, last = _VERGE, 1.0 - _VERGE
        where = np.array([position for position, _share in seen[road]])
        by_vehicles = None
        if len(where):
            # The narrower of them: two that touch, and one seen side on,
            # are both wider than a vehicle and never narrower.
            size = float(np.percentile([share for _position, share in seen[road]], 40.0))
            by_vehicles = float(np.clip(_LANE_OVER_VEHICLE * size, *_LANE_SHARE))
        lines = np.zeros(0)
        if paint is not None:
            here = paint & (frame.piece == road)
            if int(here.sum()) >= _LINE_FEWEST_PIXELS:
                # No two lines nearer than most of a lane: the vehicles say
                # how wide that is, where there are any. Counted by length
                # along the road and not by pixels, or the paint under the
                # camera would be all of it.
                narrowest = _LANE_SHARE[0] if by_vehicles is None else by_vehicles
                lines = _line_positions(
                    frame.across[here], 1.0 / frame.width_px[here], _NEAREST_LINES * narrowest
                )
        lane_by_paint = _lane_width(np.diff(lines), by_vehicles) if len(lines) >= 2 else None
        if len(lines) >= 2 and lane_by_paint is None:
            # Paint that fits no even spacing is not lane lines: a
            # crossing, hatching, words. Nothing is read off it.
            lines = np.zeros(0)
        painted[road] = len(lines)
        if lane_by_paint is not None:
            edges[road] = _lanes_between(lines, lane_by_paint, first, last)
            continue
        if by_vehicles is None:
            count = int(rng.integers(lane_count_range[0], lane_count_range[1] + 1))
            lane = (last - first) / count
            start = float(lines[0]) if len(lines) else first
            edges[road] = _lanes_between(np.array([start]), lane, first, last)
            continue
        lane = by_vehicles
        if len(lines) == 1:
            edges[road] = _lanes_between(lines, lane, first, last)
            continue
        turn = 2.0 * np.pi * where / lane
        centre = lane * float(np.angle(np.exp(1j * turn).mean())) / (2.0 * np.pi)
        if len(where) >= _ENOUGH_VEHICLES:
            first = max(_VERGE / 2.0, float(where.min()) - 0.6 * lane)
            last = min(1.0 - _VERGE / 2.0, float(where.max()) + 0.6 * lane)
        # The comb's teeth are lane lines: half a lane from the centres.
        line = centre + 0.5 * lane
        line += lane * np.ceil((first - line) / lane)
        if line > last:
            line = 0.5 * (first + last)
        edges[road] = _lanes_between(np.array([line]), lane, first, last)
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
# The plume behind a vehicle: how many puffs it is drawn as; its length as
# a multiple of the vehicle's own length in the picture; a puff's spread
# across the vehicle's heading, as a share of half the vehicle's width, at
# the wheels and at the plume's far end; and how high it hangs there, as a
# share of the vehicle's height in the picture.
_PUFFS = 7
_PLUME_LENGTH = 1.5
_PUFF_ACROSS = (0.75, 1.3)
_PUFF_UP = (0.2, 0.55)
# The most spray counted in front of one pixel, in plumes at their densest:
# ten cars abreast do not make ten times the mist of one.
_MOST_SPRAY = 1.6
# A vehicle's own heading is read off its outline when the outline is this
# many times longer than wide, and lies within this cosine of the road's.
_HEADING_ELONGATION = 1.3
_HEADING_NEAR_ROAD = 0.82
# A vehicle whose long way lies further than this cosine from the road's
# lies across it, and trails nothing. And the least share of the frame a
# road covers for its vehicles to be given spray at all.
_HEADING_ACROSS = 0.7
_LEAST_ROAD_FOR_SPRAY = 0.05
# A tail lamp: red at this level, in linear light, and this many times its
# other channels. And the fewest vehicles on a road that have to show them
# for the road's traffic to be taken as driving away.
_TAIL_LAMP_LEVEL = 0.3
_TAIL_LAMP_RED = 2.5
_FEWEST_TAIL_LAMPS = 2


def vehicle_spray(
    frame: RoadFrame,
    vehicles: np.ndarray,
    rng: np.random.Generator,
    linear_rgb: np.ndarray | None = None,
) -> np.ndarray:
    """How much spray hangs in front of each pixel, ``(H, W)`` float32, 0 and up.

    A vehicle at speed on a wet road lifts the water off it and trails it
    as a low cloud of mist that lies on the road behind it: densest at the
    wheels and no higher than they are there, as wide as the vehicle, then
    a little wider, hanging up to about its roof and thinning out over a
    length and a half. It is a thing
    in the scene: the plume of a vehicle driving away hangs between it and
    the camera and takes its wheels and its bumper with it, while that of a
    vehicle coming on is hidden by the vehicle itself.

    **Where the plume lies is the vehicle's own heading.** Its outline in
    the picture is longer one way than the other, and where that way is
    near the road's it is the vehicle's heading: a car changing lanes or
    taking a bend trails its spray along its own line and not the road's.
    A vehicle too small or too square to say is given the road's direction.
    The plume starts at the end of the outline it trails from, and is as
    wide there as the vehicle is across its heading.

    **Some vehicles get none.** One whose long way lies across the road is
    crossing it, turning or parked, or the road was misread there. One on a
    scrap of road, a slip road cut by the frame's edge, is on a road whose
    direction is a guess. Both used to shoot a jet of mist out sideways.

    **Which end is the tail** a frame says only sometimes. Tail lamps do:
    a vehicle showing red lamps is seen from behind, and is driving away.
    Where ``linear_rgb`` is given and enough of a road's vehicles show
    them, that road's traffic runs away from the camera. Otherwise the
    direction is drawn, once per road, since a carriageway goes one way.

    How fast each vehicle is going is drawn too: each is given its own
    share of spray, and some none, as a parked car has none.

    The result is a density, to be turned into an opacity by whatever knows
    how wet the road is: ``1 - exp(-strength * density)``.
    """
    height, width = vehicles.shape
    mask = (vehicles > 0.5).astype(np.uint8)
    count, parts, stats, centres = cv2.connectedComponentsWithStats(mask, connectivity=8)
    roads = int(frame.piece.max())
    # Towards the camera, or away from it: one way for each road.
    trailing = rng.choice([-1.0, 1.0], size=roads + 1)
    order = sorted(range(1, count), key=lambda index: -int(stats[index, cv2.CC_STAT_AREA]))
    order = [index for index in order if stats[index, cv2.CC_STAT_AREA] >= _FEWEST_SPRAY_PIXELS]
    order = order[:_MOST_SPRAYING]

    shapes = []
    red_lamps = np.zeros(roads + 1)
    for index in order:
        left, top, wide, tall, _area = (int(value) for value in stats[index])
        rows, columns = np.nonzero(parts[top : top + tall, left : left + wide] == index)
        rows, columns = rows + top, columns + left
        row = min(top + tall - 1, height - 1)
        column = min(left + wide // 2, width - 1)
        road = int(frame.piece[row, column])
        if road == 0:
            continue
        shapes.append((index, road, rows, columns, row, column))
        if linear_rgb is not None:
            colour = linear_rgb[rows, columns]
            lit = (colour[:, 0] > _TAIL_LAMP_LEVEL) & (
                colour[:, 0] > _TAIL_LAMP_RED * colour[:, 1:].max(axis=1)
            )
            red_lamps[road] += 1.0 if int(lit.sum()) >= 2 else 0.0
    # Seen from behind: driving away, and trailing towards the camera.
    trailing[red_lamps >= _FEWEST_TAIL_LAMPS] = 1.0

    near = np.zeros((height, width), dtype=np.float32)
    behind = np.zeros((height, width), dtype=np.float32)
    share = np.bincount(frame.piece.ravel(), minlength=roads + 1) / frame.piece.size
    for index, road, rows, columns, row, column in shapes:
        speed = float(rng.uniform(-0.6, 1.0))
        # Not on a scrap of road, whose direction is a guess: a slip road
        # cut by the frame's edge, the far carriageway behind a barrier.
        # Their plumes shot out sideways.
        if speed <= 0.0 or share[road] < _LEAST_ROAD_FOR_SPRAY:
            continue
        tall = int(stats[index, cv2.CC_STAT_HEIGHT])
        along_road = frame.along[row, column].astype(np.float64)
        centre = centres[index]
        offsets = np.stack([columns - centre[0], rows - centre[1]], axis=1)
        values, vectors = np.linalg.eigh(np.cov(offsets.T) + 1e-6 * np.eye(2))
        own = vectors[:, 1] * (1.0 if float(vectors[:, 1] @ along_road) >= 0.0 else -1.0)
        long_way = float(np.sqrt(values[1] / max(values[0], 1e-6)))
        agrees = float(own @ along_road)
        # A vehicle that lies across the road is crossing it, turning or
        # parked, or the road was read wrong there: none of them is a car
        # at speed, and none gets a plume.
        if long_way > _HEADING_ELONGATION and agrees < _HEADING_ACROSS:
            continue
        # Its own heading, where its outline is long and near the road's.
        heading = (
            own if long_way > _HEADING_ELONGATION and agrees > _HEADING_NEAR_ROAD else along_road
        )
        way = heading * trailing[road]
        side = np.array([-way[1], way[0]])
        half_wide = max(1.5, float(np.percentile(np.abs(offsets @ side), 90.0)))
        long_px = max(3.0, 2.0 * float(np.percentile(np.abs(offsets @ way), 90.0)))
        towards = way[1] > 0.0
        target = near if towards else behind
        length = _PLUME_LENGTH * long_px
        # Where the plume begins: at the end of the outline it trails from.
        # That is the vehicle's foot when it is driving away, and behind its
        # own roof, part of the way to the far end, when it is coming on.
        reach_along = float((offsets @ way).max())
        tail_x = centre[0] + way[0] * reach_along * (1.0 if towards else 0.6)
        tail_y = centre[1] + way[1] * reach_along * (1.0 if towards else 0.6)
        step = length / (_PUFFS - 1)
        for puff in range(_PUFFS):
            along = puff / (_PUFFS - 1)
            # On the road: as wide as the vehicle at its wheels and half as
            # wide again by the end, and each puff run into the next.
            low, high = _PUFF_ACROSS
            across = (low + (high - low) * along) * half_wide
            lengthways = max(1.5, 1.1 * step)
            low, high = _PUFF_UP
            rise = (low + (high - low) * along) * tall
            # And up from it: thrown to the height of the wheels at the
            # tail, hanging to about the roof further back, thinner higher.
            for lift, thin in ((0.0, 1.0), (0.5, 0.55), (1.0, 0.25)):
                centre_x = tail_x + way[0] * length * along
                centre_y = tail_y + way[1] * length * along - lift * rise
                reach = int(3.0 * max(across, lengthways)) + 1
                x0, x1 = max(0, int(centre_x) - reach), min(width, int(centre_x) + reach + 1)
                y0, y1 = max(0, int(centre_y) - reach), min(height, int(centre_y) + reach + 1)
                if x0 >= x1 or y0 >= y1:
                    continue
                dx = np.arange(x0, x1, dtype=np.float32)[None, :] - np.float32(centre_x)
                dy = np.arange(y0, y1, dtype=np.float32)[:, None] - np.float32(centre_y)
                run = (dx * way[0] + dy * way[1]) / lengthways
                spread = (dx * side[0] + dy * side[1]) / across
                blob = np.exp(-0.5 * (run**2 + spread**2))
                # Behind the tail and nowhere else: a puff is round, and
                # one centred on the tail would stand out all round the
                # vehicle as a rim of mist.
                past = (np.arange(y0, y1, dtype=np.float32)[:, None] - tail_y) * way[1] + (
                    np.arange(x0, x1, dtype=np.float32)[None, :] - tail_x
                ) * way[0]
                blob = blob * np.clip(past / max(0.15 * long_px, 1.0) + 0.3, 0.0, 1.0)
                strength = speed * thin * (1.0 - 0.75 * along) ** 1.5
                target[y0:y1, x0:x1] += np.float32(strength) * blob
    # What trails away from the camera is behind the vehicle that threw it.
    # A network's outline of a vehicle is a little inside the vehicle, and
    # the plume showed round it as a rim of light: the outline is let out.
    grow = max(3, round(0.008 * float(np.hypot(height, width))) | 1)
    grown = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow, grow)))
    hidden = cv2.GaussianBlur(grown.astype(np.float32), (0, 0), sigmaX=1.0)
    spray: np.ndarray = np.minimum(near + behind * (1.0 - hidden), _MOST_SPRAY)
    return spray.astype(np.float32)


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
    luma = np.sqrt(
        np.clip(linear_rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32), 0.0, 1.0)
    ).astype(np.float32)
    reach = max(3, round(_LINE_WIDTH * float(np.hypot(height, width))) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (reach, reach))
    without = cv2.morphologyEx(luma, cv2.MORPH_OPEN, kernel)
    where = (road > 0.5) & (frame.weight > 0.5)
    if on_road is not None:
        near = cv2.dilate((on_road > 0.5).astype(np.uint8), np.ones((5, 5), np.uint8))
        where &= near == 0
    if int(where.sum()) < 64:
        return np.zeros(road.shape, dtype=bool)
    typical = float(np.median(luma[where]))
    thin = (
        where
        & (luma - without > _LINE_STANDS_OUT * (without + 0.05))
        & (luma > _LINE_OVER_ROAD * typical)
    )
    count, parts, stats, _centroids = cv2.connectedComponentsWithStats(
        thin.astype(np.uint8), connectivity=8
    )
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


def read_road(
    linear_rgb: np.ndarray,
    road: np.ndarray,
    on_road: np.ndarray,
    vehicles: np.ndarray | None,
    rng: np.random.Generator,
    lane_count_range: tuple[int, int] = (2, 3),
) -> tuple[RoadFrame, Lanes] | None:
    """A picture's roads and their lanes, from everything that says where they are.

    The steps of this module in order: the road's frame from the mask of
    the carriageway and what stands on it (:func:`road_frame`); the lines
    painted on it (:func:`painted_lines`); the frame redrawn through the
    point those lines run to, where they run to one (:func:`straighten`),
    and the paint read again in the truer frame; and the lanes, from the
    paint where it put the road right, and otherwise from the vehicles or
    a guess (:func:`road_lanes`).

    ``road`` is the carriageway itself, ``on_road`` what stands on it, and
    ``vehicles`` the vehicles among that. ``None`` when the picture holds
    no road with two sides.
    """
    frame = road_frame((road > 0.5) | (on_road > 0.5))
    if frame is None:
        return None
    paint = painted_lines(linear_rgb, frame, road, on_road)
    trusted = np.zeros(road.shape, dtype=bool)
    if paint.any():
        truer = straighten(frame, paint)
        # Lane lines are counted off the paint only on a road its paint put
        # right: there the lines demonstrably run to one point, and each
        # lies at one position across. On a road read off its mask alone,
        # a bend or two carriageways run together, the same paint piles up
        # at the wrong positions, and lanes read off it were bunched
        # against one kerb. Those roads take their lanes from the vehicles.
        moved = np.abs(truer.across - frame.across) > 1e-6
        for index in range(1, int(frame.piece.max()) + 1):
            if moved[frame.piece == index].any():
                trusted |= frame.piece == index
        frame = truer
        paint = painted_lines(linear_rgb, frame, road, on_road) & trusted
    return frame, road_lanes(frame, vehicles, rng, lane_count_range, paint)
