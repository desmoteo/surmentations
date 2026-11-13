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
_LANE_SHARE = (0.07, 0.9)
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


class Lanes(NamedTuple):
    """Where the lanes of each road in a :class:`RoadFrame` lie across it.

    One entry per road, at the index :attr:`RoadFrame.piece` gives it; index
    0 is unused. All in the units of :attr:`RoadFrame.across`.

    Attributes:
        centre: the position across the road of one lane's centre.
        width: the width of a lane.
        first: where the lanes begin.
        last: where they end.
    """

    centre: np.ndarray
    width: np.ndarray
    first: np.ndarray
    last: np.ndarray


def road_lanes(
    frame: RoadFrame,
    vehicles: np.ndarray | None,
    rng: np.random.Generator,
    lane_count_range: tuple[int, int] = (2, 3),
) -> Lanes:
    """Where the lanes are, read off the vehicles in them.

    Nothing in a picture of a snowed-in road shows its lanes, and the
    vehicles on it are in them. Each vehicle's foot gives a position across
    the road and its width a size: a lane is some 1.7 times as wide as what
    drives in it. The lanes are then a comb of that pitch, slid across the
    road to where the vehicles sit closest to its teeth, which is the mean
    of their positions taken round the circle of one lane's width.

    A road with no vehicle on it gets a number of lanes drawn from
    ``lane_count_range``, spread between its verges.

    ``vehicles`` is a mask of the frame, ``(H, W)``, or ``None`` when
    nothing is known of them.
    """
    roads = int(frame.piece.max())
    centre = np.zeros(roads + 1)
    lane = np.ones(roads + 1)
    first = np.full(roads + 1, _VERGE)
    last = np.full(roads + 1, 1.0 - _VERGE)
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
        if not seen[road]:
            lanes = int(rng.integers(lane_count_range[0], lane_count_range[1] + 1))
            lane[road] = (1.0 - 2.0 * _VERGE) / lanes
            centre[road] = _VERGE + 0.5 * lane[road]
            continue
        where = np.array([position for position, _share in seen[road]])
        # The narrower of them: two that touch, and one seen side on, are
        # both wider than a vehicle and never narrower.
        size = float(np.percentile([share for _position, share in seen[road]], 40.0))
        lane[road] = float(np.clip(_LANE_OVER_VEHICLE * size, *_LANE_SHARE))
        turn = 2.0 * np.pi * where / lane[road]
        centre[road] = lane[road] * float(np.angle(np.exp(1j * turn).mean())) / (2.0 * np.pi)
        if len(seen[road]) >= _ENOUGH_VEHICLES:
            first[road] = max(_VERGE / 2.0, float(where.min()) - 0.6 * lane[road])
            last[road] = min(1.0 - _VERGE / 2.0, float(where.max()) + 0.6 * lane[road])
    # A road whose traffic lies across it has no lanes the way it was read.
    first[crosswise > 0.0], last[crosswise > 0.0] = 1.0, 0.0
    return Lanes(centre, lane, first, last)


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
    lane = lanes.width[frame.piece]
    centre = lanes.centre[frame.piece]
    turns = (frame.across - centre) / lane
    if ragged is not None:
        # A tenth of a lane to either side: nobody drives in a ruled line.
        turns = turns + _TRACK_WANDER * (2.0 * ragged - 1.0)
    which = np.rint(turns)
    from_centre = np.abs(turns - which)
    from_wheel = np.abs(from_centre - 0.5 * _WHEELS_APART)
    half: np.ndarray = np.full(frame.across.shape, band, dtype=np.float32)
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
    track = track * use[which.astype(np.int64) % 64]
    if worn is not None:
        track = track * np.clip(wear_floor + (1.0 - wear_floor) * 2.0 * worn, 0.0, 1.0)
    inside = (frame.across >= lanes.first[frame.piece]) & (frame.across <= lanes.last[frame.piece])
    return (track * inside * frame.weight).astype(np.float32)
