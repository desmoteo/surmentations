"""Reading the ground off a depth map and a picture.

Rain wets the ground and snow settles on it, and both need the same three
things: which pixels are level ground, what kind of ground it is, and a way
to lay a pattern on it so that it foreshortens with the road instead of
sitting on the screen.
"""

from __future__ import annotations

import cv2
import numpy as np

from surmentations import _native, mode
from surmentations.utils.channels import channel_max, channel_min
from surmentations.utils.filters import wide_blur
from surmentations.utils.noise import value_noise

# Vegetation: hue between yellow-green and blue-green, with enough
# saturation to have a hue at all.
_VEGETATION_HUE_DEG = (50.0, 170.0)
_VEGETATION_MIN_SATURATION = 0.18

# Value of ``cosine ** falloff`` from which ground counts as level.
_LEVEL_GROUND_WEIGHT = 0.35

# How far level ground is grown towards the objects standing on it, as a
# fraction of the diagonal, and how alike in linear RGB a pixel has to be to
# the ground beside it to be taken for more of the same ground.
_EDGE_FILL_REACH = 0.012
_EDGE_FILL_TOLERANCE = 0.08
# The sigma, in pixels, the nearby-ground colour is blurred at on a reduced
# copy in fast mode: coarser than the default, for an average that only has
# to be right to the tolerance above. See `wide_blur`.
_EDGE_FILL_WORK_SIGMA = 4.0

# What stands on the ground. The ground around a pixel is averaged over a
# window this fraction of the diagonal in sigma; a pixel nearer than that
# average by the first share is the start of something standing there and by
# the second all of it, and by the next pair is one whatever it looks like.
_OBJECT_WINDOW = 0.03
_OBJECT_NEARER = (0.02, 0.06)
_OBJECT_FOR_CERTAIN = (0.08, 0.14)
# How far round such a pixel the object is looked for, as a fraction of the
# diagonal, and how unlike the road there a pixel has to be to belong to it:
# the largest difference between the logarithms of its channels and the
# road's, where it starts to count and where it counts in full.
_OBJECT_REACH = 0.02
_OBJECT_UNLIKE = (0.35, 0.6)
# What is too small for a depth map to show standing on the ground. The
# road round a pixel is averaged over a window this fraction of the diagonal
# in sigma, and a pixel whose channels differ from that average by more than
# the first of these, as a logarithm, begins to count as a thing on the road
# and by the second counts in full: a factor of about one and a half to two.
_SMALL_THING_WINDOW = 0.015
_SMALL_THING_UNLIKE = (0.4, 0.7)

# The shorter side, in pixels, below which the frame is not reduced first.
_OBJECT_FULL_SIZE = 320

# Noise tile that ground patterns are cut from, in pixels and in cells.
_TILE_PX = 256
_TILE_CELLS = 8


def vegetation_weight(linear_rgb: np.ndarray) -> np.ndarray:
    """``[0, 1]`` weight of pixels that are green vegetation, ``(H, W)``.

    By colour alone, which is all that is needed to tell grass and foliage
    from paving and nothing like enough to tell them from green paint.
    """
    high = channel_max(linear_rgb)
    low = channel_min(linear_rgb)
    chroma = high - low
    saturation = chroma / np.maximum(high, 1e-4)
    red, green, blue = linear_rgb[..., 0], linear_rgb[..., 1], linear_rgb[..., 2]
    # Hue, needed only where green is the largest channel.
    hue = 60.0 * (2.0 + (blue - red) / np.maximum(chroma, 1e-6))
    in_band = (green >= high - 1e-6) & (hue > _VEGETATION_HUE_DEG[0]) & (
        hue < _VEGETATION_HUE_DEG[1]
    )
    strength = np.clip((saturation - _VEGETATION_MIN_SATURATION) / 0.12, 0.0, 1.0)
    vegetation = (in_band * strength).astype(np.float32)
    sigma = max(1.0, 0.004 * float(np.hypot(*linear_rgb.shape[:2])))
    return cv2.GaussianBlur(vegetation, (0, 0), sigmaX=sigma).astype(np.float32)


def level_ground_weight(upward_cosine: np.ndarray, falloff: float) -> np.ndarray:
    """``[0, 1]`` weight of level ground from the upward-facing cosine.

    The power says how fast the weight falls on a slope. On its own it left
    a real road, whose estimated normals wander a few degrees, at 40-70% of
    full weight, so a soaked road came out half dry and a buried one half
    bare. Level ground is level: the weight saturates.
    """
    return np.clip(upward_cosine**falloff / _LEVEL_GROUND_WEIGHT, 0.0, 1.0).astype(np.float32)


def fill_to_edges(ground: np.ndarray, linear_rgb: np.ndarray) -> np.ndarray:
    """Extend level ground over the unreadable band round each object.

    The depth map blurs the edge of everything, and the band of road inside
    that blur has no orientation to read, so it was left out: a dry, bare
    outline round each car. The ground is grown outwards by a few pixels,
    and only over pixels that look like the ground already found next to
    them, so the road under a bumper is taken in and the bumper is not.
    """
    diagonal = float(np.hypot(*ground.shape))
    reach = max(3, round(_EDGE_FILL_REACH * diagonal) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (reach, reach))
    grown = cv2.dilate(ground, kernel)
    # The colour of the ground nearby: an average over ground pixels only.
    sigma = 0.6 * reach
    weight = wide_blur(ground, sigma, work_sigma=_EDGE_FILL_WORK_SIGMA)
    colour = wide_blur(linear_rgb * ground[..., None], sigma, work_sigma=_EDGE_FILL_WORK_SIGMA)
    # The comparison and the grown mask are one compiled pass where there is
    # one, with this arithmetic in this order: the same mask.
    done = _native.grow_alike(
        linear_rgb, colour, weight, ground, grown, None, 0.0, 1.0, _EDGE_FILL_TOLERANCE, 1e-4, 0.02
    )
    if done is not None:
        return done
    colour = colour / np.maximum(weight, 1e-4)[..., None]
    distance = channel_max(np.abs(linear_rgb - colour))
    alike = np.clip(2.0 - distance / _EDGE_FILL_TOLERANCE, 0.0, 1.0)
    return np.maximum(ground, grown * alike * (weight > 0.02)).astype(np.float32)


def _ground_average(values: np.ndarray, ground: np.ndarray, sigma: float) -> np.ndarray:
    """The average of ``values`` over the ground near each pixel, by ``ground``'s weight."""
    # An average that only has to be right to a few per cent: blurred coarsely.
    coarse = _EDGE_FILL_WORK_SIGMA
    weight = wide_blur(ground, sigma, work_sigma=coarse)
    if values.ndim == 3:
        total = wide_blur(values * ground[..., None], sigma, work_sigma=coarse)
        return (total / np.maximum(weight, 1e-3)[..., None]).astype(np.float32)
    total = wide_blur(values * ground, sigma, work_sigma=coarse)
    return (total / np.maximum(weight, 1e-3)).astype(np.float32)


def standing_objects(depth_m: np.ndarray, linear_rgb: np.ndarray, ground: np.ndarray) -> np.ndarray:
    """``[0, 1]`` weight of what stands on the ground and was read as part of it.

    A car a hundred metres off is forty pixels of a depth map that was
    estimated at five hundred, and its roof, bonnet and sides come out as
    one smooth bump with the orientation of the road it stands on. Nothing
    in a surface normal tells it from road, and a wet road then darkened
    every vehicle on it and laid reflections over them.

    Two things do tell it, and neither alone. The bump itself: the car is
    **nearer than the ground around it**, by a few per cent, where a lamp's
    pool of light, a shadow or a reflection already in the road is not. And
    its look: within reach of such a bump, what is **unlike the road beside
    it** is the object, and what is like it is the road at its foot, which
    the depth map's soft edges would otherwise take with it and leave as a
    dry rim round every wheel. The look alone was tried: it also took out
    the dark half of a road lit by one lamp, and the reflections on a road
    that was wet to begin with.

    ``ground`` is the weight of level ground as found so far. The ground's
    distance is taken out row by row before anything is compared, because
    ground a row down the frame is nearer by nature, and an average taken at
    the frame's bottom edge has only further ground to draw on.
    """
    height, width = ground.shape
    # Every step below is an average over a few per cent of the frame, of
    # things a car's size or more: in fast mode it is all done on a copy
    # half the size, and the weight is enlarged.
    if mode.fast() and min(height, width) >= _OBJECT_FULL_SIZE:
        half = (width // 2, height // 2)
        small = standing_objects(
            cv2.resize(depth_m.astype(np.float32), half, interpolation=cv2.INTER_AREA),
            cv2.resize(linear_rgb, half, interpolation=cv2.INTER_AREA),
            cv2.resize(ground, half, interpolation=cv2.INTER_AREA),
        )
        enlarged = cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)
        return np.clip(enlarged, 0.0, 1.0).astype(np.float32)
    diagonal = float(np.hypot(height, width))
    weight = np.maximum(ground, 1e-3).astype(np.float32)
    log_depth = np.log(np.maximum(depth_m, 1e-6)).astype(np.float32)
    by_row = (log_depth * weight).sum(axis=1) / weight.sum(axis=1)
    relief = log_depth - by_row.reshape(height, 1).astype(np.float32)
    nearer = _ground_average(relief, weight, _OBJECT_WINDOW * diagonal) - relief

    def ramp(value: np.ndarray, bounds: tuple[float, float]) -> np.ndarray:
        """0 below the first bound, 1 above the second."""
        return np.clip((value - bounds[0]) / (bounds[1] - bounds[0]), 0.0, 1.0).astype(np.float32)

    reach = max(3, round(_OBJECT_REACH * diagonal) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (reach, reach))
    near = cv2.dilate(ramp(nearer, _OBJECT_NEARER), kernel)
    near = cv2.GaussianBlur(near, (0, 0), sigmaX=0.25 * reach)

    # Compared a little out of focus, so that the grain of a road is not
    # taken for a difference, and with a floor under the logarithm, so that
    # the noise in a night frame's blacks is not either.
    soft = cv2.GaussianBlur(linear_rgb, (0, 0), sigmaX=1.0)
    floor = np.float32(0.02 * float(soft.mean()) + 1e-4)
    road = _ground_average(soft, weight * (1.0 - near), 0.04 * diagonal)
    unlike = ramp(channel_max(np.abs(np.log((soft + floor) / (road + floor)))), _OBJECT_UNLIKE)
    return np.maximum(ramp(nearer, _OBJECT_FOR_CERTAIN), near * unlike).astype(np.float32)


def small_things(linear_rgb: np.ndarray, ground: np.ndarray) -> np.ndarray:
    """``[0, 1]`` weight of small things on the ground that are not the ground, ``(H, W)``.

    A car four hundred metres off is a dozen pixels, and no depth map shows
    it standing: :func:`standing_objects` needs a bump to start from. But
    the far end of a wet road is where it is most nearly a mirror, and a
    mirror laid over those dozen pixels takes the car out of the picture.
    What a thing that small still is, is unlike the road *immediately* round
    it: so this is every pixel that differs from the average of the ground
    within a few of its own widths.

    It is deliberately a test of small things. Told nothing about size, the
    look of a pixel also takes out the dark half of a road lit by one lamp
    and the reflections on a road that was wet to begin with; those are
    broad, and an average taken this close goes with them. It does keep a
    lane marking as it is, which under a film at a grazing angle really
    fades into the sheen, and is kept here because a marking that has been
    painted over is a worse fault in a training frame than one too clear.
    """
    height, width = ground.shape
    # On a copy half the size in fast mode, as `standing_objects` is: a
    # thing a dozen pixels across is still six, and the comparison is of
    # averages.
    if mode.fast() and min(height, width) >= _OBJECT_FULL_SIZE:
        half = (width // 2, height // 2)
        small = small_things(
            cv2.resize(linear_rgb, half, interpolation=cv2.INTER_AREA),
            cv2.resize(ground, half, interpolation=cv2.INTER_AREA),
        )
        enlarged = cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)
        return np.clip(enlarged, 0.0, 1.0).astype(np.float32)
    diagonal = float(np.hypot(height, width))
    soft = cv2.GaussianBlur(linear_rgb, (0, 0), sigmaX=1.0)
    floor = np.float32(0.02 * float(soft.mean()) + 1e-4)
    weight = np.maximum(ground, 1e-3).astype(np.float32)
    road = _ground_average(soft, weight, max(2.0, _SMALL_THING_WINDOW * diagonal))
    unlike = channel_max(np.abs(np.log((soft + floor) / (road + floor))))
    low, high = _SMALL_THING_UNLIKE
    return np.clip((unlike - low) / (high - low), 0.0, 1.0).astype(np.float32)


def ground_noise(
    depth_m: np.ndarray,
    focal_px: float,
    camera_pitch_deg: float,
    scale_m: float,
    rng: np.random.Generator,
    octaves: int = 3,
    uniform: bool = False,
) -> np.ndarray:
    """Noise in ``[0, 1]`` laid on the ground plane, ``(H, W)``.

    A pattern on a road is foreshortened with it: wide and thin towards the
    horizon, round at the camera's feet. Screen-space noise is neither, and
    reads as something floating in front of the scene. This samples a noise
    tile at each pixel's position on the ground, recovered from depth, with
    features about ``scale_m`` across.

    With ``uniform`` the values are spread evenly over ``[0, 1]``, so that
    thresholding at ``t`` selects a share ``t`` of the ground.
    """
    height, width = depth_m.shape
    noise = value_noise(_TILE_PX, _TILE_PX, rng=rng, octaves=octaves, base_cells=_TILE_CELLS)
    if uniform:
        ranks = np.argsort(np.argsort(noise, axis=None)).astype(np.float32)
        noise = (ranks / (noise.size - 1)).reshape(noise.shape)

    pitch = np.deg2rad(camera_pitch_deg)
    offset = rng.uniform(0.0, 100.0, size=2)
    period = 2.0 * (_TILE_PX - 1)
    if mode.fast():
        # Both maps in one pass, where the compiled kernels are there: the
        # arithmetic below, which is a dozen passes over the frame.
        how = [focal_px, np.cos(pitch), np.sin(pitch), _TILE_PX / (_TILE_CELLS * scale_m)]
        places = _native.ground_tile(depth_m, [*how, period, 1.0 / period, *offset])
        if places is not None:
            return cv2.remap(noise, places[0], places[1], interpolation=cv2.INTER_LINEAR)
    xs = (np.arange(width, dtype=np.float32) - width / 2.0).reshape(1, -1)
    ys = (np.arange(height, dtype=np.float32) - height / 2.0).reshape(-1, 1)
    across = xs * depth_m / focal_px
    down = ys * depth_m / focal_px
    along = depth_m * np.float32(np.cos(pitch)) - down * np.float32(np.sin(pitch))

    def tile(metres: np.ndarray) -> np.ndarray:
        # Reflected at the tile's edges, so the pattern repeats without a seam.
        """Ground coordinates in metres as coordinates in the noise tile, folded into it.

        The tile is reflected at its edges, so the pattern repeats without a
        seam. Fast mode folds with a floor where exact mode takes a remainder.
        """
        pixels = metres * (_TILE_PX / (_TILE_CELLS * scale_m))
        if mode.fast():
            # The same fold by a floor: `np.mod` on a frame of floats was a
            # tenth of a settled-snow frame, and this is a tenth of that. It
            # lands within a few millionths of a pixel of it.
            span = np.float32(period)
            folded = pixels - span * np.floor(pixels * np.float32(1.0 / period))
            return np.minimum(folded, span - folded).astype(np.float32)
        folded = np.mod(pixels, period)
        return np.where(folded > period / 2.0, period - folded, folded).astype(np.float32)

    return cv2.remap(
        noise,
        tile(across + mode.scalar(offset[0])),
        tile(along + mode.scalar(offset[1])),
        interpolation=cv2.INTER_LINEAR,
    )


def small_level_areas(level: np.ndarray, max_width_fraction: float) -> np.ndarray:
    """``[0, 1]`` weight of level patches too small to be ground, ``(H, W)``.

    A car's roof and bonnet are level, and nothing in a surface normal says
    they are not road. What does is that they are islands: patches of level
    surface a car's width across, walled off from the road by the car's
    sides. So this is every connected patch of level surface that fits in a
    box ``max_width_fraction`` of the diagonal on a side. Give it the level
    weight as read from the normals, before it is grown to object edges,
    which is what joins a roof to the road beside it.

    (Taking what a morphological opening removes was tried first. It also
    removes the road between two cars, and leaves the open road as a set of
    discs.)
    """
    limit = max_width_fraction * float(np.hypot(*level.shape))
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        (level > 0.5).astype(np.uint8), connectivity=4
    )
    small = np.zeros(count, dtype=np.float32)
    widths = stats[:, cv2.CC_STAT_WIDTH]
    heights = stats[:, cv2.CC_STAT_HEIGHT]
    small[1:] = ((widths[1:] <= limit) & (heights[1:] <= limit)).astype(np.float32)
    islands = small[labels]
    # Out to the island's own edge, which the threshold stops short of.
    grown = cv2.dilate(islands, np.ones((5, 5), np.uint8))
    return cv2.GaussianBlur(grown, (0, 0), sigmaX=1.0).astype(np.float32)
