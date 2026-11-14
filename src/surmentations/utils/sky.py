"""Finding the sky, with a segmentation model or without one.

Several effects need to know where the sky is: fog must not flatten it,
overcast light has to grey it, and a wet road reflects it. Depth alone does
not find it by being far: the depth network puts sky at the far plane in
some frames, at mid-range in others and, over a city skyline, *nearer* than
the buildings under it (relative depth 0.17). A far-depth threshold found
sky in one of nine test scenes.

What sky reliably is, in a photograph, is three things at once:

* **featureless** at the scale of a few pixels, where almost nothing on the
  ground is;
* **at one depth**: wherever the network puts the sky, it puts all of it
  there, to three decimal places in every scene measured. A ceiling or a
  wall filling the top of the picture recedes, and that is what separates
  them. It is also never the nearest thing in frame;
* **reached from the top of the frame** without crossing anything that fails
  the first two. A pale wall in the middle of the picture is not sky; nor is
  a smooth road.

By day it is also not among the darker things in view, and that is
required too. At night it is not, so the test is
dropped when the frame as a whole is dark.

This is a heuristic and is documented as one. It misses sky seen through
branches, and it will take a smooth, distant, pale surface that touches the
top edge — a snowfield, a white facade — for sky.

Given the scene's labels (:mod:`surmentations.perception.labels`) none of
it is used: the sky is what the network called sky, which is right in both
of those cases and in every sample scene. Its outline is the network's, a
few pixels soft, and is redrawn against the depth map's own edge by
:func:`sky_to_depth_edge` either way.
"""

from __future__ import annotations

import cv2
import numpy as np

from surmentations import _native
from surmentations.perception.labels import SceneClass, class_weight
from surmentations.utils.channels import channel_max
from surmentations.utils.filters import wide_blur, wide_dilate
from surmentations.utils.photometry import luminance

# Detection runs on a copy this many pixels wide: sky is a large region, and
# texture measured at a fixed size makes the thresholds resolution-independent.
_WORK_WIDTH = 256

# Local contrast above which a pixel is "textured", as a fraction of the
# local mean in gamma-encoded luminance. Clouds are a few percent; foliage,
# masonry and asphalt at this scale are several times that.
_MAX_TEXTURE = 0.045

# Spread of relative depth (20th to 80th percentile) a region may have and
# still be sky. Measured sky is flat to 0.001; a car park's ceiling spans
# several tenths. The middle of the distribution, because a region's edge
# takes in a little of whatever borders it.
_MAX_DEPTH_SPREAD = 0.03

# And it is never the nearest thing in frame, which a flat soffit right above
# the camera can be.
_MIN_RELATIVE_DEPTH = 0.08

# By day, sky is at least this bright relative to the frame's median, in
# gamma-encoded luminance. Below one on purpose: a hazy sky over a sunlit
# concrete road is the darker of the two.
_MIN_DAY_BRIGHTNESS = 0.75

# Mean gamma-encoded luminance below which the frame is taken to be night,
# and sky is no longer expected to be bright.
_NIGHT_LEVEL = 0.22


def sky_weight(
    linear_rgb: np.ndarray,
    relative_depth: np.ndarray | None = None,
    labels: np.ndarray | None = None,
) -> np.ndarray:
    """Soft ``[0, 1]`` mask of sky, shape ``(H, W)``, float32.

    ``linear_rgb`` is linear light in ``[0, 1]``. ``relative_depth`` is the
    depth provider's output; without it a ceiling cannot be told from sky.
    ``labels`` are the scene's classes, where the provider has them: the sky
    is then the pixels labelled sky, feathered as the heuristic's mask is.
    """
    height, width = linear_rgb.shape[:2]
    if labels is not None:
        feather = 1.5 * width / _WORK_WIDTH
        return class_weight(labels, (SceneClass.SKY,), soften_px=max(1.0, feather))
    scale = _WORK_WIDTH / max(width, 1)
    size = (max(8, round(width * scale)), max(8, round(height * scale)))
    small = cv2.resize(linear_rgb, size, interpolation=cv2.INTER_AREA)
    # Texture is judged the way it looks, so in gamma-encoded terms.
    luma = np.sqrt(np.clip(luminance(small), 0.0, 1.0))

    mean = cv2.blur(luma, (5, 5))
    spread = np.sqrt(np.maximum(cv2.blur(luma * luma, (5, 5)) - mean * mean, 0.0))
    candidate = spread < _MAX_TEXTURE * np.maximum(mean, 0.25)

    if float(luma.mean()) > _NIGHT_LEVEL:
        candidate &= luma > _MIN_DAY_BRIGHTNESS * float(np.median(luma))

    # Keep what is connected to the top edge and lies at one depth.
    _count, labels = cv2.connectedComponents(candidate.astype(np.uint8), connectivity=8)
    mask = np.zeros(candidate.shape, dtype=np.float32)
    depth = None
    if relative_depth is not None:
        depth = cv2.resize(relative_depth, size, interpolation=cv2.INTER_AREA)
    for label in np.unique(labels[0]):
        if label == 0:
            continue
        region = labels == label
        if depth is not None:
            low, middle, high = np.percentile(depth[region], (20.0, 50.0, 80.0))
            if high - low > _MAX_DEPTH_SPREAD or middle < _MIN_RELATIVE_DEPTH:
                continue
        mask[region] = 1.0
    if not mask.any():
        return np.zeros((height, width), dtype=np.float32)
    # Close the pinholes texture leaves in cloud, then feather.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=1.5)
    return np.clip(cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR), 0.0, 1.0)


# Koschmieder: visibility is the distance at which contrast falls to 2%,
# exp(-3.912).
KOSCHMIEDER = 3.912

# Share of what is not sky that lies nearer than a scene's extent. Short of
# all of it, so that one far pixel is not the scene.
_EXTENT_SHARE = 98.0


def scene_extent_m(depth_m: np.ndarray, sky: np.ndarray) -> float:
    """How deep a scene is: the distance of the farthest thing in it that is not sky.

    A fog of one visibility is a different picture on every scene: a hundred
    metres hides nothing of a car park and all of a valley. An effect asked
    for a visibility *relative to the scene* multiplies by this. It is the
    98th percentile of the depth of what is not sky, and the greatest depth
    in a frame that is all sky.
    """
    ground = depth_m[sky < 0.5]
    if ground.size == 0:
        return float(depth_m.max())
    return float(np.percentile(ground, _EXTENT_SHARE))


def extinction_for_visibility(visibility_m: float) -> float:
    """The extinction coefficient, per metre, of air one can see ``visibility_m`` through."""
    return KOSCHMIEDER / max(visibility_m, 1e-6)


def lift_sky(depth_m: np.ndarray, sky: np.ndarray) -> np.ndarray:
    """Put the sky at the far end of a depth map.

    The network's depth for sky is arbitrary — mid-range, or nearer than the
    skyline under it — so anything that attenuates with distance treats the
    sky as closer than the buildings in front of it: fog thickens over a
    city and leaves the sky above it blue. The far plane is not right
    either, since the sky has no distance, but it is the limit every
    distance-dependent effect here converges to.
    """
    far = float(depth_m.max())
    return (depth_m * (1.0 - sky) + far * sky).astype(depth_m.dtype)


# How far the sky mask is allowed to grow to meet the depth map's own edge,
# as a fraction of the diagonal, and how close in relative depth a pixel has
# to be to the sky's to be let in.
_SKY_EDGE_REACH = 0.04
_SKY_DEPTH_TOLERANCE = 0.04


# How unlike the sky's own colour, in linear RGB, a pixel may be and still be
# let into the mask when it is redrawn.
_SKY_COLOUR_TOLERANCE = 0.06

# How many times further from the sky's depth a pixel already in the mask
# may be than one being added to it.
_SKY_KEPT_TOLERANCE = 4.0


def sky_to_depth_edge(
    sky: np.ndarray, relative_depth: np.ndarray, linear_rgb: np.ndarray | None = None
) -> np.ndarray:
    """Redraw the sky mask's outline where the depth map puts it.

    The mask is made at low resolution and its edge wanders a few percent of
    the frame either side of the real skyline. By day that is mostly
    hidden, a veil being about as bright as the sky. At night it is not:
    fog and snow veils are brighter than the sky, so every patch of sky the
    mask misses next to a tree stays a dark hole, and every patch of tree it
    takes in is veiled as if it were miles away. The depth map has the
    skyline to the pixel; what it lacks is which side is sky, which the mask
    knows.

    Depth alone is not enough to let a pixel in. A tower a mile off is at
    the sky's depth as far as the network can tell, and growing the mask
    over its edge drew a pale outline round the whole skyline. With
    ``linear_rgb`` given, what is added must also be the sky's colour.
    """
    core = sky > 0.9
    if not core.any():
        return sky
    level = float(np.median(relative_depth[core]))
    reach = max(3, round(_SKY_EDGE_REACH * float(np.hypot(*sky.shape))) | 1)
    grown = wide_dilate(sky, reach)

    # Everything below is arithmetic on a pixel and what the mask, grown,
    # says of it. Below the lowest row the grown mask reaches both are zero
    # and so is the result, so it is done on the rows in between: the same
    # values there, and in a street scene half the frame not gone over.
    # (numpy's maximum along a row, not OpenCV's: `cv2.reduce` with
    # REDUCE_MAX across rows took 8 ms on a 1024-pixel frame, this 0.4.)
    reached = np.flatnonzero((grown.max(axis=1) > 0) | (sky.max(axis=1) > 0))
    band = slice(int(reached[0]), int(reached[-1]) + 1)
    sky_band, grown_band, depth_band = sky[band], grown[band], relative_depth[band]
    redrawn = np.zeros(sky.shape, dtype=np.float32)

    # What the mask already holds is kept unless the depth map plainly says
    # otherwise (`plainly_not`, below). Depth is blurred across every
    # skyline, and holding the mask's own pixels to the tolerance used for
    # new ones cut a band of sky out along it: left as photographed while the
    # rest was relit, a pale outline round every building.
    if linear_rgb is not None:
        # The sky's colour near each pixel, since a sky is not one colour.
        sigma = float(reach)
        weight = wide_blur(sky, sigma)[band]
        of_sky = np.zeros_like(linear_rgb)
        of_sky[band] = linear_rgb[band] * sky_band[..., None]
        nearby_sum = wide_blur(of_sky, sigma)[band]
        # What follows, the comparison of each pixel with the sky's colour
        # near it and the redrawn mask, is one compiled pass where there is
        # one, with this arithmetic in this order: the same mask.
        done = _native.grow_alike(
            linear_rgb[band],
            nearby_sum,
            weight,
            sky_band,
            grown_band,
            depth_band,
            level,
            _SKY_KEPT_TOLERANCE * _SKY_DEPTH_TOLERANCE,
            _SKY_COLOUR_TOLERANCE,
            1e-4,
            0.02,
        )
        if done is not None:
            redrawn[band] = done
            return redrawn
        plainly_not = np.clip(
            2.0 - np.abs(depth_band - level) / (_SKY_KEPT_TOLERANCE * _SKY_DEPTH_TOLERANCE),
            0.0,
            1.0,
        )
        kept = sky_band * plainly_not
        nearby = nearby_sum / np.maximum(weight, 1e-4)[..., None]
        unlike = channel_max(np.abs(linear_rgb[band] - nearby))
        alike = np.clip(2.0 - unlike / _SKY_COLOUR_TOLERANCE, 0.0, 1.0) * (weight > 0.02)
        # With colour to go by, depth only has to not contradict it: the
        # last pixels of sky before a roofline are sky-coloured and, in a
        # depth map blurred across that roofline, already half building.
        redrawn[band] = np.maximum(kept, grown_band * alike * plainly_not)
        return redrawn
    plainly_not = np.clip(
        2.0 - np.abs(depth_band - level) / (_SKY_KEPT_TOLERANCE * _SKY_DEPTH_TOLERANCE),
        0.0,
        1.0,
    )
    kept = sky_band * plainly_not
    near_level = np.clip(2.0 - np.abs(depth_band - level) / _SKY_DEPTH_TOLERANCE, 0.0, 1.0)
    redrawn[band] = np.maximum(kept, grown_band * near_level)
    return redrawn
