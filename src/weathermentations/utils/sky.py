"""Finding the sky without a segmentation model.

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
"""

from __future__ import annotations

import cv2
import numpy as np

from weathermentations import mode
from weathermentations.utils.channels import channel_max
from weathermentations.utils.photometry import luminance

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


def sky_weight(linear_rgb: np.ndarray, relative_depth: np.ndarray | None = None) -> np.ndarray:
    """Soft ``[0, 1]`` mask of sky, shape ``(H, W)``, float32.

    ``linear_rgb`` is linear light in ``[0, 1]``. ``relative_depth`` is the
    depth provider's output; without it a ceiling cannot be told from sky.
    """
    height, width = linear_rgb.shape[:2]
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


# The sigma, in pixels, a wide blur is run at on its reduced copy. At 8 the
# result is within 0.2% of the full-size blur's peak; at 4 it was 0.7%.
_WIDE_BLUR_WORK_SIGMA = 8.0


def wide_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur for a sigma of many pixels, computed on a smaller copy.

    A Gaussian filter's cost grows with its sigma, and a result blurred that
    far holds no detail a smaller image could not: the frame is reduced until
    the sigma is a few pixels, blurred there, and enlarged. The two blurs in
    :func:`sky_to_depth_edge` have a sigma of 4% of the diagonal, 58 pixels
    on a 1024-pixel frame. On one core, which is what a ``DataLoader``
    worker has, they took 400 ms of every fog, rain and snow frame; this way
    they take 5. Below twice the working sigma, and in exact mode
    (:mod:`weathermentations.mode`), it is the plain blur.
    """
    factor = int(sigma / _WIDE_BLUR_WORK_SIGMA) if mode.fast() else 1
    if factor < 2:
        return cv2.GaussianBlur(image, (0, 0), sigmaX=sigma)
    height, width = image.shape[:2]
    size = (max(1, round(width / factor)), max(1, round(height / factor)))
    small = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigmaX=sigma * size[0] / width)
    return cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)


# The diameter, in pixels, a wide dilation's disc has on its reduced copy.
_WIDE_DILATE_WORK_PX = 9


def wide_dilate(mask: np.ndarray, diameter: int) -> np.ndarray:
    """Grey dilation of an ``(H, W)`` mask by a disc many pixels across.

    A disc is not a separable structuring element, so OpenCV visits every
    pixel of it for every pixel of the image: 59 pixels across on a
    1024-pixel frame is 80 ms on one core. The mask is first reduced by
    taking the maximum of each block, which loses nothing a dilation would
    keep, then dilated by a disc of a few pixels and enlarged: 2 ms. It
    agrees with the exact dilation except in a band about one block wide
    where the reach ends, which it draws soft. Good for a bound on how far
    something may grow, which is its one use here; not for a shape whose
    outline matters. Below twice the working diameter, and in exact mode,
    it is the plain dilation.
    """
    factor = diameter // _WIDE_DILATE_WORK_PX if mode.fast() else 1
    if factor < 2:
        return cv2.dilate(
            mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
        )
    height, width = mask.shape[:2]
    # A square dilation is separable and cheap; one sample from the middle of
    # each block of it is that block's maximum.
    pooled = cv2.dilate(mask, np.ones((factor, factor), np.uint8))
    small = np.ascontiguousarray(pooled[factor // 2 :: factor, factor // 2 :: factor])
    disc = max(3, round(diameter / factor) | 1)
    grown = cv2.dilate(small, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (disc, disc)))
    return cv2.resize(grown, (width, height), interpolation=cv2.INTER_LINEAR)


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
    near_level = np.clip(
        2.0 - np.abs(relative_depth - level) / _SKY_DEPTH_TOLERANCE, 0.0, 1.0
    )
    # What the mask already holds is kept unless the depth map plainly says
    # otherwise. Depth is blurred across every skyline, and holding the
    # mask's own pixels to the tolerance used for new ones cut a band of sky
    # out along it: left as photographed while the rest was relit, a pale
    # outline round every building.
    plainly_not = np.clip(
        2.0 - np.abs(relative_depth - level) / (_SKY_KEPT_TOLERANCE * _SKY_DEPTH_TOLERANCE),
        0.0,
        1.0,
    )
    kept = sky * plainly_not
    reach = max(3, round(_SKY_EDGE_REACH * float(np.hypot(*sky.shape))) | 1)
    grown = wide_dilate(sky, reach)
    if linear_rgb is not None:
        # The sky's colour near each pixel, since a sky is not one colour.
        sigma = float(reach)
        weight = wide_blur(sky, sigma)
        nearby = wide_blur(linear_rgb * sky[..., None], sigma)
        nearby = nearby / np.maximum(weight, 1e-4)[..., None]
        unlike = channel_max(np.abs(linear_rgb - nearby))
        alike = np.clip(2.0 - unlike / _SKY_COLOUR_TOLERANCE, 0.0, 1.0) * (weight > 0.02)
        # With colour to go by, depth only has to not contradict it: the
        # last pixels of sky before a roofline are sky-coloured and, in a
        # depth map blurred across that roofline, already half building.
        return np.maximum(kept, grown * alike * plainly_not).astype(np.float32)
    return np.maximum(kept, grown * near_level).astype(np.float32)
