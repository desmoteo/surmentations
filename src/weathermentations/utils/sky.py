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
