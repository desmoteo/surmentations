"""Reading the ground off a depth map and a picture.

Rain wets the ground and snow settles on it, and both need the same three
things: which pixels are level ground, what kind of ground it is, and a way
to lay a pattern on it so that it foreshortens with the road instead of
sitting on the screen.
"""

from __future__ import annotations

import cv2
import numpy as np

from weathermentations.utils.noise import value_noise

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

# Noise tile that ground patterns are cut from, in pixels and in cells.
_TILE_PX = 256
_TILE_CELLS = 8


def vegetation_weight(linear_rgb: np.ndarray) -> np.ndarray:
    """``[0, 1]`` weight of pixels that are green vegetation, ``(H, W)``.

    By colour alone, which is all that is needed to tell grass and foliage
    from paving and nothing like enough to tell them from green paint.
    """
    high = linear_rgb.max(axis=2)
    low = linear_rgb.min(axis=2)
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
    weight = cv2.GaussianBlur(ground, (0, 0), sigmaX=sigma)
    colour = cv2.GaussianBlur(linear_rgb * ground[..., None], (0, 0), sigmaX=sigma)
    colour = colour / np.maximum(weight, 1e-4)[..., None]
    distance = np.abs(linear_rgb - colour).max(axis=2)
    alike = np.clip(2.0 - distance / _EDGE_FILL_TOLERANCE, 0.0, 1.0)
    return np.maximum(ground, grown * alike * (weight > 0.02)).astype(np.float32)


def ground_noise(
    depth_m: np.ndarray,
    focal_px: float,
    camera_pitch_deg: float,
    scale_m: float,
    rng: np.random.Generator,
    octaves: int = 3,
) -> np.ndarray:
    """Noise in ``[0, 1]`` laid on the ground plane, ``(H, W)``.

    A pattern on a road is foreshortened with it: wide and thin towards the
    horizon, round at the camera's feet. Screen-space noise is neither, and
    reads as something floating in front of the scene. This samples a noise
    tile at each pixel's position on the ground, recovered from depth, with
    features about ``scale_m`` across.
    """
    height, width = depth_m.shape
    noise = value_noise(_TILE_PX, _TILE_PX, rng=rng, octaves=octaves, base_cells=_TILE_CELLS)

    pitch = np.deg2rad(camera_pitch_deg)
    xs = (np.arange(width, dtype=np.float32) - width / 2.0).reshape(1, -1)
    ys = (np.arange(height, dtype=np.float32) - height / 2.0).reshape(-1, 1)
    across = xs * depth_m / focal_px
    down = ys * depth_m / focal_px
    along = depth_m * np.float32(np.cos(pitch)) - down * np.float32(np.sin(pitch))

    def tile(metres: np.ndarray) -> np.ndarray:
        # Reflected at the tile's edges, so the pattern repeats without a seam.
        pixels = metres * (_TILE_PX / (_TILE_CELLS * scale_m))
        period = 2.0 * (_TILE_PX - 1)
        folded = np.mod(pixels, period)
        return np.where(folded > period / 2.0, period - folded, folded).astype(np.float32)

    offset = rng.uniform(0.0, 100.0, size=2)
    return cv2.remap(
        noise, tile(across + offset[0]), tile(along + offset[1]), interpolation=cv2.INTER_LINEAR
    )
