"""Steam and exhaust: fog in one place.

Physical model
--------------
Cold weather puts small clouds into a street scene that no uniform fog
describes: steam from a manhole or a vent, the exhaust of an idling engine,
a boiler flue. Each is a plume: a volume of droplets a metre or so across
at its source, rising, spreading and thinning until it is gone a few
metres up, and leaning with whatever wind there is.

**It is somewhere.** A plume stands at one place on the ground, so it has
a distance. Its source is put on a point of the scene itself, read from
the depth map, and its width and height in pixels follow from metres at
that distance: the same vent is a wisp across the street and a wall beside
the camera.

**It hides only what is behind it.** A car nearer than the plume is in
front of it. Each pixel is veiled by the plume's optical depth there only
if the scene at that pixel is further away than the plume is.

**It spreads and thins as it rises.** The column widens with height and
the same water is spread over a wider section, so its optical depth falls
from the source up, and it ends by fading, not at an edge. Turbulence
breaks it into billows the size of its own width: a fractal noise
modulates the density and bends the column.

**It is white, and it shades itself.** Droplets scatter all colours alike
and absorb none. The plume takes the colour and level of the light
falling on it, estimated as for rain and snow from the frame;
``brightness_range`` scales that, for exhaust in shadow or steam in the
sun. The light comes from above, through the plume's own upper part, so
each billow is brightest on top and greyer underneath by the optical
depth above it: what gives steam its modelling and stops it being a flat
white shape.
"""

from __future__ import annotations

from typing import Annotated, Any

import numpy as np
from albumentations.core.pydantic import check_range_bounds, nondecreasing
from albumentations.core.transforms_interface import BaseTransformInitSchema
from pydantic import AfterValidator, ConfigDict, Field

from weathermentations.base import WeatherTransform, scene_of
from weathermentations.depth.provider import SharedDepthProvider
from weathermentations.diagnostics import record_draw
from weathermentations.utils.color import illuminant_radiance_linear
from weathermentations.utils.noise import value_noise
from weathermentations.utils.photometry import from_linear_unit, to_linear_unit
from weathermentations.utils.precipitation import focal_length_px

# How many times wider than its source a plume is where it ends.
_SPREAD = 3.5
# How far in front of the plume's distance the scene starts to be veiled,
# as a share of that distance: a plume has depth of its own.
_DEPTH_SOFTNESS = 0.08
# How much of the light from above a plume's own optical depth takes off
# what is under it, at most.
_SELF_SHADE = 0.45


def plume_depth(
    rng: np.random.Generator,
    height: int,
    width: int,
    *,
    source: tuple[float, float],
    width_px: float,
    rise_px: float,
    lean: float,
    peak: float,
) -> np.ndarray:
    """Optical depth of one plume, ``(H, W)`` float32, seen side on.

    It rises ``rise_px`` from ``source`` (x, y), leaning ``lean`` pixels
    across for each pixel up, from a Gaussian section ``width_px`` wide to
    one ``_SPREAD`` times that. ``peak`` is its optical depth through the
    middle at the source; above that it falls as the section widens, and
    to nothing at the top. Billows the size of the column break it up.
    """
    yy = np.arange(height, dtype=np.float32)[:, None]
    xx = np.arange(width, dtype=np.float32)[None, :]
    up = (source[1] - yy) / max(rise_px, 1.0)
    inside = (up >= -0.05) & (up <= 1.0)
    climbed = np.clip(up, 0.0, 1.0)
    section = width_px * (1.0 + (_SPREAD - 1.0) * climbed)

    # Billows: noise on the plume's own scale, which also bends the column.
    cells = int(np.clip(max(height, width) / max(width_px, 2.0), 2, 48))
    billow = value_noise(height, width, rng=rng, octaves=4, base_cells=cells)
    wander = value_noise(height, 1, rng=rng, octaves=2, base_cells=max(2, cells // 3))
    centre = source[0] + lean * (source[1] - yy) + (wander - 0.5) * 2.0 * width_px * climbed

    across = (xx - centre) / (0.5 * section)
    # The same water over a wider section, and gone by the top.
    thinning = (1.0 / (1.0 + (_SPREAD - 1.0) * climbed)) * np.sqrt(1.0 - climbed)
    depth = peak * np.exp(-0.5 * across * across) * thinning * np.clip(2.4 * billow - 0.3, 0.0, 2.0)
    # It leaves its source as a rounded puff, not along a line: it fades in
    # over the first tenth of its rise, from a little below the source.
    start = np.clip((up + 0.05) / 0.15, 0.0, 1.0)
    depth = depth * (start * start * (3.0 - 2.0 * start))
    out: np.ndarray = np.where(inside, depth, 0.0).astype(np.float32)
    return out


class LocalPlumes(WeatherTransform):
    """Plumes of steam or exhaust standing in the scene.

    Depth-conditioned: each plume is placed on a point of the scene and
    hides only what is behind it, so it needs a ``depth_provider``.

    Args:
        depth_provider: one shared provider.
        plume_count_range: how many plumes.
        source_width_range_m: how wide a plume is where it starts.
        rise_range_m: how high it rises before it is gone.
        opacity_range: its optical depth through the middle at the source.
            At 1 it hides two thirds of what is behind it there; at 3,
            nearly all.
        lean_range: how far it drifts sideways for each metre it rises:
            the wind.
        brightness_range: how bright the plume is, as a share of the light
            the frame is lit by.
        horizontal_fov_range_deg: field of view, which turns metres at the
            plume's distance into pixels.
        depth_scale: pseudo-metres at relative depth 1, for this transform
            alone; ``None`` uses the provider's own far plane.

    Targets:
        image

    Image types:
        uint8, float32
    """

    class InitSchema(BaseTransformInitSchema):
        """The arguments of ``LocalPlumes`` as Albumentations validates them: each
        range ordered low to high and within the bounds given here.
        """

        model_config = ConfigDict(arbitrary_types_allowed=True)

        depth_provider: SharedDepthProvider
        plume_count_range: Annotated[
            tuple[int, int],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0, 20)),
        ]
        source_width_range_m: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.05, 50.0)),
        ]
        rise_range_m: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.2, 200.0)),
        ]
        opacity_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 20.0)),
        ]
        lean_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(-5.0, 5.0)),
        ]
        brightness_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 3.0)),
        ]
        horizontal_fov_range_deg: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(2.0, 150.0)),
        ]
        depth_scale: float | None = Field(default=None, gt=0.0)

    def __init__(
        self,
        depth_provider: SharedDepthProvider,
        plume_count_range: tuple[int, int] = (1, 3),
        source_width_range_m: tuple[float, float] = (0.3, 1.0),
        rise_range_m: tuple[float, float] = (1.5, 5.0),
        opacity_range: tuple[float, float] = (0.4, 2.0),
        lean_range: tuple[float, float] = (-0.6, 0.6),
        brightness_range: tuple[float, float] = (0.3, 0.8),
        horizontal_fov_range_deg: tuple[float, float] = (48.0, 78.0),
        depth_scale: float | None = None,
        p: float = 0.5,
    ):
        """Store the provider and the ranges; each is described in the class docstring."""
        super().__init__(p=p)
        self.depth_provider = depth_provider
        self.plume_count_range = plume_count_range
        self.source_width_range_m = source_width_range_m
        self.rise_range_m = rise_range_m
        self.opacity_range = opacity_range
        self.lean_range = lean_range
        self.brightness_range = brightness_range
        self.horizontal_fov_range_deg = horizontal_fov_range_deg
        self.depth_scale = depth_scale

    def get_params_dependent_on_data(
        self, params: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, Any]:
        """Put each plume on a point of the scene and work out what of it each pixel sees.

        Returns ``veiled``, the share of each pixel's light the plumes
        replace; ``shade``, how much of the light from above reaches each
        point of them; and their ``brightness``.
        """
        scene = scene_of(data)
        height, width = params["shape"][:2]
        rng = self.random_generator
        focal = focal_length_px(
            width, height, self.py_random.uniform(*self.horizontal_fov_range_deg)
        )
        depth_m = self.depth_provider.to_metric(
            self.depth_provider.get_depth(scene), far_m=self.depth_scale
        )

        clear = np.ones((height, width), dtype=np.float32)
        above = np.zeros((height, width), dtype=np.float32)
        distances = []
        for _ in range(self.py_random.randint(*self.plume_count_range)):
            # On the ground, which in a camera's view is the lower part of
            # the frame.
            x = float(rng.uniform(0.1, 0.9) * (width - 1))
            y = float(rng.uniform(0.45, 0.85) * (height - 1))
            distance = float(depth_m[int(y), int(x)])
            per_metre = focal / max(distance, 0.1)
            tau = plume_depth(
                rng,
                height,
                width,
                source=(x, y),
                width_px=self.py_random.uniform(*self.source_width_range_m) * per_metre,
                rise_px=self.py_random.uniform(*self.rise_range_m) * per_metre,
                lean=self.py_random.uniform(*self.lean_range),
                peak=self.py_random.uniform(*self.opacity_range),
            )
            behind = np.clip(
                (depth_m - distance * (1.0 - _DEPTH_SOFTNESS)) / (_DEPTH_SOFTNESS * distance),
                0.0,
                1.0,
            )
            clear *= np.exp(-tau * behind)
            # Optical depth between each point and the light overhead:
            # the column above it, counted across the plume's own width.
            above += np.cumsum(tau, axis=0) / max(float(tau.sum(axis=0).max()), 1e-6)
            distances.append(distance)

        record_draw(
            plume_count=len(distances),
            distance_m=np.array(distances, dtype=np.float32),
            focal_px=focal,
            veiled_fraction=lambda: float((clear < 0.5).mean()),
        )
        return {
            "veiled": 1.0 - clear,
            "shade": 1.0 - _SELF_SHADE * np.clip(above, 0.0, 1.0),
            "brightness": self.py_random.uniform(*self.brightness_range),
        }

    def apply(
        self,
        img: np.ndarray,
        veiled: np.ndarray,
        shade: np.ndarray,
        brightness: float,
        **params: Any,
    ) -> np.ndarray:
        """Replace what the plumes hide by their own light, in linear light."""
        if float(veiled.max()) < 1e-3:
            return img
        linear, scale, is_integer = to_linear_unit(img)
        steam = brightness * illuminant_radiance_linear(linear)
        share = veiled[..., None]
        out = linear * (1.0 - share) + share * shade[..., None] * steam[None, None, :]
        return from_linear_unit(out, scale, is_integer, img.dtype)
