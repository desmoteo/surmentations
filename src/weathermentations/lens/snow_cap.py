"""Snow on the housing.

Physical model
--------------
Snow settles on a camera as it does on everything else. It builds a cap on
the top of the housing or its sunshield, which sags over the front edge and
hangs into the top of the view; and wind plasters wet snow onto the glass
itself in patches. Until it slides off or melts, part of the picture is
gone.

**It hides what is behind it.** Unlike a web or a film of frost, a lump of
snow is thick: it covers, and what is seen in its place is the snow.

**It is a soft shape with no detail.** It is on the housing, centimetres
from a lens focused metres away, so its edge is a gradient tens of pixels
wide and its surface is featureless. What identifies it is the outline: an
uneven lobed edge hanging from the top of the frame.

**How bright it is depends on what lights it.** By day the camera looks at
the cap's underside, which is in its own shade and lit two ways: through
the snow from above, and by the sky and the ground from below. Light
through snow is what makes it glow at its thin edge and dull in its
thick middle, and it comes through bluer than it went in, snow absorbing
red a little more than blue; so the mass is a pale blue grey, brightest
along its lower edge, and its surface is never quite even. At night under
the infrared ring it is a white diffuser a hand's width from the lamp,
and clips.
"""

from __future__ import annotations

from typing import Annotated, Any

import cv2
import numpy as np
from albumentations.core.pydantic import check_range_bounds, nondecreasing
from albumentations.core.transforms_interface import BaseTransformInitSchema
from pydantic import AfterValidator

from weathermentations.base import WeatherTransform
from weathermentations.diagnostics import record_draw
from weathermentations.utils.color import illuminant_radiance_linear
from weathermentations.utils.filters import wide_blur
from weathermentations.utils.noise import value_noise
from weathermentations.utils.photometry import from_linear_unit, to_linear_unit

# How much darker the thickest part of the cap is than its edge, by day:
# less of the sky's light comes through more snow.
_THICK_SHADE = 0.45
# What comes through, per channel, relative to green: snow passes blue best.
_THROUGH_SNOW = np.array([0.93, 1.0, 1.08], dtype=np.float32)
# How uneven the surface's brightness is, as a share of it.
_SURFACE_RELIEF = 0.12


def cap_edge(rng: np.random.Generator, width: int, lumpiness: float) -> np.ndarray:
    """How far down the cap hangs at each column, relative to its mean: ``(W,)``, mean 1.

    A slow unevenness, the cap being deeper at one end than the other, and
    a few rounded lobes where it has sagged; ``lumpiness`` (0 to 1) is how
    far both go.
    """
    across = np.linspace(0.0, 1.0, width, dtype=np.float32)
    knots = rng.uniform(-1.0, 1.0, 5).astype(np.float32)
    slow = cv2.resize(knots[None, :], (width, 1), interpolation=cv2.INTER_CUBIC)[0]
    edge = 1.0 + 0.6 * lumpiness * slow
    for _ in range(int(rng.integers(2, 6))):
        at, span = rng.uniform(0.0, 1.0), rng.uniform(0.04, 0.16)
        lobe = np.clip(1.0 - ((across - at) / span) ** 2, 0.0, None)
        edge += lumpiness * rng.uniform(0.2, 0.9) * np.sqrt(lobe)
    hang: np.ndarray = np.clip(edge / max(float(edge.mean()), 1e-3), 0.0, None)
    return hang


class SnowOnHousing(WeatherTransform):
    """A cap of snow hanging into the top of the view, and patches of it on the glass.

    Needs no depth. Under the infrared ring, set ``glow_range`` and apply it
    after ``NightIrMode``.

    Args:
        cap_depth_range: how far the cap hangs into the frame, as a fraction
            of its height, on average along the top edge. Zero is no cap.
        lumpiness_range: how uneven the cap's lower edge is, 0 to 1.
        patch_count_range: lumps of snow stuck to the glass, besides the cap.
        patch_size_range: their size, as a fraction of the diagonal.
        defocus_range: blur of the snow's outline, as a fraction of the
            diagonal. Large: it is on the housing.
        brightness_range: how bright the snow's underside is by day, as a
            share of the scene's illuminant.
        glow_range: light the snow returns from the infrared ring, in units
            of full scale. Zero by day; above 1 at night, where it clips.

    Targets:
        image

    Image types:
        uint8, float32
    """

    class InitSchema(BaseTransformInitSchema):
        """The arguments of ``SnowOnHousing`` as Albumentations validates them: each
        range ordered low to high and within the bounds given here.
        """

        cap_depth_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 1.0)),
        ]
        lumpiness_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 1.0)),
        ]
        patch_count_range: Annotated[
            tuple[int, int],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0, 30)),
        ]
        patch_size_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.005, 0.5)),
        ]
        defocus_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.002, 0.1)),
        ]
        brightness_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 2.0)),
        ]
        glow_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 5.0)),
        ]

    def __init__(
        self,
        cap_depth_range: tuple[float, float] = (0.06, 0.4),
        lumpiness_range: tuple[float, float] = (0.2, 0.9),
        patch_count_range: tuple[int, int] = (0, 4),
        patch_size_range: tuple[float, float] = (0.03, 0.14),
        defocus_range: tuple[float, float] = (0.01, 0.04),
        brightness_range: tuple[float, float] = (0.55, 1.0),
        glow_range: tuple[float, float] = (0.0, 0.0),
        p: float = 0.5,
    ):
        """Store the ranges; each argument is described in the class docstring."""
        super().__init__(p=p)
        self.cap_depth_range = cap_depth_range
        self.lumpiness_range = lumpiness_range
        self.patch_count_range = patch_count_range
        self.patch_size_range = patch_size_range
        self.defocus_range = defocus_range
        self.brightness_range = brightness_range
        self.glow_range = glow_range

    def get_params_dependent_on_data(
        self, params: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, Any]:
        """Shape the cap and the patches and blur them as the lens sees them.

        Returns ``cover``, how much of each pixel the snow hides; ``thick``,
        how deep into the snow each pixel looks, 0 at its edge to 1; the
        ``relief`` of its surface, a factor about one; and the snow's
        ``brightness`` by day and ``glow`` under the lamp.
        """
        del data
        height, width = params["shape"][:2]
        diagonal = float(np.hypot(width, height))
        rng = self.random_generator

        cap_depth = self.py_random.uniform(*self.cap_depth_range)
        lumpiness = self.py_random.uniform(*self.lumpiness_range)
        patches = self.py_random.randint(*self.patch_count_range)
        sigma = max(1.0, self.py_random.uniform(*self.defocus_range) * diagonal)

        rows = np.arange(height, dtype=np.float32)[:, None]
        hang = cap_depth * height * cap_edge(rng, width, lumpiness)[None, :]
        solid = (rows < hang).astype(np.float32)
        sizes = []
        for _ in range(patches):
            size = self.py_random.uniform(*self.patch_size_range) * diagonal
            sizes.append(size)
            cv2.ellipse(
                solid,
                (int(rng.uniform(0, width)), int(rng.uniform(0, height))),
                (max(1, int(size)), max(1, int(size * rng.uniform(0.5, 1.0)))),
                float(rng.uniform(0.0, 180.0)),
                0,
                360,
                1.0,
                -1,
            )
        cover = np.clip(wide_blur(solid, sigma), 0.0, 1.0)
        # Depth into the snow: the same shape blurred much wider, which is
        # small near any edge and large in the middle of a mass.
        thick = np.clip(wide_blur(solid, 4.0 * sigma), 0.0, 1.0) * cover
        # The surface's own lumps, a few to the frame, seen as soft shading.
        relief = value_noise(height, width, rng=rng, octaves=3, base_cells=5)
        relief = 1.0 + _SURFACE_RELIEF * 2.0 * (relief - float(relief.mean()))

        record_draw(
            cap_depth=cap_depth,
            lumpiness=lumpiness,
            patch_count=patches,
            patch_size_px=np.array(sizes, dtype=np.float32),
            defocus_sigma_px=sigma,
            covered_fraction=lambda: float((cover > 0.5).mean()),
        )
        return {
            "cover": cover,
            "thick": thick,
            "relief": relief.astype(np.float32),
            "brightness": self.py_random.uniform(*self.brightness_range),
            "glow": self.py_random.uniform(*self.glow_range),
        }

    def apply(
        self,
        img: np.ndarray,
        cover: np.ndarray,
        thick: np.ndarray,
        relief: np.ndarray,
        brightness: float,
        glow: float,
        **params: Any,
    ) -> np.ndarray:
        """Hide the scene behind the snow and show the snow in its place."""
        if float(cover.max()) < 1e-3:
            return img
        linear, scale, is_integer = to_linear_unit(img)
        daylight = brightness * illuminant_radiance_linear(linear) * _THROUGH_SNOW
        # Daylight comes through the snow, so less of it where the snow is
        # thick; the lamp lights the face the camera sees, thick or thin.
        through = ((1.0 - _THICK_SHADE * thick) * relief)[..., None]
        snow = daylight[None, None, :] * through + glow * relief[..., None]
        alpha = cover[..., None]
        out = linear * (1.0 - alpha) + alpha * snow
        return from_linear_unit(out, scale, is_integer, img.dtype)
