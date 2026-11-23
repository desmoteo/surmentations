"""Insects and spider webs on the camera housing.

Physical model
--------------
This is a mundane, extremely common surveillance failure that no augmentation
pipeline simulates. Infrared illuminators attract insects, which land on and
walk across the dome directly in front of the lens; spiders follow the
insects and build webs across the housing, often anchored to the illuminator
ring itself. Both then sit a centimetre or two from the front element.

Two consequences follow from that distance, and they are what make these look
right rather than like stickers:

**They are wildly out of focus, and much larger than they sound.** An object
on the dome is far closer than any focus distance the lens can reach, so
nothing here has a hard edge. But the same short distance also *magnifies*
it enormously, and that is the part which is easy to get wrong. Take a 4 mm
lens with a 3 mm aperture and a dome 20 mm away: a 6 mm insect projects a
**160 px** image on a 640 px frame against an 80 px penumbra, so it reads as
a large, soft, substantially opaque blob. Size it as though it were a small
object in the scene and the blur swamps it into invisibility.

The same arithmetic explains why webs behave differently: a 0.1 mm strand
projects about 2 px against that same 80 px penumbra, so it is *correctly*
near-invisible by day. What makes a web show up is not its size but the
illuminator lighting it — which is the second point.

**Whether they are dark or bright depends on the illuminator, not the
object.** By day they are backlit against the scene and read as dark
silhouettes. At night the IR ring is *behind* them, a few centimetres away,
so they are lit far more strongly than anything in the scene and read as
bright glowing shapes. A web at night is the classic case: barely visible by
day, a blazing tangle of loops on the night feed. The same geometry produces
both, so one parameter — how strongly the near field is lit — covers it.

The webs here are lines, blurred. :class:`~surmentations.lens.SpiderWeb`
is a web as geometry, thread by thread at its own distance, and
:class:`~surmentations.atmosphere.NightInsects` is the insect in
flight; this transform is what has settled on the dome.
"""

from __future__ import annotations

import math
from typing import Annotated, Any

import cv2
import numpy as np
from albumentations.core.pydantic import check_range_bounds, nondecreasing
from albumentations.core.transforms_interface import BaseTransformInitSchema
from pydantic import AfterValidator

from surmentations.base import CameraTransform
from surmentations.diagnostics import record_draw
from surmentations.utils.filters import wide_blur
from surmentations.utils.photometry import from_linear_unit, to_linear_unit


def _draw_insect(
    mask: np.ndarray, rng: np.random.Generator, cx: float, cy: float, body_px: float
) -> None:
    """One insect: an elongated body with thin legs radiating from it."""
    angle = float(rng.uniform(0.0, 180.0))
    cv2.ellipse(
        mask,
        (int(cx), int(cy)),
        (max(1, int(body_px)), max(1, int(body_px * rng.uniform(0.35, 0.6)))),
        angle,
        0,
        360,
        1.0,
        thickness=-1,
    )
    # Legs. Thin next to the body, and the first thing defocus erases — which
    # is why a real insect on a dome reads as a blob with a faint fringe
    # rather than as a recognisable insect.
    for _ in range(6):
        theta = float(rng.uniform(0.0, 2.0 * math.pi))
        reach = body_px * float(rng.uniform(1.2, 2.4))
        cv2.line(
            mask,
            (int(cx), int(cy)),
            (int(cx + reach * math.cos(theta)), int(cy + reach * math.sin(theta))),
            1.0,
            thickness=max(1, int(body_px * 0.16)),
            lineType=cv2.LINE_AA,
        )


def _draw_web(
    mask: np.ndarray,
    rng: np.random.Generator,
    cx: float,
    cy: float,
    radius_px: float,
    spokes: int,
) -> None:
    """An orb web: radial spokes plus the spiral capture thread across them."""
    points = []
    for index in range(spokes):
        theta = 2.0 * math.pi * index / spokes + float(rng.uniform(-0.15, 0.15))
        reach = radius_px * float(rng.uniform(0.75, 1.0))
        end = (cx + reach * math.cos(theta), cy + reach * math.sin(theta))
        points.append(end)
        cv2.line(mask, (int(cx), int(cy)), (int(end[0]), int(end[1])), 1.0, 1, cv2.LINE_AA)

    # The spiral: successive rings joining adjacent spokes, tightening inward.
    for ring in range(1, 5):
        scale = ring / 5.0
        ring_points = [
            (int(cx + (px - cx) * scale), int(cy + (py - cy) * scale)) for px, py in points
        ]
        cv2.polylines(mask, [np.array(ring_points, dtype=np.int32)], True, 1.0, 1, cv2.LINE_AA)


class HousingOccluders(CameraTransform):
    """Insects and spider webs on the dome, far inside the near focus limit.

    Args:
        insect_count_range: how many insects are on the dome.
        insect_size_range: body size as a fraction of the image diagonal.
            Large by default, and deliberately: an insect sitting on the dome
            is only centimetres from the lens, so it is magnified into a blob
            covering a good part of the frame. See the module docstring for
            the arithmetic.
        web_count_range: how many webs.
        web_radius_range: web radius as a fraction of the image diagonal.
        defocus_range: blur radius as a fraction of the diagonal. Large by
            construction — anything sitting on the housing is centimetres from
            a lens focused metres away, so this is not a softening but the
            dominant transformation of the shape.
        opacity_range: how much light the occluder blocks.
        glow_range: how brightly the occluder is lit from behind by the
            camera's own IR illuminator. 0 is daylight and they silhouette
            dark; high values are the night case, where the ring lights them
            far more strongly than the scene and they glow.

    Targets:
        image

    Image types:
        uint8, float32
    """

    class InitSchema(BaseTransformInitSchema):
        """The arguments of ``HousingOccluders`` as Albumentations validates them: each
        range ordered low to high and within the bounds given here.
        """
        insect_count_range: Annotated[
            tuple[int, int],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0, 30)),
        ]
        insect_size_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.002, 0.4)),
        ]
        web_count_range: Annotated[
            tuple[int, int],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0, 6)),
        ]
        web_radius_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.02, 0.6)),
        ]
        defocus_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.002, 0.1)),
        ]
        opacity_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 1.0)),
        ]
        glow_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 3.0)),
        ]

    def __init__(
        self,
        insect_count_range: tuple[int, int] = (0, 4),
        insect_size_range: tuple[float, float] = (0.03, 0.16),
        web_count_range: tuple[int, int] = (0, 2),
        web_radius_range: tuple[float, float] = (0.06, 0.3),
        defocus_range: tuple[float, float] = (0.01, 0.045),
        opacity_range: tuple[float, float] = (0.5, 0.95),
        glow_range: tuple[float, float] = (0.0, 1.6),
        p: float = 0.5,
    ):
        """Store the ranges; each argument is described in the class docstring."""
        super().__init__(p=p)
        self.insect_count_range = insect_count_range
        self.insect_size_range = insect_size_range
        self.web_count_range = web_count_range
        self.web_radius_range = web_radius_range
        self.defocus_range = defocus_range
        self.opacity_range = opacity_range
        self.glow_range = glow_range

    def get_params_dependent_on_data(
        self, params: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, Any]:
        """Draw what is on or in front of the housing, defocused as the lens sees it.

        Webs anchored near a corner or the illuminator ring, insects anywhere.
        Returns the blurred ``occluder`` mask, its ``opacity``, and its
        ``glow`` when the illuminator behind it is on.
        """
        del data
        height, width = params["shape"][:2]
        diagonal = float(np.hypot(width, height))
        rng = self.random_generator

        mask = np.zeros((height, width), dtype=np.float32)

        insect_count = self.py_random.randint(*self.insect_count_range)
        insect_sizes = []
        for _ in range(insect_count):
            size = self.py_random.uniform(*self.insect_size_range) * diagonal
            insect_sizes.append(size)
            _draw_insect(
                mask,
                rng,
                float(rng.uniform(0, width)),
                float(rng.uniform(0, height)),
                size,
            )

        web_count = self.py_random.randint(*self.web_count_range)
        for _ in range(web_count):
            # Webs are usually anchored near a corner or the illuminator ring
            # rather than floating in the middle of the view.
            _draw_web(
                mask,
                rng,
                float(rng.uniform(-0.1, 1.1) * width),
                float(rng.uniform(-0.1, 1.1) * height),
                self.py_random.uniform(*self.web_radius_range) * diagonal,
                spokes=self.py_random.randint(8, 14),
            )

        # The defining property: this close to the lens, nothing keeps an
        # edge. The blur *is* the appearance, not a finishing touch.
        sigma = max(1.0, self.py_random.uniform(*self.defocus_range) * diagonal)
        # Tens of pixels on a large frame, so on a reduced copy in fast mode.
        mask = wide_blur(mask, sigma)

        # Counts and sizes are local: only the composited mask reaches the
        # renderer, so "how many insects, how big" is unanswerable from the
        # output alone - and it is exactly what an audit of this effect asks.
        record_draw(
            insect_count=insect_count,
            insect_size_px=np.array(insect_sizes, dtype=np.float32),
            web_count=web_count,
            defocus_sigma_px=sigma,
            covered_fraction=lambda: float((mask > 0.05).mean()),
        )

        return {
            "occluder": np.clip(mask, 0.0, 1.0),
            "opacity": self.py_random.uniform(*self.opacity_range),
            "glow": self.py_random.uniform(*self.glow_range),
        }

    def legibility_planes(
        self, params: dict[str, Any], before: np.ndarray, after: np.ndarray
    ) -> dict[str, Any] | None:
        """What the occluder hides: its silhouette, by its opacity."""
        del before, after
        return {"hidden": params["occluder"] * params["opacity"]}

    def apply(
        self,
        img: np.ndarray,
        occluder: np.ndarray,
        opacity: float,
        glow: float,
        **params: Any,
    ) -> np.ndarray:
        """Block the scene behind the occluder, and add its glow where the illuminator is on."""
        if float(occluder.max()) < 1e-3:
            return img

        linear, scale, is_integer = to_linear_unit(img)
        alpha = (occluder * opacity)[..., None]

        # Blocked: the occluder hides the scene behind it.
        out = linear * (1.0 - alpha)
        # Lit: the illuminator sits behind the occluder and a centimetre away,
        # so at night it outshines anything in the scene.
        out = out + alpha * glow

        return from_linear_unit(np.clip(out, 0.0, 1.0), scale, is_integer, img.dtype)
