"""Dirt / soil smudges on or near the lens.

Physical model: dirt sitting on or very close to the front element is far
outside the camera's depth of field, so it renders as a heavily defocused,
low-contrast, semi-opaque smudge rather than a sharp mask — not a geometric
occluder. We approximate that by generating irregular noise-based blob
clusters (`weathermentations.utils.noise`), heavily blurring their alpha
mask (proxy for defocus blur), and compositing a soil-toned, reduced-contrast
patch under the resulting soft mask.
"""

from __future__ import annotations

from typing import Annotated, Any

import cv2
import numpy as np
from albumentations.core.pydantic import check_range_bounds, nondecreasing
from albumentations.core.transforms_interface import BaseTransformInitSchema
from pydantic import AfterValidator, Field

from weathermentations.base import WeatherTransform
from weathermentations.utils.compositing import alpha_composite
from weathermentations.utils.noise import blob_cluster_mask


class DirtSoil(WeatherTransform):
    """Defocused dirt/soil smudges near the lens.

    Args:
        blob_count_range: number of smudge clusters to render.
        blob_size_range: per-blob radius in pixels, before defocus blur.
        opacity_range: peak alpha of a smudge at its most opaque point.
        defocus_blur_range: Gaussian blur kernel radius (px) applied to the
            blob alpha mask, simulating out-of-focus rendering.
        color_low, color_high: RGB bounds a smudge's tint is sampled from
            (uniformly, per channel, per call). Defaults span dust/soil
            browns and grays.
        contrast_reduction_range: fraction by which local contrast under a
            smudge is pulled toward the smudge's mean (0 = no change,
            1 = fully flattened).

    Targets:
        image

    Image types:
        uint8, float32
    """

    class InitSchema(BaseTransformInitSchema):
        blob_count_range: Annotated[
            tuple[int, int],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(1, 30)),
        ]
        blob_size_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(4.0, 400.0)),
        ]
        opacity_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 1.0)),
        ]
        defocus_blur_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(1.0, 121.0)),
        ]
        color_low: tuple[int, int, int] = Field(default=(40, 32, 24))
        color_high: tuple[int, int, int] = Field(default=(95, 80, 64))
        contrast_reduction_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 1.0)),
        ]

    def __init__(
        self,
        blob_count_range: tuple[int, int] = (2, 10),
        blob_size_range: tuple[float, float] = (15.0, 120.0),
        opacity_range: tuple[float, float] = (0.25, 0.65),
        defocus_blur_range: tuple[float, float] = (9.0, 41.0),
        color_low: tuple[int, int, int] = (40, 32, 24),
        color_high: tuple[int, int, int] = (95, 80, 64),
        contrast_reduction_range: tuple[float, float] = (0.2, 0.5),
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.blob_count_range = blob_count_range
        self.blob_size_range = blob_size_range
        self.opacity_range = opacity_range
        self.defocus_blur_range = defocus_blur_range
        self.color_low = color_low
        self.color_high = color_high
        self.contrast_reduction_range = contrast_reduction_range

    def get_params_dependent_on_data(
        self, params: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, Any]:
        height, width = params["shape"][:2]

        count = self.py_random.randint(*self.blob_count_range)
        mask = blob_cluster_mask(
            height,
            width,
            rng=self.random_generator,
            count=count,
            size_range=self.blob_size_range,
        )

        blur_radius = self.py_random.uniform(*self.defocus_blur_range)
        kernel = int(blur_radius) | 1  # nearest odd kernel size
        if kernel > 1:
            mask = cv2.GaussianBlur(mask, (kernel, kernel), 0)

        opacity = self.py_random.uniform(*self.opacity_range)
        mask = np.clip(mask * opacity, 0.0, 1.0)

        color = np.array(
            [
                self.py_random.uniform(lo, hi)
                for lo, hi in zip(self.color_low, self.color_high, strict=True)
            ],
            dtype=np.float32,
        )
        contrast_reduction = self.py_random.uniform(*self.contrast_reduction_range)

        return {"alpha_mask": mask, "smudge_color": color, "contrast_reduction": contrast_reduction}

    def apply(
        self,
        img: np.ndarray,
        alpha_mask: np.ndarray,
        smudge_color: np.ndarray,
        contrast_reduction: float,
        **params: Any,
    ) -> np.ndarray:
        img_f = img.astype(np.float32)

        # Flatten local contrast under the smudge toward the smudge color before
        # compositing, so it reads as "seen through grime" rather than "painted on".
        flattened = img_f + (smudge_color - img_f) * contrast_reduction
        composited = alpha_composite(flattened.astype(img.dtype), smudge_color, alpha_mask)
        return composited
