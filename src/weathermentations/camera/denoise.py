"""Artefacts left by a surveillance ISP's noise reduction.

Physical model
--------------
At night a camera's sensor gain is high and the raw image is very noisy, so
the ISP denoises hard. Consumer and surveillance ISPs use "3D-NR": a spatial
edge-preserving filter combined with a temporal average across frames. Both
halves leave a signature, and both destroy exactly the fine detail that
detectors depend on.

**The spatial half**, which a single image *can* express and which is most of
the damage: an edge-preserving filter run aggressively does not blur the
image uniformly. It collapses low-contrast texture into flat patches while
leaving strong edges intact — the "watercolour" or "plastic" look. That is a
different operation from a blur, and the difference is the point: a blur
softens edges and texture alike, while this keeps the edges crisp and deletes
the texture between them. Any test that cannot tell those apart is not
testing this effect.

Chroma is denoised far harder than luma, because the eye is much less
sensitive to colour detail and the bit budget is better spent elsewhere. The
result is colour bleeding across boundaries and low-frequency chroma
blotching — visible as patches of wrong colour that do not line up with any
edge in the luminance.

**The temporal half is not single-frame and is not attempted here.**
Averaging across frames leaves comet trails behind anything that moved, and
the length of a trail is set by how far the object travelled between frames —
information a single image does not contain. Approximating it by smearing
high-gradient regions would be inventing motion that is not in the data. What
a clip-level API would add is exactly this: a per-frame displacement, from
which the trail follows.
"""

from __future__ import annotations

from typing import Annotated, Any

import cv2
import numpy as np
from albumentations.core.pydantic import check_range_bounds, nondecreasing
from albumentations.core.transforms_interface import BaseTransformInitSchema
from pydantic import AfterValidator

from weathermentations import _native, mode
from weathermentations.base import WeatherTransform
from weathermentations.utils.filters import wide_blur

# Whether the compiled filter has been seen to give OpenCV's bytes on this
# installation: None until it has been tried.
_kernel_is_opencv: bool | None = None


def _kernel_matches_opencv() -> bool:
    """Whether the compiled edge-preserving filter gives OpenCV's bytes here.

    It is OpenCV's arithmetic in OpenCV's order, and on x86-64 with glibc it
    gives the same bytes. That rests on the two calling the same ``powf`` and
    on neither fusing a multiply into an add, which another C library or
    another build of OpenCV could undo, so it is tried once on two small
    frames and not assumed.
    """
    global _kernel_is_opencv
    if _kernel_is_opencv is None:
        probe = np.random.default_rng(0).integers(0, 256, (48, 64, 3), dtype=np.uint8)
        same = True
        for sigma_s, sigma_r in ((3.0, 0.07), (40.0, 0.3)):
            ours = _native.edge_preserving(probe, sigma_s, sigma_r, tabled=False)
            if ours is None:
                return False  # the kernels are off: nothing learnt, ask again later
            theirs = cv2.edgePreservingFilter(
                probe, flags=cv2.RECURS_FILTER, sigma_s=sigma_s, sigma_r=sigma_r
            )
            same = same and bool(np.array_equal(ours, theirs))
        _kernel_is_opencv = same
    return _kernel_is_opencv


def _edge_preserving(image: np.ndarray, sigma_s: float, sigma_r: float) -> np.ndarray:
    """``cv2.edgePreservingFilter`` with ``RECURS_FILTER``, as fast as it can be had.

    OpenCV's own takes a power for every pixel six times a call and walks the
    frame through checked accessors: 160 ms on a 1024-pixel frame. The
    compiled kernel is the same arithmetic in 49 ms, and is used in exact
    mode where it has been seen to give the same bytes. In fast mode it
    takes the power from a table of its 766 possible values, 21 ms, and a
    few values in a million come out one level away. That makes this the one
    place where a fast-mode frame depends, by that much, on whether the
    kernels are installed.
    """
    done = None
    if mode.fast():
        done = _native.edge_preserving(image, sigma_s, sigma_r, tabled=True)
    elif _kernel_matches_opencv():
        done = _native.edge_preserving(image, sigma_s, sigma_r, tabled=False)
    if done is not None:
        return done
    filtered: np.ndarray = cv2.edgePreservingFilter(
        image, flags=cv2.RECURS_FILTER, sigma_s=sigma_s, sigma_r=sigma_r
    )
    return filtered


def _blur_plane(plane: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur of one uint8 plane.

    Exact mode is OpenCV's blur of the bytes. Fast mode blurs a reduced copy
    when the sigma is many pixels, in float32 so that reducing and enlarging
    do not each round to a whole level, and rounds once at the end.
    """
    if not mode.fast():
        return cv2.GaussianBlur(plane, (0, 0), sigmaX=sigma)
    blurred = wide_blur(plane.astype(np.float32), sigma)
    return np.clip(np.rint(blurred), 0, 255).astype(np.uint8)


class TemporalDenoiseArtifacts(WeatherTransform):
    """The "watercolour" flattening and chroma blotching left by 3D-NR.

    Args:
        strength_range: how far the result is pushed toward the fully
            denoised version. 0 is a no-op, 1 applies the filter outright.
        iterations_range: how many times the edge-preserving filter is run.
            Repetition is what turns a mild smoothing into flat patches —
            each pass pulls low-contrast neighbours closer together until
            they collapse to a single value, which a single strong pass does
            not do.
        edge_threshold_range: contrast below which detail is treated as noise
            and removed. This is the knob that separates this from a blur:
            edges above it survive intact however many iterations are run.
        spatial_radius_range: filter radius as a fraction of the image
            diagonal, i.e. how large the flat patches grow.
        chroma_blur_range: extra smoothing applied to the colour channels
            alone, as a fraction of the diagonal. Chroma is always denoised
            harder than luma, so colour bleeds across boundaries that
            luminance still resolves.

    Targets:
        image

    Image types:
        uint8, float32
    """

    class InitSchema(BaseTransformInitSchema):
        strength_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 1.0)),
        ]
        iterations_range: Annotated[
            tuple[int, int],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(1, 6)),
        ]
        edge_threshold_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(2.0, 120.0)),
        ]
        spatial_radius_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.002, 0.06)),
        ]
        chroma_blur_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 0.05)),
        ]

    def __init__(
        self,
        strength_range: tuple[float, float] = (0.4, 1.0),
        iterations_range: tuple[int, int] = (1, 3),
        edge_threshold_range: tuple[float, float] = (18.0, 60.0),
        spatial_radius_range: tuple[float, float] = (0.004, 0.018),
        chroma_blur_range: tuple[float, float] = (0.004, 0.022),
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.strength_range = strength_range
        self.iterations_range = iterations_range
        self.edge_threshold_range = edge_threshold_range
        self.spatial_radius_range = spatial_radius_range
        self.chroma_blur_range = chroma_blur_range

    def get_params_dependent_on_data(
        self, params: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, Any]:
        del data
        return {
            "strength": self.py_random.uniform(*self.strength_range),
            "iterations": self.py_random.randint(*self.iterations_range),
            "edge_threshold": self.py_random.uniform(*self.edge_threshold_range),
            "spatial_radius": self.py_random.uniform(*self.spatial_radius_range),
            "chroma_blur": self.py_random.uniform(*self.chroma_blur_range),
        }

    def apply(
        self,
        img: np.ndarray,
        strength: float,
        iterations: int,
        edge_threshold: float,
        spatial_radius: float,
        chroma_blur: float,
        **params: Any,
    ) -> np.ndarray:
        if strength < 1e-3:
            return img

        is_integer = np.issubdtype(img.dtype, np.integer)
        work = img if is_integer else np.clip(img * 255.0, 0, 255).astype(np.uint8)

        height, width = work.shape[:2]
        diagonal = float(np.hypot(width, height))
        radius_px = max(1.0, spatial_radius * diagonal)

        # Repeated edge-preserving filtering. Each pass drags low-contrast
        # neighbours together; run a few times they collapse into flat
        # patches, while anything above `edge_threshold` is left alone.
        #
        # The domain-transform filter is used rather than a bilateral one.
        # Both preserve edges, but the bilateral costs O(radius^2) per pixel
        # while this is linear in the image regardless of radius — and a
        # denoiser wants a *large* radius, which is exactly where the
        # bilateral becomes unaffordable (a 26px radius cost over five seconds
        # a frame). A real ISP has the same constraint and solves it the same
        # way: with recursive separable passes, not one enormous kernel.
        denoised = work
        for _ in range(max(1, iterations)):
            denoised = _edge_preserving(
                denoised,
                sigma_s=float(np.clip(radius_px, 1.0, 200.0)),
                sigma_r=float(np.clip(edge_threshold / 255.0, 0.01, 1.0)),
            )

        # Chroma is denoised harder than luma, so colour bleeds across
        # boundaries that luminance still resolves.
        if chroma_blur > 1e-4:
            ycrcb = cv2.cvtColor(denoised, cv2.COLOR_RGB2YCrCb)
            sigma = max(0.6, chroma_blur * diagonal)
            ycrcb[..., 1] = _blur_plane(ycrcb[..., 1], sigma)
            ycrcb[..., 2] = _blur_plane(ycrcb[..., 2], sigma)
            denoised = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)

        blended = (
            work.astype(np.float32) * (1.0 - strength)
            + denoised.astype(np.float32) * strength
        )

        if is_integer:
            return np.clip(blended, 0, 255).astype(img.dtype)
        return np.clip(blended / 255.0, 0.0, 1.0).astype(img.dtype)
