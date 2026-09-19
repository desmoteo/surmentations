"""Night mode: IR-cut filter removed, scene lit by the camera's own illuminator.

Physical model
--------------
A day/night camera swings its IR-cut filter out of the optical path after
dark and switches on a ring of 850 nm LEDs around the lens. Three things
change at once, and none of them is "the image gets darker":

**The image goes monochrome, and not by luminance.** Silicon is sensitive
well past 700 nm, and all three Bayer dyes transmit there, so every channel
sees mostly the same infrared signal and the colour collapses. But what it
collapses *to* is near-infrared **reflectance**, which is not the visible
brightness of a material. The largest discrepancy is vegetation: chlorophyll
absorbs red strongly and reflects NIR strongly — the "red edge" at ~700 nm —
so foliage that reads dark green by day is among the *brightest* things in
frame at night. Converting to luminance gets this exactly backwards, which is
the single most visible error in a naive night-mode augmentation.

We have no NIR channel to read, so reflectance is estimated from RGB: a broad
red-weighted response (silicon's sensitivity rises toward the red end) plus
an explicit boost where the pixel is green relative to red, which is where
chlorophyll is. It is an approximation, and it is calibrated to reproduce the
one large, well-known material difference rather than to be spectrally exact.

**The scene is lit by the camera, not by the sun.** The illuminator sits at
the lens, so irradiance falls as ``1/z^2`` and is shaped by the LED beam
profile. This is why night CCTV looks the way it does: a subject at two
metres is blown to featureless white while the background five metres behind
is nearly black, and no exposure setting fixes it, because the ratio is 6:1
in the illumination itself. Getting the inverse-square right is what produces
that automatically instead of having to paint it on.

**Gain goes up, so noise does.** Shot noise scales with the square root of
the signal and read noise does not scale at all, so a gained-up dark frame is
noisiest in the shadows — not uniformly noisy.
"""

from __future__ import annotations

from typing import Annotated, Any

import numpy as np
from albumentations.core.pydantic import check_range_bounds, nondecreasing
from albumentations.core.transforms_interface import BaseTransformInitSchema
from pydantic import AfterValidator, Field

from weathermentations.base import WeatherTransform, require_depth_provider
from weathermentations.depth.provider import SharedDepthProvider
from weathermentations.depth.scale import to_pseudo_metric
from weathermentations.diagnostics import record_draw
from weathermentations.utils.grids import pixel_grid
from weathermentations.utils.photometry import from_linear_unit, to_linear_unit

# Broad, red-weighted stand-in for silicon's response with the IR-cut filter
# removed. Not a luminance vector: the eye's green-heavy weighting is exactly
# what does not apply once the sensor is seeing past 700 nm.
_NIR_RESPONSE = np.array([0.50, 0.30, 0.20], dtype=np.float32)


def near_infrared_reflectance(linear_rgb: np.ndarray, vegetation_gain: float) -> np.ndarray:
    """Estimate NIR reflectance from a visible-light image.

    Broad red-weighted response plus a chlorophyll term. ``vegetation_gain``
    controls how strongly foliage lifts; 0 disables the correction and leaves
    a plain broad-band conversion.
    """
    broadband = linear_rgb @ _NIR_RESPONSE

    # Greenness relative to red is where chlorophyll is, and chlorophyll is
    # what reflects hard in the near infrared.
    red, green, blue = linear_rgb[..., 0], linear_rgb[..., 1], linear_rgb[..., 2]
    greenness = np.clip(
        (green - np.maximum(red, blue)) / np.maximum(green + red + blue, 1e-4), 0.0, 1.0
    )
    return np.clip(broadband + vegetation_gain * greenness, 0.0, 4.0)


class NightIrMode(WeatherTransform):
    """Monochrome IR night mode with the camera's own illuminator.

    Args:
        depth_provider: shared :class:`~weathermentations.depth.provider.SharedDepthProvider`.
        illuminator_range_m: distance at which the illuminator delivers unit
            exposure. Objects nearer than this blow out, objects further fade
            into the noise — and the ``1/z^2`` between them is why night CCTV
            has such a narrow band of usable distance.
        beam_falloff_range: how tightly the LED beam is focused. Higher
            concentrates light on the optical axis and darkens the corners
            further; the exponent of a ``cos^k`` beam profile.
        ambient_range: residual scene lighting (moon, streetlights, spill) as
            a fraction of unit exposure, so the far field is not pure black.
        vegetation_gain_range: how strongly foliage lifts relative to a plain
            broad-band conversion — the chlorophyll red-edge term.
        target_level_range: level the camera's auto-exposure drives the scene
            to, as a fraction of full scale measured at the 80th percentile.
            This is not a cosmetic normalisation — a night camera really does
            gain up until the scene is usable, and without modelling it the
            result depends entirely on how the depth map happened to be
            scaled: pick an illuminator range slightly too short and the whole
            lower frame saturates. Auto-exposure fixes the *level* while
            leaving the ``1/z^2`` **ratio** between near and far intact, and
            that ratio is the part that makes night footage recognisable.
        gain_range: extra sensor gain on top of the auto-exposure, which sets
            how far the noise floor is lifted with the signal.
        read_noise_range: signal-independent noise, as a fraction of unit
            exposure. Unlike shot noise this does not shrink in the shadows,
            which is why a gained-up night frame is noisiest where it is
            darkest.

    Targets:
        image

    Image types:
        uint8, float32
    """

    class InitSchema(BaseTransformInitSchema):
        depth_provider: SharedDepthProvider
        illuminator_range_m: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.5, 200.0)),
        ]
        beam_falloff_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 12.0)),
        ]
        ambient_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 1.0)),
        ]
        vegetation_gain_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 2.0)),
        ]
        target_level_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.05, 0.95)),
        ]
        gain_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.1, 20.0)),
        ]
        read_noise_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 0.2)),
        ]
        residual_tint: float = Field(ge=0.0, le=0.5)

    def __init__(
        self,
        depth_provider: SharedDepthProvider,
        illuminator_range_m: tuple[float, float] = (4.0, 14.0),
        beam_falloff_range: tuple[float, float] = (1.0, 4.0),
        ambient_range: tuple[float, float] = (0.01, 0.12),
        vegetation_gain_range: tuple[float, float] = (0.25, 0.7),
        target_level_range: tuple[float, float] = (0.35, 0.6),
        gain_range: tuple[float, float] = (0.8, 1.6),
        read_noise_range: tuple[float, float] = (0.004, 0.03),
        residual_tint: float = 0.04,
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.depth_provider = require_depth_provider(depth_provider, cls_name=type(self).__name__)
        self.illuminator_range_m = illuminator_range_m
        self.beam_falloff_range = beam_falloff_range
        self.ambient_range = ambient_range
        self.vegetation_gain_range = vegetation_gain_range
        self.target_level_range = target_level_range
        self.gain_range = gain_range
        self.read_noise_range = read_noise_range
        self.residual_tint = residual_tint

    def get_params_dependent_on_data(
        self, params: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, Any]:
        image = data["image"]
        height, width = params["shape"][:2]

        relative_depth = self.depth_provider.get_depth(image)
        depth_m = np.maximum(to_pseudo_metric(relative_depth, None), 0.3)

        reference = self.py_random.uniform(*self.illuminator_range_m)
        falloff = self.py_random.uniform(*self.beam_falloff_range)

        # Inverse-square from an illuminator sitting at the lens.
        irradiance = (reference / depth_m) ** 2

        # LED beam profile: brightest on the optical axis, falling as cos^k of
        # the field angle. Approximated from normalised image radius, which is
        # monotonic in field angle.
        yy, xx = pixel_grid(height, width)
        cx, cy = width / 2.0, height / 2.0
        radius = np.sqrt(((xx - cx) / max(cx, 1)) ** 2 + ((yy - cy) / max(cy, 1)) ** 2)
        beam = np.clip(np.cos(np.clip(radius, 0.0, 1.0) * (np.pi / 3.0)), 0.0, 1.0) ** falloff

        record_draw(
            illuminator_range_m=reference,
            beam_falloff=falloff,
            depth_min_m=float(depth_m.min()),
            depth_p50_m=float(np.median(depth_m)),
            depth_max_m=float(depth_m.max()),
        )
        return {
            "exposure": (irradiance * beam).astype(np.float32),
            "ambient": self.py_random.uniform(*self.ambient_range),
            "target_level": self.py_random.uniform(*self.target_level_range),
            "vegetation_gain": self.py_random.uniform(*self.vegetation_gain_range),
            "gain": self.py_random.uniform(*self.gain_range),
            "read_noise": self.py_random.uniform(*self.read_noise_range),
            "noise_seed": self.py_random.randint(0, 2**31 - 1),
        }

    def apply(
        self,
        img: np.ndarray,
        exposure: np.ndarray,
        ambient: float,
        target_level: float,
        vegetation_gain: float,
        gain: float,
        read_noise: float,
        noise_seed: int,
        **params: Any,
    ) -> np.ndarray:
        linear, scale, is_integer = to_linear_unit(img)

        reflectance = near_infrared_reflectance(linear, vegetation_gain)
        signal = reflectance * (exposure + ambient)

        # Auto-exposure. The camera gains up until the scene is usable, so the
        # absolute level is set by the AE loop rather than by the illuminator's
        # nominal range — while the near/far *ratio* the inverse-square law
        # produced is untouched, which is the part that reads as night footage.
        reference_level = float(np.percentile(signal, 80.0))
        if reference_level > 1e-6:
            signal = signal * (target_level / reference_level)
        signal = signal * gain

        # Shot noise scales as sqrt(signal); read noise does not scale at all,
        # which is why the shadows of a gained-up frame are the noisy part.
        rng = np.random.default_rng(noise_seed)
        noise = rng.standard_normal(signal.shape).astype(np.float32)
        signal = signal + noise * (np.sqrt(np.maximum(signal, 0.0)) * 0.02 + read_noise) * gain

        signal = np.clip(signal, 0.0, 1.0)

        # Near-monochrome, with the slight residual cast these sensors keep
        # rather than a mathematically perfect grey.
        tint = np.array([1.0 + self.residual_tint, 1.0, 1.0 - self.residual_tint * 0.5])
        out = signal[..., None] * tint.astype(np.float32)[None, None, :]
        return from_linear_unit(np.clip(out, 0.0, 1.0), scale, is_integer, img.dtype)
