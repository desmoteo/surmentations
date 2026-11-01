"""Depth-conditioned fog / haze via atmospheric scattering.

Core model (Koschmieder), evaluated **in linear light**::

    I(x) = I0(x) * t(x) + A * (1 - t(x))
    t(x) = exp(-beta * d(x))

where ``d`` is scene depth, ``beta`` the extinction coefficient (1/m) and
``A`` the airlight colour. Extinction relates to meteorological visibility
by ``MOR = 3.912 / beta`` (the 2% contrast threshold), so beta 0.005 / 0.01 /
0.02 / 0.06 correspond to roughly 780 / 390 / 195 / 65 m of visibility — the
same ladder the Foggy Cityscapes benchmark uses.

Three things beyond the bare equation are what actually make it read as fog
rather than a white wash, and all three are implemented here:

* **Linear light.** Radiance adds linearly; sRGB values do not. Compositing
  airlight directly onto gamma-encoded pixels lifts the mid-tones far too
  much, which is the single biggest cause of the flat, milky look.
* **Wavelength-dependent extinction.** Scattering goes as
  ``beta(lambda) ~ lambda^-gamma``: gamma ~ 4 for Rayleigh (clean air),
  ~1-1.5 for thin haze, ~0 for dense fog whose droplets are much larger than
  the wavelength. Thin haze therefore goes *cool and blue* with distance,
  while dense fog is neutral grey. A single achromatic beta can only ever
  produce the latter.
* **Forward-scatter blur.** Fog is a low-pass filter, not just a blend:
  multiply-scattered light smears detail, so distant edges soften as well as
  fade. Sigma grows with optical depth (``~ sqrt(beta*d)``).

Sky is special-cased. At ``d -> infinity`` the model collapses to exactly
``A``, which would flatten a sky that in reality keeps its gradient and its
brightest region near the sun; optical depth is therefore capped for
sky-like pixels so the sky desaturates and loses contrast instead of turning
into a single flat colour.

References: Koschmieder 1924; He, Sun & Tang 2009 (dark-channel prior, used
here for airlight estimation); Sakaridis et al. 2018 (Foggy Cityscapes,
beta ladder and sky handling); Quilez, "Better Fog" (separate extinction and
inscattering, sun-tinted fog colour).
"""

from __future__ import annotations

from typing import Annotated, Any

import cv2
import numpy as np
from albumentations.core.pydantic import check_range_bounds, nondecreasing
from albumentations.core.transforms_interface import BaseTransformInitSchema
from pydantic import AfterValidator, Field

from weathermentations.base import (
    WeatherTransform,
    first_frame,
    require_depth_provider,
    scene_of,
)
from weathermentations.depth.provider import SharedDepthProvider
from weathermentations.diagnostics import record_draw
from weathermentations.utils.channels import channel_mean
from weathermentations.utils.color import veil_radiance_linear
from weathermentations.utils.photometry import from_linear_unit, linear_to_srgb, to_linear_unit
from weathermentations.utils.sky import lift_sky, sky_to_depth_edge, sky_weight

# Rec.709 primaries' representative wavelengths, in nm.
_RGB_WAVELENGTHS = np.array([600.0, 550.0, 450.0], dtype=np.float32)


def channel_extinction(beta: float, gamma: float) -> np.ndarray:
    """Per-channel extinction ``beta_c = beta * (lambda_c / 550)^-gamma``.

    ``gamma = 0`` gives neutral grey fog (large droplets, dense fog);
    ``gamma ~ 1.3`` gives the cool, blue-shifted distance of thin haze;
    ``gamma = 4`` is Rayleigh scattering off clean air.
    """
    return beta * (_RGB_WAVELENGTHS / 550.0) ** (-gamma)


class RandomFog(WeatherTransform):
    """Physically-grounded fog/haze, conditioned on scene depth.

    Args:
        depth_provider: shared :class:`~weathermentations.depth.provider.SharedDepthProvider`.
        beta_range: extinction coefficient, in 1/m against ``depth_scale``'s
            pseudo-metres. Visibility is ``3.912 / beta``; 0.005 is light
            haze, 0.02 is fog, 0.06+ is dense fog.
        scattering_gamma_range: wavelength dependence of extinction. 0 =
            neutral dense fog, ~1.3 = blue-shifted thin haze, 4 = Rayleigh.
        atmospheric_light: fixed airlight RGB, or ``None`` to estimate it per
            image with a dark-channel-prior-style search for the brightest
            haze-opaque region.
        depth_scale: pseudo-metres at relative depth 1.0, for this transform
            only. ``None`` uses the provider's ``depth_range_m``, which is the
            place to calibrate a whole pipeline.
        forward_scatter_range: how strongly fog blurs distant detail, as a
            fraction of the image diagonal at full optical depth. 0 disables
            the blur and leaves fog as a pure blend.
        sky_optical_depth_cap: optical depth ceiling applied to sky-like
            pixels, so the sky desaturates instead of collapsing to a flat
            airlight colour.

    Targets:
        image

    Image types:
        uint8, float32
    """

    class InitSchema(BaseTransformInitSchema):
        depth_provider: SharedDepthProvider
        beta_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 2.0)),
        ]
        scattering_gamma_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 4.0)),
        ]
        atmospheric_light: tuple[int, int, int] | None
        depth_scale: float | None
        forward_scatter_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 0.02)),
        ]
        sky_optical_depth_cap: float = Field(ge=0.0, le=10.0)

    def __init__(
        self,
        depth_provider: SharedDepthProvider,
        beta_range: tuple[float, float] = (0.01, 0.08),
        scattering_gamma_range: tuple[float, float] = (0.0, 1.5),
        atmospheric_light: tuple[int, int, int] | None = None,
        depth_scale: float | None = None,
        forward_scatter_range: tuple[float, float] = (0.001, 0.005),
        sky_optical_depth_cap: float = 1.6,
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.depth_provider = require_depth_provider(depth_provider, cls_name=type(self).__name__)
        self.beta_range = beta_range
        self.scattering_gamma_range = scattering_gamma_range
        self.atmospheric_light = atmospheric_light
        self.depth_scale = depth_scale
        self.forward_scatter_range = forward_scatter_range
        self.sky_optical_depth_cap = sky_optical_depth_cap

    def get_params_dependent_on_data(
        self, params: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, Any]:
        scene = scene_of(data)
        image = first_frame(scene)
        height, width = params["shape"][:2]
        diagonal = float(np.hypot(width, height))

        relative_depth = self.depth_provider.get_depth(scene)
        depth_pm = self.depth_provider.to_metric(relative_depth, far_m=self.depth_scale)
        # The network's depth for sky is arbitrary, so it is found by other
        # means and put at the far plane; see utils/sky.py.
        linear_scene, _scale, _is_integer = to_linear_unit(image)
        sky = sky_to_depth_edge(
            sky_weight(linear_scene, relative_depth), relative_depth, linear_scene
        )
        depth_pm = lift_sky(depth_pm, sky)

        beta = self.py_random.uniform(*self.beta_range)
        gamma = self.py_random.uniform(*self.scattering_gamma_range)
        beta_rgb = channel_extinction(beta, gamma)

        # Optical depth per channel, with a ceiling on sky-like pixels so the
        # sky keeps its gradient instead of flattening to pure airlight.
        optical_depth = depth_pm[..., None] * beta_rgb[None, None, :]
        capped = np.minimum(optical_depth, self.sky_optical_depth_cap)
        optical_depth = optical_depth * (1.0 - sky[..., None]) + capped * sky[..., None]

        transmission = np.exp(-optical_depth).astype(np.float32)

        # beta is the whole chain here: it fixes the visibility via
        # Koschmieder and the visibility is what the effect is *about*. Logged
        # because it is a local variable that never reaches the renderer -
        # only exp(-beta z) does - so without this the one derivation worth
        # auditing has to be run backwards out of the transmission field.
        record_draw(
            beta=beta,
            scattering_gamma=gamma,
            beta_rgb=beta_rgb,
            koschmieder_visibility_m=3.912 / beta if beta > 1e-9 else float("inf"),
            depth_min_m=lambda: float(depth_pm.min()),
            depth_max_m=lambda: float(depth_pm.max()),
        )

        if self.atmospheric_light is not None:
            airlight = np.array(self.atmospheric_light, dtype=np.float32) / 255.0
        else:
            # The horizon sky's radiance: read off the sky where there is
            # one, inferred where there is not. Not the frame's brightest
            # pixels, which at night are lamps. Kept gamma-encoded in [0, 1]
            # here, like `atmospheric_light`, and decoded in `apply`.
            airlight = linear_to_srgb(veil_radiance_linear(linear_scene, sky))

        return {
            "transmission": transmission,
            "airlight": airlight.astype(np.float32),
            "scatter_sigma": self.py_random.uniform(*self.forward_scatter_range) * diagonal,
        }

    def apply(
        self,
        img: np.ndarray,
        transmission: np.ndarray,
        airlight: np.ndarray,
        scatter_sigma: float,
        **params: Any,
    ) -> np.ndarray:
        linear, scale, is_integer = to_linear_unit(img)
        airlight_linear = _srgb_scalar_to_linear(airlight)

        # Forward scatter: fog is a low-pass filter, so distant detail softens
        # as well as fades. Blur strength follows optical depth, which the
        # mean transmission already encodes.
        if scatter_sigma > 0.3:
            mean_t = channel_mean(transmission)
            softness = np.clip(1.0 - mean_t, 0.0, 1.0)[..., None]
            blurred = cv2.GaussianBlur(linear, (0, 0), sigmaX=scatter_sigma)
            linear = linear * (1.0 - softness) + blurred * softness

        out = linear * transmission + airlight_linear[None, None, :] * (1.0 - transmission)
        return from_linear_unit(out, scale, is_integer, img.dtype)


def _srgb_scalar_to_linear(color: np.ndarray) -> np.ndarray:
    from weathermentations.utils.photometry import srgb_to_linear

    return srgb_to_linear(color.reshape(1, 1, -1)).reshape(-1)


