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

from surmentations.base import (
    CameraTransform,
    first_frame,
    require_depth_provider,
    scene_of,
)
from surmentations.depth.provider import SharedDepthProvider
from surmentations.diagnostics import record_draw
from surmentations.perception.labels import SceneClass
from surmentations.utils.channels import channel_mean
from surmentations.utils.color import veil_radiance_linear
from surmentations.utils.photometry import from_linear_unit, linear_to_srgb, to_linear_unit
from surmentations.utils.road import vehicle_lamps, vehicles_as_seen
from surmentations.utils.sky import (
    extinction_for_visibility,
    lift_sky,
    scene_extent_m,
    sky_to_depth_edge,
    sky_weight,
)

# Rec.709 primaries' representative wavelengths, in nm.
_RGB_WAVELENGTHS = np.array([600.0, 550.0, 450.0], dtype=np.float32)

# How much fog a line of sight straight up runs through, in metres: the
# depth of a bank of fog. A line of sight to the sky near the horizon runs
# through far more of it, by one over the sine of how high it looks, which
# is taken no lower than the second number. The sky is held back from the
# fog's own colour (``sky_optical_depth_cap``) so that haze leaves it its
# gradient; that is for the sky overhead, and low down the fog hides the
# sky as it hides what stands in front of it. And the field of view a
# frame is taken to have where no provider says.
_FOG_BEFORE_THE_SKY_M = 150.0
_LOWEST_LOOK = 0.03
_ASSUMED_FOV_DEG = 60.0
# The share of vehicles seen from the front, and from behind, whose lamps
# are lit in fog by day: nearly all, at either end.
_LIT_IN_FOG = (0.95, 0.95)
# The halo round a lamp in fog: how far it reaches, as a share of the
# frame's diagonal, and how much of the light the fog scattered it shows.
_HALO_REACH = 0.012
_HALO_SHARE = 2.5


def channel_extinction(beta: float, gamma: float) -> np.ndarray:
    """Per-channel extinction ``beta_c = beta * (lambda_c / 550)^-gamma``.

    ``gamma = 0`` gives neutral grey fog (large droplets, dense fog);
    ``gamma ~ 1.3`` gives the cool, blue-shifted distance of thin haze;
    ``gamma = 4`` is Rayleigh scattering off clean air.
    """
    return beta * (_RGB_WAVELENGTHS / 550.0) ** (-gamma)


class RandomFog(CameraTransform):
    """Physically-grounded fog/haze, conditioned on scene depth.

    With a provider that knows more than depth
    (:mod:`surmentations.perception`): the sky is what the scene's labels
    call sky, and the fog is over real metres where the camera's height is
    known, unless ``depth_scale`` says otherwise.

    Args:
        depth_provider: shared :class:`~surmentations.depth.provider.SharedDepthProvider`.
        visibility_range_scene: visibility as a multiple of the scene's
            depth, drawn per call, in place of ``beta_range``. A fog of one
            extinction is a different picture on every scene: a visibility
            of a hundred metres hides nothing of a car park and all of a
            valley. ``(0.5, 1.0)`` is fog that takes the far half of
            whatever scene it is given. The scene's depth is the distance
            of the farthest thing in it that is not sky. ``None``, the
            default, uses ``beta_range``.
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
            airlight colour. That is for the sky overhead: a line of
            sight to the sky runs through the fog first, 150 m of it
            straight up and far more near the horizon, and the ceiling is
            never lower than that.
        vehicle_lamp_range: how brightly the traffic's lamps are lit, drawn
            per call; 1 is a dipped headlamp as a camera sees it by day, 0
            none, which is the default. Traffic in fog drives with its
            lamps on, and in thick fog a car is two haloes before it is a
            car: each lamp is seen through the fog in front of it, dimmed
            by that fog, and what the fog takes out of its light it
            scatters round it. It needs a provider that reads the
            vehicles (``vehicles=``): the lamps are lit where the reader
            found them, on the end of each vehicle it saw.

    Targets:
        image

    Image types:
        uint8, float32
    """

    class InitSchema(BaseTransformInitSchema):
        """The arguments of ``RandomFog`` as Albumentations validates them: each
        range ordered low to high and within the bounds given here.
        """
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
        visibility_range_scene: (
            Annotated[
                tuple[float, float],
                AfterValidator(nondecreasing),
                AfterValidator(check_range_bounds(0.01, 100.0)),
            ]
            | None
        )
        forward_scatter_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 0.02)),
        ]
        sky_optical_depth_cap: float = Field(ge=0.0, le=10.0)
        vehicle_lamp_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 4.0)),
        ]

    def __init__(
        self,
        depth_provider: SharedDepthProvider,
        beta_range: tuple[float, float] = (0.01, 0.08),
        scattering_gamma_range: tuple[float, float] = (0.0, 1.5),
        atmospheric_light: tuple[int, int, int] | None = None,
        depth_scale: float | None = None,
        forward_scatter_range: tuple[float, float] = (0.001, 0.005),
        sky_optical_depth_cap: float = 1.6,
        visibility_range_scene: tuple[float, float] | None = None,
        vehicle_lamp_range: tuple[float, float] = (0.0, 0.0),
        p: float = 0.5,
    ):
        """Store the ranges; each argument is described in the class docstring.

        Raises ``ValueError`` without a ``depth_provider``: the one
        :class:`~surmentations.depth.provider.SharedDepthProvider` of the
        pipeline must be passed in, so that depth is estimated once per frame.
        """
        super().__init__(p=p)
        self.depth_provider = require_depth_provider(depth_provider, cls_name=type(self).__name__)
        self.beta_range = beta_range
        self.visibility_range_scene = visibility_range_scene
        self.scattering_gamma_range = scattering_gamma_range
        self.atmospheric_light = atmospheric_light
        self.depth_scale = depth_scale
        self.forward_scatter_range = forward_scatter_range
        self.sky_optical_depth_cap = sky_optical_depth_cap
        self.vehicle_lamp_range = vehicle_lamp_range

    def get_params_dependent_on_data(
        self, params: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, Any]:
        """Draw the fog for one frame from its depth.

        The sky is found and put at the far plane, where the network's depth
        for it is meaningless. Returns the per-pixel ``transmission``
        ``exp(-beta z)`` for each channel, with ``beta`` drawn through the
        visibility it implies (Koschmieder), the ``airlight`` colour read off
        the sky or inferred where there is none, and the forward-scatter
        blur's ``scatter_sigma``.
        """
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
            sky_weight(linear_scene, relative_depth, self.depth_provider.get_labels(scene)),
            relative_depth,
            linear_scene,
        )
        depth_pm = lift_sky(depth_pm, sky)

        if self.visibility_range_scene is None:
            beta = self.py_random.uniform(*self.beta_range)
        else:
            # Set against the scene and not in metres: the same fog on a
            # car park and on a valley.
            visibility = self.py_random.uniform(*self.visibility_range_scene)
            beta = extinction_for_visibility(visibility * scene_extent_m(depth_pm, sky))
        gamma = self.py_random.uniform(*self.scattering_gamma_range)
        beta_rgb = channel_extinction(beta, gamma)

        # Optical depth per channel, with a ceiling on sky-like pixels so the
        # sky keeps its gradient instead of flattening to pure airlight.
        optical_depth = depth_pm[..., None] * beta_rgb[None, None, :]
        # The ceiling is no lower than the fog between the lens and the
        # sky is deep, which near the horizon is very deep: the sky used
        # to stay darker than the buildings in front of it, which the fog
        # had turned to its own colour, and their outline showed as a pale
        # flat shape with nothing behind it.
        horizon = self.depth_provider.horizon_row(scene)
        if horizon is None:
            # Where the sky comes lowest in the frame.
            rows_of_sky = np.flatnonzero((sky > 0.5).any(axis=1))
            horizon = float(rows_of_sky.max()) if rows_of_sky.size else 0.5 * height
        camera = self.depth_provider.settle_camera(scene)
        focal = (
            0.5 * width / np.tan(np.deg2rad(0.5 * _ASSUMED_FOV_DEG))
            if camera is None
            else camera.focal_px
        )
        looks_up = np.arctan((horizon - np.arange(height, dtype=np.float32)) / focal)
        through = _FOG_BEFORE_THE_SKY_M / np.maximum(np.sin(looks_up), _LOWEST_LOOK)
        ceiling = np.maximum(
            self.sky_optical_depth_cap, through[:, None, None] * beta_rgb[None, None, :]
        )
        capped = np.minimum(optical_depth, ceiling)
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

        drawn: dict[str, Any] = {
            "transmission": transmission,
            "airlight": airlight.astype(np.float32),
            "scatter_sigma": self.py_random.uniform(*self.forward_scatter_range) * diagonal,
        }
        # The traffic's lamps, last of the draws.
        lamps = self.py_random.uniform(*self.vehicle_lamp_range)
        lamp_seed = self.py_random.randint(0, 2**31 - 1)
        labels = self.depth_provider.get_labels(scene)
        lit = False
        if lamps > 0.0 and labels is not None:
            named = labels == int(SceneClass.VEHICLE)
            traffic = vehicles_as_seen(named, self.depth_provider.get_vehicles(scene))
            if traffic:
                light, _glow, cores = vehicle_lamps(
                    None,
                    traffic,
                    np.random.default_rng(lamp_seed),
                    lamps,
                    linear_scene,
                    shape=(height, width),
                    lit_share=_LIT_IN_FOG,
                )
                if float(cores.max()) > 0.0:
                    drawn["lamp_light"] = light
                    drawn["halo_sigma"] = _HALO_REACH * diagonal
                    lit = True
        record_draw(vehicle_lamps=lamps, lamps_lit=lit)
        return drawn

    def apply(
        self,
        img: np.ndarray,
        transmission: np.ndarray,
        airlight: np.ndarray,
        scatter_sigma: float,
        lamp_light: np.ndarray | None = None,
        halo_sigma: float = 0.0,
        **params: Any,
    ) -> np.ndarray:
        """Soften distant detail by forward scatter, then veil the frame with airlight.

        In linear light: ``linear * t + airlight * (1 - t)``, after a blur
        whose strength follows the optical depth. The traffic's lamps,
        where they are lit, are added after the veil: each through the fog
        in front of it, and round it the halo of what that fog scattered
        out of its light.
        """
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
        if lamp_light is not None:
            # A lamp is seen through the fog between it and the lens. What
            # that fog takes out of its beam it sends on at small angles:
            # a halo round the lamp, widest and brightest where about half
            # of the light has been scattered, and gone with the lamp
            # where the fog is too thick for either.
            through = channel_mean(transmission)[..., None]
            scattered = lamp_light * (through * (1.0 - through))
            wide = cv2.GaussianBlur(scattered, (0, 0), sigmaX=max(halo_sigma, 1.0))
            near = cv2.GaussianBlur(scattered, (0, 0), sigmaX=max(0.3 * halo_sigma, 0.7))
            halo = np.asarray(wide, dtype=np.float32) + np.asarray(near, dtype=np.float32)
            out = out + lamp_light * through + np.float32(_HALO_SHARE) * halo
        return from_linear_unit(out, scale, is_integer, img.dtype)


def _srgb_scalar_to_linear(color: np.ndarray) -> np.ndarray:
    """One sRGB colour, ``(C,)`` in ``[0, 1]``, in linear light."""
    from surmentations.utils.photometry import srgb_to_linear

    return srgb_to_linear(color.reshape(1, 1, -1)).reshape(-1)


