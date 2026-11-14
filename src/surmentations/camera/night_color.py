"""Night in colour: the scene lit by its street lamps and its traffic.

Physical model
--------------
Most cameras do not go to infrared at dusk. With a fast lens and a large
pixel they hold colour through an ordinary lit night, and a street at night
in colour is not a day frame turned down:

**It is lit from a few points.** Each street lamp throws a pool under
itself that falls off with the square of the distance, each vehicle throws
its beams, and between and beyond them there is only the glow of the sky.
A frame turned down is evenly dim, with the sun's shadows still in it.

**The light has a colour, and not one.** A high-pressure sodium lamp is a
black body of about 2100 K to a camera: orange, with almost no blue, and
under it a blue car is a grey one. A white LED is 3000 to 5700 K. A
headlamp is whiter than either and a tail lamp is red. The camera balances
for one of them, and everything lit by another has a cast.

**The lamps are in the frame.** They are far over full scale and bloom,
and the camera cannot expose for them and the road at once.

**The sensor is short of light.** It gains up, and the noise is in the
colour as much as in the brightness: each channel has its own.

Where the lamps are comes from the provider, as for
:class:`~surmentations.camera.night_ir.NightIrMode`: its vehicle reader
finds the street lamps and the traffic's lamps
(:mod:`surmentations.perception.vehicles`), its scene labels say what is
ground and what is sky, and the light is worked out in space
(:mod:`surmentations.camera.night_lighting`). A provider that knows of no
lamps leaves the sky's glow alone to light the scene, which is a night
with no lighting at all: dim, even and noisy.

References:

* Elvidge, C. D., Keith, D. M., Tuttle, B. T., Baugh, K. E. (2010).
  Spectral Identification of Lighting Type and Character. Sensors 10(4),
  3961-3988. https://doi.org/10.3390/s100403961 (the spectra of sodium,
  metal halide and LED street lighting; the colour temperatures here are
  the round ones commonly quoted for each, not read off its tables)
"""

from __future__ import annotations

from typing import Annotated, Any

import numpy as np
from albumentations.core.pydantic import check_range_bounds, nondecreasing
from albumentations.core.transforms_interface import BaseTransformInitSchema
from pydantic import AfterValidator

from surmentations.base import CameraTransform, first_frame, require_depth_provider, scene_of
from surmentations.camera.night_lighting import (
    SceneLighting,
    light_scene,
    lit_air_of,
    note_lit_air,
)
from surmentations.depth.provider import SharedDepthProvider
from surmentations.diagnostics import record_draw
from surmentations.perception.labels import SceneClass
from surmentations.utils.channels import percentile
from surmentations.utils.night_light import even_road
from surmentations.utils.photometry import from_linear_unit, luminance, to_linear_unit
from surmentations.utils.sky import sky_to_depth_edge, sky_weight
from surmentations.utils.spectral import blackbody_rgb

# What the camera's white balance is set for at night, in kelvin: between
# the lamps it meets, so sodium is orange to it and a cool LED a little blue.
_BALANCED_FOR_K = 4300.0
# A headlamp's colour temperature, and the sky's glow over a town: lit
# from below by the town's own lamps, and bluer than they are.
_HEADLAMP_K = 4800.0
_SKY_GLOW_K = 5500.0
# What is left of the sky itself, as a share of what the glow would make
# of a pale surface, and how much of the glow round a lamp the sensor shows.
_NIGHT_SKY_LEVEL = 0.35
_LAMP_BLOOM = 0.3
# The auto-exposure holds the brightest hundredth of the scene's surfaces
# to this level once lamps light it, and otherwise puts its 80th
# percentile at the target.
_HIGHLIGHT_LEVEL = 0.95
_HIGHLIGHT_PERCENTILE = 99.0
# What a road in a beam sends back at the least, as a share of the middle
# value of the frame's road: the day's deep shade was the light's doing.
_BEAM_ROAD = 0.35
# Shot noise against the square root of the signal, at a gain of 1.
_SHOT = 0.03


def lamp_colour(temperature_k: float) -> np.ndarray:
    """A lamp of a colour temperature as the night camera's white balance shows it.

    Linear RGB with a luminance of 1: a black body at ``temperature_k``
    against the one the camera is balanced for. 2100 K is sodium's orange,
    4300 K is white, 5700 K a cool LED's faint blue.
    """
    tint = blackbody_rgb(temperature_k) / blackbody_rgb(_BALANCED_FOR_K)
    tint = tint.astype(np.float32)
    level = float(luminance(tint.reshape(1, 1, 3)).ravel()[0])
    scaled: np.ndarray = tint / max(level, 1e-6)
    return scaled


class NightColorMode(CameraTransform):
    """A night in colour, lit by the scene's street lamps and its traffic.

    A daylight frame as a colour camera would show the same scene at night.
    See the module docstring for the model.

    With a provider that carries scene labels
    (:mod:`surmentations.perception.labels`) and a vehicle reader
    (:mod:`surmentations.perception.vehicles`): the street lamps the reader
    finds throw their pools in the lamps' own colour and burn, the
    traffic's lamps are lit, white at the front and red at the rear, each
    lit vehicle throws its beams, the road loses the day's light and shade,
    and the sky is dark. With labels alone the nearest street lamps are
    found from the lamp label. Without either there is nothing to say
    where a lamp is, and the frame is lit by the sky's glow alone.

    Args:
        depth_provider: shared :class:`~surmentations.depth.provider.SharedDepthProvider`.
        lamp_temperature_range_k: the colour temperature of the street
            lighting, drawn per call: one kind of lamp to a street. 1900 to
            2200 K is high-pressure sodium, 2700 to 3000 K a warm LED, 4000
            K a neutral one, 5000 K and over a cool one.
        street_lamp_range: how much light a street lamp throws on the road
            straight under it, in unit exposures, drawn per call; 0 is
            none.
        vehicle_lamp_range: how brightly the traffic's lamps are lit, drawn
            per call; 1 is dipped beams, 0 none. It needs a vehicle reader
            on the provider.
        sky_glow_range: the light of the night sky on everything, as a
            share of what a street lamp puts under itself, drawn per call.
            A few hundredths in a town; it is all the light there is where
            the provider knows of no lamps.
        target_level_range: the level the camera's auto-exposure drives
            the scene to, as a fraction of full scale at the 80th
            percentile. Once lamps light the scene it also holds the lit
            road below white, as a camera does.
        gain_range: sensor gain, which sets how far the noise is lifted and
            not how bright the frame is: the auto-exposure has placed that.
        read_noise_range: signal-independent noise per channel, as a
            fraction of full scale.

    Targets:
        image

    Image types:
        uint8, float32
    """

    class InitSchema(BaseTransformInitSchema):
        """The arguments of ``NightColorMode`` as Albumentations validates them: each
        range ordered low to high and within the bounds given here.
        """

        depth_provider: SharedDepthProvider
        lamp_temperature_range_k: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(1500.0, 8000.0)),
        ]
        street_lamp_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 5.0)),
        ]
        vehicle_lamp_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 2.0)),
        ]
        sky_glow_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 1.0)),
        ]
        target_level_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.02, 0.95)),
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

    def __init__(
        self,
        depth_provider: SharedDepthProvider,
        lamp_temperature_range_k: tuple[float, float] = (1900.0, 5500.0),
        street_lamp_range: tuple[float, float] = (0.6, 1.4),
        vehicle_lamp_range: tuple[float, float] = (0.7, 1.3),
        sky_glow_range: tuple[float, float] = (0.01, 0.05),
        target_level_range: tuple[float, float] = (0.12, 0.3),
        gain_range: tuple[float, float] = (1.0, 3.0),
        read_noise_range: tuple[float, float] = (0.004, 0.02),
        p: float = 0.5,
    ):
        """Store the ranges; each argument is described in the class docstring.

        Raises ``ValueError`` without a ``depth_provider``: the one
        :class:`~surmentations.depth.provider.SharedDepthProvider` of the
        pipeline must be passed in, so that depth is estimated once per frame.
        """
        super().__init__(p=p)
        self.depth_provider = require_depth_provider(depth_provider, cls_name=type(self).__name__)
        self.lamp_temperature_range_k = lamp_temperature_range_k
        self.street_lamp_range = street_lamp_range
        self.vehicle_lamp_range = vehicle_lamp_range
        self.sky_glow_range = sky_glow_range
        self.target_level_range = target_level_range
        self.gain_range = gain_range
        self.read_noise_range = read_noise_range

    def get_params_dependent_on_data(
        self, params: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, Any]:
        """Draw the night's lighting and the camera, and light the scene from its lamps.

        Returns the scene's ``lighting``
        (:func:`~surmentations.camera.night_lighting.light_scene`), the
        colours of its street lamps, its headlamps and its sky, the sky
        mask where the scene has labels, and the camera's exposure target,
        gain and noise.
        """
        scene = scene_of(data)
        depth_m = np.maximum(self.depth_provider.get_metric_depth(scene), 0.3)
        labels = self.depth_provider.get_labels(scene)

        temperature = self.py_random.uniform(*self.lamp_temperature_range_k)
        street = self.py_random.uniform(*self.street_lamp_range)
        lamps = self.py_random.uniform(*self.vehicle_lamp_range)
        lamp_seed = self.py_random.randint(0, 2**31 - 1)
        lighting = light_scene(
            self.depth_provider, scene, labels, depth_m, lamp_seed, lamps, street
        )
        street_colour, head_colour = lamp_colour(temperature), lamp_colour(_HEADLAMP_K)

        night_sky = None
        road_level = None
        if labels is not None:
            relative_depth = self.depth_provider.get_depth(scene)
            linear, _scale, _is_integer = to_linear_unit(first_frame(scene))
            night_sky = sky_to_depth_edge(
                sky_weight(linear, relative_depth, labels), relative_depth, linear
            )
            paved = labels == int(SceneClass.ROAD)
            if lighting.in_beam is not None and int(paved.sum()) > 50:
                road_level = float(np.median(luminance(linear)[paved]))

        # The light in the air, in the lamps' colours, for whatever puts
        # something in that air afterwards.
        air = lit_air_of(lighting, street_colour, head_colour)

        record_draw(
            lamp_temperature_k=temperature,
            street_lamp_level=street,
            vehicle_lamps=lamps,
            vehicles_read=lighting.vehicles_read,
            vehicles_lit=lighting.vehicles_lit,
            street_lamps=lighting.street_lamps if lighting.lit else 0,
        )
        return {
            "lighting": lighting,
            "street_colour": street_colour,
            "head_colour": head_colour,
            "sky_colour": lamp_colour(_SKY_GLOW_K),
            "sky_glow": self.py_random.uniform(*self.sky_glow_range),
            "night_sky": night_sky,
            "road_level": road_level,
            "target_level": self.py_random.uniform(*self.target_level_range),
            "gain": self.py_random.uniform(*self.gain_range),
            "read_noise": self.py_random.uniform(*self.read_noise_range),
            "noise_seed": self.py_random.randint(0, 2**31 - 1),
            "air": air,
            "exposed": [1.0],
        }

    def apply_with_params(
        self, params: dict[str, Any], *args: Any, **kwargs: Any
    ) -> dict[str, Any]:
        """Apply, then note for the frame returned what light hangs in its air.

        For the fog, rain or snow a pipeline puts in that air afterwards
        (:func:`~surmentations.camera.night_lighting.note_lit_air`).
        """
        result = super().apply_with_params(params, *args, **kwargs)
        note_lit_air(self.depth_provider, params, result)
        return result

    def apply(
        self,
        img: np.ndarray,
        lighting: SceneLighting,
        street_colour: np.ndarray,
        head_colour: np.ndarray,
        sky_colour: np.ndarray,
        sky_glow: float,
        target_level: float,
        gain: float,
        read_noise: float,
        noise_seed: int,
        night_sky: np.ndarray | None = None,
        road_level: float | None = None,
        exposed: list[float] | None = None,
        **params: Any,
    ) -> np.ndarray:
        """Render the frame as a colour camera sees the scene at night.

        The day frame's colours, taken as what each surface sends back,
        times the light that reaches it from the sky, the street lamps and
        the headlamps, each in its colour; exposed by the camera; then the
        lamps themselves, and noise in each channel.
        """
        linear, scale, is_integer = to_linear_unit(img)
        surface = linear
        if lighting.road is not None or lighting.in_beam is not None:
            # The day's light and shade come off the road, and a road in
            # the day's deep shade is not left black under a beam: both
            # were the sun's doing and not the road's.
            level = luminance(linear)
            evened = level if lighting.road is None else even_road(level, lighting.road)
            if lighting.in_beam is not None and road_level is not None:
                floor = np.float32(_BEAM_ROAD * road_level) * lighting.in_beam
                # A surface that photographed as black has no colour to
                # keep: what the beam shows of it is grey.
                surface = np.maximum(surface, floor[..., None])
                evened = np.maximum(evened, floor)
            surface = surface * (evened / np.maximum(luminance(surface), 1e-4))[..., None]

        arriving = np.float32(sky_glow) * sky_colour.reshape(1, 1, 3)
        if lighting.street is not None:
            arriving = arriving + lighting.street[..., None] * street_colour
        if lighting.beams is not None:
            arriving = arriving + lighting.beams[..., None] * head_colour
        signal = surface * arriving
        if night_sky is not None:
            signal = signal * (1.0 - np.float32(1.0 - _NIGHT_SKY_LEVEL) * night_sky)[..., None]

        # Auto-exposure, on the frame's brightness: for the 80th percentile,
        # and once lamps light the scene, no further than keeps the lit
        # road below white.
        brightness = luminance(signal)
        reference = percentile(brightness, 80.0)
        if reference > 1e-7:
            exposure = target_level / reference
            if lighting.lit:
                brightest = percentile(brightness, _HIGHLIGHT_PERCENTILE)
                exposure = min(exposure, _HIGHLIGHT_LEVEL / max(brightest, 1e-7))
            signal = signal * np.float32(exposure)
            if exposed is not None:
                exposed[0] = float(exposure)

        # The lamps themselves, at their own level: the traffic's in their
        # own colours, the street's in the street lighting's.
        if lighting.vehicle_light is not None and lighting.vehicle_glow is not None:
            signal = signal + lighting.vehicle_light
            signal = signal + np.float32(_LAMP_BLOOM) * lighting.vehicle_glow
        if lighting.head_light is not None and lighting.head_glow is not None:
            burning = lighting.head_light + np.float32(_LAMP_BLOOM) * lighting.head_glow
            signal = signal + burning[..., None] * street_colour

        # Shot noise and read noise, each channel its own: the grain of a
        # night frame is coloured.
        rng = np.random.default_rng(noise_seed)
        noise = rng.standard_normal(signal.shape, dtype=np.float32)
        grain = (np.sqrt(np.maximum(signal, 0.0)) * _SHOT + read_noise) * gain
        out = np.clip(signal + noise * grain.astype(np.float32), 0.0, 1.0).astype(np.float32)
        return from_linear_unit(out, scale, is_integer, img.dtype)
