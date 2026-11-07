"""What is in the air: fog, rain and snow, by day and under the camera's own lamps.

The daylight effects (:class:`RandomFog`, :class:`RandomRain`,
:class:`RandomSnow`) need a depth map and so a
:class:`~weathermentations.depth.SharedDepthProvider`, and so do the three
that depend on where things are: :class:`GroundFog`, a layer on the ground;
:class:`LocalPlumes`, steam standing at a point of the scene; and
:class:`HeatShimmer`, turbulence over a long path. The night effects lit by
the camera (:class:`NightRain`, :class:`NightSnow`, :class:`NightInsects`)
do not, and :class:`NightFog` does.
"""

from weathermentations.atmosphere.fog import RandomFog
from weathermentations.atmosphere.ground_fog import GroundFog
from weathermentations.atmosphere.heat_shimmer import HeatShimmer
from weathermentations.atmosphere.night_fog import NightFog
from weathermentations.atmosphere.night_insects import NightInsects
from weathermentations.atmosphere.night_rain import NightRain
from weathermentations.atmosphere.night_snow import NightSnow
from weathermentations.atmosphere.plumes import LocalPlumes
from weathermentations.atmosphere.rain import RandomRain
from weathermentations.atmosphere.snow import RandomSnow

__all__ = [
    "GroundFog",
    "HeatShimmer",
    "LocalPlumes",
    "NightFog",
    "NightInsects",
    "NightRain",
    "NightSnow",
    "RandomFog",
    "RandomRain",
    "RandomSnow",
]
