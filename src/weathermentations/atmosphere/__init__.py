"""Weather in the air: fog, rain and snow, by day and under the camera's own lamps.

The daylight effects (:class:`RandomFog`, :class:`RandomRain`,
:class:`RandomSnow`) need a depth map and so a
:class:`~weathermentations.depth.SharedDepthProvider`; the night ones lit by
the camera (:class:`NightRain`, :class:`NightSnow`) do not, and
:class:`NightFog` does.
"""

from weathermentations.atmosphere.fog import RandomFog
from weathermentations.atmosphere.night_fog import NightFog
from weathermentations.atmosphere.night_insects import NightInsects
from weathermentations.atmosphere.night_rain import NightRain
from weathermentations.atmosphere.night_snow import NightSnow
from weathermentations.atmosphere.rain import RandomRain
from weathermentations.atmosphere.snow import RandomSnow

__all__ = [
    "NightFog",
    "NightInsects",
    "NightRain",
    "NightSnow",
    "RandomFog",
    "RandomRain",
    "RandomSnow",
]
