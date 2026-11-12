"""What is on or in front of the lens, and what the lens does with bright light.

On the glass: water, dirt, frost, snow, paint or tape. On the housing or
just in front of it: insects, a spider's web, a branch. In the lens: the
flare of the sun and of the frame's own lamps, the illuminator's
reflection in the dome, and a focus that has drifted. None of these needs
a depth map; ``FocusShift`` uses one if it is given.
"""

from weathermentations.lens.dirt import DirtSoil
from weathermentations.lens.dome_glare import IrDomeGlare
from weathermentations.lens.focus import FocusShift
from weathermentations.lens.foliage import FoliageIntrusion
from weathermentations.lens.frost import FrostOnGlass
from weathermentations.lens.occluders import HousingOccluders
from weathermentations.lens.snow_cap import SnowOnHousing
from weathermentations.lens.source_glare import SourceGlare
from weathermentations.lens.spider_web import SpiderWeb
from weathermentations.lens.sun_flare import SunFlare
from weathermentations.lens.tamper import LensTamper
from weathermentations.lens.water_droplets import WaterDroplets

__all__ = [
    "DirtSoil",
    "FocusShift",
    "FoliageIntrusion",
    "FrostOnGlass",
    "HousingOccluders",
    "IrDomeGlare",
    "LensTamper",
    "SnowOnHousing",
    "SourceGlare",
    "SpiderWeb",
    "SunFlare",
    "WaterDroplets",
]
