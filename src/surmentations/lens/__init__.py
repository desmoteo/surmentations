"""What is on or in front of the lens, and what the lens does with bright light.

On the glass: water, dirt, frost, snow, paint or tape. On the housing or
just in front of it: insects, a spider's web, a branch. In the lens: the
flare of the sun and of the frame's own lamps, the illuminator's
reflection in the dome, and a focus that has drifted. None of these needs
a depth map; ``FocusShift`` uses one if it is given.
"""

from surmentations.lens.dirt import DirtSoil
from surmentations.lens.dome_glare import IrDomeGlare
from surmentations.lens.focus import FocusShift
from surmentations.lens.foliage import FoliageIntrusion
from surmentations.lens.frost import FrostOnGlass
from surmentations.lens.occluders import HousingOccluders
from surmentations.lens.snow_cap import SnowOnHousing
from surmentations.lens.source_glare import SourceGlare
from surmentations.lens.spider_web import SpiderWeb
from surmentations.lens.sun_flare import SunFlare
from surmentations.lens.tamper import LensTamper
from surmentations.lens.water_droplets import WaterDroplets

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
