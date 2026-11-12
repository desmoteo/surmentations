"""What is on or in front of the lens, and what the lens does with bright light.

Dirt, water, insects and webs on the housing; a web across it, lit by the
illuminator; the flare of the sun and of the frame's own lamps. None of these
needs a depth map.
"""

from weathermentations.lens.dirt import DirtSoil
from weathermentations.lens.occluders import HousingOccluders
from weathermentations.lens.source_glare import SourceGlare
from weathermentations.lens.spider_web import SpiderWeb
from weathermentations.lens.sun_flare import SunFlare
from weathermentations.lens.water_droplets import WaterDroplets

__all__ = ["DirtSoil", "HousingOccluders", "SourceGlare", "SpiderWeb", "SunFlare", "WaterDroplets"]
