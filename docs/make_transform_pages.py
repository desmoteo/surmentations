"""Write one page per transform, from its source, into ``docs/transforms``.

Run with::

    python docs/make_transform_pages.py

Each page says what the transform models and how, and is assembled from two
places so that it cannot drift from the code:

* **From the source**, read when the page is written: the model, which is
  the module's own docstring; every parameter with its default and its
  meaning, from the class; the stage it sorts into; whether it needs depth;
  what it does to annotations; and its row of the severity gallery.
* **From this file**: a diagram of the algorithm, and the references.

A page is therefore regenerated, not edited. Change the docstring or the
tables below and run this again;
``tests/unit/test_documentation.py`` fails while a page and its source
disagree, and when a transform has no page at all.

Ten of the diagrams are the ones ``docs/how_it_works.md`` already draws and
are read from it by the heading they stand under, so the two documents show
the same picture.
"""

from __future__ import annotations

import importlib
import inspect
import re
import sys
from pathlib import Path

DOCS = Path(__file__).resolve().parent
ROOT = DOCS.parent
OUT = DOCS / "transforms"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from docs.generate_examples import EFFECTS  # noqa: E402
from surmentations.presets.pipeline import stage_of  # noqa: E402

PACKAGES = {
    "atmosphere": "Weather and what is in the air",
    "scene": "The scene",
    "lens": "Lens and housing",
    "camera": "Sensor, processor and link",
}

STAGES = {
    0: "scene",
    1: "atmosphere",
    2: "optics: on the glass or the housing",
    3: "optics: in the lens",
    4: "sensor: day or night",
    5: "lit by the camera's own lamp, after the infrared conversion",
    6: "sensor",
    7: "processor: exposure",
    8: "processor: tone mapping",
    9: "processor: noise reduction",
    10: "overlay drawn by the camera, before the link",
    11: "link",
    12: "link: what it loses",
    13: "overlay drawn by the recorder, after the link",
}

# Papers and books, by a short key.
SOURCES = {
    "koschmieder": "Koschmieder, H. (1924). Theorie der horizontalen Sichtweite. "
    "*Beiträge zur Physik der freien Atmosphäre* 12, 33-53.",
    "middleton": "Middleton, W. E. K. (1952). *Vision through the Atmosphere*. "
    "University of Toronto Press.",
    "narasimhan2002": "Narasimhan, S. G. and Nayar, S. K. (2002). Vision and the atmosphere. "
    "*International Journal of Computer Vision* 48(3), 233-254.",
    "narasimhan2003": "Narasimhan, S. G. and Nayar, S. K. (2003). Shedding light on the weather. "
    "*IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*.",
    "marshall": "Marshall, J. S. and Palmer, W. McK. (1948). The distribution of raindrops with "
    "size. *Journal of Meteorology* 5, 165-166.",
    "ulbrich": "Ulbrich, C. W. (1983). Natural variations in the analytical form of the raindrop "
    "size distribution. *Journal of Climate and Applied Meteorology* 22, 1764-1775.",
    "gunn_kinzer": "Gunn, R. and Kinzer, G. D. (1949). The terminal velocity of fall for water "
    "droplets in stagnant air. *Journal of Meteorology* 6, 243-248.",
    "atlas": "Atlas, D. (1953). Optical extinction by rainfall. *Journal of Meteorology* 10, "
    "486-488.",
    "garg": "Garg, K. and Nayar, S. K. (2007). Vision and rain. *International Journal of "
    "Computer Vision* 75(1), 3-27.",
    "gunn_marshall": "Gunn, K. L. S. and Marshall, J. S. (1958). The distribution with size of "
    "aggregate snowflakes. *Journal of Meteorology* 15, 452-461.",
    "locatelli": "Locatelli, J. D. and Hobbs, P. V. (1974). Fall speeds and masses of solid "
    "precipitation particles. *Journal of Geophysical Research* 79(15), 2185-2197.",
    "warren": "Warren, S. G. (1982). Optical properties of snow. *Reviews of Geophysics* 20(1), "
    "67-89.",
    "bohren": "Bohren, C. F. (1983). Colors of snow, frozen waterfalls, and icebergs. *Journal of "
    "the Optical Society of America* 73(12), 1646-1652.",
    "gultepe": "Gultepe, I. et al. (2007). Fog research: a review of past achievements and future "
    "perspectives. *Pure and Applied Geophysics* 164, 1121-1159.",
    "morton": "Morton, B. R., Taylor, G. I. and Turner, J. S. (1956). Turbulent gravitational "
    "convection from maintained and instantaneous sources. *Proceedings of the Royal Society A* "
    "234, 1-23.",
    "kolmogorov": "Kolmogorov, A. N. (1941). The local structure of turbulence in incompressible "
    "viscous fluid for very large Reynolds numbers. *Doklady Akademii Nauk SSSR* 30, 301-305.",
    "tatarskii": "Tatarskii, V. I. (1961). *Wave Propagation in a Turbulent Medium*. McGraw-Hill.",
    "fried": "Fried, D. L. (1966). Optical resolution through a randomly inhomogeneous medium for "
    "very long and very short exposures. *Journal of the Optical Society of America* 56(10), "
    "1372-1379.",
    "roggemann": "Roggemann, M. C. and Welsh, B. M. (1996). *Imaging Through Turbulence*. CRC "
    "Press.",
    "greenewalt": "Greenewalt, C. H. (1962). Dimensional relationships for flying animals. "
    "*Smithsonian Miscellaneous Collections* 144(2).",
    "fabian": "Fabian, S. T. et al. (2024). Why flying insects gather at artificial light. "
    "*Nature Communications* 15, 689.",
    "prati": "Prati, A., Mikic, I., Trivedi, M. M. and Cucchiara, R. (2003). Detecting moving "
    "shadows: algorithms and evaluation. *IEEE Transactions on Pattern Analysis and Machine "
    "Intelligence* 25(7), 918-923.",
    "roser": "Roser, M. and Geiger, A. (2009). Video-based raindrop detection for improved image "
    "registration. *IEEE International Conference on Computer Vision Workshops*.",
    "halimeh": "Halimeh, J. C. and Roser, M. (2009). Raindrop detection on car windshields using "
    "geometric-photometric environment construction and intensity-based correlation. *IEEE "
    "Intelligent Vehicles Symposium*.",
    "you": "You, S., Tan, R. T., Kawakami, R., Mukaigawa, Y. and Ikeuchi, K. (2016). Adherent "
    "raindrop modeling, detection and removal in video. *IEEE Transactions on Pattern Analysis "
    "and Machine Intelligence* 38(9), 1721-1733.",
    "born": "Born, M. and Wolf, E. (1999). *Principles of Optics*, 7th edition. Cambridge "
    "University Press.",
    "hecht": "Hecht, E. (2017). *Optics*, 5th edition. Pearson.",
    "gu": "Gu, J., Ramamoorthi, R., Belhumeur, P. and Nayar, S. (2009). Removing image artifacts "
    "due to dirty camera lenses and thin occluders. *ACM Transactions on Graphics* 28(5) "
    "(SIGGRAPH Asia).",
    "hullin": "Hullin, M., Eisemann, E., Seidel, H.-P. and Lee, S. (2011). Physically-based "
    "real-time lens flare rendering. *ACM Transactions on Graphics* 30(4) (SIGGRAPH).",
    "talvala": "Talvala, E.-V., Adams, A., Horowitz, M. and Levoy, M. (2007). Veiling glare in "
    "high dynamic range imaging. *ACM Transactions on Graphics* 26(3) (SIGGRAPH).",
    "mccann": "McCann, J. J. and Rizzi, A. (2007). Veiling glare: the dynamic range limit of HDR "
    "images. *Proceedings of SPIE* 6492, Human Vision and Electronic Imaging XII.",
    "potmesil": "Potmesil, M. and Chakravarty, I. (1981). A lens and aperture camera model for "
    "synthetic image generation. *ACM SIGGRAPH Computer Graphics* 15(3), 297-305.",
    "zschokke": "Zschokke, S. (1999). Nomenclature of the orb-web. *Journal of Arachnology* 27, "
    "542-546.",
    "vollrath": "Vollrath, F. and Edmonds, D. T. (1989). Modulation of the mechanical properties "
    "of spider silk by coating with water. *Nature* 340, 305-307.",
    "foelix": "Foelix, R. F. (2011). *Biology of Spiders*, 3rd edition. Oxford University Press.",
    "libbrecht": "Libbrecht, K. G. (2005). The physics of snow crystals. *Reports on Progress in "
    "Physics* 68, 855-895.",
    "ribnick": "Ribnick, E., Atev, S., Masoud, O., Papanikolopoulos, N. and Voyles, R. (2006). "
    "Real-time detection of camera tampering. *IEEE International Conference on Advanced Video "
    "and Signal Based Surveillance (AVSS)*.",
    "saglam": "Saglam, A. and Temizel, A. (2009). Real-time adaptive camera tamper detection for "
    "video surveillance. *IEEE International Conference on Advanced Video and Signal Based "
    "Surveillance (AVSS)*.",
    "gates": "Gates, D. M., Keegan, H. J., Schleter, J. C. and Weidner, V. R. (1965). Spectral "
    "properties of plants. *Applied Optics* 4(1), 11-20.",
    "brown": "Brown, M. and Süsstrunk, S. (2011). Multi-spectral SIFT for scene category "
    "recognition. *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*.",
    "emva": "European Machine Vision Association. *EMVA Standard 1288: Standard for "
    "Characterization of Image Sensors and Cameras*.",
    "healey": "Healey, G. E. and Kondepudy, R. (1994). Radiometric CCD camera calibration and "
    "noise estimation. *IEEE Transactions on Pattern Analysis and Machine Intelligence* 16(3), "
    "267-276.",
    "janesick": "Janesick, J. R. (2007). *Photon Transfer*. SPIE Press.",
    "bayer": "Bayer, B. E. (1976). Color imaging array. US Patent 3,971,065.",
    "gunturk": "Gunturk, B. K., Glotzbach, J., Altunbasak, Y., Schafer, R. W. and Mersereau, "
    "R. M. (2005). Demosaicking: color filter array interpolation. *IEEE Signal Processing "
    "Magazine* 22(1), 44-54.",
    "poplin": "Poplin, D. (2006). An automatic flicker detection method for embedded camera "
    "systems. *IEEE Transactions on Consumer Electronics* 52(2), 308-311.",
    "liang": "Liang, C.-K., Chang, L.-W. and Chen, H. H. (2008). Analysis and compensation of "
    "rolling shutter effect. *IEEE Transactions on Image Processing* 17(8), 1323-1330.",
    "fergus": "Fergus, R., Singh, B., Hertzmann, A., Roweis, S. T. and Freeman, W. T. (2006). "
    "Removing camera shake from a single photograph. *ACM Transactions on Graphics* 25(3) "
    "(SIGGRAPH).",
    "wyszecki": "Wyszecki, G. and Stiles, W. S. (1982). *Color Science: Concepts and Methods, "
    "Quantitative Data and Formulae*, 2nd edition. Wiley.",
    "durand": "Durand, F. and Dorsey, J. (2002). Fast bilateral filtering for the display of "
    "high-dynamic-range images. *ACM Transactions on Graphics* 21(3) (SIGGRAPH).",
    "tomasi": "Tomasi, C. and Manduchi, R. (1998). Bilateral filtering for gray and color images. "
    "*IEEE International Conference on Computer Vision (ICCV)*.",
    "gastal": "Gastal, E. S. L. and Oliveira, M. M. (2011). Domain transform for edge-aware image "
    "and video processing. *ACM Transactions on Graphics* 30(4) (SIGGRAPH).",
    "wiegand": "Wiegand, T., Sullivan, G. J., Bjøntegaard, G. and Luthra, A. (2003). Overview of "
    "the H.264/AVC video coding standard. *IEEE Transactions on Circuits and Systems for Video "
    "Technology* 13(7), 560-576.",
    "sullivan": "Sullivan, G. J., Ohm, J.-R., Han, W.-J. and Wiegand, T. (2012). Overview of the "
    "High Efficiency Video Coding (HEVC) standard. *IEEE Transactions on Circuits and Systems "
    "for Video Technology* 22(12), 1649-1668.",
    "list": "List, P., Joch, A., Lainema, J., Bjøntegaard, G. and Karczewicz, M. (2003). Adaptive "
    "deblocking filter. *IEEE Transactions on Circuits and Systems for Video Technology* 13(7), "
    "614-619.",
    "wang": "Wang, Y. and Zhu, Q.-F. (1998). Error control and concealment for video "
    "communication: a review. *Proceedings of the IEEE* 86(5), 974-997.",
    "poynton": "Poynton, C. (2012). *Digital Video and HD: Algorithms and Interfaces*, 2nd "
    "edition. Morgan Kaufmann.",
    "bt601": "ITU-R Recommendation BT.601. *Studio encoding parameters of digital television for "
    "standard 4:3 and wide-screen 16:9 aspect ratios*.",
    "bt1700": "ITU-R Recommendation BT.1700. *Characteristics of composite video signals for "
    "conventional analogue television systems*.",
    "depth_anything": "Yang, L., Kang, B., Huang, Z., Zhao, Z., Xu, X., Feng, J. and Zhao, H. "
    "(2024). Depth Anything V2. *Advances in Neural Information Processing Systems (NeurIPS)*.",
}

# A diagram read from docs/how_it_works.md, by the heading it stands under.
_FROM_GUIDE = "guide:"


def _flow(*lines: str) -> str:
    """A top-down mermaid flowchart from its lines."""
    return "flowchart TD\n" + "\n".join(f"    {line}" for line in lines)


# For each transform: the diagram of its algorithm, and its references.
PAGES: dict[str, tuple[str, list[str]]] = {
    "RandomFog": (
        _FROM_GUIDE + "Fog — Koschmieder",
        ["koschmieder", "middleton", "narasimhan2002", "depth_anything"],
    ),
    "RandomRain": (
        _FROM_GUIDE + "Rain — a volume of drops",
        ["marshall", "ulbrich", "gunn_kinzer", "garg", "atlas"],
    ),
    "RandomSnow": (
        _FROM_GUIDE + "Snow — the same skeleton",
        ["gunn_marshall", "locatelli", "warren"],
    ),
    "NightRain": (_FROM_GUIDE + "Night rain", ["garg", "marshall", "gunn_kinzer"]),
    "NightSnow": (
        _flow(
            'RATE["snowfall rate, snow density"] --> SIZE["Gunn-Marshall sizes, '
            'corrected for density"]',
            'SIZE --> POP["how many flakes are in the beam and resolve"]',
            'POP --> FLAKE["each flake: distance z, size, fall speed"]',
            'FLAKE --> BLUR["bokeh: aperture disc by distance, swept along the fall"]',
            'FLAKE --> LIT["brightness by its own distance from the lamp: 1/z^2"]',
            'BLUR --> SPRITE["flake sprites, flux conserved"]',
            "LIT --> SPRITE",
            'SPRITE --> GLARE["veiling glare, before clipping"]',
            'GLARE --> OUT(["added to the frame, times the beam, plus backscatter"])',
        ),
        ["gunn_marshall", "locatelli", "warren", "potmesil"],
    ),
    "NightFog": (
        _flow(
            'IMG["night frame"] --> LAMPS["find the lamps: compact clipped regions"]',
            'IMG --> FIELD["light reaching the fog: what surfaces return, over a '
            'typical albedo, blurred wide"]',
            'DEPTH["depth z"] --> T["transmission exp(-beta z)"]',
            'FIELD --> VEIL["the fog\'s own brightness: high near lamps, their colour"]',
            'T --> MIX["frame * t + veil * (1 - t)"]',
            "VEIL --> MIX",
            'LAMPS --> HALO["a halo round each lamp, wider with more fog"]',
            "HALO --> MIX",
            'RING["infrared ring, if on"] --> BACK["backscatter: an even wall of light"]',
            "BACK --> MIX",
            "MIX --> OUT([frame])",
        ),
        ["koschmieder", "narasimhan2003", "narasimhan2002"],
    ),
    "GroundFog": (
        _flow(
            'DEPTH["depth z"] --> POINT["each pixel\'s point: range and height, from '
            'pitch and field of view"]',
            'POINT --> TAU["optical depth along the ray: closed-form integral of '
            'beta0 exp(-h / H)"]',
            'VIS["visibility at the ground"] --> BETA["beta0 = 3.912 / V"]',
            "BETA --> TAU",
            'BANKS["banks and wisps on the ground plane"] --> TAU',
            'SKY["sky pixels"] --> OPEN["beta0 exp(-hc / H) H / sin(elevation)"]',
            'TAU --> T["transmission exp(-tau)"]',
            "OPEN --> T",
            'T --> SOFT["forward scatter: blur by 1 - t"]',
            'SOFT --> OUT(["frame * t + airlight * (1 - t)"])',
        ),
        ["koschmieder", "gultepe", "narasimhan2002"],
    ),
    "LocalPlumes": (
        _flow(
            'DEPTH["depth z"] --> SRC["a source on a point of the scene: its distance"]',
            'SRC --> SIZE["width and rise in pixels: metres * f / distance"]',
            'SIZE --> COL["column: widens with height, leans with the wind"]',
            'COL --> TAU["optical depth: thins as the section widens, broken into billows"]',
            'TAU --> SHADE["self-shading: optical depth above each point"]',
            'DEPTH --> BEHIND["only where the scene is further than the plume"]',
            'TAU --> VEIL["share veiled: 1 - exp(-tau)"]',
            "BEHIND --> VEIL",
            'VEIL --> OUT(["frame * (1 - v) + v * shade * illuminant"])',
            "SHADE --> OUT",
        ),
        ["morton", "narasimhan2002"],
    ),
    "HeatShimmer": (
        _flow(
            'DEPTH["depth z = path length L"] --> TILT["tilt: sigma^2 = 2.91 D^(-1/3) Cn2 L"]',
            'DEPTH --> R0["coherence length r0 = (0.423 k^2 Cn2 L)^(-3/5)"]',
            'R0 --> BLUR["blur lambda / r0, less the lens\'s own lambda / D"]',
            'FIELD["two smooth random fields, zero mean"] --> WARP["displacement = field '
            '* sigma * f"]',
            "TILT --> WARP",
            'BLUR --> B["blur each pixel by its own sigma"]',
            'B --> W["warp the frame"]',
            "WARP --> W",
            "W --> OUT([frame])",
        ),
        ["kolmogorov", "tatarskii", "fried", "roggemann"],
    ),
    "NightInsects": (
        _flow(
            'INSECT["each insect: distance z, size, speed, wing beat"] --> PATH["path: '
            'f * v * t / z pixels, gently curved"]',
            'INSECT --> DISC["image: its size and the aperture\'s disc, in quadrature"]',
            'INSECT --> FLUX["light in one exposure: radiance by its own distance '
            '* pixels covered"]',
            'PATH --> SHARE["flux shared along the path, more where the wings are broadside"]',
            "FLUX --> SHARE",
            'DUST["dust near the lens"] --> ORB["one disc each: the aperture\'s image"]',
            'SHARE --> SUM["discs added, as light adds"]',
            "DISC --> SUM",
            "ORB --> SUM",
            'SUM --> OUT(["glare, beam, added to the frame"])',
        ),
        ["greenewalt", "fabian", "potmesil"],
    ),
    "CastShadows": (
        _flow(
            'DEPTH["depth"] --> UP["upward-facing surfaces, from normals"]',
            'SHAPES["casters: person, vehicle, foliage"] --> PROJ["projected along one '
            'sun direction"]',
            'PROJ --> PEN["penumbra: blur growing with distance from the caster"]',
            'UP --> MASK["blocked share of the direct beam, on surfaces only"]',
            "PEN --> MASK",
            'MASK --> LIGHT["surviving light per channel: direct removed, blue sky light kept"]',
            'LIGHT --> OUT(["frame * surviving light"])',
        ),
        ["prati", "depth_anything"],
    ),
    "WaterDroplets": (
        _FROM_GUIDE + "Water droplets",
        ["roser", "halimeh", "you", "born"],
    ),
    "DirtSoil": (_FROM_GUIDE + "Dirt: scattering", ["gu"]),
    "SunFlare": (_FROM_GUIDE + "Sun flare", ["hullin", "talvala"]),
    "SourceGlare": (
        _flow(
            'IMG["frame"] --> FIND["compact clipped regions: the lamps"]',
            'FIND --> STR["strength from the clipped area, times the over-range"]',
            'STR --> FLARE["a SunFlare at each: veil, starburst, ghosts"]',
            'FLARE --> OUT(["added to the frame"])',
            'FIND -.->|"no lamps"| SAME(["frame unchanged"])',
        ),
        ["hullin", "talvala", "mccann"],
    ),
    "HousingOccluders": (
        _flow(
            'DRAW["insects and webs drawn on a mask, much magnified"] --> BLUR["defocus: '
            'a blur of several percent of the frame"]',
            'BLUR --> ALPHA["how much each pixel is blocked"]',
            'ALPHA --> DARK["by day: frame * (1 - alpha), a silhouette"]',
            'ALPHA --> GLOW["under the ring: plus alpha * glow"]',
            "DARK --> OUT([frame])",
            "GLOW --> OUT",
        ),
        ["potmesil", "gu"],
    ),
    "SpiderWeb": (
        _FROM_GUIDE + "A spider's web",
        ["zschokke", "vollrath", "foelix", "potmesil"],
    ),
    "IrDomeGlare": (
        _flow(
            'RING["ring of LEDs round the lens"] --> MIRROR["mirrored by the dome\'s '
            'inner surface"]',
            'MIRROR --> BAND["defocused to a band at a radius"]',
            'TILT["lens tilted off the dome\'s axis"] --> ARC["centre displaced, only '
            'an arc in frame"]',
            'BAND --> LOBES["lobes where the LEDs are few"]',
            "ARC --> LOBES",
            'DIRT["deposits on the dome"] --> MOTTLE["patchy, about a mean of one"]',
            "LOBES --> GLOW",
            "MOTTLE --> GLOW",
            'VEIL["film on the dome"] --> GLOW["glow = ring * intensity + veil"]',
            'GLOW --> OUT(["added to the frame"])',
        ),
        ["talvala", "mccann"],
    ),
    "FrostOnGlass": (
        _flow(
            'RIM["nucleation at the cold rim"] --> FERN["ferns: stems inwards, branches '
            'at 60 degrees, and theirs"]',
            'RIM --> LAYER["an even layer behind a ragged front"]',
            'FERN --> DEPTH["optical depth of ice"]',
            "LAYER --> DEPTH",
            'DEPTH --> S["scattered share s = 1 - exp(-depth), defocused"]',
            'IMG["frame"] --> DIFF["wide blur of the scene: light with its direction lost"]',
            'S --> OUT(["frame * (1 - s) + s * (0.7 diffuse + ambient + glow)"])',
            "DIFF --> OUT",
        ),
        ["libbrecht", "gu"],
    ),
    "SnowOnHousing": (
        _flow(
            'CAP["cap from the top edge: uneven, lobed"] --> SOLID["what the snow covers"]',
            'PATCH["lumps stuck to the glass"] --> SOLID',
            'SOLID --> COVER["defocus: cover, with a soft edge"]',
            'SOLID --> THICK["a wider blur: how deep into the snow"]',
            'THICK --> DAY["daylight through it: bluer, less where thick, uneven"]',
            'COVER --> OUT(["frame * (1 - cover) + cover * snow"])',
            "DAY --> OUT",
            'LAMP["infrared ring"] --> OUT',
        ),
        ["warren", "bohren"],
    ),
    "LensTamper": (
        _flow(
            'KIND{"which kind"} --> SPRAY["spray: opaque core, speckle, runs"]',
            'KIND --> TAPE["tape: straight strips across the frame"]',
            'KIND --> COVERK["cover: the whole frame, with folds"]',
            'SPRAY --> BLUR["defocus"]',
            "TAPE --> BLUR",
            'BLUR --> EXT["extent: what is behind it; cover: what it stops"]',
            "COVERK --> EXT",
            'EXT --> OUT(["scene beside it, diffuse light through it, its own colour on it"])',
        ),
        ["ribnick", "saglam"],
    ),
    "FoliageIntrusion": (
        _flow(
            'BRANCH["each branch: a distance z"] --> SIZE["leaf length in pixels: mm * f / z"]',
            'SIZE --> TWIGS["drooping twigs, leaves on stalks"]',
            'TWIGS --> LEAF["each leaf: its own turn, lit on its face or lit through, '
            'shading those behind"]',
            'LEAF --> BLUR["aperture disc: A f (1/z - 1/s)"]',
            'BLUR --> SMEAR["wind: sway speed * shutter time, as seen from z"]',
            'SMEAR --> OUT(["frame * (1 - cover) + leaves * illuminant + lamp"])',
            'LAMP["infrared ring: by z, by how squarely a leaf faces it"] --> OUT',
        ),
        ["gates", "potmesil"],
    ),
    "FocusShift": (
        _flow(
            'ERR["focus error: the lens is focused at s"] --> DIAM["blur disc: '
            'A f abs(1/s - 1/z)"]',
            'DEPTH["depth z, if given"] --> DIAM',
            'DIAM --> LAYERS["blur the frame at a few diameters"]',
            'LAYERS --> BLEND["each pixel from the two nearest, in proportion"]',
            "BLEND --> OUT([frame])",
            'DIAM -.->|"no depth"| EVEN(["one blur: A f / s"])',
        ),
        ["potmesil", "hecht"],
    ),
    "NightIrMode": (
        _FROM_GUIDE + "Night IR",
        ["gates", "brown", "emva", "depth_anything"],
    ),
    "IrCutFilterFault": (
        _flow(
            'RGB["frame, linear"] --> NIR["infrared reflectance: red-weighted, plus '
            'the chlorophyll term"]',
            'EDGE["how far open the filter is"] --> SIDE["which part of the sensor is uncovered"]',
            'NIR --> ADD["raw = rgb + leak * nir * dyes, where uncovered"]',
            "SIDE --> ADD",
            'ADD --> WB["white balance: towards the old channel means, part-way"]',
            'WB --> AE["exposure: back to the old level"]',
            "AE --> OUT([frame])",
        ),
        ["brown", "gates"],
    ),
    "AcFlicker": (_FROM_GUIDE + "AC flicker", ["poplin"]),
    "RollingShutter": (
        _flow(
            'MOTION["camera motion during readout: pan, wobble"] --> ROW["each row read '
            'at its own time"]',
            'ROW --> DISP["displacement of each row"]',
            'DISP --> IMG["image rows resampled"]',
            'DISP --> ANN["masks resampled, boxes and keypoints moved"]',
            "IMG --> OUT([frame and annotations])",
            "ANN --> OUT",
        ),
        ["liang"],
    ),
    "CameraVibration": (
        _flow(
            'SWING["swing: amplitude, frequency"] --> PATH["pointing over the exposure: '
            'a stretch of a sinusoid"]',
            'EXP["shutter time"] --> PATH',
            'PATH --> KERNEL["blur kernel: time spent at each offset from the mean"]',
            'PATH --> SHIFT["mean position: the frame\'s displacement"]',
            'KERNEL --> IMG["frame convolved, in linear light"]',
            'SHIFT --> MOVE["frame, masks, boxes and keypoints displaced"]',
            "IMG --> MOVE",
            "MOVE --> OUT([frame and annotations])",
        ),
        ["fergus", "liang"],
    ),
    "SensorNoise": (
        _flow(
            'IMG["frame, linear"] --> E["electrons = image * full well / gain"]',
            'E --> MOSAIC["one colour a pixel: the Bayer mosaic"]',
            'MOSAIC --> NOISE["plus Normal(0, sqrt(electrons + read noise^2))"]',
            'NOISE --> DEMOSAIC["interpolated: grain coarser in red and blue"]',
            'DEMOSAIC --> OUT(["image = electrons * gain / full well"])',
        ),
        ["healey", "emva", "janesick"],
    ),
    "DemosaicArtifacts": (
        _flow(
            'IMG["frame, linear"] --> SENSOR["resampled to the sensor\'s resolution"]',
            'SENSOR --> LENS["the lens\'s blur"]',
            'LENS --> MOSAIC["one colour kept a pixel: Bayer pattern, at some phase"]',
            'MOSAIC --> INTERP["bilinear interpolation of the other two"]',
            'INTERP --> CHROMA["median on the colour differences"]',
            'CHROMA --> OUT(["enlarged to the frame"])',
        ),
        ["bayer", "gunturk"],
    ),
    "ExposureTransient": (
        _flow(
            'IMG["frame, linear"] --> GAIN["exposure error in stops: times 2^error"]',
            'T1["temperature the camera assumed"] --> RATIO["ratio of two blackbody colours"]',
            'T2["temperature of the light"] --> RATIO',
            'RATIO --> WB["white-balance error, on the blue-orange axis"]',
            'TINT["green-magenta residual"] --> WB',
            'GAIN --> OUT(["frame * gain * white-balance error"])',
            "WB --> OUT",
        ),
        ["wyszecki"],
    ),
    "WdrToneMapping": (
        _flow(
            'IMG["frame"] --> LOG["log luminance"]',
            'LOG --> BASE["base layer: edge-preserving filter, towards a Gaussian as '
            'the processor gets cheaper"]',
            'LOG --> DETAIL["detail = log - base"]',
            'BASE --> COMP["base compressed"]',
            'COMP --> SUM["compressed base + detail"]',
            "DETAIL --> SUM",
            'SUM --> OUT(["back to light: halos where the base leaked across an edge"])',
        ),
        ["durand", "tomasi"],
    ),
    "TemporalDenoiseArtifacts": (
        _flow(
            'IMG["frame"] --> SPLIT["luma and chroma"]',
            'SPLIT --> LUMA["luma: edge-preserving filter, hard: texture flattened, edges kept"]',
            'SPLIT --> CHROMA["chroma: filtered much harder: bleeding and blotches"]',
            'LUMA --> OUT(["recombined: the watercolour look"])',
            "CHROMA --> OUT",
        ),
        ["gastal", "tomasi"],
    ),
    "VideoCodecArtifacts": (
        _flow(
            'IMG["frame"] --> ENC["encoded for real: H.264 or H.265 at a CRF"]',
            'ENC --> DEC["decoded: deblocking, 4:2:0 chroma, rate control"]',
            'DEC --> STALE["optionally: stale macroblocks from an older reference"]',
            "STALE --> OUT([frame])",
        ),
        ["wiegand", "sullivan", "list"],
    ),
    "TransmissionFaults": (
        _flow(
            'IMG["frame"] --> CLIP["a short clip: the picture sliding, as a scene moves"]',
            'CLIP --> ENC["encoded as H.264"]',
            'ENC --> LOSS["packets removed from the bitstream, in bursts"]',
            'LOSS --> DEC["decoded with error concealment: holes filled, garbage, smears"]',
            'DEC --> OUT(["the last frame of the group, damage and all"])',
        ),
        ["wang", "wiegand"],
    ),
    "AnalogVideoArtifacts": (
        _flow(
            'IMG["frame, code values"] --> FIELDS["fields offset; lines jittered"]',
            'FIELDS --> YUV["brightness and two colour differences"]',
            'YUV --> ENC["colour band-limited, modulated on a subcarrier, added to brightness"]',
            'ENC --> DEC["decoded: times the carrier, low-passed: cross-colour and '
            'dot crawl remain"]',
            'DEC --> CABLE["cable: low-pass along the line, reflection, hum, noise"]',
            'CABLE --> RX["receiver: peaking; colour delayed"]',
            "RX --> OUT([frame])",
        ),
        ["poynton", "bt601", "bt1700"],
    ),
    "OnScreenOverlay": (
        _flow(
            'WHAT["what is written: date and time, a name"] --> TEXT["5 by 7 face, '
            'whole pixels a dot, fitted to the frame"]',
            'MASKS["privacy masks: filled or reduced to blocks"] --> DRAW["drawn on code values"]',
            "TEXT --> DRAW",
            'MASKS --> BOX["a box covered by 0.9 of its area is dropped"]',
            "DRAW --> OUT([frame])",
            "BOX --> OUT",
        ),
        [],
    ),
}


def _class(name: str) -> type:
    """The transform class of that name, from whichever package exports it."""
    for package in PACKAGES:
        module = importlib.import_module(f"surmentations.{package}")
        if name in module.__all__:
            found: type = getattr(module, name)
            return found
    raise KeyError(name)


def transforms() -> dict[str, list[str]]:
    """Every exported transform's name, by package, in the packages' own order."""
    return {
        package: sorted(importlib.import_module(f"surmentations.{package}").__all__)
        for package in PACKAGES
    }


def _guide_diagram(heading: str) -> str:
    """The first diagram under the heading of ``how_it_works.md`` that starts so."""
    text = (DOCS / "how_it_works.md").read_text()
    start = re.search(rf"^#+ {re.escape(heading)}.*$", text, re.M)
    if start is None:
        raise KeyError(f"how_it_works.md has no heading starting {heading!r}")
    block = re.search(r"```mermaid\n(.*?)```", text[start.end() :], re.S)
    if block is None:
        raise KeyError(f"no diagram under {heading!r}")
    return block.group(1).rstrip()


def to_markdown(docstring: str) -> str:
    """A reStructuredText docstring as Markdown: the few constructs the source uses.

    Underlined titles become headings, ``::`` blocks become fenced code,
    double backticks become single, and roles such as ``:class:`` keep
    their target's last name.
    """
    lines = inspect.cleandoc(docstring).split("\n")
    out: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        following = lines[index + 1] if index + 1 < len(lines) else ""
        if line.strip() and re.fullmatch(r"[-=~^]{3,}", following.strip()):
            out += [f"### {line.strip()}", ""]
            index += 2
            continue
        if line.rstrip().endswith("::"):
            out.append(line.rstrip()[:-2].rstrip() + ":" if line.strip() != "::" else "")
            index += 1
            block: list[str] = []
            while index < len(lines) and (not lines[index].strip() or lines[index][:1] == " "):
                block.append(lines[index])
                index += 1
            while block and not block[-1].strip():
                block.pop()
            while block and not block[0].strip():
                block.pop(0)
            indent = min(len(b) - len(b.lstrip()) for b in block if b.strip())
            out += ["", "```", *[b[indent:] for b in block], "```", ""]
            continue
        out.append(line)
        index += 1
    text = "\n".join(out)
    text = re.sub(
        r":[a-z]+:`~?([^`<]*?)(?: <[^`]*>)?`", lambda m: f"`{m.group(1).split('.')[-1]}`", text
    )
    text = re.sub(r"``([^`]+)``", r"`\1`", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _arguments(cls: type) -> list[tuple[str, str]]:
    """Each argument of a class with what its docstring says of it, in order."""
    doc = inspect.cleandoc(cls.__doc__ or "")
    section = re.search(r"^Args:\n(.*?)(?=^\S|\Z)", doc, re.M | re.S)
    found: list[tuple[str, str]] = []
    if section is None:
        return found
    for line in section.group(1).split("\n"):
        entry = re.match(r"^ {4}(\w+): (.*)$", line)
        if entry:
            found.append((entry.group(1), entry.group(2).strip()))
        elif line.strip() and found:
            name, text = found[-1]
            found[-1] = (name, f"{text} {line.strip()}")
    return found


def _summary(cls: type) -> str:
    """The first sentence of a class's docstring."""
    return inspect.cleandoc(cls.__doc__ or "").split("\n\n")[0].replace("\n", " ")


def _depth(cls: type) -> str:
    """Whether the transform needs a depth provider: yes, no or optional."""
    provider = inspect.signature(cls.__init__).parameters.get("depth_provider")
    if provider is None:
        return "no"
    return "optional" if provider.default is None else "yes"


def _annotations(cls: type) -> str:
    """What the transform does to masks, boxes and keypoints."""
    targets = {str(getattr(t, "value", t)).lower() for t in getattr(cls, "_targets", ())}
    if "keypoints" in targets:
        return "moves masks, boxes and keypoints with the pixels"
    if "bboxes" in targets:
        return "drops boxes it hides; masks and keypoints untouched"
    return "untouched: the image only"


def page(name: str) -> str:
    """The whole page for one transform, as Markdown."""
    cls = _class(name)
    module = inspect.getmodule(cls)
    assert module is not None and module.__file__ is not None
    source = Path(module.__file__).resolve().relative_to(ROOT)
    diagram, references = PAGES[name]
    if diagram.startswith(_FROM_GUIDE):
        diagram = _guide_diagram(diagram[len(_FROM_GUIDE) :])
    test = ROOT / "tests" / "unit" / f"test_{source.stem}.py"

    parts = [
        f"# `{name}`",
        "",
        f"> {_summary(cls)}",
        "",
        "| | |",
        "| --- | --- |",
        f"| Module | [`{module.__name__}`](../../{source}) |",
        f"| Stage | {STAGES[stage_of(cls)]} |",
        f"| Needs depth | {_depth(cls)} |",
        f"| Annotations | {_annotations(cls)} |",
    ]
    if test.exists():
        parts.append(f"| Tests | [`{test.relative_to(ROOT)}`](../../{test.relative_to(ROOT)}) |")
    parts += ["", "## Algorithm", "", "```mermaid", diagram, "```", ""]
    parts += ["## Model", "", to_markdown(module.__doc__ or ""), ""]

    parts += ["## Parameters", "", "| Parameter | Default | Meaning |", "| --- | --- | --- |"]
    signature = inspect.signature(cls.__init__).parameters
    described = dict(_arguments(cls))
    for parameter, spec in signature.items():
        if parameter == "self":
            continue
        default = "required" if spec.default is inspect.Parameter.empty else f"`{spec.default!r}`"
        meaning = "probability that the transform is applied" if parameter == "p" else ""
        meaning = to_markdown(described.get(parameter, meaning)).replace("|", "\\|")
        parts.append(f"| `{parameter}` | {default} | {meaning} |")
    parts.append("")

    rows = [key for key, owner in EFFECTS.items() if owner == name]
    rows = [key for key in rows if (DOCS / "examples" / f"{key}_0.png").exists()]
    if rows:
        parts += [
            "## Samples",
            "",
            "Severity 1 to 5, left to right, from the severity gallery.",
            "",
        ]
        for key in rows:
            cells = " | ".join(f"![{key} {i + 1}](../examples/{key}_{i}.png)" for i in range(5))
            parts += [f"`{key}`", "", "| 1 | 2 | 3 | 4 | 5 |", "| --- | --- | --- | --- | --- |"]
            parts += [f"| {cells} |", ""]

    parts += ["## References", ""]
    if references:
        parts += [f"- {SOURCES[key]}" for key in references]
    else:
        parts.append("None: this transform models a convention, not a physical process.")
    parts += [
        "",
        "---",
        "",
        "Generated by [`docs/make_transform_pages.py`](../make_transform_pages.py) from the "
        "source; edit the docstring or that script, not this page. "
        "[All transforms](README.md).",
        "",
    ]
    return "\n".join(parts)


def index() -> str:
    """The page that lists every transform's page, by package."""
    parts = [
        "# The transforms, one page each",
        "",
        "What each transform models, the algorithm as a diagram, every parameter, sample",
        "output and the papers behind it. For how they fit together see",
        "[`how_it_works.md`](../how_it_works.md); for using them, the",
        "[README](../../README.md).",
        "",
    ]
    for package, title in PACKAGES.items():
        names = transforms()[package]
        parts += [f"## {title}", "", "| Transform | Models | Depth |", "| --- | --- | --- |"]
        for name in names:
            cls = _class(name)
            parts.append(
                f"| [`{name}`]({name}.md) | {to_markdown(_summary(cls))} | {_depth(cls)} |"
            )
        parts.append("")
    parts += [
        "---",
        "",
        "Generated by [`docs/make_transform_pages.py`](../make_transform_pages.py).",
        "",
    ]
    return "\n".join(parts)


def pages() -> dict[str, str]:
    """Every file of ``docs/transforms``, by name, as it should be."""
    files = {"README.md": index()}
    for names in transforms().values():
        for name in names:
            files[f"{name}.md"] = page(name)
    return files


def main() -> None:
    """Write every page, and remove any that no longer has a transform."""
    OUT.mkdir(parents=True, exist_ok=True)
    files = pages()
    for stale in OUT.glob("*.md"):
        if stale.name not in files:
            stale.unlink()
    for name, text in files.items():
        (OUT / name).write_text(text)
    print(f"wrote {len(files)} pages to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
