"""Render every effect, and the shipped pipelines, over each sample scene.

``generate_examples.py`` sweeps the severity ladder over one reference
photograph. This is the other axis: one severity, many scenes. The scenes in
``sample_scenes/`` were chosen to be unlike the reference and unlike each
other — a crossing seen from straight above, a covered car park, a real
night street, mountains seen across a valley — because that is where a
weather model stops looking right, and looking is how most of the defects in
this library were found.

Output goes to ``docs/examples/scenes/<scene>/``::

    original.jpg             the scene as rendered from, 800 px wide
    reading.jpg              its labels and horizon, as a provider reads them
    effects/<effect>.jpg     one representative draw per effect
    aware/<effect>.jpg       the same draw with scene labels and geometry
    pipelines/<name>_<n>.jpg draws of the pipeline that fits the scene
    pipelines_aware/...      the same draws with scene labels and geometry
    README.md                the same images as a browsable page

Run from the repo root with ``python docs/generate_scene_examples.py``.
Requires a CUDA GPU.

Three things here are not in the picture and have to be told to the
transforms, which is the point of showing it done:

* **Scale.** Monocular depth is relative, so each scene gets its own
  ``depth_range_m`` on the provider. The values below are estimated by eye
  from the photographs, not surveyed.
* **How thick the air is for that scale.** A severity level fixes extinction
  per metre, which suits a street and blanks a landscape: fog that leaves a
  car park's far wall visible at 40 m hides a mountain at 3 km completely.
  So fog is set by visibility, as a fraction of the scene's depth, and the
  distance veils of rain and snow are capped the same way.
* **What applies.** Sunlight effects and daylight fog are skipped on the
  night scene, the two effects that act on lamps on the daylit ones, and
  rain and snow under a roof. Nothing in the library works
  that out for itself.

Some of that a provider can now read off the picture for itself
(:mod:`surmentations.perception`): what each pixel is, where the horizon
is, and where there are vehicles to judge it by, how high the camera hangs
and so how many metres the scene is deep. So every effect that depends on
what the provider knows is rendered twice, from the same parameters and the
same seed: once with the plain provider and the by-eye values above, and
once with a provider that also carries scene labels and an estimated
geometry. Each scene's page shows the pairs side by side, under what the
second provider read.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import albumentations as A
import cv2
import numpy as np
from tqdm import tqdm

# Run as a script, this file's own directory is on the path and the repo root
# is not, so `docs.generate_examples` would not resolve.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docs.generate_examples import (  # noqa: E402
    _GALLERY_DRAWS,
    _NEEDS_FFMPEG,
    EFFECTS,
    GROUND_ROW_WEATHER_LEVEL,
    GROUND_ROWS,
    LAMP_LIT,
    _build,
    _night_base,
    _snowy_night_base,
)
from surmentations.camera.codec import ffmpeg_path  # noqa: E402
from surmentations.depth import DepthAnythingV2Estimator, SharedDepthProvider  # noqa: E402
from surmentations.perception import (  # noqa: E402
    GroundPlaneGeometryEstimator,
    SceneClass,
    SegmentationLabelEstimator,
)
from surmentations.presets import surveillance, surveillance_lite  # noqa: E402
from surmentations.presets.severity import SEVERITY  # noqa: E402
from surmentations.utils.precipitation import (  # noqa: E402
    rain_extinction_coefficient,
    snow_extinction_coefficient,
)

REPO = Path(__file__).resolve().parents[1]
SCENES_DIR = REPO / "sample_scenes"
OUT_DIR = Path(__file__).parent / "examples" / "scenes"

WIDTH_PX = 800
JPEG_QUALITY = 86
# The middle of the ladder: plainly visible, not a fault report.
SEVERITY_LEVEL = 3
# Which of generate_examples' per-variant overrides to use; 2 is the middle one.
VARIANT = 2
PIPELINE_DRAWS = 4

# Lit by the camera's own illuminator, so rendered over an infrared night
# version of the scene, as in the severity gallery.
NEEDS_SUN = ("sun_flare", "cast_shadows")
NEEDS_OPEN_SKY = (
    "rain", "snow", "wet_ground", "snow_cover", "night_rain", "night_snow", *NEEDS_SUN,
    "frost_on_glass", "snow_on_housing", "foliage_intrusion", "night_foliage", "ground_fog",
)  # fmt: skip
# Faults and effects of the daytime picture.
DAYTIME_ONLY = ("ir_cut_filter_fault",)
# Turbulence over ground the sun has heated, which shows over a long path.
NEEDS_A_LONG_VIEW = ("heat_shimmer",)
LONG_VIEW_M = 500.0
DAYLIGHT_WEATHER = ("fog",)
NEEDS_LAMPS = ("night_fog", "source_glare")

# Meteorological visibility, as a multiple of the scene's farthest distance.
# Fog is set to it outright. For rain and snow it is a floor: their veils
# follow from the precipitation rate and are left alone unless that would
# hide the far end of the scene altogether. Severity-3 rain is visible to
# about 1.5 km and snow to about 110 m, so the floor binds on rain only in
# the two landscapes and on snow nearly everywhere.
FOG_VISIBILITY = 0.6
RAIN_VISIBILITY_FLOOR = 1.5
SNOW_VISIBILITY_FLOOR = 1.0
# Koschmieder: visibility is where contrast falls to 2%, exp(-3.912).
KOSCHMIEDER = 3.912

WHY_NOT = {
    "sun": "it needs the sun, and the scene is at night",
    "roof": "the scene is under a roof",
    "daylight": (
        "it is the daylight model, an even veil lit by the sky. `night_fog` is fog lit by "
        "the lamps"
    ),
    "lamps": "it acts on artificial lights at night, and this is a daylit scene",
    "daytime": "it is a fault of the colour picture by day, and the scene is at night",
    "long view": (
        "it is turbulence over sunlit ground, which shows over a kilometre or so by day: "
        "this scene is too short, under a roof or at night"
    ),
}

# The effects whose picture depends on what the provider knows of the scene,
# and so are rendered a second time with scene labels and geometry: the
# ones that read the ground, the sky, the camera or the distances. The rest
# of the lamp-lit rows differ only in the infrared frame under them, which
# `night_ir` shows.
COMPARED = (
    "fog", "rain", "snow", "wet_ground", "snow_cover", "ground_fog", "local_plumes",
    "heat_shimmer", "cast_shadows", "night_ir", "night_fog", "night_snow",
)  # fmt: skip

# One colour per scene class, for the label maps.
LABEL_COLOURS: dict[SceneClass, tuple[int, int, int]] = {
    SceneClass.OTHER: (40, 40, 40),
    SceneClass.SKY: (135, 190, 235),
    SceneClass.ROAD: (105, 105, 110),
    SceneClass.PAVEMENT: (175, 145, 175),
    SceneClass.TERRAIN: (150, 120, 70),
    SceneClass.GRASS: (125, 215, 85),
    SceneClass.FOLIAGE: (35, 125, 40),
    SceneClass.STRUCTURE: (150, 85, 65),
    SceneClass.VEHICLE: (225, 40, 40),
    SceneClass.PERSON: (250, 215, 40),
    SceneClass.WATER: (40, 70, 215),
    SceneClass.POLE: (235, 235, 235),
    SceneClass.LAMP: (255, 150, 30),
}

# An open iris, and rain metered for street lighting: see presets/pipelines.py.
NIGHT_IRIS_MM = (1.4, 3.0)
NIGHT_STREET_LUX = (5.0, 150.0)
LAMP_OVER_RANGE = (4.0, 12.0)


@dataclass(frozen=True)
class Scene:
    """What a photograph cannot tell the transforms about itself.

    Attributes:
        depth_range_m: distance of the nearest and farthest thing in frame.
        camera_pitch_deg: how far the camera looks down from level.
        night: taken at night under artificial light.
        sheltered: under a roof, so nothing falls on it and the sun is not in it.
        sky_in_frame: whether the frame shows any sky. Where it does not, the
            dirt on the glass is still lit by it.
    """

    depth_range_m: tuple[float, float]
    camera_pitch_deg: float
    night: bool = False
    sheltered: bool = False
    sky_in_frame: bool = True


SCENES: dict[str, Scene] = {
    "crosswalk_day": Scene((12.0, 30.0), 70.0, sky_in_frame=False),
    "parking_garage": Scene((2.0, 45.0), 4.0, sheltered=True, sky_in_frame=False),
    "street_night": Scene((3.0, 150.0), 5.0, night=True),
    "underpass_from_above": Scene((8.0, 160.0), 25.0, sky_in_frame=False),
    "wooded_hills": Scene((40.0, 6000.0), 6.0),
    "mountain": Scene((25.0, 3000.0), 0.0),
    "urban_road_day": Scene((8.0, 900.0), 10.0),
    "urban_street_wet": Scene((6.0, 250.0), 30.0),
}


def _veil_cap(scene: Scene, visibility_floor: float, derived_beta: float) -> dict[str, Any]:
    """Cap a precipitation veil so the scene's far end is not hidden outright."""
    cap = KOSCHMIEDER / (visibility_floor * scene.depth_range_m[1])
    if derived_beta <= cap:
        return {}
    return {"veil_beta_range": (0.8 * cap, cap)}


def _overrides(effect: str, scene: Scene) -> dict[str, Any]:
    """Tell one transform about the scene it is being applied to."""
    pitch = scene.camera_pitch_deg
    preset = SEVERITY[SEVERITY_LEVEL]
    extra: dict[str, Any] = {}
    if effect in GROUND_ROWS:
        # The ground rows are rain and snow at a light rate.
        preset = SEVERITY[GROUND_ROW_WEATHER_LEVEL]
        effect = GROUND_ROWS[effect]
    if effect in ("rain", "snow"):
        extra["camera_pitch_range_deg"] = (pitch, pitch)
    if effect == "rain":
        rate = float(np.mean(preset["RandomRain"]["rain_rate_range_mm_h"]))
        extra |= _veil_cap(scene, RAIN_VISIBILITY_FLOOR, rain_extinction_coefficient(rate))
    if effect == "snow":
        rate = float(np.mean(preset["RandomSnow"]["snowfall_rate_range_mm_h"]))
        extra |= _veil_cap(scene, SNOW_VISIBILITY_FLOOR, snow_extinction_coefficient(rate))
    if effect == "cast_shadows":
        extra["camera_pitch_deg"] = pitch
    if effect == "fog":
        beta = KOSCHMIEDER / (FOG_VISIBILITY * scene.depth_range_m[1])
        extra["beta_range"] = (0.85 * beta, 1.15 * beta)
        # Against the scene's own distances, from the provider, in place of
        # the severity gallery's 40-90 m far plane.
        extra["depth_scale"] = None
    if effect == "ground_fog":
        extra["camera_pitch_range_deg"] = (pitch, pitch)
        extra["depth_scale"] = None
    if effect in ("heat_shimmer", "local_plumes"):
        # Over the scene's own distances, in place of the severity
        # gallery's stand-ins.
        extra["depth_scale"] = None
    if effect == "night_fog":
        visibility = FOG_VISIBILITY * scene.depth_range_m[1]
        extra["visibility_range_m"] = (0.85 * visibility, 1.15 * visibility)
    if effect == "dirt_soil" and not scene.sky_in_frame:
        extra["ambient_light_range"] = (2.5, 4.0)
    if scene.night and effect == "water_droplets":
        extra["aperture_diameter_range_mm"] = NIGHT_IRIS_MM
    if scene.night and effect == "rain":
        extra["aperture_diameter_range_mm"] = NIGHT_IRIS_MM
        extra["scene_illuminance_range_lux"] = NIGHT_STREET_LUX
        # Lit by the lamps in the frame, and not relit for cloud: the light
        # comes from a pole.
        extra["lamp_glint_range"] = LAMP_OVER_RANGE
        extra["overcast_range"] = (0.0, 0.0)
    if scene.night and effect == "snow":
        extra["overcast_range"] = (0.0, 0.0)
    return extra


def _why_not(effect: str, scene: Scene) -> str | None:
    """Why ``effect`` does not belong on ``scene``, or ``None`` if it does."""
    if scene.night and effect in NEEDS_SUN:
        return WHY_NOT["sun"]
    if scene.sheltered and effect in NEEDS_OPEN_SKY:
        return WHY_NOT["roof"]
    if scene.night and effect in DAYLIGHT_WEATHER:
        return WHY_NOT["daylight"]
    if not scene.night and effect in NEEDS_LAMPS:
        return WHY_NOT["lamps"]
    if scene.night and effect in DAYTIME_ONLY:
        return WHY_NOT["daytime"]
    if effect in NEEDS_A_LONG_VIEW and (
        scene.night or scene.sheltered or scene.depth_range_m[1] < LONG_VIEW_M
    ):
        return WHY_NOT["long view"]
    return None


def _pipeline(scene: Scene, provider: SharedDepthProvider) -> tuple[str, Any]:
    """The shipped pipeline that fits the scene, and what it is called.

    ``surveillance`` treats its input as a daylit outdoor frame and decides
    for itself whether to turn it into night, so it is right for the day
    scenes and wrong for the other two. A frame that is already night, or
    under a roof, gets ``surveillance_lite`` with every frame treated as
    artificially lit: the lens and the camera chain, no weather, no sun.
    """
    if scene.night or scene.sheltered:
        return "surveillance_lite", surveillance_lite(night=1.0, severity=2)
    # Weather raised from its climatological 12% so that four draws show
    # some; severity 2 because level 1 is deliberately close to invisible.
    return "surveillance", surveillance(
        provider, rain=0.25, snow=0.10, fog=0.15, severity=2, weather_severity=3
    )


def label_picture(labels: np.ndarray) -> np.ndarray:
    """A label map as a picture, one colour per scene class."""
    table = np.zeros((256, 3), dtype=np.uint8)
    for scene_class, colour in LABEL_COLOURS.items():
        table[int(scene_class)] = colour
    return table[labels]


def label_legend(width: int) -> np.ndarray:
    """A strip ``width`` pixels across naming the colours of :func:`label_picture`."""
    strip = np.full((34, width, 3), 255, dtype=np.uint8)
    step = width // len(LABEL_COLOURS)
    for index, (scene_class, colour) in enumerate(LABEL_COLOURS.items()):
        left = index * step
        cv2.rectangle(strip, (left + 6, 9), (left + 22, 25), colour, -1)
        cv2.rectangle(strip, (left + 6, 9), (left + 22, 25), (0, 0, 0), 1)
        name = scene_class.name.lower()
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(strip, name, (left + 27, 23), font, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
    return strip


def draw_horizon(picture: np.ndarray, horizon: float, roll_deg: float) -> None:
    """Draw a horizon across ``picture``, in place: yellow on black.

    ``horizon`` is its row at the middle column as a fraction of the
    picture's height, and ``roll_deg`` how far it runs downhill to the right.
    """
    height, width = picture.shape[:2]
    slope = float(np.tan(np.radians(roll_deg)))
    left = (0, round(horizon * height - 0.5 * width * slope))
    right = (width - 1, round(horizon * height + 0.5 * width * slope))
    cv2.line(picture, left, right, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.line(picture, left, right, (255, 235, 0), 2, cv2.LINE_AA)


def _reading(
    image: np.ndarray, provider: SharedDepthProvider, scene: Scene
) -> tuple[np.ndarray, list[tuple[str, str]]]:
    """What a provider with labels and geometry read off a scene: a picture, and a table's rows.

    The picture is the photograph beside its label map, the horizon drawn
    across both. The rows say what the camera was taken to be, beside the
    by-eye values the plain provider is given, and what could not be told.
    """
    height, width = image.shape[:2]
    told = provider.describe(image)
    geometry, camera, metric = told["geometry"], told["camera"], told["metric"]
    labels = provider.get_labels(image)
    photograph, named = image.copy(), label_picture(labels)
    by_eye = f"{scene.camera_pitch_deg:g}°"

    horizon = "not found: no level ground was labelled"
    pitch = f"not found; the by-eye {by_eye} is used"
    if geometry.horizon is not None:
        for picture in (photograph, named):
            draw_horizon(picture, geometry.horizon, geometry.roll_deg)
        horizon = f"{geometry.horizon:.2f} of the frame's height from the top"
        if geometry.horizon_is_bound:
            horizon += ", or above: there is no sky in the frame to mark it"
            least = geometry.view(width, height, pitch_deg=-90.0).pitch_deg
            pitch = f"at least {least:+.0f}°, so the by-eye {by_eye} stands"
        else:
            pitch = f"{camera.pitch_deg:+.0f}° (by eye: {by_eye})"
    tall = "not found: fewer than three usable vehicles or people in view"
    if geometry.height_m is not None:
        tall = f"{geometry.height_m:.0f} m, from the vehicles and people in view"
    depth = "the by-eye range above: there is no height to measure from"
    if metric is not None:
        metres = provider.get_metric_depth(image)[labels != int(SceneClass.SKY)]
        near, far = np.percentile(metres, [1.0, 99.0])
        depth = f"in metres from the camera's height: {near:.0f} m to {far:.0f} m"
    elif geometry.height_m is not None:
        depth = "the by-eye range above: the height needs a pitch to measure with"
    shares = sorted(told["label_shares"].items(), key=lambda item: -item[1])
    named_shares = ", ".join(f"{name.lower()} {share:.0%}" for name, share in shares[:6])

    rows = [
        ("Field of view", f"{camera.horizontal_fov_deg:.0f}°, assumed: a picture does not tell it"),
        ("Horizon", horizon),
        ("Camera pitch below level", pitch),
        ("Roll", f"{geometry.roll_deg:+.1f}°"),
        ("Camera height", tall),
        ("Depth", depth),
        ("Labels", named_shares),
    ]
    pair = np.hstack([photograph, named])
    return np.vstack([pair, label_legend(pair.shape[1])]), rows


def _representative(
    transform: Any, base: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, int]:
    """One representative draw of a transform, and the seed it was drawn with.

    As ``generate_examples._render_representative``: seven seeds from
    ``rng``, and the draw kept is the median by how much it changed the
    frame. The seed is returned so that the same draw can be made again
    over a provider that knows more, and what differs between the two is
    then what the provider knew.
    """
    reference = base.astype(np.float32)
    candidates = []
    for _ in range(_GALLERY_DRAWS):
        seed = int(rng.integers(0, 2**31 - 1))
        transform.set_random_seed(seed)
        out = transform(image=base)["image"]
        candidates.append((float(np.abs(out.astype(np.float32) - reference).mean()), seed, out))
    candidates.sort(key=lambda entry: entry[0])
    _change, seed, out = candidates[len(candidates) // 2]
    return out, seed


def _base(effect: str, image: np.ndarray, provider: SharedDepthProvider, scene: Scene) -> Any:
    """The frame an effect is drawn over: the scene, or its infrared night for a lamp-lit one."""
    if effect == "night_snow":
        # Over the snow that has settled, as the pipelines draw it.
        pitch = scene.camera_pitch_deg
        return _snowy_night_base(image, provider, SEVERITY_LEVEL, (pitch, pitch))
    if effect in LAMP_LIT:
        return _night_base(image, provider, SEVERITY_LEVEL)
    return image


def _load(path: Path) -> np.ndarray:
    """A sample scene as RGB, :data:`WIDTH_PX` across."""
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Could not read {path}")
    height, width = image.shape[:2]
    size = (WIDTH_PX, round(height * WIDTH_PX / width))
    return cv2.cvtColor(cv2.resize(image, size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)


def _save(path: Path, image: np.ndarray) -> None:
    """Write an RGB image as a JPEG, making its directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(
        str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    )


def _sources() -> dict[str, dict[str, str]]:
    """Author, licence and link for each scene, from ``sample_scenes/SOURCES.md``."""
    rows = {}
    row = re.compile(r"\| `(\w+)\.jpg` \| (.+?) \| (.+?) \| (.+?) \| \[(.+?)\]\((.+?)\) \|")
    for line in (SCENES_DIR / "SOURCES.md").read_text().splitlines():
        match = row.match(line)
        if match:
            name, stresses, author, licence, title, url = match.groups()
            rows[name] = {
                "stresses": stresses, "author": author, "licence": licence,
                "title": title, "url": url,
            }
    return rows


def _table(cells: list[tuple[str, str]], columns: int = 3) -> list[str]:
    """Markdown image table: ``cells`` are ``(caption, relative path)``."""
    lines = []
    for start in range(0, len(cells), columns):
        row = cells[start : start + columns]
        row += [("", "")] * (columns - len(row))
        lines.append("| " + " | ".join(caption for caption, _ in row) + " |")
        lines.append("|" + " --- |" * columns)
        images = [f"![{caption}]({path})" if path else "" for caption, path in row]
        lines.append("| " + " | ".join(images) + " |")
        lines.append("")
    return lines


def _scene_readme(
    name: str,
    scene: Scene,
    source: dict[str, str],
    effects: list[str],
    skipped: dict[str, list[str]],
    pipeline_name: str,
    reading: list[tuple[str, str]],
    pipeline_compared: bool,
) -> str:
    """The page for one scene: its source, how it was described to the transforms, and its images.

    The distances, the fog and veil limits, and which effects were left out
    and why, are stated with the images, since they decide how the effects
    look. ``reading`` is what a provider with labels and geometry read off
    the scene, as rows of a table; the effects in :data:`COMPARED` are shown
    as pairs, without it and with it, and so are the pipeline's draws when
    ``pipeline_compared``.
    """
    near, far = scene.depth_range_m
    fog_m, rain_m, snow_m = (
        factor * far for factor in (FOG_VISIBILITY, RAIN_VISIBILITY_FLOOR, SNOW_VISIBILITY_FLOOR)
    )
    where = "night" if scene.night else "day"
    cover = "under a roof" if scene.sheltered else "open to the sky"
    lines = [
        f"# {name}",
        "",
        f"{source['stresses'].capitalize()}.",
        "",
        "![original](original.jpg)",
        "",
        f"Source: [{source['title']}]({source['url']}) by {source['author']}, "
        f"{source['licence']}. Every image in this folder is derived from it and is "
        "available under the same licence.",
        "",
        "## How the transforms were told about this scene",
        "",
        "| | |",
        "| --- | --- |",
        f"| Distance of nearest and farthest point | {near:g} m to {far:g} m (estimated by eye) |",
        f"| Camera pitch below level | {scene.camera_pitch_deg:g} degrees |",
        f"| Lighting | {where}, {cover} |",
        f"| Sky in frame | {'yes' if scene.sky_in_frame else 'no'} |",
        f"| Fog | visibility {fog_m:g} m ({FOG_VISIBILITY:g} x the farthest point) |",
        f"| Rain and snow veils | capped so visibility stays above {rain_m:g} m and {snow_m:g} m |",
        "",
        "## What a provider with scene labels and geometry read off it",
        "",
        "The photograph and its label map, with the horizon the geometry estimator",
        "found drawn across both. See",
        "[scene understanding](../../../scene_understanding.md) for how each is made",
        "and how far to trust it.",
        "",
        "![labels and horizon](reading.jpg)",
        "",
        "| | |",
        "| --- | --- |",
        *(f"| {name} | {value} |" for name, value in reading),
        "",
        "## Without and with it",
        "",
        "Every effect that depends on what the provider knows, twice: on the left with",
        "the plain provider and the by-eye values above, on the right with the provider",
        "that read the scene. The parameters and the seed are the same, so what differs",
        "is what the labels and the geometry changed.",
        "",
    ]
    compared = [effect for effect in effects if effect in COMPARED]
    for effect in compared:
        pair = [
            (f"`{effect}` without", f"effects/{effect}.jpg"),
            (f"`{effect}` with", f"aware/{effect}.jpg"),
        ]
        lines += _table(pair, columns=2)
    others = [effect for effect in effects if effect not in COMPARED]
    lines += [
        f"## The other effects at severity {SEVERITY_LEVEL}",
        "",
        "One representative draw each: the median of seven by how much the frame changed.",
        "These do not read the scene, and are the same with either provider.",
        "",
        *_table([(effect, f"effects/{effect}.jpg") for effect in others]),
    ]
    for reason, names in skipped.items():
        listed = ", ".join(f"`{name}`" for name in names)
        lines += [f"Not rendered: {listed} — {reason}.", ""]
    lines += [f"## `{pipeline_name}` pipeline", ""]
    if pipeline_compared:
        lines += [
            f"{PIPELINE_DRAWS} consecutive draws, each a different combination of effects:",
            "on the left with the plain provider, on the right the same draw with the",
            "provider that read the scene.",
            "",
        ]
        for index in range(PIPELINE_DRAWS):
            name_ = f"{pipeline_name}_{index}.jpg"
            pair = [
                (f"draw {index} without", f"pipelines/{name_}"),
                (f"draw {index} with", f"pipelines_aware/{name_}"),
            ]
            lines += _table(pair, columns=2)
    else:
        lines += [
            f"{PIPELINE_DRAWS} consecutive draws. Each is a different combination of effects.",
            "This pipeline takes no depth provider, so there is nothing to compare.",
            "",
            *_table(
                [
                    (f"draw {index}", f"pipelines/{pipeline_name}_{index}.jpg")
                    for index in range(PIPELINE_DRAWS)
                ],
                columns=2,
            ),
        ]
    return "\n".join(lines).rstrip() + "\n"


def _index_readme(sources: dict[str, dict[str, str]]) -> str:
    """The index of the scene pages, with each scene's source and licence."""
    lines = [
        "# Effects on other scenes",
        "",
        "The severity gallery one level up shows every effect at five strengths over one",
        "photograph. These are the same effects at one strength over eight scenes chosen",
        "to be unlike it: a crossing from straight above, a covered car park, a real",
        "night street, mountains across a valley.",
        "",
        "Each scene's page says how the transforms were told about it: its distances,",
        "the camera's pitch, whether it is night or under a roof, and which effects were",
        "left out as not applying. Severity fixes extinction per metre, which suits a",
        "street and blanks a landscape, so fog is set by visibility as a fraction of the",
        "scene's depth and the veils of rain and snow are capped the same way.",
        "",
        "Each page then shows what a provider with",
        "[scene labels and geometry](../../scene_understanding.md) read off the scene",
        "for itself, and every effect that depends on it twice, side by side: without",
        "it and with it, from the same parameters and the same seed.",
        "",
        "Regenerate with `python docs/generate_scene_examples.py` (needs a CUDA GPU).",
        "The inputs are in [`sample_scenes/`](../../../sample_scenes/), with their authors",
        "and licences in [`SOURCES.md`](../../../sample_scenes/SOURCES.md). The images here",
        "are derived from them and carry the same licences, not the library's.",
        "",
        "| Scene | | What it stresses |",
        "| --- | --- | --- |",
    ]
    for name in SCENES:
        lines.append(
            f"| [`{name}`]({name}/README.md) | [![{name}]({name}/original.jpg)]({name}/README.md) "
            f"| {sources[name]['stresses']} |"
        )
    lines += [
        "",
        "## What these show that is not right",
        "",
        "They are unretouched output, and some of it is wrong in ways worth knowing.",
        "Where the provider's labels or geometry change the answer, the right-hand",
        "picture of each pair shows it:",
        "",
        "- **Wet ground is level ground that is not green.** Vegetation is left dry;",
        "  bare soil, gravel and roofs seen from above still get a film of water. With",
        "  labels it is also kept off what is named as a vehicle, a person, a tree or a",
        "  lawn, and laid on what is named as paving.",
        "- **A car's reflection is drawn straight below it.** That is right for a",
        "  road running away from the camera and wrong looking across the traffic,",
        "  where the reflection lies to one side. Tall things, such as walls and",
        "  lamp posts, get no reflection of their own.",
        "- **A reflection from outside the frame is a guess.** Where the mirrored point",
        "  is above the picture the road reflects an estimate of the sky, which is",
        "  right under open sky and wrong under the flyover in `underpass_from_above`.",
        "- **Overcast light cannot remove shadows.** Rain and snow grey the sky and",
        "  lower contrast, but the hard shadows of a sunny photograph stay.",
        "- **Night fog and night rain know only the light they can see.** Fog is as",
        "  bright as the surfaces around it suggest, and rain is lit by the lamps in",
        "  the frame. A lamp outside it lights both and shows in neither.",
        "- **How bright a lamp was is a guess.** A lamp is recorded at clipping. The",
        "  strength of its halo in fog, of the streaks round it in rain and of its",
        "  reflection in a wet road all rest on an assumed factor above that.",
        "- **Packet loss has no moving objects.** `transmission_faults` conceals from a",
        "  displaced copy of the same picture, so a stale band shows the scene",
        "  shifted, never a car where it was two seconds ago.",
        "- **Depth is relative and stretched to the range it is given.** The facades in",
        "  `urban_street_wet` are placed further away than they are, so its fog is",
        "  heavier than the stated visibility would make it. Where the estimator finds",
        "  the camera's height, depth is in metres measured from it instead, and a",
        "  veil set for the by-eye range can then be far too thick or too thin: the",
        "  parameters of each pair are deliberately the same.",
        "- **Wet ground stops at a few hundred metres.** Beyond that, level ground is",
        "  taken to be terrain; a distant wet road loses its sheen with it, unless it",
        "  is labelled as a road.",
        "- **Labels are a network's opinion.** It names the crossing in `crosswalk_day`,",
        "  seen from straight above, a building; nothing is dried or bared on its",
        "  word alone, which is why that scene's pairs hardly differ.",
        "- **The camera's height is a rough estimate**, from the sizes of the vehicles",
        "  and people in view, and every distance scales with it.",
        "- **Flakes close to an infrared lamp are very large.** `night_snow` draws the",
        "  nearest ones as long white bars.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    """Render every effect and the fitting pipeline over each sample scene, with their pages.

    Raises ``SystemExit`` if a scene has no entry in
    ``sample_scenes/SOURCES.md``: an image is not published without its
    author and licence.
    """
    sources = _sources()
    missing = sorted(set(SCENES) - set(sources))
    if missing:
        raise SystemExit(f"No entry in sample_scenes/SOURCES.md for: {missing}")

    estimator = DepthAnythingV2Estimator()
    labeller = SegmentationLabelEstimator()
    names = [n for n in EFFECTS if n not in _NEEDS_FFMPEG or ffmpeg_path() is not None]
    if len(names) != len(EFFECTS):
        print("ffmpeg not found; skipping the video-codec examples")

    for scene_name, scene in tqdm(SCENES.items(), desc="scenes"):
        image = _load(SCENES_DIR / f"{scene_name}.jpg")
        out = OUT_DIR / scene_name
        # One provider per scene: the distance calibration is the scene's.
        provider = SharedDepthProvider(estimator, depth_range_m=scene.depth_range_m)
        # Seeded per scene, so adding a scene re-rolls none of the others.
        rng = np.random.default_rng(sum(scene_name.encode()))
        _save(out / "original.jpg", image)

        # First everything with the plain provider, and only then the
        # second one: a transform with no provider of its own asks whichever
        # provider holds the frame, and must not be answered here by one
        # these pictures are meant to be without.
        rendered: list[str] = []
        skipped: dict[str, list[str]] = {}
        twins: list[tuple[str, dict[str, Any], dict[str, Any], int]] = []
        for effect in names:
            reason = _why_not(effect, scene)
            if reason is not None:
                skipped.setdefault(reason, []).append(effect)
                continue
            overrides = _overrides(effect, scene)
            # The generator as it stood, so that the twin can be built from
            # it and draw the very parameters this one does.
            before = rng.bit_generator.state
            transform = _build(effect, SEVERITY_LEVEL, VARIANT, rng, provider, overrides)
            drawn, seed = _representative(transform, _base(effect, image, provider, scene), rng)
            _save(out / "effects" / f"{effect}.jpg", drawn)
            rendered.append(effect)
            if effect in COMPARED:
                twins.append((effect, overrides, before, seed))

        pipeline_name, pipeline = _pipeline(scene, provider)
        if not isinstance(pipeline, A.Compose):
            raise TypeError(f"{pipeline_name} did not return a Compose")
        pipeline_seed = int(rng.integers(0, 2**31 - 1))
        pipeline.set_random_seed(pipeline_seed)
        for index in range(PIPELINE_DRAWS):
            _save(
                out / "pipelines" / f"{pipeline_name}_{index}.jpg", pipeline(image=image)["image"]
            )

        # Then the twin, which reads the scene's labels and geometry as
        # well. It is given the same by-eye distances, and uses them
        # wherever the camera's height cannot be found.
        aware = SharedDepthProvider(
            estimator,
            depth_range_m=scene.depth_range_m,
            labels=labeller,
            geometry=GroundPlaneGeometryEstimator(),
        )
        picture, reading = _reading(image, aware, scene)
        _save(out / "reading.jpg", picture)
        for effect, overrides, before, seed in twins:
            twin_rng = np.random.default_rng()
            twin_rng.bit_generator.state = before
            twin = _build(effect, SEVERITY_LEVEL, VARIANT, twin_rng, aware, overrides)
            twin.set_random_seed(seed)
            again = twin(image=_base(effect, image, aware, scene))["image"]
            _save(out / "aware" / f"{effect}.jpg", again)

        # The pipeline that takes a provider is drawn again over the twin.
        pipeline_compared = pipeline_name == "surveillance"
        if pipeline_compared:
            _name, twin_pipeline = _pipeline(scene, aware)
            twin_pipeline.set_random_seed(pipeline_seed)
            for index in range(PIPELINE_DRAWS):
                _save(
                    out / "pipelines_aware" / f"{pipeline_name}_{index}.jpg",
                    twin_pipeline(image=image)["image"],
                )

        (out / "README.md").write_text(
            _scene_readme(
                scene_name,
                scene,
                sources[scene_name],
                rendered,
                skipped,
                pipeline_name,
                reading,
                pipeline_compared,
            )
        )

    (OUT_DIR / "README.md").write_text(_index_readme(sources))
    print(f"Wrote {len(SCENES)} scenes to {OUT_DIR}")


if __name__ == "__main__":
    main()
