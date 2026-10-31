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
    effects/<effect>.jpg     one representative draw per effect
    pipelines/<name>_<n>.jpg draws of the pipeline that fits the scene
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
    _NEEDS_FFMPEG,
    EFFECTS,
    GROUND_ROW_WEATHER_LEVEL,
    GROUND_ROWS,
    _build,
    _night_base,
    _render_representative,
)
from weathermentations.camera.codec import ffmpeg_path  # noqa: E402
from weathermentations.depth import DepthAnythingV2Estimator, SharedDepthProvider  # noqa: E402
from weathermentations.presets import surveillance, surveillance_lite  # noqa: E402
from weathermentations.presets.severity import SEVERITY  # noqa: E402
from weathermentations.utils.precipitation import (  # noqa: E402
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
LAMP_LIT = ("night_rain", "night_snow", "night_droplets")
NEEDS_SUN = ("sun_flare", "cast_shadows")
NEEDS_OPEN_SKY = (
    "rain", "snow", "wet_ground", "snow_cover", "night_rain", "night_snow", *NEEDS_SUN
)
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


def _load(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Could not read {path}")
    height, width = image.shape[:2]
    size = (WIDTH_PX, round(height * WIDTH_PX / width))
    return cv2.cvtColor(cv2.resize(image, size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)


def _save(path: Path, image: np.ndarray) -> None:
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
) -> str:
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
        f"## Effects at severity {SEVERITY_LEVEL}",
        "",
        "One representative draw each: the median of seven by how much the frame changed.",
        "",
        *_table([(effect, f"effects/{effect}.jpg") for effect in effects]),
    ]
    for reason, names in skipped.items():
        listed = ", ".join(f"`{name}`" for name in names)
        lines += [f"Not rendered: {listed} — {reason}.", ""]
    lines += [
        f"## `{pipeline_name}` pipeline",
        "",
        f"{PIPELINE_DRAWS} consecutive draws. Each is a different combination of effects.",
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
        "They are unretouched output, and some of it is wrong in ways worth knowing:",
        "",
        "- **Wet ground is level ground that is not green.** Vegetation is left dry;",
        "  bare soil, gravel and roofs seen from above still get a film of water.",
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
        "  heavier than the stated visibility would make it.",
        "- **Wet ground stops at a few hundred metres.** Beyond that, level ground is",
        "  taken to be terrain; a distant wet road loses its sheen with it.",
        "- **Flakes close to an infrared lamp are very large.** `night_snow` draws the",
        "  nearest ones as long white bars.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    sources = _sources()
    missing = sorted(set(SCENES) - set(sources))
    if missing:
        raise SystemExit(f"No entry in sample_scenes/SOURCES.md for: {missing}")

    estimator = DepthAnythingV2Estimator()
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

        rendered: list[str] = []
        skipped: dict[str, list[str]] = {}
        for effect in names:
            reason = _why_not(effect, scene)
            if reason is not None:
                skipped.setdefault(reason, []).append(effect)
                continue
            transform = _build(
                effect, SEVERITY_LEVEL, VARIANT, rng, provider, _overrides(effect, scene)
            )
            base = image
            if effect in LAMP_LIT:
                base = _night_base(image, provider, SEVERITY_LEVEL)
            _save(out / "effects" / f"{effect}.jpg", _render_representative(transform, base, rng))
            rendered.append(effect)

        pipeline_name, pipeline = _pipeline(scene, provider)
        if not isinstance(pipeline, A.Compose):
            raise TypeError(f"{pipeline_name} did not return a Compose")
        pipeline.set_random_seed(int(rng.integers(0, 2**31 - 1)))
        for index in range(PIPELINE_DRAWS):
            _save(
                out / "pipelines" / f"{pipeline_name}_{index}.jpg", pipeline(image=image)["image"]
            )

        (out / "README.md").write_text(
            _scene_readme(scene_name, scene, sources[scene_name], rendered, skipped, pipeline_name)
        )

    (OUT_DIR / "README.md").write_text(_index_readme(sources))
    print(f"Wrote {len(SCENES)} scenes to {OUT_DIR}")


if __name__ == "__main__":
    main()
