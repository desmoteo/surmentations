"""Draw the figures of ``docs/pipelines.md``.

Four pictures of what the ready-made pipeline does, each tile named in
the picture:

* the ten states of the air and the ground, by day, on one scene;
* four of them in each of the three lighting states: by day, in a night in
  colour, in an infrared night;
* two frames that are not dry daylight ones, a night photograph and a car
  park under a roof, taken for what they are not and read for what they
  are;
* eight things the pipeline draws together: rain with water on the glass,
  snow with snow on the housing, a wet lens glowing under infrared.

Each state is drawn alone, as the arm of
:func:`~surmentations.presets.weather_state` that holds it, without the
camera's own chain after it: the pipeline's draws as they come, chain and
all, are in the scene gallery (``docs/generate_scene_examples.py``).

Run from the repository root, on a machine with a CUDA GPU (without one the
networks run on the CPU, slowly)::

    python docs/make_pipeline_figures.py

It writes to ``docs/assets/pipelines/``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import albumentations as A
import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from docs.gallery import networks, save  # noqa: E402
from docs.make_road_surface_figures import _photograph, _provider  # noqa: E402
from surmentations.camera import NightColorMode  # noqa: E402
from surmentations.camera.night_ir import NightIrMode  # noqa: E402
from surmentations.depth import DepthAnythingV2Estimator  # noqa: E402
from surmentations.lens import SunFlare  # noqa: E402
from surmentations.perception import (  # noqa: E402
    OpenVocabularyVehicleEstimator,
    SegmentationLabelEstimator,
)
from surmentations.presets import (  # noqa: E402
    SEVERITY,
    WEATHER_STATES,
    surveillance,
    weather_state,
)
from surmentations.scene import CastShadows  # noqa: E402

OUT_DIR = REPO / "docs" / "assets" / "pipelines"
# The scene the states are shown on: a city road with traffic on it.
SCENE = "urban_road_day"
# What each state is called in a picture.
CALLED = {
    "dry": "dry air, dry road: the sun may be out",
    "wet_road": "road still wet, nothing falling",
    "rain_light": "light rain, to 3 mm/h",
    "rain_moderate": "moderate rain, to 10 mm/h",
    "rain_heavy": "heavy rain, 10 to 50 mm/h",
    "fog_thin": "fog, 200 m to 1 km",
    "fog_dense": "dense fog, 50 to 200 m",
    "fog_rain": "fog with drizzle",
    "snow": "snow falling",
    "snow_lying": "snow lying, none falling",
}
# Nothing on the glass or the housing, for the pictures of the states and
# of the lighting: what goes with each state has a picture of its own.
BARE = {
    "wet_rain": 0.0,
    "wet_fog": 0.0,
    "wet_snow": 0.0,
    "wet_after": 0.0,
    "wet_clear": 0.0,
    "snow_on_housing": 0.0,
}
# The camera chain kept out of the pictures of single frames, so that what
# differs between two tiles is what the caption says.
QUIET = {"codec": 0.0, "dirt": 0.0, "occluders": 0.0, "prior_power": 1.0}


def _named(tile: np.ndarray, text: str) -> np.ndarray:
    """A tile at the figures' size with its name on a dark strip along the bottom.

    Along the bottom and not the top: what hangs from a housing, a cap of
    snow or a web, hangs into the top of the view.
    """
    out = cv2.resize(tile, (640, 420), interpolation=cv2.INTER_AREA)
    out[-26:] = (0.35 * out[-26:]).astype(np.uint8)
    cv2.putText(out, text, (8, 412), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return np.asarray(out)


def _grid(name: str, tiles: list[np.ndarray], columns: int) -> None:
    """Write named tiles as one WebP, ``columns`` to a row."""
    rows = [np.hstack(tiles[start : start + columns]) for start in range(0, len(tiles), columns)]
    save(OUT_DIR / name, np.vstack(rows))


def _one_state(state: A.OneOf, name: str, seed: int) -> A.Compose:
    """The arm of a weather choice that holds one state, certain to fire."""
    arm = state.transforms[WEATHER_STATES.index(name)]
    return A.Compose([A.Sequential(list(arm.transforms), p=1.0)], seed=seed)


def _sun(provider: Any) -> dict[str, Any]:
    """Hard shadows and a flare, both certain: what only the dry states are given."""
    return {
        "sunlit": (
            CastShadows(depth_provider=provider, p=1.0, **SEVERITY[2]["CastShadows"]),
        ),
        "sunlit_after": (SunFlare(p=1.0, **SEVERITY[1]["SunFlare"]),),
    }


def states(image: np.ndarray, provider: Any) -> None:
    """The ten states of the air and the ground by day, each drawn alone."""
    by_day = weather_state(provider, **_sun(provider), **BARE)
    tiles = [
        _named(_one_state(by_day, name, seed=7)(image=image)["image"], CALLED[name])
        for name in WEATHER_STATES
    ]
    _grid("states.webp", tiles, 2)


def lighting(image: np.ndarray, provider: Any) -> None:
    """Four states in each lighting state: by day, a night in colour, an infrared night."""
    choices = {
        "by day": weather_state(provider, **_sun(provider), **BARE),
        "night in colour": weather_state(
            provider,
            night=True,
            **BARE,
            relight=NightColorMode(
                depth_provider=provider, p=1.0, **SEVERITY[1]["NightColorMode"]
            ),
        ),
        "infrared night": weather_state(
            provider,
            night=True,
            infrared=True,
            relight=NightIrMode(depth_provider=provider, p=1.0, **SEVERITY[1]["NightIrMode"]),
            **BARE,
        ),
    }
    tiles = []
    for name in ("dry", "rain_moderate", "fog_thin", "snow"):
        for light, choice in choices.items():
            out = _one_state(choice, name, seed=3)(image=image)["image"]
            tiles.append(_named(out, f"{light}: {CALLED[name].split(':')[0].split(',')[0]}"))
    _grid("lighting.webp", tiles, 3)


def source(depth: Any, labels: Any, vehicles: Any) -> None:
    """A night photograph and a car park, taken for dry daylight frames and read for what they are.

    The same pipeline and the same draw each time, with ``read_source`` off
    and on. The night photograph is given an infrared night with rain; the
    car park a day with rain.
    """
    rain = {"rain": 0.9, "snow": 0.0, "fog": 0.0, "wet_road": 0.0, "snow_lying": 0.0}
    cases = (
        ("street_night", {"day": 0.0, "lit_night": 0.0, "ir_night": 1.0}, "a night photograph"),
        ("parking_garage", {"day": 1.0, "lit_night": 0.0, "ir_night": 0.0}, "under a roof"),
    )
    tiles = []
    for name, light, what in cases:
        image = _photograph(name)
        provider = _provider(name, depth, labels, vehicles, image)
        for read, how in ((False, "taken for a dry daylight frame"), (True, "read for what it is")):
            pipeline = surveillance(provider, read_source=read, **light, **rain, **QUIET)
            pipeline.set_random_seed(2)
            tiles.append(_named(pipeline(image=image)["image"], f"{what}, {how}"))
    _grid("source.webp", tiles, 2)


def combinations(image: np.ndarray, provider: Any) -> None:
    """Eight things that go together, each drawn for certain.

    The states' own arms again, with the odds of what goes with each put
    to 1: water on the glass with rain, with fog and after rain, snow on
    the housing with snow, the sun over lying snow, and under infrared a
    wet lens in the illuminator's light.
    """
    wet = {"wet_rain": 1.0, "wet_fog": 1.0, "wet_after": 1.0, "wet_snow": 0.0, "wet_clear": 0.0}
    by_day = weather_state(provider, **wet, snow_on_housing=1.0, ground_severity=4)
    sunny = weather_state(provider, **_sun(provider), snow_on_housing=0.0, wet_clear=0.0)
    infrared = weather_state(
        provider,
        night=True,
        infrared=True,
        relight=NightIrMode(depth_provider=provider, p=1.0, **SEVERITY[1]["NightIrMode"]),
        **wet,
    )
    drawn = (
        (by_day, "rain_moderate", 4, "rain, a wet road, water on the glass"),
        (by_day, "wet_road", 4, "the rain over, the road wet, drops left on the glass"),
        (by_day, "fog_rain", 4, "drizzle through fog, the glass wet"),
        (by_day, "fog_dense", 4, "dense fog, condensation on the glass"),
        (by_day, "snow", 4, "snow falling on lying snow, and on the housing"),
        (sunny, "snow_lying", 7, "snow lying, the sun out"),
        (infrared, "rain_moderate", 4, "infrared night, rain, a wet lens in the lamp"),
        (infrared, "fog_thin", 4, "infrared night, fog, a wet lens in the lamp"),
    )
    tiles = [
        _named(_one_state(choice, name, seed=seed)(image=image)["image"], text)
        for choice, name, seed, text in drawn
    ]
    _grid("combinations.webp", tiles, 2)


def main() -> None:
    """Make the four figures."""
    where = networks()
    depth = DepthAnythingV2Estimator(**where)
    labels = SegmentationLabelEstimator(**where)
    vehicles = OpenVocabularyVehicleEstimator(**where)
    image = _photograph(SCENE)
    provider = _provider(SCENE, depth, labels, vehicles, image)
    states(image, provider)
    lighting(image, provider)
    source(depth, labels, vehicles)
    combinations(image, provider)
    print(f"Wrote the figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
