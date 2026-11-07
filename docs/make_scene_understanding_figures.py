"""Draw the figures of ``docs/scene_understanding.md``, and print its table.

Two kinds of picture, both made from the photographs the galleries use:

* what the provider reads off each scene: its labels, with the horizon the
  geometry estimator found drawn across them;
* four effects on three scenes, each without and with that reading, so that
  what the labels and the geometry change can be seen and not taken on trust.

Run from the repository root, on a machine with a CUDA GPU::

    python docs/make_scene_understanding_figures.py

It writes to ``docs/assets/scene_understanding/`` and prints, as a markdown
table, the numbers the guide quotes for each scene.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from docs.generate_scene_examples import SCENES  # noqa: E402
from surmentations.atmosphere import GroundFog, RandomRain, RandomSnow  # noqa: E402
from surmentations.camera import NightIrMode  # noqa: E402
from surmentations.depth import DepthAnythingV2Estimator, SharedDepthProvider  # noqa: E402
from surmentations.perception import (  # noqa: E402
    GroundPlaneGeometryEstimator,
    SceneClass,
    SegmentationLabelEstimator,
)
from surmentations.presets.severity import GROUND_SEVERITY, SEVERITY  # noqa: E402

OUT_DIR = REPO / "docs" / "assets" / "scene_understanding"
REFERENCE = REPO / "docs" / "examples" / "00_base_scene.png"

# One colour per scene class, for the label maps.
PALETTE: dict[SceneClass, tuple[int, int, int]] = {
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

# The three scenes the effects are compared on: a motorway from a bridge,
# a city road under a long view, a lamp-lit street at night.
COMPARED = ("reference", "urban_road_day", "street_night")
NIGHT_RAIN = {
    "lamp_glint_range": (4.0, 12.0),
    "overcast_range": (0.0, 0.0),
    "scene_illuminance_range_lux": (5.0, 30.0),
}


def _photographs() -> dict[str, np.ndarray]:
    """The reference photograph and every gallery scene, as RGB."""
    paths = {"reference": REFERENCE}
    scenes = REPO / "docs" / "examples" / "scenes"
    paths |= {name: scenes / name / "original.jpg" for name in SCENES}
    images = {}
    for name, path in paths.items():
        image = cv2.imread(str(path))
        if image is None:
            raise SystemExit(f"cannot read {path}")
        images[name] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return images


def _depth_range(name: str) -> tuple[float, float]:
    """The near and far distance a scene's gallery page gives it, by eye."""
    return SCENES[name].depth_range_m if name in SCENES else (3.0, 300.0)


def _by_eye_pitch(name: str) -> float:
    """The pitch a scene's gallery page gives it, by eye."""
    return SCENES[name].camera_pitch_deg if name in SCENES else 18.0


def _coloured(labels: np.ndarray) -> np.ndarray:
    """A label map as a picture, one colour per class."""
    table = np.zeros((256, 3), dtype=np.uint8)
    for scene_class, colour in PALETTE.items():
        table[int(scene_class)] = colour
    return table[labels]


def _tile(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """``image`` resized to a tile of the figure."""
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def _save(name: str, image: np.ndarray) -> None:
    """Write an RGB figure as a JPEG."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(
        str(OUT_DIR / name),
        cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, 88],
    )


def _legend(width: int) -> np.ndarray:
    """A strip naming the colours of the label maps."""
    strip = np.full((34, width, 3), 255, dtype=np.uint8)
    step = width // len(PALETTE)
    for index, (scene_class, colour) in enumerate(PALETTE.items()):
        left = index * step
        cv2.rectangle(strip, (left + 6, 9), (left + 22, 25), colour, -1)
        cv2.rectangle(strip, (left + 6, 9), (left + 22, 25), (0, 0, 0), 1)
        cv2.putText(
            strip,
            scene_class.name.lower(),
            (left + 27, 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return strip


def read_scenes(
    images: dict[str, np.ndarray], depth: DepthAnythingV2Estimator, labels: Any
) -> list[str]:
    """Draw every scene beside its labels and horizon, and return the table's rows."""
    provider = SharedDepthProvider(
        depth, labels=labels, geometry=GroundPlaneGeometryEstimator(horizontal_fov_deg=60.0)
    )
    tiles, rows = [], []
    for name, image in images.items():
        height, width = image.shape[:2]
        told = provider.describe(image)
        geometry, camera, metric = told["geometry"], told["camera"], told["metric"]
        named = _coloured(provider.get_labels(image))
        photograph = image.copy()
        horizon = "none found"
        if geometry.horizon is not None:
            row = geometry.horizon * height
            slope = np.tan(np.radians(geometry.roll_deg))
            ends = [
                (0, round(row - 0.5 * width * slope)),
                (width - 1, round(row + 0.5 * width * slope)),
            ]
            for picture in (photograph, named):
                cv2.line(picture, ends[0], ends[1], (0, 0, 0), 5, cv2.LINE_AA)
                cv2.line(picture, ends[0], ends[1], (255, 235, 0), 2, cv2.LINE_AA)
            horizon = f"{geometry.horizon:.2f}" + (" or above" if geometry.horizon_is_bound else "")
        tiles.append(np.hstack([_tile(photograph, 360, 240), _tile(named, 360, 240)]))
        pitch = "open" if camera.pitch_deg is None else f"{camera.pitch_deg:+.0f}°"
        if geometry.horizon_is_bound and camera.pitch_deg is None:
            least = geometry.view(width, height, pitch_deg=-90.0)
            pitch = f"at least {least.pitch_deg:+.0f}°"
        tall = "—" if geometry.height_m is None else f"{geometry.height_m:.0f} m"
        metres = "—"
        if metric is not None:
            near = float(np.percentile(provider.get_metric_depth(image), 1))
            metres = f"from {near:.0f} m"
        rows.append(
            f"| `{name}` | {horizon} | {pitch} | {_by_eye_pitch(name):+.0f}° | {tall} | {metres} |"
        )
    grid = [np.hstack(tiles[index : index + 2]) for index in range(0, len(tiles) - 1, 2)]
    if len(tiles) % 2:
        last = np.hstack([tiles[-1], np.full_like(tiles[-1], 255)])
        grid.append(last)
    figure = np.vstack(grid)
    _save("labels_and_horizon.jpg", np.vstack([figure, _legend(figure.shape[1])]))
    return rows


def _transform(effect: str, name: str, provider: SharedDepthProvider) -> Any:
    """One of the four compared effects, built for scene ``name``."""
    pitch = _by_eye_pitch(name)
    by_eye = {"camera_pitch_range_deg": (pitch, pitch)}
    if effect == "wet_road":
        night = NIGHT_RAIN if name == "street_night" else {}
        soaked = {**SEVERITY[1]["RandomRain"], **GROUND_SEVERITY[5]["RandomRain"]}
        return RandomRain(depth_provider=provider, p=1.0, **soaked, **by_eye, **night)
    if effect == "settled_snow":
        deep = {**SEVERITY[2]["RandomSnow"], **GROUND_SEVERITY[4]["RandomSnow"]}
        return RandomSnow(depth_provider=provider, p=1.0, **deep, **by_eye)
    if effect == "night_ir":
        return NightIrMode(depth_provider=provider, p=1.0, **SEVERITY[3]["NightIrMode"])
    if effect == "ground_fog":
        return GroundFog(depth_provider=provider, p=1.0, **SEVERITY[3]["GroundFog"], **by_eye)
    raise ValueError(effect)


def compare_effects(
    images: dict[str, np.ndarray], depth: DepthAnythingV2Estimator, labels: Any
) -> None:
    """Each effect on three scenes: the photograph, without the reading, with it."""
    for effect in ("wet_road", "settled_snow", "night_ir", "ground_fog"):
        rows = []
        for name in COMPARED:
            image = images[name]
            tiles = [image]
            for aware in (False, True):
                extra: dict[str, Any] = {}
                if aware:
                    extra = {"labels": labels, "geometry": GroundPlaneGeometryEstimator()}
                provider = SharedDepthProvider(depth, depth_range_m=_depth_range(name), **extra)
                transform = _transform(effect, name, provider)
                transform.set_random_seed(3)
                tiles.append(transform(image=image)["image"])
            rows.append(np.hstack([_tile(tile, 400, 300) for tile in tiles]))
        _save(f"{effect}.jpg", np.vstack(rows))


def main() -> None:
    """Make every figure, and print the table of what was read off each scene."""
    images = _photographs()
    depth = DepthAnythingV2Estimator()
    labels = SegmentationLabelEstimator()
    rows = read_scenes(images, depth, labels)
    compare_effects(images, depth, labels)
    print("| Scene | Horizon | Pitch | Pitch by eye | Height | Depth in metres |")
    print("| --- | --- | --- | --- | --- | --- |")
    print("\n".join(rows))
    print(f"Wrote the figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
