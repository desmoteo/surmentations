"""Draw the figures of ``docs/road_surface.md``.

Three pictures, made from the photographs the galleries use, with a
provider that carries scene labels and an estimated geometry:

* the roads as the library reads them: painted lines, lane lines and the
  wheel tracks laid in the lanes;
* settled snow without wheel tracks and with them;
* a soaked road without spray and with it.

Run from the repository root, on a machine with a CUDA GPU::

    python docs/make_road_surface_figures.py

It writes to ``docs/assets/road_surface/``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from docs.generate_scene_examples import SCENES, SCENES_DIR, _load  # noqa: E402
from surmentations.atmosphere import RandomRain, RandomSnow  # noqa: E402
from surmentations.depth import DepthAnythingV2Estimator, SharedDepthProvider  # noqa: E402
from surmentations.perception import (  # noqa: E402
    GroundPlaneGeometryEstimator,
    SceneClass,
    SegmentationLabelEstimator,
)
from surmentations.presets.severity import GROUND_SEVERITY, SEVERITY  # noqa: E402
from surmentations.utils.photometry import to_linear_unit  # noqa: E402
from surmentations.utils.road import painted_lines, read_road, wheel_tracks  # noqa: E402

OUT_DIR = REPO / "docs" / "assets" / "road_surface"
REFERENCE = REPO / "docs" / "examples" / "00_base_scene.png"
# The scenes shown: a motorway from a bridge, a city road, a street, a bend.
SHOWN = ("reference", "urban_road_day", "urban_street_wet", "underpass_from_above")


def _photograph(name: str) -> np.ndarray:
    """One of the shown scenes, as RGB."""
    if name == "reference":
        return cv2.cvtColor(cv2.imread(str(REFERENCE)), cv2.COLOR_BGR2RGB)
    return _load(SCENES_DIR / f"{name}.jpg")


def _provider(name: str, depth: Any, labels: Any) -> SharedDepthProvider:
    """A provider for a scene that reads its labels and geometry."""
    far = SCENES[name].depth_range_m if name in SCENES else (3.0, 300.0)
    return SharedDepthProvider(
        depth, depth_range_m=far, labels=labels, geometry=GroundPlaneGeometryEstimator()
    )


def _pitch(name: str) -> tuple[float, float]:
    """The by-eye pitch of a scene, as a range of one value."""
    pitch = SCENES[name].camera_pitch_deg if name in SCENES else 18.0
    return (pitch, pitch)


def _save(name: str, tiles: list[np.ndarray], columns: int) -> None:
    """Write tiles as one JPEG, ``columns`` to a row."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sized = [cv2.resize(tile, (640, 420), interpolation=cv2.INTER_AREA) for tile in tiles]
    rows = [np.hstack(sized[start : start + columns]) for start in range(0, len(sized), columns)]
    cv2.imwrite(
        str(OUT_DIR / name),
        cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, 88],
    )


def read_roads(images: dict[str, np.ndarray], providers: dict[str, SharedDepthProvider]) -> None:
    """Each scene with its paint in magenta, its lane lines in cyan and its tracks in orange."""
    tiles = []
    for name, image in images.items():
        labels = providers[name].get_labels(image)
        assert labels is not None
        vehicles = labels == int(SceneClass.VEHICLE)
        on_road = vehicles | (labels == int(SceneClass.PERSON))
        carriageway = labels == int(SceneClass.ROAD)
        linear = to_linear_unit(image)[0]
        read = read_road(linear, carriageway, on_road, vehicles, np.random.default_rng(0))
        out = image.copy()
        if read is not None:
            frame, lanes = read
            tracks = wheel_tracks(frame, lanes, 0.07, np.random.default_rng(0))[..., None]
            tinted = out * (1.0 - 0.6 * tracks) + np.array([255, 70, 0]) * 0.6 * tracks
            out = tinted.astype(np.uint8)
            for road in range(1, len(lanes.edges)):
                for line in lanes.edges[road]:
                    near = np.abs(frame.across - line) * frame.width_px < 1.2
                    out[near & (frame.piece == road)] = (0, 255, 255)
            out[painted_lines(linear, frame, carriageway, on_road)] = (255, 0, 255)
        tiles.append(out)
    _save("lanes.jpg", tiles, 2)


def compare(
    images: dict[str, np.ndarray], providers: dict[str, SharedDepthProvider], effect: str
) -> None:
    """Each scene twice: settled snow or a soaked road, without the new thing and with it."""
    tiles = []
    for name, image in images.items():
        for amount in (0.0, 0.6):
            if effect == "snow":
                settings = {**SEVERITY[1]["RandomSnow"], **GROUND_SEVERITY[4]["RandomSnow"]}
                transform: Any = RandomSnow(
                    depth_provider=providers[name],
                    p=1.0,
                    camera_pitch_range_deg=_pitch(name),
                    wheel_track_range=(amount, amount),
                    vehicle_cover_range=(0.1, 0.1),
                    **settings,
                )
            else:
                settings = {**SEVERITY[2]["RandomRain"], **GROUND_SEVERITY[5]["RandomRain"]}
                transform = RandomRain(
                    depth_provider=providers[name],
                    p=1.0,
                    camera_pitch_range_deg=_pitch(name),
                    spray_range=(amount, amount),
                    **settings,
                )
            transform.set_random_seed(4)
            tiles.append(transform(image=image)["image"])
    _save("wheel_tracks.jpg" if effect == "snow" else "wet_road_and_spray.jpg", tiles, 2)


def main() -> None:
    """Make the three figures."""
    depth, labels = DepthAnythingV2Estimator(), SegmentationLabelEstimator()
    images = {name: _photograph(name) for name in SHOWN}
    providers = {name: _provider(name, depth, labels) for name in SHOWN}
    read_roads(images, providers)
    compare(images, providers, "snow")
    compare(images, providers, "rain")
    print(f"Wrote the figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
