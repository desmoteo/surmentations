"""Draw the figures of ``docs/road_surface.md``.

Six pictures, made from the photographs the galleries use, with a
provider that carries scene labels and an estimated geometry:

* the roads as the library reads them: painted lines, lane lines and the
  wheel tracks laid in the lanes;
* settled snow without wheel tracks and with them;
* each scene's vehicles as a network reads them: which end, and its lamps;
* a soaked road as it was, and with its traffic's spray and lamps;
* infrared night lit evenly, as a daylight frame is, and lit by its street
  lamps and its traffic's lamps;
* rain and snow with an even veil in the distance and with its grain.

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
from surmentations.camera.night_ir import NightIrMode  # noqa: E402
from surmentations.depth import DepthAnythingV2Estimator, SharedDepthProvider  # noqa: E402
from surmentations.perception import (  # noqa: E402
    GroundPlaneGeometryEstimator,
    OpenVocabularyVehicleEstimator,
    SceneClass,
    SegmentationLabelEstimator,
)
from surmentations.presets.severity import GROUND_SEVERITY, SEVERITY  # noqa: E402
from surmentations.utils.photometry import to_linear_unit  # noqa: E402
from surmentations.utils.road import (  # noqa: E402
    painted_lines,
    read_road,
    road_cues,
    wheel_tracks,
)

OUT_DIR = REPO / "docs" / "assets" / "road_surface"
REFERENCE = REPO / "docs" / "examples" / "00_base_scene.png"
# The scenes shown: a motorway from a bridge, a city road, a street, a bend.
SHOWN = ("reference", "urban_road_day", "urban_street_wet", "underpass_from_above")
# How fast the traffic of each is taken to be going, in km/h: a frame does
# not say, and spray is mostly a matter of speed.
SPEEDS_KMH = {
    "reference": (95.0, 115.0),
    "urban_road_day": (75.0, 95.0),
    "urban_street_wet": (30.0, 45.0),
    "underpass_from_above": (30.0, 45.0),
}


def _photograph(name: str) -> np.ndarray:
    """One of the shown scenes, as RGB."""
    if name == "reference":
        return cv2.cvtColor(cv2.imread(str(REFERENCE)), cv2.COLOR_BGR2RGB)
    return _load(SCENES_DIR / f"{name}.jpg")


def _provider(name: str, depth: Any, labels: Any, vehicles: Any) -> SharedDepthProvider:
    """A provider for a scene that reads its labels, its geometry and its vehicles."""
    far = SCENES[name].depth_range_m if name in SCENES else (3.0, 300.0)
    return SharedDepthProvider(
        depth,
        depth_range_m=far,
        labels=labels,
        geometry=GroundPlaneGeometryEstimator(),
        vehicles=vehicles,
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
    """Each scene with its cues in yellow, paint in magenta, lanes in cyan, tracks in orange."""
    tiles = []
    for name, image in images.items():
        labels = providers[name].get_labels(image)
        assert labels is not None
        vehicles = labels == int(SceneClass.VEHICLE)
        on_road = vehicles | (labels == int(SceneClass.PERSON))
        carriageway = labels == int(SceneClass.ROAD)
        linear = to_linear_unit(image)[0]
        camera = providers[name].settle_camera(image)
        read = read_road(
            linear,
            carriageway,
            on_road,
            vehicles,
            np.random.default_rng(0),
            horizon_row=providers[name].horizon_row(image),
            camera_height_m=None if camera is None else camera.height_m,
        )
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
        for column, row, dx, dy, _weight in road_cues(linear, carriageway, on_road, vehicles):
            first = (int(column - 6 * dx), int(row - 6 * dy))
            last = (int(column + 6 * dx), int(row + 6 * dy))
            cv2.line(out, first, last, (255, 255, 0), 1, cv2.LINE_AA)
        tiles.append(out)
    _save("lanes.jpg", tiles, 2)


def read_vehicles(images: dict[str, np.ndarray], providers: dict[str, SharedDepthProvider]) -> None:
    """Each scene's vehicles as read: outlined green for a front and red for a rear, lamps boxed.

    Each vehicle by its own outline, as the network told it from its
    neighbours. Grey is one it did not read or leans neither way on.
    """
    tiles = []
    for name, image in images.items():
        out = image.copy()
        for view in providers[name].get_vehicles(image) or []:
            left, top, wide, tall = view.box
            colour = (150, 150, 150)
            if abs(view.facing) >= 0.03:
                colour = (255, 60, 60) if view.facing > 0.0 else (60, 255, 60)
            own = np.ones((tall, wide), dtype=np.uint8)
            if view.mask is not None:
                own = view.mask.astype(np.uint8)
            edges, _ = cv2.findContours(own, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(out, edges, -1, colour, 1, offset=(left, top))
            for x, y, half_wide, half_tall, _score in view.lamps.tolist():
                first = (round(x - half_wide), round(y - half_tall))
                last = (round(x + half_wide), round(y + half_tall))
                cv2.rectangle(out, first, last, (255, 255, 0), 1)
        tiles.append(out)
    _save("vehicle_views.jpg", tiles, 2)


def compare(
    images: dict[str, np.ndarray], providers: dict[str, SharedDepthProvider], effect: str
) -> None:
    """Each scene twice: settled snow without tracks and with, or a soaked road and its traffic's.

    The soaked road's second picture has spray at each scene's own speed
    (:data:`SPEEDS_KMH`) and the traffic's lamps lit.
    """
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
                    traffic_speed_range_kmh=SPEEDS_KMH[name],
                    vehicle_lamp_range=(amount / 0.6, amount / 0.6),
                    **settings,
                )
            transform.set_random_seed(4)
            tiles.append(transform(image=image)["image"])
    _save("wheel_tracks.jpg" if effect == "snow" else "wet_road_and_spray.jpg", tiles, 2)


def night_lamps(images: dict[str, np.ndarray], depth: Any, labels: Any, vehicles: Any) -> None:
    """Three scenes as an infrared night camera sees them: lit evenly, and lit by their lamps.

    The same draw both times. On the left with the lamps turned off, which
    is a daylight frame in grey: as bright far off as near, with the sun's
    shade still on the road. On the right the street lamps the reader
    found throw their pools and burn, the traffic's lamps are lit where it
    found them, and each lit vehicle throws its beam.
    """
    tiles = []
    for name in ("urban_road_day", "reference", "urban_street_wet"):
        provider = _provider(name, depth, labels, vehicles)
        for lamps in ({"street_lamp_range": (0.0, 0.0), "vehicle_lamp_range": (0.0, 0.0)}, {}):
            night = NightIrMode(depth_provider=provider, p=1.0, **lamps)
            night.set_random_seed(5)
            tiles.append(night(image=images[name])["image"])
    _save("night_lamps.jpg", tiles, 2)


def far_field(images: dict[str, np.ndarray], providers: dict[str, SharedDepthProvider]) -> None:
    """Rain and snow over two scenes, cut to their middle distance and enlarged.

    The same draw twice: on the left with the far veil even
    (``distant_rain_range`` and ``distant_snow_range`` at 0), which is fog
    with a few streaks over it; on the right with the grain of the drops
    and flakes too far off to draw one by one, each distance's share of it
    in front of only what lies beyond.
    """
    tiles = []
    cuts = {"underpass_from_above": (150, 60), "urban_street_wet": (380, 60)}
    for name, (left, top) in cuts.items():
        for transform, knob, level in (
            (RandomRain, "distant_rain_range", 4),
            (RandomSnow, "distant_snow_range", 3),
        ):
            for amount in (0.0, 1.0):
                settings = {
                    **SEVERITY[level][transform.__name__],
                    "camera_pitch_range_deg": _pitch(name),
                    knob: (amount, amount),
                }
                built = transform(depth_provider=providers[name], p=1.0, **settings)
                built.set_random_seed(3)
                out = built(image=images[name])["image"]
                tiles.append(
                    cv2.resize(
                        out[top : top + 210, left : left + 320],
                        (640, 420),
                        interpolation=cv2.INTER_NEAREST,
                    )
                )
    _save("far_field.jpg", tiles, 2)


def main() -> None:
    """Make the six figures."""
    depth, labels = DepthAnythingV2Estimator(), SegmentationLabelEstimator()
    images = {name: _photograph(name) for name in SHOWN}
    vehicles = OpenVocabularyVehicleEstimator()
    providers = {name: _provider(name, depth, labels, vehicles) for name in SHOWN}
    read_roads(images, providers)
    read_vehicles(images, providers)
    compare(images, providers, "snow")
    compare(images, providers, "rain")
    night_lamps(images, depth, labels, vehicles)
    far_field(images, providers)
    print(f"Wrote the figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
