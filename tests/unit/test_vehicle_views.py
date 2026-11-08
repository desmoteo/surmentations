"""Which end of each vehicle is seen, and where its lamps are.

``surmentations.perception.vehicles`` asks a network, once per scene, what
colour says poorly: whether a vehicle shows its front or its rear, and
where its lamps are. Most of what is tested here needs no network: what is
made of a detector's boxes, how a provider keeps the reading, and what the
road's traffic and its lamps do with it. The network itself is run on the
reference scene, where which cars are coming on is plain to see.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import cv2
import numpy as np
import pytest

from surmentations.atmosphere import RandomRain
from surmentations.depth.estimator import DepthEstimator
from surmentations.depth.provider import SharedDepthProvider
from surmentations.diagnostics import recording
from surmentations.perception import (
    SceneClass,
    SceneLabelEstimator,
    VehicleView,
    VehicleViewEstimator,
)
from surmentations.perception.vehicles import check_views, lamps_of, vehicle_outlines
from surmentations.utils.road import road_frame, road_vehicles, vehicle_lamps

HEIGHT, WIDTH = 240, 320
BOX = (100, 80, 60, 50)


def _found(*rows: tuple[float, float, float, float, int, float]) -> np.ndarray:
    """A detector's boxes: left, top, right, bottom, kind and score of each."""
    return np.array(rows, dtype=np.float32).reshape(-1, 6)


def _lamp(x: float, y: float, kind: int, score: float, half: float = 4.0) -> tuple:
    """One box of a lamp's size with its middle at a point of the frame."""
    return (x - half, y - half, x + half, y + half, kind, score)


def test_a_vehicle_shows_the_end_whose_lamps_are_surer_and_its_lamps_are_the_pair() -> None:
    """Two headlamps either side, at one level: a front, and those are its
    lamps. A number plate between them, a bumper as wide as the car and a
    box on the road beside it are none of them lamps."""
    outline = np.ones((BOX[3], BOX[2]), dtype=bool)
    view = lamps_of(
        BOX,
        outline,
        _found(
            _lamp(110.0, 115.0, 0, 0.4),
            _lamp(150.0, 116.0, 0, 0.35),
            _lamp(130.0, 120.0, 0, 0.3),
            (102.0, 110.0, 158.0, 128.0, 0, 0.6),
            _lamp(60.0, 115.0, 0, 0.5),
            _lamp(125.0, 100.0, 1, 0.08),
        ),
    )
    assert view.box == BOX
    assert view.front == pytest.approx(0.4) and view.rear == pytest.approx(0.08)
    assert view.facing < 0.0
    assert view.lamps.shape == (2, 5)
    assert view.lamps[:, 0].tolist() == [110.0, 150.0] and view.lamps[:, 1].tolist() == [
        115.0,
        116.0,
    ]
    assert view.lamps[:, 2:4].tolist() == [[4.0, 4.0], [4.0, 4.0]]


def test_a_rear_is_one_whose_tail_lamps_are_found_and_one_lamp_is_enough() -> None:
    """A tail lamp seen and the other hidden: a rear, with the one lamp. And
    of two boxes on one lamp the surer is kept."""
    outline = np.ones((BOX[3], BOX[2]), dtype=bool)
    view = lamps_of(
        BOX,
        outline,
        _found(
            _lamp(112.0, 110.0, 1, 0.5), _lamp(113.0, 111.0, 1, 0.3), _lamp(140.0, 112.0, 0, 0.1)
        ),
    )
    assert view.facing == pytest.approx(0.4)
    assert view.lamps.shape == (1, 5) and view.lamps[0, :2].tolist() == [112.0, 110.0]
    # Two that are not a pair, one above the other: the surer alone.
    stacked = lamps_of(
        BOX, outline, _found(_lamp(112.0, 90.0, 1, 0.5), _lamp(150.0, 125.0, 1, 0.4))
    )
    assert stacked.lamps.shape == (1, 5) and stacked.lamps[0, 4] == pytest.approx(0.5)


def test_a_vehicle_nothing_is_found_on_says_nothing() -> None:
    """No box, or none the detector believes: no end, no lamps."""
    outline = np.ones((BOX[3], BOX[2]), dtype=bool)
    for found in (_found(), _found(_lamp(120.0, 110.0, 0, 0.01))):
        view = lamps_of(BOX, outline, found)
        assert view.front == view.rear == view.facing == 0.0
        assert view.lamps.shape == (0, 5)
    # A lamp on the box and off the outline is another vehicle's.
    corner = np.zeros((BOX[3], BOX[2]), dtype=bool)
    corner[:20, :20] = True
    assert lamps_of(BOX, corner, _found(_lamp(150.0, 120.0, 1, 0.6))).rear == 0.0


def test_outlines_are_the_connected_pieces_of_the_vehicle_mask_largest_first() -> None:
    vehicles = np.zeros((HEIGHT, WIDTH), dtype=bool)
    vehicles[100:130, 40:70] = True
    vehicles[50:60, 200:210] = True
    vehicles[150:200, 150:230] = True
    vehicles[10:12, 10:12] = True
    outlines = vehicle_outlines(vehicles, min_pixels=50, most=8)
    assert [box for box, _mask in outlines] == [
        (150, 150, 80, 50),
        (40, 100, 30, 30),
        (200, 50, 10, 10),
    ]
    assert all(mask.all() and mask.shape == (box[3], box[2]) for box, mask in outlines)
    assert len(vehicle_outlines(vehicles, min_pixels=50, most=2)) == 2
    assert vehicle_outlines(np.zeros_like(vehicles), 50, 8) == []


class _Level(DepthEstimator):
    """Depth of a level road: nearer towards the bottom of the frame."""

    def estimate(self, image: np.ndarray) -> np.ndarray:
        """A ramp from far at the top to near at the bottom."""
        height, width = image.shape[:2]
        rows = np.linspace(1.0, 0.0, height, dtype=np.float32)[:, None]
        return np.ascontiguousarray(np.broadcast_to(rows, (height, width)))


class _Labels(SceneLabelEstimator):
    """A label map that is always the same one."""

    def __init__(self, labels: np.ndarray) -> None:
        """Keep the map."""
        self.labels = labels

    def estimate(self, image: np.ndarray) -> np.ndarray:
        """The map, whatever the frame."""
        return self.labels


class _Told(VehicleViewEstimator):
    """Views that are told: every outline shows one end, with a pair of lamps on it."""

    def __init__(self, facing: float, lamps: bool = True) -> None:
        """Keep which end every vehicle shows, and count the frames read."""
        self.facing = facing
        self.lamps = lamps
        self.calls = 0

    def estimate(self, image: np.ndarray, vehicles: np.ndarray) -> list[VehicleView]:
        """One view per outline, with its lamps a quarter of the way in and half way down."""
        self.calls += 1
        views = []
        for (left, top, wide, tall), _mask in vehicle_outlines(vehicles, 20, 32):
            lamps = np.zeros((0, 5), dtype=np.float32)
            if self.lamps:
                level = top + 0.5 * tall
                lamps = np.array(
                    [
                        [left + 0.25 * wide, level, 2.0, 1.5, 0.5],
                        [left + 0.75 * wide, level, 2.0, 1.5, 0.5],
                    ],
                    dtype=np.float32,
                )
            front, rear = (0.0, self.facing) if self.facing > 0.0 else (-self.facing, 0.0)
            views.append(VehicleView((left, top, wide, tall), front, rear, lamps))
        return views


def _scene() -> tuple[np.ndarray, np.ndarray]:
    """A road up the frame under a sky, with two cars on it, and its labels."""
    labels = np.full((HEIGHT, WIDTH), int(SceneClass.GRASS), dtype=np.uint8)
    labels[:60] = int(SceneClass.SKY)
    labels[60:, 60:260] = int(SceneClass.ROAD)
    labels[150:180, 96:124] = int(SceneClass.VEHICLE)
    labels[90:120, 196:224] = int(SceneClass.VEHICLE)
    image = np.full((HEIGHT, WIDTH, 3), 90, dtype=np.uint8)
    image[:60] = (150, 180, 230)
    return image, labels


def test_a_provider_reads_a_scene_s_vehicles_once_and_only_where_it_can() -> None:
    """With the scene's depth and labels, from the same frame; not at all
    without labels to say where the vehicles are, nor for a scene with none."""
    image, labels = _scene()
    told = _Told(1.0)
    provider = SharedDepthProvider(_Level(), labels=_Labels(labels), vehicles=told)
    views = provider.get_vehicles(image)
    assert views is not None and sorted(view.box for view in views) == [
        (96, 150, 28, 30),
        (196, 90, 28, 30),
    ]
    assert provider.get_vehicles(image) is views and told.calls == 1
    provider.get_depth(image)
    assert told.calls == 1

    assert SharedDepthProvider(_Level(), labels=_Labels(labels)).get_vehicles(image) is None
    unlabelled = _Told(1.0)
    assert SharedDepthProvider(_Level(), vehicles=unlabelled).get_vehicles(image) is None
    empty = _Told(1.0)
    nothing = np.full((HEIGHT, WIDTH), int(SceneClass.ROAD), dtype=np.uint8)
    assert (
        SharedDepthProvider(_Level(), labels=_Labels(nothing), vehicles=empty).get_vehicles(image)
        == []
    )
    assert unlabelled.calls == empty.calls == 0

    # Labels told for a frame afterwards are read for their vehicles too.
    late = _Told(-1.0)
    provider = SharedDepthProvider(_Level(), vehicles=late)
    provider.get_depth(image)
    provider.remember(image, labels=labels)
    views = provider.get_vehicles(image)
    assert views is not None and len(views) == 2 and late.calls == 1


def test_a_reader_of_the_wrong_kind_and_a_reading_of_the_wrong_kind_are_refused() -> None:
    image, labels = _scene()
    with pytest.raises(TypeError, match="VehicleViewEstimator"):
        SharedDepthProvider(_Level(), vehicles="owl")  # type: ignore[arg-type]

    class Boxes(VehicleViewEstimator):
        """An estimator that forgot what it was to return."""

        def estimate(self, image: np.ndarray, vehicles: np.ndarray) -> list:
            """Bare boxes."""
            return [(0, 0, 4, 4)]

    with pytest.raises(TypeError, match="Boxes gave a tuple"):
        SharedDepthProvider(_Level(), labels=_Labels(labels), vehicles=Boxes()).get_vehicles(image)
    assert check_views([], source="nobody") == []


def test_a_provider_pickles_with_its_vehicle_reader() -> None:
    image, labels = _scene()
    provider = SharedDepthProvider(_Level(), labels=_Labels(labels), vehicles=_Told(1.0))
    provider.get_vehicles(image)
    again = pickle.loads(pickle.dumps(provider))
    views = again.get_vehicles(image)
    assert views is not None and len(views) == 2


def _traffic_scene() -> tuple:
    """A straight road with two cars on its left and two on its right, all plain grey."""
    mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
    mask[:, 60:260] = True
    frame = road_frame(mask)
    assert frame is not None
    vehicles = np.zeros((HEIGHT, WIDTH), dtype=bool)
    scene = np.full((HEIGHT, WIDTH, 3), 0.1, dtype=np.float32)
    boxes = ((80, 100, 26, 30), (100, 170, 26, 30), (200, 100, 26, 30), (215, 170, 26, 30))
    for left, top, wide, tall in boxes:
        vehicles[top : top + tall, left : left + wide] = True
        scene[top : top + tall, left : left + wide] = (0.3, 0.32, 0.35)
    return frame, vehicles, scene, boxes


def _view(box: tuple[int, int, int, int], facing: float) -> VehicleView:
    """A view of one outline showing one end that surely, with a pair of lamps half way down it."""
    left, top, wide, tall = box
    lamps = np.array(
        [
            [left + 0.25 * wide, top + 0.5 * tall, 2.0, 1.5, abs(facing)],
            [left + 0.75 * wide, top + 0.5 * tall, 2.0, 1.5, abs(facing)],
        ],
        dtype=np.float32,
    )
    front, rear = (0.0, facing) if facing > 0.0 else (-facing, 0.0)
    return VehicleView(box, front, rear, lamps)


def test_a_network_s_reading_is_heard_in_place_of_colour() -> None:
    """Grey cars with no red on them read as fronts by colour. Told they are
    rears, they are rears, every one, with the lamps that were found."""
    frame, vehicles, scene, boxes = _traffic_scene()
    by_colour = road_vehicles(frame, vehicles, np.random.default_rng(0), scene)
    assert all(vehicle.away == -1.0 and vehicle.lamps is None for vehicle in by_colour)
    views = [_view(box, 0.4) for box in boxes]
    told = road_vehicles(frame, vehicles, np.random.default_rng(0), scene, views=views)
    assert len(told) == 4
    for vehicle in told:
        assert vehicle.shows == vehicle.away == 1.0
        assert vehicle.lamps is not None and vehicle.lamps.shape == (2, 5)


def test_a_vehicle_the_network_is_sure_of_keeps_its_own_way_and_the_rest_go_with_their_road() -> (
    None
):
    """Three rears and one front it is sure of, on one side of a road: the
    front is coming on whatever its neighbours do. One it only leans on goes
    with them, and its lamps, found on the end it does not show, are left."""
    frame, vehicles, scene, boxes = _traffic_scene()
    sure = [_view(boxes[0], 0.4), _view(boxes[1], 0.4), _view(boxes[2], 0.4), _view(boxes[3], -0.4)]
    ways = {
        vehicle.foot[1]: vehicle
        for vehicle in road_vehicles(frame, vehicles, np.random.default_rng(0), scene, views=sure)
    }
    front = max(ways)
    assert ways[front].away == -1.0 and ways[front].lamps is not None
    assert all(vehicle.away == 1.0 for column, vehicle in ways.items() if column != front)

    leaning = [*sure[:3], _view(boxes[3], -0.06)]
    ways = {
        vehicle.foot[1]: vehicle
        for vehicle in road_vehicles(
            frame, vehicles, np.random.default_rng(0), scene, views=leaning
        )
    }
    assert ways[front].shows == -1.0 and ways[front].away == 1.0
    assert ways[front].lamps is None, "headlamps kept for a car taken to show its rear"

    # One the network did not read is read by colour, as before.
    partly = road_vehicles(frame, vehicles, np.random.default_rng(0), scene, views=sure[:1])
    assert sorted(vehicle.shows for vehicle in partly) == [-1.0, -1.0, -1.0, 1.0]


def test_lamps_a_network_found_are_lit_where_it_found_them() -> None:
    """Not a quarter of the way up the outline, where a lamp is put when
    nothing is known: half way down it, where these were found. Red on a
    rear and white on a front, and no larger than a lamp is."""
    frame, vehicles, scene, boxes = _traffic_scene()
    for facing in (0.4, -0.4):
        views = [_view(box, facing) for box in boxes]
        lit = 0
        for seed in range(6):
            rng = np.random.default_rng(seed)
            traffic = road_vehicles(frame, vehicles, rng, scene, views=views)
            light, _glow, cores = vehicle_lamps(frame, traffic, rng, 1.0, scene)
            for left, top, wide, tall in boxes:
                car = light[top : top + tall, left : left + wide]
                if float(car.max()) == 0.0:
                    continue
                lit += 1
                rows, columns = np.nonzero(car.max(axis=2) > 0.5 * car.max())
                assert abs(float(rows.mean()) - 0.5 * tall) < 2.0, "lamps not where they were found"
                assert {int(column) for column in columns} <= set(range(3, 11)) | set(range(16, 24))
                red, green = float(car[..., 0].sum()), float(car[..., 1].sum())
                assert (green < 0.1 * red) if facing > 0.0 else (green > 0.8 * red)
            assert float(cores.max()) <= 1.0
        assert lit >= 12


def test_rain_lights_the_lamps_a_provider_s_reader_found() -> None:
    """Through the transform: the provider's reading is what the lamps are
    lit by, at the places it found them."""
    image, labels = _scene()
    settings: dict[str, object] = {
        "max_streaks": 0,
        "wetness_probability": 1.0,
        "wetness_darkening_range": (0.4, 0.4),
        "puddle_coverage_range": (0.0, 0.0),
        "camera_pitch_range_deg": (35.0, 35.0),
        "rain_rate_range_mm_h": (2.0, 2.0),
    }
    shown = {}
    for facing in (1.0, -1.0):
        provider = SharedDepthProvider(
            _Level(), depth_range_m=(4.0, 40.0), labels=_Labels(labels), vehicles=_Told(facing)
        )
        plain = RandomRain(depth_provider=provider, p=1.0, **settings)  # type: ignore[arg-type]
        lamps = RandomRain(
            depth_provider=provider,
            p=1.0,
            vehicle_lamp_range=(1.0, 1.0),
            **settings,  # type: ignore[arg-type]
        )
        change = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32)
        lit = 0
        for seed in range(6):
            plain.set_random_seed(seed)
            lamps.set_random_seed(seed)
            before = plain(image=image)["image"].astype(np.float32)
            with recording() as log:
                after = lamps(image=image)["image"].astype(np.float32)
            lit += bool(log[0]["drew"]["lamps_lit"])
            change = np.maximum(change, after - before)
        # By day most vehicles are lit, and not every one in every frame.
        assert lit >= 4
        shown[facing] = change
        # Half way down each car, where the reader put them, either side.
        for left, top in ((96, 150), (196, 90)):
            car = change[top : top + 30, left : left + 28]
            rows, _columns = np.nonzero(car.max(axis=2) > 0.6 * car.max())
            assert float(car.max()) > 100.0 and abs(float(rows.mean()) - 15.0) < 2.5
    rear, front = shown[1.0][150:180, 96:124], shown[-1.0][150:180, 96:124]
    assert float(rear[..., 1].max()) < 0.4 * float(rear[..., 0].max()), (
        "tail lamps that are not red"
    )
    assert float(front[..., 1].max()) > 0.8 * float(front[..., 0].max()), (
        "headlamps that are not white"
    )


requires_cuda = pytest.mark.skipif(
    not __import__("torch").cuda.is_available(), reason="the detection network needs a GPU"
)


@requires_cuda
def test_the_bundled_network_tells_the_reference_scene_s_fronts_from_its_rears() -> None:
    """A motorway seen from a bridge: the traffic on its left carriageway is
    coming on and that on its right is driving away. Not a test of the
    network's accuracy, which is its authors' to report: a test that what
    it finds arrives on the right vehicles, in the frame's own pixels."""
    from surmentations.perception import OpenVocabularyVehicleEstimator, SegmentationLabelEstimator

    path = Path(__file__).resolve().parents[2] / "docs" / "examples" / "00_base_scene.png"
    image = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    vehicles = SegmentationLabelEstimator().estimate(image) == int(SceneClass.VEHICLE)
    views = OpenVocabularyVehicleEstimator().estimate(image, vehicles)
    assert views and all(isinstance(view, VehicleView) for view in views)
    by_foot = {(view.box[0] + view.box[2] // 2, view.box[1] + view.box[3]): view for view in views}

    def nearest(column: int, row: int) -> VehicleView:
        """The view whose outline stands nearest a point of the road."""
        return by_foot[min(by_foot, key=lambda foot: abs(foot[0] - column) + abs(foot[1] - row))]

    # The four cars nearest the camera on the left, and three on the right.
    for column, row in ((307, 605), (418, 595), (348, 492), (403, 475)):
        view = nearest(column, row)
        assert view.facing < -0.1, f"a car coming on at {column},{row} read as {view.facing:+.2f}"
    for column, row in ((513, 461), (494, 416), (450, 419)):
        view = nearest(column, row)
        assert view.facing > 0.1, f"a car driving away at {column},{row} read as {view.facing:+.2f}"
    for column, row in ((307, 605), (418, 595), (513, 461)):
        view = nearest(column, row)
        left, top, wide, tall = view.box
        assert len(view.lamps) == 2, "no pair of lamps on a car that fills a tenth of the frame"
        for x, y, half_wide, half_tall, score in view.lamps.tolist():
            assert left - 3 <= x <= left + wide + 3 and top + 0.3 * tall <= y <= top + tall + 3
            assert 0.0 < half_wide < 0.2 * wide and 0.0 < half_tall < 0.2 * tall and score > 0.05
        assert abs(view.lamps[0, 0] - view.lamps[1, 0]) > 0.3 * wide
    # A float frame is the same frame.
    as_float = OpenVocabularyVehicleEstimator().estimate(image.astype(np.float32) / 255.0, vehicles)
    assert [view.box for view in as_float] == [view.box for view in views]
