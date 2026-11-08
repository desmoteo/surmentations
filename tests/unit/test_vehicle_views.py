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
from surmentations.camera.night_ir import NightIrMode
from surmentations.depth.estimator import DepthEstimator
from surmentations.depth.provider import SharedDepthProvider
from surmentations.diagnostics import recording
from surmentations.perception import (
    SceneClass,
    SceneLabelEstimator,
    VehicleView,
    VehicleViewEstimator,
)
from surmentations.perception.vehicles import (
    check_views,
    lamps_of,
    separate_vehicles,
    split_outlines,
    vehicle_outlines,
)
from surmentations.utils.road import (
    road_frame,
    road_vehicles,
    vehicle_lamps,
    vehicle_spray,
    vehicles_as_seen,
)

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
    none = np.zeros((0, 5), dtype=np.float32)
    with pytest.raises(ValueError, match="a mask of shape"):
        check_views([VehicleView((0, 0, 4, 6), 0.0, 0.0, none, np.ones((4, 6), bool))], source="x")


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


def _box(left: float, top: float, wide: float, tall: float, score: float, kind: int = 0) -> tuple:
    """One box a detector found: left, top, right, bottom, kind and score."""
    return (left, top, left + wide, top + tall, kind, score)


def test_boxes_that_are_each_one_vehicle_are_kept_and_the_rest_are_not() -> None:
    """A detector asked for cars answers with the same car twice, a row of
    cars as one, a hoarding, and things it is not sure of."""
    vehicles = np.zeros((HEIGHT, WIDTH), dtype=bool)
    vehicles[100:130, 40:70] = vehicles[100:130, 74:104] = vehicles[100:130, 108:138] = True
    vehicles[180:220, 200:280] = True
    vehicles[190:200, 230:240] = True
    kept = separate_vehicles(
        _found(
            _box(40, 100, 30, 30, 0.45),
            _box(74, 100, 30, 30, 0.44),
            _box(108, 100, 30, 30, 0.43),
            _box(42, 101, 30, 30, 0.3),  # the first car again
            _box(38, 98, 102, 34, 0.5),  # the three as one
            _box(200, 20, 60, 30, 0.6),  # a hoarding: nothing labelled vehicle under it
            _box(74, 100, 30, 30, 0.1),  # not sure enough
            _box(200, 180, 80, 40, 0.4, kind=3),  # a bus
            _box(205, 184, 60, 30, 0.35),  # most of the bus, called a car
            _box(228, 188, 14, 14, 0.3),  # a car in front of the bus
        ),
        vehicles,
    )
    assert kept.shape == (5, 6)
    assert sorted(kept[:, 0].tolist()) == [40.0, 74.0, 108.0, 200.0, 228.0]
    assert kept[kept[:, 0] == 200.0, 4].tolist() == [3.0], "the bus is kept as a bus"
    assert kept[:, 5].tolist() == sorted(kept[:, 5].tolist(), reverse=True)
    assert separate_vehicles(_found(), vehicles).shape == (0, 6)


def test_an_outline_is_shared_out_among_the_vehicles_found_on_it() -> None:
    """Two cars one behind the other, run together by the labels: two
    vehicles, and where their boxes overlap the pixels are the nearer
    one's, and a little of what lies just beside a box is its vehicle's. A
    piece well away from both is a third nobody found. An outline
    with one box is one vehicle of that kind, and with none one of no kind."""
    vehicles = np.zeros((HEIGHT, WIDTH), dtype=bool)
    vehicles[60:150, 100:130] = True  # two cars nose to tail, as one outline
    vehicles[60:75, 130:190] = True  # and something beside the far one
    vehicles[180:210, 30:60] = True
    vehicles[180:210, 250:280] = True
    found = _found(
        _box(100, 60, 30, 50, 0.4), _box(100, 100, 30, 50, 0.45), _box(30, 180, 30, 30, 0.4, kind=3)
    )
    apart = split_outlines(vehicles, found)
    by_box = {box: (mask, kind) for box, mask, kind in apart}
    assert set(by_box) == {
        (100, 100, 30, 50),
        (100, 60, 38, 40),
        (138, 60, 52, 15),
        (30, 180, 30, 30),
        (250, 180, 30, 30),
    }
    # The overlap, rows 100 to 110, went to the nearer car, which reaches lower.
    assert by_box[(100, 100, 30, 50)][0].all() and by_box[(100, 100, 30, 50)][1] == "car"
    assert by_box[(100, 60, 38, 40)][1] == "car"
    assert by_box[(138, 60, 52, 15)][1] == "", "a piece no box is near is of no known kind"
    assert by_box[(30, 180, 30, 30)][1] == "bus" and by_box[(250, 180, 30, 30)][1] == ""
    # Every pixel of the mask is some vehicle's, once.
    total = np.zeros((HEIGHT, WIDTH), dtype=np.int32)
    for (left, top, wide, tall), mask, _kind in apart:
        total[top : top + tall, left : left + wide] += mask
    assert (total == vehicles).all()
    sizes = [int(mask.sum()) for _box, mask, _kind in apart]
    assert sizes == sorted(sizes, reverse=True)


def _column_of_cars() -> tuple:
    """Three grey cars nose to tail up one lane, run together by the labels, and their views."""
    mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
    mask[:, 60:260] = True
    frame = road_frame(mask)
    assert frame is not None
    vehicles = np.zeros((HEIGHT, WIDTH), dtype=bool)
    scene = np.full((HEIGHT, WIDTH, 3), 0.1, dtype=np.float32)
    vehicles[70:190, 150:176] = True
    scene[70:190, 150:176] = (0.3, 0.32, 0.35)
    views = []
    for top in (70, 110, 150):
        box = (150, top, 26, 40)
        views.append(_view(box, 0.4)._replace(mask=np.ones((40, 26), dtype=bool), kind="car"))
    return frame, vehicles, scene, views


def test_vehicles_told_apart_are_each_a_vehicle_on_the_road() -> None:
    """To a semantic mask three cars nose to tail are one vehicle four cars
    long. Told apart, they are three: each with its own foot, its own lamps
    and its own wake, and none of them taken for several."""
    frame, vehicles, scene, views = _column_of_cars()
    merged = road_vehicles(frame, vehicles, np.random.default_rng(0), scene)
    assert len(merged) == 1
    apart = road_vehicles(frame, vehicles, np.random.default_rng(0), scene, views=views)
    assert sorted(vehicle.foot[0] for vehicle in apart) == [109, 149, 189]
    assert all(vehicle.alone and vehicle.away == 1.0 for vehicle in apart)
    assert all(vehicle.lamps is not None and len(vehicle.rows) == 26 * 40 for vehicle in apart)
    lit = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    for seed in range(6):
        rng = np.random.default_rng(seed)
        traffic = road_vehicles(frame, vehicles, rng, scene, views=views)
        lit = np.maximum(lit, vehicle_lamps(frame, traffic, rng, 1.0, scene)[0].max(axis=2))
    for top in (70, 110, 150):
        assert float(lit[top + 15 : top + 25, 150:176].max()) > 1.0, "a car of the column unlit"
    # Each throws its own wake towards the camera, and the first car's
    # shows either side of the second. One long vehicle throws one wake,
    # from the foot of its near end.
    rng = np.random.default_rng(1)
    traffic = road_vehicles(frame, vehicles, rng, scene, 110.0, views)
    spray = vehicle_spray(frame, vehicles, rng, scene, traffic)
    beside_the_second = float(spray[115:145, 138:149].mean() + spray[115:145, 177:188].mean())
    assert beside_the_second > 0.5, "no wake behind the first of the column"
    whole = [_view((150, 70, 26, 120), 0.4)]
    rng = np.random.default_rng(1)
    as_one = road_vehicles(frame, vehicles, rng, scene, 110.0, whole)
    merged_spray = vehicle_spray(frame, vehicles, rng, scene, as_one)
    assert len(as_one) == 1 and float(merged_spray[192:215, 152:174].mean()) > 0.2
    assert float(merged_spray[115:145, 138:149].mean()) < 0.02


def test_a_lorry_is_known_by_its_name_where_it_has_one() -> None:
    """A vehicle a network called a truck keeps to a lorry's speed however
    narrow its outline, and a wide one it called a car does not."""
    frame, vehicles, scene, boxes = _traffic_scene()
    vehicles[100:130, 140:190] = True
    scene[100:130, 140:190] = (0.3, 0.32, 0.35)
    for seed in range(8):
        views = [_view(box, 0.4)._replace(kind="car") for box in boxes]
        views[0] = views[0]._replace(kind="truck")
        views.append(_view((140, 100, 50, 30), 0.4)._replace(kind="car"))
        traffic = road_vehicles(frame, vehicles, np.random.default_rng(seed), scene, 130.0, views)
        speeds = {vehicle.foot[1]: vehicle.speed_kmh for vehicle in traffic}
        assert speeds[min(speeds)] <= 90.0, "a truck over a lorry's limit"
        wide = [vehicle for vehicle in traffic if 140 <= vehicle.foot[1] < 190]
        assert len(wide) == 1 and wide[0].alone
    assert max(vehicle.speed_kmh for vehicle in traffic) > 100.0


def test_one_lamp_found_on_a_vehicle_seen_end_on_has_its_fellow_lit_across_from_it() -> None:
    """The other of the pair is there, in shade or missed. Not on a vehicle
    seen from the side, where the one lamp is all that end shows."""
    frame, vehicles, scene, boxes = _traffic_scene()
    left, top, wide, tall = boxes[0]
    one = _view(boxes[0], 0.4)
    one = one._replace(lamps=one.lamps[:1])
    lit_columns: set[int] = set()
    for seed in range(8):
        rng = np.random.default_rng(seed)
        traffic = road_vehicles(frame, vehicles, rng, scene, views=[one])
        light, _glow, _cores = vehicle_lamps(frame, traffic, rng, 1.0, scene)
        car = light[top : top + tall, left : left + wide].max(axis=2)
        if float(car.max()) > 0.0:
            lit_columns |= {int(column) for column in np.nonzero(car > 0.5 * car.max())[1]}
    assert lit_columns & set(range(4, 10)) and lit_columns & set(range(16, 22))

    side = np.zeros((HEIGHT, WIDTH), dtype=bool)
    side[100:120, 120:190] = True
    box = (120, 100, 70, 20)
    lamp = np.array([[128.0, 112.0, 2.0, 1.5, 0.4]], dtype=np.float32)
    seen = VehicleView(box, 0.0, 0.4, lamp)
    for seed in range(8):
        rng = np.random.default_rng(seed)
        traffic = road_vehicles(frame, side, rng, None, views=[seen])
        light, _glow, _cores = vehicle_lamps(frame, traffic, rng, 1.0, None)
        assert float(light[:, 160:].max()) == 0.0, "a second lamp on a vehicle seen from the side"


def test_a_wake_ends_where_its_road_does() -> None:
    """A road that turns out of the frame half way down it: the wake of a
    car driving away up it runs back to where the road ends and no further.
    It used to run on, across the verge."""
    mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
    mask[:150, 60:260] = True
    frame = road_frame(mask)
    assert frame is not None
    vehicles = np.zeros((HEIGHT, WIDTH), dtype=bool)
    vehicles[90:120, 147:173] = True
    thrown = 0.0
    for seed in range(8):
        rng = np.random.default_rng(seed)
        traffic = road_vehicles(frame, vehicles, rng, views=[_view((147, 90, 26, 30), 0.4)])
        spray = vehicle_spray(frame, vehicles, rng, traffic=traffic)
        thrown += float(spray[121:150].sum())
        assert float(spray[185:].max()) < 0.02 * float(spray.max())
    assert thrown > 0.0


def test_vehicles_a_network_read_can_be_lit_with_no_road_under_them() -> None:
    """A junction seen from above, a car park: no road is read, and a
    vehicle still shows which end it shows. Those the network read and
    leans on are taken at their word; the rest are left out."""
    _frame, vehicles, scene, boxes = _traffic_scene()
    views = [
        _view(boxes[0], 0.4),
        _view(boxes[1], -0.4),
        _view(boxes[2], 0.01),  # leans neither way
        VehicleView(boxes[3], 0.0, 0.0, np.zeros((0, 5), np.float32)),  # not read
    ]
    seen = vehicles_as_seen(vehicles, views)
    assert sorted((vehicle.foot[1], vehicle.away) for vehicle in seen) == [(92, 1.0), (112, -1.0)]
    assert all(vehicle.road == 0 and vehicle.lamps is not None for vehicle in seen)
    assert vehicles_as_seen(vehicles, None) == []
    rng = np.random.default_rng(0)
    light, _glow, cores = vehicle_lamps(
        None, seen, rng, 1.0, scene, shape=vehicles.shape, lit_share=(1.0, 1.0)
    )
    for (left, top, wide, tall), lit in zip(boxes, (True, True, False, False), strict=True):
        assert (float(light[top : top + tall, left : left + wide].max()) > 1.0) is lit
    assert float(cores.max()) > 0.0
    with pytest.raises(ValueError, match="frame or the picture's shape"):
        vehicle_lamps(None, seen, rng, 1.0)


def test_lamps_a_night_picture_already_shows_burning_are_not_lit_again() -> None:
    """A photograph taken at night has its traffic's lamps in it: a few
    pixels at full scale on a dark car. That car is left as it is; one
    beside it with its lamps off is lit, and at night every one is."""
    _frame, vehicles, _scene_, boxes = _traffic_scene()
    night = np.full((HEIGHT, WIDTH, 3), 0.01, dtype=np.float32)
    left, top, _wide, _tall = boxes[0]
    night[top + 14 : top + 16, left + 5 : left + 8] = 1.0
    night[top + 14 : top + 16, left + 18 : left + 21] = 1.0
    seen = vehicles_as_seen(vehicles, [_view(box, 0.4) for box in boxes])
    for seed in range(6):
        rng = np.random.default_rng(seed)
        light, _glow, _cores = vehicle_lamps(None, seen, rng, 1.0, night, shape=vehicles.shape)
        for index, (left, top, wide, tall) in enumerate(boxes):
            lit = float(light[top : top + tall, left : left + wide].max()) > 1.0
            assert lit is (index != 0), "a burning lamp lit again, or a dark car left dark at night"


def _night_scene() -> tuple[np.ndarray, np.ndarray]:
    """The two-car scene in daylight, its cars dark grey, and its labels."""
    image, labels = _scene()
    image[labels == int(SceneClass.VEHICLE)] = (50, 52, 55)
    return image, labels


def test_infrared_night_lights_the_lamps_a_reader_found_and_only_then() -> None:
    """A daylight photograph has its traffic's lamps off, and by night they
    are the brightest things in the frame. With a vehicle reader on the
    provider they are lit where it found them, white whichever end they are
    on; without one the frame is what it always was."""
    image, labels = _night_scene()

    def infrared(provider: SharedDepthProvider, **settings: object) -> tuple[np.ndarray, dict]:
        """The scene as the night camera sees it, and what the transform drew."""
        night = NightIrMode(depth_provider=provider, p=1.0, **settings)  # type: ignore[arg-type]
        night.set_random_seed(3)
        with recording() as log:
            out = night(image=image)["image"].astype(np.float32)
        return out, log[0]["drew"]

    plain, drew = infrared(SharedDepthProvider(_Level(), labels=_Labels(labels)))
    assert drew["vehicles_read"] == 0
    for facing in (1.0, -1.0):
        reader = SharedDepthProvider(_Level(), labels=_Labels(labels), vehicles=_Told(facing))
        lit, drew = infrared(reader)
        assert drew["vehicles_read"] == 2 and drew["vehicle_lamps"] > 0.0
        assert 1 <= drew["vehicles_lit"] <= 2
        change = lit - plain
        on_cars = labels == int(SceneClass.VEHICLE)
        assert float(change[on_cars].max()) > 100.0, "no lamp lit on a dark car"
        # Half way down each car, where the reader put them.
        # The level depth of this scene has no cars standing in it, so a
        # beam runs on over the car ahead: the lamps are the brightest of
        # what changed, and are looked for at that.
        on_a_car = on_cars & (change.max(axis=2) > 0.9 * change[on_cars].max())
        rows, columns = np.nonzero(on_a_car)
        assert {int(row) for row in rows} <= set(range(96, 114)) | set(range(156, 174))
        # White, whichever end: the three channels differ only by the
        # sensor's residual tint.
        brightest = lit[rows, columns]
        assert float((brightest.max(axis=1) - brightest.min(axis=1)).max()) < 20.0
        off, drew = infrared(reader, vehicle_lamp_range=(0.0, 0.0))
        assert drew["vehicles_read"] == 0 and float(np.abs(off - plain).max()) == 0.0


def test_rain_lights_a_reader_s_lamps_where_no_road_was_read() -> None:
    """With the ground left dry no road is read, and the traffic's lamps are
    lit all the same from what the reader saw. Without a reader there is
    nothing to light them by."""
    image, labels = _scene()
    settings: dict[str, object] = {
        "max_streaks": 0,
        "wetness_probability": 0.0,
        "rain_rate_range_mm_h": (2.0, 2.0),
        "vehicle_lamp_range": (1.0, 1.0),
    }
    lit = {}
    for name, extra in (("reader", {"vehicles": _Told(-1.0)}), ("none", {})):
        provider = SharedDepthProvider(_Level(), labels=_Labels(labels), **extra)  # type: ignore[arg-type]
        rain = RandomRain(depth_provider=provider, p=1.0, **settings)  # type: ignore[arg-type]
        count = 0
        for seed in range(6):
            rain.set_random_seed(seed)
            with recording() as log:
                rain(image=image)
            count += bool(log[0]["drew"]["lamps_lit"])
        lit[name] = count
    assert lit["reader"] >= 4 and lit["none"] == 0


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
    # The labels run this traffic into a few long outlines. The network
    # tells its vehicles apart: three cars nose to tail in the left lane
    # are three, each its own pixels and each read.
    pieces = cv2.connectedComponents(vehicles.astype(np.uint8))[0] - 1
    assert len(views) > 2 * pieces
    assert sum(view.kind == "car" for view in views) > 20

    def holding(column: int, row: int) -> int:
        """Which view's own pixels a point of the frame is among."""
        for index, view in enumerate(views):
            left, top, wide, tall = view.box
            inside = left <= column < left + wide and top <= row < top + tall
            if inside and view.mask is not None and view.mask[row - top, column - left]:
                return index
        raise AssertionError(f"no vehicle at {column},{row}")

    column_of_three = {holding(225, 345), holding(228, 378), holding(242, 403)}
    assert len(column_of_three) == 3
    assert all(views[index].facing < -0.1 and len(views[index].lamps) for index in column_of_three)
    total = np.zeros(vehicles.shape, dtype=np.int32)
    for view in views:
        left, top, wide, tall = view.box
        assert view.mask is not None and view.mask.shape == (tall, wide)
        total[top : top + tall, left : left + wide] += view.mask
    assert int(total.max()) == 1 and not (total > vehicles).any()
    # A float frame is the same frame.
    as_float = OpenVocabularyVehicleEstimator().estimate(image.astype(np.float32) / 255.0, vehicles)
    assert [view.box for view in as_float] == [view.box for view in views]
