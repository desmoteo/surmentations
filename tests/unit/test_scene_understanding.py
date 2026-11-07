"""Labels and geometry held by the provider, and what the transforms do with them.

``SharedDepthProvider`` shares more than depth: what each pixel of the scene
is, and how the camera is mounted over it. These tests pin the three things
that make that worth having. Each is worked out once per scene, from the
frame the depth was, and shared along a pipeline. A transform given them
does better on exactly the cases depth alone gets wrong. And a transform
given neither behaves as it did before there was anything to give.
"""

from __future__ import annotations

import math
import pickle

import albumentations as A
import cv2
import numpy as np
import pytest

from surmentations.atmosphere import GroundFog, RandomFog, RandomRain, RandomSnow
from surmentations.atmosphere.rain import _read_wet_ground
from surmentations.base import scene_fov
from surmentations.camera import NightIrMode
from surmentations.camera.night_ir import near_infrared_reflectance
from surmentations.depth.estimator import DepthEstimator
from surmentations.depth.provider import SharedDepthProvider
from surmentations.diagnostics import recording
from surmentations.perception import (
    Camera,
    GeometryEstimator,
    GroundPlaneGeometryEstimator,
    InstallationGeometry,
    SceneClass,
    SceneLabelEstimator,
)
from surmentations.utils.photometry import srgb_to_linear
from surmentations.utils.sky import sky_weight
from surmentations.utils.sources import lamp_mask

WIDTH, HEIGHT = 240, 180
CAMERA = Camera(WIDTH, HEIGHT, 60.0, 12.0, 0.0, 6.0)
KNOWN = InstallationGeometry(horizontal_fov_deg=60.0, pitch_deg=12.0, height_m=6.0)


def _world(camera: Camera = CAMERA) -> tuple[np.ndarray, np.ndarray]:
    """Depth in metres of level ground under ``camera``, and which pixels are ground."""
    rows, columns = np.mgrid[0 : camera.height, 0 : camera.width]
    assert camera.height_m is not None
    depth = camera.height_m * camera.ground_depth_per_metre(rows + 0.5, columns + 0.5)
    return depth, np.isfinite(depth)


class _World(DepthEstimator):
    """A depth backend that knows the scene: level ground under ``CAMERA``, sky above.

    Its relative depth is the scene's, squeezed into ``[0, 1]`` as a
    backend's is; its disparity is the reciprocal of distance up to a scale
    and an offset it does not tell.
    """

    def __init__(self, camera: Camera = CAMERA) -> None:
        """The scene under ``camera``, and a count of the estimates asked for."""
        self.depth, self.ground = _world(camera)
        self.calls = 0

    def estimate(self, image: np.ndarray) -> np.ndarray:
        """The scene's depth squeezed into ``[0, 1]``, with the sky at 1."""
        self.calls += 1
        finite = np.where(self.ground, self.depth, 0.0)
        near, far = float(finite[self.ground].min()), 80.0
        relative = np.clip((finite - near) / (far - near), 0.0, 1.0)
        return np.where(self.ground, relative, 1.0).astype(np.float32)

    def disparity(self, relative_depth: np.ndarray) -> np.ndarray | None:
        """The reciprocal of the scene's distance, scaled and offset; the sky at the offset."""
        nearness = np.where(self.ground, 1.0 / np.where(self.ground, self.depth, 1.0), 0.0)
        return (0.6 * nearness + 0.08).astype(np.float32)


class _Named(SceneLabelEstimator):
    """A label estimator that returns one map, and remembers what it was shown."""

    def __init__(self, labels: np.ndarray) -> None:
        """Keep the map to return, and a list of the frames shown."""
        self.labels = labels
        self.seen: list[np.ndarray] = []

    def estimate(self, image: np.ndarray) -> np.ndarray:
        """The one map, whatever the frame; the frame is kept."""
        self.seen.append(image.copy())
        return self.labels


def _labels(**regions: tuple[slice, slice]) -> np.ndarray:
    """Road under ``CAMERA``'s horizon and sky above it, with named regions painted in."""
    _depth, ground = _world()
    labels = np.where(ground, int(SceneClass.ROAD), int(SceneClass.SKY)).astype(np.uint8)
    for name, region in regions.items():
        labels[region] = int(SceneClass[name])
    return labels


def _image(seed: int = 0) -> np.ndarray:
    """A grey road under a pale sky, with a little grain."""
    rng = np.random.default_rng(seed)
    _depth, ground = _world()
    image = np.where(ground[..., None], 118.0, 205.0) + rng.normal(0.0, 2.0, (HEIGHT, WIDTH, 3))
    return np.clip(image, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------- held with the depth


def test_a_provider_with_nothing_to_say_says_nothing() -> None:
    provider = SharedDepthProvider(_World())
    image = _image()
    assert provider.get_labels(image) is None
    assert provider.get_geometry(image) is None
    assert provider.settle_camera(image, 50.0, 3.0) is None
    assert provider.settle_view(image, 50.0, 3.0) == (50.0, 3.0)
    assert provider.settle_fov(image, 50.0) == 50.0
    assert provider.horizon_row(image) is None
    told = provider.describe(image)
    assert told["geometry"] is None and told["metric"] is None and told["label_shares"] == {}


def test_labels_are_made_once_from_the_frame_the_depth_was() -> None:
    """With the depth, and not when first asked for: a rain transform at the
    end of a pipeline is told about the scene, not about the fog over it."""
    named = _Named(_labels())
    world = _World()
    provider = SharedDepthProvider(world, labels=named)
    image = _image()
    provider.get_depth(image)
    assert len(named.seen) == 1 and np.array_equal(named.seen[0], image)

    fogged = np.full_like(image, 200)
    provider.carry(image, fogged)
    labels = provider.get_labels(fogged)
    assert labels is not None and np.array_equal(labels, named.labels)
    assert len(named.seen) == 1 and world.calls == 1
    assert not labels.flags.writeable, "the map is shared, so it is read-only"
    shares = provider.describe(image)["label_shares"]
    assert set(shares) == {"ROAD", "SKY"} and sum(shares.values()) == pytest.approx(1.0)


def test_asking_after_labels_or_the_camera_is_not_a_request_for_depth() -> None:
    """`cache_info` counts depth requests answered from what is held, which is
    how sharing is checked. It must not move with the rest."""
    provider = SharedDepthProvider(_World(), labels=_Named(_labels()), geometry=KNOWN)
    image = _image()
    provider.get_labels(image)
    provider.settle_camera(image)
    provider.horizon_row(image)
    assert provider.cache_info()[:2] == (0, 1)
    provider.get_depth(image)
    assert provider.cache_info()[:2] == (0, 1), "the estimate that was made was for this"
    provider.get_depth(image)
    assert provider.cache_info()[:2] == (1, 1)


def test_wrong_kinds_of_labels_and_geometry_are_refused_at_once() -> None:
    with pytest.raises(TypeError, match="labels"):
        SharedDepthProvider(_World(), labels=_labels())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="geometry"):
        SharedDepthProvider(_World(), geometry=(60.0, 12.0))  # type: ignore[arg-type]
    wrong_shape = SharedDepthProvider(_World(), labels=_Named(np.zeros((4, 4), dtype=np.uint8)))
    with pytest.raises(ValueError, match="_Named"):
        wrong_shape.get_depth(_image())


def test_a_provider_pickles_with_its_estimators_and_nothing_cached() -> None:
    provider = SharedDepthProvider(_World(), labels=_Named(_labels()), geometry=KNOWN)
    image = _image()
    provider.get_depth(image)
    # A round trip of the object built two lines up, as a DataLoader makes of it.
    restored = pickle.loads(pickle.dumps(provider))
    assert restored.cache_info().scenes == 0
    assert restored.get_geometry(image) == KNOWN
    assert np.array_equal(restored.get_labels(image), _labels())


def test_what_is_already_known_of_a_frame_can_be_told_to_the_provider() -> None:
    """Ground-truth masks, or the camera a frame came from in a dataset that
    mixes cameras: used in place of the estimators, which are not run."""
    named = _Named(_labels())
    provider = SharedDepthProvider(_World(), labels=named)
    image = _image()
    truth = _labels(VEHICLE=(slice(120, 140), slice(60, 100)))
    provider.remember(image, labels=truth, geometry=KNOWN)
    assert named.seen == []
    assert np.array_equal(provider.get_labels(image), truth)
    assert provider.get_geometry(image) == KNOWN

    # Told again, after the scene is held: the new word replaces the old.
    provider.remember(image, labels=_labels())
    assert np.array_equal(provider.get_labels(image), _labels())
    with pytest.raises(ValueError, match="remember"):
        provider.remember(image, labels=np.zeros((3, 3), dtype=np.uint8))


def test_a_geometry_told_for_one_frame_is_filled_in_by_the_estimator() -> None:
    class _Fixed(GeometryEstimator):
        """An estimator that reads the same geometry off every frame."""

        def estimate(self, image, disparity, labels):  # type: ignore[no-untyped-def]
            """A lens, a horizon and a height, whatever it is shown."""
            return InstallationGeometry(horizontal_fov_deg=77.0, horizon=0.3, height_m=9.0)

    provider = SharedDepthProvider(_World(), geometry=_Fixed())
    image = _image()
    provider.remember(image, geometry=InstallationGeometry(height_m=4.0))
    geometry = provider.get_geometry(image)
    assert geometry is not None
    assert (geometry.height_m, geometry.horizontal_fov_deg, geometry.horizon) == (4.0, 77.0, 0.3)


# ------------------------------------------------------------------------ one camera


def test_every_transform_sees_one_scene_through_one_camera() -> None:
    """Each used to draw a field of view and a pitch of its own."""
    provider = SharedDepthProvider(_World(), geometry=KNOWN)
    image = _image()
    assert provider.settle_view(image, 48.0, 3.0) == (60.0, 12.0)
    assert provider.settle_view(image, 75.0, 28.0) == (60.0, 12.0)
    assert provider.settle_fov(image, 90.0) == 60.0
    camera = provider.settle_camera(image)
    assert camera == CAMERA
    assert provider.horizon_row(image) == pytest.approx(CAMERA.horizon_row)


def test_what_a_geometry_leaves_open_the_first_draw_settles_for_the_scene() -> None:
    provider = SharedDepthProvider(_World(), geometry=InstallationGeometry(height_m=6.0))
    image = _image()
    assert provider.settle_camera(image) is None, "no lens known and none proposed yet"
    assert provider.settle_view(image, 52.0, 9.0) == (52.0, 9.0)
    assert provider.settle_view(image, 70.0, 25.0) == (52.0, 9.0), "the second caller's draw"
    # Nothing fixes this horizon: it follows from a draw, and is not offered
    # to a wet road as a fact about the frame.
    assert provider.horizon_row(image) is None
    # Another frame is another scene, with its own first draw.
    other = _image(1)
    assert provider.settle_view(other, 70.0, 25.0) == (70.0, 25.0)


def test_a_transform_with_no_provider_is_told_the_scene_s_field_of_view() -> None:
    """Rain lit by an infrared lamp takes no depth provider. It follows a
    transform that does, and sees the frame through the same lens."""
    provider = SharedDepthProvider(_World(), geometry=KNOWN)
    image = _image()
    assert scene_fov({"image": image}, 48.0) == 48.0, "no provider holds this frame yet"
    provider.get_depth(image)
    assert scene_fov({"image": image}, 48.0) == 60.0
    derived = image.copy()
    provider.carry(image, derived)
    assert scene_fov({"image": derived}, 48.0) == 60.0
    assert scene_fov({"images": np.stack([derived, derived])}, 48.0) == 48.0
    assert scene_fov({}, 48.0) == 48.0, "a transform asked without a frame keeps its draw"


def test_the_estimator_s_reading_of_the_frame_becomes_the_scene_s_camera() -> None:
    estimator = GroundPlaneGeometryEstimator(horizontal_fov_deg=60.0, estimate_height=False)
    provider = SharedDepthProvider(_World(), labels=_Named(_labels()), geometry=estimator)
    image = _image()
    fov, pitch = provider.settle_view(image, 50.0, 0.0)
    assert fov == 60.0 and pitch == pytest.approx(12.0, abs=0.5)
    horizon = provider.horizon_row(image)
    assert horizon is not None and horizon == pytest.approx(CAMERA.horizon_row, abs=1.5)


# ----------------------------------------------------------------------------- metres


def test_a_camera_of_known_height_puts_the_depth_in_metres() -> None:
    """In place of the nearest and farthest distance the provider is
    otherwise told by hand, which here are deliberately wrong."""
    world = _World()
    provider = SharedDepthProvider(world, depth_range_m=(1.0, 9.0), geometry=KNOWN)
    image = _image()
    metres = provider.get_metric_depth(image)
    near = world.ground & (world.depth < 50.0)
    assert np.abs(metres[near] / world.depth[near] - 1.0).max() < 0.03
    assert provider.describe(image)["metric"] is not None

    # An explicit far plane is the caller's own calibration, and is honoured.
    relative = provider.get_depth(image)
    pseudo = provider.to_metric(relative, far_m=40.0)
    assert float(pseudo.max()) == pytest.approx(40.0) and float(pseudo.min()) == pytest.approx(1.0)
    # And a depth map that is not this provider's is not a scene it can measure.
    assert float(provider.to_metric(relative.copy()).max()) == pytest.approx(9.0)


def test_without_a_height_the_depth_is_what_it_always_was() -> None:
    plain = SharedDepthProvider(_World(), depth_range_m=(2.0, 30.0))
    no_height = SharedDepthProvider(
        _World(),
        depth_range_m=(2.0, 30.0),
        geometry=InstallationGeometry(horizontal_fov_deg=60.0, pitch_deg=12.0),
    )
    image = _image()
    assert np.array_equal(plain.get_metric_depth(image), no_height.get_metric_depth(image))
    assert no_height.describe(image)["metric"] is None


def test_a_scale_waits_for_a_lens_and_is_found_once_there_is_one() -> None:
    """Asked for metres before any field of view is settled, the provider
    answers in pseudo-metres and does not write the scene off."""
    world = _World()
    provider = SharedDepthProvider(
        world, geometry=InstallationGeometry(pitch_deg=12.0, height_m=6.0)
    )
    image = _image()
    relative = provider.get_depth(image)
    assert float(provider.to_metric(relative).max()) == pytest.approx(50.0)
    provider.settle_fov(image, 60.0)
    metres = provider.to_metric(relative)
    near = world.ground & (world.depth < 50.0)
    assert np.abs(metres[near] / world.depth[near] - 1.0).max() < 0.03


# ------------------------------------------------------------------- the transforms


def _linear(image: np.ndarray) -> np.ndarray:
    """An 8-bit frame in linear light, as the transforms read it."""
    return srgb_to_linear(image)


def test_a_labelled_vehicle_stays_dry_where_nothing_else_could_tell_it_from_road() -> None:
    """Flat in the depth map and the colour of the road: neither the bump
    nor the look gives it away. The network's name for it does."""
    depth, _ground = _world()
    depth = np.where(np.isfinite(depth), depth, 400.0).astype(np.float32)
    scene = _linear(_image())
    car = (slice(120, 150), slice(80, 140))
    without, *_ = _read_wet_ground(depth, scene, 60.0, 12.0, 6.0)
    with_labels, *_ = _read_wet_ground(depth, scene, 60.0, 12.0, 6.0, _labels(VEHICLE=car))
    assert float(without[car].mean()) > 0.9, "the fixture's car is not hidden in the road"
    assert float(with_labels[130:140, 95:125].mean()) < 0.05
    assert float(with_labels[160:, :60].mean()) > 0.9, "the road beside it is still wet"


def test_a_labelled_lawn_holds_no_film_whatever_colour_it_is() -> None:
    depth, _ground = _world()
    depth = np.where(np.isfinite(depth), depth, 400.0).astype(np.float32)
    lawn = (slice(110, 180), slice(0, 70))
    wet, *_ = _read_wet_ground(depth, _linear(_image()), 60.0, 12.0, 6.0, _labels(GRASS=lawn))
    assert float(wet[125:170, 5:55].mean()) < 0.05
    assert float(wet[125:170, 120:].mean()) > 0.9


def test_a_labelled_road_is_wet_where_its_normals_wander() -> None:
    """A camber, a rut, the depth map's own unevenness: the surface normal
    there says "not level", and the road used to be wet in patches."""
    depth, _ground = _world()
    depth = np.where(np.isfinite(depth), depth, 400.0).astype(np.float32)
    rows, columns = np.mgrid[0:HEIGHT, 0:WIDTH]
    patch = (slice(100, 170), slice(60, 180))
    rippled = depth.copy()
    rippled[patch] *= 1.0 + 0.1 * np.sin(columns[patch] * 0.3) * np.sin(rows[patch] * 0.3)
    scene = _linear(_image())
    without, *_ = _read_wet_ground(rippled, scene, 60.0, 12.0, 6.0)
    with_labels, *_ = _read_wet_ground(rippled, scene, 60.0, 12.0, 6.0, _labels())
    inner = (slice(110, 160), slice(75, 165))
    assert float(without[inner].mean()) < 0.7, "the fixture's ripple does not disturb the normals"
    assert float(with_labels[inner].mean()) > 0.9


def test_a_road_far_off_keeps_its_sheen_when_it_is_known_to_be_a_road() -> None:
    """Level ground hundreds of metres away is taken for terrain and left
    dry, or every landscape gets a mirror along its horizon. A road that far
    off is still a road."""
    depth, ground = _world()
    far_m = np.where(ground, depth * 12.0, 4000.0).astype(np.float32)
    scene = _linear(_image())
    without, *_ = _read_wet_ground(far_m, scene, 60.0, 12.0, 6.0)
    with_labels, *_ = _read_wet_ground(far_m, scene, 60.0, 12.0, 6.0, _labels())
    band = (slice(62, 78), slice(40, 200))
    assert float(far_m[band].min()) > 420.0
    assert float(without[band].mean()) < 0.05 and float(with_labels[band].mean()) > 0.9


def test_a_wet_road_takes_its_horizon_from_the_camera_s_mounting() -> None:
    provider = SharedDepthProvider(_World(), geometry=KNOWN)
    rain = RandomRain(
        depth_provider=provider,
        p=1.0,
        wetness_probability=1.0,
        camera_pitch_range_deg=(25.0, 25.0),
        horizontal_fov_range_deg=(75.0, 75.0),
    )
    rain.set_random_seed(0)
    with recording() as log:
        rain(image=_image())
    drew = log[0]["drew"]
    assert drew["horizon_row"] == pytest.approx(CAMERA.horizon_row, abs=1.0)
    assert drew["horizontal_fov_deg"] == 60.0 and drew["camera_pitch_deg"] == 12.0


def _snowed(labels: np.ndarray | None, **kwargs: object) -> np.ndarray:
    """The frame after deep snow has settled, with or without labels."""
    extra = {} if labels is None else {"labels": _Named(labels)}
    provider = SharedDepthProvider(_World(), geometry=KNOWN, **extra)  # type: ignore[arg-type]
    snow = RandomSnow(
        depth_provider=provider,
        p=1.0,
        max_flakes=0,
        accumulation_probability=1.0,
        accumulation_strength_range=(1.0, 1.0),
        accumulation_patchiness_range=(0.0, 0.0),
        overcast_range=(0.0, 0.0),
        **kwargs,  # type: ignore[arg-type]
    )
    snow.set_random_seed(0)
    return snow(image=_image())["image"].astype(np.float32)


def test_snow_settles_on_no_one_and_on_a_car_only_as_much_as_it_is_told() -> None:
    person = (slice(120, 150), slice(40, 60))
    car = (slice(120, 150), slice(140, 200))
    plain = _snowed(None, vehicle_cover_range=(0.0, 0.0))
    named = _snowed(_labels(PERSON=person, VEHICLE=car), vehicle_cover_range=(0.0, 0.0))
    road = (slice(155, 175), slice(80, 130))
    assert plain[person].mean() > 200 and plain[car].mean() > 200, "the fixture is not snowed in"
    assert named[road].mean() > 200
    # Bare is not as the frame was: the air between is full of snow.
    assert named[128:142, 45:55].mean() < 190, "snow lies on a person"
    assert named[128:142, 155:185].mean() < 190, "a car that has just driven in is bare"
    covered = _snowed(_labels(VEHICLE=car), vehicle_cover_range=(1.0, 1.0))
    assert covered[128:142, 155:185].mean() > 230, "and one that has stood all night is not"


def test_the_sky_is_what_the_network_called_sky() -> None:
    """Where there are labels. The heuristic it replaces takes a pale wall
    that reaches the top of the frame for sky; a name does not."""
    image = np.full((HEIGHT, WIDTH, 3), 230, dtype=np.uint8)
    labels = np.full((HEIGHT, WIDTH), int(SceneClass.STRUCTURE), dtype=np.uint8)
    labels[:60, :100] = int(SceneClass.SKY)
    linear = _linear(image)
    flat = np.full((HEIGHT, WIDTH), 0.5, dtype=np.float32)
    assert float(sky_weight(linear, flat).mean()) > 0.9, "the heuristic no longer takes the wall"
    named = sky_weight(linear, flat, labels)
    assert named.dtype == np.float32 and named.shape == (HEIGHT, WIDTH)
    assert float(named[:50, :90].mean()) > 0.95 and float(named[80:, :].max()) < 0.01


def test_labelled_vegetation_is_bright_in_the_near_infrared_whatever_its_colour() -> None:
    """Chlorophyll is read off colour, and colour says "leaf" only of a green
    leaf in good light. Under a sodium lamp nothing is green."""
    linear = np.full((40, 60, 3), 0.2, dtype=np.float32)
    linear[..., 0] = 0.3
    vegetation = np.zeros((40, 60), dtype=np.float32)
    vegetation[:, 30:] = 1.0
    by_colour = near_infrared_reflectance(linear, 0.6)
    by_name = near_infrared_reflectance(linear, 0.6, vegetation)
    assert np.allclose(by_name[:, :30], by_colour[:, :30])
    assert float(by_name[:, 30:].mean()) > 1.4 * float(by_colour[:, 30:].mean())
    # A leaf in deep shade is lifted as a leaf in shade.
    dark = linear * 0.05
    shaded = near_infrared_reflectance(np.concatenate([dark, linear]), 0.6, np.ones((80, 60)))
    assert float(shaded[:40].mean()) < 0.3 * float(shaded[40:].mean())


def test_the_night_sky_is_dark_once_it_is_known_to_be_sky() -> None:
    """The ambient term is light reflected off surfaces. The sky reflects
    nothing, and in a frame the illuminator cannot reach it came out as
    bright as by day."""

    def night(labels: np.ndarray | None) -> np.ndarray:
        """The frame in infrared night mode, with or without labels, noise off."""
        extra = {} if labels is None else {"labels": _Named(labels)}
        provider = SharedDepthProvider(_World(), geometry=KNOWN, **extra)  # type: ignore[arg-type]
        mode_ = NightIrMode(depth_provider=provider, p=1.0, read_noise_range=(0.0, 0.0))
        mode_.set_random_seed(0)
        return mode_(image=_image())["image"].astype(np.float32)

    _depth, ground = _world()
    plain, named = night(None), night(_labels())
    sky = ~ground
    sky[int(CAMERA.horizon_row or 0) - 6 :] = False
    assert named[sky].mean() < 0.4 * plain[sky].mean()


def test_ground_fog_lies_under_the_camera_at_the_height_it_is_known_to_hang() -> None:
    provider = SharedDepthProvider(_World(), geometry=KNOWN)
    fog = GroundFog(depth_provider=provider, p=1.0, camera_height_range_m=(2.0, 2.0))
    fog.set_random_seed(0)
    with recording() as log:
        fog(image=_image())
    drew = log[0]["drew"]
    assert drew["camera_height_m"] == 6.0 and drew["camera_pitch_deg"] == 12.0


def test_fog_reads_real_metres_from_a_known_camera_along_a_pipeline() -> None:
    """One estimate of depth, one of labels, and every transform measuring
    from the same camera."""
    world, named = _World(), _Named(_labels())
    provider = SharedDepthProvider(world, labels=named, geometry=KNOWN)
    pipeline = A.Compose(
        [
            RandomFog(depth_provider=provider, p=1.0),
            RandomRain(depth_provider=provider, p=1.0, max_streaks=50, wetness_probability=1.0),
        ],
        seed=1,
    )
    with recording() as log:
        pipeline(image=_image())
    assert world.calls == 1 and len(named.seen) == 1
    rain = next(entry["drew"] for entry in log if entry["t"] == "RandomRain")
    assert rain["horizontal_fov_deg"] == 60.0 and rain["camera_pitch_deg"] == 12.0


# ------------------------------------------------------------------------------ lamps


def test_a_lamp_is_compact_and_a_lit_road_is_not() -> None:
    """Under a street lamp the road, the white line and the leaves are
    clipped too. Raised by a lamp's over-range factor and reflected, they
    filled a wet road's puddles with slabs of yellow."""
    linear = np.full((200, 300, 3), 0.05, dtype=np.float32)
    cv2.circle(linear, (60, 50), 6, (1.0, 1.0, 1.0), -1)  # a lamp
    linear[120:130, 40:260] = 1.0  # a lit strip of road
    cv2.line(linear, (200, 20), (290, 110), (1.0, 1.0, 1.0), 2)  # a white line
    linear[170, 30] = 1.0  # a glint
    mask = lamp_mask(linear)
    assert mask.shape == (200, 300) and mask.dtype == np.float32
    assert float(mask[50, 60]) == 1.0
    assert float(mask[125, 150]) == 0.0 and float(mask[65, 245]) == 0.0
    assert float(mask[170, 30]) == 0.0
    assert math.isclose(float(lamp_mask(np.zeros((8, 8, 3), np.float32)).max()), 0.0)
