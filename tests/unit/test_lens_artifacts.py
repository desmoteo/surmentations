import numpy as np
import pytest

from weathermentations.lens.dirt import DirtSoil
from weathermentations.lens.sun_flare import SunFlare
from weathermentations.lens.water_droplets import WaterDroplets


@pytest.fixture
def image() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(128, 128, 3), dtype=np.uint8)


@pytest.mark.parametrize(
    "transform",
    [
        DirtSoil(p=1.0),
        WaterDroplets(p=1.0),
        SunFlare(p=1.0),
    ],
    ids=["dirt_soil", "water_droplets", "sun_flare"],
)
def test_output_shape_dtype_and_range(transform, image: np.ndarray) -> None:
    out = transform(image=image)["image"]
    assert out.shape == image.shape
    assert out.dtype == image.dtype
    assert out.min() >= 0
    assert out.max() <= 255


@pytest.mark.parametrize(
    "transform",
    [DirtSoil(p=1.0), WaterDroplets(p=1.0), SunFlare(p=1.0)],
    ids=["dirt_soil", "water_droplets", "sun_flare"],
)
def test_p_zero_is_identity(transform, image: np.ndarray) -> None:
    transform.p = 0.0
    out = transform(image=image)["image"]
    np.testing.assert_array_equal(out, image)


def test_dirt_soil_alpha_mask_in_unit_range(image: np.ndarray) -> None:
    t = DirtSoil(p=1.0, blob_count_range=(3, 6))
    for _ in range(20):
        out_params = t.get_params_dependent_on_data({"shape": image.shape}, {})
        mask = out_params["alpha_mask"]
        assert mask.shape == image.shape[:2]
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0


def test_water_droplets_refraction_stays_in_bounds(image: np.ndarray) -> None:
    # cv2.remap with BORDER_REFLECT never reads out of bounds; this test
    # documents/guards that no exception is raised even at extreme cap ratios.
    t = WaterDroplets(
        p=1.0,
        droplet_count_range=(20, 20),
        droplet_radius_range=(5.0, 60.0),
        cap_height_ratio_range=(0.9, 1.0),
        streak_probability=0.5,
    )
    out = t(image=image)["image"]
    assert out.shape == image.shape


def test_sun_flare_ghosts_lie_on_source_center_line(image: np.ndarray) -> None:
    t = SunFlare(p=1.0, flare_position=(0.9, 0.1), ghost_count_range=(5, 5))
    params = t.get_params_dependent_on_data({"shape": image.shape}, {})
    source = np.array(params["source"])
    center = np.array(params["center"])
    direction = center - source
    direction /= np.linalg.norm(direction)

    for gx, gy, _gsize, _hue in params["ghosts"]:
        ghost = np.array([gx, gy])
        to_ghost = ghost - source
        # Cross product magnitude ~0 => collinear with the source-center line.
        cross = direction[0] * to_ghost[1] - direction[1] * to_ghost[0]
        assert abs(cross) < 1e-3


def test_sun_flare_intensity_is_monotonic(image: np.ndarray) -> None:
    dim_image = SunFlare(p=1.0, flare_position=(0.5, 0.5), intensity_range=(0.2, 0.2))(
        image=image
    )["image"]
    bright_image = SunFlare(p=1.0, flare_position=(0.5, 0.5), intensity_range=(2.0, 2.0))(
        image=image
    )["image"]
    assert bright_image.astype(np.float32).mean() >= dim_image.astype(np.float32).mean()
