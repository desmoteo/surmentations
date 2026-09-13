import math

import cv2
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


def test_dirt_mask_in_unit_range(image: np.ndarray) -> None:
    t = DirtSoil(p=1.0, smudge_count_range=(3, 6))
    for _ in range(20):
        out_params = t.get_params_dependent_on_data({"shape": image.shape}, {})
        mask = out_params["dirt_mask"]
        assert mask.shape == image.shape[:2]
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0


def test_dirt_is_far_more_visible_against_a_bright_source(image: np.ndarray) -> None:
    """Grime scatters light rather than tinting it, so it should be close to
    invisible in flat light and obvious when something bright is in frame."""
    flat = np.full((128, 128, 3), 90, dtype=np.uint8)
    backlit = flat.copy()
    cv2.circle(backlit, (64, 30), 18, (255, 255, 255), -1)

    t = DirtSoil(p=1.0)
    t.set_random_seed(0)
    params = t.get_params_dependent_on_data({"shape": flat.shape}, {})

    flat_delta = np.abs(
        t.apply(flat, **params).astype(np.float32) - flat.astype(np.float32)
    ).mean()
    backlit_delta = np.abs(
        t.apply(backlit, **params).astype(np.float32) - backlit.astype(np.float32)
    ).mean()
    assert backlit_delta > flat_delta * 1.5


def test_water_droplets_refraction_stays_in_bounds(image: np.ndarray) -> None:
    # cv2.remap with BORDER_REFLECT never reads out of bounds; this test
    # documents/guards that no exception is raised even at extreme cap ratios.
    t = WaterDroplets(
        p=1.0,
        droplet_count_range=(20, 20),
        droplet_radius_range=(0.01, 0.12),
        cap_height_ratio_range=(0.9, 1.0),
        streak_probability=0.5,
    )
    out = t(image=image)["image"]
    assert out.shape == image.shape


def test_water_droplets_never_overlap(image: np.ndarray) -> None:
    """Physical invariant: sessile droplets don't merge in this model, so no
    two placed droplets' footprints may touch or overlap."""
    t = WaterDroplets(
        p=1.0,
        droplet_count_range=(40, 40),
        droplet_radius_range=(0.006, 0.02),
        streak_probability=0.4,
        streak_length_range=(1.5, 4.0),
    )
    for seed in range(10):
        t.set_random_seed(seed)
        params = t.get_params_dependent_on_data({"shape": image.shape}, {})
        droplets = params["droplets"]
        for i in range(len(droplets)):
            cx1, cy1, r1, _cap1, s1, _irr1, _ph1 = droplets[i]
            footprint1 = r1 * max(1.0, s1)
            for j in range(i + 1, len(droplets)):
                cx2, cy2, r2, _cap2, s2, _irr2, _ph2 = droplets[j]
                footprint2 = r2 * max(1.0, s2)
                dist = math.hypot(cx1 - cx2, cy1 - cy2)
                assert dist >= footprint1 + footprint2 - 1e-6, (
                    f"seed {seed}: droplets {i} and {j} overlap "
                    f"(dist={dist:.2f}, footprints={footprint1:.2f}+{footprint2:.2f})"
                )


def test_water_droplets_degrade_gracefully_when_canvas_too_small() -> None:
    """Requesting more droplets than can fit without overlap should place
    fewer, not raise or hang."""
    t = WaterDroplets(p=1.0, droplet_count_range=(60, 60), droplet_radius_range=(0.2, 0.25))
    small_image = np.zeros((100, 100, 3), dtype=np.uint8)
    params = t.get_params_dependent_on_data({"shape": small_image.shape}, {})
    assert 0 < len(params["droplets"]) < 60


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


def test_condensation_stain_edges_are_soft_but_wipe_edges_are_sharp() -> None:
    """Two different physical boundaries, two different edge profiles.

    A stain's border is a nucleation gradient — droplet density thins out
    until the glass reads clear — so it must fade over a wide band. A wiped
    track is the opposite: a sliding drop picks the film up mechanically, so
    its contact line leaves a crisp boundary. Getting these the wrong way
    round (soft wipes, hard stains) is what makes condensation read as
    pasted-on blobs.
    """
    image = np.zeros((400, 400, 3), dtype=np.uint8)
    t = WaterDroplets(
        p=1.0,
        droplet_count_range=(18, 18),
        condensation_coverage_range=(0.6, 0.6),
        trail_probability=1.0,
        trail_length_range=(0.4, 0.6),
    )
    t.set_random_seed(3)
    field = t.get_params_dependent_on_data({"shape": image.shape}, {})["condensation"]

    gradient = np.abs(cv2.Sobel(field, cv2.CV_32F, 1, 0, ksize=3))
    # Wipe tracks are vertical strokes, so they are the only source of steep
    # horizontal gradients; stain borders contribute the shallow ones.
    steep = np.percentile(gradient[gradient > 1e-4], 99.5)
    typical = np.percentile(gradient[gradient > 1e-4], 50)
    assert steep > 8.0 * typical, (
        f"wipe edges ({steep:.3f}) should be far sharper than stain borders ({typical:.3f})"
    )
