"""FoliageIntrusion: a branch in front of the lens, sized, blurred and lit by where it hangs."""

from __future__ import annotations

import numpy as np

from weathermentations.lens import FoliageIntrusion

SHAPE = (240, 320)


def _params(seed: int = 0, **settings: object) -> dict:
    """The foliage one seed draws, with one still branch unless asked otherwise."""
    arguments: dict[str, object] = {
        "branch_count_range": (1, 1),
        "sway_speed_range_m_s": (0.0, 0.0),
        "reach_range": (0.6, 0.6),
        **settings,
    }
    transform = FoliageIntrusion(p=1.0, **arguments)
    transform.set_random_seed(seed)
    return transform.get_params_dependent_on_data({"shape": (*SHAPE, 3)}, {})


def _covered(**settings: object) -> float:
    """Share of the frame foliage hides, averaged over a few branches."""
    return float(np.mean([(_params(seed, **settings)["cover"] > 0.5).mean() for seed in range(8)]))


def _soft(**settings: object) -> float:
    """Share of the covered pixels that are only partly covered: the width of the edges."""
    shares = []
    for seed in range(8):
        cover = _params(seed, **settings)["cover"]
        shares.append(((cover > 0.1) & (cover < 0.9)).sum() / max((cover > 0.1).sum(), 1))
    return float(np.mean(shares))


def _scene() -> np.ndarray:
    """A bright daylit frame."""
    rng = np.random.default_rng(0)
    return rng.integers(150, 230, (*SHAPE, 3), dtype=np.uint8)


def test_a_nearer_branch_has_larger_leaves_and_hides_more() -> None:
    near = _covered(distance_range_m=(0.25, 0.25), leaf_length_range_mm=(60.0, 60.0))
    far = _covered(distance_range_m=(1.0, 1.0), leaf_length_range_mm=(60.0, 60.0))
    # Four times the leaf, on a branch of the same length that holds fewer of them.
    assert near > 1.3 * far > 0.0


def test_a_wider_stop_blurs_the_leaves_more() -> None:
    fixed = {"distance_range_m": (0.3, 0.3), "leaf_length_range_mm": (60.0, 60.0)}
    stopped = _soft(aperture_diameter_range_mm=(0.5, 0.5), **fixed)
    open_wide = _soft(aperture_diameter_range_mm=(4.0, 4.0), **fixed)
    assert open_wide > 1.5 * stopped


def test_wind_smears_the_branch_over_the_exposure() -> None:
    fixed = {"distance_range_m": (0.4, 0.4), "aperture_diameter_range_mm": (0.5, 0.5)}
    still = _soft(**fixed)
    short = _soft(sway_speed_range_m_s=(0.4, 0.4), exposure_time_range=(0.0005, 0.0005), **fixed)
    long = _soft(sway_speed_range_m_s=(0.4, 0.4), exposure_time_range=(0.04, 0.04), **fixed)
    assert abs(short - still) < 0.02
    assert long > 1.3 * still


def test_by_day_leaves_are_dark_and_green_against_the_scene() -> None:
    image = _scene()
    transform = FoliageIntrusion(p=1.0, distance_range_m=(0.3, 0.3), reach_range=(0.8, 0.8))
    transform.set_random_seed(1)
    params = transform.get_params_dependent_on_data({"shape": image.shape}, {})
    out = transform.apply(image, **params).astype(np.float32)
    leaf = params["cover"] > 0.95
    assert leaf.sum() > 200
    assert out[leaf].mean() < 0.7 * image[leaf].mean()
    assert out[leaf][:, 1].mean() > out[leaf][:, 0].mean()
    assert out[leaf][:, 1].mean() > out[leaf][:, 2].mean()
    clear = params["cover"] < 1e-4
    assert np.array_equal(out[clear], image[clear].astype(np.float32))


def test_under_the_lamp_leaves_are_bright_by_their_own_distance() -> None:
    night = np.full((*SHAPE, 3), 15, dtype=np.uint8)

    def brightness(distance: float) -> float:
        """Mean level of the leaves of a branch at a distance, under the ring."""
        transform = FoliageIntrusion(
            p=1.0,
            distance_range_m=(distance, distance),
            leaf_light_range=(0.0, 0.0),
            illuminator_gain_range=(0.2, 0.2),
            reach_range=(0.8, 0.8),
        )
        transform.set_random_seed(1)
        params = transform.get_params_dependent_on_data({"shape": night.shape}, {})
        out = transform.apply(night, **params)
        return float(out[params["cover"] > 0.95].mean())

    assert brightness(0.25) == 255.0
    assert 15.0 < brightness(1.2) < 0.7 * brightness(0.25)


def test_no_branches_leave_the_frame_alone() -> None:
    image = _scene()
    transform = FoliageIntrusion(p=1.0, branch_count_range=(0, 0))
    transform.set_random_seed(0)
    assert np.array_equal(transform(image=image)["image"], image)


def test_cover_and_colour_stay_within_what_can_be() -> None:
    for seed in range(4):
        params = _params(seed, branch_count_range=(3, 3))
        assert 0.0 <= params["cover"].min() and params["cover"].max() <= 1.0
        # Premultiplied: a pixel's colour is at most its cover times the
        # brightest a leaf can be.
        assert (params["reflectance"].max(axis=2) <= params["cover"] * 0.2 + 1e-4).all()
