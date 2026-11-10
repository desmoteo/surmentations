"""What the weather models infer about a scene the photograph does not state.

Sky, the colour of a veil, what is paving, and the light. Each of these was
wrong in a way that showed on real scenes, and each test names that way.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from weathermentations.atmosphere.rain import _apply_ground_wetness, _paved_weight
from weathermentations.utils.color import veil_radiance_linear
from weathermentations.utils.overcast import apply_overcast, cloud_cover
from weathermentations.utils.photometry import luminance
from weathermentations.utils.sky import lift_sky, sky_weight

H, W = 120, 160


def _outdoor(sky_colour=(0.35, 0.5, 0.8), seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Smooth sky over textured ground, with the depth a network gives it:
    sky flat at an arbitrary mid value, ground receding."""
    rng = np.random.default_rng(seed)
    image = np.empty((H, W, 3), dtype=np.float32)
    image[:50] = sky_colour
    image[50:] = 0.1 + 0.25 * rng.random((H - 50, W, 1)).astype(np.float32)
    depth = np.empty((H, W), dtype=np.float32)
    depth[:50] = 0.42
    depth[50:] = np.linspace(1.0, 0.0, H - 50, dtype=np.float32)[:, None]
    return image, depth


# ---------------------------------------------------------------------- sky


def test_sky_is_found_where_the_depth_network_puts_it_at_mid_range() -> None:
    """A far-depth threshold found sky in one test scene of nine, because the
    network does not put sky far. It puts it all at one depth."""
    image, depth = _outdoor()
    sky = sky_weight(image, depth)
    assert float(sky[:40].mean()) > 0.95
    assert float(sky[60:].mean()) < 0.02


def test_a_ceiling_is_not_sky() -> None:
    """Smooth, pale and at the top of the frame — but it recedes."""
    image, depth = _outdoor(sky_colour=(0.6, 0.6, 0.6))
    depth[:50] = np.linspace(0.1, 0.6, 50, dtype=np.float32)[:, None]
    assert float(sky_weight(image, depth).max()) < 0.05


def test_a_smooth_patch_that_does_not_reach_the_top_is_not_sky() -> None:
    image, depth = _outdoor()
    rng = np.random.default_rng(1)
    image[:12] = 0.1 + 0.25 * rng.random((12, W, 1)).astype(np.float32)
    assert float(sky_weight(image, depth).mean()) < 0.01


def test_night_sky_is_found_though_it_is_dark() -> None:
    image, depth = _outdoor(sky_colour=(0.004, 0.004, 0.012))
    image[50:] *= 0.05
    assert float(sky_weight(image, depth)[:40].mean()) > 0.9


def test_lifted_sky_is_the_farthest_thing_in_frame() -> None:
    """Left at the network's depth, the sky is nearer than the horizon under
    it and fog stops at the roofline."""
    image, depth = _outdoor()
    metres = 3.0 + 47.0 * depth
    lifted = lift_sky(metres, sky_weight(image, depth))
    assert float(lifted[:40].min()) == pytest.approx(50.0, abs=0.5)
    np.testing.assert_allclose(lifted[70:], metres[70:], atol=1e-3)


# --------------------------------------------------------------- veil colour


def test_lamps_do_not_colour_the_veil() -> None:
    """On a night street the brightest pixels are sodium lamps. Read off
    them, the veil was 17.8 times the frame's median and orange."""
    image, depth = _outdoor(sky_colour=(0.004, 0.004, 0.012))
    image[50:] *= 0.05
    for x in (30, 80, 130):
        cv2.circle(image, (x, 70), 3, (1.0, 0.6, 0.1), -1)
    veil = veil_radiance_linear(image, sky_weight(image, depth))
    assert float(luminance(veil.reshape(1, 1, 3))[0, 0]) < 0.02
    # With no sky to read, a few lamps still do not set it.
    no_sky = veil_radiance_linear(image[50:], None)
    assert float(no_sky.max()) < 0.2
    assert float(no_sky.max() - no_sky.min()) < 0.05, "an inferred veil is neutral"


def test_the_veil_under_a_blue_sky_is_nearly_white() -> None:
    """Droplets scatter all colours alike. Taking the sky's own colour made
    fog under a clear sky into blue smoke."""
    image, depth = _outdoor()
    veil = veil_radiance_linear(image, sky_weight(image, depth))
    saturation = float((veil.max() - veil.min()) / veil.max())
    assert saturation < 0.2
    assert float(luminance(veil.reshape(1, 1, 3))[0, 0]) >= 0.45


def test_a_frame_of_ground_alone_implies_a_brighter_sky() -> None:
    """A camera looking down at a road sees no sky, and is lit by one about
    five times brighter than the road looks."""
    road = np.full((H, W, 3), 0.08, dtype=np.float32)
    veil = veil_radiance_linear(road, None)
    assert float(veil.mean()) == pytest.approx(0.4, rel=0.05)


# ------------------------------------------------------------------- paving


def test_vegetation_is_not_paved() -> None:
    image = np.empty((40, 120, 3), dtype=np.float32)
    image[:, :40] = (0.2, 0.2, 0.21)  # asphalt
    image[:, 40:80] = (0.12, 0.3, 0.06)  # grass
    image[:, 80:] = (0.5, 0.1, 0.08)  # a red-painted lane is still a road
    paved = _paved_weight(image)
    assert float(paved[:, 5:35].mean()) > 0.98
    assert float(paved[:, 45:75].mean()) < 0.05
    assert float(paved[:, 85:115].mean()) > 0.98


def test_a_reflection_from_above_the_frame_is_sky_not_the_top_row() -> None:
    """On a steeply pitched camera most of the road mirrors a point above
    the picture. Repeating the top row there drew whatever was on it — here
    a red and a blue block — as stripes down the whole road."""
    image = np.full((100, 100, 3), 0.1, dtype=np.float32)
    image[0:4, :50] = (0.9, 0.0, 0.0)
    image[0:4, 50:] = (0.0, 0.0, 0.9)
    wetness = np.ones((100, 100), dtype=np.float32)
    sky = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    # Horizon far above the frame: every row mirrors something off it.
    out = _apply_ground_wetness(
        image, wetness, 0.5, horizon_px=-200.0, focal_px=60.0, sky_radiance=sky
    )
    road = out[20:]
    assert float(np.abs(road[:, :50] - road[:, 50:]).max()) < 1e-3, "the two halves must match"
    assert float(road[..., 0].mean()) == pytest.approx(float(road[..., 2].mean()), abs=1e-3)


# -------------------------------------------------------------------- light


def test_heavier_precipitation_means_more_cloud() -> None:
    assert cloud_cover(0.0, 8.0) == 0.0
    assert cloud_cover(8.0, 8.0) == pytest.approx(1.0 - np.exp(-1.0))
    assert cloud_cover(2.0, 8.0) < cloud_cover(20.0, 8.0) < 1.0


def test_overcast_greys_the_sky_and_fills_the_shadows() -> None:
    """Rain used to be drawn on a sunny day: blue sky, hard shadows, full
    colour. The light changes first."""
    image, depth = _outdoor()
    image[50:, :80] *= 0.3  # a shadowed half
    image[50:, :, 1] *= 1.6  # and some colour on the ground
    sky = sky_weight(image, depth)
    out = apply_overcast(image, 1.0, sky)

    def saturation(pixels: np.ndarray) -> float:
        mean = pixels.reshape(-1, 3).mean(axis=0)
        return float((mean.max() - mean.min()) / mean.max())

    assert saturation(out[:40]) < 0.25 * saturation(image[:40])
    assert saturation(out[60:]) < 0.8 * saturation(image[60:])
    before = image[60:, 100:].mean() / image[60:, :60].mean()
    after = out[60:, 100:].mean() / out[60:, :60].mean()
    assert 1.0 < after < 0.9 * before, "shadows should fill, and stay shadows"


def test_overcast_is_graded_and_can_be_off() -> None:
    image, depth = _outdoor()
    sky = sky_weight(image, depth)
    assert apply_overcast(image, 0.0, sky) is image
    half = np.abs(apply_overcast(image, 0.5, sky) - image).mean()
    full = np.abs(apply_overcast(image, 1.0, sky) - image).mean()
    assert 0.0 < half < full


def test_a_sky_that_is_already_grey_changes_little() -> None:
    blue, depth = _outdoor()
    grey, _ = _outdoor(sky_colour=(0.55, 0.55, 0.56))
    change_blue = np.abs(apply_overcast(blue, 1.0, sky_weight(blue, depth)) - blue)[60:].mean()
    change_grey = np.abs(apply_overcast(grey, 1.0, sky_weight(grey, depth)) - grey)[60:].mean()
    assert change_grey < 0.5 * change_blue
