"""Lens dirt, sun flare and their kin: the output contract and each effect's physical signature."""

import math

import cv2
import numpy as np
import pytest

from surmentations.lens.dirt import DirtSoil
from surmentations.lens.sun_flare import SunFlare, diffraction_starburst
from surmentations.lens.water_droplets import (
    WaterDroplets,
    capillary_length_mm,
    power_law_radii,
)
from surmentations.utils.spectral import ghost_tint


@pytest.fixture
def image() -> np.ndarray:
    """A frame of uniform noise, as a fixture."""
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
    invisible in flat light and obvious when something bright is in frame.

    Visibility is measured as the *spatial structure* the grime introduces
    (the standard deviation of the change), not the mean absolute change.
    The two stopped being interchangeable once deposits carried a real
    single-scattering albedo: an absorbing deposit such as soot also removes
    a few percent of the light uniformly, which shows up in a mean-absolute
    metric but is exactly the part a camera's auto-exposure cancels and is
    not what makes dirt conspicuous. What makes it conspicuous is a bright
    source being smeared across the mask, which is structure.
    """
    flat = np.full((128, 128, 3), 90, dtype=np.uint8)
    backlit = flat.copy()
    cv2.circle(backlit, (64, 30), 18, (255, 255, 255), -1)

    t = DirtSoil(p=1.0)
    t.set_random_seed(0)
    params = t.get_params_dependent_on_data({"shape": flat.shape}, {})

    flat_structure = (
        t.apply(flat, **params).astype(np.float32) - flat.astype(np.float32)
    ).std()
    backlit_structure = (
        t.apply(backlit, **params).astype(np.float32) - backlit.astype(np.float32)
    ).std()
    assert backlit_structure > flat_structure * 1.5


def test_dirt_deposits_absorb_in_albedo_order() -> None:
    """A deposit's single-scattering albedo is what it does *not* absorb, so
    dimming in flat light must follow 1 - albedo across the palette."""
    flat = np.full((128, 128, 3), 120, dtype=np.uint8)

    def mean_shift(deposit: str) -> float:
        shifts = []
        for seed in range(6):
            t = DirtSoil(p=1.0, deposits=(deposit,))
            t.set_random_seed(seed)
            params = t.get_params_dependent_on_data({"shape": flat.shape}, {})
            out = t.apply(flat, **params).astype(np.float32)
            shifts.append(float((out - flat.astype(np.float32)).mean()))
        return float(np.mean(shifts))

    # Ordered by increasing absorption (decreasing albedo), so each must dim
    # the frame strictly more than the one before it.
    ladder = ["limescale", "mineral_dust", "road_film", "soot"]
    shifts = [mean_shift(name) for name in ladder]
    # Deliberately pairwise, so the shorter second argument is the point.
    assert all(a > b for a, b in zip(shifts, shifts[1:], strict=False)), shifts
    # A near-non-absorbing deposit must be close to invisible in flat light.
    assert shifts[0] > -1.5


def test_dirt_deposit_hue_matches_the_substance() -> None:
    """Mineral dust absorbs toward the blue and so must leave a warmer cast
    than soot, which absorbs almost flat across the visible band."""
    flat = np.full((128, 128, 3), 120, dtype=np.uint8)

    def warmth(deposit: str) -> float:
        values = []
        for seed in range(6):
            t = DirtSoil(p=1.0, deposits=(deposit,), coverage_range=(0.3, 0.3))
            t.set_random_seed(seed)
            params = t.get_params_dependent_on_data({"shape": flat.shape}, {})
            out = t.apply(flat, **params).astype(np.float32).reshape(-1, 3).mean(axis=0)
            values.append(float(out[0] - out[2]))
        return float(np.mean(values))

    assert warmth("mineral_dust") > warmth("soot") + 1.0
    assert abs(warmth("soot")) < 1.0


def test_dirt_glows_under_a_sky_the_camera_cannot_see() -> None:
    """Dirt scatters light in from the whole hemisphere in front of the glass,
    and most of that is outside the frame. A camera looking down at a road
    has grime lit by a sky several times brighter than its view, which is the
    milky lift a dirty dome shows by day — and what the model lacked while it
    scattered only the frame's own light: a patch over asphalt was replaced
    by a blur of the same asphalt, and the effect was all but invisible.

    With the surroundings no brighter than the frame, flat light still shows
    nothing: there is no brighter light to bring in.
    """
    road = np.full((128, 128, 3), 70, dtype=np.uint8)

    def render(ambient: float) -> tuple[np.ndarray, np.ndarray]:
        t = DirtSoil(
            p=1.0,
            deposits=("limescale",),
            ambient_light_range=(ambient, ambient),
            dust_film_range=(0.0, 0.0),
            absorption_range=(0.0, 0.0),
        )
        t.set_random_seed(4)
        params = t.get_params_dependent_on_data({"shape": road.shape}, {})
        return t.apply(road, **params).astype(np.float32), params["dirt_mask"]

    level, _mask = render(1.0)
    assert float(np.abs(level - 70.0).max()) < 4.0, "flat light, nothing brighter to scatter in"

    lit, mask = render(4.0)
    thick, clean = mask > 0.5, mask < 0.02
    assert thick.any() and clean.any()
    assert float(lit[thick].mean()) > 70.0 + 5.0, "grime under an open sky should lift the road"
    assert float(np.abs(lit[clean] - 70.0).max()) < 2.0, "clean glass must stay clean"
    # Bounded by what is there to scatter: it cannot outshine the sky it
    # stands in for (four times the frame, here level 133).
    assert float(lit.max()) < 133.0


def test_a_dust_film_leaves_no_clean_glass() -> None:
    """Patches are what splashed or was smeared; the film is what settled.
    It is even, so it veils every pixel and lowers contrast everywhere."""
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    image[:, 64:] = 200

    def render(film: float) -> tuple[np.ndarray, np.ndarray]:
        t = DirtSoil(p=1.0, dust_film_range=(film, film), ambient_light_range=(1.0, 1.0))
        t.set_random_seed(2)
        params = t.get_params_dependent_on_data({"shape": image.shape}, {})
        return t.apply(image, **params).astype(np.float32), params["dirt_mask"]

    _bare, bare_mask = render(0.0)
    filmed, filmed_mask = render(0.3)
    assert float(bare_mask.min()) < 0.01
    assert float(filmed_mask.min()) >= 0.3 - 1e-6
    assert float(filmed_mask.max()) <= 1.0
    # The dark half is lifted by light from the bright half, everywhere.
    assert float(filmed[:, :40].min()) > 3.0


def test_dirt_rejects_unknown_deposit() -> None:
    with pytest.raises(ValueError, match="Unknown deposit"):
        DirtSoil(p=1.0, deposits=("unobtanium",))


def test_water_droplets_refraction_stays_in_bounds(image: np.ndarray) -> None:
    # A steep drop bends rays by up to 41 degrees, which from near the edge
    # of the frame points well outside it. The remap reflects at the border,
    # so that must neither raise nor leave a hole.
    t = WaterDroplets(
        p=1.0,
        droplet_count_range=(20, 20),
        droplet_radius_range=(0.01, 0.12),
        cap_height_ratio_range=(0.9, 1.0),
    )
    out = t(image=image)["image"]
    assert out.shape == image.shape


def test_water_droplets_never_overlap(image: np.ndarray) -> None:
    """Separate drops stay separate: two that touched would have become one.

    Merged drops do exist — a share of them are caught mid-coalescence — but
    each is one entry here with extra lobes of the same water surface, so
    this still holds for everything listed. `test_water_on_glass.py` checks
    that the lobes join and that drops this far apart do not."""
    t = WaterDroplets(
        p=1.0,
        droplet_count_range=(40, 40),
        droplet_radius_range=(0.006, 0.02),
        streak_length_range=(1.5, 4.0),
        # A wide pane means each pixel spans more millimetres, so a drop of a
        # given pixel size is physically larger and more of them pass the
        # shedding threshold — which is what puts elongated runners in the
        # field, so the non-overlap rule is exercised on those too.
        pane_width_range_mm=(300.0, 300.0),
    )
    for seed in range(10):
        t.set_random_seed(seed)
        params = t.get_params_dependent_on_data({"shape": image.shape}, {})
        droplets = params["droplets"]
        for i in range(len(droplets)):
            cx1, cy1, r1, _cap1, s1, *_rest1 = droplets[i]
            footprint1 = r1 * max(1.0, s1)
            for j in range(i + 1, len(droplets)):
                cx2, cy2, r2, _cap2, s2, *_rest2 = droplets[j]
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

    for gx, gy, *_optics in params["ghosts"]:
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
        # Only drops past the shedding threshold leave a wiped track, so the
        # pane has to be wide enough that some of them do — otherwise there
        # are no wipe edges for this test to measure at all.
        pane_width_range_mm=(120.0, 120.0),
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


def test_flare_ghost_tints_stay_in_the_coating_palette() -> None:
    """Ghost colour comes from thin-film interference in the lens coatings,
    not from a random hue rotation. A single-layer coating's residual
    reflectance always suppresses the middle of the band relative to at least
    one end, so ghosts land in the violet/amber/cyan family — green-dominant
    ghosts are the one thing the physics cannot produce."""
    for design_a in (460.0, 520.0, 550.0, 620.0):
        for design_b in (460.0, 520.0, 550.0, 620.0):
            tint = ghost_tint(design_a, design_b, incidence_deg=20.0, bounces=4)
            assert tint.max() == pytest.approx(1.0)
            assert tint.min() >= 0.0
            # Green is never strictly the brightest channel on its own.
            assert tint[1] <= max(tint[0], tint[2]) + 1e-6


def test_flare_higher_order_ghosts_are_more_saturated() -> None:
    """Each extra pair of internal bounces multiplies in another copy of the
    coating's residual spectrum, so high-order ghosts are more saturated —
    and correspondingly fainter, which the renderer pairs them with."""

    def saturation(bounces: int) -> float:
        tint = ghost_tint(480.0, 600.0, incidence_deg=20.0, bounces=bounces)
        return float(tint.max() - tint.min())

    assert saturation(6) > saturation(4) > saturation(2)


def test_starburst_spike_count_follows_blade_parity() -> None:
    """The starburst is the aperture's Fraunhofer diffraction pattern, so an
    even blade count gives N spikes (opposite edges are parallel and their
    spikes coincide) and an odd count gives 2N."""

    def spike_count(blades: int) -> int:
        pattern = diffraction_starburst(blades)
        size = pattern.shape[0]
        centre = size / 2.0
        angles = np.linspace(0.0, 2 * np.pi, 1440, endpoint=False)

        # Average over a radial *band*, not a single ring. Each spike carries
        # its own fringe structure and so passes through zeros along its
        # length; sampling one radius lands in those nulls and splits or drops
        # spikes, which is a property of the sampling, not of the pattern.
        profile = np.zeros_like(angles)
        for radius in np.linspace(size * 0.20, size * 0.40, 40):
            xs = np.clip((centre + radius * np.cos(angles)).astype(int), 0, size - 1)
            ys = np.clip((centre + radius * np.sin(angles)).astype(int), 0, size - 1)
            profile += pattern[ys, xs]

        above = profile > profile.mean() + 1.5 * profile.std()
        # Count contiguous runs around the (circular) profile.
        return int(np.sum(above & ~np.roll(above, 1)))

    # Even blade counts give N spikes because opposite edges are parallel;
    # odd counts give 2N.
    assert spike_count(6) == 6
    assert spike_count(8) == 8
    assert spike_count(5) == 10
    assert spike_count(7) == 14


def test_capillary_length_matches_water() -> None:
    """The shedding threshold is built on water's capillary length, which is
    a measured constant (~2.7 mm), not a tuning parameter."""
    assert capillary_length_mm() == pytest.approx(2.7, abs=0.05)


def test_droplet_sizes_follow_a_power_law() -> None:
    """Dropwise condensation reaches a self-similar regime with N(r) ~ r^-2,
    which is far heavier-tailed than a reshaped uniform: most of the glass
    carries fine spatter and only a few beads are large."""
    rng = np.random.default_rng(0)
    radii = power_law_radii(rng, 40000, 2.0, 40.0, tau=2.0)

    assert radii.min() >= 2.0
    assert radii.max() <= 40.0
    # A true r^-2 law puts the median near the small end, nothing like the
    # midpoint a uniform draw would give.
    assert np.median(radii) < 6.0
    # Counts in successive octaves must fall roughly like r^-2 integrated,
    # i.e. each octave up holds about half the previous one's population.
    low = ((radii >= 4) & (radii < 8)).sum()
    high = ((radii >= 8) & (radii < 16)).sum()
    assert 1.6 < low / max(high, 1) < 2.6


def test_only_large_droplets_slide() -> None:
    """Sliding is not a coin flip: a drop runs once gravity beats contact-line
    pinning, which happens above a size set by the capillary length. So every
    elongated drop must be larger than every round one is required to be."""
    t = WaterDroplets(
        p=1.0,
        droplet_count_range=(300, 300),
        droplet_radius_range=(0.002, 0.04),
        pane_width_range_mm=(30.0, 30.0),
    )
    t.set_random_seed(0)
    params = t.get_params_dependent_on_data({"shape": (512, 512, 3)}, {})

    radii = np.array([d[2] for d in params["droplets"]])
    stretch = np.array([d[4] for d in params["droplets"]])
    sliding = stretch > 1.6

    assert sliding.any(), "no droplet passed the shedding threshold"
    assert not sliding.all(), "every droplet slid; threshold is not discriminating"
    # The smallest runner must still be bigger than the median bead.
    assert radii[sliding].min() > np.median(radii[~sliding])


def test_stickier_glass_pins_more_droplets() -> None:
    """Higher contact-angle hysteresis is a stickier surface, so it holds
    larger drops in place and fewer of them run."""

    def sliding_fraction(hysteresis: float) -> float:
        t = WaterDroplets(
            p=1.0,
            droplet_count_range=(300, 300),
            droplet_radius_range=(0.002, 0.04),
            pane_width_range_mm=(30.0, 30.0),
            contact_angle_hysteresis_range=(hysteresis, hysteresis),
        )
        t.set_random_seed(0)
        params = t.get_params_dependent_on_data({"shape": (512, 512, 3)}, {})
        stretch = np.array([d[4] for d in params["droplets"]])
        return float((stretch > 1.6).mean())

    assert sliding_fraction(0.1) > sliding_fraction(1.0)


def test_ccd_smear_is_a_full_height_column_not_a_glow() -> None:
    """CCD smear is a charge-transfer effect, not an optical one.

    Charge leaks into the vertical shift register *while* the frame is being
    shifted out, so it is dragged uniformly down the whole column. That gives
    two properties no glow has, and both are the point of the artefact: it
    reaches the top and bottom edges of the frame, and it does **not** fade
    with distance from the light.
    """
    dark = np.full((300, 400, 3), 10, dtype=np.uint8)
    off = dict(
        streak_probability=0.0,
        ghost_count_range=(0, 0),
        dispersion_ring_probability=0.0,
        scratch_count_range=(0, 0),
        ir_leak_probability=0.0,
        window_ghost_probability=0.0,
        halo_radius_range=(0.95, 0.95),
        intensity_range=(1.0, 1.0),
    )
    source_y = 0.25

    t = SunFlare(
        p=1.0,
        flare_position=(0.5, source_y),
        sensor_smear_probability=1.0,
        sensor_smear_strength_range=(0.35, 0.35),
        sensor_smear_width_range=(4.0, 4.0),
        **off,
    )
    t.set_random_seed(0)
    smeared = t(image=dark)["image"].astype(np.float32)

    baseline = SunFlare(p=1.0, flare_position=(0.5, source_y), sensor_smear_probability=0.0, **off)
    baseline.set_random_seed(0)
    plain = baseline(image=dark)["image"].astype(np.float32)

    column = (smeared - plain).mean(axis=2)[:, 200]

    # Reaches both edges of the frame.
    assert column[0] > 3.0, column[0]
    assert column[-1] > 3.0, column[-1]
    # And is near-uniform down it: a glow centred on the source would be far
    # brighter at the source row than at the far edge.
    near_source = column[int(300 * source_y)]
    far_edge = column[-1]
    assert near_source == pytest.approx(far_edge, rel=0.25)


def test_ir_leak_blooms_magenta() -> None:
    """Silicon stays sensitive past 700 nm, where all three Bayer dyes leak but
    green leaks least — so IR bloom reads magenta-red, never green."""
    dark = np.full((240, 320, 3), 12, dtype=np.uint8)
    off = dict(
        streak_probability=0.0,
        ghost_count_range=(0, 0),
        dispersion_ring_probability=0.0,
        scratch_count_range=(0, 0),
        sensor_smear_probability=0.0,
        window_ghost_probability=0.0,
        halo_radius_range=(0.95, 0.95),
        intensity_range=(1.0, 1.0),
    )
    t = SunFlare(
        p=1.0,
        flare_position=(0.5, 0.5),
        ir_leak_probability=1.0,
        ir_leak_strength_range=(0.5, 0.5),
        **off,
    )
    t.set_random_seed(0)
    leaked = t(image=dark)["image"].astype(np.float32)

    baseline = SunFlare(p=1.0, flare_position=(0.5, 0.5), ir_leak_probability=0.0, **off)
    baseline.set_random_seed(0)
    plain = baseline(image=dark)["image"].astype(np.float32)

    added = (leaked - plain).reshape(-1, 3).mean(axis=0)
    red, green, blue = added
    assert red > green and blue > green, added
    assert red > blue, added


def test_scratch_count_controls_streak_count() -> None:
    """Dome scratches are independent scatterers, so asking for more of them
    must put more light on the frame — unlike the iris starburst, whose spike
    count is fixed by the blade geometry and cannot be varied this way."""
    dark = np.full((240, 320, 3), 12, dtype=np.uint8)
    off = dict(
        streak_probability=0.0,
        ghost_count_range=(0, 0),
        dispersion_ring_probability=0.0,
        sensor_smear_probability=0.0,
        ir_leak_probability=0.0,
        window_ghost_probability=0.0,
        halo_radius_range=(0.95, 0.95),
        intensity_range=(1.0, 1.0),
    )

    def scattered_light(count: int) -> float:
        t = SunFlare(
            p=1.0,
            flare_position=(0.5, 0.45),
            scratch_count_range=(count, count),
            scratch_strength_range=(0.3, 0.3),
            **off,
        )
        t.set_random_seed(1)
        return float(t(image=dark)["image"].astype(np.float32).mean())

    assert scattered_light(0) < scattered_light(3) < scattered_light(9)
