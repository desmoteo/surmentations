"""Water on the glass of a camera that is focused far beyond it.

`test_lens_artifacts.py` keeps the invariants of where drops are and which of
them run. This file pins what they look like, and each test is a way the
earlier renderer was wrong:

* condensation was blended towards a fixed white, so a fogged lens at night
  lit the frame from nowhere — thick-film areas of a night street went from
  level 35 to 102;
* drops were blurred by a hand-set fraction of their own radius, where the
  blur is the aperture's width whatever the drop's size, so beads far
  narrower than the pupil were drawn crisp;
* every drop carried a +210 highlight from a fixed light and a -85 rim,
  whatever the scene, so drops over a dark frame were lit by nothing;
* drops were circles kept apart by a forced gap, and runners were stretched
  circles on ruler-straight tracks;
* and, once those were fixed, every large drop was still the same shape at a
  different size: one ellipse for all the pinned ones, one teardrop formula
  for all the runners.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from weathermentations.diagnostics import recording
from weathermentations.lens.water_droplets import (
    _LOBE_REACH,
    _OUTLINE_HARMONICS,
    _OUTLINE_LEVEL,
    WaterDroplets,
    _aperture_blur,
    _condensation_field,
    _drop_optics,
    _drop_outline,
    _lobe_profile,
    _runner_body,
    _runner_path,
    _water_surface,
)
from weathermentations.utils.photometry import srgb_to_linear

# A pane with nothing on it but film: one drop too small to cover a pixel.
NO_DROPS = {"droplet_count_range": (1, 1), "droplet_radius_range": (0.0005, 0.0006)}
FULL_FILM = {
    "condensation_coverage_range": (1.0, 1.0),
    "condensation_edge_softness_range": (0.01, 0.01),
    "condensation_milkiness_range": (0.7, 0.7),
}
NO_FILM = {"condensation_coverage_range": (0.0, 0.0)}


def _lobe(
    x: float, y: float, radius: float, cap: float = 0.5, egg: float = 0.0, sag: float = 0.0
) -> tuple[float, ...]:
    """One drop lobe as the water surface takes it: round, with the given cap, egg and sag."""
    return (x, y, radius, 1.0, cap, egg, sag)


def _round(count: int = 1) -> np.ndarray:
    """Outline harmonics that leave ``count`` lobes as plain ellipses."""
    return np.zeros((count, len(_OUTLINE_HARMONICS), 2))


def _night_street(size: int = 256) -> np.ndarray:
    """A dark frame with two lamps that are bright but not clipped."""
    image = np.full((size, size, 3), 14, dtype=np.uint8)
    cv2.circle(image, (size // 3, size // 3), 5, (230, 200, 120), -1)
    cv2.circle(image, (2 * size // 3, size // 2), 4, (230, 200, 120), -1)
    return image


def _smooth(size: int = 256) -> np.ndarray:
    """Gradients only, so every edge in the output was put there by water."""
    ramp = np.linspace(0.0, 1.0, size, dtype=np.float32)
    across = np.broadcast_to(60.0 + 150.0 * ramp[None, :], (size, size))
    down = np.broadcast_to(200.0 - 140.0 * ramp[:, None], (size, size))
    return np.stack([across, down, np.full((size, size), 120.0)], axis=2).astype(np.uint8)


def _textured(size: int = 256, seed: int = 0) -> np.ndarray:
    """A frame of coarse random blocks, texture that refraction visibly moves."""
    rng = np.random.default_rng(seed)
    coarse = rng.integers(40, 215, size=(size // 8, size // 8, 3)).astype(np.uint8)
    return cv2.resize(coarse, (size, size), interpolation=cv2.INTER_NEAREST)


def _run(image: np.ndarray, seed: int = 0, **kwargs: object) -> np.ndarray:
    """``image`` through ``WaterDroplets`` with ``kwargs``, seeded."""
    transform = WaterDroplets(p=1.0, **kwargs)  # type: ignore[arg-type]
    transform.set_random_seed(seed)
    return np.asarray(transform(image=image)["image"])


# --------------------------------------------------------------------- film


def test_condensation_redistributes_light_and_adds_none() -> None:
    """A film scatters the light that reaches it. Over a night scene the
    frame must stay as dark as it was, on average: whatever a pixel gains
    was taken from somewhere else."""
    image = _night_street()
    out = _run(image, **NO_DROPS, **FULL_FILM)

    before = float(srgb_to_linear(image).mean())
    after = float(srgb_to_linear(out).mean())
    assert after <= before * 1.02, f"mean linear radiance rose from {before:.5f} to {after:.5f}"

    # Far from both lamps the frame was level 14 and has to stay about there.
    corner = out[-40:, :40].astype(np.float32).mean()
    assert corner < 22.0, f"a dark corner far from any light was lifted to level {corner:.1f}"


def test_a_lamp_behind_condensation_gets_a_glow() -> None:
    """The night-time signature of a fogged lens: the film's wide scattering
    lobe spreads each lamp over a large area, well outside the lamp itself."""
    image = _night_street()
    out = _run(image, **NO_DROPS, **FULL_FILM).astype(np.float32)
    clean = image.astype(np.float32)

    lamp = (256 // 3, 256 // 3)
    yy, xx = np.mgrid[0:256, 0:256]
    distance = np.hypot(xx - lamp[0], yy - lamp[1])
    ring = (distance > 14) & (distance < 30)
    gain = float((out[ring] - clean[ring]).mean())
    assert gain > 2.0, f"no glow around the lamp: +{gain:.2f} levels at 14-30 px"
    # And the lamp pays for it.
    core = distance < 3
    assert out[core].mean() < clean[core].mean()


def test_condensation_cannot_brighten_a_flat_frame() -> None:
    """With nothing brighter than anything else there is nothing to spread;
    all a film can do to an even frame is lose a little of it."""
    image = np.full((128, 128, 3), 120, dtype=np.uint8)
    out = _run(image, **NO_DROPS, **FULL_FILM)
    assert int(out.max()) <= 120
    assert int(out.min()) >= 108


@pytest.mark.parametrize("seed", range(10))
def test_condensation_is_one_film_not_scattered_patches(seed: int) -> None:
    """Glass temperature varies smoothly, so the fogged area is one region
    with a single front."""
    field = _condensation_field(
        240, 320, np.random.default_rng(seed), coverage=0.4, edge_softness=0.3, irregularity=0.4
    )
    assert field.shape == (240, 320)
    assert 0.0 <= float(field.min()) and float(field.max()) <= 1.0

    fogged = (field > 0.5).astype(np.uint8)
    assert 0.25 < fogged.mean() < 0.55, f"coverage {fogged.mean():.2f} for a request of 0.4"
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(fogged, connectivity=8)
    areas = stats[1:count, cv2.CC_STAT_AREA]
    assert areas.max() > 0.95 * areas.sum(), f"film is in {count - 1} pieces: {sorted(areas)}"


# ----------------------------------------------------------------- surface


def test_one_lobe_is_a_spherical_cap_of_its_own_radius() -> None:
    level, height_sq = _water_surface(np.array([_lobe(64.0, 64.0, 20.0, cap=0.5)]), 128, 128)

    outline = level > _OUTLINE_LEVEL
    assert outline.sum() == pytest.approx(np.pi * 20.0**2, rel=0.03)
    assert float(np.sqrt(height_sq[64, 64])) == pytest.approx(0.5 * 20.0, rel=0.01)
    # h = c * sqrt(r^2 - d^2), so half-way out it is sqrt(3)/2 of the peak.
    assert float(np.sqrt(height_sq[64, 74])) == pytest.approx(0.5 * 20.0 * np.sqrt(0.75), rel=0.02)
    assert float(height_sq[64, 90]) == 0.0


def _pieces(level: np.ndarray) -> int:
    """How many separate patches of water a level field outlines."""
    count, _labels = cv2.connectedComponents((level > _OUTLINE_LEVEL).astype(np.uint8))
    return int(count) - 1


def test_touching_lobes_merge_and_separate_drops_do_not() -> None:
    """Two drops caught mid-coalescence are one body of water with a neck;
    two held apart by the placement gap are two."""
    r1, r2 = 18.0, 12.0

    def pair(spacing: float) -> np.ndarray:
        lobes = np.array([_lobe(70.0, 80.0, r1), _lobe(70.0 + spacing * (r1 + r2), 80.0, r2)])
        return _water_surface(lobes, 160, 200)[0]

    merged = pair(0.95)
    assert _pieces(merged) == 1
    # The neck is narrower than either drop: a peanut, not an ellipse.
    neck_column = int(round(70.0 + r1 * 0.95 + 1))
    neck = int((merged[:, neck_column] > _OUTLINE_LEVEL).sum())
    assert 0 < neck < 2 * r2

    # 0.12 is the transform's default `min_gap_ratio`.
    assert _pieces(pair(1.12)) == 2


def test_a_lobe_ends_without_a_corner() -> None:
    """Where one lobe ends inside another, a corner in its profile becomes a
    crease in the water surface, and a crease refracts as a line: the outline
    of one drop drawn across the other. The profile has to arrive at its rim
    and at zero with no jump in slope."""
    distance = np.linspace(0.0, _LOBE_REACH + 0.2, 4001)
    profile = _lobe_profile(distance**2)

    assert float(profile[0]) == 1.0
    assert float(np.interp(1.0, distance, profile)) == pytest.approx(_OUTLINE_LEVEL, abs=1e-4)
    assert float(profile[distance >= _LOBE_REACH].max()) == 0.0
    assert np.all(np.diff(profile) <= 1e-12), "a lobe must only fall with distance"
    slope = np.diff(profile) / np.diff(distance)
    # The cap's own slope changes by 0.0006 per step here; a corner at the
    # rim would be a jump of 1.5.
    assert float(np.abs(np.diff(slope)).max()) < 0.01


def _runner(seed: int, radius: float = 16.0, tail: float = 90.0):
    """A running drop's outline, drawn from one seed."""
    return _runner_body(
        np.random.default_rng(seed), 60.0, 130.0, radius, tail, cap=0.5, path=None
    )


def test_a_runner_is_one_body_of_water_with_its_head() -> None:
    """The head is a drop; what it trails is joined to it, not laid over it."""
    radius = 16.0
    for seed in range(12):
        body = _runner(seed)
        level, height_sq = _water_surface(
            np.array([_lobe(60.0, 130.0, radius)]), 180, 120, _round(), [body]
        )
        wet = (level > _OUTLINE_LEVEL).astype(np.uint8)
        _count, labels = cv2.connectedComponents(wet)
        main = labels == labels[130, 60]
        assert main[130, 60], f"seed {seed}: the head is missing"
        rows = np.flatnonzero(main.any(axis=1))
        # Blunt below, where the head advances; drawn out above it.
        assert rows.max() == pytest.approx(130 + radius, abs=2)
        assert 130 - rows.min() > 1.6 * radius, f"seed {seed}: nothing trails the head"
        # Widest at the head, and tallest there too.
        widths = main.sum(axis=1)
        assert abs(int(widths.argmax()) - 130) <= radius
        peak_row, _column = np.unravel_index(int(height_sq.argmax()), height_sq.shape)
        assert abs(int(peak_row) - 130) <= 1.2 * radius


def test_no_two_runners_are_the_same_shape() -> None:
    """They used to be: one formula, scaled. On a real pane each runner is
    at a different stage — a slug that has just let go, a pear, a head with
    a thread, a thread already broken into beads — and each is lopsided in
    its own way."""
    bodies = [_runner(seed) for seed in range(24)]

    def profile(body) -> np.ndarray:
        return (body.left + body.right) / (2.0 * 16.0)

    profiles = [profile(body) for body in bodies]
    for first in range(len(profiles)):
        for second in range(first + 1, len(profiles)):
            gap = float(np.abs(profiles[first] - profiles[second]).max())
            assert gap > 0.05, f"runners {first} and {second} have the same outline"

    def pieces(body) -> int:
        present = (body.left + body.right) > 0.6
        return int(np.sum(present[1:] & ~present[:-1]) + present[0])

    broken = sum(pieces(body) > 1 for body in bodies)
    whole = sum(pieces(body) == 1 for body in bodies)
    assert broken >= 3, "no runner's thread had broken into beads"
    assert whole >= 3, "every runner's thread had broken up"

    # Neither side mirrors the other, and the tail does not stay on the axis.
    lopsided = [float(np.abs(body.left - body.right).max()) / 16.0 for body in bodies]
    assert min(lopsided) > 0.01
    leans = [float(np.ptp(body.centre)) for body in bodies]
    assert max(leans) > 4.0 and min(leans) < max(leans) / 3.0


def test_small_drops_are_round_and_large_ones_are_not() -> None:
    """Surface tension holds a drop far below the capillary length in a
    near-perfect cap. A large one keeps whatever outline its pinned contact
    line was left with, and gravity pools it towards its lower edge."""
    rng = np.random.default_rng(0)
    fine = [_drop_outline(rng, 0.1, deformation=0.25) for _ in range(50)]
    large = [_drop_outline(rng, 0.1, deformation=2.4) for _ in range(50)]

    def departure(outlines) -> float:
        return float(np.mean([h[:, 0].sum() + egg for egg, _sag, h in outlines]))

    assert departure(fine) < 0.05
    assert departure(large) > 5.0 * departure(fine)
    assert max(h[:, 0].sum() for _egg, _sag, h in large) <= 0.34 + 1e-9

    def circularity(harmonics: np.ndarray, egg: float) -> float:
        level, _ = _water_surface(
            np.array([_lobe(100.0, 100.0, 50.0, egg=egg)]), 200, 200, harmonics[None]
        )
        wet = (level > _OUTLINE_LEVEL).astype(np.uint8)
        contours, _hierarchy = cv2.findContours(wet, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        outline = max(contours, key=cv2.contourArea)
        return float(4.0 * np.pi * cv2.contourArea(outline) / cv2.arcLength(outline, True) ** 2)

    round_drop = circularity(_round()[0], 0.0)
    shaped = [circularity(h, egg) for egg, _sag, h in large[:12]]
    assert round_drop > 0.88
    assert max(shaped) < round_drop
    # Not one shape at twelve sizes: the outlines differ from each other.
    assert float(np.ptp(shaped)) > 0.02


def test_gravity_makes_a_large_drop_bottom_heavy() -> None:
    lobes = np.array([_lobe(100.0, 100.0, 50.0, egg=0.15, sag=0.4)])
    level, height_sq = _water_surface(lobes, 200, 200, _round())
    wet = level > _OUTLINE_LEVEL

    assert int(wet[125].sum()) > int(wet[75].sum()) + 6, "the outline should widen downward"
    peak_row, _column = np.unravel_index(int(height_sq.argmax()), height_sq.shape)
    assert int(peak_row) > 104, "the water should pool below the drop's middle"


def test_resting_drops_sag_in_proportion_to_their_weight() -> None:
    """Gravity against surface tension goes as the square of a drop's size,
    so only drops near the shedding threshold are visibly stretched."""
    transform = WaterDroplets(
        p=1.0,
        droplet_count_range=(300, 300),
        droplet_radius_range=(0.002, 0.03),
        pane_width_range_mm=(60.0, 60.0),
        vertical_stretch_range=(1.5, 1.5),
        trail_probability=0.0,
    )
    transform.set_random_seed(1)
    with recording() as log:
        transform(image=_textured(512))
    shedding = float(log[0]["drew"]["shedding_radius_px"])
    transform.set_random_seed(1)
    params = transform.get_params_dependent_on_data({"shape": (512, 512, 3)}, {})

    pinned = [(d[2], d[4]) for d in params["droplets"] if not d[7]]
    assert len(pinned) > 50
    for radius, elongation in pinned:
        assert elongation == pytest.approx(1.0 + 0.5 * min(1.0, (radius / shedding) ** 2))
    small = [elongation for radius, elongation in pinned if radius < 0.3 * shedding]
    assert small and max(small) < 1.05


def test_runner_tracks_meander_but_never_climb() -> None:
    rng = np.random.default_rng(4)
    path = _runner_path(rng, x=200.0, y=380.0, radius=10.0, length_px=320.0, straight_px=40.0)

    assert np.all(np.diff(path[:, 1]) < 0), "the track must run monotonically up the pane"
    assert path[0].tolist() == [200.0, 380.0]
    # No sideways jog inside the drop's own tail, only its slow wander.
    own_tail = path[path[:, 1] > 380.0 - 40.0]
    assert float(np.abs(own_tail[:, 0] - 200.0).max()) < 0.5 * 10.0
    assert float(np.ptp(path[:, 0])) > 5.0, "a 320 px track with no sideways movement at all"
    # Jogs are about a drop width, not a wander across the pane.
    assert float(np.abs(path[:, 0] - 200.0).max()) < 12 * 10.0


# ------------------------------------------------------------------ optics


def _single_drop_optics(cap: float = 0.5, focal_px: float = 600.0):
    """The optics of one round drop in the middle of a 200-pixel pane."""
    level, height_sq = _water_surface(np.array([_lobe(100.0, 100.0, 40.0, cap=cap)]), 200, 200)
    optics = _drop_optics(level, height_sq, focal_px=focal_px, light_direction=(0.0, -0.5, 0.866))
    assert optics is not None
    return optics


def test_the_picture_inside_a_drop_is_inverted() -> None:
    """A convex drop is a short-focal-length lens: what is above the drop
    appears at its bottom. Nothing sets that directly — it has to come out of
    the refraction.

    It also has to stop being true when it should. The picture is inverted
    only if the lens sits further behind the glass than the drop's own focal
    length, ``r / (c (n - 1))``; a drop wide enough that it does not is a
    weak magnifier, and the camera sees through it the right way up.
    """
    weak = _single_drop_optics(focal_px=150.0)
    weak_y = weak.map_y.ravel()[: weak.wet.size]
    weak_rows = (weak.wet // 200).astype(np.float32)
    weak_columns = (weak.wet % 200).astype(np.float32)
    inner = np.hypot(weak_columns - 100.0, weak_rows - 100.0) < 12.0
    assert np.all((weak_y[inner] - 100.0) * (weak_rows[inner] - 100.0) >= 0.0)

    optics = _single_drop_optics()
    ys, xs = (optics.wet // 200).astype(np.float32), (optics.wet % 200).astype(np.float32)
    sample_x = optics.map_x.ravel()[: optics.wet.size]
    sample_y = optics.map_y.ravel()[: optics.wet.size]

    radius = np.hypot(xs - 100.0, ys - 100.0)
    body = (radius > 8.0) & (radius < 30.0)
    below, right = body & (ys > 108.0), body & (xs > 108.0)
    assert np.all(sample_y[below] < 100.0), "the lower half of a drop must look upward"
    assert np.all(sample_x[right] < 100.0), "the right half of a drop must look left"
    # The centre looks straight ahead.
    centre = radius < 1.5
    assert np.abs(sample_y[centre] - ys[centre]).max() < 12.0


def test_the_dark_rim_is_total_internal_reflection() -> None:
    """Past 48.8 degrees of surface tilt no light leaves the water. That band
    is the outline of a drop, and it is wider on a steeper drop."""
    def rim_share(cap: float) -> float:
        optics = _single_drop_optics(cap)
        covered = optics.alpha.ravel() > 0.99
        return float((optics.reflect.ravel()[covered] > 0.999).mean())

    optics = _single_drop_optics()
    ys, xs = optics.wet // 200, optics.wet % 200
    radius = np.hypot(xs - 100.0, ys - 100.0)
    reflect = optics.reflect.ravel()
    # Fresnel at normal incidence, water to air: ((n-1)/(n+1))^2 = 2%.
    assert float(reflect[radius < 3.0].mean()) == pytest.approx(0.02, abs=0.005)
    assert float(reflect[(radius > 38.0) & (radius < 39.5)].min()) == 1.0

    shallow, steep = rim_share(0.35), rim_share(0.65)
    assert 0.0 < shallow < steep < 0.6, (shallow, steep)


def test_the_aperture_dissolves_drops_narrower_than_itself() -> None:
    """Focused far away, the lens images a point on the glass as a disc the
    width of its aperture. The same drops are therefore crisp behind a
    stopped-down iris and all but gone behind an open one."""
    image = _smooth()
    common = dict(
        droplet_count_range=(60, 60),
        droplet_radius_range=(0.008, 0.024),  # 0.35 to 1 mm on this pane
        pane_width_range_mm=(30.0, 30.0),
        glint_range=(0.0, 0.0),
        **NO_FILM,
    )

    def edge_energy(aperture_mm: float) -> float:
        out = _run(image, seed=3, aperture_diameter_range_mm=(aperture_mm, aperture_mm), **common)
        change = out.astype(np.float32).mean(axis=2) - image.astype(np.float32).mean(axis=2)
        # The strongest edges, not the mean: a soft change still rounds to
        # 8-bit steps, and those put a faint Laplacian on every pixel.
        return float(np.percentile(np.abs(cv2.Laplacian(change, cv2.CV_32F)), 99.0))

    stopped, open_iris = edge_energy(0.2), edge_energy(3.0)
    assert stopped > 5.0 * open_iris, (
        f"edges added behind a 0.2 mm aperture ({stopped:.2f}) should far exceed those "
        f"behind a 3 mm one ({open_iris:.2f})"
    )
    # Dissolved, not removed: the water still costs the frame contrast.
    assert open_iris > 0.0


def test_the_aperture_disc_is_flat_topped_and_conserves_light() -> None:
    point = np.zeros((65, 65), dtype=np.float32)
    point[32, 32] = 1.0
    disc = _aperture_blur(point, 9.0)

    assert float(disc.sum()) == pytest.approx(1.0, abs=1e-4)
    inside = disc[32, 32 - 3 : 32 + 4]
    assert float(inside.max() - inside.min()) < 1e-6, "an out-of-focus point is a uniform disc"
    assert float(disc[32, 32 + 6]) < 1e-6
    # Wide enough to be computed at reduced size, and still the same light.
    wide = _aperture_blur(np.pad(point, 100), 60.0)
    assert float(wide.sum()) == pytest.approx(1.0, abs=0.02)


def test_water_over_a_black_frame_stays_black() -> None:
    """Every bright thing in a drop is the image of a light in the scene.
    With no light and the illuminator off there is nothing to show — the
    fixed highlight this replaces put a glint in every drop regardless."""
    black = np.zeros((160, 160, 3), dtype=np.uint8)
    out = _run(black, seed=1, glint_range=(1.0, 1.0), **FULL_FILM)
    assert int(out.max()) == 0


def test_the_infrared_ring_lights_water_on_the_glass() -> None:
    """The one light that does come from the camera. On, a fogged or wet
    dome glows; that is why such a camera whites out at night."""
    dark = np.full((200, 200, 3), 10, dtype=np.uint8)
    common = dict(
        droplet_count_range=(40, 40),
        droplet_radius_range=(0.01, 0.04),
        aperture_diameter_range_mm=(1.0, 1.0),
        condensation_coverage_range=(0.5, 0.5),
        condensation_milkiness_range=(0.6, 0.6),
    )
    off = _run(dark, seed=2, illuminator_glow_range=(0.0, 0.0), **common).astype(np.float32)
    on = _run(dark, seed=2, illuminator_glow_range=(1.0, 1.0), **common).astype(np.float32)

    assert off.max() <= 12.0, "with the illuminator off nothing may outshine the scene"
    lit = (on - off).mean(axis=2)
    assert float(lit.min()) >= -1.0, "the illuminator can only add light"
    assert float((lit > 20.0).mean()) > 0.15, "too little of the wet glass lit up"
    # Clear, dry glass returns nothing.
    assert float((lit < 1.0).mean()) > 0.05


# --------------------------------------------------------------- transform


def test_runners_sweep_their_tracks_clear() -> None:
    """A drop that has slid down has wiped the film off its track and
    collected every bead in the way, so no pinned drop may sit on one."""
    transform = WaterDroplets(
        p=1.0,
        droplet_count_range=(250, 250),
        droplet_radius_range=(0.003, 0.03),
        pane_width_range_mm=(200.0, 200.0),
        trail_probability=1.0,
        trail_length_range=(0.4, 0.7),
        **FULL_FILM,
    )
    swept_total = 0.0
    for seed in range(4):
        transform.set_random_seed(seed)
        params = transform.get_params_dependent_on_data({"shape": (400, 400, 3)}, {})
        film = params["condensation"]
        swept_total += float((film < 0.05).mean())
        for x, y, _radius, _cap, elongation, *_rest, sliding in params["droplets"]:
            # Pinned drops sag a little; the round beads a runner leaves in
            # its own track are the exception and have no sag at all.
            if sliding or elongation == 1.0:
                continue
            row, column = int(round(y)), int(round(x))
            if 0 <= row < 400 and 0 <= column < 400:
                assert film[row, column] > 0.4, f"seed {seed}: a pinned drop sits on a wiped track"
    assert swept_total > 0.02, "no track was wiped at all"


def test_some_drops_are_caught_merged_and_the_rate_is_a_parameter() -> None:
    def merged(rate: float) -> int:
        transform = WaterDroplets(
            p=1.0, droplet_count_range=(120, 120), coalescence_range=(rate, rate)
        )
        transform.set_random_seed(0)
        with recording() as log:
            transform(image=_textured())
        return int(log[0]["drew"]["merged_lobes"])

    assert merged(0.0) == 0
    assert merged(0.6) > merged(0.1) > 0


def test_a_clip_is_one_pane_of_glass() -> None:
    """The water does not move between the frames of a clip, and its optics
    are worked out once for all of them."""
    frame = _textured()
    clip = np.stack([frame, frame, frame[:, ::-1].copy()])
    transform = WaterDroplets(p=1.0)
    transform.set_random_seed(5)
    out = transform(images=clip)["images"]

    np.testing.assert_array_equal(out[0], out[1])
    assert not np.array_equal(out[0], out[2])


def test_focusing_on_the_glass_draws_the_water_sharp() -> None:
    """`background_blur_range` is the other camera: focused on the pane, so
    the scene is soft and the aperture disc no longer applies to the water."""
    image = _textured()
    common = dict(
        droplet_count_range=(40, 40),
        droplet_radius_range=(0.01, 0.03),
        aperture_diameter_range_mm=(3.0, 3.0),
        pane_width_range_mm=(30.0, 30.0),
        **NO_FILM,
    )
    far = _run(image, seed=7, **common).astype(np.float32)
    near = _run(image, seed=7, background_blur_range=(0.02, 0.02), **common).astype(np.float32)

    def sharpest_edge(out: np.ndarray) -> float:
        return float(np.percentile(np.abs(cv2.Laplacian(out.mean(axis=2), cv2.CV_32F)), 99.9))

    # The scene's own block edges are gone, and what is left is drop outlines.
    blurred_scene = cv2.GaussianBlur(image, (0, 0), 0.02 * np.hypot(256, 256)).astype(np.float32)
    assert sharpest_edge(near) > 4.0 * sharpest_edge(blurred_scene)
    assert np.abs(near - far).mean() > 5.0
