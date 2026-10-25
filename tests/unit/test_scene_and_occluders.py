"""Cast shadows and housing occluders (plan.md 6.2 / Phase 7)."""

from __future__ import annotations

import numpy as np
import pytest

from weathermentations.depth.estimator import DepthEstimator
from weathermentations.depth.provider import SharedDepthProvider
from weathermentations.lens import HousingOccluders
from weathermentations.scene import CastShadows

_FOV_DEG = 60.0
_MAX_DEPTH_M = 50.0


class _GroundPlaneDepth(DepthEstimator):
    """Level camera above a horizontal plane, sky above the horizon."""

    def estimate(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        focal = width / (2.0 * np.tan(np.deg2rad(_FOV_DEG) / 2.0))
        cy = height / 2.0
        rows = np.arange(height, dtype=np.float32).reshape(height, 1)
        depth = np.where(
            rows > cy + 2, 1.5 * focal / np.maximum(rows - cy, 1e-3), _MAX_DEPTH_M
        )
        return np.broadcast_to(
            np.minimum(depth, _MAX_DEPTH_M) / _MAX_DEPTH_M, (height, width)
        ).astype(np.float32)


@pytest.fixture
def scene() -> np.ndarray:
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    img[:120] = (150, 170, 200)  # sky
    img[120:] = (120, 118, 110)  # ground
    return img


def test_shadow_darkens_without_flattening_texture(scene: np.ndarray) -> None:
    """A shadow removes the direct beam and leaves the surface's albedo and
    texture intact. Multiplying radiance does that; blending toward grey
    destroys it, which is the tell-tale of a drawn-on shadow.

    So a textured patch must keep its *relative* contrast when shadowed, even
    though its absolute level drops.
    """
    rng = np.random.default_rng(0)
    textured = scene.copy()
    noise = rng.integers(-25, 26, textured.shape, dtype=np.int16)
    textured = np.clip(textured.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    t = CastShadows(
        depth_provider=SharedDepthProvider(_GroundPlaneDepth()),
        p=1.0,
        shadow_count_range=(6, 6),
        shadow_scale_range=(0.3, 0.3),
        direct_fraction_range=(0.7, 0.7),
    )
    t.set_random_seed(0)
    out = t(image=textured)["image"].astype(np.float32)
    before = textured.astype(np.float32)

    ground_before, ground_after = before[170:230], out[170:230]
    assert ground_after.mean() < ground_before.mean() - 5.0

    # Relative contrast survives: the ratio of spread to level is preserved.
    ratio_before = ground_before.std() / ground_before.mean()
    ratio_after = ground_after.std() / ground_after.mean()
    assert ratio_after > 0.75 * ratio_before


def test_shadows_do_not_land_on_sky(scene: np.ndarray) -> None:
    """Shadows fall on surfaces. The sky is not a surface, and a shadow drawn
    across it is the most obvious possible giveaway."""
    t = CastShadows(
        depth_provider=SharedDepthProvider(_GroundPlaneDepth()),
        p=1.0,
        shadow_count_range=(8, 8),
        shadow_scale_range=(0.35, 0.35),
        direct_fraction_range=(0.9, 0.9),
    )
    t.set_random_seed(0)
    out = t(image=scene)["image"].astype(np.float32)

    sky_change = np.abs(out[:100] - scene[:100].astype(np.float32)).mean()
    ground_change = np.abs(out[160:] - scene[160:].astype(np.float32)).mean()
    assert sky_change < 1.0
    assert ground_change > 5.0


def test_shadow_is_bluer_than_the_light_it_replaces(scene: np.ndarray) -> None:
    """What survives in shadow is skylight, which is blue. A daylight shadow is
    therefore measurably bluer than its surroundings, and rendering it as
    neutral grey misses that."""
    t = CastShadows(
        depth_provider=SharedDepthProvider(_GroundPlaneDepth()),
        p=1.0,
        shadow_count_range=(8, 8),
        shadow_scale_range=(0.4, 0.4),
        direct_fraction_range=(0.85, 0.85),
    )
    t.set_random_seed(0)
    out = t(image=scene)["image"].astype(np.float32)
    before = scene.astype(np.float32)

    ground_before, ground_after = before[160:], out[160:]
    # Blue survives better than red, so the red/blue ratio falls.
    ratio_before = ground_before[..., 0].mean() / max(ground_before[..., 2].mean(), 1e-6)
    ratio_after = ground_after[..., 0].mean() / max(ground_after[..., 2].mean(), 1e-6)
    assert ratio_after < ratio_before


def test_occluders_are_soft_edged(scene: np.ndarray) -> None:
    """An object on the dome is centimetres from a lens focused metres away, so
    it cannot have a sharp edge. Anything crisp here would be wrong."""
    t = HousingOccluders(
        p=1.0,
        insect_count_range=(3, 3),
        web_count_range=(0, 0),
        defocus_range=(0.03, 0.03),
        opacity_range=(0.9, 0.9),
        glow_range=(0.0, 0.0),
    )
    t.set_random_seed(0)
    params = t.get_params_dependent_on_data({"shape": scene.shape}, {})
    mask = params["occluder"]

    assert mask.max() > 0.05, "no occluder was drawn"
    # A hard-edged mask would have gradients approaching its own peak value;
    # a heavily defocused one cannot.
    gradient = np.abs(np.diff(mask, axis=1)).max()
    assert gradient < 0.1 * mask.max()


def test_occluders_darken_by_day_and_glow_under_ir(scene: np.ndarray) -> None:
    """Same geometry, opposite appearance: backlit by the scene they
    silhouette, lit from behind by the IR ring they glow. One parameter covers
    both because one mechanism does."""

    def mean_change(glow: float) -> float:
        t = HousingOccluders(
            p=1.0,
            insect_count_range=(4, 4),
            web_count_range=(1, 1),
            opacity_range=(0.85, 0.85),
            glow_range=(glow, glow),
        )
        t.set_random_seed(1)
        out = t(image=scene)["image"].astype(np.float32)
        return float(out.mean() - scene.astype(np.float32).mean())

    assert mean_change(0.0) < -1.0  # daylight: darker
    assert mean_change(2.0) > 1.0  # IR: brighter


# --- what casts a shadow, and the one sun that throws them all ------------


def _shadow_shape(kind: str, seed: int, azimuth: float, elongation: float) -> np.ndarray | None:
    """One caster's shadow on an empty canvas, or ``None`` if it left the frame."""
    import cv2

    from weathermentations.scene.shadows import _draw_cast_shape

    mask = np.zeros((600, 600), dtype=np.float32)
    _draw_cast_shape(
        mask, np.random.default_rng(seed), 600, 600, 40.0,
        kind=kind, sun_azimuth_deg=azimuth, elongation=elongation,
    )
    filled = mask > 0.5
    edges = (filled[0], filled[-1], filled[:, 0], filled[:, -1])
    if any(edge.any() for edge in edges) or filled.sum() < 200:
        return None
    if kind != "foliage":
        # A figure or a vehicle is one object. Foliage is allowed to come
        # apart: a crown and a stray branch really do cast separate patches.
        pieces = cv2.connectedComponents(filled.astype(np.uint8))[0] - 1
        assert pieces == 1, f"a {kind}'s shadow fell into {pieces} pieces"
    return filled


def _solidity(filled: np.ndarray) -> float:
    """Area over convex-hull area: 1 for a blob, lower for limbs and a neck."""
    import cv2

    points = np.argwhere(filled)[:, ::-1].astype(np.float32)
    return float(filled.sum() / cv2.contourArea(cv2.convexHull(points)))


def _axis_deg(filled: np.ndarray) -> float:
    """Direction of the shape's long axis, in [0, 180)."""
    ys, xs = np.nonzero(filled)
    centred = np.stack([xs - xs.mean(), ys - ys.mean()])
    _values, vectors = np.linalg.eigh(np.cov(centred))
    major = vectors[:, -1]
    return float(np.rad2deg(np.arctan2(major[1], major[0])) % 180.0)


def test_a_persons_shadow_has_a_persons_outline() -> None:
    """A cast shadow is a false alarm because it is *shaped* like the thing a
    detector looks for. The ellipse-with-lobes this drew for everything has no
    head, no neck and no legs, and so tested robustness to dark patches rather
    than to shadows of people."""
    people = [_shadow_shape("person", seed, 40.0, 1.6) for seed in range(40)]
    blobs = [_shadow_shape("foliage", seed, 40.0, 1.6) for seed in range(40)]
    people = [shape for shape in people if shape is not None]
    blobs = [shape for shape in blobs if shape is not None]
    assert len(people) >= 8 and len(blobs) >= 3

    person_solidity = np.median([_solidity(shape) for shape in people])
    blob_solidity = np.median([_solidity(shape) for shape in blobs])
    # Measured: figures 0.79, foliage blobs 0.86, vehicles 1.0.
    assert person_solidity < 0.84, f"a figure's shadow is {person_solidity:.2f} solid"
    assert blob_solidity > person_solidity + 0.03
    vehicles = [_shadow_shape("vehicle", seed, 40.0, 1.0) for seed in range(40)]
    vehicles = [shape for shape in vehicles if shape is not None]
    assert len(vehicles) >= 5
    assert np.median([_solidity(shape) for shape in vehicles]) > 0.93, "a vehicle is a box"


@pytest.mark.parametrize("azimuth", [20.0, 75.0, 140.0])
def test_upright_things_cast_their_shadows_along_the_suns_azimuth(azimuth: float) -> None:
    shapes = [_shadow_shape("person", seed, azimuth, 2.2) for seed in range(40)]
    axes = np.array([_axis_deg(shape) for shape in shapes if shape is not None])
    assert len(axes) >= 8
    error = np.abs((axes - azimuth + 90.0) % 180.0 - 90.0)
    assert error.max() < 6.0, f"shadows point up to {error.max():.0f} degrees off the sun"


def test_a_low_sun_throws_a_longer_shadow() -> None:
    def length(elongation: float) -> float:
        shapes = [_shadow_shape("person", seed, 30.0, elongation) for seed in range(40)]
        return float(np.median([shape.sum() for shape in shapes if shape is not None]))

    assert length(2.4) > 2.5 * length(0.6)


def test_one_frame_has_one_sun() -> None:
    """Each shadow used to get its own random angle, which describes a scene
    lit from several directions at once."""
    from weathermentations.diagnostics import recording

    provider = SharedDepthProvider(_GroundPlaneDepth())
    transform = CastShadows(depth_provider=provider, p=1.0, shadow_count_range=(6, 6))
    scene = np.full((200, 240, 3), 150, dtype=np.uint8)

    elevations = []
    for seed in range(12):
        transform.set_random_seed(seed)
        with recording() as log:
            transform(image=scene)
        drew = log[0]["drew"]
        assert drew["shadow_count"] == 6
        assert 0.0 <= drew["sun_azimuth_deg"] < 360.0
        elevations.append(drew["sun_elevation_deg"])
    assert min(elevations) >= 20.0 and max(elevations) <= 65.0
    assert max(elevations) - min(elevations) > 10.0, "the sun must vary between frames"

    with pytest.raises(ValueError, match="casters"):
        CastShadows(depth_provider=provider, casters=("person", "cloud"))
    with pytest.raises(ValueError, match="casters"):
        CastShadows(depth_provider=provider, casters=())
