"""DemosaicArtifacts: colour that is not in the scene, where the detail is as fine as the pixels."""

from __future__ import annotations

import numpy as np

from surmentations.camera import DemosaicArtifacts
from surmentations.camera.demosaic import smooth_chroma, through_mosaic


def _stripes(period: int, size: int = 96) -> np.ndarray:
    """Grey vertical stripes ``period`` pixels to a cycle, linear light."""
    across = (np.arange(size) % period < period / 2).astype(np.float32)
    plane = np.broadcast_to(0.2 + 0.6 * across[None, :], (size, size))
    return np.ascontiguousarray(np.repeat(plane[..., None], 3, axis=2))


def _false_colour(image: np.ndarray) -> float:
    """Mean spread between the channels: zero for a picture that is grey."""
    inner = image[4:-4, 4:-4]
    return float((inner.max(axis=2) - inner.min(axis=2)).mean())


def _camera(**settings: object) -> DemosaicArtifacts:
    """A sharp lens on a sensor at the picture's resolution, unfiltered, unless changed."""
    arguments: dict[str, object] = {
        "optical_blur_range_px": (0.0, 0.0),
        "sensor_scale_range": (1.0, 1.0),
        "chroma_filter_range_px": (0, 0),
        **settings,
    }
    return DemosaicArtifacts(p=1.0, **arguments)


def _through(camera: DemosaicArtifacts, image: np.ndarray, seed: int = 0) -> np.ndarray:
    """A float frame through a camera, with a seed."""
    camera.set_random_seed(seed)
    out: np.ndarray = camera(image=image)["image"]
    return out


def test_a_grey_scene_with_no_fine_detail_stays_grey() -> None:
    flat = np.full((64, 64, 3), 0.4, dtype=np.float32)
    assert _false_colour(through_mosaic(flat, (0, 0))) < 1e-6
    # Coarse stripes are coloured only along their edges, a pixel or two wide.
    assert _false_colour(through_mosaic(_stripes(32), (0, 0))) < 0.05


def test_detail_as_fine_as_the_pixels_comes_out_coloured() -> None:
    fine, coarse = _stripes(2), _stripes(16)
    assert _false_colour(through_mosaic(fine, (0, 0))) > 0.2
    assert _false_colour(through_mosaic(fine, (0, 0))) > 5.0 * _false_colour(
        through_mosaic(coarse, (0, 0))
    )


def test_which_colour_depends_on_where_the_pattern_falls() -> None:
    fine = _stripes(2)
    one = through_mosaic(fine, (0, 0))[8:-8, 8:-8].reshape(-1, 3).mean(axis=0)
    other = through_mosaic(fine, (0, 1))[8:-8, 8:-8].reshape(-1, 3).mean(axis=0)
    # Shifted by a pixel, the red sites see the other half of each stripe.
    assert (one[0] - one[2]) * (other[0] - other[2]) < 0.0


def test_the_lens_s_blur_is_what_limits_it() -> None:
    fine = _stripes(3)
    sharp = _false_colour(_through(_camera(), fine))
    soft = _false_colour(_through(_camera(optical_blur_range_px=(1.2, 1.2)), fine))
    assert soft < 0.4 * sharp


def test_a_coarser_sensor_makes_wider_fringes() -> None:
    # Stripes two pixels wide: an edge every other pixel for a sensor at this
    # resolution, and as fine as the pixels of one with half as many.
    stripes = _stripes(4, size=128)
    native = _false_colour(_through(_camera(), stripes))
    coarse = _false_colour(_through(_camera(sensor_scale_range=(2.0, 2.0)), stripes))
    assert coarse > 1.5 * native


def test_the_processor_s_filter_hides_fringes() -> None:
    recorded = through_mosaic(_stripes(3), (0, 0))
    assert _false_colour(smooth_chroma(recorded, 2)) < 0.6 * _false_colour(recorded)
    assert np.array_equal(smooth_chroma(recorded, 0), recorded)
    # Green is what the differences are taken against, and is not touched.
    assert np.array_equal(smooth_chroma(recorded, 2)[..., 1], recorded[..., 1])


def test_the_frame_keeps_its_size_and_level() -> None:
    import cv2

    # Smooth, as a photograph is: averaging noise in linear light would
    # raise its level by itself.
    rng = np.random.default_rng(0)
    small = rng.integers(0, 256, (13, 17, 3), dtype=np.uint8)
    image = cv2.resize(small, (131, 97), interpolation=cv2.INTER_CUBIC)
    for seed in range(4):
        camera = DemosaicArtifacts(p=1.0)
        camera.set_random_seed(seed)
        out = camera(image=image)["image"]
        assert out.shape == image.shape and out.dtype == image.dtype
        assert abs(float(out.mean()) - float(image.mean())) < 8.0
