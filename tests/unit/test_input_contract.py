"""What every transform accepts, and that it behaves the same for all of it.

Each transform's docstring says ``Image types: uint8, float32``. That was true
of the linear-light effects, which convert through one shared helper, and
false of the ones that work on 8-bit code values directly:

* ``WaterDroplets`` added its highlight and rim shading as 8-bit constants to
  whatever it was given, so a float frame in ``[0, 1]`` came back with values
  up to 124.
* ``RandomFog`` divided the estimated airlight by 255 whether or not the image
  was already in ``[0, 1]``, so on float input the fog was black.

And nothing said what the transforms do *not* accept. A greyscale frame got as
far as the renderer, where five transforms returned an image of a different
shape rather than failing.

So the contract is asserted for every transform rather than per transform:
the float path must agree with the uint8 path, and unsupported input must be
refused before anything is rendered.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from weathermentations.atmosphere import (
    NightFog,
    NightRain,
    NightSnow,
    RandomFog,
    RandomRain,
    RandomSnow,
)
from weathermentations.camera import (
    AcFlicker,
    ExposureTransient,
    NightIrMode,
    RollingShutter,
    SensorNoise,
    TemporalDenoiseArtifacts,
    TransmissionFaults,
    VideoCodecArtifacts,
    WdrToneMapping,
)
from weathermentations.camera.codec import ffmpeg_path
from weathermentations.depth.estimator import DepthEstimator
from weathermentations.depth.provider import SharedDepthProvider
from weathermentations.lens import DirtSoil, HousingOccluders, SunFlare, WaterDroplets
from weathermentations.scene import CastShadows


class _GroundPlane(DepthEstimator):
    def estimate(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        rows = np.arange(height, dtype=np.float32).reshape(height, 1)
        depth = np.clip(1.5 * 300.0 / np.maximum(rows - height / 2.0, 1e-3), 0.0, 50.0)
        return np.broadcast_to(depth / 50.0, (height, width)).astype(np.float32)


_PROVIDER = SharedDepthProvider(_GroundPlane())

_needs_ffmpeg = pytest.mark.skipif(ffmpeg_path() is None, reason="needs the ffmpeg binary")

BUILDERS: dict[str, Callable[[], Any]] = {
    "RandomFog": lambda: RandomFog(depth_provider=_PROVIDER, p=1.0),
    "NightFog": lambda: NightFog(depth_provider=_PROVIDER, p=1.0),
    "RandomRain": lambda: RandomRain(depth_provider=_PROVIDER, p=1.0, wetness_probability=1.0),
    "RandomSnow": lambda: RandomSnow(
        depth_provider=_PROVIDER, p=1.0, accumulation_probability=1.0
    ),
    "NightRain": lambda: NightRain(p=1.0),
    "NightSnow": lambda: NightSnow(p=1.0),
    "CastShadows": lambda: CastShadows(depth_provider=_PROVIDER, p=1.0),
    "NightIrMode": lambda: NightIrMode(depth_provider=_PROVIDER, p=1.0),
    "WaterDroplets": lambda: WaterDroplets(p=1.0),
    "DirtSoil": lambda: DirtSoil(p=1.0),
    "SunFlare": lambda: SunFlare(p=1.0),
    "HousingOccluders": lambda: HousingOccluders(p=1.0),
    "AcFlicker": lambda: AcFlicker(p=1.0),
    "RollingShutter": lambda: RollingShutter(p=1.0),
    "SensorNoise": lambda: SensorNoise(p=1.0),
    "ExposureTransient": lambda: ExposureTransient(p=1.0),
    "WdrToneMapping": lambda: WdrToneMapping(p=1.0),
    "TemporalDenoiseArtifacts": lambda: TemporalDenoiseArtifacts(p=1.0),
    "VideoCodecArtifacts": lambda: VideoCodecArtifacts(p=1.0),
    "TransmissionFaults": lambda: TransmissionFaults(p=1.0),
}
ALL = [
    pytest.param(name, marks=_needs_ffmpeg)
    if name in ("VideoCodecArtifacts", "TransmissionFaults")
    else name
    for name in BUILDERS
]
# Every effect that models light. `RollingShutter` only moves pixels, so it is
# indifferent to what they hold and is held to a wider contract below.
RGB_ONLY = [param for param in ALL if param != "RollingShutter"]


def _scene() -> np.ndarray:
    """Sky over textured ground with a saturated object: something for every
    effect to act on, including the ones that read an illuminant."""
    rng = np.random.default_rng(0)
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    image[:45] = (190, 205, 225)
    image[45:] = (70, 74, 66)
    image[70:95, 40:100] = (150, 40, 35)
    noise = rng.integers(-12, 13, image.shape, dtype=np.int16)
    return np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def _run(name: str, image: np.ndarray) -> np.ndarray:
    transform = BUILDERS[name]()
    transform.set_random_seed(3)
    result: np.ndarray = transform(image=image)["image"]
    return result


def test_every_exported_transform_is_covered() -> None:
    """A transform added without an entry here would skip the whole contract."""
    import weathermentations.atmosphere as atmosphere
    import weathermentations.camera as camera
    import weathermentations.lens as lens
    import weathermentations.scene as scene

    exported = {
        name for module in (atmosphere, camera, lens, scene) for name in module.__all__
    }
    # SourceGlare acts only on lights in the frame and this file's scene has
    # none, so its contract (dtype parity included) is in test_night_lights.py.
    assert exported == set(BUILDERS) | {"SourceGlare"}


@pytest.mark.parametrize("name", ALL)
def test_float32_input_gives_the_same_picture_as_uint8(name: str) -> None:
    image = _scene()
    from_uint8 = _run(name, image)
    from_float = _run(name, image.astype(np.float32) / 255.0)

    assert from_float.dtype == np.float32
    assert from_float.shape == image.shape
    assert float(from_float.min()) >= 0.0 and float(from_float.max()) <= 1.0, (
        f"float output left [0, 1]: [{from_float.min():.3f}, {from_float.max():.3f}]"
    )
    # One 8-bit level of mean difference is the quantisation the uint8 path
    # goes through and the float path does not; the bugs this guards against
    # measured 134 levels (fog) and 3576 (droplets).
    difference = np.abs(from_float * 255.0 - from_uint8.astype(np.float32)).mean()
    assert difference < 1.5, f"float and uint8 paths disagree by {difference:.2f} levels"
    assert not np.array_equal(from_uint8, image), "transform did nothing; parity is vacuous"


@pytest.mark.parametrize("name", ALL)
def test_input_is_not_modified_and_output_keeps_shape_and_dtype(name: str) -> None:
    image = _scene()
    original = image.copy()
    out = _run(name, image)
    np.testing.assert_array_equal(image, original)
    assert out.shape == image.shape and out.dtype == image.dtype


@pytest.mark.parametrize("name", RGB_ONLY)
@pytest.mark.parametrize(
    "bad",
    [
        pytest.param(lambda image: image[..., 0].copy(), id="greyscale-2d"),
        pytest.param(lambda image: image[..., :1].copy(), id="greyscale-1ch"),
        pytest.param(lambda image: np.dstack([image, image[..., :1]]), id="rgba"),
    ],
)
def test_images_that_are_not_rgb_are_refused(name: str, bad: Callable[..., np.ndarray]) -> None:
    """Refused, not rendered: several transforms used to return a frame of a
    different shape for these, which a training loop would stack and train on."""
    with pytest.raises(ValueError, match=r"\(H, W, 3\)"):
        _run(name, bad(_scene()))


@pytest.mark.parametrize("name", RGB_ONLY)
def test_unsupported_dtypes_are_refused(name: str) -> None:
    """A 16-bit image used to be read as if full scale were 255 and came back
    clipped to white."""
    sixteen_bit = _scene().astype(np.uint16) * 257
    with pytest.raises(TypeError, match="uint8"):
        _run(name, sixteen_bit)


def test_a_transform_that_does_not_fire_does_not_inspect_its_input() -> None:
    """``p=0`` must stay a true no-op, so a pipeline can carry transforms that
    are switched off without them vetoing input they will never touch."""
    grey = _scene()[..., 0].copy()
    transform = DirtSoil(p=0.0)
    assert transform(image=grey)["image"] is grey


@pytest.mark.parametrize(
    "variant",
    [
        pytest.param(lambda image: image[..., 0].copy(), id="greyscale-2d"),
        pytest.param(lambda image: image[..., :1].copy(), id="greyscale-1ch"),
        pytest.param(lambda image: image.astype(np.uint16) * 257, id="uint16"),
    ],
)
def test_rolling_shutter_accepts_whatever_can_be_resampled(
    variant: Callable[..., np.ndarray],
) -> None:
    """Greyscale is an infrared camera's native output, and a shear is as
    meaningful on it as on colour. The single-channel case used to come back
    as ``(H, W)``: OpenCV drops a trailing axis of one."""
    image = variant(_scene())
    transform = RollingShutter(p=1.0, pan_shift_range_px=(15.0, 15.0))
    transform.set_random_seed(0)
    out = transform(image=image)["image"]
    assert out.shape == image.shape and out.dtype == image.dtype
    assert not np.array_equal(out, image)
