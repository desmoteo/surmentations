"""Throughput benchmark for every transform in the library.

Run with::

    .venv/bin/python scripts/benchmark.py
    .venv/bin/python scripts/benchmark.py --sizes 512 1024 --repeats 20
    .venv/bin/python scripts/benchmark.py --only rain fog --csv results.csv

What it measures, and why it is split that way
----------------------------------------------
These transforms run per-sample inside a ``DataLoader`` worker, so the number
that matters is images/second *per worker*. But a single total would hide the
one thing worth knowing about this library's cost model: for the
depth-conditioned effects (fog, rain, snow) almost all the time goes into
monocular depth estimation, not into rendering. So each transform is timed in
two halves:

* **params** — ``get_params_dependent_on_data``, which is where the depth
  estimator runs and where per-particle sampling happens.
* **apply** — the actual rendering onto the image.

Depth is cached per ``SharedDepthProvider`` and keyed on the *identity* of the
image array, so benchmarking the same array repeatedly would hit that cache
every time after the first call and report a depth cost of ~zero. That is a
real speed-up when several depth-conditioned transforms are chained in one
``Compose``, but it is not what a fresh sample costs. Both are therefore
reported: ``cold`` feeds a distinct array each iteration (a new sample),
``warm`` reuses one (a second depth-conditioned transform in the same
pipeline, reusing the first one's depth).
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from weathermentations.atmosphere import (  # noqa: E402
    NightFog,
    NightRain,
    NightSnow,
    RandomFog,
    RandomRain,
    RandomSnow,
)
from weathermentations.atmosphere.night_insects import NightInsects  # noqa: E402
from weathermentations.camera import (  # noqa: E402
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
from weathermentations.camera.analog import AnalogVideoArtifacts  # noqa: E402
from weathermentations.camera.codec import ffmpeg_path  # noqa: E402
from weathermentations.camera.demosaic import DemosaicArtifacts  # noqa: E402
from weathermentations.camera.ir_cut import IrCutFilterFault  # noqa: E402
from weathermentations.camera.overlay import OnScreenOverlay  # noqa: E402
from weathermentations.camera.vibration import CameraVibration  # noqa: E402
from weathermentations.depth import (  # noqa: E402
    DepthAnythingV2Estimator,
    SharedDepthProvider,
)
from weathermentations.lens import (  # noqa: E402
    DirtSoil,
    HousingOccluders,
    SourceGlare,
    SpiderWeb,
    SunFlare,
    WaterDroplets,
)
from weathermentations.lens.dome_glare import IrDomeGlare  # noqa: E402
from weathermentations.lens.focus import FocusShift  # noqa: E402
from weathermentations.lens.foliage import FoliageIntrusion  # noqa: E402
from weathermentations.lens.frost import FrostOnGlass  # noqa: E402
from weathermentations.lens.snow_cap import SnowOnHousing  # noqa: E402
from weathermentations.lens.tamper import LensTamper  # noqa: E402
from weathermentations.scene import CastShadows  # noqa: E402

DEPTH_FREE = (
    "water_droplets",
    "dirt_soil",
    "sun_flare",
    "housing_occluders",
    "ac_flicker",
    "rolling_shutter",
    "sensor_noise",
    "exposure_transient",
    "wdr_tone_mapping",
    "denoise_artifacts",
    "video_codec",
    "transmission_faults",
    "night_rain",
    "night_snow",
)
DEPTH_CONDITIONED = ("fog", "rain", "snow", "night_ir", "cast_shadows")


@dataclass
class Result:
    """One transform at one size: the median time to draw its parameters and to render."""
    name: str
    size: int
    params_ms: float
    apply_ms: float

    @property
    def total_ms(self) -> float:
        """Drawing and rendering together, in milliseconds."""
        return self.params_ms + self.apply_ms

    @property
    def images_per_second(self) -> float:
        """How many frames one worker gets through a second at this cost."""
        return 1000.0 / self.total_ms if self.total_ms > 0 else float("inf")


def build_transforms(depth_provider: SharedDepthProvider | None) -> dict[str, object]:
    """Every transform of the library at its defaults, keyed by a short name.

    The two codec transforms only where there is an ffmpeg binary; the
    depth-conditioned ones only with a ``depth_provider``. This is also what
    ``benchmark_single_core.py`` times, in whichever checkout it is given.
    """
    transforms: dict[str, object] = {
        "water_droplets": WaterDroplets(p=1.0),
        "dirt_soil": DirtSoil(p=1.0),
        "sun_flare": SunFlare(p=1.0),
        "housing_occluders": HousingOccluders(p=1.0),
        "source_glare": SourceGlare(p=1.0),
        "ac_flicker": AcFlicker(p=1.0),
        "rolling_shutter": RollingShutter(p=1.0),
        "sensor_noise": SensorNoise(p=1.0),
        "exposure_transient": ExposureTransient(p=1.0),
        "wdr_tone_mapping": WdrToneMapping(p=1.0),
        "denoise_artifacts": TemporalDenoiseArtifacts(p=1.0),
        # Lit by the camera rather than the scene, so it needs no depth.
        "night_rain": NightRain(p=1.0),
        "night_snow": NightSnow(p=1.0),
        "spider_web": SpiderWeb(p=1.0),
        "on_screen_overlay": OnScreenOverlay(p=1.0),
        "camera_vibration": CameraVibration(p=1.0),
        "focus_shift": FocusShift(p=1.0),
        "analog_video_artifacts": AnalogVideoArtifacts(p=1.0),
        "demosaic_artifacts": DemosaicArtifacts(p=1.0),
        "ir_cut_filter_fault": IrCutFilterFault(p=1.0),
        "foliage_intrusion": FoliageIntrusion(p=1.0),
        "lens_tamper": LensTamper(p=1.0),
        "snow_on_housing": SnowOnHousing(p=1.0),
        "frost_on_glass": FrostOnGlass(p=1.0),
        "ir_dome_glare": IrDomeGlare(p=1.0),
        "night_insects": NightInsects(p=1.0),
    }
    # Two ffmpeg subprocesses per call, so it is both the slowest transform
    # here and the one whose cost is least about this library.
    if ffmpeg_path() is not None:
        transforms["video_codec"] = VideoCodecArtifacts(p=1.0)
        transforms["transmission_faults"] = TransmissionFaults(p=1.0)
    if depth_provider is not None:
        transforms |= {
            "fog": RandomFog(depth_provider=depth_provider, p=1.0),
            "rain": RandomRain(depth_provider=depth_provider, p=1.0),
            "snow": RandomSnow(depth_provider=depth_provider, p=1.0),
            "night_ir": NightIrMode(depth_provider=depth_provider, p=1.0),
            "night_fog": NightFog(depth_provider=depth_provider, p=1.0),
            "cast_shadows": CastShadows(depth_provider=depth_provider, p=1.0),
        }
    return transforms


def make_scene(size: int, rng: np.random.Generator) -> np.ndarray:
    """A scene with real structure — flat noise would make the depth model and
    the bright-pass/airlight estimators behave unlike they do on photographs."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / size
    sky = np.stack([0.45 + 0.3 * yy, 0.6 + 0.25 * yy, 0.95 - 0.1 * yy], axis=-1)
    ground = np.stack([0.25 + 0.2 * yy, 0.35 + 0.1 * yy, 0.2 + 0.1 * yy], axis=-1)
    horizon = (yy > 0.55)[..., None]
    scene = np.where(horizon, ground, sky)
    scene += 0.04 * rng.standard_normal(scene.shape)
    # A few hard edges so blurs and gradients have something to bite on.
    scene[int(size * 0.6) : int(size * 0.64), :] = 0.12
    scene[:, int(size * 0.45) : int(size * 0.47)] = 0.8
    return np.clip(scene * 255, 0, 255).astype(np.uint8)


def timed(fn: Callable[[], object], repeats: int) -> float:
    """Median wall-clock milliseconds over `repeats` calls."""
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples)


def scene_source(scenes: list[np.ndarray], reuse: bool) -> Iterator[np.ndarray]:
    """Yield a fresh array each call, or the same one, forever."""
    index = 0
    while True:
        yield scenes[0] if reuse else scenes[index % len(scenes)]
        index += 1


def benchmark(
    name: str,
    transform: object,
    scenes: list[np.ndarray],
    size: int,
    repeats: int,
    reuse_scene: bool,
) -> Result:
    """Time one transform at one size: drawing its parameters, and rendering.

    Each half is the median of ``repeats`` calls after a warm-up. Drawing is
    timed on a fresh array each call unless ``reuse_scene``, so that the
    depth estimate is paid as a new frame pays it.
    """
    transform.set_random_seed(0)  # type: ignore[attr-defined]
    source = scene_source(scenes, reuse_scene)
    shape_params = {"shape": scenes[0].shape}

    # Warm up: first call pays for lazy model init, kernel allocation and
    # any one-off CUDA context setup, none of which is per-image cost.
    warm = next(source)
    params = transform.get_params_dependent_on_data(shape_params, {"image": warm})  # type: ignore[attr-defined]
    transform.apply(warm, **params)  # type: ignore[attr-defined]

    params_ms = timed(
        lambda: transform.get_params_dependent_on_data(  # type: ignore[attr-defined]
            shape_params, {"image": next(source)}
        ),
        repeats,
    )
    # Render timing uses one fixed parameter set so it measures rendering
    # only, not the sampling that produced it.
    fixed_params = transform.get_params_dependent_on_data(shape_params, {"image": warm})  # type: ignore[attr-defined]
    apply_ms = timed(lambda: transform.apply(warm, **fixed_params), repeats)  # type: ignore[attr-defined]

    return Result(name=name, size=size, params_ms=params_ms, apply_ms=apply_ms)


def print_table(results: list[Result], title: str) -> None:
    """Print results as a table: transform, size, the two halves, the total and frames a second."""
    print(f"\n{title}")
    header = (
        f"{'transform':<16}{'size':>7}{'params ms':>12}"
        f"{'apply ms':>11}{'total ms':>11}{'img/s':>9}"
    )
    print(header)
    print("-" * 66)
    for r in results:
        print(
            f"{r.name:<16}{r.size:>7}{r.params_ms:>12.2f}{r.apply_ms:>11.2f}"
            f"{r.total_ms:>11.2f}{r.images_per_second:>9.1f}"
        )


def main() -> int:
    """Parse the command line, time every transform asked for, and print or save the results."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=[512, 1024], help="square image sizes to test"
    )
    parser.add_argument("--repeats", type=int, default=10, help="timed calls per measurement")
    parser.add_argument("--only", nargs="+", default=None, help="benchmark only these transforms")
    parser.add_argument("--csv", type=Path, default=None, help="also write results to this CSV")
    parser.add_argument(
        "--no-depth",
        action="store_true",
        help="skip the depth-conditioned transforms (no GPU needed)",
    )
    args = parser.parse_args()

    depth_provider = None
    if not args.no_depth:
        try:
            depth_provider = SharedDepthProvider(DepthAnythingV2Estimator())
        except RuntimeError as exc:
            print(f"Depth-conditioned transforms skipped: {exc}\n")

    rng = np.random.default_rng(0)
    results: list[Result] = []

    for size in args.sizes:
        # Several distinct arrays so the "cold" runs cannot hit the depth
        # cache, which keys on array identity.
        scenes = [make_scene(size, rng) for _ in range(max(4, args.repeats))]
        transforms = build_transforms(depth_provider)
        for name, transform in transforms.items():
            if args.only and name not in args.only:
                continue
            reuse = name not in DEPTH_CONDITIONED
            results.append(benchmark(name, transform, scenes, size, args.repeats, reuse))

    print_table(results, "Per-transform cost (fresh sample; depth computed per image)")

    if depth_provider is not None and (not args.only or set(args.only) & set(DEPTH_CONDITIONED)):
        warm_results = []
        for size in args.sizes:
            scenes = [make_scene(size, rng) for _ in range(max(4, args.repeats))]
            transforms = build_transforms(depth_provider)
            for name in DEPTH_CONDITIONED:
                if args.only and name not in args.only:
                    continue
                warm_results.append(
                    benchmark(name, transforms[name], scenes, size, args.repeats, reuse_scene=True)
                )
        print_table(
            warm_results,
            "Depth-conditioned, depth already cached (2nd+ such transform in one Compose)",
        )

        for size in args.sizes:
            cold = [r for r in results if r.size == size and r.name in DEPTH_CONDITIONED]
            warm = [r for r in warm_results if r.size == size]
            if cold and warm:
                saved = statistics.median(
                    [c.total_ms for c in cold]
                ) - statistics.median([w.total_ms for w in warm])
                print(
                    f"\n  At {size}px, sharing one depth provider saves ~{saved:.0f} ms "
                    f"per extra depth-conditioned transform in the pipeline."
                )

    if args.csv:
        with args.csv.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["transform", "size", "params_ms", "apply_ms", "total_ms", "img_per_s"])
            for r in results:
                writer.writerow(
                    [
                        r.name,
                        r.size,
                        f"{r.params_ms:.3f}",
                        f"{r.apply_ms:.3f}",
                        f"{r.total_ms:.3f}",
                        f"{r.images_per_second:.2f}",
                    ]
                )
        print(f"\nWrote {args.csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
