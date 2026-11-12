"""Time every transform the way a ``DataLoader`` worker runs it, and compare checkouts.

This is the driver behind the timing table in ``docs/native_kernels.md``. It
differs from ``benchmark.py`` in what it is for. ``benchmark.py`` splits each
transform into parameter drawing and rendering and reports throughput on
synthetic scenes. This one answers "how much faster is this checkout than that
one", so it times whole calls on real photographs, on one thread, with depth
already estimated, and keeps the least of several calls.

Run it pinned to one core, each checkout in turn, then compare::

    taskset -c 2 python scripts/benchmark_single_core.py run --out now.json
    taskset -c 2 python scripts/benchmark_single_core.py run \\
        --tree ../weathermentations-original --out before.json
    python scripts/benchmark_single_core.py compare --before before.json --after now.json

Repeat the two ``run`` lines in turn, two or three times, with a different
``--out`` each, and give ``compare`` every file: it keeps each figure's least
over the runs of each side. A machine shared with other work slows some runs
and not others, and alternating the checkouts spreads that over both.

What each figure is
-------------------
For each transform, scene and seed: three calls with the same seed, the first
discarded as a warm-up, the faster of the other two kept. The figure is the
mean of that over four example scenes and two seeds, in milliseconds. OpenCV
runs on one thread and so does any BLAS numpy uses. The depth map is estimated
before timing starts and shared, as a pipeline's
:class:`~weathermentations.depth.SharedDepthProvider` shares it, so the
depth-conditioned transforms are timed without the network; the estimate's
own cost is a row of its own when the network is run here.

``RandomRain`` is timed twice, with the wet road always drawn and never
drawn, because whether it is drawn is a random choice that dominates its cost.

Depth without the network
-------------------------
The depth model needs torch and a CUDA GPU. ``save-depth FILE`` stores the
maps it makes, for every scene and size, in both precisions the library uses
(full, and half in fast mode); ``run --load-depth FILE`` reads them back in
place of the network, so the same frames can be timed on a machine with
neither, an arm64 board for one. Each checkout is given the maps its own mode would
have made: half precision for this branch in fast mode, full precision
otherwise.

``--tree`` times another checkout of this repository: its ``src`` is imported
in place of this one's, and its ``scripts/benchmark.py`` decides which
transforms there are and how they are configured. The script itself only
needs what every checkout has had: ``build_transforms`` and the depth
estimator's interface.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
SCENES = ("mountain", "street_night", "urban_road_day", "crosswalk_day")
SEEDS = (0, 1)

# The names the comparison table prints, in place of the keys of
# ``benchmark.build_transforms``.
LABELS = {
    "fog": "`RandomFog`",
    "night_fog": "`NightFog`",
    "rain_wet": "`RandomRain`, wet road drawn",
    "rain_dry": "`RandomRain`, dry road",
    "snow": "`RandomSnow`",
    "night_rain": "`NightRain`",
    "night_snow": "`NightSnow`",
    "cast_shadows": "`CastShadows`",
    "night_ir": "`NightIrMode`",
    "housing_occluders": "`HousingOccluders`",
    "spider_web": "`SpiderWeb`",
    "ground_fog": "`GroundFog`",
    "local_plumes": "`LocalPlumes`",
    "heat_shimmer": "`HeatShimmer`",
    "on_screen_overlay": "`OnScreenOverlay`",
    "camera_vibration": "`CameraVibration`",
    "focus_shift": "`FocusShift`",
    "analog_video_artifacts": "`AnalogVideoArtifacts`",
    "demosaic_artifacts": "`DemosaicArtifacts`",
    "ir_cut_filter_fault": "`IrCutFilterFault`",
    "foliage_intrusion": "`FoliageIntrusion`",
    "lens_tamper": "`LensTamper`",
    "snow_on_housing": "`SnowOnHousing`",
    "frost_on_glass": "`FrostOnGlass`",
    "ir_dome_glare": "`IrDomeGlare`",
    "night_insects": "`NightInsects`",
    "dirt_soil": "`DirtSoil`",
    "sun_flare": "`SunFlare`",
    "source_glare": "`SourceGlare`",
    "water_droplets": "`WaterDroplets`",
    "sensor_noise": "`SensorNoise`",
    "ac_flicker": "`AcFlicker`",
    "exposure_transient": "`ExposureTransient`",
    "wdr_tone_mapping": "`WdrToneMapping`",
    "denoise_artifacts": "`TemporalDenoiseArtifacts`",
    "rolling_shutter": "`RollingShutter`",
    "video_codec": "`VideoCodecArtifacts`",
    "transmission_faults": "`TransmissionFaults`",
    "depth_estimate": "the depth estimate, once per new frame",
}


def _scene_image(scene: str, size: int) -> np.ndarray:
    """An example photograph as RGB, resized to ``size`` by ``size``."""
    import cv2

    path = REPO / "docs" / "examples" / "scenes" / scene / "original.jpg"
    image = cv2.imread(str(path))
    if image is None:
        raise SystemExit(f"cannot read {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(cv2.resize(image, (size, size), interpolation=cv2.INTER_CUBIC))


def _fast_mode() -> bool:
    """Whether the checkout imported runs in fast mode; False if it has no modes."""
    try:
        from weathermentations import mode
    except ImportError:
        return False
    return bool(mode.fast())


def _stored_estimator(path: Path, precision: str) -> Any:
    """A depth estimator that hands back maps saved by ``save-depth``.

    The map is the one stored for the scene and size named in its
    ``current`` attribute, which the caller sets before each scene. Not the
    frame's bytes: another machine's OpenCV resizes a photograph to slightly
    different ones, and the depth of the scene is what is wanted.
    """
    from weathermentations.depth.estimator import DepthEstimator

    stored = np.load(path)
    maps = {
        name.split("/", 1)[1]: stored[name] for name in stored.files if name.startswith(precision)
    }

    class StoredDepth(DepthEstimator):
        """The estimator interface over the stored maps."""
        current = ""

        def estimate(self, image: np.ndarray) -> np.ndarray:
            """The map stored for the scene and size in ``current``, checked against the frame."""
            if self.current not in maps:
                raise KeyError(f"no stored depth for {self.current!r}: saved at this size?")
            depth = maps[self.current]
            if depth.shape != image.shape[:2]:
                raise ValueError(f"stored depth for {self.current!r} is {depth.shape}")
            return depth.copy()

    return StoredDepth()


def save_depth(path: Path, sizes: list[int]) -> None:
    """Estimate depth for every scene and size, in both precisions, and store it."""
    from weathermentations import mode
    from weathermentations.depth import DepthAnythingV2Estimator

    estimator = DepthAnythingV2Estimator()
    maps: dict[str, np.ndarray] = {}
    for size in sizes:
        for scene in SCENES:
            image = _scene_image(scene, size)
            for precision, fast in (("half", True), ("full", False)):
                mode.set_fast(fast)
                depth = estimator.estimate(image).astype(np.float32)
                maps[f"{precision}/{scene}/{size}"] = depth
    np.savez_compressed(path, **maps)
    print(f"saved {len(maps)} depth maps to {path}")


def run(args: argparse.Namespace) -> None:
    """Time every transform of one checkout on every scene, size and seed, and write the JSON.

    ``args.tree`` is the checkout, imported in place of this one; with
    ``args.load_depth`` its depth comes from stored maps, in the precision its
    own mode would have used, and otherwise from the network, whose own time
    is then a row of the results.
    """
    tree = Path(args.tree).resolve()
    sys.path.insert(0, str(tree / "scripts"))
    sys.path.insert(0, str(tree / "src"))
    import cv2

    cv2.setNumThreads(1)
    import benchmark as tree_benchmark

    import weathermentations

    if not Path(weathermentations.__file__).resolve().is_relative_to(tree):
        raise SystemExit(f"imported {weathermentations.__file__}, not the checkout at {tree}")
    if hasattr(weathermentations, "keep_freed_memory"):
        weathermentations.keep_freed_memory()
    from weathermentations.atmosphere import RandomRain
    from weathermentations.depth import SharedDepthProvider

    if args.load_depth:
        precision = "half" if _fast_mode() else "full"
        estimator = _stored_estimator(Path(args.load_depth), precision)
    else:
        import torch

        from weathermentations.depth import DepthAnythingV2Estimator

        torch.set_num_threads(1)
        estimator = DepthAnythingV2Estimator()

    provider = SharedDepthProvider(estimator)
    times: dict[str, list[float]] = {}
    for size in args.sizes:
        transforms = tree_benchmark.build_transforms(provider)
        transforms.pop("rain", None)
        transforms["rain_wet"] = RandomRain(depth_provider=provider, p=1.0, wetness_probability=1.0)
        transforms["rain_dry"] = RandomRain(depth_provider=provider, p=1.0, wetness_probability=0.0)
        if args.only:
            transforms = {name: t for name, t in transforms.items() if name in args.only}
        for scene in SCENES:
            image = _scene_image(scene, size)
            estimator.current = f"{scene}/{size}"
            if not args.load_depth:
                times.setdefault(f"depth_estimate/{size}", []).append(
                    _least(lambda image=image: _estimate_synchronised(estimator, image), 5)
                )
            for name, transform in transforms.items():
                for seed in SEEDS:

                    def call(
                        transform: Any = transform, seed: int = seed, image: np.ndarray = image
                    ) -> None:
                        """One call of the transform on this scene with this seed."""
                        transform.set_random_seed(seed)
                        transform(image=image)

                    call()
                    times.setdefault(f"{name}/{size}", []).append(_least(call, 2))
            print(f"{size} px {scene} done", file=sys.stderr)
    result = {
        "tree": str(tree),
        "fast_mode": _fast_mode(),
        "ms": {key: float(np.mean(values)) for key, values in times.items()},
    }
    Path(args.out).write_text(json.dumps(result, indent=1) + "\n")
    print(f"wrote {args.out}")


def _estimate_synchronised(estimator: Any, image: np.ndarray) -> None:
    """Estimate depth for ``image``, with the GPU synchronised before and after."""
    import torch

    torch.cuda.synchronize()
    estimator.estimate(image)
    torch.cuda.synchronize()


def _least(call: Any, repeats: int) -> float:
    """The least of ``repeats`` timed calls, in milliseconds."""
    best = math.inf
    for _ in range(repeats):
        start = time.perf_counter()
        call()
        best = min(best, time.perf_counter() - start)
    return best * 1e3


def _least_over_runs(paths: list[str]) -> dict[str, float]:
    """Each figure's least over several result files, for the figures they all hold."""
    runs = [json.loads(Path(path).read_text())["ms"] for path in paths]
    keys = set(runs[0]).intersection(*runs[1:])
    return {key: min(run[key] for run in runs) for key in keys}


def compare(args: argparse.Namespace) -> None:
    """Print the timing table of ``docs/native_kernels.md``: before, after, ratio."""
    before = _least_over_runs(args.before)
    after = _least_over_runs(args.after)
    sizes = sorted({int(key.split("/")[1]) for key in before if key in after})
    if len(sizes) < 1:
        raise SystemExit("the two sides have no transform and size in common")
    names = sorted({key.split("/")[0] for key in before if key in after})
    small, large = sizes[0], sizes[-1]

    def ratio(name: str, size: int) -> float:
        """How many times faster ``after`` is than ``before`` for a transform at a size."""
        return before[f"{name}/{size}"] / after[f"{name}/{size}"]

    print(f"| Transform | {small}: before | after | | {large}: before | after | |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for name in sorted(names, key=lambda n: -ratio(n, large)):
        cells = []
        for size in (small, large):
            cells += [
                f"{before[f'{name}/{size}']:.1f}",
                f"{after[f'{name}/{size}']:.1f}",
                f"{ratio(name, size):.1f}x",
            ]
        print(f"| {LABELS.get(name, name)} | " + " | ".join(cells) + " |")
    transforms = [name for name in names if name != "depth_estimate"]
    count = len(transforms)
    means = [
        math.exp(np.mean([math.log(ratio(n, size)) for n in transforms])) for size in (small, large)
    ]
    totals = [
        (
            sum(before[f"{n}/{size}"] for n in transforms),
            sum(after[f"{n}/{size}"] for n in transforms),
        )
        for size in (small, large)
    ]
    print(
        f"| **Geometric mean of the {count} transforms** | | | **{means[0]:.2f}x** "
        f"| | | **{means[1]:.2f}x** |"
    )
    print(
        f"| **All {count} once each** | {totals[0][0]:.0f} | {totals[0][1]:.0f} | "
        f"**{totals[0][0] / totals[0][1]:.2f}x** | {totals[1][0]:.0f} | {totals[1][1]:.0f} | "
        f"**{totals[1][0] / totals[1][1]:.2f}x** |"
    )


def main() -> None:
    """Parse the command line and run ``run``, ``save-depth`` or ``compare``."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)

    timing = commands.add_parser("run", help="time every transform of one checkout")
    timing.add_argument(
        "--tree", default=str(REPO), help="the checkout to time (default: this one)"
    )
    timing.add_argument("--sizes", type=int, nargs="+", default=[512, 1024])
    timing.add_argument("--only", nargs="+", default=None, help="time only these transforms")
    timing.add_argument("--out", required=True, help="JSON file to write")
    timing.add_argument("--load-depth", default=None, help="depth maps stored by save-depth")

    saving = commands.add_parser("save-depth", help="estimate depth for every scene and store it")
    saving.add_argument("file")
    saving.add_argument("--sizes", type=int, nargs="+", default=[512, 1024])

    table = commands.add_parser("compare", help="print the before/after table")
    table.add_argument("--before", nargs="+", required=True)
    table.add_argument("--after", nargs="+", required=True)

    args = parser.parse_args()
    if args.command == "run":
        run(args)
    elif args.command == "save-depth":
        sys.path.insert(0, str(REPO / "src"))
        save_depth(Path(args.file), args.sizes)
    else:
        compare(args)


if __name__ == "__main__":
    main()
