"""Time the ready-made pipelines and save what they produce, for inspection.

Run with::

    PYTHONPATH=. .venv/bin/python scripts/benchmark_pipelines.py
    PYTHONPATH=. .venv/bin/python scripts/benchmark_pipelines.py --runs 40 --size 1024

For each pipeline in ``weathermentations.presets`` it renders the reference
photograph N times, records how long each call took and which effects actually
fired, and writes every frame out. Two numbers matter and they are not the
same one: the **median** says what a typical frame costs, and the **p90** says
what a ``DataLoader`` worker has to keep up with — cost here depends on which
effects the draw selected, so the distribution is wide and skewed by design.

The per-frame effect list is saved alongside, because a timing on its own
cannot tell you whether a slow frame was slow for a good reason.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from weathermentations.diagnostics import recording  # noqa: E402
from weathermentations.presets import surveillance, surveillance_lite  # noqa: E402

REFERENCE = Path(__file__).resolve().parent.parent / "docs" / "examples" / "reference.webp"


def _scene(size: int) -> np.ndarray:
    image = cv2.imread(str(REFERENCE))
    if image is None:  # pragma: no cover - the asset ships with the repo
        raise SystemExit(f"reference image not found at {REFERENCE}")
    return cv2.cvtColor(cv2.resize(image, (size, size)), cv2.COLOR_BGR2RGB)


def _contact_sheet(frames: list[np.ndarray], columns: int) -> np.ndarray:
    """Tiles at native resolution, deliberately.

    Downscaling the tiles hides exactly the effects that are hardest to judge:
    a rain streak is one pixel wide, so 2:1 area averaging removes it
    entirely, and snow flakes go the same way. A sheet built that way said
    "rain and snow are missing" about frames that contained a thousand
    streaks. The file is large; that is the cost of it being honest.
    """
    tile = frames[0].shape[0]
    rows = (len(frames) + columns - 1) // columns
    sheet = np.zeros((rows * tile, columns * tile, 3), dtype=np.uint8)
    for index, frame in enumerate(frames):
        y, x = divmod(index, columns)
        sheet[y * tile : (y + 1) * tile, x * tile : (x + 1) * tile] = frame
    return sheet


def _run(
    name: str, transform: Any, scene: np.ndarray, runs: int, out: Path, seed: int
) -> dict[str, Any]:
    directory = out / name
    directory.mkdir(parents=True, exist_ok=True)

    # Seeded **once**, then left to run. Re-seeding before every frame is the
    # obvious thing to write and it silently biases the result: a fresh seed
    # correlates the per-transform probability checks, so frames come out with
    # nearly every effect or nearly none. Measured on `surveillance_lite`,
    # re-seeding each call gives 27% empty frames and a variance of 7.25
    # effects per frame, against 8% and 1.81 when seeded once. A training run
    # seeds its Compose once and streams samples through it, so that is what
    # is timed here.
    transform.set_random_seed(seed)
    # One untimed call: the first invocation of a depth-conditioned pipeline
    # pays for weights and CUDA context, a startup cost no training step sees.
    transform(image=scene.copy())

    timings: list[float] = []
    manifest: list[dict[str, Any]] = []
    frames: list[np.ndarray] = []

    for index in range(runs):
        image = scene.copy()
        start = time.perf_counter()
        with recording() as log:
            result = transform(image=image)["image"]
        elapsed = (time.perf_counter() - start) * 1000.0

        timings.append(elapsed)
        frames.append(result)
        effects = sorted({record["t"] for record in log})
        delta = float(np.abs(result.astype(np.float32) - scene.astype(np.float32)).mean())
        manifest.append(
            {"frame": index, "ms": round(elapsed, 2), "mean_abs_delta": round(delta, 2),
             "effects": effects}
        )
        cv2.imwrite(
            str(directory / f"{index:02d}.png"), cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        )

    cv2.imwrite(
        str(out / f"{name}_sheet.png"),
        cv2.cvtColor(_contact_sheet(frames, columns=5), cv2.COLOR_RGB2BGR),
    )
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=1))

    ordered = sorted(timings)
    summary = {
        "pipeline": name,
        "runs": runs,
        "median_ms": round(statistics.median(ordered), 1),
        "mean_ms": round(statistics.fmean(ordered), 1),
        "p90_ms": round(ordered[int(0.9 * (len(ordered) - 1))], 1),
        "min_ms": round(ordered[0], 1),
        "max_ms": round(ordered[-1], 1),
        "effects_per_frame": round(statistics.fmean(len(m["effects"]) for m in manifest), 2),
        "unchanged_frames": sum(m["mean_abs_delta"] < 0.5 for m in manifest),
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs", type=int, default=50, help="renders per pipeline")
    parser.add_argument("--size", type=int, default=512, help="square frame size")
    parser.add_argument("--out", type=Path, default=Path("pipeline_samples"))
    parser.add_argument("--seed", type=int, default=7, help="seed, set once per pipeline")
    parser.add_argument(
        "--no-weather-demo",
        action="store_true",
        help="skip the weather-forced variant (the default priors put weather on 12%% of "
        "frames, so a 20-frame sample shows almost none)",
    )
    parser.add_argument("--no-depth", action="store_true", help="skip the CUDA pipeline")
    args = parser.parse_args()

    scene = _scene(args.size)
    args.out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out / "00_reference.png"), cv2.cvtColor(scene, cv2.COLOR_RGB2BGR))

    builders: list[tuple[str, Any]] = [("surveillance_lite", surveillance_lite())]
    if not args.no_depth:
        from weathermentations.depth import DepthAnythingV2Estimator, SharedDepthProvider

        provider = SharedDepthProvider(DepthAnythingV2Estimator())
        builders.append(("surveillance", surveillance(provider)))
        if not args.no_weather_demo:
            # The shipped priors are climatological: 12% of frames carry
            # weather, so a 20-frame sample typically shows one rain and no
            # snow at all, which reads as "the weather is broken". This
            # variant raises the three states so they can actually be looked
            # at. It is a demo of the effects, not a recommended setting.
            builders.append(
                (
                    "surveillance_weather",
                    surveillance(provider, rain=0.33, snow=0.32, fog=0.32, ir_night=0.0,
                                 lit_night=0.0, day=1.0),
                )
            )

    summaries = [_run(name, t, scene, args.runs, args.out, args.seed) for name, t in builders]

    header = (
        f"{'pipeline':<20}{'median':>9}{'mean':>9}{'p90':>9}"
        f"{'min':>9}{'max':>9}{'fx/frame':>10}"
    )
    print(f"\n{args.runs} renders each at {args.size}px, times in ms\n")
    print(header)
    print("-" * len(header))
    for s in summaries:
        print(
            f"{s['pipeline']:<20}{s['median_ms']:>9}{s['mean_ms']:>9}{s['p90_ms']:>9}"
            f"{s['min_ms']:>9}{s['max_ms']:>9}{s['effects_per_frame']:>10}"
        )
    for s in summaries:
        if s["unchanged_frames"]:
            print(f"\n  note: {s['pipeline']} left {s['unchanged_frames']} frame(s) unchanged")

    (args.out / "summary.json").write_text(json.dumps(summaries, indent=1))
    print(f"\nFrames, contact sheets and per-frame manifests in {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
