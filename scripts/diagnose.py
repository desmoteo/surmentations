"""Render a sweep of every transform and log the random draws behind each frame.

Run with: .venv/bin/python scripts/diagnose.py [--out DIR] [--seeds N]

Produces, in the output directory:

* ``draws.jsonl`` — one compact record per transform call: the constructor
  ranges, the physical quantities actually drawn, and a summary of every
  parameter handed to the renderer.
* ``<effect>_sev<N>_seed<M>.png`` — the frame each record produced.

The point is to make the *derived chain* auditable. Most quantities here are
consequences rather than choices — a rain rate fixes a size distribution,
which fixes a drop count, which fixes each streak's opacity — and an image
alone cannot tell you whether that chain held, only whether the result looked
plausible. Pairing each image with the numbers behind it can.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

# Reuse the gallery's own construction, so what is audited is what ships.
from docs.generate_examples import EFFECTS, LEVELS, _base_scene, _build  # noqa: E402
from weathermentations.camera.codec import ffmpeg_path  # noqa: E402
from weathermentations.depth.estimator import DepthAnythingV2Estimator  # noqa: E402
from weathermentations.depth.provider import SharedDepthProvider  # noqa: E402
from weathermentations.diagnostics import recording, write_jsonl


def main() -> None:
    """Render every effect at every severity with a few seeds, and record what each drew.

    Each frame is saved beside its record, what the transform drew and how
    far the frame moved, in ``draws.jsonl``, so that the chain of derived
    quantities behind each image can be audited.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="diagnostics", type=Path)
    parser.add_argument("--seeds", default=3, type=int, help="seeds per severity")
    parser.add_argument("--size", default=480, type=int)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    scene = _base_scene(args.size)
    cv2.imwrite(str(args.out / "00_scene.png"), cv2.cvtColor(scene, cv2.COLOR_RGB2BGR))

    provider = SharedDepthProvider(DepthAnythingV2Estimator())
    names = [
        n
        for n in EFFECTS
        if n not in ("video_codec", "transmission_faults") or ffmpeg_path() is not None
    ]

    rng = np.random.default_rng(20260914)
    records: list[dict[str, Any]] = []

    for name in tqdm(names, desc="effects"):
        for index, level in enumerate(LEVELS):
            for seed_index in range(args.seeds):
                transform = _build(name, level, index, rng, provider)
                seed = int(rng.integers(0, 2**31 - 1))
                transform.set_random_seed(seed)

                with recording() as log:
                    out = transform(image=scene)["image"]

                stem = f"{name}_sev{level}_seed{seed_index}"
                cv2.imwrite(str(args.out / f"{stem}.png"), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))

                for record in log:
                    record["effect"] = name
                    record["severity"] = level
                    record["seed"] = seed
                    record["image"] = f"{stem}.png"
                    # Cheap whole-frame summary, so the log alone says how much
                    # the effect actually changed and in which direction.
                    delta = out.astype(np.float32) - scene.astype(np.float32)
                    record["delta"] = {
                        "mean_abs": round(float(np.abs(delta).mean()), 3),
                        "mean_signed": round(float(delta.mean()), 3),
                        "std": round(float(delta.std()), 3),
                        "frac_changed": round(float((np.abs(delta) > 2).mean()), 4),
                    }
                records.extend(log)

    write_jsonl(records, args.out / "draws.jsonl")
    print(f"Wrote {len(records)} records and {len(records)} frames to {args.out}")


if __name__ == "__main__":
    main()
