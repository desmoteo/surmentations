"""Render a spread of parameter values per transform into docs/examples/.

Run with: .venv/bin/python docs/generate_examples.py

Each effect gets one image per severity level (1-5), because severity is this
library's own variability axis: it moves the *physical* driver of an effect —
a precipitation rate, an extinction coefficient, how much grime is on the
glass — and lets everything else follow from that.

On top of the ladder, each variant re-rolls the knobs severity deliberately
leaves alone. Severity describes the **weather**; it says nothing about where
the camera points, what lens is on it, or what happens to be on the glass, and
those change the picture just as much:

* where the sun sits relative to the frame, which relocates the whole ghost
  chain and changes each ghost's angle of incidence (hence its coating tint),
* how far the camera is tilted down, which is what makes rain streaks splay
  and which decides which surfaces collect snow,
* which deposits are on the lens, which sets the colour of the grime,
* how much glass the frame spans, which decides how many droplets are large
  enough to beat contact-line pinning and run.

Without that second axis, five images per effect differ only by RNG seed
within one set of ranges, and every row reads as five takes of one picture.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

from weathermentations.atmosphere.fog import RandomFog
from weathermentations.atmosphere.rain import MIN_RAIN_RATE_MM_H, RandomRain
from weathermentations.atmosphere.snow import RandomSnow
from weathermentations.camera import (
    AcFlicker,
    ExposureTransient,
    NightIrMode,
    RollingShutter,
    TemporalDenoiseArtifacts,
    VideoCodecArtifacts,
    WdrToneMapping,
)
from weathermentations.camera.codec import ffmpeg_path
from weathermentations.depth.estimator import DepthAnythingV2Estimator
from weathermentations.depth.provider import SharedDepthProvider
from weathermentations.lens import DirtSoil, HousingOccluders, SunFlare, WaterDroplets
from weathermentations.presets.severity import SEVERITY
from weathermentations.scene import CastShadows

OUT_DIR = Path(__file__).parent / "examples"
REFERENCE_IMAGE = OUT_DIR / "reference.webp"

# Output file stem -> the key that effect uses in the severity presets.
EFFECTS: dict[str, str] = {
    "water_droplets": "WaterDroplets",
    "dirt_soil": "DirtSoil",
    "sun_flare": "SunFlare",
    "fog": "RandomFog",
    "rain": "RandomRain",
    "snow": "RandomSnow",
    # Surveillance-stream effects (plan.md 6.2).
    "housing_occluders": "HousingOccluders",
    "cast_shadows": "CastShadows",
    "night_ir": "NightIrMode",
    "ac_flicker": "AcFlicker",
    "rolling_shutter": "RollingShutter",
    "exposure_transient": "ExposureTransient",
    "wdr_tone_mapping": "WdrToneMapping",
    "denoise_artifacts": "TemporalDenoiseArtifacts",
    "video_codec": "VideoCodecArtifacts",
}

# Needs the ffmpeg binary; skipped with a warning rather than crashing the run.
_NEEDS_FFMPEG = {"video_codec"}

LEVELS = sorted(SEVERITY)

# Deposit mixes that correspond to recognisable environments, so the dirt row
# shows the palette's actual range instead of five draws from the same bag.
DEPOSIT_MIXES: tuple[tuple[str, ...], ...] = (
    ("limescale",),                           # dried sea spray, hard water
    ("mineral_dust", "limescale"),            # dry dusty road
    ("soot", "road_film"),                    # urban traffic
    ("clay_soil", "mineral_dust"),            # unsealed track
    ("pollen", "mineral_dust", "road_film"),  # spring, everything at once
)


def _base_scene(max_dim: int = 640) -> np.ndarray:
    """Real reference photo (docs/examples/reference.webp), downscaled and in RGB.

    A real highway/traffic scene with genuine depth structure (near road ->
    mid-ground cars/guardrail -> distant buildings/skyline), which is what the
    depth-conditioned effects (fog/rain/snow) actually need to look
    convincing, unlike a flat synthetic scene.
    """
    img_bgr = cv2.imread(str(REFERENCE_IMAGE))
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read reference image at {REFERENCE_IMAGE}")
    height, width = img_bgr.shape[:2]
    scale = max_dim / max(height, width)
    img_bgr = cv2.resize(img_bgr, (int(width * scale), int(height * scale)))
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def _span(
    rng: np.random.Generator, low: float, high: float, width: float = 0.35
) -> tuple[float, float]:
    """A random sub-interval of ``[low, high]``, not a single value.

    Overrides here have to stay *ranges*. Collapsing one to a scalar would make
    the transform deterministic in that quantity for the whole variant, which
    is exactly the failure the transforms were fixed to avoid — a docs gallery
    is no reason to reintroduce it.
    """
    span = (high - low) * width
    centre = float(rng.uniform(low + span / 2.0, high - span / 2.0))
    return (centre - span / 2.0, centre + span / 2.0)


def _jitter_band(
    band: tuple[float, float], rng: np.random.Generator, floor: float = 0.0
) -> tuple[float, float]:
    """Shift and stretch a ``(min, max)`` band, keeping it a band.

    Used to vary a quantity the severity preset already sets, without
    discarding what severity chose: the level still decides roughly how bad
    the weather is, and this moves the window around that.
    """
    low, high = band
    shift = float(rng.uniform(0.75, 1.4))
    stretch = float(rng.uniform(0.9, 1.6))
    centre = 0.5 * (low + high) * shift
    half = 0.5 * (high - low) * stretch
    return (max(floor, centre - half), max(floor + 1e-3, centre + half))


def _variant_kwargs(
    name: str, index: int, rng: np.random.Generator, preset: dict[str, Any]
) -> dict[str, Any]:
    """Per-variant overrides for the knobs severity intentionally excludes.

    ``index`` runs alongside the severity level, so a variant differs in both
    how bad the weather is and how the shot is set up. ``preset`` is the
    severity entry, so an override can *move* a value severity already set
    rather than replacing it outright.
    """
    if name == "water_droplets":
        return {
            # How much glass the frame spans fixes a drop's size in
            # millimetres, and so whether it beats contact-line pinning and
            # runs. A wide pane puts several runners on the glass; a narrow one
            # leaves every drop pinned as a round bead.
            "pane_width_range_mm": _span(rng, 25.0, 260.0),
            "contact_angle_hysteresis_range": _span(rng, 0.15, 0.7),
        }
    if name == "dirt_soil":
        return {"deposits": DEPOSIT_MIXES[index % len(DEPOSIT_MIXES)]}
    if name == "sun_flare":
        return {
            # Every ghost sits on the source-to-centre line, so moving the
            # source relocates the whole chain — and changes the angle of
            # incidence at the internal surfaces, which blue-shifts the
            # coatings' notch and retints the ghosts.
            "flare_position": (
                float(rng.uniform(0.05, 0.95)),
                float(rng.uniform(0.02, 0.35)),
            ),
            "aperture_blades_range": (int(rng.integers(5, 8)), int(rng.integers(8, 10))),
            "streak_probability": 1.0,
            "coating_wavelength_range": (
                float(rng.uniform(440.0, 500.0)),
                float(rng.uniform(560.0, 640.0)),
            ),
        }
    if name == "fog":
        # How far the scene is assumed to extend, which scales the whole
        # optical depth.
        return {"depth_scale": float(rng.uniform(40.0, 90.0))}
    if name == "rain":
        return {
            # Camera pitch is what makes streaks splay: parallel world lines
            # project through a vanishing point, which a level camera puts at
            # infinity and a tilted one brings into view.
            "camera_pitch_range_deg": _span(rng, 0.0, 28.0),
            "wind_angle_range": (
                float(rng.uniform(-22.0, -2.0)),
                float(rng.uniform(2.0, 22.0)),
            ),
            # Move the severity's rain-rate window rather than replacing it,
            # so the ladder still orders the rows while no two samples share a
            # rate. Clamped to the transform's own validated floor.
            "rain_rate_range_mm_h": _jitter_band(
                preset["rain_rate_range_mm_h"], rng, floor=MIN_RAIN_RATE_MM_H
            ),
            # Severity does not touch the drop-size window, but it changes the
            # picture: raising the lower bound drops the fine drizzle texture
            # and leaves only the bold streaks, while the upper bound sets how
            # heavy the heaviest drops get.
            "drop_diameter_range_mm": (
                float(rng.uniform(0.6, 1.2)),
                float(rng.uniform(3.5, 6.0)),
            ),
        }
    if name == "night_ir":
        return {
            # How far the illuminator reaches is the dominant variable in a
            # night scene: it decides where the usable band of distance sits.
            "illuminator_range_m": _span(rng, 3.0, 20.0),
            "beam_falloff_range": _span(rng, 0.5, 5.0),
        }
    if name == "housing_occluders":
        # Day (silhouettes) or night under the IR ring (glowing).
        return {"glow_range": (0.0, 0.0) if index % 2 == 0 else _span(rng, 0.8, 2.0)}
    if name == "cast_shadows":
        return {"camera_pitch_deg": float(rng.uniform(2.0, 20.0))}
    if name == "ac_flicker":
        # 50 Hz and 60 Hz mains give different band counts for the same
        # readout, so alternate rather than picking one.
        return {
            "mains_frequency_hz": 50.0 if index % 2 == 0 else 60.0,
            "readout_time_range_s": _span(rng, 0.008, 0.033),
        }
    if name == "video_codec":
        return {"codec": "h264" if index % 2 == 0 else "h265"}
    if name == "snow":
        return {
            # Pitch also decides which surfaces face up, and so where snow
            # settles.
            "camera_pitch_range_deg": _span(rng, 2.0, 25.0),
            "fall_angle_range": (
                float(rng.uniform(-25.0, -5.0)),
                float(rng.uniform(5.0, 25.0)),
            ),
            "flutter_range": (
                float(rng.uniform(5.0, 15.0)),
                float(rng.uniform(20.0, 40.0)),
            ),
        }
    return {}


def _build(
    name: str, level: int, index: int, rng: np.random.Generator, provider: SharedDepthProvider
) -> Any:
    """Construct one transform at ``level`` severity with this variant's overrides."""
    preset: dict[str, Any] = SEVERITY[level][EFFECTS[name]]
    kwargs: dict[str, Any] = {"p": 1.0, **preset}
    kwargs.update(_variant_kwargs(name, index, rng, preset))

    if name == "water_droplets":
        return WaterDroplets(**kwargs)
    if name == "dirt_soil":
        return DirtSoil(**kwargs)
    if name == "sun_flare":
        return SunFlare(**kwargs)
    if name == "fog":
        return RandomFog(depth_provider=provider, **kwargs)
    if name == "rain":
        return RandomRain(depth_provider=provider, **kwargs)
    if name == "housing_occluders":
        return HousingOccluders(**kwargs)
    if name == "cast_shadows":
        return CastShadows(depth_provider=provider, **kwargs)
    if name == "night_ir":
        return NightIrMode(depth_provider=provider, **kwargs)
    if name == "ac_flicker":
        return AcFlicker(**kwargs)
    if name == "rolling_shutter":
        return RollingShutter(**kwargs)
    if name == "exposure_transient":
        return ExposureTransient(**kwargs)
    if name == "wdr_tone_mapping":
        return WdrToneMapping(**kwargs)
    if name == "denoise_artifacts":
        return TemporalDenoiseArtifacts(**kwargs)
    if name == "video_codec":
        return VideoCodecArtifacts(**kwargs)
    # Settled snow is half of what the transform does, so the gallery always
    # shows it rather than leaving it to a coin flip.
    return RandomSnow(depth_provider=provider, accumulation_probability=1.0, **kwargs)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scene = _base_scene()
    cv2.imwrite(str(OUT_DIR / "00_base_scene.png"), cv2.cvtColor(scene, cv2.COLOR_RGB2BGR))

    # One provider for every depth-conditioned transform, so depth is estimated
    # once for the whole run rather than per transform — the same sharing the
    # library asks pipelines to do.
    depth_provider = SharedDepthProvider(DepthAnythingV2Estimator())

    rng = np.random.default_rng(42)
    # Most of the time here goes into depth estimation, so show progress
    # rather than appearing to hang.
    names = [n for n in EFFECTS if n not in _NEEDS_FFMPEG or ffmpeg_path() is not None]
    if len(names) != len(EFFECTS):
        print("ffmpeg not found; skipping the video-codec examples")

    for name in tqdm(names, total=len(names), desc="effects"):
        for index, level in enumerate(LEVELS):
            transform = _build(name, level, index, rng, depth_provider)
            transform.set_random_seed(int(rng.integers(0, 2**31 - 1)))
            out = transform(image=scene)["image"]
            cv2.imwrite(
                str(OUT_DIR / f"{name}_{index}.png"), cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
            )

    print(f"Wrote {len(names)} effects x {len(LEVELS)} severities to {OUT_DIR}")


if __name__ == "__main__":
    main()
