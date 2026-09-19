# weathermentations

Albumentations-compatible, physically-grounded augmentations for training
models that run on real camera feeds: weather (rain, snow, fog), lens
artifacts (water droplets, dirt, flare, insects on the housing), and the
sensor/ISP/codec chain that degrades a CCTV stream far more often than
weather does.

Effects are parameterised by **physical quantities** — a rain rate in mm/h, a
drop diameter in millimetres, a shutter time in seconds — not opaque 0-1
"strength" dials, and everything that follows from those is derived rather
than exposed.

- **[`docs/how_it_works.md`](docs/how_it_works.md)** — what each effect
  models and why, with diagrams. Start here to decide which knobs matter.
- **[`docs/pipelines.md`](docs/pipelines.md)** — a worked composite whose
  effects cannot contradict each other, with measured frequencies and cost.
  Start here if you want a pipeline rather than a transform.
- **[`docs/physical_review.md`](docs/physical_review.md)** — a critical audit
  of whether the implementation lives up to that, including where it did not.
- **[`optimizations.md`](optimizations.md)** — where the CPU time actually
  goes, and an assessment of what is worth optimising.
- **[`plan.md`](plan.md)** — design record and roadmap.

---

## Install

```bash
pip install -e ".[dev]"   # editable install with lint/type/test tooling
```

Optional runtime dependencies:

| Need | Required for |
| --- | --- |
| CUDA GPU | the depth-conditioned effects (fog, rain, snow, night IR, cast shadows) |
| `ffmpeg` on PATH | `VideoCodecArtifacts` only |

Everything else runs on CPU with no extra setup.

---

## Quickstart

### Camera-local effects — no GPU

```python
import albumentations as A
from weathermentations.lens import WaterDroplets, DirtSoil, SunFlare

transform = A.Compose([
    WaterDroplets(p=0.3),
    DirtSoil(p=0.2),
    SunFlare(p=0.1),
])

augmented = transform(image=image)["image"]
```

### Depth-conditioned effects — needs CUDA

Construct **one** `SharedDepthProvider` and pass that same instance to every
depth-conditioned transform, so depth is estimated once per image instead of
once per transform. This is the single biggest throughput decision in a
pipeline here — see [Performance](#performance).

```python
import albumentations as A
from weathermentations.atmosphere import RandomFog, RandomRain, RandomSnow
from weathermentations.depth import DepthAnythingV2Estimator, SharedDepthProvider

depth_provider = SharedDepthProvider(DepthAnythingV2Estimator())

transform = A.Compose([
    RandomFog(depth_provider=depth_provider, p=0.2),
    RandomRain(depth_provider=depth_provider, p=0.2),
    RandomSnow(depth_provider=depth_provider, p=0.1),
])
```

With no CUDA device, constructing `DepthAnythingV2Estimator` raises a
`RuntimeError` immediately with an actionable message, rather than silently
falling back to a CPU path slow enough to blow a training-step budget.

---

## The effects

### Weather and scene

| Transform | Models | Needs depth |
| --- | --- | --- |
| `RandomFog` | Koschmieder extinction with per-channel scattering, and a cap that keeps the sky's gradient | yes |
| `RandomRain` | Marshall-Palmer drop sizes, Gunn-Kinzer fall speeds, Garg-Nayar streak photometry, a distance veil, and optional wet ground with Fresnel reflection | yes |
| `RandomSnow` | Gunn-Marshall sizes corrected for snow density, Locatelli-Hobbs fall speeds, and accumulation on upward-facing surfaces | yes |
| `CastShadows` | Hard cast shadows on upward-facing surfaces — a classic false-positive source. A loss of the direct beam, so texture survives and the shadow is bluer | yes |

### Lens and housing

| Transform | Models | Needs depth |
| --- | --- | --- |
| `WaterDroplets` | Droplets as short-focal-length lenses holding an inverted, demagnified image of the scene; shedding decided by capillary length | no |
| `DirtSoil` | Grime as a *scattering* layer, not paint — energy-conserving, with per-deposit albedos from grey dust through yellow pollen to brown spray | no |
| `SunFlare` | Veiling glare, halo, coating-tinted ghost chain, diffraction starburst, dispersion corona, plus CCTV-specific dome scratches, smear and IR leak | no |
| `HousingOccluders` | Insects and webs on the dome — hugely magnified and defocused, silhouetted by day, glowing under the IR ring | no |

### Sensor, ISP and transmission

| Transform | Models | Needs depth |
| --- | --- | --- |
| `NightIrMode` | IR-cut filter out: monochrome by *infrared* reflectance (foliage goes bright, which a luma conversion gets backwards), an illuminator falling as 1/z², gain-scaled noise | yes |
| `AcFlicker` | Mains banding. Cancels exactly when the exposure is a whole number of flicker periods — the basis of a camera's anti-flicker setting | no |
| `RollingShutter` | Row-dependent shear and wobble from camera motion during readout | no |
| `ExposureTransient` | AE/AWB caught mid-correction. White-balance error is the ratio of two blackbody colours, so it stays on the blue-orange axis real illuminants vary on | no |
| `WdrToneMapping` | Local tone mapping, with the black halos a cheap edge-preserving filter leaves | no |
| `TemporalDenoiseArtifacts` | 3D-NR's "watercolour" flattening and chroma blotching | no |
| `VideoCodecArtifacts` | H.264/H.265 **encoded for real via ffmpeg**, so deblocking, 4:2:0 chroma and rate control come out correct rather than approximated. Optionally simulates stale macroblocks from a lost P-frame | no |

---

## Configuration

### Parameter conventions

Three rules hold across every transform.

**Physical `(min, max)` ranges.** Constructors take ranges of physical
quantities — fog's extinction coefficient, a raindrop's diameter in
millimetres, an aperture's diameter and blade count, a shutter time in
seconds. Ranges are validated by a pydantic `InitSchema`, so an impossible
value fails at construction rather than at render.

**Derived quantities are not exposed.** A rain rate settles the drop-size
distribution, the particle count, and — through the standard visibility
relations — how dense the veil is. These are not independent in reality, and
making them independent here is what allows a frame with a downpour's haze
and a drizzle's streaks.

**Sizes are resolution-independent.** Camera-local effects express sizes as
fractions of the image diagonal; effects with a real scene scale work in
metres and project through a stated field of view. A feature's *geometry* is
the same on a 256 px crop and a 4K frame.

### Configuration reference

The knobs that change the picture most, per transform. Every one is a
`(min, max)` range unless noted; see each module's docstring for the full set
and the physics behind it.

| Transform | Start with |
| --- | --- |
| `RandomFog` | `beta_range` (extinction, 1/m), `scattering_gamma_range` (0 grey, ~1.3 blue haze, 4 Rayleigh), `depth_scale` |
| `RandomRain` | `rain_rate_range_mm_h`, `drop_diameter_range_mm`, `exposure_time_range`, `wetness_probability`, `min_streak_width_px`, `max_streaks` |
| `RandomSnow` | `snowfall_rate_range_mm_h`, `snow_density_range_g_cm3`, `accumulation_probability`, `accumulation_strength_range`, `min_flake_width_px`, `max_flakes` |
| `CastShadows` | `shadow_count_range`, `shadow_scale_range`, `direct_fraction_range`, `penumbra_range`, `camera_pitch_deg` |
| `WaterDroplets` | `droplet_count_range`, `droplet_radius_range`, `pane_width_range_mm` (sets the physical scale), `background_blur_range`, `condensation_coverage_range` |
| `DirtSoil` | `coverage_range`, `veil_strength_range`, `highlight_boost_range`, `deposits` (which substances), `smudge_count_range` |
| `SunFlare` | `flare_position`, `intensity_range`, `ghost_count_range`, `aperture_blades_range`, `dispersion_ring_probability`, `scratch_count_range` |
| `HousingOccluders` | `insect_count_range`, `insect_size_range`, `web_count_range`, `defocus_range`, `glow_range` (0 by day, >0 under the IR ring) |
| `NightIrMode` | `illuminator_range_m`, `beam_falloff_range`, `gain_range` (noise, not level), `read_noise_range`, `vegetation_gain_range` |
| `AcFlicker` | `mains_frequency_hz` (50 or 60), `readout_time_range_s` (sets the band *count*), `exposure_time_range_s` (sets the depth), `pulse_exponent_range`, `modulation_depth_range` |
| `RollingShutter` | `pan_shift_range_px`, `wobble_amplitude_range_px`, `wobble_cycles_range` |
| `ExposureTransient` | `exposure_error_stops_range`, `assumed_temperature_range_k`, `actual_temperature_range_k` |
| `WdrToneMapping` | `compression_range`, `detail_gain_range`, `halo_strength_range` |
| `TemporalDenoiseArtifacts` | `strength_range`, `iterations_range`, `edge_threshold_range`, `chroma_blur_range` |
| `VideoCodecArtifacts` | `codec` (`"h264"` / `"h265"`), `crf_range`, `stale_block_probability` |

Two of these are worth singling out because their effect is not what the name
suggests. `AcFlicker`'s `readout_time_range_s` matters more than
`modulation_depth_range`: band *count* is what makes flicker legible, and two
deep bands read as an exposure difference. `NightIrMode`'s `gain_range` scales
the **noise**, not the signal — the auto-exposure has already placed the
level, and gain is one of the levers it used to get there.

### Severity presets

For ImageNet-C-style sweeps, unpack a preset instead of picking ranges by
hand. Levels are 1-5 and every ladder is tested for monotonicity.

```python
from weathermentations.presets import SEVERITY
from weathermentations.lens import WaterDroplets

transform = WaterDroplets(p=1.0, **SEVERITY[3]["WaterDroplets"])
```

### Composition order

These model stages of one imaging chain — scene → atmosphere → optics →
sensor → ISP → codec — and applying them out of order produces images that
could not exist (codec blocking underneath lens blur means the lens blurred
an already-compressed image). `physical_order` sorts a list into it:

```python
from weathermentations.presets import physical_order

transform = A.Compose(physical_order([
    VideoCodecArtifacts(p=0.4),
    RandomRain(depth_provider=provider, p=0.3),
    DirtSoil(p=0.2),
]))
```

It sorts a *list*; it does not build the `Compose`, so you keep control of
probabilities and seeding. Sorting is stable, so transforms sharing a stage
keep the order you gave them.

### Seeding

Randomness goes through each transform's own RNG, so pipeline-level seeding
reaches these transforms exactly as it does built-in ones:

```python
transform = A.Compose([...], seed=42)
```

---

## Performance

Measured with `scripts/benchmark.py` on one machine (Intel Core Ultra 9 285H,
RTX PRO 1000 Blackwell Laptop GPU, torch 2.11 + CUDA 12.8), median of 15
calls per measurement. Treat these as *relative* costs; absolute numbers will
differ on your hardware.

```bash
python scripts/benchmark.py                        # 512px and 1024px
python scripts/benchmark.py --sizes 1024 --csv out.csv
python scripts/benchmark.py --no-depth             # CPU-only subset
```

### CPU-only transforms — total ms per image

| Transform | 512 px | 1024 px |
| --- | --- | --- |
| `RollingShutter` | **0.2** | 1.1 |
| `AcFlicker` | 4.1 | 21.3 |
| `ExposureTransient` | 4.8 | 24.3 |
| `HousingOccluders` | 10.8 | 66.7 |
| `SunFlare` | 21.4 | 125.5 |
| `DirtSoil` | 32.8 | 171.8 |
| `TemporalDenoiseArtifacts` | 38.6 | 237.4 |
| `WaterDroplets` | 96.3 | 209.4 |
| `WdrToneMapping` | 102.9 | 146.2 |
| `VideoCodecArtifacts` | 170.4 | 251.9 |

### Depth-conditioned transforms — total ms per image

"Cold" pays for depth estimation; "warm" is what the second and subsequent
depth-conditioned transform costs when they share one provider.

| Transform | 512 cold | 512 warm | 1024 cold | 1024 warm |
| --- | --- | --- | --- | --- |
| `CastShadows` | 61.5 | **24.2** | 209.0 | 126.7 |
| `NightIrMode` | 81.4 | **13.5** | 136.0 | 77.3 |
| `RandomFog` | 86.7 | **19.8** | 212.7 | 114.0 |
| `RandomRain` | 126.0 | 67.3 | 349.3 | 269.7 |
| `RandomSnow` | 135.4 | **32.1** | 245.8 | 177.0 |

Sharing one provider saves roughly **62 ms per extra transform at 512 px and
86 ms at 1024 px**. Constructing a provider per transform throws that away.

These are a laptop under thermal limits: absolute numbers move by up to 2x
between runs, and the relative ordering plus the cold/warm gap are the parts
worth planning around.

### Choosing a budget

A `DataLoader` worker must keep ahead of the GPU. Rough guidance:

- **Cheap enough to apply liberally** (under ~25 ms at 512 px):
  `RollingShutter`, `AcFlicker`, `ExposureTransient`, `HousingOccluders`,
  `SunFlare`.
- **Moderate** (25-110 ms): `DirtSoil`, `TemporalDenoiseArtifacts`,
  `WaterDroplets`, `WdrToneMapping`, and any *warm* depth-conditioned
  transform.
- **Expensive, keep `p` low**: `VideoCodecArtifacts` (two ffmpeg subprocesses
  per call), and the first depth-conditioned transform in a pipeline.

Cost is paid only when `p` fires, so a pipeline of ten effects at `p=0.2`
costs about two of them per image, not ten. The depth estimate is the
exception: it is paid once if *any* depth-conditioned transform fires.

---

## Usage considerations

**These are per-image transforms.** Anything whose mechanism is genuinely
*temporal* — 3D-NR's motion trails, banding drifting between frames — is
rendered only as the single-frame state it leaves behind. Each module says so
where it applies rather than approximating silently. If you train on video and
need temporal coherence, you need to hold the draws fixed across a clip
yourself.

**`p` decides whether; parameters decide how strongly.** A transform
constructed with `p=1.0` will change the image. Ranges are not permitted to
include "no effect" as a legal draw — that was a bug twice, and both are now
guarded by tests. If you want an effect to sometimes not apply, lower `p`.

**Precipitation intensity is resolution-dependent, by design.** Geometry is
resolution-independent, but the amount of precipitation visible as *discrete
particles* rises with image size (roughly 6-8× over 256→1920 px), because the
cut deciding what is drawn individually rather than left to the veil is in
pixels — a bigger sensor really does resolve more drops. Total extinction is
unaffected (rendered particles carry ~0.3% of it). If you train multi-scale
and want the look held constant, `min_streak_width_px` and
`min_flake_width_px` are the knobs.

**Depth is pseudo-metric, not measured.** Monocular depth has no absolute
scale, so it is mapped onto a near and a far plane — 3 m and 50 m by default,
set in `depth/scale.py`. Effects that depend on distance *linearly* (fog,
rain, snow) tolerate a wrong guess; `NightIrMode`'s inverse-square illuminator
does not, and a scene whose real near plane is 10 m will read too contrasty
at the bottom of the frame.

Only `RandomFog` currently exposes the calibration, as `depth_scale`; the
other depth-conditioned transforms use the module defaults. If your scenes
have a known and unusual scale, change the defaults rather than expecting a
per-transform argument.

**Compositing is in linear light.** If you chain these with other
augmentation libraries that work on gamma-encoded values, the ordering matters
more than usual — put colour-space-naive transforms outside this chain, not in
the middle of it.

**GPU memory.** One `DepthAnythingV2Estimator` holds a model on the GPU.
With multiple `DataLoader` workers each constructing their own, memory scales
with worker count — construct the provider inside the worker init, and size
`num_workers` accordingly.

---

## Effect gallery

Rendered by [`docs/generate_examples.py`](docs/generate_examples.py) on the
reference photo below. Each effect gets one image per severity level, so
`<effect>_0.png` is severity 1 and `<effect>_4.png` is severity 5. Each
variant also re-rolls the setup knobs severity leaves alone — sun position,
camera pitch, deposit mix, mains frequency — since those change the picture as
much as the weather does.

Each cell shows the **median of seven draws**, not the first: effects that can
legitimately draw themselves out of existence otherwise produce a cell the
reader cannot tell from a bug. See [`docs/examples/`](docs/examples/) for the
full sweep.

| Base scene | Water droplets | Dirt / soil |
| --- | --- | --- |
| ![base](docs/examples/00_base_scene.png) | ![water droplets](docs/examples/water_droplets_2.png) | ![dirt/soil](docs/examples/dirt_soil_2.png) |

| Sun flare | Fog | Rain | Snow |
| --- | --- | --- | --- |
| ![sun flare](docs/examples/sun_flare_3.png) | ![fog](docs/examples/fog_2.png) | ![rain](docs/examples/rain_3.png) | ![snow](docs/examples/snow_3.png) |

| Night IR | AC flicker | Cast shadows | Housing occluders |
| --- | --- | --- | --- |
| ![night ir](docs/examples/night_ir_3.png) | ![ac flicker](docs/examples/ac_flicker_4.png) | ![cast shadows](docs/examples/cast_shadows_3.png) | ![housing occluders](docs/examples/housing_occluders_3.png) |

---

## Auditing the physics

Most quantities here are *derived* — a rain rate fixes a size distribution,
which fixes a drop count, which fixes each streak's opacity. An image only
shows whether the result looks plausible, not whether that chain held, so the
draws behind each frame can be recorded and checked:

```bash
PYTHONPATH=. python scripts/diagnose.py --out diagnostics --seeds 3
```

That writes one frame plus one compact JSON record per call: the constructor
ranges, the physical quantities actually sampled, and a summary of every
parameter handed to the renderer. Recording is opt-in and costs one flag check
when off; transforms do not opt in, `WeatherTransform` instruments every
subclass automatically.

```python
from weathermentations.diagnostics import recording, write_jsonl

with recording() as log:
    augmented = transform(image=image)["image"]
write_jsonl(log, "draws.jsonl")
```

[`docs/physical_review.md`](docs/physical_review.md) is a critical assessment
built from such sweeps across four passes. Every derived chain that can be
checked closes to float precision — drop counts, flake sizes, flicker depth,
veil extinction — and the document is equally explicit about what did not hold
and what the failures had in common.

---

## Development

```bash
pip install -e ".[dev]"
ruff check src tests docs scripts
mypy src/weathermentations
pytest tests -q
```

Depth-conditioned tests skip automatically when no CUDA device is present.
Regenerate visual-regression goldens deliberately — only when a render
intentionally changes:

```bash
python -m tests.visual_regression.generate_goldens
python docs/generate_examples.py          # and the gallery
```

Two test files are worth knowing about because they guard properties rather
than pixels: `test_severity_ladders.py` checks every severity ladder is
monotonic and that level 1 is visible, and `test_gallery_configs.py` renders
the configuration the *gallery* builds — overrides included — and requires
each cell to change the image. The second exists because a stale override in
the gallery once discarded a library fix while every library test passed.

---

## License

MIT — see [LICENSE](LICENSE).
