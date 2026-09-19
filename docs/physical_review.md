# Physical review: are the outputs actually consistent with their inputs?

A critical assessment of the rendered effects against the **random draws that
produced them**, rather than against how the images look.

The distinction matters here more than in most libraries. Almost nothing in
this codebase is chosen directly: a rain rate fixes a Marshall–Palmer slope,
which with the simulated volume fixes a drop count, which with each drop's
diameter and distance fixes its streak's length, width and opacity. An image
can only tell you whether the result *seems* plausible. It cannot tell you
whether that chain held — and a chain that quietly breaks still produces a
plausible-looking picture, which is the failure mode worth hunting.

## Method

`src/weathermentations/diagnostics.py` records, per transform call:

- **`init`** — the constructor ranges, i.e. what each draw was supposed to
  come from.
- **`drew`** — the physical quantities actually sampled, including ones that
  are local variables and never reach the renderer (the rain rate, the
  exposure, the aperture). These are exactly the values needed to audit the
  derived chain and would otherwise be invisible.
- **`out`** — every parameter handed to the renderer, with arrays reduced to
  `n / min / p50 / max / mean`.
- **`delta`** — a whole-frame summary of what the effect did to the image.

Recording is opt-in and free when off (one flag check). Transforms do not opt
in: `WeatherTransform.__init_subclass__` wraps every subclass's
`get_params_dependent_on_data` automatically.

```bash
PYTHONPATH=. .venv/bin/python scripts/diagnose.py --out diagnostics --seeds 3
```

This review is based on **225 records / 225 frames** — 15 effects × 5
severities × 3 seeds — at 480 px on the reference highway photograph.
The log is 209 KB, about 950 bytes per frame.

**Scope limits, stated up front.** One scene, one resolution for the sweep,
three seeds per cell. That is enough to audit the arithmetic exactly and to
catch structural problems, and *not* enough to characterise distributions or
to claim anything about scene-dependence. Where a finding rests on three
samples it says so.

## What verified exactly

These are closed-form checks against the log, not impressions.

| Check | Result |
| --- | --- |
| Every sampled value lies inside the range it was drawn from | **315 draws, 0 violations** |
| Rain: `count == number_density(rate) × volume × resolvable_fraction` | max relative error **3.6e-5** |
| Snow: melted→physical factor `== (ρ_water/ρ_snow)^(1/3)` | max absolute error **1.2e-5** |
| Flicker: recorded depth `== nominal × |sinc(T·f)|` | max absolute error **5.4e-7** |
| Rain and snow veil extinction derived from the rate (±[0.75, 1.3]) | holds, 30/30 records |

The residuals are float32 round-off. The derived chains are sound: the drop
count really is a consequence of the rain rate and the simulated volume, and
the flake size really is the melted diameter corrected for snow density.

Streak photometry also lands where Garg & Nayar put it. Median streak opacity
across all rain frames is **0.0155 – 0.069** (median 0.025), against their
~0.017 for a 1 mm drop at 1/60 s. Higher values track the larger drops heavy
rain produces, as `α = D/(vT)` requires.

## Findings

### F1 — Apparent intensity is resolution-dependent (rain, snow)

**The most consequential finding, and it contradicts a documented claim.**

The README states sizes are resolution-independent so "an effect looks the
same on a 256px crop and a 4K frame". The *geometry* is. The **amount of
visible precipitation is not**, because the resolvability cut
(`min_streak_width_px`, `min_flake_width_px`) is in pixels. Same physical
weather, varying only image size:

| size | snow particles | snow coverage | rain streaks | rain coverage |
| --- | --- | --- | --- | --- |
| 256 | 52 | 0.34% | 134 | 0.06% |
| 480 | 341 | 0.80% | 1 066 | 0.17% |
| 640 | 789 | 0.95% | 2 570 | 0.24% |
| 1024 | 2 685 | 1.45% | 10 220 | 0.42% |
| 1920 | 8 543 | 2.05% | 16 000 | 0.48% |

A **6× swing in snow coverage and 8× in rain** across a range of sizes a
training pipeline hits routinely with random crops and multi-scale training.

> **Correction (2026-09-15).** The first version of this section claimed the
> veil double-counts the rendered particles by "up to 7.4% for rain and 14.2%
> for snow", and recommended scaling the veil by `1 − resolvable_fraction`.
> **That number was wrong and the recommendation was unnecessary.** It used
> the *count* fraction of particles inside the simulated column as a proxy for
> their share of the *extinction* — two different quantities. Extinction is an
> optical depth accumulated along the whole path, and the rendered particles
> occupy only a few metres of a 50 m one.
>
> Measured directly instead, by comparing the rendered layer's occupancy
> against the veil's optical depth (they add, since the two are composited
> sequentially):
>
> | rain rate | veil τ | rendered occupancy | rendered share |
> | --- | --- | --- | --- |
> | 5 mm/h | 0.154 | 0.00019 | 0.12% |
> | 20 mm/h | 0.368 | 0.00075 | 0.20% |
> | 60 mm/h | 0.735 | 0.00203 | 0.28% |
> | 120 mm/h | 1.138 | 0.00363 | 0.32% |
>
> The double-count is **0.12–0.32%** — a rounding error. The veil was left
> alone. The lesson is the one this document exists to make: a plausible
> proxy is not a measurement, and I should have measured the quantity I was
> making a claim about rather than one correlated with it.

What remains true is that the underlying behaviour is *correct physics* — a
higher-resolution sensor at the same field of view genuinely does resolve more
individual drops — while **the documented claim is false as written**. The
resolution dependence is not an energy-conservation error. It is that the
share of precipitation visible as *discrete particles* rises with resolution,
while the veil (correctly) does not change. Since the rendered particles carry
a third of a percent of the extinction but most of the visual signature, the
image changes a lot while the physics barely does.

There is accordingly nothing to fix in the renderer. The documentation needs
to stop claiming otherwise, and a caller doing multi-scale training should
know that `min_streak_width_px` / `min_flake_width_px` are the knobs that
control it.

### F2 — The pseudo-metric depth scale puts the scene 0.3 m from the camera

`to_pseudo_metric` maps relative depth linearly onto 0–50 m, so the nearest
scene point lands at 0 m and is only saved by a 0.3 m clamp. On the reference
frame `NightIrMode` therefore sees the scene spanning **0.30 m to 50 m**,
median 6.3 m.

For a bridge-mounted highway camera the nearest road surface is perhaps 10 m
away. The consequence for an inverse-square illuminator is not subtle: the
near:far irradiance ratio comes out at **27 778 ×**, and the recorded
exposure field spans `p50 ≈ 1.3` to `max ≈ 1510`. The auto-exposure added
earlier hides the absolute level, but it cannot fix the *gradient* — which is
precisely the quantity that gives night footage its character. The rendered
result (see `night_ir_sev3_seed0.png`) has a correctly blown-out near field
but an unrealistically abrupt transition to black.

This is a **pre-existing limitation of the depth calibration**, documented in
`depth/scale.py` as "a documented, overridable guess", now with a measured
consequence. It affects every depth-conditioned effect but bites hardest where
the dependence is quadratic.

> **Resolved (2026-09-15).** `to_pseudo_metric` now maps onto `[near, far]`
> with `DEFAULT_MIN_DISTANCE_METERS = 3.0`, so the nearest scene point lands
> on a near plane instead of at zero. The near:far irradiance ratio fell from
> **27 778× to 278×**, and on the reference frame the night-mode output went
> from 11.6% saturated / 24.1% black to **6.5% / 8.0%**, with scene structure
> surviving across the whole depth range rather than only in a narrow band.
> Every depth-conditioned effect reads through this function, so fog, rain,
> snow and cast shadows all changed slightly too; goldens were regenerated.

### F3 — The docs generator's jitter escapes its own severity band

`_jitter_band` in `docs/generate_examples.py` clamps the **floor** to the
transform's validated minimum but does not clamp the ceiling. Recorded rain
rates by severity:

| severity | nominal preset | effective range after jitter | actually drawn |
| --- | --- | --- | --- |
| 1 | 2–8 | 2.0–10.9 | 2.0–9.3 |
| 3 | 28–54 | 14.1–70.3 | 47.7–64.8 |
| 5 | 55–100 | 48.7–137.4 | **105.0–130.6** |

Severity 5 drew up to **130 mm/h against a nominal ceiling of 100**, and the
effective bands for levels 3–5 overlap heavily. The ladder still orders
correctly on average, but it is looser than it claims. This is a gallery
concern, not a library one — the transforms themselves are unaffected.

### F4 — `NightIrMode` severity is invisible to any whole-frame metric

Mean |Δ| by severity: **114, 112, 113, 111, 107** — flat, and mildly
*decreasing*. This is not a bug: the auto-exposure deliberately normalises the
level, and the day→night conversion (≈110 levels) swamps everything severity
controls (illuminator reach, gain, noise floor). But it means the severity
ladder for this effect cannot be validated the way the others were, and the
monotonicity test used elsewhere would be meaningless here. It needs its own
metric — noise floor in the shadows, or the near/far falloff gradient.

### F5 — Non-monotonic severity: three real causes, only one worth fixing

Five effects came out non-monotonic over three seeds. Separating them matters,
because two are correct behaviour being measured badly:

- **`exposure_transient` (10.6, 33.2, 18.0, 31.4, 51.9)** — the error is
  *signed and symmetric*, so severity widens the spread rather than raising
  the magnitude. A draw near zero gives a small delta at any level. Correct;
  three seeds is simply too few to see it. Needs ~20 seeds, or a metric on the
  spread rather than the mean.
- **`ac_flicker` (0.46, 4.00, 3.20, 1.29, 5.77)** — driven by the `sinc`
  cancellation: an exposure that happens to land near a whole flicker period
  kills the banding outright, whatever the severity. This is the physics
  working as intended (it is the basis of a camera's anti-flicker setting) and
  should *not* be smoothed away.
- **`housing_occluders` (1.21, 6.71, 1.87, 7.30, 5.72)** — alternates because
  the generator alternates day/night by variant index, and the IR case is much
  brighter. An artefact of the gallery, not the transform.
- **`fog`, `sun_flare`, `cast_shadows`** — single small dips consistent with
  seed noise at n=3 (fog's `depth_scale` and the flare's source position are
  both jittered per variant).

Nothing here needs a code change. It does mean **the severity ladders are not
currently verified for most effects** — only rain and snow have monotonicity
tests, and those were added because their ladders actually broke.

### F6 — Severity 1 is below the visible threshold for three effects

Mean |Δ| at severity 1: `dirt_soil` 0.94, `denoise_artifacts` 0.71,
`ac_flicker` 0.46 — at or under one 8-bit level. The same problem already
fixed for `SunFlare` (which was 0.33 before its floor was raised). A bottom
rung that does nothing wastes a rung and reads as a bug rather than a setting.
`ac_flicker` is partly excused by F5, but `dirt_soil` and `denoise_artifacts`
are simply set too low.

### F7 — Rain streaks are 1 px wide in every frame

Recorded streak thickness has `p50 == 1` in **15 of 15** rain frames (max
observed 6). This is the documented sub-pixel handling — drops narrower than a
pixel are drawn one pixel wide with alpha scaled by `width/thickness` to
conserve deposited light — and the arithmetic is right. But it means the
*width* channel of the drop-size distribution is almost entirely collapsed at
these resolutions: drop diameter reaches the image only through opacity, not
through visible width. Worth knowing before concluding that drop-size variety
is visible in the output. It is not; it is visible in the statistics.

### F8 — Night mode's shadow noise is at the edge of plausible

`night_ir` frames come out **11.6% saturated and 24.1% pure black**, with very
strong noise in the black region. The direction is right — read noise does not
scale with signal, so a gained-up frame is noisiest where it is darkest — but
a quarter of the frame at zero with visible speckle is heavier than typical
surveillance footage, which usually shows a noise floor above black. Combined
with F2 (the exaggerated depth gradient), the near/far contrast is likely
overstated.

## What holds up

Worth stating explicitly, since the above is a list of complaints:

- **Every arithmetic chain checked was exact.** Drop counts, flake sizes,
  flicker depth and veil extinction all follow from their inputs to float
  precision. There is no place where a "physical" quantity is quietly
  disconnected from the rendering.
- **No draw ever escaped its declared range** across 315 checks.
- **Streak photometry matches the literature** without having been fitted to
  it — `α = D/(vT)` lands at 1.5–7%, against Garg & Nayar's ~1.7% reference.
- **The effects darken and brighten in the right directions**: fog (+35),
  snow (+33) and rain (+7) add airlight; shadows (−11) and night mode (−70)
  remove light; rolling shutter (+0.03) is geometry and moves nothing.
- **Severity is monotonic where it was tested** — `water_droplets`, `rain`,
  `snow`, `rolling_shutter`, `wdr_tone_mapping`, `denoise_artifacts`,
  `video_codec` and `dirt_soil` all rise cleanly across the ladder.

## Recommendations, in priority order

1. ~~Scale the veil by `1 − resolvable_fraction`~~ — **withdrawn**, see the
   correction under F1. The double-count is 0.3%, not 7–14%.
2. **Correct the README's resolution-independence claim** (F1) to say that
   geometry is resolution-independent while the resolved/veiled split is not.
   — **done**
3. **Give `to_pseudo_metric` a near-plane** (F2) instead of mapping relative
   depth onto `[0, max]`. — **done**: `DEFAULT_MIN_DISTANCE_METERS = 3.0`.
   The night-mode near:far irradiance ratio went from 27 778× to 278×, and
   measured on the reference frame the output went from 11.6% saturated /
   24.1% black to 6.5% / 8.0%, with scene structure surviving across the whole
   depth range instead of only in a narrow band.
4. **Clamp `_jitter_band` at both ends** (F3).
5. **Raise the severity-1 floor for `dirt_soil` and `denoise_artifacts`** (F6),
   and measure it rather than eyeballing — the check is one line against a
   clean frame.
6. **Add severity-monotonicity tests for the remaining effects** (F5), with
   enough seeds to be meaningful and an effect-appropriate metric for
   `night_ir` (F4) and `ac_flicker`.
7. **Reconsider the night-mode noise floor** (F8) once F2 is addressed, since
   the two interact.

None of these is a correctness bug in the sense of a broken formula. F1 and F2
are calibration and bookkeeping issues that change how the output *looks* at
scale; the rest are test coverage and preset tuning.
