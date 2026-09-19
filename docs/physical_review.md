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

> **Resolved (2026-09-15).** `_jitter_band` now clamps at both ends, with 10%
> headroom above the preset band. Re-measured over a fresh 300-frame sweep:
> severity 5's effective band tops out at 110 mm/h and draws reach 104.7,
> against 137 / 130.6 before. Bands still overlap a little, deliberately — the
> jitter exists to stop two samples sharing a rate — but each now stays inside
> the level it belongs to.

### F4 — `NightIrMode` severity is invisible to any whole-frame metric

Mean |Δ| by severity: **114, 112, 113, 111, 107** — flat, and mildly
*decreasing*. This is not a bug: the auto-exposure deliberately normalises the
level, and the day→night conversion (≈110 levels) swamps everything severity
controls (illuminator reach, gain, noise floor). But it means the severity
ladder for this effect cannot be validated the way the others were, and the
monotonicity test used elsewhere would be meaningless here. It needs its own
metric — noise floor in the shadows, or the near/far falloff gradient.

### F5 — Non-monotonic severity: three real causes, none a code defect

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

> **Resolved (2026-09-15).** `tests/unit/test_severity_ladders.py` now covers
> all 14 (the codec keeps its own CRF test, since two ffmpeg subprocesses per
> render would make the suite minutes long). Re-measured at **12 seeds instead
> of 3, every one is monotonic** — including `exposure_transient` and
> `ac_flicker`, confirming both were sampling noise rather than defects, which
> is what the diagnosis above predicted.

### F6 — Severity 1 is below the visible threshold for three effects

Mean |Δ| at severity 1: `dirt_soil` 0.94, `denoise_artifacts` 0.71,
`ac_flicker` 0.46 — at or under one 8-bit level. The same problem already
fixed for `SunFlare` (which was 0.33 before its floor was raised). A bottom
rung that does nothing wastes a rung and reads as a bug rather than a setting.
`ac_flicker` is partly excused by F5, but `dirt_soil` and `denoise_artifacts`
are simply set too low.

> **Resolved (2026-09-15).** Floors raised for all three; measured on the
> reference photograph they now read **1.79, 2.21 and 1.38** levels, and
> `tests/unit/test_severity_ladders.py` pins the property so it cannot
> regress silently.
>
> Calibrating this surfaced something worth recording. `DirtSoil` measures
> 0.75 levels on a synthetic sky-over-ground scene and 1.79 on a photograph at
> the *same* settings, because it scatters light rather than tinting it and so
> is near-invisible in flat light **by design** — its own unit test pins
> exactly that behaviour. Tuning the preset until it passed on the synthetic
> scene would have over-driven it everywhere real. The severity tests
> therefore calibrate against the reference photograph, while the golden tests
> keep their synthetic scene, where the requirement is reproducibility rather
> than realism.

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
4. **Clamp `_jitter_band` at both ends** (F3). — **done**
5. **Raise the severity-1 floor** for `dirt_soil`, `denoise_artifacts` and
   `ac_flicker` (F6). — **done**, and now pinned by a test.
6. **Add severity-monotonicity tests for the remaining effects** (F5/F4). —
   **done**, 12 seeds, with a noise-floor metric for `night_ir` whose
   auto-exposure makes any whole-frame metric run backwards.
7. **Reconsider the night-mode noise floor** (F8) once F2 is addressed, since
   the two interact.

None of these is a correctness bug in the sense of a broken formula. F1 and F2
are calibration and bookkeeping issues that change how the output *looks* at
scale; the rest are test coverage and preset tuning.

**Status:** F1 withdrawn (the claim was wrong, see the correction); F2–F6
resolved; F7 and F8 recorded as known characteristics rather than defects —
F7 is the documented sub-pixel handling and F8 should be re-examined now that
F2 has changed the depth gradient it interacts with.

---

# Second pass (2026-09-15)

A fresh audit after F2–F6 landed, on an **independent 300-record sweep** (15
effects × 5 severities × 4 seeds, new seed stream) plus a reading of the
rendered gallery in `docs/examples/`. The first pass audited the arithmetic;
this one re-runs those checks against new draws and then looks harder at the
places where correct arithmetic can still produce an inconsistent picture.

## The first pass's checks still hold on new draws

Re-derived from scratch on the new log, nothing carried over:

| Check | Result |
| --- | --- |
| Every sampled value lies inside the range it was drawn from | **180 draws, 0 violations** |
| Rain: `count == (N₀/Λ)e^(−ΛD_min) × V × f_resolvable` | max rel. error **5.8e-5** |
| Snow: `count == n × V × f_resolvable` | residual **< 0.5 particles** (count is an int) |
| Snow: melted→physical `== (ρ_w/ρ_s)^(1/3)` | max abs. error **7.8e-6** |
| Flicker: `depth == nominal × \|sinc(T·f)\|` | max abs. error **7.8e-7** |
| Rain and snow veil β derived from the rate, ±[0.75, 1.3] | **20/20 and 20/20** |
| Streak opacity `α = D/(vT)` | p50 **0.0155–0.0455** (median 0.027), vs Garg & Nayar ~0.017 |

Fog does not log its extinction coefficient (see F11), but it can be recovered
from the logged transmission field, and the ladder it implies is sensible:

| severity | β (1/m) | Koschmieder visibility `3.912/β` |
| --- | --- | --- |
| 1 | 0.0125 | 381 m — light haze |
| 2 | 0.0569 | 74 m |
| 3 | 0.0666 | 61 m |
| 4 | 0.1035 | 40 m — dense fog |
| 5 | 0.1015 | 42 m |

**A note on my own method first.** Three of these checks failed on the first
attempt — rain's count chain by 60%, snow's by 4.9% — and in all three cases
the error was in *my check*, not the code. The rain one is instructive: I
reused the first pass's formula, which assumed a 1 mm minimum drop diameter,
but the gallery now jitters `drop_diameter_range_mm` per variant (0.60–0.80 mm
across the sweep). Reading the bound out of each record instead closed the
check to 5.8e-5. Same failure mode as the withdrawn F1 claim, in the opposite
direction: there I trusted a proxy and reported a defect that did not exist;
here I nearly did it again. **A failing check is a claim about the checker
too.**

## New findings

### F9 — Road wetness is a screen-space band, not the ground

`_apply_ground_wetness` takes no depth argument. It wets the bottom 12–32% of
*image rows* and fakes the reflection by mirroring the band immediately above
it. `RandomRain` cannot be constructed without a depth provider, and this is
the one part of it that ignores the depth map entirely.

On the reference frame, measured against the surface normals the library
already computes:

- the band it does wet **is** ground — mean upward-facing weight **0.82–0.86**
  against 0.65 for the whole frame, so there are few false positives;
- but the widest band covers only **49% of the ground pixels**. Real ground
  runs from row 45 to row 477 (9%–99% of frame height). **51% of the wet road
  is left dry**, with a horizontal seam where the band ends.

This is the failure mode this document exists to hunt: on a bridge-mounted
camera the bottom of the frame *is* the near road, so the picture looks right
and the chain behind it does not hold. Point a camera further down — the
normal CCTV mounting — and the ground fills the frame while the wet band stays
pinned to the bottom third.

The fix is already in the codebase and already used by the neighbouring
effect: `depth/normals.py::upward_facing` is what `RandomSnow` masks
accumulation with. Wetness should read the same mask.

Two smaller things in the same pass. The reflection mirrors about the band's
top edge rather than about the horizon the depth map knows, and
`wetness_probability` is decoupled from the drawn rain rate — the severity
preset ladders it 0.15→0.9, but a caller writing
`RandomRain(rain_rate_range_mm_h=(5, 200))` gets the same 40% chance of a wet
road whether the call drew 5 mm/h or 200. Some decoupling is right (a road
stays wet after the rain stops, and is dry in the first seconds of a
downpour), so this is worth a documented note more than a coupling.

### F10 — Night mode still clips a quarter of the frame (F8, re-measured)

F2 gave the depth map a near plane and F8 was left open pending exactly this
re-measurement. Across the full ladder, 4 seeds per level:

| severity | % saturated (≥250) | % black (≤5) | mean | noise floor |
| --- | --- | --- | --- | --- |
| 1 | 10.4 | 1.7 | 115.1 | 39.3 |
| 2 | 12.4 | 3.6 | 119.0 | 53.8 |
| 3 | 16.9 | 6.2 | 131.7 | 76.5 |
| 4 | **24.4** | 10.6 | 135.6 | 94.4 |
| 5 | 17.8 | 15.5 | 123.7 | 108.8 |

F8's original complaint — **24.1% of the frame at pure black** — is fixed: the
worst level is now 15.5% and the typical one 6%. The noise floor rises cleanly
across the ladder, which is why the severity test uses it.

But the problem moved rather than went away. Clipping now runs to **24.4% of
the frame at severity 4**, and the gallery images show why: the near road
washes to featureless white across the bottom third. A real IR installation
clips on nearby plates and the closest metre or two of tarmac, not on a
quarter of the scene. The earlier "6.5% saturated / 8.0% black" figure quoted
under F2 was a single severity and seed, so it was real but narrower than it
read; the ladder average is the honest number.

The remaining cause is not the depth gradient but the auto-exposure target:
it meters for the whole frame, so an illuminator that blows out the near field
still hits its target and nothing pulls the exposure back. Metering on a
high percentile rather than the mean, or clamping the illuminator's near-field
gain, would fix it. **Open.**

### F11 — Eight of fifteen transforms log no draws at all

The Method section says the log records "the physical quantities actually
sampled, including ones that are local variables and never reach the
renderer". That is true for 7 of 15 transforms. `drew` is absent for
`fog`, `dirt_soil`, `cast_shadows`, `housing_occluders`, `rolling_shutter`,
`denoise_artifacts`, `wdr_tone_mapping` and `video_codec`.

For four of them nothing is actually lost — `rolling_shutter`,
`denoise_artifacts`, `wdr_tone_mapping` and `video_codec` sample their
parameters and hand all of them straight to the renderer, so `out` *is* the
draw list. The gap is real for the other four:

- **`fog`** logs `transmission`, `airlight` and `scatter_sigma` but neither the
  visibility it drew nor the β it derived — so the one chain worth auditing
  (visibility → β → `exp(−βz)`) has to be reconstructed backwards from the
  transmission field, as above, and cannot be closed exactly.
- **`cast_shadows`** logs `direct_fraction` but not the sun elevation or
  azimuth behind it.
- **`housing_occluders`** logs the composited mask but not the insect count or
  sizes.
- **`dirt_soil`** logs the deposit albedo field but not the species it drew.

This bounds what the first pass could have verified, so it belongs in the
record as a limit on the method rather than as a defect in the effects.

### F12 — The recorder changed a field's schema with the particle count — fixed

`_summarise` returned a `{n, shape, min, p50, max, mean}` dict for arrays
larger than 4 elements and a **bare list** for smaller ones. So
`out.flakes.radius.p50` read correctly on a heavy snow frame and raised
`TypeError` on a light one — and the light frames are exactly the ones an
audit of "is severity 1 doing anything" has to read. It broke three of this
pass's own analysis scripts before I noticed it was the log and not the data.

Fixed: every array now summarises to the same dict, with small ones
additionally carrying a `values` key. `tests/unit/test_diagnostics.py` pins
the schema across sizes 1 → 1000.

## What I checked and did not find

Recorded because a measurement that refutes a suspicion is worth as much as
one that confirms it, and three of the four below came from looking at the
gallery and guessing wrong.

- **Particles rendered behind scene geometry.** Distances are sampled
  volume-uniformly over 0.4–6 m while the scene starts at 3 m, so naively
  ~87% of the column sits past the near plane and streaks should be painting
  over solid road. Measured against the depth map at each particle's own
  pixel: **1.6% of rain streaks and 1.8% of snow flakes**. The resolvability
  cut concentrates particles near the camera (median distance 1.5 m), which
  the naive volume argument misses entirely. Not worth an occlusion test.
- **Snow flakes looked like hard-edged white discs** in the gallery. They are
  not: logged `blur_radius / radius` runs **0.68–2.2** (p50 ≈ 1.3), so flakes
  are softened by at least their own width. The hard edges were an artefact of
  the nearest-neighbour upscale in my own inspection crop.
- **Dirt looked colourless.** The palette is genuinely varied — measured over
  16 seeds, deposit albedo spans hue **30°–52°** (grey through yellow to
  brown) at saturations **0.8%–45.5%**, which is what was asked for. The
  rendered result looks neutral at severity 3 because coverage and veil
  strength are low there, not because the palette is narrow. Dirt also carries
  real high-frequency structure, not just a veil: high-pass/low-pass energy
  ratio **0.87–1.37** across the ladder.
- **Precipitation leaving the scene contrast untouched.** It does not — near
  road detail contrast falls to **83% / 64% / 50%** of clear-sky at rain
  severities 1/3/5, and to 34% for fog severity 5. I had also expected to find
  hard sun shadows surviving under heavy rain; the reference photograph is
  diffusely lit to begin with, so there was nothing to find and the question
  needs a different scene.

## Severity ladders: preset vs gallery

`tests/unit/test_severity_ladders.py` (12 seeds, clean presets) has all 14
ladders monotonic. This sweep, on **gallery-jittered** configs at 4 seeds,
shows 6 non-monotonic: `ac_flicker`, `dirt_soil` (6.51 → 6.49), `sun_flare`
(4.24 → 4.23), `exposure_transient`, `video_codec` and `night_ir`.

These do not contradict each other and both are wanted. The presets must
ladder cleanly; the gallery deliberately jitters each level's band so two
samples never share a configuration, which lets adjacent levels cross. Two of
the six cross by under 0.02 levels. The rest are the causes already diagnosed
under F4 and F5 — signed-symmetric error, `sinc` cancellation, auto-exposure —
none of which the jitter creates.

## Status after the second pass

| | |
| --- | --- |
| F1 | Withdrawn — the claim was wrong; documentation corrected |
| F2 | Resolved — near plane added |
| F3 | Resolved — jitter clamped |
| F4 | Resolved — noise-floor metric |
| F5 | Resolved — all 14 ladders tested |
| F6 | Resolved — severity-1 floors raised |
| F7 | Confirmed, not a defect — and it extends to snow: flake radius p50 is **0.89–1.00 px** in 19 of 19 frames, so flake size reaches the image through opacity and defocus, not width |
| F8 | Superseded by F10 — the black crush is fixed, the clipping is not |
| F9 | **Open** — wetness ignores the depth map; 51% of the ground left dry |
| F10 | **Open** — 24% of the frame clipped at severity 4; auto-exposure meters the mean |
| F11 | **Open** — 4 transforms log neither their draws nor enough to reconstruct them |
| F12 | Resolved — recorder schema made consistent and pinned by a test |

Ranked by what they cost: **F9** is the only one that makes a physically wrong
picture on a camera geometry the library explicitly targets, and its fix is to
reuse a mask the neighbouring transform already uses. **F10** is a tuning
problem with a clear cause. **F11** is a gap in the instrument, which matters
mostly because it silently bounds what any future pass of this document can
claim.

Still nothing here is a broken formula. Every derived chain that can be
checked, checks — twice now, on independent draws.
