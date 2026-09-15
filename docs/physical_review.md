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

---

# Third pass (2026-09-15): closing F9, F10 and F11

## F9 — Wetness now reads the depth map

`_apply_ground_wetness` no longer takes a band fraction. It takes the ground
mask, the horizon row and the focal length, and two things changed with it.

**Where the wetness goes.** The mask is `upward_facing(...) ** falloff`, the
same cosine `RandomSnow` already uses for accumulation, with a steeper
exponent (4–9 against snow's 2–5) because water runs off a slope more readily
than snow slides off it. Measured on the reference frame:

| | ground pixels covered | false positives |
| --- | --- | --- |
| screen-space band (widest, 32% of rows) | 49% | — |
| depth-derived mask | **93% ± 3%** | **0.0%** |

**Where the reflection comes from.** A horizontal mirror flips a ray's
vertical angle about the horizontal, so the reflection at row `y` is the scene
at row `2·horizon − y`. The old code mirrored about the band's own top edge,
which put the reflection of a car somewhere the car is not. The horizon comes
from a new `depth/normals.py::horizon_row`: on a ground plane
`1/z = (y − y_h)/(h·f)`, so **inverse** depth is linear in the row index and
crosses zero at the horizon. Fitting that line and reading its root recovers
`y_h` without knowing the camera height, the focal length or the pitch —
verified exact (±0.05 rows) across nine synthetic planes spanning all three.

I nearly "fixed" this estimator for a bias it does not have. On the reference
frame it returns row 120 while the treeline sits near row 185, which looked
wrong until I computed what it should be: for a camera pitched down by θ the
horizon is at `cy − f·tan θ`, which is row 136 at 14° and row 72 at 22°. Row
120 corresponds to ~17°, entirely plausible for a bridge-mounted shot. **The
treeline is scene content above the horizon, not the horizon.** I had also
built a second estimator from the ground-plane normal to replace it; measured
side by side it was worse on both synthetic data (2–4 rows off, from the
normals' smoothing) and the real frame (row 59), so it was discarded.

**How strong the reflection is.** Reflectance is not a constant. Schlick's
approximation over the depression angle below the horizon,
`R = R₀ + (1 − R₀)(1 − cos θ)⁵` with `R₀ = 0.02` for water, gives what a wet
road actually looks like — mirror-bright toward the horizon, merely dark and
matte at your feet. Measured down the frame (horizon at row 121, f = 416 px):

| row | Fresnel R | change vs dry |
| --- | --- | --- |
| 160 | 0.62 | **+4.0%** (sky mirror) |
| 240 | 0.22 | −3.3% |
| 320 | 0.08 | **−7.2%** |
| 440 | 0.03 | −2.1% |

That gradient is the effect. A single blend weight, which is what the band
had, cannot produce it in either direction.

One calibration change came out of looking at the result: the near-field
darkening was too timid. Dry asphalt albedo runs 0.10–0.15 and wet 0.05–0.08,
about a halving, so `wetness_darkening_range` moved from (0.55, 0.85) to
(0.4, 0.7). Near-road darkening went from −4.5% to **−9% … −13%**, which is
where it reads as a wet road rather than a slightly grey one.

`wetness_band_fraction_range` is gone, replaced by `wetness_darkening_range`
and `wetness_falloff_range`. `wetness_probability` is deliberately still
independent of the drawn rain rate, and now says why in its docstring: road
wetness follows the rain's *history*, which a single-frame transform cannot
know — a road stays wet after the rain stops and is dry for the first seconds
of a downpour.

## F10 — The cause was not what I said it was

The second pass concluded the clipping came from "the auto-exposure metering
the frame mean". **It does not — it meters the 80th percentile, and always
has.** I asserted the mechanism from the symptom without reading the six lines
that implement it, which is the same error as the withdrawn F1 claim for the
third time in this document.

The actual cause, visible immediately on reading them, is that gain was
applied **twice**: the AE drives the 80th percentile to `target_level`, and
then `signal = signal * gain` multiplies it again. With `target_level` up to
0.6 and `gain` up to 1.6, the 80th percentile landed at 0.96, so a fifth of
the frame sat against the clipping point by construction.

That is also wrong as physics. Gain is one of the levers an AE loop uses to
*reach* its target, not something applied on top of a level it already hit.
What gain does control is the noise, which the line below it already handled
independently. Removing the signal multiply:

| severity | % saturated before → after | % black | noise floor |
| --- | --- | --- | --- |
| 1 | 10.4 → **7.2** | 3.9 | 42.0 |
| 2 | 12.4 → **7.1** | 7.9 | 60.6 |
| 3 | 16.9 → **7.1** | 11.1 | 80.4 |
| 4 | **24.4 → 7.0** | 13.8 | 101.3 |
| 5 | 17.8 → **7.0** | 16.8 | 123.2 |

Clipping is now flat at ~7% across the ladder — headlights, plates and the
nearest metre of tarmac, which is what a real IR installation clips — instead
of scaling to a quarter of the frame. The noise floor still rises cleanly, so
the severity test's metric is unaffected. Black rises with severity, which is
correct: severity 5 is a long-range scene with a weak illuminator, and its far
field genuinely is black.

## F11 — The four silent transforms now log their draws

`fog`, `cast_shadows`, `housing_occluders` and `dirt_soil` record what they
sample. Fog was the one that mattered: it now logs `beta`,
`scattering_gamma`, `beta_rgb` and the Koschmieder visibility, so the
visibility → extinction → `exp(−βz)` chain can be checked forwards instead of
reconstructed backwards out of the transmission field. `cast_shadows` logs the
shadow count and penumbra, `housing_occluders` the insect count and sizes and
the web count, `dirt_soil` which deposit species it drew.

The other four transforms without `drew` — `rolling_shutter`,
`denoise_artifacts`, `wdr_tone_mapping`, `video_codec` — are left alone
deliberately: they hand every sampled parameter straight to the renderer, so
`out` already **is** the draw list and a second copy would only be a second
thing to keep in sync. `tests/unit/test_diagnostics.py` pins the four that
needed it.

None of this changes a rendered pixel; the goldens confirm it, with only
`rain.png` moving (from F9).

## F13 — A distant flat plane reads as "ground" past ~18° of pitch

Found while checking F9's horizon estimates across the sweep rather than on
one frame. The estimate sat stably at row 120.5 for every drawn pitch from
5.6° to 17.3° — correctly, since it reads the depth map, which encodes the
photograph's true geometry rather than the pitch that happened to be drawn —
and then **collapsed to row 49 at 18.1° and 26.5°**.

The cause is in `upward_facing`, not in the fit. A sky is a flat far plane
whose normal points straight at the camera, so `n · up` there is exactly
`sin(pitch)`, which crosses a lenient 0.3 threshold at 17.5°. Past that the
"ground" mask jumps from 63% to 94.5% of the frame and starts at row 1.

The rendered output was never affected: the falloff exponent (6.5 for
wetness, 3.5 for snow accumulation) drives those pixels to **1–3% weight**
against the road's 41–73%, so nothing visible was ever wet or snowed-on that
should not have been. Only the horizon fit, which used the raw cosine at a
lenient threshold, saw them. Fitting on the same falloff-weighted mask the
renderer uses holds the estimate at **122.5 ± 0.7 rows across 16 seeds**
spanning the whole 0–22° pitch range, against a 71-row bimodal swing before.

Worth recording as a caveat on `upward_facing` rather than a defect in it: the
function is behaving correctly given a pitch, and a surface facing a camera
tilted 26° down genuinely is partly facing up. The trouble is that
`camera_pitch_deg` is a per-call *draw* while the depth map encodes the
scene's actual, unknown geometry, so the two can disagree. Any future consumer
of `upward_facing` should apply its falloff before thresholding, as both
current callers now do.

## Verification sweep

A fresh 300-record sweep after all of the above:

| Check | Result |
| --- | --- |
| Draws inside their declared ranges | **340 draws, 0 violations** (was 180 — the new logging nearly doubled what is checkable) |
| Transforms recording their draws | **11/15**, the remaining four by design |
| Fog: `visibility == 3.912/β` logged forwards | max error **1.1e-2 m** on a 409–672 m visibility |
| Goldens | only `rain.png` and `snow.png` moved, both from F9 |
| Tests | **193 passing**, ruff and mypy clean |

Rain's whole-frame effect rose across the ladder (severity 1: 10.4 → 13.5)
now that wetness covers the road rather than a band of rows.

## Status

| | |
| --- | --- |
| F1 | Withdrawn — the claim was wrong |
| F2–F6 | Resolved |
| F7 | Confirmed, not a defect (sub-pixel handling, rain and snow both) |
| F8 | Superseded by F10 |
| F9 | **Resolved** — wetness reads the depth map; 49% → 93% of the ground, 0% false positives |
| F10 | **Resolved** — gain was applied twice; clipping 24.4% → 7.0% |
| F11 | **Resolved** — four transforms now log their draws, four by design do not |
| F12 | Resolved — recorder schema consistent |
| F13 | **Resolved** — horizon fit no longer sees the sky; render was never affected |

## What three passes of this actually taught

Every wrong call in this document has the same shape, and it is worth naming
because it recurred after being named twice.

- **F1**: claimed the veil double-counts by 7–14%, using particle *count* as a
  proxy for *extinction*. Measured: 0.3%.
- **F10**: claimed night mode clips because the auto-exposure "meters the
  frame mean". It meters the 80th percentile, and six lines of source say so.
  The real cause — gain applied on top of a level the AE had already hit — was
  visible immediately on reading them.
- **F9's horizon**: nearly "fixed" a 65-row bias that was my own bad reference
  point, and had a replacement estimator built and measured before discovering
  it was worse than the one it would have replaced.

In each case a plausible mechanism was inferred from a real symptom and then
reported as the cause. The symptoms were all genuine; **the diagnoses were
guesses wearing the clothes of measurements.** What consistently worked was
the opposite order — measure the specific quantity in the claim, and read the
code that implements the mechanism before naming it.

The same order also refuted four suspicions cheaply in the second pass
(occluded particles, flake hardness, dirt colour, contrast loss), and found
F13, which no amount of looking at pictures would have surfaced because it
never reached a picture.

---

# Fourth pass (2026-09-15): two effects that were invisible in the gallery

Prompted by a report that only one `housing_occluders` frame showed anything
and that `ac_flicker` was hard to notice. Both were real, and they had
different causes — plus a third, in how the gallery chooses what to show.

## F14 — `housing_occluders` produced bit-identical output at three severities

Not "too subtle": **max absolute change of 0**, at severities 1, 2 and 3.

The preset drew counts from `insect_count_range = (0, …)` and
`web_count_range = (0, …)`, so an empty frame was a legal draw at every
level. Measured over 200 seeds:

| severity | insects | webs | empty output |
| --- | --- | --- | --- |
| 1 | (0, 1) | (0, 1) | **28.0%** |
| 2 | (0, 2) | (0, 1) | 20.0% |
| 3 | (0, 3) | (0, 2) | 9.5% |
| 4 | (0, 4) | (0, 2) | 8.0% |
| 5 | (0, 6) | (0, 3) | 3.0% |

The rendering was never at fault: on a non-empty draw the effect is strong
(mean |Δ| 2.3–12.3, peak 93–174 levels). The gallery simply drew three empty
frames in a row, which at those rates is about a 0.5% event — unlucky, and a
fair symptom of a design that permits it at all.

**Whether an effect applies is what `p` is for.** A caller who sets `p=1.0`
has explicitly asked for it and should not get a bit-identical frame back a
quarter of the time; the count should decide how *much* there is, not
*whether* there is any. Insect counts now start at 1 at every level. Webs
stay optional, because a housing can easily carry insects and no web.
Empty output is now **0.0%** at all five levels, and the ladder runs
4.00 → 5.67 → 13.9 → 15.5 → 19.3.

## F15 — Exposure was sampled uniformly, and the sinc cancels the long end

`AcFlicker`'s banding depth is `nominal × |sinc(T·f)|`, which is correct
physics — an exposure spanning a whole flicker period averages the
oscillation away, and arranging exactly that is what a camera's anti-flicker
setting does. The bug was in how `T` was drawn.

Exposure is a **logarithmic** quantity. It is counted in stops, and
1/1000 → 1/500 s is as large a change as 1/8 → 1/4 s. Drawing it uniformly
over (0.001, 0.016) s puts most of the mass at the long end — which is
precisely the end the sinc cancels:

| sampling | median exposure | median `T·f` | median \|sinc\| | frames with \|sinc\| < 0.1 |
| --- | --- | --- | --- | --- |
| uniform (was) | 8.47 ms | 0.85 | **0.217** | 13.7% |
| log-uniform (now) | 4.00 ms | 0.40 | **0.757** | 7.4% |

A 3.5× increase in typical band depth, from correcting the sampling rather
than from raising any amplitude. Measured end to end, mean |Δ| by severity
went from 1.9 / 6.4 / 1.4 / 4.4 / 10.2 (non-monotonic, two levels under two
8-bit levels) to **3.2 / 5.1 / 7.0 / 8.9 / 10.8** — monotonic, with the
fraction of near-invisible frames down from ~14% to 3–8%.

What remains is honest about the effect: the rendered banding is **±3.9% at
severity 1 rising to ±13.5%, across 2–3 broad bands**. That is what 50 Hz
mains and a 12–33 ms readout produce, and it reads as a gentle vertical
gradient at thumbnail scale even though it is plainly visible at full size.
The sinc cancellation is still there and still zeroes the occasional frame,
because that is the physics and it is worth training against.

## F16 — The gallery showed whichever draw came up first

Both findings above were reported from the gallery, and the gallery made
each look worse than it was by rendering exactly one sample per cell.

A cell is one draw from a distribution, and showing the first one
misrepresents it at both ends. Effects that can legitimately draw themselves
out of existence — flicker cancelling on the sinc, a flare whose geometry
puts every artefact off frame — then produce a cell the reader cannot
distinguish from a bug.

`docs/generate_examples.py` now takes **seven draws per cell and keeps the
median** by mean absolute change. Taking the median rather than the strongest
is the point: it drops the degenerate tail without reaching for the
flattering one, so a cell shows what that severity typically looks like.
Only the seed varies — the preset and the per-variant jitter are fixed — so
nothing here can make a level appear stronger than it is.

Across all 75 cells, **none is now under one 8-bit level** (the same
threshold F6 established), against four before.

Also removed: `sun_flare_fixed_*.png`, five stale files from a naming the
generator dropped when the flare's position was randomised. They had not been
regenerated since September 14, and one of them was the near-blank frame that
would have been the next thing to look wrong.

## Status

F1 withdrawn. F2–F6, F9–F16 resolved. F7 and F8 stand as documented
characteristics (sub-pixel particle width; superseded by F10).

The pattern F14 and F15 share is worth naming alongside the diagnostic one
from the third pass: **both are distribution bugs, not formula bugs.** The
occluder renderer and the sinc term are each exactly right, and each was
being fed a sampling choice that made the correct formula produce nothing
most of the time. Every check in this document's first three passes verified
that outputs follow from their inputs; none of them asked whether the inputs
were drawn from a sensible distribution. That is the gap the fourth pass
found, and the severity tests now guard the visible end of it.

## F17 — Band *count*, not band depth, is what makes flicker legible

F15 tripled the flicker's band depth and it was still reported as having no
effect. It did have one: measured peak-to-peak modulation was **27%**. The
picture disagreed with the number, and the picture was right — **the metric
was measuring the wrong thing.**

Two broad bands across a frame is a smooth luminance gradient, and the eye
discounts slow gradients as an exposure difference no matter how deep they
are. What makes flicker read as an artefact is the number of bands, and

    bands across the frame = 2 · f_mains · readout_time

`readout_time_range_s` defaulted to (0.012, 0.033) s, giving **1.2–3.3
bands**. That range is a fast sensor: 12 ms is an 83 fps readout. A rolling
shutter reads out over roughly its frame period, so a 25–30 fps camera takes
33–40 ms and a bandwidth-limited 17 fps stream takes 60 ms. The default is
now (0.020, 0.060) s — 50 fps down to 17 fps — for **2–6 bands** at 50 Hz
mains and 2.4–7.2 at 60 Hz.

Second, the lamp was a pure sinusoid. Only an incandescent filament is one,
and only because its thermal mass low-passes the supply; a discharge lamp
follows `|sin|` and a cheap LED driver conducts in narrow pulses near the
mains peaks, both of which carry harmonics that give bands visible edges. The
lamp is now `|sin(πu)| ** pulse_exponent` over the flicker period, with a new
`pulse_exponent_range` defaulting to (1, 5): 1 is a discharge lamp, 2 is
exactly a sinusoid, 4+ an LED pulse train.

**Each harmonic is box-filtered by the exposure independently**, so each gets
its own `sinc(n·T·f)`. That is not a detail — it is what keeps anti-flicker
cancellation exact: when `T·f` is a whole number so is every multiple, and
the entire profile vanishes together rather than leaving harmonics ringing.
A test pins that at a strongly harmonic lamp setting.

| | bands | mean \|Δ\| by severity | peak-to-peak at sev 5 |
| --- | --- | --- | --- |
| originally reported | 2.2 | 1.9 / 6.4 / 1.4 / 4.4 / 10.2 | — |
| after F15 (log-uniform exposure) | 2.2 | 3.2 / 5.1 / 7.0 / 8.9 / 10.8 | 20.6% |
| after F17 | **3.9** | 2.8 / 4.5 / 6.2 / 7.8 / 9.4 | **21.2%** |

The mean |Δ| barely moved — correctly, since nothing about the amount of
light changed. What moved is entirely in how it is distributed down the
frame, which is the whole of the complaint and none of what the metric saw.

Severity 1 remains genuinely subtle (6.3% peak-to-peak), which is what a
bottom rung is for, and the sinc still zeroes the occasional frame outright,
which is the physics and is worth training against.

**The lesson is a new one for this document.** The earlier failures were
mechanisms asserted without measurement. This was the reverse: a measurement,
correctly executed, of a quantity that did not answer the question. Peak-to-
peak modulation is a real property of the output and says nothing about
whether a human can see it. The check that would have caught it — and the one
that did, eventually — was looking at the rendered frame at full size next to
a clean one.

## F18 — The gallery override discarded the library fix, twice

F17's readout-time widening never reached a single rendered frame. The
gallery's variant override for `ac_flicker` set

    "readout_time_range_s": _span(rng, 0.008, 0.033)

— a hardcoded copy of an older library default, which silently overrode the
new (0.020, 0.060). Every published frame kept rendering **0.8–3.3 bands**,
the exact regime F17 identified as illegible, while the library's own tests
passed throughout because they test the library.

**An override that duplicates a range the transform already owns is a copy
waiting to go stale.** The override exists to alternate 50/60 Hz mains; the
readout was incidental and is now gone, so the library's range stands.

This is also how the same report could be right three times running. Each
round I measured the library and the library was fine.

`tests/unit/test_gallery_configs.py` closes it: for every effect and
severity, it builds the configuration **the gallery builds** — overrides
included — renders the reference photograph, and requires the median of 8
seeds to move it by at least one 8-bit level. It fails on the old override,
and it would have failed on F14's empty occluders too.

Writing that test immediately surfaced one more thing worth fixing rather
than asserting around. Even with a corrected readout, `ac_flicker`'s draws
were bimodal — half of them near zero — because the default exposure range
(1–16 ms) straddles the sinc's first null at 10 ms (50 Hz) and 8.33 ms
(60 Hz), and at 60 Hz nearly reaches the second. The default now stops at
**8 ms, below the first null**.

That is not a fudge to make the effect appear. A shutter shorter than one
flicker period is the condition under which banding exists at all, and a
camera sitting on the null is a camera whose anti-flicker setting is
working — not one you would put in a set labelled "shows flicker banding".
It is the same principle as F14: **whether the effect happens is what `p` is
for; the parameters decide how strongly.** Cancellation remains fully
modelled, tested, and available to anyone who widens the range deliberately.

| exposure range | \|sinc\| median, 50 Hz | 60 Hz | draws essentially blank |
| --- | --- | --- | --- |
| 1–16 ms (was) | 0.758 | 0.663 | 17–24% |
| 1–8 ms (now) | **0.875** | **0.820** | 0% / 7% |

Published frames now carry **8.0 / 15.0 / 13.8 / 22.9 / 23.0%** peak-to-peak
across 3–6 bands, against 6.2 / 5.2 / 15.3 / 16.6 / 21.4% over 2 bands.

**What this one is really about.** Three consecutive rounds of "still no
effect" were three correct reports, and each time I verified the fix against
the layer I had just edited rather than against the artefact the report was
about. The library was never the thing being looked at. The rule that falls
out is narrower than "measure more": **verify at the layer the complaint
names.** The test now does that, so the next such regression fails in CI
rather than in the gallery.
