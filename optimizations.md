# Optimization assessment

An assessment of where CPU time actually goes, and of three specific
proposals: precomputed LUTs over the physically reasonable parameter space, a
provider that precomputes and batches randomised model results for transforms
to draw from, and running depth estimation at reduced resolution.

**No code has been changed.** Everything below is measured against the current
implementation so that the proposals can be judged against numbers rather than
intuition.

> **Partly overtaken.** This was written before the `native-kernels` branch,
> and none of its figures have been re-measured. That branch draws rain
> streaks and snowflakes in C, which supersedes D1, and brings the colour
> round trip from 31 ms to about 2 ms at 1024 px, which removes most of the
> case for D2. A2 was done earlier. C2, C3, A1 and D3 stand as written. See
> [`docs/native_kernels.md`](docs/native_kernels.md).

## Headline

> **The sampling of physical quantities is 8.5–11.6% of the cost of rain and
> snow. Rasterisation and filtering are the other ~90%.**

That bounds the two precompute proposals: done perfectly, with zero overhead
and no quality cost, precomputing every drawn quantity would make rain about
11% faster. It is not where the time is.

The depth proposal is more interesting than it looks, but not for the stated
reason — feeding the estimator a smaller image saves nothing, because the
model already resizes to a fixed 518×518 internally. The **corollary** is
where the win is: the depth map has no information above 518×518, so every
downstream computation performed on it at full resolution is work on
interpolated data.

Ranked by measured payoff, the opportunities are:

| # | Change | Measured effect | Quality risk |
| --- | --- | --- | --- |
| 1 | Batch streak rasterisation by alpha | **4.3×** on the primitive | quantisation, bounded and measurable |
| 2 | Replace `AutoImageProcessor` with a direct resize | **31.2 → 4.3 ms** per depth estimate | resize kernel differs |
| 3 | Surface normals at 518 px with scale-matched smoothing | **6.6×**, 99.4% mask agreement | small, at silhouettes |
| 4 | Keep the pipeline in linear light between transforms | **~31 ms per transform** at 1024 px | none, if done properly |
| 5 | Precompute droplet shading fields on a canonical grid | removes most per-droplet work | none — they are analytic |
| 6 | 16-bit LUT for `linear_to_srgb` | 12.4 → 8.9 ms | **exact** |
| 7 | Precompute/batch the physical draws | ≤ 11% ceiling | diversity, if pooled |

---

## Method

All measurements on one machine (Intel Core Ultra 9 285H, RTX PRO 1000
Blackwell Laptop GPU, torch 2.11 + CUDA 12.8), on the reference photograph at
1024×1024 unless stated. `cProfile` for attribution, `perf_counter` medians
for timings. Depth-conditioned transforms use a cached provider so the numbers
isolate the transform's own work.

Reproduce the attribution with:

```bash
PYTHONPATH=. python -m cProfile -s tottime scripts/benchmark.py --only rain --sizes 1024
```

---

## Where the time actually goes

### Rain — 297 ms at 1024 px

| Stage | ms | Note |
| --- | --- | --- |
| sampling the physical draws | 34.3 | **11.6%** |
| rasterising the streak layer | 262.2 | 88.4% |

Inside the rasteriser, the profile is unambiguous: **16 000 individual
`cv2.polylines` calls per frame**, one per streak, each into its own freshly
allocated patch. The per-streak Python loop, not the drawing, is the cost.

### Snow — 314 ms at 1024 px

Sampling 26.7 ms (**8.5%**), rasterisation 287.7 ms. Same shape as rain.

### Water droplets — 204 ms at 1024 px

> **Superseded.** The renderer measured here drew each droplet as its own
> patch. It now builds one height field for all the water and refracts the
> frame through it in a single remap, which is what A1 and D3 below were
> reaching for by other means: there is no per-droplet shading or blur left
> to tabulate or batch. It did not make the transform cheaper overall. Timed
> back to back against the renderer it replaced, it costs 17% more at 512 px
> and 7% less at 1024 px: the per-droplet work went, and a full-frame
> refraction, an aperture-disc convolution and three scatter lobes came in.

~280 droplets rendered per call (the condensation pass contributes most of
them), each with **2 `GaussianBlur` calls and a `remap`** — 561 blur calls per
frame. Sampling is negligible.

### WDR tone mapping — 130 ms

84% is one `bilateralFilter`. It already computes the base layer on a
downsampled copy above a radius threshold, which is why it is 130 ms and not
seconds.

### Temporal denoise — 224 ms

85% is `edgePreservingFilter`, already the linear-time alternative to the
bilateral it replaced.

### Colour conversion — paid by every transform

`to_linear_unit` → `from_linear_unit` costs **31.4 ms at 1024 px** (3.9 ms at
512). Every transform pays it independently, so a `Compose` of five pays it
five times.

---

## Proposal A — precomputed LUTs over the parameter space

**Verdict: right idea, wrong target as stated. Two specific places where it
is genuinely worth doing.**

The proposal imagines tabulating the physical models — drop-size
distributions, terminal velocities, droplet distributions — over the plausible
parameter space. Measured, those evaluations are inside the 8.5–11.6% sampling
slice, and most of them are already closed-form numpy over a few hundred
elements. Tabulating `raindrop_terminal_velocity` saves microseconds.

Two places in the same spirit *do* pay, because they tabulate **rasterisation
work**, not sampling:

### A1. Droplet shading fields (recommended)

`_render_droplet` computes, per droplet: a centred grid, `dist_norm`, `angle`,
the local radius, the Snell rim bend, a surface normal, a Blinn-Phong
specular term and a crescent term. Expressed in **normalised** coordinates
(offset ÷ radius), every one of those except the `cv2.remap` sample is
independent of the droplet's radius, position and the scene behind it.

They are analytic functions of normalised position, so evaluating them once on
a canonical grid (say 256×256) and resampling per droplet is exact up to
resampling error — which is negligible if the canonical grid is larger than
any droplet drawn. The only per-droplet work left is the `remap` that samples
the scene, which is irreducible because it is the scene-dependent part.

Irregularity and phase perturb the rim per droplet, but only through
`local_radius(angle)`, which is a **1-D** angular modulation and can be applied
on top of the 2-D field.

### A2. `linear_to_srgb` (recommended, exact)

Currently a `pow` over the whole float array. A 16-bit LUT indexed by
quantised linear value is **12.4 → 8.9 ms** at 1024 px and, measured against
the current implementation, has **zero error in the 8-bit output** — 65 536
linear steps are finer than the sRGB output quantisation everywhere,
including the steep near-black region.

| LUT size | ms | max error (8-bit levels) |
| --- | --- | --- |
| 12-bit | 10.6 | 0.804 |
| 14-bit | 11.1 | 0.201 |
| 16-bit | **8.9** | **0.000** |

The forward direction already uses a 256-entry LUT for `uint8` input.

### What not to tabulate

Anything keyed on more than two continuous parameters. A LUT over
`(rate, diameter, distance, focal, exposure)` is a five-dimensional table
whose interpolation error is unbounded and whose memory cost is absurd, to
replace arithmetic that is already vectorised. `dispersion_ring_profile`
records the same lesson: continuous random arguments gave an `lru_cache`
there a **0% hit rate** (see B1).

---

## Proposal B — a provider that precomputes and batches randomised results

**Verdict: the ceiling is ~11%, and the diversity cost is real. Not
recommended in the general form; one narrow version is worth considering.**

The ceiling is the sampling share measured above. Against that, the costs:

**Diversity.** A pool of precomputed populations is drawn from repeatedly
across an epoch. Two images sharing a population share their drop count, size
distribution and per-drop opacities. Re-randomising positions hides this
visually while leaving the *statistics* correlated — and correlated
augmentation is precisely the failure mode `test_randomization.py` exists to
catch. It would pass, because it checks that two seeds differ, not that N
samples are independent.

**Seeding.** Transforms draw through their own RNG so `A.Compose(..., seed=)`
is reproducible. A shared pre-filled pool is process-global state; making it
reproducible per-sample means indexing it by the sample's seed, at which point
it is a cache keyed on something that never repeats.

**Worker duplication.** Each `DataLoader` worker is a separate process, so the
pool is built and held per worker. At 8 workers, an 8× memory cost for a ≤11%
saving.

### B1. The narrow version worth considering

Precompute **only** the parts that are genuinely invariant across images and
parameters, and do it lazily per process:

- The sRGB↔linear LUTs (already done for the forward direction).
- The droplet shading fields of A1, which depend on nothing drawn.

That is not "precomputed randomness". It is caching the deterministic part,
which carries none of the diversity or seeding problems above.

The codebase already contains both the positive and the negative example of
exactly this, and they are worth reading before adding a third.
`diffraction_starburst` is `lru_cache`d on blade count — a small discrete
parameter space, so it hits. `dispersion_ring_profile` deliberately is not,
and says why in its docstring: its two parameters are drawn from continuous
ranges, an `lru_cache` measured a **0% hit rate** over 40 varied calls, and
quantising the key enough to collide would have been coarse enough to see.
The discriminator is whether the key is discrete and low-cardinality, not
whether the computation looks expensive.

---

## Proposal C — depth at lower resolution

**Verdict: the stated form saves nothing. The corollary is the second-largest
win available.**

### C1. Downsampling the input does nothing

`AutoImageProcessor` resizes to the model's fixed 518×518 regardless of input
size, so inference cost is already resolution-independent:

| input | preprocess | model | interpolate + copy | numpy | total |
| --- | --- | --- | --- | --- | --- |
| 512 px | 19.9 | **28.3** | 0.5 | 0.5 | 49.3 ms |
| 1024 px | 14.9 | **28.3** | 0.9 | 2.0 | 46.2 ms |
| 1920 px | 19.0 | **28.1** | 2.5 | 7.8 | 57.4 ms |

Feeding a 512 px copy of a 1024 px frame saves the difference between two
resizes — under a millisecond — and loses the scene detail the model would
have seen.

### C2. Replace `AutoImageProcessor` (recommended)

Preprocessing is 15–20 ms, a third of the total, for a resize and a
normalise. Doing it directly:

```
AutoImageProcessor : 31.2 ms
direct cv2 resize  :  4.3 ms
```

The output tensors differ by mean 0.015 because the resize kernels differ
(`INTER_AREA` vs the processor's default). That is a change in what the model
sees, so it needs validating against depth output rather than assumed — but a
different resize kernel is well within the noise of a monocular depth
estimate, and the saving is ~27 ms on every depth-conditioned frame.

### C3. Do downstream work at 518 px (recommended)

This is the real version of the proposal. The depth map is bicubic-upsampled
from 518×518, so it carries **no information above that resolution**:

> Round-tripping the 1024 px depth map through 518 px changes it by
> **0.017 m on a 3–50 m range — 0.036%.**

Yet `upward_facing` — Sobel gradients, back-projection, normalisation,
smoothing — runs at full resolution for every consumer (snow accumulation,
road wetness, cast shadows, the horizon fit). At 1024 px that is **98.7 ms**.

Computing it at 518 px and upsampling the result, **with the smoothing sigma
scaled to match**, gives:

| resolution | ms | speed-up | mean abs diff | ground-mask agreement |
| --- | --- | --- | --- | --- |
| 256 px | 3.1 | 31.8× | 0.0196 | 98.01% |
| **518 px** | **15.0** | **6.6×** | **0.0060** | **99.39%** |
| 724 px | 39.0 | 2.5× | 0.0028 | 99.79% |

Scaling the sigma matters: at the default `smooth_sigma_px=3.0` regardless of
resolution, 518 px gives only 96.55% agreement, because the smoothing then
covers twice the scene extent. With it scaled, 518 px is the natural choice —
it is exactly the resolution the depth data actually has.

The residual disagreement is at object silhouettes, where the depth gradient
is enormous and the normals are least trustworthy anyway.

---

## Opportunities not proposed, ranked above most of the above

### D1. Batch streak rasterisation by alpha (largest single win)

Rain issues **16 000 `cv2.polylines` calls per frame**, one per streak.
Bucketing streaks by opacity and issuing one call per bucket, on a shared
canvas:

| approach | ms (16 000 streaks) | speed-up |
| --- | --- | --- |
| one call per streak | 13.4 | — |
| 8 or 16 buckets | **3.1** | **4.3×** |

Alpha quantisation error at 16 buckets is **0.0009 mean, 0.0019 max** against
opacities spanning 0.02–0.08 — around 2% of the value, well under the
variation the model already draws.

Two caveats worth stating. This measures the drawing primitive in isolation;
the real `_render_streak_layer` also allocates a patch and blurs per streak,
so the end-to-end gain will be smaller than 4.3× until those are batched too.
And drawing a bucket onto a shared canvas **overwrites** where streaks cross,
whereas the current model accumulates occupancy before
`coverage = 1 − e^−occupancy`. At the measured coverages (0.17–0.48% of
pixels) crossings are rare, but the effect is a systematic under-count that
should be measured, not assumed — rendering to a per-bucket layer and summing
preserves it exactly, at 1.5–1.8× instead of 4.3×.

### D2. Keep the pipeline in linear light

Every transform does sRGB → linear → sRGB independently: **31.4 ms per
transform at 1024 px**. A five-transform `Compose` spends ~157 ms converting
back and forth between operations that all want linear.

A linear-light pipeline mode — convert once on entry, once on exit — would
remove four of those five round trips. This is the largest *structural* saving
available and the only one that grows with pipeline length.

It is also the most invasive: it changes the contract between transforms, and
interleaving a third-party augmentation that expects sRGB would then be
wrong. A `LinearLightCompose` that converts at its boundaries and passes
float32 linear internally would keep the ordinary path untouched. A2's LUT is
the cheap partial version of the same idea.

### D3. Water droplet blur batching

561 `GaussianBlur` calls per frame, two per droplet, on small patches. Droplets
within a defocus bucket could share one separable blur applied to a composited
layer. Combined with A1 this is most of the droplet transform's cost.

---

## What to avoid

**Do not cache across images by content.** `SharedDepthProvider` recognises
a scene by array *identity*, deliberately: it hits within one sample and never
across samples. A content hash would appear to help on a benchmark that reuses
one array and do nothing in training.

(Identity here means a weak reference, with a sparse fingerprint of the pixels
as a guard — not `id()`, which is reused the moment a frame is freed and served
the previous frame's depth to the next one. And identity is carried from each
transform's input to its output, which is what makes the within-sample hit
actually happen; without that the second transform in a chain never saw the
array depth was estimated for. See `depth/provider.py`.)

**Do not reduce particle counts to buy speed.** `max_streaks` and
`max_flakes` are render budgets, and `width_cut_for_budget` already trades
them against the resolvability cut in a way that keeps the severity ladder
monotonic. Lowering them changes the physics the transform claims.

**Do not approximate the filters by shrinking their radius.** WDR's halo width
and the denoiser's flattening scale *are* the artefacts. Downsampling the
input to the filter and scaling its radius preserves them; shrinking the
radius does not. Measured on the bilateral base layer, a ¼-resolution copy
with a ¼ radius differs by 5.3 levels mean and 146 max from full resolution —
so even the legitimate version needs its error measured per use, not assumed.

**Measure end to end, not the primitive.** Every number above for a *primitive*
(4.3× for polylines, 6.6× for normals) is an upper bound on the end-to-end
gain. The session that produced this library has a documented history of
plausible-looking proxies being wrong by an order of magnitude; see
`docs/physical_review.md`.

---

## Does any of it matter? A realistic pipeline

Per-transform costs do not say whether optimising is worth the risk. A full
pipeline does: 14 effects in `physical_order`, each at `p=0.15`, measured over
30 images.

| size | median | mean | p90 | per worker | 8 workers |
| --- | --- | --- | --- | --- | --- |
| 512 px | 104 ms | 193 ms | 384 ms | 5.2 img/s | **41 img/s** |
| 1024 px | 235 ms | 284 ms | 574 ms | 3.5 img/s | **28 img/s** |

**41 images/second from eight workers is a bottleneck**, not a rounding error.
A mid-size detection model on a current GPU consumes several times that, so
these transforms would decide the training step time rather than the network.
The mean sits well above the median because cost is driven by which effects
happen to fire, and the tail reaches 1.2 s.

So: yes, optimising is worth doing. But the ranking depends on resolution in a
way the per-transform numbers hide.

### Where the 104 ms goes at 512 px

Expected cost per image = per-transform cost × probability, with the depth
estimate charged once whenever any of the five depth-conditioned effects
fires (56% of images at `p=0.15`). This model predicts **101 ms** against the
**104 ms** measured median, so it can be trusted to target work:

| | ms | share |
| --- | --- | --- |
| **depth estimate** | **25.6** | **25.4%** |
| WDR tone mapping | 16.4 | 16.2% |
| water droplets | 15.2 | 15.1% |
| rain | 11.4 | 11.3% |
| everything else (10 effects) | 32.2 | 32.0% |

### The ranking is resolution-dependent

**C3 saves nothing at 512 px.** The depth map is 518×518, so at a 512 px crop
there is nothing to downsample — `upward_facing` costs 22.5 ms and would cost
22.5 ms after the change. It is the top item at 1024 px and worthless at the
resolution many pipelines actually train at. Measured:

| training size | C3 saves |
| --- | --- |
| 512 px | **0** |
| 1024 px | ~84 ms per call, ~4% of pipeline |
| 1920 px | more |

**C2 is the one that pays at every resolution**, because the depth estimate
costs the same 46 ms whatever the frame size — which is exactly why
preprocessing being a third of it matters. Replacing `AutoImageProcessor`
takes the estimate from ~46 ms to ~19 ms, and the amortised depth cost from
25.6 ms to ~10.6 ms: a **~15% cut in total pipeline cost from one function**.

## Suggested order

1. **C2 — direct preprocessing.** The only item that is both large and
   resolution-independent: ~15% of the whole pipeline, one function, and the
   depth map is easy to diff before and after. Do this first regardless of
   what else gets done.
2. **A2 — 16-bit LUT for `linear_to_srgb`.** Exact, local, an hour's work.
   Small in absolute terms (~1 ms at 512 px) but it is free and correct.
3. **C3 — normals at 518 px with scale-matched sigma.** Only if you train
   above ~724 px. At 512 px, skip it.
4. **WDR's bilateral and the droplet renderer (A1 + D3).** Together 31% of
   the 512 px budget, and the largest remaining targets. Both are real work.
5. **D1 — bucketed streak rasterisation.** 11% of the budget, and needs the
   occupancy-accumulation question settled first.
6. **D2 — linear-light pipeline.** Highest ceiling, most invasive. Only worth
   it if pipelines routinely chain several of these.

Proposal B is not on the list. Its ceiling is 11% of two transforms, and it
trades sample independence to get there.

### The cheapest optimization is not in this document

At `p=0.15` the pipeline already costs 104 ms because 14 effects are
configured. Dropping the two most expensive effects you do not actually need —
or halving their `p` — costs nothing to implement and is available today.
`WdrToneMapping` and `WaterDroplets` are 31% of the budget between them.

## Verifying any of this

Every change above must leave the images identical or provably close. The
suite already has the instruments:

- `tests/visual_regression/` — golden images; an optimization that changes a
  pixel has to say why.
- `tests/unit/test_severity_ladders.py` — a quantisation that flattens a
  severity step fails here.
- `tests/unit/test_gallery_configs.py` — catches an effect becoming invisible.
- `scripts/diagnose.py` — the drawn quantities must be unchanged for anything
  that touches sampling.
- `scripts/benchmark.py` — before and after, same machine, same sizes.

For anything affecting depth (C2, C3), compare the depth map and the derived
masks directly rather than trusting the rendered frame: a wrong mask is easy
to miss by eye and trivial to measure.
