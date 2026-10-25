# Why rain streaks are faint, and what actually changes that

Rain rendered by this library reads as a wet road and drops on the glass,
with a distance haze in long views, but rarely as streaks. This is the investigation into whether that is a
modelling error, and what the levers really are. Short version: **the streak
photometry is correct, the drop-size distribution is not the lever, and the
binding constraint is that a raindrop is sub-pixel at surveillance
resolutions.**

Everything here is measured on the reference photograph at 512 px unless
stated.

## The photometry checks out

A streak's opacity is `α = w² / L` — drop image width `w` px, streak length
`L` px. That follows from Garg & Nayar: the drop's image sweeps `L` pixels
during the exposure while covering `w²` px² at any instant, so each pixel
along the path sees it for `w²/L` of the exposure. Light per streak is
`α · L · w = w²`, independent of exposure, which is the conservation the
model needs.

Measured against the implementation at 50 mm/h: α p50 **0.027**, streak
length p50 **9.6 px**, thickness p50 **1 px**. Garg & Nayar's reference case
is ~0.017 for a 1 mm drop at 1/60 s. The radiance is generous rather than
mean: a streak takes the illuminant, **6.0× the road's radiance** and 3.3×
the sky's on this frame.

So a streak lifts the pixels it crosses by about **8 sRGB levels**.

## Why that is invisible

The asphalt's own texture is **17 sRGB levels** at one sigma.

> A correct streak sits **0.48 σ** into the background texture, on a line one
> pixel wide.

That is the whole finding. Nothing is missing from the render; the signal is
genuinely below the scene's own variation.

## Four levers, three of which do not work

**Exposure does not help.** Lengthening the shutter from 1/333–1/100 s to
1/30–1/15 s takes the median streak from 10 px to 76 px — and drops α from
0.027 to 0.0034, because `α = w²/L`. Measured high-frequency energy added by
the rain: **4.21 → 4.13**. Flat. A longer exposure spreads one drop's light
further; it does not produce more of it.

**A shorter one does change the picture, though it is the same light.** That
measurement only went longer. Going the other way, on a 1280 px road scene at
30 mm/h with a 0.4 mm aperture, the high-frequency energy the rain adds is
**5.9** at 1/30 s, **7.8** at 1/250 s and **9.7** at 1/2000 s (5.4, 6.7 and
8.5 behind a 2.5 mm aperture, which defocuses the near drops). A fast shutter
concentrates each drop's light into a fleck instead of spreading it along a
streak, and a fleck clears the background's texture where a streak does not.
Which shutter a camera runs is set by the light and its aperture, so
`RandomRain` now meters it — `T = K N² / E` — instead of drawing it from
1/333–1/100 s, where every frame got the same middling streak.

**Rendering more of the column does not help much.** Dropping the
resolvability cut from 0.40 px to 0.08 px and raising the budget takes the
drawn count from 1 756 to 60 000 — 34× — for a high-frequency change of
**5.14 → 6.29**, 22%. Each extra drop is fainter in exactly the proportion
that there are more of them.

**The drop-size distribution does not help.** Replacing Marshall & Palmer's
exponential with Ulbrich's gamma at `mu = 3`, at a fixed rain rate:

| | drops > 2 mm | Σ D² (projected area) | resolvable fraction |
| --- | --- | --- | --- |
| exponential (`mu = 0`) | 11.5% | 2040 | 0.04% |
| gamma `mu = 3` | **14.8%** | 2092 (**+2.5%**) | 0.04% |

The shape is genuinely better — see below — but visibility follows the summed
projected area, and that barely moves.

**Resolution and focal length are the real lever, and they are constrained.**
`w = f·D/z`, so a 2 mm drop reaches 1 px only within `z = f·D`:

| sensor / field of view | focal (px) | 2 mm drop is 1 px wide within |
| --- | --- | --- |
| 512 px, 60° | 443 | **0.44 m** |
| 2048 px, 60° | 1774 | 1.8 m |
| 4000 px, 60° | 3464 | 3.5 m |

This is why photographs show rain and 512 px CCTV crops do not. But rendering
at 2048 px does not recover it either, because **defocus eats the gain**: at
1.4 m with the lens focused at 20 m the circle of confusion is ~2 px, so a
2 px drop is drawn 4 px wide at half the opacity. That is correct optics for a
wide-angle lens focused at distance — the same deep depth of field that keeps
the scene sharp is what smears the near drops.

## So when *is* CCTV rain visible?

Against a dark background, which is where the 8-level lift stops competing
with 17 levels of asphalt texture — and above all **at night under the
camera's own illuminator**, where drops a metre from the lens are lit
directly while the scene behind them is not. That is the footage everyone
pictures when they picture visible CCTV rain.

`NightRain` renders that case. It is the same drop physics — the same
Marshall-Palmer sampling, the same `α = w²/L` — with two terms changed:

* the streak's radiance comes from the lamp and falls as `1/z²`, so a drop at
  0.4 m returns a hundred times what one at 4 m does and saturates, and
* it **adds** rather than occludes, because a drop scattering the lamp's
  light back is not hiding anything behind it.

Both sides of the contrast move at once: the drop gets brighter than anything
in the scene while the scene goes dark. That is the whole difference between
invisible and dramatic, and it needed no change to the drop model.

It also carries the backscatter glow from the unresolved drops the beam
lights — the wall of light that makes an IR camera nearly useless in heavy
rain — as a separable term, because it is the part that destroys contrast.

`NightRain` needs no depth provider (the drops are in front of the scene and
the beam is a property of the lamp), so unlike every other atmospheric effect
here it runs on CPU. It sorts **after** `NightIrMode` in `physical_order`:
the streaks are already in infrared terms, so passing them through the
scene's reflectance-to-IR conversion would convert them twice.

## What changed

`RandomRain` gained `dsd_shape_mu`, defaulting to **0** — Marshall & Palmer,
exactly as before, because every published measurement in this repository was
taken against it. Setting it to 2–4 selects Ulbrich's gamma, which is the
better fit to observed spectra: it removes the pile-up of very small drops an
exponential predicts and puts the mode at a finite diameter.

Two details worth recording, both found while implementing it:

**Marshall & Palmer's constants are not self-consistent.** Integrating their
own `N(D) = 8000 exp(-ΛD)` with `Λ = 4.1 R^-0.21` returns a rain rate 4% high
at 100 mm/h and **20% high at 5 mm/h**, because the intercept and the slope
relation were fitted separately. Calibrating the gamma path absolutely would
therefore have put a 10% step at `mu = 0` that is an artefact of that
inconsistency, not a property of the shape. The gamma density is calibrated
*relative* to Marshall & Palmer at the same rate instead, so `mu = 0` is exact.

**The sampler and the resolvable fraction share one description.** The gamma
has no closed-form inverse CDF, so it is inverted on a grid — and the
survival function used to size the sample is built by quadrature on the same
grid. Keeping those in separate closed forms is how a sampler and its
acceptance rate drift apart.

**The transform has to pass the shape on, too.** For a while it did not:
`RandomRain` computed the gamma *slope* and then sized and sampled its drops
as if the distribution were still exponential. A steeper slope without the
`D^mu` term that balances it belongs to neither model, so asking for the
gamma gave fewer and *smaller* drops — at 50 mm/h, 62% fewer streaks and 2% of
drops above 2 mm against the gamma's 14.8%. The helpers and their tests were
correct throughout, which is why nothing caught it; the test that does now
asserts on what the transform draws rather than on what the helpers return.
`NightRain` was unaffected.

## The veil was eight times too strong

The haze that distant rain adds was derived from `V = 1.9 R^-0.63` km,
labelled the standard visibility-in-rain relation. It is not one. Atlas
(1953), which is, gives an extinction of `0.25 R^0.63` per km for ordinary
rain and `V = 11.6 R^-0.63` km. The exponent was right and the constant was
8.2 times too large in extinction, at every rain rate.

Three routes that do not share an error agree with Atlas:

| rain rate | old constant | Atlas (1953) | Carbonneau (optical links) | Marshall-Palmer, summed |
| --- | --- | --- | --- | --- |
| 5 mm/h | 0.69 km | 5.7 km | 5.4 km | 3.9 km |
| 20 mm/h | 0.29 km | 2.4 km | 2.1 km | 1.6 km |
| 40 mm/h | 0.19 km | 1.5 km | 1.3 km | 1.1 km |
| 80 mm/h | 0.12 km | 1.0 km | 0.8 km | 0.7 km |

Visibility at 2% contrast. Carbonneau is `1.076 R^0.67` dB/km, the
temperate-climate fit used for free-space optical links. The last column
sums geometric cross-sections over the drop-size distribution the library
already samples from, `pi N0 / Lambda^3`; it sits above the others because
it counts light diffracted forward into the lens as lost. The two published
figures are taken from papers quoting them, not from the originals.

`rain_extinction_coefficient` now returns Atlas's value. Two consequences:

* **Against the default 3-50 m depth range the veil nearly disappears.** At
  40 mm/h the light surviving to 50 m goes from 35% to 88%. The old veil was
  doing most of the work of making daylight rain visible, and it was doing
  it with the haze of rain eight times heavier. What is left by day is the
  wet road, the streaks and flecks, and the water on the glass, which is the
  finding of this document rather than a regression against it.
* **Against real distances rain now looks like rain.** With a provider
  calibrated to a city road 900 m deep, severity-3 rain used to hide
  everything beyond two hundred metres; it now greys the skyline.

**The severity ladder was re-pitched to match.** With the veil at its true
strength, level 3 (28-54 mm/h) left a 3-50 m frame close to clean. Rates now
run from 2-12 mm/h at level 1 to 80-150 mm/h at level 5, and the ground
carries the ladder with them: `wetness_probability` rises from 0.45 to 0.95
and `wetness_darkening_range` from damp (0.75-0.9) to soaked (0.4-0.6). The
mirror on wet ground now follows the darkening, so light rain darkens a road
without turning it into one. Mean change on the reference photograph, levels
1 to 5: 5.8, 12.3, 16.3, 19.3 and 24.3 levels.

A note on snow, which was checked against the same error and not changed:
`V = 0.6/R` km could not be confirmed against Rasmussen et al. directly, and
their own data scatter by a factor of 3 to 10 between storms. The old
docstring's remark that snow comes out at six times rain's extinction,
offered as a check on both, held only because the rain figure was inflated;
the ratio is about thirty.

## Verifying

`tests/unit/test_rain.py` pins all three: that `mu = 0` reproduces the
exponential exactly and without a step in the calibration, that `mu = 3`
carries meaningfully more large drops, and that the projected area — the
quantity that actually decides visibility — varies by under 10% across
`mu = 0..4`, which is the claim this document rests on.
