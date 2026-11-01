# Native kernels: an assessment, and what was built

Two questions were asked of this library, with
[Pillow-AllSIMD](https://github.com/desmoteo/pillow-allsimd) as the model:

1. Would its strategies pay here, including translating code to C or C++
   written for the compiler to optimise?
2. Would explicit SIMD kernels for x86 and arm64 NEON pay?

This page answers both from measurements and describes what the
`native-kernels` branch does about them. It follows on from
[`optimizations.md`](../optimizations.md), which assessed the same code
before any of it was changed.

## The answers

**Its method, yes, all of it. Its kernels, no.** Pillow-AllSIMD's design
notes prescribe an order: find where the time goes, fix the algorithm, help
the compiler, and only then write SIMD by hand. Followed here, the first
three steps made fog, rain and snow 6 to 13 times faster on one core at
1024 pixels, and the 22 transforms 3.8 times faster as a geometric mean
(2.8 at 512 pixels), and left nothing for the fourth to do.

**C, yes, in a few places.** Colour conversion, snowflake sprites, rain
streaks, the wet road's reflection, the denoise filter, and the compositing
of lens dirt, of rain and snow over their veil and of overcast light are
now plain C, compiled per instruction-set tier and loaded
through `ctypes`, with the numpy code kept as the definition and the
fallback. The loops they replace were slow because of the interpreter, not
the arithmetic: one Python call per snowflake, a dozen numpy and OpenCV
calls per rain streak, five passes over the frame to turn linear light into
code values. A wholesale translation would gain little more, because most of
what is left is inside OpenCV filters.

**Fast and exact, both.** As in Pillow-AllSIMD, the forms that are faster
but not quite the same operation are a mode, on by default, and
`WEATHERMENTATIONS_FAST=0` or `weathermentations.set_fast(False)` turns them
off. Exact mode reproduces the parent branch byte for byte on every frame
compared. See [Fast and exact mode](#fast-and-exact-mode).

**Explicit SIMD, no, not now.** Three loops of the kernels were written a
second time with AVX2 and NEON intrinsics and timed against the plain C. On
x86-64 the intrinsics were between 0.74 and 1.14 times the compiler's speed.
The best of those is worth about 1% of one transform. The arm64 half of the
question is still open: the NEON code is verified correct under emulation,
but nothing here could time it.

## What it bought

Milliseconds per call on one pinned performance core with OpenCV held to
one thread, which is what a `DataLoader` worker has. "Original" is the
branch this one started from, plain numpy and OpenCV; "now" is this branch
as built, in fast mode with the kernels. Each figure is the mean over four
of the example scenes and two seeds, depth already estimated, and the best
of three interleaved rounds. The rows marked "with PyAV" are the same
branch with the optional `codec` extra installed, which changes only the
two codec transforms.

| Transform | 512: original | now | | 1024: original | now | |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `RandomRain`, dry road | 158.1 | 23.3 | 6.8x | 1035.2 | 81.8 | 12.6x |
| `RandomRain`, wet road drawn | 270.5 | 68.1 | 4.0x | 1692.1 | 181.7 | 9.3x |
| `DirtSoil` | 48.8 | 7.0 | 7.0x | 254.0 | 28.9 | 8.8x |
| `AcFlicker` | 6.7 | 1.0 | 6.5x | 31.8 | 3.8 | 8.3x |
| `NightFog` | 97.0 | 26.7 | 3.6x | 832.3 | 112.6 | 7.4x |
| `TemporalDenoiseArtifacts` | 66.5 | 14.8 | 4.5x | 333.6 | 46.8 | 7.1x |
| `RandomSnow` | 179.2 | 51.5 | 3.5x | 1298.6 | 188.4 | 6.9x |
| `NightSnow` | 45.2 | 6.2 | 7.3x | 254.0 | 37.2 | 6.8x |
| `HousingOccluders` | 14.6 | 6.1 | 2.4x | 104.2 | 16.1 | 6.5x |
| `SunFlare` | 27.8 | 6.7 | 4.2x | 130.5 | 21.4 | 6.1x |
| `RandomFog` | 88.1 | 22.9 | 3.8x | 800.1 | 132.8 | 6.0x |
| `SourceGlare` | 48.7 | 11.6 | 4.2x | 221.5 | 39.0 | 5.7x |
| `NightRain` | 116.8 | 25.0 | 4.7x | 341.7 | 60.5 | 5.6x |
| `ExposureTransient` | 7.6 | 2.1 | 3.6x | 35.0 | 8.6 | 4.1x |
| the depth estimate, once per new frame | 31.1 | 12.0 | 2.6x | 36.0 | 15.8 | 2.3x |
| `SensorNoise` | 13.4 | 6.0 | 2.2x | 65.2 | 31.1 | 2.1x |
| `CastShadows` | 26.7 | 21.0 | 1.3x | 138.6 | 76.0 | 1.8x |
| `NightIrMode` | 19.0 | 12.3 | 1.5x | 83.8 | 51.3 | 1.6x |
| `WdrToneMapping` | 121.8 | 116.5 | 1.0x | 164.4 | 127.7 | 1.3x |
| `WaterDroplets` | 52.4 | 45.8 | 1.1x | 140.2 | 125.4 | 1.1x |
| `RollingShutter` | 2.0 | 1.9 | 1.0x | 7.7 | 7.7 | 1.0x |
| `VideoCodecArtifacts` | 143.6 | 145.8 | 1.0x | 182.3 | 186.7 | 1.0x |
| `VideoCodecArtifacts`, with PyAV | 143.6 | 9.2 | 15.7x | 182.3 | 29.4 | 6.2x |
| `TransmissionFaults` | 171.9 | 172.3 | 1.0x | 281.6 | 293.4 | 1.0x |
| `TransmissionFaults`, with PyAV | 171.9 | 23.1 | 7.4x | 281.6 | 79.1 | 3.6x |
| **Geometric mean of the 22 transforms** | | | **2.80x** | | | **3.80x** |
| **All 22 once each** | 1727 | 795 | **2.17x** | 8428 | 1859 | **4.53x** |
| **Geometric mean, with PyAV** | | | **3.48x** | | | **4.38x** |
| **All once each, with PyAV** | 1727 | 509 | **3.39x** | 8428 | 1487 | **5.67x** |

- **The geometric mean** is over the 22 transforms, each counted once
  whatever it costs: 2.8 times at 512 pixels and 3.8 at 1024. Three of them
  did not change. Over the 19 that changed by more than a tenth it is 3.3
  and 4.7. With PyAV the two codec transforms change as well, and it is 3.5
  and 4.4 over all 22.
- **"All 22 once each"** adds the times up instead, so the heavy transforms
  weigh more: 2.2 and 4.5 times, 3.4 and 5.7 with PyAV. Neither figure is a
  pipeline's: that depends on which transforms it holds and how often each
  fires.
- **`RandomRain` is two rows** because its default changed. The original
  drew the wet road in 70% of frames and this branch draws it in 40%, which
  is a choice about pictures and not a speed-up; the rows show the same
  work in both. Weighted by each branch's own default, a rain frame at
  1024 pixels went from about 1490 ms to about 120.
- **The depth estimate** is paid once for each new frame, by the first
  depth-conditioned transform that fires, and is not in the means.
- **Part of the first four rows is a regression being undone.** The sky-edge
  redraw that the round before this branch shared between fog, rain, snow
  and night fog ran two Gaussian blurs with a sigma of 58 pixels and a
  59-pixel disc dilation at full resolution. With OpenCV's sixteen threads
  that was about a fifth of a second at 1024 pixels. On one thread it was
  most of a second. The README's timing table was measured before that
  change and with all threads, so it never showed it.
- **The machine was not idle.** Other jobs were running on it throughout and
  the hardware counters were not readable (`perf_event_paranoid` is 4). The
  same pinned loop measured 87 ms in one hour and 144 ms in another. Ratios
  within a row hold; absolute figures are good to perhaps 40%, and differ
  between the tables on this page, which were made on different days.
- **The README's table is all-threads and stale.** It has not been
  re-measured on this branch.

### After the first round

How much of that came from numpy alone and how much from the compiled
kernels, measured after the first round of work on the benchmark's
synthetic scene: "before" is the original, "numpy only" the branch without
the compiled library, "kernels" with it. Best of six interleaved rounds.

| Transform | 512: before | numpy only | kernels | 1024: before | numpy only | kernels | 1024 speed-up |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `RandomFog` (depth cached) | 62.5 | 17.3 | 15.3 | 605.0 | 100.3 | 86.3 | 7.0x |
| `NightFog` (depth cached) | 66.7 | 19.4 | 16.8 | 625.1 | 95.6 | 83.3 | 7.5x |
| `RandomSnow` (depth cached) | 85.7 | 40.7 | 31.9 | 734.6 | 213.8 | 154.5 | 4.8x |
| `RandomRain` (depth cached) | 128.8 | 85.5 | 63.3 | 1006.9 | 495.1 | 338.3 | 3.0x |
| `NightRain` | 87.3 | 86.1 | 41.0 | 281.2 | 270.0 | 74.5 | 3.8x |
| `NightSnow` | 12.7 | 11.6 | 5.9 | 85.0 | 77.2 | 30.1 | 2.8x |
| `AcFlicker` | 3.8 | 1.7 | 0.6 | 17.9 | 13.2 | 2.7 | 6.7x |
| `ExposureTransient` | 4.4 | 2.4 | 1.2 | 20.6 | 15.3 | 6.0 | 3.4x |
| `SourceGlare` | 20.8 | 11.4 | 9.1 | 102.2 | 64.0 | 42.8 | 2.4x |
| `NightIrMode` (depth cached) | 10.8 | 8.9 | 7.6 | 63.9 | 53.8 | 38.4 | 1.7x |
| `SensorNoise` | 8.0 | 6.6 | 4.6 | 51.0 | 42.2 | 32.2 | 1.6x |
| `CastShadows` (depth cached) | 13.9 | 12.2 | 11.2 | 102.6 | 89.5 | 73.1 | 1.4x |
| `WaterDroplets` | 34.6 | 33.8 | 29.2 | 99.8 | 89.2 | 77.1 | 1.3x |
| `SunFlare` | 26.7 | 23.7 | 21.6 | 137.6 | 125.5 | 107.0 | 1.3x |
| `WdrToneMapping` | 27.5 | 26.4 | 23.9 | 112.2 | 103.4 | 84.4 | 1.3x |
| `DirtSoil` | 33.5 | 32.0 | 29.2 | 169.5 | 150.2 | 144.8 | 1.2x |
| `HousingOccluders` | 5.2 | 5.2 | 5.2 | 52.3 | 52.5 | 52.3 | 1.0x |
| `TemporalDenoiseArtifacts` | 76.5 | 76.1 | 76.7 | 438.9 | 438.6 | 447.4 | 1.0x |
| `RollingShutter` | 1.1 | 1.1 | 1.1 | 4.4 | 4.5 | 4.4 | 1.0x |
| `VideoCodecArtifacts` | 86.7 | 86.4 | 86.2 | 106.1 | 105.9 | 105.6 | 1.0x |
| `TransmissionFaults` | 95.3 | 95.8 | 95.6 | 133.0 | 132.7 | 133.6 | 1.0x |

The table is fast mode, measured before the mode existed, when its sky-edge
filters were the only ones. 

## Where the time was

Profiled per transform at 1024 pixels, the time fell into four kinds, and
only one of them is the kind Pillow-AllSIMD's kernels address.

| Kind | Example | What fixes it |
| --- | --- | --- |
| An OpenCV filter asked the wrong question | a blur with sigma 58 px on the full frame: 366 ms on one thread | a smaller copy of the frame |
| The interpreter, once per particle | 16,000 streaks at a dozen numpy and OpenCV calls each: 140 ms of Python around 18 ms of drawing | one call for the field, in C |
| numpy passes over the frame | `image.max(axis=2)`: 25 ms, against 1 ms for two `np.maximum` calls | the right numpy call, or one fused pass in C |
| An OpenCV filter that is the effect | `edgePreservingFilter` in `TemporalDenoiseArtifacts`: 75% of 439 ms | nothing here; it is already SIMD |

Pillow's kernels were hot loops of pixel arithmetic bound by the core. This
library has few of those of its own: its filters are OpenCV's, which are
already vectorised, and its own time went to how they were called and to
Python.

## What was done, in the method's order

### 1. Algorithm

- **Wide blurs and dilations on a reduced copy**
  ([`utils/sky.py`](../src/weathermentations/utils/sky.py), `wide_blur` and
  `wide_dilate`). The sky-edge redraw went from 875 ms to 38 ms on one
  thread, timed back to back. It alters pictures along the skyline, so it
  is fast mode only.
- **Channel reductions as two ufunc calls**
  ([`utils/channels.py`](../src/weathermentations/utils/channels.py)). Exact.
- **The Bayer mosaic by strided copies.** `SensorNoise` looked each pixel's
  colour up in a frame-sized map of site labels, four times a frame. Copying
  along the mosaic's own stride gives the same arrays in 2.5 ms where the
  lookups took 8.1 ms at 1024 pixels. Exact.

### 2. numpy

- **A table of code values for the encode.** `from_linear_unit` was clip,
  scale, gather a float, scale and round, clip and cast. The last three
  depend only on the table entry, so they are folded into a second,
  `uint8` table: 18.1 ms to 5.9 ms at 1024 pixels, the same bytes.

### 3. C, written for the compiler

| Kernel | Replaces | numpy | C (AVX2 tier) | Agreement with numpy |
| --- | --- | ---: | ---: | --- |
| `encode_u8` | linear light to sRGB bytes, 3 M values | 5.9 to 7.3 ms | 0.8 to 1.4 ms | exact |
| `decode_u8` | sRGB bytes to linear light | 2.9 to 3.3 ms | 0.5 to 0.8 ms | exact |
| `render_flakes`, round aperture | 3,000 sprites, 10 M pixels | 144 ms | 17 ms | exact below a 17 px radius, else 6e-9 |
| `render_flakes`, six-blade iris | the same | 382 ms | 22 ms | 1.5e-6 |
| `render_streaks`, one pixel wide | 15,000 streaks | 211 ms | 11 ms | exact |
| `render_streaks`, wide and blurred | 15,000 streaks | 251 ms | 24 ms | exact unblurred, 6e-8 blurred |
| `wet_composite` (fast mode) | the wet road's compositing, 600 rows | 24 ms | 1.8 ms | exact |
| `mirror_rows` | what each wet pixel mirrors | 41 ms | 9.5 ms with its two OpenCV filters | exact |
| `wet_reflection` (fast mode) | the wet road's mirror image | two warps and three blends | one pass | exact against the numpy it shares its arithmetic with |

Getting those figures took three corrections to how the C was written or
built, none of them intrinsics. They are the substance of the answer to the
second question, so they are spelled out.

- **The first C encode was slower than numpy**: 9.4 ms against 5.9. GCC had
  compiled the clamp as two branches, which a photograph's highlights and
  shadows mispredict, and vectorised nothing. Splitting the loop into index
  arithmetic and table lookups, and passing `-fno-trapping-math` so the
  float-to-int conversion could be computed for every lane, made it 1.0 ms.
- **One loop was vectorised in one function and not in another.** The loop
  that shades a sprite row is three clamps in sequence. Inlined into its
  caller it was left scalar; out of line it was vectorised; out of line but
  called with constants it was left scalar again, three times slower. The
  cause was jump threading turning the clamps into branches before the
  vectoriser ran. `-fno-thread-jumps` makes it stable in all three places.
- **Two of its three clamps could never bind.** Removing them gave the same
  floats a fifth faster.

Because of the second, [`tests/unit/test_native_build.py`](../tests/unit/test_native_build.py)
compiles the kernels for every tier and fails if any loop marked
`WM_VECTORISED` was not vectorised. It is this branch's version of
Pillow-AllSIMD's instruction-count gate: a timing on a CI runner is noise,
and the compiler's own remark is not.

### 4. Explicit SIMD: measured, not adopted

[`native/experiments/simd_explicit.c`](../native/experiments/simd_explicit.c)
holds three loops in plain C and again in intrinsics that do the same
operations in the same order. It checks the two give identical bits, then
times them. On the development machine (Core Ultra 9 285H performance core,
GCC 13.3):

| Loop | plain C, SSE2 tier | plain C, AVX2 tier | AVX2 intrinsics | intrinsics / plain C |
| --- | ---: | ---: | ---: | ---: |
| encode (ns per value) | 0.56 | 0.52 | 0.49 | 1.07x |
| capsule distance (ns per pixel) | 0.44 | 0.22 | 0.30 | 0.74x |
| sprite shading (ns per pixel) | 0.85 | 0.49 | 0.43 | 1.13x |

- **Encode** is bound by its table lookups. An AVX2 gather does not beat
  eight scalar loads.
- **Capsule**: the compiler's code is faster than a straightforward
  transcription into intrinsics. A tuned version would at best tie.
- **Shading** is the one win, 13%. That loop is about a third of the flake
  kernel, which is about a tenth of `RandomSnow`.

The columns that do differ are the first two. The AVX2 tier is up to twice
the SSE2 tier on the same source, so compiling per tier and choosing at load
is worth its complexity, and that part of Pillow-AllSIMD's design is kept.

**On arm64 this is unanswered.** The NEON versions give the same bits as the
plain C in an arm64 container under QEMU, where timings mean nothing. On the
Jetson, one command answers it:

```bash
taskset -c <a big core> sh native/experiments/run.sh
```

If the NEON intrinsics win there by a margin that matters, the experiment's
functions can be moved into `wm_kernels.c` under `#if defined(__ARM_NEON)`
as they stand; they are already checked bit for bit.

**When this would change.** Explicit kernels become worth writing when a
loop is hot, bound by the core, and the compiler cannot be brought to
vectorise it: Pillow-AllSIMD's own rule. Nothing left in this library meets
all three. What remains is listed under [Not done](#not-done).

## What carried over from Pillow-AllSIMD, and what did not

| Its practice | Here |
| --- | --- |
| Time on one pinned core, old and new interleaved, outputs hashed in the same run | Done; it is how the single-thread regression was found |
| Fix the algorithm and the data path before the instructions | The largest gain on the branch |
| Read the compiler's remarks before replacing it | Three fixes, above |
| One translation unit per tier, a table of kernels chosen at load | `wm_kernels.c` compiled per tier, `wm_dispatch.c` |
| Plain loops compiled per tier, not only the hand-written ones | All of the kernels are that |
| Every tier gives the same pixels, checked on every build | Tests, and hashes that match across x86-64 and arm64 |
| A kernel takes a whole row or image, never a fragment | A whole frame or a whole field of particles per call |
| A deterministic gate in CI for silent slow-downs | The vectorisation test |
| Check the tails for over-reads | Address and undefined-behaviour sanitizer run |
| Hand-written intrinsics per architecture | Not adopted: measured at 0.74x to 1.13x |
| Fast mode on by default, exact mode on request, the bounds tested | Adopted: `weathermentations.mode` |
| Integer fixed-point tricks, AVX-VNNI | Not applicable: this code is float32 |

## How it is built and loaded

- **C through `ctypes`, not the C API, C++, Cython or numba.** The library
  takes raw pointers and includes no Python or numpy header, so one binary
  serves every Python version on a platform and the wheel is tagged
  `py3-none-<platform>`. `ctypes` releases the GIL for the call. The cost
  is a few microseconds per call, which is nothing for kernels called once
  per frame. Nothing here needs C++. numba would add a large runtime
  dependency and a compilation in every `DataLoader` worker.
- **Tiers.** x86-64: `sse2` (the baseline) and `avx2`, chosen by
  `__builtin_cpu_supports` when the library loads. arm64: `neon`, the
  baseline. Elsewhere: `portable`. Only a tier's own object is compiled
  with more than the baseline, so the binary runs on any CPU of its
  architecture.
- **Flags.** `-O3 -ffp-contract=off -fno-math-errno -fno-trapping-math
  -fno-thread-jumps -funroll-loops`. No fused multiply-add and no
  `-ffast-math`, so every tier and compiler rounds as numpy does. What
  other flags were measured is [below](#compiler-flags-and-the-types-left-in-the-c-measured).
- **Optional.** [`hatch_build.py`](../hatch_build.py) compiles it when a
  wheel or an editable install is made, and goes on without it if there is
  no C compiler. `python hatch_build.py` builds it in place in a checkout.
  Without it every transform runs its numpy code and gives the same picture.

| Variable | Effect |
| --- | --- |
| `WEATHERMENTATIONS_NATIVE=0` | run the numpy code even when the library is there |
| `WEATHERMENTATIONS_NATIVE=require` | a missing library is an `ImportError` |
| `WEATHERMENTATIONS_ISA=sse2` | use that tier, to compare tiers on one machine |
| `WEATHERMENTATIONS_FAST=0` | exact mode |
| `WEATHERMENTATIONS_CODEC=ffmpeg` | encode with the ffmpeg binary even when PyAV is installed |
| `WEATHERMENTATIONS_BUILD_NATIVE=0` / `require` | skip the build, or fail if it fails |
| `WEATHERMENTATIONS_BUILD_MARCH=native` | at build time, add a tier compiled for this machine only |

`weathermentations._native` has the same switches at run time
(`set_enabled`, `disabled()`, `set_tier`, `tier()`, `tiers()`).

## What was verified

- **Against numpy**, per kernel, in
  [`tests/unit/test_native_kernels.py`](../tests/unit/test_native_kernels.py):
  bit for bit where the table above says exact, to the stated bound where a
  blur's sums are taken in another order. The streak kernel is a port of
  OpenCV's line, polygon-fill and circle rasterisers and is compared with
  `cv2.polylines` over 36,000 streaks at every heading and width.
- **On real scenes**: every transform on four of the example scenes with two
  seeds, 152 frames. With the kernels on and off all 152 are byte-identical
  in exact mode, and exact mode without the kernels is byte-identical to
  the parent branch. In fast mode 149 are, and three differ by one level:
  the denoise filter, whose tabled form needs the kernels, in 3 values of
  the 9 million in its eight frames, and one snow frame in one value of
  1.2 million. The SSE2 and AVX2 tiers are identical on all 152.
- **Blurred sprites and streaks are the kernels' one tolerance, and it can
  reach a frame.** They agree with numpy to the last bit of a float and
  not always in it, because a blur's sums are taken in another order. Now
  and then that bit decides a rounding to a code value: the snow frame
  above, and in exact mode one value in one of 24 further snow frames tried
  on other seeds. Everything else a kernel computes is numpy's to the bit.
- **Across machines**: [`native/check/run.sh`](../native/check/run.sh) prints
  a hash per kernel, and one more each for the fast-mode sprites and the
  tabled filter. The SSE2 and AVX2 builds (glibc, GCC 13.3) and an arm64
  build under QEMU (musl, GCC 13.2) print the same twelve, the denoise
  filter's among them, which also holds one C library's `powf` to the
  other's.
- **Against OpenCV and against ffmpeg**, where a kernel or PyAV stands in
  for one of them: tried once on each installation and relied on in exact
  mode only where it gave the same bytes
  ([the fifth round](#a-fifth-round-the-remaining-heavy-effects)).
- **Memory**: the kernel tests pass with the library built with
  `-fsanitize=address,undefined`.
- **Refused input**: a NaN in a frame still raises, as the table lookup
  always has, instead of becoming a black pixel in C.
- **The full suite** passes in fast mode with the kernels, in exact mode
  with them, in exact mode without them, and in fast mode with PyAV
  installed.

## Fast and exact mode

Some operations have a faster form that is not quite the same operation.
Those forms are **fast mode**, which is the default. **Exact mode** computes
what the reference numpy and OpenCV code computes.

```python
import weathermentations

weathermentations.set_fast(False)      # exact from here on
with weathermentations.exact():        # or for one block
    frame = transform(image=image)["image"]
```

`WEATHERMENTATIONS_FAST=0` selects exact mode at import. Use that with a
`DataLoader` whose workers are spawned, since a spawned worker does not see a
`set_fast` call made in its parent. The mode is independent of the compiled
kernels: either mode runs with or without them, and gives the same frame
with them as without, with one exception in fast mode, the denoise filter
in the table below.

What fast mode changes, and by how much, on 152 frames (every transform on
four example scenes, two seeds):

| Fast form | Where | Speed | Difference from exact |
| --- | --- | --- | --- |
| Blur and dilation on a reduced copy of the frame | the sky's edge, in `RandomFog`, `RandomRain`, `RandomSnow`, `NightFog` | 5 to 7 times the whole transform at 1024 px | 0.8 to 1.4% of pixels, 0.01 to 0.02 levels on average, 22 at most, along the skyline |
| A weighted mean colour as a matrix-vector product | the same four | 12 ms to under 1 ms at 1024 px | float32 rounding of a sum |
| float32 where the reference is float64 by accident | aperture and scratch geometry in `SunFlare`, `SourceGlare`; polygonal flakes without the kernels | `SunFlare` 1.16 times | one level, in 0.02% of values |
| Multiplying by a reciprocal for a division | polygonal sprites in the flake kernel | the kernel 1.2 times (AVX2) to 1.4 times (SSE2) | 2e-6 of a sprite's peak |
| Blur on a reduced copy, once its sigma reaches 8 or 16 px | defocus in `HousingOccluders` and `DirtSoil`, chroma in `TemporalDenoiseArtifacts`, settled snow and ground filling in `RandomSnow` and `RandomRain` | 1.2 to 4 times at 1024 px, less at 512 | 2 levels at most in the first three |
| The depth network in half precision | every depth-conditioned transform | the estimate 2.2 to 2.5 times | see below |
| Surface normals read at the depth map's 518 px | `RandomRain` with a wet road, `RandomSnow`, `CastShadows` | 70 ms to 15 ms at 1024 px | see below |
| The wet road's reflection blurred on a copy reduced down the frame only | `RandomRain` | 95 ms to 11 ms at 1024 px | within the figures below |
| A cube as two multiplications, not a power | the wet road's darkening | 20 ms at 1024 px | a rounding |
| Folding the ground-noise tile with a floor, not a remainder | settled snow, puddles | 26 ms to 3 ms per field at 1024 px | millionths of a pixel |
| The denoise filter's power function from a table of its 766 values | `TemporalDenoiseArtifacts`, with the kernels | 48 ms a pass to 19 at 1024 px | one level, in a few values in a million a pass; two at most over a frame |
| The deposit field built at a quarter of the size | `DirtSoil` | 107 ms to 13 at 1024 px | 0.15 levels on average, 10 at most: the same deposits, placed to within two pixels |
| Broad light on a coarse grid, streaks and glows evaluated only where they reach, an affine warp for the starburst | `SunFlare` | 115 ms to 31 at 1024 px | one level at most at 1024 px, two below |
| The lamp's beam as a product of two profiles | `NightRain`, `NightSnow` | 5 ms to 0.6 at 1024 px | one level, in two values in a million |
| float32 where two scalars had widened the settled-snow fields | `RandomSnow` | 6% of a frame with settled snow | one level, in 0.002% of values |
| Overcast light's contrast gain from a table of 65536 values, and the relighting in one pass | `RandomRain`, `RandomSnow` | 46 ms to 13 at 1024 px | within a level; one level in 0.1% of values on the scenes |
| The brightest pixels' mean from a histogram; a frame's mean colour summed by OpenCV | `RandomRain`, `RandomSnow`, `RandomFog` | 37 ms to 12 at 1024 px | the fifth digit of a colour |
| H.265 encoded in the process, and damaged streams decoded there | `VideoCodecArtifacts(codec="h265")`, `TransmissionFaults`, with PyAV | 2 to 9 times at 512 px and below | another release of libx265; an unconcealed loss is always the zeroed hole |

Six transforms give the same bytes in both modes on these frames:
`WaterDroplets`, `AcFlicker`, `RollingShutter`, `SensorNoise`,
`ExposureTransient` and `WdrToneMapping`. `VideoCodecArtifacts` with H.264
does as well, with or without PyAV. `NightRain` and `NightSnow` are one
level away in two values in a million.

**What is read off depth moves at its edges.** Half precision changes the
network's output by 0.01% of its range on average, and reading the normals
at 518 pixels moves what counts as level ground on 2 to 3% of pixels, all of
them where the ground meets something standing on it. Everything drawn from
those moves with them. Against exact mode, on the same 152 frames:

| | Pixels that differ | Mean difference (levels) | Largest |
| --- | ---: | ---: | ---: |
| `RandomRain`, wet road drawn | 30% | 0.74 | 42 |
| `RandomSnow` | 14% | 0.45 | 94 |
| `CastShadows` | 8% | 0.15 | 62 |
| `NightIrMode` | 3% | 0.03 | 2 |
| `RandomFog`, `NightFog`, `RandomRain` with a dry road | 1 to 2% | 0.02 | 9 |

These are the largest differences fast mode makes. Side by side the worst
frames of each are not distinguishable: the same road is wet, the same
ground is covered, and the outline of each differs by a pixel or two round
a car or a kerb. `DepthAnythingV2Estimator(half_precision=False)` keeps the
network at full precision in fast mode; the normals follow the mode.

**Exact mode is the parent branch.** With the kernels off, all 152 frames
are byte-identical to the frames the parent branch makes, and with the
kernels on they are byte-identical again. That was checked after each round
of changes, the fifth round's included. The two codec transforms are not
among the 152; with PyAV installed their exact-mode frames were compared
with the binary's separately, and were identical on every seed and size
tried.

**The skyline is the difference that can matter.** The reduced dilation
decides how far the sky mask may grow, and it is largest in the night street
scene. Side by side the frames are not distinguishable, and the golden images
pass in either mode, but the skyline is where the previous round's artefacts
were. If one comes back, exact mode is the first thing to try.

Each bound above is asserted in
[`tests/unit/test_modes.py`](../tests/unit/test_modes.py), except the
skyline figures, which need the depth model and were measured as described.

## A second round: the heavy effects

After the kernels, the transforms were profiled again on one thread, on real
scenes and not only the benchmark's synthetic one, which never draws settled
snow. Milliseconds per call at 1024 pixels, the mean of four example scenes
and two seeds, depth already estimated, before this round and after:

| | 512: before | after | | 1024: before | after | |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `HousingOccluders` | 9.1 | 6.1 | 1.5x | 71.5 | 17.5 | 4.1x |
| the depth estimate itself | 31.7 | 12.5 | 2.5x | 36.4 | 16.7 | 2.2x |
| `NightRain` | 68.2 | 35.6 | 1.9x | 109.5 | 70.8 | 1.5x |
| `DirtSoil` | 43.6 | 43.1 | 1.0x | 228.7 | 165.1 | 1.4x |
| `RandomSnow` | 82.3 | 72.9 | 1.1x | 425.5 | 318.4 | 1.3x |
| `TemporalDenoiseArtifacts` | 66.8 | 63.0 | 1.1x | 330.6 | 267.4 | 1.2x |
| `RandomRain` | 49.1 | 42.2 | 1.2x | 179.3 | 171.5 | 1.05x |

The other twelve measured are within 5% either way, which is the noise of
the run.

What was done:

- **Reduced-size blurs** in the five places the fast-mode table lists. The
  blur in `HousingOccluders` was most of that transform. Settled snow, added
  in an earlier round, had six blurs of 10 to 29 pixels at full size, five
  of which are now reduced.
- **Half-precision depth**, 28 ms to 11 ms for the network at 518 pixels.
- **The depth model's preprocessing without its processor.** The Hugging Face
  processor spent 3 to 5 ms a frame on a resize and a normalisation. The
  estimator now makes the same calls itself in 1 to 2 ms, after checking
  when it is built that it reproduces the processor's tensor bit for bit;
  with a version of `transformers` where it does not, the processor is used
  as before. Exact, in both modes.
- **Particle sampling that skips hopeless candidates.** `NightRain` drew
  2,000,000 candidate drops to keep 16,000, and computed a diameter and a
  distance for every one. Nearly all can be ruled out from their two
  uniforms by a table lookup. The same uniforms are drawn in the same order
  and the same particles come back: 54 ms to 18 ms. Exact.
- **Diagnostics that cost nothing when nobody is recording.** The median of
  the depth map, and eight other statistics of a whole frame, were computed
  on every call to be handed to a recorder that was not there. They are now
  passed as functions. Exact.

What was tried and dropped:

- **Resizing for the depth model on the GPU.** A millisecond faster than the
  exact preprocessing, and it changed the far-field depth of the night
  scene by several percent, because the mapping from disparity to depth is
  steep there.
- **Percentiles from a thinned grid of pixels.** 1 to 5 ms a frame, and it
  moved `NightIrMode`, whose exposure hangs on one percentile, by 3 levels.
- **The edge-preserving filter at half size**, which is three quarters of
  `TemporalDenoiseArtifacts`: 169 ms to 37 ms, and a different picture. It
  differs from the full-size filter by 1.7 levels on average where the
  filter itself changes the frame by 3.0.

## A third round: the wet road

`RandomRain` looked slow in the tables above, and the streaks were not why:
the kernel draws them in 20 ms. It is the wet road, which the transform drew
in 70% of frames and which cost three times a frame without it.
Milliseconds per call on real scenes, one thread, depth already estimated:

| | 512: before | after | | 1024: before | after | |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `RandomRain`, wet road drawn | 136.1 | 107.3 | 1.3x | 642.5 | 407.0 | 1.6x |
| `RandomRain`, dry road | 45.0 | 45.1 | 1.0x | 172.0 | 172.8 | 1.0x |
| `CastShadows` | 21.0 | 21.6 | 1.0x | 120.5 | 82.0 | 1.5x |
| `RandomSnow` | 95.2 | 99.8 | 1.0x | 368.9 | 340.2 | 1.1x |

What was done:

- **Only the rows that hold wet ground are worked on.** A road is the lower
  part of a frame, and the wet-road pass was arithmetic over all of it. The
  reflection is blurred down the frame, so the rows worked on include a
  margin as deep as the blur's kernel, and the result is the same bytes.
  Exact, in both modes.
- **The reflection's blur on a reduced copy**, 11 pixels across and 36 down
  at 1024 pixels: reduced four times in height and not in width. The blur
  was 136 ms over the whole frame, is 95 ms over the wet rows, and 11 ms
  reduced.
- **Surface normals at the depth map's resolution.** They were differences
  between neighbouring pixels of a map interpolated up from 518.
- **A kernel for the compositing** (`wet_composite`): the darkened surface,
  the mirror image over it and the blend with the dry frame, nine numpy
  expressions in one pass, bit for bit what they give in fast mode.
- **A wet road less often.** `wetness_probability` defaults to 0.4, from 0.7,
  and the pipelines' rain branch asks for 0.5, from 0.85. The mean cost of
  a default `RandomRain` frame at 1024 pixels goes from about 500 ms to
  about 270: 0.7 of 642 and 0.3 of 172 before, 0.4 of 407 and 0.6 of 173
  now. The example images of the pipelines were made before this.

### The wet road, simplified

That left a wet frame at 400 ms, 231 ms of it the wet road, in some twenty
steps of 5 to 30 ms. It was revised once more, into three stages:

1. **Read the ground once, at the depth map's size.** Where the road is, how
   far it reaches up to each object, where it is paved and where puddles
   lie were five passes over the full frame, all of them reading a depth map
   interpolated up from 518 pixels. In fast mode they run on a copy that
   size and the map of wet ground is enlarged: 125 ms to 36.
2. **Build the mirror image in one pass.** What each pixel mirrors was found
   with six frame-sized arrays, and the image itself with two warps of the
   frame and three blends. A mirror about a horizontal line moves whole
   rows, so the frame read at a fractional row is two rows weighted. One
   sweep down the frame finds what each pixel mirrors (`mirror_rows`, 41 ms
   to 9.5, exact) and one pass makes the image (`wet_reflection`).
3. **Composite**, with the kernel already there.

The blur of the puddles' reflection is also skipped when no water stands on
the road, which changes nothing. Against the commit before:

| | 512: before | after | | 1024: before | after | |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `RandomRain`, wet road drawn | 105.4 | 95.0 | 1.1x | 389.8 | 272.0 | 1.4x |
| `RandomRain`, dry road | 44.3 | 44.1 | 1.0x | 167.4 | 170.6 | 1.0x |

The wet road itself went from 231 ms to 97 ms at 1024 pixels, and the pass
that renders it from 103 ms to 49. Since the start of this round a wet frame
has gone from 642 ms to 272.

Two things to know about the mirror image:

- **In fast mode its arithmetic is the library's own**, the same in numpy and
  in C, so a frame does not depend on the kernels. Exact mode keeps OpenCV's
  warp. The two could not be made one: OpenCV rounds the interpolation
  weight to a 32nd of a pixel in version 4 and does not in version 5, so
  "what OpenCV gives" is not one thing to reproduce.
- **Fast mode now differs from exact by more on a wet road** than on anything
  else: 30% of pixels, 0.7 levels on average, 42 at most, where the ground
  read at 518 pixels meets what stands on it. The worst frames show the same
  road with the same reflections; the difference is under and beside the
  vehicles.

**Is there more for C here?** Not much. What is left of a wet frame's 100 ms
is two small blurs, a median filter, the ground reading at 518 pixels and
the arithmetic around the kernels, none above 13 ms.

## A fifth round: the remaining heavy effects

Five transforms were still slow after the wet road: the three camera and
lens effects with no hot loop of their own, and the two that start ffmpeg.
All five were first assessed and then built. One pinned core, 1024 pixels
unless said. The figures in this section are of one scene over a dozen
seeds or more, so they are not the table's at the top, which is four scenes
and two seeds.

| Transform | Where the time was | What was done | Result |
| --- | --- | --- | --- |
| `TemporalDenoiseArtifacts` | OpenCV's `edgePreservingFilter`, 157 ms a pass, one to three passes | the same filter as one C function; in fast mode its power function from a table | 48 ms a pass with OpenCV's bytes, 19 ms tabled |
| `DirtSoil` | building the deposit field at full size, 107 ms; compositing it, 49 ms | the field at a quarter of the size in fast mode; the compositing as two C passes | 156 ms to 25 |
| `SunFlare` | one float64 expression, 21 ms; scratch streaks tested over the whole frame, 14 ms; the starburst, 10 ms; the final conversion, 10 ms | each rewritten for fast mode, in numpy | 115 ms to 31 |
| `VideoCodecArtifacts` | starting two ffmpeg processes | PyAV, where installed, encodes in the process | H.264: 217 ms to 12 at 512 px, the same bytes |
| `TransmissionFaults` | the same two processes | the same | 168 ms to 20 at 512 px in fast mode |

### The denoise filter

OpenCV's `edgePreservingFilter` with `RECURS_FILTER` is Gastal and
Oliveira's domain transform: three passes, each smoothing every row there
and back and then every column, with a weight per pixel that is a power of
how unlike the pixel is from its neighbour. OpenCV takes that power for
every pixel twice in each pass and walks the frame through checked element
accessors.

- **The port is OpenCV's arithmetic in OpenCV's order**, and gives OpenCV's
  bytes: on 40 random frames down to one pixel, on the example scenes, on
  both tiers and on arm64. That rests on the two calling the same `powf`
  and neither fusing a multiply into an add, so the transform tries it once
  on two small frames (`denoise._kernel_matches_opencv`) and uses the
  kernel in exact mode only where it passed. 157 ms to 48.
- **The power only ever takes 766 values**, one for each possible sum of
  three byte differences. Fast mode takes it once for each and looks it up:
  19 ms, and 2 values in 3 million one level away. This is the one place
  where a fast-mode frame depends on whether the kernels are installed,
  by that much: without them fast mode runs OpenCV's filter.
- **Half its time was one recurrence waiting for itself.** Smoothing along
  a row is `p[x] += (p[x-1] - p[x]) * w[x]`: each step needs the one
  before, twelve cycles of subtract, multiply and add, while the core could
  have done several times as much. Four rows are taken together with their
  steps interleaved, the running pixels in local variables: 30 ms to 21.
  Two rows gave nothing and eight no more than four.

### Lens dirt

- **The deposit field is built at a quarter of the size** in fast mode and
  enlarged, as asked. Everything on the glass is out of focus: the field's
  last step is a blur of 0.6% of the diagonal, 8.7 pixels on a 1024-pixel
  frame, so a quarter of the size holds all of it. 107 ms to 13.
  - *The draws are the same ones.* Positions and sizes are drawn as shares
    of the frame and scaled, so a seed puts the same deposits in the same
    places.
  - *Thin shapes are not drawn small.* Wipe marks and water-spot rings are
    a pixel or two wide in places. Drawn at a quarter of the size with
    their widths rounded they came out up to twice too heavy or too light
    (a frame 32 levels off at worst). They are drawn at full size, which is
    cheap, and averaged down, so each pixel of the field takes the share of
    it they cover.
  - *The reduction follows the blur.* It is 4 wherever the field's blur is
    still a pixel wide at that size, a diagonal of 670 pixels or more;
    smaller frames are reduced by less, and one under 240 pixels not at
    all.
  - *Enlarged bilinearly.* Bicubic measured no closer to the full-size
    field and slower.
  - *Against exact mode*: as much grime (coverage within 1%), a frame 0.10
    levels away on average at 1024 pixels and 8 at most, 0.18 and 10 at
    512. What is left is each patch's corner being rounded to a pixel of
    the smaller field, two pixels of the frame.
- **The compositing is two C passes** either side of the scatter blur, where
  numpy made eleven frame-sized results. They are numpy's arithmetic in
  numpy's order, so the frame is the same to the bit with and without them,
  in both modes. 49 ms to 11.

### Sun flare

No kernel: each of these is the numpy code doing less. Milliseconds at 1024
pixels, from a line profile, so a little above what the stages cost
untraced.

| Stage | Was | Now | How |
| --- | ---: | ---: | --- |
| veiling glare, halo, infrared glow | 32 ms | 5 ms | float32 colours; drawn on a grid four times coarser and enlarged |
| scratch streaks | 14 ms | 1.5 ms | evaluated on the few thousand pixels within reach of each line, found from where it crosses each row |
| starburst | 10 ms | 5 ms | an affine warp in place of two frame-sized coordinate maps a channel |
| window ghost | 7 ms | under 1 ms | in the square the glow reaches, not over the frame |
| the final conversion | 13 ms | 8 ms | intensity and scale in one pass, clipped in place |

- **The reference was float64 by accident again**: the flare's colour was a
  float64 array, so every field it multiplied was widened. The round before
  fixed the scalars; this fixes the colour.
- **Against exact mode** the frame is one level away at most at 1024
  pixels, in 0.3% of values, and two at 256 and 512. The coarse grid is
  within a quarter of a level of the full one, and that much only at the
  glare's very peak.

### The codecs in process

PyAV binds FFmpeg's libraries into Python. It is an optional extra
(`pip install "weathermentations[codec]"`), used when present.

- **For H.264 it is the binary's frame to the byte.** The first probe
  differed from the binary by 2.4 to 4.7 levels on average. Three settings
  account for all of it: the colour conversion's interpolation (the binary's
  is bicubic, PyAV's default bilinear), one encoder thread (PyAV's default
  lets libx264 cut the frame into a slice per thread), and the binary's
  frame rate. With those, 75 of 75 round trips were identical, three images
  at five sizes and five qualities, although the two carry different
  releases of libx264 and libswscale.
- **That is checked on each installation, not assumed.** It rests on two
  builds of FFmpeg, neither of them this library's. Each encoder is tried
  once on a small frame against the binary, and exact mode encodes in
  process only where it passed. Fast mode does whenever PyAV is there.
- **H.265 is not the binary's frame**, and is used in fast mode only. The
  two carry libx265 3.5 and 4.3, which make different decisions: 3 to 4
  levels apart on average, where the codec changes the frame by 5. It is as
  much real H.265 either way. It gains less, because there the encoder is
  the cost: 218 ms to 30 at 256 pixels, 250 to 92 at 512, nothing at 1024.
  libx265 is told to be quiet and to use one thread; by default it starts
  one for every core, for each frame.
- **A damaged stream is decoded by the binary in exact mode.** The encoded
  slices are the binary's to the byte (the streams differ in one message,
  which names the libx264 release that wrote them and is never put at risk
  of loss). With error concealment on, the two decoders gave the same frame
  every time. With it off, a lost slice shows whatever the decoder's frame
  buffer held: in process that is the zeroed, green hole every time, and
  from the binary sometimes that and sometimes a stale picture, depending
  on which buffer it reused. 15 of 120 streams differed, all of them
  unconcealed. Neither is wrong and neither can be held to the other, so
  exact mode keeps one of its two processes: 184 ms to 104 at 512 pixels,
  where fast mode is 19.
- **What it costs**: about 130 MB installed, wheels that bundle FFmpeg
  built with GPL components (libx264, libx265), which matters to anyone
  redistributing an environment, and Python 3.12 for the release tested
  (19.0.2). `WEATHERMENTATIONS_CODEC=ffmpeg` keeps to the binary. Without
  the binary PyAV alone is enough, in either mode.

## A sixth round: rain with a dry road

A rain frame with a dry road was 174 ms at 1024 pixels, and the streaks
were 5% of it. The rest was relighting the scene, reading its colours and
blending, in numpy. By line, one pinned core, milliseconds a frame over
four scenes and two seeds:

| Stage | Was | Now | What changed |
| --- | ---: | ---: | --- |
| overcast light | 45.9 | 13.4 | the contrast gain from a table; luminance once; one C pass |
| the light a streak is lit by | 28.5 | 7.2 | the frame's mean colour summed by OpenCV; the brightest pixels found from a histogram |
| the veil and the streaks over it | 26.0 | 2.4 | one C pass, numpy's arithmetic |
| the sky's edge | 25.7 | 22.2 | redrawn on the rows the mask reaches |
| the veil's colour | 8.8 | 5.2 | the mean colour, as above |
| streaks, their opacity, the veil's transmission, sampling | 26.6 | 25.3 | nothing |
| **a frame, untraced** | **175** | **86** | |

- **The blend is exact.** `linear * t + airlight * (1 - t)` and then
  `out * (1 - a) + light * a` are two expressions and ten frame-sized
  results. One kernel does both with the same arithmetic in the same
  order, for `RandomRain` and `RandomSnow`, so the frame is the same to
  the bit in both modes.
- **Overcast light took a power for every pixel.** The contrast gain is
  `(level / luminance) ** exponent`, and luminance is in `[0, 1]`, so fast
  mode takes the power for 65536 values of it and looks the gain up, a
  luminance within 8 millionths of the pixel's own. The reference also
  computed luminance three times, the third being the first again, and the
  grey of the relit frame, which is the gain times the luminance it came
  from. What is left is one pass, by a kernel that gives the floats of the
  numpy written beside it. Within a level of exact mode, and that in 0.1%
  of values on the scenes.
- **Half of one estimate was a mean.** `frame.reshape(-1, 3).mean(axis=0)`
  runs numpy's reduction down a million three-element rows: 13 ms, and
  good to five digits, since it adds float32 values in sequence. OpenCV
  does it in 0.6 ms in double. The mean of the brightest 3% of pixels was
  a percentile, which is a partial sort, and a copy of those pixels out of
  the frame; fast mode counts a histogram and averages above its threshold
  in place.
- **The sky's edge gave least.** Its blurs were already on a reduced copy.
  What remains is arithmetic on each pixel and what the grown mask says of
  it, and below the lowest row that mask reaches the answer is zero, so it
  is done on the rows above: the same bytes in both modes, and 14% here
  because in these scenes the sky's reach is most of the frame. It is
  shared by fog, snow and night fog.
- **Not done**: the veil's transmission is an exponential of depth with
  the sky's sharp edge in it, which a reduced copy would blur where
  earlier artefacts were; and taking the streaks' opacity only where there
  are streaks measured slower (see [Found on the way](#found-on-the-way)).
- **What it carried with it**: `RandomSnow` 296 ms to 199 and the wet road
  265 to 175 at 1024 pixels, from the same blend, overcast and statistics.

## Three suggestions, measured

Three general prescriptions were tried against this code. Each was worth
something in one place and nothing in the others.

**Floats in place of doubles.**
- *The kernels* already use float32 in every per-pixel loop. Doubles are left
  only in the set-up of a particle, which is under 6% of the flake kernel.
- *The numpy code* was audited by running every transform with a hook that
  reports frame-sized float64 arrays. It found four fields in `SunFlare` and
  one in the snow code, all float64 for the same accidental reason: `np.cos`
  returns an `np.float64`, and numpy widens a float32 array that meets one.
  Fast mode keeps them float32: 1.16 times on `SunFlare`.
- *Random draws* stay float64. Drawing float32 directly changes the stream,
  so a seed would give a different picture, not a rounded one, for 10% of
  one call.
- *A second audit, of every allocation.* The first looked at a function's
  local variables when it returned, and so missed an array that was float64
  for three lines and then cast. The second interposes `malloc` and counts
  allocations of exactly 8, 16 and 24 bytes a pixel on a frame whose sides
  are primes, which nothing float32 or uint8 comes to, and a line tracer
  says which line of the library was running. It sees temporaries with no
  name. In fast mode, over every transform, it found three things:

  | Where | Why it was wide | Made float32 | Gain |
  | --- | --- | --- | --- |
  | settled snow: six fields a frame, and four more in each ground pattern | two scalars: `np.ptp` of a pair of constants, and an offset drawn with `rng.uniform`, both `np.float64` | fast mode | 409 ms to 385 at 1024 px on a frame with settled snow; one level in 0.002% of values |
  | the lamp's beam in `NightRain` and `NightSnow` | coordinates made as two int64 grids and cast | both modes, the same floats: a row and a column that broadcast | 13 ms to 5 at 1024 px |
  | Gaussian noise in `SensorNoise` and `NightIrMode`; the particle sampler's 2 million uniforms | drawn as float64 and cast | no | drawn as float32 the draw is 10% faster and another stream |

  The beam is also a Gaussian of a sum of squares, which is a profile
  across the frame times one down it. Fast mode takes it that way, two
  exponentials of a line for one of a frame: 5 ms to 0.6, a float's last
  bit away. With both, `NightRain` is 1.2 to 1.3 times faster and
  `NightSnow` 1.5 times. Nothing else in fast mode allocates a float64 or
  int64 frame; what is left is index arrays, which numpy needs wide.

**Multiplications in place of divisions.**
- *Polygonal sprites* divided by a per-edge constant for every pixel and
  every blade. Folding the reciprocal into the edge normal once per sprite
  is the 1.2 to 1.4 times in the table above.
- *Round sprites* have two divisions by constants per pixel. Replacing them
  gains 3%.
- *numpy over a frame*: `frame / c` is 1.71 ms and `frame * (1 / c)` 1.56 ms
  at 1024 pixels. Not worth a changed result; left alone.
- *An integer division per pixel* in the streak kernel is gone, with no
  change in time that could be measured.

**Narrower integers.**
- *The Bayer site map* was int64. As uint8 it is an eighth the memory and
  indexes no faster (numpy widens an index array again). What paid was not
  having the map: the strided copies above.
- *The streak kernel's pixel list* is now two int16 coordinates in place of
  one int32 index. Same size, same speed.
- *Table indices* as uint16 convert three times faster than int32 in the
  numpy encode (0.7 ms against 2.1 ms), but a NaN then becomes index 0
  instead of an error. Left as int32; the kernel does not use that path.

## Compiler flags and the types left in the C, measured

Each flag was applied to the whole library, every kernel timed with it on
one pinned core, and its output compared with the standard build's. Times in
milliseconds, AVX2 tier, best of nine interleaved runs; "changed" means the
output is no longer the bytes the standard build gives.

| Build | encode | decode | small sprites | large round | large iris, exact | large iris, fast | streaks | wet road | Output |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| standard | 1.33 | 0.98 | 2.85 | 18.4 | 23.1 | 19.7 | 24.7 | 88.2 | |
| `-fstrict-overflow` | 1.30 | 0.98 | 2.84 | 18.5 | 23.2 | 19.7 | 24.7 | 87.5 | same; identical machine code |
| `-fomit-frame-pointer` | 1.28 | 0.96 | 2.87 | 18.5 | 23.2 | 19.8 | 24.7 | 88.2 | same; identical machine code |
| `-ffp-contract=fast` | 1.27 | 0.92 | 2.87 | 18.3 | 23.2 | 19.8 | 24.6 | 88.7 | same; identical machine code |
| `-ffp-contract=fast -mfma` | 1.29 | 0.93 | 2.69 | 18.0 | 22.9 | 19.0 | 24.6 | 89.5 | changed |
| `-flto` | 1.29 | 0.92 | 2.86 | 18.5 | 23.2 | 19.8 | 24.4 | 89.3 | same |
| `-Ofast` | 1.17 | 0.94 | 2.50 | 17.0 | 18.5 | 17.5 | 24.5 | 88.4 | changed |
| `-Ofast -ffp-contract=fast -fomit-frame-pointer -flto` | 1.18 | 0.93 | 2.51 | 16.9 | 18.7 | 17.5 | 24.6 | 87.8 | changed |
| the same four with `-mfma` | 1.19 | 0.95 | 2.39 | 16.6 | 18.5 | 17.2 | 24.5 | 87.7 | changed |
| `-march=native -mtune=native` | 1.28 | 0.94 | 2.85 | 18.5 | 23.1 | 19.7 | 24.7 | 88.2 | same |
| without `-funroll-loops` | 1.59 | 0.93 | 2.78 | 18.2 | 22.9 | 19.7 | 24.4 | 89.6 | same |
| `-O2` | 8.09 | 1.09 | 6.34 | 65.7 | 97.2 | 65.9 | 29.9 | 97.1 | same |

"Wet road" is the whole wet-road pass, of which the kernel is 2 ms. The
colour columns were timed with `keep_freed_memory()` on; without it the cost
of mapping their output arrays hid the flags.

- **Three of the flags change nothing at all.** `-fstrict-overflow` and
  `-fomit-frame-pointer` are what GCC does at `-O3` on x86-64 already.
  (Python's own build passes `-fno-strict-overflow`, which is why that one
  mattered in Pillow-AllSIMD; this library is not compiled with Python's
  flags, and its loops count in `ptrdiff_t` and `size_t` in any case.)
  `-ffp-contract=fast` has nothing to contract into until the target has a
  fused multiply-add, which the AVX2 tier's `-mavx2` does not give it.
- **`-flto` changes no time.** Each tier is one translation unit whose
  functions are static, so the compiler could already inline everything
  that link-time optimisation would.
- **`-funroll-loops` is adopted**: 16% on the encode's table lookups here,
  8 to 18% over the runs made, nothing elsewhere, the same output. The
  library grows from 63 to 104 KB.
- **`-Ofast` is not.** Its best case is the exact-mode iris, 20%, where it
  does on its own what fast mode already does by hand. It gives 8 to 12% on
  the encode and the other sprites, and nothing on streaks or the wet road. Of a snow frame that is about half a
  percent. The price is the build's one promise: with `-ffast-math` each
  tier and each compiler may reorder arithmetic its own way, so SSE2, AVX2
  and NEON stop giving the same bytes; the NaN that the encode refuses is
  assumed not to exist; and there is no exact mode left in that binary.
- **Fused multiply-add is not**, for the same reason: 6% on the smallest
  sprites and 4% at most elsewhere, and an AVX2 tier that no longer agrees
  with the SSE2 one.
- **All four of the flags together are `-Ofast`**, to within the noise, and
  with `-mfma` a further 2 to 5% on sprites.
- **`-march=native -mtune=native` changes no time on this machine**, and no
  output. GCC 13 does not know the CPU (a Core Ultra 9 285H) and takes it
  for `sierraforest`; what `native` adds over `-mavx2` here is FMA, which
  the build does not contract into, and AVX-VNNI, which is integer. There
  is no AVX-512 for it to find. See [below](#building-for-one-machine).
- **`-O2` is two to six times slower** on everything vectorised.

### Building for one machine

`WEATHERMENTATIONS_BUILD_MARCH=native` at build time adds a third tier,
`native`, compiled with `-march=native -mtune=native` (`-mcpu=native` on
arm64), and the library prefers it:

```bash
WEATHERMENTATIONS_BUILD_MARCH=native python hatch_build.py
python -c "from weathermentations import _native; print(_native.tiers(), _native.tier())"
# ('sse2', 'avx2', 'native') native
```

It gives the same bytes as the other tiers, which the tests and the hash
check confirm for it like any other. It gained nothing on the machine it was
tried on; a CPU with AVX-512, where the compiler can use vectors twice as
wide, is where it might. A library built this way is for the machine that
built it. Before using the tier the dispatcher checks the CPU for the main
instruction sets it was compiled with and falls back to `avx2` without
them, but that check is not a guarantee for every extension `native` can
turn on, so do not put such a build in a wheel.

On the types still visible in the source:

- **The doubles are per particle, not per pixel.** A sprite's size and
  corner are rounded from Python floats in the numpy code, so the kernel
  computes them in double to land on the same pixel. Every per-pixel loop
  is float. Rewriting the set-up in float, with a reciprocal for its one
  division per column, measured no faster even on six thousand of the
  smallest sprites, where set-up is the largest share, and moved sprites.
- **The remaining divisions** are per pixel only in exact mode, where they
  are numpy's own; fast mode multiplies. The rest are per particle, or are
  OpenCV's integer arithmetic for a line's slope, reproduced so that the
  same pixels are set.
- **int16 for int32 gains nothing here.** C widens a 16-bit integer to `int`
  for arithmetic anyway. A narrower type pays where it halves a large array
  or doubles the lanes of a vector loop, and the kernels have neither: their
  integer inputs are a megabyte per 16,000 streaks, read once, and their
  vector loops are all float. Tried twice, in the pixel list and in the
  line walker, with no change either time.

## Loop-level changes and an all-float build, measured

Two more prescriptions were tried on the kernels. As before, each was
applied, every kernel timed with and without it on one pinned core, best of
eleven interleaved runs, and the outputs compared.

**Hoisting, pointer walks and shared indices.** Every kernel's loops were
gone over for what the usual list asks: values that do not change inside a
loop taken out of it, row addresses stepped instead of multiplied, an index
shared between arrays computed once. Milliseconds at 1024 pixels:

| Kernel | Change | AVX2: before | after | SSE2: before | after |
| --- | --- | ---: | ---: | ---: | ---: |
| what each pixel mirrors | whole numbers in place of a double per pixel | 2.18 | 1.55 | 3.29 | 3.16 |
| the mirror image | no remainder by 3 per value; a row's sky share taken once | 2.07 | 1.13 | 2.11 | 1.28 |
| the denoise filter, tabled | one sweep along and down, a row smoothed while it is in the cache; no test per pixel | 20.6 | 19.4 | 25.7 | 24.2 |
| the denoise filter, exact | the same | 48.0 | 48.0 | 53.8 | 53.2 |
| streaks | a span's row address and bounds once per span; the patch's corner in the frame once per streak | 24.9 | 25.4 | 24.6 | 24.4 |
| wet composite | rows by stepped pointers | 1.97 | 2.03 | 2.75 | 2.76 |
| large sprites and their blur | rows by stepped pointers; a row's borders found once | 18.3 | 18.1 | 23.6 | 23.3 |
| colour | already one loop per block | 1.38 | 1.37 | 1.93 | 1.98 |

- **It paid where the loop held something the compiler may not remove.**
  A remainder by three on every value, and a conversion to double and back
  on every pixel, are arithmetic the C asks for. Taking them out is 29% and
  45% on the AVX2 tier.
- **It paid nothing where it only restated the loop.** Stepping a pointer
  in place of `base + row * width`, or naming a loop-invariant outside the
  loop, is what `-O3` does already: every such change measured between
  0.95 and 1.03, which is the noise. They are kept, since they are no
  harder to read.
- **Fusing the filter's sweeps is the cache.** The frame as floats is 12 MB
  at 1024 pixels. Smoothing all rows and then sweeping down read it twice;
  drawing each row towards the one above as soon as it has been smoothed
  reads it once. 6% here, and more on a larger frame.
- **Carrying a pixel's three channels in one four-wide register was
  slower**, by 1.1 to 2.1 times. Written in plain C as a four-float load
  and a three-float store, each load overlaps the store before it and
  waits for it. It would need intrinsics or a four-channel frame.
- **Every output is the same bytes** as before the changes.

**Every `double` made `float`.** For a test, as asked: every `double` in
the kernels became `float`, every maths call its `f` form, and constants
were compiled single precision.

| | encode | decode | small sprites | large round | iris, exact | iris, fast | streaks | mirror | reflection | composite | denoise | denoise, tabled |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AVX2, float / double | 1.02 | 1.00 | 0.97 | 0.99 | 1.00 | 1.00 | 1.00 | 0.95 | 0.97 | 0.97 | 1.00 | 1.02 |
| SSE2, float / double | 0.99 | 0.98 | 0.97 | 1.00 | 1.00 | 1.00 | 0.99 | 0.99 | 1.03 | 1.01 | 1.00 | 1.00 |
| values that changed | none | none | 16% | 60% | 65% | 65% | 29% | none | none | none | none | none |

- **No kernel is faster by more than the noise.** The doubles are in a
  particle's set-up, once per flake or streak against thousands of pixels
  each, and in a filter's set-up, once per frame. The per-pixel loops were
  float already.
- **Sprites and streaks change**, by a part in a million of their peak at
  most (1.2e-5 for polygonal sprites). That is nothing to the eye and it
  is the end of exact mode for them: 6 of the 49 kernel tests fail, the
  ones that hold a kernel to numpy's bytes, because numpy's set-up is in
  double.
- **Not adopted**, for a cost with no gain to set against it.

## A memory pool for the kernels

Asked: would keeping the kernels' scratch buffers between calls, in place of
allocating them each time, be faster?

**Inside the kernels, no.** A variant that keeps its scratch in thread-local
storage was timed against the one that allocates:

| 3,000 flakes | allocating | pooled |
| --- | ---: | ---: |
| sprites up to 60 px | 16.68 ms | 16.58 ms |
| up to 15 px | 1.90 ms | 1.90 ms |
| a few pixels | 0.97 ms | 0.97 ms |

The kernels allocate a handful of times per call, not per particle, and
60 calls made 199 allocator system calls in all. A pool would add state
shared between calls to save 0.6%.

**The kernels' output arrays, barely.** Writing into a reused buffer in
place of a fresh array is 1.37 ms against 1.40 ms for the encode at 1024
pixels.

**The numpy code around them, yes, and without a pool.** Every numpy
expression over a frame allocates a frame and frees it. glibc returns blocks
of that size to the operating system and maps them again, and the pages are
zeroed each time. Telling it to keep them
(`weathermentations.keep_freed_memory()`, which sets `M_MMAP_THRESHOLD` and
`M_TRIM_THRESHOLD`) measured:

| Transform | default | kept |
| --- | ---: | ---: |
| `SensorNoise`, 512 px | 12.7 ms | 5.9 ms |
| `SensorNoise`, 1024 px | 39 ms | 28 ms |
| `SunFlare`, 1024 px | 111 ms | 100 ms |
| `DirtSoil`, 1024 px | 255 ms | 240 ms |
| `WaterDroplets`, 1024 px | 125 ms | 116 ms |
| `AcFlicker`, 1024 px | 4.5 ms | 4.5 ms |

It is a setting of the whole process, so the library does not make it on
import; call it in a `DataLoader`'s `worker_init_fn`. Both settings are
needed: either alone measured slower than neither. Peak memory did not move
in these runs, but the process no longer shrinks back after a peak. Each
figure is one transform called in a loop; a pipeline's mix of sizes was not
measured.

## Found on the way

- **Exact mode could be handed fast mode's depth.** The shared depth
  provider keyed its cache on the frame alone. The bundled estimator runs
  in half precision in fast mode, so a frame whose depth had been
  estimated there and was asked for again inside `with exact():` got the
  half-precision map, and exact mode depended on what had run before it.
  A cached map now carries the mode it was made in. It was found by a
  check that switched modes in one process; the checks before it ran each
  mode in its own.
- **numpy's mean down a long axis is good to five digits.**
  `frame.reshape(-1, 3).mean(axis=0)` adds a million float32 values one
  after another, without the pairwise blocks its other reductions use, and
  comes out 1e-5 from the mean taken in double. It was also 13 ms. Fast
  mode's `frame_mean` is closer to the truth than the reference it departs
  from.
- **A ufunc's `where=` is not a way to skip work.** Taking `expm1` only
  where a streak layer is not zero was 1.3 to 2 times slower than taking
  it everywhere, which numpy already does fastest where the layer is zero.
- **Rain streaks are not antialiased.** `render_streak_layer` asks
  `cv2.polylines` for `LINE_AA`, and OpenCV ignores that on a float32 image
  and draws 8-connected lines. The kernel reproduces what is drawn today.
  Whether streaks should be antialiased is a rendering decision, and it is
  now a change to one C function.
- **A polygonal aperture is evaluated in float64**, by an accident of numpy
  scalar types in `_smeared_aperture_distance`. The kernel uses float32,
  which is where the 1.5e-6 comes from.
- **Two items of `optimizations.md` are overtaken.** D1 proposed drawing
  streaks in opacity buckets, 4.3 times faster on the drawing primitive at
  the cost of quantising opacity; the kernel draws the whole field 10 to 20
  times faster with nothing quantised. D2 proposed keeping a pipeline in
  linear light to save a 31 ms round trip per transform; the round trip is
  now about 2 ms, which no longer justifies changing the contract between
  transforms.

## Not done

- **Binary wheels.** A wheel holding the library is tagged for the machine
  that built it, which PyPI does not accept, so the publish workflow builds
  a pure-Python wheel. Publishing binaries needs a `cibuildwheel` job for
  manylinux x86-64 and arm64 and macOS arm64. Until then a user gets the
  kernels by installing from the sdist with a C compiler present.
- **Windows.** The build is skipped there (the source uses variable-length
  arrays and GCC flags), and the numpy code runs.
- **CI has never run.** The workflow now builds the kernels, runs the suite
  both ways and adds an arm64 job, and none of that has been executed: the
  repository has no remote.
- **NEON on real hardware**, above.
- **What is left in the frame time**, at 1024 pixels on one thread, none of
  it a case for intrinsics:
  - `bilateralFilter`, most of `WdrToneMapping`: OpenCV's, and the effect
    itself. No shortcut found that keeps the picture.
  - The sky's edge, a quarter of a dry-road rain frame and shared by fog,
    snow and night fog: a mean of the sky's colour near each pixel, and a
    comparison with it, over most of the frame.
  - The veil's transmission and the streaks' opacity: an exponential over
    the frame each.
  - The blob patches of `DirtSoil`'s deposit field, placed on whole pixels:
    the two pixels fast mode moves them by would go with sub-pixel
    placement.
  - H.265 at 1024 pixels and above, where the encoder is the cost.
- **PyAV on other versions.** The in-process codec was run against PyAV
  19.0.2 only, which needs Python 3.12. The extra asks for `av>=12` and the
  code keeps to calls that are old, but no older release was tried. A
  release that encodes differently is caught by the check against the
  binary and left out of exact mode.
- **Threads.** The kernels run on one. A `DataLoader` already uses one
  worker per core, so splitting a frame across threads would oversubscribe.

## Reproducing

```bash
python hatch_build.py                         # build the kernels in place
pytest tests/unit/test_native_kernels.py tests/unit/test_native_build.py -q
sh native/check/run.sh                        # hashes, per tier
taskset -c 2 sh native/experiments/run.sh     # plain C against intrinsics
WEATHERMENTATIONS_NATIVE=0 pytest tests -q    # the numpy path alone
pip install -e ".[codec]"                     # PyAV, then:
pytest tests/unit/test_inprocess_codec.py -q  # the codecs in process against ffmpeg
```

The table at the top came from a short driver around the transforms that
`scripts/benchmark.py` builds: each on four of the example scenes at 512
and 1024 pixels with two seeds, the best of three calls after a warm-up,
with `cv2.setNumThreads(1)` and pinned with `taskset`. It was run from a
worktree of the parent branch, from this branch, and from this branch with
PyAV on the path, in that order, three times over, and each cell is the
least of its three. The driver is not in the repository.
