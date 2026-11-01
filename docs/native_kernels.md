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
three steps took the depth-conditioned weather transforms from 3 to 7.5
times faster on one core and left nothing for the fourth to do.

**C, yes, in three places.** Colour conversion, snowflake sprites and rain
streaks are now plain C, compiled per instruction-set tier and loaded
through `ctypes`, with the numpy code kept as the definition and the
fallback. The loops they replace were slow because of the interpreter, not
the arithmetic: one Python call per snowflake, a dozen numpy and OpenCV
calls per rain streak, five passes over the frame to turn linear light into
code values. A wholesale translation would gain little more, because most of
what is left is inside OpenCV filters.

**Explicit SIMD, no, not now.** Three loops of the kernels were written a
second time with AVX2 and NEON intrinsics and timed against the plain C. On
x86-64 the intrinsics were between 0.74 and 1.14 times the compiler's speed.
The best of those is worth about 1% of one transform. The arm64 half of the
question is still open: the NEON code is verified correct under emulation,
but nothing here could time it.

## What it bought

Milliseconds per frame on one pinned performance core with OpenCV held to
one thread, which is what a `DataLoader` worker has. "Before" is the parent
branch, "numpy only" is this branch without the compiled library, "kernels"
is with it. Each figure is the best of six interleaved rounds.

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

Three things to know before reading it.

- **Part of the first four rows is a regression being undone.** The sky-edge
  redraw that the previous round shared between fog, rain, snow and night
  fog ran two Gaussian blurs with a sigma of 58 pixels and a 59-pixel disc
  dilation at full resolution. With OpenCV's sixteen threads that was about
  a fifth of a second at 1024 pixels. On one thread it was most of a
  second. The README's timing table was measured before that change and
  with all threads, so it never showed it.
- **The machine was not idle.** Other jobs were running on it throughout and
  the hardware counters were not readable (`perf_event_paranoid` is 4). The
  same pinned loop measured 87 ms in one hour and 144 ms in another. Ratios
  within a row hold; absolute figures are good to perhaps 40%.
- **The README's table is all-threads and now stale.** It has not been
  re-measured on this branch.

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
  thread, timed back to back. **This is the one change on the branch that
  alters pictures**; see [What changed in the pictures](#what-changed-in-the-pictures).
- **Channel reductions as two ufunc calls**
  ([`utils/channels.py`](../src/weathermentations/utils/channels.py)). Exact.

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
| Fast mode, with bounded differences from exact | Not adopted: the kernels match numpy, and the one approximation is in Python |
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
  -fno-thread-jumps`. No fused multiply-add and no `-ffast-math`, so every
  tier and compiler rounds as numpy does.
- **Optional.** [`hatch_build.py`](../hatch_build.py) compiles it when a
  wheel or an editable install is made, and goes on without it if there is
  no C compiler. `python hatch_build.py` builds it in place in a checkout.
  Without it every transform runs its numpy code and gives the same picture.

| Variable | Effect |
| --- | --- |
| `WEATHERMENTATIONS_NATIVE=0` | run the numpy code even when the library is there |
| `WEATHERMENTATIONS_NATIVE=require` | a missing library is an `ImportError` |
| `WEATHERMENTATIONS_ISA=sse2` | use that tier, to compare tiers on one machine |
| `WEATHERMENTATIONS_BUILD_NATIVE=0` / `require` | skip the build, or fail if it fails |

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
  seeds, 152 frames. With the kernels on and off, 151 are byte-identical and
  one snow frame differs in one value of 1,152,000, by one level. The SSE2
  and AVX2 tiers are identical on all 152.
- **Across machines**: [`native/check/run.sh`](../native/check/run.sh) prints
  a hash per kernel. The SSE2 and AVX2 builds (glibc, GCC 13.3) and an arm64
  build under QEMU (musl, GCC 13.2) print the same four hashes.
- **Memory**: the kernel tests pass with the library built with
  `-fsanitize=address,undefined`.
- **Refused input**: a NaN in a frame still raises, as the table lookup
  always has, instead of becoming a black pixel in C.
- **The full suite** passes with the kernels on and with
  `WEATHERMENTATIONS_NATIVE=0`.

## What changed in the pictures

Only the reduced-size blur and dilation in the sky-edge redraw, and only in
`RandomFog`, `RandomRain`, `RandomSnow` and `NightFog`. Against the parent
branch on the 152 frames:

| Transform | Pixels changed | Mean difference (levels) | Largest (levels) |
| --- | ---: | ---: | ---: |
| fog | 0.9% | 0.010 | 7 |
| rain | 1.4% | 0.017 | 9 |
| snow | 1.2% | 0.014 | 22 |
| night fog | 0.8% | 0.013 | 8 |

The differences lie along the skyline, where the redraw decides how far the
sky mask may grow, and are largest in the night street scene. The blur
accounts for almost none of it (two levels at most, in 0.1% of pixels); the
dilation accounts for the rest. Side by side the frames are not
distinguishable, and the golden images pass unregenerated, but the skyline
is where the previous round's artefacts were, so this change is in a commit
of its own. If the skyline is to stay exactly as it was, `wide_dilate` can
be swapped back for `cv2.dilate` alone, at about 80 ms per frame at 1024
pixels on one thread; reverting the whole commit brings back the blurs as
well, which cost several times that.

## Found on the way

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
  - `edgePreservingFilter` and `bilateralFilter` (`TemporalDenoiseArtifacts`
    439 ms, `WdrToneMapping` 84 ms): OpenCV's, and the effect itself.
  - `apply_overcast`, a quarter of `RandomSnow`: `pow`, `exp` and `log` over
    the frame. A fused C pass would need a table or a vector maths library
    to stay exact.
  - `np.percentile` on whole frames, about 10% of snow: a histogram would do.
  - Wet ground and surface normals in `RandomRain`, about a fifth of it:
    Gaussian blurs and numpy passes, candidates for `wide_blur` and for the
    518-pixel normals `optimizations.md` proposes (C3).
  - Sampling the drops for `NightRain`: a third of what is left of it.
- **Threads.** The kernels run on one. A `DataLoader` already uses one
  worker per core, so splitting a frame across threads would oversubscribe.

## Reproducing

```bash
python hatch_build.py                         # build the kernels in place
pytest tests/unit/test_native_kernels.py tests/unit/test_native_build.py -q
sh native/check/run.sh                        # hashes, per tier
taskset -c 2 sh native/experiments/run.sh     # plain C against intrinsics
WEATHERMENTATIONS_NATIVE=0 pytest tests -q    # the numpy path alone
```

The table at the top came from `scripts/benchmark.py --sizes 512 1024
--repeats 7` with `cv2.setNumThreads(1)`, pinned with `taskset`, run six
times each from a worktree of the parent branch, from this branch with
`WEATHERMENTATIONS_NATIVE=0`, and from this branch as built, in that order
and repeated.
