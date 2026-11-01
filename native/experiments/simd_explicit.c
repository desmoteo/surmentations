/** Would hand-written SIMD beat the compiler on these kernels?
 *
 * Three loops of wm_kernels.c, each written twice: in plain C, as the
 * library has it, and with intrinsics (AVX2 on x86-64, NEON on arm64). The
 * intrinsics do the same operations in the same order, so the outputs must
 * be equal bit for bit, and the program checks that before it prints a time.
 *
 * Build and run with run.sh, which compiles this file once per set of flags.
 * The results on the machines tried, and what was concluded from them, are
 * in docs/native_kernels.md.
 */
/**/

/** Asks <time.h> for clock_gettime under -std=c11. */
#define _POSIX_C_SOURCE 199309L
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(__AVX2__)
#include <immintrin.h>
/** Whether this target has intrinsics to compare the plain C with. */
#define HAVE_EXPLICIT 1
/** What the intrinsic versions are written in, for the report. */
#define EXPLICIT "avx2 intrinsics"
#elif defined(__ARM_NEON)
#include <arm_neon.h>
/** Whether this target has intrinsics to compare the plain C with. */
#define HAVE_EXPLICIT 1
/** What the intrinsic versions are written in, for the report. */
#define EXPLICIT "neon intrinsics"
#else
/** No intrinsics for this target: the plain C alone is timed. */
#define EXPLICIT "none"
#endif

/** Keeps each timed loop a function of its own, as the library's are. */
#define NOINLINE __attribute__((noinline))

/** ``v`` clamped to ``[0, 1]``; a NaN comes out as 0. */
static inline float clamp01(float v) {
    v = v > 0.0f ? v : 0.0f;
    return v < 1.0f ? v : 1.0f;
}

/* ------------------------------------------------------------ plain C */

/** Values the colour loop converts at a time, as the library's does. */
#define BLOCK 512

/** The colour kernel's table lookup as the library writes it: plain C, vectorised by the compiler.
 */
static NOINLINE void encode_plain(
    const float *restrict in, uint8_t *restrict out, size_t n, const uint8_t *restrict table) {
    int32_t index[BLOCK];
    for (size_t done = 0; done < n; done += BLOCK) {
        const size_t count = n - done < BLOCK ? n - done : BLOCK;
        const float *restrict src = in + done;
        for (size_t i = 0; i < count; i++) {
            index[i] = (int32_t)(clamp01(src[i]) * 65535.0f);
        }
        uint8_t *restrict block = out + done;
        for (size_t i = 0; i < count; i++) {
            block[i] = table[index[i]];
        }
    }
}

/** A row of a round sprite's distance to its smear segment, as the library writes it. */
static NOINLINE void capsule_plain(
    float *restrict row, const float *restrict dx, ptrdiff_t size,
    float dy, float ux, float uy, float hs) {
    const float dyuy = dy * uy;
    for (ptrdiff_t c = 0; c < size; c++) {
        float along = dx[c] * ux + dyuy;
        along = along < -hs ? -hs : along;
        along = along > hs ? hs : along;
        const float ex = dx[c] - along * ux, ey = dy - along * uy;
        row[c] = sqrtf(ex * ex + ey * ey);
    }
}

/** A row of distance to brightness, as the library writes it. */
static NOINLINE void shade_plain(float *restrict row, ptrdiff_t size, float rim, float level) {
    for (ptrdiff_t c = 0; c < size; c++) {
        /* `inside` is in [0, 1], so the edge can only pass 1 and the fall
         * only pass 0: the other two clamps numpy's code has never bind, and
         * leaving them out is a fifth faster for the same floats. */
        const float inside = clamp01(1.0f - row[c]);
        float edge = inside / 0.12f;
        edge = edge < 1.0f ? edge : 1.0f;
        float fall = 1.0f - inside / 0.35f;
        fall = fall > 0.0f ? fall : 0.0f;
        row[c] = edge * (1.0f + rim * fall) * level;
    }
}

/* --------------------------------------------------------- intrinsics */

#if defined(__AVX2__)

/** Eight values clamped to ``[0, 1]``. */
static inline __m256 clamp01_v(__m256 v) {
    /* max(v, 0) then min(v, 1) with the scalar code's operand order, so a
     * NaN comes out as 0 here too. */
    v = _mm256_max_ps(v, _mm256_setzero_ps());
    return _mm256_min_ps(v, _mm256_set1_ps(1.0f));
}

/** ``encode_plain`` with AVX2 intrinsics: eight indices and eight lookups, by a gather, at a time.
 */
static NOINLINE void encode_explicit(
    const float *in, uint8_t *out, size_t n, const uint8_t *table) {
    const __m256 scale = _mm256_set1_ps(65535.0f);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256i index =
            _mm256_cvttps_epi32(_mm256_mul_ps(clamp01_v(_mm256_loadu_ps(in + i)), scale));
        /* Four bytes are read at each index and the low one kept; the
         * caller's table has three bytes of padding after its last entry. */
        __m256i bytes = _mm256_i32gather_epi32((const int *)table, index, 1);
        bytes = _mm256_and_si256(bytes, _mm256_set1_epi32(0xff));
        __m128i narrow = _mm_packus_epi32(
            _mm256_castsi256_si128(bytes), _mm256_extracti128_si256(bytes, 1));
        narrow = _mm_packus_epi16(narrow, narrow);
        _mm_storel_epi64((__m128i *)(out + i), narrow);
    }
    for (; i < n; i++) {
        out[i] = table[(int32_t)(clamp01(in[i]) * 65535.0f)];
    }
}

/** ``capsule_plain`` with AVX2 intrinsics, eight pixels at a time. */
static NOINLINE void capsule_explicit(
    float *row, const float *dx, ptrdiff_t size, float dy, float ux, float uy, float hs) {
    const __m256 vux = _mm256_set1_ps(ux), vuy = _mm256_set1_ps(uy), vdy = _mm256_set1_ps(dy);
    const __m256 dyuy = _mm256_set1_ps(dy * uy);
    const __m256 lo = _mm256_set1_ps(-hs), hi = _mm256_set1_ps(hs);
    ptrdiff_t c = 0;
    for (; c + 8 <= size; c += 8) {
        const __m256 x = _mm256_loadu_ps(dx + c);
        __m256 along = _mm256_add_ps(_mm256_mul_ps(x, vux), dyuy);
        along = _mm256_blendv_ps(along, lo, _mm256_cmp_ps(along, lo, _CMP_LT_OQ));
        along = _mm256_blendv_ps(along, hi, _mm256_cmp_ps(along, hi, _CMP_GT_OQ));
        const __m256 ex = _mm256_sub_ps(x, _mm256_mul_ps(along, vux));
        const __m256 ey = _mm256_sub_ps(vdy, _mm256_mul_ps(along, vuy));
        _mm256_storeu_ps(
            row + c, _mm256_sqrt_ps(_mm256_add_ps(_mm256_mul_ps(ex, ex), _mm256_mul_ps(ey, ey))));
    }
    if (c < size) {
        capsule_plain(row + c, dx + c, size - c, dy, ux, uy, hs);
    }
}

/** ``shade_plain`` with AVX2 intrinsics, eight pixels at a time. */
static NOINLINE void shade_explicit(float *row, ptrdiff_t size, float rim, float level) {
    const __m256 one = _mm256_set1_ps(1.0f), vrim = _mm256_set1_ps(rim);
    const __m256 vlevel = _mm256_set1_ps(level);
    const __m256 edge_width = _mm256_set1_ps(0.12f), core_width = _mm256_set1_ps(0.35f);
    ptrdiff_t c = 0;
    for (; c + 8 <= size; c += 8) {
        const __m256 inside = clamp01_v(_mm256_sub_ps(one, _mm256_loadu_ps(row + c)));
        const __m256 edge = clamp01_v(_mm256_div_ps(inside, edge_width));
        const __m256 fall = clamp01_v(_mm256_sub_ps(one, _mm256_div_ps(inside, core_width)));
        const __m256 profile = _mm256_add_ps(one, _mm256_mul_ps(vrim, fall));
        _mm256_storeu_ps(row + c, _mm256_mul_ps(_mm256_mul_ps(edge, profile), vlevel));
    }
    if (c < size) {
        shade_plain(row + c, size - c, rim, level);
    }
}

#elif defined(__ARM_NEON)

/** Four values clamped to ``[0, 1]``. */
static inline float32x4_t clamp01_v(float32x4_t v) {
    /* Selects, not vmaxq/vminq: those return NaN for a NaN operand, and the
     * scalar code returns 0. */
    v = vbslq_f32(vcgtq_f32(v, vdupq_n_f32(0.0f)), v, vdupq_n_f32(0.0f));
    return vbslq_f32(vcltq_f32(v, vdupq_n_f32(1.0f)), v, vdupq_n_f32(1.0f));
}

/** ``encode_plain`` with NEON intrinsics: four indices at a time, the lookups scalar. */
static NOINLINE void encode_explicit(
    const float *in, uint8_t *out, size_t n, const uint8_t *table) {
    const float32x4_t scale = vdupq_n_f32(65535.0f);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        const int32x4_t index = vcvtq_s32_f32(vmulq_f32(clamp01_v(vld1q_f32(in + i)), scale));
        out[i] = table[vgetq_lane_s32(index, 0)];
        out[i + 1] = table[vgetq_lane_s32(index, 1)];
        out[i + 2] = table[vgetq_lane_s32(index, 2)];
        out[i + 3] = table[vgetq_lane_s32(index, 3)];
    }
    for (; i < n; i++) {
        out[i] = table[(int32_t)(clamp01(in[i]) * 65535.0f)];
    }
}

/** ``capsule_plain`` with NEON intrinsics, four pixels at a time. */
static NOINLINE void capsule_explicit(
    float *row, const float *dx, ptrdiff_t size, float dy, float ux, float uy, float hs) {
    const float32x4_t vux = vdupq_n_f32(ux), vuy = vdupq_n_f32(uy), vdy = vdupq_n_f32(dy);
    const float32x4_t dyuy = vdupq_n_f32(dy * uy);
    const float32x4_t lo = vdupq_n_f32(-hs), hi = vdupq_n_f32(hs);
    ptrdiff_t c = 0;
    for (; c + 4 <= size; c += 4) {
        const float32x4_t x = vld1q_f32(dx + c);
        /* vmulq then vaddq, not vmlaq: no fused multiply-add. */
        float32x4_t along = vaddq_f32(vmulq_f32(x, vux), dyuy);
        along = vbslq_f32(vcltq_f32(along, lo), lo, along);
        along = vbslq_f32(vcgtq_f32(along, hi), hi, along);
        const float32x4_t ex = vsubq_f32(x, vmulq_f32(along, vux));
        const float32x4_t ey = vsubq_f32(vdy, vmulq_f32(along, vuy));
        vst1q_f32(row + c, vsqrtq_f32(vaddq_f32(vmulq_f32(ex, ex), vmulq_f32(ey, ey))));
    }
    if (c < size) {
        capsule_plain(row + c, dx + c, size - c, dy, ux, uy, hs);
    }
}

/** ``shade_plain`` with NEON intrinsics, four pixels at a time. */
static NOINLINE void shade_explicit(float *row, ptrdiff_t size, float rim, float level) {
    const float32x4_t one = vdupq_n_f32(1.0f), vrim = vdupq_n_f32(rim);
    const float32x4_t vlevel = vdupq_n_f32(level);
    const float32x4_t edge_width = vdupq_n_f32(0.12f), core_width = vdupq_n_f32(0.35f);
    ptrdiff_t c = 0;
    for (; c + 4 <= size; c += 4) {
        const float32x4_t inside = clamp01_v(vsubq_f32(one, vld1q_f32(row + c)));
        const float32x4_t edge = clamp01_v(vdivq_f32(inside, edge_width));
        const float32x4_t fall = clamp01_v(vsubq_f32(one, vdivq_f32(inside, core_width)));
        const float32x4_t profile = vaddq_f32(one, vmulq_f32(vrim, fall));
        vst1q_f32(row + c, vmulq_f32(vmulq_f32(edge, profile), vlevel));
    }
    if (c < size) {
        shade_plain(row + c, size - c, rim, level);
    }
}

#endif

/* ------------------------------------------------------------ harness */

/** A monotonic clock, in milliseconds. */
static double now_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec * 1e-6;
}

/** A uniform value in ``[0, 1)`` from a linear congruential generator. */
static float uniform(uint32_t *state) {
    *state = *state * 1664525u + 1013904223u;
    return (float)(*state >> 8) / 16777216.0f;
}

/** Times each loop is run; the fastest is kept. */
#define ROUNDS 15
/** Run ``statement`` ROUNDS times and set ``best`` to the fastest, in milliseconds. */
#define TIME(best, statement)              \
    do {                                   \
        best = 1e30;                       \
        for (int r = 0; r < ROUNDS; r++) { \
            const double t0 = now_ms();    \
            statement;                     \
            const double t1 = now_ms();    \
            best = t1 - t0 < best ? t1 - t0 : best; \
        }                                  \
    } while (0)

/** Time each loop in plain C and with intrinsics on the same inputs, and check they agree.
 *
 * Prints nanoseconds a value for both and their ratio; the first argument
 * labels the build.
 */
int main(int argc, char **argv) {
    const char *label = argc > 1 ? argv[1] : "this build";
    /* Not constants: the library's come from the data, and a compiler that
     * knows them specialises the loops into something else. */
    volatile float knob = 1.0f;
    const float ux = 0.6f * knob, uy = 0.8f * knob, hs = 0.4f * knob;
    const float rim = 0.4f * knob, level = 0.7f * knob;
    enum { PIXELS = 1024 * 1024 * 3, ROW = 96, ROWS = 4096 };
    uint32_t seed = 1;
    int failed = 0;
    double plain, explicit_ms = -1.0;

    /* Linear light as a renderer leaves it: some of it under 0 and over 1. */
    float *linear = malloc(PIXELS * sizeof(float));
    uint8_t *codes = malloc(PIXELS), *codes2 = malloc(PIXELS);
    uint8_t *table = malloc(65536 + 3);
    for (size_t i = 0; i < PIXELS; i++) {
        linear[i] = uniform(&seed) * 1.6f - 0.2f;
    }
    for (int i = 0; i < 65536; i++) {
        const double v = i / 65535.0;
        const double s = v <= 0.0031308 ? v * 12.92 : 1.055 * pow(v, 1.0 / 2.4) - 0.055;
        table[i] = (uint8_t)(s * 255.0 + 0.5);
    }
    memset(table + 65536, 0, 3);
    TIME(plain, encode_plain(linear, codes, PIXELS, table));
#ifdef HAVE_EXPLICIT
    TIME(explicit_ms, encode_explicit(linear, codes2, PIXELS, table));
    failed |= memcmp(codes, codes2, PIXELS) != 0;
#endif
    printf("%-22s encode   plain C %6.3f ns/value", label, plain * 1e6 / PIXELS);
    if (explicit_ms >= 0.0) {
        printf("   %s %6.3f ns/value (%.2fx)", EXPLICIT, explicit_ms * 1e6 / PIXELS, plain / explicit_ms);
    }
    printf("\n");

    /* Sprite rows of the width a mid-sized flake has. */
    float *axis = malloc(ROW * sizeof(float));
    float *rows = malloc((size_t)ROW * ROWS * sizeof(float));
    float *rows2 = malloc((size_t)ROW * ROWS * sizeof(float));
    for (int c = 0; c < ROW; c++) {
        axis[c] = ((float)c - ROW / 2.0f) / 37.0f;
    }
    TIME(plain, for (int r = 0; r < ROWS; r++) capsule_plain(
        rows + (size_t)r * ROW, axis, ROW, axis[r % ROW], ux, uy, hs));
#ifdef HAVE_EXPLICIT
    TIME(explicit_ms, for (int r = 0; r < ROWS; r++) capsule_explicit(
        rows2 + (size_t)r * ROW, axis, ROW, axis[r % ROW], ux, uy, hs));
    failed |= memcmp(rows, rows2, (size_t)ROW * ROWS * sizeof(float)) != 0;
#endif
    printf("%-22s capsule  plain C %6.3f ns/pixel", label, plain * 1e6 / (ROW * ROWS));
    if (explicit_ms >= 0.0) {
        printf("   %s %6.3f ns/pixel (%.2fx)", EXPLICIT, explicit_ms * 1e6 / (ROW * ROWS), plain / explicit_ms);
    }
    printf("\n");

    /* Shading overwrites its input, so each round starts from a copy. */
    float *distance = malloc((size_t)ROW * ROWS * sizeof(float));
    memcpy(distance, rows, (size_t)ROW * ROWS * sizeof(float));
    TIME(plain, memcpy(rows, distance, (size_t)ROW * ROWS * sizeof(float));
         for (int r = 0; r < ROWS; r++) shade_plain(rows + (size_t)r * ROW, ROW, rim, level));
#ifdef HAVE_EXPLICIT
    TIME(explicit_ms, memcpy(rows2, distance, (size_t)ROW * ROWS * sizeof(float));
         for (int r = 0; r < ROWS; r++) shade_explicit(rows2 + (size_t)r * ROW, ROW, rim, level));
    failed |= memcmp(rows, rows2, (size_t)ROW * ROWS * sizeof(float)) != 0;
#endif
    printf("%-22s shade    plain C %6.3f ns/pixel", label, plain * 1e6 / (ROW * ROWS));
    if (explicit_ms >= 0.0) {
        printf("   %s %6.3f ns/pixel (%.2fx)", EXPLICIT, explicit_ms * 1e6 / (ROW * ROWS), plain / explicit_ms);
    }
    printf("\n");

    if (failed) {
        printf("%-22s FAILED: the intrinsics and the plain C gave different results\n", label);
    }
    return failed;
}
