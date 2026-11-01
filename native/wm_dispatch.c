/* Entry points of the native library, and the choice of tier.
 *
 * wm_kernels.c is compiled once per tier; each compilation exports one
 * table. On x86-64 there are two, the psABI baseline (SSE2) and AVX2, and
 * the best one the CPU and the operating system support is chosen when the
 * library is loaded. Elsewhere there is one. A build may add a "native"
 * tier, compiled for the machine it was built on. Every tier gives the same
 * floats, so which one runs is a matter of speed only.
 */
#include <string.h>

#include "wm_kernels.h"

#define WM_EXPORT __attribute__((visibility("default")))
#define WM_CAT_(a, b) a##_##b
#define WM_CAT(a, b) WM_CAT_(a, b)

#ifndef WM_BASE
#define WM_BASE portable
#endif
#define WM_BASE_TABLE WM_CAT(wm_table, WM_BASE)

extern const wm_kernel_table WM_BASE_TABLE;
#ifdef WM_HAVE_AVX2
extern const wm_kernel_table wm_table_avx2;
#endif
#ifdef WM_HAVE_NATIVE
extern const wm_kernel_table wm_table_native;
/* Defined with the native tier: whether this CPU has what it was built for. */
extern int wm_native_usable(void);
#endif

static int always(void) { return 1; }

#ifdef WM_HAVE_AVX2
static int cpu_has_avx2(void) {
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2");
}
#endif

/* The tiers of this build, slowest first. */
static const struct {
    const wm_kernel_table *table;
    int (*usable)(void);
} tiers[] = {
    {&WM_BASE_TABLE, always},
#ifdef WM_HAVE_AVX2
    {&wm_table_avx2, cpu_has_avx2},
#endif
#ifdef WM_HAVE_NATIVE
    {&wm_table_native, wm_native_usable},
#endif
};
enum { N_TIERS = sizeof(tiers) / sizeof(tiers[0]) };

static const wm_kernel_table *active = &WM_BASE_TABLE;

WM_EXPORT int wm_abi_version(void) { return WM_ABI_VERSION; }

/* The tiers this build holds, best last, separated by commas. */
WM_EXPORT const char *wm_tiers(void) {
    static char names[96];
    if (!names[0]) {
        for (int i = 0; i < N_TIERS; i++) {
            if (i) {
                strcat(names, ",");
            }
            strcat(names, tiers[i].table->name);
        }
    }
    return names;
}

WM_EXPORT const char *wm_tier(void) { return active->name; }

/* Choose a tier by name, or the best the CPU has with "auto". Returns 0 when
 * the tier is now active, 1 when this build or this CPU does not have it. */
WM_EXPORT int wm_select(const char *name) {
    const int best = strcmp(name, "auto") == 0;
    for (int i = N_TIERS - 1; i >= 0; i--) {
        if ((best || strcmp(name, tiers[i].table->name) == 0) && tiers[i].usable()) {
            active = tiers[i].table;
            return 0;
        }
    }
    return 1;
}

WM_EXPORT int wm_encode_u8(const float *in, uint8_t *out, size_t n, const uint8_t *table) {
    return active->encode_u8(in, out, n, table);
}

WM_EXPORT int wm_encode_f32(const float *in, float *out, size_t n, const float *table) {
    return active->encode_f32(in, out, n, table);
}

WM_EXPORT void wm_decode_u8(const uint8_t *in, float *out, size_t n, const float *table) {
    active->decode_u8(in, out, n, table);
}

WM_EXPORT int wm_render_flakes(
    float *canvas, int height, int width, ptrdiff_t count,
    const float *x, const float *y, const float *radius, const float *blur_radius,
    const float *smear, const float *dir_x, const float *dir_y, const float *gain,
    int blades, double blade_rotation, double rim_gain, double cats_eye,
    double centre_x, double centre_y, int fast) {
    return active->render_flakes(
        canvas, height, width, count, x, y, radius, blur_radius, smear, dir_x, dir_y, gain,
        blades, blade_rotation, rim_gain, cats_eye, centre_x, centre_y, fast);
}

WM_EXPORT int wm_render_streaks(
    float *layer, int height, int width,
    const int32_t *select, ptrdiff_t n_select,
    const int32_t *x0, const int32_t *y0, const int32_t *patch_w, const int32_t *patch_h,
    const int32_t *vertices, int n_vertices, const int32_t *thickness,
    const float *alpha, const float *col_offset, const float *col_rate,
    const float *row_offset, const float *row_rate, const float *phase,
    const float *blur_sigma, const float *beading, int beading_size) {
    return active->render_streaks(
        layer, height, width, select, n_select, x0, y0, patch_w, patch_h, vertices, n_vertices,
        thickness, alpha, col_offset, col_rate, row_offset, row_rate, phase, blur_sigma, beading,
        beading_size);
}

WM_EXPORT int wm_wet_composite(
    float *out, const float *dry, const float *sharp, const float *rough,
    const float *wetness, const float *puddles, const float *fresnel,
    const float *film_fresnel, ptrdiff_t rows, ptrdiff_t width,
    float ratio, float spare, float film, float gloss) {
    return active->wet_composite(
        out, dry, sharp, rough, wetness, puddles, fresnel, film_fresnel, rows, width, ratio,
        spare, film, gloss);
}
