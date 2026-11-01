/* Entry points of the native library, and the choice of tier.
 *
 * wm_kernels.c is compiled once per tier; each compilation exports one
 * table. On x86-64 there are two, the psABI baseline (SSE2) and AVX2, and
 * the best one the CPU and the operating system support is chosen when the
 * library is loaded. Elsewhere there is one. Every tier gives the same
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

static const wm_kernel_table *active = &WM_BASE_TABLE;

static int cpu_has_avx2(void) {
#if defined(WM_HAVE_AVX2)
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2");
#else
    return 0;
#endif
}

WM_EXPORT int wm_abi_version(void) { return WM_ABI_VERSION; }

/* The tiers this build holds, best last, separated by commas. */
WM_EXPORT const char *wm_tiers(void) {
#ifdef WM_HAVE_AVX2
    static char names[64];
    if (!names[0]) {
        strcpy(names, WM_BASE_TABLE.name);
        strcat(names, ",avx2");
    }
    return names;
#else
    return WM_BASE_TABLE.name;
#endif
}

WM_EXPORT const char *wm_tier(void) { return active->name; }

/* Choose a tier by name, or the best the CPU has with "auto". Returns 0 when
 * the tier is now active, 1 when this build or this CPU does not have it. */
WM_EXPORT int wm_select(const char *name) {
    if (strcmp(name, "auto") == 0) {
        active = &WM_BASE_TABLE;
#ifdef WM_HAVE_AVX2
        if (cpu_has_avx2()) {
            active = &wm_table_avx2;
        }
#endif
        return 0;
    }
    if (strcmp(name, WM_BASE_TABLE.name) == 0) {
        active = &WM_BASE_TABLE;
        return 0;
    }
#ifdef WM_HAVE_AVX2
    if (strcmp(name, "avx2") == 0 && cpu_has_avx2()) {
        active = &wm_table_avx2;
        return 0;
    }
#endif
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
