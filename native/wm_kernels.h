/* Kernels of weathermentations' optional native library.
 *
 * Plain C, called through ctypes with raw pointers: no Python or numpy
 * headers, so one binary serves every Python version on a platform. Every
 * kernel takes a whole image or a whole field of particles; none is called
 * per pixel or per particle, which is where the Python implementations they
 * replace spend their time.
 *
 * wm_kernels.c holds the kernels and is compiled once per instruction-set
 * tier (WM_TIER); wm_dispatch.c exports the entry points and chooses the
 * tier's table when the library is loaded. See docs/native_kernels.md.
 */
#ifndef WM_KERNELS_H
#define WM_KERNELS_H

#include <stddef.h>
#include <stdint.h>

#define WM_ABI_VERSION 3

typedef struct {
    const char *name;

    /* Linear light to sRGB code values through a 65536-entry table, as
     * photometry.from_linear_unit does it. Returns nonzero if a NaN was
     * seen, which the caller hands to the Python path to report. */
    int (*encode_u8)(const float *in, uint8_t *out, size_t n, const uint8_t *table);
    int (*encode_f32)(const float *in, float *out, size_t n, const float *table);
    /* sRGB code values to linear light through a 256-entry table. */
    void (*decode_u8)(const uint8_t *in, float *out, size_t n, const float *table);

    /* Max-blend a field of snowflake sprites into `canvas`
     * (snow._paste_flake_sprite, for every flake in one call). `gain` may be
     * NULL. With `fast`, divisions become multiplications by a reciprocal,
     * which moves a sprite by a few units in the last place. Returns nonzero
     * when it declines the field or cannot allocate its scratch space. */
    int (*render_flakes)(
        float *canvas, int height, int width, ptrdiff_t count,
        const float *x, const float *y, const float *radius, const float *blur_radius,
        const float *smear, const float *dir_x, const float *dir_y, const float *gain,
        int blades, double blade_rotation, double rim_gain, double cats_eye,
        double centre_x, double centre_y, int fast);

    /* Add a field of rain streaks to `layer` (rain.render_streak_layer):
     * each a polyline drawn as cv2.polylines draws it on a float image,
     * modulated by the beading table, blurred if its sigma is above 0.3, and
     * accumulated. `vertices` is (count, n_vertices, 2) of patch-local x, y,
     * which must lie inside the patch with room for the line's width.
     * `select` holds the indices to draw. Nonzero on allocation failure. */
    int (*render_streaks)(
        float *layer, int height, int width,
        const int32_t *select, ptrdiff_t n_select,
        const int32_t *x0, const int32_t *y0, const int32_t *patch_w, const int32_t *patch_h,
        const int32_t *vertices, int n_vertices, const int32_t *thickness,
        const float *alpha, const float *col_offset, const float *col_rate,
        const float *row_offset, const float *row_rate, const float *phase,
        const float *blur_sigma, const float *beading, int beading_size);

    /* The wet road, for the rows that hold any (rain._apply_ground_wetness
     * in fast mode): `dry`, `sharp`, `rough` and `out` are (rows, width, 3),
     * `wetness` and `puddles` (rows, width), `puddles` may be NULL, and
     * `fresnel` and `film_fresnel` hold one value a row. Nonzero if it could
     * not allocate a row of scratch. */
    int (*wet_composite)(
        float *out, const float *dry, const float *sharp, const float *rough,
        const float *wetness, const float *puddles, const float *fresnel,
        const float *film_fresnel, ptrdiff_t rows, ptrdiff_t width,
        float ratio, float spare, float film, float gloss);
} wm_kernel_table;

#endif
