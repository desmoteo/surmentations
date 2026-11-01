/** Kernels of weathermentations' optional native library.
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
/**/

#ifndef WM_KERNELS_H
#define WM_KERNELS_H

#include <stddef.h>
#include <stdint.h>

#define WM_CAT_(a, b) a##_##b
/** Pastes two names with an underscore, after expanding them: a kernel's
 * name and a tier's, or ``wm_table`` and a tier's.
 */
#define WM_CAT(a, b) WM_CAT_(a, b)

/** The version of the table's layout and the kernels' arguments.
 *
 * ``weathermentations._native`` refuses a library whose version is not its
 * own: a stale build is not called with the wrong arguments.
 */
#define WM_ABI_VERSION 11

/** One instruction-set tier's kernels: what ``wm_kernels.c`` exports, once per tier.
 *
 * Every tier computes the same floats; only how fast differs. The members
 * are documented one by one below.
 */
typedef struct {
    /** The tier's name: ``sse2``, ``avx2``, ``neon``, ``portable`` or ``native``. */
    const char *name;

    /** Linear light to sRGB code values through a 65536-entry table, as
     * photometry.from_linear_unit does it. Returns nonzero if a NaN was
     * seen, which the caller hands to the Python path to report.
     */
    int (*encode_u8)(const float *in, uint8_t *out, size_t n, const uint8_t *table);
    /** As ``encode_u8``, through a float32 table: sRGB values in ``[0, 1]``. */
    int (*encode_f32)(const float *in, float *out, size_t n, const float *table);
    /** sRGB code values to linear light through a 256-entry table. */
    void (*decode_u8)(const uint8_t *in, float *out, size_t n, const float *table);

    /** Max-blend a field of snowflake sprites into `canvas`
     * (snow._paste_flake_sprite, for every flake in one call). `gain` may be
     * NULL. With `fast`, divisions become multiplications by a reciprocal,
     * which moves a sprite by a few units in the last place. Returns nonzero
     * when it declines the field or cannot allocate its scratch space.
     */
    int (*render_flakes)(
        float *canvas, int height, int width, ptrdiff_t count,
        const float *x, const float *y, const float *radius, const float *blur_radius,
        const float *smear, const float *dir_x, const float *dir_y, const float *gain,
        int blades, double blade_rotation, double rim_gain, double cats_eye,
        double centre_x, double centre_y, int fast);

    /** Add a field of rain streaks to `layer` (rain.render_streak_layer):
     * each a polyline drawn as cv2.polylines draws it on a float image,
     * modulated by the beading table, blurred if its sigma is above 0.3, and
     * accumulated. `vertices` is (count, n_vertices, 2) of patch-local x, y,
     * which must lie inside the patch with room for the line's width.
     * `select` holds the indices to draw. Nonzero on allocation failure.
     */
    int (*render_streaks)(
        float *layer, int height, int width,
        const int32_t *select, ptrdiff_t n_select,
        const int32_t *x0, const int32_t *y0, const int32_t *patch_w, const int32_t *patch_h,
        const int32_t *vertices, int n_vertices, const int32_t *thickness,
        const float *alpha, const float *col_offset, const float *col_rate,
        const float *row_offset, const float *row_rate, const float *phase,
        const float *blur_sigma, const float *beading, int beading_size);

    /** The wet road, for the rows that hold any (rain._apply_ground_wetness
     * in fast mode): `dry`, `sharp`, `rough` and `out` are (rows, width, 3),
     * `wetness` and `puddles` (rows, width), `puddles` may be NULL, and
     * `fresnel` and `film_fresnel` hold one value a row. Nonzero if it could
     * not allocate a row of scratch.
     */
    int (*wet_composite)(
        float *out, const float *dry, const float *sharp, const float *rough,
        const float *wetness, const float *puddles, const float *fresnel,
        const float *film_fresnel, ptrdiff_t rows, ptrdiff_t width,
        float ratio, float spare, float film, float gloss);

    /** rain._mirrored_rows: from `solid` (height x width of 0 or 1), the row
     * each pixel is mirrored about the foot of what stands above it and
     * whether it is in reach of that. Nonzero if it could not allocate.
     */
    int (*mirror_rows)(
        const uint8_t *solid, ptrdiff_t height, ptrdiff_t width, double limit,
        float *about_foot, float *in_reach);

    /** The wet road's mirror image for `rows` rows: `source` is the whole
     * frame, (height, width, 3); `about_horizon` and `off_frame` hold one
     * value a row, `about_foot` and `under` (which may be NULL) one a
     * pixel, `sky` three.
     */
    void (*wet_reflection)(
        float *out, const float *source, ptrdiff_t height, ptrdiff_t width, ptrdiff_t rows,
        const float *about_horizon, const float *off_frame, const float *about_foot,
        const float *under, const float *sky);

    /** cv2.edgePreservingFilter(src, flags=RECURS_FILTER, sigma_s, sigma_r)
     * for an (height, width, 3) uint8 image. With `tabled`, the filter's
     * power function is taken from a table: faster, and a level off in a few
     * pixels. Nonzero if it could not allocate.
     */
    int (*edge_preserving)(
        const uint8_t *src, uint8_t *dst, ptrdiff_t height, ptrdiff_t width,
        float sigma_s, float sigma_r, int tabled);

    /** DirtSoil before its scatter blur: the scene with its highlights
     * weighted up, ``linear * (1 + boost * clip((luma - 0.35) / 0.65, 0, 1))``,
     * in one pass with numpy's arithmetic. `linear` and `out` are
     * (pixels, 3), `luma` one value a pixel.
     */
    void (*dirt_emphasise)(
        float *out, const float *linear, const float *luma, ptrdiff_t pixels, float boost);

    /** DirtSoil after its scatter blur: the frame lerped towards the light
     * the deposit scatters in, then towards its crust, in one pass with
     * numpy's arithmetic. `linear`, `out`, `spread` and `albedo` are
     * (pixels, 3); `mask` one value a pixel; `surroundings` three.
     */
    void (*dirt_composite)(
        float *out, const float *linear, const float *spread, const float *mask,
        const float *albedo, const float *surroundings, ptrdiff_t pixels,
        float forward, float veil_strength, float absorption, float body);

    /** A frame seen through a veil, and then through a layer of particles:
     * out = linear * t + airlight * (1 - t), and with `alpha` not NULL
     * out = out * (1 - alpha) + light * alpha, as RandomRain and RandomSnow
     * write them. `transmission` and `alpha` hold one value a pixel,
     * `airlight` three; the particles' light is the three values of `light`
     * or, with `light_map` not NULL, a value for every one of the frame's.
     */
    void (*veil_blend)(
        float *out, const float *linear, const float *transmission, const float *airlight,
        const float *alpha, const float *light, const float *light_map, ptrdiff_t pixels);

    /** overcast.apply_overcast in fast mode, after its statistics: each
     * pixel scaled by `gain`, drawn towards its grey, covered by cloud where
     * `sky` (which may be NULL) says, and clipped to [0, 1].
     */
    void (*overcast_blend)(
        float *out, const float *linear, const float *gain, const float *grey,
        const float *sky, const float *cloud, ptrdiff_t pixels,
        float desaturate, float cover_scale, float mean_grey);

    /** A mask grown over the pixels that look like it: for each pixel,
     * whether it is the mask's colour where the mask's own colour near it is
     * `nearby_sum` over `weight`, and the mask redrawn from that and the mask
     * `grown`. With `depth`, a pixel far from the mask's depth `level` is
     * also let go (sky.sky_to_depth_edge); without it, it is not
     * (ground.fill_to_edges). `linear` and `nearby_sum` are (pixels, 3),
     * the rest one value a pixel.
     */
    void (*grow_alike)(
        float *out, const float *linear, const float *nearby_sum, const float *weight,
        const float *mask, const float *grown, const float *depth, ptrdiff_t pixels,
        float level, float depth_tolerance, float colour_tolerance, float least_weight,
        float some_weight);

    /** water_droplets._sample_positions' acceptance: candidates in `order`
     * are kept while there are fewer than `count`, if no kept one is nearer
     * than (r_kept + r) * (1 + gap). Writes the kept candidates' indices to
     * `kept` and returns how many; the arithmetic is the numpy code's, in
     * double.
     */
    ptrdiff_t (*place_apart)(
        const double *x, const double *y, const double *footprint, const int64_t *order,
        ptrdiff_t candidates, ptrdiff_t count, double gap, int64_t *kept);

    /** water_droplets._water_surface's lobes, in fast mode: each drop's level
     * and thickness added into the two (height, width) fields. `lobes` is
     * (count, 7) and `harmonics` (count, n_harmonics, 2) or NULL, as there;
     * `orders` holds each harmonic's order. The angle's sines come from a
     * recurrence, not from atan2 and sin, so the fields agree with numpy's
     * to float rounding.
     */
    void (*water_lobes)(
        float *level, float *weighted, ptrdiff_t height, ptrdiff_t width, const double *lobes,
        const double *harmonics, ptrdiff_t count, int n_harmonics, const int32_t *orders,
        double wobble_radius, double outline_level);

    /** The end of WaterDroplets.apply: `base` blended towards `scattered` by
     * `fraction` (both may be NULL), then the drops' `layers`, (pixels, 4)
     * premultiplied colour and coverage, laid over it (may be NULL). The
     * numpy code's arithmetic in its order.
     */
    void (*droplet_composite)(
        float *out, const float *base, const float *scattered, const float *fraction,
        const float *layers, ptrdiff_t pixels);
} wm_kernel_table;

#endif
