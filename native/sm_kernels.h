/** Kernels of surmentations' optional native library.
 *
 * Plain C, called through ctypes with raw pointers: no Python or numpy
 * headers, so one binary serves every Python version on a platform. Every
 * kernel takes a whole image or a whole field of particles; none is called
 * per pixel or per particle, which is where the Python implementations they
 * replace spend their time.
 *
 * sm_kernels.c holds the kernels and is compiled once per instruction-set
 * tier (SM_TIER); sm_dispatch.c exports the entry points and chooses the
 * tier's table when the library is loaded. See docs/native_kernels.md.
 */
/**/

#ifndef SM_KERNELS_H
#define SM_KERNELS_H

#include <stddef.h>
#include <stdint.h>

#define SM_CAT_(a, b) a##_##b
/** Pastes two names with an underscore, after expanding them: a kernel's
 * name and a tier's, or ``sm_table`` and a tier's.
 */
#define SM_CAT(a, b) SM_CAT_(a, b)

/** The version of the table's layout and the kernels' arguments.
 *
 * ``surmentations._native`` refuses a library whose version is not its
 * own: a stale build is not called with the wrong arguments.
 */
#define SM_ABI_VERSION 14

/** One instruction-set tier's kernels: what ``sm_kernels.c`` exports, once per tier.
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

    /** illuminator.splat_discs: `count` discs of light, taken in `order`,
     * added into `total`, (height, width) doubles. Disc i is centred on
     * (x[i], y[i]) with radius[i] and an edge a pixel wide, and holds
     * weight[i] in all. The numpy code's arithmetic, float for a pixel's
     * share and double for the sum, in its order.
     */
    void (*splat_discs)(
        double *total, ptrdiff_t height, ptrdiff_t width,
        const float *x, const float *y, const float *radius, const float *weight,
        const int64_t *order, ptrdiff_t count);

    /** night_light.street_air: the street lamps' light on the air along
     * each of `pixels` lines of sight, added into `air`, and that times how
     * far along the line it is into `reach`. A line is its unit direction
     * (`rx`, `ry`, `rz`), its length `far` and its direction against `up`,
     * `rising`; `lamps` is (count, 4), a head's place and its height.
     * `nearest2` is the square of the nearest a line is taken to pass a
     * lamp, `spill` what a lamp sends upwards and `forward` the asymmetry
     * of the air's scattering. The numpy code's float arithmetic in its
     * order.
     */
    void (*street_air)(
        float *air, float *reach, const float *rx, const float *ry, const float *rz,
        const float *far, const float *rising, ptrdiff_t pixels, const float *lamps,
        ptrdiff_t count, const float *up, float nearest2, float spill, float forward);

    /** road._add_puff for a whole table: `count` puffs of spray, each a
     * Gaussian with its own axes and radii, added into `spray` less what
     * hides them. `puffs` is (count, 15) doubles, a row as
     * road._PUFF_COLUMNS has it; `stands`, `hidden` and `depth` (which may
     * be NULL) are (height, width) like `spray`. numpy's float arithmetic
     * with reciprocals for its divisions and a polynomial for its
     * exponential: within a few millionths of a puff's density.
     */
    void (*spray_puffs)(
        float *spray, const float *stands, const float *hidden, const float *depth,
        ptrdiff_t height, ptrdiff_t width, const double *puffs, ptrdiff_t count,
        float feather, float hides_wholly, float stands_nearer);

    /** normals._normals_and_jump after its blur: each pixel's unit surface
     * normal as the planes `nx`, `ny`, `nz`, and in `jump` the relative
     * step in depth it was read across. `depth` is (height, width), and
     * `u` (width) and `v` (height) each column's and row's line of sight.
     * Differences are `step` apart and one-sided; height and width must
     * be more than twice `step`. The numpy code's arithmetic to the bit.
     */
    void (*normals)(
        float *nx, float *ny, float *nz, float *jump, const float *depth, const float *u,
        const float *v, ptrdiff_t height, ptrdiff_t width, ptrdiff_t step);

    /** night_ir.near_infrared_reflectance: a broad red-weighted `response`
     * of `linear`, (pixels, 3), lifted by `gain` where the colour says
     * leaf or `vegetation` (which may be NULL) does, by `labelled` and by
     * the pixel's brightness against `usual`, the frame's mean response.
     */
    void (*nir_reflectance)(
        float *out, const float *linear, const float *vegetation, ptrdiff_t pixels,
        const float *response, float gain, float labelled, float usual);

    /** NightIrMode.apply before its exposure: `signal` is `reflectance`,
     * held up to `floor_scale` times `in_beam`, times the light arriving
     * (`exposure` and `ambient`, `street`, `beams`), less `sky_scale` of
     * the night `sky`; `lifted` is the reflectance so held. `in_beam`,
     * `street`, `beams` and `sky` may be NULL.
     */
    void (*night_ir_light)(
        float *signal, float *lifted, const float *reflectance, const float *exposure,
        const float *in_beam, const float *street, const float *beams, const float *sky,
        ptrdiff_t pixels, float ambient, float floor_scale, float sky_scale);

    /** NightIrMode.apply from its exposure on, into `out`, (pixels, 3):
     * the signal at the camera's level, what `retro` returns of `lifted`
     * and `exposure`, the `glare`, the lamps (the traffic's (pixels, 3),
     * the street's a plane each), the grain from `noise`, clipped and
     * tinted. Each of those but `signal` and `noise` may be NULL. `how` is
     * eight floats: level, bloom, shot, read noise, gain and the tint.
     */
    void (*night_ir_expose)(
        float *out, const float *signal, const float *retro, const float *lifted,
        const float *exposure, const float *glare, const float *vehicle_light,
        const float *vehicle_glow, const float *head_light, const float *head_glow,
        const float *noise, ptrdiff_t pixels, const float *how);

    /** NightColorMode.apply before its exposure: `signal`, (pixels, 3), is
     * each surface's colour times the light reaching it, and `brightness`
     * its luminance. With `evened` a surface is first held up to the
     * beam's floor and scaled to that lightness. `evened`, `in_beam`,
     * `street`, `beams` and `sky` may be NULL. `how` is fourteen floats:
     * the sky's glow, the street's colour and the headlamps', the floor's
     * scale, the sky's, and the luminance's weights.
     */
    void (*night_colour_light)(
        float *signal, float *brightness, const float *linear, const float *evened,
        const float *in_beam, const float *street, const float *beams, const float *sky,
        ptrdiff_t pixels, const float *how);

    /** NightColorMode.apply from its exposure on: `out`, `signal` and
     * `noise` are (pixels, 3); the traffic's lamps, (pixels, 3) each, and
     * the street's, a plane each and in the colour `street`, may be NULL.
     * `how` is five floats: level, bloom, shot, read noise and gain.
     */
    void (*night_colour_expose)(
        float *out, const float *signal, const float *vehicle_light,
        const float *vehicle_glow, const float *head_light, const float *head_glow,
        const float *noise, ptrdiff_t pixels, const float *street, const float *how);

    /** NightFog.apply before its halo: `radiance`, (pixels, 3), is
     * `linear` with its lamps (`clipped`, a plane) lifted by `over`;
     * `removed` is what the fog, of transmission `one_way`, takes of the
     * lamps, and `weight` that summed over the channels. The numpy code's
     * arithmetic to the bit.
     */
    void (*fog_lost)(
        float *radiance, float *removed, float *weight, const float *linear,
        const float *clipped, const float *one_way, ptrdiff_t pixels, float over);

    /** NightFog.apply after its halo, into `out`, (pixels, 3): `radiance`
     * through the fog, the `halo`, the lamps' `cones`, the `lit` volume
     * and the illuminator's `wall`, a plane; the last three may be NULL.
     * `how` is ten floats: the halo's share, whether the light crosses
     * the fog twice, the cones' level a channel, the veil's share, its
     * lit part and its diffuse part a channel.
     */
    void (*fog_veil)(
        float *out, const float *radiance, const float *halo, const float *one_way,
        const float *cones, const float *lit, const float *wall, ptrdiff_t pixels,
        const float *how);

    /** snow.RandomSnow's lying snow from the planes it is read from: the
     * `level` ground, the picture's `relief`, how `steady` the level
     * reads, how `green` it is, the labelled `lawn` and canopy (`named`),
     * the `patch` noise, the `grain`, the `luma` and its mean `around`,
     * and the `sky`. `lawn`, `named` and `patch` may be NULL. `how` is
     * seven floats: where relief begins and its span, the patchiness, the
     * patches' contrast, the strength, the grain's depth and a canopy's
     * cover. The numpy code's fast-mode arithmetic to the bit.
     */
    void (*snow_settled)(
        float *settled, const float *level, const float *relief, const float *steady,
        const float *green, const float *lawn, const float *named, const float *patch,
        const float *grain, const float *luma, const float *around, const float *sky,
        ptrdiff_t pixels, const float *how);

    /** ground.ground_noise's two maps into its noise tile, in fast mode:
     * each pixel's place on the ground across and along, from `depth`,
     * (height, width), in tile pixels and folded at the tile's edges.
     * `how` is eight floats: the focal length, the pitch's cosine and
     * sine, tile pixels a metre, the fold's span and its reciprocal, and
     * the two offsets. The numpy code's arithmetic to the bit.
     */
    void (*ground_tile)(
        float *tile_x, float *tile_y, const float *depth, ptrdiff_t height,
        ptrdiff_t width, const float *how);

    /** night_light.street_light: every street lamp's light on every point,
     * added into `light`. `points` is (pixels, 3), `raised` how high each
     * stands along `up`, and `lamps` (count, 4), a head's place and its
     * height; `spill` is what a lamp sends upwards. numpy's arithmetic to
     * the last bit of a float or one off it.
     */
    void (*street_light)(
        float *light, const float *points, const float *raised, ptrdiff_t pixels,
        const float *lamps, ptrdiff_t count, const float *up, float spill);

    /** night_light.headlamp_light's sum: every lit vehicle's two dipped
     * beams on every point, added into `light`. A point is its place along
     * the ground's two axes (`along_first`, `along_second`) and its height
     * `above`; `low` is nought well above any beam. `beams` is (count, 6)
     * floats: a vehicle's front along the two axes, its heading along
     * them, its scale and what the lens gets of its pool. `how` is seven:
     * a lamp's width and widening, half the lamps' spacing, each lamp's
     * level, where it halves, how far it reaches and how high. numpy's
     * arithmetic within a few millionths of a pool.
     */
    void (*headlamp_beams)(
        float *light, const float *along_first, const float *along_second,
        const float *above, const float *low, ptrdiff_t pixels, const float *beams,
        ptrdiff_t count, const float *how);

    /** road.wheel_tracks: the weight of the two wheel tracks down each
     * lane, into `out`. `piece` is each pixel's road, `across` its place
     * across it, `width_px` the road's width there and `weight` whether it
     * is road at all; `ragged` and `worn` may be NULL. `edges` holds every
     * road's lane lines one after another, road r's from `first_edge[r]`
     * to `first_edge[r + 1]`, for roads 1 to `roads`. `use` is how much
     * each of 64 lanes is driven in, and `how` five floats: the band, the
     * wander, half the wheels' spacing, the wear's floor and its gain. The
     * numpy code's arithmetic to the bit.
     */
    void (*wheel_tracks)(
        float *out, const int32_t *piece, const float *across, const float *width_px,
        const float *weight, const float *ragged, const float *worn, ptrdiff_t pixels,
        const double *edges, const int32_t *first_edge, int32_t roads, const float *use,
        const float *how);

    /** RandomRain.apply's lamps before its road: `out`, (pixels, 3), gains
     * `lamp_light` up to full scale through `through`, a plane. The numpy
     * code's arithmetic to the bit.
     */
    void (*lamps_lit)(
        float *out, const float *lamp_light, const float *through, ptrdiff_t pixels);

    /** RandomRain.apply between its road and its streaks, on `out`,
     * (pixels, 3): the `spray` laid over it in the light `lit`, three
     * values, then what of `lamp_light` is over full scale, through
     * `through` and the spray, and `lamp_glow` by how wet the air is,
     * from `glow_in_rain` up. `spray`, and the two lamp fields together,
     * may be NULL. The numpy code's arithmetic to the bit.
     */
    void (*spray_and_lamps)(
        float *out, const float *spray, const float *lamp_light, const float *lamp_glow,
        const float *through, ptrdiff_t pixels, const float *lit, float glow_in_rain);

    /** road.model_frame's `along`: the way a road runs at every pixel of a
     * (height, width) frame, a unit vector into `along`, (height, width,
     * 2), for a road that runs to `centre` on a horizon at row `horizon`
     * with a `bend`. The numpy code's double arithmetic to the bit.
     */
    void (*road_along)(
        float *along, ptrdiff_t height, ptrdiff_t width, double centre, double horizon,
        double bend);

    /** road.model_frame's `piece`, `across` and width in pixels (`span`),
     * each (rows, width), for the rows a road can be on. `offset` is each
     * pixel's place across the road and `at` the position it is counted
     * as; `of_position` gives a position's carriageway, 0 being none, and
     * `firsts` and `widths` each carriageway's first place and its width.
     * `whole` is 1 where the pixel is road, `below` each row's distance
     * below the horizon. The numpy code's arithmetic to the bit.
     */
    void (*road_places)(
        int32_t *piece, float *across, float *span, const double *offset, const int64_t *at,
        const uint8_t *whole, const double *below, ptrdiff_t rows, ptrdiff_t width,
        const int32_t *of_position, const double *firsts, const double *widths);
} sm_kernel_table;

#endif
