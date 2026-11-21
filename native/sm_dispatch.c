/** Entry points of the native library, and the choice of tier.
 *
 * sm_kernels.c is compiled once per tier; each compilation exports one
 * table. On x86-64 there are two, the psABI baseline (SSE2) and AVX2, and
 * the best one the CPU and the operating system support is chosen when the
 * library is loaded. Elsewhere there is one. A build may add a "native"
 * tier, compiled for the machine it was built on. Every tier gives the same
 * floats, so which one runs is a matter of speed only.
 */
/**/

#include <string.h>

#include "sm_kernels.h"

/** Makes a function visible outside the library; everything else is hidden. */
#define SM_EXPORT __attribute__((visibility("default")))

#ifndef SM_BASE
/** The baseline tier, which every CPU of the architecture runs; set by the build. */
#define SM_BASE portable
#endif
/** The baseline tier's table. */
#define SM_BASE_TABLE SM_CAT(sm_table, SM_BASE)

extern const sm_kernel_table SM_BASE_TABLE;
#ifdef SM_HAVE_AVX2
extern const sm_kernel_table sm_table_avx2;
#endif
#ifdef SM_HAVE_NATIVE
extern const sm_kernel_table sm_table_native;
/** Defined with the native tier: whether this CPU has what it was built for. */
extern int sm_native_usable(void);
#endif

/** A tier every CPU of the architecture can run. */
static int always(void) { return 1; }

#ifdef SM_HAVE_AVX2
/** Whether this CPU and its operating system support AVX2. */
static int cpu_has_avx2(void) {
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2");
}
#endif

/** The tiers of this build, slowest first. */
static const struct {
    const sm_kernel_table *table;
    int (*usable)(void);
} tiers[] = {
    {&SM_BASE_TABLE, always},
#ifdef SM_HAVE_AVX2
    {&sm_table_avx2, cpu_has_avx2},
#endif
#ifdef SM_HAVE_NATIVE
    {&sm_table_native, sm_native_usable},
#endif
};
enum { N_TIERS = sizeof(tiers) / sizeof(tiers[0]) };

/** The table of the tier in use; the best the CPU supports until one is chosen. */
static const sm_kernel_table *active = &SM_BASE_TABLE;

/** The library's ``SM_ABI_VERSION``, which the Python loader checks against its own. */
SM_EXPORT int sm_abi_version(void) { return SM_ABI_VERSION; }

/** The tiers this build holds, best last, separated by commas. */
SM_EXPORT const char *sm_tiers(void) {
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

/** The name of the tier in use. */
SM_EXPORT const char *sm_tier(void) { return active->name; }

/** Choose a tier by name, or the best the CPU has with "auto". Returns 0 when
 * the tier is now active, 1 when this build or this CPU does not have it.
 */
SM_EXPORT int sm_select(const char *name) {
    const int best = strcmp(name, "auto") == 0;
    for (int i = N_TIERS - 1; i >= 0; i--) {
        if ((best || strcmp(name, tiers[i].table->name) == 0) && tiers[i].usable()) {
            active = tiers[i].table;
            return 0;
        }
    }
    return 1;
}

/** The active tier's ``encode_u8``; see ``sm_kernel_table``. */
SM_EXPORT int sm_encode_u8(const float *in, uint8_t *out, size_t n, const uint8_t *table) {
    return active->encode_u8(in, out, n, table);
}

/** The active tier's ``encode_f32``; see ``sm_kernel_table``. */
SM_EXPORT int sm_encode_f32(const float *in, float *out, size_t n, const float *table) {
    return active->encode_f32(in, out, n, table);
}

/** The active tier's ``decode_u8``; see ``sm_kernel_table``. */
SM_EXPORT void sm_decode_u8(const uint8_t *in, float *out, size_t n, const float *table) {
    active->decode_u8(in, out, n, table);
}

/** The active tier's ``render_flakes``; see ``sm_kernel_table``. */
SM_EXPORT int sm_render_flakes(
    float *canvas, int height, int width, ptrdiff_t count,
    const float *x, const float *y, const float *radius, const float *blur_radius,
    const float *smear, const float *dir_x, const float *dir_y, const float *gain,
    int blades, double blade_rotation, double rim_gain, double cats_eye,
    double centre_x, double centre_y, int fast) {
    return active->render_flakes(
        canvas, height, width, count, x, y, radius, blur_radius, smear, dir_x, dir_y, gain,
        blades, blade_rotation, rim_gain, cats_eye, centre_x, centre_y, fast);
}

/** The active tier's ``render_streaks``; see ``sm_kernel_table``. */
SM_EXPORT int sm_render_streaks(
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

/** The active tier's ``wet_composite``; see ``sm_kernel_table``. */
SM_EXPORT int sm_wet_composite(
    float *out, const float *dry, const float *sharp, const float *rough,
    const float *wetness, const float *puddles, const float *fresnel,
    const float *film_fresnel, ptrdiff_t rows, ptrdiff_t width,
    float ratio, float spare, float film, float gloss) {
    return active->wet_composite(
        out, dry, sharp, rough, wetness, puddles, fresnel, film_fresnel, rows, width, ratio,
        spare, film, gloss);
}

/** The active tier's ``mirror_rows``; see ``sm_kernel_table``. */
SM_EXPORT int sm_mirror_rows(
    const uint8_t *solid, ptrdiff_t height, ptrdiff_t width, double limit,
    float *about_foot, float *in_reach) {
    return active->mirror_rows(solid, height, width, limit, about_foot, in_reach);
}

/** The active tier's ``wet_reflection``; see ``sm_kernel_table``. */
SM_EXPORT void sm_wet_reflection(
    float *out, const float *source, ptrdiff_t height, ptrdiff_t width, ptrdiff_t rows,
    const float *about_horizon, const float *off_frame, const float *about_foot,
    const float *under, const float *sky) {
    active->wet_reflection(
        out, source, height, width, rows, about_horizon, off_frame, about_foot, under, sky);
}

/** The active tier's ``edge_preserving``; see ``sm_kernel_table``. */
SM_EXPORT int sm_edge_preserving(
    const uint8_t *src, uint8_t *dst, ptrdiff_t height, ptrdiff_t width,
    float sigma_s, float sigma_r, int tabled) {
    return active->edge_preserving(src, dst, height, width, sigma_s, sigma_r, tabled);
}

/** The active tier's ``dirt_emphasise``; see ``sm_kernel_table``. */
SM_EXPORT void sm_dirt_emphasise(
    float *out, const float *linear, const float *luma, ptrdiff_t pixels, float boost) {
    active->dirt_emphasise(out, linear, luma, pixels, boost);
}

/** The active tier's ``dirt_composite``; see ``sm_kernel_table``. */
SM_EXPORT void sm_dirt_composite(
    float *out, const float *linear, const float *spread, const float *mask,
    const float *albedo, const float *surroundings, ptrdiff_t pixels,
    float forward, float veil_strength, float absorption, float body) {
    active->dirt_composite(
        out, linear, spread, mask, albedo, surroundings, pixels, forward, veil_strength,
        absorption, body);
}

/** The active tier's ``veil_blend``; see ``sm_kernel_table``. */
SM_EXPORT void sm_veil_blend(
    float *out, const float *linear, const float *transmission, const float *airlight,
    const float *alpha, const float *light, const float *light_map, ptrdiff_t pixels) {
    active->veil_blend(out, linear, transmission, airlight, alpha, light, light_map, pixels);
}

/** The active tier's ``overcast_blend``; see ``sm_kernel_table``. */
SM_EXPORT void sm_overcast_blend(
    float *out, const float *linear, const float *gain, const float *grey,
    const float *sky, const float *cloud, ptrdiff_t pixels,
    float desaturate, float cover_scale, float mean_grey) {
    active->overcast_blend(
        out, linear, gain, grey, sky, cloud, pixels, desaturate, cover_scale, mean_grey);
}

/** The active tier's ``grow_alike``; see ``sm_kernel_table``. */
SM_EXPORT void sm_grow_alike(
    float *out, const float *linear, const float *nearby_sum, const float *weight,
    const float *mask, const float *grown, const float *depth, ptrdiff_t pixels,
    float level, float depth_tolerance, float colour_tolerance, float least_weight,
    float some_weight) {
    active->grow_alike(
        out, linear, nearby_sum, weight, mask, grown, depth, pixels, level, depth_tolerance,
        colour_tolerance, least_weight, some_weight);
}

/** The active tier's ``place_apart``; see ``sm_kernel_table``. */
SM_EXPORT ptrdiff_t sm_place_apart(
    const double *x, const double *y, const double *footprint, const int64_t *order,
    ptrdiff_t candidates, ptrdiff_t count, double gap, int64_t *kept) {
    return active->place_apart(x, y, footprint, order, candidates, count, gap, kept);
}

/** The active tier's ``water_lobes``; see ``sm_kernel_table``. */
SM_EXPORT void sm_water_lobes(
    float *level, float *weighted, ptrdiff_t height, ptrdiff_t width, const double *lobes,
    const double *harmonics, ptrdiff_t count, int n_harmonics, const int32_t *orders,
    double wobble_radius, double outline_level) {
    active->water_lobes(
        level, weighted, height, width, lobes, harmonics, count, n_harmonics, orders,
        wobble_radius, outline_level);
}

/** The active tier's ``droplet_composite``; see ``sm_kernel_table``. */
SM_EXPORT void sm_droplet_composite(
    float *out, const float *base, const float *scattered, const float *fraction,
    const float *layers, ptrdiff_t pixels) {
    active->droplet_composite(out, base, scattered, fraction, layers, pixels);
}

/** The active tier's ``splat_discs``; see ``sm_kernel_table``. */
SM_EXPORT void sm_splat_discs(
    double *total, ptrdiff_t height, ptrdiff_t width,
    const float *x, const float *y, const float *radius, const float *weight,
    const int64_t *order, ptrdiff_t count) {
    active->splat_discs(total, height, width, x, y, radius, weight, order, count);
}

/** The active tier's ``street_air``; see ``sm_kernel_table``. */
SM_EXPORT void sm_street_air(
    float *air, float *reach, const float *rx, const float *ry, const float *rz,
    const float *far, const float *rising, ptrdiff_t pixels, const float *lamps,
    ptrdiff_t count, const float *up, float nearest2, float spill, float forward) {
    active->street_air(
        air, reach, rx, ry, rz, far, rising, pixels, lamps, count, up, nearest2, spill, forward);
}

/** The active tier's ``spray_puffs``; see ``sm_kernel_table``. */
SM_EXPORT void sm_spray_puffs(
    float *spray, const float *stands, const float *hidden, const float *depth,
    ptrdiff_t height, ptrdiff_t width, const double *puffs, ptrdiff_t count,
    float feather, float hides_wholly, float stands_nearer) {
    active->spray_puffs(
        spray, stands, hidden, depth, height, width, puffs, count, feather, hides_wholly,
        stands_nearer);
}

/** The active tier's ``normals``; see ``sm_kernel_table``. */
SM_EXPORT void sm_normals(
    float *nx, float *ny, float *nz, float *jump, const float *depth, const float *u,
    const float *v, ptrdiff_t height, ptrdiff_t width, ptrdiff_t step) {
    active->normals(nx, ny, nz, jump, depth, u, v, height, width, step);
}

/** The active tier's ``nir_reflectance``; see ``sm_kernel_table``. */
SM_EXPORT void sm_nir_reflectance(
    float *out, const float *linear, const float *vegetation, ptrdiff_t pixels,
    const float *response, float gain, float labelled, float usual) {
    active->nir_reflectance(out, linear, vegetation, pixels, response, gain, labelled, usual);
}

/** The active tier's ``night_ir_light``; see ``sm_kernel_table``. */
SM_EXPORT void sm_night_ir_light(
    float *signal, float *lifted, const float *reflectance, const float *exposure,
    const float *in_beam, const float *street, const float *beams, const float *sky,
    ptrdiff_t pixels, float ambient, float floor_scale, float sky_scale) {
    active->night_ir_light(
        signal, lifted, reflectance, exposure, in_beam, street, beams, sky, pixels, ambient,
        floor_scale, sky_scale);
}

/** The active tier's ``night_ir_expose``; see ``sm_kernel_table``. */
SM_EXPORT void sm_night_ir_expose(
    float *out, const float *signal, const float *retro, const float *lifted,
    const float *exposure, const float *glare, const float *vehicle_light,
    const float *vehicle_glow, const float *head_light, const float *head_glow,
    const float *noise, ptrdiff_t pixels, const float *how) {
    active->night_ir_expose(
        out, signal, retro, lifted, exposure, glare, vehicle_light, vehicle_glow, head_light,
        head_glow, noise, pixels, how);
}

/** The active tier's ``night_colour_light``; see ``sm_kernel_table``. */
SM_EXPORT void sm_night_colour_light(
    float *signal, float *brightness, const float *linear, const float *evened,
    const float *in_beam, const float *street, const float *beams, const float *sky,
    ptrdiff_t pixels, const float *how) {
    active->night_colour_light(
        signal, brightness, linear, evened, in_beam, street, beams, sky, pixels, how);
}

/** The active tier's ``night_colour_expose``; see ``sm_kernel_table``. */
SM_EXPORT void sm_night_colour_expose(
    float *out, const float *signal, const float *vehicle_light, const float *vehicle_glow,
    const float *head_light, const float *head_glow, const float *noise, ptrdiff_t pixels,
    const float *street, const float *how) {
    active->night_colour_expose(
        out, signal, vehicle_light, vehicle_glow, head_light, head_glow, noise, pixels, street,
        how);
}

/** The active tier's ``fog_lost``; see ``sm_kernel_table``. */
SM_EXPORT void sm_fog_lost(
    float *radiance, float *removed, float *weight, const float *linear,
    const float *clipped, const float *one_way, ptrdiff_t pixels, float over) {
    active->fog_lost(radiance, removed, weight, linear, clipped, one_way, pixels, over);
}

/** The active tier's ``fog_veil``; see ``sm_kernel_table``. */
SM_EXPORT void sm_fog_veil(
    float *out, const float *radiance, const float *halo, const float *one_way,
    const float *cones, const float *lit, const float *wall, ptrdiff_t pixels,
    const float *how) {
    active->fog_veil(out, radiance, halo, one_way, cones, lit, wall, pixels, how);
}

/** The active tier's ``snow_settled``; see ``sm_kernel_table``. */
SM_EXPORT void sm_snow_settled(
    float *settled, const float *level, const float *relief, const float *steady,
    const float *green, const float *lawn, const float *named, const float *patch,
    const float *grain, const float *luma, const float *around, const float *sky,
    ptrdiff_t pixels, const float *how) {
    active->snow_settled(
        settled, level, relief, steady, green, lawn, named, patch, grain, luma, around, sky,
        pixels, how);
}

/** The active tier's ``ground_tile``; see ``sm_kernel_table``. */
SM_EXPORT void sm_ground_tile(
    float *tile_x, float *tile_y, const float *depth, ptrdiff_t height, ptrdiff_t width,
    const float *how) {
    active->ground_tile(tile_x, tile_y, depth, height, width, how);
}

/** The active tier's ``street_light``; see ``sm_kernel_table``. */
SM_EXPORT void sm_street_light(
    float *light, const float *points, const float *raised, ptrdiff_t pixels,
    const float *lamps, ptrdiff_t count, const float *up, float spill) {
    active->street_light(light, points, raised, pixels, lamps, count, up, spill);
}

/** The active tier's ``headlamp_beams``; see ``sm_kernel_table``. */
SM_EXPORT void sm_headlamp_beams(
    float *light, const float *along_first, const float *along_second, const float *above,
    const float *low, ptrdiff_t pixels, const float *beams, ptrdiff_t count, const float *how) {
    active->headlamp_beams(
        light, along_first, along_second, above, low, pixels, beams, count, how);
}

/** The active tier's ``wheel_tracks``; see ``sm_kernel_table``. */
SM_EXPORT void sm_wheel_tracks(
    float *out, const int32_t *piece, const float *across, const float *width_px,
    const float *weight, const float *ragged, const float *worn, ptrdiff_t pixels,
    const double *edges, const int32_t *first_edge, int32_t roads, const float *use,
    const float *how) {
    active->wheel_tracks(
        out, piece, across, width_px, weight, ragged, worn, pixels, edges, first_edge, roads,
        use, how);
}

/** The active tier's ``lamps_lit``; see ``sm_kernel_table``. */
SM_EXPORT void sm_lamps_lit(
    float *out, const float *lamp_light, const float *through, ptrdiff_t pixels) {
    active->lamps_lit(out, lamp_light, through, pixels);
}

/** The active tier's ``spray_and_lamps``; see ``sm_kernel_table``. */
SM_EXPORT void sm_spray_and_lamps(
    float *out, const float *spray, const float *lamp_light, const float *lamp_glow,
    const float *through, ptrdiff_t pixels, const float *lit, float glow_in_rain) {
    active->spray_and_lamps(
        out, spray, lamp_light, lamp_glow, through, pixels, lit, glow_in_rain);
}

/** The active tier's ``road_along``; see ``sm_kernel_table``. */
SM_EXPORT void sm_road_along(
    float *along, ptrdiff_t height, ptrdiff_t width, double centre, double horizon,
    double bend) {
    active->road_along(along, height, width, centre, horizon, bend);
}

/** The active tier's ``road_places``; see ``sm_kernel_table``. */
SM_EXPORT void sm_road_places(
    int32_t *piece, float *across, float *span, const double *offset, const int64_t *at,
    const uint8_t *whole, const double *below, ptrdiff_t rows, ptrdiff_t width,
    const int32_t *of_position, const double *firsts, const double *widths) {
    active->road_places(
        piece, across, span, offset, at, whole, below, rows, width, of_position, firsts, widths);
}
