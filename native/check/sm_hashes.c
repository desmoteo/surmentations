/** Hashes of what each kernel computes on fixed inputs.
 *
 * The Python tests hold the kernels to the numpy code on the machine they
 * run on. This holds one machine to another: built for any tier on any
 * architecture, the program must print the same twenty-four lines. It needs only a
 * C compiler, so it runs where the Python stack is slow or missing, an
 * arm64 container under QEMU for one. run.sh builds it for every tier of
 * the machine it is on.
 */
/**/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sm_kernels.h"

extern const sm_kernel_table SM_TABLE;

/** The generator's state: fixed, so every machine draws the same inputs. */
static uint32_t state = 12345u;

/** A uniform value in ``[0, 1)`` from a linear congruential generator: the same on every machine.
 */
static float uniform(void) {
    state = state * 1664525u + 1013904223u;
    return (float)(state >> 8) / 16777216.0f;
}

/** ``v`` clamped to ``[0, 1]``. */
static float clamp_unit(float v) { return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v); }

/** The 64-bit FNV-1a hash of ``bytes`` bytes: a kernel's output, as one number to compare. */
static uint64_t fnv(const void *data, size_t bytes) {
    const unsigned char *p = data;
    uint64_t hash = 1469598103934665603ull;
    for (size_t i = 0; i < bytes; i++) {
        hash = (hash ^ p[i]) * 1099511628211ull;
    }
    return hash;
}

/** Run every kernel of this tier on fixed inputs and print one hash per kernel.
 *
 * Returns nonzero if a kernel declines its input or cannot allocate.
 */
int main(void) {
    enum { W = 320, H = 240, N = W * H * 3, FLAKES = 600, STREAKS = 3000, VERTICES = 7 };

    /* Colour. The tables' contents do not matter, only that they are fixed. */
    float *linear = malloc(N * sizeof(float));
    uint8_t *codes = malloc(N), *table8 = malloc(65536);
    float *decoded = malloc(N * sizeof(float)), *table32 = malloc(65536 * sizeof(float));
    for (int i = 0; i < N; i++) {
        linear[i] = uniform() * 1.6f - 0.3f;
    }
    for (int i = 0; i < 65536; i++) {
        table8[i] = (uint8_t)((i * 7919u) >> 8);
        table32[i] = (float)i / 65535.0f;
    }
    SM_TABLE.encode_u8(linear, codes, N, table8);
    SM_TABLE.decode_u8(codes, decoded, N, table32);
    uint64_t colour = fnv(codes, N) ^ fnv(decoded, N * sizeof(float));
    SM_TABLE.encode_f32(linear, decoded, N, table32);
    colour ^= fnv(decoded, N * sizeof(float)) << 1;
    printf("%-9s colour   %016llx\n", SM_TABLE.name, (unsigned long long)colour);

    /* Flakes: a round aperture, then an iris with a cat's-eye stop. */
    float *f[8];
    for (int k = 0; k < 8; k++) {
        f[k] = malloc(FLAKES * sizeof(float));
    }
    for (int i = 0; i < FLAKES; i++) {
        const float big = uniform();
        f[0][i] = uniform() * (W + 60) - 30;
        f[1][i] = uniform() * (H + 60) - 30;
        f[2][i] = 0.3f + 3.7f * uniform();
        f[3][i] = big * big * big * big * 70.0f;
        f[4][i] = 12.0f * uniform();
        /* A direction from a table of exact values: no libm in the inputs. */
        static const float dirs[4][2] = {{0.0f, 1.0f}, {0.6f, 0.8f}, {-0.8f, 0.6f}, {1.0f, 0.0f}};
        f[5][i] = dirs[i & 3][0];
        f[6][i] = dirs[i & 3][1];
        f[7][i] = 0.2f + 1.8f * uniform();
    }
    float *canvas = calloc(W * H, sizeof(float));
    SM_TABLE.render_flakes(
        canvas, H, W, FLAKES, f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7],
        0, 0.0, 1.0, 0.0, W / 2.0, H / 2.0, 0);
    printf("%-9s round    %016llx\n", SM_TABLE.name,
           (unsigned long long)fnv(canvas, W * H * sizeof(float)));
    memset(canvas, 0, W * H * sizeof(float));
    SM_TABLE.render_flakes(
        canvas, H, W, FLAKES, f[0], f[1], f[2], f[3], f[4], f[5], f[6], NULL,
        6, 0.3, 1.4, 0.5, W / 2.0, H / 2.0, 0);
    printf("%-9s iris     %016llx\n", SM_TABLE.name,
           (unsigned long long)fnv(canvas, W * H * sizeof(float)));

    memset(canvas, 0, W * H * sizeof(float));
    SM_TABLE.render_flakes(
        canvas, H, W, FLAKES, f[0], f[1], f[2], f[3], f[4], f[5], f[6], NULL,
        6, 0.3, 1.4, 0.5, W / 2.0, H / 2.0, 1);
    printf("%-9s iris-fast %015llx\n", SM_TABLE.name,
           (unsigned long long)fnv(canvas, W * H * sizeof(float)) >> 4);

    /* Streaks: thin and wide, sharp and blurred, with room in every patch. */
    int32_t *select = malloc(STREAKS * sizeof(int32_t));
    int32_t *geometry[5];
    for (int k = 0; k < 5; k++) {
        geometry[k] = malloc(STREAKS * sizeof(int32_t));
    }
    int32_t *vertices = malloc(STREAKS * VERTICES * 2 * sizeof(int32_t));
    float *s[7];
    for (int k = 0; k < 7; k++) {
        s[k] = malloc(STREAKS * sizeof(float));
    }
    float beading[256];
    for (int i = 0; i < 256; i++) {
        beading[i] = 0.25f + 0.75f * uniform();
    }
    for (int i = 0; i < STREAKS; i++) {
        const int thickness = 1 + (int)(uniform() * 7.99f);
        const int room = thickness + 2;
        const int pw = 2 * room + 2 + (int)(uniform() * 40), ph = 2 * room + 2 + (int)(uniform() * 60);
        select[i] = i;
        geometry[0][i] = (int)(uniform() * (W + 40)) - 40;
        geometry[1][i] = (int)(uniform() * (H + 60)) - 60;
        geometry[2][i] = pw;
        geometry[3][i] = ph;
        geometry[4][i] = thickness;
        for (int k = 0; k < VERTICES; k++) {
            vertices[(i * VERTICES + k) * 2] = room + (int)(uniform() * (pw - 2 * room));
            vertices[(i * VERTICES + k) * 2 + 1] = room + (int)(uniform() * (ph - 2 * room));
        }
        s[0][i] = 0.01f + 0.2f * uniform();
        s[1][i] = uniform() - 0.5f;
        s[2][i] = 4.0f * (uniform() - 0.5f);
        s[3][i] = uniform() - 0.5f;
        s[4][i] = 6.0f * (uniform() - 0.3f);
        s[5][i] = 60.0f * uniform();
        s[6][i] = uniform() < 0.3f ? 0.3f + 2.0f * uniform() : 0.0f;
    }
    memset(canvas, 0, W * H * sizeof(float));
    SM_TABLE.render_streaks(
        canvas, H, W, select, STREAKS, geometry[0], geometry[1], geometry[2], geometry[3],
        vertices, VERTICES, geometry[4], s[0], s[1], s[2], s[3], s[4], s[5], s[6], beading, 256);
    printf("%-9s streaks  %016llx\n", SM_TABLE.name,
           (unsigned long long)fnv(canvas, W * H * sizeof(float)));

    /* The wet road, over the colour test's frame: a row's reflectance from a
     * short table, puddles from the beading values. */
    float *dry = malloc(N * sizeof(float)), *sharp = malloc(N * sizeof(float));
    float *road = malloc(N * sizeof(float)), *wet = malloc(W * H * sizeof(float));
    float *pools = malloc(W * H * sizeof(float)), flat[H], tilted[H];
    for (int i = 0; i < N; i++) {
        dry[i] = uniform() * 1.1f;
        sharp[i] = uniform();
    }
    for (int i = 0; i < W * H; i++) {
        wet[i] = uniform();
        pools[i] = beading[i & 255] > 0.8f ? uniform() : 0.0f;
    }
    for (int r = 0; r < H; r++) {
        flat[r] = 0.02f + 0.98f * uniform() * uniform();
        tilted[r] = 0.02f + 0.98f * uniform();
    }
    SM_TABLE.wet_composite(
        road, dry, sharp, linear, wet, pools, flat, tilted, H, W, 0.55f, 0.45f, 0.7f, 0.6f);
    printf("%-9s wet      %016llx\n", SM_TABLE.name, (unsigned long long)fnv(road, N * sizeof(float)));

    /* What each pixel mirrors, and the mirror image: solid things of several
     * heights on the road, rows mirrored at fractional heights. */
    uint8_t *solid = calloc(W * H, 1);
    float *about_foot = malloc(W * H * sizeof(float)), *reach = malloc(W * H * sizeof(float));
    float about_horizon[H], off_frame[H];
    const float sky[3] = {0.5f, 0.6f, 0.7f};
    for (int k = 0; k < 40; k++) {
        const int y0 = (int)(uniform() * (H - 4)), x0 = (int)(uniform() * (W - 4));
        const int tall = 2 + (int)(uniform() * 120), wide = 2 + (int)(uniform() * 60);
        for (int y = y0; y < y0 + tall && y < H; y++) {
            for (int x = x0; x < x0 + wide && x < W; x++) {
                solid[y * W + x] = 1;
            }
        }
    }
    SM_TABLE.mirror_rows(solid, H, W, 0.3 * H, about_foot, reach);
    for (int r = 0; r < H; r++) {
        about_horizon[r] = 2.0f * 71.3f - (float)r;
        off_frame[r] = clamp_unit((19.2f - about_horizon[r]) / 19.2f);
    }
    SM_TABLE.wet_reflection(road, dry, H, W, H, about_horizon, off_frame, about_foot, reach, sky);
    printf("%-9s mirror   %016llx\n", SM_TABLE.name,
           (unsigned long long)(fnv(road, N * sizeof(float)) ^ fnv(about_foot, W * H * sizeof(float))));

    /* The denoise filter, on a frame with flat areas, a gradient and noise,
     * with the power taken per pixel and from the table. Both call the
     * machine's powf, so these two lines also hold one C library to
     * another. */
    uint8_t *frame = malloc(N), *filtered = malloc(N);
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            const float block = (float)(((x / 40) * 53 + (y / 30) * 97) % 200);
            for (int c = 0; c < 3; c++) {
                frame[(y * W + x) * 3 + c] =
                    (uint8_t)(clamp_unit((block + 0.1f * (float)x + 30.0f * uniform()) / 255.0f) * 255.0f);
            }
        }
    }
    if (SM_TABLE.edge_preserving(frame, filtered, H, W, 25.0f, 0.2f, 0)) {
        return 1;
    }
    printf("%-9s denoise  %016llx\n", SM_TABLE.name, (unsigned long long)fnv(filtered, N));
    if (SM_TABLE.edge_preserving(frame, filtered, H, W, 25.0f, 0.2f, 1)) {
        return 1;
    }
    printf("%-9s denoise-tabled %010llx\n", SM_TABLE.name, (unsigned long long)fnv(filtered, N));

    /* Lens dirt: the highlight-weighted scene, then the deposit over it. */
    float *luma = malloc(W * H * sizeof(float)), *cover = malloc(W * H * sizeof(float));
    for (int i = 0; i < W * H; i++) {
        luma[i] = uniform() * 1.2f - 0.1f;
        cover[i] = uniform() * 1.5f - 0.2f;
    }
    SM_TABLE.dirt_emphasise(road, linear, luma, W * H, 3.7f);
    const uint64_t emphasised = fnv(road, N * sizeof(float));
    SM_TABLE.dirt_composite(road, linear, dry, cover, sharp, sky, W * H, 0.6f, 0.7f, 0.09f, 0.35f);
    printf("%-9s dirt     %016llx\n", SM_TABLE.name,
           (unsigned long long)(emphasised ^ fnv(road, N * sizeof(float))));

    /* A frame through a veil and a layer of particles, lit by one colour and
     * then by a field; and the overcast relighting, with and without sky. */
    for (int i = 0; i < W * H; i++) {
        luma[i] = uniform();
        cover[i] = uniform();
    }
    SM_TABLE.veil_blend(road, linear, luma, sky, cover, sky, NULL, W * H);
    uint64_t veiled = fnv(road, N * sizeof(float));
    SM_TABLE.veil_blend(road, linear, luma, sky, cover, NULL, dry, W * H);
    veiled ^= fnv(road, N * sizeof(float));
    SM_TABLE.veil_blend(road, linear, luma, sky, NULL, NULL, NULL, W * H);
    printf("%-9s veil     %016llx\n", SM_TABLE.name,
           (unsigned long long)(veiled ^ fnv(road, N * sizeof(float))));
    SM_TABLE.overcast_blend(road, linear, luma, cover, wet, sky, W * H, 0.3f, 0.6f, 0.41f);
    const uint64_t relit = fnv(road, N * sizeof(float));
    SM_TABLE.overcast_blend(road, linear, luma, cover, NULL, sky, W * H, 0.3f, 0.6f, 0.41f);
    printf("%-9s overcast %016llx\n", SM_TABLE.name,
           (unsigned long long)(relit ^ fnv(road, N * sizeof(float))));

    /* A mask grown by colour, with the depth test and without. */
    float *edge = malloc(W * H * sizeof(float));
    SM_TABLE.grow_alike(edge, linear, dry, cover, luma, wet, about_foot, W * H, 0.4f, 0.16f, 0.06f,
                        1e-4f, 0.02f);
    uint64_t edges = fnv(edge, W * H * sizeof(float));
    SM_TABLE.grow_alike(edge, linear, dry, cover, luma, wet, NULL, W * H, 0.0f, 1.0f, 0.06f, 1e-4f,
                        0.02f);
    printf("%-9s grow     %016llx\n", SM_TABLE.name,
           (unsigned long long)(edges ^ fnv(edge, W * H * sizeof(float))));

    /* Drops: placed apart, their water surface, and the composite. */
    enum { CANDIDATES = 4000, DROPS = 300, ORDERS = 4 };
    double *px = malloc(CANDIDATES * sizeof(double)), *py = malloc(CANDIDATES * sizeof(double));
    double *pr = malloc(CANDIDATES * sizeof(double));
    int64_t *order = malloc(CANDIDATES * sizeof(int64_t)), *kept = malloc(DROPS * sizeof(int64_t));
    for (int i = 0; i < CANDIDATES; i++) {
        px[i] = uniform() * W, py[i] = uniform() * H, pr[i] = 0.5 + 12.0 * uniform() * uniform();
        order[i] = CANDIDATES - 1 - i;
    }
    const ptrdiff_t placed = SM_TABLE.place_apart(px, py, pr, order, CANDIDATES, DROPS, 0.15, kept);
    double lobes[DROPS * 7], shapes[DROPS * ORDERS * 2];
    const int32_t orders[ORDERS] = {2, 3, 4, 5};
    for (ptrdiff_t d = 0; d < placed; d++) {
        const int64_t i = kept[d];
        const double row[7] = {px[i], py[i], pr[i], 1.0 + 0.4 * uniform(), 0.3 + 0.4 * uniform(),
                               0.1 * uniform(), 0.2 * uniform()};
        for (int k = 0; k < 7; k++) {
            lobes[7 * d + k] = row[k];
        }
        for (int k = 0; k < ORDERS; k++) {
            shapes[(d * ORDERS + k) * 2] = 0.05 * uniform();
            shapes[(d * ORDERS + k) * 2 + 1] = 6.28 * uniform();
        }
    }
    float *level = calloc(W * H, sizeof(float)), *weighted = calloc(W * H, sizeof(float));
    SM_TABLE.water_lobes(level, weighted, H, W, lobes, shapes, placed, ORDERS, orders, 4.0, 0.25);
    float *layers = malloc(W * H * 4 * sizeof(float));
    for (int i = 0; i < W * H; i++) {
        const float alpha = clamp_unit(level[i]);
        layers[4 * i] = alpha * 0.3f, layers[4 * i + 1] = alpha * 0.4f;
        layers[4 * i + 2] = alpha * 0.5f, layers[4 * i + 3] = alpha;
    }
    SM_TABLE.droplet_composite(road, linear, dry, cover, layers, W * H);
    printf("%-9s drops    %016llx\n", SM_TABLE.name,
           (unsigned long long)(fnv(kept, (size_t)placed * sizeof(int64_t))
                                ^ fnv(level, W * H * sizeof(float))
                                ^ fnv(weighted, W * H * sizeof(float))
                                ^ fnv(road, N * sizeof(float))));

    /* Discs of light, as a web's threads are drawn: small and large, some
     * over the frame's edge, the largest first. */
    enum { DISCS = 5000 };
    float *disc[4];
    for (int k = 0; k < 4; k++) {
        disc[k] = malloc(DISCS * sizeof(float));
    }
    int64_t *taken = malloc(DISCS * sizeof(int64_t));
    for (int i = 0; i < DISCS; i++) {
        const float big = uniform();
        disc[0][i] = uniform() * (W + 40) - 20;
        disc[1][i] = uniform() * (H + 40) - 20;
        disc[2][i] = 0.5f + 20.0f * big * big;
        disc[3][i] = 0.1f + 40.0f * uniform();
        taken[i] = DISCS - 1 - i;
    }
    double *total = calloc(W * H, sizeof(double));
    SM_TABLE.splat_discs(total, H, W, disc[0], disc[1], disc[2], disc[3], taken, DISCS);
    printf("%-9s discs    %016llx\n", SM_TABLE.name,
           (unsigned long long)fnv(total, W * H * sizeof(double)));

    /* What reads the scene. Planes in [0, 1) to draw every input from: a
     * scene is W by H, its lines of sight fan out from the lens, and its
     * depth is some metres and up. */
    enum { P = W * H, LAMPS = 9, PUFFS = 400 };
    float *plane[12];
    for (int k = 0; k < 12; k++) {
        plane[k] = malloc(P * sizeof(float));
        for (int i = 0; i < P; i++) {
            plane[k][i] = uniform();
        }
    }
    float *field[5];
    for (int k = 0; k < 5; k++) {
        field[k] = malloc(N * sizeof(float));
        for (int i = 0; i < N; i++) {
            field[k][i] = uniform();
        }
    }
    float *deep = malloc(P * sizeof(float));
    float *ray[5];
    for (int k = 0; k < 5; k++) {
        ray[k] = malloc(P * sizeof(float));
    }
    const float upright[3] = {0.0f, -0.9396926f, -0.3420201f};
    for (int v = 0; v < H; v++) {
        for (int u = 0; u < W; u++) {
            const int i = v * W + u;
            const float x = ((float)u - W / 2) / 280.0f, y = ((float)v - H / 2) / 280.0f;
            const float length = sqrtf(x * x + y * y + 1.0f);
            deep[i] = 4.0f + 90.0f * plane[0][i] * (float)(H - v) / H;
            ray[0][i] = x / length, ray[1][i] = y / length, ray[2][i] = 1.0f / length;
            ray[3][i] = deep[i] * length;
            ray[4][i] = ray[0][i] * upright[0] + ray[1][i] * upright[1] + ray[2][i] * upright[2];
        }
    }

    /* The street lamps' light on the air. */
    float heads[LAMPS * 4];
    for (int k = 0; k < LAMPS; k++) {
        heads[4 * k] = 40.0f * uniform() - 20.0f;
        heads[4 * k + 1] = -5.0f - 4.0f * uniform();
        heads[4 * k + 2] = 10.0f + 80.0f * uniform();
        heads[4 * k + 3] = 8.0f + 3.0f * uniform();
    }
    float *air = calloc(P, sizeof(float)), *air_far = calloc(P, sizeof(float));
    SM_TABLE.street_air(
        air, air_far, ray[0], ray[1], ray[2], ray[3], ray[4], P, heads, LAMPS, upright, 0.36f,
        0.03f, 0.5f);
    printf("%-9s air      %016llx\n", SM_TABLE.name,
           (unsigned long long)(fnv(air, P * sizeof(float))
                                ^ (fnv(air_far, P * sizeof(float)) << 1)));

    /* Puffs of spray: round and long, some cut off behind a line, some
     * over the frame's edge, with and without a depth to hide behind. */
    double *puffs = malloc(PUFFS * 15 * sizeof(double));
    for (int k = 0; k < PUFFS; k++) {
        double *puff = puffs + 15 * k;
        const double east = 2.0 * uniform() - 1.0, south = 0.1 + uniform();
        const double long_way = sqrt(east * east + south * south);
        puff[0] = uniform() * (W + 40) - 20, puff[1] = uniform() * (H + 40) - 20;
        puff[2] = east / long_way, puff[3] = south / long_way;
        puff[4] = 0.5 + 30.0 * uniform(), puff[5] = 0.5 + 12.0 * uniform();
        puff[6] = 0.01 + uniform(), puff[7] = uniform() * H;
        puff[8] = k % 3 ? 6.0 + 60.0 * uniform() : 0.0, puff[9] = 1.0 + 2.0 * uniform();
        puff[10] = puff[0] - 6.0, puff[11] = puff[1] + 4.0;
        puff[12] = puff[2], puff[13] = puff[3], puff[14] = k % 4 ? 0.0 : 1.0 + 9.0 * uniform();
    }
    for (int i = 0; i < P; i++) {
        plane[1][i] = plane[1][i] * H;
    }
    float *spray = calloc(P, sizeof(float));
    SM_TABLE.spray_puffs(spray, plane[1], plane[2], deep, H, W, puffs, PUFFS, 2.4f, 0.5f, 0.7f);
    uint64_t sprayed = fnv(spray, P * sizeof(float));
    SM_TABLE.spray_puffs(spray, plane[1], plane[2], NULL, H, W, puffs, PUFFS, 2.4f, 0.5f, 0.7f);
    printf("%-9s spray    %016llx\n", SM_TABLE.name,
           (unsigned long long)(sprayed ^ (fnv(spray, P * sizeof(float)) << 1)));

    /* Surface normals, differenced two pixels apart. */
    float *across = malloc(W * sizeof(float)), *down = malloc(H * sizeof(float));
    for (int u = 0; u < W; u++) {
        across[u] = ((float)u - W / 2.0f) / 280.0f;
    }
    for (int v = 0; v < H; v++) {
        down[v] = ((float)v - H / 2.0f) / 280.0f;
    }
    float *normal[4];
    for (int k = 0; k < 4; k++) {
        normal[k] = malloc(P * sizeof(float));
    }
    SM_TABLE.normals(normal[0], normal[1], normal[2], normal[3], deep, across, down, H, W, 2);
    printf("%-9s normals  %016llx\n", SM_TABLE.name,
           (unsigned long long)(fnv(normal[0], P * sizeof(float))
                                ^ (fnv(normal[1], P * sizeof(float)) << 1)
                                ^ (fnv(normal[2], P * sizeof(float)) << 2)
                                ^ (fnv(normal[3], P * sizeof(float)) << 3)));

    /* The two nights: the infrared one with every lamp and return there
     * is, then with none; the colour one evened and lit, then bare. */
    const float response[3] = {0.5f, 0.3f, 0.2f};
    const float ir_how[8] = {1.7f, 0.3f, 0.02f, 0.01f, 1.4f, 1.04f, 1.0f, 0.98f};
    const float colour_how[14] = {
        0.02f, 0.025f, 0.04f, 1.0f, 0.75f, 0.45f, 1.0f, 0.95f, 0.85f, 0.21f, 0.65f,
        0.2126f, 0.7152f, 0.0722f};
    const float expose_how[5] = {2.3f, 0.3f, 0.03f, 0.01f, 1.4f};
    float *lifted = malloc(P * sizeof(float)), *signal = malloc(P * sizeof(float));
    float *seen = malloc(P * sizeof(float)), *night = malloc(N * sizeof(float));
    float *lit = malloc(N * sizeof(float));
    SM_TABLE.nir_reflectance(seen, field[0], plane[3], P, response, 0.6f, 0.5f, 0.31f);
    uint64_t dark = fnv(seen, P * sizeof(float));
    SM_TABLE.night_ir_light(
        signal, lifted, seen, plane[4], plane[5], plane[6], plane[7], plane[8], P, 0.02f, 0.3f,
        0.88f);
    SM_TABLE.night_ir_expose(
        night, signal, plane[9], lifted, plane[4], plane[10], field[1], field[2], plane[11],
        plane[3], plane[0], P, ir_how);
    dark ^= fnv(night, N * sizeof(float)) << 1;
    SM_TABLE.nir_reflectance(seen, field[0], NULL, P, response, 0.6f, 0.5f, 0.31f);
    SM_TABLE.night_ir_light(
        signal, lifted, seen, plane[4], NULL, NULL, NULL, NULL, P, 0.02f, 0.0f, 0.88f);
    SM_TABLE.night_ir_expose(
        night, signal, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, plane[0], P, ir_how);
    dark ^= fnv(night, N * sizeof(float)) << 2;
    SM_TABLE.night_colour_light(
        lit, seen, field[0], plane[4], plane[5], plane[6], plane[7], plane[8], P, colour_how);
    SM_TABLE.night_colour_expose(
        night, lit, field[1], field[2], plane[11], plane[3], field[3], P, colour_how + 3,
        expose_how);
    dark ^= (fnv(night, N * sizeof(float)) << 3) ^ fnv(seen, P * sizeof(float));
    SM_TABLE.night_colour_light(lit, seen, field[0], NULL, NULL, NULL, NULL, NULL, P, colour_how);
    SM_TABLE.night_colour_expose(
        night, lit, NULL, NULL, NULL, NULL, field[3], P, colour_how + 3, expose_how);
    dark ^= fnv(night, N * sizeof(float)) << 4;
    printf("%-9s night    %016llx\n", SM_TABLE.name, (unsigned long long)dark);

    /* Fog at night: with the lamps' cones under an illuminator, then as a veil. */
    const float cone_how[10] = {0.6f, 1.0f, 0.05f, 0.04f, 0.03f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    const float veil_how[10] = {0.6f, 0.0f, 0.0f, 0.0f, 0.0f, 2.2f, 0.75f, 0.01f, 0.012f, 0.014f};
    float *radiance = malloc(N * sizeof(float)), *removed = malloc(N * sizeof(float));
    SM_TABLE.fog_lost(radiance, removed, seen, field[0], plane[5], plane[6], P, 5.0f);
    uint64_t fogged = fnv(radiance, N * sizeof(float)) ^ (fnv(removed, N * sizeof(float)) << 1)
        ^ (fnv(seen, P * sizeof(float)) << 2);
    SM_TABLE.fog_veil(night, radiance, field[4], plane[6], field[1], NULL, plane[7], P, cone_how);
    fogged ^= fnv(night, N * sizeof(float)) << 3;
    SM_TABLE.fog_veil(night, radiance, field[4], plane[6], NULL, field[2], NULL, P, veil_how);
    fogged ^= fnv(night, N * sizeof(float)) << 4;
    printf("%-9s fog      %016llx\n", SM_TABLE.name, (unsigned long long)fogged);

    /* Lying snow, with patches and labels and without, and the ground's
     * noise laid through each pixel's depth. */
    const float snow_how[7] = {0.18f, 0.18f, 0.5f, 2.5f, 0.7f, 0.06f, 0.49f};
    const float tile_how[8] = {
        280.0f, 0.9396926f, 0.3420201f, 6.4f, 510.0f, 1.0f / 510.0f, 37.25f, 81.5f};
    for (int i = 0; i < P; i++) {
        plane[10][i] = 0.05f + plane[10][i];
    }
    SM_TABLE.snow_settled(
        seen, plane[3], plane[4], plane[5], plane[6], plane[7], plane[8], plane[9],
        plane[11], plane[0], plane[10], plane[2], P, snow_how);
    uint64_t lying = fnv(seen, P * sizeof(float));
    SM_TABLE.snow_settled(
        seen, plane[3], plane[4], plane[5], plane[6], NULL, NULL, NULL, plane[11], plane[0],
        plane[10], plane[2], P, snow_how);
    lying ^= fnv(seen, P * sizeof(float)) << 1;
    SM_TABLE.ground_tile(signal, lifted, deep, H, W, tile_how);
    lying ^= (fnv(signal, P * sizeof(float)) << 2) ^ (fnv(lifted, P * sizeof(float)) << 3);
    printf("%-9s snow     %016llx\n", SM_TABLE.name, (unsigned long long)lying);

    /* Lamps on surfaces: the street's pools on every point of the scene,
     * and the dipped beams of vehicles coming on and driving away. */
    float *spot = malloc(N * sizeof(float));
    for (int i = 0; i < P; i++) {
        spot[3 * i] = ray[0][i] * ray[3][i], spot[3 * i + 1] = ray[1][i] * ray[3][i];
        spot[3 * i + 2] = ray[2][i] * ray[3][i];
        signal[i] = spot[3 * i] * upright[0] + spot[3 * i + 1] * upright[1]
            + spot[3 * i + 2] * upright[2];
        lifted[i] = 30.0f * plane[4][i] - 15.0f, seen[i] = 4.0f + 60.0f * plane[5][i];
        deep[i] = 3.0f * plane[6][i], plane[7][i] = plane[7][i] > 0.2f ? 1.0f : 0.0f;
    }
    memset(air, 0, P * sizeof(float));
    SM_TABLE.street_light(air, spot, signal, P, heads, LAMPS, upright, 0.03f);
    uint64_t lamp_pools = fnv(air, P * sizeof(float));
    enum { BEAMS = 7 };
    float beams[BEAMS * 6];
    for (int k = 0; k < BEAMS; k++) {
        /* A heading with no sine or cosine in it: those are the C
         * library's, and differ by a bit from one library to the next. */
        const float east = 2.0f * uniform() - 1.0f, south = 0.1f + uniform();
        const float long_way = sqrtf(east * east + south * south);
        beams[6 * k] = 24.0f * uniform() - 12.0f, beams[6 * k + 1] = 8.0f + 50.0f * uniform();
        beams[6 * k + 2] = east / long_way, beams[6 * k + 3] = south / long_way;
        beams[6 * k + 4] = 0.6f + uniform(), beams[6 * k + 5] = 0.4f + 0.6f * uniform();
    }
    const float dipped[7] = {0.35f, 0.12f, 0.65f, 0.6f, 14.0f, 75.0f, 1.2f};
    memset(air, 0, P * sizeof(float));
    SM_TABLE.headlamp_beams(air, lifted, seen, deep, plane[7], P, beams, BEAMS, dipped);
    printf("%-9s lamps    %016llx\n", SM_TABLE.name,
           (unsigned long long)(lamp_pools ^ (fnv(air, P * sizeof(float)) << 1)));

    /* Wheel tracks down the lanes of two roads, even and then ragged and
     * worn; and the rain's spray and lamps laid over a frame. */
    int32_t *on_road = malloc(P * sizeof(int32_t));
    for (int i = 0; i < P; i++) {
        on_road[i] = (int32_t)(3.0f * plane[8][i]);
        plane[9][i] = 1.1f * plane[9][i] - 0.05f, plane[10][i] = 2.0f + 80.0f * plane[11][i];
    }
    const double lines[8] = {0.02, 0.27, 0.5, 0.74, 0.97, 0.1, 0.52, 0.9};
    const int32_t first_line[4] = {0, 0, 5, 8};
    float lane_use[64];
    for (int k = 0; k < 64; k++) {
        lane_use[k] = 0.35f + 0.65f * uniform();
    }
    const float track_how[5] = {0.09f, 0.1f, 0.22f, 0.3f, 1.4f};
    SM_TABLE.wheel_tracks(
        air, on_road, plane[9], plane[10], plane[7], NULL, NULL, P, lines, first_line, 2, lane_use,
        track_how);
    uint64_t tracks = fnv(air, P * sizeof(float));
    SM_TABLE.wheel_tracks(
        air, on_road, plane[9], plane[10], plane[7], plane[3], plane[4], P, lines, first_line, 2,
        lane_use, track_how);
    tracks ^= fnv(air, P * sizeof(float)) << 1;
    const float spray_lit[3] = {0.45f, 0.5f, 0.54f};
    for (int i = 0; i < N; i++) {
        field[1][i] = field[1][i] > 0.97f ? 6.0f * field[1][i] : 0.0f;
    }
    memcpy(night, field[0], N * sizeof(float));
    SM_TABLE.lamps_lit(night, field[1], plane[6], P);
    SM_TABLE.spray_and_lamps(night, plane[5], field[1], field[2], plane[6], P, spray_lit, 0.35f);
    tracks ^= fnv(night, N * sizeof(float)) << 2;
    memcpy(night, field[0], N * sizeof(float));
    SM_TABLE.spray_and_lamps(night, NULL, field[1], field[2], plane[6], P, spray_lit, 0.35f);
    SM_TABLE.spray_and_lamps(night, plane[5], NULL, NULL, plane[6], P, spray_lit, 0.35f);
    tracks ^= fnv(night, N * sizeof(float)) << 3;
    printf("%-9s tracks   %016llx\n", SM_TABLE.name, (unsigned long long)tracks);

    /* A road's frame: the way it runs at every pixel, on a bend, and each
     * pixel's carriageway, its place across it and its width there. */
    float *runs = malloc(2 * P * sizeof(float));
    SM_TABLE.road_along(runs, H, W, 151.5, -20.25, 700.0);
    uint64_t framed = fnv(runs, 2 * P * sizeof(float));
    double *place = malloc(P * sizeof(double)), *under = malloc(H * sizeof(double));
    int64_t *counted = malloc(P * sizeof(int64_t));
    uint8_t *is_road = malloc(P);
    for (int v = 0; v < H; v++) {
        under[v] = 20.25 + v;
    }
    for (int i = 0; i < P; i++) {
        place[i] = 4.0 * (double)plane[0][i] - 2.0;
        counted[i] = (int64_t)(10.0f * plane[2][i]);
        is_road[i] = plane[3][i] > 0.3f;
    }
    const int32_t carriageway[10] = {0, 1, 1, 1, 0, 0, 2, 2, 2, 0};
    const double begins[3] = {0.0, -1.6, 0.4}, spans[3] = {1.0, 1.2, 1.2};
    SM_TABLE.road_places(
        on_road, air, air_far, place, counted, is_road, under, H, W, carriageway, begins, spans);
    framed ^= (fnv(on_road, P * sizeof(int32_t)) << 1) ^ (fnv(air, P * sizeof(float)) << 2)
        ^ (fnv(air_far, P * sizeof(float)) << 3);
    printf("%-9s road     %016llx\n", SM_TABLE.name, (unsigned long long)framed);
    return 0;
}
