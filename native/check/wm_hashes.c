/* Hashes of what each kernel computes on fixed inputs.
 *
 * The Python tests hold the kernels to the numpy code on the machine they
 * run on. This holds one machine to another: built for any tier on any
 * architecture, the program must print the same fourteen lines. It needs only a
 * C compiler, so it runs where the Python stack is slow or missing, an
 * arm64 container under QEMU for one. run.sh builds it for every tier of
 * the machine it is on.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "wm_kernels.h"

extern const wm_kernel_table WM_TABLE;

static uint32_t state = 12345u;

static float uniform(void) {
    state = state * 1664525u + 1013904223u;
    return (float)(state >> 8) / 16777216.0f;
}

static float clamp_unit(float v) { return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v); }

static uint64_t fnv(const void *data, size_t bytes) {
    const unsigned char *p = data;
    uint64_t hash = 1469598103934665603ull;
    for (size_t i = 0; i < bytes; i++) {
        hash = (hash ^ p[i]) * 1099511628211ull;
    }
    return hash;
}

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
    WM_TABLE.encode_u8(linear, codes, N, table8);
    WM_TABLE.decode_u8(codes, decoded, N, table32);
    uint64_t colour = fnv(codes, N) ^ fnv(decoded, N * sizeof(float));
    WM_TABLE.encode_f32(linear, decoded, N, table32);
    colour ^= fnv(decoded, N * sizeof(float)) << 1;
    printf("%-9s colour   %016llx\n", WM_TABLE.name, (unsigned long long)colour);

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
    WM_TABLE.render_flakes(
        canvas, H, W, FLAKES, f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7],
        0, 0.0, 1.0, 0.0, W / 2.0, H / 2.0, 0);
    printf("%-9s round    %016llx\n", WM_TABLE.name,
           (unsigned long long)fnv(canvas, W * H * sizeof(float)));
    memset(canvas, 0, W * H * sizeof(float));
    WM_TABLE.render_flakes(
        canvas, H, W, FLAKES, f[0], f[1], f[2], f[3], f[4], f[5], f[6], NULL,
        6, 0.3, 1.4, 0.5, W / 2.0, H / 2.0, 0);
    printf("%-9s iris     %016llx\n", WM_TABLE.name,
           (unsigned long long)fnv(canvas, W * H * sizeof(float)));

    memset(canvas, 0, W * H * sizeof(float));
    WM_TABLE.render_flakes(
        canvas, H, W, FLAKES, f[0], f[1], f[2], f[3], f[4], f[5], f[6], NULL,
        6, 0.3, 1.4, 0.5, W / 2.0, H / 2.0, 1);
    printf("%-9s iris-fast %015llx\n", WM_TABLE.name,
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
    WM_TABLE.render_streaks(
        canvas, H, W, select, STREAKS, geometry[0], geometry[1], geometry[2], geometry[3],
        vertices, VERTICES, geometry[4], s[0], s[1], s[2], s[3], s[4], s[5], s[6], beading, 256);
    printf("%-9s streaks  %016llx\n", WM_TABLE.name,
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
    WM_TABLE.wet_composite(
        road, dry, sharp, linear, wet, pools, flat, tilted, H, W, 0.55f, 0.45f, 0.7f, 0.6f);
    printf("%-9s wet      %016llx\n", WM_TABLE.name, (unsigned long long)fnv(road, N * sizeof(float)));

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
    WM_TABLE.mirror_rows(solid, H, W, 0.3 * H, about_foot, reach);
    for (int r = 0; r < H; r++) {
        about_horizon[r] = 2.0f * 71.3f - (float)r;
        off_frame[r] = clamp_unit((19.2f - about_horizon[r]) / 19.2f);
    }
    WM_TABLE.wet_reflection(road, dry, H, W, H, about_horizon, off_frame, about_foot, reach, sky);
    printf("%-9s mirror   %016llx\n", WM_TABLE.name,
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
    if (WM_TABLE.edge_preserving(frame, filtered, H, W, 25.0f, 0.2f, 0)) {
        return 1;
    }
    printf("%-9s denoise  %016llx\n", WM_TABLE.name, (unsigned long long)fnv(filtered, N));
    if (WM_TABLE.edge_preserving(frame, filtered, H, W, 25.0f, 0.2f, 1)) {
        return 1;
    }
    printf("%-9s denoise-tabled %010llx\n", WM_TABLE.name, (unsigned long long)fnv(filtered, N));

    /* Lens dirt: the highlight-weighted scene, then the deposit over it. */
    float *luma = malloc(W * H * sizeof(float)), *cover = malloc(W * H * sizeof(float));
    for (int i = 0; i < W * H; i++) {
        luma[i] = uniform() * 1.2f - 0.1f;
        cover[i] = uniform() * 1.5f - 0.2f;
    }
    WM_TABLE.dirt_emphasise(road, linear, luma, W * H, 3.7f);
    const uint64_t emphasised = fnv(road, N * sizeof(float));
    WM_TABLE.dirt_composite(road, linear, dry, cover, sharp, sky, W * H, 0.6f, 0.7f, 0.09f, 0.35f);
    printf("%-9s dirt     %016llx\n", WM_TABLE.name,
           (unsigned long long)(emphasised ^ fnv(road, N * sizeof(float))));

    /* A frame through a veil and a layer of particles, lit by one colour and
     * then by a field; and the overcast relighting, with and without sky. */
    for (int i = 0; i < W * H; i++) {
        luma[i] = uniform();
        cover[i] = uniform();
    }
    WM_TABLE.veil_blend(road, linear, luma, sky, cover, sky, NULL, W * H);
    uint64_t veiled = fnv(road, N * sizeof(float));
    WM_TABLE.veil_blend(road, linear, luma, sky, cover, NULL, dry, W * H);
    veiled ^= fnv(road, N * sizeof(float));
    WM_TABLE.veil_blend(road, linear, luma, sky, NULL, NULL, NULL, W * H);
    printf("%-9s veil     %016llx\n", WM_TABLE.name,
           (unsigned long long)(veiled ^ fnv(road, N * sizeof(float))));
    WM_TABLE.overcast_blend(road, linear, luma, cover, wet, sky, W * H, 0.3f, 0.6f, 0.41f);
    const uint64_t relit = fnv(road, N * sizeof(float));
    WM_TABLE.overcast_blend(road, linear, luma, cover, NULL, sky, W * H, 0.3f, 0.6f, 0.41f);
    printf("%-9s overcast %016llx\n", WM_TABLE.name,
           (unsigned long long)(relit ^ fnv(road, N * sizeof(float))));

    /* A mask grown by colour, with the depth test and without. */
    float *edge = malloc(W * H * sizeof(float));
    WM_TABLE.grow_alike(edge, linear, dry, cover, luma, wet, about_foot, W * H, 0.4f, 0.16f, 0.06f,
                        1e-4f, 0.02f);
    uint64_t edges = fnv(edge, W * H * sizeof(float));
    WM_TABLE.grow_alike(edge, linear, dry, cover, luma, wet, NULL, W * H, 0.0f, 1.0f, 0.06f, 1e-4f,
                        0.02f);
    printf("%-9s grow     %016llx\n", WM_TABLE.name,
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
    const ptrdiff_t placed = WM_TABLE.place_apart(px, py, pr, order, CANDIDATES, DROPS, 0.15, kept);
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
    WM_TABLE.water_lobes(level, weighted, H, W, lobes, shapes, placed, ORDERS, orders, 4.0, 0.25);
    float *layers = malloc(W * H * 4 * sizeof(float));
    for (int i = 0; i < W * H; i++) {
        const float alpha = clamp_unit(level[i]);
        layers[4 * i] = alpha * 0.3f, layers[4 * i + 1] = alpha * 0.4f;
        layers[4 * i + 2] = alpha * 0.5f, layers[4 * i + 3] = alpha;
    }
    WM_TABLE.droplet_composite(road, linear, dry, cover, layers, W * H);
    printf("%-9s drops    %016llx\n", WM_TABLE.name,
           (unsigned long long)(fnv(kept, (size_t)placed * sizeof(int64_t))
                                ^ fnv(level, W * H * sizeof(float))
                                ^ fnv(weighted, W * H * sizeof(float))
                                ^ fnv(road, N * sizeof(float))));
    return 0;
}
