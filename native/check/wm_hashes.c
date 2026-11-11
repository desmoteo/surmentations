/* Hashes of what each kernel computes on fixed inputs.
 *
 * The Python tests hold the kernels to the numpy code on the machine they
 * run on. This holds one machine to another: built for any tier on any
 * architecture, the program must print the same four lines. It needs only a
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
        0, 0.0, 1.0, 0.0, W / 2.0, H / 2.0);
    printf("%-9s round    %016llx\n", WM_TABLE.name,
           (unsigned long long)fnv(canvas, W * H * sizeof(float)));
    memset(canvas, 0, W * H * sizeof(float));
    WM_TABLE.render_flakes(
        canvas, H, W, FLAKES, f[0], f[1], f[2], f[3], f[4], f[5], f[6], NULL,
        6, 0.3, 1.4, 0.5, W / 2.0, H / 2.0);
    printf("%-9s iris     %016llx\n", WM_TABLE.name,
           (unsigned long long)fnv(canvas, W * H * sizeof(float)));

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
    return 0;
}
