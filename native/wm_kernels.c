/** The kernels, in plain C written for the compiler to vectorise.
 *
 * Compiled once per tier with -DWM_TIER=<name> and that tier's flags, always
 * with -ffp-contract=off and never with -ffast-math: every tier must give the
 * same floats, and the results are checked against the numpy code these
 * replace. What that asks of the source:
 *
 * - pointer parameters are `restrict`, and loops run over `ptrdiff_t`, not
 *   `int`, so the vectoriser does not have to prove an index cannot wrap;
 * - a row of a sprite is one loop with no call and no branch in it;
 * - arithmetic is float where numpy's is float32 and in numpy's order.
 */
/**/

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "wm_kernels.h"

#ifndef WM_TIER
/** The tier this compilation is for, set by the build with ``-DWM_TIER=``. */
#define WM_TIER portable
#endif
/** A kernel's name for this tier: ``WM_FN(encode_u8)`` is ``encode_u8_avx2`` in the AVX2 object.
 *
 * Each tier's object defines its own copies, and only its table is exported.
 */
#define WM_FN(name) WM_CAT(name, WM_TIER)
#define WM_STR_(a) #a
/** The tier's name as a string, for the table. */
#define WM_STR(a) WM_STR_(a)
/** Keeps a function out of line: inlined into a caller with constant arguments, GCC lost the vectorisation of some loops (see ``shade_row``).
 */
#define WM_NOINLINE __attribute__((noinline))
/** Marks a loop the build is checked to have vectorised (test_native_build.py). */
#define WM_VECTORISED

/** ``v`` clamped to ``[0, 1]``; a NaN comes out as 0. */
static inline float clamp01(float v) {
    /* NaN compares false both times and comes out as 0. */
    v = v > 0.0f ? v : 0.0f;
    return v < 1.0f ? v : 1.0f;
}

/* ------------------------------------------------------------ colour */

/** Values a colour kernel converts at a time: their indices fit on the stack. */
#define WM_BLOCK 512

/** Table indices for a block of values. Kept apart from the lookups so that
 * it is one loop of float arithmetic and nothing else: written as a single
 * loop with the lookup in it, GCC compiled the clamp as two branches, which
 * a picture's highlights and shadows mispredict, and vectorised nothing.
 * Returns nonzero if a value was NaN.
 */
static inline int table_indices(const float *restrict in, int32_t *restrict index, size_t n) {
    int bad = 0;
    WM_VECTORISED for (size_t i = 0; i < n; i++) {
        const float v = in[i];
        bad |= (v != v);
        index[i] = (int32_t)(clamp01(v) * 65535.0f);
    }
    return bad;
}

/** Linear light to 8-bit sRGB: each value clamped, scaled to a table index and looked up.
 *
 * Block by block, the indices of a block first and then its lookups.
 * Returns nonzero if a value was NaN, which the caller refuses.
 */
static int WM_FN(encode_u8)(
    const float *restrict in, uint8_t *restrict out, size_t n, const uint8_t *restrict table) {
    int32_t index[WM_BLOCK];
    int bad = 0;
    for (size_t done = 0; done < n; done += WM_BLOCK) {
        const size_t count = n - done < WM_BLOCK ? n - done : WM_BLOCK;
        bad |= table_indices(in + done, index, count);
        uint8_t *restrict block = out + done;
        for (size_t i = 0; i < count; i++) {
            block[i] = table[index[i]];
        }
    }
    return bad;
}

/** As ``encode_u8``, looking up float32 sRGB values in ``[0, 1]``. */
static int WM_FN(encode_f32)(
    const float *restrict in, float *restrict out, size_t n, const float *restrict table) {
    int32_t index[WM_BLOCK];
    int bad = 0;
    for (size_t done = 0; done < n; done += WM_BLOCK) {
        const size_t count = n - done < WM_BLOCK ? n - done : WM_BLOCK;
        bad |= table_indices(in + done, index, count);
        float *restrict block = out + done;
        for (size_t i = 0; i < count; i++) {
            block[i] = table[index[i]];
        }
    }
    return bad;
}

/** 8-bit sRGB to linear light: each byte looked up in a 256-entry table. */
static void WM_FN(decode_u8)(
    const uint8_t *restrict in, float *restrict out, size_t n, const float *restrict table) {
    for (size_t i = 0; i < n; i++) {
        out[i] = table[in[i]];
    }
}

/* ------------------------------------------------------------- flakes */

/** The most aperture blades a polygonal sprite may have. */
#define WM_MAX_BLADES 32
/** Half the side, in pixels, of the largest sprite the kernel will draw. */
#define WM_MAX_REACH 4096.0
static const double WM_PI = 3.14159265358979323846;

/** cv2.getGaussianKernel(n, sigma <= 0), which is what
 * cv2.GaussianBlur(src, (n, n), 0) filters with.
 */
static void gaussian_kernel(float *kernel, int n) {
    static const float small[4][7] = {
        {1.0f},
        {0.25f, 0.5f, 0.25f},
        {0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f},
        {0.03125f, 0.109375f, 0.21875f, 0.28125f, 0.21875f, 0.109375f, 0.03125f},
    };
    if (n <= 7) {
        memcpy(kernel, small[n >> 1], (size_t)n * sizeof(float));
        return;
    }
    double sigma = ((n - 1) * 0.5 - 1.0) * 0.3 + 0.8;
    double scale = -0.5 / (sigma * sigma);
    double sum = 0.0;
    double values[n];
    for (int i = 0; i < n; i++) {
        double t = i - (n - 1) * 0.5;
        values[i] = exp(scale * t * t);
        sum += values[i];
    }
    for (int i = 0; i < n; i++) {
        kernel[i] = (float)(values[i] / sum);
    }
}

/** The index ``i`` reflected into ``[0, n)`` as BORDER_REFLECT_101 reflects it: ``-1`` is ``1``. */
static inline ptrdiff_t reflect101(ptrdiff_t i, ptrdiff_t n) {
    if (n == 1) {
        return 0;
    }
    while (i < 0 || i >= n) {
        i = i < 0 ? -i : 2 * n - 2 - i;
    }
    return i;
}

/** cv2.getGaussianKernel(n, sigma) for sigma > 0, as float32. */
static void gaussian_kernel_sigma(float *kernel, int n, double sigma) {
    const double scale = -0.5 / (sigma * sigma);
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        const double t = i - (n - 1) * 0.5;
        kernel[i] = (float)exp(scale * t * t);
        sum += kernel[i];
    }
    for (int i = 0; i < n; i++) {
        kernel[i] = (float)(kernel[i] / sum);
    }
}

/** Separable blur of a width x height patch with a symmetric kernel and
 * BORDER_REFLECT_101, in place: cv2.GaussianBlur's filter. `tmp` is scratch
 * the size of the patch, `line` of width + ksize floats.
 */
static void blur_patch(
    float *restrict patch, float *restrict tmp, float *restrict line,
    ptrdiff_t width, ptrdiff_t height, const float *kernel, int ksize) {
    const int half = ksize >> 1;
    const float centre = kernel[half];
    const float *const taps = kernel + half;
    /* The row with its borders, and where in it a row's own values start
     * and end: the same for every row, so found once. */
    float *const inner = line + half;
    float *const last = inner + width - 1;
    const size_t row_bytes = (size_t)width * sizeof(float);
    const float *restrict src = patch;
    float *restrict dst = tmp;
    for (ptrdiff_t r = 0; r < height; r++, src += width, dst += width) {
        memcpy(inner, src, row_bytes);
        for (ptrdiff_t c = 1; c <= half; c++) {
            inner[-c] = src[reflect101(-c, width)];
            last[c] = src[reflect101(width - 1 + c, width)];
        }
        for (ptrdiff_t c = 0; c < width; c++) {
            dst[c] = centre * inner[c];
        }
        for (int k = 1; k <= half; k++) {
            const float weight = taps[k];
            const float *restrict left = inner - k;
            const float *restrict right = inner + k;
            for (ptrdiff_t c = 0; c < width; c++) {
                dst[c] += weight * (left[c] + right[c]);
            }
        }
    }
    dst = patch;
    const float *restrict mid = tmp;
    for (ptrdiff_t r = 0; r < height; r++, dst += width, mid += width) {
        for (ptrdiff_t c = 0; c < width; c++) {
            dst[c] = centre * mid[c];
        }
        for (int k = 1; k <= half; k++) {
            const float weight = taps[k];
            const float *restrict above = tmp + reflect101(r - k, height) * width;
            const float *restrict below = tmp + reflect101(r + k, height) * width;
            for (ptrdiff_t c = 0; c < width; c++) {
                dst[c] += weight * (above[c] + below[c]);
            }
        }
    }
}

/* One row of a sprite at a time, each step a function with nothing in it but
 * its loop: straight-line float arithmetic, which every tier vectorises.
 *
 * That is less robust than it looks. The last step, three clamps in a row,
 * was left scalar when it was written inline in render_flakes, and again in
 * a test program that called it with constants ("control flow in loop"):
 * GCC's jump threading had turned the clamps into branches. The build
 * passes -fno-thread-jumps for that, shade_row is kept out of line as well,
 * and tests/unit/test_native_build.py fails if a loop marked WM_VECTORISED
 * stops being vectorised. */
/** Distance to a segment: a circle swept along its motion is a capsule. */
static inline void capsule_row(
    float *restrict row, const float *restrict dx, ptrdiff_t size,
    float dy, float ux, float uy, float hs) {
    const float dyuy = dy * uy;
    WM_VECTORISED for (ptrdiff_t c = 0; c < size; c++) {
        float along = dx[c] * ux + dyuy;
        along = along < -hs ? -hs : along;
        along = along > hs ? hs : along;
        const float ex = dx[c] - along * ux, ey = dy - along * uy;
        row[c] = sqrtf(ex * ex + ey * ey);
    }
}

/** A polygon swept along a segment: each edge's support grows by the
 * segment's reach along that edge's normal.
 */
static inline void first_plane_row(
    float *restrict row, const float *restrict dx, ptrdiff_t size,
    float dy, float nx, float ny, float support) {
    WM_VECTORISED for (ptrdiff_t c = 0; c < size; c++) {
        row[c] = (dx[c] * nx + dy * ny) / support;
    }
}

/** As ``first_plane_row``, keeping the larger of the row and this edge's distance. */
static inline void next_plane_row(
    float *restrict row, const float *restrict dx, ptrdiff_t size,
    float dy, float nx, float ny, float support) {
    WM_VECTORISED for (ptrdiff_t c = 0; c < size; c++) {
        const float plane = (dx[c] * nx + dy * ny) / support;
        row[c] = row[c] > plane ? row[c] : plane;
    }
}

/** Fast mode: the division by the support is folded into the normal, once
 * per sprite, so a plane is two multiplies and an add. The quotient and the
 * product with a rounded reciprocal differ in the last bit or two.
 */
static inline void first_plane_row_fast(
    float *restrict row, const float *restrict dx, ptrdiff_t size, float dy, float nx, float ny) {
    const float dyny = dy * ny;
    WM_VECTORISED for (ptrdiff_t c = 0; c < size; c++) {
        row[c] = dx[c] * nx + dyny;
    }
}

/** As ``first_plane_row_fast``, keeping the larger of the row and this edge's distance. */
static inline void next_plane_row_fast(
    float *restrict row, const float *restrict dx, ptrdiff_t size, float dy, float nx, float ny) {
    const float dyny = dy * ny;
    WM_VECTORISED for (ptrdiff_t c = 0; c < size; c++) {
        const float plane = dx[c] * nx + dyny;
        row[c] = row[c] > plane ? row[c] : plane;
    }
}

/** The cat's-eye stop: a second, displaced aperture. */
static inline void stop_row(
    float *restrict row, const float *restrict dx, ptrdiff_t size,
    float dy, float clip_x, float clip_y) {
    const float ey = dy - clip_y;
    const float ey2 = ey * ey;
    WM_VECTORISED for (ptrdiff_t c = 0; c < size; c++) {
        const float ex = dx[c] - clip_x;
        const float stop = sqrtf(ex * ex + ey2);
        row[c] = row[c] > stop ? row[c] : stop;
    }
}

/** Distance to brightness: a defined rim, then the aberration profile. */
static WM_NOINLINE void shade_row(float *restrict row, ptrdiff_t size, float rim, float level) {
    WM_VECTORISED for (ptrdiff_t c = 0; c < size; c++) {
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

/** Fast mode: the two divisions by constants as multiplications. */
static WM_NOINLINE void shade_row_fast(
    float *restrict row, ptrdiff_t size, float rim, float level) {
    const float per_edge = 1.0f / 0.12f, per_core = 1.0f / 0.35f;
    WM_VECTORISED for (ptrdiff_t c = 0; c < size; c++) {
        const float inside = clamp01(1.0f - row[c]);
        float edge = inside * per_edge;
        edge = edge < 1.0f ? edge : 1.0f;
        float fall = 1.0f - inside * per_core;
        fall = fall > 0.0f ? fall : 0.0f;
        row[c] = edge * (1.0f + rim * fall) * level;
    }
}

/** ``dst`` set to the larger of itself and ``src``, value by value: max-blending a sprite. */
static inline void max_row(float *restrict dst, const float *restrict src, ptrdiff_t size) {
    WM_VECTORISED for (ptrdiff_t c = 0; c < size; c++) {
        dst[c] = dst[c] > src[c] ? dst[c] : src[c];
    }
}

/** Max-blend every flake's sprite into ``canvas`` (snow.render_flake_field).
 *
 * For each flake: its patch's size and place from its radius, defocus and
 * smear, in double as the numpy code works them out; its distance field,
 * round (a capsule) or polygonal (the iris' blades), with a cat's-eye stop
 * when asked; that distance shaded into brightness; the sprite blurred as
 * cv2.GaussianBlur would; and the sprite max-blended into the canvas.
 * Returns nonzero, having drawn nothing more, on a NaN or absurd flake or
 * if scratch space cannot be had.
 */
static int WM_FN(render_flakes)(
    float *canvas, int height, int width, ptrdiff_t count,
    const float *x, const float *y, const float *radius, const float *blur_radius,
    const float *smear, const float *dir_x, const float *dir_y, const float *gain,
    int blades, double blade_rotation, double rim_gain, double cats_eye,
    double centre_x, double centre_y, int fast) {
    float normal_x[WM_MAX_BLADES], normal_y[WM_MAX_BLADES];
    double base_support = 0.0;
    if (blades > WM_MAX_BLADES) {
        blades = WM_MAX_BLADES;
    }
    if (blades >= 3) {
        base_support = cos(WM_PI / blades);
        for (int b = 0; b < blades; b++) {
            double angle = blade_rotation + 2.0 * WM_PI * b / blades;
            normal_x[b] = (float)cos(angle);
            normal_y[b] = (float)sin(angle);
        }
    }

    ptrdiff_t capacity = 0;
    float *sprite = NULL, *tmp = NULL, *axis = NULL, *line = NULL;
    const float rim = (float)(rim_gain - 1.0);

    for (ptrdiff_t i = 0; i < count; i++) {
        /* A sprite's set-up is in double, divisions and all, because the
         * numpy code's is: its scalars are Python floats, and a patch's size
         * and corner are rounded from them. It runs once per flake, against
         * thousands of pixels each. Done in float it measured no faster,
         * even on a field of the smallest sprites, and moved patches. */
        const double fx = x[i], fy = y[i];
        double render_radius = hypot((double)radius[i], (double)blur_radius[i]);
        if (render_radius < 0.5) {
            render_radius = 0.5;
        }
        const double half_smear = 0.5 * smear[i] / render_radius;
        const double reach = render_radius * (1.3 + half_smear);
        /* A NaN, an infinity or a sprite of absurd size: decline, and let
         * the numpy code report it as it always has. Sprites already drawn
         * do no harm, since drawing one twice is drawing it once. */
        if (!(reach < WM_MAX_REACH) || !(fabs(fx) < 1e9) || !(fabs(fy) < 1e9)) {
            free(sprite), free(tmp), free(axis), free(line);
            return 1;
        }
        /* Python's round(): half to even, which is nearbyint's default. */
        ptrdiff_t size = (ptrdiff_t)nearbyint(reach * 2.0) | 1;
        if (size < 3) {
            size = 3;
        }
        const double centre = size / 2.0;
        const ptrdiff_t left = (ptrdiff_t)nearbyint(fx - centre);
        const ptrdiff_t top = (ptrdiff_t)nearbyint(fy - centre);
        const ptrdiff_t dst_x0 = left > 0 ? left : 0;
        const ptrdiff_t dst_y0 = top > 0 ? top : 0;
        const ptrdiff_t dst_x1 = left + size < width ? left + size : width;
        const ptrdiff_t dst_y1 = top + size < height ? top + size : height;
        if (dst_x1 <= dst_x0 || dst_y1 <= dst_y0) {
            continue;
        }

        if (size > capacity) {
            capacity = size + size / 2;
            free(sprite), free(tmp), free(axis), free(line);
            sprite = malloc((size_t)(capacity * capacity) * sizeof(float));
            tmp = malloc((size_t)(capacity * capacity) * sizeof(float));
            axis = malloc((size_t)capacity * sizeof(float));
            line = malloc((size_t)(3 * capacity) * sizeof(float));
            if (!sprite || !tmp || !axis || !line) {
                free(sprite), free(tmp), free(axis), free(line);
                return 1;
            }
        }

        const float centre_f = (float)centre, radius_f = (float)render_radius;
        for (ptrdiff_t k = 0; k < size; k++) {
            axis[k] = ((float)k - centre_f) / radius_f;
        }

        const float ux = dir_x[i], uy = dir_y[i];
        const float hs = (float)half_smear;

        /* The cat's-eye stop is displaced towards the frame's centre. */
        int clipped = 0;
        float clip_x = 0.0f, clip_y = 0.0f;
        if (cats_eye > 0.0) {
            const double to_x = centre_x - fx, to_y = centre_y - fy;
            const double norm = hypot(to_x, to_y);
            if (norm > 1e-3) {
                const double longest = centre_x > centre_y ? centre_x : centre_y;
                double share = norm / longest;
                if (share > 1.0) {
                    share = 1.0;
                }
                const double offset = cats_eye * share;
                clip_x = (float)(to_x / norm * offset);
                clip_y = (float)(to_y / norm * offset);
                clipped = 1;
            }
        }

        double coverage = radius[i] / render_radius;
        coverage = coverage * coverage;
        if (coverage > 1.0) {
            coverage = 1.0;
        }
        const double smear_area = 1.0 + 2.0 * half_smear / WM_PI;
        const float level = (float)((gain ? (double)gain[i] : 1.0) * coverage / smear_area);

        float support[WM_MAX_BLADES], scaled_x[WM_MAX_BLADES], scaled_y[WM_MAX_BLADES];
        for (int b = 0; b < blades && blades >= 3; b++) {
            const double reach_b =
                base_support + half_smear * fabs((double)ux * normal_x[b] + (double)uy * normal_y[b]);
            support[b] = (float)reach_b;
            scaled_x[b] = (float)(normal_x[b] / reach_b);
            scaled_y[b] = (float)(normal_y[b] / reach_b);
        }

        const int polygon = blades >= 3;
        float *row = sprite;
        for (ptrdiff_t r = 0; r < size; r++, row += size) {
            const float dy = axis[r];
            if (polygon && fast) {
                first_plane_row_fast(row, axis, size, dy, scaled_x[0], scaled_y[0]);
                for (int b = 1; b < blades; b++) {
                    next_plane_row_fast(row, axis, size, dy, scaled_x[b], scaled_y[b]);
                }
            } else if (polygon) {
                first_plane_row(row, axis, size, dy, normal_x[0], normal_y[0], support[0]);
                for (int b = 1; b < blades; b++) {
                    next_plane_row(row, axis, size, dy, normal_x[b], normal_y[b], support[b]);
                }
            } else {
                capsule_row(row, axis, size, dy, ux, uy, hs);
            }
            if (clipped) {
                stop_row(row, axis, size, dy, clip_x, clip_y);
            }
            if (fast) {
                shade_row_fast(row, size, rim, level);
            } else {
                shade_row(row, size, rim, level);
            }
        }

        int ksize = (int)(render_radius * 0.12);
        ksize = (ksize < 1 ? 1 : ksize) | 1;
        if (ksize > 1) {
            float kernel[ksize];
            gaussian_kernel(kernel, ksize);
            blur_patch(sprite, tmp, line, size, size, kernel, ksize);
        }

        /* The part of the sprite that is in the frame: where it starts in
         * each, and how wide it is, are the same for all its rows. */
        float *restrict into = canvas + dst_y0 * width + dst_x0;
        const float *restrict from = sprite + (dst_y0 - top) * size + (dst_x0 - left);
        const ptrdiff_t visible = dst_x1 - dst_x0;
        for (ptrdiff_t r = dst_y0; r < dst_y1; r++, into += width, from += size) {
            max_row(into, from, visible);
        }
    }
    free(sprite), free(tmp), free(axis), free(line);
    return 0;
}

/* ------------------------------------------------------------ streaks */

/** A streak is a polyline that cv2.polylines draws into a small patch. What
 * follows sets the pixels OpenCV sets, by the arithmetic OpenCV uses
 * (modules/imgproc/src/drawing.cpp: Line, Line2, FillConvexPoly, Circle,
 * ThickLine, PolyLine), for the case this library has: a float32 image, on
 * which OpenCV ignores LINE_AA and draws 8-connected. The tests draw
 * thousands of streaks both ways and compare the pixels.
 *
 * The pixels are not written as they are found. They are collected, each
 * once, and the streak's brightness is laid down afterwards: drawing only
 * ever sets a pixel to the same value again, and the list is all that the
 * beading and the accumulation need to visit.
 */
/**/

/** Fractional bits of the fixed-point coordinates OpenCV's wide lines are drawn in. */
#define WM_XY_SHIFT 16
/** One pixel in those fixed-point coordinates. */
#define WM_XY_ONE (1 << WM_XY_SHIFT)

/** A pixel of a patch. Sixteen bits a coordinate: the pair is the size of
 * the flat index it replaces, and reading it back needs no division. (That
 * made no difference that could be measured.) Patches are held to
 * WM_MAX_SIDE, well inside sixteen bits.
 */
typedef struct {
    int16_t x, y;
} wm_pixel;

/** The pixels a streak covers in its patch, each listed once.
 *
 * ``seen`` marks the patch's pixels already listed; ``pixels`` lists them
 * in the order they were drawn, which is the order their values are added.
 */
typedef struct {
    uint8_t *seen;
    wm_pixel *pixels;
    ptrdiff_t n_pixels;
    int32_t patch_w, patch_h;
} wm_patch;

/** List the patch pixel ``(px, py)`` if it is inside the patch and not listed yet. */
static inline void plot(wm_patch *patch, int64_t px, int64_t py) {
    if (px < 0 || px >= patch->patch_w || py < 0 || py >= patch->patch_h) {
        return;
    }
    const int32_t at = (int32_t)py * patch->patch_w + (int32_t)px;
    if (!patch->seen[at]) {
        patch->seen[at] = 1;
        patch->pixels[patch->n_pixels].x = (int16_t)px;
        patch->pixels[patch->n_pixels].y = (int16_t)py;
        patch->n_pixels++;
    }
}

/** A run of pixels along one row. What the row shares, its address and
 * whether it is in the patch at all, is settled once and not per pixel.
 */
static inline void span(wm_patch *patch, int64_t py, int64_t left, int64_t right) {
    if (py < 0 || py >= patch->patch_h) {
        return;
    }
    left = left < 0 ? 0 : left;
    right = right >= patch->patch_w ? patch->patch_w - 1 : right;
    uint8_t *restrict seen = patch->seen + (int32_t)py * patch->patch_w;
    wm_pixel *restrict next = patch->pixels + patch->n_pixels;
    const int16_t row = (int16_t)py;
    for (int64_t px = left; px <= right; px++) {
        if (!seen[px]) {
            seen[px] = 1;
            next->x = (int16_t)px;
            next->y = row;
            next++;
        }
    }
    patch->n_pixels = next - patch->pixels;
}

/** Line: OpenCV's LineIterator, 8-connected, left to right. Both ends must
 * lie inside the patch, which the caller checks, since OpenCV clips a line
 * before walking it and a clipped line takes other pixels.
 */
static void line8(wm_patch *patch, int32_t x1, int32_t y1, int32_t x2, int32_t y2) {
    int32_t dx = x2 - x1, dy = y2 - y1;
    int32_t step_x = 1, step_y = 1;
    if (dx < 0) {
        dx = -dx, dy = -dy;
        x1 = x2, y1 = y2;
    }
    if (dy < 0) {
        dy = -dy, step_y = -1;
    }
    const int steep = dy > dx;
    if (steep) {
        int32_t t = dx;
        dx = dy, dy = t;
    }
    int32_t err = dx - (dy + dy);
    const int32_t plus = dx + dx, minus = -(dy + dy);
    int32_t px = x1, py = y1;
    for (int32_t i = 0; i <= dx; i++) {
        plot(patch, px, py);
        const int turn = err < 0;
        err += minus + (turn ? plus : 0);
        if (steep) {
            py += step_y;
            px += turn ? step_x : 0;
        } else {
            px += step_x;
            py += turn ? step_y : 0;
        }
    }
}

/** Line2: a line between points in 16.16 fixed point, the outline of a
 * filled polygon.
 */
static void line2(wm_patch *patch, int64_t x1, int64_t y1, int64_t x2, int64_t y2) {
    int64_t dx = x2 - x1, dy = y2 - y1;
    const int64_t ax = dx < 0 ? -dx : dx, ay = dy < 0 ? -dy : dy;
    int64_t x_step, y_step, count;
    if (ax > ay) {
        if (dx < 0) {
            int64_t t = x1;
            x1 = x2, x2 = t;
            t = y1, y1 = y2, y2 = t;
            dy = -dy;
        }
        x_step = WM_XY_ONE;
        y_step = dy * WM_XY_ONE / (ax | 1);
        count = (x2 - x1) >> WM_XY_SHIFT;
    } else {
        if (dy < 0) {
            int64_t t = x1;
            x1 = x2, x2 = t;
            t = y1, y1 = y2, y2 = t;
            dx = -dx;
        }
        x_step = dx * WM_XY_ONE / (ay | 1);
        y_step = WM_XY_ONE;
        count = (y2 - y1) >> WM_XY_SHIFT;
    }
    x1 += WM_XY_ONE >> 1;
    y1 += WM_XY_ONE >> 1;
    plot(patch, (x2 + (WM_XY_ONE >> 1)) >> WM_XY_SHIFT, (y2 + (WM_XY_ONE >> 1)) >> WM_XY_SHIFT);
    if (ax > ay) {
        x1 >>= WM_XY_SHIFT;
        for (; count >= 0; count--) {
            plot(patch, x1, y1 >> WM_XY_SHIFT);
            x1++;
            y1 += y_step;
        }
    } else {
        y1 >>= WM_XY_SHIFT;
        for (; count >= 0; count--) {
            plot(patch, x1 >> WM_XY_SHIFT, y1);
            x1 += x_step;
            y1++;
        }
    }
}

/** FillConvexPoly for four vertices in 16.16 fixed point, not antialiased. */
static void fill_quad(wm_patch *patch, const int64_t vx[4], const int64_t vy[4]) {
    enum { N = 4 };
    const int64_t delta = WM_XY_ONE >> 1;
    struct {
        int idx, di;
        int64_t x, dx;
        int64_t ye;
    } edge[2];
    int imin = 0, edges = N;
    int64_t ymin = vy[0], ymax = vy[0];
    int64_t px = vx[N - 1], py = vy[N - 1];
    for (int i = 0; i < N; i++) {
        if (vy[i] < ymin) {
            ymin = vy[i];
            imin = i;
        }
        ymax = vy[i] > ymax ? vy[i] : ymax;
        line2(patch, px, py, vx[i], vy[i]);
        px = vx[i], py = vy[i];
    }
    ymin = (ymin + delta) >> WM_XY_SHIFT;
    ymax = (ymax + delta) >> WM_XY_SHIFT;
    if (ymax < 0 || ymin >= patch->patch_h) {
        return;
    }
    ymax = ymax < patch->patch_h - 1 ? ymax : patch->patch_h - 1;
    int64_t y = ymin;
    edge[0].idx = edge[1].idx = imin;
    edge[0].ye = edge[1].ye = y;
    edge[0].di = 1;
    edge[1].di = N - 1;
    edge[0].x = edge[1].x = -WM_XY_ONE;
    edge[0].dx = edge[1].dx = 0;
    do {
        for (int i = 0; i < 2; i++) {
            if (y >= edge[i].ye) {
                int idx0 = edge[i].idx;
                const int di = edge[i].di;
                int idx = idx0 + di;
                idx -= idx >= N ? N : 0;
                for (; edges-- > 0;) {
                    const int64_t ty = (vy[idx] + delta) >> WM_XY_SHIFT;
                    if (ty > y) {
                        const int64_t xs = vx[idx0], xe = vx[idx];
                        edge[i].ye = ty;
                        edge[i].dx = ((xe - xs) * 2 + (ty - y)) / (2 * (ty - y));
                        edge[i].x = xs;
                        edge[i].idx = idx;
                        break;
                    }
                    idx0 = idx;
                    idx += di;
                    idx -= idx >= N ? N : 0;
                }
            }
        }
        if (edges < 0) {
            break;
        }
        if (y >= 0) {
            const int left = edge[0].x > edge[1].x;
            span(
                patch, y, (edge[left].x + delta) >> WM_XY_SHIFT,
                (edge[!left].x + delta) >> WM_XY_SHIFT);
        }
        edge[0].x += edge[0].dx;
        edge[1].x += edge[1].dx;
    } while (++y <= ymax);
}

/** Circle, filled: the round cap at the end of a wide segment. */
static void disc(wm_patch *patch, int64_t cx, int64_t cy, int64_t radius) {
    int64_t err = 0, dx = radius, dy = 0, plus = 1, minus = (radius << 1) - 1;
    while (dx >= dy) {
        span(patch, cy - dy, cx - dx, cx + dx);
        span(patch, cy + dy, cx - dx, cx + dx);
        span(patch, cy - dx, cx - dy, cx + dy);
        span(patch, cy + dx, cx - dy, cx + dy);
        dy++;
        err += plus;
        plus += 2;
        if (err > 0) {
            err -= minus;
            dx--;
            minus -= 2;
        }
    }
}

/** ThickLine for a thickness above one: a filled rectangle about the segment
 * and a disc at each end `caps` asks for (bit 0 the start, bit 1 the end).
 */
static void wide_line(
    wm_patch *patch, int32_t x1, int32_t y1, int32_t x2, int32_t y2, int thickness, int caps) {
    int64_t p0x = (int64_t)x1 * WM_XY_ONE, p0y = (int64_t)y1 * WM_XY_ONE;
    const int64_t p1x = (int64_t)x2 * WM_XY_ONE, p1y = (int64_t)y2 * WM_XY_ONE;
    const double dx = (double)(p0x - p1x) * (1.0 / WM_XY_ONE);
    const double dy = (double)(p1y - p0y) * (1.0 / WM_XY_ONE);
    double r = dx * dx + dy * dy;
    const int odd = thickness & 1;
    const int64_t half = (int64_t)thickness << (WM_XY_SHIFT - 1);
    if (fabs(r) > 2.2204460492503131e-16) {
        r = ((double)half + odd * WM_XY_ONE * 0.5) / sqrt(r);
        /* cvRound: to nearest, halves to even. */
        const int64_t ox = (int64_t)nearbyint(dy * r), oy = (int64_t)nearbyint(dx * r);
        const int64_t vx[4] = {p0x + ox, p0x - ox, p1x - ox, p1x + ox};
        const int64_t vy[4] = {p0y + oy, p0y - oy, p1y - oy, p1y + oy};
        fill_quad(patch, vx, vy);
    }
    const int64_t radius = (half + (WM_XY_ONE >> 1)) >> WM_XY_SHIFT;
    for (int end = 0; end < 2; end++) {
        if (caps & (end + 1)) {
            disc(
                patch, (p0x + (WM_XY_ONE >> 1)) >> WM_XY_SHIFT,
                (p0y + (WM_XY_ONE >> 1)) >> WM_XY_SHIFT, radius);
        }
        p0x = p1x, p0y = p1y;
    }
}

/** The longest side of a patch, in pixels, and the widest blur the kernel
 * will draw.
 */
#define WM_MAX_SIDE 8192
/** The widest blur, in pixels, the streak kernel draws; a wider one is left to numpy. */
#define WM_MAX_SIGMA 256.0f

/** The kernel size cv2.GaussianBlur(patch, (0, 0), sigma) takes on float32. */
static inline int streak_kernel_size(float sigma) {
    return (int)nearbyint((double)sigma * 8.0 + 1.0) | 1;
}

/** Add the streaks listed in ``select`` to ``layer`` (rain.render_streak_layer).
 *
 * Each streak's polyline is drawn into its patch as cv2.polylines draws it
 * on a float image, thin or wide; each pixel takes the streak's opacity
 * times the beading table at its place along the streak; a streak whose
 * sigma is above 0.3 is blurred in its patch first; and the patch is added
 * to the layer. Everything that could make it give up is checked before a
 * pixel is added, since streaks accumulate.
 */
static int WM_FN(render_streaks)(
    float *layer, int height, int width,
    const int32_t *select, ptrdiff_t n_select,
    const int32_t *x0, const int32_t *y0, const int32_t *patch_w, const int32_t *patch_h,
    const int32_t *vertices, int n_vertices, const int32_t *thickness,
    const float *alpha, const float *col_offset, const float *col_rate,
    const float *row_offset, const float *row_rate, const float *phase,
    const float *blur_sigma, const float *beading, int beading_size) {
    /* Everything that could make the kernel give up is checked here, before
     * a pixel is added: streaks accumulate, so a field half drawn and then
     * handed back to the numpy loop would be half drawn twice. */
    ptrdiff_t largest = 0, widest = 0;
    int blurred = 0, widest_kernel = 1;
    for (ptrdiff_t s = 0; s < n_select; s++) {
        const int32_t i = select[s];
        if (patch_w[i] <= 0 || patch_h[i] <= 0) {
            continue;
        }
        const ptrdiff_t area = (ptrdiff_t)patch_w[i] * patch_h[i];
        if (patch_w[i] > WM_MAX_SIDE || patch_h[i] > WM_MAX_SIDE) {
            return 1;
        }
        largest = area > largest ? area : largest;
        widest = patch_w[i] > widest ? patch_w[i] : widest;
        if (blur_sigma[i] > 0.3f) {
            if (!(blur_sigma[i] < WM_MAX_SIGMA)) {
                return 1;
            }
            const int ksize = streak_kernel_size(blur_sigma[i]);
            widest_kernel = ksize > widest_kernel ? ksize : widest_kernel;
            blurred = 1;
        }
    }
    if (largest <= 0) {
        return 0;
    }
    wm_patch patch;
    patch.seen = calloc((size_t)largest, 1);
    /* Each pixel of a patch is listed at most once. */
    patch.pixels = malloc((size_t)largest * sizeof(wm_pixel));
    float *soft = NULL, *tmp = NULL, *line = NULL;
    if (blurred) {
        soft = malloc((size_t)largest * sizeof(float));
        tmp = malloc((size_t)largest * sizeof(float));
        line = malloc((size_t)(widest + widest_kernel) * sizeof(float));
    }
    if (!patch.seen || !patch.pixels || (blurred && (!soft || !tmp || !line))) {
        free(patch.seen), free(patch.pixels), free(soft), free(tmp), free(line);
        return 1;
    }

    for (ptrdiff_t s = 0; s < n_select; s++) {
        const int32_t i = select[s];
        const int32_t pw = patch_w[i], ph = patch_h[i];
        patch.patch_w = pw, patch.patch_h = ph;
        patch.n_pixels = 0;
        if (pw <= 0 || ph <= 0) {
            continue;
        }
        const int32_t left = x0[i], top = y0[i];
        if (left + pw <= 0 || top + ph <= 0 || left >= width || top >= height) {
            continue;
        }
        const int32_t *v = vertices + (ptrdiff_t)i * n_vertices * 2;
        for (int k = 1; k < n_vertices; k++) {
            if (thickness[i] <= 1) {
                line8(&patch, v[2 * k - 2], v[2 * k - 1], v[2 * k], v[2 * k + 1]);
            } else {
                /* PolyLine caps the start of the first segment only. */
                wide_line(
                    &patch, v[2 * k - 2], v[2 * k - 1], v[2 * k], v[2 * k + 1], thickness[i],
                    k == 1 ? 3 : 2);
            }
        }

        const float a = alpha[i];
        const float co = col_offset[i], cr = col_rate[i];
        const float ro = row_offset[i], rr = row_rate[i], phi = phase[i];
        const int soften = blur_sigma[i] > 0.3f;
        if (soften) {
            memset(soft, 0, (size_t)pw * (size_t)ph * sizeof(float));
        }
        /* Where the patch's corner is in the frame, taken once: a pixel's
         * address is then its own two coordinates away. */
        float *const corner = layer + (ptrdiff_t)top * width + left;
        const wm_pixel *restrict listed = patch.pixels;
        const ptrdiff_t n_listed = patch.n_pixels;
        for (ptrdiff_t k = 0; k < n_listed; k++) {
            const int32_t px = listed[k].x, py = listed[k].y;
            const int32_t at = py * pw + px;
            patch.seen[at] = 0;
            const int32_t fx = left + px, fy = top + py;
            const int in_frame =
                (uint32_t)fx < (uint32_t)width && (uint32_t)fy < (uint32_t)height;
            if (!soften && !in_frame) {
                continue;
            }
            /* Where along the streak this pixel is, in table entries: the
             * float32 sum numpy makes of a row term and a column term. */
            const float row_term = ((float)py + ro) * rr + phi;
            const float col_term = ((float)px + co) * cr;
            const float where = row_term + col_term;
            int32_t entry = 0;
            if (where >= (float)(beading_size - 1)) {
                entry = beading_size - 1;
            } else if (where > 0.0f) {
                entry = (int32_t)where;
            }
            const float value = a * beading[entry];
            if (soften) {
                soft[at] = value;
            } else {
                corner[(ptrdiff_t)py * width + px] += value;
            }
        }
        if (soften) {
            const double sigma = blur_sigma[i];
            const int ksize = streak_kernel_size(blur_sigma[i]);
            float kernel[ksize];
            gaussian_kernel_sigma(kernel, ksize, sigma);
            blur_patch(soft, tmp, line, pw, ph, kernel, ksize);
            const int32_t c0 = left < 0 ? -left : 0, r0 = top < 0 ? -top : 0;
            const int32_t c1 = left + pw > width ? width - left : pw;
            const int32_t r1 = top + ph > height ? height - top : ph;
            float *restrict dst = corner + (ptrdiff_t)r0 * width + c0;
            const float *restrict src = soft + (ptrdiff_t)r0 * pw + c0;
            const int32_t visible = c1 - c0;
            for (int32_t r = r0; r < r1; r++, dst += width, src += pw) {
                for (int32_t c = 0; c < visible; c++) {
                    dst[c] += src[c];
                }
            }
        }
    }
    free(patch.seen), free(patch.pixels), free(soft), free(tmp), free(line);
    return 0;
}

/* ----------------------------------------------------------- wet road */

/** One row of the wet road: the darkened surface, the mirror image over it,
 * and the blend with the dry frame, for values that have each been given
 * their pixel's weights. The numpy code this replaces is nine expressions
 * over the frame, each with a frame-sized result; the arithmetic and its
 * order are that code's, in fast mode, where the cube is two
 * multiplications.
 */
static WM_NOINLINE void wet_row(
    float *restrict out, const float *restrict dry, const float *restrict sharp,
    const float *restrict rough, const float *restrict water, const float *restrict mirror,
    const float *restrict wetness, ptrdiff_t n, float ratio, float spare) {
    WM_VECTORISED for (ptrdiff_t i = 0; i < n; i++) {
        const float d = dry[i];
        const float c = clamp01(d);
        float kept = 1.0f - spare * (c * c * c);
        kept = kept > 1e-6f ? kept : 1e-6f;
        const float albedo = ratio * c / kept;
        const float mirrored = sharp[i] * water[i] + rough[i] * (1.0f - water[i]);
        const float wet = albedo * (1.0f - mirror[i]) + mirrored * mirror[i];
        out[i] = d * (1.0f - wetness[i]) + wet * wetness[i];
    }
}

/** The wet road over the dry frame, row by row (rain._apply_ground_wetness, fast mode).
 *
 * Each pixel's weights, how much standing water and how much mirror it
 * takes and how wet it is, are written out once per channel, and
 * ``wet_row`` blends the row. ``spare`` is ``1 - ratio`` taken in double by
 * the caller, as numpy takes it.
 */
static int WM_FN(wet_composite)(
    float *out, const float *dry, const float *sharp, const float *rough,
    const float *wetness, const float *puddles, const float *fresnel,
    const float *film_fresnel, ptrdiff_t rows, ptrdiff_t width,
    float ratio, float spare, float film, float gloss) {
    /* `spare` is one minus `ratio`, taken by the caller in double as numpy
     * takes it: taken here in float it is a rounding off, often enough to
     * show. A pixel's weights are written out once per channel so that a
     * row is one loop over its values. */
    const ptrdiff_t n = 3 * width;
    float *scratch = malloc((size_t)(3 * n) * sizeof(float));
    if (!scratch) {
        return 1;
    }
    float *restrict water = scratch, *restrict mirror = scratch + n, *restrict wet = scratch + 2 * n;
    const float *pool = puddles, *soaked = wetness;
    for (ptrdiff_t r = 0; r < rows; r++, soaked += width, out += n, dry += n, sharp += n, rough += n) {
        const float flat = fresnel[r], tilted = film_fresnel[r];
        float *restrict to_water = water, *restrict to_mirror = mirror, *restrict to_wet = wet;
        for (ptrdiff_t x = 0; x < width; x++, to_water += 3, to_mirror += 3, to_wet += 3) {
            const float standing = pool ? pool[x] * film : 0.0f;
            const float weight = flat * standing + tilted * (1.0f - standing) * film * gloss;
            const float share = soaked[x];
            to_water[0] = to_water[1] = to_water[2] = standing;
            to_mirror[0] = to_mirror[1] = to_mirror[2] = weight;
            to_wet[0] = to_wet[1] = to_wet[2] = share;
        }
        if (pool) {
            pool += width;
        }
        wet_row(out, dry, sharp, rough, water, mirror, wet, n, ratio, spare);
    }
    free(scratch);
    return 0;
}

/** Where a wet pixel's reflection comes from (rain._mirrored_rows): for
 * each pixel, the lowest solid row at or above it, the "foot" of whatever
 * stands there, and whether that thing is compact enough and the pixel near
 * enough below it to mirror it about its foot. One sweep down the frame with
 * a few integers of state per column, where numpy made six frame-sized
 * arrays. `limit` is the tallest run of solid rows that counts as compact.
 */
static int WM_FN(mirror_rows)(
    const uint8_t *solid, ptrdiff_t height, ptrdiff_t width, double limit,
    float *about_foot, float *in_reach) {
    int32_t *state = malloc((size_t)(3 * width) * sizeof(int32_t));
    uint8_t *was_solid = calloc((size_t)width, 1);
    if (!state || !was_solid) {
        free(state), free(was_solid);
        return 1;
    }
    int32_t *restrict foot = state, *restrict run = state + width, *restrict top = state + 2 * width;
    for (ptrdiff_t x = 0; x < 3 * width; x++) {
        state[x] = -1;
    }
    /* A run is compact if it is shorter than `limit` rows: the smallest
     * whole number of rows that is not, for comparing whole numbers with. */
    const int32_t tallest = (int32_t)ceil(limit);
    const uint8_t *restrict row = solid;
    float *restrict mirrored = about_foot;
    float *restrict reach = in_reach;
    for (ptrdiff_t y = 0; y < height; y++, row += width, mirrored += width, reach += width) {
        const int32_t here = (int32_t)y;
        for (ptrdiff_t x = 0; x < width; x++) {
            if (row[x]) {
                if (!was_solid[x]) {
                    run[x] = here; /* a run of solid rows starts here */
                }
                foot[x] = here;
                top[x] = run[x];
            }
            was_solid[x] = row[x];
            /* 2 (foot + 1/2) - y is a whole number, so it is kept as one:
             * no double in the loop, and the same value. */
            const int32_t about = 2 * foot[x] + 1 - here;
            const int compact = foot[x] >= 0 && top[x] > 0 && foot[x] - top[x] < tallest;
            mirrored[x] = (float)about;
            reach[x] = compact && about >= top[x] ? 1.0f : 0.0f;
        }
    }
    free(state), free(was_solid);
    return 0;
}

/** A row of an image at a fractional height: the two rows about it and
 * their weights, the lower one's being the fraction. (OpenCV's warp does
 * the same, with the fraction rounded to a 32nd before version 5 and not
 * since, which is why this is defined here and not copied from it.)
 */
typedef struct {
    const float *upper, *lower;
    float w_upper, w_lower;
} wm_row_pair;

/** The two rows about the fractional row ``row``, clamped to the image, and their weights. */
static inline wm_row_pair row_pair(
    const float *image, ptrdiff_t height, ptrdiff_t stride, float row) {
    const float last = (float)(height - 1);
    row = row > 0.0f ? row : 0.0f; /* a NaN lands on the first row */
    row = row < last ? row : last;
    const float below = floorf(row);
    const ptrdiff_t whole = (ptrdiff_t)below;
    const ptrdiff_t next = whole + 1 < height ? whole + 1 : height - 1;
    wm_row_pair pair;
    pair.w_lower = row - below;
    pair.w_upper = 1.0f - pair.w_lower;
    pair.upper = image + whole * stride;
    pair.lower = image + next * stride;
    return pair;
}

/** The mirror image a wet road shows (rain._apply_ground_wetness): each row
 * mirrored about the horizon, fading to the sky's radiance where it would
 * look above the frame, and under a compact object mirrored about the
 * object's foot instead. Two warps of the frame and three blends, as one
 * pass. `under` may be NULL.
 */
static void WM_FN(wet_reflection)(
    float *out, const float *source, ptrdiff_t height, ptrdiff_t width, ptrdiff_t rows,
    const float *about_horizon, const float *off_frame, const float *about_foot,
    const float *under, const float *sky) {
    const ptrdiff_t stride = 3 * width;
    float *restrict row = out;
    const float *weight = under, *foot = about_foot;
    for (ptrdiff_t r = 0; r < rows; r++, row += stride, weight += width, foot += width) {
        const wm_row_pair far = row_pair(source, height, stride, about_horizon[r]);
        const float *restrict upper = far.upper, *restrict lower = far.lower;
        const float w_upper = far.w_upper, w_lower = far.w_lower;
        /* The sky's share of this row, one value a channel, taken once. */
        const float off = off_frame[r], on = 1.0f - off;
        const float sky_r = sky[0] * off, sky_g = sky[1] * off, sky_b = sky[2] * off;
        for (ptrdiff_t i = 0; i < stride; i += 3) {
            row[i] = (upper[i] * w_upper + lower[i] * w_lower) * on + sky_r;
            row[i + 1] = (upper[i + 1] * w_upper + lower[i + 1] * w_lower) * on + sky_g;
            row[i + 2] = (upper[i + 2] * w_upper + lower[i + 2] * w_lower) * on + sky_b;
        }
        if (!under) {
            continue;
        }
        for (ptrdiff_t x = 0; x < width; x++) {
            const float w = weight[x];
            if (w == 0.0f) {
                continue; /* numpy's blend with a weight of nothing changes nothing */
            }
            const wm_row_pair near = row_pair(source, height, stride, foot[x]);
            for (int c = 0; c < 3; c++) {
                const ptrdiff_t i = 3 * x + c;
                const float mirrored = near.upper[i] * near.w_upper + near.lower[i] * near.w_lower;
                row[i] = row[i] * (1.0f - w) + mirrored * w;
            }
        }
    }
}

/* ------------------------------------------------- edge-preserving filter */

/** Rows ``smooth_along`` interleaves; see there. Defined for an experiment to
 * set another.
 */
#ifndef WM_TOGETHER
#define WM_TOGETHER 4
#endif

/** Rows smoothed there and back along themselves, WM_TOGETHER at a time.
 * `weight[x]` joins pixel x to pixel x - 1.
 *
 * Each step of the recurrence waits for the one before it, twelve cycles of
 * subtract, multiply and add, while the core could have done several times
 * as much. So the rows are taken a few at a time, their steps interleaved:
 * the chains are independent and run side by side. The running pixels are
 * held in local variables, so the chains run through registers and not
 * through memory. Four rows together took the tabled filter from 30 ms to
 * 21 on a 1024-pixel frame; two gave nothing, and eight no more than four.
 */
static inline void smooth_along(
    float *restrict image, const float *restrict weights, ptrdiff_t width, ptrdiff_t n,
    ptrdiff_t rows) {
    float last[WM_TOGETHER][3];
    float *q[WM_TOGETHER];
    const float *w[WM_TOGETHER];
    for (ptrdiff_t y = 0; y < rows; y += WM_TOGETHER) {
        /* A short last group works its final row more than once, which
         * is harmless only if done on a copy: so it is done one at a time. */
        const int together = rows - y >= WM_TOGETHER ? WM_TOGETHER : 1;
        for (int k = 0; k < together; k++) {
            q[k] = image + (y + k) * n;
            w[k] = weights + (y + k) * width;
            last[k][0] = q[k][0], last[k][1] = q[k][1], last[k][2] = q[k][2];
        }
        for (ptrdiff_t x = 1; x < width; x++) {
            const ptrdiff_t i = 3 * x;
            for (int k = 0; k < together; k++) {
                const float a = w[k][x];
                float *restrict pixel = q[k] + i;
                last[k][0] = pixel[0] + (last[k][0] - pixel[0]) * a;
                last[k][1] = pixel[1] + (last[k][1] - pixel[1]) * a;
                last[k][2] = pixel[2] + (last[k][2] - pixel[2]) * a;
                pixel[0] = last[k][0], pixel[1] = last[k][1], pixel[2] = last[k][2];
            }
        }
        for (ptrdiff_t x = width - 2; x >= 0; x--) {
            const ptrdiff_t i = 3 * x;
            for (int k = 0; k < together; k++) {
                const float a = w[k][x + 1];
                float *restrict pixel = q[k] + i;
                last[k][0] = pixel[0] + (last[k][0] - pixel[0]) * a;
                last[k][1] = pixel[1] + (last[k][1] - pixel[1]) * a;
                last[k][2] = pixel[2] + (last[k][2] - pixel[2]) * a;
                pixel[0] = last[k][0], pixel[1] = last[k][1], pixel[2] = last[k][2];
            }
        }
        if (together == 1) {
            y -= WM_TOGETHER - 1; /* and the next row is the one after this */
        }
    }
}

/** One row drawn towards the row before it in the sweep. `weight3` is the
 * pixel's weight written out for each of its three values, so this is one
 * loop over a row's values with nothing shared between them.
 */
static inline void smooth_towards(
    float *restrict row, const float *restrict other, const float *restrict weight3, ptrdiff_t n) {
    WM_VECTORISED for (ptrdiff_t i = 0; i < n; i++) {
        row[i] = row[i] + (other[i] - row[i]) * weight3[i];
    }
}

/** Each of a row's values written three times, once for each channel of its pixel. */
static inline void thrice(float *restrict out, const float *restrict in, ptrdiff_t width) {
    for (ptrdiff_t x = 0; x < width; x++, out += 3) {
        out[0] = out[1] = out[2] = in[x];
    }
}

/** A row of weights, each written out for its pixel's three values: from
 * the table by the row's byte sums, or as they were worked out before.
 */
static inline void thrice_tabled(
    float *restrict out, const uint16_t *restrict sums, const float *restrict table,
    ptrdiff_t width) {
    for (ptrdiff_t x = 0; x < width; x++, out += 3) {
        out[0] = out[1] = out[2] = table[sums[x]];
    }
}

/** A row of weights from the table, by the row's byte sums. */
static inline void lookup_row(
    float *restrict out, const uint16_t *restrict sums, const float *restrict table,
    ptrdiff_t width) {
    for (ptrdiff_t x = 0; x < width; x++) {
        out[x] = table[sums[x]];
    }
}

/** A row of weights as ``base`` to each exponent, by ``powf`` as OpenCV takes them. */
static inline void power_row(
    float *restrict out, const float *restrict exponents, float base, ptrdiff_t width) {
    for (ptrdiff_t x = 0; x < width; x++) {
        out[x] = powf(base, exponents[x]);
    }
}

/** The byte sums a row's weights are looked up by: how unlike each pixel is
 * from the one before it (`along`) and from the one above it (`down`). The
 * first pixel is its own neighbour and is taken apart from the loop, which
 * is then the same three differences for every pixel, with no test in it.
 */
static inline void unlikeness_bytes(
    uint16_t *restrict along, uint16_t *restrict down, const uint8_t *restrict row,
    const uint8_t *restrict above, ptrdiff_t n) {
    along[0] = 0;
    down[0] = (uint16_t)(abs(row[0] - above[0]) + abs(row[1] - above[1]) + abs(row[2] - above[2]));
    const uint8_t *restrict before = row;
    row += 3, above += 3;
    for (ptrdiff_t i = 3; i < n; i += 3, row += 3, above += 3, before += 3) {
        *++along = (uint16_t)(abs(row[0] - before[0]) + abs(row[1] - before[1]) + abs(row[2] - before[2]));
        *++down = (uint16_t)(abs(row[0] - above[0]) + abs(row[1] - above[1]) + abs(row[2] - above[2]));
    }
}

/** The same as exponents, in OpenCV's arithmetic and its order of summing. */
static inline void unlikeness_exponents(
    float *restrict along, float *restrict down, const float *restrict row,
    const float *restrict above, ptrdiff_t n, float ratio) {
    for (ptrdiff_t i = 0; i < n; i += 3) {
        const float *restrict here = row + i;
        const float *restrict before = i ? here - 3 : here;
        const float *restrict over = above + i;
        float dx = 0.0f, dy = 0.0f;
        dx = dx + fabsf(here[0] - before[0]);
        dx = dx + fabsf(here[1] - before[1]);
        dx = dx + fabsf(here[2] - before[2]);
        dy = dy + fabsf(here[0] - over[0]);
        dy = dy + fabsf(here[1] - over[1]);
        dy = dy + fabsf(here[2] - over[2]);
        *along++ = 1.0f + dx * ratio;
        *down++ = 1.0f + dy * ratio;
    }
}

/** OpenCV's edgePreservingFilter with RECURS_FILTER (Gastal and Oliveira's
 * domain transform, recursive form), which TemporalDenoiseArtifacts runs one
 * to three times a frame. The arithmetic and its order are OpenCV's
 * (modules/photo/src/npr.hpp), so that with `tabled` off the output is
 * OpenCV's to the byte wherever the two share a powf and neither fuses a
 * multiply-add; the caller checks that before relying on it.
 *
 * OpenCV takes a power for every pixel, twice, in each of three iterations,
 * and that is most of its time. The exponent is one plus a multiple of the
 * summed byte differences between a pixel and its neighbour, which is one of
 * 766 values. With `tabled` the power is taken once for each of those and
 * looked up: a rounding away from the per-pixel exponent, and a level away
 * in the output for a handful of pixels in a million.
 */
static int WM_FN(edge_preserving)(
    const uint8_t *src, uint8_t *dst, ptrdiff_t height, ptrdiff_t width,
    float sigma_s, float sigma_r, int tabled) {
    const ptrdiff_t n = 3 * width, pixels = height * width, values = height * n;
    const float unit = (float)(1.0 / 255.0);
    const float ratio = sigma_s / sigma_r;
    float *image = malloc((size_t)values * sizeof(float));
    /* The weights along the rows, for the few rows smoothed together. */
    float *across = malloc((size_t)(WM_TOGETHER * width) * sizeof(float));
    float *weight3 = malloc((size_t)n * sizeof(float));
    /* What the weights are a function of: the byte sums, or the exponents
     * themselves and with them the weights down the frame, which are wanted
     * twice a pass and are a power each. */
    uint16_t *sum = tabled ? malloc((size_t)(2 * pixels) * sizeof(uint16_t)) : NULL;
    float *exponent = tabled ? NULL : malloc((size_t)(2 * pixels) * sizeof(float));
    float *down = tabled ? NULL : malloc((size_t)pixels * sizeof(float));
    if (!image || !across || !weight3 || (tabled ? !sum : !exponent || !down)) {
        free(image), free(across), free(weight3), free(sum), free(exponent), free(down);
        return 1;
    }

    {
        const uint8_t *bytes = src;
        float *row = image;
        for (ptrdiff_t y = 0; y < height; y++, bytes += n, row += n) {
            for (ptrdiff_t i = 0; i < n; i++) {
                row[i] = (float)bytes[i] * unit;
            }
            const ptrdiff_t at = y * width;
            if (tabled) {
                unlikeness_bytes(sum + at, sum + pixels + at, bytes, y ? bytes - n : bytes, n);
            } else {
                unlikeness_exponents(
                    exponent + at, exponent + pixels + at, row, y ? row - n : row, n, ratio);
            }
        }
    }

    for (int pass = 0; pass < 3; pass++) {
        const float sigma_h =
            (float)(sigma_s * sqrt(3.0) * pow(2.0, 3 - (pass + 1)) / sqrt(pow(4.0, 3) - 1));
        const float base = (float)exp((-1.0 * sqrt(2.0)) / sigma_h);
        float table[766];
        if (tabled) {
            for (int k = 0; k < 766; k++) {
                table[k] = powf(base, 1.0f + ((float)k * unit) * ratio);
            }
        } else {
            power_row(down, exponent + pixels, base, pixels);
        }

        /* Along the rows and down the frame in one sweep. A row is drawn
         * towards the one above as soon as it has been smoothed along
         * itself, while it is still in the cache; the row above had its turn
         * a moment before, so the values are those of two separate sweeps. */
        for (ptrdiff_t y = 0; y < height; y += WM_TOGETHER) {
            const ptrdiff_t group = height - y < WM_TOGETHER ? height - y : WM_TOGETHER;
            float *row = image + y * n;
            for (ptrdiff_t k = 0; k < group; k++) {
                const ptrdiff_t at = (y + k) * width;
                if (tabled) {
                    lookup_row(across + k * width, sum + at, table, width);
                } else {
                    power_row(across + k * width, exponent + at, base, width);
                }
            }
            smooth_along(row, across, width, n, group);
            for (ptrdiff_t k = 0; k < group; k++, row += n) {
                const ptrdiff_t at = (y + k) * width;
                if (y + k == 0) {
                    continue;
                }
                if (tabled) {
                    thrice_tabled(weight3, sum + pixels + at, table, width);
                } else {
                    thrice(weight3, down + at, width);
                }
                smooth_towards(row, row - n, weight3, n);
            }
        }
        /* And back up. */
        float *row = image + (height - 2) * n;
        for (ptrdiff_t y = height - 2; y >= 0; y--, row -= n) {
            const ptrdiff_t at = (y + 1) * width;
            if (tabled) {
                thrice_tabled(weight3, sum + pixels + at, table, width);
            } else {
                thrice(weight3, down + at, width);
            }
            smooth_towards(row, row + n, weight3, n);
        }
    }

    /* cv::convertScaleAbs: the magnitude, to the nearest level, saturated. */
    for (ptrdiff_t i = 0; i < values; i++) {
        const float level = nearbyintf(fabsf(image[i] * 255.0f));
        dst[i] = (uint8_t)(level > 255.0f ? 255.0f : level);
    }
    free(image), free(across), free(weight3), free(sum), free(exponent), free(down);
    return 0;
}

/* ----------------------------------------------------------- lens dirt */

/** One value a pixel written out for each of its three channels, so that
 * what follows is one loop over a block's values.
 */
static inline void thrice_block(float *restrict out, const float *restrict in, ptrdiff_t count) {
    for (ptrdiff_t i = 0; i < count; i++, out += 3) {
        out[0] = out[1] = out[2] = in[i];
    }
}

/** Each pixel's highlight gain: ``1 + boost * clip((luma - 0.35) / 0.65, 0, 1)``. */
static inline void emphasis_gains(
    float *restrict gain, const float *restrict luma, ptrdiff_t count, float boost) {
    WM_VECTORISED for (ptrdiff_t i = 0; i < count; i++) {
        /* np.clip: a NaN passes through both comparisons, as it does there. */
        float highlight = (luma[i] - 0.35f) / 0.65f;
        highlight = highlight < 0.0f ? 0.0f : highlight;
        highlight = highlight > 1.0f ? 1.0f : highlight;
        gain[i] = 1.0f + boost * highlight;
    }
}

/** ``out = in * gain``, value by value. */
static inline void scale_block(
    float *restrict out, const float *restrict in, const float *restrict gain, ptrdiff_t n) {
    WM_VECTORISED for (ptrdiff_t i = 0; i < n; i++) {
        out[i] = in[i] * gain[i];
    }
}

/** linear * (1 + boost * clip((luma - 0.35) / 0.65, 0, 1)): the scene with
 * its highlights weighted up, which is what a deposit scatters most of.
 */
static void WM_FN(dirt_emphasise)(
    float *out, const float *linear, const float *luma, ptrdiff_t pixels, float boost) {
    float gain[WM_BLOCK], gain3[3 * WM_BLOCK];
    for (ptrdiff_t done = 0; done < pixels; done += WM_BLOCK) {
        const ptrdiff_t count = pixels - done < WM_BLOCK ? pixels - done : WM_BLOCK;
        emphasis_gains(gain, luma + done, count, boost);
        thrice_block(gain3, gain, count);
        scale_block(out + 3 * done, linear + 3 * done, gain3, 3 * count);
    }
}

/** Each pixel's veil, ``clip(strength * mask, 0, 0.95)``, and the share its crust blocks. */
static inline void dirt_weights(
    float *restrict veil, float *restrict blocked, const float *restrict mask, ptrdiff_t count,
    float veil_strength, float absorption) {
    WM_VECTORISED for (ptrdiff_t i = 0; i < count; i++) {
        float v = veil_strength * mask[i];
        v = v < 0.0f ? 0.0f : v;
        veil[i] = v > 0.95f ? 0.95f : v;
        blocked[i] = absorption * mask[i];
    }
}

/** A block of the deposit over the frame: lerp towards the scattered light, then towards the crust.
 */
static WM_NOINLINE void dirt_block(
    float *restrict out, const float *restrict linear, const float *restrict spread,
    const float *restrict albedo, const float *restrict veil, const float *restrict blocked,
    const float *restrict around, ptrdiff_t n, float forward, float body) {
    WM_VECTORISED for (ptrdiff_t i = 0; i < n; i++) {
        const float arriving = spread[i] * forward + around[i];
        const float scattered = arriving * albedo[i];
        const float veiled = linear[i] * (1.0f - veil[i]) + scattered * veil[i];
        out[i] = veiled * (1.0f - blocked[i]) + (albedo[i] * body) * blocked[i];
    }
}

/** What the deposit does to the frame: a lerp towards the light it scatters
 * in, then towards its own dark body where it is caked. The numpy code is
 * eleven expressions over the frame, each with a frame-sized result.
 */
static void WM_FN(dirt_composite)(
    float *out, const float *linear, const float *spread, const float *mask,
    const float *albedo, const float *surroundings, ptrdiff_t pixels,
    float forward, float veil_strength, float absorption, float body) {
    float veil[WM_BLOCK], blocked[WM_BLOCK];
    float veil3[3 * WM_BLOCK], blocked3[3 * WM_BLOCK], around[3 * WM_BLOCK];
    /* The light from outside the frame, the same three values for every
     * pixel, laid out once in the order a block's values come in. */
    for (ptrdiff_t i = 0; i < 3 * WM_BLOCK; i += 3) {
        around[i] = surroundings[0], around[i + 1] = surroundings[1], around[i + 2] = surroundings[2];
    }
    for (ptrdiff_t done = 0; done < pixels; done += WM_BLOCK) {
        const ptrdiff_t count = pixels - done < WM_BLOCK ? pixels - done : WM_BLOCK;
        const ptrdiff_t at = 3 * done;
        dirt_weights(veil, blocked, mask + done, count, veil_strength, absorption);
        thrice_block(veil3, veil, count);
        thrice_block(blocked3, blocked, count);
        dirt_block(
            out + at, linear + at, spread + at, albedo + at, veil3, blocked3, around, 3 * count,
            forward, body);
    }
}

/* ------------------------------------------------ veils and relighting */

/** Three values repeated down a block, in the order its values come in. */
static inline void pattern_block(float *restrict out, const float *restrict three) {
    for (ptrdiff_t i = 0; i < 3 * WM_BLOCK; i += 3) {
        out[i] = three[0], out[i + 1] = three[1], out[i + 2] = three[2];
    }
}

/** A block of ``linear * t + air * (1 - t)``: the frame seen through the veil. */
static WM_NOINLINE void veil_block(
    float *restrict out, const float *restrict linear, const float *restrict through,
    const float *restrict air, ptrdiff_t n) {
    WM_VECTORISED for (ptrdiff_t i = 0; i < n; i++) {
        out[i] = linear[i] * through[i] + air[i] * (1.0f - through[i]);
    }
}

/** The veil, then the particles over it: ``veiled * (1 - alpha) + light * alpha``. */
static WM_NOINLINE void veil_and_layer_block(
    float *restrict out, const float *restrict linear, const float *restrict through,
    const float *restrict air, const float *restrict alpha, const float *restrict light,
    ptrdiff_t n) {
    WM_VECTORISED for (ptrdiff_t i = 0; i < n; i++) {
        const float veiled = linear[i] * through[i] + air[i] * (1.0f - through[i]);
        out[i] = veiled * (1.0f - alpha[i]) + light[i] * alpha[i];
    }
}

/** The numpy code this replaces is two expressions over the frame and ten
 * frame-sized results; the arithmetic and its order are that code's.
 */
static void WM_FN(veil_blend)(
    float *out, const float *linear, const float *transmission, const float *airlight,
    const float *alpha, const float *light, const float *light_map, ptrdiff_t pixels) {
    float through[3 * WM_BLOCK], cover[3 * WM_BLOCK];
    float air[3 * WM_BLOCK], lit[3 * WM_BLOCK];
    pattern_block(air, airlight);
    if (alpha && !light_map) {
        pattern_block(lit, light);
    }
    for (ptrdiff_t done = 0; done < pixels; done += WM_BLOCK) {
        const ptrdiff_t count = pixels - done < WM_BLOCK ? pixels - done : WM_BLOCK;
        const ptrdiff_t at = 3 * done;
        thrice_block(through, transmission + done, count);
        if (!alpha) {
            veil_block(out + at, linear + at, through, air, 3 * count);
            continue;
        }
        thrice_block(cover, alpha + done, count);
        veil_and_layer_block(
            out + at, linear + at, through, air, cover, light_map ? light_map + at : lit,
            3 * count);
    }
}

/** Each pixel's cloud cover and the cloud's shading by the scene's grey around it. */
static inline void cloud_weights(
    float *restrict cover, float *restrict shade, const float *restrict sky,
    const float *restrict grey, ptrdiff_t count, float cover_scale, float mean_grey) {
    WM_VECTORISED for (ptrdiff_t i = 0; i < count; i++) {
        cover[i] = sky[i] * cover_scale;
        shade[i] = 0.85f + (0.15f * grey[i]) / mean_grey;
    }
}

/** A block scaled by its gain and drawn towards its grey: the overcast's flatter light. */
static WM_NOINLINE void relight_block(
    float *restrict out, const float *restrict linear, const float *restrict gain,
    const float *restrict grey, ptrdiff_t n, float desaturate) {
    WM_VECTORISED for (ptrdiff_t i = 0; i < n; i++) {
        const float lifted = linear[i] * gain[i];
        out[i] = lifted + desaturate * (grey[i] - lifted);
    }
}

/** A block covered by cloud where the sky is: ``out * (1 - cover) + tint * shade * cover``. */
static WM_NOINLINE void cloud_block(
    float *restrict out, const float *restrict cover, const float *restrict shade,
    const float *restrict tint, ptrdiff_t n) {
    WM_VECTORISED for (ptrdiff_t i = 0; i < n; i++) {
        out[i] = out[i] * (1.0f - cover[i]) + (tint[i] * shade[i]) * cover[i];
    }
}

/** A block clipped to ``[0, 1]`` as ``np.clip`` clips it. */
static WM_NOINLINE void clip_block(float *restrict out, ptrdiff_t n) {
    WM_VECTORISED for (ptrdiff_t i = 0; i < n; i++) {
        /* np.clip: a NaN passes through both comparisons, as it does there. */
        float v = out[i];
        v = v < 0.0f ? 0.0f : v;
        out[i] = v > 1.0f ? 1.0f : v;
    }
}

/** The numpy code is eight expressions over the frame; a block of it stays
 * in the cache from the first to the last.
 */
static void WM_FN(overcast_blend)(
    float *out, const float *linear, const float *gain, const float *grey,
    const float *sky, const float *cloud, ptrdiff_t pixels,
    float desaturate, float cover_scale, float mean_grey) {
    float gain3[3 * WM_BLOCK], grey3[3 * WM_BLOCK];
    float cover[WM_BLOCK], shade[WM_BLOCK], cover3[3 * WM_BLOCK], shade3[3 * WM_BLOCK];
    float tint[3 * WM_BLOCK];
    if (sky) {
        pattern_block(tint, cloud);
    }
    for (ptrdiff_t done = 0; done < pixels; done += WM_BLOCK) {
        const ptrdiff_t count = pixels - done < WM_BLOCK ? pixels - done : WM_BLOCK;
        const ptrdiff_t at = 3 * done, n = 3 * count;
        thrice_block(gain3, gain + done, count);
        thrice_block(grey3, grey + done, count);
        relight_block(out + at, linear + at, gain3, grey3, n, desaturate);
        if (sky) {
            cloud_weights(cover, shade, sky + done, grey + done, count, cover_scale, mean_grey);
            thrice_block(cover3, cover, count);
            thrice_block(shade3, shade, count);
            cloud_block(out + at, cover3, shade3, tint, n);
        }
        clip_block(out + at, n);
    }
}

/* ------------------------------------------- masks grown by colour */

/** The arithmetic of sky.sky_to_depth_edge and ground.fill_to_edges after
 * their blurs, in their order::
 *
 *   plainly_not = clip(2 - |depth - level| / depth_tolerance, 0, 1)   (1 without depth)
 *   nearby      = nearby_sum / max(weight, least_weight)
 *   unlike      = max over channels of |linear - nearby|
 *   alike       = clip(2 - unlike / colour_tolerance, 0, 1) * (weight > some_weight)
 *   out         = max(mask * plainly_not, grown * alike * plainly_not)
 *
 * Without depth the multiplications by plainly_not are by 1.0 and change
 * nothing, which is what fill_to_edges computes. The numpy code makes a
 * frame-sized result of each step, ten of them, three with three channels.
 */
static void WM_FN(grow_alike)(
    float *out, const float *linear, const float *nearby_sum, const float *weight,
    const float *mask, const float *grown, const float *depth, ptrdiff_t pixels,
    float level, float depth_tolerance, float colour_tolerance, float least_weight,
    float some_weight) {
    const float *restrict rgb = linear, *restrict around = nearby_sum;
    for (ptrdiff_t i = 0; i < pixels; i++, rgb += 3, around += 3) {
        const float plainly_not =
            depth ? clamp01(2.0f - fabsf(depth[i] - level) / depth_tolerance) : 1.0f;
        const float w = weight[i];
        const float share = w > least_weight ? w : least_weight;
        const float d0 = fabsf(rgb[0] - around[0] / share);
        const float d1 = fabsf(rgb[1] - around[1] / share);
        const float d2 = fabsf(rgb[2] - around[2] / share);
        float unlike = d0 > d1 ? d0 : d1;
        unlike = unlike > d2 ? unlike : d2;
        const float alike =
            clamp01(2.0f - unlike / colour_tolerance) * (w > some_weight ? 1.0f : 0.0f);
        const float kept = mask[i] * plainly_not;
        const float added = grown[i] * alike * plainly_not;
        out[i] = kept > added ? kept : added;
    }
}

/* ------------------------------------------------------- water droplets */

/** Greedy placement: the numpy loop this replaces checked each candidate
 * against every drop kept so far with five array operations, a Python
 * iteration and an `any` per candidate. The arithmetic is the same, in
 * double and in the same order, so the drops kept are the same.
 */
static ptrdiff_t WM_FN(place_apart)(
    const double *x, const double *y, const double *footprint, const int64_t *order,
    ptrdiff_t candidates, ptrdiff_t count, double gap, int64_t *kept) {
    const double widen = 1.0 + gap;
    ptrdiff_t n = 0;
    for (ptrdiff_t k = 0; k < candidates && n < count; k++) {
        const int64_t i = order[k];
        const double fx = x[i], fy = y[i], fr = footprint[i];
        int clear = 1;
        for (ptrdiff_t j = 0; j < n; j++) {
            const int64_t other = kept[j];
            const double reach = (footprint[other] + fr) * widen;
            const double dx = x[other] - fx, dy = y[other] - fy;
            if (dx * dx + dy * dy < reach * reach) {
                clear = 0;
                break;
            }
        }
        if (clear) {
            kept[n++] = i;
        }
    }
    return n;
}

/** The most outline harmonics a drop may have. */
#define WM_MAX_HARMONICS 8

/** A lobe's contribution from its normalised squared distance q:
 * 1 - (1 - cut) q inside its rim, a parabola landing at zero outside.
 */
static inline float lobe_value(float q, float falloff, float reach, float skirt_scale) {
    if (q <= 1.0f) {
        return 1.0f - falloff * q;
    }
    const float limit = reach * reach;
    const float distance = sqrtf(q < limit ? q : limit);
    const float gap = reach - distance;
    return skirt_scale * (gap * gap);
}

/** Add each drop's lobe into the water's ``level`` and ``weighted`` thickness.
 *
 * For each drop, over its patch: its normalised distance, through an
 * outline that egg, sag and harmonics deform, gives its lobe; the lobe
 * adds to the level, and the lobe times the drop's thickness, pooled
 * towards its low side, to the weighted field. The harmonics' sines come
 * from a recurrence on the angle's sine and cosine, not from atan2 and sin,
 * which is why this is used in fast mode only.
 */
static void WM_FN(water_lobes)(
    float *level, float *weighted, ptrdiff_t height, ptrdiff_t width, const double *lobes,
    const double *harmonics, ptrdiff_t count, int n_harmonics, const int32_t *orders,
    double wobble_radius, double outline_level) {
    const float falloff = (float)(1.0 - outline_level);
    const double reach_d = 1.0 + outline_level / (1.0 - outline_level);
    const float reach = (float)reach_d;
    const float skirt_scale = (float)((1.0 - outline_level) * (1.0 - outline_level) / outline_level);
    if (n_harmonics > WM_MAX_HARMONICS) {
        n_harmonics = WM_MAX_HARMONICS;
    }
    for (ptrdiff_t d = 0; d < count; d++) {
        const double *lobe = lobes + 7 * d;
        const double cx = lobe[0], cy = lobe[1], radius = lobe[2], elongation = lobe[3];
        const double cap = lobe[4], egg = lobe[5], sag = lobe[6];
        const double *outline = NULL;
        if (harmonics && radius >= wobble_radius) {
            outline = harmonics + (ptrdiff_t)2 * n_harmonics * d;
        }
        double stretch = 1.0 + egg;
        float amp[WM_MAX_HARMONICS], cos_phase[WM_MAX_HARMONICS], sin_phase[WM_MAX_HARMONICS];
        if (outline) {
            double total = 0.0;
            for (int k = 0; k < n_harmonics; k++) {
                total += outline[2 * k];
                amp[k] = (float)outline[2 * k];
                cos_phase[k] = (float)cos(outline[2 * k + 1]);
                sin_phase[k] = (float)sin(outline[2 * k + 1]);
            }
            stretch += total;
        }
        const double reach_x = radius * stretch * reach_d;
        const double reach_y = reach_x * elongation;
        ptrdiff_t x0 = (ptrdiff_t)(cx - reach_x), x1 = (ptrdiff_t)(cx + reach_x) + 2;
        ptrdiff_t y0 = (ptrdiff_t)(cy - reach_y), y1 = (ptrdiff_t)(cy + reach_y) + 2;
        x0 = x0 > 0 ? x0 : 0, y0 = y0 > 0 ? y0 : 0;
        x1 = x1 < width ? x1 : width, y1 = y1 < height ? y1 : height;
        if (x1 <= x0 || y1 <= y0) {
            continue;
        }
        const float fcx = (float)cx, fcy = (float)cy, fel = (float)elongation;
        const float thickness = (float)((cap * radius) * (cap * radius));
        const float fradius = (float)radius, inv_r2 = 1.0f / (float)(radius * radius);
        const float fegg = (float)egg, fsag = (float)sag;
        for (ptrdiff_t y = y0; y < y1; y++) {
            const float dy = ((float)y - fcy) / fel;
            float *restrict lrow = level + y * width;
            float *restrict wrow = weighted + y * width;
            for (ptrdiff_t x = x0; x < x1; x++) {
                const float dx = (float)x - fcx;
                const float dsq = dx * dx + dy * dy;
                if (!outline) {
                    const float v = lobe_value(dsq * inv_r2, falloff, reach, skirt_scale);
                    lrow[x] += v;
                    wrow[x] += v * thickness;
                    continue;
                }
                /* The angle's cosine and sine from the offset; at the centre
                 * atan2 gives 0, so cosine 1 and sine 0. */
                const float r = sqrtf(dsq);
                const float c1 = r > 0.0f ? dx / r : 1.0f, s1 = r > 0.0f ? dy / r : 0.0f;
                float local = 1.0f + fegg * s1;
                float ck = c1, sk = s1;
                int order = 1;
                for (int k = 0; k < n_harmonics; k++) {
                    while (order < orders[k]) {
                        const float next_c = ck * c1 - sk * s1;
                        sk = sk * c1 + ck * s1;
                        ck = next_c;
                        order++;
                    }
                    local += amp[k] * (sk * cos_phase[k] + ck * sin_phase[k]);
                }
                local = fradius * local;
                const float v = lobe_value(dsq / (local * local), falloff, reach, skirt_scale);
                const float edge = r / local;
                const float pooled = 1.0f + fsag * s1 * (edge < 1.0f ? edge : 1.0f);
                lrow[x] += v;
                wrow[x] += v * thickness * pooled * pooled;
            }
        }
    }
}

/** out + (scattered - out) * fraction, then out * (1 - coverage) + colour:
 * the condensation film and the drops over it. numpy makes five frame-sized
 * results of these two lines, and reads the drops' colour and coverage as
 * strided slices of one four-channel array.
 */
static void WM_FN(droplet_composite)(
    float *out, const float *base, const float *scattered, const float *fraction,
    const float *layers, ptrdiff_t pixels) {
    const float *restrict b = base;
    const float *restrict s = scattered;
    const float *restrict layer = layers;
    float *restrict o = out;
    for (ptrdiff_t i = 0; i < pixels; i++, b += 3, o += 3) {
        float v0 = b[0], v1 = b[1], v2 = b[2];
        if (s) {
            const float f = fraction[i];
            v0 = v0 + (s[0] - v0) * f;
            v1 = v1 + (s[1] - v1) * f;
            v2 = v2 + (s[2] - v2) * f;
            s += 3;
        }
        if (layer) {
            const float clear = 1.0f - layer[3];
            v0 = v0 * clear + layer[0];
            v1 = v1 * clear + layer[1];
            v2 = v2 * clear + layer[2];
            layer += 4;
        }
        o[0] = v0, o[1] = v1, o[2] = v2;
    }
}

/** spider_web.splat_discs: discs of light added into `total`, a
 * (height, width) field of doubles. Disc i is centred on (x[i], y[i]), pixel
 * centres being whole numbers, with an edge a pixel wide, and holds weight[i]
 * in all: a pixel gains its coverage, clip(r + 0.5 - distance, 0, 1), times
 * weight / (pi (r^2 + 1/12)). Discs are taken in `order`. Coverage is float
 * arithmetic and the sum is double, both as the numpy code has them, so the
 * field is the same to the last bit.
 */
static void WM_FN(splat_discs)(
    double *total, ptrdiff_t height, ptrdiff_t width,
    const float *x, const float *y, const float *radius, const float *weight,
    const int64_t *order, ptrdiff_t count) {
    const float pi = 3.14159274101257324f;
    const float edge = (float)(1.0 / 12.0);
    for (ptrdiff_t k = 0; k < count; k++) {
        const int64_t i = order[k];
        const float cx = x[i], cy = y[i], r = radius[i];
        const float reach = r + 0.5f;
        const ptrdiff_t half = (ptrdiff_t)ceilf(reach);
        const ptrdiff_t column = (ptrdiff_t)rintf(cx), row = (ptrdiff_t)rintf(cy);
        const float peak = weight[i] / (pi * (r * r + edge));
        const ptrdiff_t top = row - half < 0 ? 0 : row - half;
        const ptrdiff_t bottom = row + half >= height ? height - 1 : row + half;
        const ptrdiff_t left = column - half < 0 ? 0 : column - half;
        const ptrdiff_t right = column + half >= width ? width - 1 : column + half;
        for (ptrdiff_t v = top; v <= bottom; v++) {
            const float dy = (float)v - cy;
            const float dy2 = dy * dy;
            double *restrict line = total + v * width;
            for (ptrdiff_t u = left; u <= right; u++) {
                const float dx = (float)u - cx;
                float cover = reach - sqrtf(dx * dx + dy2);
                cover = cover < 0.0f ? 0.0f : (cover > 1.0f ? 1.0f : cover);
                line[u] += (double)(cover * peak);
            }
        }
    }
}

#ifdef WM_NATIVE
/** The native tier is compiled with -march=native, for whatever the build
 * machine had. Run on a CPU without one of those instruction sets it would
 * fault, so the dispatcher asks first. The list is the sets a compiler uses
 * for the loops in this file; a build copied to a different CPU is not
 * supported beyond that, and says so in docs/native_kernels.md.
 */
int wm_native_usable(void) {
#if defined(__x86_64__) && defined(__GNUC__)
    __builtin_cpu_init();
    int usable = 1;
#ifdef __AVX__
    usable &= __builtin_cpu_supports("avx") != 0;
#endif
#ifdef __AVX2__
    usable &= __builtin_cpu_supports("avx2") != 0;
#endif
#ifdef __FMA__
    usable &= __builtin_cpu_supports("fma") != 0;
#endif
#ifdef __BMI__
    usable &= __builtin_cpu_supports("bmi") != 0;
#endif
#ifdef __BMI2__
    usable &= __builtin_cpu_supports("bmi2") != 0;
#endif
#ifdef __AVX512F__
    usable &= __builtin_cpu_supports("avx512f") != 0;
#endif
#ifdef __AVX512BW__
    usable &= __builtin_cpu_supports("avx512bw") != 0;
#endif
#ifdef __AVX512DQ__
    usable &= __builtin_cpu_supports("avx512dq") != 0;
#endif
#ifdef __AVX512VL__
    usable &= __builtin_cpu_supports("avx512vl") != 0;
#endif
    return usable;
#else
    return 1;
#endif
}
#endif

/** This tier's kernels, exported to ``wm_dispatch.c`` as ``wm_table_<tier>``. */
const wm_kernel_table WM_FN(wm_table) = {
    WM_STR(WM_TIER),
    WM_FN(encode_u8),
    WM_FN(encode_f32),
    WM_FN(decode_u8),
    WM_FN(render_flakes),
    WM_FN(render_streaks),
    WM_FN(wet_composite),
    WM_FN(mirror_rows),
    WM_FN(wet_reflection),
    WM_FN(edge_preserving),
    WM_FN(dirt_emphasise),
    WM_FN(dirt_composite),
    WM_FN(veil_blend),
    WM_FN(overcast_blend),
    WM_FN(grow_alike),
    WM_FN(place_apart),
    WM_FN(water_lobes),
    WM_FN(droplet_composite),
    WM_FN(splat_discs),
};
