/** The kernels, in plain C written for the compiler to vectorise.
 *
 * Compiled once per tier with -DSM_TIER=<name> and that tier's flags, always
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

#include "sm_kernels.h"

#ifndef SM_TIER
/** The tier this compilation is for, set by the build with ``-DSM_TIER=``. */
#define SM_TIER portable
#endif
/** A kernel's name for this tier: ``SM_FN(encode_u8)`` is ``encode_u8_avx2`` in the AVX2 object.
 *
 * Each tier's object defines its own copies, and only its table is exported.
 */
#define SM_FN(name) SM_CAT(name, SM_TIER)
#define SM_STR_(a) #a
/** The tier's name as a string, for the table. */
#define SM_STR(a) SM_STR_(a)
/** Keeps a function out of line: inlined into a caller with constant arguments, GCC lost the vectorisation of some loops (see ``shade_row``).
 */
#define SM_NOINLINE __attribute__((noinline))
/** Marks a loop the build is checked to have vectorised (test_native_build.py). */
#define SM_VECTORISED

/** ``v`` clamped to ``[0, 1]``; a NaN comes out as 0. */
static inline float clamp01(float v) {
    /* NaN compares false both times and comes out as 0. */
    v = v > 0.0f ? v : 0.0f;
    return v < 1.0f ? v : 1.0f;
}

/** The most a Gaussian's exponent is taken to be: past it, it is 1e-7 of
 * its peak and is drawn as that.
 */
#define SM_DEEPEST 16.0f

/** exp(-q) for q of 0 and up, without a call: what a loop of Gaussians
 * needs to be vectorised, the build having no vector `exp`. exp(-q) is
 * exp(-q / 32) raised to the 32nd power, by five squarings, and exp(-t)
 * for t in [0, 1/2] is its series to the seventh power, good to 1e-7:
 * with the squarings' own rounding, a few millionths of the value. Past
 * `SM_DEEPEST` it is what it is there, 1e-7.
 */
static inline float exp_of_minus(float q) {
    const float c2 = 1.0f / 2.0f, c3 = (float)(1.0 / 6.0), c4 = (float)(1.0 / 24.0);
    const float c5 = (float)(1.0 / 120.0), c6 = (float)(1.0 / 720.0);
    const float c7 = (float)(1.0 / 5040.0);
    q = q < SM_DEEPEST ? q : SM_DEEPEST;
    const float t = q * (1.0f / 32.0f);
    float e = 1.0f - t * (1.0f - t * (c2 - t * (c3 - t * (c4 - t * (c5 - t * (c6 - t * c7))))));
    e *= e;
    e *= e;
    e *= e;
    e *= e;
    return e * e;
}

/* ------------------------------------------------------------ colour */

/** Values a colour kernel converts at a time: their indices fit on the stack. */
#define SM_BLOCK 512

/** Table indices for a block of values. Kept apart from the lookups so that
 * it is one loop of float arithmetic and nothing else: written as a single
 * loop with the lookup in it, GCC compiled the clamp as two branches, which
 * a picture's highlights and shadows mispredict, and vectorised nothing.
 * Returns nonzero if a value was NaN.
 */
static inline int table_indices(const float *restrict in, int32_t *restrict index, size_t n) {
    int bad = 0;
    SM_VECTORISED for (size_t i = 0; i < n; i++) {
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
static int SM_FN(encode_u8)(
    const float *restrict in, uint8_t *restrict out, size_t n, const uint8_t *restrict table) {
    int32_t index[SM_BLOCK];
    int bad = 0;
    for (size_t done = 0; done < n; done += SM_BLOCK) {
        const size_t count = n - done < SM_BLOCK ? n - done : SM_BLOCK;
        bad |= table_indices(in + done, index, count);
        uint8_t *restrict block = out + done;
        for (size_t i = 0; i < count; i++) {
            block[i] = table[index[i]];
        }
    }
    return bad;
}

/** As ``encode_u8``, looking up float32 sRGB values in ``[0, 1]``. */
static int SM_FN(encode_f32)(
    const float *restrict in, float *restrict out, size_t n, const float *restrict table) {
    int32_t index[SM_BLOCK];
    int bad = 0;
    for (size_t done = 0; done < n; done += SM_BLOCK) {
        const size_t count = n - done < SM_BLOCK ? n - done : SM_BLOCK;
        bad |= table_indices(in + done, index, count);
        float *restrict block = out + done;
        for (size_t i = 0; i < count; i++) {
            block[i] = table[index[i]];
        }
    }
    return bad;
}

/** 8-bit sRGB to linear light: each byte looked up in a 256-entry table. */
static void SM_FN(decode_u8)(
    const uint8_t *restrict in, float *restrict out, size_t n, const float *restrict table) {
    for (size_t i = 0; i < n; i++) {
        out[i] = table[in[i]];
    }
}

/* ------------------------------------------------------------- flakes */

/** The most aperture blades a polygonal sprite may have. */
#define SM_MAX_BLADES 32
/** Half the side, in pixels, of the largest sprite the kernel will draw. */
#define SM_MAX_REACH 4096.0
static const double SM_PI = 3.14159265358979323846;

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
 * and tests/unit/test_native_build.py fails if a loop marked SM_VECTORISED
 * stops being vectorised. */
/** Distance to a segment: a circle swept along its motion is a capsule. */
static inline void capsule_row(
    float *restrict row, const float *restrict dx, ptrdiff_t size,
    float dy, float ux, float uy, float hs) {
    const float dyuy = dy * uy;
    SM_VECTORISED for (ptrdiff_t c = 0; c < size; c++) {
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
    SM_VECTORISED for (ptrdiff_t c = 0; c < size; c++) {
        row[c] = (dx[c] * nx + dy * ny) / support;
    }
}

/** As ``first_plane_row``, keeping the larger of the row and this edge's distance. */
static inline void next_plane_row(
    float *restrict row, const float *restrict dx, ptrdiff_t size,
    float dy, float nx, float ny, float support) {
    SM_VECTORISED for (ptrdiff_t c = 0; c < size; c++) {
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
    SM_VECTORISED for (ptrdiff_t c = 0; c < size; c++) {
        row[c] = dx[c] * nx + dyny;
    }
}

/** As ``first_plane_row_fast``, keeping the larger of the row and this edge's distance. */
static inline void next_plane_row_fast(
    float *restrict row, const float *restrict dx, ptrdiff_t size, float dy, float nx, float ny) {
    const float dyny = dy * ny;
    SM_VECTORISED for (ptrdiff_t c = 0; c < size; c++) {
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
    SM_VECTORISED for (ptrdiff_t c = 0; c < size; c++) {
        const float ex = dx[c] - clip_x;
        const float stop = sqrtf(ex * ex + ey2);
        row[c] = row[c] > stop ? row[c] : stop;
    }
}

/** Distance to brightness: a defined rim, then the aberration profile. */
static SM_NOINLINE void shade_row(float *restrict row, ptrdiff_t size, float rim, float level) {
    SM_VECTORISED for (ptrdiff_t c = 0; c < size; c++) {
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
static SM_NOINLINE void shade_row_fast(
    float *restrict row, ptrdiff_t size, float rim, float level) {
    const float per_edge = 1.0f / 0.12f, per_core = 1.0f / 0.35f;
    SM_VECTORISED for (ptrdiff_t c = 0; c < size; c++) {
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
    SM_VECTORISED for (ptrdiff_t c = 0; c < size; c++) {
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
static int SM_FN(render_flakes)(
    float *canvas, int height, int width, ptrdiff_t count,
    const float *x, const float *y, const float *radius, const float *blur_radius,
    const float *smear, const float *dir_x, const float *dir_y, const float *gain,
    int blades, double blade_rotation, double rim_gain, double cats_eye,
    double centre_x, double centre_y, int fast) {
    float normal_x[SM_MAX_BLADES], normal_y[SM_MAX_BLADES];
    double base_support = 0.0;
    if (blades > SM_MAX_BLADES) {
        blades = SM_MAX_BLADES;
    }
    if (blades >= 3) {
        base_support = cos(SM_PI / blades);
        for (int b = 0; b < blades; b++) {
            double angle = blade_rotation + 2.0 * SM_PI * b / blades;
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
        if (!(reach < SM_MAX_REACH) || !(fabs(fx) < 1e9) || !(fabs(fy) < 1e9)) {
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
        const double smear_area = 1.0 + 2.0 * half_smear / SM_PI;
        const float level = (float)((gain ? (double)gain[i] : 1.0) * coverage / smear_area);

        float support[SM_MAX_BLADES], scaled_x[SM_MAX_BLADES], scaled_y[SM_MAX_BLADES];
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
#define SM_XY_SHIFT 16
/** One pixel in those fixed-point coordinates. */
#define SM_XY_ONE (1 << SM_XY_SHIFT)

/** A pixel of a patch. Sixteen bits a coordinate: the pair is the size of
 * the flat index it replaces, and reading it back needs no division. (That
 * made no difference that could be measured.) Patches are held to
 * SM_MAX_SIDE, well inside sixteen bits.
 */
typedef struct {
    int16_t x, y;
} sm_pixel;

/** The pixels a streak covers in its patch, each listed once.
 *
 * ``seen`` marks the patch's pixels already listed; ``pixels`` lists them
 * in the order they were drawn, which is the order their values are added.
 */
typedef struct {
    uint8_t *seen;
    sm_pixel *pixels;
    ptrdiff_t n_pixels;
    int32_t patch_w, patch_h;
} sm_patch;

/** List the patch pixel ``(px, py)`` if it is inside the patch and not listed yet. */
static inline void plot(sm_patch *patch, int64_t px, int64_t py) {
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
static inline void span(sm_patch *patch, int64_t py, int64_t left, int64_t right) {
    if (py < 0 || py >= patch->patch_h) {
        return;
    }
    left = left < 0 ? 0 : left;
    right = right >= patch->patch_w ? patch->patch_w - 1 : right;
    uint8_t *restrict seen = patch->seen + (int32_t)py * patch->patch_w;
    sm_pixel *restrict next = patch->pixels + patch->n_pixels;
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
static void line8(sm_patch *patch, int32_t x1, int32_t y1, int32_t x2, int32_t y2) {
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
static void line2(sm_patch *patch, int64_t x1, int64_t y1, int64_t x2, int64_t y2) {
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
        x_step = SM_XY_ONE;
        y_step = dy * SM_XY_ONE / (ax | 1);
        count = (x2 - x1) >> SM_XY_SHIFT;
    } else {
        if (dy < 0) {
            int64_t t = x1;
            x1 = x2, x2 = t;
            t = y1, y1 = y2, y2 = t;
            dx = -dx;
        }
        x_step = dx * SM_XY_ONE / (ay | 1);
        y_step = SM_XY_ONE;
        count = (y2 - y1) >> SM_XY_SHIFT;
    }
    x1 += SM_XY_ONE >> 1;
    y1 += SM_XY_ONE >> 1;
    plot(patch, (x2 + (SM_XY_ONE >> 1)) >> SM_XY_SHIFT, (y2 + (SM_XY_ONE >> 1)) >> SM_XY_SHIFT);
    if (ax > ay) {
        x1 >>= SM_XY_SHIFT;
        for (; count >= 0; count--) {
            plot(patch, x1, y1 >> SM_XY_SHIFT);
            x1++;
            y1 += y_step;
        }
    } else {
        y1 >>= SM_XY_SHIFT;
        for (; count >= 0; count--) {
            plot(patch, x1 >> SM_XY_SHIFT, y1);
            x1 += x_step;
            y1++;
        }
    }
}

/** FillConvexPoly for four vertices in 16.16 fixed point, not antialiased. */
static void fill_quad(sm_patch *patch, const int64_t vx[4], const int64_t vy[4]) {
    enum { N = 4 };
    const int64_t delta = SM_XY_ONE >> 1;
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
    ymin = (ymin + delta) >> SM_XY_SHIFT;
    ymax = (ymax + delta) >> SM_XY_SHIFT;
    if (ymax < 0 || ymin >= patch->patch_h) {
        return;
    }
    ymax = ymax < patch->patch_h - 1 ? ymax : patch->patch_h - 1;
    int64_t y = ymin;
    edge[0].idx = edge[1].idx = imin;
    edge[0].ye = edge[1].ye = y;
    edge[0].di = 1;
    edge[1].di = N - 1;
    edge[0].x = edge[1].x = -SM_XY_ONE;
    edge[0].dx = edge[1].dx = 0;
    do {
        for (int i = 0; i < 2; i++) {
            if (y >= edge[i].ye) {
                int idx0 = edge[i].idx;
                const int di = edge[i].di;
                int idx = idx0 + di;
                idx -= idx >= N ? N : 0;
                for (; edges-- > 0;) {
                    const int64_t ty = (vy[idx] + delta) >> SM_XY_SHIFT;
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
                patch, y, (edge[left].x + delta) >> SM_XY_SHIFT,
                (edge[!left].x + delta) >> SM_XY_SHIFT);
        }
        edge[0].x += edge[0].dx;
        edge[1].x += edge[1].dx;
    } while (++y <= ymax);
}

/** Circle, filled: the round cap at the end of a wide segment. */
static void disc(sm_patch *patch, int64_t cx, int64_t cy, int64_t radius) {
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
    sm_patch *patch, int32_t x1, int32_t y1, int32_t x2, int32_t y2, int thickness, int caps) {
    int64_t p0x = (int64_t)x1 * SM_XY_ONE, p0y = (int64_t)y1 * SM_XY_ONE;
    const int64_t p1x = (int64_t)x2 * SM_XY_ONE, p1y = (int64_t)y2 * SM_XY_ONE;
    const double dx = (double)(p0x - p1x) * (1.0 / SM_XY_ONE);
    const double dy = (double)(p1y - p0y) * (1.0 / SM_XY_ONE);
    double r = dx * dx + dy * dy;
    const int odd = thickness & 1;
    const int64_t half = (int64_t)thickness << (SM_XY_SHIFT - 1);
    if (fabs(r) > 2.2204460492503131e-16) {
        r = ((double)half + odd * SM_XY_ONE * 0.5) / sqrt(r);
        /* cvRound: to nearest, halves to even. */
        const int64_t ox = (int64_t)nearbyint(dy * r), oy = (int64_t)nearbyint(dx * r);
        const int64_t vx[4] = {p0x + ox, p0x - ox, p1x - ox, p1x + ox};
        const int64_t vy[4] = {p0y + oy, p0y - oy, p1y - oy, p1y + oy};
        fill_quad(patch, vx, vy);
    }
    const int64_t radius = (half + (SM_XY_ONE >> 1)) >> SM_XY_SHIFT;
    for (int end = 0; end < 2; end++) {
        if (caps & (end + 1)) {
            disc(
                patch, (p0x + (SM_XY_ONE >> 1)) >> SM_XY_SHIFT,
                (p0y + (SM_XY_ONE >> 1)) >> SM_XY_SHIFT, radius);
        }
        p0x = p1x, p0y = p1y;
    }
}

/** The longest side of a patch, in pixels, and the widest blur the kernel
 * will draw.
 */
#define SM_MAX_SIDE 8192
/** The widest blur, in pixels, the streak kernel draws; a wider one is left to numpy. */
#define SM_MAX_SIGMA 256.0f

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
static int SM_FN(render_streaks)(
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
        if (patch_w[i] > SM_MAX_SIDE || patch_h[i] > SM_MAX_SIDE) {
            return 1;
        }
        largest = area > largest ? area : largest;
        widest = patch_w[i] > widest ? patch_w[i] : widest;
        if (blur_sigma[i] > 0.3f) {
            if (!(blur_sigma[i] < SM_MAX_SIGMA)) {
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
    sm_patch patch;
    patch.seen = calloc((size_t)largest, 1);
    /* Each pixel of a patch is listed at most once. */
    patch.pixels = malloc((size_t)largest * sizeof(sm_pixel));
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
        const sm_pixel *restrict listed = patch.pixels;
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
static SM_NOINLINE void wet_row(
    float *restrict out, const float *restrict dry, const float *restrict sharp,
    const float *restrict rough, const float *restrict water, const float *restrict mirror,
    const float *restrict wetness, ptrdiff_t n, float ratio, float spare) {
    SM_VECTORISED for (ptrdiff_t i = 0; i < n; i++) {
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
static int SM_FN(wet_composite)(
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
static int SM_FN(mirror_rows)(
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
} sm_row_pair;

/** The two rows about the fractional row ``row``, clamped to the image, and their weights. */
static inline sm_row_pair row_pair(
    const float *image, ptrdiff_t height, ptrdiff_t stride, float row) {
    const float last = (float)(height - 1);
    row = row > 0.0f ? row : 0.0f; /* a NaN lands on the first row */
    row = row < last ? row : last;
    const float below = floorf(row);
    const ptrdiff_t whole = (ptrdiff_t)below;
    const ptrdiff_t next = whole + 1 < height ? whole + 1 : height - 1;
    sm_row_pair pair;
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
static void SM_FN(wet_reflection)(
    float *out, const float *source, ptrdiff_t height, ptrdiff_t width, ptrdiff_t rows,
    const float *about_horizon, const float *off_frame, const float *about_foot,
    const float *under, const float *sky) {
    const ptrdiff_t stride = 3 * width;
    float *restrict row = out;
    const float *weight = under, *foot = about_foot;
    for (ptrdiff_t r = 0; r < rows; r++, row += stride, weight += width, foot += width) {
        const sm_row_pair far = row_pair(source, height, stride, about_horizon[r]);
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
            const sm_row_pair near = row_pair(source, height, stride, foot[x]);
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
#ifndef SM_TOGETHER
#define SM_TOGETHER 4
#endif

/** Rows smoothed there and back along themselves, SM_TOGETHER at a time.
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
    float last[SM_TOGETHER][3];
    float *q[SM_TOGETHER];
    const float *w[SM_TOGETHER];
    for (ptrdiff_t y = 0; y < rows; y += SM_TOGETHER) {
        /* A short last group works its final row more than once, which
         * is harmless only if done on a copy: so it is done one at a time. */
        const int together = rows - y >= SM_TOGETHER ? SM_TOGETHER : 1;
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
            y -= SM_TOGETHER - 1; /* and the next row is the one after this */
        }
    }
}

/** One row drawn towards the row before it in the sweep. `weight3` is the
 * pixel's weight written out for each of its three values, so this is one
 * loop over a row's values with nothing shared between them.
 */
static inline void smooth_towards(
    float *restrict row, const float *restrict other, const float *restrict weight3, ptrdiff_t n) {
    SM_VECTORISED for (ptrdiff_t i = 0; i < n; i++) {
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
static int SM_FN(edge_preserving)(
    const uint8_t *src, uint8_t *dst, ptrdiff_t height, ptrdiff_t width,
    float sigma_s, float sigma_r, int tabled) {
    const ptrdiff_t n = 3 * width, pixels = height * width, values = height * n;
    const float unit = (float)(1.0 / 255.0);
    const float ratio = sigma_s / sigma_r;
    float *image = malloc((size_t)values * sizeof(float));
    /* The weights along the rows, for the few rows smoothed together. */
    float *across = malloc((size_t)(SM_TOGETHER * width) * sizeof(float));
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
        for (ptrdiff_t y = 0; y < height; y += SM_TOGETHER) {
            const ptrdiff_t group = height - y < SM_TOGETHER ? height - y : SM_TOGETHER;
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
    SM_VECTORISED for (ptrdiff_t i = 0; i < count; i++) {
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
    SM_VECTORISED for (ptrdiff_t i = 0; i < n; i++) {
        out[i] = in[i] * gain[i];
    }
}

/** linear * (1 + boost * clip((luma - 0.35) / 0.65, 0, 1)): the scene with
 * its highlights weighted up, which is what a deposit scatters most of.
 */
static void SM_FN(dirt_emphasise)(
    float *out, const float *linear, const float *luma, ptrdiff_t pixels, float boost) {
    float gain[SM_BLOCK], gain3[3 * SM_BLOCK];
    for (ptrdiff_t done = 0; done < pixels; done += SM_BLOCK) {
        const ptrdiff_t count = pixels - done < SM_BLOCK ? pixels - done : SM_BLOCK;
        emphasis_gains(gain, luma + done, count, boost);
        thrice_block(gain3, gain, count);
        scale_block(out + 3 * done, linear + 3 * done, gain3, 3 * count);
    }
}

/** Each pixel's veil, ``clip(strength * mask, 0, 0.95)``, and the share its crust blocks. */
static inline void dirt_weights(
    float *restrict veil, float *restrict blocked, const float *restrict mask, ptrdiff_t count,
    float veil_strength, float absorption) {
    SM_VECTORISED for (ptrdiff_t i = 0; i < count; i++) {
        float v = veil_strength * mask[i];
        v = v < 0.0f ? 0.0f : v;
        veil[i] = v > 0.95f ? 0.95f : v;
        blocked[i] = absorption * mask[i];
    }
}

/** A block of the deposit over the frame: lerp towards the scattered light, then towards the crust.
 */
static SM_NOINLINE void dirt_block(
    float *restrict out, const float *restrict linear, const float *restrict spread,
    const float *restrict albedo, const float *restrict veil, const float *restrict blocked,
    const float *restrict around, ptrdiff_t n, float forward, float body) {
    SM_VECTORISED for (ptrdiff_t i = 0; i < n; i++) {
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
static void SM_FN(dirt_composite)(
    float *out, const float *linear, const float *spread, const float *mask,
    const float *albedo, const float *surroundings, ptrdiff_t pixels,
    float forward, float veil_strength, float absorption, float body) {
    float veil[SM_BLOCK], blocked[SM_BLOCK];
    float veil3[3 * SM_BLOCK], blocked3[3 * SM_BLOCK], around[3 * SM_BLOCK];
    /* The light from outside the frame, the same three values for every
     * pixel, laid out once in the order a block's values come in. */
    for (ptrdiff_t i = 0; i < 3 * SM_BLOCK; i += 3) {
        around[i] = surroundings[0], around[i + 1] = surroundings[1], around[i + 2] = surroundings[2];
    }
    for (ptrdiff_t done = 0; done < pixels; done += SM_BLOCK) {
        const ptrdiff_t count = pixels - done < SM_BLOCK ? pixels - done : SM_BLOCK;
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
    for (ptrdiff_t i = 0; i < 3 * SM_BLOCK; i += 3) {
        out[i] = three[0], out[i + 1] = three[1], out[i + 2] = three[2];
    }
}

/** A block of ``linear * t + air * (1 - t)``: the frame seen through the veil. */
static SM_NOINLINE void veil_block(
    float *restrict out, const float *restrict linear, const float *restrict through,
    const float *restrict air, ptrdiff_t n) {
    SM_VECTORISED for (ptrdiff_t i = 0; i < n; i++) {
        out[i] = linear[i] * through[i] + air[i] * (1.0f - through[i]);
    }
}

/** The veil, then the particles over it: ``veiled * (1 - alpha) + light * alpha``. */
static SM_NOINLINE void veil_and_layer_block(
    float *restrict out, const float *restrict linear, const float *restrict through,
    const float *restrict air, const float *restrict alpha, const float *restrict light,
    ptrdiff_t n) {
    SM_VECTORISED for (ptrdiff_t i = 0; i < n; i++) {
        const float veiled = linear[i] * through[i] + air[i] * (1.0f - through[i]);
        out[i] = veiled * (1.0f - alpha[i]) + light[i] * alpha[i];
    }
}

/** The numpy code this replaces is two expressions over the frame and ten
 * frame-sized results; the arithmetic and its order are that code's.
 */
static void SM_FN(veil_blend)(
    float *out, const float *linear, const float *transmission, const float *airlight,
    const float *alpha, const float *light, const float *light_map, ptrdiff_t pixels) {
    float through[3 * SM_BLOCK], cover[3 * SM_BLOCK];
    float air[3 * SM_BLOCK], lit[3 * SM_BLOCK];
    pattern_block(air, airlight);
    if (alpha && !light_map) {
        pattern_block(lit, light);
    }
    for (ptrdiff_t done = 0; done < pixels; done += SM_BLOCK) {
        const ptrdiff_t count = pixels - done < SM_BLOCK ? pixels - done : SM_BLOCK;
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
    SM_VECTORISED for (ptrdiff_t i = 0; i < count; i++) {
        cover[i] = sky[i] * cover_scale;
        shade[i] = 0.85f + (0.15f * grey[i]) / mean_grey;
    }
}

/** A block scaled by its gain and drawn towards its grey: the overcast's flatter light. */
static SM_NOINLINE void relight_block(
    float *restrict out, const float *restrict linear, const float *restrict gain,
    const float *restrict grey, ptrdiff_t n, float desaturate) {
    SM_VECTORISED for (ptrdiff_t i = 0; i < n; i++) {
        const float lifted = linear[i] * gain[i];
        out[i] = lifted + desaturate * (grey[i] - lifted);
    }
}

/** A block covered by cloud where the sky is: ``out * (1 - cover) + tint * shade * cover``. */
static SM_NOINLINE void cloud_block(
    float *restrict out, const float *restrict cover, const float *restrict shade,
    const float *restrict tint, ptrdiff_t n) {
    SM_VECTORISED for (ptrdiff_t i = 0; i < n; i++) {
        out[i] = out[i] * (1.0f - cover[i]) + (tint[i] * shade[i]) * cover[i];
    }
}

/** A block clipped to ``[0, 1]`` as ``np.clip`` clips it. */
static SM_NOINLINE void clip_block(float *restrict out, ptrdiff_t n) {
    SM_VECTORISED for (ptrdiff_t i = 0; i < n; i++) {
        /* np.clip: a NaN passes through both comparisons, as it does there. */
        float v = out[i];
        v = v < 0.0f ? 0.0f : v;
        out[i] = v > 1.0f ? 1.0f : v;
    }
}

/** The numpy code is eight expressions over the frame; a block of it stays
 * in the cache from the first to the last.
 */
static void SM_FN(overcast_blend)(
    float *out, const float *linear, const float *gain, const float *grey,
    const float *sky, const float *cloud, ptrdiff_t pixels,
    float desaturate, float cover_scale, float mean_grey) {
    float gain3[3 * SM_BLOCK], grey3[3 * SM_BLOCK];
    float cover[SM_BLOCK], shade[SM_BLOCK], cover3[3 * SM_BLOCK], shade3[3 * SM_BLOCK];
    float tint[3 * SM_BLOCK];
    if (sky) {
        pattern_block(tint, cloud);
    }
    for (ptrdiff_t done = 0; done < pixels; done += SM_BLOCK) {
        const ptrdiff_t count = pixels - done < SM_BLOCK ? pixels - done : SM_BLOCK;
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
static void SM_FN(grow_alike)(
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
static ptrdiff_t SM_FN(place_apart)(
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
#define SM_MAX_HARMONICS 8

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
static void SM_FN(water_lobes)(
    float *level, float *weighted, ptrdiff_t height, ptrdiff_t width, const double *lobes,
    const double *harmonics, ptrdiff_t count, int n_harmonics, const int32_t *orders,
    double wobble_radius, double outline_level) {
    const float falloff = (float)(1.0 - outline_level);
    const double reach_d = 1.0 + outline_level / (1.0 - outline_level);
    const float reach = (float)reach_d;
    const float skirt_scale = (float)((1.0 - outline_level) * (1.0 - outline_level) / outline_level);
    if (n_harmonics > SM_MAX_HARMONICS) {
        n_harmonics = SM_MAX_HARMONICS;
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
        float amp[SM_MAX_HARMONICS], cos_phase[SM_MAX_HARMONICS], sin_phase[SM_MAX_HARMONICS];
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
static void SM_FN(droplet_composite)(
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

/** illuminator.splat_discs: discs of light added into `total`, a
 * (height, width) field of doubles. Disc i is centred on (x[i], y[i]), pixel
 * centres being whole numbers, with an edge a pixel wide, and holds weight[i]
 * in all: a pixel gains its coverage, clip(r + 0.5 - distance, 0, 1), times
 * weight / (pi (r^2 + 1/12)). Discs are taken in `order`. Coverage is float
 * arithmetic and the sum is double, both as the numpy code has them, so the
 * field is the same to the last bit.
 */
static void SM_FN(splat_discs)(
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

/* ------------------------------------------------- lamps on surfaces */

/** A block of night_light.street_light: one lamp's light on each of `n`
 * points, added into `light`. It falls off with the square of the
 * distance from the lamp's head and with how far the point is from
 * straight under it; a point above the lamp gets its spill.
 */
static SM_NOINLINE void street_light_block(
    float *restrict light, const float *restrict points, const float *restrict raised,
    ptrdiff_t n, float x, float y, float z, float tall, float lamp_up, float spill) {
    SM_VECTORISED for (ptrdiff_t i = 0; i < n; i++) {
        const float dx = x - points[3 * i], dy = y - points[3 * i + 1];
        const float dz = z - points[3 * i + 2];
        const float away = sqrtf(dx * dx + dy * dy + dz * dz);
        const float far = away > 0.5f ? away : 0.5f;
        const float below = clamp01((lamp_up - raised[i]) / far);
        const float reach = (tall / far) * (tall / far);
        /* The cosine to the power of one and a half: itself by its root. */
        light[i] += reach * (spill + (1.0f - spill) * (below * sqrtf(below)));
    }
}

/** night_light.street_light: every lamp's light on every point. `points`
 * is (pixels, 3) and `raised` how high each stands along `up`; `lamps`
 * is (count, 4), a head's place and its height. numpy's float arithmetic
 * in its order, but for the power of one and a half, which is a root and
 * a product here: to the last bit of a float or one off it.
 */
static void SM_FN(street_light)(
    float *light, const float *points, const float *raised, ptrdiff_t pixels,
    const float *lamps, ptrdiff_t count, const float *up, float spill) {
    for (ptrdiff_t done = 0; done < pixels; done += SM_BLOCK) {
        const ptrdiff_t n = pixels - done < SM_BLOCK ? pixels - done : SM_BLOCK;
        for (ptrdiff_t k = 0; k < count; k++) {
            const float x = lamps[4 * k], y = lamps[4 * k + 1], z = lamps[4 * k + 2];
            street_light_block(
                light + done, points + 3 * done, raised + done, n, x, y, z, lamps[4 * k + 3],
                x * up[0] + y * up[1] + z * up[2], spill);
        }
    }
}

/** A vehicle's pair of dipped beams, as night_light.headlamp_light lays it
 * on the ground: a row of its table.
 */
typedef struct {
    /** Where its front is along the ground's two axes, and which way it points along them. */
    float front_first, front_second, ahead_first, ahead_second;
    /** A metre of it in the scene's units, and what the lens gets of its pool. */
    float scale, facing;
} sm_beam;

/** What every dipped beam has in common. */
typedef struct {
    /** A lamp's half width at the vehicle and how fast it grows; half the lamps' spacing. */
    float width, widening, half_apart;
    /** What each lamp puts on the road, where it has halved, how far it reaches, how high. */
    float each, halves_m, reach_m, high_m;
} sm_dipped;

/** A block of night_light.headlamp_light: one vehicle's two beams on each
 * of `n` points, each point by its two distances along the ground and
 * its height above it. Two pools just ahead of the vehicle that run into
 * one, fading with distance, cut off behind it, at the beam's reach and
 * above the horizontal.
 */
static SM_NOINLINE void beam_block(
    float *restrict light, const float *restrict along_first,
    const float *restrict along_second, const float *restrict above,
    const float *restrict low, ptrdiff_t n, const sm_beam *beam, const sm_dipped *dipped) {
    const sm_beam of = *beam;
    const sm_dipped lamp = *dipped;
    const float high = of.scale * lamp.high_m;
    SM_VECTORISED for (ptrdiff_t i = 0; i < n; i++) {
        const float du = along_first[i] - of.front_first;
        const float dv = along_second[i] - of.front_second;
        const float along = (du * of.ahead_first + dv * of.ahead_second) / of.scale;
        const float aside = (du * of.ahead_second - dv * of.ahead_first) / of.scale;
        const float forward = along > 0.0f ? along : 0.0f;
        const float wide = lamp.width + lamp.widening * forward;
        const float one = (aside - lamp.half_apart) / wide;
        const float other = (aside + lamp.half_apart) / wide;
        float pool = exp_of_minus(one * one) + exp_of_minus(other * other);
        const float fades = forward / lamp.halves_m;
        pool *= lamp.each / (1.0f + fades * fades);
        pool *= clamp01(along) * clamp01((lamp.reach_m - along) / 15.0f);
        pool *= of.facing;
        light[i] += pool * low[i] * clamp01(1.5f - above[i] / high);
    }
}

/** night_light.headlamp_light's sum: every lit vehicle's beams on every
 * point. `along_first`, `along_second` and `above` are each point's place
 * along the ground's two axes and over it; `low` is nought well above any
 * beam. `beams` is (count, 6) floats, a row an `sm_beam`, and `how` seven,
 * an `sm_dipped`. numpy's float arithmetic in its order but for its
 * exponential, which is a polynomial here: a few millionths of a pool.
 */
static void SM_FN(headlamp_beams)(
    float *light, const float *along_first, const float *along_second, const float *above,
    const float *low, ptrdiff_t pixels, const float *beams, ptrdiff_t count, const float *how) {
    const sm_dipped dipped = {how[0], how[1], how[2], how[3], how[4], how[5], how[6]};
    for (ptrdiff_t done = 0; done < pixels; done += SM_BLOCK) {
        const ptrdiff_t n = pixels - done < SM_BLOCK ? pixels - done : SM_BLOCK;
        for (ptrdiff_t k = 0; k < count; k++) {
            const float *row = beams + 6 * k;
            const sm_beam beam = {row[0], row[1], row[2], row[3], row[4], row[5]};
            beam_block(
                light + done, along_first + done, along_second + done, above + done,
                low + done, n, &beam, &dipped);
        }
    }
}

/* -------------------------------------------------------------- night */

/** A block of nothing, for a plane a caller has not got: a street with no
 * lamps, a frame with no sky. Added or multiplied in, it changes no float,
 * and the loops need no second form without it.
 */
static const float sm_nothing[3 * SM_BLOCK];

/** `plane` from `at` on, or nothing where there is no plane. */
static inline const float *block_of(const float *plane, ptrdiff_t at) {
    return plane ? plane + at : sm_nothing;
}

/** A block of night_ir.near_infrared_reflectance: a broad red-weighted
 * response, lifted where the colour or the labels say leaf.
 */
static SM_NOINLINE void nir_block(
    float *restrict out, const float *restrict linear, const float *restrict vegetation,
    ptrdiff_t n, const float *restrict response, float gain, float labelled, float usual) {
    const float w0 = response[0], w1 = response[1], w2 = response[2];
    SM_VECTORISED for (ptrdiff_t i = 0; i < n; i++) {
        const float r = linear[3 * i], g = linear[3 * i + 1], b = linear[3 * i + 2];
        const float broadband = r * w0 + g * w1 + b * w2;
        const float most = r > b ? r : b, all = g + r + b;
        const float green = clamp01((g - most) / (all > 1e-4f ? all : 1e-4f));
        const float lit = clamp01(broadband / usual);
        const float named = labelled * vegetation[i] * lit;
        const float lifted = broadband + gain * (green > named ? green : named);
        const float held = lifted > 0.0f ? lifted : 0.0f;
        out[i] = held < 4.0f ? held : 4.0f;
    }
}

/** night_ir.near_infrared_reflectance in one pass. `usual` is the mean of
 * the broadband response over the frame, which the caller has.
 */
static void SM_FN(nir_reflectance)(
    float *out, const float *linear, const float *vegetation, ptrdiff_t pixels,
    const float *response, float gain, float labelled, float usual) {
    usual = usual > 1e-4f ? usual : 1e-4f;
    for (ptrdiff_t done = 0; done < pixels; done += SM_BLOCK) {
        const ptrdiff_t n = pixels - done < SM_BLOCK ? pixels - done : SM_BLOCK;
        nir_block(
            out + done, linear + 3 * done, block_of(vegetation, done), n, response, gain,
            labelled, usual);
    }
}

/** A block of the infrared night's light: what each surface sends back of
 * the illuminator, the ambient light, the street lamps and the beams.
 */
static SM_NOINLINE void ir_light_block(
    float *restrict signal, float *restrict lifted, const float *restrict reflectance,
    const float *restrict exposure, const float *restrict in_beam,
    const float *restrict street, const float *restrict beams, const float *restrict sky,
    ptrdiff_t n, float ambient, float floor_scale, float sky_scale) {
    SM_VECTORISED for (ptrdiff_t i = 0; i < n; i++) {
        const float floor = floor_scale * in_beam[i];
        const float surface = reflectance[i] > floor ? reflectance[i] : floor;
        const float arriving = exposure[i] + ambient + street[i] + beams[i];
        lifted[i] = surface;
        signal[i] = surface * arriving * (1.0f - sky_scale * sky[i]);
    }
}

/** NightIrMode.apply before its exposure: `signal` is the reflectance,
 * held up to a floor where a beam falls, times the light that arrives,
 * less the night sky; `lifted` is that reflectance. Any of `in_beam`,
 * `street`, `beams` and `sky` may be NULL.
 */
static void SM_FN(night_ir_light)(
    float *signal, float *lifted, const float *reflectance, const float *exposure,
    const float *in_beam, const float *street, const float *beams, const float *sky,
    ptrdiff_t pixels, float ambient, float floor_scale, float sky_scale) {
    for (ptrdiff_t done = 0; done < pixels; done += SM_BLOCK) {
        const ptrdiff_t n = pixels - done < SM_BLOCK ? pixels - done : SM_BLOCK;
        ir_light_block(
            signal + done, lifted + done, reflectance + done, exposure + done,
            block_of(in_beam, done), block_of(street, done), block_of(beams, done),
            block_of(sky, done), n, ambient, floor_scale, sky_scale);
    }
}

/** What an infrared night's exposure, lamps and grain share. */
typedef struct {
    /** The exposure's gain, the lamps' bloom, and the grain: shot, read, gain. */
    float level, bloom, shot, read_noise, gain;
    /** The residual tint of the three channels. */
    float tint[3];
} sm_ir_exposure;

/** A block of the infrared night after its exposure: the scene at the
 * camera's level, the paint's and the sheeting's return, the lamps, the
 * grain, and the grey tinted into three channels.
 */
static SM_NOINLINE void ir_expose_block(
    float *restrict out, const float *restrict signal, const float *restrict returned,
    const float *restrict glare, const float *restrict vehicle_light,
    const float *restrict vehicle_glow, const float *restrict head_light,
    const float *restrict head_glow, const float *restrict noise, ptrdiff_t n,
    const sm_ir_exposure *how) {
    const sm_ir_exposure of = *how;
    SM_VECTORISED for (ptrdiff_t i = 0; i < n; i++) {
        float s = signal[i] * of.level + returned[i] + glare[i];
        const float *restrict light = vehicle_light + 3 * i;
        const float *restrict glow = vehicle_glow + 3 * i;
        const float lamp = light[0] > light[1] ? light[0] : light[1];
        const float halo = glow[0] > glow[1] ? glow[0] : glow[1];
        s = s + (lamp > light[2] ? lamp : light[2]) + of.bloom * (halo > glow[2] ? halo : glow[2]);
        s = s + head_light[i] + of.bloom * head_glow[i];
        const float root = sqrtf(s > 0.0f ? s : 0.0f);
        const float grey = clamp01(s + noise[i] * (root * of.shot + of.read_noise) * of.gain);
        out[3 * i] = clamp01(grey * of.tint[0]);
        out[3 * i + 1] = clamp01(grey * of.tint[1]);
        out[3 * i + 2] = clamp01(grey * of.tint[2]);
    }
}

/** The retroreflected light of a block: the paint's glass beads send the
 * illuminator's light back down its own beam.
 */
static SM_NOINLINE void ir_returned_block(
    float *restrict returned, const float *restrict retro, const float *restrict lifted,
    const float *restrict exposure, ptrdiff_t n, float level) {
    SM_VECTORISED for (ptrdiff_t i = 0; i < n; i++) {
        returned[i] = level * retro[i] * lifted[i] * exposure[i];
    }
}

/** NightIrMode.apply from its exposure on: `out` is (pixels, 3). `retro`
 * with `lifted` and `exposure`, `glare`, the two (pixels, 3) fields of the
 * traffic's lamps and the two planes of the street's may each be NULL.
 * `how` is the level, the bloom, the grain's shot, read and gain, and the
 * tint's three.
 */
static void SM_FN(night_ir_expose)(
    float *out, const float *signal, const float *retro, const float *lifted,
    const float *exposure, const float *glare, const float *vehicle_light,
    const float *vehicle_glow, const float *head_light, const float *head_glow,
    const float *noise, ptrdiff_t pixels, const float *how) {
    const sm_ir_exposure of = {how[0], how[1], how[2], how[3], how[4], {how[5], how[6], how[7]}};
    float returned[SM_BLOCK];
    for (ptrdiff_t done = 0; done < pixels; done += SM_BLOCK) {
        const ptrdiff_t n = pixels - done < SM_BLOCK ? pixels - done : SM_BLOCK;
        if (retro) {
            ir_returned_block(returned, retro + done, lifted + done, exposure + done, n, of.level);
        }
        ir_expose_block(
            out + 3 * done, signal + done, retro ? returned : sm_nothing, block_of(glare, done),
            block_of(vehicle_light, 3 * done), block_of(vehicle_glow, 3 * done),
            block_of(head_light, done), block_of(head_glow, done), noise + done, n, &of);
    }
}

/** What a colour night's light has in common over a frame. */
typedef struct {
    /** The sky's glow, the street lamps' colour and the headlamps', a channel each. */
    float glow[3], street[3], head[3];
    /** The beam's floor, how dark the night sky is, and the luminance's weights. */
    float floor_scale, sky_scale, luma[3];
} sm_colour_light;

/** A block of the colour night's light, with the day's light and shade
 * taken off the surfaces first: each is held up to the beam's floor and
 * scaled to the lightness `evened` gives it.
 */
static SM_NOINLINE void colour_light_block(
    float *restrict signal, float *restrict brightness, const float *restrict linear,
    const float *restrict evened, const float *restrict in_beam,
    const float *restrict street, const float *restrict beams, const float *restrict sky,
    ptrdiff_t n, const sm_colour_light *how, int even) {
    const sm_colour_light of = *how;
    SM_VECTORISED for (ptrdiff_t i = 0; i < n; i++) {
        float r = linear[3 * i], g = linear[3 * i + 1], b = linear[3 * i + 2];
        const float floor = of.floor_scale * in_beam[i];
        const float fr = r > floor ? r : floor, fg = g > floor ? g : floor;
        const float fb = b > floor ? b : floor;
        const float level = evened[i] > floor ? evened[i] : floor;
        const float own = fr * of.luma[0] + fg * of.luma[1] + fb * of.luma[2];
        const float scale = level / (own > 1e-4f ? own : 1e-4f);
        r = even ? fr * scale : r;
        g = even ? fg * scale : g;
        b = even ? fb * scale : b;
        const float dark = 1.0f - of.sky_scale * sky[i];
        r = r * (of.glow[0] + street[i] * of.street[0] + beams[i] * of.head[0]) * dark;
        g = g * (of.glow[1] + street[i] * of.street[1] + beams[i] * of.head[1]) * dark;
        b = b * (of.glow[2] + street[i] * of.street[2] + beams[i] * of.head[2]) * dark;
        signal[3 * i] = r, signal[3 * i + 1] = g, signal[3 * i + 2] = b;
        brightness[i] = r * of.luma[0] + g * of.luma[1] + b * of.luma[2];
    }
}

/** NightColorMode.apply before its exposure: `signal`, (pixels, 3), is
 * each surface's colour times the light that reaches it, and `brightness`
 * its luminance. With `evened`, a surface is first held up to the beam's
 * floor and scaled to that lightness. `in_beam`, `street`, `beams` and
 * `sky` may be NULL. `how` is the sky's glow, the street's colour and the
 * headlamps', three each, then the floor's scale, the sky's, and the
 * luminance's three weights.
 */
static void SM_FN(night_colour_light)(
    float *signal, float *brightness, const float *linear, const float *evened,
    const float *in_beam, const float *street, const float *beams, const float *sky,
    ptrdiff_t pixels, const float *how) {
    const sm_colour_light of = {
        {how[0], how[1], how[2]}, {how[3], how[4], how[5]}, {how[6], how[7], how[8]},
        how[9], how[10], {how[11], how[12], how[13]},
    };
    for (ptrdiff_t done = 0; done < pixels; done += SM_BLOCK) {
        const ptrdiff_t n = pixels - done < SM_BLOCK ? pixels - done : SM_BLOCK;
        colour_light_block(
            signal + 3 * done, brightness + done, linear + 3 * done, block_of(evened, done),
            block_of(in_beam, done), block_of(street, done), block_of(beams, done),
            block_of(sky, done), n, &of, evened != NULL);
    }
}

/** A block of the colour night after its exposure: the scene at the
 * camera's level, the lamps in their colours, and each channel's grain.
 */
static SM_NOINLINE void colour_expose_block(
    float *restrict out, const float *restrict signal, const float *restrict vehicle_light,
    const float *restrict vehicle_glow, const float *restrict head_light,
    const float *restrict head_glow, const float *restrict noise, ptrdiff_t n,
    const float *restrict street, float level, float bloom, float shot, float read_noise,
    float gain) {
    const float s0 = street[0], s1 = street[1], s2 = street[2];
    SM_VECTORISED for (ptrdiff_t i = 0; i < n; i++) {
        const float burning = head_light[i] + bloom * head_glow[i];
        const float tint[3] = {s0, s1, s2};
        for (int c = 0; c < 3; c++) {
            const ptrdiff_t at = 3 * i + c;
            float s = signal[at] * level + vehicle_light[at];
            s = s + bloom * vehicle_glow[at];
            s = s + burning * tint[c];
            const float grain = (sqrtf(s > 0.0f ? s : 0.0f) * shot + read_noise) * gain;
            out[at] = clamp01(s + noise[at] * grain);
        }
    }
}

/** NightColorMode.apply from its exposure on: `out`, `signal` and `noise`
 * are (pixels, 3). The traffic's lamps, (pixels, 3) each, and the
 * street's, a plane each, may be NULL. `how` is the level, the bloom and
 * the grain's shot, read and gain.
 */
static void SM_FN(night_colour_expose)(
    float *out, const float *signal, const float *vehicle_light, const float *vehicle_glow,
    const float *head_light, const float *head_glow, const float *noise, ptrdiff_t pixels,
    const float *street, const float *how) {
    for (ptrdiff_t done = 0; done < pixels; done += SM_BLOCK) {
        const ptrdiff_t n = pixels - done < SM_BLOCK ? pixels - done : SM_BLOCK;
        colour_expose_block(
            out + 3 * done, signal + 3 * done, block_of(vehicle_light, 3 * done),
            block_of(vehicle_glow, 3 * done), block_of(head_light, done),
            block_of(head_glow, done), noise + 3 * done, n, street, how[0], how[1], how[2],
            how[3], how[4]);
    }
}

/** A block of the night fog before its halo: each lamp counted at what
 * it plausibly was, and what the fog takes of it.
 */
static SM_NOINLINE void fog_lost_block(
    float *restrict radiance, float *restrict removed, float *restrict weight,
    const float *restrict linear, const float *restrict clipped,
    const float *restrict one_way, ptrdiff_t n, float over) {
    SM_VECTORISED for (ptrdiff_t i = 0; i < n; i++) {
        const float lamp = clipped[i], lost = 1.0f - one_way[i];
        const float lift = 1.0f + over * lamp;
        const float r = linear[3 * i] * lift, g = linear[3 * i + 1] * lift;
        const float b = linear[3 * i + 2] * lift;
        const float gone_r = r * lamp * lost, gone_g = g * lamp * lost, gone_b = b * lamp * lost;
        radiance[3 * i] = r, radiance[3 * i + 1] = g, radiance[3 * i + 2] = b;
        removed[3 * i] = gone_r, removed[3 * i + 1] = gone_g, removed[3 * i + 2] = gone_b;
        weight[i] = gone_r + gone_g + gone_b;
    }
}

/** NightFog.apply before its halo: `radiance`, (pixels, 3), is `linear`
 * with its lamps (`clipped`, a plane) lifted by `over`; `removed` what the
 * fog (`one_way`, its transmission) takes of them, and `weight` that
 * summed over the channels.
 */
static void SM_FN(fog_lost)(
    float *radiance, float *removed, float *weight, const float *linear,
    const float *clipped, const float *one_way, ptrdiff_t pixels, float over) {
    for (ptrdiff_t done = 0; done < pixels; done += SM_BLOCK) {
        const ptrdiff_t n = pixels - done < SM_BLOCK ? pixels - done : SM_BLOCK;
        fog_lost_block(
            radiance + 3 * done, removed + 3 * done, weight + done, linear + 3 * done,
            clipped + done, one_way + done, n, over);
    }
}

/** What a night fog's veil has in common over a frame. */
typedef struct {
    /** How much of the halo comes forward, and whether the light crosses the fog twice. */
    float forward;
    int twice;
    /** The fog's glow by the scene's level, with lamps whose places are known. */
    float level[3];
    /** The veil's share, the lit volume's part of it, and the diffuse part a channel. */
    float share, lit_part, diffuse[3];
} sm_fog_light;

/** A block of the night fog after its halo: the scene through the fog,
 * the halo, the lamps' cones or the veil, and the illuminator's wall.
 */
static SM_NOINLINE void fog_veil_block(
    float *restrict out, const float *restrict radiance, const float *restrict halo,
    const float *restrict one_way, const float *restrict cones, const float *restrict lit,
    const float *restrict wall, ptrdiff_t n, const sm_fog_light *how) {
    const sm_fog_light of = *how;
    SM_VECTORISED for (ptrdiff_t i = 0; i < n; i++) {
        const float through = of.twice ? one_way[i] * one_way[i] : one_way[i];
        const float lost = 1.0f - one_way[i];
        for (int c = 0; c < 3; c++) {
            const ptrdiff_t at = 3 * i + c;
            float v = radiance[at] * through + of.forward * halo[at];
            v = v + cones[at] + of.level[c] * lost;
            v = v + of.share * (of.lit_part * lit[at] + of.diffuse[c]) * lost;
            out[at] = v + wall[i];
        }
    }
}

/** NightFog.apply after its halo, into `out`, (pixels, 3): `radiance`
 * through the fog (`one_way`, crossed twice with `how`'s second value not
 * zero), the `halo`, and whichever of the lamps' `cones`, the `lit`
 * volume and the illuminator's `wall` (a plane) there is; each of the
 * three may be NULL. `how` is ten floats: the halo's share, whether
 * twice, the cones' level a channel, the veil's share, its lit part and
 * its diffuse part a channel.
 */
static void SM_FN(fog_veil)(
    float *out, const float *radiance, const float *halo, const float *one_way,
    const float *cones, const float *lit, const float *wall, ptrdiff_t pixels,
    const float *how) {
    const sm_fog_light of = {
        how[0], how[1] != 0.0f, {how[2], how[3], how[4]}, how[5], how[6],
        {how[7], how[8], how[9]},
    };
    for (ptrdiff_t done = 0; done < pixels; done += SM_BLOCK) {
        const ptrdiff_t n = pixels - done < SM_BLOCK ? pixels - done : SM_BLOCK;
        fog_veil_block(
            out + 3 * done, radiance + 3 * done, halo + 3 * done, one_way + done,
            block_of(cones, 3 * done), block_of(lit, 3 * done), block_of(wall, done), n, &of);
    }
}

/* --------------------------------------------------------------- snow */

/** What a frame of lying snow has in common. */
typedef struct {
    /** Where texture begins to say canopy and over what span it comes to say so wholly. */
    float relief_from, relief_span;
    /** How much the snow lies in patches, their contrast, and whether there are any. */
    float patchiness, contrast;
    /** How much there is, the unevenness of a thin covering, and a canopy's share. */
    float strength, grain, canopy_cover;
} sm_snow_cover;

/** A block of snow.RandomSnow's lying snow, once every field it is read
 * from is there: where the ground is a canopy, how the snow lies in
 * patches and in the surface's grain, what a canopy's proud clumps hold,
 * and nothing on the sky.
 */
static SM_NOINLINE void snow_settled_block(
    float *restrict settled, const float *restrict level, const float *restrict relief,
    const float *restrict steady, const float *restrict green, const float *restrict lawn,
    const float *restrict named, const float *restrict patch, const float *restrict grain,
    const float *restrict luma, const float *restrict around, const float *restrict sky,
    ptrdiff_t n, const sm_snow_cover *how) {
    const sm_snow_cover of = *how;
    SM_VECTORISED for (ptrdiff_t i = 0; i < n; i++) {
        const float rough = clamp01((relief[i] - of.relief_from) / of.relief_span);
        const float unsteady =
            clamp01((0.85f - steady[i]) / 0.25f) * clamp01((steady[i] - 0.15f) / 0.2f);
        const float grown = rough * (green[i] > unsteady ? green[i] : unsteady);
        const float kept = grown * (1.0f - lawn[i]);
        const float canopy = kept > named[i] ? kept : named[i];
        const float patches = clamp01((patch[i] - 0.5f) * of.contrast + 0.5f);
        const float ground =
            level[i] * (1.0f - canopy) * (1.0f - of.patchiness * (1.0f - patches));
        const float uneven = 1.0f + of.grain * (2.0f * grain[i] - 1.0f);
        const float proud = clamp01((luma[i] / around[i] - 0.95f) / 0.35f);
        const float lying = ground * of.strength * uneven + of.canopy_cover * canopy * proud;
        settled[i] = lying * (1.0f - sky[i]);
    }
}

/** snow.RandomSnow's lying snow from the fields it is read from, a plane
 * each: `level` ground, the picture's `relief`, how `steady` the level
 * reads, how `green` it is, the labelled `lawn` and canopy (`named`), the
 * `patch` noise, the `grain`, the `luma` and its mean `around`, the `sky`.
 * `lawn`, `named` and `patch` may be NULL. `how` is seven floats: where
 * relief begins and its span, the patchiness, the patches' contrast, the
 * strength, the grain's depth and a canopy's cover, the last two already
 * times what the caller multiplies them by. The numpy code's arithmetic
 * in its order.
 */
static void SM_FN(snow_settled)(
    float *settled, const float *level, const float *relief, const float *steady,
    const float *green, const float *lawn, const float *named, const float *patch,
    const float *grain, const float *luma, const float *around, const float *sky,
    ptrdiff_t pixels, const float *how) {
    /* With no patches, none is taken off: a contrast of nothing and no
     * patchiness leave the ground as it is, to the bit. */
    const sm_snow_cover of = {
        how[0], how[1], patch ? how[2] : 0.0f, how[3], how[4], how[5], how[6]};
    for (ptrdiff_t done = 0; done < pixels; done += SM_BLOCK) {
        const ptrdiff_t n = pixels - done < SM_BLOCK ? pixels - done : SM_BLOCK;
        snow_settled_block(
            settled + done, level + done, relief + done, steady + done, green + done,
            block_of(lawn, done), block_of(named, done), block_of(patch, done), grain + done,
            luma + done, around + done, sky + done, n, &of);
    }
}

/** The largest whole number not over `x`, for an `x` that an int holds:
 * written so that every tier works it eight or four at a time, which the
 * library's `floorf` is not before SSE4.1.
 */
static inline float floor_of(float x) {
    const float whole = (float)(int32_t)x;
    return whole - (whole > x ? 1.0f : 0.0f);
}

/** A coordinate on the ground as one in a noise tile, folded at the tile's edges. */
static inline float folded(float metres, float per_metre, float span, float per_span) {
    const float pixels = metres * per_metre;
    const float within = pixels - span * floor_of(pixels * per_span);
    const float back = span - within;
    return within < back ? within : back;
}

/** A row of ground.ground_noise's places in its tile: where each pixel
 * of the row stands on the ground, across and along, from its depth.
 */
static SM_NOINLINE void ground_tile_row(
    float *restrict tile_x, float *restrict tile_y, const float *restrict depth,
    const float *restrict column, ptrdiff_t n, float left, float y, const float *restrict how) {
    const float focal = how[0], cosine = how[1], sine = how[2], per_metre = how[3];
    const float span = how[4], per_span = how[5], off_x = how[6], off_y = how[7];
    SM_VECTORISED for (ptrdiff_t i = 0; i < n; i++) {
        const float x = column[i] + left;
        const float across = x * depth[i] / focal;
        const float down = y * depth[i] / focal;
        const float along = depth[i] * cosine - down * sine;
        tile_x[i] = folded(across + off_x, per_metre, span, per_span);
        tile_y[i] = folded(along + off_y, per_metre, span, per_span);
    }
}

/** ground.ground_noise's two maps into its noise tile, in fast mode: for
 * each pixel of `depth`, (height, width), its place on the ground across
 * and along, in tile pixels, folded so that the tile repeats without a
 * seam. `how` is eight floats: the focal length, the pitch's cosine and
 * sine, tile pixels a metre, the fold's span and its reciprocal, and the
 * two offsets. The numpy code's arithmetic to the bit.
 */
static void SM_FN(ground_tile)(
    float *tile_x, float *tile_y, const float *depth, ptrdiff_t height, ptrdiff_t width,
    const float *how) {
    float column[SM_BLOCK];
    for (ptrdiff_t i = 0; i < SM_BLOCK; i++) {
        column[i] = (float)i;
    }
    const float half_wide = (float)((double)width / 2.0);
    const float half_high = (float)((double)height / 2.0);
    for (ptrdiff_t v = 0; v < height; v++) {
        const float y = (float)v - half_high;
        for (ptrdiff_t u = 0; u < width; u += SM_BLOCK) {
            const ptrdiff_t n = width - u < SM_BLOCK ? width - u : SM_BLOCK;
            const ptrdiff_t at = v * width + u;
            /* The column's number less half the width: a block's first, which
             * is a whole number or a half and exact, and so is each one's. */
            ground_tile_row(tile_x + at, tile_y + at, depth + at, column, n,
                            (float)u - half_wide, y, how);
        }
    }
}

/* ------------------------------------------------------ a road's frame */

/** A row of which way a road runs at each pixel, as a unit vector that
 * points down the frame. `column` holds the columns' numbers less the
 * block's first, which is `left`; `under` is the row's distance below the
 * horizon and `bent` what the road's bend adds to its slope there. In
 * double, as the numpy code is, and narrowed at the end.
 */
static SM_NOINLINE void road_along_row(
    float *restrict along, const double *restrict column, ptrdiff_t n, double left,
    double centre, double under, double bent) {
    SM_VECTORISED for (ptrdiff_t i = 0; i < n; i++) {
        const double slope = (column[i] + left - centre) / under - bent;
        const double length = sqrt(1.0 + slope * slope);
        along[2 * i] = (float)(slope / length);
        along[2 * i + 1] = (float)(1.0 / length);
    }
}

/** road.model_frame's last field: the way the road runs at every pixel of
 * a (height, width) frame, into `along`, (height, width, 2). A line along
 * a road of constant curvature runs, at a pixel, (u - centre) / t - 2
 * bend / t^2 columns a row, t being the row's distance below the horizon.
 * The numpy code's double arithmetic to the bit.
 */
static void SM_FN(road_along)(
    float *along, ptrdiff_t height, ptrdiff_t width, double centre, double horizon,
    double bend) {
    double column[SM_BLOCK];
    for (ptrdiff_t i = 0; i < SM_BLOCK; i++) {
        column[i] = (double)i;
    }
    for (ptrdiff_t v = 0; v < height; v++) {
        const double below = (double)v - horizon;
        const double under = below > 1e-6 ? below : 1e-6;
        const double bent = 2.0 * bend / (under * under);
        for (ptrdiff_t u = 0; u < width; u += SM_BLOCK) {
            const ptrdiff_t n = width - u < SM_BLOCK ? width - u : SM_BLOCK;
            road_along_row(
                along + 2 * (v * width + u), column, n, (double)u, centre, under, bent);
        }
    }
}

/** road.model_frame's other fields, for the rows a road can be on: which
 * carriageway each pixel is on (`piece`), how far across it (`across`)
 * and how wide it is there in pixels (`span`), each (rows, width). A
 * pixel's place across the road is `offset`, and `at` the position that
 * is counted as; `of_position` says which carriageway a position is on,
 * 0 being none, and `firsts` and `widths` where each carriageway begins
 * and how wide it is. `whole` is 1 where the pixel is road at all and
 * `below` each row's distance below the horizon. A lookup a pixel, so
 * not a loop a compiler vectorises; the numpy code's arithmetic to the
 * bit, in double and narrowed at the end.
 */
static void SM_FN(road_places)(
    int32_t *piece, float *across, float *span, const double *offset, const int64_t *at,
    const uint8_t *whole, const double *below, ptrdiff_t rows, ptrdiff_t width,
    const int32_t *of_position, const double *firsts, const double *widths) {
    for (ptrdiff_t v = 0; v < rows; v++) {
        const double under = below[v];
        for (ptrdiff_t u = 0; u < width; u++) {
            const ptrdiff_t i = v * width + u;
            const int32_t on = of_position[at[i]] * (int32_t)whole[i];
            const double wide = widths[on];
            const double place = (offset[i] - firsts[on]) / wide;
            const double held = place > 0.0 ? (place < 1.0 ? place : 1.0) : 0.0;
            piece[i] = on;
            across[i] = on > 0 ? (float)held : 0.0f;
            span[i] = on > 0 ? (float)(wide * under) : 1.0f;
        }
    }
}

/* --------------------------------------------------------- wheel tracks */

/** Whether any of `n` pixels is on road `road`. */
static inline int any_on(const int32_t *restrict piece, ptrdiff_t n, int32_t road) {
    int found = 0;
    for (ptrdiff_t i = 0; i < n; i++) {
        found |= piece[i] == road;
    }
    return found;
}

/** For a block of pixels, the lane line at or below each one's place
 * across its road and the next one up, by trying `line` and the one
 * `above` it: a pixel past `line` takes them, the rest keep what they
 * had. Tried for each line in turn from the lowest, this is the search
 * numpy does, without a branch or a lookup a pixel.
 */
static SM_NOINLINE void lane_step(
    double *restrict lower, double *restrict upper, int32_t *restrict lane,
    const float *restrict across, ptrdiff_t n, double line, double above, int32_t index) {
    SM_VECTORISED for (ptrdiff_t i = 0; i < n; i++) {
        const int past = line < (double)across[i];
        lower[i] = past ? line : lower[i];
        upper[i] = past ? above : upper[i];
        lane[i] = past ? index : lane[i];
    }
}

/** What road `road` gives a block of pixels: where each is in its lane,
 * in lanes from the lane's middle (`turns`), how wide the lane is, which
 * lane it is, and whether it is within the road's lanes at all. A pixel
 * of another road keeps what it had.
 */
static SM_NOINLINE void lane_place(
    float *restrict turns, float *restrict wide, int32_t *restrict which,
    float *restrict inside, const double *restrict lower, const double *restrict upper,
    const int32_t *restrict lane, const int32_t *restrict piece, const float *restrict across,
    ptrdiff_t n, int32_t road, double first, double last) {
    SM_VECTORISED for (ptrdiff_t i = 0; i < n; i++) {
        const int here = piece[i] == road;
        const double at = (double)across[i];
        const float lane_wide = (float)(upper[i] - lower[i]);
        const float place = (float)((at - 0.5 * (lower[i] + upper[i])) / (double)lane_wide);
        /* Everything a pixel of this road would take is read and worked
         * out for every pixel, and then taken or not: read inside the
         * choice, the lane's number was a load only some pixels make,
         * which is a branch, and the SSE2 tier left the loop scalar. */
        const float within = (float)((at >= first) & (at <= last));
        const int32_t numbered = lane[i] + 17 * road;
        turns[i] = here ? place : turns[i];
        wide[i] = here ? lane_wide : wide[i];
        which[i] = here ? numbered : which[i];
        inside[i] = here ? within : inside[i];
    }
}

/** What a frame's wheel tracks have in common. */
typedef struct {
    /** Half a track's width in lanes, how far a track wanders, half the wheels' spacing. */
    float band, wander, half_apart;
    /** The least of a track that wear leaves, and how fast wear restores it. */
    float wear_floor, wear_gain;
    /** Whether the tracks are ragged, and whether they are worn. */
    int ragged, worn;
} sm_tracks;

/** A block of the tracks themselves, before each lane's own use: a soft
 * band either side of the lane's middle, no sharper than a pixel shows.
 */
static SM_NOINLINE void track_block(
    float *restrict track, const float *restrict turns, const float *restrict wide,
    const float *restrict width_px, const float *restrict ragged, ptrdiff_t n,
    const sm_tracks *how) {
    const sm_tracks of = *how;
    SM_VECTORISED for (ptrdiff_t i = 0; i < n; i++) {
        const float wandered = turns[i] + of.wander * (2.0f * ragged[i] - 1.0f);
        const float turn = of.ragged ? wandered : turns[i];
        const float half = of.ragged ? of.band * (0.6f + 0.8f * ragged[i]) : of.band;
        const float from_wheel = fabsf(fabsf(turn) - of.half_apart);
        const float lane_px = wide[i] * width_px[i];
        const float pixel = 1.0f / (lane_px > 1.0f ? lane_px : 1.0f);
        const float soft = 1.6f * half, sharp = 1.5f * pixel;
        const float reach = soft > sharp ? soft : sharp;
        const float edge = clamp01(1.0f - from_wheel / reach);
        track[i] = edge * edge * (3.0f - 2.0f * edge) * clamp01(soft / reach);
    }
}

/** A block of tracks by how much has been driven over each stretch, and
 * only on the road's own lanes.
 */
static SM_NOINLINE void track_wear_block(
    float *restrict out, const float *restrict track, const float *restrict worn,
    const float *restrict inside, const float *restrict weight, ptrdiff_t n,
    const sm_tracks *how) {
    const sm_tracks of = *how;
    SM_VECTORISED for (ptrdiff_t i = 0; i < n; i++) {
        const float kept = clamp01(of.wear_floor + of.wear_gain * worn[i]);
        const float driven = of.worn ? track[i] * kept : track[i];
        out[i] = driven * inside[i] * weight[i];
    }
}

/** road.wheel_tracks: the weight of the two wheel tracks down each lane,
 * for every pixel. `piece` is each pixel's road, `across` its place
 * across it and `width_px` the road's width there; `edges` holds every
 * road's lane lines one after another, road r's from `first_edge[r]` to
 * `first_edge[r + 1]`, for roads 1 to `roads`. `ragged` and `worn` may be
 * NULL. `use` is how much each of 64 lanes is driven in, and `how` five
 * floats: the band, the wander, half the wheels' spacing, the wear's
 * floor and its gain. The numpy code's arithmetic, in double where it is
 * and float where it is, to the bit.
 */
static void SM_FN(wheel_tracks)(
    float *out, const int32_t *piece, const float *across, const float *width_px,
    const float *weight, const float *ragged, const float *worn, ptrdiff_t pixels,
    const double *edges, const int32_t *first_edge, int32_t roads, const float *use,
    const float *how) {
    const sm_tracks of = {how[0], how[1], how[2], how[3], how[4], ragged != NULL, worn != NULL};
    double lower[SM_BLOCK], upper[SM_BLOCK];
    int32_t lane[SM_BLOCK], which[SM_BLOCK];
    float turns[SM_BLOCK], wide[SM_BLOCK], inside[SM_BLOCK], track[SM_BLOCK];
    for (ptrdiff_t done = 0; done < pixels; done += SM_BLOCK) {
        const ptrdiff_t n = pixels - done < SM_BLOCK ? pixels - done : SM_BLOCK;
        for (ptrdiff_t i = 0; i < n; i++) {
            turns[i] = 0.0f, wide[i] = 1.0f, inside[i] = 0.0f, which[i] = 0;
        }
        for (int32_t road = 1; road <= roads; road++) {
            const double *lines = edges + first_edge[road];
            const int32_t count = first_edge[road + 1] - first_edge[road];
            if (count < 2 || !any_on(piece + done, n, road)) {
                continue;
            }
            for (ptrdiff_t i = 0; i < n; i++) {
                lower[i] = lines[0], upper[i] = lines[1], lane[i] = 0;
            }
            for (int32_t k = 1; k < count - 1; k++) {
                lane_step(lower, upper, lane, across + done, n, lines[k], lines[k + 1], k);
            }
            lane_place(
                turns, wide, which, inside, lower, upper, lane, piece + done, across + done, n,
                road, lines[0], lines[count - 1]);
        }
        track_block(track, turns, wide, width_px + done, block_of(ragged, done), n, &of);
        for (ptrdiff_t i = 0; i < n; i++) {
            track[i] *= use[which[i] & 63];
        }
        track_wear_block(
            out + done, track, block_of(worn, done), inside, weight + done, n, &of);
    }
}

/* ------------------------------------------------- rain, lamps and spray */

/** A block of the traffic's lamps through the rain between them and the
 * lens, up to full scale: what the wet road then reflects of them.
 */
static SM_NOINLINE void lamps_lit_block(
    float *restrict out, const float *restrict lamp_light, const float *restrict through,
    ptrdiff_t n) {
    SM_VECTORISED for (ptrdiff_t i = 0; i < n; i++) {
        for (int c = 0; c < 3; c++) {
            const float lamp = lamp_light[3 * i + c];
            out[3 * i + c] += (lamp < 1.0f ? lamp : 1.0f) * through[i];
        }
    }
}

/** RandomRain.apply's lamps before its road: `out`, (pixels, 3), gains
 * each lamp up to full scale through the veil's transmission `through`.
 * The numpy code's arithmetic to the bit.
 */
static void SM_FN(lamps_lit)(
    float *out, const float *lamp_light, const float *through, ptrdiff_t pixels) {
    for (ptrdiff_t done = 0; done < pixels; done += SM_BLOCK) {
        const ptrdiff_t n = pixels - done < SM_BLOCK ? pixels - done : SM_BLOCK;
        lamps_lit_block(out + 3 * done, lamp_light + 3 * done, through + done, n);
    }
}

/** A block of the spray over the road and of the lamps through it: the
 * spray lit as the rain's veil is, then what of each lamp is over full
 * scale, through the rain and the spray, and its glow in the wet air.
 */
static SM_NOINLINE void spray_and_lamps_block(
    float *restrict out, const float *restrict spray, const float *restrict lamp_light,
    const float *restrict lamp_glow, const float *restrict through, ptrdiff_t n,
    const float *restrict lit, float glow_in_rain) {
    const float l0 = lit[0], l1 = lit[1], l2 = lit[2];
    SM_VECTORISED for (ptrdiff_t i = 0; i < n; i++) {
        const float share = spray[i];
        const float clear = through[i] * (1.0f - share);
        const float wet_air = glow_in_rain + (1.0f - glow_in_rain) * share;
        const float air[3] = {l0, l1, l2};
        for (int c = 0; c < 3; c++) {
            const ptrdiff_t at = 3 * i + c;
            const float lamp = lamp_light[at];
            const float veiled = out[at] * (1.0f - share) + air[c] * share;
            out[at] = veiled + (lamp - (lamp < 1.0f ? lamp : 1.0f)) * clear
                + lamp_glow[at] * wet_air;
        }
    }
}

/** RandomRain.apply between its road and its streaks, on `out`, (pixels,
 * 3): the `spray`, a plane, laid over it in the light `lit` (the veil's
 * radiance times the spray's albedo, a channel each), and then what of
 * `lamp_light` is over full scale, through `through` and the spray, with
 * `lamp_glow` by how wet the air is. `spray`, and the two lamp fields
 * together, may be NULL. The numpy code's arithmetic to the bit.
 */
static void SM_FN(spray_and_lamps)(
    float *out, const float *spray, const float *lamp_light, const float *lamp_glow,
    const float *through, ptrdiff_t pixels, const float *lit, float glow_in_rain) {
    for (ptrdiff_t done = 0; done < pixels; done += SM_BLOCK) {
        const ptrdiff_t n = pixels - done < SM_BLOCK ? pixels - done : SM_BLOCK;
        spray_and_lamps_block(
            out + 3 * done, block_of(spray, done), block_of(lamp_light, 3 * done),
            block_of(lamp_glow, 3 * done), through + done, n, lit, glow_in_rain);
    }
}

/* ------------------------------------------------------------ normals */

/** A surface's unit normal at a pixel, and the relative step in depth it was read across. */
typedef struct {
    float x, y, z, jump;
} sm_normal;

/** normals._normals_and_jump at one pixel. `d` is its depth and (u, v) its
 * line of sight, so that it stands at (u d, v d, d). Along the row it has
 * a tangent forwards, from the pixel at `h_a1` to the one at `h_b1`, and
 * one backwards, from `h_a2` to `h_b2`: each a depth and its own u. Down
 * the column likewise, each a depth and its own v. Of each pair the one
 * with the smaller step in depth is the pixel's own surface. Their cross
 * product, turned to face the lens and made a unit, is the normal. The
 * numpy code's float arithmetic in its order.
 */
static inline sm_normal normal_at(
    float d, float u, float v,
    float h_a1, float h_b1, float u_a1, float u_b1,
    float h_a2, float h_b2, float u_a2, float u_b2,
    float v_a1, float v_b1, float w_a1, float w_b1,
    float v_a2, float v_b2, float w_a2, float w_b2) {
    /* Along the row: the forward difference, the backward one, the choice. */
    const float fz = h_b1 - h_a1, bz = h_b2 - h_a2;
    const int forward = fabsf(fz) <= fabsf(bz);
    const float ux = forward ? u_b1 * h_b1 - u_a1 * h_a1 : u_b2 * h_b2 - u_a2 * h_a2;
    const float uy = forward ? v * h_b1 - v * h_a1 : v * h_b2 - v * h_a2;
    const float uz = forward ? fz : bz;
    /* Down the column. */
    const float gz = v_b1 - v_a1, cz = v_b2 - v_a2;
    const int downward = fabsf(gz) <= fabsf(cz);
    const float vx = downward ? u * v_b1 - u * v_a1 : u * v_b2 - u * v_a2;
    const float vy = downward ? w_b1 * v_b1 - w_a1 * v_a1 : w_b2 * v_b2 - w_a2 * v_a2;
    const float vz = downward ? gz : cz;
    const float nx = uy * vz - uz * vy;
    const float ny = uz * vx - ux * vz;
    const float nz = ux * vy - uy * vx;
    const float length = sqrtf(nx * nx + ny * ny + nz * nz);
    /* Towards the lens, which is at the origin: n . P < 0. */
    const float facing = nx * (u * d) + ny * (v * d) + nz * d;
    float scale = (facing > 0.0f ? -1.0f : 1.0f) / (length > 1e-8f ? length : 1e-8f);
    scale = length <= 1e-8f ? 0.0f : scale;
    const float az = fabsf(uz), bv = fabsf(vz);
    const sm_normal normal = {
        nx * scale, ny * scale, nz * scale, (az > bv ? az : bv) / (d > 1e-6f ? d : 1e-6f)};
    return normal;
}

/** One row of the normals. `mid` is the row's own depth; `a1` to `b1` is
 * the pair of rows its forward difference is taken across and `a2` to `b2`
 * its backward one, with `w` those rows' own v. On the frame's top and
 * bottom rows the two pairs are the same pair. Within `step` of the left
 * and right edges a pixel has one neighbour along the row and takes it;
 * the rest of the row is one loop.
 */
static SM_NOINLINE void normals_row(
    float *restrict nx, float *restrict ny, float *restrict nz, float *restrict jump,
    const float *restrict mid, const float *restrict a1, const float *restrict b1,
    const float *restrict a2, const float *restrict b2, const float *restrict u, float v,
    float w_a1, float w_b1, float w_a2, float w_b2, ptrdiff_t width, ptrdiff_t step) {
    for (ptrdiff_t c = 0; c < step; c++) {
        const ptrdiff_t e = width - 1 - c;
        const sm_normal left = normal_at(
            mid[c], u[c], v, mid[c], mid[c + step], u[c], u[c + step],
            mid[c], mid[c + step], u[c], u[c + step],
            a1[c], b1[c], w_a1, w_b1, a2[c], b2[c], w_a2, w_b2);
        nx[c] = left.x, ny[c] = left.y, nz[c] = left.z, jump[c] = left.jump;
        const sm_normal right = normal_at(
            mid[e], u[e], v, mid[e - step], mid[e], u[e - step], u[e],
            mid[e - step], mid[e], u[e - step], u[e],
            a1[e], b1[e], w_a1, w_b1, a2[e], b2[e], w_a2, w_b2);
        nx[e] = right.x, ny[e] = right.y, nz[e] = right.z, jump[e] = right.jump;
    }
    const float *restrict before = mid - step;
    const float *restrict after = mid + step;
    const float *restrict u_before = u - step;
    const float *restrict u_after = u + step;
    SM_VECTORISED for (ptrdiff_t c = step; c < width - step; c++) {
        const sm_normal here = normal_at(
            mid[c], u[c], v, mid[c], after[c], u[c], u_after[c],
            before[c], mid[c], u_before[c], u[c],
            a1[c], b1[c], w_a1, w_b1, a2[c], b2[c], w_a2, w_b2);
        nx[c] = here.x, ny[c] = here.y, nz[c] = here.z, jump[c] = here.jump;
    }
}

/** normals._normals_and_jump after its blur: each pixel's unit normal as
 * three planes, and the relative step in depth it was read across.
 * `depth` is (height, width); `u` and `v` are each column's and each row's
 * line of sight, (x - cx) / f and (y - cy) / f. Differences are taken
 * `step` pixels apart, on whichever side of a pixel the depth changes
 * less, and both sides must exist somewhere: height and width are more
 * than twice `step`.
 */
static void SM_FN(normals)(
    float *nx, float *ny, float *nz, float *jump, const float *depth, const float *u,
    const float *v, ptrdiff_t height, ptrdiff_t width, ptrdiff_t step) {
    for (ptrdiff_t r = 0; r < height; r++) {
        /* A row within `step` of the top has only the rows below it to
         * difference with, and one that near the bottom only those above. */
        const ptrdiff_t fa = r + step < height ? r : r - step;
        const ptrdiff_t ba = r - step >= 0 ? r - step : r;
        const ptrdiff_t at = r * width;
        normals_row(
            nx + at, ny + at, nz + at, jump + at, depth + at,
            depth + fa * width, depth + (fa + step) * width,
            depth + ba * width, depth + (ba + step) * width,
            u, v[r], v[fa], v[fa + step], v[ba], v[ba + step], width, step);
    }
}

/* -------------------------------------------------------------- spray */

/** The columns of a puff's row in the table: see road._PUFF_COLUMNS. */
enum {
    PUFF_X, PUFF_Y, PUFF_LONG_X, PUFF_LONG_Y, PUFF_LONG_RADIUS, PUFF_SHORT_RADIUS,
    PUFF_STRENGTH, PUFF_GROUND_ROW, PUFF_REACH_M, PUFF_RADII_OUT,
    PUFF_GATE_X, PUFF_GATE_Y, PUFF_KEPT_X, PUFF_KEPT_Y, PUFF_FEATHERED, PUFF_COLUMNS
};

/** One row of a puff's Gaussian over `n` columns: exp(-(a^2 + b^2) / 2),
 * with a and b the place along the puff's two axes in units of its radii.
 * `column` holds the columns' numbers less the window's first, which is
 * `left` from the puff's middle less that; `dy` is the row's own distance.
 * Multiplications and a clamp (`exp_of_minus`), and no call.
 */
static SM_NOINLINE void puff_row(
    float *restrict blob, const float *restrict column, ptrdiff_t n, float left, float dy,
    float long_x, float long_y, float per_long, float per_short) {
    SM_VECTORISED for (ptrdiff_t i = 0; i < n; i++) {
        const float dx = column[i] + left;
        const float a = (dx * long_x + dy * long_y) * per_long;
        const float b = (dy * long_x - dx * long_y) * per_short;
        blob[i] = exp_of_minus(0.5f * (a * a + b * b));
    }
}

/** A row of a puff cut off behind a line, softly: `past` the line at the
 * window's first column, growing by `kept_x` a column.
 */
static SM_NOINLINE void puff_gate_row(
    float *restrict blob, const float *restrict column, ptrdiff_t n, float past, float kept_x,
    float per_feathered) {
    SM_VECTORISED for (ptrdiff_t i = 0; i < n; i++) {
        const float start = clamp01((past + column[i] * kept_x) * per_feathered);
        blob[i] = blob[i] * start * start * (3.0f - 2.0f * start);
    }
}

/** A row of a puff added into the spray, less what the vehicles standing
 * nearer than it hide: over a few rows, and not at a line.
 */
static SM_NOINLINE void puff_add_row(
    float *restrict spray, const float *restrict blob, const float *restrict stands,
    const float *restrict hidden, ptrdiff_t n, float strength, float ground_row,
    float per_feather) {
    SM_VECTORISED for (ptrdiff_t i = 0; i < n; i++) {
        const float in_front = clamp01((stands[i] - ground_row) * per_feather);
        spray[i] += strength * (blob[i] * (1.0f - hidden[i] * in_front));
    }
}

/** As ``puff_add_row``, less also what the scene hides: what is much
 * nearer the lens than the puff's vehicle, unless it is a vehicle itself.
 */
static SM_NOINLINE void puff_add_row_behind(
    float *restrict spray, const float *restrict blob, const float *restrict stands,
    const float *restrict hidden, const float *restrict depth, ptrdiff_t n, float strength,
    float ground_row, float per_feather, float per_reach, float hides_wholly, float per_span) {
    SM_VECTORISED for (ptrdiff_t i = 0; i < n; i++) {
        const float in_front = clamp01((stands[i] - ground_row) * per_feather);
        const float clear = clamp01((depth[i] * per_reach - hides_wholly) * per_span);
        const float inside = clamp01(2.0f * hidden[i] - 1.0f);
        const float shown = blob[i] * (1.0f - hidden[i] * in_front);
        spray[i] += strength * (shown * (clear > inside ? clear : inside));
    }
}

/** road._add_puff for every row of `puffs`, (count, 15) doubles, in their
 * order: each a Gaussian with its own axes and radii, added into `spray`
 * over the window it reaches, less what hides it. `stands`, `hidden` and
 * `depth` (which may be NULL) are (height, width) like `spray`.
 */
static void SM_FN(spray_puffs)(
    float *spray, const float *stands, const float *hidden, const float *depth,
    ptrdiff_t height, ptrdiff_t width, const double *puffs, ptrdiff_t count, float feather,
    float hides_wholly, float stands_nearer) {
    float column[SM_BLOCK], blob[SM_BLOCK];
    for (ptrdiff_t i = 0; i < SM_BLOCK; i++) {
        column[i] = (float)i;
    }
    const float per_feather = 1.0f / feather;
    const float per_span = 1.0f / (stands_nearer - hides_wholly);
    for (ptrdiff_t k = 0; k < count; k++) {
        const double *puff = puffs + k * PUFF_COLUMNS;
        const double widest = puff[PUFF_LONG_RADIUS] > puff[PUFF_SHORT_RADIUS]
            ? puff[PUFF_LONG_RADIUS] : puff[PUFF_SHORT_RADIUS];
        const double out = puff[PUFF_RADII_OUT] * widest;
        /* Not a number, or wider than any frame: nothing a picture holds. */
        if (!(out >= 0.0 && out < 1e6 && fabs(puff[PUFF_X]) < 1e7 && fabs(puff[PUFF_Y]) < 1e7)) {
            continue;
        }
        const ptrdiff_t reach = (ptrdiff_t)out + 1;
        const ptrdiff_t at_x = (ptrdiff_t)puff[PUFF_X], at_y = (ptrdiff_t)puff[PUFF_Y];
        const ptrdiff_t x0 = at_x - reach < 0 ? 0 : at_x - reach;
        const ptrdiff_t x1 = at_x + reach + 1 > width ? width : at_x + reach + 1;
        const ptrdiff_t y0 = at_y - reach < 0 ? 0 : at_y - reach;
        const ptrdiff_t y1 = at_y + reach + 1 > height ? height : at_y + reach + 1;
        if (x0 >= x1 || y0 >= y1) {
            continue;
        }
        const float x = (float)puff[PUFF_X], y = (float)puff[PUFF_Y];
        const float long_x = (float)puff[PUFF_LONG_X], long_y = (float)puff[PUFF_LONG_Y];
        const double long_radius = puff[PUFF_LONG_RADIUS], short_radius = puff[PUFF_SHORT_RADIUS];
        const float per_long = 1.0f / (float)(long_radius > 1.0 ? long_radius : 1.0);
        const float per_short = 1.0f / (float)(short_radius > 1.0 ? short_radius : 1.0);
        const float strength = (float)puff[PUFF_STRENGTH];
        const float ground_row = (float)puff[PUFF_GROUND_ROW];
        const int gated = puff[PUFF_FEATHERED] > 0.0;
        const float kept_x = (float)puff[PUFF_KEPT_X], kept_y = (float)puff[PUFF_KEPT_Y];
        const float gate_x = (float)puff[PUFF_GATE_X], gate_y = (float)puff[PUFF_GATE_Y];
        const float per_feathered =
            1.0f / (float)(puff[PUFF_FEATHERED] > 1.0 ? puff[PUFF_FEATHERED] : 1.0);
        const int behind = depth != NULL && puff[PUFF_REACH_M] > 0.0;
        const float per_reach = behind ? 1.0f / (float)puff[PUFF_REACH_M] : 0.0f;
        for (ptrdiff_t v = y0; v < y1; v++) {
            const float dy = (float)v - y;
            for (ptrdiff_t u = x0; u < x1; u += SM_BLOCK) {
                const ptrdiff_t n = x1 - u < SM_BLOCK ? x1 - u : SM_BLOCK;
                const ptrdiff_t here = v * width + u;
                puff_row(blob, column, n, (float)u - x, dy, long_x, long_y, per_long, per_short);
                if (gated) {
                    const float past = ((float)v - gate_y) * kept_y + ((float)u - gate_x) * kept_x;
                    puff_gate_row(blob, column, n, past, kept_x, per_feathered);
                }
                if (behind) {
                    puff_add_row_behind(
                        spray + here, blob, stands + here, hidden + here, depth + here, n,
                        strength, ground_row, per_feather, per_reach, hides_wholly, per_span);
                } else {
                    puff_add_row(
                        spray + here, blob, stands + here, hidden + here, n, strength,
                        ground_row, per_feather);
                }
            }
        }
    }
}

/* ------------------------------------------------------------ lit air */

/** A direction in a plane, as the cosine and the sine of its angle. */
typedef struct {
    float c, s;
} sm_turn;

/** The direction halfway between two: their sum, made a unit. Good while
 * the two are less than half a turn apart, which the two ends of a line of
 * sight seen from a lamp always are.
 */
static inline sm_turn halfway(sm_turn a, sm_turn b) {
    const float c = a.c + b.c, s = a.s + b.s;
    const float square = c * c + s * s;
    const float unit = 1.0f / sqrtf(square > 1e-12f ? square : 1e-12f);
    const sm_turn mid = {c * unit, s * unit};
    return mid;
}

/** What one lamp, a block of lines of sight and the air have in common. */
typedef struct {
    /** The lamp's head, the square of its height and of its distance, and how high it is. */
    float x, y, z, tall2, apart, lamp_up;
    /** The square of the nearest a line is taken to pass, and the lamp's spill. */
    float nearest2, spill;
    /** The forward scattering's ``1 + g^2``, ``2 g`` and ``1 - g^2``. */
    float least, twice, most;
} sm_lamp;

/** The lit air at one point of a line: how far below the lamp it is, and
 * how much of the light going that way is scattered on to the lens.
 */
static inline float lit_at(sm_turn at, float over, float rising, const sm_lamp *lamp) {
    const float below = clamp01(at.c * over - at.s * rising);
    const float turned = lamp->least + lamp->twice * at.s;
    const float forward = lamp->most / (turned * sqrtf(turned));
    return forward * (lamp->spill + (1.0f - lamp->spill) * (below * below));
}

/** One lamp's light on the air along each of `n` lines of sight, added
 * into `air`, and that times how far off into `reach`.
 *
 * A line that passes the lamp at `passes` is seen from the lamp, at `s`
 * along it from the nearest approach, in the direction (passes, s). The
 * line's two ends are two such directions; seven more between them come by
 * halving, and Simpson's rule sums the nine. The angle between neighbours
 * comes from its sine by the arcsine's series. So there is no call in the
 * loop: multiplications, divisions, square roots and clamps, in the order
 * night_light._air_of_lamp has them.
 */
static SM_NOINLINE void air_of_lamp_block(
    float *restrict air, float *restrict reach, const float *restrict rx,
    const float *restrict ry, const float *restrict rz, const float *restrict far,
    const float *restrict rising, ptrdiff_t n, const sm_lamp *lamp) {
    const sm_lamp of = *lamp;
    const float sixth = (float)(1.0 / 6.0), fortieth = (float)(3.0 / 40.0);
    const float last_term = (float)(15.0 / 336.0), third = (float)(1.0 / 3.0);
    SM_VECTORISED for (ptrdiff_t i = 0; i < n; i++) {
        const float along = rx[i] * of.x + ry[i] * of.y + rz[i] * of.z;
        const float clear = of.apart - along * along;
        const float nearness = 1.0f / sqrtf(clear > of.nearest2 ? clear : of.nearest2);
        const float over = (of.lamp_up - along * rising[i]) * nearness;
        const float first = -along * nearness, last = (far[i] - along) * nearness;
        const float unit_first = 1.0f / sqrtf(1.0f + first * first);
        const float unit_last = 1.0f / sqrtf(1.0f + last * last);
        const sm_turn t0 = {unit_first, first * unit_first};
        const sm_turn t8 = {unit_last, last * unit_last};
        const sm_turn t4 = halfway(t0, t8);
        const sm_turn t2 = halfway(t0, t4), t6 = halfway(t4, t8);
        const sm_turn t1 = halfway(t0, t2), t3 = halfway(t2, t4);
        const sm_turn t5 = halfway(t4, t6), t7 = halfway(t6, t8);
        const float sine = t0.c * t1.s - t0.s * t1.c;
        const float square = sine * sine;
        const float step =
            sine * (1.0f + square * (sixth + square * (fortieth + square * last_term)));
        float summed = lit_at(t0, over, rising[i], &of);
        summed += 4.0f * lit_at(t1, over, rising[i], &of);
        summed += 2.0f * lit_at(t2, over, rising[i], &of);
        summed += 4.0f * lit_at(t3, over, rising[i], &of);
        summed += 2.0f * lit_at(t4, over, rising[i], &of);
        summed += 4.0f * lit_at(t5, over, rising[i], &of);
        summed += 2.0f * lit_at(t6, over, rising[i], &of);
        summed += 4.0f * lit_at(t7, over, rising[i], &of);
        summed += lit_at(t8, over, rising[i], &of);
        const float lit = of.tall2 * (step * third * summed * nearness);
        const float nearest = along > 0.0f ? along : 0.0f;
        air[i] += lit;
        reach[i] += lit * (nearest < far[i] ? nearest : far[i]);
    }
}

/** night_light.street_air's sum: every lamp's light on the air along every
 * line of sight. `rx`, `ry`, `rz` are the lines' unit directions, `far`
 * how long each is and `rising` its direction against `up`; `lamps` is
 * (count, 4), a head's place and its height. A block of lines at a time,
 * every lamp over it, so that the seven planes of a block stay in the
 * cache; a pixel's lamps are still summed in their order.
 */
static void SM_FN(street_air)(
    float *air, float *reach, const float *rx, const float *ry, const float *rz,
    const float *far, const float *rising, ptrdiff_t pixels, const float *lamps,
    ptrdiff_t count, const float *up, float nearest2, float spill, float forward) {
    for (ptrdiff_t done = 0; done < pixels; done += SM_BLOCK) {
        const ptrdiff_t n = pixels - done < SM_BLOCK ? pixels - done : SM_BLOCK;
        for (ptrdiff_t k = 0; k < count; k++) {
            const float x = lamps[4 * k], y = lamps[4 * k + 1], z = lamps[4 * k + 2];
            const float tall = lamps[4 * k + 3];
            const sm_lamp lamp = {
                x, y, z, tall * tall, x * x + y * y + z * z,
                x * up[0] + y * up[1] + z * up[2], nearest2, spill,
                1.0f + forward * forward, 2.0f * forward, 1.0f - forward * forward,
            };
            air_of_lamp_block(
                air + done, reach + done, rx + done, ry + done, rz + done, far + done,
                rising + done, n, &lamp);
        }
    }
}

#ifdef SM_NATIVE
/** The native tier is compiled with -march=native, for whatever the build
 * machine had. Run on a CPU without one of those instruction sets it would
 * fault, so the dispatcher asks first. The list is the sets a compiler uses
 * for the loops in this file; a build copied to a different CPU is not
 * supported beyond that, and says so in docs/native_kernels.md.
 */
int sm_native_usable(void) {
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

/** This tier's kernels, exported to ``sm_dispatch.c`` as ``sm_table_<tier>``. */
const sm_kernel_table SM_FN(sm_table) = {
    SM_STR(SM_TIER),
    SM_FN(encode_u8),
    SM_FN(encode_f32),
    SM_FN(decode_u8),
    SM_FN(render_flakes),
    SM_FN(render_streaks),
    SM_FN(wet_composite),
    SM_FN(mirror_rows),
    SM_FN(wet_reflection),
    SM_FN(edge_preserving),
    SM_FN(dirt_emphasise),
    SM_FN(dirt_composite),
    SM_FN(veil_blend),
    SM_FN(overcast_blend),
    SM_FN(grow_alike),
    SM_FN(place_apart),
    SM_FN(water_lobes),
    SM_FN(droplet_composite),
    SM_FN(splat_discs),
    SM_FN(street_air),
    SM_FN(spray_puffs),
    SM_FN(normals),
    SM_FN(nir_reflectance),
    SM_FN(night_ir_light),
    SM_FN(night_ir_expose),
    SM_FN(night_colour_light),
    SM_FN(night_colour_expose),
    SM_FN(fog_lost),
    SM_FN(fog_veil),
    SM_FN(snow_settled),
    SM_FN(ground_tile),
    SM_FN(street_light),
    SM_FN(headlamp_beams),
    SM_FN(wheel_tracks),
    SM_FN(lamps_lit),
    SM_FN(spray_and_lamps),
    SM_FN(road_along),
    SM_FN(road_places),
};
