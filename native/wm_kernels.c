/* The kernels, in plain C written for the compiler to vectorise.
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
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "wm_kernels.h"

#ifndef WM_TIER
#define WM_TIER portable
#endif
#define WM_CAT_(a, b) a##_##b
#define WM_CAT(a, b) WM_CAT_(a, b)
#define WM_FN(name) WM_CAT(name, WM_TIER)
#define WM_STR_(a) #a
#define WM_STR(a) WM_STR_(a)
#define WM_NOINLINE __attribute__((noinline))
/* Marks a loop the build is checked to have vectorised (test_native_build.py). */
#define WM_VECTORISED

static inline float clamp01(float v) {
    /* NaN compares false both times and comes out as 0. */
    v = v > 0.0f ? v : 0.0f;
    return v < 1.0f ? v : 1.0f;
}

/* ------------------------------------------------------------ colour */

/* Table indices for a block of values. Kept apart from the lookups so that
 * it is one loop of float arithmetic and nothing else: written as a single
 * loop with the lookup in it, GCC compiled the clamp as two branches, which
 * a picture's highlights and shadows mispredict, and vectorised nothing. */
#define WM_BLOCK 512

static inline int table_indices(const float *restrict in, int32_t *restrict index, size_t n) {
    int bad = 0;
    WM_VECTORISED for (size_t i = 0; i < n; i++) {
        const float v = in[i];
        bad |= (v != v);
        index[i] = (int32_t)(clamp01(v) * 65535.0f);
    }
    return bad;
}

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

static void WM_FN(decode_u8)(
    const uint8_t *restrict in, float *restrict out, size_t n, const float *restrict table) {
    for (size_t i = 0; i < n; i++) {
        out[i] = table[in[i]];
    }
}

/* ------------------------------------------------------------- flakes */

#define WM_MAX_BLADES 32
/* Half the side, in pixels, of the largest sprite the kernel will draw. */
#define WM_MAX_REACH 4096.0
static const double WM_PI = 3.14159265358979323846;

/* cv2.getGaussianKernel(n, sigma <= 0), which is what
 * cv2.GaussianBlur(src, (n, n), 0) filters with. */
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

static inline ptrdiff_t reflect101(ptrdiff_t i, ptrdiff_t n) {
    if (n == 1) {
        return 0;
    }
    while (i < 0 || i >= n) {
        i = i < 0 ? -i : 2 * n - 2 - i;
    }
    return i;
}

/* cv2.getGaussianKernel(n, sigma) for sigma > 0, as float32. */
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

/* Separable blur of a width x height patch with a symmetric kernel and
 * BORDER_REFLECT_101, in place: cv2.GaussianBlur's filter. `tmp` is scratch
 * the size of the patch, `line` of width + ksize floats. */
static void blur_patch(
    float *restrict patch, float *restrict tmp, float *restrict line,
    ptrdiff_t width, ptrdiff_t height, const float *kernel, int ksize) {
    const int half = ksize >> 1;
    const float centre = kernel[half];
    for (ptrdiff_t r = 0; r < height; r++) {
        const float *restrict src = patch + r * width;
        float *restrict dst = tmp + r * width;
        memcpy(line + half, src, (size_t)width * sizeof(float));
        for (ptrdiff_t c = 1; c <= half; c++) {
            line[half - c] = src[reflect101(-c, width)];
            line[half + width - 1 + c] = src[reflect101(width - 1 + c, width)];
        }
        for (ptrdiff_t c = 0; c < width; c++) {
            dst[c] = centre * line[c + half];
        }
        for (int k = 1; k <= half; k++) {
            const float weight = kernel[half + k];
            const float *restrict left = line + half - k;
            const float *restrict right = line + half + k;
            for (ptrdiff_t c = 0; c < width; c++) {
                dst[c] += weight * (left[c] + right[c]);
            }
        }
    }
    for (ptrdiff_t r = 0; r < height; r++) {
        float *restrict dst = patch + r * width;
        const float *restrict mid = tmp + r * width;
        for (ptrdiff_t c = 0; c < width; c++) {
            dst[c] = centre * mid[c];
        }
        for (int k = 1; k <= half; k++) {
            const float weight = kernel[half + k];
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
/* Distance to a segment: a circle swept along its motion is a capsule. */
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

/* A polygon swept along a segment: each edge's support grows by the
 * segment's reach along that edge's normal. */
static inline void first_plane_row(
    float *restrict row, const float *restrict dx, ptrdiff_t size,
    float dy, float nx, float ny, float support) {
    WM_VECTORISED for (ptrdiff_t c = 0; c < size; c++) {
        row[c] = (dx[c] * nx + dy * ny) / support;
    }
}

static inline void next_plane_row(
    float *restrict row, const float *restrict dx, ptrdiff_t size,
    float dy, float nx, float ny, float support) {
    WM_VECTORISED for (ptrdiff_t c = 0; c < size; c++) {
        const float plane = (dx[c] * nx + dy * ny) / support;
        row[c] = row[c] > plane ? row[c] : plane;
    }
}

/* Fast mode: the division by the support is folded into the normal, once
 * per sprite, so a plane is two multiplies and an add. The quotient and the
 * product with a rounded reciprocal differ in the last bit or two. */
static inline void first_plane_row_fast(
    float *restrict row, const float *restrict dx, ptrdiff_t size, float dy, float nx, float ny) {
    const float dyny = dy * ny;
    WM_VECTORISED for (ptrdiff_t c = 0; c < size; c++) {
        row[c] = dx[c] * nx + dyny;
    }
}

static inline void next_plane_row_fast(
    float *restrict row, const float *restrict dx, ptrdiff_t size, float dy, float nx, float ny) {
    const float dyny = dy * ny;
    WM_VECTORISED for (ptrdiff_t c = 0; c < size; c++) {
        const float plane = dx[c] * nx + dyny;
        row[c] = row[c] > plane ? row[c] : plane;
    }
}

/* The cat's-eye stop: a second, displaced aperture. */
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

/* Distance to brightness: a defined rim, then the aberration profile. */
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

/* Fast mode: the two divisions by constants as multiplications. */
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

static inline void max_row(float *restrict dst, const float *restrict src, ptrdiff_t size) {
    WM_VECTORISED for (ptrdiff_t c = 0; c < size; c++) {
        dst[c] = dst[c] > src[c] ? dst[c] : src[c];
    }
}

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

        for (ptrdiff_t r = 0; r < size; r++) {
            float *row = sprite + r * size;
            const float dy = axis[r];
            if (blades >= 3 && fast) {
                first_plane_row_fast(row, axis, size, dy, scaled_x[0], scaled_y[0]);
                for (int b = 1; b < blades; b++) {
                    next_plane_row_fast(row, axis, size, dy, scaled_x[b], scaled_y[b]);
                }
            } else if (blades >= 3) {
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

        for (ptrdiff_t r = dst_y0; r < dst_y1; r++) {
            max_row(
                canvas + r * width + dst_x0,
                sprite + (r - top) * size + (dst_x0 - left),
                dst_x1 - dst_x0);
        }
    }
    free(sprite), free(tmp), free(axis), free(line);
    return 0;
}

/* ------------------------------------------------------------ streaks */

/* A streak is a polyline that cv2.polylines draws into a small patch. What
 * follows sets the pixels OpenCV sets, by the arithmetic OpenCV uses
 * (modules/imgproc/src/drawing.cpp: Line, Line2, FillConvexPoly, Circle,
 * ThickLine, PolyLine), for the case this library has: a float32 image, on
 * which OpenCV ignores LINE_AA and draws 8-connected. The tests draw
 * thousands of streaks both ways and compare the pixels.
 *
 * The pixels are not written as they are found. They are collected, each
 * once, and the streak's brightness is laid down afterwards: drawing only
 * ever sets a pixel to the same value again, and the list is all that the
 * beading and the accumulation need to visit. */

#define WM_XY_SHIFT 16
#define WM_XY_ONE (1 << WM_XY_SHIFT)

/* A pixel of a patch. Sixteen bits a coordinate: the pair is the size of
 * the flat index it replaces, and reading it back needs no division. (That
 * made no difference that could be measured.) Patches are held to
 * WM_MAX_SIDE, well inside sixteen bits. */
typedef struct {
    int16_t x, y;
} wm_pixel;

typedef struct {
    uint8_t *seen;
    wm_pixel *pixels;
    ptrdiff_t n_pixels;
    int32_t patch_w, patch_h;
} wm_patch;

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

static inline void span(wm_patch *patch, int64_t py, int64_t left, int64_t right) {
    left = left < 0 ? 0 : left;
    right = right >= patch->patch_w ? patch->patch_w - 1 : right;
    for (int64_t px = left; px <= right; px++) {
        plot(patch, px, py);
    }
}

/* Line: OpenCV's LineIterator, 8-connected, left to right. Both ends must
 * lie inside the patch, which the caller checks, since OpenCV clips a line
 * before walking it and a clipped line takes other pixels. */
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

/* Line2: a line between points in 16.16 fixed point, the outline of a
 * filled polygon. */
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

/* FillConvexPoly for four vertices in 16.16 fixed point, not antialiased. */
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

/* Circle, filled: the round cap at the end of a wide segment. */
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

/* ThickLine for a thickness above one: a filled rectangle about the segment
 * and a disc at each end `caps` asks for (bit 0 the start, bit 1 the end). */
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

/* The longest side of a patch, in pixels, and the widest blur the kernel
 * will draw. */
#define WM_MAX_SIDE 8192
#define WM_MAX_SIGMA 256.0f

/* The kernel size cv2.GaussianBlur(patch, (0, 0), sigma) takes on float32. */
static inline int streak_kernel_size(float sigma) {
    return (int)nearbyint((double)sigma * 8.0 + 1.0) | 1;
}

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
        for (ptrdiff_t k = 0; k < patch.n_pixels; k++) {
            const int32_t px = patch.pixels[k].x, py = patch.pixels[k].y;
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
                layer[(ptrdiff_t)fy * width + fx] += value;
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
            for (int32_t r = r0; r < r1; r++) {
                float *restrict dst = layer + (ptrdiff_t)(top + r) * width + left;
                const float *restrict src = soft + (ptrdiff_t)r * pw;
                for (int32_t c = c0; c < c1; c++) {
                    dst[c] += src[c];
                }
            }
        }
    }
    free(patch.seen), free(patch.pixels), free(soft), free(tmp), free(line);
    return 0;
}

const wm_kernel_table WM_FN(wm_table) = {
    WM_STR(WM_TIER),
    WM_FN(encode_u8),
    WM_FN(encode_f32),
    WM_FN(decode_u8),
    WM_FN(render_flakes),
    WM_FN(render_streaks),
};
