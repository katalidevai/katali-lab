/* Katali-GGUF x86 SIMD kernels — Apache-2.0 */
#include "katali_gguf_simd.h"
#include "katali_gguf_dtype.h"
#include "katali_gguf_fp.h"

#include <string.h>
#include <stdlib.h>
#include <math.h>

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#define KATALI_X86 1
#else
#define KATALI_X86 0
#endif

#if KATALI_X86
#include <immintrin.h>
#if defined(_MSC_VER)
#include <intrin.h>
#endif
#endif

/* GCC/Clang: compile these functions for AVX2 even though the translation unit
 * is built for the baseline ISA, so one binary runs everywhere. MSVC always
 * compiles intrinsics for the enabled ISA, so no attribute is needed there. */
#if KATALI_X86 && (defined(__GNUC__) || defined(__clang__))
#define KATALI_AVX2 __attribute__((target("avx2,fma,f16c")))
#else
#define KATALI_AVX2
#endif

/* ==================================================================== *
 * Runtime detection                                                     *
 * ==================================================================== */
static int g_simd = -1;

static int simd_detect(void) {
    const char *off = getenv("KATALI_GGUF_NO_SIMD");
    if (off && *off && off[0] != '0') return 0;
#if KATALI_X86
#if defined(__GNUC__) || defined(__clang__)
    __builtin_cpu_init();
    if (!__builtin_cpu_supports("avx2")) return 0;
    if (!__builtin_cpu_supports("fma")) return 0;
    /* Every AVX2-capable CPU also has F16C, so no separate probe is needed;
     * GCC 10's __builtin_cpu_supports does not accept "f16c" anyway. */
    return 1;
#elif defined(_MSC_VER)
    int r[4];
    __cpuid(r, 0);
    if (r[0] < 7) return 0;
    __cpuidex(r, 7, 0);
    if (!(r[1] & (1 << 5))) return 0;   /* AVX2 */
    __cpuid(r, 1);
    if (!(r[2] & (1 << 12))) return 0;  /* FMA  */
    if (!(r[2] & (1 << 29))) return 0;  /* F16C */
    return 1;
#else
    return 0;
#endif
#else
    return 0;
#endif
}

int katali_ggml_simd_available(void) {
    if (g_simd < 0) g_simd = simd_detect();
    return g_simd;
}

const char *katali_ggml_simd_name(void) {
    return katali_ggml_simd_available() ? "avx2+fma" : "scalar";
}

#if KATALI_X86

/* ==================================================================== *
 * Helpers                                                               *
 * ==================================================================== */
KATALI_AVX2 static inline float hsum256_ps(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    __m128 sh = _mm_movehdup_ps(lo);
    __m128 s = _mm_add_ps(lo, sh);
    sh = _mm_movehl_ps(sh, s);
    s = _mm_add_ss(s, sh);
    return _mm_cvtss_f32(s);
}

static inline uint16_t simd_ld_u16(const uint8_t *p) {
    uint16_t v; memcpy(&v, p, 2); return v;
}

/* Expand 8 unsigned bytes to 8 floats. */
KATALI_AVX2 static inline __m256 u8x8_to_ps(__m128i bytes8) {
    return _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(bytes8));
}

/* Expand 8 signed bytes to 8 floats. */
KATALI_AVX2 static inline __m256 i8x8_to_ps(__m128i bytes8) {
    return _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(bytes8));
}

/* ==================================================================== *
 * F32 / F16 / BF16                                                      *
 * ==================================================================== */
/* F32 rows are loaded through a float-typed pointer on purpose: loading floats
 * through a pointer derived from const uint8_t* was observed to be miscompiled
 * by GCC 10 (wrong lanes for n >= 16), and this is the only correct spelling. */
KATALI_AVX2 static float dot_f32_avx2(const float *s, const float *x, uint64_t n) {
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    uint64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(s + i), _mm256_loadu_ps(x + i), a0);
        a1 = _mm256_fmadd_ps(_mm256_loadu_ps(s + i + 8), _mm256_loadu_ps(x + i + 8), a1);
    }
    float acc = hsum256_ps(_mm256_add_ps(a0, a1));
    double tail = 0.0;
    for (; i < n; i++) tail += (double)s[i] * (double)x[i];
    return acc + (float)tail;
}

/* Several rows against one activation vector, sharing the x loads.
 *
 * The prefill matmul decodes a weight row once and then dots it against every
 * prompt position; measured on 8B ffn_gate (12288x4096, B=32) that sweep of the
 * activation matrix is the dominant cost (~6.4 GB re-read per call at ~195 GB/s
 * against 28 MB of weights), so reading x once for several rows is the win.
 *
 * Each row keeps its own accumulator pair, the same k-order and the same tail
 * handling as dot_f32_avx2, so every output is bit-identical to the single-row
 * kernel (asserted in test_core, not assumed). */
KATALI_AVX2 static void dot_f32_rows2_avx2(const float *r0, const float *r1,
                                           const float *x, uint64_t n, float *out) {
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    __m256 b0 = _mm256_setzero_ps(), b1 = _mm256_setzero_ps();
    uint64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m256 x0 = _mm256_loadu_ps(x + i);
        __m256 x1 = _mm256_loadu_ps(x + i + 8);
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(r0 + i), x0, a0);
        a1 = _mm256_fmadd_ps(_mm256_loadu_ps(r0 + i + 8), x1, a1);
        b0 = _mm256_fmadd_ps(_mm256_loadu_ps(r1 + i), x0, b0);
        b1 = _mm256_fmadd_ps(_mm256_loadu_ps(r1 + i + 8), x1, b1);
    }
    double t0 = 0.0, t1 = 0.0;
    for (; i < n; i++) {
        t0 += (double)r0[i] * (double)x[i];
        t1 += (double)r1[i] * (double)x[i];
    }
    out[0] = hsum256_ps(_mm256_add_ps(a0, a1)) + (float)t0;
    out[1] = hsum256_ps(_mm256_add_ps(b0, b1)) + (float)t1;
}

KATALI_AVX2 static void dot_f32_rows4_avx2(const float *r0, const float *r1,
                                           const float *r2, const float *r3,
                                           const float *x, uint64_t n, float *out) {
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    __m256 b0 = _mm256_setzero_ps(), b1 = _mm256_setzero_ps();
    __m256 c0 = _mm256_setzero_ps(), c1 = _mm256_setzero_ps();
    __m256 d0 = _mm256_setzero_ps(), d1 = _mm256_setzero_ps();
    uint64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m256 x0 = _mm256_loadu_ps(x + i);
        __m256 x1 = _mm256_loadu_ps(x + i + 8);
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(r0 + i), x0, a0);
        a1 = _mm256_fmadd_ps(_mm256_loadu_ps(r0 + i + 8), x1, a1);
        b0 = _mm256_fmadd_ps(_mm256_loadu_ps(r1 + i), x0, b0);
        b1 = _mm256_fmadd_ps(_mm256_loadu_ps(r1 + i + 8), x1, b1);
        c0 = _mm256_fmadd_ps(_mm256_loadu_ps(r2 + i), x0, c0);
        c1 = _mm256_fmadd_ps(_mm256_loadu_ps(r2 + i + 8), x1, c1);
        d0 = _mm256_fmadd_ps(_mm256_loadu_ps(r3 + i), x0, d0);
        d1 = _mm256_fmadd_ps(_mm256_loadu_ps(r3 + i + 8), x1, d1);
    }
    double t0 = 0.0, t1 = 0.0, t2 = 0.0, t3 = 0.0;
    for (; i < n; i++) {
        t0 += (double)r0[i] * (double)x[i];
        t1 += (double)r1[i] * (double)x[i];
        t2 += (double)r2[i] * (double)x[i];
        t3 += (double)r3[i] * (double)x[i];
    }
    out[0] = hsum256_ps(_mm256_add_ps(a0, a1)) + (float)t0;
    out[1] = hsum256_ps(_mm256_add_ps(b0, b1)) + (float)t1;
    out[2] = hsum256_ps(_mm256_add_ps(c0, c1)) + (float)t2;
    out[3] = hsum256_ps(_mm256_add_ps(d0, d1)) + (float)t3;
}

int katali_ggml_f32_rows_dot_avx2(const float *const *rows, int nr,
                                  const float *x, uint64_t n, float *out) {
#if KATALI_X86
    if (!katali_ggml_simd_available()) return -1;
    switch (nr) {
        case 1: out[0] = dot_f32_avx2(rows[0], x, n); return 0;
        case 2: dot_f32_rows2_avx2(rows[0], rows[1], x, n, out); return 0;
        case 4: dot_f32_rows4_avx2(rows[0], rows[1], rows[2], rows[3], x, n, out); return 0;
        default: return -1;
    }
#else
    (void)rows; (void)nr; (void)x; (void)n; (void)out;
    return -1;
#endif
}

KATALI_AVX2 static float dot_f16_avx2(const uint8_t *s, const float *x, uint64_t n) {
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    uint64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m128i h0 = _mm_loadu_si128((const __m128i *)(s + 2 * i));
        __m128i h1 = _mm_loadu_si128((const __m128i *)(s + 2 * i + 16));
        a0 = _mm256_fmadd_ps(_mm256_cvtph_ps(h0), _mm256_loadu_ps(x + i), a0);
        a1 = _mm256_fmadd_ps(_mm256_cvtph_ps(h1), _mm256_loadu_ps(x + i + 8), a1);
    }
    float acc = hsum256_ps(_mm256_add_ps(a0, a1));
    double tail = 0.0;
    for (; i < n; i++)
        tail += (double)katali_f16_to_f32(simd_ld_u16(s + 2 * i)) * (double)x[i];
    return acc + (float)tail;
}

KATALI_AVX2 static float dot_bf16_avx2(const uint8_t *s, const float *x, uint64_t n) {
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    uint64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        /* bf16 is the high half of f32: widen 16 bits then shift into place. */
        __m256i w0 = _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)(s + 2 * i)));
        __m256i w1 = _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)(s + 2 * i + 16)));
        __m256 f0 = _mm256_castsi256_ps(_mm256_slli_epi32(w0, 16));
        __m256 f1 = _mm256_castsi256_ps(_mm256_slli_epi32(w1, 16));
        a0 = _mm256_fmadd_ps(f0, _mm256_loadu_ps(x + i), a0);
        a1 = _mm256_fmadd_ps(f1, _mm256_loadu_ps(x + i + 8), a1);
    }
    float acc = hsum256_ps(_mm256_add_ps(a0, a1));
    double tail = 0.0;
    for (; i < n; i++)
        tail += (double)katali_bf16_to_f32(simd_ld_u16(s + 2 * i)) * (double)x[i];
    return acc + (float)tail;
}

/* ==================================================================== *
 * Q8_0 and Q4_0                                                         *
 * ==================================================================== */
KATALI_AVX2 static float dot_q8_0_avx2(const uint8_t *s, const float *x, uint64_t nb) {
    double acc = 0.0;
    for (uint64_t b = 0; b < nb; b++) {
        const uint8_t *blk = s + b * 34;
        float d = katali_f16_to_f32(simd_ld_u16(blk));
        const int8_t *qs = (const int8_t *)(blk + 2);
        const float *xb = x + b * 32;
        __m256 a = _mm256_setzero_ps();
        for (int k = 0; k < 4; k++) {
            __m128i b8 = _mm_loadl_epi64((const __m128i *)(qs + 8 * k));
            a = _mm256_fmadd_ps(i8x8_to_ps(b8), _mm256_loadu_ps(xb + 8 * k), a);
        }
        acc += (double)d * (double)hsum256_ps(a);
    }
    return (float)acc;
}

KATALI_AVX2 static float dot_q4_0_avx2(const uint8_t *s, const float *x, uint64_t nb) {
    const __m128i mask0F = _mm_set1_epi8(0x0F);
    double acc = 0.0;
    for (uint64_t b = 0; b < nb; b++) {
        const uint8_t *blk = s + b * 18;
        float d = katali_f16_to_f32(simd_ld_u16(blk));
        const uint8_t *qs = blk + 2;
        const float *xb = x + b * 32;
        __m256 aL = _mm256_setzero_ps(), aH = _mm256_setzero_ps();
        __m256 xs = _mm256_setzero_ps();
        for (int k = 0; k < 2; k++) {
            __m128i b8 = _mm_loadl_epi64((const __m128i *)(qs + 8 * k));
            __m256 lo = u8x8_to_ps(_mm_and_si128(b8, mask0F));
            __m256 hi = u8x8_to_ps(_mm_and_si128(_mm_srli_epi16(b8, 4), mask0F));
            __m256 xl = _mm256_loadu_ps(xb + 8 * k);
            __m256 xh = _mm256_loadu_ps(xb + 16 + 8 * k);
            aL = _mm256_fmadd_ps(lo, xl, aL);
            aH = _mm256_fmadd_ps(hi, xh, aH);
            xs = _mm256_add_ps(xs, _mm256_add_ps(xl, xh));
        }
        float qsum = hsum256_ps(_mm256_add_ps(aL, aH));
        float xsum = hsum256_ps(xs);
        acc += (double)d * ((double)qsum - 8.0 * (double)xsum);
    }
    return (float)acc;
}

/* ==================================================================== *
 * Q4_K                                                                  *
 * ==================================================================== */
/* 6-bit scale/min unpacking for Q4_K / Q5_K (format definition). */
static inline void simd_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63u;
        *m = q[j + 4] & 63u;
    } else {
        *d = (uint8_t)((q[j + 4] & 0x0Fu) | ((q[j - 4] >> 6) << 4));
        *m = (uint8_t)((q[j + 4] >> 4)   | ((q[j]     >> 6) << 4));
    }
}

/* Per block: 4 groups of 64 outputs from 32 packed bytes each.
 * low nibble  -> first 32 outputs (scale/min index is+0)
 * high nibble -> next 32 outputs  (scale/min index is+1)
 *
 * The per-group scale is applied to the vector accumulator and the horizontal
 * reduction happens once per block, not once per group. */
KATALI_AVX2 static float dot_q4_K_avx2(const uint8_t *s, const float *x, uint64_t nb) {
    const __m128i mask0F = _mm_set1_epi8(0x0F);
    double acc = 0.0;
    for (uint64_t b = 0; b < nb; b++) {
        const uint8_t *blk = s + b * 144;
        float d  = katali_f16_to_f32(simd_ld_u16(blk));
        float mn = katali_f16_to_f32(simd_ld_u16(blk + 2));
        const uint8_t *sc = blk + 4;
        const uint8_t *q  = blk + 16;
        const float *xb = x + b * 256;
        __m256 accL = _mm256_setzero_ps(), accH = _mm256_setzero_ps();
        int is = 0;
        for (int g = 0; g < 4; g++) {
            uint8_t s0, m0, s1, m1;
            simd_scale_min_k4(is + 0, sc, &s0, &m0);
            simd_scale_min_k4(is + 1, sc, &s1, &m1);

            __m256 aL = _mm256_setzero_ps(), aH = _mm256_setzero_ps();
            __m256 xL = _mm256_setzero_ps(), xH = _mm256_setzero_ps();
            for (int k = 0; k < 4; k++) {
                __m128i b8 = _mm_loadl_epi64((const __m128i *)(q + 8 * k));
                __m256 lo = u8x8_to_ps(_mm_and_si128(b8, mask0F));
                __m256 hi = u8x8_to_ps(_mm_and_si128(_mm_srli_epi16(b8, 4), mask0F));
                __m256 xl = _mm256_loadu_ps(xb + g * 64 + 8 * k);
                __m256 xh = _mm256_loadu_ps(xb + g * 64 + 32 + 8 * k);
                aL = _mm256_fmadd_ps(lo, xl, aL);
                aH = _mm256_fmadd_ps(hi, xh, aH);
                xL = _mm256_add_ps(xL, xl);
                xH = _mm256_add_ps(xH, xh);
            }
            accL = _mm256_fmadd_ps(_mm256_set1_ps(d * (float)s0), aL, accL);
            accL = _mm256_fmadd_ps(_mm256_set1_ps(-(mn * (float)m0)), xL, accL);
            accH = _mm256_fmadd_ps(_mm256_set1_ps(d * (float)s1), aH, accH);
            accH = _mm256_fmadd_ps(_mm256_set1_ps(-(mn * (float)m1)), xH, accH);
            q += 32;
            is += 2;
        }
        acc += (double)(hsum256_ps(accL) + hsum256_ps(accH));
    }
    return (float)acc;
}

/* ==================================================================== *
 * Q4_K activation-sum hoisting                                          *
 *                                                                       *
 * dot_q4_K_avx2 rebuilds xL/xH -- the two 8-lane partial sums of the     *
 * activation vector per 64-value group -- for every weight row, although *
 * they depend only on x. Here they are computed once per matvec call and *
 * reused by every row. The accumulation order is identical to the        *
 * in-kernel loop, so results are bit-identical (verified by test_core).  *
 *                                                                       *
 * dot_q4_K_avx2 above is deliberately left unmodified: it stays the      *
 * cross-check reference for the hoisted kernel.                          *
 * ==================================================================== */
static int q4k_sumcache_enabled(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("KATALI_GGUF_NO_Q4K_SUMCACHE");
        v = (e && *e && e[0] != '0') ? 0 : 1;
    }
    return v;
}

static int q4k_dualrow_enabled(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("KATALI_GGUF_NO_Q4K_DUALROW");
        v = (e && *e && e[0] != '0') ? 0 : 1;
    }
    return v;
}

/* One 256-value block -> 4 groups x 2 halves x 8 lanes = 64 floats. */
KATALI_AVX2 static void q4k_sums_avx2(const float *x, uint64_t nb, float *xs) {
    for (uint64_t b = 0; b < nb; b++) {
        const float *xb = x + b * 256;
        float *out = xs + b * 64;
        for (int g = 0; g < 4; g++) {
            __m256 xL = _mm256_setzero_ps(), xH = _mm256_setzero_ps();
            for (int k = 0; k < 4; k++) { /* same order as dot_q4_K_avx2 */
                xL = _mm256_add_ps(xL, _mm256_loadu_ps(xb + g * 64 + 8 * k));
                xH = _mm256_add_ps(xH, _mm256_loadu_ps(xb + g * 64 + 32 + 8 * k));
            }
            _mm256_storeu_ps(out + g * 16 + 0, xL);
            _mm256_storeu_ps(out + g * 16 + 8, xH);
        }
    }
}

/* Same arithmetic as dot_q4_K_avx2; only the source of xL/xH differs. */
KATALI_AVX2 static float dot_q4_K_sums_avx2(const uint8_t *s, const float *x,
                                            uint64_t nb, const float *xs) {
    const __m128i mask0F = _mm_set1_epi8(0x0F);
    double acc = 0.0;
    for (uint64_t b = 0; b < nb; b++) {
        const uint8_t *blk = s + b * 144;
        float d  = katali_f16_to_f32(simd_ld_u16(blk));
        float mn = katali_f16_to_f32(simd_ld_u16(blk + 2));
        const uint8_t *sc = blk + 4;
        const uint8_t *q  = blk + 16;
        const float *xb  = x + b * 256;
        const float *xsb = xs + b * 64;
        __m256 accL = _mm256_setzero_ps(), accH = _mm256_setzero_ps();
        int is = 0;
        for (int g = 0; g < 4; g++) {
            uint8_t s0, m0, s1, m1;
            simd_scale_min_k4(is + 0, sc, &s0, &m0);
            simd_scale_min_k4(is + 1, sc, &s1, &m1);

            __m256 aL = _mm256_setzero_ps(), aH = _mm256_setzero_ps();
            for (int k = 0; k < 4; k++) {
                __m128i b8 = _mm_loadl_epi64((const __m128i *)(q + 8 * k));
                __m256 lo = u8x8_to_ps(_mm_and_si128(b8, mask0F));
                __m256 hi = u8x8_to_ps(_mm_and_si128(_mm_srli_epi16(b8, 4), mask0F));
                aL = _mm256_fmadd_ps(lo, _mm256_loadu_ps(xb + g * 64 + 8 * k), aL);
                aH = _mm256_fmadd_ps(hi, _mm256_loadu_ps(xb + g * 64 + 32 + 8 * k), aH);
            }
            accL = _mm256_fmadd_ps(_mm256_set1_ps(d * (float)s0), aL, accL);
            accL = _mm256_fmadd_ps(_mm256_set1_ps(-(mn * (float)m0)),
                                   _mm256_loadu_ps(xsb + g * 16 + 0), accL);
            accH = _mm256_fmadd_ps(_mm256_set1_ps(d * (float)s1), aH, accH);
            accH = _mm256_fmadd_ps(_mm256_set1_ps(-(mn * (float)m1)),
                                   _mm256_loadu_ps(xsb + g * 16 + 8), accH);
            q += 32;
            is += 2;
        }
        acc += (double)(hsum256_ps(accL) + hsum256_ps(accH));
    }
    return (float)acc;
}

/* Two rows at once, for memory-level parallelism: each row keeps its own
 * weights, scales, minima and accumulators (no cross-row dependency), and the
 * two independent weight-load chains are interleaved in one block/group loop so
 * the CPU can have more loads in flight. Each x/xs slice is loaded once and
 * used by both rows, which also halves the activation reads.
 *
 * The per-row arithmetic is the same sequence of operations as
 * dot_q4_K_sums_avx2, so both outputs are bit-identical to the single-row
 * kernel (asserted in test_core, not assumed). */
KATALI_AVX2 static void dot_q4_K_sums2_avx2(const uint8_t *s1, const uint8_t *s2,
                                            const float *x, uint64_t nb,
                                            const float *xs,
                                            float *y1, float *y2) {
    const __m128i mask0F = _mm_set1_epi8(0x0F);
    double acc1 = 0.0, acc2 = 0.0;
    for (uint64_t b = 0; b < nb; b++) {
        const uint8_t *blk1 = s1 + b * 144;
        const uint8_t *blk2 = s2 + b * 144;
        const float d1  = katali_f16_to_f32(simd_ld_u16(blk1));
        const float mn1 = katali_f16_to_f32(simd_ld_u16(blk1 + 2));
        const float d2  = katali_f16_to_f32(simd_ld_u16(blk2));
        const float mn2 = katali_f16_to_f32(simd_ld_u16(blk2 + 2));
        const uint8_t *sc1 = blk1 + 4, *sc2 = blk2 + 4;
        const uint8_t *q1 = blk1 + 16, *q2 = blk2 + 16;
        const float *xb  = x + b * 256;
        const float *xsb = xs + b * 64;
        __m256 accL1 = _mm256_setzero_ps(), accH1 = _mm256_setzero_ps();
        __m256 accL2 = _mm256_setzero_ps(), accH2 = _mm256_setzero_ps();
        int is = 0;
        for (int g = 0; g < 4; g++) {
            uint8_t as0, am0, as1, am1, bs0, bm0, bs1, bm1;
            simd_scale_min_k4(is + 0, sc1, &as0, &am0);
            simd_scale_min_k4(is + 1, sc1, &as1, &am1);
            simd_scale_min_k4(is + 0, sc2, &bs0, &bm0);
            simd_scale_min_k4(is + 1, sc2, &bs1, &bm1);

            __m256 aL1 = _mm256_setzero_ps(), aH1 = _mm256_setzero_ps();
            __m256 aL2 = _mm256_setzero_ps(), aH2 = _mm256_setzero_ps();
            for (int k = 0; k < 4; k++) {
                /* one load per activation slice serves both rows */
                __m256 xl = _mm256_loadu_ps(xb + g * 64 + 8 * k);
                __m256 xh = _mm256_loadu_ps(xb + g * 64 + 32 + 8 * k);
                __m128i p1 = _mm_loadl_epi64((const __m128i *)(q1 + 8 * k));
                __m128i p2 = _mm_loadl_epi64((const __m128i *)(q2 + 8 * k));
                aL1 = _mm256_fmadd_ps(u8x8_to_ps(_mm_and_si128(p1, mask0F)), xl, aL1);
                aH1 = _mm256_fmadd_ps(u8x8_to_ps(_mm_and_si128(_mm_srli_epi16(p1, 4), mask0F)), xh, aH1);
                aL2 = _mm256_fmadd_ps(u8x8_to_ps(_mm_and_si128(p2, mask0F)), xl, aL2);
                aH2 = _mm256_fmadd_ps(u8x8_to_ps(_mm_and_si128(_mm_srli_epi16(p2, 4), mask0F)), xh, aH2);
            }
            __m256 xL = _mm256_loadu_ps(xsb + g * 16 + 0); /* shared by both rows */
            __m256 xH = _mm256_loadu_ps(xsb + g * 16 + 8);
            accL1 = _mm256_fmadd_ps(_mm256_set1_ps(d1 * (float)as0), aL1, accL1);
            accL1 = _mm256_fmadd_ps(_mm256_set1_ps(-(mn1 * (float)am0)), xL, accL1);
            accH1 = _mm256_fmadd_ps(_mm256_set1_ps(d1 * (float)as1), aH1, accH1);
            accH1 = _mm256_fmadd_ps(_mm256_set1_ps(-(mn1 * (float)am1)), xH, accH1);
            accL2 = _mm256_fmadd_ps(_mm256_set1_ps(d2 * (float)bs0), aL2, accL2);
            accL2 = _mm256_fmadd_ps(_mm256_set1_ps(-(mn2 * (float)bm0)), xL, accL2);
            accH2 = _mm256_fmadd_ps(_mm256_set1_ps(d2 * (float)bs1), aH2, accH2);
            accH2 = _mm256_fmadd_ps(_mm256_set1_ps(-(mn2 * (float)bm1)), xH, accH2);
            q1 += 32;
            q2 += 32;
            is += 2;
        }
        acc1 += (double)(hsum256_ps(accL1) + hsum256_ps(accH1));
        acc2 += (double)(hsum256_ps(accL2) + hsum256_ps(accH2));
    }
    *y1 = (float)acc1;
    *y2 = (float)acc2;
}

#endif /* KATALI_X86 */

int katali_ggml_q4k_sums(const float *x, uint64_t cols, float *sums,
                         uint64_t sums_cap) {
#if KATALI_X86
    if (!x || !sums) return -1;
    if (!q4k_sumcache_enabled() || !katali_ggml_simd_available()) return -1;
    if (cols == 0 || cols % 256 != 0) return -1;
    if (KATALI_Q4K_SUMS_FLOATS(cols) > sums_cap) return -1;
    q4k_sums_avx2(x, cols / 256, sums);
    return 0;
#else
    (void)x; (void)cols; (void)sums; (void)sums_cap;
    return -1;
#endif
}

int katali_ggml_q4k_dot_sums(const void *row, const float *x, uint64_t cols,
                             const float *sums, float *out) {
#if KATALI_X86
    if (!row || !x || !sums || !out) return -1;
    if (!katali_ggml_simd_available()) return -1;
    if (cols == 0 || cols % 256 != 0) return -1;
    *out = dot_q4_K_sums_avx2((const uint8_t *)row, x, cols / 256, sums);
    return 0;
#else
    (void)row; (void)x; (void)cols; (void)sums; (void)out;
    return -1;
#endif
}

/* Two rows in one call, sharing the activation slices. Returns 0 when handled;
 * the two outputs must be bit-identical to two katali_ggml_q4k_dot_sums calls
 * (the row loop relies on that, and test_core asserts it). */
int katali_ggml_q4k_dot_sums2(const void *row1, const void *row2, const float *x,
                              uint64_t cols, const float *sums,
                              float *out1, float *out2) {
#if KATALI_X86
    if (!row1 || !row2 || !x || !sums || !out1 || !out2) return -1;
    if (!katali_ggml_simd_available()) return -1;
    if (cols == 0 || cols % 256 != 0) return -1;
    dot_q4_K_sums2_avx2((const uint8_t *)row1, (const uint8_t *)row2, x,
                        cols / 256, sums, out1, out2);
    return 0;
#else
    (void)row1; (void)row2; (void)x; (void)cols; (void)sums; (void)out1; (void)out2;
    return -1;
#endif
}

#if KATALI_X86

/* ==================================================================== *
 * Q6_K                                                                  *
 * ==================================================================== */
/* Per block: 2 halves of 128 outputs. Each half consumes 64 ql bytes,
 * 32 qh bytes and 8 int8 scales; two 16-value chunks per half, each chunk
 * using scale indices {c, c+2, c+4, c+6}. */
KATALI_AVX2 static float dot_q6_K_avx2(const uint8_t *s, const float *x, uint64_t nb) {
    const __m128i m0F = _mm_set1_epi8(0x0F);
    const __m128i m03 = _mm_set1_epi8(0x03);
    const __m128i m32 = _mm_set1_epi8(32);
    double acc = 0.0;
    for (uint64_t b = 0; b < nb; b++) {
        const uint8_t *blk = s + b * 210;
        const uint8_t *ql = blk;
        const uint8_t *qh = blk + 128;
        const int8_t  *sc = (const int8_t *)(blk + 192);
        float d = katali_f16_to_f32(simd_ld_u16(blk + 208));
        const float *xb = x + b * 256;
        __m256 A1 = _mm256_setzero_ps(), A2 = _mm256_setzero_ps();
        __m256 A3 = _mm256_setzero_ps(), A4 = _mm256_setzero_ps();

        for (int half = 0; half < 2; half++) {
            const uint8_t *qln = ql + half * 64;
            const uint8_t *qhn = qh + half * 32;
            const int8_t  *scn = sc + half * 8;
            const float *xn = xb + half * 128;

            for (int c = 0; c < 2; c++) {
                __m128i ql0  = _mm_loadu_si128((const __m128i *)(qln + c * 16));
                __m128i ql32 = _mm_loadu_si128((const __m128i *)(qln + 32 + c * 16));
                __m128i qh16 = _mm_loadu_si128((const __m128i *)(qhn + c * 16));

                __m128i lo0  = _mm_and_si128(ql0, m0F);
                __m128i lo32 = _mm_and_si128(ql32, m0F);
                __m128i hi0  = _mm_and_si128(_mm_srli_epi16(ql0, 4), m0F);
                __m128i hi32 = _mm_and_si128(_mm_srli_epi16(ql32, 4), m0F);

                __m128i q1 = _mm_or_si128(lo0,  _mm_slli_epi16(_mm_and_si128(qh16, m03), 4));
                __m128i q2 = _mm_or_si128(lo32, _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qh16, 2), m03), 4));
                __m128i q3 = _mm_or_si128(hi0,  _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qh16, 4), m03), 4));
                __m128i q4 = _mm_or_si128(hi32, _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qh16, 6), m03), 4));

                q1 = _mm_sub_epi8(q1, m32);
                q2 = _mm_sub_epi8(q2, m32);
                q3 = _mm_sub_epi8(q3, m32);
                q4 = _mm_sub_epi8(q4, m32);

                const float *xp = xn + c * 16;
                __m256 a1 = _mm256_setzero_ps(), a2 = _mm256_setzero_ps();
                __m256 a3 = _mm256_setzero_ps(), a4 = _mm256_setzero_ps();
                a1 = _mm256_fmadd_ps(i8x8_to_ps(q1), _mm256_loadu_ps(xp + 0), a1);
                a1 = _mm256_fmadd_ps(i8x8_to_ps(_mm_srli_si128(q1, 8)), _mm256_loadu_ps(xp + 8), a1);
                a2 = _mm256_fmadd_ps(i8x8_to_ps(q2), _mm256_loadu_ps(xp + 32), a2);
                a2 = _mm256_fmadd_ps(i8x8_to_ps(_mm_srli_si128(q2, 8)), _mm256_loadu_ps(xp + 40), a2);
                a3 = _mm256_fmadd_ps(i8x8_to_ps(q3), _mm256_loadu_ps(xp + 64), a3);
                a3 = _mm256_fmadd_ps(i8x8_to_ps(_mm_srli_si128(q3, 8)), _mm256_loadu_ps(xp + 72), a3);
                a4 = _mm256_fmadd_ps(i8x8_to_ps(q4), _mm256_loadu_ps(xp + 96), a4);
                a4 = _mm256_fmadd_ps(i8x8_to_ps(_mm_srli_si128(q4, 8)), _mm256_loadu_ps(xp + 104), a4);

                A1 = _mm256_fmadd_ps(_mm256_set1_ps(d * (float)scn[c + 0]), a1, A1);
                A2 = _mm256_fmadd_ps(_mm256_set1_ps(d * (float)scn[c + 2]), a2, A2);
                A3 = _mm256_fmadd_ps(_mm256_set1_ps(d * (float)scn[c + 4]), a3, A3);
                A4 = _mm256_fmadd_ps(_mm256_set1_ps(d * (float)scn[c + 6]), a4, A4);
            }
        }
        acc += (double)(hsum256_ps(A1) + hsum256_ps(A2) +
                        hsum256_ps(A3) + hsum256_ps(A4));
    }
    return (float)acc;
}
/* ==================================================================== *
 * Attention primitives                                                  *
 * ==================================================================== */
KATALI_AVX2 static int attn_scores_avx2(const float *q, const float *k,
                                        size_t kv_len, size_t head_dim,
                                        float scale, float *scores) {
    if (head_dim % 8 != 0) return -1;
    for (size_t t = 0; t < kv_len; t++) {
        const float *kt = k + t * head_dim;
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
        size_t d = 0;
        for (; d + 16 <= head_dim; d += 16) {
            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(q + d), _mm256_loadu_ps(kt + d), a0);
            a1 = _mm256_fmadd_ps(_mm256_loadu_ps(q + d + 8), _mm256_loadu_ps(kt + d + 8), a1);
        }
        float acc = hsum256_ps(_mm256_add_ps(a0, a1));
        for (; d < head_dim; d++) acc += q[d] * kt[d];
        scores[t] = acc * scale;
    }
    return 0;
}

KATALI_AVX2 static int attn_accum_avx2(const float *scores, size_t kv_len,
                                       const float *v, size_t head_dim, float *out) {
    if (head_dim % 8 != 0) return -1;
    for (size_t d = 0; d < head_dim; d++) out[d] = 0.0f;
    for (size_t t = 0; t < kv_len; t++) {
        const __m256 p = _mm256_set1_ps(scores[t]);
        const float *vt = v + t * head_dim;
        size_t d = 0;
        for (; d + 8 <= head_dim; d += 8) {
            __m256 o = _mm256_loadu_ps(out + d);
            o = _mm256_fmadd_ps(p, _mm256_loadu_ps(vt + d), o);
            _mm256_storeu_ps(out + d, o);
        }
        for (; d < head_dim; d++) out[d] += scores[t] * vt[d];
    }
    return 0;
}

/* Per-type row loops: one entry point per (matvec, worker); the kernel is
 * called directly and can be inlined, and no per-row dispatch happens. Each
 * worker gets a CONTIGUOUS row range so its weight stream is sequential (one
 * prefetch stream per worker instead of one per interlaced row). */
KATALI_AVX2 static void rows_f32(const float *w, uint64_t row_bytes,
                                 const float *x, uint64_t cols, float *y,
                                 uint64_t r0, uint64_t r1, volatile int *cancel) {
    const uint64_t stride = row_bytes / sizeof(float);
    for (uint64_t r = r0; r < r1; r++) {
        if (cancel && *cancel) return;
        y[r] = dot_f32_avx2(w + r * stride, x, cols);
    }
}

#define KATALI_ROW_LOOP(NAME, KERNEL)                                           \
    KATALI_AVX2 static void NAME(const uint8_t *w, uint64_t row_bytes,          \
                                 const float *x, uint64_t nb, float *y,         \
                                 uint64_t r0, uint64_t r1,                      \
                                 volatile int *cancel) {                        \
        for (uint64_t r = r0; r < r1; r++) {                                    \
            if (cancel && *cancel) return;                                      \
            y[r] = KERNEL(w + r * row_bytes, x, nb);                            \
        }                                                                       \
    }

KATALI_ROW_LOOP(rows_f16,  dot_f16_avx2)
KATALI_ROW_LOOP(rows_bf16, dot_bf16_avx2)
KATALI_ROW_LOOP(rows_q8_0, dot_q8_0_avx2)
KATALI_ROW_LOOP(rows_q4_0, dot_q4_0_avx2)
KATALI_ROW_LOOP(rows_q6_K, dot_q6_K_avx2)
#undef KATALI_ROW_LOOP

/* Q4_K gets a hand-written row loop so it can use the precomputed activation
 * sums (`sums`, see katali_ggml_q4k_sums) and, when they are available, process
 * two rows per iteration (dot_q4_K_sums2_avx2) so each worker keeps two
 * independent weight-load chains in flight. An odd row count is finished by the
 * single-row loop below, which is also the only path when `sums` is NULL.
 *
 * Runtime A/B switch: KATALI_GGUF_NO_Q4K_DUALROW=1 forces the single-row loop,
 * KATALI_GGUF_NO_Q4K_SUMCACHE=1 disables the sums (and therefore dual-row, which
 * needs them). With sums == NULL the per-row kernel is exactly dot_q4_K_avx2,
 * i.e. the pre-hoist behaviour.
 *
 * A software prefetch of a later row (leads of 1, 2, 4 and 8 rows, first two
 * cache lines) was tried here and measured neutral-to-slower at every lead
 * (2.2-9.4% slower at 12 threads on a 37 MB DRAM-resident Q4_K matvec), so it
 * was removed: the rows are contiguous, so the hardware prefetcher already
 * streams them and the extra instructions only add pressure. */
KATALI_AVX2 static void rows_q4_K(const uint8_t *w, uint64_t row_bytes,
                                  const float *x, uint64_t nb, float *y,
                                  uint64_t r0, uint64_t r1, volatile int *cancel,
                                  const float *sums) {
    uint64_t r = r0;
    if (sums && q4k_dualrow_enabled()) {
        for (; r1 - r >= 2; r += 2) {
            if (cancel && *cancel) return;
            const uint8_t *row1 = w + r * row_bytes;
            dot_q4_K_sums2_avx2(row1, row1 + row_bytes, x, nb, sums,
                                &y[r], &y[r + 1]);
        }
    }
    for (; r < r1; r++) {
        if (cancel && *cancel) return;
        y[r] = sums ? dot_q4_K_sums_avx2(w + r * row_bytes, x, nb, sums)
                    : dot_q4_K_avx2(w + r * row_bytes, x, nb);
    }
}

/* ==================================================================== *
 * Activation: silu(gate) * up                                           *
 * ==================================================================== */
/* exp(x) via 2^round(x*log2e) with a degree-6 polynomial for 2^r on
 * r in [-0.5, 0.5]. Relative error ~1e-7, far below the quantisation noise of
 * the weights, and the same formula everywhere so it stays deterministic. */
KATALI_AVX2 static __m256 exp256_ps(__m256 x) {
    x = _mm256_min_ps(x, _mm256_set1_ps(88.0f));
    x = _mm256_max_ps(x, _mm256_set1_ps(-88.0f));
    const __m256 log2e = _mm256_set1_ps(1.44269504088896341f);
    __m256 t = _mm256_mul_ps(x, log2e);
    __m256 n = _mm256_round_ps(t, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 r = _mm256_sub_ps(t, n);
    __m256 p = _mm256_set1_ps(0.00015403530393381608f);
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.0013333558146428443f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.009618129107628477f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.05550410866482158f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.2402265069591007f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.6931471805599453f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f));
    __m256i ni = _mm256_cvtps_epi32(n);
    ni = _mm256_slli_epi32(_mm256_add_epi32(ni, _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(p, _mm256_castsi256_ps(ni));
}

KATALI_AVX2 static void silu_mul_avx2(float *gate, const float *up, size_t n) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 g = _mm256_loadu_ps(gate + i);
        __m256 e = exp256_ps(_mm256_sub_ps(_mm256_setzero_ps(), g));
        __m256 s = _mm256_div_ps(g, _mm256_add_ps(_mm256_set1_ps(1.0f), e));
        _mm256_storeu_ps(gate + i, _mm256_mul_ps(s, _mm256_loadu_ps(up + i)));
    }
    for (; i < n; i++)
        gate[i] = (gate[i] / (1.0f + expf(-gate[i]))) * up[i];
}

/* ==================================================================== *
 * Softmax: max-reduce / exp(x-m)+sum / normalize                        *
 * ==================================================================== */
/* Same three passes and the same max-subtraction guarantee as the scalar
 * version, vectorized:
 *   1. max-reduce: max is exact and order-independent, so this pass is
 *      bit-identical to the scalar one.
 *   2. exp(x - m) via exp256_ps (the same polynomial silu uses, ~1e-7 relative)
 *      with the sum accumulated in 8 lanes and folded to double. This is the
 *      only pass whose rounding differs from the scalar reference: libm expf
 *      and the left-to-right double summation are not reproduced exactly.
 *   3. normalize: a single multiply by the same `inv`, bit-identical.
 * The max subtraction is preserved exactly, so values stay in [0,1] and cannot
 * overflow - the numerical-stability property of the scalar version. */
KATALI_AVX2 static void softmax_avx2(float *x, size_t n) {
    size_t i = 0;
    float m = x[0];

    /* pass 1: max */
    __m256 mv = _mm256_set1_ps(x[0]);
    for (; i + 8 <= n; i += 8)
        mv = _mm256_max_ps(mv, _mm256_loadu_ps(x + i));
    {
        float lanes[8];
        _mm256_storeu_ps(lanes, mv);
        for (int k = 0; k < 8; k++) if (lanes[k] > m) m = lanes[k];
    }
    for (; i < n; i++) if (x[i] > m) m = x[i];

    /* pass 2: exp(x - m) in place, sum in 8 lanes */
    const __m256 vm = _mm256_set1_ps(m);
    __m256 sv = _mm256_setzero_ps();
    i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 e = exp256_ps(_mm256_sub_ps(_mm256_loadu_ps(x + i), vm));
        _mm256_storeu_ps(x + i, e);
        sv = _mm256_add_ps(sv, e);
    }
    double sum = 0.0;
    {
        float lanes[8];
        _mm256_storeu_ps(lanes, sv);
        for (int k = 0; k < 8; k++) sum += (double)lanes[k];
    }
    for (; i < n; i++) { x[i] = expf(x[i] - m); sum += (double)x[i]; }
    if (sum <= 0.0) return;

    /* pass 3: normalize */
    const float inv = (float)(1.0 / sum);
    const __m256 vi = _mm256_set1_ps(inv);
    i = 0;
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(x + i, _mm256_mul_ps(_mm256_loadu_ps(x + i), vi));
    for (; i < n; i++) x[i] *= inv;
}

#endif /* KATALI_X86 */

int katali_ggml_softmax_avx2(float *x, size_t n) {
#if KATALI_X86
    static int enabled = -1;
    if (enabled < 0) {
        const char *e = getenv("KATALI_GGUF_NO_SIMD_SOFTMAX");
        enabled = (e && *e && e[0] != '0') ? 0 : 1;
    }
    if (!enabled || !katali_ggml_simd_available() || n < 8) return -1;
    softmax_avx2(x, n);
    return 0;
#else
    (void)x; (void)n;
    return -1;
#endif
}

int katali_ggml_silu_mul_avx2(float *gate, const float *up, size_t n) {
#if KATALI_X86
    static int enabled = -1;
    if (enabled < 0) {
        const char *e = getenv("KATALI_GGUF_NO_SIMD_SILU");
        enabled = (e && *e && e[0] != '0') ? 0 : 1;
    }
    if (!enabled || !katali_ggml_simd_available()) return -1;
    silu_mul_avx2(gate, up, n);
    return 0;
#else
    (void)gate; (void)up; (void)n;
    return -1;
#endif
}

int katali_ggml_matvec_rows(uint32_t type, const void *w, uint64_t rows,
                            uint64_t cols, uint64_t row_bytes, const float *x,
                            float *y, uint64_t row_begin, uint64_t row_end,
                            volatile int *cancel, const float *q4k_sums) {
#if KATALI_X86
    if (!katali_ggml_simd_available()) return -1;
    if (row_begin >= row_end) return 0;
    if (row_end > rows) return -1;
    const uint8_t *ww = (const uint8_t *)w;
    switch (type) {
        case KGGML_F32:
            rows_f32((const float *)(const void *)w, row_bytes, x, cols, y,
                     row_begin, row_end, cancel);
            return 0;
        case KGGML_F16:  rows_f16(ww, row_bytes, x, cols, y, row_begin, row_end, cancel); return 0;
        case KGGML_BF16: rows_bf16(ww, row_bytes, x, cols, y, row_begin, row_end, cancel); return 0;
        case KGGML_Q8_0: rows_q8_0(ww, row_bytes, x, cols / 32, y, row_begin, row_end, cancel); return 0;
        case KGGML_Q4_0: rows_q4_0(ww, row_bytes, x, cols / 32, y, row_begin, row_end, cancel); return 0;
        case KGGML_Q4_K: rows_q4_K(ww, row_bytes, x, cols / 256, y, row_begin, row_end,
                                   cancel, q4k_sums); return 0;
        case KGGML_Q6_K: rows_q6_K(ww, row_bytes, x, cols / 256, y, row_begin, row_end, cancel); return 0;
        default: return -1;
    }
#else
    (void)type; (void)w; (void)rows; (void)cols; (void)row_bytes; (void)x;
    (void)y; (void)row_begin; (void)row_end; (void)cancel; (void)q4k_sums;
    return -1;
#endif
}

/* Public wrappers that dispatch to the accelerated primitives when available. */
int katali_ggml_attn_scores_avx2(const float *q, const float *k,
                                 size_t kv_len, size_t head_dim,
                                 float scale, float *scores) {
#if KATALI_X86
    if (!katali_ggml_simd_available()) return -1;
    return attn_scores_avx2(q, k, kv_len, head_dim, scale, scores);
#else
    (void)q; (void)k; (void)kv_len; (void)head_dim; (void)scale; (void)scores;
    return -1;
#endif
}

int katali_ggml_attn_accum_avx2(const float *scores, size_t kv_len,
                                const float *v, size_t head_dim, float *out) {
#if KATALI_X86
    if (!katali_ggml_simd_available()) return -1;
    return attn_accum_avx2(scores, kv_len, v, head_dim, out);
#else
    (void)scores; (void)kv_len; (void)v; (void)head_dim; (void)out;
    return -1;
#endif
}

/* ==================================================================== *
 * Row-materializing Q4_K decoder (prefill row decode)                    *
 * ==================================================================== */
/* Decode nb Q4_K blocks (144 bytes -> 256 floats each) into `y`.
 *
 * Same block layout and the same 2x32-nibble grouping as dot_q4_K_avx2 and as
 * the scalar reference dequant_q4_K, but writing floats out instead of
 * accumulating a dot. Each element's arithmetic is the scalar reference's
 * sequence with the same single-rounding steps and no fused multiply-add -
 * convert the nibble, multiply by d*scale, subtract dmin*min - so the output is
 * bit-identical to dequant_q4_K (asserted in test_core, not assumed). */
KATALI_AVX2 static void dequant_q4_K_avx2(const uint8_t *s, uint64_t nb, float *y) {
    const __m256i mask0F = _mm256_set1_epi32(0x0F);
    for (uint64_t b = 0; b < nb; b++) {
        const uint8_t *blk = s + b * 144;
        const float d  = katali_f16_to_f32(simd_ld_u16(blk));
        const float mn = katali_f16_to_f32(simd_ld_u16(blk + 2));
        const uint8_t *sc = blk + 4;
        const uint8_t *q = blk + 16;
        float *out = y + b * 256;
        int is = 0;
        for (int n = 0; n < 256; n += 64) {
            uint8_t s0, m0, s1, m1;
            simd_scale_min_k4(is + 0, sc, &s0, &m0);
            simd_scale_min_k4(is + 1, sc, &s1, &m1);
            const __m256 d1v = _mm256_set1_ps(d * (float)s0);
            const __m256 m1v = _mm256_set1_ps(mn * (float)m0);
            const __m256 d2v = _mm256_set1_ps(d * (float)s1);
            const __m256 m2v = _mm256_set1_ps(mn * (float)m1);
            for (int l = 0; l < 32; l += 8) {
                __m256i bytes = _mm256_cvtepu8_epi32(
                    _mm_loadl_epi64((const __m128i *)(q + l)));
                __m256 lo = _mm256_cvtepi32_ps(_mm256_and_si256(bytes, mask0F));
                __m256 hi = _mm256_cvtepi32_ps(_mm256_srli_epi32(bytes, 4));
                _mm256_storeu_ps(out + n + l,
                                 _mm256_sub_ps(_mm256_mul_ps(d1v, lo), m1v));
                _mm256_storeu_ps(out + n + 32 + l,
                                 _mm256_sub_ps(_mm256_mul_ps(d2v, hi), m2v));
            }
            q += 32;
            is += 2;
        }
    }
}

/* Decode nb Q6_K blocks (210 bytes -> 256 floats each) into `y`.
 *
 * Layout follows the reference dequant_q6_K exactly: ql[128], qh[32] and 16 int8
 * scales at 192, with d as f16 at 208; within each 128-value half, output l gets
 * q1 from ql[l]'s low nibble and qh[l]'s bits 0-1, q2 from ql[l+32] and bits 2-3,
 * q3 from ql[l]'s high nibble and bits 4-5, q4 from ql[l+32]'s high nibble and
 * bits 6-7, each minus 32, with the scale index l/16.
 *
 * The reference evaluates `d * (float)sc * (float)q` left to right, so this
 * computes the same product as two separate multiplies in the same order (scale
 * times d broadcast first, then multiplied by the widened integer), with no fused
 * multiply-add: the output is bit-identical to the reference (test_core asserts
 * it), not merely close. */
KATALI_AVX2 static void dequant_q6_K_avx2(const uint8_t *s, uint64_t nb, float *y) {
    const __m256i maskF = _mm256_set1_epi32(0x0F);
    const __m256i mask3 = _mm256_set1_epi32(0x03);
    const __m256i off32 = _mm256_set1_epi32(32);
    for (uint64_t b = 0; b < nb; b++) {
        const uint8_t *blk = s + b * 210;
        const uint8_t *ql = blk;
        const uint8_t *qh = blk + 128;
        const int8_t *sc = (const int8_t *)(blk + 192);
        const float d = katali_f16_to_f32(simd_ld_u16(blk + 208));
        float *out = y + b * 256;
        for (int n = 0; n < 256; n += 128) {
            for (int l = 0; l < 32; l += 8) {
                const int is = l / 16;          /* constant inside an 8-wide chunk */
                const float t1 = d * (float)sc[is + 0];
                const float t2 = d * (float)sc[is + 2];
                const float t3 = d * (float)sc[is + 4];
                const float t4 = d * (float)sc[is + 6];
                __m256i lo0 = _mm256_cvtepu8_epi32(
                    _mm_loadl_epi64((const __m128i *)(ql + l)));
                __m256i lo32 = _mm256_cvtepu8_epi32(
                    _mm_loadl_epi64((const __m128i *)(ql + 32 + l)));
                __m256i hi = _mm256_cvtepu8_epi32(
                    _mm_loadl_epi64((const __m128i *)(qh + l)));
                __m256i q1 = _mm256_sub_epi32(
                    _mm256_or_si256(_mm256_and_si256(lo0, maskF),
                                    _mm256_slli_epi32(_mm256_and_si256(hi, mask3), 4)),
                    off32);
                __m256i q2 = _mm256_sub_epi32(
                    _mm256_or_si256(_mm256_and_si256(lo32, maskF),
                                    _mm256_slli_epi32(
                                        _mm256_and_si256(_mm256_srli_epi32(hi, 2), mask3), 4)),
                    off32);
                __m256i q3 = _mm256_sub_epi32(
                    _mm256_or_si256(_mm256_srli_epi32(lo0, 4),
                                    _mm256_slli_epi32(
                                        _mm256_and_si256(_mm256_srli_epi32(hi, 4), mask3), 4)),
                    off32);
                __m256i q4 = _mm256_sub_epi32(
                    _mm256_or_si256(_mm256_srli_epi32(lo32, 4),
                                    _mm256_slli_epi32(
                                        _mm256_and_si256(_mm256_srli_epi32(hi, 6), mask3), 4)),
                    off32);
                _mm256_storeu_ps(out + n + l,
                                 _mm256_mul_ps(_mm256_set1_ps(t1), _mm256_cvtepi32_ps(q1)));
                _mm256_storeu_ps(out + n + 32 + l,
                                 _mm256_mul_ps(_mm256_set1_ps(t2), _mm256_cvtepi32_ps(q2)));
                _mm256_storeu_ps(out + n + 64 + l,
                                 _mm256_mul_ps(_mm256_set1_ps(t3), _mm256_cvtepi32_ps(q3)));
                _mm256_storeu_ps(out + n + 96 + l,
                                 _mm256_mul_ps(_mm256_set1_ps(t4), _mm256_cvtepi32_ps(q4)));
            }
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

int katali_ggml_dequant_rows_avx2(uint32_t type, const void *src, uint64_t n,
                                  float *out) {
#if KATALI_X86
    if (!katali_ggml_simd_available()) return -1;
    switch (type) {
        case KGGML_Q4_K:
            if (n == 0 || n % 256 != 0) return -1;
            dequant_q4_K_avx2((const uint8_t *)src, n / 256, out);
            return 0;
        case KGGML_Q6_K:
            if (n == 0 || n % 256 != 0) return -1;
            dequant_q6_K_avx2((const uint8_t *)src, n / 256, out);
            return 0;
        default:
            return -1;
    }
#else
    (void)type; (void)src; (void)n; (void)out;
    return -1;
#endif
}

/* ==================================================================== *
 * Dispatcher                                                            *
 * ==================================================================== */
int katali_ggml_vec_dot_avx2(uint32_t type, const void *src,
                             const float *x, uint64_t n, float *out) {
#if KATALI_X86
    if (!katali_ggml_simd_available()) return -1;
    const uint8_t *s = (const uint8_t *)src;
    switch (type) {
        case KGGML_F32:  *out = dot_f32_avx2((const float *)(const void *)src, x, n); return 0;
        case KGGML_F16:  *out = dot_f16_avx2(s, x, n); return 0;
        case KGGML_BF16: *out = dot_bf16_avx2(s, x, n); return 0;
        case KGGML_Q8_0:
            if (n % 32 != 0) return -1;
            *out = dot_q8_0_avx2(s, x, n / 32);
            return 0;
        case KGGML_Q4_0:
            if (n % 32 != 0) return -1;
            *out = dot_q4_0_avx2(s, x, n / 32);
            return 0;
        case KGGML_Q4_K:
            if (n % 256 != 0) return -1;
            *out = dot_q4_K_avx2(s, x, n / 256);
            return 0;
        case KGGML_Q6_K:
            if (n % 256 != 0) return -1;
            *out = dot_q6_K_avx2(s, x, n / 256);
            return 0;
        default:
            return -1;
    }
#else
    (void)type; (void)src; (void)x; (void)n; (void)out;
    return -1;
#endif
}




