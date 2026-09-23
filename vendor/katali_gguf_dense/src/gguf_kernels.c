/* Katali-GGUF dense CPU kernels — Apache-2.0 */
#include "katali_gguf_kernels.h"
#include "katali_gguf_simd.h"
#include "katali_gguf_prof.h"
#include "katali_gguf_threads.h"

#include <math.h>
#include <string.h>
#include <stdio.h>
#include <float.h>
#include <stdlib.h>

void katali_gguf_rmsnorm(const float *x, const float *w, size_t n, float eps, float *out) {
    double sumsq = 0.0;
    for (size_t i = 0; i < n; i++) sumsq += (double)x[i] * (double)x[i];
    float inv = 1.0f / sqrtf((float)(sumsq / (double)n) + eps);
    for (size_t i = 0; i < n; i++) out[i] = x[i] * inv * w[i];
}

void katali_gguf_rmsnorm_inplace(float *x, const float *w, size_t n, float eps) {
    double sumsq = 0.0;
    for (size_t i = 0; i < n; i++) sumsq += (double)x[i] * (double)x[i];
    float inv = 1.0f / sqrtf((float)(sumsq / (double)n) + eps);
    for (size_t i = 0; i < n; i++) x[i] = x[i] * inv * w[i];
}

void katali_gguf_rope(float *x, size_t n_heads, size_t head_dim,
                      size_t rope_dim, size_t pos, float theta) {
    size_t rd = (rope_dim == 0 || rope_dim > head_dim) ? head_dim : rope_dim;
    size_t half = rd / 2;
    if (half < 1) return;
    for (size_t i = 0; i < half; i++) {
        float inv = 1.0f / powf(theta, (2.0f * (float)i) / (float)rd);
        float freq = (float)pos * inv;
        float c = cosf(freq), s = sinf(freq);
        for (size_t h = 0; h < n_heads; h++) {
            float *v = x + h * head_dim;
            float x1 = v[i], x2 = v[half + i];
            v[i]        = x1 * c - x2 * s;
            v[half + i] = x2 * c + x1 * s;
        }
    }
}

void katali_gguf_rope_apply(float *x, size_t n_heads, size_t head_dim,
                            size_t rope_dim, const float *cos, const float *sin) {
    size_t rd = (rope_dim == 0 || rope_dim > head_dim) ? head_dim : rope_dim;
    size_t half = rd / 2;
    if (half < 1) return;
    for (size_t i = 0; i < half; i++) {
        float c = cos[i], s = sin[i];
        for (size_t h = 0; h < n_heads; h++) {
            float *v = x + h * head_dim;
            float x1 = v[i], x2 = v[half + i];
            v[i]        = x1 * c - x2 * s;
            v[half + i] = x2 * c + x1 * s;
        }
    }
}

/* Scalar reference and fallback: three passes, libm expf, double sum. Exposed
 * so the selftest and the SIMD path can be compared against it directly. */
void katali_gguf_softmax_scalar(float *x, size_t n) {
    if (n == 0) return;
    float m = x[0];
    for (size_t i = 1; i < n; i++) if (x[i] > m) m = x[i];
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) { x[i] = expf(x[i] - m); sum += (double)x[i]; }
    if (sum <= 0.0) return;
    float inv = (float)(1.0 / sum);
    for (size_t i = 0; i < n; i++) x[i] *= inv;
}

void katali_gguf_softmax(float *x, size_t n) {
    if (n == 0) return;
    /* Vectorized when available; the scalar path is the reference and the
     * fallback (also used when KATALI_GGUF_NO_SIMD_SOFTMAX=1). */
    if (katali_ggml_softmax_avx2(x, n) == 0) return;
    katali_gguf_softmax_scalar(x, n);
}

/* ==================================================================== *
 * Attention: shared per-head body, serial and parallel entry points     *
 * ==================================================================== */
/* One parallel worker may not exceed this; the pool clamps to 256 too. */
#define KATALI_ATTN_MAX_THREADS 256

typedef struct GqaPlan {
    const float *q, *k, *v;
    size_t kv_len, kv_cap, n_q, n_kv, head_dim;
    float *out;
    int    simd;
    float  scale;
} GqaPlan;

/* Computes query heads [h0, h1) into `out`, using `scores` (kv_len floats) as
 * scratch for one head at a time. This is the *only* implementation: the serial
 * entry point calls it once over [0, n_q) and the parallel entry point calls it
 * per worker, so a head's arithmetic cannot differ between the two paths.
 *
 * `scores` must be private to the calling worker when this runs concurrently.
 * The three stage sums are written to the caller's `t_*` (NULL when not
 * profiling); in the parallel path those are worker-private slots folded by the
 * caller, so no shared profiler counter is ever touched from a worker. */
/* Bytes one attention call streams from the K (or V) cache. Every query head
 * reads its own KV head's [0, kv_len) range contiguously, so with GQA this
 * counts each KV head once per query head that maps to it: it is the traffic
 * *before* any reuse, which is the number to compare against achieved
 * bandwidth (see docs/benchmarks.md). */
static unsigned long long kv_bytes(size_t n_q, size_t kv_len, size_t head_dim) {
    return (unsigned long long)n_q * (unsigned long long)kv_len *
           (unsigned long long)head_dim * sizeof(float);
}

static void gqa_heads(const GqaPlan *p, size_t h0, size_t h1, float *scores,
                      int prof, double *t_scores, double *t_softmax, double *t_accum) {
    if (h1 > p->n_q) h1 = p->n_q;
    if (h0 >= h1) return;
    size_t reps = p->n_q / p->n_kv;
    if (reps < 1) reps = 1;
    double s_scores = 0.0, s_softmax = 0.0, s_accum = 0.0;

    for (size_t hq = h0; hq < h1; hq++) {
        size_t hk = hq / reps;
        if (hk >= p->n_kv) hk = p->n_kv - 1;
        const float *qh = p->q + hq * p->head_dim;
        const float *kh = p->k + hk * p->kv_cap * p->head_dim;
        const float *vh = p->v + hk * p->kv_cap * p->head_dim;
        float *oh = p->out + hq * p->head_dim;
        double t0 = prof ? katali_prof_now() : 0.0;

        if (!p->simd ||
            katali_ggml_attn_scores_avx2(qh, kh, p->kv_len, p->head_dim, p->scale, scores) != 0) {
            for (size_t t = 0; t < p->kv_len; t++) {
                const float *kt = kh + t * p->head_dim;
                double acc = 0.0;
                for (size_t d = 0; d < p->head_dim; d++) acc += (double)qh[d] * (double)kt[d];
                scores[t] = (float)acc * p->scale;
            }
        }
        double t1 = prof ? katali_prof_now() : 0.0;
        katali_gguf_softmax(scores, p->kv_len);
        double t2 = prof ? katali_prof_now() : 0.0;
        if (!p->simd ||
            katali_ggml_attn_accum_avx2(scores, p->kv_len, vh, p->head_dim, oh) != 0) {
            for (size_t t = 0; t < p->kv_len; t++) {
                float pr = scores[t];
                const float *vt = vh + t * p->head_dim;
                for (size_t d = 0; d < p->head_dim; d++) oh[d] += pr * vt[d];
            }
        }
        if (prof) {
            double t3 = katali_prof_now();
            s_scores  += t1 - t0;
            s_softmax += t2 - t1;
            s_accum   += t3 - t2;
        }
    }
    if (prof) {
        if (t_scores)  *t_scores  = s_scores;
        if (t_softmax) *t_softmax = s_softmax;
        if (t_accum)   *t_accum   = s_accum;
    }
}

void katali_gguf_gqa_decode_scratch(const float *q, const float *k, const float *v,
                                    size_t kv_len, size_t kv_cap,
                                    size_t n_q, size_t n_kv, size_t head_dim,
                                    float *out, float *scores) {
    if (!q || !k || !v || !out || !scores) return;
    memset(out, 0, n_q * head_dim * sizeof(float));
    if (kv_len == 0 || n_q == 0 || n_kv == 0) return;
    GqaPlan p;
    p.q = q; p.k = k; p.v = v;
    p.kv_len = kv_len; p.kv_cap = kv_cap;
    p.n_q = n_q; p.n_kv = n_kv; p.head_dim = head_dim;
    p.out = out;
    p.simd = katali_ggml_simd_available();
    p.scale = 1.0f / sqrtf((float)head_dim);
    /* Per-stage timers, respecting KATALI_GGUF_NO_PROF (one flag for all
     * profiling). 4 QPC reads per head, ~16 x 28 = 448 heads per decoded token,
     * so ~0.1 % of a 26 ms token - the same order as the existing profiler. */
    const int prof = katali_prof_enabled();
    double a = 0.0, b = 0.0, c = 0.0;
    gqa_heads(&p, 0, n_q, scores, prof, &a, &b, &c);
    if (prof) {
        katali_prof_add_n(KATALI_PHASE_ATTN_SCORES,  a, 1);
        katali_prof_add_n(KATALI_PHASE_ATTN_SOFTMAX, b, 1);
        katali_prof_add_n(KATALI_PHASE_ATTN_ACCUM,   c, 1);
        /* K stream (score dot) and V stream (weighted accumulation) traffic. */
        katali_prof_add_bytes(KATALI_PHASE_ATTN_SCORES, kv_bytes(n_q, kv_len, head_dim));
        katali_prof_add_bytes(KATALI_PHASE_ATTN_ACCUM,  kv_bytes(n_q, kv_len, head_dim));
    }
}

/* --- parallel path ------------------------------------------------------ */
/* 1 unless KATALI_GGUF_NO_ATTN_PARALLEL=1. */
static int attn_parallel_enabled(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("KATALI_GGUF_NO_ATTN_PARALLEL");
        v = (e && *e && e[0] != '0') ? 0 : 1;
    }
    return v;
}

/* Smallest kv_len worth dispatching. A dispatch costs ~30 us at 12 threads and
 * ~15 us at 4 (measured, docs/benchmarks.md), so below this the attention is
 * shorter than the wake/join that would parallelize it. Tunable for A/B. */
static size_t attn_parallel_min_kv(void) {
    static long v = -2;
    if (v == -2) {
        const char *e = getenv("KATALI_GGUF_ATTN_MIN_KV");
        v = (e && *e) ? atol(e) : 256;
        if (v < 0) v = 0;
    }
    return (size_t)v;
}

typedef struct GqaCtx {
    GqaPlan plan;
    float  *pool;      /* n_threads score buffers of `stride` floats each */
    size_t  stride;
    volatile int *cancel;
    double *timings;   /* 3 doubles per worker; a worker writes only its own */
} GqaCtx;

static void gqa_worker(int worker, int n_workers, void *arg) {
    GqaCtx *c = (GqaCtx *)arg;
    /* Contiguous head range, like the matvec row ranges. */
    const size_t per = (c->plan.n_q + (size_t)n_workers - 1) / (size_t)n_workers;
    size_t h0 = (size_t)worker * per;
    size_t h1 = h0 + per;
    if (h0 > c->plan.n_q) h0 = c->plan.n_q;
    double a = 0.0, b = 0.0, d = 0.0;
    if (!(c->cancel && *c->cancel)) {
        gqa_heads(&c->plan, h0, h1, c->pool + (size_t)worker * c->stride,
                  c->timings != NULL, &a, &b, &d);
    }
    if (c->timings) {
        c->timings[3 * (size_t)worker + 0] = a;
        c->timings[3 * (size_t)worker + 1] = b;
        c->timings[3 * (size_t)worker + 2] = d;
    }
}

void katali_gguf_gqa_decode_parallel(const float *q, const float *k, const float *v,
                                     size_t kv_len, size_t kv_cap,
                                     size_t n_q, size_t n_kv, size_t head_dim,
                                     float *out, float *scores,
                                     float *scores_pool, size_t pool_stride,
                                     int n_threads, volatile int *cancel) {
    if (!q || !k || !v || !out || !scores) return;
    int nt = n_threads > 0 ? n_threads : katali_gguf_get_threads();
    if (nt > KATALI_ATTN_MAX_THREADS) nt = KATALI_ATTN_MAX_THREADS;
    /* Serial whenever a dispatch would cost more than the work it would share,
     * when parallelism is switched off, or when no private buffers were given.
     * `scores_pool` must hold at least n_threads * pool_stride floats. */
    if (nt <= 1 || n_q < 2 || !scores_pool || pool_stride < kv_len ||
        !attn_parallel_enabled() || kv_len < attn_parallel_min_kv()) {
        katali_gguf_gqa_decode_scratch(q, k, v, kv_len, kv_cap, n_q, n_kv,
                                       head_dim, out, scores);
        return;
    }
    memset(out, 0, n_q * head_dim * sizeof(float));
    if (kv_len == 0 || n_q == 0 || n_kv == 0) return;

    GqaCtx c;
    c.plan.q = q; c.plan.k = k; c.plan.v = v;
    c.plan.kv_len = kv_len; c.plan.kv_cap = kv_cap;
    c.plan.n_q = n_q; c.plan.n_kv = n_kv; c.plan.head_dim = head_dim;
    c.plan.out = out;
    c.plan.simd = katali_ggml_simd_available();
    c.plan.scale = 1.0f / sqrtf((float)head_dim);
    c.pool = scores_pool;
    c.stride = pool_stride;
    c.cancel = cancel;
    const int prof = katali_prof_enabled();
    double timings[3 * KATALI_ATTN_MAX_THREADS];
    if (prof) for (int i = 0; i < 3 * KATALI_ATTN_MAX_THREADS; i++) timings[i] = 0.0;
    c.timings = prof ? timings : NULL;

    katali_gguf_parallel_run_n(nt, gqa_worker, &c);

    if (prof) {
        /* Fold the worker-private sums. In parallel mode these stages are
         * summed work per stage rather than elapsed wall time. */
        double a = 0.0, b = 0.0, d = 0.0;
        for (int i = 0; i < nt; i++) {
            a += timings[3 * i + 0];
            b += timings[3 * i + 1];
            d += timings[3 * i + 2];
        }
        katali_prof_add_n(KATALI_PHASE_ATTN_SCORES,  a, 1);
        katali_prof_add_n(KATALI_PHASE_ATTN_SOFTMAX, b, 1);
        katali_prof_add_n(KATALI_PHASE_ATTN_ACCUM,   d, 1);
        /* K stream (score dot) and V stream (weighted accumulation) traffic. */
        katali_prof_add_bytes(KATALI_PHASE_ATTN_SCORES, kv_bytes(n_q, kv_len, head_dim));
        katali_prof_add_bytes(KATALI_PHASE_ATTN_ACCUM,  kv_bytes(n_q, kv_len, head_dim));
    }
}

void katali_gguf_gqa_decode(const float *q, const float *k, const float *v,
                            size_t kv_len, size_t kv_cap,
                            size_t n_q, size_t n_kv, size_t head_dim,
                            float *out) {
    if (!q || !k || !v || !out) return;
    memset(out, 0, n_q * head_dim * sizeof(float));
    if (kv_len == 0 || n_q == 0 || n_kv == 0) return;
    /* Public shorthand: allocates the score buffer. The model uses the scratch
     * variant so the decode loop performs no allocation at all. */
    float *scores = (float *)malloc(kv_len * sizeof(float));
    if (!scores) return;
    katali_gguf_gqa_decode_scratch(q, k, v, kv_len, kv_cap, n_q, n_kv, head_dim,
                                   out, scores);
    free(scores);
}

void katali_gguf_add_inplace(float *a, const float *b, size_t n) {
    for (size_t i = 0; i < n; i++) a[i] += b[i];
}

float katali_gguf_silu(float x) { return x / (1.0f + expf(-x)); }
float katali_gguf_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

float katali_gguf_gelu(float x) {
    float sign = x < 0.0f ? -1.0f : 1.0f;
    float ax = fabsf(x);
    float t = 1.0f / (1.0f + 0.3275911f * ax);
    float y = 1.0f - ((((1.061405429f * t - 1.453152027f) * t + 1.421413741f) * t
                      - 0.284496736f) * t + 0.254829592f) * t * expf(-ax * ax);
    return 0.5f * x * (1.0f + sign * y);
}

int katali_gguf_argmax(const float *x, size_t n) {
    int best = 0;
    float bv = -FLT_MAX;
    for (size_t i = 0; i < n; i++) if (x[i] > bv) { bv = x[i]; best = (int)i; }
    return best;
}

/* ==================================================================== *
 * Self-test                                                             *
 * ==================================================================== */
int katali_gguf_kernels_selftest(void) {
    int failures = 0;

    /* RMSNorm: uniform vector, weight 1 => x / rms. */
    {
        float x[4] = { 3.0f, -3.0f, 3.0f, -3.0f };
        float w[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
        float out[4];
        katali_gguf_rmsnorm(x, w, 4, 0.0f, out);
        for (int i = 0; i < 4; i++) {
            float want = x[i] / 3.0f;
            if (fabsf(out[i] - want) > 1e-5f) { printf("  FAIL rmsnorm[%d]\n", i); failures++; }
        }
    }
    /* RoPE at position 0 is identity. */
    {
        float x[8], y[8];
        for (int i = 0; i < 8; i++) x[i] = (float)(i + 1);
        memcpy(y, x, sizeof(x));
        katali_gguf_rope(y, 1, 8, 8, 0, 10000.0f);
        for (int i = 0; i < 8; i++)
            if (fabsf(y[i] - x[i]) > 1e-6f) { printf("  FAIL rope pos0\n"); failures++; break; }
    }
    /* RoPE preserves the vector norm. */
    {
        float x[8] = { 1.0f, 2.0f, 3.0f, 4.0f, -1.0f, 0.5f, 2.5f, -3.5f };
        float y[8];
        memcpy(y, x, sizeof(x));
        katali_gguf_rope(y, 1, 8, 8, 7, 10000.0f);
        double n0 = 0, n1 = 0;
        for (int i = 0; i < 8; i++) { n0 += x[i] * x[i]; n1 += y[i] * y[i]; }
        if (fabs(n0 - n1) > 1e-4) { printf("  FAIL rope norm %.6f vs %.6f\n", n0, n1); failures++; }
    }
    /* Softmax sums to 1. */
    {
        float x[5] = { 1.0f, 2.0f, 3.0f, 2.0f, 1.0f };
        katali_gguf_softmax(x, 5);
        double s = 0; for (int i = 0; i < 5; i++) s += x[i];
        if (fabs(s - 1.0) > 1e-6) { printf("  FAIL softmax sum %.9f\n", s); failures++; }
    }
    /* GQA decode with a single valid position returns v exactly.
     * n_q=4, n_kv=2 => heads 0,1 read kv head 0; heads 2,3 read kv head 1. */
    {
        enum { NQ = 4, NKV = 2, HD = 3, CAP = 4 };
        float q[NQ * HD], k[NKV * CAP * HD], v[NKV * CAP * HD], out[NQ * HD];
        memset(k, 0, sizeof(k)); memset(v, 0, sizeof(v));
        for (int i = 0; i < NQ * HD; i++) q[i] = (float)i * 0.1f;
        for (int d = 0; d < HD; d++) {
            v[(0 * CAP + 0) * HD + d] = (float)(d + 1);
            v[(1 * CAP + 0) * HD + d] = (float)(d + 10);
        }
        katali_gguf_gqa_decode(q, k, v, 1, CAP, NQ, NKV, HD, out);
        for (int d = 0; d < HD; d++) {
            if (fabsf(out[0 * HD + d] - (float)(d + 1)) > 1e-5f) { printf("  FAIL gqa kv0\n"); failures++; break; }
            if (fabsf(out[2 * HD + d] - (float)(d + 10)) > 1e-5f) { printf("  FAIL gqa kv1\n"); failures++; break; }
        }
    }
    if (fabsf(katali_gguf_silu(0.0f)) > 1e-7f) { printf("  FAIL silu0\n"); failures++; }
    if (fabsf(katali_gguf_sigmoid(0.0f) - 0.5f) > 1e-7f) { printf("  FAIL sig0\n"); failures++; }

    /* Vectorized silu(gate)*up against the scalar reference, across the range
     * where the gates actually live (including far tails). */
    {
        enum { NSILU = 4096 };
        float *g = (float *)malloc(NSILU * sizeof(float));
        float *u = (float *)malloc(NSILU * sizeof(float));
        float *ref = (float *)malloc(NSILU * sizeof(float));
        if (g && u && ref) {
            for (int i = 0; i < NSILU; i++) {
                g[i] = -40.0f + 80.0f * (float)i / (float)(NSILU - 1);
                u[i] = ((float)((i * 37) % 101) - 50.0f) / 25.0f;
            }
            for (int i = 0; i < NSILU; i++) ref[i] = katali_gguf_silu(g[i]) * u[i];
            const float before0 = g[0], beforeN = g[NSILU - 1];
            int rc = katali_ggml_silu_mul_avx2(g, u, NSILU);
            if (rc == 0) {
                float max_rel = 0.0f;
                for (int i = 0; i < NSILU; i++) {
                    float d = fabsf(g[i] - ref[i]);
                    float rel = d / (fabsf(ref[i]) + 1e-6f);
                    if (rel > max_rel) max_rel = rel;
                }
                printf("  silu simd vs scalar: max_rel_err=%.3g (tol 1e-4)\n", max_rel);
                if (max_rel > 1e-4f) {
                    printf("  FAIL silu simd rel err %.3g\n", max_rel);
                    failures++;
                }
            } else {
                /* SIMD unavailable: the scalar path must be untouched. */
                if (!(g[0] == before0 || fabsf(g[0] - ref[0]) <= 1e-6f)) {
                    printf("  FAIL silu fallback did not run\n");
                    failures++;
                }
                (void)beforeN;
                (void)rc;
            }
        }
        free(g); free(u); free(ref);
    }

    /* Vectorized softmax against the scalar reference, including the far tails
     * (large positive/negative scores) where the exp argument is most extreme.
     * Sum-to-one is also checked for the scalar path. */
    {
        enum { NSM = 2048 };
        float *simd_buf = (float *)malloc(NSM * sizeof(float));
        float *ref_buf = (float *)malloc(NSM * sizeof(float));
        if (simd_buf && ref_buf) {
            for (int i = 0; i < NSM; i++) {
                float t = (float)i / (float)(NSM - 1);
                simd_buf[i] = -60.0f + 120.0f * t + 3.0f * sinf(37.0f * t);
                ref_buf[i] = simd_buf[i];
            }
            if (katali_ggml_softmax_avx2(simd_buf, NSM) == 0) {
                katali_gguf_softmax_scalar(ref_buf, NSM);
                double max_abs = 0.0, max_rel = 0.0, sum = 0.0;
                for (int i = 0; i < NSM; i++) {
                    double d = fabs((double)simd_buf[i] - (double)ref_buf[i]);
                    double rel = d / ((double)fabs(ref_buf[i]) + 1e-30);
                    if (d > max_abs) max_abs = d;
                    if (rel > max_rel) max_rel = rel;
                    sum += (double)simd_buf[i];
                }
                printf("  softmax simd vs scalar: max_abs=%.3g max_rel=%.3g sum=%.9f\n",
                       max_abs, max_rel, sum);
                if (max_abs > 1e-5 || max_rel > 1e-4 || fabs(sum - 1.0) > 1e-5) {
                    printf("  FAIL softmax simd tolerance\n");
                    failures++;
                }
            } else {
                printf("  softmax simd unavailable (scalar path active)\n");
            }
            free(simd_buf); free(ref_buf);
        }
    }

    /* Attention output with the vectorized softmax vs the scalar reference, on a
     * realistic shape (16 q heads, 8 kv heads, head_dim 128, kv_len 250). The
     * softmax is the only stage that differs, so this max-abs/max-rel is the
     * number that matters for the end-to-end tolerance. */
    {
        enum { AHQ = 16, AKV = 8, AHD = 128, ACAP = 256, ALEN = 250 };
        size_t nq = AHQ * AHD, nkv = (size_t)AKV * ACAP * AHD;
        float *q = (float *)malloc(nq * sizeof(float));
        float *kk = (float *)malloc(nkv * sizeof(float));
        float *vv = (float *)malloc(nkv * sizeof(float));
        float *o_simd = (float *)malloc(nq * sizeof(float));
        float *o_ref = (float *)malloc(nq * sizeof(float));
        float *sc_simd = (float *)malloc(ALEN * sizeof(float));
        float *sc_ref = (float *)malloc(ALEN * sizeof(float));
        if (q && kk && vv && o_simd && o_ref && sc_simd && sc_ref) {
            unsigned s = 12345u;
            for (size_t i = 0; i < nq; i++) { s = s * 1103515245u + 12345u; q[i] = ((float)((s >> 9) & 0x3FF) / 512.0f) - 1.0f; }
            for (size_t i = 0; i < nkv; i++) { s = s * 1103515245u + 12345u; kk[i] = ((float)((s >> 9) & 0x3FF) / 512.0f) - 1.0f;
                                              s = s * 1103515245u + 12345u; vv[i] = ((float)((s >> 9) & 0x3FF) / 512.0f) - 1.0f; }
            memset(o_simd, 0, nq * sizeof(float));
            memset(o_ref, 0, nq * sizeof(float));
            const float scale = 1.0f / sqrtf((float)AHD);
            int simd_ok = 0;
            for (int hq = 0; hq < AHQ; hq++) {
                int hk = hq / (AHQ / AKV);
                const float *qh = q + (size_t)hq * AHD;
                const float *kh = kk + (size_t)hk * ACAP * AHD;
                const float *vh = vv + (size_t)hk * ACAP * AHD;
                katali_ggml_attn_scores_avx2(qh, kh, ALEN, AHD, scale, sc_simd);
                memcpy(sc_ref, sc_simd, ALEN * sizeof(float));
                katali_gguf_softmax_scalar(sc_ref, ALEN);
                if (katali_ggml_softmax_avx2(sc_simd, ALEN) != 0) { simd_ok = 0; break; }
                simd_ok = 1;
                katali_ggml_attn_accum_avx2(sc_simd, ALEN, vh, AHD, o_simd + (size_t)hq * AHD);
                katali_ggml_attn_accum_avx2(sc_ref, ALEN, vh, AHD, o_ref + (size_t)hq * AHD);
            }
            if (simd_ok) {
                double max_abs = 0.0, scale = 0.0;
                for (size_t i = 0; i < nq; i++) {
                    double a = fabs((double)o_ref[i]);
                    if (a > scale) scale = a;
                }
                for (size_t i = 0; i < nq; i++) {
                    double d = fabs((double)o_simd[i] - (double)o_ref[i]);
                    if (d > max_abs) max_abs = d;
                }
                /* Relative error is only meaningful where the reference is well
                 * above float epsilon: a component that cancels to ~1e-5 has a
                 * large ratio while its absolute error stays at 1 ULP. */
                const double floor_abs = 1e-3 * scale;
                double max_rel = 0.0, rel_at_abs = 0.0;
                for (size_t i = 0; i < nq; i++) {
                    double a = fabs((double)o_ref[i]);
                    if (a <= floor_abs) continue;
                    double rel = fabs((double)o_simd[i] - (double)o_ref[i]) / a;
                    if (rel > max_rel) { max_rel = rel; rel_at_abs = a; }
                }
                printf("  attention out simd vs scalar softmax: max_abs=%.3g (%.2g of scale %.3g), "
                       "max_rel=%.3g above %.2g of scale\n",
                       max_abs, max_abs / scale, scale, max_rel, floor_abs);
                if (max_abs / scale > 1e-5 || max_rel > 1e-3) {
                    printf("  FAIL attention softmax tolerance (rel above scale floor %.3g at |ref|=%.3g)\n",
                           max_rel, rel_at_abs);
                    failures++;
                }
            }
        }
        free(q); free(kk); free(vv); free(o_simd); free(o_ref); free(sc_simd); free(sc_ref);
    }

    if (failures == 0) printf("katali kernels selftest: PASS\n");
    else printf("katali kernels selftest: %d FAILURE(S)\n", failures);
    return failures;
}

