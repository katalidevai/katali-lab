/* Katali-GGUF sampler — Apache-2.0 */
#include "katali_gguf_sampler.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

void katali_gguf_sampler_init(KataliGgufSampler *s) {
    if (!s) return;
    memset(s, 0, sizeof(*s));
    s->temperature = 0.0f;
    s->top_k = 0;
    s->top_p = 1.0f;
    s->repetition_penalty = 1.0f;
    s->seed = 1u;
}

void katali_gguf_sampler_free(KataliGgufSampler *s) {
    if (!s) return;
    free(s->work);
    free(s->order);
    memset(s, 0, sizeof(*s));
}

static unsigned xorshift32(unsigned *st) {
    unsigned x = *st;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    *st = x ? x : 1u;
    return *st;
}

static int ensure_scratch(KataliGgufSampler *s, int n) {
    if (s->work_cap >= n) return 0;
    float *w = (float *)realloc(s->work, (size_t)n * sizeof(float));
    if (!w) return -1;
    s->work = w;
    int *o = (int *)realloc(s->order, (size_t)n * sizeof(int));
    if (!o) return -1;
    s->order = o;
    s->work_cap = n;
    return 0;
}

typedef struct CmpCtx { const float *v; } CmpCtx;

static int cmp_desc(const void *a, const void *b);

static CmpCtx g_cmp;

static int cmp_desc(const void *a, const void *b) {
    float va = g_cmp.v[*(const int *)a];
    float vb = g_cmp.v[*(const int *)b];
    if (va < vb) return 1;
    if (va > vb) return -1;
    return (*(const int *)a) - (*(const int *)b);
}

int katali_gguf_sample(KataliGgufSampler *s, const float *logits, int n,
                       const int *recent, int n_recent) {
    if (!s || !logits || n <= 0) return -1;
    if (s->temperature <= 0.0f) {
        int best = 0;
        float bv = -FLT_MAX;
        for (int i = 0; i < n; i++) if (logits[i] > bv) { bv = logits[i]; best = i; }
        return best;
    }
    if (ensure_scratch(s, n) != 0) return -1;

    memcpy(s->work, logits, (size_t)n * sizeof(float));

    /* Repetition penalty (positive logits divided, negative multiplied). */
    if (s->repetition_penalty > 1.0f && recent && n_recent > 0) {
        float p = s->repetition_penalty;
        for (int i = 0; i < n_recent; i++) {
            int id = recent[i];
            if (id < 0 || id >= n) continue;
            if (s->work[id] > 0.0f) s->work[id] /= p;
            else s->work[id] *= p;
        }
    }

    float inv_t = 1.0f / s->temperature;
    for (int i = 0; i < n; i++) s->work[i] *= inv_t;

    for (int i = 0; i < n; i++) s->order[i] = i;
    g_cmp.v = s->work;
    qsort(s->order, (size_t)n, sizeof(int), cmp_desc);

    int keep = n;
    if (s->top_k > 0 && s->top_k < keep) keep = s->top_k;

    if (s->top_p < 1.0f) {
        /* Softmax over the sorted prefix, keep the smallest prefix whose
         * cumulative probability reaches top_p. */
        float maxv = s->work[s->order[0]];
        double sum = 0.0;
        for (int i = 0; i < keep; i++) sum += exp((double)(s->work[s->order[i]] - maxv));
        if (sum <= 0.0) return s->order[0];
        double cum = 0.0;
        int cut = keep;
        for (int i = 0; i < keep; i++) {
            cum += exp((double)(s->work[s->order[i]] - maxv)) / sum;
            if (cum >= (double)s->top_p) { cut = i + 1; break; }
        }
        keep = cut;
    }

    /* Sample from the kept prefix. */
    float maxv = s->work[s->order[0]];
    double sum = 0.0;
    for (int i = 0; i < keep; i++) sum += exp((double)(s->work[s->order[i]] - maxv));
    if (sum <= 0.0) return s->order[0];
    double r = (double)(xorshift32(&s->seed) >> 8) / 16777216.0 * sum;
    double cum = 0.0;
    for (int i = 0; i < keep; i++) {
        cum += exp((double)(s->work[s->order[i]] - maxv));
        if (r <= cum) return s->order[i];
    }
    return s->order[keep - 1];
}
