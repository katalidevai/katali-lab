/* Katali-GGUF Qwen2/Qwen3 dense model ? Apache-2.0 */
#include "katali_gguf_model.h"
#include "katali_dense_cuda.h"
#include "katali_gguf_dtype.h"
#include "katali_gguf_kernels.h"
#include "katali_gguf_simd.h"
#include "katali_gguf_threads.h"
#include "katali_gguf_prof.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <math.h>

#ifdef _WIN32
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0600
#endif
#include <windows.h>
static double now_s(void) {
    static LARGE_INTEGER freq;
    static int have = 0;
    if (!have) { QueryPerformanceFrequency(&freq); have = 1; }
    LARGE_INTEGER c;
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)freq.QuadPart;
}
#else
#include <time.h>
static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
#endif

static void set_err(char *err, size_t cap, const char *fmt, ...) {
    if (!err || !cap) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(err, cap, fmt, ap);
    va_end(ap);
}

static const KataliGgufTensor *T(const KataliGgufFile *f, const char *fmt, ...) {
    char name[256];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(name, sizeof(name), fmt, ap);
    va_end(ap);
    return katali_gguf_find_tensor(f, name);
}

/* Required tensor: missing, or present with an unsupported type. */
static int need_tensor(const KataliGgufTensor *t, const char *name,
                       char *err, size_t cap) {
    if (!t) { set_err(err, cap, "missing tensor '%s'", name); return -1; }
    if (!t->data) {
        set_err(err, cap, "tensor '%s' uses unsupported type %s",
                name, katali_ggml_type_name(t->type));
        return -1;
    }
    return 0;
}

/* ==================================================================== *
 * Architecture from GGUF metadata                                       *
 * ==================================================================== */
static int64_t arch_i(const KataliGgufFile *f, const char *arch, const char *field,
                      int64_t def) {
    char key[128];
    snprintf(key, sizeof(key), "%s.%s", arch, field);
    int64_t v = katali_gguf_get_int(f, key, INT64_MIN);
    if (v != INT64_MIN) return v;
    return katali_gguf_get_int(f, field, def);
}

static double arch_f(const KataliGgufFile *f, const char *arch, const char *field,
                     double def) {
    char key[128];
    snprintf(key, sizeof(key), "%s.%s", arch, field);
    const KataliGgufKv *kv = katali_gguf_kv_find(f, key);
    if (kv && kv->type != KGGUF_ARRAY) {
        if (kv->type == KGGUF_FLOAT32 || kv->type == KGGUF_FLOAT64) return kv->f;
        return (double)kv->i;
    }
    return katali_gguf_get_float(f, field, def);
}

/* ==================================================================== *
 * Open / close                                                          *
 * ==================================================================== */
int katali_gguf_model_open(KataliGgufModel *m, const KataliBackendOptions *opts,
                           char *err, size_t err_cap) {
    if (!m || !opts || !opts->model_path) return -1;
    memset(m, 0, sizeof(*m));
    katali_gguf_sampler_init(&m->sampler);
    double t0 = now_s();

    int use_mmap = opts->use_mmap ? 1 : 0;
    if (katali_gguf_open(&m->file, opts->model_path, use_mmap, err, err_cap) != 0)
        return -1;
    m->file_open = 1;
    const KataliGgufFile *f = &m->file;
    m->stats.map_s = f->map_s;
    m->stats.parse_s = f->parse_s;

    const char *arch = katali_gguf_get_str(f, "general.architecture", "", NULL);
    snprintf(m->arch, sizeof(m->arch), "%s", arch);
    if (m->arch[0] == '\0') {
        set_err(err, err_cap, "GGUF has no general.architecture");
        goto fail;
    }
    if (strstr(m->arch, "moe") != NULL || arch_i(f, m->arch, "expert_count", 0) > 0) {
        set_err(err, err_cap, "architecture '%s' is a MoE model; Katali-GGUF Phase 1 "
                              "supports dense Qwen2/Qwen3 only", m->arch);
        goto fail;
    }
    if (strncmp(m->arch, "qwen", 4) != 0 && strcmp(m->arch, "llama") != 0) {
        set_err(err, err_cap, "architecture '%s' is not a supported dense Qwen/Llama model",
                m->arch);
        goto fail;
    }

    m->n_layers = (int)arch_i(f, m->arch, "block_count", 0);
    m->hidden   = (int)arch_i(f, m->arch, "embedding_length", 0);
    m->n_heads  = (int)arch_i(f, m->arch, "attention.head_count", 0);
    m->n_kv_heads = (int)arch_i(f, m->arch, "attention.head_count_kv", m->n_heads);
    m->ffn_dim  = (int)arch_i(f, m->arch, "feed_forward_length", 0);
    m->head_dim = (int)arch_i(f, m->arch, "attention.key_length", 0);
    if (m->head_dim <= 0 && m->n_heads > 0 && m->hidden > 0)
        m->head_dim = m->hidden / m->n_heads;
    m->rope_dim = (int)arch_i(f, m->arch, "rope.dimension_count", m->head_dim);
    m->rope_theta = (float)arch_f(f, m->arch, "rope.freq_base", 10000.0);
    m->rms_eps = (float)arch_f(f, m->arch, "attention.layer_norm_rms_epsilon", 1e-5);
    m->ctx_len = (int)arch_i(f, m->arch, "context_length", 4096);
    m->vocab = (int)arch_i(f, m->arch, "vocab_size", 0);
    m->embedding_scale = 1.0f;
    m->residual_scale = 1.0f;
    m->logit_scale = 1.0f;
    if (strcmp(m->arch, "llama") == 0 && m->hidden == 2048 &&
        m->n_layers == 42 && m->vocab == 130560) {
        m->embedding_scale = 12.0f;
        m->residual_scale = 1.4f / sqrtf((float)m->n_layers);
        m->logit_scale = 256.0f / (float)m->hidden;
    }

    if (m->n_layers <= 0 || m->n_layers > KATALI_GGUF_MAX_LAYERS ||
        m->hidden <= 0 || m->n_heads <= 0 || m->head_dim <= 0 ||
        m->n_kv_heads <= 0 || m->n_kv_heads > m->n_heads || m->ffn_dim <= 0) {
        set_err(err, err_cap, "invalid architecture dimensions "
                "(layers=%d hidden=%d heads=%d kv=%d head_dim=%d ffn=%d)",
                m->n_layers, m->hidden, m->n_heads, m->n_kv_heads,
                m->head_dim, m->ffn_dim);
        goto fail;
    }
    if (m->n_heads % m->n_kv_heads != 0) {
        set_err(err, err_cap, "attention.head_count (%d) is not a multiple of "
                "head_count_kv (%d)", m->n_heads, m->n_kv_heads);
        goto fail;
    }
    m->q_dim = m->n_heads * m->head_dim;
    m->kv_dim = m->n_kv_heads * m->head_dim;

    if (katali_gguf_tokenizer_init(&m->tok, f, err, err_cap) != 0) goto fail;
    if (m->vocab <= 0) m->vocab = m->tok.vocab_size;

    /* Keep the public default laptop-safe; larger contexts remain opt-in. */
    m->kv_cap = m->ctx_len < 4096 ? m->ctx_len : 4096;
    if (opts->context_length > 0 && opts->context_length < m->kv_cap)
        m->kv_cap = opts->context_length;
    else if (opts->context_length > m->kv_cap && opts->context_length < m->ctx_len)
        m->kv_cap = opts->context_length;
    if (m->kv_cap < 16) m->kv_cap = 16;
    if (m->kv_cap > m->ctx_len) m->kv_cap = m->ctx_len;

    /* --- weights --- */
    m->embed = T(f, "token_embd.weight");
    if (need_tensor(m->embed, "token_embd.weight", err, err_cap) != 0) goto fail;
    m->out_norm = T(f, "output_norm.weight");
    if (need_tensor(m->out_norm, "output_norm.weight", err, err_cap) != 0) goto fail;
    m->out_w = T(f, "output.weight");
    m->tied = (m->out_w == NULL || m->out_w->data == NULL);
    if (m->tied) m->out_w = m->embed;

    for (int L = 0; L < m->n_layers; L++) {
        KataliGgufLayer *lw = &m->layers[L];
        lw->attn_norm = T(f, "blk.%d.attn_norm.weight", L);
        lw->ffn_norm  = T(f, "blk.%d.ffn_norm.weight", L);
        lw->wq = T(f, "blk.%d.attn_q.weight", L);
        lw->wk = T(f, "blk.%d.attn_k.weight", L);
        lw->wv = T(f, "blk.%d.attn_v.weight", L);
        lw->wo = T(f, "blk.%d.attn_output.weight", L);
        lw->gate = T(f, "blk.%d.ffn_gate.weight", L);
        lw->up   = T(f, "blk.%d.ffn_up.weight", L);
        lw->down = T(f, "blk.%d.ffn_down.weight", L);
        lw->bq = T(f, "blk.%d.attn_q.bias", L);
        lw->bk = T(f, "blk.%d.attn_k.bias", L);
        lw->bv = T(f, "blk.%d.attn_v.bias", L);
        lw->bo = T(f, "blk.%d.attn_output.bias", L);
        lw->rope_freqs = T(f, "blk.%d.rope_freqs.weight", L);
        lw->bgate = T(f, "blk.%d.ffn_gate.bias", L);
        lw->bup = T(f, "blk.%d.ffn_up.bias", L);
        lw->bdown = T(f, "blk.%d.ffn_down.bias", L);
        lw->q_norm = T(f, "blk.%d.attn_q_norm.weight", L);
        lw->k_norm = T(f, "blk.%d.attn_k_norm.weight", L);
        char nm[128];
        struct { const KataliGgufTensor *t; const char *sfx; const char *tag; } req[] = {
            { lw->attn_norm, ".weight", "attn_norm.weight" },
            { lw->ffn_norm,  ".weight", "ffn_norm.weight"  },
            { lw->wq,        ".weight", "attn_q.weight"     },
            { lw->wk,        ".weight", "attn_k.weight"     },
            { lw->wv,        ".weight", "attn_v.weight"     },
            { lw->wo,        ".weight", "attn_output.weight" },
            { lw->gate,      ".weight", "ffn_gate.weight"   },
            { lw->up,        ".weight", "ffn_up.weight"     },
            { lw->down,      ".weight", "ffn_down.weight"   },
        };
        for (size_t ri = 0; ri < sizeof(req) / sizeof(req[0]); ri++) {
            snprintf(nm, sizeof(nm), "blk.%d.%s", L, req[ri].tag);
            if (need_tensor(req[ri].t, nm, err, err_cap) != 0) goto fail;
        }
        (void)nm;
        if (lw->q_norm || lw->k_norm) m->has_qk_norm = 1;
        if (lw->bq || lw->bk || lw->bv) m->has_bias = 1;
    }
    if (m->embed->dims[0] != (uint64_t)m->hidden) {
        set_err(err, err_cap, "token_embd.weight has %llu columns but hidden_size is %d",
                (unsigned long long)m->embed->dims[0], m->hidden);
        goto fail;
    }

    /* --- scratch --- */
    {
        size_t nq = (size_t)m->q_dim, nk = (size_t)m->kv_dim;
        size_t nh = (size_t)m->hidden, nf = (size_t)m->ffn_dim;
        size_t nv = (size_t)m->vocab;
        m->x     = (float *)malloc(nh * sizeof(float));
        m->xb    = (float *)malloc(nh * sizeof(float));
        m->q     = (float *)malloc(nq * sizeof(float));
        m->k     = (float *)malloc(nk * sizeof(float));
        m->v     = (float *)malloc(nk * sizeof(float));
        m->attn  = (float *)malloc(nq * sizeof(float));
        m->gate  = (float *)malloc(nf * sizeof(float));
        m->up    = (float *)malloc(nf * sizeof(float));
        m->down  = (float *)malloc(nh * sizeof(float));
        m->logits = (float *)malloc(nv * sizeof(float));
        size_t nwcap = nh;
        if (nf > nwcap) nwcap = nf;
        if ((size_t)m->q_dim > nwcap) nwcap = (size_t)m->q_dim;
        if ((size_t)m->kv_dim > nwcap) nwcap = (size_t)m->kv_dim;
        if ((size_t)m->head_dim > nwcap) nwcap = (size_t)m->head_dim;
        m->nw = (float *)malloc(nwcap * sizeof(float));
        if (!m->x || !m->xb || !m->q || !m->k || !m->v || !m->attn ||
            !m->gate || !m->up || !m->down || !m->logits || !m->nw) {
            set_err(err, err_cap, "out of memory allocating model scratch");
            goto fail;
        }
    }

    /* --- KV cache --- */
    {
        double tk = now_s();
        size_t per_layer = (size_t)m->n_kv_heads * (size_t)m->kv_cap * (size_t)m->head_dim;
        size_t total = per_layer * (size_t)m->n_layers;
        if (total == 0 || total > (size_t)((uint64_t)1 << 34)) {
            set_err(err, err_cap, "KV cache size is implausible (%llu floats)",
                    (unsigned long long)total);
            goto fail;
        }
        m->kv_k = (float *)calloc(total, sizeof(float));
        m->kv_v = (float *)calloc(total, sizeof(float));
        if (!m->kv_k || !m->kv_v) {
            set_err(err, err_cap, "out of memory allocating KV cache (%llu MB)",
                    (unsigned long long)(total * 8 / (1024 * 1024)));
            goto fail;
        }
        /* One score buffer for the whole run: attention otherwise allocates per
         * layer per token. */
        m->attn_scores = (float *)malloc((size_t)m->kv_cap * sizeof(float));
        if (!m->attn_scores) {
            set_err(err, err_cap, "out of memory allocating attention scratch");
            goto fail;
        }
        /* Parallel attention needs one score buffer per worker. Sized once here
         * (never per token) and only when there is more than one worker; the
         * serial path keeps using attn_scores.
         *
         * NOTE: this must agree with the count the matvec dispatches use, or the
         * pool would be resized on every layer (attention 12, matvec 4, ...).
         * `m->n_threads` is assigned further down, so it is read from `opts`
         * directly here - the same expression matvec uses at call time. */
        m->attn_threads = opts->n_threads > 0 ? opts->n_threads
                                              : katali_gguf_get_threads();
        if (m->attn_threads > 256) m->attn_threads = 256;
        m->attn_pool_stride = (size_t)m->kv_cap;
        if (m->attn_threads > 1) {
            size_t pool = (size_t)m->attn_threads * m->attn_pool_stride;
            m->attn_scores_pool = (float *)malloc(pool * sizeof(float));
            if (!m->attn_scores_pool) {
                set_err(err, err_cap, "out of memory allocating parallel attention scratch");
                goto fail;
            }
        }
        m->stats.kv_alloc_s = now_s() - tk;
    }

    /* --- Q4_K activation-sum scratch ------------------------------------- */
    /* Sized once for the largest Q4_K tensor in the file so the decode loop
     * never allocates. If there is no Q4_K tensor the pointer stays NULL and
     * the matvec keeps its per-row recomputation path. */
    {
        uint64_t max_cols = 0;
        for (uint64_t i = 0; i < m->file.tensor_count; i++) {
            const KataliGgufTensor *t = &m->file.tensors[i];
            if (t->type == KGGML_Q4_K && t->n_dims >= 1 && t->dims[0] > max_cols)
                max_cols = t->dims[0];
        }
        m->q4k_sums_cap = KATALI_Q4K_SUMS_FLOATS(max_cols);
        if (m->q4k_sums_cap) {
            m->q4k_sums = (float *)malloc((size_t)m->q4k_sums_cap * sizeof(float));
            if (!m->q4k_sums) {
                set_err(err, err_cap, "out of memory allocating Q4_K sum scratch");
                goto fail;
            }
        }
    }

    /* --- RoPE tables ------------------------------------------------------ */
    {
        size_t rd = (m->rope_dim == 0 || m->rope_dim > m->head_dim)
                        ? (size_t)m->head_dim : (size_t)m->rope_dim;
        size_t half = rd / 2;
        if (half < 1) half = 1;
        m->rope_inv = (float *)malloc(half * sizeof(float));
        m->rope_cos = (float *)malloc(half * sizeof(float));
        m->rope_sin = (float *)malloc(half * sizeof(float));
        if (!m->rope_inv || !m->rope_cos || !m->rope_sin) {
            set_err(err, err_cap, "out of memory allocating RoPE tables");
            goto fail;
        }
        for (size_t i = 0; i < half; i++)
            m->rope_inv[i] = 1.0f / powf(m->rope_theta, (2.0f * (float)i) / (float)rd);
    }

    m->n_threads = opts->n_threads;
    m->max_tokens = opts->max_tokens > 0 ? opts->max_tokens : 128;
    m->think = opts->think;
    m->batch_prefill = -1; /* auto: env/default decides per call */
    katali_gguf_sampler_init(&m->sampler);
    m->sampler.temperature = (float)opts->temperature_milli / 1000.0f;
    m->sampler.top_k = opts->top_k;
    m->sampler.top_p = opts->top_p_milli > 0 ? (float)opts->top_p_milli / 1000.0f : 1.0f;
    m->sampler.repetition_penalty =
        opts->repetition_penalty_milli > 1000
            ? (float)opts->repetition_penalty_milli / 1000.0f : 1.0f;
    m->sampler.seed = opts->seed ? (unsigned)opts->seed : 1u;

    m->stats.open_s = now_s() - t0;
    m->stats.total_s = m->stats.open_s;
    m->stats.resident_bytes = katali_gguf_model_resident_bytes(m);
    return 0;

fail:
    katali_gguf_model_close(m);
    return -1;
}

void katali_gguf_model_close(KataliGgufModel *m) {
    if (!m) return;
    free(m->kv_k);
    free(m->kv_v);
    free(m->x); free(m->xb); free(m->q); free(m->k); free(m->v);
    free(m->attn); free(m->gate); free(m->up); free(m->down);
    free(m->logits); free(m->nw);
    free(m->rope_inv); free(m->rope_cos); free(m->rope_sin);
    free(m->attn_scores);
    free(m->attn_scores_pool);
    free(m->q4k_sums);
    if (m->file_open) {
        katali_gguf_tokenizer_free(&m->tok);
        katali_gguf_close(&m->file);
    }
    memset(m, 0, sizeof(*m));
}

void katali_gguf_model_reset(KataliGgufModel *m) {
    if (!m) return;
    double t = now_s();
    /* Only the logical length gates attention reads, and it is the single
     * piece of per-question state. Clearing it removes every trace of the
     * previous question without touching the mapped weights. */
    m->kv_len = 0;
    m->cancel = 0;
    m->stats.last_reset_s = now_s() - t;
    m->stats.reset_count++;
}

void katali_gguf_model_last_report(const KataliGgufModel *m, KataliGgufGenReport *out) {
    if (!out) return;
    if (!m) { memset(out, 0, sizeof(*out)); return; }
    *out = m->last;
}

void katali_gguf_model_cancel(KataliGgufModel *m) {
    if (m) m->cancel = 1;
}

void katali_gguf_model_set_token_callback(KataliGgufModel *m,
                                          int (*fn)(const char *bytes, size_t len, void *user),
                                          void *user) {
    if (!m) return;
    m->on_token = fn;
    m->on_token_user = user;
}

int katali_gguf_model_stopped_by_callback(const KataliGgufModel *m) {
    return m ? m->stopped_by_cb : 0;
}

int katali_gguf_model_cancelled(const KataliGgufModel *m) {
    return m && m->cancel ? 1 : 0;
}

size_t katali_gguf_model_resident_bytes(const KataliGgufModel *m) {
    if (!m) return 0;
    size_t bytes = 0;
    if (m->kv_k && m->kv_v) {
        size_t per_layer = (size_t)m->n_kv_heads * (size_t)m->kv_cap * (size_t)m->head_dim;
        bytes += per_layer * (size_t)m->n_layers * sizeof(float) * 2;
    }
    size_t scratch = (size_t)m->hidden * 2 + (size_t)m->q_dim + (size_t)m->kv_dim * 2 +
                     (size_t)m->ffn_dim * 2 + (size_t)m->hidden + (size_t)m->vocab;
    bytes += scratch * sizeof(float);
    bytes += (size_t)m->hidden * sizeof(float); /* nw */
    {
        size_t rd = (m->rope_dim == 0 || m->rope_dim > m->head_dim)
                        ? (size_t)m->head_dim : (size_t)m->rope_dim;
        size_t half = rd / 2; if (half < 1) half = 1;
        bytes += half * sizeof(float) * 3; /* rope_inv/cos/sin */
    }
    bytes += (size_t)m->kv_cap * sizeof(float); /* attention score scratch */
    bytes += (size_t)m->q4k_sums_cap * sizeof(float); /* Q4_K activation sums */
    bytes += (size_t)m->tok.vocab_size * (sizeof(void *) + sizeof(size_t) + sizeof(int));
    bytes += (size_t)m->tok.hash_cap * sizeof(int);
    if (!m->file.mapped) bytes += (size_t)m->file.size; /* heap-loaded weights */
    return bytes;
}

/* ==================================================================== *
 * Forward                                                              *
 * ==================================================================== */
/* Phase-timer helpers. t0 == 0 means profiling is disabled, so a disabled
 * build pays only an integer compare per group. */
static inline double prof0(void) {
    return katali_prof_enabled() ? katali_prof_now() : 0.0;
}
static inline void prof1(int phase, double t0) {
    if (t0 > 0.0) katali_prof_add_n(phase, katali_prof_now() - t0, 1);
}

/* cos/sin for one position, from the cached inverse frequencies. The values are
 * identical to computing them inside katali_gguf_rope(), but they are shared by
 * the q and k rotations and by every layer of the token. */
static void rope_prepare(KataliGgufModel *m, int pos) {
    size_t rd = (m->rope_dim == 0 || m->rope_dim > m->head_dim)
                    ? (size_t)m->head_dim : (size_t)m->rope_dim;
    size_t half = rd / 2;
    for (size_t i = 0; i < half; i++) {
        float freq = (float)pos * m->rope_inv[i];
        m->rope_cos[i] = cosf(freq);
        m->rope_sin[i] = sinf(freq);
    }
}

static const float *load_norm(KataliGgufModel *m, const KataliGgufTensor *t, int count) {
    if (!t || !t->data) return NULL;
    double t0 = prof0();
    int rc = katali_ggml_dequant_ref(t->type, t->data, (uint64_t)count, m->nw);
    prof1(KATALI_PHASE_NORMDQ, t0);
    if (rc != 0) return NULL;
    return m->nw;
}

static void add_bias(KataliGgufModel *m, float *y, const KataliGgufTensor *t, int count) {
    if (!t || !t->data) return;
    if (katali_ggml_dequant_ref(t->type, t->data, (uint64_t)count, m->nw) != 0) return;
    for (int i = 0; i < count; i++) y[i] += m->nw[i];
}

/* y = W @ x, where W is a GGUF matrix with dims[0]=cols (contiguous).
 * `role` is the KataliMatvecRole this projection plays, used only for the
 * per-role kernel accounting (never for behaviour). */
static int matvec_t(KataliGgufModel *m, const KataliGgufTensor *t,
                    const float *x, float *y, int role) {
    if (!t || !t->data) return -1;
    uint64_t rows = t->dims[1];
    uint64_t cols = t->dims[0];
    if (strcmp(m->arch, "qwen3") == 0 && katali_dense_cuda_matvec(t, x, y) == 0) return 0;
    katali_ggml_matvec_role(t->type, t->data, rows, cols, x, y, m->n_threads,
                            (volatile int *)&m->cancel,
                            m->q4k_sums, m->q4k_sums_cap, role);
    return 0;
}

/* 1 unless KATALI_GGUF_NO_FUSE_DISPATCH=1. Several projections in the decode
 * path read the same activation vector and have no dependency on each other
 * (q/k/v all read m->xb; gate and up all read m->xb), so they can share one
 * dispatch. Fusing q/k/v does not reorder any arithmetic: the bias, q/k norm
 * and RoPE steps that follow each projection still run in their original order
 * on their own buffers. */
static int fuse_dispatch_enabled(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("KATALI_GGUF_NO_FUSE_DISPATCH");
        v = (e && *e && e[0] != '0') ? 0 : 1;
    }
    return v;
}

/* Computes several same-input projections, sharing one dispatch when enabled.
 * Returns -1 if any tensor is missing, matching matvec_t(). */
static int matvec_multi_t(KataliGgufModel *m, const KataliGgufTensor **ts,
                          float **ys, const int *roles, int n, const float *x) {
    if (n <= 0) return 0;
    uint64_t cols = ts[0] ? ts[0]->dims[0] : 0;
    for (int i = 0; i < n; i++) {
        if (!ts[i] || !ts[i]->data) return -1;
        /* All jobs must share the activation vector's length. */
        if (ts[i]->dims[0] != cols) return -1;
    }
    if (strcmp(m->arch, "qwen3") == 0 && n > 1 && katali_dense_cuda_matvec_multi(ts, ys, n, x) == 0) return 0;
    if (n == 1 || cols == 0 || !fuse_dispatch_enabled()) {
        for (int i = 0; i < n; i++) {
            katali_ggml_matvec_role(ts[i]->type, ts[i]->data, ts[i]->dims[1], cols,
                                    x, ys[i], m->n_threads, (volatile int *)&m->cancel,
                                    m->q4k_sums, m->q4k_sums_cap, roles[i]);
        }
        return 0;
    }
    KataliMatvecJob jobs[4];
    if (n > (int)(sizeof(jobs) / sizeof(jobs[0]))) return -1;
    for (int i = 0; i < n; i++) {
        jobs[i].type = ts[i]->type;
        jobs[i].w = ts[i]->data;
        jobs[i].rows = ts[i]->dims[1];
        jobs[i].y = ys[i];
        jobs[i].role = roles[i];
    }
    katali_ggml_matvec_fused(jobs, n, cols, x, m->n_threads,
                             (volatile int *)&m->cancel,
                             m->q4k_sums, m->q4k_sums_cap);
    return 0;
}

/* Y[B][rows] = X[B][cols] @ W^T, W a GGUF matrix (dims[0]=cols). */
static int matmul_t(KataliGgufModel *m, const KataliGgufTensor *t,
                    const float *X, uint64_t B, float *Y, int role) {
    if (!t || !t->data) return -1;
    uint64_t rows = t->dims[1];
    uint64_t cols = t->dims[0];
    katali_ggml_matmul_role(t->type, t->data, rows, cols, X, B, Y, m->n_threads,
                            (volatile int *)&m->cancel, role);
    return 0;
}

static void rope_prepare_layer(KataliGgufModel *m, const KataliGgufTensor *t, int pos);

static int gguf_forward_token(KataliGgufModel *m, int token_id, int pos) {
    if (token_id < 0 || token_id >= m->vocab) return -1;
    size_t row_bytes = katali_ggml_row_bytes(m->embed->type, (uint64_t)m->hidden);
    if (row_bytes == 0) return -1;
    const uint8_t *row = m->embed->data + (size_t)token_id * row_bytes;
    /* Timed separately from the LM head: this is a single-row dequantization of
     * O(hidden) values, not a matvec over 151936 rows, and grouping the two would
     * hide the LM head's real cost behind a near-free operation. */
    double t_embed = prof0();
    if (katali_ggml_dequant_ref(m->embed->type, row, (uint64_t)m->hidden, m->x) != 0)
        return -1;
    if (m->embedding_scale != 1.0f)
        for (int i = 0; i < m->hidden; i++) m->x[i] *= m->embedding_scale;
    prof1(KATALI_PHASE_EMBED, t_embed);

    const int H = m->hidden, NH = m->n_heads, NKV = m->n_kv_heads, HD = m->head_dim;
    const size_t per_layer = (size_t)NKV * (size_t)m->kv_cap * (size_t)HD;

    /* Position-dependent frequencies are computed once per token, not once per
     * layer, and shared by the q and k rotations. */
    {
        double t_rp = prof0();
        rope_prepare(m, pos);
        prof1(KATALI_PHASE_ROPE, t_rp);
    }

    for (int L = 0; L < m->n_layers; L++) {
        if (m->cancel) return -1;
        KataliGgufLayer *lw = &m->layers[L];
        rope_prepare_layer(m, lw->rope_freqs, pos);

        const float *nw = load_norm(m, lw->attn_norm, H);
        if (!nw) return -1;
        double t_norm = prof0();
        katali_gguf_rmsnorm(m->x, nw, (size_t)H, m->rms_eps, m->xb);
        prof1(KATALI_PHASE_RMSNORM, t_norm);

        /* q, k and v all read m->xb and do not depend on each other, so they
         * share one dispatch; the bias / q-norm / RoPE steps below are
         * unchanged and still run in their original order on their own buffers. */
        {
            const KataliGgufTensor *qkv_t[3] = { lw->wq, lw->wk, lw->wv };
            float *qkv_y[3] = { m->q, m->k, m->v };
            const int qkv_r[3] = { KATALI_ROLE_WQ, KATALI_ROLE_WK, KATALI_ROLE_WV };
            if (matvec_multi_t(m, qkv_t, qkv_y, qkv_r, 3, m->xb) != 0) return -1;
        }
        add_bias(m, m->q, lw->bq, m->q_dim);
        if (lw->q_norm) {
            const float *qn = load_norm(m, lw->q_norm, HD);
            if (!qn) return -1;
            t_norm = prof0();
            for (int h = 0; h < NH; h++)
                katali_gguf_rmsnorm_inplace(m->q + (size_t)h * HD, qn, (size_t)HD, m->rms_eps);
            prof1(KATALI_PHASE_RMSNORM, t_norm);
        }
        double t_rope = prof0();
        katali_gguf_rope_apply(m->q, (size_t)NH, (size_t)HD, (size_t)m->rope_dim,
                               m->rope_cos, m->rope_sin);
        prof1(KATALI_PHASE_ROPE, t_rope);

        add_bias(m, m->k, lw->bk, m->kv_dim);
        if (lw->k_norm) {
            const float *kn = load_norm(m, lw->k_norm, HD);
            if (!kn) return -1;
            t_norm = prof0();
            for (int h = 0; h < NKV; h++)
                katali_gguf_rmsnorm_inplace(m->k + (size_t)h * HD, kn, (size_t)HD, m->rms_eps);
            prof1(KATALI_PHASE_RMSNORM, t_norm);
        }
        t_rope = prof0();
        katali_gguf_rope_apply(m->k, (size_t)NKV, (size_t)HD, (size_t)m->rope_dim,
                               m->rope_cos, m->rope_sin);
        prof1(KATALI_PHASE_ROPE, t_rope);

        add_bias(m, m->v, lw->bv, m->kv_dim);

        float *lk = m->kv_k + (size_t)L * per_layer;
        float *lv = m->kv_v + (size_t)L * per_layer;
        double t_kv = prof0();
        for (int h = 0; h < NKV; h++) {
            float *dst_k = lk + ((size_t)h * m->kv_cap + (size_t)pos) * HD;
            float *dst_v = lv + ((size_t)h * m->kv_cap + (size_t)pos) * HD;
            memcpy(dst_k, m->k + (size_t)h * HD, (size_t)HD * sizeof(float));
            memcpy(dst_v, m->v + (size_t)h * HD, (size_t)HD * sizeof(float));
        }
        prof1(KATALI_PHASE_KV, t_kv);

        double t_attn = prof0();
        katali_gguf_gqa_decode_parallel(m->q, lk, lv, (size_t)pos + 1, (size_t)m->kv_cap,
                                        (size_t)NH, (size_t)NKV, (size_t)HD, m->attn,
                                        m->attn_scores, m->attn_scores_pool,
                                        m->attn_pool_stride, m->attn_threads,
                                        (volatile int *)&m->cancel);
        prof1(KATALI_PHASE_ATTENTION, t_attn);
        if (matvec_t(m, lw->wo, m->attn, m->down, KATALI_ROLE_WO) != 0) return -1;
        add_bias(m, m->down, lw->bo, H);
        if (m->residual_scale == 1.0f) katali_gguf_add_inplace(m->x, m->down, (size_t)H);
        else for (int i = 0; i < H; i++) m->x[i] += m->residual_scale * m->down[i];

        nw = load_norm(m, lw->ffn_norm, H);
        if (!nw) return -1;
        t_norm = prof0();
        katali_gguf_rmsnorm(m->x, nw, (size_t)H, m->rms_eps, m->xb);
        prof1(KATALI_PHASE_RMSNORM, t_norm);

        /* gate and up share m->xb and are independent, so one dispatch. */
        {
            const KataliGgufTensor *gu_t[2] = { lw->gate, lw->up };
            float *gu_y[2] = { m->gate, m->up };
            const int gu_r[2] = { KATALI_ROLE_GATE, KATALI_ROLE_UP };
            if (matvec_multi_t(m, gu_t, gu_y, gu_r, 2, m->xb) != 0) return -1;
            add_bias(m, m->gate, lw->bgate, m->ffn_dim);
            add_bias(m, m->up, lw->bup, m->ffn_dim);
        }
        double t_act = prof0();
        if (katali_ggml_silu_mul_avx2(m->gate, m->up, (size_t)m->ffn_dim) != 0) {
            for (int i = 0; i < m->ffn_dim; i++)
                m->gate[i] = katali_gguf_silu(m->gate[i]) * m->up[i];
        }
        prof1(KATALI_PHASE_ACTIVATION, t_act);
        if (matvec_t(m, lw->down, m->gate, m->down, KATALI_ROLE_DOWN) != 0) return -1;
        add_bias(m, m->down, lw->bdown, H);
        if (m->residual_scale == 1.0f) katali_gguf_add_inplace(m->x, m->down, (size_t)H);
        else for (int i = 0; i < H; i++) m->x[i] += m->residual_scale * m->down[i];
    }

    const float *nw = load_norm(m, m->out_norm, H);
    if (!nw) return -1;
    katali_gguf_rmsnorm(m->x, nw, (size_t)H, m->rms_eps, m->xb);
    if (matvec_t(m, m->out_w, m->xb, m->logits, KATALI_ROLE_LM_HEAD) != 0) return -1;
    if (m->logit_scale != 1.0f)
        for (int i = 0; i < m->vocab; i++) m->logits[i] *= m->logit_scale;
    return 0;
}

/* ==================================================================== *
 * Batched prompt prefill                                                *
 *                                                                       *
 * Evaluates a block of prompt tokens together. Only the projections are *
 * batched; attention reuses the exact decode kernel with a growing      *
 * kv_len, so causal masking and the tested attention path are unchanged.*
 * The token-by-token path is still used for decoding and as a fallback. *
 * ==================================================================== */
#define KATALI_PREFILL_CHUNK 64
#define KATALI_PREFILL_BATCH_MIN 8

static int gguf_prefill_chunk(KataliGgufModel *m, const int *ids, int n0, int B,
                              float *X, float *XB, float *Q, float *K, float *V,
                              float *ATT, float *O, float *G, float *U) {
    const int H = m->hidden, FFN = m->ffn_dim, QD = m->q_dim, KVD = m->kv_dim;
    const int NH = m->n_heads, NKV = m->n_kv_heads, HD = m->head_dim;
    const size_t per_layer = (size_t)NKV * (size_t)m->kv_cap * (size_t)HD;

    size_t row_bytes = katali_ggml_row_bytes(m->embed->type, (uint64_t)H);
    if (row_bytes == 0) return -1;
    for (int t = 0; t < B; t++) {
        int id = ids[n0 + t];
        if (id < 0 || id >= m->vocab) return -1;
        const uint8_t *row = m->embed->data + (size_t)id * row_bytes;
        if (katali_ggml_dequant_ref(m->embed->type, row, (uint64_t)H,
                                    X + (size_t)t * H) != 0) return -1;
        if (m->embedding_scale != 1.0f)
            for (int i = 0; i < H; i++) X[(size_t)t * H + i] *= m->embedding_scale;
    }

    for (int L = 0; L < m->n_layers; L++) {
        if (m->cancel) return -1;
        KataliGgufLayer *lw = &m->layers[L];

        const float *nw = load_norm(m, lw->attn_norm, H);
        if (!nw) return -1;
        for (int t = 0; t < B; t++)
            katali_gguf_rmsnorm(X + (size_t)t * H, nw, (size_t)H, m->rms_eps,
                                XB + (size_t)t * H);

        if (matmul_t(m, lw->wq, XB, (uint64_t)B, Q, KATALI_ROLE_WQ) != 0) return -1;
        if (matmul_t(m, lw->wk, XB, (uint64_t)B, K, KATALI_ROLE_WK) != 0) return -1;
        if (matmul_t(m, lw->wv, XB, (uint64_t)B, V, KATALI_ROLE_WV) != 0) return -1;

        float *lk = m->kv_k + (size_t)L * per_layer;
        float *lv = m->kv_v + (size_t)L * per_layer;
        for (int t = 0; t < B; t++) {
            int pos = n0 + t;
            float *qt = Q + (size_t)t * QD;
            float *kt = K + (size_t)t * KVD;
            float *vt = V + (size_t)t * KVD;
            add_bias(m, qt, lw->bq, QD);
            add_bias(m, kt, lw->bk, KVD);
            add_bias(m, vt, lw->bv, KVD);
            if (lw->q_norm) {
                const float *qn = load_norm(m, lw->q_norm, HD);
                if (!qn) return -1;
                for (int h = 0; h < NH; h++)
                    katali_gguf_rmsnorm_inplace(qt + (size_t)h * HD, qn,
                                                (size_t)HD, m->rms_eps);
            }
            if (lw->k_norm) {
                const float *kn = load_norm(m, lw->k_norm, HD);
                if (!kn) return -1;
                for (int h = 0; h < NKV; h++)
                    katali_gguf_rmsnorm_inplace(kt + (size_t)h * HD, kn,
                                                (size_t)HD, m->rms_eps);
            }
            rope_prepare_layer(m, lw->rope_freqs, pos);
            katali_gguf_rope_apply(qt, (size_t)NH, (size_t)HD, (size_t)m->rope_dim,
                                   m->rope_cos, m->rope_sin);
            katali_gguf_rope_apply(kt, (size_t)NKV, (size_t)HD, (size_t)m->rope_dim,
                                   m->rope_cos, m->rope_sin);
            for (int h = 0; h < NKV; h++) {
                memcpy(lk + ((size_t)h * m->kv_cap + (size_t)pos) * HD,
                       kt + (size_t)h * HD, (size_t)HD * sizeof(float));
                memcpy(lv + ((size_t)h * m->kv_cap + (size_t)pos) * HD,
                       vt + (size_t)h * HD, (size_t)HD * sizeof(float));
            }
        }

        /* Causal attention: each token reads only positions <= its own, which
         * is exactly what the decode kernel does with kv_len = pos + 1. */
        for (int t = 0; t < B; t++) {
            int pos = n0 + t;
            katali_gguf_gqa_decode_parallel(Q + (size_t)t * QD, lk, lv, (size_t)pos + 1,
                                            (size_t)m->kv_cap, (size_t)NH, (size_t)NKV,
                                            (size_t)HD, ATT + (size_t)t * QD,
                                            m->attn_scores, m->attn_scores_pool,
                                            m->attn_pool_stride, m->attn_threads,
                                            (volatile int *)&m->cancel);
        }

        if (matmul_t(m, lw->wo, ATT, (uint64_t)B, O, KATALI_ROLE_WO) != 0) return -1;
        for (int t = 0; t < B; t++)
            add_bias(m, O + (size_t)t * H, lw->bo, H);
        for (int t = 0; t < B; t++)
            if (m->residual_scale == 1.0f) katali_gguf_add_inplace(X + (size_t)t * H, O + (size_t)t * H, (size_t)H);
            else for (int i = 0; i < H; i++) X[(size_t)t * H + i] += m->residual_scale * O[(size_t)t * H + i];

        nw = load_norm(m, lw->ffn_norm, H);
        if (!nw) return -1;
        for (int t = 0; t < B; t++)
            katali_gguf_rmsnorm(X + (size_t)t * H, nw, (size_t)H, m->rms_eps,
                                XB + (size_t)t * H);

        if (matmul_t(m, lw->gate, XB, (uint64_t)B, G, KATALI_ROLE_GATE) != 0) return -1;
        if (matmul_t(m, lw->up, XB, (uint64_t)B, U, KATALI_ROLE_UP) != 0) return -1;
        for (int t = 0; t < B; t++) {
            add_bias(m, G + (size_t)t * FFN, lw->bgate, FFN);
            add_bias(m, U + (size_t)t * FFN, lw->bup, FFN);
        }
        for (int t = 0; t < B; t++) {
            float *g = G + (size_t)t * FFN;
            const float *u = U + (size_t)t * FFN;
            if (katali_ggml_silu_mul_avx2(g, u, (size_t)FFN) != 0) {
                for (int i = 0; i < FFN; i++) g[i] = katali_gguf_silu(g[i]) * u[i];
            }
        }
        if (matmul_t(m, lw->down, G, (uint64_t)B, O, KATALI_ROLE_DOWN) != 0) return -1;
        for (int t = 0; t < B; t++)
            add_bias(m, O + (size_t)t * H, lw->bdown, H);
        for (int t = 0; t < B; t++)
            if (m->residual_scale == 1.0f) katali_gguf_add_inplace(X + (size_t)t * H, O + (size_t)t * H, (size_t)H);
            else for (int i = 0; i < H; i++) X[(size_t)t * H + i] += m->residual_scale * O[(size_t)t * H + i];
    }

    /* Only the last prompt token needs logits. */
    const float *nw = load_norm(m, m->out_norm, H);
    if (!nw) return -1;
    katali_gguf_rmsnorm(X + (size_t)(B - 1) * H, nw, (size_t)H, m->rms_eps,
                        XB + (size_t)(B - 1) * H);
    if (matvec_t(m, m->out_w, XB + (size_t)(B - 1) * H, m->logits, KATALI_ROLE_LM_HEAD) != 0) return -1;
    if (m->logit_scale != 1.0f)
        for (int i = 0; i < m->vocab; i++) m->logits[i] *= m->logit_scale;
    return 0;
}

static int gguf_prefill_batched(KataliGgufModel *m, const int *ids, int n) {
    const size_t H = (size_t)m->hidden, FFN = (size_t)m->ffn_dim;
    const size_t QD = (size_t)m->q_dim, KVD = (size_t)m->kv_dim;
    const size_t CH = KATALI_PREFILL_CHUNK;
    size_t need = CH * (H * 3 + QD * 2 + KVD * 2 + FFN * 2);
    float *buf = (float *)malloc(need * sizeof(float));
    if (!buf) return -1;
    float *X   = buf;
    float *XB  = X   + CH * H;
    float *Q   = XB  + CH * H;
    float *K   = Q   + CH * QD;
    float *V   = K   + CH * KVD;
    float *ATT = V   + CH * KVD;
    float *O   = ATT + CH * QD;
    float *G   = O   + CH * H;
    float *U   = G   + CH * FFN;

    int rc = 0;
    m->kv_len = 0;
    for (int n0 = 0; n0 < n && rc == 0; n0 += KATALI_PREFILL_CHUNK) {
        int B = n - n0;
        if (B > KATALI_PREFILL_CHUNK) B = KATALI_PREFILL_CHUNK;
        rc = gguf_prefill_chunk(m, ids, n0, B, X, XB, Q, K, V, ATT, O, G, U);
        if (rc == 0) m->kv_len = n0 + B;
    }
    free(buf);
    return rc;
}

void katali_gguf_model_set_batch_prefill(KataliGgufModel *m, int enable) {
    if (m) m->batch_prefill = enable ? 1 : 0;
}

int katali_gguf_model_batch_prefill_enabled(const KataliGgufModel *m) {
    if (!m) return 0;
    if (m->batch_prefill >= 0) return m->batch_prefill;
    return getenv("KATALI_GGUF_NO_BATCH_PREFILL") == NULL;
}

/* ==================================================================== *
 * Prompt rendering + generation                                         *
 * ==================================================================== */

static int append_id(int *ids, int at, int cap, int id) {
    if (id < 0) return at;
    if (at >= cap) return -1;
    ids[at++] = id;
    return at;
}

static int append_text(const KataliGgufTokenizer *tk, const char *s,
                       int *ids, int at, int cap) {
    int tmp[2048];
    int n = katali_gguf_tokenizer_encode(tk, s, tmp, 2048);
    if (n < 0) return -1;
    for (int i = 0; i < n; i++) {
        if (at >= cap) return -1;
        ids[at++] = tmp[i];
    }
    return at;
}

static int template_has(const KataliGgufTokenizer *tk, const char *needle) {
    if (!tk->has_chat_template || !tk->chat_template) return 0;
    size_t nl = strlen(needle);
    if (nl == 0 || tk->chat_template_len < nl) return 0;
    for (size_t i = 0; i + nl <= tk->chat_template_len; i++)
        if (memcmp(tk->chat_template + i, needle, nl) == 0) return 1;
    return 0;
}

static void rope_prepare_layer(KataliGgufModel *m, const KataliGgufTensor *t, int pos) {
    if (!t || !t->data) { rope_prepare(m, pos); return; }
    size_t rd = (m->rope_dim == 0 || m->rope_dim > m->head_dim) ? (size_t)m->head_dim : (size_t)m->rope_dim;
    size_t half = rd / 2;
    if (katali_ggml_dequant_ref(t->type, t->data, (uint64_t)half, m->nw) != 0) {
        rope_prepare(m, pos); return;
    }
    for (size_t i = 0; i < half; i++) {
        float freq = (float)pos * m->nw[i];
        m->rope_cos[i] = cosf(freq);
        m->rope_sin[i] = sinf(freq);
    }
}

static int is_minicpm5_model(const KataliGgufModel *m) {
    return m && strcmp(m->arch, "llama") == 0 &&
           m->hidden == 2048 && m->n_layers == 42 && m->vocab == 130560;
}

int katali_gguf_model_format_chat(KataliGgufModel *m, const char *user_text,
                                  int think, int fallback_raw,
                                  int *ids, int max_ids) {
    if (!m || !user_text || !ids) return -1;
    const KataliGgufTokenizer *tk = &m->tok;
    /* ChatML is recognised from the file's own vocabulary, never assumed from
     * a hard-coded id. */
    if (tk->im_start_id >= 0 && tk->im_end_id >= 0) {
        int at = 0;
        const int minicpm5 = is_minicpm5_model(m);
        if (minicpm5 && tk->bos_id >= 0) {
            at = append_id(ids, at, max_ids, tk->bos_id);
            if (at < 0) return -1;
        }
        at = append_id(ids, at, max_ids, tk->im_start_id);
        if (at < 0) return -1;
        at = append_text(tk, minicpm5 ? "user\r\n" : "user\n", ids, at, max_ids);
        if (at < 0) return -1;
        at = append_text(tk, user_text, ids, at, max_ids);
        if (at < 0) return -1;
        at = append_id(ids, at, max_ids, tk->im_end_id);
        if (at < 0) return -1;
        at = append_text(tk, minicpm5 ? "\r\n" : "\n", ids, at, max_ids);
        if (at < 0) return -1;
        at = append_id(ids, at, max_ids, tk->im_start_id);
        if (at < 0) return -1;
        at = append_text(tk, minicpm5 ? "assistant\r\n" : "assistant\n", ids, at, max_ids);
        if (at < 0) return -1;
        /* Qwen3's template emits an empty think block to disable thinking; the
         * behaviour is only applied when the file's template actually declares
         * the switch, and the markers are resolved from this file's vocabulary. */
        if (!think && template_has(tk, "enable_thinking")) {
            if (tk->think_open_id >= 0 && tk->think_close_id >= 0) {
                at = append_id(ids, at, max_ids, tk->think_open_id);
                if (at < 0) return -1;
                at = append_text(tk, minicpm5 ? "\r\n\r\n" : "\n\n", ids, at, max_ids);
                if (at < 0) return -1;
                at = append_id(ids, at, max_ids, tk->think_close_id);
                if (at < 0) return -1;
                at = append_text(tk, minicpm5 ? "\r\n\r\n" : "\n\n", ids, at, max_ids);
                if (at < 0) return -1;
            } else {
                at = append_text(tk, "\n\n", ids, at, max_ids);
                if (at < 0) return -1;
            }
        }
        return at;
    }
    if (!fallback_raw) return -1;
    return katali_gguf_tokenizer_encode(tk, user_text, ids, max_ids);
}

int katali_gguf_model_generate(KataliGgufModel *m, const char *prompt,
                               char *out, size_t out_cap) {
    if (!m || !prompt || !out || out_cap == 0) return -1;
    double t0 = now_s();
    memset(&m->last, 0, sizeof(m->last));
    /* Cancellation is not cleared here on purpose: reset_state() is the single
     * place that clears it, so a cancel request can never be lost. */
    m->kv_len = 0;

    int ids[8192];
    double tt = now_s();
    int n = katali_gguf_model_format_chat(m, prompt, m->think, 1, ids, 8192);
    if (getenv("KATALI_DEBUG")) {
        fprintf(stderr, "dense_prompt_tokens=%d:", n);
        for (int i = 0; i < n; i++) fprintf(stderr, " %d", ids[i]);
        fputc('\n', stderr);
    }
    m->last.tokenize_s = now_s() - tt;
    m->stats.tokenize_s += m->last.tokenize_s;
    katali_prof_add_n(KATALI_PHASE_TOKENIZE, m->last.tokenize_s, 1);
    if (n <= 0) {
        m->last.total_s = now_s() - t0;
        m->stats.total_s += m->last.total_s;
        return -3;
    }

    /* Leave at least one KV slot free for the generated tokens. */
    if (n > m->kv_cap - 1) n = m->kv_cap - 1;
    if (n <= 0) {
        m->last.total_s = now_s() - t0;
        m->stats.total_s += m->last.total_s;
        return -3;
    }
    m->last.prompt_tokens = n;

    double tp = now_s();
    int batched = 0;
    if (katali_gguf_model_batch_prefill_enabled(m) && n >= KATALI_PREFILL_BATCH_MIN) {
        if (gguf_prefill_batched(m, ids, n) == 0) {
            batched = 1;
            m->kv_len = n;
        } else if (!m->cancel) {
            m->kv_len = 0; /* fall back to the token-at-a-time path below */
        }
    }
    if (!batched) {
        for (int i = 0; i < n; i++) {
            if (m->cancel) break;
            if (gguf_forward_token(m, ids[i], m->kv_len) != 0) return -4;
            m->kv_len++;
        }
    }
    m->last.batched_prefill = batched;
    m->last.prefill_s = now_s() - tp;
    m->stats.prefill_s += m->last.prefill_s;
    katali_prof_add_n(KATALI_PHASE_PREFILL, m->last.prefill_s, 1);
    m->stats.prefill_calls++;
    m->stats.n_prompt = n;

    double td = now_s();
    int gen = 0;
    size_t used = 0;
    m->stopped_by_cb = 0;
    int recent[64];
    int nrec = 0;
    int in_think = 0;
    char piece[256];
    const int eos1 = m->tok.eos_id, eos2 = m->tok.im_end_id, eos3 = m->tok.endoftext_id;
    double t_first = 0.0;

    while (gen < m->max_tokens && m->kv_len < m->kv_cap) {
        if (m->cancel) break;
        int next;
        double t_samp = prof0();
        next = katali_gguf_sample(&m->sampler, m->logits, m->vocab, recent, nrec);
        if (getenv("KATALI_DEBUG")) fprintf(stderr, "dense_next[%d]=%d\n", gen, next);
        prof1(KATALI_PHASE_SAMPLER, t_samp);
        if (next < 0) break;
        if (next == eos1 || next == eos2 || next == eos3) break;
        if (gen == 0) t_first = now_s(); /* model produced its first token */

        if (nrec < 64) recent[nrec++] = next;
        else { memmove(recent, recent + 1, 63 * sizeof(int)); recent[63] = next; }

        int emit = 1;
        if (next == m->tok.think_open_id) { in_think = 1; emit = 0; }
        else if (next == m->tok.think_close_id) { in_think = 0; emit = 0; }
        if (in_think && !m->think) emit = 0;
        if (emit) {
            int nb = katali_gguf_tokenizer_decode(&m->tok, next, piece, sizeof(piece));
            for (int k = 0; k < nb && used + 1 < out_cap; k++)
                out[used++] = piece[k];
            /* Streaming observer: sees exactly the bytes just appended. Returning
             * non-zero stops generation after this token, like a cancel. */
            if (m->on_token && m->on_token(piece, (size_t)nb, m->on_token_user)) {
                m->stopped_by_cb = 1;
                break;
            }
        }
        if (gguf_forward_token(m, next, m->kv_len) != 0) break;
        m->kv_len++;
        gen++;
    }
    out[used] = '\0';

    m->last.ttft_s = t_first > 0.0 ? (t_first - t0) : (now_s() - t0);
    m->last.decode_s = now_s() - td;
    m->last.generated_tokens = gen;
    m->last.total_s = now_s() - t0;
    m->last.decode_tps = m->last.decode_s > 0.0
                             ? (double)gen / m->last.decode_s : 0.0;
    m->stats.decode_s += m->last.decode_s;
    m->stats.n_gen = gen;
    m->stats.total_s += m->last.total_s;
    m->stats.resident_bytes = katali_gguf_model_resident_bytes(m);
    return gen;
}











