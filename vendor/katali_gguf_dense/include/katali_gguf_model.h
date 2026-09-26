/* Katali-GGUF Qwen2/Qwen3 dense model — Apache-2.0 */
#ifndef KATALI_GGUF_MODEL_H
#define KATALI_GGUF_MODEL_H

#include "katali_gguf.h"
#include "katali_gguf_tokenizer.h"
#include "katali_gguf_sampler.h"
#include "katali_backend.h"

#ifdef __cplusplus
extern "C" {
#endif

#define KATALI_GGUF_MAX_LAYERS 256

typedef struct KataliGgufLayer {
    const KataliGgufTensor *attn_norm;
    const KataliGgufTensor *ffn_norm;
    const KataliGgufTensor *wq, *wk, *wv, *wo;
    const KataliGgufTensor *bq, *bk, *bv, *bo, *rope_freqs;
    const KataliGgufTensor *bgate, *bup, *bdown;
    const KataliGgufTensor *q_norm, *k_norm;
    const KataliGgufTensor *gate, *up, *down;
} KataliGgufLayer;

typedef struct KataliGgufStats {
    double open_s;         /* total model open */
    double parse_s;        /* GGUF metadata + directory parse */
    double map_s;          /* file mapping */
    double tokenize_s;     /* cumulative prompt tokenization */
    double prefill_s;      /* cumulative */
    double decode_s;       /* cumulative */
    double kv_alloc_s;
    double total_s;        /* cumulative generate time */
    double last_reset_s;   /* cost of the most recent KV reset */
    int    n_prompt;
    int    n_gen;
    int    reset_count;
    int    prefill_calls;  /* cumulative generate calls that ran a prefill */
    unsigned long long resident_bytes;
} KataliGgufStats;

/* Per-generation breakdown (overwritten by each generate call, unlike the
 * cumulative stats above). This is what the session/benchmark layer reports. */
typedef struct KataliGgufGenReport {
    double tokenize_s;      /* prompt formatting + tokenization */
    double prefill_s;       /* prompt evaluation */
    double ttft_s;          /* generate entry -> first generated token */
    double decode_s;        /* sampling + forward for generated tokens */
    double total_s;         /* whole generate call */
    int    prompt_tokens;
    int    generated_tokens;
    double decode_tps;
    int    batched_prefill; /* 1 when this call used the batched path */
} KataliGgufGenReport;


typedef struct KataliGgufModel {
    KataliGgufFile      file;
    KataliGgufTokenizer tok;
    int                 file_open;

    char arch[64];
    int  n_layers, hidden, n_heads, n_kv_heads, head_dim, ffn_dim, vocab, ctx_len;
    int  q_dim, kv_dim;
    int  rope_dim;
    float rope_theta;
    float embedding_scale;
    float residual_scale;
    float logit_scale;
    float rms_eps;
    int  has_bias;
    int  has_qk_norm;
    int  tied;
    int  eos_id, eos_id2, im_end_id;

    KataliGgufLayer layers[KATALI_GGUF_MAX_LAYERS];
    const KataliGgufTensor *embed;
    const KataliGgufTensor *out_norm;
    const KataliGgufTensor *out_w;

    /* KV cache: layer-major, each layer [n_kv * cap * head_dim]. */
    float *kv_k;
    float *kv_v;
    int    kv_cap;
    int    kv_len;

    /* Scratch (allocated once). */
    float *x, *xb, *q, *k, *v, *attn, *gate, *up, *down, *logits;
    float *nw;   /* dequantized norm/bias weights */
    /* RoPE: inverse frequencies are immutable, cos/sin are per-position and
     * shared by every layer of the token being decoded. */
    float *rope_inv;  /* [rope_dim/2] */
    float *rope_cos;  /* [rope_dim/2] for the current position */
    float *rope_sin;  /* [rope_dim/2] for the current position */
    float *attn_scores; /* [kv_cap] attention score scratch (no per-layer malloc) */
    /* Per-worker attention score buffers for the parallel attention path:
     * attn_threads * attn_pool_stride floats. One shared buffer would be
     * corrupted by concurrent workers, so each worker gets its own slice. */
    float *attn_scores_pool;
    size_t attn_pool_stride;
    int    attn_threads;
    /* Q4_K activation-group sums: depends only on the activation vector, so it
     * is computed once per matvec instead of once per weight row. Sized at open
     * for the largest Q4_K tensor in the file; never allocated in the token
     * loop. NULL/0 disables the hoist (per-row recomputation). */
    float   *q4k_sums;
    uint64_t q4k_sums_cap;

    int  n_threads;
    int  max_tokens;
    volatile int cancel;
    int  think;
    int  batch_prefill;   /* -1 = auto (env/default), 0 = off, 1 = on */
    /* Optional streaming observer (SDK/server). Called on the generating thread
     * with the bytes decoded for one token (UTF-8, possibly ending mid-sequence),
     * after they have been appended to the output buffer. Returning non-zero stops
     * generation, the same contract as cancel. NULL (the default) leaves the loop
     * unchanged. */
    int (*on_token)(const char *bytes, size_t len, void *user);
    void *on_token_user;
    int   stopped_by_cb;  /* last generate ended because on_token asked it to */
    KataliGgufSampler sampler;
    KataliGgufStats    stats;
    KataliGgufGenReport last;
} KataliGgufModel;

/* Open a GGUF model. Returns 0 on success; -1 with a reason in err. */
int  katali_gguf_model_open(KataliGgufModel *m, const KataliBackendOptions *opts,
                            char *err, size_t err_cap);
void katali_gguf_model_close(KataliGgufModel *m);
/* Reset only the KV state, keeping weights mapped (persistent loading). */
void katali_gguf_model_reset(KataliGgufModel *m);

/* Generate one answer. Returns tokens generated, or <0 on error. */
int  katali_gguf_model_generate(KataliGgufModel *m, const char *prompt,
                                char *out, size_t out_cap);
void katali_gguf_model_cancel(KataliGgufModel *m);
size_t katali_gguf_model_resident_bytes(const KataliGgufModel *m);

/* Install (or clear, with fn = NULL) a per-token observer for streaming clients.
 * The callback runs on the generating thread and receives the UTF-8 bytes
 * decoded for one generated token, which by then are already in the output
 * buffer; returning non-zero stops generation early. It is observation only: it
 * cannot change what the model produces. */
void katali_gguf_model_set_token_callback(KataliGgufModel *m,
                                          int (*fn)(const char *bytes, size_t len, void *user),
                                          void *user);
/* 1 when the most recent generate stopped because the token callback asked it to
 * (as opposed to EOS, a token limit, cancellation or an error). */
int  katali_gguf_model_stopped_by_callback(const KataliGgufModel *m);
int  katali_gguf_model_cancelled(const KataliGgufModel *m);

/* Breakdown of the most recent generate call (see KataliGgufGenReport). */
void katali_gguf_model_last_report(const KataliGgufModel *m, KataliGgufGenReport *out);

/* Batched prefill toggle. Default: enabled for prompts at or above
 * KATALI_GGUF_PREFILL_BATCH_MIN tokens, unless KATALI_GGUF_NO_BATCH_PREFILL=1.
 * Exposed so tests/benchmarks can force either path on the same binary. */
void katali_gguf_model_set_batch_prefill(KataliGgufModel *m, int enable);
int  katali_gguf_model_batch_prefill_enabled(const KataliGgufModel *m);

/* Prompt rendering using the model's own tokenizer metadata. Returns the number
 * of ids written, or <0 when usable chat metadata is absent and `fallback_raw`
 * is 0. Exposed for tests. */
int  katali_gguf_model_format_chat(KataliGgufModel *m, const char *user_text,
                                   int think, int fallback_raw,
                                   int *ids, int max_ids);

#ifdef __cplusplus
}
#endif
#endif /* KATALI_GGUF_MODEL_H */
