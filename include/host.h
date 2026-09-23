#ifndef KATALI_HOST_H
#define KATALI_HOST_H
#include "katali.h"
#include "katali_gguf.h"
#include "model.h"
#include "ecache.h"
#include "moe_index.h"
#include "vram_cache.h"
#include "katali_gguf_tokenizer.h"
#include "katali_gguf_sampler.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef struct HostState {
    float *x, *xb, *xb2;
    float *q, *k, *v, *att_out, *gate, *q_gate;
    float *logits;
    float *kv_k, *kv_v;       /* [n_full][n_kv][max_seq][head_dim] */
    float *attn_scores;
    float *rope_cos, *rope_sin;
    float *ssm_state;         /* [n_recurrent][v_heads][kd][vd] */
    float *conv_hist;         /* [n_recurrent][conv_ch][kernel-1] */
    float *qwen_qkv, *qwen_z, *qwen_alpha, *qwen_beta;
    float *qwen_conv, *qwen_recurrent, *tmp;
    float *norm_buf;
    float *q4k_scratch;       /* optional Q4_K activation-sum buffer */
    int    q4k_scratch_cap;
    int    pos;
    int    max_seq;
    int    n_full;
    int    n_recurrent;
    size_t kv_layer_stride;   /* n_kv * max_seq * head_dim */
    size_t ssm_state_stride;
    size_t conv_hist_stride;
    int   *full_index;        /* layer -> full slot, -1 if recurrent */
    int   *rec_index;         /* layer -> recurrent slot, -1 if full */
} HostState;

typedef struct HostModel {
    KataliGgufFile gguf;
    KataliArch arch;
    ECache ecache;
    void *ecache_fill_ud; /* HostEcacheFillCtx*, freed in host_close */
    MoeIndex moe;
    KataliGgufTokenizer tok;
    KataliGgufSampler samp;
    HostState st;
    /* Prior-token top-k expert ids for speculative prefetch: [n_layers * last_topk] */
    int *last_eids;
    int last_topk;
    int last_routes_valid;
    int opened;
    int tok_ok;
    /* Optional VRAM tier (third residency level). All of these stay zero unless
     * the CUDA backend loaded AND the tier opened, so the CPU path is untouched. */
    VramCache vram;
    int   vram_ready;
    void *vram_x;   /* device: activation in     [hidden] */
    void *vram_gy;  /* device: gate out          [moe_intermediate] */
    void *vram_uy;  /* device: up out            [moe_intermediate] */
    void *vram_iy;  /* device: silu(gate)*up     [moe_intermediate] */
    void *vram_dy;  /* device: down out          [hidden] */
    int   vram_fallbacks; /* count of GPU-attempted calls that fell back */
    /*
     * Phase 1 timeline instrumentation (KATALI_CUDA_TIMELINE=1). Seconds
     * accumulated per decode token and printed as one compact line so the real
     * CPU/GPU critical path can be seen instead of assumed.
     */
    double tl_attn, tl_router, tl_shared, tl_gpu, tl_rest;
    double tl_wait;   /* Phase 4/5: explicit residual wait after a submit */
    double tl_vram;   /* Phase 9: vram_cache_get (lookup + malloc + upload) */
    double tl_xup;    /* Phase 9: per-layer activation upload */
    double tl_sub;    /* Phase 9: the launch/submit call itself */
    char path[1024];
    char err[256];
} HostModel;

KATALI_API int  host_open(HostModel *m, const char *gguf_path, size_t ecache_bytes, int pin_frac);
KATALI_API void host_close(HostModel *m);
KATALI_API int  host_probe_expert(HostModel *m, int layer, int expert);
KATALI_API int  host_forward_token(HostModel *m, int token_id, float *logits_out);
KATALI_API int  host_generate(HostModel *m, const char *prompt, int max_tokens);
/* Port #5 layer-major batch prefill (also in forward.h). */
KATALI_API int  host_prefill_wanted(const HostModel *m, int n_prompt);
KATALI_API int  host_prefill_batch(HostModel *m, const int *ids, int n);
#ifdef __cplusplus
}
#endif
#endif
