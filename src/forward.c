#include "forward.h"
#include "platform.h"
#include "attn_gqa.h"
#include "attn_delta.h"
#include "moe_ffn.h"
#include "katali_gguf_dtype.h"
#include "katali_gguf_kernels.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdlib.h>

static void *xcalloc(size_t n, size_t sz) {
    void *p = calloc(n ? n : 1, sz);
    return p;
}

int host_state_alloc(HostModel *m, int max_seq) {
    if (!m) return KATALI_ERR;
    host_state_free(m);
    HostState *st = &m->st;
    const KataliArch *a = &m->arch;
    if (max_seq <= 0) max_seq = 2048;
    if (max_seq > a->ctx_train && a->ctx_train > 0) max_seq = a->ctx_train;
    st->max_seq = max_seq;
    st->n_full = a->n_full_layers;
    st->n_recurrent = a->n_recurrent;
    st->pos = 0;

    const int H = a->hidden;
    const int NH = a->n_heads, NKV = a->n_kv_heads, HD = a->head_dim;
    const int QD = a->q_dim > 0 ? a->q_dim : NH * HD;
    const int QPD = a->q_proj_dim > 0 ? a->q_proj_dim : QD * 2;
    const int KVD = a->kv_dim > 0 ? a->kv_dim : NKV * HD;
    const int V = a->vocab > 0 ? a->vocab : 1;
    const int rd = (a->rope_dim > 0 ? a->rope_dim : HD) / 2;
    const int vh = a->ssm_value_heads;
    const int kd = a->ssm_head_dim;
    const int cc = a->ssm_conv_channels;
    const int ksz = a->ssm_conv_kernel > 1 ? a->ssm_conv_kernel - 1 : 1;

    st->full_index = (int *)xcalloc((size_t)a->n_layers, sizeof(int));
    st->rec_index = (int *)xcalloc((size_t)a->n_layers, sizeof(int));
    if (!st->full_index || !st->rec_index) return KATALI_ERR_NOMEM;
    int fi = 0, ri = 0;
    for (int i = 0; i < a->n_layers; i++) {
        st->full_index[i] = -1;
        st->rec_index[i] = -1;
        if (i >= a->n_trunk) continue;
        if (a->layer_kind && a->layer_kind[i] == KLAYER_FULL_ATTN)
            st->full_index[i] = fi++;
        else
            st->rec_index[i] = ri++;
    }
    st->n_full = fi;
    st->n_recurrent = ri;

    st->kv_layer_stride = (size_t)NKV * (size_t)max_seq * (size_t)HD;
    st->ssm_state_stride = (size_t)vh * (size_t)kd * (size_t)kd;
    st->conv_hist_stride = (size_t)cc * (size_t)ksz;

    st->x = (float *)xcalloc((size_t)H, sizeof(float));
    st->xb = (float *)xcalloc((size_t)H, sizeof(float));
    st->xb2 = (float *)xcalloc((size_t)H, sizeof(float));
    st->q = (float *)xcalloc((size_t)(QPD > QD ? QPD : QD), sizeof(float));
    st->k = (float *)xcalloc((size_t)KVD, sizeof(float));
    st->v = (float *)xcalloc((size_t)KVD, sizeof(float));
    st->att_out = (float *)xcalloc((size_t)QD, sizeof(float));
    st->gate = (float *)xcalloc((size_t)QD, sizeof(float));
    st->q_gate = (float *)xcalloc((size_t)QD, sizeof(float));
    st->logits = (float *)xcalloc((size_t)V, sizeof(float));
    st->attn_scores = (float *)xcalloc((size_t)max_seq, sizeof(float));
    st->rope_cos = (float *)xcalloc((size_t)(rd > 0 ? rd : 1), sizeof(float));
    st->rope_sin = (float *)xcalloc((size_t)(rd > 0 ? rd : 1), sizeof(float));
    st->norm_buf = (float *)xcalloc((size_t)(H > HD ? H : HD), sizeof(float));
    st->q4k_scratch_cap = H;
    st->q4k_scratch = (float *)xcalloc((size_t)st->q4k_scratch_cap, sizeof(float));
    st->tmp = (float *)xcalloc((size_t)H, sizeof(float));

    if (st->n_full > 0) {
        size_t kv_n = (size_t)st->n_full * st->kv_layer_stride;
        st->kv_k = (float *)xcalloc(kv_n, sizeof(float));
        st->kv_v = (float *)xcalloc(kv_n, sizeof(float));
    }
    if (st->n_recurrent > 0) {
        st->ssm_state = (float *)xcalloc((size_t)st->n_recurrent * st->ssm_state_stride, sizeof(float));
        st->conv_hist = (float *)xcalloc((size_t)st->n_recurrent * st->conv_hist_stride, sizeof(float));
        st->qwen_qkv = (float *)xcalloc((size_t)cc, sizeof(float));
        st->qwen_z = (float *)xcalloc((size_t)vh * (size_t)kd, sizeof(float));
        st->qwen_alpha = (float *)xcalloc((size_t)vh, sizeof(float));
        st->qwen_beta = (float *)xcalloc((size_t)vh, sizeof(float));
        st->qwen_conv = (float *)xcalloc((size_t)cc, sizeof(float));
        st->qwen_recurrent = (float *)xcalloc((size_t)vh * (size_t)kd, sizeof(float));
    }

    if (!st->x || !st->xb || !st->xb2 || !st->q || !st->k || !st->v ||
        !st->att_out || !st->gate || !st->q_gate || !st->logits ||
        !st->attn_scores || !st->rope_cos || !st->rope_sin || !st->norm_buf || !st->tmp ||
        (st->n_full > 0 && (!st->kv_k || !st->kv_v)) ||
        (st->n_recurrent > 0 && (!st->ssm_state || !st->conv_hist || !st->qwen_qkv ||
         !st->qwen_z || !st->qwen_alpha || !st->qwen_beta || !st->qwen_conv ||
         !st->qwen_recurrent))) {
        host_state_free(m);
        return KATALI_ERR_NOMEM;
    }
    return KATALI_OK;
}

void host_state_free(HostModel *m) {
    if (!m) return;
    HostState *st = &m->st;
    free(st->x); free(st->xb); free(st->xb2);
    free(st->q); free(st->k); free(st->v);
    free(st->att_out); free(st->gate); free(st->q_gate);
    free(st->logits); free(st->kv_k); free(st->kv_v);
    free(st->attn_scores); free(st->rope_cos); free(st->rope_sin);
    free(st->ssm_state); free(st->conv_hist);
    free(st->qwen_qkv); free(st->qwen_z); free(st->qwen_alpha); free(st->qwen_beta);
    free(st->qwen_conv); free(st->qwen_recurrent); free(st->tmp); free(st->norm_buf);
    free(st->q4k_scratch);
    free(st->full_index); free(st->rec_index);
    memset(st, 0, sizeof(*st));
}

void host_state_reset(HostModel *m) {
    if (!m) return;
    HostState *st = &m->st;
    st->pos = 0;
    m->last_routes_valid = 0;
    if (st->kv_k && st->n_full > 0)
        memset(st->kv_k, 0, (size_t)st->n_full * st->kv_layer_stride * sizeof(float));
    if (st->kv_v && st->n_full > 0)
        memset(st->kv_v, 0, (size_t)st->n_full * st->kv_layer_stride * sizeof(float));
    if (st->ssm_state && st->n_recurrent > 0)
        memset(st->ssm_state, 0, (size_t)st->n_recurrent * st->ssm_state_stride * sizeof(float));
    if (st->conv_hist && st->n_recurrent > 0)
        memset(st->conv_hist, 0, (size_t)st->n_recurrent * st->conv_hist_stride * sizeof(float));
}

static const float *norm_w(const KataliGgufTensor *t, float *buf, int n) {
    if (!t || !t->data || n <= 0) return NULL;
    if (t->type == KGGML_F32) return (const float *)t->data;
    if (katali_ggml_dequant_ref(t->type, t->data, (uint64_t)n, buf) != 0) return NULL;
    return buf;
}


/* Prefetch 3 weight keys for each prior-token expert id at layer. */
static void prefetch_layer_routes(HostModel *m, int layer) {
    if (!m || !m->last_routes_valid || !m->last_eids || m->last_topk <= 0) return;
    if (getenv("KATALI_NO_ECACHE")) return;
    if (layer < 0 || layer >= m->moe.n_layers) return;
    int *slot = m->last_eids + (size_t)layer * (size_t)m->last_topk;
    for (int t = 0; t < m->last_topk; t++) {
        int eid = slot[t];
        if (eid < 0) continue;
        for (int w = 0; w < 3; w++)
            ecache_prefetch(&m->ecache, layer, eid * 3 + w);
    }
}

int host_forward_token(HostModel *m, int token_id, float *logits_out) {
    if (!m || !m->opened) return KATALI_ERR;
    HostState *st = &m->st;
    const KataliArch *a = &m->arch;
    if (!st->x || st->max_seq <= 0) return KATALI_ERR;
    if (st->pos >= st->max_seq) return KATALI_ERR;
    if (!m->moe.tok_embd || !m->moe.tok_embd->data) return KATALI_ERR;
    if (token_id < 0 || (a->vocab > 0 && token_id >= a->vocab)) return KATALI_ERR;

    const int H = a->hidden;
    uint64_t row_b = katali_ggml_row_bytes(m->moe.tok_embd->type, (uint64_t)H);
    if (row_b == 0) return KATALI_ERR;
    const uint8_t *row = m->moe.tok_embd->data + (size_t)token_id * (size_t)row_b;
    if (katali_ggml_dequant_ref(m->moe.tok_embd->type, row, (uint64_t)H, st->x) != 0)
        return KATALI_ERR;

    const int n_trunk = a->n_trunk > 0 ? a->n_trunk : a->n_layers;
    int max_layer = n_trunk;
    {
        const char *ml = getenv("KATALI_MAX_LAYER");
        if (ml && ml[0]) {
            int v = atoi(ml);
            if (v >= 0 && v < max_layer) max_layer = v;
        }
    }
    for (int L = 0; L < max_layer; L++) {
        MoeLayerTensors *lt = &m->moe.layers[L];
        /* Overlap SSD with attn: speculative prefetch this + next layer from prior token. */
        /* Decode-only hot-set mode keeps prompt routing from feeding the
         * speculative prior-token prefetch policy. Batch ensure_many still
         * loads exactly the experts needed by this prompt. */
        if (!getenv("KATALI_EC_DECODE_ONLY")) {
            prefetch_layer_routes(m, L);
            if (L + 1 < max_layer) prefetch_layer_routes(m, L + 1);
        }
        const float *nw = norm_w(lt->attn_norm, st->norm_buf, H);
        if (!nw) return KATALI_ERR;
        katali_gguf_rmsnorm(st->x, nw, (size_t)H, a->rms_eps, st->xb);

        int rc;
        int skip_delta = getenv("KATALI_SKIP_DELTA") != NULL;
        int skip_gqa = getenv("KATALI_SKIP_GQA") != NULL;
        int skip_moe = getenv("KATALI_SKIP_MOE") != NULL;
        /* Phase 1: attention is the biggest per-layer CPU block, so its span is
         * measured directly rather than inferred. */
        const double tl_attn_t0 = katali_time_s();
        if (a->layer_kind && a->layer_kind[L] == KLAYER_LINEAR_ATTN) {
            if (skip_delta) { memset(st->tmp, 0, (size_t)H * sizeof(float)); rc = KATALI_OK; }
            else rc = attn_delta_forward(m, L, st->xb, st->tmp);
        } else {
            if (skip_gqa) { memset(st->tmp, 0, (size_t)H * sizeof(float)); rc = KATALI_OK; }
            else rc = attn_gqa_forward(m, L, st->xb, st->tmp);
        }
        m->tl_attn += katali_time_s() - tl_attn_t0;
        if (rc != KATALI_OK) return rc;
        katali_gguf_add_inplace(st->x, st->tmp, (size_t)H);

        nw = norm_w(lt->post_attn_norm, st->norm_buf, H);
        if (!nw) return KATALI_ERR;
        katali_gguf_rmsnorm(st->x, nw, (size_t)H, a->rms_eps, st->xb);
        if (skip_moe) {
            memset(st->tmp, 0, (size_t)H * sizeof(float));
        } else if (moe_ffn_forward(m, L, st->xb, st->tmp) != KATALI_OK) {
            return KATALI_ERR;
        }
        katali_gguf_add_inplace(st->x, st->tmp, (size_t)H);
    }

    const float *on = norm_w(m->moe.output_norm, st->norm_buf, H);
    if (!on) return KATALI_ERR;
    katali_gguf_rmsnorm(st->x, on, (size_t)H, a->rms_eps, st->xb);

    const KataliGgufTensor *out_w = m->moe.output ? m->moe.output : m->moe.tok_embd;
    if (!out_w || !out_w->data) return KATALI_ERR;
    int vocab = a->vocab > 0 ? a->vocab : (int)out_w->dims[1];
    katali_ggml_matvec(out_w->type, out_w->data, (uint64_t)vocab, (uint64_t)H,
                       st->xb, st->logits, 0, NULL, NULL, 0);

    if (logits_out) memcpy(logits_out, st->logits, (size_t)vocab * sizeof(float));
    st->pos++;

    /* Soft-pin prior-token routes (katali2 host_qwen STREAM=0 + PIN_FRAC budget).
     * GGUF: each expert expands to 3 weight keys (gate/up/down). */
    if (m->last_eids && m->last_topk > 0) {
        m->last_routes_valid = 1;
        if (!getenv("KATALI_NO_ECACHE")) {
            ecache_unpin_all(&m->ecache);
            if (ecache_pins_enabled()) {
                int nL = max_layer;
                int lim = ecache_spec_pin_limit(&m->ecache);
                int per = (nL > 0) ? (lim / nL) : lim;
                if (per < 1 && lim > 0) per = 1;
                if (per > m->last_topk) per = m->last_topk;
                int left = lim;
                for (int L = 0; L < nL && left > 0; L++) {
                    int *slot = m->last_eids + (size_t)L * (size_t)m->last_topk;
                    int n = per < left ? per : left;
                    for (int t = 0; t < n; t++) {
                        int eid = slot[t];
                        if (eid < 0) continue;
                        for (int w = 0; w < 3; w++)
                            ecache_pin(&m->ecache, L, eid * 3 + w);
                    }
                    left -= n;
                }
            }
        }
    }
    return KATALI_OK;
}


/* ---- Port #5: layer-major batch prefill (katali2 host_qwen_prefill_batch) ----
 * Walk layer -> positions. Attention stays position-sequential (GQA/delta KV).
 * MoE: route all positions (dense), UNION unique experts for the layer, one
 * ecache_ensure_many (workers queue fills), then MLP each position via get.
 * Decode path (host_forward_token) unchanged. Default ON; KATALI_PREFILL_BATCH=0
 * forces per-token. Do not invent READY/PIPE/STREAM. */

#define LAB_PREFILL_MAX_STATE (256u * 1024u * 1024u)

int host_prefill_wanted(const HostModel *m, int n_prompt) {
    const char *env;
    if (!m || !m->opened) return 0;
    if (n_prompt < 2) return 0;
    if (getenv("KATALI_NO_ECACHE")) return 0;
    if (getenv("KATALI_SKIP_MOE")) return 0;
    if (m->ecache.n_workers <= 0) return 0;
    if (!m->ecache.default_fill || !m->ecache.default_fill_ud) return 0;
    if (m->arch.n_experts_active <= 0 || m->arch.n_experts <= 0) return 0;
    env = getenv("KATALI_PREFILL_BATCH");
    /* Default ON: only explicit 0 disables (match current katali2 Qwen). */
    if (env && atoi(env) == 0) return 0;
    return 1;
}

static int batch_final_logits(HostModel *m) {
    HostState *st = &m->st;
    const KataliArch *a = &m->arch;
    const int H = a->hidden;
    const float *on = norm_w(m->moe.output_norm, st->norm_buf, H);
    if (!on) return KATALI_ERR;
    katali_gguf_rmsnorm(st->x, on, (size_t)H, a->rms_eps, st->xb);
    const KataliGgufTensor *out_w = m->moe.output ? m->moe.output : m->moe.tok_embd;
    if (!out_w || !out_w->data) return KATALI_ERR;
    int vocab = a->vocab > 0 ? a->vocab : (int)out_w->dims[1];
    katali_ggml_matvec(out_w->type, out_w->data, (uint64_t)vocab, (uint64_t)H,
                       st->xb, st->logits, 0, NULL, NULL, 0);
    return KATALI_OK;
}

/* Returns 0 = done, 1 = declined, -1 = hard fail. */
int host_prefill_batch(HostModel *m, const int *ids, int n) {
    if (!m || !ids || n < 2) return 1;
    if (!host_prefill_wanted(m, n)) return 1;

    HostState *st = &m->st;
    const KataliArch *a = &m->arch;
    const int H = a->hidden;
    const int K = a->n_experts_active > 0 ? a->n_experts_active : 8;
    if (K <= 0 || K > 64) return 1;
    if (!m->moe.tok_embd || !m->moe.tok_embd->data) return 1;
    if (!st->x || st->max_seq <= 0) return 1;
    if (n >= st->max_seq) n = st->max_seq - 1;
    if (n < 2) return 1;

    const int n_trunk = a->n_trunk > 0 ? a->n_trunk : a->n_layers;
    int max_layer = n_trunk;
    {
        const char *ml = getenv("KATALI_MAX_LAYER");
        if (ml && ml[0]) {
            int v = atoi(ml);
            if (v >= 0 && v < max_layer) max_layer = v;
        }
    }
    if (max_layer <= 0) return 1;

    size_t bytes = (size_t)n * (size_t)H * 2u * sizeof(float);
    if (bytes > LAB_PREFILL_MAX_STATE) return 1;

    float *res_st = (float *)malloc((size_t)n * (size_t)H * sizeof(float));
    float *moe_in = (float *)malloc((size_t)n * (size_t)H * sizeof(float));
    int *rid = (int *)malloc((size_t)n * (size_t)K * sizeof(int));
    float *rwt = (float *)malloc((size_t)n * (size_t)K * sizeof(float));
    /* Union scratch: at most n*K unique experts. */
    int *uniq = (int *)malloc((size_t)n * (size_t)K * sizeof(int));
    int *seen = (int *)calloc((size_t)(a->n_experts > 0 ? a->n_experts : 1), sizeof(int));
    if (!res_st || !moe_in || !rid || !rwt || !uniq || !seen) {
        free(res_st); free(moe_in); free(rid); free(rwt); free(uniq); free(seen);
        return 1;
    }

    uint64_t row_b = katali_ggml_row_bytes(m->moe.tok_embd->type, (uint64_t)H);
    if (row_b == 0) {
        free(res_st); free(moe_in); free(rid); free(rwt); free(uniq); free(seen);
        return -1;
    }

    int skip_delta = getenv("KATALI_SKIP_DELTA") != NULL;
    int skip_gqa = getenv("KATALI_SKIP_GQA") != NULL;
    int verbose = getenv("KATALI_DEBUG") != NULL || getenv("KATALI_PREFILL_VERBOSE") != NULL;

    /* Embed all prompt tokens into residual stream slots. */
    for (int p = 0; p < n; p++) {
        int tid = ids[p];
        if (tid < 0 || (a->vocab > 0 && tid >= a->vocab)) tid = 0;
        const uint8_t *row = m->moe.tok_embd->data + (size_t)tid * (size_t)row_b;
        if (katali_ggml_dequant_ref(m->moe.tok_embd->type, row, (uint64_t)H,
                                   res_st + (size_t)p * (size_t)H) != 0) {
            free(res_st); free(moe_in); free(rid); free(rwt); free(uniq); free(seen);
            return -1;
        }
    }

    fprintf(stderr, "phase: prefill layer-major batch tokens=%d layers=%d topk=%d\n",
            n, max_layer, K);
    fflush(stderr);

    for (int L = 0; L < max_layer; L++) {
        MoeLayerTensors *lt = &m->moe.layers[L];
        if (verbose && (L == 0 || (L & 15) == 15 || L + 1 == max_layer)) {
            fprintf(stderr, "host: prefill-batch layer=%d/%d\n", L, max_layer);
            fflush(stderr);
        }

        /* Speculative prefetch from prior-token routes (same as per-token). */
        prefetch_layer_routes(m, L);
        if (L + 1 < max_layer) prefetch_layer_routes(m, L + 1);

        /* ---- attention + post-attn norm, position-sequential ---- */
        for (int p = 0; p < n; p++) {
            float *resp = res_st + (size_t)p * (size_t)H;
            float *mip = moe_in + (size_t)p * (size_t)H;
            memcpy(st->x, resp, (size_t)H * sizeof(float));
            st->pos = p; /* attn_gqa / delta read st->pos */

            const float *nw = norm_w(lt->attn_norm, st->norm_buf, H);
            if (!nw) goto fail;
            katali_gguf_rmsnorm(st->x, nw, (size_t)H, a->rms_eps, st->xb);

            int rc;
            if (a->layer_kind && a->layer_kind[L] == KLAYER_LINEAR_ATTN) {
                if (skip_delta) { memset(st->tmp, 0, (size_t)H * sizeof(float)); rc = KATALI_OK; }
                else rc = attn_delta_forward(m, L, st->xb, st->tmp);
            } else {
                if (skip_gqa) { memset(st->tmp, 0, (size_t)H * sizeof(float)); rc = KATALI_OK; }
                else rc = attn_gqa_forward(m, L, st->xb, st->tmp);
            }
            if (rc != KATALI_OK) goto fail;
            katali_gguf_add_inplace(st->x, st->tmp, (size_t)H);

            nw = norm_w(lt->post_attn_norm, st->norm_buf, H);
            if (!nw) goto fail;
            katali_gguf_rmsnorm(st->x, nw, (size_t)H, a->rms_eps, mip);
            memcpy(resp, st->x, (size_t)H * sizeof(float)); /* post-attn residual */
        }

        /* ---- MoE: route all positions, UNION experts, ensure_many, MLP ---- */
        for (int p = 0; p < n; p++) {
            float *mip = moe_in + (size_t)p * (size_t)H;
            if (moe_ffn_route(m, L, mip, rid + (size_t)p * (size_t)K,
                              rwt + (size_t)p * (size_t)K) != KATALI_OK)
                goto fail;
        }

        /* Build unique expert set for this layer (prompt-union). */
        int n_uniq = 0;
        if (seen) memset(seen, 0, (size_t)a->n_experts * sizeof(int));
        for (int p = 0; p < n; p++) {
            for (int t = 0; t < K; t++) {
                int e = rid[(size_t)p * (size_t)K + (size_t)t];
                if (e < 0 || e >= a->n_experts) continue;
                if (seen[e]) continue;
                seen[e] = 1;
                uniq[n_uniq++] = e;
            }
        }

        if (n_uniq > 0 && !getenv("KATALI_NO_ECACHE") &&
            m->ecache.default_fill && m->ecache.default_fill_ud) {
            int nkeys = n_uniq * 3;
            int *layers_arr = (int *)malloc((size_t)nkeys * sizeof(int));
            int *keys_arr = (int *)malloc((size_t)nkeys * sizeof(int));
            if (!layers_arr || !keys_arr) {
                free(layers_arr); free(keys_arr);
                goto fail;
            }
            for (int i = 0; i < n_uniq; i++) {
                int eid = uniq[i];
                for (int w = 0; w < 3; w++) {
                    int idx = i * 3 + w;
                    layers_arr[idx] = L;
                    keys_arr[idx] = eid * 3 + w;
                    ecache_pin(&m->ecache, L, keys_arr[idx]);
                }
            }
            (void)ecache_ensure_many(&m->ecache, layers_arr, keys_arr, nkeys,
                                     m->ecache.default_fill, m->ecache.default_fill_ud);
            free(layers_arr);
            free(keys_arr);
            if (verbose) {
                fprintf(stderr, "host: prefill-batch L%d union_experts=%d keys=%d\n",
                        L, n_uniq, nkeys);
                fflush(stderr);
            }
        }

        for (int p = 0; p < n; p++) {
            float *resp = res_st + (size_t)p * (size_t)H;
            float *mip = moe_in + (size_t)p * (size_t)H;
            if (moe_ffn_mlp_routed(m, L, mip,
                                  rid + (size_t)p * (size_t)K,
                                  rwt + (size_t)p * (size_t)K,
                                  st->tmp) != KATALI_OK)
                goto fail;
            memcpy(st->x, resp, (size_t)H * sizeof(float));
            katali_gguf_add_inplace(st->x, st->tmp, (size_t)H);
            memcpy(resp, st->x, (size_t)H * sizeof(float));
        }

        /* Release pin window except last position's routes (decode warm). */
        for (int p = 0; p < n - 1; p++) {
            for (int t = 0; t < K; t++) {
                int e = rid[(size_t)p * (size_t)K + (size_t)t];
                int keep = 0;
                for (int u = 0; u < K; u++) {
                    if (e == rid[(size_t)(n - 1) * (size_t)K + (size_t)u]) {
                        keep = 1;
                        break;
                    }
                }
                if (!keep) {
                    for (int w = 0; w < 3; w++)
                        ecache_unpin(&m->ecache, L, e * 3 + w);
                }
            }
        }
    }

    /* Last residual -> st->x; compute logits once (katali2 host_final_logits). */
    memcpy(st->x, res_st + (size_t)(n - 1) * (size_t)H, (size_t)H * sizeof(float));
    if (batch_final_logits(m) != KATALI_OK) goto fail;
    st->pos = n;

    /* Soft-pin last-position routes for decode (same as host_qwen_prefill_batch).
     * Decay LFU first so prompt-union experts don't steal decode slots. */
    if (m->last_eids && m->last_topk > 0 && !getenv("KATALI_NO_ECACHE")) {
        m->last_routes_valid = 1;
        if (!getenv("KATALI_EC_DECODE_ONLY"))
            ecache_freq_decay(&m->ecache, 4);
        ecache_unpin_all(&m->ecache);
        if (ecache_pins_enabled()) {
            int lim = ecache_spec_pin_limit(&m->ecache);
            int per = (max_layer > 0) ? (lim / max_layer) : lim;
            if (per < 1 && lim > 0) per = 1;
            if (per > m->last_topk) per = m->last_topk;
            int left = lim;
            for (int Li = 0; Li < max_layer && left > 0; Li++) {
                int *slot = m->last_eids + (size_t)Li * (size_t)m->last_topk;
                int n_pin = per < left ? per : left;
                for (int t = 0; t < n_pin; t++) {
                    int eid = slot[t];
                    if (eid < 0) continue;
                    for (int w = 0; w < 3; w++)
                        ecache_pin(&m->ecache, Li, eid * 3 + w);
                }
                left -= n_pin;
            }
            ecache_drop_unpinned(&m->ecache);
            fprintf(stderr,
                    "phase: ecache post-prefill drop_unpinned lim=%d per_layer=%d "
                    "pin_frac=%d (batch)\n",
                    lim, per, m->ecache.pin_frac);
        }
    }

    free(res_st); free(moe_in); free(rid); free(rwt); free(uniq); free(seen);
    return 0;

fail:
    free(res_st); free(moe_in); free(rid); free(rwt); free(uniq); free(seen);
    return -1;
}
