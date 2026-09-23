#include "forward.h"
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
    st->norm_buf = (float *)xcalloc((size_t)(HD > kd ? HD : kd), sizeof(float));
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
    free(st->full_index); free(st->rec_index);
    memset(st, 0, sizeof(*st));
}

void host_state_reset(HostModel *m) {
    if (!m) return;
    HostState *st = &m->st;
    st->pos = 0;
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

    const int n_trunk = getenv("KATALI_PROBE_LM") ? 0 : (a->n_trunk > 0 ? a->n_trunk : a->n_layers);
    for (int L = 0; L < n_trunk; L++) {
        MoeLayerTensors *lt = &m->moe.layers[L];
        const float *nw = norm_w(lt->attn_norm, st->norm_buf, H);
        if (!nw) return KATALI_ERR;
        katali_gguf_rmsnorm(st->x, nw, (size_t)H, a->rms_eps, st->xb);

        int rc;
        if (a->layer_kind && a->layer_kind[L] == KLAYER_LINEAR_ATTN)
            rc = attn_delta_forward(m, L, st->xb, st->tmp);
        else
            rc = attn_gqa_forward(m, L, st->xb, st->tmp);
        if (rc != KATALI_OK) return rc;
        katali_gguf_add_inplace(st->x, st->tmp, (size_t)H);

        nw = norm_w(lt->post_attn_norm, st->norm_buf, H);
        if (!nw) return KATALI_ERR;
        katali_gguf_rmsnorm(st->x, nw, (size_t)H, a->rms_eps, st->xb);
        if (getenv("KATALI_SKIP_MOE")) {
            memset(st->tmp, 0, (size_t)H * sizeof(float));
        } else if (moe_ffn_forward(m, L, st->xb, st->tmp) != KATALI_OK) return KATALI_ERR;
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
    return KATALI_OK;
}
