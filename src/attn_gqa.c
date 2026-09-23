#include "attn_gqa.h"
#include "katali_gguf_dtype.h"
#include "katali_gguf_kernels.h"
#include <math.h>
#include <string.h>

static const float *norm_w(const KataliGgufTensor *t, float *buf, int n) {
    if (!t || !t->data || n <= 0) return NULL;
    if (t->type == KGGML_F32) return (const float *)t->data;
    if (katali_ggml_dequant_ref(t->type, t->data, (uint64_t)n, buf) != 0) return NULL;
    return buf;
}

int attn_gqa_forward(HostModel *m, int layer, const float *xb, float *y) {
    MoeLayerTensors *L = &m->moe.layers[layer];
    HostState *st = &m->st;
    const KataliArch *a = &m->arch;
    if (!L->attn_q || !L->attn_k || !L->attn_v || !L->attn_o) return KATALI_ERR;
    const int H = a->hidden, NH = a->n_heads, NKV = a->n_kv_heads, HD = a->head_dim;
    const int QD = a->q_dim, QPD = a->q_proj_dim, KVD = a->kv_dim;
    const int pos = st->pos, max_seq = st->max_seq;
    if (pos < 0 || pos >= max_seq) return KATALI_ERR;
    int full_i = st->full_index[layer];
    if (full_i < 0) return KATALI_ERR;

    /* st->q is sized max(QPD, QD) so it can hold the fused Q+gate projection. */
    katali_ggml_matvec(L->attn_q->type, L->attn_q->data, (uint64_t)QPD, (uint64_t)H, xb, st->q, 0, NULL, m->st.q4k_scratch, m->st.q4k_scratch_cap);
    katali_ggml_matvec(L->attn_k->type, L->attn_k->data, (uint64_t)KVD, (uint64_t)H, xb, st->k, 0, NULL, m->st.q4k_scratch, m->st.q4k_scratch_cap);
    katali_ggml_matvec(L->attn_v->type, L->attn_v->data, (uint64_t)KVD, (uint64_t)H, xb, st->v, 0, NULL, m->st.q4k_scratch, m->st.q4k_scratch_cap);

    /* Unpack interleaved [q|gate] per head without in-place corruption:
     * packing high heads first would overwrite still-needed low-head sources. */
    for (int h = 0; h < NH; h++) {
        size_t src = (size_t)h * (size_t)HD * 2;
        memcpy(st->att_out + (size_t)h * HD, st->q + src, (size_t)HD * sizeof(float));
        memcpy(st->q_gate + (size_t)h * HD, st->q + src + HD, (size_t)HD * sizeof(float));
    }
    memcpy(st->q, st->att_out, (size_t)QD * sizeof(float));

    const float *qn = norm_w(L->attn_q_norm, st->norm_buf, HD);
    if (qn) for (int h = 0; h < NH; h++)
        katali_gguf_rmsnorm_inplace(st->q + (size_t)h * HD, qn, (size_t)HD, a->rms_eps);
    const float *kn = norm_w(L->attn_k_norm, st->norm_buf, HD);
    if (kn) for (int h = 0; h < NKV; h++)
        katali_gguf_rmsnorm_inplace(st->k + (size_t)h * HD, kn, (size_t)HD, a->rms_eps);

    int rd = a->rope_dim > 0 ? a->rope_dim : HD;
    if (rd > HD) rd = HD;
    int half = rd / 2;
    for (int i = 0; i < half; i++) {
        float inv = 1.f / powf(a->rope_theta, (2.f * (float)i) / (float)rd);
        float freq = (float)pos * inv;
        st->rope_cos[i] = cosf(freq);
        st->rope_sin[i] = sinf(freq);
    }
    katali_gguf_rope_apply(st->q, (size_t)NH, (size_t)HD, (size_t)rd, st->rope_cos, st->rope_sin);
    katali_gguf_rope_apply(st->k, (size_t)NKV, (size_t)HD, (size_t)rd, st->rope_cos, st->rope_sin);

    float *lk = st->kv_k + (size_t)full_i * st->kv_layer_stride;
    float *lv = st->kv_v + (size_t)full_i * st->kv_layer_stride;
    for (int h = 0; h < NKV; h++) {
        memcpy(lk + ((size_t)h * max_seq + (size_t)pos) * HD, st->k + (size_t)h * HD, (size_t)HD * sizeof(float));
        memcpy(lv + ((size_t)h * max_seq + (size_t)pos) * HD, st->v + (size_t)h * HD, (size_t)HD * sizeof(float));
    }

    katali_gguf_gqa_decode_scratch(st->q, lk, lv, (size_t)pos + 1, (size_t)max_seq,
                                   (size_t)NH, (size_t)NKV, (size_t)HD,
                                   st->att_out, st->attn_scores);
    for (int i = 0; i < QD; i++) st->att_out[i] *= katali_gguf_sigmoid(st->q_gate[i]);
    katali_ggml_matvec(L->attn_o->type, L->attn_o->data, (uint64_t)H, (uint64_t)QD, st->att_out, y, 0, NULL, m->st.q4k_scratch, m->st.q4k_scratch_cap);
    return KATALI_OK;
}
