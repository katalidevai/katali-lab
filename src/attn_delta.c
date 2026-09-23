#include "attn_delta.h"
#include "katali_gguf_dtype.h"
#include "katali_gguf_kernels.h"
#include "katali_gguf_qwen35.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const float *norm_w(const KataliGgufTensor *t, float *buf, int n) {
    if (!t || !t->data || n <= 0) return NULL;
    if (t->type == KGGML_F32) return (const float *)t->data;
    if (katali_ggml_dequant_ref(t->type, t->data, (uint64_t)n, buf) != 0) return NULL;
    return buf;
}

static float l2(const float *x, int n) {
    double s = 0.0;
    for (int i = 0; i < n; i++) s += (double)x[i] * (double)x[i];
    return (float)sqrt(s);
}

int attn_delta_forward(HostModel *m, int layer, const float *xb, float *y) {
    MoeLayerTensors *L = &m->moe.layers[layer];
    HostState *st = &m->st;
    const KataliArch *a = &m->arch;
    const int H = a->hidden;
    const int key_dim = a->ssm_head_dim;
    const int value_heads = a->ssm_value_heads;
    const int key_heads = a->ssm_groups;
    const int value_dim = a->ssm_head_dim;
    const int conv_channels = a->ssm_conv_channels;
    const int kernel = a->ssm_conv_kernel;

    if (!L->attn_qkv || !L->attn_gate || !L->ssm_alpha || !L->ssm_beta ||
        !L->ssm_conv1d || !L->ssm_a || !L->ssm_dt || !L->ssm_norm || !L->ssm_out)
        return KATALI_ERR;
    if (key_dim <= 0 || value_heads <= 0 || key_heads <= 0 || value_heads % key_heads != 0)
        return KATALI_ERR;
    if (L->ssm_conv1d->type != KGGML_F32 || L->ssm_a->type != KGGML_F32 || L->ssm_dt->type != KGGML_F32)
        return KATALI_ERR;
    int rec_i = st->rec_index[layer];
    if (rec_i < 0) return KATALI_ERR;

    katali_ggml_matvec(L->attn_qkv->type, L->attn_qkv->data, (uint64_t)conv_channels, (uint64_t)H, xb, st->qwen_qkv, 0, NULL, m->st.q4k_scratch, m->st.q4k_scratch_cap);
    katali_ggml_matvec(L->attn_gate->type, L->attn_gate->data, (uint64_t)(value_heads * value_dim), (uint64_t)H, xb, st->qwen_z, 0, NULL, m->st.q4k_scratch, m->st.q4k_scratch_cap);
    katali_ggml_matvec(L->ssm_alpha->type, L->ssm_alpha->data, (uint64_t)value_heads, (uint64_t)H, xb, st->qwen_alpha, 0, NULL, m->st.q4k_scratch, m->st.q4k_scratch_cap);
    katali_ggml_matvec(L->ssm_beta->type, L->ssm_beta->data, (uint64_t)value_heads, (uint64_t)H, xb, st->qwen_beta, 0, NULL, m->st.q4k_scratch, m->st.q4k_scratch_cap);

    float *history = st->conv_hist + (size_t)rec_i * st->conv_hist_stride;
    katali_qwen35_causal_conv1d(st->qwen_qkv, history, (const float *)L->ssm_conv1d->data,
                                st->qwen_conv, (size_t)conv_channels, (size_t)kernel);
    for (int i = 0; i < conv_channels; i++) st->qwen_conv[i] = katali_gguf_silu(st->qwen_conv[i]);

    float *q = st->qwen_conv;
    float *k = q + (size_t)key_heads * key_dim;
    float *v = k + (size_t)key_heads * key_dim;
    for (int h = 0; h < key_heads; h++) {
        float qn = 0.f, kn = 0.f;
        for (int i = 0; i < key_dim; i++) {
            qn += q[(size_t)h * key_dim + i] * q[(size_t)h * key_dim + i];
            kn += k[(size_t)h * key_dim + i] * k[(size_t)h * key_dim + i];
        }
        qn = 1.f / sqrtf(qn + 1e-6f); kn = 1.f / sqrtf(kn + 1e-6f);
        for (int i = 0; i < key_dim; i++) {
            q[(size_t)h * key_dim + i] *= qn;
            k[(size_t)h * key_dim + i] *= kn;
        }
    }

    float *decay = st->qwen_alpha, *beta = st->qwen_beta;
    /* Default: ssm_a is -exp(A_log) (GGUF dump). KATALI_SSM_ALOG=1 treats as A_log. */
    int a_is_log = getenv("KATALI_SSM_ALOG") != NULL;
    for (int h = 0; h < value_heads; h++) {
        float x = decay[h] + ((const float *)L->ssm_dt->data)[h];
        float sp = (x > 20.f) ? x : ((x > 0.f) ? x + log1pf(expf(-x)) : log1pf(expf(x)));
        float aa = ((const float *)L->ssm_a->data)[h];
        decay[h] = a_is_log ? expf(-expf(aa) * sp) : expf(aa * sp);
        beta[h] = 1.0f / (1.0f + expf(-beta[h]));
    }
    /* HF / llama fused GDN: scale q (equiv. scale output) by 1/sqrt(key_dim).
     * Disable with KATALI_NO_QSCALE=1. */
    if (!getenv("KATALI_NO_QSCALE")) {
        float inv = 1.f / sqrtf((float)key_dim);
        for (int h = 0; h < key_heads; h++)
            for (int i = 0; i < key_dim; i++)
                q[(size_t)h * key_dim + i] *= inv;
    }

    float *state = st->ssm_state + (size_t)rec_i * st->ssm_state_stride;
    katali_qwen35_delta_heads(state, q, k, v, decay, beta, st->qwen_recurrent,
                              (size_t)key_heads, (size_t)value_heads,
                              (size_t)key_dim, (size_t)value_dim);

    if (getenv("KATALI_DUMP_GDN") && (layer == 0 || layer == 1 || layer == 2)) {
        float dmin = decay[0], dmax = decay[0];
        double dsum = 0.0;
        for (int h = 0; h < value_heads; h++) {
            if (decay[h] < dmin) dmin = decay[h];
            if (decay[h] > dmax) dmax = decay[h];
            dsum += (double)decay[h];
        }
        float sa0 = ((const float *)L->ssm_a->data)[0];
        fprintf(stderr,
                "gdn L%d pos=%d conv_l2=%.4f out_l2=%.4f state_l2=%.4f "
                "decay[min/mean/max]=%.5f/%.5f/%.5f ssm_a[0]=%.6f flip=%d alog=%d interleave=%d\n",
                layer, st->pos,
                l2(st->qwen_conv, conv_channels),
                l2(st->qwen_recurrent, value_heads * value_dim),
                l2(state, value_heads * key_dim * value_dim),
                dmin, (float)(dsum / (double)value_heads), dmax,
                sa0,
                getenv("KATALI_CONV_FLIP") != NULL,
                a_is_log,
                getenv("KATALI_GDN_INTERLEAVE") != NULL);
    }

    const float *sn = norm_w(L->ssm_norm, st->norm_buf, value_dim);
    if (!sn) return KATALI_ERR;
    for (int h = 0; h < value_heads; h++) {
        float *o = st->qwen_recurrent + (size_t)h * value_dim;
        katali_gguf_rmsnorm_inplace(o, sn, (size_t)value_dim, a->rms_eps);
        for (int i = 0; i < value_dim; i++)
            o[i] *= katali_gguf_silu(st->qwen_z[(size_t)h * value_dim + i]);
    }
    katali_ggml_matvec(L->ssm_out->type, L->ssm_out->data, (uint64_t)H,
                       (uint64_t)(value_heads * value_dim), st->qwen_recurrent, y, 0, NULL, m->st.q4k_scratch, m->st.q4k_scratch_cap);
    return KATALI_OK;
}
