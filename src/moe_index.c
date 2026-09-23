#include "moe_index.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const KataliGgufTensor *T(const KataliGgufFile *f, char *buf, size_t bufn,
                                int layer, const char *suffix) {
    snprintf(buf, bufn, "blk.%d.%s", layer, suffix);
    return katali_gguf_find_tensor(f, buf);
}

int moe_index_build(MoeIndex *idx, const KataliGgufFile *f, const KataliArch *arch) {
    memset(idx, 0, sizeof(*idx));
    if (!f || !f->ok || !arch) return KATALI_ERR;
    idx->n_layers = arch->n_layers;
    idx->n_experts = arch->n_experts;
    idx->layers = (MoeLayerTensors *)calloc((size_t)idx->n_layers, sizeof(MoeLayerTensors));
    if (!idx->layers) return KATALI_ERR_NOMEM;
    char name[160];
    for (int i = 0; i < idx->n_layers; i++) {
        MoeLayerTensors *L = &idx->layers[i];
        L->layer = i;
        L->kind = arch->layer_kind ? arch->layer_kind[i] : KLAYER_FULL_ATTN;

        L->gate_exps = T(f, name, sizeof(name), i, "ffn_gate_exps.weight");
        L->up_exps   = T(f, name, sizeof(name), i, "ffn_up_exps.weight");
        L->down_exps = T(f, name, sizeof(name), i, "ffn_down_exps.weight");
        L->gate_inp  = T(f, name, sizeof(name), i, "ffn_gate_inp.weight");
        L->gate_inp_shexp = T(f, name, sizeof(name), i, "ffn_gate_inp_shexp.weight");
        L->gate_shexp = T(f, name, sizeof(name), i, "ffn_gate_shexp.weight");
        L->up_shexp   = T(f, name, sizeof(name), i, "ffn_up_shexp.weight");
        L->down_shexp = T(f, name, sizeof(name), i, "ffn_down_shexp.weight");

        L->attn_norm = T(f, name, sizeof(name), i, "attn_norm.weight");
        L->post_attn_norm = T(f, name, sizeof(name), i, "post_attention_norm.weight");
        if (!L->post_attn_norm)
            L->post_attn_norm = T(f, name, sizeof(name), i, "ffn_norm.weight");

        if (L->kind == KLAYER_FULL_ATTN) {
            L->attn_q = T(f, name, sizeof(name), i, "attn_q.weight");
            L->attn_k = T(f, name, sizeof(name), i, "attn_k.weight");
            L->attn_v = T(f, name, sizeof(name), i, "attn_v.weight");
            L->attn_o = T(f, name, sizeof(name), i, "attn_output.weight");
            L->attn_q_norm = T(f, name, sizeof(name), i, "attn_q_norm.weight");
            L->attn_k_norm = T(f, name, sizeof(name), i, "attn_k_norm.weight");
        } else {
            L->attn_qkv = T(f, name, sizeof(name), i, "attn_qkv.weight");
            L->attn_gate = T(f, name, sizeof(name), i, "attn_gate.weight");
            L->ssm_conv1d = T(f, name, sizeof(name), i, "ssm_conv1d.weight");
            L->ssm_dt = T(f, name, sizeof(name), i, "ssm_dt.bias");
            L->ssm_a = T(f, name, sizeof(name), i, "ssm_a");
            L->ssm_alpha = T(f, name, sizeof(name), i, "ssm_alpha.weight");
            L->ssm_beta = T(f, name, sizeof(name), i, "ssm_beta.weight");
            L->ssm_norm = T(f, name, sizeof(name), i, "ssm_norm.weight");
            L->ssm_out = T(f, name, sizeof(name), i, "ssm_out.weight");
        }

        if (L->gate_exps && idx->n_experts > 0)
            L->expert_bytes_gate = L->gate_exps->nbytes / (uint64_t)idx->n_experts;
        if (L->up_exps && idx->n_experts > 0)
            L->expert_bytes_up = L->up_exps->nbytes / (uint64_t)idx->n_experts;
        if (L->down_exps && idx->n_experts > 0)
            L->expert_bytes_down = L->down_exps->nbytes / (uint64_t)idx->n_experts;
    }
    idx->tok_embd = katali_gguf_find_tensor(f, "token_embd.weight");
    idx->output = katali_gguf_find_tensor(f, "output.weight");
    idx->output_norm = katali_gguf_find_tensor(f, "output_norm.weight");
    return KATALI_OK;
}

void moe_index_free(MoeIndex *idx) {
    if (!idx) return;
    free(idx->layers);
    memset(idx, 0, sizeof(*idx));
}

int moe_expert_span(const MoeLayerTensors *L, int which, int expert,
                    uint64_t *rel_off, uint64_t *len) {
    if (!L || expert < 0) return KATALI_ERR;
    const KataliGgufTensor *t = NULL;
    uint64_t eb = 0;
    if (which == 0) { t = L->gate_exps; eb = L->expert_bytes_gate; }
    else if (which == 1) { t = L->up_exps; eb = L->expert_bytes_up; }
    else if (which == 2) { t = L->down_exps; eb = L->expert_bytes_down; }
    else return KATALI_ERR;
    if (!t || !t->data || eb == 0) return KATALI_ERR;
    *len = eb;
    *rel_off = (uint64_t)expert * eb;
    if (*rel_off + *len > t->nbytes) return KATALI_ERR;
    return KATALI_OK;
}
