#ifndef KATALI_MOE_INDEX_H
#define KATALI_MOE_INDEX_H
#include "katali_gguf.h"
#include "model.h"
#include "katali.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct MoeLayerTensors {
    int layer;
    KataliLayerKind kind;
    const KataliGgufTensor *gate_exps, *up_exps, *down_exps;
    const KataliGgufTensor *gate_inp, *gate_inp_shexp;
    const KataliGgufTensor *gate_shexp, *up_shexp, *down_shexp;
    const KataliGgufTensor *attn_norm, *post_attn_norm;
    const KataliGgufTensor *attn_q, *attn_k, *attn_v, *attn_o;
    const KataliGgufTensor *attn_q_norm, *attn_k_norm;
    const KataliGgufTensor *attn_qkv, *attn_gate;
    const KataliGgufTensor *ssm_conv1d, *ssm_dt, *ssm_a, *ssm_alpha, *ssm_beta;
    const KataliGgufTensor *ssm_norm, *ssm_out;
    uint64_t expert_bytes_gate, expert_bytes_up, expert_bytes_down;
} MoeLayerTensors;
typedef struct MoeIndex {
    MoeLayerTensors *layers;
    int n_layers, n_experts;
    const KataliGgufTensor *tok_embd, *output, *output_norm;
} MoeIndex;
KATALI_API int  moe_index_build(MoeIndex *idx, const KataliGgufFile *f, const KataliArch *arch);
KATALI_API void moe_index_free(MoeIndex *idx);
KATALI_API int  moe_expert_span(const MoeLayerTensors *L, int which, int expert, uint64_t *rel_off, uint64_t *len);
#ifdef __cplusplus
}
#endif
#endif
