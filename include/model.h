#ifndef KATALI_MODEL_H
#define KATALI_MODEL_H
#include "katali.h"
#include "katali_gguf.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    KLAYER_FULL_ATTN = 0,
    KLAYER_LINEAR_ATTN = 1
} KataliLayerKind;

typedef struct KataliArch {
    char     name[64];
    int      n_layers;
    int      n_trunk;
    int      hidden;
    int      n_heads;
    int      n_kv_heads;
    int      head_dim;
    int      q_proj_dim;
    int      kv_dim;
    int      q_dim;
    int      vocab;
    int      n_experts;
    int      n_experts_active;
    int      shared_expert;
    int      moe_intermediate;
    float    rms_eps;
    float    rope_theta;
    int      rope_dim;
    int      ctx_train;
    int      full_attn_interval;
    int      n_full_layers;
    int      n_recurrent;

    int      ssm_inner;
    int      ssm_state;
    int      ssm_conv_kernel;
    int      ssm_dt_rank;
    int      ssm_groups;
    int      ssm_value_heads;
    int      ssm_head_dim;
    int      ssm_conv_channels;

    KataliLayerKind *layer_kind;
} KataliArch;

KATALI_API int  katali_arch_from_gguf(const KataliGgufFile *f, KataliArch *out);
KATALI_API void katali_arch_qwen36_a3b_defaults(KataliArch *out);
KATALI_API void katali_arch_free(KataliArch *a);
KATALI_API void katali_arch_print(const KataliArch *a);

#ifdef __cplusplus
}
#endif
#endif
