#include "model.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void katali_arch_qwen36_a3b_defaults(KataliArch *out) {
    memset(out, 0, sizeof(*out));
    snprintf(out->name, sizeof(out->name), "qwen35moe");
    out->n_layers = 41;
    out->n_trunk = 40;
    out->hidden = 2048;
    out->n_heads = 16;
    out->n_kv_heads = 2;
    out->head_dim = 256;
    out->q_dim = out->n_heads * out->head_dim;
    out->kv_dim = out->n_kv_heads * out->head_dim;
    out->q_proj_dim = out->q_dim * 2;
    out->vocab = 248320;
    out->n_experts = 256;
    out->n_experts_active = 8;
    out->shared_expert = 1;
    out->moe_intermediate = 512;
    out->rms_eps = 1e-6f;
    out->rope_theta = 10000000.f;
    out->rope_dim = 64;
    out->ctx_train = 262144;
    out->full_attn_interval = 4;
    out->ssm_inner = 4096;
    out->ssm_state = 128;
    out->ssm_conv_kernel = 4;
    out->ssm_dt_rank = 32;
    out->ssm_groups = 16;
    out->ssm_value_heads = 32;
    out->ssm_head_dim = 128;
    out->ssm_conv_channels = 2 * out->ssm_groups * out->ssm_head_dim
                           + out->ssm_value_heads * out->ssm_head_dim;
    out->layer_kind = (KataliLayerKind *)calloc((size_t)out->n_layers, sizeof(KataliLayerKind));
    if (out->layer_kind) {
        for (int i = 0; i < out->n_layers; i++) {
            int is_full = ((i % out->full_attn_interval) == (out->full_attn_interval - 1))
                       || (i == out->n_layers - 1);
            out->layer_kind[i] = is_full ? KLAYER_FULL_ATTN : KLAYER_LINEAR_ATTN;
        }
    }
    out->n_full_layers = 0;
    out->n_recurrent = 0;
    for (int i = 0; i < out->n_trunk; i++) {
        if (out->layer_kind[i] == KLAYER_FULL_ATTN) out->n_full_layers++;
        else out->n_recurrent++;
    }
}

static int64_t get_i(const KataliGgufFile *f, const char *key, int64_t def) {
    return katali_gguf_get_int(f, key, def);
}

static double get_f(const KataliGgufFile *f, const char *key, double def) {
    return katali_gguf_get_float(f, key, def);
}

int katali_arch_from_gguf(const KataliGgufFile *f, KataliArch *out) {
    katali_arch_qwen36_a3b_defaults(out);
    if (!f || !f->ok) return KATALI_OK;

    const char *arch = katali_gguf_get_str(f, "general.architecture", NULL, NULL);
    if (arch) snprintf(out->name, sizeof(out->name), "%s", arch);

    const char *pfx = out->name[0] ? out->name : "qwen35moe";
    char key[128];

#define ARCH_I(field, suffix, def) do { \
    snprintf(key, sizeof(key), "%s.%s", pfx, suffix); \
    out->field = (int)get_i(f, key, def); \
} while (0)

    ARCH_I(n_layers, "block_count", out->n_layers);
    ARCH_I(hidden, "embedding_length", out->hidden);
    ARCH_I(n_heads, "attention.head_count", out->n_heads);
    ARCH_I(n_kv_heads, "attention.head_count_kv", out->n_kv_heads);
    ARCH_I(head_dim, "attention.key_length", out->head_dim);
    if (out->head_dim <= 0 && out->n_heads > 0)
        out->head_dim = out->hidden / out->n_heads;
    ARCH_I(n_experts, "expert_count", out->n_experts);
    ARCH_I(n_experts_active, "expert_used_count", out->n_experts_active);
    ARCH_I(moe_intermediate, "expert_feed_forward_length", out->moe_intermediate);
    ARCH_I(ctx_train, "context_length", out->ctx_train);
    ARCH_I(full_attn_interval, "full_attention_interval", out->full_attn_interval);
    ARCH_I(rope_dim, "rope.dimension_count", out->rope_dim);
    ARCH_I(ssm_inner, "ssm.inner_size", out->ssm_inner);
    ARCH_I(ssm_state, "ssm.state_size", out->ssm_state);
    ARCH_I(ssm_conv_kernel, "ssm.conv_kernel", out->ssm_conv_kernel);
    ARCH_I(ssm_dt_rank, "ssm.time_step_rank", out->ssm_dt_rank);
    ARCH_I(ssm_groups, "ssm.group_count", out->ssm_groups);
    /* 122B: time_step_rank == v_heads (64); 35B uses 32. Prefer metadata when
     * tensor dims are not yet available. Recompute conv_channels so the later
     * rem→groups inference does not poison groups from a stale 35B default. */
    if (out->ssm_dt_rank > 0) out->ssm_value_heads = out->ssm_dt_rank;
    if (out->ssm_groups > 0 && out->ssm_head_dim > 0 && out->ssm_value_heads > 0)
        out->ssm_conv_channels = 2 * out->ssm_groups * out->ssm_head_dim
                               + out->ssm_value_heads * out->ssm_head_dim;
    {
        char key2[128];
        snprintf(key2, sizeof(key2), "%s.expert_shared_feed_forward_length", pfx);
        int64_t sh = get_i(f, key2, 0);
        if (sh <= 0) {
            snprintf(key2, sizeof(key2), "%s.shared_expert_feed_forward_length", pfx);
            sh = get_i(f, key2, 0);
        }
        (void)sh; /* shared uses same moe_intermediate in this engine */
    }

    snprintf(key, sizeof(key), "%s.rope.freq_base", pfx);
    out->rope_theta = (float)get_f(f, key, out->rope_theta);
    snprintf(key, sizeof(key), "%s.attention.layer_norm_rms_epsilon", pfx);
    out->rms_eps = (float)get_f(f, key, out->rms_eps);

    /* Infer head_dim from full-attn q_norm when present (more reliable). */
    {
        char nm[64];
        snprintf(nm, sizeof(nm), "blk.3.attn_q_norm.weight");
        const KataliGgufTensor *qn = katali_gguf_find_tensor(f, nm);
        if (qn && qn->n_dims >= 1 && qn->dims[0] > 0)
            out->head_dim = (int)qn->dims[0];
        snprintf(nm, sizeof(nm), "blk.3.attn_q.weight");
        const KataliGgufTensor *wq = katali_gguf_find_tensor(f, nm);
        if (wq && wq->n_dims >= 2)
            out->q_proj_dim = (int)wq->dims[1];
        snprintf(nm, sizeof(nm), "blk.3.attn_k.weight");
        const KataliGgufTensor *wk = katali_gguf_find_tensor(f, nm);
        if (wk && wk->n_dims >= 2 && out->head_dim > 0)
            out->n_kv_heads = (int)(wk->dims[1] / (uint64_t)out->head_dim);
    }

    /* SSM geometry from tensors when present. */
    {
        const KataliGgufTensor *sn = katali_gguf_find_tensor(f, "blk.0.ssm_norm.weight");
        const KataliGgufTensor *sa = katali_gguf_find_tensor(f, "blk.0.ssm_alpha.weight");
        const KataliGgufTensor *sc = katali_gguf_find_tensor(f, "blk.0.ssm_conv1d.weight");
        if (sn && sn->n_dims >= 1) out->ssm_head_dim = (int)sn->dims[0];
        if (sa && sa->n_dims >= 2) out->ssm_value_heads = (int)sa->dims[1];
        if (sc && sc->n_dims >= 2) {
            out->ssm_conv_kernel = (int)sc->dims[0];
            out->ssm_conv_channels = (int)sc->dims[1];
        }
        if (out->ssm_head_dim > 0 && out->ssm_value_heads > 0 && out->ssm_conv_channels > 0) {
            int rem = out->ssm_conv_channels - out->ssm_value_heads * out->ssm_head_dim;
            if (rem > 0 && (rem % (2 * out->ssm_head_dim)) == 0)
                out->ssm_groups = rem / (2 * out->ssm_head_dim);
        }
    }

    out->q_dim = out->n_heads * out->head_dim;
    out->kv_dim = out->n_kv_heads * out->head_dim;
    if (out->q_proj_dim <= 0) out->q_proj_dim = out->q_dim * 2;
    if (out->rope_dim <= 0 || out->rope_dim > out->head_dim)
        out->rope_dim = out->head_dim;
    if (out->ssm_value_heads <= 0 && out->ssm_dt_rank > 0)
        out->ssm_value_heads = out->ssm_dt_rank;
    if (out->ssm_head_dim <= 0 && out->ssm_state > 0)
        out->ssm_head_dim = out->ssm_state;
    if (out->ssm_conv_channels <= 0)
        out->ssm_conv_channels = 2 * out->ssm_groups * out->ssm_head_dim
                               + out->ssm_value_heads * out->ssm_head_dim;

    /* MTP tail: GGUF block_count = HF num_hidden_layers + mtp_num_hidden_layers.
     * 35B-A3B: 41 = 40+1; 122B-A10B: 49 = 48+1. The MTP block must NOT run in the
     * main decoder — doing so feeds a NextN eh_proj block as if it were a normal
     * full-attn layer and NaNs the residual (after_prefill top_i=-1 / !!!!). */
    out->n_trunk = out->n_layers;
    {
        int has_mtp = 0;
        int last = out->n_layers - 1;
        if (last >= 0) {
            static const char *sfx[] = {
                "nextn.eh_proj.weight", "nextn_eh_proj.weight", "eh_proj.weight",
                "nextn.enorm.weight", "nextn_enorm.weight", "enorm.weight",
                "nextn.hnorm.weight", "nextn_hnorm.weight", "hnorm.weight",
                NULL
            };
            char nm[96];
            for (int i = 0; sfx[i]; i++) {
                snprintf(nm, sizeof(nm), "blk.%d.%s", last, sfx[i]);
                if (katali_gguf_find_tensor(f, nm)) { has_mtp = 1; break; }
            }
        }
        if (!has_mtp && (out->n_layers == 41 || out->n_layers == 49))
            has_mtp = 1; /* known Qwen3.5 MoE + MTP packs */
        if (has_mtp && out->n_layers > 1)
            out->n_trunk = out->n_layers - 1;
    }

    /* Prefer expert tensor dims over metadata when present (FF = dims[1] for
     * gate/up exps laid out [H, FF, E]). */
    {
        const KataliGgufTensor *ge = katali_gguf_find_tensor(f, "blk.0.ffn_gate_exps.weight");
        if (ge && ge->n_dims >= 2 && ge->dims[1] > 0)
            out->moe_intermediate = (int)ge->dims[1];
    }

    free(out->layer_kind);
    out->layer_kind = (KataliLayerKind *)calloc((size_t)out->n_layers, sizeof(KataliLayerKind));
    int interval = out->full_attn_interval > 0 ? out->full_attn_interval : 4;
    out->n_full_layers = 0;
    out->n_recurrent = 0;
    if (out->layer_kind) {
        for (int i = 0; i < out->n_layers; i++) {
            int is_full;
            if (i >= out->n_trunk) {
                /* MTP tail — unused by forward; bookkeeping only. */
                is_full = 1;
            } else {
                is_full = ((i % interval) == (interval - 1)) || (i == out->n_trunk - 1);
            }
            out->layer_kind[i] = is_full ? KLAYER_FULL_ATTN : KLAYER_LINEAR_ATTN;
        }
        for (int i = 0; i < out->n_trunk; i++) {
            if (out->layer_kind[i] == KLAYER_FULL_ATTN) out->n_full_layers++;
            else out->n_recurrent++;
        }
    }

    /* Vocab from tokenizer token count when available. */
    {
        uint32_t et = 0;
        int64_t n = katali_gguf_array_len(f, "tokenizer.ggml.tokens", &et);
        if (n > 0) out->vocab = (int)n;
    }
    return KATALI_OK;
}

void katali_arch_free(KataliArch *a) {
    if (!a) return;
    free(a->layer_kind);
    memset(a, 0, sizeof(*a));
}

void katali_arch_print(const KataliArch *a) {
    if (!a) return;
    printf("arch: %s\n", a->name);
    printf("  layers=%d trunk=%d hidden=%d heads=%d kv=%d head_dim=%d q_proj=%d vocab=%d\n",
           a->n_layers, a->n_trunk, a->hidden, a->n_heads, a->n_kv_heads,
           a->head_dim, a->q_proj_dim, a->vocab);
    printf("  moe experts=%d active=%d shared=%d intermediate=%d\n",
           a->n_experts, a->n_experts_active, a->shared_expert, a->moe_intermediate);
    printf("  ctx_train=%d rope_theta=%g rope_dim=%d rms_eps=%g\n",
           a->ctx_train, a->rope_theta, a->rope_dim, a->rms_eps);
    printf("  ssm: groups=%d v_heads=%d head_dim=%d conv_ch=%d kernel=%d\n",
           a->ssm_groups, a->ssm_value_heads, a->ssm_head_dim,
           a->ssm_conv_channels, a->ssm_conv_kernel);
    printf("  layer_kinds (trunk): linear/DeltaNet=%d full_attn=%d\n",
           a->n_recurrent, a->n_full_layers);
}
