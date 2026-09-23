#include "moe_ffn.h"
#include "katali_cuda.h"
#include "platform.h"
#include "katali_gguf_dtype.h"
#include <stdio.h>
#include "katali_gguf_kernels.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct { float score; int id; } MoeRoute;

/* Match katali2 host_topk_route: higher logit wins; tie -> lower expert id. */
static int cmp_route_desc(const void *a, const void *b) {
    const MoeRoute *ra = (const MoeRoute *)a;
    const MoeRoute *rb = (const MoeRoute *)b;
    if (ra->score < rb->score) return 1;
    if (ra->score > rb->score) return -1;
    return (ra->id > rb->id) - (ra->id < rb->id);
}

/*
 * VRAM expert path: run an expert's gate/up/down GEMVs on the GPU against
 * device-resident weights (uploaded on first use, LRU-evicted under budget).
 *
 * Only the small activation vectors cross PCIe (H + FF + FF + H floats per
 * expert, ~12 KiB), never the ~2 MB of weights. That is the whole point: with
 * Gen3 x8 (~6.5 GB/s) streaming weights per token would cap decode at ~3.8
 * tok/s, so weights must stay resident and only activations move.
 *
 * Returns 1 when the GPU handled the expert, 0 to ask the caller for the CPU
 * path. Any CUDA failure returns 0, so behaviour degrades to the CPU rather
 * than producing a wrong result.
 */
static int expert_matvecs_gpu(HostModel *m, MoeLayerTensors *L, int expert,
                              const float *x, float *scratch_ff, float *y_out) {
    if (!m->vram_ready || !vram_cache_enabled(&m->vram)) return 0;
    if (!L->gate_exps || !L->up_exps || !L->down_exps) return 0;
    const int H = m->arch.hidden, FF = m->arch.moe_intermediate;
    if (H <= 0 || FF <= 0) return 0;
    uint64_t rel = 0, len = 0;
    const uint8_t *g = NULL, *u = NULL, *d = NULL;
    size_t glen = 0, ulen = 0, dlen = 0;
    if (moe_expert_span(L, 0, expert, &rel, &len) != KATALI_OK) return 0;
    g = L->gate_exps->data + rel; glen = (size_t)len;
    if (moe_expert_span(L, 1, expert, &rel, &len) != KATALI_OK) return 0;
    u = L->up_exps->data + rel; ulen = (size_t)len;
    if (moe_expert_span(L, 2, expert, &rel, &len) != KATALI_OK) return 0;
    d = L->down_exps->data + rel; dlen = (size_t)len;

    void *dg = vram_cache_get(&m->vram, L->layer, expert * 3 + 0, g, glen);
    void *du = vram_cache_get(&m->vram, L->layer, expert * 3 + 1, u, ulen);
    void *dd = vram_cache_get(&m->vram, L->layer, expert * 3 + 2, d, dlen);
    if (!dg || !du || !dd) return 0;   /* not enough VRAM: CPU path */

    float *gate_out = scratch_ff, *up_out = scratch_ff + FF;
    float *inter = scratch_ff + 2 * FF, *down_out = scratch_ff + 3 * FF;
    int ok =
        katali_cuda_upload(m->vram_x, x, (size_t)H * sizeof(float)) == KATALI_OK &&
        katali_cuda_matvec(L->gate_exps->type, dg, (uint64_t)FF, (uint64_t)H,
                           (const float *)m->vram_x, (float *)m->vram_gy) == KATALI_OK &&
        katali_cuda_matvec(L->up_exps->type, du, (uint64_t)FF, (uint64_t)H,
                           (const float *)m->vram_x, (float *)m->vram_uy) == KATALI_OK &&
        katali_cuda_download(gate_out, m->vram_gy, (size_t)FF * sizeof(float)) == KATALI_OK &&
        katali_cuda_download(up_out, m->vram_uy, (size_t)FF * sizeof(float)) == KATALI_OK;
    if (!ok) return 0;

    /* Same SiLU(gate)*up the CPU path computes, so numerics stay comparable. */
    for (int i = 0; i < FF; i++) {
        float gg = gate_out[i];
        inter[i] = (gg / (1.f + expf(-gg))) * up_out[i];
    }
    ok = katali_cuda_upload(m->vram_iy, inter, (size_t)FF * sizeof(float)) == KATALI_OK &&
         katali_cuda_matvec(L->down_exps->type, dd, (uint64_t)H, (uint64_t)FF,
                            (const float *)m->vram_iy, (float *)m->vram_dy) == KATALI_OK &&
         katali_cuda_download(down_out, m->vram_dy, (size_t)H * sizeof(float)) == KATALI_OK;
    if (!ok) return 0;

    memcpy(y_out, down_out, (size_t)H * sizeof(float));
    return 1;
}

static int expert_matvecs(HostModel *m, MoeLayerTensors *L, int expert,
                          const float *x, float *scratch_ff, float *y_out) {
    if (m->vram_ready &&
        expert_matvecs_gpu(m, L, expert, x, scratch_ff, y_out)) {
        katali_cuda_count_expert(1);
        return KATALI_OK;
    }
    uint64_t rel = 0, len = 0;
    if (!L->gate_exps || !L->up_exps || !L->down_exps) return KATALI_ERR;
    if (moe_expert_span(L, 0, expert, &rel, &len) != KATALI_OK) return KATALI_ERR;
    const uint8_t *gate, *up, *down;
    if (getenv("KATALI_NO_ECACHE")) {
        gate = L->gate_exps->data + rel;
        if (moe_expert_span(L, 1, expert, &rel, &len) != KATALI_OK) return KATALI_ERR;
        up = L->up_exps->data + rel;
        if (moe_expert_span(L, 2, expert, &rel, &len) != KATALI_OK) return KATALI_ERR;
        down = L->down_exps->data + rel;
    } else {
        /* Absolute pointers — works across multi-shard maps.
         * Port #2: three keys; prefer hit after ensure_many. See docs/OPTIMIZE.md. */
        gate = ecache_get(&m->ecache, L->layer, expert * 3 + 0, L->gate_exps->data + rel, len);
        if (!gate) return KATALI_ERR;
        if (moe_expert_span(L, 1, expert, &rel, &len) != KATALI_OK) return KATALI_ERR;
        up = ecache_get(&m->ecache, L->layer, expert * 3 + 1, L->up_exps->data + rel, len);
        if (!up) return KATALI_ERR;
        if (moe_expert_span(L, 2, expert, &rel, &len) != KATALI_OK) return KATALI_ERR;
        down = ecache_get(&m->ecache, L->layer, expert * 3 + 2, L->down_exps->data + rel, len);
        if (!down) return KATALI_ERR;
    }

    const int H = m->arch.hidden, FF = m->arch.moe_intermediate;
    float *gate_out = scratch_ff, *up_out = scratch_ff + FF;
    float *inter = scratch_ff + 2 * FF, *down_out = scratch_ff + 3 * FF;
    katali_ggml_matvec(L->gate_exps->type, gate, (uint64_t)FF, (uint64_t)H, x, gate_out, 0, NULL, NULL, 0);
    katali_ggml_matvec(L->up_exps->type, up, (uint64_t)FF, (uint64_t)H, x, up_out, 0, NULL, NULL, 0);
    for (int i = 0; i < FF; i++) {
        float g = gate_out[i];
        inter[i] = (g / (1.f + expf(-g))) * up_out[i];
    }
    katali_ggml_matvec(L->down_exps->type, down, (uint64_t)H, (uint64_t)FF, inter, down_out, 0, NULL, NULL, 0);
    memcpy(y_out, down_out, (size_t)H * sizeof(float));
    katali_cuda_count_expert(0);
    return KATALI_OK;
}

static int shared_ff_dim(const MoeLayerTensors *L, int fallback) {
    /* gate/up shexp: GGUF dims [H, FF] → dims[1]=FF; down: [FF, H] → dims[0]=FF. */
    if (L->gate_shexp && L->gate_shexp->n_dims >= 2 && L->gate_shexp->dims[1] > 0)
        return (int)L->gate_shexp->dims[1];
    if (L->down_shexp && L->down_shexp->n_dims >= 1 && L->down_shexp->dims[0] > 0)
        return (int)L->down_shexp->dims[0];
    return fallback;
}

static int shared_expert(HostModel *m, MoeLayerTensors *L, const float *x,
                         float *scratch_ff, float *y_acc) {
    if (!L->gate_shexp || !L->up_shexp || !L->down_shexp) return KATALI_OK;
    if (!L->gate_shexp->data || !L->up_shexp->data || !L->down_shexp->data) return KATALI_ERR;
    const int H = m->arch.hidden;
    const int FF = shared_ff_dim(L, m->arch.moe_intermediate);
    float *gate_out = scratch_ff, *up_out = scratch_ff + FF;
    float *inter = scratch_ff + 2 * FF, *down_out = scratch_ff + 3 * FF;
    katali_ggml_matvec(L->gate_shexp->type, L->gate_shexp->data, (uint64_t)FF, (uint64_t)H, x, gate_out, 0, NULL, NULL, 0);
    katali_ggml_matvec(L->up_shexp->type, L->up_shexp->data, (uint64_t)FF, (uint64_t)H, x, up_out, 0, NULL, NULL, 0);
    for (int i = 0; i < FF; i++) {
        float g = gate_out[i];
        inter[i] = (g / (1.f + expf(-g))) * up_out[i];
    }
    katali_ggml_matvec(L->down_shexp->type, L->down_shexp->data, (uint64_t)H, (uint64_t)FF, inter, down_out, 0, NULL, NULL, 0);
    float g = 1.f;
    if (L->gate_inp_shexp && L->gate_inp_shexp->data && L->gate_inp_shexp->type == KGGML_F32) {
        const float *w = (const float *)L->gate_inp_shexp->data;
        float dot = 0.f;
        for (int i = 0; i < H; i++) dot += x[i] * w[i];
        g = katali_gguf_sigmoid(dot);
    }
    for (int i = 0; i < H; i++) y_acc[i] += g * down_out[i];
    return KATALI_OK;
}

static int moe_alloc_scratch(HostModel *m, MoeLayerTensors *L,
                             float **logits, MoeRoute **routes,
                             float **scratch, float **expert_y) {
    const int H = m->arch.hidden, E = m->arch.n_experts;
    int ff_scratch = m->arch.moe_intermediate;
    {
        int sff = shared_ff_dim(L, m->arch.moe_intermediate);
        if (sff > ff_scratch) ff_scratch = sff;
    }
    *logits = (float *)malloc((size_t)E * sizeof(float));
    *routes = (MoeRoute *)malloc((size_t)E * sizeof(MoeRoute));
    *scratch = (float *)malloc((size_t)(3 * ff_scratch + H) * sizeof(float));
    *expert_y = (float *)malloc((size_t)H * sizeof(float));
    if (!*logits || !*routes || !*scratch || !*expert_y) {
        free(*logits); free(*routes); free(*scratch); free(*expert_y);
        *logits = NULL; *routes = NULL; *scratch = NULL; *expert_y = NULL;
        return KATALI_ERR_NOMEM;
    }
    return KATALI_OK;
}

/* Fill eids/wts[0..K) from gate logits; update last_eids. Shared by forward + batch. */
static int moe_route_into(HostModel *m, int layer, const float *x,
                          int *eids_out, float *wts_out) {
    MoeLayerTensors *L = &m->moe.layers[layer];
    const int H = m->arch.hidden, E = m->arch.n_experts, K = m->arch.n_experts_active;
    if (!L->gate_inp || !L->gate_inp->data) return KATALI_ERR;
    if (!eids_out || !wts_out || K <= 0 || E <= 0) return KATALI_ERR;

    float *logits = NULL;
    MoeRoute *routes = NULL;
    float *scratch = NULL, *expert_y = NULL;
    if (moe_alloc_scratch(m, L, &logits, &routes, &scratch, &expert_y) != KATALI_OK)
        return KATALI_ERR_NOMEM;
    (void)scratch; (void)expert_y;

    katali_ggml_matvec(KGGML_F32, L->gate_inp->data, (uint64_t)E, (uint64_t)H, x, logits, 0, NULL, NULL, 0);
    if (getenv("KATALI_MOE_LOGIT_TOPK")) {
        for (int i = 0; i < E; i++) { routes[i].id = i; routes[i].score = logits[i]; }
        qsort(routes, (size_t)E, sizeof(MoeRoute), cmp_route_desc);
        float maxv = routes[0].score, sum = 0.f;
        for (int i = 0; i < K; i++) { routes[i].score = expf(routes[i].score - maxv); sum += routes[i].score; }
        if (sum < 1e-20f) sum = 1.f;
        for (int i = 0; i < K; i++) routes[i].score /= sum;
    } else {
        float maxv = logits[0];
        for (int i = 1; i < E; i++) if (logits[i] > maxv) maxv = logits[i];
        float sum = 0.f;
        for (int i = 0; i < E; i++) {
            float e = expf(logits[i] - maxv);
            logits[i] = e;
            sum += e;
        }
        if (sum < 1e-20f) sum = 1.f;
        for (int i = 0; i < E; i++) {
            routes[i].id = i;
            routes[i].score = logits[i] / sum;
        }
        qsort(routes, (size_t)E, sizeof(MoeRoute), cmp_route_desc);
        sum = 0.f;
        for (int i = 0; i < K; i++) sum += routes[i].score;
        if (sum < 1e-20f) sum = 1.f;
        for (int i = 0; i < K; i++) routes[i].score /= sum;
    }
    for (int i = 0; i < K; i++) {
        eids_out[i] = routes[i].id;
        wts_out[i] = routes[i].score;
    }
    if (m->last_eids && m->last_topk >= K) {
        int *slot = m->last_eids + (size_t)layer * (size_t)m->last_topk;
        for (int i = 0; i < K; i++) slot[i] = routes[i].id;
        for (int i = K; i < m->last_topk; i++) slot[i] = -1;
    }
    free(logits); free(routes); free(scratch); free(expert_y);
    return KATALI_OK;
}

int moe_ffn_route(HostModel *m, int layer, const float *x,
                  int *eids_out, float *wts_out) {
    if (!m || !m->opened || !x) return KATALI_ERR;
    if (layer < 0 || layer >= m->moe.n_layers) return KATALI_ERR;
    return moe_route_into(m, layer, x, eids_out, wts_out);
}

static int expert_matvecs_fused(HostModel *m, MoeLayerTensors *L, int layer,
                                const float *x, const int *eids, const float *wts,
                                int K, float *scratch_ff, float *y_out) {
    const int H = m->arch.hidden, FF = m->arch.moe_intermediate;
    const uint8_t *gates[64], *ups[64], *downs[64];
    float *gall = (float *)malloc((size_t)K * (size_t)FF * sizeof(float));
    float *uall = (float *)malloc((size_t)K * (size_t)FF * sizeof(float));
    KataliMatvecJob jobs[128];
    if (!gall || !uall) { free(gall); free(uall); return KATALI_ERR_NOMEM; }
    for (int i = 0; i < K; i++) {
        uint64_t rel = 0, len = 0;
        if (!L->gate_exps || !L->up_exps || !L->down_exps ||
            moe_expert_span(L, 0, eids[i], &rel, &len) != KATALI_OK) goto fail;
        gates[i] = getenv("KATALI_NO_ECACHE") ? L->gate_exps->data + rel :
            ecache_get(&m->ecache, layer, eids[i] * 3 + 0, L->gate_exps->data + rel, len);
        if (moe_expert_span(L, 1, eids[i], &rel, &len) != KATALI_OK) goto fail;
        ups[i] = getenv("KATALI_NO_ECACHE") ? L->up_exps->data + rel :
            ecache_get(&m->ecache, layer, eids[i] * 3 + 1, L->up_exps->data + rel, len);
        if (moe_expert_span(L, 2, eids[i], &rel, &len) != KATALI_OK) goto fail;
        downs[i] = getenv("KATALI_NO_ECACHE") ? L->down_exps->data + rel :
            ecache_get(&m->ecache, layer, eids[i] * 3 + 2, L->down_exps->data + rel, len);
        if (!gates[i] || !ups[i] || !downs[i]) goto fail;
        jobs[i * 2 + 0] = (KataliMatvecJob){ L->gate_exps->type, gates[i], (uint64_t)FF,
                                             gall + (size_t)i * (size_t)FF, KATALI_ROLE_GATE };
        jobs[i * 2 + 1] = (KataliMatvecJob){ L->up_exps->type, ups[i], (uint64_t)FF,
                                             uall + (size_t)i * (size_t)FF, KATALI_ROLE_UP };
    }
    katali_ggml_matvec_fused(jobs, K * 2, (uint64_t)H, x, 0, NULL, NULL, 0);
    for (int i = 0; i < K; i++) {
        float *inter = scratch_ff + 2 * FF;
        float *down_out = scratch_ff + 3 * FF;
        for (int d = 0; d < FF; d++) {
            float g = gall[(size_t)i * (size_t)FF + (size_t)d];
            float u = uall[(size_t)i * (size_t)FF + (size_t)d];
            inter[d] = (g / (1.f + expf(-g))) * u;
        }
        katali_ggml_matvec(L->down_exps->type, downs[i], (uint64_t)H,
                           (uint64_t)FF, inter, down_out, 0, NULL, NULL, 0);
        for (int d = 0; d < H; d++) y_out[d] += wts[i] * down_out[d];
    }
    free(gall); free(uall);
    return KATALI_OK;
fail:
    free(gall); free(uall);
    return KATALI_ERR;
}

/*
 * Fused layer-level MoE on the GPU (Phase 2-7).
 *
 * One call per MoE layer instead of one call per expert per projection. The
 * measured motivation: the per-expert path issued 960 kernel launches and 960
 * blocking syncs per token (2 826 CUDA API calls, 354 ms of a 626 ms token).
 * This path costs, per layer: 1 upload + 4-5 fused launches + 1 sync + 1
 * download.
 *
 * All intermediate activations stay on the device; the only PCIe traffic is the
 * layer input and the layer output. Expert weights are device-resident via the
 * VRAM tier, so a hit costs nothing but a hash lookup.
 *
 * Returns 1 when `y` holds the routed-expert sum, 0 for the caller to use the
 * CPU path. Any failure returns 0 - never a wrong result.
 */
static int moe_ffn_routed_gpu_submit(HostModel *m, MoeLayerTensors *L,
                                     const float *x, const int *eids,
                                     const float *wts, int K, int async) {
    if (!m->vram_ready || !vram_cache_enabled(&m->vram)) return 0;
    if (!L->gate_exps || !L->up_exps || !L->down_exps) return 0;
    if (K < 1 || K > 16) return 0;
    const uint64_t H  = (uint64_t)m->arch.hidden;
    const uint64_t FF = (uint64_t)m->arch.moe_intermediate;
    if (H == 0 || FF == 0) return 0;
    /* The fused kernel assumes exactly the CPU path's shapes. */
    if (L->gate_exps->dims[0] != H || L->up_exps->dims[0] != H) return 0;
    if (L->down_exps->dims[0] != FF || L->down_exps->dims[1] != H) return 0;
    if (!katali_cuda_supports_type(L->gate_exps->type) ||
        !katali_cuda_supports_type(L->up_exps->type) ||
        !katali_cuda_supports_type(L->down_exps->type)) return 0;

    uint64_t rb_g = katali_ggml_row_bytes(L->gate_exps->type, H);
    uint64_t rb_u = katali_ggml_row_bytes(L->up_exps->type, H);
    uint64_t rb_d = katali_ggml_row_bytes(L->down_exps->type, FF);
    if (!rb_g || !rb_u || !rb_d) return 0;
    const uint64_t rows_g = L->gate_exps->dims[1];
    const uint64_t rows_u = L->up_exps->dims[1];
    const uint64_t rows_d = L->down_exps->dims[1];

    /* Make every selected expert's three slabs device-resident. */
    const void *dg[16], *du[16], *dd[16];
    const double tl_vram_t0 = katali_time_s();
    for (int j = 0; j < K; j++) {
        const int e = eids[j];
        if (e < 0 || e >= m->moe.n_experts) return 0;
        uint64_t r = 0, l = 0;
        if (moe_expert_span(L, 0, e, &r, &l) != KATALI_OK) return 0;
        if (l != rb_g * rows_g) return 0;
        dg[j] = vram_cache_get(&m->vram, L->layer, e * 3 + 0,
                               L->gate_exps->data + r, (size_t)l);
        if (!dg[j]) return 0;
        if (moe_expert_span(L, 1, e, &r, &l) != KATALI_OK) return 0;
        if (l != rb_u * rows_u) return 0;
        du[j] = vram_cache_get(&m->vram, L->layer, e * 3 + 1,
                               L->up_exps->data + r, (size_t)l);
        if (!du[j]) return 0;
        if (moe_expert_span(L, 2, e, &r, &l) != KATALI_OK) return 0;
        if (l != rb_d * rows_d) return 0;
        dd[j] = vram_cache_get(&m->vram, L->layer, e * 3 + 2,
                               L->down_exps->data + r, (size_t)l);
        if (!dd[j]) return 0;
    }
    m->tl_vram += katali_time_s() - tl_vram_t0;

    KataliCudaMoeDesc desc;
    memset(&desc, 0, sizeof(desc));
    desc.gate_type = L->gate_exps->type;
    desc.up_type   = L->up_exps->type;
    desc.down_type = L->down_exps->type;
    desc.gate_row_bytes = rb_g;
    desc.up_row_bytes   = rb_u;
    desc.down_row_bytes = rb_d;
    desc.gate_rows = rows_g;
    desc.up_rows   = rows_u;
    desc.down_rows = rows_d;
    desc.hidden    = H;
    desc.inter     = FF;

    {
        const double tl_x_t0 = katali_time_s();
        const int xrc = katali_cuda_upload(m->vram_x, x, (size_t)H * sizeof(float));
        m->tl_xup += katali_time_s() - tl_x_t0;
        if (xrc != KATALI_OK) return 0;
    }

    /*
     * Phase 11: optional per-layer profile. Correlates each layer's GPU time
     * with its VRAM hit/miss/eviction counts, so a slow layer can be attributed
     * to churn rather than guessed at. Off unless KATALI_CUDA_LAYER_PROFILE=1.
     */
    static int prof = -1;
    if (prof < 0) {
        const char *e = getenv("KATALI_CUDA_LAYER_PROFILE");
        prof = (e && e[0] && e[0] != '0') ? 1 : 0;
    }
    KataliCudaTelemetry t0, t1;
    uint64_t vh0 = 0, vm0 = 0, ve0 = 0, vu0 = 0;
    size_t used0 = 0, bud0 = 0, nent0 = 0;
    if (prof) {
        katali_cuda_telemetry_get(&t0);
        vram_cache_stats(&m->vram, &vh0, &vm0, &ve0, &vu0, &used0, &bud0, &nent0);
    }

    /* Phase 4: queue the layer and return, or block here when async is off. */
    const double tl_sub_t0 = katali_time_s();
    const int rc = async
        ? katali_cuda_moe_submit(&desc, dg, du, dd, wts, K,
                                 (const float *)m->vram_x, (float *)m->vram_dy)
        : katali_cuda_moe_layer(&desc, dg, du, dd, wts, K,
                                (const float *)m->vram_x, (float *)m->vram_dy);
    m->tl_sub += katali_time_s() - tl_sub_t0;

    if (prof) {
        katali_cuda_telemetry_get(&t1);
        uint64_t vh1 = 0, vm1 = 0, ve1 = 0, vu1 = 0;
        size_t used1 = 0, bud1 = 0, nent1 = 0;
        vram_cache_stats(&m->vram, &vh1, &vm1, &ve1, &vu1, &used1, &bud1, &nent1);
        const double gpu_ms = (t1.gpu_seconds - t0.gpu_seconds) * 1000.0;
        const double wmb = (double)K * (double)(rb_g * rows_g + rb_u * rows_u +
                                                rb_d * rows_d) / (1024.0 * 1024.0);
        const double gbs = (gpu_ms > 0.0) ? wmb / 1024.0 / (gpu_ms / 1000.0) : 0.0;
        fprintf(stderr,
                "cu-layer L%-3d n_sel=%d bytes=%.2fMiB gpu=%.3fms %.1fGB/s "
                "vhit=%llu vmiss=%llu vevict=%llu vupload=%llu resident=%zu\n",
                L->layer, K, wmb, gpu_ms, gbs,
                (unsigned long long)(vh1 - vh0), (unsigned long long)(vm1 - vm0),
                (unsigned long long)(ve1 - ve0), (unsigned long long)(vu1 - vu0),
                nent1);
    }
    if (rc != KATALI_OK) return 0;
    return 1;      /* queued (or completed when !async) into m->vram_dy */
}

/*
 * Finish a submitted layer: wait once, copy the result down, accumulate it and
 * attribute the experts to the GPU. Split out from the submit so the caller can
 * run independent CPU work (the shared expert) in between.
 *
 * The accumulation order is `routed + shared`, which is arithmetically identical
 * to the original `y = routed; y += shared` because y starts zeroed.
 */
static int moe_ffn_routed_gpu_finish(HostModel *m, int K,
                                     float *y, float *expert_y) {
    const int H = m->arch.hidden;
    if (katali_cuda_moe_wait() != KATALI_OK) return 0;
    if (katali_cuda_download(expert_y, m->vram_dy, (size_t)H * sizeof(float))
            != KATALI_OK) return 0;
    for (int d = 0; d < H; d++) y[d] += expert_y[d];
    for (int j = 0; j < K; j++) katali_cuda_count_expert(1);
    return 1;
}

/* Blocking convenience path: submit then immediately finish, i.e. the original
 * synchronous behaviour, used when async overlap is disabled for A/B. */
static int moe_ffn_routed_gpu(HostModel *m, MoeLayerTensors *L, const float *x,
                              const int *eids, const float *wts, int K,
                              float *y, float *expert_y) {
    if (!moe_ffn_routed_gpu_submit(m, L, x, eids, wts, K, 0)) return 0;
    return moe_ffn_routed_gpu_finish(m, K, y, expert_y);
}

int moe_ffn_mlp_routed(HostModel *m, int layer, const float *x,
                       const int *eids, const float *wts, float *y) {
    if (!m || !m->opened || !x || !y || !eids || !wts) return KATALI_ERR;
    if (layer < 0 || layer >= m->moe.n_layers) return KATALI_ERR;
    MoeLayerTensors *L = &m->moe.layers[layer];
    const int H = m->arch.hidden, K = m->arch.n_experts_active;
    if (K <= 0) return KATALI_ERR;

    float *logits = NULL;
    MoeRoute *routes = NULL;
    float *scratch = NULL, *expert_y = NULL;
    if (moe_alloc_scratch(m, L, &logits, &routes, &scratch, &expert_y) != KATALI_OK)
        return KATALI_ERR_NOMEM;
    (void)logits; (void)routes;

    memset(y, 0, (size_t)H * sizeof(float));

    /*
     * Fused layer-level GPU path (default when the VRAM tier is live).
     * KATALI_CUDA_MOE_LAYER=0 keeps the Phase 1 per-expert GPU path instead, so
     * the two can be A/B'd with everything else identical.
     *
     * Phase 4/5: with KATALI_CUDA_MOE_ASYNC=1 (default) the layer is SUBMITTED
     * and the CPU runs the shared expert - which depends only on `x`, not on the
     * routed experts - while the GPU executes. The routed result is collected
     * afterwards. KATALI_CUDA_MOE_ASYNC=0 restores the fully synchronous path.
     */
    int routed_gpu = 0, routed_pending = 0;
    const double tl_gpu_t0 = katali_time_s();
    {
        const char *lt = getenv("KATALI_CUDA_MOE_LAYER");
        const int want_layer = m->vram_ready && !(lt && lt[0] == '0');
        const char *ae = getenv("KATALI_CUDA_MOE_ASYNC");
        const int async_on = !(ae && ae[0] == '0');
        if (want_layer) {
            if (async_on)
                routed_pending = moe_ffn_routed_gpu_submit(m, L, x, eids, wts, K, 1);
            else
                routed_gpu = moe_ffn_routed_gpu(m, L, x, eids, wts, K, y, expert_y);
        }
    }
    m->tl_gpu += katali_time_s() - tl_gpu_t0;

    int use_fused = 0;
    int fused_explicit = 0;
    {
        const char *fe = getenv("KATALI_MOE_FUSED_EXPERTS");
        fused_explicit = fe && fe[0] != '\0';
        use_fused = fused_explicit && fe[0] != '0';
    }
    /* 35B-A3B benefits in sustained decode; 122B-A10B remains opt-in because
     * its longer validation run regressed despite short-run gains. */
    if (!fused_explicit && !use_fused && m->arch.n_experts == 256 && K == 8 &&
        H == 2048)
        use_fused = 1;
    /* The VRAM tier supersedes CPU-side expert fusion: fusion exists to amortize
     * CPU weight traffic, which the GPU path removes entirely. */
    if (m->vram_ready) use_fused = 0;
    if (routed_gpu || routed_pending) {
        /* Routed experts are on the GPU. With `pending` the result is not read
         * yet; the shared expert below runs while the kernel is still in flight. */
    } else if (use_fused) {
        if (expert_matvecs_fused(m, L, layer, x, eids, wts, K,
                                 scratch, y) != KATALI_OK) {
            free(logits); free(routes); free(scratch); free(expert_y);
            return KATALI_ERR;
        }
    } else {
        for (int i = 0; i < K; i++) {
            if (expert_matvecs(m, L, eids[i], x, scratch, expert_y) != KATALI_OK) {
                free(logits); free(routes); free(scratch); free(expert_y);
                return KATALI_ERR;
            }
            float w = wts[i];
            for (int d = 0; d < H; d++) y[d] += w * expert_y[d];
        }
    }
    if (!getenv("KATALI_SKIP_SHARED")) {
        /* Phase 1: the shared expert depends only on `x`, not on the routed
         * experts, so it is the one real CPU/GPU overlap candidate per layer. */
        const double tl_sh_t0 = katali_time_s();
        const int sh_rc = shared_expert(m, L, x, scratch, y);
        m->tl_shared += katali_time_s() - tl_sh_t0;
        if (sh_rc != KATALI_OK) {
            free(logits); free(routes); free(scratch); free(expert_y);
            return KATALI_ERR;
        }
    }
    /*
     * Phase 4/5: collect the submitted GPU work now. Everything that could run
     * while the kernel was in flight has already run, so this wait is the
     * residual - by construction shorter than the old blocking call.
     */
    if (routed_pending) {
        const double tl_w_t0 = katali_time_s();
        const int okf = moe_ffn_routed_gpu_finish(m, K, y, expert_y);
        m->tl_wait += katali_time_s() - tl_w_t0;
        if (!okf) {
            free(logits); free(routes); free(scratch); free(expert_y);
            return KATALI_ERR;
        }
    }
    free(logits); free(routes); free(scratch); free(expert_y);
    return KATALI_OK;
}

int moe_ffn_forward(HostModel *m, int layer, const float *x, float *y) {
    if (!m || !m->opened || !x || !y) return KATALI_ERR;
    if (layer < 0 || layer >= m->moe.n_layers) return KATALI_ERR;
    const int K = m->arch.n_experts_active;
    if (K <= 0 || K > 64) return KATALI_ERR;

    int eids[64];
    float wts[64];
    /* Phase 1: the router is a CPU matvec that must finish before the GPU can
     * be told which experts to run; it is on the critical path either way. */
    const double tl_route_t0 = katali_time_s();
    if (moe_route_into(m, layer, x, eids, wts) != KATALI_OK) return KATALI_ERR;
    m->tl_router += katali_time_s() - tl_route_t0;

    /*
     * Port #2 demand path (katali2 STREAM=0): soft-pin current experts' 3 weight
     * keys → ecache_ensure_many for cold fills → MLP via ecache_get (hits).
     * Prefetch of prior-token routes is issued from forward.c during attn.
     */
    /*
     * The RAM (ECache) tier stays ACTIVE alongside the VRAM tier. It is the
     * middle level of SSD -> RAM -> VRAM, and its async workers/prefetch are
     * what keep a VRAM miss cheap. Disabling it here made every VRAM miss a
     * synchronous mmap fill and measured SLOWER than CPU-only (2.15 vs 3.10
     * tok/s), which is why it is left in place.
     */
    if (!getenv("KATALI_NO_ECACHE") && m->ecache.default_fill && m->ecache.default_fill_ud) {
        int nkeys = K * 3;
        int *layers_arr = (int *)malloc((size_t)nkeys * sizeof(int));
        int *keys_arr = (int *)malloc((size_t)nkeys * sizeof(int));
        if (layers_arr && keys_arr) {
            for (int i = 0; i < K; i++) {
                int eid = eids[i];
                for (int w = 0; w < 3; w++) {
                    int idx = i * 3 + w;
                    layers_arr[idx] = layer;
                    keys_arr[idx] = eid * 3 + w;
                    ecache_pin(&m->ecache, layer, keys_arr[idx]);
                }
            }
            (void)ecache_ensure_many(&m->ecache, layers_arr, keys_arr, nkeys,
                                     m->ecache.default_fill, m->ecache.default_fill_ud);
        }
        free(layers_arr);
        free(keys_arr);
    }

    return moe_ffn_mlp_routed(m, layer, x, eids, wts, y);
}
