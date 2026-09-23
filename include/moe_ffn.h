#ifndef KATALI_MOE_FFN_H
#define KATALI_MOE_FFN_H
#include "host.h"

#ifdef __cplusplus
extern "C" {
#endif

/* One MoE FFN block: x[hidden] -> y[hidden], CPU only, elastic expert cache.
 * Decode / per-token path: pin → ensure_many → get + MLP + shared. */
KATALI_API int moe_ffn_forward(HostModel *m, int layer, const float *x, float *y);

/* Port #5 layer-major batch helpers (same numerics as moe_ffn_forward).
 * Route: softmax-all then top-k (unless KATALI_MOE_LOGIT_TOPK); writes eids/wts
 * (length K = n_experts_active) and updates last_eids[layer]. */
KATALI_API int moe_ffn_route(HostModel *m, int layer, const float *x,
                             int *eids_out, float *wts_out);

/* MLP + shared only — assumes ecache already primed (union ensure_many). */
KATALI_API int moe_ffn_mlp_routed(HostModel *m, int layer, const float *x,
                                  const int *eids, const float *wts, float *y);

#ifdef __cplusplus
}
#endif
#endif
