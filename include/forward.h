#ifndef KATALI_FORWARD_H
#define KATALI_FORWARD_H
#include "host.h"
#ifdef __cplusplus
extern "C" {
#endif
KATALI_API int  host_state_alloc(HostModel *m, int max_seq);
KATALI_API void host_state_free(HostModel *m);
KATALI_API void host_state_reset(HostModel *m);
/* host_forward_token is declared in host.h */

/* Port #5: layer-major batch prefill (katali2 host_qwen_prefill_batch §12.3).
 * Returns 1 if wanted (caller may call host_prefill_batch). */
KATALI_API int  host_prefill_wanted(const HostModel *m, int n_prompt);
/* Returns 0=done, 1=declined (use per-token), -1=hard fail. */
KATALI_API int  host_prefill_batch(HostModel *m, const int *ids, int n);
#ifdef __cplusplus
}
#endif
#endif
