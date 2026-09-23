/* Katali session — Apache-2.0
 *
 * A persistent, backend-agnostic chat/inference session: the model is opened
 * once and reused for every question, with the KV state reset between
 * independent questions. Katali-GGUF's CLI, GUI and the router's residency
 * layer can all drive the same object instead of each re-implementing the
 * open/generate/reset lifecycle.
 */
#ifndef KATALI_SESSION_H
#define KATALI_SESSION_H

#include "katali_backend.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct KataliSessionStats {
    /* lifecycle */
    double load_s;            /* model open (paid once) */
    int    model_loads;       /* 1 after open; never grows while reused */
    int    questions;         /* questions answered by this session */
    int    kv_resets;         /* cumulative reset_state() calls */
    double last_reset_s;      /* cost of the most recent KV reset */

    /* most recent question */
    double last_tokenize_s;
    double last_prefill_s;
    double last_ttft_s;
    double last_decode_s;
    double last_total_s;
    int    last_prompt_tokens;
    int    last_generated_tokens;
    double last_decode_tps;
    int    last_batched_prefill;

    /* aggregates */
    double first_question_s;  /* total latency of question 1 */
    double later_total_s;     /* sum of total latency for questions 2..N */
    double later_avg_s;       /* average total latency for questions 2..N */
    int    later_count;
    double total_generate_s;  /* sum of full per-question times */
    double total_decode_s;    /* sum of decode-only times */
    double decode_avg_tps;    /* tokens / total_decode_s */
    long long total_tokens;
    unsigned long long resident_bytes;
} KataliSessionStats;

typedef struct KataliSession KataliSession;

/* Open a backend and keep it resident. `backend_name` may be NULL for "gguf". */
KataliSession *katali_session_open(const char *backend_name,
                                   const KataliBackendOptions *opts,
                                   char *err, size_t err_cap);
void katali_session_close(KataliSession *s);

/* Ask one independent question. The KV state is reset first, so questions do
 * not contaminate each other and the model is never reloaded. Returns the
 * number of tokens generated, or <0 on error. */
int  katali_session_ask(KataliSession *s, const char *question,
                        char *out, size_t out_cap);

/* Cooperative cancellation, safe to call from a poller thread. */
void katali_session_cancel(KataliSession *s);

const KataliSessionStats *katali_session_stats(const KataliSession *s);
size_t      katali_session_resident_bytes(const KataliSession *s);
const char *katali_session_backend(const KataliSession *s);

#ifdef __cplusplus
}
#endif
#endif /* KATALI_SESSION_H */
