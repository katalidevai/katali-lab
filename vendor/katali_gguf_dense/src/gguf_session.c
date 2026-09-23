/* Katali session — Apache-2.0 */
#include "katali_session.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct KataliSession {
    KataliBackend *b;
    char backend[32];
    KataliSessionStats st;
};

static unsigned long long backend_resident(const KataliBackend *b) {
    if (!b || !b->ops || !b->ops->resident_bytes) return 0;
    return (unsigned long long)b->ops->resident_bytes(b);
}

KataliSession *katali_session_open(const char *backend_name,
                                   const KataliBackendOptions *opts,
                                   char *err, size_t err_cap) {
    const char *name = backend_name ? backend_name : "gguf";
    if (!opts) {
        if (err && err_cap) snprintf(err, err_cap, "no backend options supplied");
        return NULL;
    }
    KataliBackend *b = katali_backend_create(name, opts);
    if (!b) {
        if (err && err_cap) snprintf(err, err_cap,
                                     "backend '%s' could not open %s", name,
                                     opts->model_path ? opts->model_path : "(null)");
        return NULL;
    }
    KataliSession *s = (KataliSession *)calloc(1, sizeof(*s));
    if (!s) { katali_backend_destroy(b); return NULL; }
    s->b = b;
    snprintf(s->backend, sizeof(s->backend), "%s", b->name);

    KataliBackendReport r;
    if (katali_backend_last_report(b, &r) == 0) {
        s->st.load_s = r.load_s;
        s->st.model_loads = r.model_loads;
    }
    s->st.resident_bytes = backend_resident(b);
    return s;
}

void katali_session_close(KataliSession *s) {
    if (!s) return;
    if (s->b) katali_backend_destroy(s->b);
    free(s);
}

int katali_session_ask(KataliSession *s, const char *question,
                       char *out, size_t out_cap) {
    if (!s || !s->b || !question || !out || out_cap == 0) return -1;

    /* Independent questions: drop the previous KV state first. The weights stay
     * mapped, so this is the only per-question setup cost. */
    if (s->b->ops && s->b->ops->reset_state) {
        if (s->b->ops->reset_state(s->b) != 0) return -1;
    }
    KataliBackendReport r;
    if (katali_backend_last_report(s->b, &r) == 0) {
        s->st.kv_resets = r.kv_resets;
        s->st.last_reset_s = r.last_reset_s;
    }

    int n = (s->b->ops && s->b->ops->generate)
                ? s->b->ops->generate(s->b, question, out, out_cap)
                : -1;

    if (katali_backend_last_report(s->b, &r) == 0) {
        s->st.last_tokenize_s = r.tokenize_s;
        s->st.last_prefill_s = r.prefill_s;
        s->st.last_ttft_s = r.ttft_s;
        s->st.last_decode_s = r.decode_s;
        s->st.last_total_s = r.total_s;
        s->st.last_prompt_tokens = r.prompt_tokens;
        s->st.last_generated_tokens = r.generated_tokens;
        s->st.last_decode_tps = r.decode_tps;
        s->st.last_batched_prefill = r.batched_prefill;
        s->st.model_loads = r.model_loads;
        s->st.kv_resets = r.kv_resets;
        s->st.resident_bytes = r.resident_bytes;
    }

    if (n >= 0) {
        s->st.questions++;
        if (s->st.questions == 1) {
            s->st.first_question_s = s->st.last_total_s;
        } else {
            s->st.later_total_s += s->st.last_total_s;
            s->st.later_count++;
            s->st.later_avg_s = s->st.later_total_s / (double)s->st.later_count;
        }
        s->st.total_generate_s += s->st.last_total_s;
        s->st.total_decode_s += s->st.last_decode_s;
        s->st.total_tokens += n;
        s->st.decode_avg_tps = s->st.total_decode_s > 0.0
                                   ? (double)s->st.total_tokens / s->st.total_decode_s
                                   : 0.0;
    }
    return n;
}

void katali_session_cancel(KataliSession *s) {
    if (!s || !s->b || !s->b->ops || !s->b->ops->cancel) return;
    s->b->ops->cancel(s->b);
}

const KataliSessionStats *katali_session_stats(const KataliSession *s) {
    return s ? &s->st : NULL;
}

size_t katali_session_resident_bytes(const KataliSession *s) {
    return s ? backend_resident(s->b) : 0;
}

const char *katali_session_backend(const KataliSession *s) {
    return s ? s->backend : "";
}
