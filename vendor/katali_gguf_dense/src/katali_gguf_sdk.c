/* Katali-GGUF binary SDK implementation — Apache-2.0
 *
 * Thin ABI layer. It wraps the private GGUF model object (the same one the CLI
 * and Katali Session use) behind the opaque handles declared in
 * include/katali_gguf_sdk.h. No private type is exposed to SDK consumers and no
 * model runtime is duplicated here: generation, reset, cancellation, sampling
 * and statistics all come from the engine.
 *
 * Threading: one CRITICAL_SECTION per model serializes generations, because the
 * KV cache belongs to the model; a per-session in-flight flag turns a second
 * concurrent call into KATALI_SDK_ERR_BUSY instead of interleaving output.
 */
#define KATALI_GGUF_SDK_BUILD 1
#include "katali_gguf_sdk.h"
#include "katali_gguf_model.h"

#include <windows.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define KATALI_MODEL_MAGIC   0x4B4D4F44u   /* 'KMOD' */
#define KATALI_SESSION_MAGIC 0x4B534553u   /* 'KSES' */
#define KATALI_SDK_DEFAULT_MAX_TOKENS 256
#define KATALI_SDK_ALLOC_SLACK 64

struct katali_model {
    uint32_t        magic;
    KataliGgufModel model;
    CRITICAL_SECTION lock;
    volatile LONG   busy;
    double          load_s;
    int             open;
};

struct katali_session {
    uint32_t      magic;
    katali_model *m;
    int  max_tokens;
    int  temperature_milli;
    int  top_k;
    int  top_p_milli;
    int  repetition_penalty_milli;
    int  seed;
    int  think;
    unsigned long long questions;
    unsigned long long kv_resets;
    int  last_stopped;
    int  last_cancelled;
};

/* Last error is thread-local and SDK-owned; it stays valid until the next SDK
 * call on the same thread. */
static __thread char g_err[512];

static const char *set_err(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_err, sizeof(g_err), fmt, ap);
    va_end(ap);
    return g_err;
}

static int32_t fail(int32_t code, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_err, sizeof(g_err), fmt, ap);
    va_end(ap);
    return code;
}

uint32_t katali_sdk_abi_version(void) { return KATALI_GGUF_SDK_ABI_MAJOR; }
const char *katali_sdk_version_string(void) { return KATALI_GGUF_SDK_VERSION; }
const char *katali_sdk_last_error(void) { return g_err[0] ? g_err : "ok"; }

/* Copying helper that respects a caller's struct_size field: only the fields the
 * caller declares are touched, so later minor versions may append fields. */
static size_t copy_out(void *dst, const void *src, size_t dst_size) {
    memcpy(dst, src, dst_size);
    return dst_size;
}


/* ==================================================================== *
 * Model handle                                                          *
 * ==================================================================== */

katali_model *katali_sdk_model_open(const katali_model_options *opts) {
    if (!opts || !opts->model_path || !opts->model_path[0]) {
        set_err("model_path is required");
        return NULL;
    }
    if (opts->struct_size < sizeof(katali_model_options)) {
        set_err("model_options.struct_size = %u is smaller than this SDK expects (%u)",
                (unsigned)opts->struct_size, (unsigned)sizeof(katali_model_options));
        return NULL;
    }
    if (opts->reserved != 0) {
        set_err("model_options.reserved must be 0");
        return NULL;
    }

    katali_model *m = (katali_model *)calloc(1, sizeof(*m));
    if (!m) { set_err("out of memory for the model handle"); return NULL; }
    m->magic = KATALI_MODEL_MAGIC;
    InitializeCriticalSection(&m->lock);

    KataliBackendOptions bo;
    memset(&bo, 0, sizeof(bo));
    bo.model_path = opts->model_path;
    bo.n_threads = opts->n_threads > 0 ? opts->n_threads : 0;
    bo.context_length = opts->context_length > 0 ? opts->context_length : 0;
    bo.max_tokens = KATALI_SDK_DEFAULT_MAX_TOKENS;
    bo.use_mmap = 1;
    if (opts->use_mmap == 0) bo.use_mmap = 0;   /* 0 is only honoured when explicit */
    bo.temperature_milli = 0;                   /* greedy default */
    bo.top_p_milli = 0;
    bo.think = 0;

    char err[512];
    err[0] = '\0';
    LARGE_INTEGER f, c0, c1;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c0);
    if (katali_gguf_model_open(&m->model, &bo, err, sizeof(err)) != 0) {
        set_err("model open failed: %s", err[0] ? err : "unspecified");
        DeleteCriticalSection(&m->lock);
        m->magic = 0;
        free(m);
        return NULL;
    }
    QueryPerformanceCounter(&c1);
    m->load_s = (double)(c1.QuadPart - c0.QuadPart) / (double)f.QuadPart;
    m->open = 1;
    g_err[0] = '\0';
    return m;
}

void katali_sdk_model_close(katali_model *m) {
    if (!m) return;
    if (m->magic != KATALI_MODEL_MAGIC) return;   /* already closed / not ours */
    if (m->open) katali_gguf_model_close(&m->model);
    m->open = 0;
    m->magic = 0;
    DeleteCriticalSection(&m->lock);
    free(m);
}

int32_t katali_sdk_model_info(const katali_model *m, katali_model_info *out) {
    if (!m || m->magic != KATALI_MODEL_MAGIC)
        return fail(KATALI_SDK_ERR_STATE, "model handle is not open");
    if (!out) return fail(KATALI_SDK_ERR_ARG, "out is NULL");
    if (out->struct_size < sizeof(uint32_t))
        return fail(KATALI_SDK_ERR_ARG, "model_info.struct_size is not set");
    katali_model_info info;
    memset(&info, 0, sizeof(info));
    info.struct_size = (uint32_t)sizeof(info);
    snprintf(info.arch, sizeof(info.arch), "%s", m->model.arch);
    info.n_layers = m->model.n_layers;
    info.hidden = m->model.hidden;
    info.n_heads = m->model.n_heads;
    info.n_kv_heads = m->model.n_kv_heads;
    info.head_dim = m->model.head_dim;
    info.ffn_dim = m->model.ffn_dim;
    info.vocab = m->model.vocab;
    info.ctx_len = m->model.ctx_len;
    info.kv_cap = m->model.kv_cap;
    info.tied = m->model.tied;
    info.resident_bytes = (uint64_t)katali_gguf_model_resident_bytes(&m->model);
    copy_out(out, &info, out->struct_size < sizeof(info) ? out->struct_size : sizeof(info));
    return KATALI_SDK_OK;
}

/* ==================================================================== *
 * Session handle                                                        *
 * ==================================================================== */

katali_session *katali_sdk_session_create(katali_model *m, const katali_gen_options *opts) {
    if (!m || m->magic != KATALI_MODEL_MAGIC) {
        set_err("model handle is not open");
        return NULL;
    }
    if (opts && opts->struct_size < sizeof(katali_gen_options)) {
        set_err("gen_options.struct_size = %u is smaller than this SDK expects (%u)",
                (unsigned)opts->struct_size, (unsigned)sizeof(katali_gen_options));
        return NULL;
    }
    if (opts && opts->reserved_ptr) {
        set_err("gen_options.reserved_ptr must be NULL");
        return NULL;
    }
    katali_session *s = (katali_session *)calloc(1, sizeof(*s));
    if (!s) { set_err("out of memory for the session handle"); return NULL; }
    s->magic = KATALI_SESSION_MAGIC;
    s->m = m;
    s->max_tokens = KATALI_SDK_DEFAULT_MAX_TOKENS;
    if (opts) {
        if (opts->max_tokens > 0) s->max_tokens = opts->max_tokens;
        s->temperature_milli = opts->temperature_milli;
        s->top_k = opts->top_k;
        s->top_p_milli = opts->top_p_milli;
        s->repetition_penalty_milli = opts->repetition_penalty_milli;
        s->seed = opts->seed;
        s->think = opts->think;
    }
    g_err[0] = '\0';
    return s;
}

void katali_sdk_session_destroy(katali_session *s) {
    if (!s || s->magic != KATALI_SESSION_MAGIC) return;
    s->magic = 0;
    s->m = NULL;
    free(s);
}

int32_t katali_sdk_session_reset(katali_session *s) {
    if (!s || s->magic != KATALI_SESSION_MAGIC)
        return fail(KATALI_SDK_ERR_STATE, "session handle is not valid");
    katali_model *m = s->m;
    if (!m || m->magic != KATALI_MODEL_MAGIC)
        return fail(KATALI_SDK_ERR_STATE, "model handle is not open");
    if (m->busy)
        return fail(KATALI_SDK_ERR_BUSY, "a generation is running on this model");
    EnterCriticalSection(&m->lock);
    katali_gguf_model_reset(&m->model);
    LeaveCriticalSection(&m->lock);
    s->kv_resets++;
    return KATALI_SDK_OK;
}

int32_t katali_sdk_cancel(katali_session *s) {
    if (!s || s->magic != KATALI_SESSION_MAGIC)
        return fail(KATALI_SDK_ERR_ARG, "session handle is not valid");
    katali_model *m = s->m;
    if (!m || m->magic != KATALI_MODEL_MAGIC)
        return fail(KATALI_SDK_ERR_ARG, "model handle is not open");
    /* Deliberately lock-free: this is the one call allowed from another thread
     * while a generation is in flight, and the flag is a single volatile word. */
    katali_gguf_model_cancel(&m->model);
    return KATALI_SDK_OK;
}


/* ==================================================================== *
 * Generation and streaming                                              *
 * ==================================================================== */

/* UTF-8 chunker: the engine hands over one token's bytes at a time, which may
 * split a multi-byte character. Bytes accumulate until the sequence starting at
 * pend[0] is complete, so a callback never sees a partial character. Any tail
 * left when generation ends is flushed by the caller. */
typedef struct StreamCtx {
    katali_chunk_fn fn;
    void  *user;
    char   pend[4];
    size_t npend;
    int    stop;         /* non-zero once the callback asked to stop */
} StreamCtx;

static size_t utf8_seq_len(unsigned char c) {
    if (c < 0x80) return 1;
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return 4;
    return 1;            /* invalid lead byte: forward it as-is */
}

static int on_token_cb(const char *bytes, size_t len, void *user) {
    StreamCtx *st = (StreamCtx *)user;
    if (!bytes || len == 0) return 0;
    for (size_t i = 0; i < len; i++) {
        if (st->npend < sizeof(st->pend)) st->pend[st->npend++] = bytes[i];
        size_t need = utf8_seq_len((unsigned char)st->pend[0]);
        if (st->npend >= need) {
            int32_t rc = st->fn(st->pend, need, st->user);
            st->npend = 0;
            if (rc) { st->stop = 1; return 1; }
        }
    }
    return 0;
}

/* The one generation path; fn == NULL means one-shot. */
static int32_t run_generate(katali_session *s, const char *prompt,
                            char *out, size_t cap, size_t *out_len,
                            katali_chunk_fn fn, void *user) {
    if (!s || s->magic != KATALI_SESSION_MAGIC)
        return fail(KATALI_SDK_ERR_STATE, "session handle is not valid");
    if (!prompt) return fail(KATALI_SDK_ERR_ARG, "prompt is NULL");
    katali_model *m = s->m;
    if (!m || m->magic != KATALI_MODEL_MAGIC)
        return fail(KATALI_SDK_ERR_STATE, "model handle is not open");
    if (!out || cap < 2)
        return fail(KATALI_SDK_ERR_ARG, "out buffer is NULL or too small");
    if (InterlockedCompareExchange(&m->busy, 1, 0) != 0)
        return fail(KATALI_SDK_ERR_BUSY,
                    "another generation is already running on this model");

    StreamCtx st;
    memset(&st, 0, sizeof(st));
    st.fn = fn;
    st.user = user;

    EnterCriticalSection(&m->lock);
    /* Per-session sampling and limits are applied on a clean sampler state, then
     * the KV state is dropped so independent questions cannot contaminate each
     * other. The weights are never reloaded. */
    katali_gguf_sampler_init(&m->model.sampler);
    m->model.sampler.temperature = (float)s->temperature_milli / 1000.0f;
    m->model.sampler.top_k = s->top_k;
    m->model.sampler.top_p = s->top_p_milli > 0 ? (float)s->top_p_milli / 1000.0f : 1.0f;
    m->model.sampler.repetition_penalty =
        s->repetition_penalty_milli > 0
            ? (float)s->repetition_penalty_milli / 1000.0f : 1.0f;
    m->model.sampler.seed = s->seed ? (unsigned)s->seed : 1u;
    m->model.max_tokens = s->max_tokens;
    m->model.think = s->think;
    katali_gguf_model_reset(&m->model);
    s->kv_resets++;
    if (fn) katali_gguf_model_set_token_callback(&m->model, on_token_cb, &st);

    int gen = katali_gguf_model_generate(&m->model, prompt, out, cap);

    if (fn) {
        katali_gguf_model_set_token_callback(&m->model, NULL, NULL);
        if (st.npend > 0) {              /* flush an incomplete trailing tail */
            (void)fn(st.pend, st.npend, st.user);
            st.npend = 0;
        }
    }
    int stopped = katali_gguf_model_stopped_by_callback(&m->model);
    int cancelled = katali_gguf_model_cancelled(&m->model);
    LeaveCriticalSection(&m->lock);
    InterlockedExchange(&m->busy, 0);

    s->questions++;
    s->last_stopped = stopped;
    s->last_cancelled = cancelled;

    size_t used = strlen(out);
    if (out_len) *out_len = used;
    if (gen < 0) return fail(KATALI_SDK_ERR_GENERATE, "generation failed (engine code %d)", gen);
    if (cancelled) {
        set_err("cancelled after %d token(s)", gen);
        return KATALI_SDK_ERR_CANCELLED;
    }
    /* The engine stops at the buffer's last byte when the answer does not fit. */
    if (used + 1 >= cap) {
        set_err("answer did not fit in %llu bytes (truncated after %d token(s))",
                (unsigned long long)cap, gen);
        return KATALI_SDK_ERR_BUFFER;
    }
    g_err[0] = '\0';
    return (int32_t)gen;
}


int32_t katali_sdk_generate(katali_session *s, const char *prompt,
                            char *out, size_t out_cap, size_t *out_len) {
    return run_generate(s, prompt, out, out_cap, out_len, NULL, NULL);
}

int32_t katali_sdk_generate_stream(katali_session *s, const char *prompt,
                                   katali_chunk_fn fn, void *user) {
    if (!fn) return fail(KATALI_SDK_ERR_ARG, "streaming callback is NULL");
    /* The engine materializes the answer in a buffer, so the SDK supplies one and
     * a streaming caller does not have to size it. */
    int max_tokens = (s && s->magic == KATALI_SESSION_MAGIC)
                         ? s->max_tokens : KATALI_SDK_DEFAULT_MAX_TOKENS;
    size_t cap = (size_t)max_tokens * 8u + KATALI_SDK_ALLOC_SLACK;
    if (cap < 1024) cap = 1024;
    char *buf = (char *)malloc(cap);
    if (!buf) return fail(KATALI_SDK_ERR_INTERNAL, "out of memory for the stream buffer");
    size_t used = 0;
    int32_t rc = run_generate(s, prompt, buf, cap, &used, fn, user);
    free(buf);
    if (rc == KATALI_SDK_ERR_BUFFER) rc = 0;   /* streaming callers never see this */
    return rc;
}

char *katali_sdk_generate_alloc(katali_session *s, const char *prompt, size_t *out_len) {
    if (!s || s->magic != KATALI_SESSION_MAGIC) {
        set_err("session handle is not valid");
        return NULL;
    }
    size_t cap = (size_t)s->max_tokens * 8u + KATALI_SDK_ALLOC_SLACK;
    if (cap < 1024) cap = 1024;
    for (int attempt = 0; attempt < 4; attempt++) {
        char *buf = (char *)malloc(cap);
        if (!buf) { set_err("out of memory for the answer"); return NULL; }
        size_t used = 0;
        int32_t rc = run_generate(s, prompt, buf, cap, &used, NULL, NULL);
        if (rc == KATALI_SDK_ERR_BUFFER) {     /* grow and try again */
            free(buf);
            cap *= 2;
            continue;
        }
        if (rc < 0) {
            char msg[256];
            snprintf(msg, sizeof(msg), "%s", g_err);
            free(buf);
            set_err("%s", msg);
            return NULL;
        }
        if (out_len) *out_len = used;
        return buf;
    }
    set_err("answer exceeded %llu bytes", (unsigned long long)cap);
    return NULL;
}

void katali_sdk_string_free(char *s) {
    if (s) free(s);
}

int32_t katali_sdk_session_stats(const katali_session *s, katali_sdk_stats *out) {
    if (!s || s->magic != KATALI_SESSION_MAGIC)
        return fail(KATALI_SDK_ERR_STATE, "session handle is not valid");
    if (!out || out->struct_size < sizeof(uint32_t))
        return fail(KATALI_SDK_ERR_ARG, "stats.struct_size is not set");
    const katali_model *m = s->m;
    if (!m || m->magic != KATALI_MODEL_MAGIC)
        return fail(KATALI_SDK_ERR_STATE, "model handle is not open");
    KataliGgufGenReport rep;
    katali_gguf_model_last_report(&m->model, &rep);
    katali_sdk_stats st;
    memset(&st, 0, sizeof(st));
    st.struct_size = (uint32_t)sizeof(st);
    st.load_s = m->load_s;
    st.model_loads = 1;                 /* the model is opened once per handle */
    st.questions = (int32_t)s->questions;
    st.kv_resets = (int32_t)s->kv_resets;
    st.last_tokenize_s = rep.tokenize_s;
    st.last_prefill_s = rep.prefill_s;
    st.last_ttft_s = rep.ttft_s;
    st.last_decode_s = rep.decode_s;
    st.last_total_s = rep.total_s;
    st.last_decode_tps = rep.decode_tps;
    st.last_prompt_tokens = rep.prompt_tokens;
    st.last_generated_tokens = rep.generated_tokens;
    st.last_batched_prefill = rep.batched_prefill;
    st.last_stopped = s->last_stopped;
    st.last_cancelled = s->last_cancelled;
    st.resident_bytes = (uint64_t)katali_gguf_model_resident_bytes(&m->model);
    copy_out(out, &st, out->struct_size < sizeof(st) ? out->struct_size : sizeof(st));
    return KATALI_SDK_OK;
}

