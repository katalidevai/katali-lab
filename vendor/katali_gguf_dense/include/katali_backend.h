/* Katali backend interface — Apache-2.0
 *
 * Format-independent contract between Katali Route / Katali Forge and an
 * inference backend. The public shape is stable so `backend=eqs` and
 * `backend=gguf` are interchangeable to the router.
 */
#ifndef KATALI_BACKEND_H
#define KATALI_BACKEND_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct KataliBackend KataliBackend;

typedef struct {
    const char *model_path;
    int n_threads;       /* 0 = auto */
    int context_length;  /* 0 = model default */
    int max_tokens;
    int use_mmap;
    /* Extended Katali options (safe to leave zero-initialised). */
    int   temperature_milli; /* temperature * 1000; 0 = greedy */
    int   top_k;             /* 0 = off */
    int   top_p_milli;       /* top_p * 1000; 0 = off */
    int   repetition_penalty_milli; /* 0/1000 = off */
    int   seed;
    int   think;             /* 1 = allow thinking block, 0 = suppress */
    const char *chat_template; /* optional override */
} KataliBackendOptions;

typedef struct {
    int    (*open)(KataliBackend *, const KataliBackendOptions *);
    int    (*reset_state)(KataliBackend *);
    int    (*generate)(KataliBackend *, const char *prompt,
                       char *output, size_t output_cap);
    int    (*cancel)(KataliBackend *);
    size_t (*resident_bytes)(const KataliBackend *);
    void   (*close)(KataliBackend *);
} KataliBackendOps;

struct KataliBackend {
    const KataliBackendOps *ops;
    void *impl;
    char  name[32];
};

/* --- output-boundary hooks (Phase 5 preparation) -------------------------
 * A structured-output consumer (tool calling, JSON validation, Katali Forge)
 * registers a boundary callback and receives typed events at token
 * boundaries instead of parsing arbitrary prose. The hook is invoked with the
 * text accumulated for one segment; returning non-zero requests a stop, the
 * same contract as the router's structural stop predicate. */
typedef enum KataliOutputKind {
    KATALI_OUT_TEXT = 0,      /* free text                        */
    KATALI_OUT_JSON,          /* structured JSON object/array     */
    KATALI_OUT_TOOL_CALL,     /* a validated tool call            */
    KATALI_OUT_FINAL,         /* the final answer                 */
    KATALI_OUT_CLARIFICATION  /* a request for clarification      */
} KataliOutputKind;

typedef int (*KataliOutputBoundaryFn)(KataliOutputKind kind, const char *text,
                                      void *user);
const char *katali_output_kind_name(KataliOutputKind kind);

/* --- registry ------------------------------------------------------------ */

/* Returns the ops for a named backend ("eqs", "gguf"), or NULL. */
const KataliBackendOps *katali_backend_lookup(const char *name);
/* Register/replace a backend implementation. Katali Route uses this to plug in
 * the existing EQS engine without Katali-GGUF depending on it. Returns 0 on
 * success, -1 when the table is full. */
int katali_backend_register(const char *name, const KataliBackendOps *ops);
/* Convenience: create + open a backend handle. Returns NULL on failure. */
KataliBackend *katali_backend_create(const char *name,
                                     const KataliBackendOptions *opts);
void katali_backend_destroy(KataliBackend *b);

/* The Katali-GGUF backend implementation. */
const KataliBackendOps *katali_gguf_backend_ops(void);

/* --- GUI / diagnostics accessors for the GGUF backend -------------------- */
struct KataliGgufInfo;
struct KataliGgufStats;
/* Fill the metadata summary for the GUI model panel. Returns 0 on success,
 * -1 when `b` is not an open GGUF handle. */
int katali_gguf_backend_describe(const KataliBackend *b, struct KataliGgufInfo *out);
/* Live generation stats (load/prefill/decode/memory) for status and benchmark
 * displays. Returns NULL for non-GGUF handles. */
const struct KataliGgufStats *katali_gguf_backend_stats(const KataliBackend *b);
/* Backend name reported in GUI status strings. */
const char *katali_gguf_backend_name(void);

/* --- per-generation report (session / benchmark / GUI) -------------------
 * A generic mirror of the GGUF model report so callers that only hold a
 * KataliBackend do not need GGUF internals. */
typedef struct KataliBackendReport {
    double load_s;           /* model open (backend lifetime) */
    double tokenize_s;       /* last generation */
    double prefill_s;
    double ttft_s;           /* generate entry -> first generated token */
    double decode_s;
    double total_s;
    int    prompt_tokens;
    int    generated_tokens;
    double decode_tps;
    int    batched_prefill;  /* 1 when the last prefill was batched */
    int    model_loads;      /* 1 for a handle created by open */
    int    kv_resets;        /* cumulative reset_state() calls */
    double last_reset_s;     /* cost of the most recent reset */
    unsigned long long resident_bytes;
} KataliBackendReport;

/* Fill the report for the backend's most recent generate call. Returns 0 on
 * success, -1 when the backend does not provide one. */
int katali_backend_last_report(const KataliBackend *b, KataliBackendReport *out);

#ifdef __cplusplus
}
#endif
#endif /* KATALI_BACKEND_H */
