/* Katali-GGUF binary SDK — public ABI v1 (Apache-2.0)
 *
 * The ONLY header an embedding application needs. It exposes opaque handles and
 * plain data structures; no internal engine type, tensor layout, thread-pool or
 * allocator detail is visible, and the implementation source is not required to
 * build against it.
 *
 * Quick start (see examples/sdk_min.c):
 *
 *     katali_model_options mo = {0}; mo.struct_size = sizeof mo;
 *     mo.model_path = "Qwen3-0.6B-Q4_K_M.gguf";
 *     katali_model *m = katali_sdk_model_open(&mo);
 *     katali_session *s = katali_sdk_session_create(m, NULL);
 *     char answer[4096]; size_t n = 0;
 *     katali_sdk_generate(s, "What is 12 times 7?", answer, sizeof answer, &n);
 *     katali_sdk_session_destroy(s);
 *     katali_sdk_model_close(m);
 *
 * ---------------------------------------------------------------------------
 * Contract summary
 * ---------------------------------------------------------------------------
 * Ownership
 *   - model_open / session_create return SDK-owned handles; free them exactly
 *     once with model_close / session_destroy. Passing NULL to any destroy/free
 *     is a no-op.
 *   - Buffers you pass in (model_path, prompt, out) are yours and are used only
 *     for the duration of the call.
 *   - last_error() returns SDK-owned thread-local storage, valid until the next
 *     SDK call on the same thread; do not free it.
 *   - generate_alloc() returns SDK storage; release with string_free(). Nothing
 *     else the SDK returns needs freeing.
 *
 * Threading
 *   - Handles are not thread-safe for concurrent mutation. One generation may be
 *     in flight per session; a second concurrent call on the same session or its
 *     model returns KATALI_SDK_ERR_BUSY instead of interleaving output.
 *   - Generation on any session takes that model's lock, so sessions sharing one
 *     model are serialized (the KV cache belongs to the model). Sessions over
 *     different models are independent.
 *   - cancel() is the exception: it is safe from another thread while a
 *     generation runs.
 *   - The streaming callback runs on the generating thread, never concurrently.
 *
 * Cancellation and stopping
 *   - cancel() is cooperative: generation stops within one token and returns
 *     KATALI_SDK_ERR_CANCELLED. Handle, weights and KV state stay usable (the
 *     next generate resets the KV state anyway).
 *   - A streaming callback returning non-zero stops generation the same way and
 *     is reported as success with the partial text.
 *
 * Reset semantics
 *   - session_reset() drops the shared model's KV state, keeping weights loaded.
 *     Every generate resets the KV state first, so independent questions never
 *     contaminate each other and the model is never reloaded.
 *
 * Encoding
 *   - All strings are UTF-8. A streaming callback never receives an invalid
 *     partial character: the engine emits one token at a time and the SDK buffers
 *     incomplete multi-byte sequences, flushing the final tail at the end.
 *
 * Errors and versioning
 *   - Fallible calls return KATALI_SDK_OK or a KATALI_SDK_ERR_* code; last_error()
 *     gives the reason on the calling thread. Codes are stable within a major ABI
 *     version.
 *   - Structures carry struct_size: set it to sizeof(your struct) and the SDK
 *     touches only the fields that fit, so fields may be appended in a later minor
 *     version without breaking older callers. ABI_MAJOR changes only for
 *     incompatible changes.
 *   - Supported platform: Windows x86-64, AVX2+FMA with the scalar fallback.
 */
#ifndef KATALI_GGUF_SDK_H
#define KATALI_GGUF_SDK_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define KATALI_GGUF_SDK_ABI_MAJOR 1
#define KATALI_GGUF_SDK_ABI_MINOR 0
#define KATALI_GGUF_SDK_VERSION   "1.0.0"

#if defined(_WIN32)
#  ifdef KATALI_GGUF_SDK_BUILD
#    define KATALI_SDK_API __declspec(dllexport)
#  else
#    define KATALI_SDK_API __declspec(dllimport)
#  endif
#  define KATALI_SDK_CALL __cdecl
#else
#  define KATALI_SDK_API
#  define KATALI_SDK_CALL
#endif

/* Result codes. 0 is success; negative values are errors. */
enum {
    KATALI_SDK_OK            =  0,
    KATALI_SDK_ERR_ARG       = -1,  /* NULL or inconsistent argument          */
    KATALI_SDK_ERR_LOAD      = -2,  /* model open failed (see last_error)     */
    KATALI_SDK_ERR_STATE     = -3,  /* handle not open / already closed       */
    KATALI_SDK_ERR_GENERATE  = -4,  /* generation failed                      */
    KATALI_SDK_ERR_CANCELLED = -5,  /* cancelled via katali_sdk_cancel()      */
    KATALI_SDK_ERR_BUFFER    = -6,  /* output buffer too small for the answer */
    KATALI_SDK_ERR_BUSY      = -7,  /* a generation is already running here   */
    KATALI_SDK_ERR_INTERNAL  = -8   /* unexpected internal failure            */
};

/* Opaque handles. */
typedef struct katali_model   katali_model;
typedef struct katali_session katali_session;

/* --- version and errors -------------------------------------------------- */

/* ABI number the library was built with (compare to KATALI_GGUF_SDK_ABI_MAJOR). */
KATALI_SDK_API uint32_t KATALI_SDK_CALL katali_sdk_abi_version(void);
/* Human-readable version, e.g. "1.0.0". Static storage, never freed. */
KATALI_SDK_API const char *KATALI_SDK_CALL katali_sdk_version_string(void);
/* Why the most recent call on this thread failed. Never NULL. Thread-local. */
KATALI_SDK_API const char *KATALI_SDK_CALL katali_sdk_last_error(void);


/* --- model -------------------------------------------------------------- */

typedef struct katali_model_options {
    uint32_t    struct_size;    /* = sizeof(katali_model_options)             */
    const char *model_path;     /* local GGUF path, UTF-8; required           */
    int32_t     n_threads;      /* 0 = automatic                              */
    int32_t     context_length; /* 0 = model default; clamped to the model's  */
    int32_t     use_mmap;       /* non-zero = map the weights (default)       */
    int32_t     reserved;       /* must be 0                                  */
} katali_model_options;

typedef struct katali_model_info {
    uint32_t struct_size;       /* = sizeof(katali_model_info)                */
    char     arch[64];          /* e.g. "qwen3"                               */
    int32_t  n_layers;
    int32_t  hidden;
    int32_t  n_heads;
    int32_t  n_kv_heads;
    int32_t  head_dim;
    int32_t  ffn_dim;
    int32_t  vocab;
    int32_t  ctx_len;           /* trained context length                     */
    int32_t  kv_cap;            /* KV capacity actually allocated             */
    int32_t  tied;              /* embedding and LM head are the same tensor  */
    uint64_t resident_bytes;    /* weights + KV + scratch                     */
} katali_model_info;

/* Open a model from a local path. NULL on failure (see last_error). The weights
 * stay mapped for the handle's lifetime and are shared by its sessions. */
KATALI_SDK_API katali_model *KATALI_SDK_CALL
katali_sdk_model_open(const katali_model_options *opts);
/* Close a model handle. Destroy its sessions first. */
KATALI_SDK_API void KATALI_SDK_CALL katali_sdk_model_close(katali_model *m);
/* Metadata for status displays. Returns KATALI_SDK_OK or an error code. */
KATALI_SDK_API int32_t KATALI_SDK_CALL
katali_sdk_model_info(const katali_model *m, katali_model_info *out);

/* --- sessions ----------------------------------------------------------- */

typedef struct katali_gen_options {
    uint32_t struct_size;             /* = sizeof(katali_gen_options)        */
    int32_t  max_tokens;              /* 0 = SDK default (256)               */
    int32_t  temperature_milli;       /* temperature * 1000; 0 = greedy      */
    int32_t  top_k;                   /* 0 = off                             */
    int32_t  top_p_milli;             /* top_p * 1000; 0 = off               */
    int32_t  repetition_penalty_milli;/* 0 or 1000 = off                     */
    int32_t  seed;                    /* 0 = default                         */
    int32_t  think;                   /* 1 = allow thinking block            */
    const char *reserved_ptr;         /* must be NULL (v1)                   */
} katali_gen_options;

/* Create an independent session over an open model. `opts` may be NULL for
 * defaults. Sessions of one model are serialized (see the contract above). */
KATALI_SDK_API katali_session *KATALI_SDK_CALL
katali_sdk_session_create(katali_model *m, const katali_gen_options *opts);
/* Destroy a session. Generation on it must have finished. */
KATALI_SDK_API void KATALI_SDK_CALL katali_sdk_session_destroy(katali_session *s);
/* Drop the KV state, keeping the model loaded. */
KATALI_SDK_API int32_t KATALI_SDK_CALL katali_sdk_session_reset(katali_session *s);

/* --- generation --------------------------------------------------------- */

/* One-shot generation. The answer is written to `out` NUL-terminated (UTF-8),
 * with its length excluding the NUL stored to `out_len` when non-NULL. Returns
 * the number of tokens generated (>= 0) or an error code.
 * KATALI_SDK_ERR_BUFFER means the answer did not fit; the truncated text is
 * still valid and NUL-terminated, and its length is reported. */
KATALI_SDK_API int32_t KATALI_SDK_CALL
katali_sdk_generate(katali_session *s, const char *prompt,
                    char *out, size_t out_cap, size_t *out_len);

/* Convenience: generate into SDK-allocated storage, returned NUL-terminated.
 * Release with katali_sdk_string_free(). NULL on failure. */
KATALI_SDK_API char *KATALI_SDK_CALL
katali_sdk_generate_alloc(katali_session *s, const char *prompt, size_t *out_len);
KATALI_SDK_API void KATALI_SDK_CALL katali_sdk_string_free(char *s);

/* Streaming generation. `fn` is called on the generating thread with each UTF-8
 * chunk, never a partial multi-byte character. Returning non-zero stops
 * generation, which is reported as success. Returns tokens generated, or an
 * error code. Both `fn` and `s` must be non-NULL. */
typedef int32_t (KATALI_SDK_CALL *katali_chunk_fn)(const char *utf8, size_t len,
                                                   void *user);
KATALI_SDK_API int32_t KATALI_SDK_CALL
katali_sdk_generate_stream(katali_session *s, const char *prompt,
                           katali_chunk_fn fn, void *user);

/* Cooperative cancellation, safe from another thread. */
KATALI_SDK_API int32_t KATALI_SDK_CALL katali_sdk_cancel(katali_session *s);

/* --- statistics --------------------------------------------------------- */

typedef struct katali_sdk_stats {
    uint32_t struct_size;       /* = sizeof(katali_sdk_stats)                 */
    double   load_s;            /* model open (paid once per model handle)    */
    int32_t  model_loads;       /* 1 while the model is reused                */
    int32_t  questions;         /* generations answered by this session       */
    int32_t  kv_resets;         /* cumulative KV resets                       */
    /* most recent generation */
    double   last_tokenize_s;
    double   last_prefill_s;
    double   last_ttft_s;
    double   last_decode_s;
    double   last_total_s;
    double   last_decode_tps;
    int32_t  last_prompt_tokens;
    int32_t  last_generated_tokens;
    int32_t  last_batched_prefill;
    int32_t  last_stopped;      /* 1 = stopped by the stream callback         */
    int32_t  last_cancelled;    /* 1 = stopped by katali_sdk_cancel()         */
    uint64_t resident_bytes;    /* meters for the shared model                */
} katali_sdk_stats;

KATALI_SDK_API int32_t KATALI_SDK_CALL
katali_sdk_session_stats(const katali_session *s, katali_sdk_stats *out);

#ifdef __cplusplus
}
#endif
#endif /* KATALI_GGUF_SDK_H */

