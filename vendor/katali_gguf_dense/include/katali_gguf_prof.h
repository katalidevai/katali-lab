/* Katali-GGUF phase profiler — Apache-2.0
 *
 * A tiny, allocation-free accumulator that attributes generation time to
 * phases (matvec, attention, RMSNorm, norm-weight dequantization, RoPE, KV
 * store, activations, sampling, thread synchronisation, tokenization) and
 * counts calls. It exists so optimisations are chosen from measurements
 * instead of guesses, and so `benchmark` can report where time actually goes.
 *
 * Overhead is one QueryPerformanceCounter pair per phase group per layer
 * (~200 pairs per decoded token); set KATALI_GGUF_NO_PROF=1 to disable it for
 * pure-speed runs.
 */
#ifndef KATALI_GGUF_PROF_H
#define KATALI_GGUF_PROF_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum KataliPhase {
    KATALI_PHASE_MATVEC = 0,   /* all matvec/matmul wall time (includes SYNC) */
    KATALI_PHASE_SYNC,         /* caller wait inside parallel dispatch (subset of MATVEC) */
    KATALI_PHASE_ATTENTION,
    KATALI_PHASE_RMSNORM,
    KATALI_PHASE_NORMDQ,       /* norm/bias weight dequantization */
    KATALI_PHASE_ROPE,
    KATALI_PHASE_KV,           /* KV store/copy */
    KATALI_PHASE_ACTIVATION,   /* silu(gate)*up */
    KATALI_PHASE_SAMPLER,
    KATALI_PHASE_TOKENIZE,
    KATALI_PHASE_PREFILL,      /* whole prefill (informational) */
    /* Attention decomposition: subsets of KATALI_PHASE_ATTENTION, measured inside
     * katali_gguf_gqa_decode_scratch() per head. Scaling is folded into the
     * scores kernel (it takes `scale`); decode has no masking; "output assembly"
     * is the single memset at the top of the call, which is included in
     * KATALI_PHASE_ATTENTION but not in any of these three. */
    KATALI_PHASE_ATTN_SCORES,  /* q.k dot products (scale applied inside) */
    KATALI_PHASE_ATTN_SOFTMAX, /* max-reduce + exp/sum + normalize */
    KATALI_PHASE_ATTN_ACCUM,   /* weighted sum of V rows */
    KATALI_PHASE_EMBED,        /* token-embedding row dequantization (O(hidden), not a matvec) */
    KATALI_PHASE_COUNT
} KataliPhase;

/* 1 unless KATALI_GGUF_NO_PROF=1. */
int    katali_prof_enabled(void);
void   katali_prof_reset(void);
void   katali_prof_add(int phase, double seconds);
void   katali_prof_inc(int phase);                  /* count one call/event */
void   katali_prof_add_n(int phase, double seconds, unsigned long long calls);
double katali_prof_seconds(int phase);
unsigned long long katali_prof_calls(int phase);
const char *katali_prof_name(int phase);

/* Bytes streamed by a phase, so a memory-bound phase can be reported as
 * achieved bandwidth instead of only as time. Attention uses it to separate the
 * K stream (read by the score dot) from the V stream (read by the weighted
 * accumulation): with GQA the same KV head is logically read once per query
 * head that maps to it, so these counters are the traffic *before* any reuse.
 * Kept separate from `calls` because a phase's bytes and its call count have
 * nothing to do with each other. Zero means "not tracked" and is not printed. */
void   katali_prof_add_bytes(int phase, unsigned long long bytes);
unsigned long long katali_prof_bytes(int phase);

/* Timestamp helper for callers that want to time a region themselves. */
double katali_prof_now(void);

#ifdef __cplusplus
}
#endif
#endif /* KATALI_GGUF_PROF_H */
