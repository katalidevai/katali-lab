/* Katali-GGUF tensor type + quantization decoding — Apache-2.0
 *
 * Reference decoders are the source of truth. The accelerated vec_dot path is
 * only allowed to be used once the reference and the optimized output agree;
 * tests/test_dtype.c asserts exactly that.
 */
#ifndef KATALI_GGUF_DTYPE_H
#define KATALI_GGUF_DTYPE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ggml tensor type ids (subset needed by Phase 1/2, matching GGUF). */
enum {
    KGGML_F32  = 0,
    KGGML_F16  = 1,
    KGGML_Q4_0 = 2,
    KGGML_Q4_1 = 3,
    KGGML_Q5_0 = 6,
    KGGML_Q5_1 = 7,
    KGGML_Q8_0 = 8,
    KGGML_Q8_1 = 9,
    KGGML_Q2_K = 10,
    KGGML_Q3_K = 11,
    KGGML_Q4_K = 12,
    KGGML_Q5_K = 13,
    KGGML_Q6_K = 14,
    KGGML_Q8_K = 15,
    KGGML_I8   = 24,
    KGGML_I16  = 25,
    KGGML_I32  = 26,
    KGGML_I64  = 27,
    KGGML_F64  = 28,
    KGGML_BF16 = 30
};

const char *katali_ggml_type_name(uint32_t type);
int         katali_ggml_type_supported(uint32_t type);
int         katali_ggml_type_is_quant(uint32_t type);
uint64_t    katali_ggml_type_block_size(uint32_t type);  /* elements per block */
uint64_t    katali_ggml_type_block_bytes(uint32_t type); /* bytes per block */

/* Bytes needed to store `n` elements of `type` (0 when unsupported). */
uint64_t    katali_ggml_row_bytes(uint32_t type, uint64_t n);

/* Reference: decode `n` elements (a multiple of the block size) into floats.
 * Returns 0 on success, -1 on unsupported type or misaligned n. */
int katali_ggml_dequant_ref(uint32_t type, const void *src, uint64_t n, float *out);

/* Same contract, but allowed to use the AVX2 row-materializing decoder when one
 * exists for the type; falls back to the reference otherwise. Output is
 * bit-identical to katali_ggml_dequant_ref either way (asserted in test_core).
 * This is what the prefill matmul uses for its row decode. */
int katali_ggml_dequant_rows(uint32_t type, const void *src, uint64_t n, float *out);

/* Reference: dot(dequant(src), x) over `n` elements. */
float katali_ggml_vec_dot_ref(uint32_t type, const void *src,
                              const float *x, uint64_t n);

/* Optimized scalar/blocked dot. Must match the reference exactly enough for the
 * test tolerance; falls back to the reference for exotic types. */
float katali_ggml_vec_dot(uint32_t type, const void *src,
                          const float *x, uint64_t n);

/* Dot `nr` F32 weight rows against one activation vector, sharing the activation
 * loads. This is the prefill matmul's inner step. Each output is bit-identical to
 * calling katali_ggml_vec_dot(KGGML_F32, rows[r], x, n) individually, on the
 * accelerated path and on the scalar fallback alike. */
void katali_ggml_f32_rows_dot(const float *const *rows, int nr,
                              const float *x, uint64_t n, float *out);

/* The scalar/blocked path without SIMD dispatch. Exposed so tests and the
 * kernel benchmark can compare the accelerated path against it directly. */
float katali_ggml_vec_dot_scalar(uint32_t type, const void *src,
                                 const float *x, uint64_t n);

/* y[r] = dot(row r of W, x), where W is a GGUF tensor with dims[0] = cols
 * (contiguous) and dims[1] = rows. Multithreaded across rows; a non-NULL
 * `cancel` is polled between rows.
 *
 * `q4k_scratch`/`q4k_scratch_cap` are an optional caller-owned buffer used to
 * precompute the Q4_K activation-group sums once per call instead of once per
 * row (see katali_ggml_q4k_sums). Pass NULL/0 to keep the previous behaviour;
 * the buffer must hold at least KATALI_Q4K_SUMS_FLOATS(cols) floats. It is
 * sized once by the caller (model open), never allocated here. */
void katali_ggml_matvec(uint32_t type, const void *w,
                        uint64_t rows, uint64_t cols,
                        const float *x, float *y,
                        int n_threads, volatile int *cancel,
                        float *q4k_scratch, uint64_t q4k_scratch_cap);

/* Same, additionally attributing the work to `role` in the per-(role,type)
 * accounting table. The model uses this for every projection so the kernel
 * contribution can be read off per role and per quantization. */
void katali_ggml_matvec_role(uint32_t type, const void *w,
                             uint64_t rows, uint64_t cols,
                             const float *x, float *y,
                             int n_threads, volatile int *cancel,
                             float *q4k_scratch, uint64_t q4k_scratch_cap,
                             int role);

/* Batched matrix-multiply: Y[t][r] = dot(W row r, X[t]) for t in [0,B).
 *
 * W is a GGUF matrix with dims[0]=cols (contiguous) and `rows` rows.
 * X is [B][cols] row-major; Y is [B][rows] row-major.
 *
 * Each weight row is decoded once and reused for all B rows of X, so the
 * quantized unpacking cost is paid once per row instead of once per (row,token).
 * This is what makes prompt prefill cheaper than B separate matvecs; the
 * token-by-token decode path keeps using katali_ggml_matvec(). */
void katali_ggml_matmul(uint32_t type, const void *w,
                        uint64_t rows, uint64_t cols,
                        const float *X, uint64_t B, float *Y,
                        int n_threads, volatile int *cancel);

/* Same, additionally attributing the work to `role`. */
void katali_ggml_matmul_role(uint32_t type, const void *w,
                             uint64_t rows, uint64_t cols,
                             const float *X, uint64_t B, float *Y,
                             int n_threads, volatile int *cancel, int role);

/* Role of a matvec/matmul for per-role kernel accounting. The caller (the model)
 * passes these explicitly, keyed by tensor identity - never by tensor name,
 * because `output.weight` may be absent and the LM head may then be the tied
 * `token_embd.weight` instead (see m->out_w / m->tied). */
typedef enum KataliMatvecRole {
    KATALI_ROLE_OTHER = 0,
    KATALI_ROLE_LM_HEAD,   /* m->out_w: the final logits matvec */
    KATALI_ROLE_WQ,
    KATALI_ROLE_WK,
    KATALI_ROLE_WV,
    KATALI_ROLE_WO,
    KATALI_ROLE_GATE,
    KATALI_ROLE_UP,
    KATALI_ROLE_DOWN,
    KATALI_ROLE_COUNT
} KataliMatvecRole;

const char *katali_ggml_role_name(int role);

/* --- per-(role, type) kernel accounting ---------------------------------
 * The global profiler below sums every matvec into one bucket, which cannot
 * separate e.g. the LM head (151936 rows, issued once per token) from a dozen
 * small per-layer projections, nor Q4_K from Q6_K within the same role. These
 * accessors expose the combinations that actually occurred, so a contribution
 * can be read off by role and by quantization. Reset by katali_ggml_prof_reset(). */
int  katali_ggml_prof_role_count(void);
void katali_ggml_prof_role_get(int i, int *role, uint32_t *type,
                               unsigned long long *calls, double *seconds,
                               unsigned long long *bytes, unsigned long long *flops);
/* 1 when slot i was produced by the batched prefill matmul, 0 for the
 * token-by-token path, so the two can be reported separately. */
int  katali_ggml_prof_role_is_prefill(int i);

/* One matvec job for katali_ggml_matvec_fused(). */
typedef struct KataliMatvecJob {
    uint32_t     type;   /* GGML tensor type of w */
    const void  *w;      /* GGUF matrix, dims[0] = cols, dims[1] = rows */
    uint64_t     rows;
    float       *y;      /* receives `rows` outputs */
    int          role;   /* KataliMatvecRole, for per-role accounting */
} KataliMatvecJob;

/* Fused multi-matvec: computes every job's y[] for the same `x` in ONE parallel
 * dispatch. All jobs must share `cols` (the caller checks) and `x`.
 *
 * The row partition is taken over the concatenation of the jobs' rows, so the
 * dispatch's wake and tail costs are paid once for the whole group instead of
 * once per job; running k jobs separately costs k round-trips and k tails.
 * Every row is still produced by exactly the same per-row kernel with the same
 * inputs, and rows are independent, so the output is bit-identical to calling
 * katali_ggml_matvec() once per job. The caller must keep `jobs` and the
 * tensors/vectors it points at alive for the duration of the call.
 *
 * `q4k_scratch` is the same optional caller-owned Q4_K activation-sum buffer as
 * katali_ggml_matvec(); one buffer serves every job because the sums depend only
 * on `x` and `cols`. n_threads <= 1 runs the whole group inline. */
void katali_ggml_matvec_fused(const KataliMatvecJob *jobs, int n_jobs,
                              uint64_t cols, const float *x,
                              int n_threads, volatile int *cancel,
                              float *q4k_scratch, uint64_t q4k_scratch_cap);

/* y = W_f32 [rows,cols] @ x, using a plain float matrix with GGUF layout. */
void katali_f32_matvec(const float *w, uint64_t rows, uint64_t cols,
                       const float *x, float *y, int n_threads);

/* Self-test: reference vs optimized agreement for every supported type. */
int katali_dtype_selftest(void);

/* Self-test: accelerated path vs scalar path vs reference for the types that
 * have a vector kernel. Reports the active simd name. */
int katali_dtype_simd_selftest(void);

/* --- lightweight kernel profiler -----------------------------------------
 * Accumulates matvec work so `benchmark` can separate kernel time from
 * model/threading overhead. Enabled always (one timer read per matvec call,
 * ~250 calls per token); reset before a measured run. */
void katali_ggml_prof_reset(void);
void katali_ggml_prof_stats(unsigned long long *calls, double *seconds,
                            unsigned long long *bytes, unsigned long long *flops);

#ifdef __cplusplus
}
#endif
#endif /* KATALI_GGUF_DTYPE_H */
