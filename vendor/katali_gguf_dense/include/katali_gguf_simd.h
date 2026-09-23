/* Katali-GGUF x86 SIMD kernels — Apache-2.0
 *
 * Runtime-dispatched AVX2/FMA kernels with a scalar fallback. The rest of the
 * engine only calls katali_ggml_vec_dot(); this header exposes the dispatch
 * probe plus one untyped entry point so no AVX2 type leaks into callers.
 *
 * Portability: on non-x86 builds everything compiles to stubs and
 * katali_ggml_simd_available() returns 0. Set KATALI_GGUF_NO_SIMD=1 in the
 * environment to force the scalar path (used for A/B measurement).
 */
#ifndef KATALI_GGUF_SIMD_H
#define KATALI_GGUF_SIMD_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 1 when an accelerated path is usable on this CPU. Cached after first call. */
int katali_ggml_simd_available(void);
/* "avx2+fma" or "scalar" (for diagnostics/benchmark output). */
const char *katali_ggml_simd_name(void);

/* Accelerated dot product. Returns 0 and writes *out when the type is handled,
 * non-zero when the caller must use its own path. Only call when
 * katali_ggml_simd_available() is non-zero. */
int katali_ggml_vec_dot_avx2(uint32_t type, const void *src,
                             const float *x, uint64_t n, float *out);

/* Dot `nr` weight rows (already F32) against one activation vector, sharing the
 * activation loads across the rows. This is the prefill matmul's inner step,
 * where the activation re-read traffic dominates, so reading x once for several
 * rows is the win. `nr` is 1, 2 or 4; each output is bit-identical to
 * katali_ggml_vec_dot(KGGML_F32, rows[r], x, n). Returns 0 when handled, non-zero
 * to ask the caller for its per-row path. */
int katali_ggml_f32_rows_dot_avx2(const float *const *rows, int nr,
                                  const float *x, uint64_t n, float *out);

/* Materialize one quantized weight row of `n` values into floats, for the
 * prefill path's row decode. Returns 0 when handled, -1 when the caller must
 * fall back. katali_ggml_dequant_rows() wraps this with the scalar reference and
 * both produce bit-identical output. */
int katali_ggml_dequant_rows_avx2(uint32_t type, const void *src, uint64_t n,
                                  float *out);

/* Attention primitives. `k`/`v` point at one KV head's base; positions are
 * contiguous with stride head_dim. Both return 0 when handled, non-zero to ask
 * the caller for its scalar fallback. */
int katali_ggml_attn_scores_avx2(const float *q, const float *k,
                                 size_t kv_len, size_t head_dim,
                                 float scale, float *scores);
int katali_ggml_attn_accum_avx2(const float *scores, size_t kv_len,
                                const float *v, size_t head_dim, float *out);

/* gate[i] = silu(gate[i]) * up[i], vectorized. Returns 0 when handled. */
int katali_ggml_silu_mul_avx2(float *gate, const float *up, size_t n);

/* Numerically stable softmax in place, vectorized (max-reduce, then
 * exp(x-max) and sum, then normalize). Returns 0 when handled, non-zero to ask
 * the caller for katali_gguf_softmax()'s scalar path. The max subtraction is
 * preserved exactly; the exp/sum pass uses the vector polynomial exp and lane
 * accumulation, so results agree with the scalar reference to ~1e-7 relative
 * rather than bit-exactly. */
int katali_ggml_softmax_avx2(float *x, size_t n);

/* Run matvec rows r in [row_begin, row_end) for one tensor type using the
 * accelerated kernels. One call per (matvec, worker) — the per-row work is a
 * direct kernel call that the compiler can inline, and no per-row dispatch
 * happens. Returns 0 when the type was handled, non-zero to ask the caller for
 * its generic path.
 *
 * `q4k_sums` is an optional precomputed buffer of KATALI_Q4K_SUMS_FLOATS(cols)
 * floats holding the Q4_K activation-group sums for `x` (see
 * katali_ggml_q4k_sums). Pass NULL to fall back to the per-row recomputation. */
int katali_ggml_matvec_rows(uint32_t type, const void *w, uint64_t rows,
                            uint64_t cols, uint64_t row_bytes, const float *x,
                            float *y, uint64_t row_begin, uint64_t row_end,
                            volatile int *cancel, const float *q4k_sums);

/* --- Q4_K activation-sum hoisting ---------------------------------------
 * dot_q4_K_avx2 accumulates, per 256-value block, the two 8-lane partial sums
 * of the activation vector (`xL`/`xH`) used for the min-subtraction term. Those
 * sums depend only on `x`, not on the weight row, yet they were recomputed for
 * every row of a matvec. These helpers compute them once per matvec call.
 *
 * Floats needed: KATALI_Q4K_SUMS_FLOATS(cols). */
#define KATALI_Q4K_SUMS_FLOATS(cols) (((uint64_t)(cols) / 256u) * 64u)

/* Fill `sums` (capacity `sums_cap` floats). Returns 0 when handled; non-zero
 * when the caller must use the per-row path (no SIMD, disabled by
 * KATALI_GGUF_NO_Q4K_SUMCACHE=1, or unsupported cols). */
int katali_ggml_q4k_sums(const float *x, uint64_t cols, float *sums,
                         uint64_t sums_cap);

/* Single-row Q4_K dot using precomputed sums. Returns 0 when handled. Used by
 * the row loop and by the correctness test that compares it against the
 * unmodified dot_q4_K_avx2. */
int katali_ggml_q4k_dot_sums(const void *row, const float *x, uint64_t cols,
                             const float *sums, float *out);

/* Two-row Q4_K dot sharing the activation slices, for the row loop's dual-row
 * path. Returns 0 when handled; each output is bit-identical to the
 * corresponding katali_ggml_q4k_dot_sums call (asserted in test_core). */
int katali_ggml_q4k_dot_sums2(const void *row1, const void *row2, const float *x,
                              uint64_t cols, const float *sums,
                              float *out1, float *out2);

#ifdef __cplusplus
}
#endif
#endif /* KATALI_GGUF_SIMD_H */
