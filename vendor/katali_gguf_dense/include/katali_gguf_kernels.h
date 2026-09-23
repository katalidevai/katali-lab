/* Katali-GGUF dense CPU kernels — Apache-2.0
 *
 * Same math as katali2's ops.h (ops_rms_norm / ops_rope_* / ops_gqa_*), kept as
 * a separate deterministic implementation so Katali-GGUF is self-contained.
 */
#ifndef KATALI_GGUF_KERNELS_H
#define KATALI_GGUF_KERNELS_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void  katali_gguf_rmsnorm(const float *x, const float *w, size_t n, float eps, float *out);
void  katali_gguf_rmsnorm_inplace(float *x, const float *w, size_t n, float eps);

/* In-place RoPE over head-major q/k. Rotates the first `rope_dim` of each head;
 * pairs (i, i + rope_dim/2). rope_dim == 0 means head_dim. */
void  katali_gguf_rope(float *x, size_t n_heads, size_t head_dim,
                       size_t rope_dim, size_t pos, float theta);

/* Same rotation using caller-prepared cos/sin tables (rope_dim/2 entries each,
 * cos[i]/sin[i] for pair (i, i + rope_dim/2)). Lets the caller compute the
 * frequencies once per position instead of once per layer, without changing the
 * result. */
void  katali_gguf_rope_apply(float *x, size_t n_heads, size_t head_dim,
                             size_t rope_dim, const float *cos, const float *sin);

/* Softmax in place, numerically stable. Uses the AVX2 path when available. */
void  katali_gguf_softmax(float *x, size_t n);

/* The scalar reference/fallback used by the selftest and by builds or A/B runs
 * without the vectorized softmax. */
void  katali_gguf_softmax_scalar(float *x, size_t n);

/* Single-token grouped-query attention.
 * q: [n_q * head_dim].  k/v: [n_kv * kv_cap * head_dim], position t at
 * (head * kv_cap + t) * head_dim. Attends positions [0, kv_len). */
void  katali_gguf_gqa_decode(const float *q, const float *k, const float *v,
                             size_t kv_len, size_t kv_cap,
                             size_t n_q, size_t n_kv, size_t head_dim,
                             float *out);

/* Same, with a caller-owned score buffer of at least kv_len floats. Lets the
 * decode loop avoid a heap allocation per layer per token. */
void  katali_gguf_gqa_decode_scratch(const float *q, const float *k, const float *v,
                                     size_t kv_len, size_t kv_cap,
                                     size_t n_q, size_t n_kv, size_t head_dim,
                                     float *out, float *scores);

/* Parallel variant: query heads are partitioned across `n_threads` workers via
 * the existing worker pool (no second pool), so each concurrent worker needs its
 * own score buffer - `scores_pool` must hold at least
 * `n_threads * pool_stride` floats with `pool_stride >= kv_len`. Sharing one
 * buffer across workers corrupts output.
 *
 * Output is bit-identical to the serial entry point: heads are independent and
 * every head runs the identical kernels in the same order; only which thread
 * runs them changes. `scores` is still required as the serial fallback buffer.
 *
 * Falls back to the serial path when n_threads <= 1, when fewer than 2 query
 * heads are present, when no pool was supplied, or when kv_len is below
 * KATALI_GGUF_ATTN_MIN_KV (default 256) - below that threshold the dispatch
 * costs more than the work it would share. KATALI_GGUF_NO_ATTN_PARALLEL=1
 * forces the serial path for A/B. `cancel` is polled once per worker in the
 * parallel path; the serial fallback ignores it, as the serial entry point
 * always has (the model cancels at layer granularity). */
void  katali_gguf_gqa_decode_parallel(const float *q, const float *k, const float *v,
                                      size_t kv_len, size_t kv_cap,
                                      size_t n_q, size_t n_kv, size_t head_dim,
                                      float *out, float *scores,
                                      float *scores_pool, size_t pool_stride,
                                      int n_threads, volatile int *cancel);

/* Add a scaled residual: a[i] += b[i]. */
void  katali_gguf_add_inplace(float *a, const float *b, size_t n);

float katali_gguf_silu(float x);
float katali_gguf_sigmoid(float x);
float katali_gguf_gelu(float x);
int   katali_gguf_argmax(const float *x, size_t n);

/* Self-test of the kernels against closed-form expectations. */
int katali_gguf_kernels_selftest(void);

#ifdef __cplusplus
}
#endif
#endif /* KATALI_GGUF_KERNELS_H */
