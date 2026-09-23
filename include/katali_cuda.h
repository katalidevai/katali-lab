/* katali-lab — optional CUDA backend ABI + runtime loader — Apache-2.0
 *
 * CUDA is OPTIONAL. Katali-lab is built by MinGW gcc (build.bat) and never links
 * against any CUDA library. The CUDA backend is a separate DLL
 * (katali_cuda.dll) built by nvcc + MSVC, because nvcc on Windows requires
 * cl.exe and cannot use a MinGW host compiler. It is discovered at runtime with
 * LoadLibrary/GetProcAddress, so a machine with no NVIDIA GPU or no CUDA runtime
 * keeps running the CPU + system RAM + SSD path unchanged.
 *
 * The DLL exports a flat `extern "C"` ABI of plain C types only (no C++ types,
 * no caller-allocated STL, no exceptions across the boundary), so a MinGW-built
 * caller can drive an MSVC-built backend safely.
 *
 * Elastic tiers: this ABI is deliberately the same shape as the existing
 * katali_ggml_matvec/matmul entry points, so VRAM becomes a third residency
 * level next to the ECache RAM slots and the GGUF mmap.
 */
#ifndef KATALI_CUDA_H
#define KATALI_CUDA_H

#include <stddef.h>
#include <stdint.h>
#include "katali.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Bumped whenever the struct layout or any signature below changes. The loader
 * refuses a DLL whose version differs, and falls back to CPU. */
/* ABI version 5 adds the Phase 8 clock keep-warm experiment.
 * ABI version 6 adds the DP4A + q8_1 activation quantization path and its
 * counters (q8_1 quant launches/seconds), plus the two switches that select it. */
#define KATALI_CUDA_ABI_VERSION 6

/* Filled by the backend during katali_cuda_backend_init(). Fixed-size POD only. */
typedef struct KataliCudaDeviceInfo {
    int abi_version;
    int cc_major;                /* compute capability major, e.g. 8 */
    int cc_minor;                /* compute capability minor, e.g. 9 */
    int sm_count;                /* multiprocessor count */
    int driver_version;          /* CUDA driver UMD version, e.g. 13040 = 13.4 */
    int runtime_version;         /* CUDA runtime (cudart) version */
    uint64_t vram_total_bytes;
    uint64_t vram_free_bytes;
    char name[128];              /* e.g. "NVIDIA GeForce RTX 4060" */
    char arch[48];               /* e.g. "Ada Lovelace" */
    char err[192];               /* backend failure detail ("" on success) */
} KataliCudaDeviceInfo;

typedef struct KataliCudaMoeDesc {
    uint32_t gate_type, up_type, down_type;
    uint64_t gate_row_bytes, up_row_bytes, down_row_bytes;
    uint64_t gate_rows, up_rows, down_rows;       /* rows per expert */
    uint64_t hidden, inter;                       /* columns */
} KataliCudaMoeDesc;

/*
 * The host-side API below shares names with the backend's exported ABI (they
 * are the same surface, so a caller does not need to care which side it lands
 * on). The backend build must therefore skip these declarations, or nvcc sees
 * "redeclaration cannot add dllexport" for every one of them.
 */
#ifndef KATALI_CUDA_BACKEND_BUILD

/*
 * ---------------------------------------------------------------------------
 * Host side (implemented in src/katali_cuda.c, always compiled, never needs CUDA)
 * ---------------------------------------------------------------------------
 * Behaviour with no DLL / no GPU / no driver:
 *   katali_cuda_available() -> 0
 *   katali_cuda_status()    -> human-readable reason
 * and every other call is a no-op returning KATALI_ERR. Nothing aborts.
 */

/* 1 = a usable backend DLL was found AND device init succeeded. Cached.
 * KATALI_CUDA=0 forces 0 without touching the filesystem. */
KATALI_API int katali_cuda_available(void);

/* Force the (cached) probe and return its result. Same value as
 * katali_cuda_available(). */
KATALI_API int katali_cuda_probe(void);

/* Human-readable current state: why CUDA is or is not usable. Never NULL. */
KATALI_API const char *katali_cuda_status(void);

/* Resolved path of the backend DLL that was attempted/loaded ("" if none). */
KATALI_API const char *katali_cuda_lib_path(void);

/* Device details, valid only when katali_cuda_available() is 1. */
KATALI_API const KataliCudaDeviceInfo *katali_cuda_device(void);

/* Drop the backend and disable further use (used by tests / A-B runs). */
KATALI_API void katali_cuda_shutdown(void);

/*
 * Thin dispatch wrappers. Each returns KATALI_ERR when no backend is usable, so
 * callers fall back to the CPU path. Memory arguments are DEVICE pointers
 * obtained from katali_cuda_malloc(); the host never dereferences them.
 */
KATALI_API int katali_cuda_malloc(void **out_dev, size_t nbytes);
KATALI_API int katali_cuda_free(void *dev);
KATALI_API int katali_cuda_upload(void *dev_dst, const void *host_src, size_t nbytes);
KATALI_API int katali_cuda_download(void *host_dst, const void *dev_src, size_t nbytes);
/* 1 when the backend has a kernel for this ggml type id. */
KATALI_API int katali_cuda_supports_type(uint32_t ggml_type);
/* y[rows] = W[rows,cols] @ x[cols]; w/x/y are device pointers. */
KATALI_API int katali_cuda_matvec(uint32_t ggml_type, const void *w_dev,
                                  uint64_t rows, uint64_t cols,
                                  const float *x_dev, float *y_dev);
/* Y[B][rows] = W[rows,cols] @ X[B][cols]; all device pointers, row-major. */
KATALI_API int katali_cuda_matmul(uint32_t ggml_type, const void *w_dev,
                                  uint64_t rows, uint64_t cols,
                                  const float *X_dev, uint64_t B, float *Y_dev);

/*
 * ---------------------------------------------------------------------------
 * Phase 1 telemetry.
 *
 * Counts the things that dominate the hybrid's cost: how many times the host
 * crosses into CUDA, and how much wall time that costs. `host_seconds` is the
 * actionable number — it is time spent inside our wrappers, i.e. dispatch
 * overhead the CPU could have spent computing. `gpu_seconds` is only populated
 * when KATALI_CUDA_GPU_TIME=1, because per-call CUDA events themselves add API
 * calls and would distort the counts they are meant to explain.
 * ---------------------------------------------------------------------------
 */
typedef struct KataliCudaTelemetry {
    /* host side */
    uint64_t api_calls;      /* every host->backend call we make */
    uint64_t uploads, downloads, matvecs, matmuls, mallocs, frees;
    uint64_t h2d_bytes, d2h_bytes;
    double   host_seconds;   /* cumulative wall time inside those calls */
    double   upload_s, download_s, matvec_s, matmul_s;
    /* device side (reported by the DLL) */
    uint64_t kernel_launches, sync_calls, memcpy_calls;
    double   gpu_seconds;
    /* ABI 6: DP4A's activation-quantization cost (subset of gpu_seconds). */
    uint64_t q8_1_quant_launches;
    double   q8_1_quant_seconds;
    /* engine side, filled by moe_ffn */
    uint64_t experts_gpu, experts_cpu, layer_batches;
} KataliCudaTelemetry;

/* Zeroes the counters. Called at the start of a measured run. */
KATALI_API void katali_cuda_telemetry_reset(void);
/* Snapshot current counters (never resets). Safe when CUDA is unavailable. */
KATALI_API void katali_cuda_telemetry_get(KataliCudaTelemetry *out);
/* Counters the engine increments from the MoE path. */
KATALI_API void katali_cuda_count_expert(int handled_on_gpu);
KATALI_API void katali_cuda_count_layer_batch(void);
/* One concise stderr line, prefixed with `tag` (e.g. "phase: cuda"). */
KATALI_API void katali_cuda_telemetry_print(const char *tag);
/* 1 when KATALI_CUDA_TELEMETRY=1 (per-token lines wanted). */
KATALI_API int katali_cuda_telemetry_enabled(void);
/* Enable device-event greedy kernel timing. Must be set before the first
 * measured call; see the note in the backend about the DLL's CRT env snapshot. */
KATALI_API void katali_cuda_set_gpu_timing(int on);

/*
 * ---------------------------------------------------------------------------
 * Fused layer-level MoE (Phase 2-7).
 *
 * One call executes an entire MoE layer's routed experts on the GPU, instead of
 * one call per expert per projection. The measured reason: the per-expert path
 * cost 960 launches + 960 device syncs per token (one each per GEMV), i.e.
 * 2 826 CUDA API calls/token and 354 ms of a 626 ms token.
 *
 * Everything intermediate stays on the device and only the layer's input and
 * final output cross PCIe. All selected experts consume one uploaded `x`, and
 * the expert weight is folded into the activation so no separate weighting pass
 * is needed.
 *
 * Requirements/behaviour:
 *  - `gate_dev`/`up_dev`/`down_dev` are arrays of `n_sel` DEVICE pointers, one
 *    per selected expert, each holding that expert's weight slab. Per-expert
 *    pointers (rather than a base + index*stride) keep the call independent of
 *    how the slabs are laid out or cached.
 *  - `sel_weights[j]` is the router weight for selected expert j.
 *  - `x_dev` has `hidden` floats, `y_dev` receives `hidden` floats and holds the
 *    SAME value the CPU path accumulates (sum over j of w_j * down(silu(g)·u)).
 *  - The reduction is done in selected-expert order, matching the CPU's
 *    accumulation order, so results stay deterministic run to run.
 *  - Returns KATALI_OK, or KATALI_ERR for the caller to use its CPU path.
 */
KATALI_API int katali_cuda_moe_layer(const KataliCudaMoeDesc *desc,
                                     const void *const *gate_dev,
                                     const void *const *up_dev,
                                     const void *const *down_dev,
                                     const float *sel_weights, int n_sel,
                                     const float *x_dev, float *y_dev);

/* Phase 8 probe: EXACTLY the same access pattern as katali_cuda_moe_layer with
 * the dequantization and per-element math removed. Comparing the two separates
 * "our memory access pattern is bad" from "we are paying for decode". */
KATALI_API int katali_cuda_moe_layer_nomath(const KataliCudaMoeDesc *desc,
                                            const void *const *gate_dev,
                                            const void *const *up_dev,
                                            const void *const *down_dev,
                                            const float *sel_weights, int n_sel,
                                            const float *x_dev, float *y_dev);

/*
 * Asynchronous variant (Phase 4). Identical work to katali_cuda_moe_layer, but
 * returns as soon as the launches are QUEUED instead of blocking, so the caller
 * can run independent CPU work (the shared expert) while the GPU executes.
 *
 * Contract: exactly one submission may be outstanding at a time, because the
 * fused path reuses one set of internal device scratch buffers. The caller MUST
 * call katali_cuda_moe_wait() before reading `y_dev`. If the work cannot be
 * queued the call returns KATALI_ERR and nothing is outstanding.
 */
KATALI_API int katali_cuda_moe_submit(const KataliCudaMoeDesc *desc,
                                      const void *const *gate_dev,
                                      const void *const *up_dev,
                                      const void *const *down_dev,
                                      const float *sel_weights, int n_sel,
                                      const float *x_dev, float *y_dev);
/* Blocks until the outstanding submission completes. No-op when none is. */
KATALI_API int katali_cuda_moe_wait(void);

/*
 * Phase 8 EXPERIMENT — clock keep-warm. Off by default.
 *
 * Schedules ONE tiny kernel (grid=1, block=1, one FMA) every `interval_ms` on a
 * non-blocking stream, to test whether the measured memory-clock collapse
 * (8501 -> 405 MHz) is caused by the *presence* of GPU activity rather than by
 * load. interval_ms <= 0 stops it. Returns KATALI_OK when configured.
 * This is a diagnostic, never a substitute for real work.
 */
KATALI_API int katali_cuda_keepwarm(int interval_ms);
/* Number of keep-warm launches so far (its own GPU cost, for reporting). */
KATALI_API uint64_t katali_cuda_keepwarm_launches(void);

/* Phase 9 probe: raw VRAM streaming ceiling over a caller-provided buffer that
 * is much larger than L2. `read_gbs` is a pure sequential read; `copy_gbs`
 * counts each byte twice (read + write). Either output may be NULL. */
KATALI_API int katali_cuda_bench_stream(void *dev_a, void *dev_b, size_t bytes,
                                        int reps, double *read_gbs,
                                        double *copy_gbs);

/*
 * ---------------------------------------------------------------------------
 * ABI 6 — DP4A + q8_1 activation quantization (Stage A).
 *
 * What it changes: the routed-expert GEMV no longer dequantizes a weight
 * element to float and multiplies it by an fp32 activation. Instead the layer's
 * activation is quantized ONCE to q8_1 on the device (32 values per block:
 * fp32 scale, fp32 integer sum, int8 quants) and every selected expert's
 * quantized weights are dotted against those int8 values with `__dp4a`, so four
 * MACs happen per instruction and no float dequantization appears in the inner
 * loop. Only the per-32-value scale factors are applied in fp32 afterwards.
 *
 * Default OFF: with KATALI_CUDA_DP4A unset (and no explicit setter call) every
 * call keeps the original fp32-dequantization kernel, byte-for-byte.
 * ---------------------------------------------------------------------------
 */
/* -1 = automatic (env KATALI_CUDA_DP4A; default off), 0 = forced off,
 * 1 = forced on. Used by the A/B harness so both arms can run in one process. */
KATALI_API void katali_cuda_set_dp4a(int on);
/* Per-weight-type gates for the DP4A path, so Q4_K and Q6_K can be measured
 * separately. Defaults to 1/1 when DP4A is enabled. Either may be 0/1. */
KATALI_API void katali_cuda_set_dp4a_types(int q4_k, int q6_k);
/* 1 when the most recent fused MoE call actually used the DP4A kernels; 0 when
 * it fell back (type / dimension / switch gate). Lets a caller prove which path
 * ran instead of inferring it from the timing. */
KATALI_API int katali_cuda_dp4a_active(void);

#endif /* !KATALI_CUDA_BACKEND_BUILD */

/*
 * ---------------------------------------------------------------------------
 * Backend ABI (implemented in the nvcc-built DLL; declared for the loader and
 * for the backend sources). Not available in the Unix/MinGW link path.
 * ---------------------------------------------------------------------------
 */

/* Must equal KATALI_CUDA_ABI_VERSION, or the loader rejects the DLL. */
typedef int (*katali_cuda_abi_version_fn)(void);
/* 0 on success; on failure writes out->err and returns non-zero. */
typedef int (*katali_cuda_backend_init_fn)(KataliCudaDeviceInfo *out);
typedef void (*katali_cuda_backend_shutdown_fn)(void);
/* 1 when the backend has a kernel for this GGML type id. */
typedef int (*katali_cuda_supports_type_fn)(uint32_t type);
/* y[rows] = W[rows,cols] @ x[cols]. Same contract as katali_ggml_matvec().
 * W is GGUF-encoded with dims[0] = cols contiguous. Returns 0 when handled,
 * non-zero to tell the caller to use its own path. */
typedef int (*katali_cuda_matvec_fn)(uint32_t type, const void *w,
                                     uint64_t rows, uint64_t cols,
                                     const float *x, float *y);
/* Y[B][rows] = W[rows,cols] @ X[B][cols], row-major. Same contract as
 * katali_ggml_matmul(). */
typedef int (*katali_cuda_matmul_fn)(uint32_t type, const void *w,
                                     uint64_t rows, uint64_t cols,
                                     const float *X, uint64_t B, float *Y);
/* VRAM tier primitives. */
typedef int (*katali_cuda_device_alloc_fn)(void **out, size_t nbytes);
typedef int (*katali_cuda_device_free_fn)(void *dev);
typedef int (*katali_cuda_upload_fn)(void *dev_dst, const void *host_src,
                                     size_t nbytes);
typedef int (*katali_cuda_download_fn)(void *host_dst, const void *dev_src,
                                       size_t nbytes);

/* Counters the DLL keeps about itself (things the host cannot observe).
 * `gpu_seconds` is only filled when KATALI_CUDA_GPU_TIME=1. */
typedef struct KataliCudaDeviceCounters {
    uint64_t kernel_launches;
    uint64_t sync_calls;
    uint64_t memcpy_calls;
    double   gpu_seconds;
    /* ABI 6: the DP4A path's own overhead, reported separately so a faster GEMV
     * cannot hide the activation-quantization cost it depends on. */
    uint64_t q8_1_quant_launches;
    double   q8_1_quant_seconds;
} KataliCudaDeviceCounters;

typedef int (*katali_cuda_device_telemetry_fn)(KataliCudaDeviceCounters *out,
                                               int reset);

/* Fused layer-level MoE; same shape as the host wrapper of the same name. */
typedef int (*katali_cuda_moe_layer_fn)(const KataliCudaMoeDesc *desc,
                                        const void *const *gate_dev,
                                        const void *const *up_dev,
                                        const void *const *down_dev,
                                        const float *sel_weights, int n_sel,
                                        const float *x_dev, float *y_dev);

typedef int (*katali_cuda_bench_stream_fn)(void *dev_a, void *dev_b,
                                           size_t bytes, int reps,
                                           double *read_gbs, double *copy_gbs);

/* Optional: absent in older DLLs, so the loader resolves it without requiring it. */
typedef void (*katali_cuda_set_gpu_timing_fn)(int on);

/* Phase 4 async MoE. Same shape as the host wrappers of the same names. */
typedef int (*katali_cuda_moe_submit_fn)(const KataliCudaMoeDesc *desc,
                                         const void *const *gate_dev,
                                         const void *const *up_dev,
                                         const void *const *down_dev,
                                         const float *sel_weights, int n_sel,
                                         const float *x_dev, float *y_dev);
typedef int (*katali_cuda_moe_wait_fn)(void);

/* Phase 8 experiment. */
typedef int (*katali_cuda_keepwarm_fn)(int interval_ms);
typedef uint64_t (*katali_cuda_keepwarm_launches_fn)(void);

/* ABI 6: DP4A path selection (optional; absent in older DLLs). */
typedef void (*katali_cuda_set_dp4a_fn)(int on);
typedef void (*katali_cuda_set_dp4a_types_fn)(int q4_k, int q6_k);
typedef int  (*katali_cuda_dp4a_active_fn)(void);

#ifdef __cplusplus
}
#endif
#endif /* KATALI_CUDA_H */