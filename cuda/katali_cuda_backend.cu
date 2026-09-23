/* katali-lab CUDA backend — implementation of the katali_cuda.h ABI v1.
 *
 * Built by nvcc + MSVC into katali_cuda.dll and loaded at runtime by the
 * MinGW-built katali-lab.exe via LoadLibrary/GetProcAddress. Nothing links
 * against this DLL, so CUDA stays optional.
 *
 * Status: Stage A — device/VRAM tier + F32 GEMM path (validates the whole
 * pipeline). Quantized (Q4_K/Q6_K/Q8_0/Q4_0) kernels follow in Stage B.
 */
#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <thread>
#include <atomic>
#include <chrono>
/* Skip the host-side declarations that share names with our own exports. */
#define KATALI_CUDA_BACKEND_BUILD 1
#include "katali_cuda.h"

#define KCE_API extern "C" __declspec(dllexport)

/* GGML type ids (mirror of katali_gguf_dtype.h; kept local so the backend does
 * not depend on the engine tree). */
enum {
    G_F32 = 0, G_F16 = 1, G_Q4_0 = 2, G_Q4_1 = 3, G_Q5_0 = 6, G_Q5_1 = 7,
    G_Q8_0 = 8, G_Q8_1 = 9, G_Q2_K = 10, G_Q3_K = 11, G_Q4_K = 12,
    G_Q5_K = 13, G_Q6_K = 14, G_Q8_K = 15, G_BF16 = 30
};

static int g_device = -1;
static int g_ready = 0;

/* Defined alongside the fused-MoE kernels further down; releases the persisted
 * device scratch. Declared here because shutdown() calls it. */
static void scr_release(void);

static const char *arch_name(int major, int minor) {
    if (major == 12) return "Blackwell";
    if (major == 11) return "Blackwell";
    if (major == 10) return "Blackwell";
    if (major == 9)  return "Hopper";
    if (major == 8)  return (minor == 9) ? "Ada Lovelace" : "Ampere";
    if (major == 7)  return (minor == 5) ? "Turing" : "Volta";
    if (major == 6)  return "Pascal";
    if (major == 5)  return "Maxwell";
    return "unknown";
}

KCE_API int katali_cuda_abi_version(void) { return KATALI_CUDA_ABI_VERSION; }

/* ---------------------------------------------------------------------------
 * Phase 1 telemetry. The DLL counts the CUDA interactions it performs itself;
 * the host counts the ones it makes through the ABI. Together they give the
 * number that matters: how many times per token the CPU crosses into CUDA, and
 * how much wall time that costs.
 * ------------------------------------------------------------------------- */
static uint64_t g_launches = 0;
static uint64_t g_syncs = 0;
static uint64_t g_memcpys = 0;
static double   g_gpu_seconds = 0.0;
/* Stage A: the DP4A path's own overhead, counted separately so a faster GEMV
 * cannot hide it. `q8_1_seconds` is a subset of gpu_seconds when device timing
 * is on (it is measured with its own event pair). */
static uint64_t g_q8_launches = 0;
static double   g_q8_seconds = 0.0;
static int      g_q8_pending = 0;
static int      g_gpu_time_env = -1;
static cudaEvent_t g_ev_start = NULL, g_ev_stop = NULL;
static cudaEvent_t g_ev_q8a = NULL, g_ev_q8b = NULL;

static int gpu_time_enabled(void) {
    if (g_gpu_time_env < 0) {
        const char *e = getenv("KATALI_CUDA_GPU_TIME");
        g_gpu_time_env = (e && e[0] && e[0] != '0') ? 1 : 0;
    }
    return g_gpu_time_env;
}

/* Explicit switch for the GPU-timing mode.
 *
 * Needed because this DLL is loaded lazily by the MinGW host: the DLL's CRT
 * snapshots the process environment at LoadLibrary time, so an env var set by
 * the host AFTER the first CUDA call never reaches getenv() here. A function
 * call has no such ambiguity. */
KCE_API void katali_cuda_set_gpu_timing(int on) {
    g_gpu_time_env = on ? 1 : 0;
    if (!on && g_ev_start) {
        cudaEventDestroy(g_ev_start); g_ev_start = NULL;
        if (g_ev_stop) { cudaEventDestroy(g_ev_stop); g_ev_stop = NULL; }
    }
}

/* Time one kernel by bracketing it with events. Only used when explicitly
 * enabled, because 2 extra event API calls per kernel would inflate exactly the
 * dispatch counts the measurement is supposed to explain. */
static void q8_tel_flush(void);   /* defined with the q8_1 telemetry below */

static void tel_begin(void) {
    if (!gpu_time_enabled()) return;
    if (!g_ev_start) {
        if (cudaEventCreate(&g_ev_start) != cudaSuccess) { g_ev_start = NULL; return; }
        if (cudaEventCreate(&g_ev_stop) != cudaSuccess) return;
    }
    cudaEventRecord(g_ev_start, 0);
}

static void tel_end(void) {
    q8_tel_flush();
    if (!gpu_time_enabled() || !g_ev_start) return;
    cudaEventRecord(g_ev_stop, 0);
    cudaEventSynchronize(g_ev_stop);
    float ms = 0.f;
    if (cudaEventElapsedTime(&ms, g_ev_start, g_ev_stop) == cudaSuccess)
        g_gpu_seconds += (double)ms / 1000.0;
}

KCE_API int katali_cuda_device_telemetry(KataliCudaDeviceCounters *out, int reset) {
    if (!out) return -1;
    out->kernel_launches = g_launches;
    out->sync_calls = g_syncs;
    out->memcpy_calls = g_memcpys;
    out->gpu_seconds = g_gpu_seconds;
    out->q8_1_quant_launches = g_q8_launches;
    out->q8_1_quant_seconds = g_q8_seconds;
    if (reset) {
        g_launches = g_syncs = g_memcpys = 0;
        g_gpu_seconds = 0.0;
        g_q8_launches = 0;
        g_q8_seconds = 0.0;
    }
    return 0;
}

/* Device-time the q8_1 quantization kernels on their own event pair, so the
 * activation-quantization overhead is a measured number rather than something
 * folded invisibly into the layer total. */
static void q8_tel_begin(void) {
    if (!gpu_time_enabled()) return;
    if (!g_ev_q8a) {
        if (cudaEventCreate(&g_ev_q8a) != cudaSuccess) { g_ev_q8a = NULL; return; }
        if (cudaEventCreate(&g_ev_q8b) != cudaSuccess) return;
    }
    cudaEventRecord(g_ev_q8a, 0);
}

static void q8_tel_end(void) {
    g_q8_launches++;
    if (!gpu_time_enabled() || !g_ev_q8a) return;
    cudaEventRecord(g_ev_q8b, 0);
    /* Deliberately NOT synchronizing here: a sync mid-layer would drain the
     * pipeline half-way through the layer, which is the exact defect fixed in
     * §15.4. The elapsed time is read in tel_end(), which already waits for the
     * layer and therefore finds both events completed. */
    g_q8_pending = 1;
}

/* Fold the pending quantization window into the q8_1 counter. Called after the
 * layer has been drained, so no extra synchronization is introduced. */
static void q8_tel_flush(void) {
    if (!g_q8_pending || !g_ev_q8a || !g_ev_q8b) { g_q8_pending = 0; return; }
    g_q8_pending = 0;
    float ms = 0.f;
    if (cudaEventElapsedTime(&ms, g_ev_q8a, g_ev_q8b) == cudaSuccess)
        g_q8_seconds += (double)ms / 1000.0;
}

/* Count one kernel launch; also the single place a launch is declared. */
static void tel_launch(void) { g_launches++; }
static void tel_sync(void) {
    g_syncs++;
    if (cudaDeviceSynchronize() != cudaSuccess) { /* caller checks */ }
}

/* Report a CUDA failure from inside the backend. Deliberately unbuffered
 * stderr: a device fault inside a kernel is asynchronous, so a caller that
 * crashes on the sticky error would otherwise lose its own buffered output and
 * leave nothing to diagnose. Called on every failure path in the fused layer.
 * The error code must be captured with cudaGetLastError() EXACTLY ONCE per
 * check: that call also clears the error, so a second read reports "no error"
 * and hides the cause. */
#define KCE_CHECK(where)                                                       \
    do {                                                                       \
        const cudaError_t kce_e_ = cudaGetLastError();                          \
        if (kce_e_ != cudaSuccess) {                                            \
            fprintf(stderr, "kce: %s: CUDA error: %s\n", (where),               \
                    cudaGetErrorString(kce_e_));                                \
            fflush(stderr);                                                     \
            return -1;                                                          \
        }                                                                       \
    } while (0)

KCE_API int katali_cuda_backend_init(KataliCudaDeviceInfo *out) {
    if (!out) return -1;
    memset(out, 0, sizeof(*out));
    out->abi_version = KATALI_CUDA_ABI_VERSION;

    int n = 0;
    cudaError_t e = cudaGetDeviceCount(&n);
    if (e != cudaSuccess) {
        snprintf(out->err, sizeof(out->err), "cudaGetDeviceCount: %s",
                 cudaGetErrorString(e));
        return -1;
    }
    if (n <= 0) {
        snprintf(out->err, sizeof(out->err), "no CUDA-capable device present");
        return -1;
    }
    /* Pick the device with the most free VRAM. */
    int best = 0;
    size_t best_free = 0;
    for (int i = 0; i < n; i++) {
        size_t f = 0, t = 0;
        if (cudaSetDevice(i) != cudaSuccess) continue;
        if (cudaMemGetInfo(&f, &t) != cudaSuccess) continue;
        if (f > best_free) { best_free = f; best = i; }
    }
    e = cudaSetDevice(best);
    if (e != cudaSuccess) {
        snprintf(out->err, sizeof(out->err), "cudaSetDevice(%d): %s",
                 best, cudaGetErrorString(e));
        return -1;
    }
    cudaDeviceProp p;
    e = cudaGetDeviceProperties(&p, best);
    if (e != cudaSuccess) {
        snprintf(out->err, sizeof(out->err), "cudaGetDeviceProperties: %s",
                 cudaGetErrorString(e));
        return -1;
    }
    int drv = 0, rt = 0;
    cudaDriverGetVersion(&drv);
    cudaRuntimeGetVersion(&rt);
    size_t fr = 0, tot = 0;
    cudaMemGetInfo(&fr, &tot);

    out->cc_major = p.major;
    out->cc_minor = p.minor;
    out->sm_count = p.multiProcessorCount;
    out->driver_version = drv;
    out->runtime_version = rt;
    out->vram_total_bytes = (uint64_t)tot;
    out->vram_free_bytes = (uint64_t)fr;
    snprintf(out->name, sizeof(out->name), "%s", p.name);
    snprintf(out->arch, sizeof(out->arch), "%s", arch_name(p.major, p.minor));

    g_device = best;
    g_ready = 1;
    return 0;
}

/* Defined with the Phase 8 experiment below; called by shutdown. */
KCE_API int katali_cuda_keepwarm(int interval_ms);

KCE_API void katali_cuda_backend_shutdown(void) {
    if (g_ready) {
        (void)katali_cuda_keepwarm(0);   /* stop the keep-warm thread first */
        scr_release();
        cudaDeviceSynchronize();
        cudaDeviceReset();
        g_ready = 0;
        g_device = -1;
    }
}

/* --- VRAM tier primitives ------------------------------------------------ */

KCE_API int katali_cuda_device_alloc(void **outp, size_t nbytes) {
    if (!outp || !g_ready) return -1;
    *outp = NULL;
    if (nbytes == 0) return -1;
    if (cudaMalloc(outp, nbytes) != cudaSuccess) { *outp = NULL; return -1; }
    return 0;
}

KCE_API int katali_cuda_device_free(void *dev) {
    if (!dev) return 0;
    return (cudaFree(dev) == cudaSuccess) ? 0 : -1;
}

KCE_API int katali_cuda_upload(void *dev_dst, const void *host_src, size_t n) {
    if (!dev_dst || !host_src || n == 0) return -1;
    g_memcpys++;
    return (cudaMemcpy(dev_dst, host_src, n, cudaMemcpyHostToDevice) == cudaSuccess)
               ? 0 : -1;
}

KCE_API int katali_cuda_download(void *host_dst, const void *dev_src, size_t n) {
    if (!host_dst || !dev_src || n == 0) return -1;
    g_memcpys++;
    return (cudaMemcpy(host_dst, dev_src, n, cudaMemcpyDeviceToHost) == cudaSuccess)
               ? 0 : -1;
}

/* --- type support ------------------------------------------------------- */

/* ===================================================================== *
 * Device-side GGML dequantization.
 *
 * Layouts mirror vendor/katali_gguf/src/gguf_dtype.c exactly; the CPU
 * reference (katali_ggml_dequant_ref / katali_ggml_vec_dot) is the oracle the
 * correctness harness compares against.
 * ===================================================================== */

__device__ __forceinline__ float kce_f16(uint16_t h) {
    uint32_t s = (h >> 15) & 1u, e = (h >> 10) & 0x1fu, f = h & 0x3ffu, o;
    if (e == 0) {
        if (f == 0) { o = s << 31; }
        else {
            e = 127 - 15 + 1;
            while ((f & 0x400u) == 0) { f <<= 1; e--; }
            f &= 0x3ffu;
            o = (s << 31) | (e << 23) | (f << 13);
        }
    } else if (e == 31) {
        o = (s << 31) | 0x7f800000u | (f << 13);
    } else {
        o = (s << 31) | ((e + 112u) << 23) | (f << 13);
    }
    float r; memcpy(&r, &o, 4); return r;
}

__device__ __forceinline__ uint16_t kce_ld16(const uint8_t *p) {
    uint16_t v; memcpy(&v, p, 2); return v;
}

/* 4 bytes from a pointer that is only guaranteed 2-byte aligned (Q6_K rows move
 * in 210-byte super-blocks). Built from two half-word loads instead of one
 * 4-byte load, which is what the hardware requires. */
__device__ __forceinline__ uint32_t kce_ld32_2aligned(const uint8_t *p) {
    return (uint32_t)kce_ld16(p) | ((uint32_t)kce_ld16(p + 2) << 16);
}

/* Q4_K: super-block of 256 values, 144 bytes:
 *   d(f16) dmin(f16) scales[12] qs[128]
 * 6-bit scale/min unpacking is the ggml k-quant scheme. */
__device__ __forceinline__ void kce_q4k_scale_min(int j, const uint8_t *q,
                                                  uint8_t *d, uint8_t *m) {
    if (j < 4) { *d = q[j] & 63u; *m = q[j + 4] & 63u; }
    else {
        *d = (uint8_t)((q[j + 4] & 0x0Fu) | ((q[j - 4] >> 6) << 4));
        *m = (uint8_t)((q[j + 4] >> 4)    | ((q[j]     >> 6) << 4));
    }
}

/* Value at flat column `c` inside a row of `type` (row byte-aligned). */
template <int TYPE>
__device__ __forceinline__ float kce_dequant_at(const uint8_t *row, unsigned c);

template <>
__device__ __forceinline__ float kce_dequant_at<G_F32>(const uint8_t *row, unsigned c) {
    return reinterpret_cast<const float *>(row)[c];
}

template <>
__device__ __forceinline__ float kce_dequant_at<G_Q4_0>(const uint8_t *row, unsigned c) {
    const uint8_t *blk = row + (size_t)(c >> 5) * 18u;   /* 32 values / 18 bytes */
    float d = kce_f16(kce_ld16(blk));
    unsigned l = c & 31u;
    uint8_t b = blk[2 + (l & 15u)];
    int q = (l < 16u) ? (int)(b & 0x0Fu) : (int)(b >> 4);
    return (float)(q - 8) * d;
}

template <>
__device__ __forceinline__ float kce_dequant_at<G_Q8_0>(const uint8_t *row, unsigned c) {
    const uint8_t *blk = row + (size_t)(c >> 5) * 34u;   /* 32 values / 34 bytes */
    float d = kce_f16(kce_ld16(blk));
    signed char q = (signed char)blk[2 + (c & 31u)];
    return d * (float)q;
}

template <>
__device__ __forceinline__ float kce_dequant_at<G_Q4_K>(const uint8_t *row, unsigned c) {
    const uint8_t *blk = row + (size_t)(c >> 8) * 144u;  /* 256 values / 144 bytes */
    float d    = kce_f16(kce_ld16(blk));
    float dmin = kce_f16(kce_ld16(blk + 2));
    const uint8_t *scales = blk + 4;
    const uint8_t *qs     = blk + 16;
    unsigned within = c & 255u;
    uint8_t sc, m;
    kce_q4k_scale_min((int)(within >> 5), scales, &sc, &m);
    unsigned half64 = within & 63u;          /* position inside a 64-value chunk */
    uint8_t byte = qs[(within >> 6) * 32u + (half64 & 31u)];
    int q = (half64 < 32u) ? (int)(byte & 0x0Fu) : (int)(byte >> 4);
    return (d * (float)sc) * (float)q - (dmin * (float)m);
}

template <>
__device__ __forceinline__ float kce_dequant_at<G_Q6_K>(const uint8_t *row, unsigned c) {
    const uint8_t *blk = row + (size_t)(c >> 8) * 210u;  /* 256 values / 210 bytes */
    float d = kce_f16(kce_ld16(blk + 208));
    unsigned within = c & 255u;
    unsigned group = within >> 7;            /* two 128-value halves */
    const uint8_t *ql = blk + group * 64u;
    const uint8_t *qh = blk + 128u + group * 32u;
    const signed char *sc = (const signed char *)(blk + 192u) + group * 8u;
    unsigned idx = within & 127u;
    unsigned l = idx & 31u;                  /* 0..31 */
    unsigned which = idx >> 5;               /* 0..3 selects q1..q4 */
    unsigned is = l >> 4;                    /* 0 or 1 inside the 8-scale group */
    int q1 = (int)((ql[l]      & 0x0Fu) | (((qh[l] >> 0) & 3u) << 4)) - 32;
    int q2 = (int)((ql[l + 32] & 0x0Fu) | (((qh[l] >> 2) & 3u) << 4)) - 32;
    int q3 = (int)((ql[l]      >> 4)    | (((qh[l] >> 4) & 3u) << 4)) - 32;
    int q4 = (int)((ql[l + 32] >> 4)    | (((qh[l] >> 6) & 3u) << 4)) - 32;
    int qv = (which == 0u) ? q1 : (which == 1u) ? q2 : (which == 2u) ? q3 : q4;
    return d * (float)sc[is + 2u * which] * (float)qv;
}

/* ===================================================================== *
 * Stage A — q8_1 activations + DP4A.
 *
 * The measured problem this addresses: the fused kernel reaches 32 GB/s cold
 * while the same access pattern without dequantization reaches 106+ GB/s, and
 * 63 % of that gap is the per-element float dequantization the inner loop does
 * before every multiply. llama.cpp's matvec kernels avoid it by quantizing the
 * ACTIVATION to q8_1 and using `__dp4a` (4-way int8 dot) against the packed
 * weights, so the inner loop contains no float conversion at all.
 *
 * KATALI-specific twist: all selected experts of a layer consume the SAME
 * activation, so it is quantized once per layer and reused by every expert
 * instead of being re-derived per expert. That is only valid because the dot
 * product is linear in the activation, and it is what amortizes the
 * quantization cost across the selected experts.
 *
 * Device-only format (never written to disk, so it does not have to match
 * ggml's on-disk block_q8_1):
 *
 *   float d;      amax/127, so x_i ~= d * qs[i]
 *   float s;      INTEGER sum of qs[0..31]   (for the `- dmin*m*sum(x)` term)
 *   float s_lo;   integer sum of qs[0..15]   (Q6_K's 16-value scale groups)
 *   int8_t qs[32];
 *
 * 44 bytes (vs ggml's f16/f16 + int8[32] = 36): fp32 scales keep the integer
 * sums exact (a 32-value sum of int8 reaches 4064, which f16 cannot represent
 * exactly) and avoid half<->float conversion in the inner loop. qs stays
 * 4-byte aligned, which is all the int32 dp4a loads need.
 * ===================================================================== */
#define KCE_QK8_1 32
/* Block size of the job-batched GEMV family. Both the fp32 and the DP4A
 * variants use it, and the DP4A thread mapping is defined in terms of it, so it
 * is declared before either. */
#define KCE_JOBS_BLOCK 128
#define KCE_BLOCK      128
typedef struct __align__(4) KceQ8_1 {
    float  d;
    float  s;
    float  s_lo;
    int8_t qs[KCE_QK8_1];
} KceQ8_1;
#define KCE_Q8_1_BYTES ((unsigned)sizeof(KceQ8_1))

/* One WARP per 32-value block: the block's amax and its integer sum are warp
 * reductions, so a warp must never straddle two q8_1 blocks. That is why the
 * block size is a multiple of 32 and the launch is sized in warps. */
__global__ void kce_quant_q8_1_kernel(const float *__restrict__ x,
                                      KceQ8_1 *__restrict__ q,
                                      unsigned n) {
    const unsigned lane = threadIdx.x & 31u;
    const unsigned warp = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
    const size_t i = (size_t)warp * KCE_QK8_1 + lane;
    float xi = (i < n) ? x[i] : 0.f;

    float amax = fabsf(xi);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));

    const float d = amax / 127.0f;
    int qi = 0;
    if (d > 0.f) {
        qi = (int)lrintf(xi / d);
        if (qi > 127) qi = 127;
        if (qi < -127) qi = -127;
    }
    int qs = qi;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        qs += __shfl_xor_sync(0xffffffffu, qs, o);

    if (i < n) q[(size_t)warp].qs[lane] = (int8_t)qi;
    if (lane == 0u) {
        q[warp].d = d;
        q[warp].s = (float)qs;
    }
    /* Half-block sum, signed. Used by Q6_K, whose scale/min groups are 16 wide
     * while a q8_1 block is 32 wide. */
    int qs_lo = (lane < 16u) ? qi : 0;
#pragma unroll
    for (int o = 8; o > 0; o >>= 1)
        qs_lo += __shfl_xor_sync(0xffffffffu, qs_lo, o);
    if (lane == 0u) q[warp].s_lo = (float)qs_lo;
}

/* ===================================================================== *
 * Quantized GEMV: one thread block per output row.
 *
 * Weights are read exactly once per row (the memory-bound optimum); `x` is
 * read once per row as well. Threads stride over the row's columns, so a warp
 * touches 32 consecutive column values and the dequant reads land in adjacent
 * cache lines. This is the operation decode is built from, so it is the one
 * worth measuring first.
 * ===================================================================== */
template <int TYPE, int BLOCK>
__global__ void kce_gemv_kernel(const uint8_t *__restrict__ w,
                                const float *__restrict__ x,
                                float *__restrict__ y,
                                unsigned rows, unsigned cols, unsigned row_bytes) {
    unsigned row = blockIdx.x;
    if (row >= rows) return;
    const uint8_t *rp = w + (size_t)row * row_bytes;
    float acc = 0.f;
    for (unsigned c = threadIdx.x; c < cols; c += BLOCK)
        acc += kce_dequant_at<TYPE>(rp, c) * x[c];

    __shared__ float sh[BLOCK / 32];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if ((threadIdx.x & 31u) == 0u) sh[threadIdx.x >> 5] = acc;
    __syncthreads();
    if (threadIdx.x < 32u) {
        acc = (threadIdx.x < (unsigned)(BLOCK / 32)) ? sh[threadIdx.x] : 0.f;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
        if (threadIdx.x == 0u) y[row] = acc;
    }
}

/* ===================================================================== *
 * Phase 2-7: fused layer-level MoE.
 *
 * One layer's routed experts are executed with a handful of launches instead of
 * one launch+sync per expert per projection. Nothing intermediate leaves the
 * device, and the expert weight is folded into the activation.
 *
 *   launch 1  gate+up for all selected experts   (job-batched GEMV)
 *   launch 2  silu(gate)*up, scaled by router w   (fused activation+weighting)
 *   launch 3  down for all selected experts       (job-batched GEMV)
 *   launch 4  deterministic reduce over experts   -> layer output
 *
 * 4-5 launches and ONE sync per layer, versus 24 launches and 24 syncs before.
 * ===================================================================== */

#define KCE_MAX_SEL  16
#define KCE_MAX_JOBS (2 * KCE_MAX_SEL)

typedef struct KceJobDesc {
    const unsigned char *w;
    const float         *x;   /* PER-JOB input: the down jobs must read their own
                               * expert's intermediate slice (is + j*FF), not a
                               * shared base. Sharing one x made every expert's
                               * down projection consume expert 0's activation,
                               * which still looked correct at n_sel=1. */
    /* Stage A: the same activation in q8_1 form, consumed by the DP4A kernels.
     * NULL on the fp32 path (which then reads `x`). */
    const KceQ8_1       *xq;
    float               *out;
    unsigned             rows;
    unsigned             cols;
    unsigned             row_bytes;
} KceJobDesc;

typedef struct KceJobSet {
    int          n;
    unsigned     max_rows;
    KceJobDesc   job[KCE_MAX_JOBS];
} KceJobSet;

typedef struct KceWeights {
    int   n;
    float w[KCE_MAX_SEL];
} KceWeights;

/* One block per (job, row). Blocks whose row is past the job's row count exit
 * immediately, which lets jobs of different sizes share one launch.
 *
 * The job table is passed BY VALUE as a kernel argument (~1 KB). That is
 * deliberate and measured: it used to live in a device buffer filled by
 * jobs_upload(), but a synchronous cudaMemcpy from pageable host memory
 * synchronizes the stream first, which force-drained the layer *mid-way*
 * (after gate+up+silu, before the down launch) on every layer. Passing it as a
 * parameter removes both the copy and that forced synchronisation point.
 *
 * UNROLL > 1 restructures the column loop into UNROLL independent accumulator
 * chains with no data dependence between them, so several weight loads are in
 * flight at once instead of one. cols is a runtime value, so the compiler
 * cannot unroll this loop on its own - which is exactly the memory-level
 * parallelism the cold kernel was missing. Both variants are instantiated and
 * selected at launch, so the A/B needs no recompile. */
template <int TYPE, int BLOCK, int UNROLL>
__global__ void kce_gemv_jobs_kernel(KceJobSet js) {
    const int j = blockIdx.y;
    if (j >= js.n) return;
    const unsigned char *w = js.job[j].w;
    const unsigned row = blockIdx.x;
    if (row >= js.job[j].rows) return;
    const uint8_t *rp = w + (size_t)row * js.job[j].row_bytes;
    const unsigned cols = js.job[j].cols;
    const float *x = js.job[j].x;

    float acc;
    if (UNROLL == 1) {
        acc = 0.f;
        for (unsigned c = threadIdx.x; c < cols; c += BLOCK)
            acc += kce_dequant_at<TYPE>(rp, c) * x[c];
    } else {
        float a[UNROLL];
#pragma unroll
        for (int u = 0; u < UNROLL; u++) a[u] = 0.f;
        unsigned c = threadIdx.x;
        const unsigned step = (unsigned)UNROLL * BLOCK;
        for (; c + (unsigned)(UNROLL - 1) * BLOCK < cols; c += step) {
#pragma unroll
            for (int u = 0; u < UNROLL; u++) {
                const unsigned cc = c + (unsigned)u * BLOCK;
                a[u] += kce_dequant_at<TYPE>(rp, cc) * x[cc];
            }
        }
        for (; c < cols; c += BLOCK) a[0] += kce_dequant_at<TYPE>(rp, c) * x[c];
        /* Sum in a fixed order so the result is deterministic regardless of
         * which UNROLL variant ran. */
        acc = 0.f;
#pragma unroll
        for (int u = 0; u < UNROLL; u++) acc += a[u];
    }

    __shared__ float sh[BLOCK / 32];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if ((threadIdx.x & 31u) == 0u) sh[threadIdx.x >> 5] = acc;
    __syncthreads();
    if (threadIdx.x < 32u) {
        acc = (threadIdx.x < (unsigned)(BLOCK / 32)) ? sh[threadIdx.x] : 0.f;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
        if (threadIdx.x == 0u) js.job[j].out[row] = acc;
    }
}

/* Block-wide sum, used by the DP4A kernels. Returns a meaningful value only on
 * thread 0 (the caller tests threadIdx.x). Same reduction shape as the fp32
 * kernels so the two paths' rounding behaviour is comparable. */
template <int BLOCK>
__device__ __forceinline__ float kce_block_sum(float acc) {
    __shared__ float sh[BLOCK / 32];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if ((threadIdx.x & 31u) == 0u) sh[threadIdx.x >> 5] = acc;
    __syncthreads();
    if (threadIdx.x < 32u) {
        acc = (threadIdx.x < (unsigned)(BLOCK / 32)) ? sh[threadIdx.x] : 0.f;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    }
    return acc;
}

/* ===================================================================== *
 * Stage A — DP4A job GEMV for Q4_K.
 *
 * Per 32-value group the Q4_K dequantization is
 *     y_i = (d * sc_g) * q_i - (dmin * m_g)
 * so the dot against the activation is
 *     sum_i y_i a_i = (d*sc_g) * sum_i q_i a_i  -  (dmin*m_g) * sum_i a_i
 * The first term is an integer dot of 4-bit weights with int8 activations
 * (`__dp4a`), and the second is the group's activation sum, which the q8_1
 * block already carries as an exact integer in `s`. Both scale factors are
 * applied once per group, outside the inner loop.
 *
 * Thread mapping (checked by dp4a_ok() before any launch): cols % 512 == 0, so
 * (cols/4)/BLOCK is a whole number of 4-value words per thread and no thread
 * straddles a q8_1 block or a super-block. Block = 128 threads, one row per
 * block, one block per (job, row) — the same shape as the fp32 job kernel, so
 * the two can be A/B'd on identical schedules.
 * ===================================================================== */
template <int BLOCK, int NW>
__global__ void kce_gemv_jobs_q4k_dp4a_kernel(KceJobSet js) {
    const int j = blockIdx.y;
    if (j >= js.n) return;
    const KceJobDesc *jd = &js.job[j];
    const unsigned row = blockIdx.x;
    if (row >= jd->rows) return;

    const uint8_t *rp = (const uint8_t *)jd->w + (size_t)row * jd->row_bytes;
    const unsigned base  = threadIdx.x * (unsigned)NW;  /* first 4-value word */
    const unsigned g     = base >> 3;                   /* 32-value group (row-wide) */
    const unsigned sb    = base >> 6;                   /* 256-value super-block */
    /* Weight-side indices are LOCAL to the super-block: 8 groups per block, two
     * groups per 32-byte qs chunk. Using the row-wide group here would walk off
     * the end of the block (measured: illegal address at every layer). */
    const unsigned gi    = g & 7u;                      /* group inside the block */
    const unsigned chunk = gi >> 1;                     /* 32-byte qs chunk */
    const unsigned hi    = gi & 1u;                     /* which nibble half */
    const unsigned off   = (base & 7u) * 4u;            /* byte offset in chunk */

    const uint8_t *blk = rp + (size_t)sb * 144u;
    const uint32_t *qw = (const uint32_t *)(blk + 16u + chunk * 32u + off);
    const KceQ8_1  *xq = jd->xq + g;
    const int32_t  *aw = (const int32_t *)(xq->qs + off);

    int sumi = 0;
#pragma unroll
    for (int u = 0; u < NW; u++) {
        uint32_t v = __ldg(qw + u);
        v = hi ? ((v >> 4) & 0x0F0F0F0Fu) : (v & 0x0F0F0F0Fu);
        sumi = __dp4a((int)v, (int)__ldg(aw + u), sumi);
    }

    const float d  = kce_f16(kce_ld16(blk));
    const float dm = kce_f16(kce_ld16(blk + 2));
    uint8_t sc, m;
    kce_q4k_scale_min((int)gi, blk + 4, &sc, &m);   /* block-local group */

    float acc = (d * (float)sc * xq->d) * (float)sumi;
    /* The `-dmin*m*sum(x)` term belongs to the whole group: only the thread that
     * owns its first word adds it, so it is counted exactly once. */
    if ((base & 7u) == 0u)
        acc -= (dm * (float)m) * (xq->d * xq->s);

    const float red = kce_block_sum<BLOCK>(acc);
    if (threadIdx.x == 0u) jd->out[row] = red;
}

/* ===================================================================== *
 * Stage A — DP4A job GEMV for Q6_K.
 *
 * Per 16-value scale group: y_i = (d * sc) * (q6_i - 32), with the stored
 * 6-bit value q6 = lo4 + 16*hi2. So
 *     sum_i y_i a_i = (d*sc) * [ sum lo4*a + 16 * sum hi2*a - 32 * sum a ]
 * where the two integer dots are `__dp4a` calls against the same int8
 * activation word (lo4 is a 4-bit field, hi2 is a 2-bit field, both treated as
 * unsigned bytes) and `sum a` is d8 * (integer sum of the activation), taken
 * from the q8_1 block. Q6_K's scale groups are 16 values wide while a q8_1
 * block is 32, which is why the activation block carries `s_lo` (the sum of its
 * first half) as well as `s`.
 * ===================================================================== */
template <int BLOCK, int NW>
__global__ void kce_gemv_jobs_q6k_dp4a_kernel(KceJobSet js) {
    const int j = blockIdx.y;
    if (j >= js.n) return;
    const KceJobDesc *jd = &js.job[j];
    const unsigned row = blockIdx.x;
    if (row >= jd->rows) return;

    const uint8_t *rp = (const uint8_t *)jd->w + (size_t)row * jd->row_bytes;
    const unsigned base  = threadIdx.x * (unsigned)NW;
    const unsigned g     = base >> 3;      /* 32-value group (row-wide) == q8_1 block */
    const unsigned sb    = base >> 6;      /* 256-value super-block */
    const unsigned gi    = g & 7u;         /* group inside the super-block */
    const unsigned half  = gi >> 2;        /* 128-value half (block-local!) */
    const unsigned which = g & 3u;         /* selects q1..q4 (same either way) */
    const unsigned l0    = (base & 7u) * 4u;

    const uint8_t *blk = rp + (size_t)sb * 210u;
    const uint8_t *ql = blk + half * 64u + (((which & 1u) != 0u) ? 32u : 0u) + l0;
    const uint8_t *qh = blk + 128u + half * 32u + l0;
    const KceQ8_1 *xq = jd->xq + g;
    const int32_t *aw = (const int32_t *)(xq->qs + l0);
    const unsigned sh = 2u * which;
    const int nib_hi = (which >= 2u);

    int sumi_lo = 0, sumi_hi = 0;
#pragma unroll
    for (int u = 0; u < NW; u++) {
        /* Q6_K super-blocks are 210 bytes, so a row's blocks are only 2-byte
         * aligned: a 4-byte load out of ql/qh faults with "misaligned address"
         * on odd super-blocks (measured). Two half-word loads are always safe. */
        uint32_t v = kce_ld32_2aligned(ql + 4 * u);
        v = nib_hi ? ((v >> 4) & 0x0F0F0F0Fu) : (v & 0x0F0F0F0Fu);
        const uint32_t h = (kce_ld32_2aligned(qh + 4 * u) >> sh) & 0x03030303u;
        const int a = (int)__ldg(aw + u);
        sumi_lo = __dp4a((int)v, a, sumi_lo);
        sumi_hi = __dp4a((int)h, a, sumi_hi);
    }

    const unsigned is = l0 >> 4;                      /* 16-value scale group */
    const float d   = kce_f16(kce_ld16(blk + 208u));
    const float scv = (float)((const int8_t *)(blk + 192u + half * 8u))[2u * which + is];

    const float w = d * scv * xq->d;
    float acc = w * (float)sumi_lo + 16.0f * w * (float)sumi_hi;
    if ((base & 3u) == 0u) {                          /* group's first word: once */
        const float s16 = (is == 0u) ? xq->s_lo : (xq->s - xq->s_lo);
        acc -= (d * scv) * (xq->d * s16) * 32.0f;
    }

    const float red = kce_block_sum<BLOCK>(acc);
    if (threadIdx.x == 0u) jd->out[row] = red;
}

/* Dispatch one DP4A job launch. `nw` is words-per-thread, computed identically
 * on both sides by dp4a_words_per_thread(). */
static int kce_launch_jobs_dp4a(uint32_t type, unsigned nw, const KceJobSet &js,
                                dim3 gdim, dim3 bdim) {
    if (type == G_Q4_K) {
        switch (nw) {
            case 1: kce_gemv_jobs_q4k_dp4a_kernel<KCE_JOBS_BLOCK, 1><<<gdim, bdim>>>(js); return 0;
            case 2: kce_gemv_jobs_q4k_dp4a_kernel<KCE_JOBS_BLOCK, 2><<<gdim, bdim>>>(js); return 0;
            case 4: kce_gemv_jobs_q4k_dp4a_kernel<KCE_JOBS_BLOCK, 4><<<gdim, bdim>>>(js); return 0;
            case 8: kce_gemv_jobs_q4k_dp4a_kernel<KCE_JOBS_BLOCK, 8><<<gdim, bdim>>>(js); return 0;
            default: return -1;
        }
    }
    if (type == G_Q6_K) {
        switch (nw) {
            case 1: kce_gemv_jobs_q6k_dp4a_kernel<KCE_JOBS_BLOCK, 1><<<gdim, bdim>>>(js); return 0;
            case 2: kce_gemv_jobs_q6k_dp4a_kernel<KCE_JOBS_BLOCK, 2><<<gdim, bdim>>>(js); return 0;
            case 4: kce_gemv_jobs_q6k_dp4a_kernel<KCE_JOBS_BLOCK, 4><<<gdim, bdim>>>(js); return 0;
            default: return -1;
        }
    }
    return -1;
}

/* out[j*ff + k] = w[j] * silu(gate) * up. Folding the router weight here means
 * the separate weighting pass the CPU path does is not needed. */
__global__ void kce_silu_weight_kernel(const float *__restrict__ g,
                                       const float *__restrict__ u,
                                       KceWeights wt,
                                       float *__restrict__ out,
                                       unsigned ff, unsigned total) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    unsigned j = i / ff;
    if (j >= (unsigned)wt.n) return;
    float gv = g[i], uv = u[i];
    float sv = gv / (1.f + expf(-gv));
    out[i] = wt.w[j] * sv * uv;
}

/* Stage A: the same silu*router-weight pass, additionally emitting the q8_1
 * activation the DP4A down-projection kernels consume. Fusing it here removes a
 * whole kernel launch AND a re-read of the intermediate per layer, and it is
 * exactly valid because this kernel is elementwise with one thread per value:
 * as long as the intermediate width is a multiple of 32 and the block size is,
 * a warp owns exactly one 32-value q8_1 block of a single expert's slice.
 * Caller checks both; `out` is still written because it is the fp32-path
 * contract. */
__global__ void kce_silu_weight_q8_kernel(const float *__restrict__ g,
                                          const float *__restrict__ u,
                                          KceWeights wt,
                                          float *__restrict__ out,
                                          KceQ8_1 *__restrict__ q,
                                          unsigned ff, unsigned total) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned lane = threadIdx.x & 31u;
    float sv = 0.f;
    bool valid = false;
    if (i < total) {
        const unsigned j = i / ff;
        if (j < (unsigned)wt.n) {
            const float gv = g[i], uv = u[i];
            sv = gv / (1.f + expf(-gv));
            sv = wt.w[j] * sv * uv;
            out[i] = sv;
            valid = true;
        }
    }
    /* The reductions below use the full mask, so every lane must reach them;
     * hence no early return. Invalid values contribute 0. */
    float amax = fabsf(sv);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const float d = amax / 127.0f;
    int qi = 0;
    if (d > 0.f) {
        qi = (int)lrintf(sv / d);
        if (qi > 127) qi = 127;
        if (qi < -127) qi = -127;
    }
    int qs = qi;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        qs += __shfl_xor_sync(0xffffffffu, qs, o);
    int qs_lo = (lane < 16u) ? qi : 0;
#pragma unroll
    for (int o = 8; o > 0; o >>= 1)
        qs_lo += __shfl_xor_sync(0xffffffffu, qs_lo, o);
    if (valid) {
        const unsigned blk = i >> 5;
        q[blk].qs[lane] = (int8_t)qi;
        if (lane == 0u) {
            q[blk].d = d;
            q[blk].s = (float)qs;
            q[blk].s_lo = (float)qs_lo;
        }
    }
}

/* Deterministic sum in selected-expert order, matching the CPU accumulation. */
__global__ void kce_reduce_experts_kernel(const float *__restrict__ src,
                                          float *__restrict__ dst,
                                          unsigned h, int n_sel) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= h) return;
    float a = 0.f;
    for (int j = 0; j < n_sel; j++) a += src[(size_t)j * h + i];
    dst[i] = a;
}

/* Persisted device scratch for the fused path (reallocated only when the
 * required size grows, so steady-state has zero cudaMalloc traffic). */
static void  *g_scr_gate = NULL, *g_scr_up = NULL, *g_scr_int = NULL, *g_scr_down = NULL;
/* Stage A: the q8_1 activations (layer input, and the per-expert intermediate)
 * live on the device too, so no quantized activation ever crosses PCIe. */
static void  *g_scr_q8x = NULL, *g_scr_q8i = NULL;
/* Recorded by katali_cuda_moe_submit(); awaited by katali_cuda_moe_wait().
 * DisableTiming: this is a fence, not a measurement. */
static cudaEvent_t g_moe_done = NULL;
static size_t g_scr_sel = 0, g_scr_h = 0, g_scr_ff = 0;

static void scr_release(void) {
    if (g_scr_gate) { cudaFree(g_scr_gate); g_scr_gate = NULL; }
    if (g_scr_up)   { cudaFree(g_scr_up);   g_scr_up = NULL; }
    if (g_scr_int)  { cudaFree(g_scr_int);  g_scr_int = NULL; }
    if (g_scr_down) { cudaFree(g_scr_down); g_scr_down = NULL; }
    if (g_scr_q8x)  { cudaFree(g_scr_q8x);  g_scr_q8x = NULL; }
    if (g_scr_q8i)  { cudaFree(g_scr_q8i);  g_scr_q8i = NULL; }
    g_scr_sel = g_scr_h = g_scr_ff = 0;
}

static int scr_ensure(size_t sel, size_t h, size_t ff) {
    if (g_scr_gate && sel <= g_scr_sel && h <= g_scr_h && ff <= g_scr_ff) return 1;
    scr_release();
    size_t s = sel, hh = h, f = ff;
    if (cudaMalloc(&g_scr_gate, s * f * sizeof(float)) != cudaSuccess) { scr_release(); return 0; }
    if (cudaMalloc(&g_scr_up,   s * f * sizeof(float)) != cudaSuccess) { scr_release(); return 0; }
    if (cudaMalloc(&g_scr_int,  s * f * sizeof(float)) != cudaSuccess) { scr_release(); return 0; }
    if (cudaMalloc(&g_scr_down, s * hh * sizeof(float)) != cudaSuccess) { scr_release(); return 0; }
    /* q8_1 scratch, rounded up to whole 32-value blocks. */
    const size_t q8x = ((hh + KCE_QK8_1 - 1) / KCE_QK8_1) * KCE_Q8_1_BYTES;
    const size_t q8i = s * (((f + KCE_QK8_1 - 1) / KCE_QK8_1) * KCE_Q8_1_BYTES);
    if (q8x && cudaMalloc(&g_scr_q8x, q8x) != cudaSuccess) { scr_release(); return 0; }
    if (q8i && cudaMalloc(&g_scr_q8i, q8i) != cudaSuccess) { scr_release(); return 0; }
    g_scr_sel = s; g_scr_h = hh; g_scr_ff = f;
    return 1;
}

/* KCE_BLOCK / KCE_JOBS_BLOCK are defined with the Stage A section above. */

/* ===================================================================== *
 * Phase 8 — clock warm-keeping (EXPERIMENT, off by default).
 *
 * Measured problem: during real inference the 4060 sits at 210 MHz SM /
 * 405 MHz memory (its idle state) because the GPU is only ~6% duty-cycled, so
 * the memory clock collapses 21x and every memory-bound kernel starves.
 *
 * This launches ONE thread performing ONE float add every `interval_ms` on a
 * non-blocking stream, and asks whether that alone is enough to stop the
 * memory clock from collapsing. It consumes a negligible amount of GPU work by
 * construction (grid=1, block=1, one FMA); the question is purely whether the
 * driver's power state responds to activity *presence* or to *load*.
 *
 * Not enabled by default. Enable with KATALI_CUDA_KEEPWARM=<ms>, 0/false to
 * disable. Never a substitute for real useful work.
 * ===================================================================== */
__global__ void kce_keepwarm_kernel(float *p) { p[0] = p[0] + 1.0e-30f; }

static std::thread          g_kw_thread;
static std::atomic<int>     g_kw_stop(1);
static std::atomic<int>     g_kw_interval(0);
static cudaStream_t         g_kw_stream = NULL;
static float               *g_kw_buf = NULL;
static uint64_t             g_kw_launches = 0;

static void kw_loop(void) {
    while (g_kw_stop.load() == 0) {
        if (g_kw_buf && g_kw_stream) {
            kce_keepwarm_kernel<<<1, 1, 0, g_kw_stream>>>(g_kw_buf);
            g_kw_launches++;
        }
        int iv = g_kw_interval.load();
        if (iv < 1) iv = 1;
        std::this_thread::sleep_for(std::chrono::milliseconds(iv));
    }
}

/* interval_ms <= 0 stops it. Returns 0 on success. */
KCE_API int katali_cuda_keepwarm(int interval_ms) {
    if (interval_ms <= 0) {
        if (g_kw_stop.load() == 0) {
            g_kw_stop.store(1);
            if (g_kw_thread.joinable()) g_kw_thread.join();
        }
        return 0;
    }
    if (!g_ready) return -1;
    if (g_kw_stop.load() == 0) {          /* already running: retune only */
        g_kw_interval.store(interval_ms);
        return 0;
    }
    if (!g_kw_buf) {
        if (cudaMalloc(&g_kw_buf, sizeof(float)) != cudaSuccess) { g_kw_buf = NULL; return -1; }
        (void)cudaMemset(g_kw_buf, 0, sizeof(float));
    }
    if (!g_kw_stream) {
        if (cudaStreamCreateWithFlags(&g_kw_stream, cudaStreamNonBlocking) != cudaSuccess) {
            g_kw_stream = NULL;
            return -1;
        }
    }
    g_kw_interval.store(interval_ms);
    g_kw_stop.store(0);
    g_kw_thread = std::thread(kw_loop);
    return 0;
}

/* Launch count, so the experiment's own GPU cost can be reported. */
KCE_API uint64_t katali_cuda_keepwarm_launches(void) { return g_kw_launches; }

/* ===================================================================== *
 * Phase 9 — raw VRAM streaming ceiling.
 *
 * A working set far larger than L2, read with 16-byte vector loads and a
 * grid-stride loop: the maximum number of independent requests the card can
 * have outstanding. This is the hardware ceiling every GEMV number is judged
 * against.
 * ===================================================================== */
__global__ void kce_stream_read_kernel(const float4 *__restrict__ src,
                                       unsigned long long n4,
                                       float *__restrict__ sink) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    float acc = 0.f;
    for (; i < n4; i += stride) {
        const float4 v = __ldg(&src[i]);
        acc += v.x + v.y + v.z + v.w;
    }
    /* Never true; stops the compiler from deleting the whole loop. */
    if (acc == 1234567.891f) sink[0] = acc;
}

__global__ void kce_stream_copy_kernel(const float4 *__restrict__ src,
                                       float4 *__restrict__ dst,
                                       unsigned long long n4) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    for (; i < n4; i += stride) dst[i] = __ldg(&src[i]);
}

/* ===================================================================== *
 * Phase 8 — the SAME access pattern as the GEMV, with the math removed.
 *
 * Reads exactly the same weight bytes (16-byte vector loads instead of the
 * scalar per-element dequant reads) and exactly the same activation elements,
 * but performs no dequantization and no per-element MAC. Comparing this against
 * the real GEMV separates "memory access pattern" from "decode cost".
 *
 * row_bytes is not always a multiple of 16 (Q6_K: 420), so the read starts at
 * the 16-byte-aligned floor of the row and reads ceil(row_bytes/16) chunks —
 * at most 15 bytes of over-read per row, immaterial for a bandwidth probe.
 * ===================================================================== */
template <int BLOCK>
__global__ void kce_jobs_readonly_kernel(KceJobSet js) {
    const int j = blockIdx.y;
    if (j >= js.n) return;
    const unsigned row = blockIdx.x;
    if (row >= js.job[j].rows) return;
    const unsigned char *wbase = js.job[j].w;
    const unsigned rb = js.job[j].row_bytes;
    const size_t byte_off = (size_t)row * rb;
    const unsigned char *rp = wbase + (byte_off & ~(size_t)15);
    const unsigned n16 = (unsigned)((rb + 15u) >> 4);
    const unsigned cols = js.job[j].cols;
    const float *x = js.job[j].x;

    float acc = 0.f;
    /* Activation traffic, same as the real kernel (trivial math). */
    for (unsigned c = threadIdx.x; c < cols; c += BLOCK)
        acc += x[c] * 0.0009765625f;
    /* Weight traffic, same bytes, no dequant. */
    for (unsigned k = threadIdx.x; k < n16; k += BLOCK) {
        const uint4 v = __ldg(((const uint4 *)rp) + k);
        acc += (float)(v.x + v.y + v.z + v.w);
    }

    __shared__ float sh[BLOCK / 32];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if ((threadIdx.x & 31u) == 0u) sh[threadIdx.x >> 5] = acc;
    __syncthreads();
    if (threadIdx.x < 32u) {
        acc = (threadIdx.x < (unsigned)(BLOCK / 32)) ? sh[threadIdx.x] : 0.f;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
        if (threadIdx.x == 0u) js.job[j].out[row] = acc;
    }
}

static uint64_t kce_row_bytes(uint32_t type, uint64_t cols) {
    switch (type) {
        case G_F32:  return cols * 4u;
        case G_Q4_0: return (cols / 32u)  * 18u;
        case G_Q8_0: return (cols / 32u)  * 34u;
        case G_Q4_K: return (cols / 256u) * 144u;
        case G_Q6_K: return (cols / 256u) * 210u;
        default: return 0;
    }
}

/* One GEMV against device-resident weights. Synchronizes before returning, so
 * `y` is valid immediately (the pipelined variant arrives in a later stage). */
KCE_API int katali_cuda_matvec(uint32_t type, const void *w,
                               uint64_t rows, uint64_t cols,
                               const float *x, float *y) {
    if (!g_ready || !w || !x || !y) return -1;
    if (rows == 0 || cols == 0) return -1;
    if (rows > 0x7fffffffu || cols > 0x7fffffffu) return -1;
    uint64_t rb = kce_row_bytes(type, cols);
    if (rb == 0 || rb > 0xffffffffu) return -1;

    dim3 grid((unsigned)rows), block(KCE_BLOCK);
    const uint8_t *wp = (const uint8_t *)w;
    unsigned r32 = (unsigned)rows, c32 = (unsigned)cols, rb32 = (unsigned)rb;

    tel_begin();
    switch (type) {
        case G_F32:  kce_gemv_kernel<G_F32,  KCE_BLOCK><<<grid, block>>>(wp, x, y, r32, c32, rb32); break;
        case G_Q4_0: kce_gemv_kernel<G_Q4_0, KCE_BLOCK><<<grid, block>>>(wp, x, y, r32, c32, rb32); break;
        case G_Q8_0: kce_gemv_kernel<G_Q8_0, KCE_BLOCK><<<grid, block>>>(wp, x, y, r32, c32, rb32); break;
        case G_Q4_K: kce_gemv_kernel<G_Q4_K, KCE_BLOCK><<<grid, block>>>(wp, x, y, r32, c32, rb32); break;
        case G_Q6_K: kce_gemv_kernel<G_Q6_K, KCE_BLOCK><<<grid, block>>>(wp, x, y, r32, c32, rb32); break;
        default: return -1;
    }
    tel_launch();
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return -1;
    tel_sync();
    tel_end();
    return 0;
}

/* Batched matmul. KNOWN LIMITATION: each batch row re-reads W, so weight
 * traffic is B x that of a properly tiled GEMM. Correct, and fine for the
 * decode-shaped A/B, but it is NOT expected to beat the CPU's
 * katali_ggml_matmul on prefill — that needs a shared-memory tiled GEMM. */
KCE_API int katali_cuda_matmul(uint32_t type, const void *w,
                               uint64_t rows, uint64_t cols,
                               const float *X, uint64_t B, float *Y) {
    if (!g_ready || !w || !X || !Y || B == 0) return -1;
    for (uint64_t b = 0; b < B; b++) {
        int rc = katali_cuda_matvec(type, w, rows, cols, X + b * cols, Y + b * rows);
        if (rc != 0) return rc;
    }
    return 0;
}
KCE_API int katali_cuda_supports_type(uint32_t type) {
    switch (type) {
        case G_F32: case G_Q4_0: case G_Q4_K: case G_Q6_K: case G_Q8_0:
            return 1;
        default:
            return 0;
    }
}

/* Dispatch one job-batched GEMV launch for the given GGML type and unroll
 * factor. */
#define KCE_LAUNCH_JOBS_U(TYPE, U, JS, GRID, BLK)                              \
    do {                                                                       \
        switch (TYPE) {                                                        \
            case G_F32:  kce_gemv_jobs_kernel<G_F32,  KCE_JOBS_BLOCK, U>       \
                             <<<(GRID), (BLK)>>>((JS)); break;                 \
            case G_Q4_0: kce_gemv_jobs_kernel<G_Q4_0, KCE_JOBS_BLOCK, U>       \
                             <<<(GRID), (BLK)>>>((JS)); break;                 \
            case G_Q8_0: kce_gemv_jobs_kernel<G_Q8_0, KCE_JOBS_BLOCK, U>       \
                             <<<(GRID), (BLK)>>>((JS)); break;                 \
            case G_Q4_K: kce_gemv_jobs_kernel<G_Q4_K, KCE_JOBS_BLOCK, U>       \
                             <<<(GRID), (BLK)>>>((JS)); break;                 \
            case G_Q6_K: kce_gemv_jobs_kernel<G_Q6_K, KCE_JOBS_BLOCK, U>       \
                             <<<(GRID), (BLK)>>>((JS)); break;                 \
            default: return -1;                                                \
        }                                                                      \
    } while (0)

/* Unroll factor for the job GEMV. Both variants are compiled in so the A/B is
 * a single env change; the winner becomes the default after measurement.
 * KATALI_GEMV_UNROLL must be set before the process starts (the DLL's CRT
 * snapshots the environment at LoadLibrary time). */
static int gemv_unroll(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("KATALI_GEMV_UNROLL");
        v = (e && *e) ? atoi(e) : 1;
        if (v != 1 && v != 4) v = 1;
    }
    return v;
}

#define KCE_LAUNCH_JOBS(TYPE, JS, GRID, BLK)                                   \
    do {                                                                       \
        if (gemv_unroll() > 1) KCE_LAUNCH_JOBS_U(TYPE, 4, JS, GRID, BLK);      \
        else                   KCE_LAUNCH_JOBS_U(TYPE, 1, JS, GRID, BLK);      \
    } while (0)

/* Read-only variant of the job-batched GEMV: identical access pattern, no
 * dequant. Type-independent, so no dispatch switch is needed. */
#define KCE_LAUNCH_JOBS_NM(JS, GRID, BLK)                                      \
    kce_jobs_readonly_kernel<KCE_JOBS_BLOCK><<<(GRID), (BLK)>>>((JS))

/* ===================================================================== *
 * Stage A — DP4A path selection.
 *
 * Default OFF: KATALI_CUDA_DP4A unset (or 0) keeps the fp32-dequantization
 * kernels, byte-for-byte. `katali_cuda_set_dp4a()` overrides the environment so
 * a single process can A/B both arms; the per-type switches exist so Q4_K and
 * Q6_K can be measured separately (Q4_K is 60 % of this model's expert bytes,
 * but Q6_K's dequantization is the more expensive one per element, so which one
 * to attack first is a measurement, not an assumption).
 * ===================================================================== */
static int g_dp4a_force = -1;   /* -1 = env, 0 = forced off, 1 = forced on */
static int g_dp4a_q4k = -1;     /* -1 = env, else forced */
static int g_dp4a_q6k = -1;
static int g_dp4a_active = 0;   /* 1 when the last fused layer used DP4A */

KCE_API void katali_cuda_set_dp4a(int on) {
    g_dp4a_force = (on < 0) ? -1 : (on ? 1 : 0);
}
KCE_API void katali_cuda_set_dp4a_types(int q4_k, int q6_k) {
    g_dp4a_q4k = q4_k ? 1 : 0;
    g_dp4a_q6k = q6_k ? 1 : 0;
}
KCE_API int katali_cuda_dp4a_active(void) { return g_dp4a_active; }

static int env_flag(const char *name, int dflt) {
    const char *e = getenv(name);
    if (!e || !*e) return dflt;
    return !(e[0] == '0' || e[0] == 'n' || e[0] == 'N' || e[0] == 'f' || e[0] == 'F');
}

static int dp4a_on(void) {
    static int v = -1;
    if (g_dp4a_force >= 0) return g_dp4a_force;
    if (v < 0) v = env_flag("KATALI_CUDA_DP4A", 0);
    return v;
}

static int dp4a_type_on(uint32_t type) {
    static int q4 = -1, q6 = -1;
    if (type == G_Q4_K) {
        if (g_dp4a_q4k >= 0) return g_dp4a_q4k;
        if (q4 < 0) q4 = env_flag("KATALI_CUDA_DP4A_Q4K", 1);
        return q4;
    }
    if (type == G_Q6_K) {
        if (g_dp4a_q6k >= 0) return g_dp4a_q6k;
        if (q6 < 0) q6 = env_flag("KATALI_CUDA_DP4A_Q6K", 1);
        return q6;
    }
    return 0;
}

/* Words (4 activation values each) owned by one thread. The kernel is written
 * so that a thread's `NW` words are consecutive and lie inside a single q8_1
 * block (8 words) and, for Q6_K, inside a single 16-value scale group (4 words)
 * — which is what makes the scale/min terms countable exactly once. */
static unsigned dp4a_words_per_thread(uint64_t cols) {
    if (cols == 0 || (cols % 512u) != 0) return 0;   /* 512 = BLOCK * 4 values */
    return (unsigned)((cols / 4u) / KCE_JOBS_BLOCK);
}

static int dp4a_ok(uint32_t type, uint64_t cols) {
    if (!dp4a_type_on(type)) return 0;
    const unsigned nw = dp4a_words_per_thread(cols);
    if (type == G_Q4_K) return (nw == 1 || nw == 2 || nw == 4 || nw == 8);
    if (type == G_Q6_K) return (nw == 1 || nw == 2 || nw == 4);
    return 0;
}

/* Fused layer-level MoE. See include/katali_cuda.h for the full contract.
 *
 * Launch plan (one layer):
 *   1  gate + up, job-batched   (one launch when both are the same type)
 *   1  silu(gate)*up * router weight
 *   1  down, job-batched
 *   1  deterministic reduce over the selected experts
 *   1  single device synchronize for the whole layer
 * The caller owns x/y and therefore the only two PCIe transfers. */
static int moe_layer_impl(const KataliCudaMoeDesc *d,
                          const void *const *gate_dev,
                          const void *const *up_dev,
                          const void *const *down_dev,
                          const float *w, int n_sel,
                          const float *x, float *y, int nomath, int nosync) {
    if (!g_ready || !d || !gate_dev || !up_dev || !down_dev) return -1;
    if (!w || !x || !y) return -1;
    if (n_sel <= 0 || n_sel > KCE_MAX_SEL) return -1;
    if (d->hidden == 0 || d->inter == 0) return -1;
    if (d->hidden > 0x7fffffffu || d->inter > 0x7fffffffu) return -1;
    if (d->gate_rows == 0 || d->up_rows == 0 || d->down_rows == 0) return -1;
    if (d->gate_row_bytes == 0 || d->up_row_bytes == 0 ||
        d->down_row_bytes == 0) return -1;
    if (d->gate_row_bytes > 0xffffffffu || d->up_row_bytes > 0xffffffffu ||
        d->down_row_bytes > 0xffffffffu) return -1;
    for (int j = 0; j < n_sel; j++)
        if (!gate_dev[j] || !up_dev[j] || !down_dev[j]) return -1;
    if (!katali_cuda_supports_type(d->gate_type) ||
        !katali_cuda_supports_type(d->up_type) ||
        !katali_cuda_supports_type(d->down_type)) return -1;
    if (!scr_ensure((size_t)n_sel, (size_t)d->hidden, (size_t)d->inter)) return -1;

    const unsigned H = (unsigned)d->hidden, FF = (unsigned)d->inter;
    const unsigned gr = (unsigned)d->gate_rows, ur = (unsigned)d->up_rows;
    const unsigned dr = (unsigned)d->down_rows;
    const unsigned rb_g = (unsigned)d->gate_row_bytes;
    const unsigned rb_u = (unsigned)d->up_row_bytes;
    const unsigned rb_d = (unsigned)d->down_row_bytes;
    float *const gs = (float *)g_scr_gate;
    float *const us = (float *)g_scr_up;
    float *const is = (float *)g_scr_int;
    float *const ds = (float *)g_scr_down;

    KceWeights wt;
    memset(&wt, 0, sizeof(wt));
    wt.n = n_sel;
    for (int j = 0; j < n_sel; j++) wt.w[j] = w[j];

    /*
     * Stage A: decide the whole layer's quantization strategy BEFORE the first
     * launch. A half-quantized layer is not a thing — every gate below is
     * checked up front, and any failure silently means "this projection keeps
     * the fp32 kernel", so the path can never produce a mixed-up result.
     *
     *   use_gu   gate+up share one launch (same type) and both are DP4A-capable
     *   use_up   separate up launch when the types differ
     *   use_dn   the down projection is DP4A-capable
     *
     * The activation quantization itself is per LAYER (both q8_1 buffers are
     * device-resident scratch): all selected experts consume the same layer
     * input, which is what amortizes that cost across n_sel experts.
     */
    int use_gu = 0, use_up = 0, use_dn = 0;
    const int up_separate = (d->up_type != d->gate_type);
    if (!nomath && dp4a_on()) {
        /* Each projection is decided independently: disabling one type must not
         * silently disable the other, or the per-type A/B would measure fp32
         * twice and report "no speedup" as if it were a kernel property. */
        const int gate_ok = dp4a_ok(d->gate_type, H);
        const int up_ok   = dp4a_ok(d->up_type,   H);
        const int down_ok = dp4a_ok(d->down_type, FF);
        const int fused_silu_ok = ((FF % KCE_QK8_1) == 0 && (H % KCE_QK8_1) == 0 &&
                                   (FF % 32u) == 0 && g_scr_q8i && g_scr_q8x);
        if (fused_silu_ok) {
            use_gu = up_separate ? gate_ok : (gate_ok && up_ok);
            use_up = up_separate ? up_ok : 0;
            use_dn = down_ok;
        }
    }
    g_dp4a_active = (use_gu || use_up || use_dn) ? 1 : 0;
    const unsigned nw_gu = dp4a_words_per_thread(H);
    const unsigned nw_dn = dp4a_words_per_thread(FF);
    const int need_q8x = (use_gu || use_up);

    tel_begin();

    /* --- one-time activation quantization, on the device --- */
    if (need_q8x) {
        const unsigned warps = (H + KCE_QK8_1 - 1u) / KCE_QK8_1;
        const unsigned blk = 256u;                      /* 8 warps per block */
        const unsigned blocks = (warps + 7u) / 8u;
        q8_tel_begin();
        kce_quant_q8_1_kernel<<<blocks, blk>>>(x, (KceQ8_1 *)g_scr_q8x, H);
        tel_launch();
        q8_tel_end();
        KCE_CHECK("after q8_1 quantize");
    }

    /* --- gate (+ up when the types match) --- */
    KceJobSet gu;
    memset(&gu, 0, sizeof(gu));
    gu.max_rows = (gr > ur) ? gr : ur;
    for (int j = 0; j < n_sel; j++) {
        KceJobDesc *jd = &gu.job[gu.n++];
        jd->w = (const unsigned char *)gate_dev[j];
        jd->x = x;
        jd->xq = (const KceQ8_1 *)g_scr_q8x;
        jd->out = gs + (size_t)j * FF;
        jd->rows = gr; jd->cols = H; jd->row_bytes = rb_g;
    }
    if (!up_separate) {
        for (int j = 0; j < n_sel; j++) {
            KceJobDesc *jd = &gu.job[gu.n++];
            jd->w = (const unsigned char *)up_dev[j];
            jd->x = x;
            jd->xq = (const KceQ8_1 *)g_scr_q8x;
            jd->out = us + (size_t)j * FF;
            jd->rows = ur; jd->cols = H; jd->row_bytes = rb_u;
        }
    }
    dim3 g1(gu.max_rows, (unsigned)gu.n), b1(KCE_JOBS_BLOCK);
    if (nomath) KCE_LAUNCH_JOBS_NM(gu, g1, b1);
    else if (use_gu) {
        if (kce_launch_jobs_dp4a(d->gate_type, nw_gu, gu, g1, b1) != 0) return -1;
    } else KCE_LAUNCH_JOBS(d->gate_type, gu, g1, b1);
    tel_launch();
    KCE_CHECK("after gate/up");

    if (up_separate) {
        KceJobSet uu;
        memset(&uu, 0, sizeof(uu));
        uu.max_rows = ur; uu.n = n_sel;
        for (int j = 0; j < n_sel; j++) {
            KceJobDesc *jd = &uu.job[j];
            jd->w = (const unsigned char *)up_dev[j];
            jd->x = x;
            jd->xq = (const KceQ8_1 *)g_scr_q8x;
            jd->out = us + (size_t)j * FF;
            jd->rows = ur; jd->cols = H; jd->row_bytes = rb_u;
        }
        dim3 g2(ur, (unsigned)n_sel), b2(KCE_JOBS_BLOCK);
        if (nomath) KCE_LAUNCH_JOBS_NM(uu, g2, b2);
        else if (use_up) {
            if (kce_launch_jobs_dp4a(d->up_type, nw_gu, uu, g2, b2) != 0) return -1;
        } else KCE_LAUNCH_JOBS(d->up_type, uu, g2, b2);
        tel_launch();
        KCE_CHECK("after up");
    }

    /* --- activation + router weighting, entirely on device --- */
    {
        const unsigned total = (unsigned)n_sel * FF;
        const unsigned blk = 256u;
        const unsigned blocks = (total + blk - 1u) / blk;
        if (use_dn)
            kce_silu_weight_q8_kernel<<<blocks, blk>>>(gs, us, wt, is,
                                                       (KceQ8_1 *)g_scr_q8i, FF, total);
        else
            kce_silu_weight_kernel<<<blocks, blk>>>(gs, us, wt, is, FF, total);
        tel_launch();
        KCE_CHECK("after silu+q8");
    }

    /* --- down projection for all selected experts --- */
    {
        KceJobSet dn;
        memset(&dn, 0, sizeof(dn));
        dn.max_rows = dr; dn.n = n_sel;
        for (int j = 0; j < n_sel; j++) {
            KceJobDesc *jd = &dn.job[j];
            jd->w = (const unsigned char *)down_dev[j];
            jd->x = is + (size_t)j * FF;
            jd->xq = (const KceQ8_1 *)g_scr_q8i + (size_t)j * (FF / KCE_QK8_1);
            jd->out = ds + (size_t)j * H;
            jd->rows = dr; jd->cols = FF; jd->row_bytes = rb_d;
        }
        dim3 g3(dr, (unsigned)n_sel), b3(KCE_JOBS_BLOCK);
        if (nomath) KCE_LAUNCH_JOBS_NM(dn, g3, b3);
        else if (use_dn) {
            if (kce_launch_jobs_dp4a(d->down_type, nw_dn, dn, g3, b3) != 0) return -1;
        } else KCE_LAUNCH_JOBS(d->down_type, dn, g3, b3);
        tel_launch();
        KCE_CHECK("after down");
    }

    /* --- deterministic reduce in selected-expert order --- */
    {
        const unsigned blk = 256u;
        const unsigned blocks = (H + blk - 1u) / blk;
        kce_reduce_experts_kernel<<<blocks, blk>>>(ds, y, H, n_sel);
        tel_launch();
        KCE_CHECK("fused layer launch");
    }

    /*
     * Synchronisation policy.
     *
     * nosync=0: block here (the original behaviour, kept for the synchronous
     *           API and for A/B).
     * nosync=1: record an event instead and return immediately, so the caller
     *           can do useful CPU work (the shared expert) while the GPU runs.
     *           katali_cuda_moe_wait() then blocks once. This is the only
     *           synchronisation removed; the caller MUST wait before reading
     *           `y`, and the D2H copy that follows is stream-ordered anyway.
     */
    if (nosync) {
        if (!g_moe_done)
            (void)cudaEventCreateWithFlags(&g_moe_done, cudaEventDisableTiming);
        if (g_moe_done) {
            (void)cudaEventRecord(g_moe_done, 0);
            /* The event-timing window is closed in katali_cuda_moe_wait(), so the
             * async path still reports a GPU span comparable to the sync path. */
            return 0;
        }
        /* Event creation failed: fall through to the blocking path rather than
         * returning an unsynchronised result. */
    }
    /* ONE synchronization for the whole layer, not one per projection. */
    tel_sync();
    /* An illegal address inside a kernel surfaces here, not at launch time. */
    KCE_CHECK("fused layer sync");
    tel_end();
    return 0;
}

/* Public entry points: the real fused layer, and the no-math variant used to
 * separate memory access from dequantization cost (Phase 8). */
KCE_API int katali_cuda_moe_layer(const KataliCudaMoeDesc *d,
                                  const void *const *gate_dev,
                                  const void *const *up_dev,
                                  const void *const *down_dev,
                                  const float *w, int n_sel,
                                  const float *x, float *y) {
    return moe_layer_impl(d, gate_dev, up_dev, down_dev, w, n_sel, x, y, 0, 0);
}

/* Asynchronous variant (Phase 4). Returns as soon as the work is QUEUED; the
 * caller must call katali_cuda_moe_wait() before reading `y`. Only one
 * submission may be outstanding at a time, because the fused path reuses one
 * set of internal device scratch buffers. */
KCE_API int katali_cuda_moe_submit(const KataliCudaMoeDesc *d,
                                   const void *const *gate_dev,
                                   const void *const *up_dev,
                                   const void *const *down_dev,
                                   const float *w, int n_sel,
                                   const float *x, float *y) {
    return moe_layer_impl(d, gate_dev, up_dev, down_dev, w, n_sel, x, y, 0, 1);
}

/* Blocks until the outstanding submission (if any) has completed. */
KCE_API int katali_cuda_moe_wait(void) {
    if (!g_moe_done) return 0;      /* nothing outstanding */
    g_syncs++;
    const int rc = (cudaEventSynchronize(g_moe_done) == cudaSuccess) ? 0 : -1;
    /* Close the device-event window opened in moe_layer_impl's nosync branch, so
     * the async path reports the same kind of GPU span as the synchronous one. */
    if (gpu_time_enabled()) tel_end();
    return rc;
}

KCE_API int katali_cuda_moe_layer_nomath(const KataliCudaMoeDesc *d,
                                         const void *const *gate_dev,
                                         const void *const *up_dev,
                                         const void *const *down_dev,
                                         const float *w, int n_sel,
                                         const float *x, float *y) {
    return moe_layer_impl(d, gate_dev, up_dev, down_dev, w, n_sel, x, y, 1, 0);
}

/* Phase 9: raw VRAM streaming ceiling. `a` is the source buffer, `b` a
 * same-sized scratch buffer for the copy test. Both must be device pointers. */
KCE_API int katali_cuda_bench_stream(void *a, void *b, size_t bytes, int reps,
                                     double *read_gbs, double *copy_gbs) {
    if (!g_ready || !a || !b || bytes < (1u << 20) || reps < 1) return -1;
    if (bytes % 16u) return -1;
    const unsigned long long n4 = (unsigned long long)(bytes / 16u);
    const unsigned blocks = 2048u, threads = 256u;
    const double gb = (double)bytes / (1024.0 * 1024.0 * 1024.0);
    float *sink = NULL;
    cudaEvent_t e0 = NULL, e1 = NULL;
    if (cudaMalloc(&sink, sizeof(float)) != cudaSuccess) return -1;
    if (cudaEventCreate(&e0) != cudaSuccess ||
        cudaEventCreate(&e1) != cudaSuccess) {
        cudaFree(sink);
        return -1;
    }
    float ms = 0.f;

    if (read_gbs) {
        kce_stream_read_kernel<<<blocks, threads>>>((const float4 *)a, n4, sink);
        tel_launch();
        if (cudaGetLastError() != cudaSuccess) {
            cudaFree(sink); cudaEventDestroy(e0); cudaEventDestroy(e1);
            return -1;
        }
        cudaEventRecord(e0, 0);
        for (int r = 0; r < reps; r++)
            kce_stream_read_kernel<<<blocks, threads>>>((const float4 *)a, n4, sink);
        tel_launch();
        cudaEventRecord(e1, 0);
        cudaEventSynchronize(e1);
        cudaEventElapsedTime(&ms, e0, e1);
        const double dt = (double)ms / 1000.0;
        *read_gbs = (dt > 0.0) ? (gb * (double)reps) / dt : 0.0;
    }
    if (copy_gbs) {
        cudaMemcpy(b, a, bytes, cudaMemcpyDeviceToDevice);   /* warm dst pages */
        g_memcpys++;
        cudaEventRecord(e0, 0);
        for (int r = 0; r < reps; r++)
            kce_stream_copy_kernel<<<blocks, threads>>>((const float4 *)a,
                                                        (float4 *)b, n4);
        tel_launch();
        cudaEventRecord(e1, 0);
        cudaEventSynchronize(e1);
        cudaEventElapsedTime(&ms, e0, e1);
        const double dt = (double)ms / 1000.0;
        /* A copy moves each byte twice: read + write. */
        *copy_gbs = (dt > 0.0) ? (2.0 * gb * (double)reps) / dt : 0.0;
    }
    cudaFree(sink);
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    return 0;
}
