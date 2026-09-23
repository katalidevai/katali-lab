/* katali-lab — optional CUDA backend runtime loader — Apache-2.0
 *
 * Never links against CUDA. Finds katali_cuda.dll at runtime and resolves a flat
 * C ABI. Any failure is reported as a status string and leaves the engine on the
 * CPU + system RAM + SSD path. Nothing here aborts the process.
 *
 * Search order for the backend DLL:
 *   1. $KATALI_CUDA_DLL (explicit path/name)
 *   2. <directory of katali-lab.exe>\katali_cuda.dll
 *   3. "katali_cuda.dll" (default library search order)
 * KATALI_CUDA=0 disables the probe entirely (no filesystem access).
 */
#include "katali_cuda.h"
#include "platform.h"
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
typedef HMODULE klib_t;
#define KATALI_PATH_SEP '\\'
static klib_t klib_open(const char *p) { return LoadLibraryA(p); }
static void  *klib_sym(klib_t h, const char *n) { return (void *)GetProcAddress(h, n); }
static void   klib_close(klib_t h) { if (h) FreeLibrary(h); }
static int    klib_err(void) { return (int)GetLastError(); }
static int    klib_file_exists(const char *p) {
    DWORD a = GetFileAttributesA(p);
    return a != INVALID_FILE_ATTRIBUTES && !(a & FILE_ATTRIBUTE_DIRECTORY);
}
#else
#include <dlfcn.h>
#include <unistd.h>
typedef void *klib_t;
#define KATALI_PATH_SEP '/'
static klib_t klib_open(const char *p) { return dlopen(p, RTLD_NOW | RTLD_LOCAL); }
static void  *klib_sym(klib_t h, const char *n) { return dlsym(h, n); }
static void   klib_close(klib_t h) { if (h) dlclose(h); }
static int    klib_err(void) { return 0; }
static int    klib_file_exists(const char *p) { return access(p, R_OK) == 0; }
#endif

/* 126 = ERROR_MOD_NOT_FOUND; found but a dependency is missing.
 * 193 = ERROR_BAD_EXE_FORMAT; found but wrong architecture (x86 vs x64). */
#define KLIB_MOD_NOT_FOUND    126
#define KLIB_BAD_EXE_FORMAT   193

/* Resolve a symbol into a function pointer without a function-pointer cast. */
#define KATALI_SYM(h, name, fnptr) \
    do { *(void **)(&(fnptr)) = klib_sym((h), (name)); } while (0)

typedef struct CudaCtx {
    int      probed;     /* probe ran (result cached) */
    int      disabled;   /* katali_cuda_shutdown() called */
    int      ok;         /* backend usable */
    klib_t   lib;
    char     lib_path[1024];
    char     status[512];
    KataliCudaDeviceInfo dev;

    katali_cuda_abi_version_fn        abi_version;
    katali_cuda_backend_init_fn       backend_init;
    katali_cuda_backend_shutdown_fn   backend_shutdown;
    katali_cuda_supports_type_fn      supports_type;
    katali_cuda_matvec_fn             matvec;
    katali_cuda_matmul_fn             matmul;
    katali_cuda_device_alloc_fn       device_alloc;
    katali_cuda_device_free_fn        device_free;
    katali_cuda_upload_fn             upload;
    katali_cuda_download_fn           download;
    katali_cuda_device_telemetry_fn   device_telemetry;
    katali_cuda_moe_layer_fn          moe_layer;
    katali_cuda_moe_layer_fn          moe_layer_nomath;
    katali_cuda_bench_stream_fn       bench_stream;
    katali_cuda_set_gpu_timing_fn     set_gpu_timing;   /* optional */
    katali_cuda_moe_submit_fn         moe_submit;
    katali_cuda_moe_wait_fn           moe_wait;
    katali_cuda_keepwarm_fn           keepwarm;
    katali_cuda_keepwarm_launches_fn  keepwarm_launches;
    katali_cuda_set_dp4a_fn           set_dp4a;          /* ABI 6 (optional) */
    katali_cuda_set_dp4a_types_fn     set_dp4a_types;    /* ABI 6 (optional) */
    katali_cuda_dp4a_active_fn        dp4a_active;       /* ABI 6 (optional) */
} CudaCtx;

static CudaCtx g_cuda;

/* --- Phase 1 host counters ------------------------------------------------ */
static KataliCudaTelemetry g_tel;

static int cuda_env_enabled(void) {
    const char *e = getenv("KATALI_CUDA");
    if (!e || !*e) return 1; /* default: auto-detect */
    return !(*e == '0' || *e == 'n' || *e == 'N' || *e == 'f' || *e == 'F');
}

/* Directory holding this executable, so the DLL can sit next to the exe. */
static void klib_self_dir(char *out, size_t cap) {
    out[0] = '\0';
#ifdef _WIN32
    char p[1024];
    DWORD n = GetModuleFileNameA(NULL, p, (DWORD)sizeof(p));
    if (n == 0 || n >= sizeof(p)) return;
    for (int i = (int)n - 1; i >= 0; i--) {
        if (p[i] == '\\' || p[i] == '/') { p[i] = '\0'; break; }
    }
    snprintf(out, cap, "%s", p);
#else
    char p[1024];
    ssize_t n = readlink("/proc/self/exe", p, sizeof(p) - 1);
    if (n <= 0) return;
    p[n] = '\0';
    for (int i = (int)n - 1; i >= 0; i--) {
        if (p[i] == '/') { p[i] = '\0'; break; }
    }
    snprintf(out, cap, "%s", p);
#endif
}

static const char *backend_dll_name(void) {
#ifdef _WIN32
    return "katali_cuda.dll";
#else
    return "libkatali_cuda.so";
#endif
}

static void cuda_set_status(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_cuda.status, sizeof(g_cuda.status), fmt, ap);
    va_end(ap);
}

/* Try to load the backend and resolve every ABI entry point.
 * Returns 1 on success; on failure leaves g_cuda.status explaining why. */
static int cuda_load_backend(void) {
    const char *env = getenv("KATALI_CUDA_DLL");
    char selfdir[1024];
    klib_self_dir(selfdir, sizeof(selfdir));

    char beside_exe[1200];
    beside_exe[0] = '\0';
    if (selfdir[0])
        snprintf(beside_exe, sizeof(beside_exe), "%s%c%s",
                 selfdir, KATALI_PATH_SEP, backend_dll_name());

    const char *candidates[3];
    int n = 0;
    if (env && *env) candidates[n++] = env;
    if (beside_exe[0]) candidates[n++] = beside_exe;
    candidates[n++] = backend_dll_name();

    CudaCtx *c = &g_cuda;
    c->lib = NULL;
    c->lib_path[0] = '\0';
    int last_err = 0, last_existed = 0;
    for (int i = 0; i < n; i++) {
        last_existed = klib_file_exists(candidates[i]);
        klib_t h = klib_open(candidates[i]);
        snprintf(c->lib_path, sizeof(c->lib_path), "%s", candidates[i]);
        if (h) { c->lib = h; break; }
        last_err = klib_err();
    }
    if (!c->lib) {
        if (!last_existed) {
            cuda_set_status("CUDA backend DLL not present (%s). Put %s next to "
                            "katali-lab.exe to enable the GPU tier. Using CPU + "
                            "system RAM + SSD.", c->lib_path, backend_dll_name());
        } else if (last_err == KLIB_MOD_NOT_FOUND) {
            cuda_set_status("CUDA backend load failed: %s exists but its "
                            "dependencies are missing (cudart64_*.dll). Install "
                            "the CUDA Toolkit or put the CUDA runtime DLLs next "
                            "to katali-lab.exe. Using CPU + system RAM + SSD.",
                            c->lib_path);
        } else if (last_err == KLIB_BAD_EXE_FORMAT) {
            cuda_set_status("CUDA backend %s has the wrong architecture for this "
                            "process (error %d). Using CPU + system RAM + SSD.",
                            c->lib_path, last_err);
        } else {
            cuda_set_status("CUDA backend load failed (error %d) for %s. Using "
                            "CPU + system RAM + SSD.", last_err, c->lib_path);
        }
        return 0;
    }

    KATALI_SYM(c->lib, "katali_cuda_abi_version",      c->abi_version);
    KATALI_SYM(c->lib, "katali_cuda_backend_init",     c->backend_init);
    KATALI_SYM(c->lib, "katali_cuda_backend_shutdown", c->backend_shutdown);
    KATALI_SYM(c->lib, "katali_cuda_supports_type",    c->supports_type);
    KATALI_SYM(c->lib, "katali_cuda_matvec",           c->matvec);
    KATALI_SYM(c->lib, "katali_cuda_matmul",           c->matmul);
    KATALI_SYM(c->lib, "katali_cuda_device_alloc",     c->device_alloc);
    KATALI_SYM(c->lib, "katali_cuda_device_free",      c->device_free);
    KATALI_SYM(c->lib, "katali_cuda_upload",           c->upload);
    KATALI_SYM(c->lib, "katali_cuda_download",         c->download);
    KATALI_SYM(c->lib, "katali_cuda_device_telemetry", c->device_telemetry);
    KATALI_SYM(c->lib, "katali_cuda_moe_layer",       c->moe_layer);
    KATALI_SYM(c->lib, "katali_cuda_moe_layer_nomath", c->moe_layer_nomath);
    KATALI_SYM(c->lib, "katali_cuda_bench_stream",     c->bench_stream);
    KATALI_SYM(c->lib, "katali_cuda_moe_submit",       c->moe_submit);
    KATALI_SYM(c->lib, "katali_cuda_moe_wait",         c->moe_wait);
    KATALI_SYM(c->lib, "katali_cuda_keepwarm",         c->keepwarm);
    KATALI_SYM(c->lib, "katali_cuda_keepwarm_launches", c->keepwarm_launches);
    /* optional; a missing symbol is not fatal */
    KATALI_SYM(c->lib, "katali_cuda_set_gpu_timing",  c->set_gpu_timing);

#define KATALI_NEED(fn, nm)                                                   \
    do {                                                                      \
        if (!c->fn) {                                                         \
            cuda_set_status("CUDA backend DLL is missing symbol %s (%s)",     \
                            (nm), c->lib_path);                               \
            klib_close(c->lib); c->lib = NULL;                                \
            return 0;                                                         \
        }                                                                     \
    } while (0)
    KATALI_NEED(abi_version, "katali_cuda_abi_version");
    KATALI_NEED(backend_init, "katali_cuda_backend_init");
    KATALI_NEED(backend_shutdown, "katali_cuda_backend_shutdown");
    KATALI_NEED(supports_type, "katali_cuda_supports_type");
    KATALI_NEED(matvec, "katali_cuda_matvec");
    KATALI_NEED(matmul, "katali_cuda_matmul");
    KATALI_NEED(device_alloc, "katali_cuda_device_alloc");
    KATALI_NEED(device_free, "katali_cuda_device_free");
    KATALI_NEED(upload, "katali_cuda_upload");
    KATALI_NEED(download, "katali_cuda_download");
    KATALI_NEED(device_telemetry, "katali_cuda_device_telemetry");
    KATALI_NEED(moe_layer, "katali_cuda_moe_layer");
    KATALI_NEED(moe_layer_nomath, "katali_cuda_moe_layer_nomath");
    KATALI_NEED(moe_submit, "katali_cuda_moe_submit");
    KATALI_NEED(moe_wait, "katali_cuda_moe_wait");
    KATALI_NEED(keepwarm, "katali_cuda_keepwarm");
    KATALI_NEED(keepwarm_launches, "katali_cuda_keepwarm_launches");
    KATALI_NEED(bench_stream, "katali_cuda_bench_stream");
#undef KATALI_NEED

    /* ABI 6, resolved without requiring it so an older DLL still loads (the ABI
     * version check below already rejects anything truly incompatible). */
    KATALI_SYM(c->lib, "katali_cuda_set_dp4a", c->set_dp4a);
    KATALI_SYM(c->lib, "katali_cuda_set_dp4a_types", c->set_dp4a_types);
    KATALI_SYM(c->lib, "katali_cuda_dp4a_active", c->dp4a_active);

    int dll_abi = c->abi_version();
    if (dll_abi != KATALI_CUDA_ABI_VERSION) {
        cuda_set_status("CUDA backend ABI mismatch: DLL reports %d, host expects "
                        "%d (%s). Rebuild the backend. Using CPU path.",
                        dll_abi, KATALI_CUDA_ABI_VERSION, c->lib_path);
        klib_close(c->lib); c->lib = NULL;
        return 0;
    }

    memset(&c->dev, 0, sizeof(c->dev));
    c->dev.abi_version = dll_abi;
    if (c->backend_init(&c->dev) != 0) {
        cuda_set_status("CUDA device init failed: %s",
                        c->dev.err[0] ? c->dev.err : "(no detail)");
        klib_close(c->lib); c->lib = NULL;
        return 0;
    }
    cuda_set_status("usable: %s (sm_%d%d, %d SMs, %.0f MiB free of %.0f MiB)",
                    c->dev.name[0] ? c->dev.name : "NVIDIA GPU",
                    c->dev.cc_major, c->dev.cc_minor, c->dev.sm_count,
                    (double)c->dev.vram_free_bytes / (1024.0 * 1024.0),
                    (double)c->dev.vram_total_bytes / (1024.0 * 1024.0));
    return 1;
}

int katali_cuda_probe(void) {
    if (g_cuda.disabled) return 0;
    if (g_cuda.probed) return g_cuda.ok;
    g_cuda.probed = 1;
    g_cuda.ok = 0;
    if (!cuda_env_enabled()) {
        cuda_set_status("CUDA disabled by KATALI_CUDA=0; using CPU + system RAM + SSD.");
        return 0;
    }
    g_cuda.ok = cuda_load_backend();
    return g_cuda.ok;
}

int katali_cuda_available(void) { return katali_cuda_probe(); }

const char *katali_cuda_status(void) {
    (void)katali_cuda_probe();
    return g_cuda.status[0] ? g_cuda.status : "CUDA not probed";
}

const char *katali_cuda_lib_path(void) { return g_cuda.lib_path; }

const KataliCudaDeviceInfo *katali_cuda_device(void) {
    if (!katali_cuda_probe()) return NULL;
    return &g_cuda.dev;
}

void katali_cuda_shutdown(void) {
    if (g_cuda.lib) {
        if (g_cuda.backend_shutdown) g_cuda.backend_shutdown();
        klib_close(g_cuda.lib);
        g_cuda.lib = NULL;
    }
    memset(&g_cuda, 0, sizeof(g_cuda));
    g_cuda.disabled = 1;
    g_cuda.probed = 1;
    cuda_set_status("CUDA backend shut down; using CPU + system RAM + SSD.");
}

/* --- dispatch wrappers --------------------------------------------------- */

int katali_cuda_malloc(void **out_dev, size_t nbytes) {
    if (!out_dev) return KATALI_ERR;
    *out_dev = NULL;
    if (!katali_cuda_probe() || !g_cuda.device_alloc) return KATALI_ERR;
    double t0 = katali_time_s();
    int rc = g_cuda.device_alloc(out_dev, nbytes);
    g_tel.api_calls++; g_tel.mallocs++;
    g_tel.host_seconds += katali_time_s() - t0;
    return (rc == 0) ? KATALI_OK : KATALI_ERR;
}

int katali_cuda_free(void *dev) {
    if (!dev) return KATALI_OK;
    if (!g_cuda.device_free) return KATALI_ERR;
    double t0 = katali_time_s();
    int rc = g_cuda.device_free(dev);
    g_tel.api_calls++; g_tel.frees++;
    g_tel.host_seconds += katali_time_s() - t0;
    return (rc == 0) ? KATALI_OK : KATALI_ERR;
}

int katali_cuda_upload(void *dev_dst, const void *host_src, size_t nbytes) {
    if (!katali_cuda_probe() || !g_cuda.upload) return KATALI_ERR;
    double t0 = katali_time_s();
    int rc = g_cuda.upload(dev_dst, host_src, nbytes);
    double dt = katali_time_s() - t0;
    g_tel.api_calls++; g_tel.uploads++; g_tel.h2d_bytes += nbytes;
    g_tel.host_seconds += dt; g_tel.upload_s += dt;
    return (rc == 0) ? KATALI_OK : KATALI_ERR;
}

int katali_cuda_download(void *host_dst, const void *dev_src, size_t nbytes) {
    if (!katali_cuda_probe() || !g_cuda.download) return KATALI_ERR;
    double t0 = katali_time_s();
    int rc = g_cuda.download(host_dst, dev_src, nbytes);
    double dt = katali_time_s() - t0;
    g_tel.api_calls++; g_tel.downloads++; g_tel.d2h_bytes += nbytes;
    g_tel.host_seconds += dt; g_tel.download_s += dt;
    return (rc == 0) ? KATALI_OK : KATALI_ERR;
}

int katali_cuda_supports_type(uint32_t ggml_type) {
    if (!katali_cuda_probe() || !g_cuda.supports_type) return 0;
    return g_cuda.supports_type(ggml_type);
}

int katali_cuda_matvec(uint32_t ggml_type, const void *w_dev,
                       uint64_t rows, uint64_t cols,
                       const float *x_dev, float *y_dev) {
    if (!katali_cuda_probe() || !g_cuda.matvec) return KATALI_ERR;
    double t0 = katali_time_s();
    int rc = g_cuda.matvec(ggml_type, w_dev, rows, cols, x_dev, y_dev);
    double dt = katali_time_s() - t0;
    g_tel.api_calls++; g_tel.matvecs++;
    g_tel.host_seconds += dt; g_tel.matvec_s += dt;
    return (rc == 0) ? KATALI_OK : KATALI_ERR;
}

int katali_cuda_matmul(uint32_t ggml_type, const void *w_dev,
                       uint64_t rows, uint64_t cols,
                       const float *X_dev, uint64_t B, float *Y_dev) {
    if (!katali_cuda_probe() || !g_cuda.matmul) return KATALI_ERR;
    double t0 = katali_time_s();
    int rc = g_cuda.matmul(ggml_type, w_dev, rows, cols, X_dev, B, Y_dev);
    double dt = katali_time_s() - t0;
    g_tel.api_calls++; g_tel.matmuls++;
    g_tel.host_seconds += dt; g_tel.matmul_s += dt;
    return (rc == 0) ? KATALI_OK : KATALI_ERR;
}

/* Fused layer-level MoE. Counted like a matmul (one host->CUDA crossing that
 * internally performs a handful of launches) so the telemetry shows the drop in
 * call count rather than hiding it. */
int katali_cuda_moe_layer(const KataliCudaMoeDesc *desc,
                          const void *const *gate_dev,
                          const void *const *up_dev,
                          const void *const *down_dev,
                          const float *sel_weights,
                          int n_sel, const float *x_dev, float *y_dev) {
    if (!katali_cuda_probe() || !g_cuda.moe_layer) return KATALI_ERR;
    double t0 = katali_time_s();
    int rc = g_cuda.moe_layer(desc, gate_dev, up_dev, down_dev,
                              sel_weights, n_sel, x_dev, y_dev);
    double dt = katali_time_s() - t0;
    g_tel.api_calls++;
    g_tel.host_seconds += dt; g_tel.matmul_s += dt;
    g_tel.layer_batches++;
    return (rc == 0) ? KATALI_OK : KATALI_ERR;
}

/* Phase 4 async MoE. `submit` only queues; `wait` is where the CPU blocks, so
 * tracking them separately makes the overlap visible in the telemetry. */
int katali_cuda_moe_submit(const KataliCudaMoeDesc *desc,
                           const void *const *gate_dev,
                           const void *const *up_dev,
                           const void *const *down_dev,
                           const float *sel_weights,
                           int n_sel, const float *x_dev, float *y_dev) {
    if (!katali_cuda_probe() || !g_cuda.moe_submit) return KATALI_ERR;
    double t0 = katali_time_s();
    int rc = g_cuda.moe_submit(desc, gate_dev, up_dev, down_dev,
                               sel_weights, n_sel, x_dev, y_dev);
    double dt = katali_time_s() - t0;
    g_tel.api_calls++;
    g_tel.host_seconds += dt; g_tel.matmul_s += dt;
    if (rc == 0) g_tel.layer_batches++;
    return (rc == 0) ? KATALI_OK : KATALI_ERR;
}

int katali_cuda_moe_wait(void) {
    if (!katali_cuda_probe() || !g_cuda.moe_wait) return KATALI_ERR;
    double t0 = katali_time_s();
    int rc = g_cuda.moe_wait();
    g_tel.api_calls++;
    g_tel.host_seconds += katali_time_s() - t0;
    return (rc == 0) ? KATALI_OK : KATALI_ERR;
}

int katali_cuda_keepwarm(int interval_ms) {
    if (!katali_cuda_probe() || !g_cuda.keepwarm) return KATALI_ERR;
    return (g_cuda.keepwarm(interval_ms) == 0) ? KATALI_OK : KATALI_ERR;
}

uint64_t katali_cuda_keepwarm_launches(void) {
    if (!g_cuda.keepwarm_launches) return 0;
    return g_cuda.keepwarm_launches();
}

int katali_cuda_moe_layer_nomath(const KataliCudaMoeDesc *desc,
                                 const void *const *gate_dev,
                                 const void *const *up_dev,
                                 const void *const *down_dev,
                                 const float *sel_weights,
                                 int n_sel, const float *x_dev, float *y_dev) {
    if (!katali_cuda_probe() || !g_cuda.moe_layer_nomath) return KATALI_ERR;
    double t0 = katali_time_s();
    int rc = g_cuda.moe_layer_nomath(desc, gate_dev, up_dev, down_dev,
                                     sel_weights, n_sel, x_dev, y_dev);
    g_tel.api_calls++;
    g_tel.host_seconds += katali_time_s() - t0;
    g_tel.layer_batches++;
    return (rc == 0) ? KATALI_OK : KATALI_ERR;
}

int katali_cuda_bench_stream(void *dev_a, void *dev_b, size_t bytes, int reps,
                            double *read_gbs, double *copy_gbs) {
    if (!katali_cuda_probe() || !g_cuda.bench_stream) return KATALI_ERR;
    g_tel.api_calls++;
    return (g_cuda.bench_stream(dev_a, dev_b, bytes, reps, read_gbs, copy_gbs) == 0)
               ? KATALI_OK : KATALI_ERR;
}

/* --- Phase 1 telemetry ---------------------------------------------------- */

void katali_cuda_telemetry_reset(void) {
    memset(&g_tel, 0, sizeof(g_tel));
    if (katali_cuda_probe() && g_cuda.device_telemetry) {
        KataliCudaDeviceCounters dc;
        g_cuda.device_telemetry(&dc, 1);
    }
}

void katali_cuda_telemetry_get(KataliCudaTelemetry *out) {
    if (!out) return;
    *out = g_tel;
    if (g_cuda.device_telemetry) {
        KataliCudaDeviceCounters dc;
        if (g_cuda.device_telemetry(&dc, 0) == 0) {
            out->kernel_launches = dc.kernel_launches;
            out->sync_calls = dc.sync_calls;
            out->memcpy_calls = dc.memcpy_calls;
            out->gpu_seconds = dc.gpu_seconds;
            out->q8_1_quant_launches = dc.q8_1_quant_launches;
            out->q8_1_quant_seconds = dc.q8_1_quant_seconds;
        }
    }
}

void katali_cuda_count_expert(int handled_on_gpu) {
    if (handled_on_gpu) g_tel.experts_gpu++;
    else                g_tel.experts_cpu++;
}

void katali_cuda_count_layer_batch(void) { g_tel.layer_batches++; }

int katali_cuda_telemetry_enabled(void) {
    const char *e = getenv("KATALI_CUDA_TELEMETRY");
    return (e && e[0] && e[0] != '0') ? 1 : 0;
}

void katali_cuda_set_gpu_timing(int on) {
    if (katali_cuda_probe() && g_cuda.set_gpu_timing)
        g_cuda.set_gpu_timing(on);
}

/* --- ABI 6: DP4A path selection ------------------------------------------ */

void katali_cuda_set_dp4a(int on) {
    if (katali_cuda_probe() && g_cuda.set_dp4a)
        g_cuda.set_dp4a(on);
}

void katali_cuda_set_dp4a_types(int q4_k, int q6_k) {
    if (katali_cuda_probe() && g_cuda.set_dp4a_types)
        g_cuda.set_dp4a_types(q4_k, q6_k);
}

int katali_cuda_dp4a_active(void) {
    if (katali_cuda_probe() && g_cuda.dp4a_active)
        return g_cuda.dp4a_active();
    return 0;
}

void katali_cuda_telemetry_print(const char *tag) {
    KataliCudaTelemetry t;
    katali_cuda_telemetry_get(&t);
    if (!tag) tag = "cuda-telemetry";
    fprintf(stderr,
            "%s: calls=%llu (matvec=%llu matmul=%llu up=%llu dn=%llu alloc=%llu free=%llu) "
            "launch=%llu sync=%llu memcpy=%llu | h2d=%.2f MiB d2h=%.2f MiB | "
            "host=%.2f ms (mv=%.2f up=%.2f dn=%.2f) gpu=%.2f ms | "
            "experts_gpu=%llu experts_cpu=%llu layer_batches=%llu\n",
            tag,
            (unsigned long long)t.api_calls,
            (unsigned long long)t.matvecs, (unsigned long long)t.matmuls,
            (unsigned long long)t.uploads, (unsigned long long)t.downloads,
            (unsigned long long)t.mallocs, (unsigned long long)t.frees,
            (unsigned long long)t.kernel_launches, (unsigned long long)t.sync_calls,
            (unsigned long long)t.memcpy_calls,
            (double)t.h2d_bytes / (1024.0 * 1024.0),
            (double)t.d2h_bytes / (1024.0 * 1024.0),
            t.host_seconds * 1000.0, t.matvec_s * 1000.0,
            t.upload_s * 1000.0, t.download_s * 1000.0,
            t.gpu_seconds * 1000.0,
            (unsigned long long)t.experts_gpu, (unsigned long long)t.experts_cpu,
            (unsigned long long)t.layer_batches);
}