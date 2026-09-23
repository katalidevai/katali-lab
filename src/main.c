#include "host.h"
#include "moe_ffn.h"
#include "forward.h"
#include "katali_gguf_dtype.h"
#include "katali_gguf_simd.h"
#include "katali_gguf_threads.h"
#include <math.h>
#include "q4.h"
#include "platform.h"
#include "katali_cuda.h"
#include "api_server.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* 35B single-file default. Override with path/dir for 122B split GGUF. */
static const char *DEFAULT_GGUF = "C:\\models\\Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf";
static const char *DEFAULT_GGUF_122B =
    "C:\\models\\Qwen_Qwen3.5-122B-A10B-Q4_K_M\\Qwen_Qwen3.5-122B-A10B-Q4_K_M";

static void usage(const char *argv0) {
    printf("katali-lab — elastic GGUF MoE (CPU + RAM + SSD, optional CUDA)\n");
    printf("Usage:\n");
    printf("  %s info\n", argv0);
    printf("  %s cuda-info\n", argv0);
    printf("  %s cuda-check [model.gguf]\n", argv0);
    printf("  %s cuda-bench [model.gguf]\n", argv0);
    printf("  %s cuda-bench-cold [model.gguf] [layer]\n", argv0);
    printf("  %s cuda-dp4a-selftest          (synthetic exactness test)\n", argv0);
    printf("  %s cuda-check-moe [model.gguf] [layer] [n_sel]\n", argv0);
    printf("  %s selftest\n", argv0);
    printf("  %s inspect [model.gguf]\n", argv0);
    printf("  %s tensors [model.gguf] [substr]\n", argv0);
    printf("  %s open [model.gguf] [--pin N] [--cache-gb N]\n", argv0);
    printf("  %s probe-expert [model.gguf] <layer> <expert>\n", argv0);
    printf("  %s moe-ffn [model.gguf] <layer>\n", argv0);
    printf("  %s generate [model.gguf|dir|shard1] \"prompt\" [--max N] [--pin N] [--cache-gb N]\n", argv0);
    printf("  %s dump-emb [model.gguf] <token_id>\n", argv0);
    printf("  %s api [--port N]\n", argv0);
    printf("\nDefault GGUF: %s\n", DEFAULT_GGUF);
    printf("122B split dir: %s\n", DEFAULT_GGUF_122B);
    printf("Format: GGUF only (Q4_K_M mix, multi-shard OK). No EQS.\n");
    printf("Tiers: VRAM + CPU + system RAM + SSD; CUDA is optional and auto-detected.\n");
}

static const char *gguf_arg(int argc, char **argv, int idx) {
    if (idx < argc && argv[idx] && argv[idx][0] != '-') return argv[idx];
    return DEFAULT_GGUF;
}

static void print_gguf_info(const KataliGgufFile *f) {
    printf("file: %s\n", f->path);
    printf("version: %u  tensors: %llu  kv: %llu  size: %.2f GiB  shards: %d\n",
           f->version,
           (unsigned long long)f->tensor_count,
           (unsigned long long)f->kv_count,
           (double)f->size / (1024.0 * 1024.0 * 1024.0),
           f->n_shards > 0 ? f->n_shards : 1);
    const char *arch = katali_gguf_get_str(f, "general.architecture", NULL, NULL);
    const char *name = katali_gguf_get_str(f, "general.name", NULL, NULL);
    if (arch) printf("architecture: %s\n", arch);
    if (name) printf("name: %s\n", name);
}

static void print_tensors(const KataliGgufFile *f, const char *substr) {
    for (uint64_t i = 0; i < f->tensor_count; i++) {
        const KataliGgufTensor *t = &f->tensors[i];
        if (substr && substr[0] && (!t->name || !strstr(t->name, substr))) continue;
        printf("%s  %s  dims=[", t->name ? t->name : "?", katali_ggml_type_name(t->type));
        for (uint32_t d = 0; d < t->n_dims; d++) {
            if (d) printf(",");
            printf("%llu", (unsigned long long)t->dims[d]);
        }
        printf("]  nbytes=%llu\n", (unsigned long long)t->nbytes);
    }
}

/* -------------------------------------------------------------------------
 * cuda-check — numerical validation of the CUDA backend.
 *
 * Compares the GPU kernels against the engine's OWN CPU kernels
 * (katali_ggml_vec_dot, the documented correctness oracle) using real
 * quantized tensors read straight from the GGUF, so the actual Q4_K/Q6_K/Q8_0
 * block layouts in the model are what gets tested — not synthetic data.
 *
 * Metric: L2 relative error ||y_gpu - y_cpu|| / ||y_cpu||, plus max abs error.
 * PASS is <= 1e-3 relative, which is tight for f32 accumulation with a
 * different summation order.
 * ------------------------------------------------------------------------- */
static float cuda_lcg(uint32_t *s) {
    *s = (*s * 1664525u) + 1013904223u;
    return (float)((*s >> 8) & 0xFFFFu) / 32768.0f - 1.0f;
}

/* Returns 0 on PASS, 1 on FAIL, -1 when the tensor shape/type is unusable. */
static int cuda_check_tensor(const KataliGgufTensor *t, int max_rows, double *rel_out) {
    if (!t || !t->data || t->n_dims < 2) return -1;
    uint64_t cols = t->dims[0];
    uint64_t rows_full = t->dims[1];
    if (cols == 0 || rows_full == 0 || cols > 1048576u) return -1;
    if (!katali_cuda_supports_type(t->type)) return -1;
    if (t->type == KGGML_Q4_K || t->type == KGGML_Q6_K) {
        if (cols % 256u) return -1;
    } else if (t->type == KGGML_Q4_0 || t->type == KGGML_Q8_0) {
        if (cols % 32u) return -1;
    }
    uint64_t rows = rows_full < (uint64_t)max_rows ? rows_full : (uint64_t)max_rows;
    uint64_t rb = katali_ggml_row_bytes(t->type, cols);
    if (rb == 0) return -1;
    size_t wbytes = (size_t)(rb * rows);

    float *x  = (float *)malloc((size_t)cols * sizeof(float));
    float *yc = (float *)malloc((size_t)rows * sizeof(float));
    float *yg = (float *)malloc((size_t)rows * sizeof(float));
    if (!x || !yc || !yg) { free(x); free(yc); free(yg); return -1; }

    uint32_t seed = 12345u;
    for (uint64_t i = 0; i < cols; i++) x[i] = cuda_lcg(&seed);

    /* CPU oracle — the same function the engine's forward path calls. */
    for (uint64_t r = 0; r < rows; r++)
        yc[r] = katali_ggml_vec_dot(t->type, t->data + (size_t)(r * rb), x, cols);

    void *dw = NULL, *dx = NULL, *dy = NULL;
    int rc = -1;
    if (katali_cuda_malloc(&dw, wbytes) != KATALI_OK) goto done;
    if (katali_cuda_malloc(&dx, (size_t)cols * sizeof(float)) != KATALI_OK) goto done;
    if (katali_cuda_malloc(&dy, (size_t)rows * sizeof(float)) != KATALI_OK) goto done;
    if (katali_cuda_upload(dw, t->data, wbytes) != KATALI_OK) goto done;
    if (katali_cuda_upload(dx, x, (size_t)cols * sizeof(float)) != KATALI_OK) goto done;
    if (katali_cuda_matvec(t->type, dw, rows, cols,
                           (const float *)dx, (float *)dy) != KATALI_OK) goto done;
    if (katali_cuda_download(yg, dy, (size_t)rows * sizeof(float)) != KATALI_OK) goto done;

    {
        double num = 0.0, den = 0.0, maxabs = 0.0;
        for (uint64_t r = 0; r < rows; r++) {
            double d = (double)yg[r] - (double)yc[r];
            num += d * d;
            den += (double)yc[r] * (double)yc[r];
            if (fabs(d) > maxabs) maxabs = fabs(d);
        }
        double rel = (den > 0.0) ? sqrt(num / den) : sqrt(num);
        if (rel_out) *rel_out = rel;
        printf("  %-40s %-6s cols=%-6llu rows=%-4llu rel_L2=%.3e max_abs=%.3e  %s\n",
               t->name ? t->name : "?", katali_ggml_type_name(t->type),
               (unsigned long long)cols, (unsigned long long)rows,
               rel, maxabs, rel <= 1e-3 ? "PASS" : "FAIL");
        rc = rel <= 1e-3 ? 0 : 1;
    }
done:
    if (dw) katali_cuda_free(dw);
    if (dx) katali_cuda_free(dx);
    if (dy) katali_cuda_free(dy);
    free(x); free(yc); free(yg);
    return rc;
}

static int cmd_cuda_check(const char *path) {
    printf("cuda-check: GPU kernels vs engine CPU kernels (katali_ggml_vec_dot)\n");
    if (!katali_cuda_available()) {
        printf("CUDA unavailable: %s\n", katali_cuda_status());
        printf("nothing to validate; CPU path unaffected\n");
        return 0;
    }
    printf("device: %s\n", katali_cuda_status());
    KataliGgufFile f;
    char err[256];
    if (katali_gguf_open(&f, path, 1, err, sizeof(err)) != 0) {
        fprintf(stderr, "open failed: %s (%s)\n", path, err);
        return 2;
    }
    int seen[256];
    memset(seen, 0, sizeof(seen));
    uint64_t counts[256];
    memset(counts, 0, sizeof(counts));
    for (uint64_t i = 0; i < f.tensor_count; i++) {
        uint32_t ty = f.tensors[i].type;
        if (ty < 256u && f.tensors[i].n_dims >= 2) counts[ty]++;
    }
    printf("tensors by type present in this GGUF (supported only):\n");
    for (int ty = 0; ty < 256; ty++) {
        if (counts[ty] && katali_cuda_supports_type((uint32_t)ty))
            printf("  %-6s %llu tensors\n", katali_ggml_type_name((uint32_t)ty),
                   (unsigned long long)counts[ty]);
    }
    printf("per-type validation (8 rows each):\n");
    int tested = 0, failed = 0;
    double worst = 0.0;
    for (uint64_t i = 0; i < f.tensor_count; i++) {
        uint32_t ty = f.tensors[i].type;
        if (ty >= 256u || seen[ty]) continue;
        if (!katali_cuda_supports_type(ty)) continue;
        double rel = 0.0;
        int rc = cuda_check_tensor(&f.tensors[i], 8, &rel);
        if (rc < 0) continue;              /* shape/type not usable here */
        seen[ty] = 1;
        tested++;
        if (rel > worst) worst = rel;
        if (rc != 0) failed++;
    }
    printf("types validated=%d  failed=%d  worst rel_L2=%.3e\n", tested, failed, worst);
    katali_gguf_close(&f);
    if (tested == 0) { printf("no usable tensor found\n"); return 2; }
    printf(failed == 0 ? "cuda-check: ALL PASS\n" : "cuda-check: FAILURES PRESENT\n");
    return failed == 0 ? 0 : 1;
}

/* -------------------------------------------------------------------------
 * cuda-bench — Step 4 measurement: CPU vs GPU for the real decode operations.
 *
 * Weights are uploaded to VRAM ONCE and stay device-resident; the timed GPU
 * region is the kernel plus its synchronize. That is the deployment shape
 * (VRAM hot tier), and it is deliberately NOT a per-call host->device copy,
 * because transferring expert weights over PCIe every token is exactly the
 * access pattern the design must avoid.
 * ------------------------------------------------------------------------- */
static const KataliGgufTensor *find_tensor(const KataliGgufFile *f, const char *name) {
    for (uint64_t i = 0; i < f->tensor_count; i++)
        if (f->tensors[i].name && strcmp(f->tensors[i].name, name) == 0)
            return &f->tensors[i];
    return NULL;
}

static double bench_cpu(const KataliGgufTensor *t, uint64_t rows, uint64_t cols,
                        const float *x, float *y, int threads, float *scratch,
                        uint64_t scratch_cap, int reps) {
    double best = 1e30;
    for (int i = 0; i < reps; i++) {
        double t0 = katali_time_s();
        katali_ggml_matvec(t->type, t->data, rows, cols, x, y, threads, NULL,
                           scratch, scratch_cap);
        double dt = katali_time_s() - t0;
        if (dt < best) best = dt;
    }
    return best;
}

static int bench_one(const char *label, const KataliGgufTensor *t,
                     uint64_t rows, uint64_t cols, int cpu_threads, int reps) {
    if (!t || !t->data) { printf("  %-34s (tensor not found)\n", label); return 0; }
    if (rows > t->dims[1]) return 0;
    uint64_t rb = katali_ggml_row_bytes(t->type, cols);
    if (rb == 0) return 0;
    double wmb = (double)(rb * rows) / (1024.0 * 1024.0);

    float *x  = (float *)malloc((size_t)cols * sizeof(float));
    float *yc = (float *)malloc((size_t)rows * sizeof(float));
    float *yg = (float *)malloc((size_t)rows * sizeof(float));
    uint64_t sc_cap = KATALI_Q4K_SUMS_FLOATS(cols);
    if (sc_cap < 64) sc_cap = 64;
    float *scratch = (float *)malloc((size_t)sc_cap * sizeof(float));
    if (!x || !yc || !yg || !scratch) { free(x); free(yc); free(yg); free(scratch); return 0; }
    uint32_t seed = 999u;
    for (uint64_t i = 0; i < cols; i++) x[i] = cuda_lcg(&seed);

    double cpu_s = bench_cpu(t, rows, cols, x, yc, cpu_threads, scratch, sc_cap, reps);

    /* GPU: upload once, then time the resident-weight kernel. */
    void *dw = NULL, *dx = NULL, *dy = NULL;
    double gpu_s = -1.0;
    if (katali_cuda_malloc(&dw, (size_t)(rb * rows)) == KATALI_OK &&
        katali_cuda_malloc(&dx, (size_t)cols * sizeof(float)) == KATALI_OK &&
        katali_cuda_malloc(&dy, (size_t)rows * sizeof(float)) == KATALI_OK &&
        katali_cuda_upload(dw, t->data, (size_t)(rb * rows)) == KATALI_OK &&
        katali_cuda_upload(dx, x, (size_t)cols * sizeof(float)) == KATALI_OK) {
        double best = 1e30;
        int ok = 1;
        for (int i = 0; i < reps; i++) {
            double t0 = katali_time_s();
            if (katali_cuda_matvec(t->type, dw, rows, cols,
                                   (const float *)dx, (float *)dy) != KATALI_OK) { ok = 0; break; }
            double dt = katali_time_s() - t0;
            if (dt < best) best = dt;
        }
        if (ok && katali_cuda_download(yg, dy, (size_t)rows * sizeof(float)) == KATALI_OK)
            gpu_s = best;
    }
    if (dw) katali_cuda_free(dw);
    if (dx) katali_cuda_free(dx);
    if (dy) katali_cuda_free(dy);

    double cpu_ms = cpu_s * 1000.0, gpu_ms = gpu_s * 1000.0;
    double cpu_gbs = cpu_s > 0 ? wmb / 1024.0 / cpu_s : 0.0;
    double gpu_gbs = gpu_s > 0 ? wmb / 1024.0 / gpu_s : 0.0;
    double num = 0, den = 0;
    if (gpu_s > 0) for (uint64_t r = 0; r < rows; r++) {
        double d = (double)yg[r] - (double)yc[r]; num += d * d; den += (double)yc[r] * (double)yc[r];
    }
    double rel = (gpu_s > 0 && den > 0) ? sqrt(num / den) : 0.0;

    printf("  %-32s %-5s %6.1f MB  cols=%-6llu rows=%-7llu\n",
           label, katali_ggml_type_name(t->type), wmb,
           (unsigned long long)cols, (unsigned long long)rows);
    printf("      CPU %8.2f ms (%6.1f GB/s)   GPU %8.2f ms (%6.1f GB/s)   speedup %5.2fx   rel=%.2e\n",
           cpu_ms, cpu_gbs, gpu_ms, gpu_gbs, gpu_ms > 0 ? cpu_ms / gpu_ms : 0.0, rel);
    free(x); free(yc); free(yg); free(scratch);
    return 1;
}

static int bench_multi(const char *label, const KataliGgufTensor *t,
                       uint64_t rows, uint64_t cols, int variants,
                       int cpu_threads, int reps);

static int cmd_cuda_bench(const char *path) {
    printf("cuda-bench: CPU vs GPU, weights resident in VRAM (Step 4)\n");
    if (!katali_cuda_available()) {
        printf("CUDA unavailable: %s\n", katali_cuda_status());
        return 0;
    }
    printf("device: %s\n", katali_cuda_status());
    int threads = katali_gguf_get_threads();
    printf("CPU threads: %d\n", threads);

    KataliGgufFile f;
    char err[256];
    if (katali_gguf_open(&f, path, 1, err, sizeof(err)) != 0) {
        fprintf(stderr, "open failed: %s (%s)\n", path, err);
        return 2;
    }
    const KataliGgufTensor *gate = find_tensor(&f, "blk.0.ffn_gate_exps.weight");
    const KataliGgufTensor *up   = find_tensor(&f, "blk.0.ffn_up_exps.weight");
    const KataliGgufTensor *down = find_tensor(&f, "blk.0.ffn_down_exps.weight");
    const KataliGgufTensor *lmh  = find_tensor(&f, "output.weight");
    if (!lmh) lmh = find_tensor(&f, "token_embd.weight");

    int done = 0;
    /* MoE expert tensors are 3-D: dims[0]=cols, dims[1]=rows per expert,
     * dims[2]=expert count. So one expert slice = dims[1] rows. */
    uint64_t g_rows = gate ? gate->dims[1] : 0, g_cols = gate ? gate->dims[0] : 0;
    uint64_t u_rows = up   ? up->dims[1]   : 0, u_cols = up   ? up->dims[0]   : 0;
    uint64_t d_rows = down ? down->dims[1] : 0, d_cols = down ? down->dims[0] : 0;
    printf("shapes (rows x cols, per expert): gate %llux%llu  up %llux%llu  down %llux%llu\n",
           (unsigned long long)g_rows, (unsigned long long)g_cols,
           (unsigned long long)u_rows, (unsigned long long)u_cols,
           (unsigned long long)d_rows, (unsigned long long)d_cols);

    printf("-- part 1: single op, data re-read (kernel speed on resident data) --\n");
    if (gate) done += bench_one("expert gate (1 of 256)", gate, g_rows, g_cols, threads, 30);
    if (up)   done += bench_one("expert up   (1 of 256)", up,   u_rows, u_cols, threads, 30);
    if (down) done += bench_one("expert down (1 of 256)", down, d_rows, d_cols, threads, 30);
    if (lmh)  done += bench_one("LM head (full vocab)", lmh, lmh->dims[1], lmh->dims[0], threads, 5);

    printf("-- part 2: decode-shaped, DISTINCT experts each pass (defeats L3) --\n");
    if (gate) done += bench_multi("64 experts: ffn_gate_exps", gate, g_rows, g_cols, 64, threads, 3);
    if (up)   done += bench_multi("64 experts: ffn_up_exps",   up,   u_rows, u_cols, 64, threads, 3);
    if (down) done += bench_multi("64 experts: ffn_down_exps", down, d_rows, d_cols, 64, threads, 3);
    if (gate) done += bench_multi("all 256: ffn_gate_exps",    gate, g_rows, g_cols, 256, threads, 1);

    katali_gguf_close(&f);
    if (!done) { printf("no usable tensor found\n"); return 2; }
    return 0;
}

/* Total row count of a possibly-3D MoE tensor: dims[1] * dims[2] * ... */
static uint64_t tensor_total_rows(const KataliGgufTensor *t) {
    uint64_t total = t->dims[1];
    for (uint32_t d = 2; d < t->n_dims; d++) total *= t->dims[d];
    return total;
}

/*
 * Multi-variant benchmark — the HONEST decode-shaped comparison.
 *
 * Re-reading one 0.6 MB expert 30 times keeps it in the i5's 12 MB L3, which
 * makes the CPU look far faster than it is during real decode, where experts
 * stream from RAM/SSD. Here we walk `variants` DISTINCT expert slices
 * (> L3, so the CPU is genuinely DRAM-bound), upload that whole set to VRAM
 * once, and report aggregate throughput for both. This is the number that
 * reflects deployment.
 */
static int bench_multi(const char *label, const KataliGgufTensor *t,
                       uint64_t rows, uint64_t cols, int variants,
                       int cpu_threads, int reps) {
    if (!t || !t->data || rows == 0 || cols == 0) return 0;
    uint64_t total = tensor_total_rows(t);
    if (rows > total) return 0;
    uint64_t rb = katali_ggml_row_bytes(t->type, cols);
    if (rb == 0) return 0;
    int nv = variants;
    if ((uint64_t)nv * rows > total) nv = (int)(total / rows);
    if (nv < 1) return 0;
    size_t slice = (size_t)(rb * rows);
    size_t all = slice * (size_t)nv;
    double set_mb = (double)all / (1024.0 * 1024.0);

    float *x  = (float *)malloc((size_t)cols * sizeof(float));
    float *y  = (float *)malloc((size_t)rows * sizeof(float));
    uint64_t sc_cap = KATALI_Q4K_SUMS_FLOATS(cols);
    if (sc_cap < 64) sc_cap = 64;
    float *scratch = (float *)malloc((size_t)sc_cap * sizeof(float));
    if (!x || !y || !scratch) { free(x); free(y); free(scratch); return 0; }
    uint32_t seed = 4242u;
    for (uint64_t i = 0; i < cols; i++) x[i] = cuda_lcg(&seed);

    double cpu_s = -1.0;
    if (reps > 0) {
        double t0 = katali_time_s();
        for (int r = 0; r < reps; r++)
            for (int v = 0; v < nv; v++)
                katali_ggml_matvec(t->type, t->data + (size_t)v * slice, rows, cols,
                                   x, y, cpu_threads, NULL, scratch, sc_cap);
        cpu_s = (katali_time_s() - t0) / (double)reps;
    }

    void *dw = NULL, *dx = NULL, *dy = NULL;
    double gpu_s = -1.0;
    if (katali_cuda_malloc(&dw, all) == KATALI_OK &&
        katali_cuda_malloc(&dx, (size_t)cols * sizeof(float)) == KATALI_OK &&
        katali_cuda_malloc(&dy, (size_t)rows * sizeof(float)) == KATALI_OK &&
        katali_cuda_upload(dw, t->data, all) == KATALI_OK &&
        katali_cuda_upload(dx, x, (size_t)cols * sizeof(float)) == KATALI_OK) {
        double t0 = katali_time_s();
        int ok = 1;
        for (int r = 0; r < reps; r++)
            for (int v = 0; v < nv; v++)
                if (katali_cuda_matvec(t->type, (const char *)dw + (size_t)v * slice,
                                       rows, cols, (const float *)dx,
                                       (float *)dy) != KATALI_OK) { ok = 0; break; }
        if (ok) gpu_s = (katali_time_s() - t0) / (double)reps;
    }
    if (dw) katali_cuda_free(dw);
    if (dx) katali_cuda_free(dx);
    if (dy) katali_cuda_free(dy);

    printf("  %-32s %-5s %6.1f MB working set, %d distinct slices of %llux%llu\n",
           label, katali_ggml_type_name(t->type), set_mb, nv,
           (unsigned long long)rows, (unsigned long long)cols);
    if (cpu_s > 0)
        printf("      CPU %8.2f ms/set (%6.1f GB/s)", cpu_s * 1000.0,
               set_mb / 1024.0 / cpu_s);
    else
        printf("      CPU      (skipped)        ");
    if (gpu_s > 0)
        printf("   GPU %8.2f ms/set (%6.1f GB/s)   speedup %5.2fx%s\n",
               gpu_s * 1000.0, set_mb / 1024.0 / gpu_s,
               (cpu_s > 0 ? cpu_s / gpu_s : 0.0),
               set_mb > 12.0 ? "  [DRAM-bound]" : "  [fits in L3 - not DRAM-bound]");
    else
        printf("   GPU unavailable\n");
    free(x); free(y); free(scratch);
    return 1;
}

/* -------------------------------------------------------------------------
 * cuda-check-moe — numerical validation of the FUSED layer-level MoE path.
 *
 * Reproduces exactly what moe_ffn does: for n_sel experts, gate and up GEMVs,
 * silu(gate)*up, router weighting, down GEMV, and accumulation — once on the CPU
 * with the engine's own kernels, once through katali_cuda_moe_layer().
 *
 * This exists because the fused path produced incoherent text end-to-end while
 * its telemetry looked perfect; a value-level comparison is the only way to see
 * where the numbers diverge.
 * ------------------------------------------------------------------------- */
static int cmd_cuda_check_moe(const char *path, int layer, int n_sel) {
    printf("cuda-check-moe: fused layer MoE vs CPU reference\n");
    if (!katali_cuda_available()) {
        printf("CUDA unavailable: %s\n", katali_cuda_status());
        return 0;
    }
    if (n_sel < 1) n_sel = 8;
    if (n_sel > 16) n_sel = 16;

    KataliGgufFile f;
    char err[256];
    if (katali_gguf_open(&f, path, 1, err, sizeof(err)) != 0) {
        fprintf(stderr, "open failed: %s (%s)\n", path, err);
        return 2;
    }
    char nm[96];
    snprintf(nm, sizeof(nm), "blk.%d.ffn_gate_exps.weight", layer);
    const KataliGgufTensor *gate = find_tensor(&f, nm);
    snprintf(nm, sizeof(nm), "blk.%d.ffn_up_exps.weight", layer);
    const KataliGgufTensor *up = find_tensor(&f, nm);
    snprintf(nm, sizeof(nm), "blk.%d.ffn_down_exps.weight", layer);
    const KataliGgufTensor *down = find_tensor(&f, nm);
    if (!gate || !up || !down) {
        printf("layer %d expert tensors not found\n", layer);
        katali_gguf_close(&f);
        return 2;
    }
    const uint64_t H  = gate->dims[0];
    const uint64_t FF = gate->dims[1];
    const uint64_t n_exp = (gate->n_dims >= 3) ? gate->dims[2] : 1;
    const uint64_t rb_gate = katali_ggml_row_bytes(gate->type, H);
    const uint64_t rb_up   = katali_ggml_row_bytes(up->type, H);
    const uint64_t rb_down = katali_ggml_row_bytes(down->type, FF);
    printf("layer=%d H=%llu FF=%llu experts=%llu  gate/up/down types=%s/%s/%s\n",
           layer, (unsigned long long)H, (unsigned long long)FF,
           (unsigned long long)n_exp, katali_ggml_type_name(gate->type),
           katali_ggml_type_name(up->type), katali_ggml_type_name(down->type));
    if (down->dims[0] != FF || down->dims[1] != H) {
        printf("unexpected down shape [%llu,%llu]\n",
               (unsigned long long)down->dims[0],
               (unsigned long long)down->dims[1]);
        katali_gguf_close(&f);
        return 2;
    }

    float *x  = (float *)malloc((size_t)H * sizeof(float));
    float *yc = (float *)malloc((size_t)H * sizeof(float));
    float *yg = (float *)malloc((size_t)H * sizeof(float));
    float *scr = (float *)malloc((size_t)(3 * FF + 2 * H) * sizeof(float));
    uint64_t sc_cap = KATALI_Q4K_SUMS_FLOATS(H > FF ? H : FF);
    if (sc_cap < 64) sc_cap = 64;
    float *q4k = (float *)malloc((size_t)sc_cap * sizeof(float));
    int *eids = (int *)malloc((size_t)n_sel * sizeof(int));
    float *wts = (float *)malloc((size_t)n_sel * sizeof(float));
    if (!x || !yc || !yg || !scr || !q4k || !eids || !wts) {
        printf("oom\n"); katali_gguf_close(&f); return 2;
    }
    uint32_t seed = 777u;
    for (uint64_t i = 0; i < H; i++) x[i] = cuda_lcg(&seed);
    for (int j = 0; j < n_sel; j++) {
        /* Spread experts across the bank rather than taking 0..n_sel-1, so a
         * wrong per-expert offset cannot look correct by accident. */
        eids[j] = (int)(((uint64_t)j * 37u) % n_exp);
        /* Diagnostic: force every job onto the same expert. If the fused path
         * then matches, per-job pointer handling is fine and the fault involves
         * distinct per-job bases. */
        if (getenv("KATALI_CHECKMOE_SAME")) eids[j] = 0;
        wts[j] = 1.0f / (float)n_sel;
    }

    float *g = scr, *u = scr + FF, *inter = scr + 2 * FF, *dout = scr + 3 * FF;
    /* acc must NOT overlap dout: dout spans [3*FF, 3*FF+H), so acc starts after
     * it. (An earlier revision placed acc at 4*FF, which overlapped dout's second
     * half and silently corrupted this CPU reference.) */
    float *acc = scr + 3 * FF + H;
    memset(acc, 0, (size_t)H * sizeof(float));
    for (int j = 0; j < n_sel; j++) {
        const int e = eids[j];
        const uint8_t *gp  = gate->data + (size_t)e * (size_t)(FF * rb_gate);
        const uint8_t *upp = up->data   + (size_t)e * (size_t)(FF * rb_up);
        const uint8_t *dp  = down->data + (size_t)e * (size_t)(H * rb_down);
        katali_ggml_matvec(gate->type, gp, FF, H, x, g, 0, NULL, q4k, sc_cap);
        katali_ggml_matvec(up->type, upp, FF, H, x, u, 0, NULL, q4k, sc_cap);
        for (uint64_t i = 0; i < FF; i++) {
            float gv = g[i];
            inter[i] = (gv / (1.f + expf(-gv))) * u[i];
        }
        katali_ggml_matvec(down->type, dp, H, FF, inter, dout, 0, NULL, q4k, sc_cap);
        for (uint64_t i = 0; i < H; i++) acc[i] += wts[j] * dout[i];
    }
    for (uint64_t i = 0; i < H; i++) yc[i] = acc[i];
    printf("CPU ref: first4 = %.5f %.5f %.5f %.5f\n", yc[0], yc[1], yc[2], yc[3]);

    /* --- GPU fused path (tensors still mapped) --- */
    void *dg[16], *du[16], *dd[16], *dx = NULL, *dy = NULL;
    int rc = 1;
    for (int j = 0; j < n_sel; j++) { dg[j] = du[j] = dd[j] = NULL; }
    if (katali_cuda_malloc(&dx, (size_t)H * sizeof(float)) != KATALI_OK) goto moe_done;
    if (katali_cuda_malloc(&dy, (size_t)H * sizeof(float)) != KATALI_OK) goto moe_done;
    if (katali_cuda_upload(dx, x, (size_t)H * sizeof(float)) != KATALI_OK) goto moe_done;
    for (int j = 0; j < n_sel; j++) {
        const int e = eids[j];
        const uint8_t *gp  = gate->data + (size_t)e * (size_t)(FF * rb_gate);
        const uint8_t *upp = up->data   + (size_t)e * (size_t)(FF * rb_up);
        const uint8_t *dp  = down->data + (size_t)e * (size_t)(H * rb_down);
        if (katali_cuda_malloc(&dg[j], (size_t)(FF * rb_gate)) != KATALI_OK) goto moe_done;
        if (katali_cuda_malloc(&du[j], (size_t)(FF * rb_up)) != KATALI_OK) goto moe_done;
        if (katali_cuda_malloc(&dd[j], (size_t)(H * rb_down)) != KATALI_OK) goto moe_done;
        if (katali_cuda_upload(dg[j], gp,  (size_t)(FF * rb_gate)) != KATALI_OK) goto moe_done;
        if (katali_cuda_upload(du[j], upp, (size_t)(FF * rb_up)) != KATALI_OK) goto moe_done;
        if (katali_cuda_upload(dd[j], dp,  (size_t)(H * rb_down)) != KATALI_OK) goto moe_done;
    }
    {
        KataliCudaMoeDesc desc;
        memset(&desc, 0, sizeof(desc));
        desc.gate_type = gate->type; desc.up_type = up->type;
        desc.down_type = down->type;
        desc.gate_row_bytes = rb_gate; desc.up_row_bytes = rb_up;
        desc.down_row_bytes = rb_down;
        desc.gate_rows = FF; desc.up_rows = FF; desc.down_rows = H;
        desc.hidden = H; desc.inter = FF;
        if (katali_cuda_moe_layer(&desc, (const void *const *)dg,
                                  (const void *const *)du,
                                  (const void *const *)dd, wts, n_sel,
                                  (const float *)dx, (float *)dy) != KATALI_OK) {
            printf("katali_cuda_moe_layer FAILED\n");
            goto moe_done;
        }
    }
    if (katali_cuda_download(yg, dy, (size_t)H * sizeof(float)) != KATALI_OK) goto moe_done;
    {
        double num = 0.0, den = 0.0, maxabs = 0.0;
        uint64_t bad = 0;
        for (uint64_t i = 0; i < H; i++) {
            double dv = (double)yg[i] - (double)yc[i];
            num += dv * dv;
            den += (double)yc[i] * (double)yc[i];
            if (fabs(dv) > maxabs) maxabs = fabs(dv);
            if (!(fabs(dv) <= 1e-3 * (1.0 + fabs((double)yc[i])))) bad++;
        }
        double rel2 = (den > 0.0) ? sqrt(num / den) : sqrt(num);
        /*
         * The tolerance depends on which kernel actually ran, and the line says
         * which one and which criterion was applied. The q8_1/DP4A path rounds
         * the activation once per 32 values on purpose, so its error is ~1e-2
         * *relative* on outputs whose own magnitude here is ~1e-3 — a relative
         * test on cancellation-prone values says nothing. What is meaningful is
         * the element-wise absolute criterion (1e-3*(1+|ref|)), which is counted
         * in both cases. The fp32 tolerance is unchanged at 1e-3.
         * Documented in docs/CUDA_PLAN.md §17.5.
         */
        const int is_dp4a = katali_cuda_dp4a_active();
        const double tol = is_dp4a ? 2e-2 : 1e-3;
        const int ok = (rel2 <= tol) && (bad == 0);
        printf("GPU fused: first4 = %.5f %.5f %.5f %.5f\n", yg[0], yg[1], yg[2], yg[3]);
        printf("rel_L2=%.3e max_abs=%.3e elems_bad=%llu/%llu  %s "
               "(kernel=%s, rel_L2 tolerance %.0e, element tolerance 1e-3*(1+|ref|))\n",
               rel2, maxabs, (unsigned long long)bad, (unsigned long long)H,
               ok ? "PASS" : "FAIL",
               is_dp4a ? "dp4a+q8_1" : "fp32 dequant", tol);
        rc = ok ? 0 : 1;
    }
    /*
     * Kernel-vs-dispatch isolation. Runs LAST so it cannot disturb `dy` before
     * the comparison above. Same GPU work, two shapes:
     *   fused : 4-5 launches + ONE sync
     *   per-op: 3 launches + THREE syncs
     * NOTE: the same slabs are re-read every rep, so this measures the L2-warm
     * rate (an upper bound), not the streaming rate the engine sees.
     */
    {
        KataliCudaMoeDesc d2;
        memset(&d2, 0, sizeof(d2));
        d2.gate_type = gate->type; d2.up_type = up->type; d2.down_type = down->type;
        d2.gate_row_bytes = rb_gate; d2.up_row_bytes = rb_up;
        d2.down_row_bytes = rb_down;
        d2.gate_rows = FF; d2.up_rows = FF; d2.down_rows = H;
        d2.hidden = H; d2.inter = FF;
        const int reps = 30;
        double t0 = katali_time_s();
        for (int r = 0; r < reps; r++)
            (void)katali_cuda_moe_layer(&d2, (const void *const *)dg,
                                        (const void *const *)du,
                                        (const void *const *)dd, wts, n_sel,
                                        (const float *)dx, (float *)dy);
        double fused_ms = (katali_time_s() - t0) * 1000.0 / reps;
        t0 = katali_time_s();
        for (int r = 0; r < reps; r++)
            katali_cuda_matvec(gate->type, dg[0], FF, H, (const float *)dx, (float *)dy);
        double one_ms = (katali_time_s() - t0) * 1000.0 / reps;
        double wmb = (double)n_sel * (double)(FF * rb_gate + FF * rb_up + H * rb_down)
                     / (1024.0 * 1024.0);
        printf("timing fused n_sel=%d x%d: %.3f ms/call  %.1f MB/call  %.1f GB/s "
               "(L2-warm upper bound; 1 gate matvec=%.3f ms)\n",
               n_sel, reps, fused_ms, wmb,
               (fused_ms > 0.0) ? wmb / 1024.0 / (fused_ms / 1000.0) : 0.0, one_ms);
    }
moe_done:
    for (int j = 0; j < n_sel; j++) {
        if (dg[j]) katali_cuda_free(dg[j]);
        if (du[j]) katali_cuda_free(du[j]);
        if (dd[j]) katali_cuda_free(dd[j]);
    }
    if (dx) katali_cuda_free(dx);
    if (dy) katali_cuda_free(dy);
    free(x); free(yc); free(yg); free(scr); free(q4k); free(eids); free(wts);
    katali_gguf_close(&f);
    return rc;
}

/* -------------------------------------------------------------------------
 * Stage A cold-benchmark helpers.
 *
 * One timing arm = `sets` fused-layer calls over `ns` rotating experts with a
 * given kernel strategy. Reported per call, so arms with different call counts
 * stay comparable, and the q8_1 quantization is reported from the device's own
 * event pair rather than inferred — the point of the exercise is that a faster
 * GEMV must not be allowed to hide the quantization it depends on.
 * ------------------------------------------------------------------------- */
typedef struct ColdArm {
    double gpu_s;         /* device seconds across the arm */
    double q8_s;          /* device seconds inside q8_1 quantization */
    double host_s;        /* wall time inside the ABI calls */
    double launches;      /* kernel launches per fused call */
    double q8_launches;   /* q8_1 quantization launches per fused call */
    int    dp4a;          /* what the backend reports it actually used */
} ColdArm;

static void cold_run_arm(const KataliCudaMoeDesc *desc, void *const *dg,
                         void *const *du, void *const *dd, const float *wts,
                         int ns, int sets, int slots, void *dx, void *dy,
                         int dp4a, int q4k, int q6k, ColdArm *out) {
    const void *sg[16], *su[16], *sd[16];
    memset(out, 0, sizeof(*out));
    katali_cuda_set_dp4a(dp4a);
    katali_cuda_set_dp4a_types(q4k, q6k);
    katali_cuda_telemetry_reset();
    for (int s = 0; s < sets; s++) {
        for (int j = 0; j < ns; j++) {
            const int e = (s * ns + j) % slots;
            sg[j] = dg[e]; su[j] = du[e]; sd[j] = dd[e];
        }
        (void)katali_cuda_moe_layer(desc, sg, su, sd, wts, ns,
                                    (const float *)dx, (float *)dy);
    }
    KataliCudaTelemetry t;
    katali_cuda_telemetry_get(&t);
    out->gpu_s = t.gpu_seconds;
    out->q8_s = t.q8_1_quant_seconds;
    out->host_s = t.host_seconds;
    out->launches = (double)t.kernel_launches / (double)sets;
    out->q8_launches = (double)t.q8_1_quant_launches / (double)sets;
    out->dp4a = katali_cuda_dp4a_active();
}

/* Numeric comparison of two device outputs, downloaded to host buffers. */
typedef struct ColdErr {
    double rel_l2;
    double max_abs;
    uint64_t bad;         /* |a-b| > 1e-3 */
    uint64_t n;
} ColdErr;

static ColdErr cold_cmp(const float *a, const float *b, uint64_t n) {
    ColdErr e;
    memset(&e, 0, sizeof(e));
    double d2 = 0.0, r2 = 0.0, mx = 0.0;
    for (uint64_t i = 0; i < n; i++) {
        const double d = (double)a[i] - (double)b[i];
        if (d < 0) { if (-d > mx) mx = -d; } else if (d > mx) mx = d;
        if (d > 1e-3 || d < -1e-3) e.bad++;
        d2 += d * d;
        r2 += (double)b[i] * (double)b[i];
    }
    e.rel_l2 = (r2 > 0.0) ? sqrt(d2 / r2) : 0.0;
    e.max_abs = mx;
    e.n = n;
    return e;
}

/* -------------------------------------------------------------------------
 * cuda-bench-cold — Phase 1/8/9 diagnostics.
 *
 * Builds the three-number ladder the optimisation work needs:
 *   1. raw VRAM streaming ceiling (Phase 9)   — what the card can actually do
 *   2. the same access pattern with no dequant (Phase 8) — memory vs decode
 *   3. the real fused GEMV (Phase 1)          — the number that must improve
 *
 * All three use a working set far larger than the ~24 MB L2, and the MoE cases
 * rotate through DIFFERENT experts every call so consecutive calls touch
 * disjoint slabs. That is what makes it represent L2-cold production traffic
 * instead of a warm benchmark that flatters the kernel.
 * ------------------------------------------------------------------------- */
static int cmd_cuda_bench_cold(const char *path, int layer, int slots_want) {
    printf("cuda-bench-cold: cold-memory ladder (Phase 1 / 8 / 9)\n");
    if (!katali_cuda_available()) {
        printf("CUDA unavailable: %s\n", katali_cuda_status());
        return 0;
    }
    /* Device-side event timing is what separates kernel time from host API
     * time. This must go through the ABI rather than an env var: the DLL is
     * loaded lazily and its CRT snapshots the environment at LoadLibrary time,
     * so a late _putenv() never reaches it. */
    katali_cuda_set_gpu_timing(1);

    KataliGgufFile f;
    char err[256];
    if (katali_gguf_open(&f, path, 1, err, sizeof(err)) != 0) {
        fprintf(stderr, "open failed: %s (%s)\n", path, err);
        return 2;
    }
    char nm[96];
    snprintf(nm, sizeof(nm), "blk.%d.ffn_gate_exps.weight", layer);
    const KataliGgufTensor *gate = find_tensor(&f, nm);
    snprintf(nm, sizeof(nm), "blk.%d.ffn_up_exps.weight", layer);
    const KataliGgufTensor *up = find_tensor(&f, nm);
    snprintf(nm, sizeof(nm), "blk.%d.ffn_down_exps.weight", layer);
    const KataliGgufTensor *down = find_tensor(&f, nm);
    if (!gate || !up || !down) {
        printf("layer %d expert tensors not found\n", layer);
        katali_gguf_close(&f);
        return 2;
    }
    const uint64_t H  = gate->dims[0];
    const uint64_t FF = gate->dims[1];
    const uint64_t n_exp = (gate->n_dims >= 3) ? gate->dims[2] : 1;
    const uint64_t rb_gate = katali_ggml_row_bytes(gate->type, H);
    const uint64_t rb_up   = katali_ggml_row_bytes(up->type, H);
    const uint64_t rb_down = katali_ggml_row_bytes(down->type, FF);
    const uint64_t slab = FF * rb_gate + FF * rb_up + H * rb_down;
    printf("model: H=%llu FF=%llu experts=%llu  types %s/%s/%s  slab=%.2f MiB/expert\n",
           (unsigned long long)H, (unsigned long long)FF,
           (unsigned long long)n_exp, katali_ggml_type_name(gate->type),
           katali_ggml_type_name(up->type), katali_ggml_type_name(down->type),
           (double)slab / (1024.0 * 1024.0));

    /* ---- Phase 9: raw streaming ceiling ---- */
    {
        const size_t BUFSZ = 256u << 20;   /* 256 MiB, >> 24 MiB L2 */
        void *ba = NULL, *bb = NULL;
        if (katali_cuda_malloc(&ba, BUFSZ) == KATALI_OK &&
            katali_cuda_malloc(&bb, BUFSZ) == KATALI_OK) {
            double rd = 0.0, cp = 0.0;
            if (katali_cuda_bench_stream(ba, bb, BUFSZ, 10, &rd, &cp) == KATALI_OK)
                printf("Phase 9 raw VRAM: seq_read=%.1f GB/s  copy(r+w)=%.1f GB/s  "
                       "[256 MiB buffer]\n", rd, cp);
            else
                printf("Phase 9 raw VRAM: FAILED\n");
        } else {
            printf("Phase 9 raw VRAM: could not allocate 2x256 MiB\n");
        }
        if (ba) katali_cuda_free(ba);
        if (bb) katali_cuda_free(bb);
    }

    /* ---- reserve a resident expert working set ----
     *
     * `slots` is the number of resident expert slabs. Its default mimics the
     * engine's steady state closely enough, but the interesting knob is raising
     * it: with the same per-call bytes but the slabs spread over several GB of
     * address space, we can tell whether the engine's 2.8 GB/s is an
     * address-space/TLB effect or a property of the kernel. Data is repeated
     * (slot % n_exp) so a large slot count is possible with 256 real experts. */
    int slots = slots_want;
    if (slots <= 0) {
        slots = (int)n_exp;
        if (slots > 192) slots = 192;
        if (slots < 8)   slots = (int)n_exp;
    }
    if (slots > (int)n_exp) slots = slots;   /* allowed: slabs repeat the data */
    void **dg = (void **)calloc((size_t)slots, sizeof(void *));
    void **du = (void **)calloc((size_t)slots, sizeof(void *));
    void **dd = (void **)calloc((size_t)slots, sizeof(void *));
    float *x   = (float *)malloc((size_t)H * sizeof(float));
    float *y   = (float *)malloc((size_t)H * sizeof(float));
    float *wts = (float *)malloc(16 * sizeof(float));
    const void *sel_g[16], *sel_u[16], *sel_d[16];
    void *dx = NULL, *dy = NULL;
    if (!dg || !du || !dd || !x || !y || !wts) { printf("oom\n"); goto cold_done; }
    {
        uint32_t seed = 31337u;
        for (uint64_t i = 0; i < H; i++) x[i] = cuda_lcg(&seed);
        for (int j = 0; j < 16; j++) wts[j] = 1.0f / 8.0f;
    }
    if (katali_cuda_malloc(&dx, (size_t)H * sizeof(float)) != KATALI_OK) goto cold_done;
    if (katali_cuda_malloc(&dy, (size_t)H * sizeof(float)) != KATALI_OK) goto cold_done;
    if (katali_cuda_upload(dx, x, (size_t)H * sizeof(float)) != KATALI_OK) goto cold_done;
    {
        int ok = 1;
        for (int e = 0; e < slots && ok; e++) {
            /* Repeat the real expert data when slots > n_exp so a multi-GB
             * address footprint can be built with only 256 distinct experts. */
            const int ex = e % (int)n_exp;
            if (katali_cuda_malloc(&dg[e], (size_t)(FF * rb_gate)) != KATALI_OK) ok = 0;
            else if (katali_cuda_malloc(&du[e], (size_t)(FF * rb_up)) != KATALI_OK) ok = 0;
            else if (katali_cuda_malloc(&dd[e], (size_t)(H * rb_down)) != KATALI_OK) ok = 0;
            else if (katali_cuda_upload(dg[e], gate->data + (size_t)ex * (FF * rb_gate),
                                        (size_t)(FF * rb_gate)) != KATALI_OK) ok = 0;
            else if (katali_cuda_upload(du[e], up->data + (size_t)ex * (FF * rb_up),
                                        (size_t)(FF * rb_up)) != KATALI_OK) ok = 0;
            else if (katali_cuda_upload(dd[e], down->data + (size_t)ex * (H * rb_down),
                                        (size_t)(H * rb_down)) != KATALI_OK) ok = 0;
        }
        if (!ok) { printf("could not stage %d experts in VRAM\n", slots); goto cold_done; }
    }
    printf("Phase 1/8 resident working set: %d experts = %.0f MiB (L2 ~24 MiB)\n",
           slots, (double)slots * (double)slab / (1024.0 * 1024.0));

    KataliCudaMoeDesc desc;
    memset(&desc, 0, sizeof(desc));
    desc.gate_type = gate->type; desc.up_type = up->type; desc.down_type = down->type;
    desc.gate_row_bytes = rb_gate; desc.up_row_bytes = rb_up;
    desc.down_row_bytes = rb_down;
    desc.gate_rows = FF; desc.up_rows = FF; desc.down_rows = H;
    desc.hidden = H; desc.inter = FF;
    (void)sel_d; (void)sel_g; (void)sel_u;

    printf("Phase 1/8 fused MoE, rotating experts every call:\n");
    printf("  %-5s %-6s %-8s %-9s %-9s %-7s %-8s %-7s %-7s %-7s %-4s\n",
           "n_sel", "calls", "dqGB/s", "dp4aGB/s", "nmGB/s", "dq_ms",
           "dp4a_ms", "q8_ms", "q8_lch", "launch", "d4");
    for (int ns = 1; ns <= 8; ns *= 2) {
        /* Keep the total experts touched ~constant across n_sel so every case
         * streams the same amount of cold data. */
        int sets = slots / ns;
        if (sets < 4) sets = 4;
        const double bytes = (double)sets * (double)ns * (double)slab;
        const double gb = bytes / (1024.0 * 1024.0 * 1024.0);
        const void *sg[16], *su[16], *sd[16];
        (void)sg; (void)su; (void)sd;

        ColdArm a_dq, a_dp, a_nm;
        /* fp32 dequantization (the pre-Stage-A kernel), DP4A + q8_1, and the
         * no-math read probe that bounds both from above. */
        cold_run_arm(&desc, dg, du, dd, wts, ns, sets, slots, dx, dy, 0, 1, 1, &a_dq);
        cold_run_arm(&desc, dg, du, dd, wts, ns, sets, slots, dx, dy, 1, 1, 1, &a_dp);
        /* The nomath arm is a separate entry point, not a flag. */
        katali_cuda_telemetry_reset();
        for (int s = 0; s < sets; s++) {
            for (int j = 0; j < ns; j++) {
                const int e = (s * ns + j) % slots;
                sg[j] = dg[e]; su[j] = du[e]; sd[j] = dd[e];
            }
            (void)katali_cuda_moe_layer_nomath(&desc, sg, su, sd, wts, ns,
                                               (const float *)dx, (float *)dy);
        }
        {
            KataliCudaTelemetry t2;
            katali_cuda_telemetry_get(&t2);
            a_nm.gpu_s = t2.gpu_seconds;
        }
        /* Restore the default (fp32) selection for anything that follows. */
        katali_cuda_set_dp4a(-1);
        katali_cuda_set_dp4a_types(-1, -1);

        const double dq_gbs = (a_dq.gpu_s > 0.0) ? gb / a_dq.gpu_s : 0.0;
        const double dp_gbs = (a_dp.gpu_s > 0.0) ? gb / a_dp.gpu_s : 0.0;
        const double nm_gbs = (a_nm.gpu_s > 0.0) ? gb / a_nm.gpu_s : 0.0;
        printf("  %-5d %-6d %-8.1f %-9.1f %-9.1f %-7.1f %-8.1f %-7.2f %-7.2f %-7.0f %-4d\n",
               ns, sets, dq_gbs, dp_gbs, nm_gbs, a_dq.gpu_s * 1000.0,
               a_dp.gpu_s * 1000.0, a_dp.q8_s * 1000.0, a_dp.q8_launches,
               a_dp.launches, a_dp.dp4a);

        /*
         * Churn probe. The engine's moe_layer call is always preceded by
         * vram_cache_get(): a few cudaMalloc/cudaFree plus a slab upload. Under
         * WDDM those are device-wide operations. This measures the SAME fused
         * call with that pattern injected, to find out whether the engine's
         * 14x slower submission is caused by the churn that precedes it.
         * Off unless KATALI_COLD_CHURN=1.
         */
        {
            const char *ce = getenv("KATALI_COLD_CHURN");
            if (ce && ce[0] && ce[0] != '0') {
                void *keep = NULL;
                double churn_s = 0.0;
                katali_cuda_telemetry_reset();
                for (int s = 0; s < sets; s++) {
                    for (int j = 0; j < ns; j++) {
                        const int e = (s * ns + j) % slots;
                        sg[j] = dg[e]; su[j] = du[e]; sd[j] = dd[e];
                    }
                    void *fresh = NULL;
                    (void)katali_cuda_malloc(&fresh, (size_t)(FF * rb_gate));
                    (void)katali_cuda_upload(fresh, gate->data,
                                             (size_t)(FF * rb_gate));
                    (void)katali_cuda_moe_layer(&desc, sg, su, sd, wts, ns,
                                                (const float *)dx, (float *)dy);
                    if (keep) { (void)katali_cuda_free(keep); }
                    keep = fresh;
                }
                if (keep) (void)katali_cuda_free(keep);
                KataliCudaTelemetry t3;
                katali_cuda_telemetry_get(&t3);
                churn_s = t3.gpu_seconds;
                printf("        churn: gpu=%.1f ms host=%.1f ms  %.1f GB/s\n",
                       t3.gpu_seconds * 1000.0, t3.host_seconds * 1000.0,
                       (churn_s > 0.0) ? gb / churn_s : 0.0);
            }
        }
    }

    /* ---------------------------------------------------------------------
     * Stage A: per-type A/B and correctness.
     *
     * Q4_K is the larger share of this model's expert bytes (gate+up = 1.18 MiB
     * versus down = 0.86 MiB per expert) but Q6_K's dequantization is the more
     * expensive one per element. Measuring each type alone settles which one to
     * attack first instead of assuming.
     *
     * The correctness arms compare the DP4A output against the fp32-dequant
     * output on the SAME inputs and the SAME experts, so the only difference is
     * the kernel. That is the tolerance basis for the whole path.
     * ------------------------------------------------------------------- */
    {
        const int ns = 8;
        int sets = slots / ns;
        if (sets < 4) sets = 4;
        const double gb = ((double)sets * (double)ns * (double)slab) /
                          (1024.0 * 1024.0 * 1024.0);
        const char *an[4] = { "fp32 dequant (baseline)", "dp4a Q4_K only",
                              "dp4a Q6_K only", "dp4a Q4_K + Q6_K" };
        const int ad4[4] = { 0, 1, 1, 1 };
        const int aq4[4] = { 1, 1, 0, 1 };
        const int aq6[4] = { 1, 0, 1, 1 };
        printf("Stage A per-type A/B (n_sel=%d, %d calls = %.0f MiB each):\n",
               ns, sets, gb * 1024.0);
        printf("  %-24s %-9s %-8s %-7s %-7s %-6s %-4s\n",
               "arm", "GB/s", "gpu_ms", "q8_ms", "q8_lch", "launch", "d4");
        for (int a = 0; a < 4; a++) {
            ColdArm r;
            cold_run_arm(&desc, dg, du, dd, wts, ns, sets, slots, dx, dy,
                         ad4[a], aq4[a], aq6[a], &r);
            const double gbs = (r.gpu_s > 0.0) ? gb / r.gpu_s : 0.0;
            printf("  %-24s %-9.1f %-8.2f %-7.2f %-7.2f %-6.0f %-4d\n",
                   an[a], gbs, r.gpu_s * 1000.0, r.q8_s * 1000.0,
                   r.q8_launches, r.launches, r.dp4a);
        }
        katali_cuda_set_dp4a(-1);
        katali_cuda_set_dp4a_types(-1, -1);

        void  *dy2 = NULL;
        float *h_ref = (float *)malloc((size_t)H * sizeof(float));
        float *h_got = (float *)malloc((size_t)H * sizeof(float));
        if (h_ref && h_got && katali_cuda_malloc(&dy2, (size_t)H * sizeof(float)) == KATALI_OK) {
            const void *cg[8], *cu[8], *cd[8];
            for (int j = 0; j < ns; j++) {
                const int e = j % slots;
                cg[j] = dg[e]; cu[j] = du[e]; cd[j] = dd[e];
            }
            /* Reference: the fp32-dequantization kernel. */
            katali_cuda_set_dp4a(0);
            (void)katali_cuda_moe_layer(&desc, cg, cu, cd, wts, ns,
                                        (const float *)dx, (float *)dy2);
            (void)katali_cuda_download(h_ref, dy2, (size_t)H * sizeof(float));
            printf("Stage A correctness vs the fp32-dequant kernel "
                   "(H=%llu, n_sel=%d, rel_L2 tolerance 1e-3):\n",
                   (unsigned long long)H, ns);
            printf("  %-24s %-11s %-11s %-8s %-6s\n",
                   "arm", "rel_L2", "max_abs", "bad(1e-3)", "d4");
            for (int a = 1; a < 4; a++) {
                katali_cuda_set_dp4a(ad4[a]);
                katali_cuda_set_dp4a_types(aq4[a], aq6[a]);
                (void)katali_cuda_moe_layer(&desc, cg, cu, cd, wts, ns,
                                            (const float *)dx, (float *)dy2);
                (void)katali_cuda_download(h_got, dy2, (size_t)H * sizeof(float));
                const ColdErr e = cold_cmp(h_got, h_ref, H);
                printf("  %-24s %-11.3e %-11.3e %-8llu %-6d\n", an[a],
                       e.rel_l2, e.max_abs, (unsigned long long)e.bad,
                       katali_cuda_dp4a_active());
            }
            katali_cuda_set_dp4a(-1);
            katali_cuda_set_dp4a_types(-1, -1);

            /* --- exactness test ------------------------------------------------
             * An activation that q8_1 represents EXACTLY: integers in [-127,127]
             * with a +-127 in EVERY 32-value block, so d = amax/127 = 1 exactly
             * and no rounding happens anywhere. (A first attempt used
             * (i*37)%255-127, which is integer-valued but whose per-block amax
             * is usually below 127 — so d < 1 and the values were still rounded,
             * and the test reported a quantization error that was not there.)
             * With d = 1 any residual difference between the DP4A and fp32
             * kernels is a kernel bug, not quantization error.
             *
             * The Q4_K arm here is a whole-layer test because its downstream
             * silu and down projection stay on the fp32 path; the Q6_K arm still
             * quantizes the intermediate by design, so it is reference-only. */
            float *xi = (float *)malloc((size_t)H * sizeof(float));
            if (xi) {
                for (uint64_t i = 0; i < H; i++) {
                    const int m4 = (int)(i & 3u), m13 = (int)(i % 13u) - 6;
                    xi[i] = (m4 == 0) ? 127.f : (m4 == 1) ? -127.f : (float)(m13 * 10);
                }
                if (katali_cuda_upload(dx, xi, (size_t)H * sizeof(float)) == KATALI_OK) {
                    katali_cuda_set_dp4a(0);
                    (void)katali_cuda_moe_layer(&desc, cg, cu, cd, wts, ns,
                                                (const float *)dx, (float *)dy2);
                    (void)katali_cuda_download(h_ref, dy2, (size_t)H * sizeof(float));
                    printf("Stage A exactness (integer activation, q8_1 is exact):\n");
                    printf("  %-24s %-11s %-11s %-8s\n", "arm", "rel_L2", "max_abs", "bad(1e-3)");
                    for (int a = 1; a < 4; a++) {
                        katali_cuda_set_dp4a(ad4[a]);
                        katali_cuda_set_dp4a_types(aq4[a], aq6[a]);
                        (void)katali_cuda_moe_layer(&desc, cg, cu, cd, wts, ns,
                                                    (const float *)dx, (float *)dy2);
                        (void)katali_cuda_download(h_got, dy2, (size_t)H * sizeof(float));
                        const ColdErr e = cold_cmp(h_got, h_ref, H);
                        printf("  %-24s %-11.3e %-11.3e %-8llu\n", an[a],
                               e.rel_l2, e.max_abs, (unsigned long long)e.bad);
                    }
                    katali_cuda_set_dp4a(0);
                    katali_cuda_set_dp4a_types(1, 1);
                    (void)katali_cuda_upload(dx, x, (size_t)H * sizeof(float));
                }
                free(xi);
            }
            katali_cuda_set_dp4a(-1);
            katali_cuda_set_dp4a_types(-1, -1);
        } else {
            printf("Stage A correctness: buffers unavailable\n");
        }
        if (dy2) katali_cuda_free(dy2);
        free(h_ref);
        free(h_got);
    }

cold_done:
    for (int e = 0; e < slots; e++) {
        if (dg[e]) katali_cuda_free(dg[e]);
        if (du[e]) katali_cuda_free(du[e]);
        if (dd[e]) katali_cuda_free(dd[e]);
    }
    if (dx) katali_cuda_free(dx);
    if (dy) katali_cuda_free(dy);
    free(dg); free(du); free(dd); free(x); free(y); free(wts);
    katali_gguf_close(&f);
    return 0;
}

/* ---------------------------------------------------------------------------
 * cuda-dp4a-selftest — synthetic exactness test for the DP4A kernels.
 *
 * Why it exists: the cold benchmark can only compare the DP4A path against the
 * fp32 path on real weights with a real activation, and that difference always
 * contains the q8_1 rounding of the activation. That is fine for a tolerance
 * statement but useless for proving a kernel right.
 *
 * This test removes the quantization from the equation. It builds synthetic
 * expert slabs whose Q4_K blocks decode to exactly 1.0, feeds a constant
 * activation, and therefore produces an intermediate that is constant inside
 * every 32-value block — which q8_1 represents exactly (d = amax/127, every
 * q = 127, so the round trip is exact to ~1e-7). With no rounding anywhere, the
 * DP4A result and the fp32 result must agree to float precision; any larger
 * difference is a kernel bug in the weight-side layout (nibble halves, 6-bit
 * scales, 16-value scale groups, the `-dmin*m*sum(x)` term, ...).
 *
 * The down slab is deliberately made of ARBITRARY quantized bytes with only `d`
 * set to a finite fp16: both kernels must interpret the same arbitrary layout
 * identically, so every field of the Q6_K block is exercised.
 * ------------------------------------------------------------------------- */
static int cmd_cuda_dp4a_selftest(void) {
    printf("cuda-dp4a-selftest: synthetic exactness test for DP4A kernels\n");
    if (!katali_cuda_available()) {
        printf("CUDA unavailable: %s\n", katali_cuda_status());
        return 0;
    }
    katali_cuda_set_gpu_timing(1);

    const uint64_t H = 2048, FF = 512;              /* same shape as the 35B */
    const uint64_t rb_gate = (H / 256) * 144;       /* Q4_K row  */
    const uint64_t rb_down = (FF / 256) * 210;      /* Q6_K row  */
    const size_t sz_g = (size_t)FF * rb_gate;       /* 860 KiB   */
    const size_t sz_d = (size_t)H * rb_down;        /* 860 KiB   */

    uint8_t *h_gate = (uint8_t *)malloc(sz_g);
    uint8_t *h_up   = (uint8_t *)malloc(sz_g);
    uint8_t *h_down = (uint8_t *)malloc(sz_d);
    float   *hx     = (float *)malloc((size_t)H * sizeof(float));
    float   *ref    = (float *)malloc((size_t)H * sizeof(float));
    float   *got    = (float *)malloc((size_t)H * sizeof(float));
    if (!h_gate || !h_up || !h_down || !hx || !ref || !got) {
        printf("oom\n");
        free(h_gate); free(h_up); free(h_down); free(hx); free(ref); free(got);
        return 1;
    }

    /* Q4_K block that decodes to exactly 1.0: d = 1, dmin = 0, all 6-bit
     * scales = 1 (which needs scales[0..3] = 1 and scales[8..11] = 1, since
     * j >= 4 takes its scale from the low nibble of scales[j+4]), quants = 1. */
    for (size_t b = 0; b < sz_g / 144; b++) {
        uint8_t *blk = h_gate + b * 144;
        blk[0] = 0x00; blk[1] = 0x3C;   /* fp16 1.0  */
        blk[2] = 0x00; blk[3] = 0x00;   /* fp16 0.0  */
        memset(blk + 4, 0, 12);
        blk[4] = 1; blk[5] = 1; blk[6] = 1; blk[7] = 1;
        blk[12] = 1; blk[13] = 1; blk[14] = 1; blk[15] = 1;
        memset(blk + 16, 0x11, 128);
    }
    memcpy(h_up, h_gate, sz_g);

    /* Q6_K blocks with arbitrary quants and small scales, d = fp16 0.5. */
    {
        uint32_t seed = 12345u;
        for (size_t b = 0; b < sz_d / 210; b++) {
            uint8_t *blk = h_down + b * 210;
            for (int i = 0; i < 192; i++) blk[i] = (uint8_t)(cuda_lcg(&seed) * 256.0f);
            for (int i = 0; i < 16; i++)
                blk[192 + i] = (uint8_t)(int8_t)((int)(cuda_lcg(&seed) * 17.0f) - 8);
            blk[208] = 0x00; blk[209] = 0x38;   /* fp16 0.5 */
        }
    }

    /* Constant activation: the intermediates below are then constant per block. */
    for (uint64_t i = 0; i < H; i++) hx[i] = 0.001f;

    void *dg = NULL, *du = NULL, *dd = NULL, *dx = NULL, *dy = NULL;
    int rc = 1;
    if (katali_cuda_malloc(&dg, sz_g) != KATALI_OK) goto done;
    if (katali_cuda_malloc(&du, sz_g) != KATALI_OK) goto done;
    if (katali_cuda_malloc(&dd, sz_d) != KATALI_OK) goto done;
    if (katali_cuda_malloc(&dx, (size_t)H * sizeof(float)) != KATALI_OK) goto done;
    if (katali_cuda_malloc(&dy, (size_t)H * sizeof(float)) != KATALI_OK) goto done;
    if (katali_cuda_upload(dg, h_gate, sz_g) != KATALI_OK) goto done;
    if (katali_cuda_upload(du, h_up, sz_g) != KATALI_OK) goto done;
    if (katali_cuda_upload(dd, h_down, sz_d) != KATALI_OK) goto done;
    if (katali_cuda_upload(dx, hx, (size_t)H * sizeof(float)) != KATALI_OK) goto done;
    printf("slabs: gate/up %.2f MiB Q4_K, down %.2f MiB Q6_K, x constant\n",
           (double)sz_g / (1024.0 * 1024.0), (double)sz_d / (1024.0 * 1024.0));

    KataliCudaMoeDesc d;
    memset(&d, 0, sizeof(d));
    d.gate_type = 12;  /* Q4_K */
    d.up_type   = 12;
    d.down_type = 14;  /* Q6_K */
    d.gate_row_bytes = rb_gate; d.up_row_bytes = rb_gate; d.down_row_bytes = rb_down;
    d.gate_rows = FF; d.up_rows = FF; d.down_rows = H;
    d.hidden = H; d.inter = FF;
    const void *cg = dg, *cu = du, *cd = dd;
    const float wts1[1] = { 1.0f };

    katali_cuda_set_dp4a(0);
    if (katali_cuda_moe_layer(&d, &cg, &cu, &cd, wts1, 1,
                              (const float *)dx, (float *)dy) != KATALI_OK) {
        printf("fp32 reference call FAILED\n");
        goto done;
    }
    if (katali_cuda_download(ref, dy, (size_t)H * sizeof(float)) != KATALI_OK) goto done;

    katali_cuda_set_dp4a(1);
    if (katali_cuda_moe_layer(&d, &cg, &cu, &cd, wts1, 1,
                              (const float *)dx, (float *)dy) != KATALI_OK) {
        printf("dp4a call FAILED\n");
        goto done;
    }
    if (katali_cuda_download(got, dy, (size_t)H * sizeof(float)) != KATALI_OK) goto done;
    katali_cuda_set_dp4a(-1);

    {
        const ColdErr e = cold_cmp(got, ref, H);
        /* The reference values here are ~1e4, so an absolute "bad element"
         * criterion is meaningless and the pass test must be relative. */
        uint64_t bad_rel = 0;
        double mx = 0.0;
        for (uint64_t i = 0; i < H; i++) {
            const double r = fabs((double)ref[i]);
            const double dv = fabs((double)got[i] - (double)ref[i]);
            if (dv > 1e-5 * (r > 1.0 ? r : 1.0)) bad_rel++;
            if (dv > mx) mx = dv;
        }
        printf("ref[0..3] = %.6f %.6f %.6f %.6f\n", ref[0], ref[1], ref[2], ref[3]);
        printf("dp4a[0..3]= %.6f %.6f %.6f %.6f  (dp4a_active=%d)\n",
               got[0], got[1], got[2], got[3], katali_cuda_dp4a_active());
        printf("rel_L2=%.3e max_abs=%.3e bad(|d|>1e-5*|ref|)=%llu/%llu  %s\n",
               e.rel_l2, mx, (unsigned long long)bad_rel, (unsigned long long)H,
               (e.rel_l2 <= 1e-5) ? "PASS" : "FAIL");
        /* The pass criterion is the GLOBAL relative L2 error: an element-wise
         * relative test is meaningless here because individual outputs can be
         * near-zero by cancellation while their absolute error stays at fp32
         * rounding level. */
        rc = (e.rel_l2 <= 1e-5) ? 0 : 1;
    }

done:
    if (dg) katali_cuda_free(dg);
    if (du) katali_cuda_free(du);
    if (dd) katali_cuda_free(dd);
    if (dx) katali_cuda_free(dx);
    if (dy) katali_cuda_free(dy);
    free(h_gate); free(h_up); free(h_down); free(hx); free(ref); free(got);
    return rc;
}

/* -------------------------------------------------------------------------
 * cuda-hold — keep the GPU under continuous memory load for N seconds.
 *
 * Diagnostic only. NVIDIA's power management keeps the 4060 at its IDLE clocks
 * (210 MHz SM / 405 MHz mem) when the workload is a low-duty-cycle burst, which
 * is what real inference looks like here: ~0.5 ms of GPU per ~8 ms of CPU work.
 * Running this alongside the engine proves whether that clock state is what
 * limits cold GEMV throughput.
 * ------------------------------------------------------------------------- */
static int cmd_cuda_hold(int seconds) {
    if (!katali_cuda_available()) {
        printf("CUDA unavailable: %s\n", katali_cuda_status());
        return 0;
    }
    const size_t BUFSZ = 256u << 20;
    void *ba = NULL, *bb = NULL;
    if (katali_cuda_malloc(&ba, BUFSZ) != KATALI_OK) { printf("hold: oom\n"); return 1; }
    if (katali_cuda_malloc(&bb, BUFSZ) != KATALI_OK) { printf("hold: oom\n"); return 1; }
    const double t0 = katali_time_s();
    double rd = 0.0, n = 0.0, sum = 0.0;
    while ((katali_time_s() - t0) < (double)seconds) {
        if (katali_cuda_bench_stream(ba, bb, BUFSZ, 5, &rd, NULL) != KATALI_OK) break;
        sum += rd;
        n += 1.0;
    }
    printf("cuda-hold: %.1f s, %d passes, mean read %.1f GB/s\n",
           katali_time_s() - t0, (int)n, n > 0.0 ? sum / n : 0.0);
    katali_cuda_free(ba);
    katali_cuda_free(bb);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) { usage(argv[0]); return 1; }
    if (strcmp(argv[1], "api") == 0) {
        int port = 8080;
        for (int i=2;i+1<argc;i++) if (strcmp(argv[i], "--port") == 0) port=atoi(argv[++i]);
        return katali_api_serve(argv[0], port);
    }

    if (strcmp(argv[1], "info") == 0) {
        KataliArch a;
        katali_arch_qwen36_a3b_defaults(&a);
        katali_arch_print(&a);
        katali_arch_free(&a);
        uint64_t avail = katali_ram_avail_bytes();
        printf("avail RAM: %.2f GiB\n", (double)avail / (1024.0 * 1024.0 * 1024.0));
        printf("default gguf: %s\n", DEFAULT_GGUF);
        printf("122B split dir: %s\n", DEFAULT_GGUF_122B);
        return 0;
    }

    if (strcmp(argv[1], "cuda-info") == 0 || strcmp(argv[1], "gpu") == 0) {
        int avail = katali_cuda_available();
        const KataliCudaDeviceInfo *d;
        printf("cuda: %s\n", avail ? "available" : "unavailable");
        printf("status: %s\n", katali_cuda_status());
        if (katali_cuda_lib_path()[0])
            printf("backend dll: %s\n", katali_cuda_lib_path());
        d = katali_cuda_device();
        if (d) {
            printf("device: %s\n", d->name);
            if (d->arch[0]) printf("architecture: %s\n", d->arch);
            printf("compute capability: %d.%d\n", d->cc_major, d->cc_minor);
            printf("multiprocessors: %d\n", d->sm_count);
            printf("VRAM total: %.2f GiB\n",
                   (double)d->vram_total_bytes / (1024.0 * 1024.0 * 1024.0));
            printf("VRAM free: %.2f GiB\n",
                   (double)d->vram_free_bytes / (1024.0 * 1024.0 * 1024.0));
            printf("CUDA driver version: %d.%d\n",
                   d->driver_version / 1000, (d->driver_version % 1000) / 10);
            printf("CUDA runtime version: %d.%d\n",
                   d->runtime_version / 1000, (d->runtime_version % 1000) / 10);
            printf("tier: VRAM + CPU + system RAM + SSD\n");
        } else {
            printf("tier: CPU + system RAM + SSD (no GPU acceleration)\n");
        }
        /* Exercise teardown so the path is covered even when CUDA is absent. */
        katali_cuda_shutdown();
        printf("after shutdown: %s\n", katali_cuda_status());
        return 0; /* missing GPU is never an error */
    }

    if (strcmp(argv[1], "cuda-check") == 0) {
        const char *p = gguf_arg(argc, argv, 2);
        return cmd_cuda_check(p);
    }
    if (strcmp(argv[1], "cuda-bench") == 0) {
        const char *p = gguf_arg(argc, argv, 2);
        return cmd_cuda_bench(p);
    }
    if (strcmp(argv[1], "cuda-hold") == 0) {
        int secs = (argc > 2) ? atoi(argv[2]) : 30;
        if (secs < 1) secs = 30;
        return cmd_cuda_hold(secs);
    }
    if (strcmp(argv[1], "cuda-dp4a-selftest") == 0) {
        return cmd_cuda_dp4a_selftest();
    }
    if (strcmp(argv[1], "cuda-bench-cold") == 0) {
        const char *p = DEFAULT_GGUF;
        int layer = 0, slots = 0, ai = 2;
        if (argc >= 3 && argv[2][0] != '-') { p = argv[2]; ai = 3; }
        if (argc > ai) layer = atoi(argv[ai]);
        if (argc > ai + 1) slots = atoi(argv[ai + 1]);
        return cmd_cuda_bench_cold(p, layer, slots);
    }
    if (strcmp(argv[1], "cuda-check-moe") == 0) {
        const char *p = DEFAULT_GGUF;
        int layer = 0, n_sel = 8, ai = 2;
        if (argc >= 3 && argv[2][0] != '-') { p = argv[2]; ai = 3; }
        if (argc > ai) layer = atoi(argv[ai]);
        if (argc > ai + 1) n_sel = atoi(argv[ai + 1]);
        return cmd_cuda_check_moe(p, layer, n_sel);
    }

    if (strcmp(argv[1], "selftest") == 0) {
        int rc = q4_selftest();
        printf("q4_selftest: %s\n", rc == KATALI_OK ? "PASS" : "FAIL");
        return rc == KATALI_OK ? 0 : 2;
    }

    if (strcmp(argv[1], "inspect") == 0) {
        const char *path = gguf_arg(argc, argv, 2);
        KataliGgufFile f;
        char err[256];
        if (katali_gguf_open(&f, path, 1, err, sizeof(err)) != 0) {
            fprintf(stderr, "open failed: %s (%s)\n", path, err);
            return 2;
        }
        print_gguf_info(&f);
        KataliArch a;
        katali_arch_from_gguf(&f, &a);
        katali_arch_print(&a);
        katali_arch_free(&a);
        katali_gguf_close(&f);
        return 0;
    }

    if (strcmp(argv[1], "tensors") == 0) {
        const char *path = DEFAULT_GGUF;
        const char *sub = NULL;
        if (argc >= 3 && argv[2][0] != '-') {
            path = argv[2];
            if (argc >= 4) sub = argv[3];
        } else if (argc >= 3) sub = argv[2];
        KataliGgufFile f;
        char err[256];
        if (katali_gguf_open(&f, path, 1, err, sizeof(err)) != 0) {
            fprintf(stderr, "open failed: %s\n", err);
            return 2;
        }
        print_tensors(&f, sub);
        katali_gguf_close(&f);
        return 0;
    }

    if (strcmp(argv[1], "open") == 0) {
        const char *path = gguf_arg(argc, argv, 2);
        int pin = 25;
        size_t cache_gb = 0;
        for (int i = 2; i < argc; i++) {
            if (strcmp(argv[i], "--pin") == 0 && i + 1 < argc) pin = atoi(argv[++i]);
            else if (strcmp(argv[i], "--cache-gb") == 0 && i + 1 < argc) cache_gb = (size_t)atoi(argv[++i]);
        }
        HostModel m;
        size_t cache = cache_gb ? cache_gb * (1024ull * 1024ull * 1024ull) : 0;
        int rc = host_open(&m, path, cache, pin);
        if (rc != KATALI_OK) {
            fprintf(stderr, "host_open failed (%d): %s\n", rc, m.err);
            return 2;
        }
        print_gguf_info(&m.gguf);
        katali_arch_print(&m.arch);
        size_t used = 0, cap = 0; int n = 0;
        ecache_stats(&m.ecache, &used, &cap, &n);
        printf("ecache: cap=%.2f GiB pin=%d%% resident=%d\n",
               (double)cap / (1024.0 * 1024.0 * 1024.0), pin, n);
        printf("moe layers indexed: %d experts: %d\n", m.moe.n_layers, m.moe.n_experts);
        printf("tok_ok=%d vocab=%d max_seq=%d n_full=%d n_recurrent=%d\n",
               m.tok_ok, m.arch.vocab, m.st.max_seq, m.st.n_full, m.st.n_recurrent);
        host_close(&m);
        return 0;
    }

    if (strcmp(argv[1], "probe-expert") == 0) {
        const char *path = DEFAULT_GGUF;
        int ai = 2;
        if (argc >= 5) { path = argv[2]; ai = 3; }
        if (argc < ai + 2) { usage(argv[0]); return 1; }
        int layer = atoi(argv[ai]);
        int expert = atoi(argv[ai + 1]);
        HostModel m;
        int rc = host_open(&m, path, 0, 25);
        if (rc != KATALI_OK) { fprintf(stderr, "host_open failed (%d)\n", rc); return 2; }
        rc = host_probe_expert(&m, layer, expert);
        host_close(&m);
        return rc == KATALI_OK ? 0 : 2;
    }

    if (strcmp(argv[1], "moe-ffn") == 0) {
        const char *path = DEFAULT_GGUF;
        int ai = 2;
        if (argc >= 4) { path = argv[2]; ai = 3; }
        if (argc < ai + 1) { usage(argv[0]); return 1; }
        int layer = atoi(argv[ai]);
        HostModel m;
        int rc = host_open(&m, path, 0, 25);
        if (rc != KATALI_OK) { fprintf(stderr, "host_open failed (%d)\n", rc); return 2; }
        int H = m.arch.hidden;
        float *x = (float *)calloc((size_t)H, sizeof(float));
        float *y = (float *)calloc((size_t)H, sizeof(float));
        if (!x || !y) { host_close(&m); return 2; }
        x[0] = 1.f;
        double t0 = katali_time_s();
        rc = moe_ffn_forward(&m, layer, x, y);
        double t1 = katali_time_s();
        if (rc != KATALI_OK) {
            fprintf(stderr, "moe_ffn_forward failed (%d)\n", rc);
            free(x); free(y); host_close(&m); return 2;
        }
        double nrm = 0.0;
        for (int i = 0; i < H; i++) nrm += (double)y[i] * (double)y[i];
        nrm = sqrt(nrm);
        size_t used=0,cap=0; int n=0;
        ecache_stats(&m.ecache, &used, &cap, &n);
        printf("moe-ffn layer=%d ok  ||y||=%.6g  time=%.3fs  ecache=%.2f MiB resident=%d\n",
               layer, nrm, t1 - t0, (double)used/(1024.0*1024.0), n);
        free(x); free(y); host_close(&m);
        return 0;
    }

    if (strcmp(argv[1], "dump-emb") == 0) {
        const char *path = DEFAULT_GGUF;
        int tid = 0;
        if (argc >= 4 && argv[2][0] != '-' && argv[3][0] != '-') { path = argv[2]; tid = atoi(argv[3]); }
        else if (argc >= 3) tid = atoi(argv[2]);
        else { usage(argv[0]); return 1; }
        HostModel m; memset(&m, 0, sizeof(m));
        int rc = host_open(&m, path, 0, 25);
        if (rc != KATALI_OK) { fprintf(stderr, "open failed (%d): %s\n", rc, m.err); return 1; }
        const KataliGgufTensor *te = m.moe.tok_embd;
        int H = m.arch.hidden;
        uint64_t row_b = katali_ggml_row_bytes(te->type, (uint64_t)H);
        const uint8_t *row = te->data + (size_t)tid * (size_t)row_b;
        float *x = (float *)malloc((size_t)H * sizeof(float));
        if (katali_ggml_dequant_ref(te->type, row, (uint64_t)H, x) != 0) { fprintf(stderr, "dequant fail\n"); return 1; }
        printf("token=%d type=%u row_bytes=%llu H=%d\n", tid, te->type, (unsigned long long)row_b, H);
        printf("first16:");
        for (int i = 0; i < 16; i++) printf(" %.6g", x[i]);
        printf("\n");
        double sum=0, sq=0; for (int i=0;i<H;i++){sum+=x[i];sq+=x[i]*x[i];}
        printf("mean=%.6g rms=%.6g\n", sum/H, sqrt(sq/H));
        free(x); host_close(&m); return 0;
    }
    if (strcmp(argv[1], "generate") == 0) {
        const char *path = DEFAULT_GGUF;
        const char *prompt = NULL;
        char *prompt_owned = NULL;
        int max_tokens = 64;
        int pin = 25;
        size_t cache_gb = 0;
        int ai = 2;
        if (argc >= 4 && argv[2][0] != '-' && argv[3][0] != '-') {
            path = argv[2];
            prompt = argv[3];
            ai = 4;
        } else if (argc >= 3 && argv[2][0] != '-') {
            prompt = argv[2];
            ai = 3;
        }
        for (int i = 2; i < argc; i++) {
            if (strcmp(argv[i], "--max") == 0 && i + 1 < argc) max_tokens = atoi(argv[++i]);
            else if (strcmp(argv[i], "--pin") == 0 && i + 1 < argc) pin = atoi(argv[++i]);
            else if (strcmp(argv[i], "--cache-gb") == 0 && i + 1 < argc) cache_gb = (size_t)atoi(argv[++i]);
            else if (strcmp(argv[i], "--file") == 0 && i + 1 < argc) {
                const char *fp = argv[++i];
                FILE *f = fopen(fp, "rb");
                if (!f) { fprintf(stderr, "cannot open --file %s\n", fp); return 1; }
                fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
                if (sz < 0) { fclose(f); return 1; }
                prompt_owned = (char *)malloc((size_t)sz + 1);
                if (!prompt_owned) { fclose(f); return 1; }
                if (fread(prompt_owned, 1, (size_t)sz, f) != (size_t)sz) { fclose(f); free(prompt_owned); return 1; }
                prompt_owned[sz] = 0; fclose(f); prompt = prompt_owned;
            }
        }
        if (!prompt) { usage(argv[0]); return 1; }
        HostModel m;
        size_t cache = cache_gb ? cache_gb * (1024ull * 1024ull * 1024ull) : 0;
        int rc = host_open(&m, path, cache, pin);
        if (rc != KATALI_OK) {
            fprintf(stderr, "host_open failed (%d): %s\n", rc, m.err);
            free(prompt_owned);
            return 2;
        }
        {
            size_t used = 0, cap = 0; int n = 0;
            ecache_stats(&m.ecache, &used, &cap, &n);
            fprintf(stderr, "phase: ecache cap=%.2f GiB pin=%d%% resident=%d\n",
                    (double)cap / (1024.0 * 1024.0 * 1024.0), pin, n);
        }
        {
            const char *e = getenv("KATALI_DEBUG");
            if (!e || !*e || *e == '0') e = getenv("KATALI_DEBUG_GEN");
            if (e && *e && *e != '0')
                fprintf(stderr, "generate max=%d prompt_len=%zu pin=%d cache_gb=%zu\n",
                        max_tokens, strlen(prompt), pin, cache_gb);
        }
        rc = host_generate(&m, prompt, max_tokens);
        host_close(&m);
        free(prompt_owned);
        return rc == KATALI_OK ? 0 : 2;
    }

    usage(argv[0]);
    return 1;
}
