/* Katali-GGUF diagnostic CLI — Apache-2.0
 *
 *   katali-gguf inspect   MODEL.gguf
 *   katali-gguf validate  MODEL.gguf
 *   katali-gguf run       MODEL.gguf --prompt "Hi" [--max-tokens N] [--ctx N]
 *                                         [--threads N] [--no-think] [--temp T]
 *   katali-gguf benchmark MODEL.gguf [--prompt "..."] [--max-tokens N]
 */
#include "katali_gguf.h"
#include "katali_gguf_dtype.h"
#include "katali_gguf_simd.h"
#include "katali_gguf_prof.h"
#include "katali_gguf_model.h"
#include "katali_session.h"
#include "katali_gguf_threads.h"
#include "katali_backend.h"
#include "katali_gguf_server.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0600
#endif
#include <windows.h>
#include <psapi.h>
static unsigned long long proc_rss_bytes(void) {
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
        return (unsigned long long)pmc.WorkingSetSize;
    return 0;
}
#else
static unsigned long long proc_rss_bytes(void) { return 0; }
#endif

static void usage(void) {
    printf(
        "Katali-GGUF native GGUF inference engine\n"
        "\n"
        "usage:\n"
        "  katali-gguf inspect   MODEL.gguf [--tensors]\n"
        "  katali-gguf validate  MODEL.gguf\n"
        "  katali-gguf tokenize  MODEL.gguf [--text TXT] [--grep S] [--range A B]\n"
        "  katali-gguf serve     MODEL.gguf [--port N]   (local HTTP API, 127.0.0.1)\n"
        "  katali-gguf run       MODEL.gguf --prompt \"Hi\" [options]\n"
        "  katali-gguf benchmark MODEL.gguf [--prompt \"Hi\"] [options]\n"
        "  katali-gguf kernels   MODEL.gguf [--iters N]\n"
        "  katali-gguf chat      MODEL.gguf [options]   (persistent session, stdin)\n"
        "  katali-gguf bench-session MODEL.gguf [--prompt Q]... [options]\n"
        "\n"
        "options:\n"
        "  --prompt TEXT        prompt (run/benchmark)\n"
        "  --max-tokens N       generation cap (default 128)\n"
        "  --ctx N              KV context length\n"
        "  --threads N          0 = auto\n"
        "  --temp T             temperature (0 = greedy)\n"
        "  --top-k N            top-k sampling\n"
        "  --top-p T            nucleus sampling\n"
        "  --rep P              repetition penalty (e.g. 1.1)\n"
        "  --seed N             sampler seed\n"
        "  --iters N            kernel microbenchmark iterations (default 200)\n"
        "  --think / --no-think allow / suppress a thinking block\n"
        "  --no-mmap            read the file into the heap instead of mapping\n"
        "\n"
        "environment:\n"
        "  KATALI_GGUF_THREADS   default thread count\n"
        "  KATALI_GGUF_NO_SIMD   1 = force the scalar kernels (A/B measurement)\n"
        "  KATALI_GGUF_NO_BATCH_PREFILL 1 = force token-at-a-time prompt prefill\n"
        "  KATALI_GGUF_NO_Q4K_SUMCACHE 1 = recompute Q4_K activation sums per row (A/B)\n"
        "  KATALI_GGUF_NO_Q4K_DUALROW 1 = single-row Q4_K loop instead of two rows per iteration\n"
        "  KATALI_GGUF_SYNC_STATS 1 = print a per-dispatch wake/compute/wait breakdown\n"
        "  KATALI_GGUF_NO_FUSE_DISPATCH 1 = one dispatch per projection instead of fusing q/k/v and gate/up (A/B)\n"
        "  KATALI_GGUF_SPLIT_DISPATCH K = split each matvec into K dispatches (measurement only)\n");
}

static void print_info(const KataliGgufInfo *in) {
    printf("format: GGUF\n");
    printf("version: %u\n", in->version);
    printf("architecture: %s\n", in->architecture);
    printf("vocabulary: %d\n", in->vocab);
    printf("layers: %d\n", in->layers);
    printf("hidden_size: %d\n", in->hidden);
    printf("attention_heads: %d\n", in->attention_heads);
    printf("key_value_heads: %d\n", in->key_value_heads);
    printf("context_length: %d\n", in->context_length);
    printf("quantization: %s\n", in->quantization);
    printf("tensor_count: %llu\n", (unsigned long long)in->tensor_count);
    printf("estimated_memory: %llu\n", (unsigned long long)in->estimated_memory);
}

static int cmd_inspect(const char *path, int show_tensors) {
    KataliGgufFile f;
    char err[256];
    if (katali_gguf_open(&f, path, 1, err, sizeof(err)) != 0) {
        fprintf(stderr, "katali-gguf: %s\n", err);
        return 2;
    }
    KataliGgufInfo in;
    katali_gguf_describe(&f, &in);
    print_info(&in);
    printf("mapped: %s\n", f.mapped ? "yes" : "no");
    printf("file_size: %llu\n", (unsigned long long)f.size);
    printf("alignment: %llu\n", (unsigned long long)f.alignment);
    printf("metadata_keys: %llu\n", (unsigned long long)f.kv_count);
    printf("map_time_ms: %.2f\n", f.map_s * 1000.0);
    printf("parse_time_ms: %.2f\n", f.parse_s * 1000.0);
    if (show_tensors) {
        printf("\ntensors (%llu):\n", (unsigned long long)f.tensor_count);
        printf("%-44s %-6s %-26s %14s\n", "name", "type", "shape", "bytes");
        for (uint64_t i = 0; i < f.tensor_count; i++) {
            const KataliGgufTensor *t = &f.tensors[i];
            char shape[64] = "";
            size_t off = 0;
            for (uint32_t d = 0; d < t->n_dims; d++) {
                int w = snprintf(shape + off, sizeof(shape) - off, "%s%llu",
                                 d ? "x" : "", (unsigned long long)t->dims[d]);
                if (w > 0) off += (size_t)w;
                if (off >= sizeof(shape)) break;
            }
            printf("%-44s %-6s %-26s %14llu%s\n", t->name,
                   katali_ggml_type_name(t->type), shape,
                   (unsigned long long)t->nbytes,
                   t->data ? "" : "  [UNSUPPORTED]");
        }
    }
    katali_gguf_close(&f);
    return 0;
}

static int cmd_validate(const char *path) {
    KataliGgufFile f;
    char err[256];
    int failures = 0;
    if (katali_gguf_open(&f, path, 1, err, sizeof(err)) != 0) {
        printf("FAIL  parse: %s\n", err);
        return 2;
    }
    printf("PASS  parse: %llu tensors, %llu metadata keys\n",
           (unsigned long long)f.tensor_count, (unsigned long long)f.kv_count);

    const char *arch = katali_gguf_get_str(&f, "general.architecture", "", NULL);
    printf("%s  architecture: %s\n", arch[0] ? "PASS" : "FAIL", arch[0] ? arch : "(missing)");
    if (!arch[0]) failures++;

    int dense = (strstr(arch, "moe") == NULL);
    if (!dense) { printf("FAIL  dense architecture required for Phase 1\n"); failures++; }
    if (strncmp(arch, "qwen", 4) != 0) {
        printf("FAIL  architecture is not a supported Qwen model\n");
        failures++;
    }

    int bad_types = 0;
    for (uint64_t i = 0; i < f.tensor_count; i++) {
        if (!f.tensors[i].data) {
            if (bad_types < 8)
                printf("FAIL  tensor '%s' uses unsupported type %s\n",
                       f.tensors[i].name, katali_ggml_type_name(f.tensors[i].type));
            bad_types++;
        }
    }
    if (bad_types) { printf("FAIL  %d tensor(s) unsupported\n", bad_types); failures++; }
    else printf("PASS  all tensor types supported\n");

    int64_t vocab = katali_gguf_array_len(&f, "tokenizer.ggml.tokens", NULL);
    if (vocab > 0) printf("PASS  tokenizer vocabulary: %lld tokens\n", (long long)vocab);
    else { printf("FAIL  tokenizer.ggml.tokens missing\n"); failures++; }

    const char *tmpl = katali_gguf_get_str(&f, "tokenizer.chat_template", NULL, NULL);
    printf("%s  chat template: %s\n", tmpl ? "PASS" : "WARN",
           tmpl ? "present" : "absent (raw prompting only)");

    katali_gguf_close(&f);
    printf("%s  %s\n", failures ? "INVALID" : "VALID", path);
    return failures ? 1 : 0;
}

/* ==================================================================== *
 * run / benchmark                                                       *
 * ==================================================================== */
typedef struct CliOpts {
    const char *prompt;
    int max_tokens;
    int ctx;
    int threads;
    int think;
    int temp_milli;
    int top_k;
    int top_p_milli;
    int rep_milli;
    int seed;
    int use_mmap;
    int iters;
} CliOpts;

static void opts_default(CliOpts *o) {
    memset(o, 0, sizeof(*o));
    o->max_tokens = 128;
    o->ctx = 0;
    o->threads = 0;
    o->think = 0;
    o->temp_milli = 0;
    o->top_p_milli = 0;
    o->rep_milli = 0;
    o->seed = 1;
    o->use_mmap = 1;
    o->iters = 200;
}

static int parse_opts(int argc, char **argv, int start, CliOpts *o) {
    for (int i = start; i < argc; i++) {
        const char *a = argv[i];
        if (strcmp(a, "--prompt") == 0 && i + 1 < argc) o->prompt = argv[++i];
        else if (strcmp(a, "--max-tokens") == 0 && i + 1 < argc) o->max_tokens = atoi(argv[++i]);
        else if (strcmp(a, "--ctx") == 0 && i + 1 < argc) o->ctx = atoi(argv[++i]);
        else if (strcmp(a, "--threads") == 0 && i + 1 < argc) o->threads = atoi(argv[++i]);
        else if (strcmp(a, "--temp") == 0 && i + 1 < argc) o->temp_milli = (int)(atof(argv[++i]) * 1000.0);
        else if (strcmp(a, "--top-k") == 0 && i + 1 < argc) o->top_k = atoi(argv[++i]);
        else if (strcmp(a, "--top-p") == 0 && i + 1 < argc) o->top_p_milli = (int)(atof(argv[++i]) * 1000.0);
        else if (strcmp(a, "--rep") == 0 && i + 1 < argc) o->rep_milli = (int)(atof(argv[++i]) * 1000.0);
        else if (strcmp(a, "--seed") == 0 && i + 1 < argc) o->seed = atoi(argv[++i]);
        else if (strcmp(a, "--iters") == 0 && i + 1 < argc) o->iters = atoi(argv[++i]);
        else if (strcmp(a, "--think") == 0) o->think = 1;
        else if (strcmp(a, "--no-think") == 0) o->think = 0;
        else if (strcmp(a, "--no-mmap") == 0) o->use_mmap = 0;
        else if (strcmp(a, "--tensors") == 0) { /* handled by inspect */ }
        else { fprintf(stderr, "katali-gguf: unknown option '%s'\n", a); return -1; }
    }
    return 0;
}

static void fill_backend_opts(KataliBackendOptions *bo, const CliOpts *o,
                              const char *path) {
    memset(bo, 0, sizeof(*bo));
    bo->model_path = path;
    bo->n_threads = o->threads;
    bo->context_length = o->ctx;
    bo->max_tokens = o->max_tokens;
    bo->use_mmap = o->use_mmap;
    bo->temperature_milli = o->temp_milli;
    bo->top_k = o->top_k;
    bo->top_p_milli = o->top_p_milli;
    bo->repetition_penalty_milli = o->rep_milli;
    bo->seed = o->seed;
    bo->think = o->think;
}

static double cli_now_s(void);

#ifdef _WIN32
static double cli_now_s(void) {
    static LARGE_INTEGER freq;
    static int have = 0;
    if (!have) { QueryPerformanceFrequency(&freq); have = 1; }
    LARGE_INTEGER c;
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)freq.QuadPart;
}
#else
#include <time.h>
static double cli_now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
#endif

static int cmd_run(const char *path, const CliOpts *o) {
    KataliBackendOptions bo;
    fill_backend_opts(&bo, o, path);
    const char *prompt = o->prompt ? o->prompt : "Hi";

    double t0 = cli_now_s();
    KataliBackend *b = katali_backend_create("gguf", &bo);
    double t_load = cli_now_s() - t0;
    if (!b) {
        fprintf(stderr, "katali-gguf: could not open %s with backend=gguf\n", path);
        return 2;
    }
    size_t cap = (size_t)1 << 20;
    char *out = (char *)malloc(cap);
    if (!out) { katali_backend_destroy(b); return 2; }

    double t1 = cli_now_s();
    int n = b->ops->generate(b, prompt, out, cap);
    double t_gen = cli_now_s() - t1;
    if (n < 0) {
        fprintf(stderr, "katali-gguf: generation failed (code %d)\n", n);
        free(out);
        katali_backend_destroy(b);
        return 2;
    }
    printf("%s\n", out);
    size_t resident = b->ops->resident_bytes ? b->ops->resident_bytes(b) : 0;
    fprintf(stderr,
            "[katali-gguf] backend=gguf load=%.3fs generate=%.3fs tokens=%d "
            "resident=%.1f MB\n",
            t_load, t_gen, n, (double)resident / (1024.0 * 1024.0));
    free(out);
    katali_backend_destroy(b);
    return 0;
}

static int cmd_bench(const char *path, const CliOpts *o) {
    KataliBackendOptions bo;
    fill_backend_opts(&bo, o, path);
    const char *prompt = o->prompt ? o->prompt : "Hi";

    KataliGgufModel m;
    char err[320];
    double t0 = cli_now_s();
    if (katali_gguf_model_open(&m, &bo, err, sizeof(err)) != 0) {
        fprintf(stderr, "katali-gguf: %s\n", err);
        return 2;
    }
    double t_open = cli_now_s() - t0;
    unsigned long long rss_before = proc_rss_bytes();

    size_t cap = (size_t)1 << 20;
    char *out = (char *)malloc(cap);
    if (!out) { katali_gguf_model_close(&m); return 2; }
    katali_ggml_prof_reset();
    katali_prof_reset();
    katali_gguf_sync_stats_reset();
    int n = katali_gguf_model_generate(&m, prompt, out, cap);
    unsigned long long rss_after = proc_rss_bytes();
    unsigned long long kcalls = 0, kbytes = 0, kflops = 0;
    double ksec = 0.0;
    katali_ggml_prof_stats(&kcalls, &ksec, &kbytes, &kflops);
    if (n < 0) {
        fprintf(stderr, "katali-gguf: generation failed (code %d)\n", n);
        free(out);
        katali_gguf_model_close(&m);
        return 2;
    }

    const KataliGgufStats *s = &m.stats;
    double decode = s->decode_s > 0 ? s->decode_s : 1e-9;
    printf("model_path: %s\n", path);
    printf("architecture: %s\n", m.arch);
    printf("quantization: %s\n", katali_ggml_type_name(m.embed->type));
    printf("threads: %d\n", m.n_threads > 0 ? m.n_threads : katali_gguf_get_threads());
    printf("context_length: %d\n", m.kv_cap);
    printf("file_size_bytes: %llu\n", (unsigned long long)m.file.size);
    printf("resident_bytes: %llu\n", (unsigned long long)katali_gguf_model_resident_bytes(&m));
    printf("process_rss_before_bytes: %llu\n", rss_before);
    printf("process_rss_after_bytes: %llu\n", rss_after);
    printf("load_s: %.4f\n", t_open);
    printf("map_s: %.4f\n", s->map_s);
    printf("parse_s: %.4f\n", s->parse_s);
    printf("kv_alloc_s: %.4f\n", s->kv_alloc_s);
    printf("tokenize_s: %.4f\n", s->tokenize_s);
    printf("prompt_tokens: %d\n", s->n_prompt);
    printf("batched_prefill: %d\n", m.last.batched_prefill);
    printf("prefill_s: %.4f\n", s->prefill_s);
    printf("generated_tokens: %d\n", s->n_gen);
    printf("decode_s: %.4f\n", s->decode_s);
    printf("decode_tokens_per_s: %.3f\n", (double)s->n_gen / decode);
    printf("ttft_s: %.4f\n", m.last.ttft_s);
    printf("total_s: %.4f\n", s->total_s);
    printf("simd: %s\n", katali_ggml_simd_name());
    printf("kernel_calls: %llu\n", kcalls);
    printf("kernel_s: %.4f\n", ksec);
    printf("kernel_gflop_per_s: %.2f\n", ksec > 0 ? (double)kflops / ksec / 1e9 : 0.0);
    printf("kernel_gbyte_per_s: %.2f\n", ksec > 0 ? (double)kbytes / ksec / 1e9 : 0.0);
    printf("kernel_share_pct: %.1f\n",
           s->total_s > 0 ? 100.0 * ksec / s->total_s : 0.0);

    /* Phase breakdown. SYNC is a subset of MATVEC (the caller's wait inside a
     * parallel dispatch), and NORMDQ is a subset of the non-kernel time. */
    printf("\nphase,seconds,calls,share_pct\n");
    for (int ph = 0; ph < KATALI_PHASE_COUNT; ph++) {
        if (ph == KATALI_PHASE_PREFILL) continue; /* reported separately */
        double ps = katali_prof_seconds(ph);
        unsigned long long pc = katali_prof_calls(ph);
        if (ps <= 0.0 && pc == 0) continue;
        printf("phase_%s_s: %.4f\n", katali_prof_name(ph), ps);
        printf("phase_%s_calls: %llu\n", katali_prof_name(ph), pc);
        printf("phase_%s_share_pct: %.1f\n", katali_prof_name(ph),
               s->total_s > 0 ? 100.0 * ps / s->total_s : 0.0);
        /* Bytes and achieved bandwidth, for phases that track traffic (the
         * attention K and V streams). */
        unsigned long long pb = katali_prof_bytes(ph);
        if (pb > 0) {
            printf("phase_%s_bytes: %llu\n", katali_prof_name(ph), pb);
            printf("phase_%s_gbyte_per_s: %.2f\n", katali_prof_name(ph),
                   ps > 0.0 ? (double)pb / ps / 1e9 : 0.0);
        }
    }
    printf("prefill_phase_s: %.4f\n", katali_prof_seconds(KATALI_PHASE_PREFILL));

    /* Kernel time by (role, quantization). The global kernel_* numbers sum every
     * matvec into one bucket; this separates the LM head (151936 rows, once per
     * token) from the per-layer projections and Q4_K from Q6_K within a role.
     * Roles are keyed on tensor identity by the model (m->out_w / m->tied), not on
     * tensor names, so a tied embedding/LM head is attributed correctly. */
    {
        int nroles = katali_ggml_prof_role_count();
        if (nroles > 0) {
            int idx[64];
            if (nroles > 64) nroles = 64;
            for (int i = 0; i < nroles; i++) idx[i] = i;
            /* simple insertion sort by descending seconds (n is tiny) */
            for (int i = 1; i < nroles; i++) {
                int k = idx[i];
                int j = i - 1;
                double ks = 0.0;
                katali_ggml_prof_role_get(k, NULL, NULL, NULL, &ks, NULL, NULL);
                while (j >= 0) {
                    double js = 0.0;
                    katali_ggml_prof_role_get(idx[j], NULL, NULL, NULL, &js, NULL, NULL);
                    if (js >= ks) break;
                    idx[j + 1] = idx[j];
                    j--;
                }
                idx[j + 1] = k;
            }
            printf("\nkernel_by_role (sorted by seconds)\n");
            printf("%-10s %-8s %-6s %8s %10s %8s %12s %9s %9s\n",
                   "role", "kind", "type", "calls", "seconds", "share%", "bytes",
                   "GB/s", "GF/s");
            for (int i = 0; i < nroles; i++) {
                int role = 0;
                uint32_t type = 0;
                unsigned long long calls = 0, bytes = 0, flops = 0;
                double secs = 0.0;
                katali_ggml_prof_role_get(idx[i], &role, &type, &calls, &secs,
                                          &bytes, &flops);
                /* A prefill matmul streams the weights once but does B tokens of
                 * arithmetic, so its GB/s is low by construction: read GF/s for
                 * prefill rows and GB/s for decode rows. */
                printf("%-10s %-8s %-6s %8llu %10.4f %8.1f %12llu %9.2f %9.1f\n",
                       katali_ggml_role_name(role),
                       katali_ggml_prof_role_is_prefill(idx[i]) ? "prefill" : "decode",
                       katali_ggml_type_name(type),
                       calls, secs,
                       s->total_s > 0 ? 100.0 * secs / s->total_s : 0.0,
                       bytes, secs > 0.0 ? (double)bytes / secs / 1e9 : 0.0,
                       secs > 0.0 ? (double)flops / secs / 1e9 : 0.0);
            }
        }
    }

    /* Optional dispatch breakdown (KATALI_GGUF_SYNC_STATS=1). Printed only when
     * the instrumentation is enabled and dispatches were recorded, so clean
     * runs produce byte-identical output to before. */
    if (katali_gguf_sync_stats_enabled()) {
        KataliSyncStats ss;
        katali_gguf_sync_stats_get(&ss);
        if (ss.dispatches > 0) {
            double d = (double)ss.dispatches;
            printf("\ndispatch_breakdown (us per dispatch, %llu dispatches)\n",
                   ss.dispatches);
            printf("dispatch_events_us: %.2f\n", 1e6 * ss.events_s / d);
            printf("dispatch_wall_us: %.2f\n", 1e6 * ss.wall_s / d);
            printf("dispatch_wake_max_us: %.2f\n", 1e6 * ss.wake_max_s / d);
            printf("dispatch_work_max_us: %.2f\n", 1e6 * ss.work_max_s / d);
            printf("dispatch_work_min_us: %.2f\n", 1e6 * ss.work_min_s / d);
            printf("dispatch_end_spread_us: %.2f\n", 1e6 * ss.end_spread_s / d);
            printf("dispatch_residual_us: %.2f\n", 1e6 * ss.residual_s / d);
            printf("dispatch_caller_wait_us: %.2f\n", 1e6 * ss.caller_wait_s / d);
        }
    }

    /* Two different memory numbers, labelled so they cannot be confused. */
    printf("memory_resident_bytes_nominal: %llu\n",
           (unsigned long long)katali_gguf_model_resident_bytes(&m));
    printf("memory_process_rss_bytes: %llu\n", rss_after);
    printf("memory_note: resident_bytes is a nominal allocation estimate "
           "(KV capacity + scratch); process_rss is the OS working set\n");
    printf("sample_output: %.120s%s\n", out, strlen(out) > 120 ? "..." : "");
    free(out);
    katali_gguf_model_close(&m);
    return 0;
}

static int cmd_tokenize(const char *path, const char *text, const char *grep,
                        int range_lo, int range_hi) {
    KataliGgufFile f;
    KataliGgufTokenizer tk;
    char err[256];
    if (katali_gguf_open(&f, path, 1, err, sizeof(err)) != 0) {
        fprintf(stderr, "katali-gguf: %s\n", err);
        return 2;
    }
    if (katali_gguf_tokenizer_init(&tk, &f, err, sizeof(err)) != 0) {
        fprintf(stderr, "katali-gguf: %s\n", err);
        katali_gguf_close(&f);
        return 2;
    }
    printf("vocab_size: %d\n", tk.vocab_size);
    printf("n_merges: %d\n", tk.n_merges);
    printf("bos_id: %d\n", tk.bos_id);
    printf("eos_id: %d\n", tk.eos_id);
    printf("pad_id: %d\n", tk.pad_id);
    printf("im_start_id: %d\n", tk.im_start_id);
    printf("im_end_id: %d\n", tk.im_end_id);
    printf("endoftext_id: %d\n", tk.endoftext_id);
    printf("think_open_id: %d\n", tk.think_open_id);
    printf("think_close_id: %d\n", tk.think_close_id);
    printf("add_bos: %d\n", tk.add_bos);
    printf("has_chat_template: %d\n", tk.has_chat_template);
    if (tk.has_chat_template && tk.chat_template) {
        /* Show the template region around the thinking switch, escaped. */
        const char *needle = "think";
        size_t nl = strlen(needle);
        for (size_t i = 0; i + nl <= tk.chat_template_len; i++) {
            if (memcmp(tk.chat_template + i, needle, nl) == 0) {
                size_t lo = i > 24 ? i - 24 : 0;
                size_t hi = i + nl + 24;
                if (hi > tk.chat_template_len) hi = tk.chat_template_len;
                printf("template@%llu hex:", (unsigned long long)i);
                for (size_t k = lo; k < hi; k++)
                    printf(" %02x", (unsigned char)tk.chat_template[k]);
                printf("\n");
                break;
            }
        }
    }

    if (range_lo >= 0 && range_hi >= range_lo) {
        if (range_hi >= tk.vocab_size) range_hi = tk.vocab_size - 1;
        for (int i = range_lo; i <= range_hi; i++) {
            size_t L = tk.tok_len[i];
            printf("  id=%d len=%llu type=%d hex=", i,
                   (unsigned long long)L, tk.tok_type ? tk.tok_type[i] : -1);
            for (size_t k = 0; k < L && k < 24; k++) printf("%02x ", tk.tok_bytes[i][k]);
            printf("| ");
            for (size_t k = 0; k < L && k < 48; k++) {
                unsigned char c = tk.tok_bytes[i][k];
                if (c >= 32 && c < 127) putchar(c);
                else putchar('.');
            }
            printf("\n");
        }
    }

    if (grep) {
        size_t gl = strlen(grep);
        printf("grep '%s':\n", grep);
        int shown = 0;
        for (int i = 0; i < tk.vocab_size && shown < 40; i++) {
            size_t L = tk.tok_len[i];
            if (L < gl) continue;
            for (size_t j = 0; j + gl <= L; j++) {
                if (memcmp(tk.tok_bytes[i] + j, grep, gl) == 0) {
                    printf("  id=%d len=%llu type=%d bytes=", i,
                           (unsigned long long)L, tk.tok_type ? tk.tok_type[i] : -1);
                    for (size_t k = 0; k < L && k < 32; k++) {
                        unsigned char c = tk.tok_bytes[i][k];
                        if (c >= 32 && c < 127) putchar(c);
                        else printf("\\x%02x", c);
                    }
                    printf("\n");
                    shown++;
                    break;
                }
            }
        }
        if (!shown) printf("  (no match)\n");
    }

    if (text) {
        int ids[1024];
        int n = katali_gguf_tokenizer_encode(&tk, text, ids, 1024);
        printf("text: %s\n", text);
        printf("token_count: %d\n", n);
        if (n >= 0) {
            printf("ids:");
            for (int i = 0; i < n && i < 64; i++) printf(" %d", ids[i]);
            printf("\n");
            char back[2048];
            size_t used = 0;
            for (int i = 0; i < n && used + 16 < sizeof(back); i++) {
                int w = katali_gguf_tokenizer_decode(&tk, ids[i], back + used, sizeof(back) - used - 1);
                if (w > 0) used += (size_t)w;
            }
            back[used] = '\0';
            printf("roundtrip: %s\n", back);
        }
    }
    katali_gguf_tokenizer_free(&tk);
    katali_gguf_close(&f);
    return 0;
}

/* ==================================================================== *
 * kernels: per-type matvec-row microbenchmark (scalar vs accelerated)   *
 * ==================================================================== */
static int cmd_kernels(const char *path, int iters) {
    if (iters <= 0) iters = 200;
    KataliGgufFile f;
    char err[256];
    if (katali_gguf_open(&f, path, 1, err, sizeof(err)) != 0) {
        fprintf(stderr, "katali-gguf: %s\n", err);
        return 2;
    }
    printf("file: %s\n", path);
    printf("simd: %s\n", katali_ggml_simd_name());
    printf("iters: %d\n\n", iters);

    enum { MAXK = 24 };
    struct KbBest { uint32_t type; const KataliGgufTensor *t; uint64_t bytes; };
    struct KbBest best[MAXK];
    int nbest = 0;
    for (uint64_t i = 0; i < f.tensor_count; i++) {
        const KataliGgufTensor *t = &f.tensors[i];
        if (!t->data || t->n_dims < 2) continue;
        uint64_t cols = t->dims[0];
        uint64_t bs = katali_ggml_type_block_size(t->type);
        if (bs == 0 || cols == 0 || cols % bs != 0) continue;
        int slot = -1;
        for (int k = 0; k < nbest; k++) if (best[k].type == t->type) { slot = k; break; }
        if (slot < 0) {
            if (nbest >= MAXK) continue;
            slot = nbest++;
            best[slot].type = t->type;
            best[slot].t = t;
            best[slot].bytes = 0;
        }
        if (t->nbytes > best[slot].bytes) { best[slot].t = t; best[slot].bytes = t->nbytes; }
    }
    for (int a = 0; a < nbest; a++)
        for (int b = a + 1; b < nbest; b++)
            if (best[b].bytes > best[a].bytes) {
                struct KbBest tmp = best[a]; best[a] = best[b]; best[b] = tmp;
            }

    printf("%-6s %-38s %7s %11s %11s %10s %10s %8s\n",
           "type", "tensor", "cols", "scalar_us", "simd_us", "scal_GB/s", "simd_GB/s", "speedup");
    for (int k = 0; k < nbest; k++) {
        const KataliGgufTensor *t = best[k].t;
        uint64_t cols = t->dims[0];
        uint64_t rowbytes = katali_ggml_row_bytes(t->type, cols);
        float *x = (float *)malloc((size_t)cols * sizeof(float));
        if (!x) { katali_gguf_close(&f); return 2; }
        for (uint64_t i = 0; i < cols; i++)
            x[i] = (float)(((i * 37) % 101) - 50) / 50.0f;

        volatile float sink = 0.0f;
        sink += katali_ggml_vec_dot_scalar(t->type, t->data, x, cols);
        sink += katali_ggml_vec_dot(t->type, t->data, x, cols);

        double ts0 = cli_now_s();
        for (int it = 0; it < iters; it++)
            sink += katali_ggml_vec_dot_scalar(t->type, t->data, x, cols);
        double ts1 = cli_now_s() - ts0;

        double tv0 = cli_now_s();
        for (int it = 0; it < iters; it++)
            sink += katali_ggml_vec_dot(t->type, t->data, x, cols);
        double tv1 = cli_now_s() - tv0;

        double us_scalar = ts1 / iters * 1e6;
        double us_simd = tv1 / iters * 1e6;
        double gbs_scalar = ts1 > 0 ? (double)rowbytes * iters / ts1 / 1e9 : 0.0;
        double gbs_simd = tv1 > 0 ? (double)rowbytes * iters / tv1 / 1e9 : 0.0;
        printf("%-6s %-38s %7llu %11.3f %11.3f %10.2f %10.2f %7.2fx\n",
               katali_ggml_type_name(t->type), t->name,
               (unsigned long long)cols, us_scalar, us_simd,
               gbs_scalar, gbs_simd, us_simd > 0 ? us_scalar / us_simd : 0.0);
        free(x);
        (void)sink;
    }
    katali_gguf_close(&f);
    return 0;
}

/* ==================================================================== *
 * chat: persistent session through the session API (model loaded once)  *
 * ==================================================================== */
static int cmd_chat(const char *path, const CliOpts *o) {
    KataliBackendOptions bo;
    fill_backend_opts(&bo, o, path);
    char err[320];
    double t0 = cli_now_s();
    KataliSession *sess = katali_session_open("gguf", &bo, err, sizeof(err));
    double t_load = cli_now_s() - t0;
    if (!sess) {
        fprintf(stderr, "katali-gguf: %s\n", err);
        return 2;
    }
    size_t cap = (size_t)1 << 20;
    char *out = (char *)malloc(cap);
    if (!out) { katali_session_close(sess); return 2; }
    const KataliSessionStats *st = katali_session_stats(sess);
    unsigned long long rss0 = proc_rss_bytes();
    fprintf(stderr,
            "[katali-gguf] backend=%s session load=%.3fs resident=%.1f MB "
            "rss=%.1f MB threads=%d ctx=%d max_tokens=%d batched_prefill=auto\n",
            katali_session_backend(sess), t_load,
            (double)katali_session_resident_bytes(sess) / (1024.0 * 1024.0),
            (double)rss0 / (1024.0 * 1024.0),
            bo.n_threads > 0 ? bo.n_threads : katali_gguf_get_threads(),
            bo.context_length, bo.max_tokens);
    printf("Katali-GGUF chat (one question per line; /quit or EOF exits)\n");

    char line[4096];
    unsigned long long rss_prev = rss0;
    while (fgets(line, sizeof(line), stdin)) {
        size_t L = strlen(line);
        while (L && (line[L - 1] == '\n' || line[L - 1] == '\r')) line[--L] = '\0';
        if (L == 0) continue;
        if (strcmp(line, "/quit") == 0 || strcmp(line, "/exit") == 0 ||
            strcmp(line, "exit") == 0 || strcmp(line, "quit") == 0 ||
            strcmp(line, "bye") == 0) break;
        if (line[0] == '/') {
            fprintf(stderr, "  (unknown command '%s'; /quit exits)\n", line);
            continue;
        }

        katali_ggml_prof_reset();
        int n = katali_session_ask(sess, line, out, cap);
        if (n < 0) {
            printf("[generation failed: code %d]\n", n);
            fflush(stdout);
            continue;
        }
        printf("%s\n", out);
        fflush(stdout);
        unsigned long long rss = proc_rss_bytes();
        fprintf(stderr,
                "[q%d] total=%.2fs ttft=%.2fs prefill=%.2fs decode=%.2fs "
                "tok=%d tok/s=%.2f reset=%.5fs batch=%d rss=%.1fMB (%+.1fMB)\n",
                st->questions, st->last_total_s, st->last_ttft_s,
                st->last_prefill_s, st->last_decode_s, n, st->last_decode_tps,
                st->last_reset_s, st->last_batched_prefill,
                (double)rss / (1024.0 * 1024.0),
                (double)((long long)rss - (long long)rss_prev) / (1024.0 * 1024.0));
        rss_prev = rss;
    }

    unsigned long long rss1 = proc_rss_bytes();
    fprintf(stderr,
            "[katali-gguf] session summary: backend=%s loads=%d questions=%d "
            "load=%.3fs first_q=%.2fs later_avg=%.2fs later_total=%.2fs "
            "kv_resets=%d reset_cost=%.5fs tokens=%lld gen_total=%.3fs "
            "decode_avg=%.2f tok/s resident=%.1fMB rss=%.1fMB (%+.1fMB)\n",
            katali_session_backend(sess), st->model_loads, st->questions,
            st->load_s, st->first_question_s, st->later_avg_s, st->later_total_s,
            st->kv_resets, st->last_reset_s, st->total_tokens,
            st->total_generate_s, st->decode_avg_tps,
            (double)katali_session_resident_bytes(sess) / (1024.0 * 1024.0),
            (double)rss1 / (1024.0 * 1024.0),
            (double)((long long)rss1 - (long long)rss0) / (1024.0 * 1024.0));
    free(out);
    katali_session_close(sess);
    return 0;
}

/* ==================================================================== *
 * bench-session: first vs later question latency through one process    *
 * ==================================================================== */
static int cmd_bench_session(const char *path, const CliOpts *o,
                             char **qs, int nq) {
    KataliBackendOptions bo;
    fill_backend_opts(&bo, o, path);
    char err[320];
    double t0 = cli_now_s();
    KataliSession *sess = katali_session_open("gguf", &bo, err, sizeof(err));
    double t_load_wall = cli_now_s() - t0;
    if (!sess) {
        fprintf(stderr, "katali-gguf: %s\n", err);
        return 2;
    }
    size_t cap = (size_t)1 << 20;
    char *out = (char *)malloc(cap);
    if (!out) { katali_session_close(sess); return 2; }
    const KataliSessionStats *st = katali_session_stats(sess);
    unsigned long long rss0 = proc_rss_bytes();

    printf("backend: %s\n", katali_session_backend(sess));
    printf("model: %s\n", path);
    printf("threads: %d\n", bo.n_threads > 0 ? bo.n_threads : katali_gguf_get_threads());
    printf("context_length: %d\n", bo.context_length);
    printf("max_tokens: %d\n", bo.max_tokens);
    printf("load_s: %.4f\n", t_load_wall);
    printf("resident_bytes: %llu\n",
           (unsigned long long)katali_session_resident_bytes(sess));
    printf("rss_before_bytes: %llu\n", rss0);
    printf("\n%-3s %-40s %7s %7s %7s %7s %7s %5s %8s %9s\n",
           "#", "question", "total", "ttft", "pref", "decode", "tok/s", "tok",
           "reset_s", "rss_MB");

    double first_total = 0.0, later_sum = 0.0;
    int later_n = 0;
    for (int i = 0; i < nq; i++) {
        int n = katali_session_ask(sess, qs[i], out, cap);
        unsigned long long rss = proc_rss_bytes();
        if (n < 0) {
            printf("%-3d %-40.40s FAILED (code %d)\n", i + 1, qs[i], n);
            continue;
        }
        if (i == 0) first_total = st->last_total_s;
        else { later_sum += st->last_total_s; later_n++; }
        printf("%-3d %-40.40s %7.2f %7.2f %7.2f %7.2f %7.2f %5d %8.5f %9.1f\n",
               i + 1, qs[i], st->last_total_s, st->last_ttft_s,
               st->last_prefill_s, st->last_decode_s, st->last_decode_tps, n,
               st->last_reset_s, (double)rss / (1024.0 * 1024.0));
    }
    unsigned long long rss1 = proc_rss_bytes();

    printf("\n--- session summary ---\n");
    printf("model_loads: %d\n", st->model_loads);
    printf("questions: %d\n", st->questions);
    printf("load_s: %.4f\n", t_load_wall);
    printf("first_question_total_s: %.4f\n", first_total);
    printf("later_questions_avg_s: %.4f\n",
           later_n ? later_sum / (double)later_n : 0.0);
    printf("later_questions_total_s: %.4f\n", later_sum);
    printf("load_excluded_from_later_questions: yes\n");
    printf("kv_resets: %d\n", st->kv_resets);
    printf("kv_reset_cost_s: %.6f\n", st->last_reset_s);
    printf("batched_prefill_last: %d\n", st->last_batched_prefill);
    printf("generated_tokens: %lld\n", st->total_tokens);
    printf("generation_total_s: %.4f\n", st->total_generate_s);
    printf("generation_avg_tokens_per_s: %.3f\n",
           st->total_generate_s > 0
               ? (double)st->total_tokens / st->total_generate_s : 0.0);
    printf("decode_avg_tokens_per_s: %.3f\n", st->decode_avg_tps);
    printf("rss_after_bytes: %llu\n", rss1);
    printf("rss_delta_bytes: %lld\n", (long long)rss1 - (long long)rss0);
    printf("resident_bytes: %llu\n",
           (unsigned long long)katali_session_resident_bytes(sess));
    printf("last_answer: %.160s\n", out);
    free(out);
    katali_session_close(sess);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 3) { usage(); return 1; }
    const char *cmd = argv[1];
    const char *path = argv[2];

    if (strcmp(cmd, "inspect") == 0) {
        int show = 0;
        for (int i = 3; i < argc; i++) if (strcmp(argv[i], "--tensors") == 0) show = 1;
        return cmd_inspect(path, show);
    }
    if (strcmp(cmd, "validate") == 0) return cmd_validate(path);
    if (strcmp(cmd, "serve") == 0) {
        int port = 8080, threads = 0;
        for (int i = 3; i < argc; i++)
            if (strcmp(argv[i], "--port") == 0 && i + 1 < argc) port = atoi(argv[++i]);
            else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) threads = atoi(argv[++i]);
        if (port <= 0 || port > 65535) {
            printf("serve: --port must be 1..65535\n");
            return 1;
        }
        char err[256];
        err[0] = '\0';
        if (katali_gguf_serve(path, port, threads, err, sizeof(err)) != 0) {
            printf("serve: %s\n", err);
            return 1;
        }
        return 0;
    }
    if (strcmp(cmd, "tokenize") == 0) {
        const char *text = NULL;
        const char *grep = NULL;
        int lo = -1, hi = -1;
        for (int i = 3; i < argc; i++) {
            if (strcmp(argv[i], "--text") == 0 && i + 1 < argc) text = argv[++i];
            else if (strcmp(argv[i], "--grep") == 0 && i + 1 < argc) grep = argv[++i];
            else if (strcmp(argv[i], "--range") == 0 && i + 2 < argc) { lo = atoi(argv[++i]); hi = atoi(argv[++i]); }
        }
        return cmd_tokenize(path, text, grep, lo, hi);
    }

    CliOpts o;
    opts_default(&o);
    /* bench-session runs several questions, so default to a shorter cap. */
    if (strcmp(cmd, "bench-session") == 0) o.max_tokens = 48;
    if (parse_opts(argc, argv, 3, &o) != 0) return 1;

    if (strcmp(cmd, "run") == 0) return cmd_run(path, &o);
    if (strcmp(cmd, "benchmark") == 0) return cmd_bench(path, &o);
    if (strcmp(cmd, "kernels") == 0) return cmd_kernels(path, o.iters);
    if (strcmp(cmd, "chat") == 0) return cmd_chat(path, &o);
    if (strcmp(cmd, "bench-session") == 0) {
        char *qs[16];
        int nq = 0;
        for (int i = 3; i < argc; i++)
            if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc && nq < 16)
                qs[nq++] = argv[++i];
        if (nq == 0) {
            static char *defs[5] = {
                "What is the capital of France?",
                "What is 12 times 7?",
                "Explain what a loop does in Python.",
                "What is the capital of Japan?",
                "What is 2 + 2?"
            };
            for (int i = 0; i < 5; i++) qs[nq++] = defs[i];
        }
        return cmd_bench_session(path, &o, qs, nq);
    }

    usage();
    return 1;
}



