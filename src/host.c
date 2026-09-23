#include "host.h"
#include "platform.h"
#include "forward.h"
#include "katali_cuda.h"
#include "katali_gguf_dtype.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>


/* GGUF mmap FillFn / ResolveFn: key = expert*3+which.
 * Private mode: memcpy into owned slot. Mmap mode: return pointer into map. */
typedef struct HostEcacheFillCtx {
    HostModel *m;
} HostEcacheFillCtx;

static int host_expert_span_ptr(HostModel *m, int layer, int key,
                                const uint8_t **out_ptr, size_t *out_len) {
    if (!m || !out_ptr || !out_len) return -1;
    if (layer < 0 || layer >= m->moe.n_layers) return -1;
    int expert = key / 3;
    int which = key % 3;
    if (which < 0 || which > 2 || expert < 0 || expert >= m->moe.n_experts) return -1;
    MoeLayerTensors *L = &m->moe.layers[layer];
    const KataliGgufTensor *ts[3] = { L->gate_exps, L->up_exps, L->down_exps };
    uint64_t rel = 0, len = 0;
    if (moe_expert_span(L, which, expert, &rel, &len) != KATALI_OK) return -1;
    if (!ts[which] || !ts[which]->data || len == 0) return -1;
    *out_ptr = ts[which]->data + rel;
    *out_len = (size_t)len;
    return 0;
}


/* Resolve hot-expert freq file path (katali2: <model_dir>/hot_experts.kcache).
 * Lab (single GGUF file): KATALI_EC_FREQ if set, else <gguf_dir>/hot_experts.kcache. */
static void host_ecache_freq_path(const HostModel *m, char *out, size_t cap) {
    const char *env = getenv("KATALI_EC_FREQ");
    if (env && env[0]) {
        snprintf(out, cap, "%s", env);
        return;
    }
    /* dirname of gguf_path */
    const char *path = m->path;
    const char *slash = strrchr(path, '/');
#ifdef _WIN32
    const char *bslash = strrchr(path, '\\');
    if (!slash || (bslash && bslash > slash)) slash = bslash;
#endif
    if (slash && slash > path) {
        size_t dlen = (size_t)(slash - path);
        if (dlen + 1 + sizeof("hot_experts.kcache") < cap) {
            memcpy(out, path, dlen);
            out[dlen] = '\0';
#ifdef _WIN32
            snprintf(out + dlen, cap - dlen, "\\hot_experts.kcache");
#else
            snprintf(out + dlen, cap - dlen, "/hot_experts.kcache");
#endif
            return;
        }
    }
    snprintf(out, cap, "hot_experts.kcache");
}

static int host_ecache_fill(void *userdata, int layer, int key,
                            uint8_t *dest, size_t nbytes) {
    HostEcacheFillCtx *ctx = (HostEcacheFillCtx *)userdata;
    if (!ctx || !ctx->m || !dest || nbytes == 0) return -1;
    const uint8_t *src = NULL;
    size_t len = 0;
    if (host_expert_span_ptr(ctx->m, layer, key, &src, &len) != 0) return -1;
    size_t n = len < nbytes ? len : nbytes;
    memcpy(dest, src, n);
    if (n < nbytes) memset(dest + n, 0, nbytes - n);
    return 0;
}

static int host_ecache_resolve(void *userdata, int layer, int key,
                               const uint8_t **out_ptr, size_t *out_nbytes) {
    HostEcacheFillCtx *ctx = (HostEcacheFillCtx *)userdata;
    if (!ctx || !ctx->m) return -1;
    return host_expert_span_ptr(ctx->m, layer, key, out_ptr, out_nbytes);
}

static uint64_t host_ecache_offset(void *userdata, int layer, int key) {
    HostEcacheFillCtx *ctx = (HostEcacheFillCtx *)userdata;
    if (!ctx || !ctx->m) return 0;
    HostModel *m = ctx->m;
    if (layer < 0 || layer >= m->moe.n_layers) return 0;
    int expert = key / 3;
    int which = key % 3;
    if (which < 0 || which > 2 || expert < 0) return 0;
    MoeLayerTensors *L = &m->moe.layers[layer];
    const KataliGgufTensor *ts[3] = { L->gate_exps, L->up_exps, L->down_exps };
    uint64_t rel = 0, len = 0;
    if (moe_expert_span(L, which, expert, &rel, &len) != KATALI_OK) return 0;
    if (!ts[which] || !ts[which]->data || !m->gguf.map) return 0;
    /* Absolute file offset of this weight span (for ensure_many sort). */
    return (uint64_t)(ts[which]->data - m->gguf.map) + rel;
}

/* Port #6: one contiguous GGUF mmap span → split to dests (private), or
 * touch-only when dests[i]==NULL (mmap-resident; caller resolves pointers). */
static int host_ecache_range_fill(void *userdata, const int *layers,
                                  const int *expert_ids,
                                  const uint64_t *offsets,
                                  const size_t *lengths,
                                  uint8_t *const *dests, int n) {
    HostEcacheFillCtx *ctx = (HostEcacheFillCtx *)userdata;
    if (!ctx || !ctx->m || !offsets || !lengths || !dests || n <= 0) return -1;
    HostModel *m = ctx->m;
    if (!m->gguf.map) return -1;
    uint64_t lo = offsets[0], hi = offsets[0] + (uint64_t)lengths[0];
    for (int i = 1; i < n; i++) {
        if (offsets[i] < lo) lo = offsets[i];
        if (offsets[i] + (uint64_t)lengths[i] > hi)
            hi = offsets[i] + (uint64_t)lengths[i];
    }
    if (hi <= lo || hi > m->gguf.size) return -1;
    size_t span = (size_t)(hi - lo);
    const uint8_t *base = m->gguf.map + (size_t)lo;

    /* Single sequential touch/fault of the coalesced run. */
    {
        volatile uint8_t sink = 0;
        sink ^= base[0];
        if (span > 1) sink ^= base[span - 1];
        for (size_t off = 4096; off < span; off += 4096)
            sink ^= base[off];
        (void)sink;
    }

    for (int i = 0; i < n; i++) {
        if (!dests[i] || lengths[i] == 0) continue;
        uint64_t rel = offsets[i] - lo;
        if (rel + lengths[i] > span) return -1;
        memcpy(dests[i], base + (size_t)rel, lengths[i]);
    }
    (void)layers;
    (void)expert_ids;
    return 0;
}

static size_t host_ecache_block_bytes(const HostModel *m) {
    size_t block = 0;
    for (int i = 0; i < m->moe.n_layers; i++) {
        const MoeLayerTensors *L = &m->moe.layers[i];
        if (L->expert_bytes_gate > block) block = (size_t)L->expert_bytes_gate;
        if (L->expert_bytes_up > block) block = (size_t)L->expert_bytes_up;
        if (L->expert_bytes_down > block) block = (size_t)L->expert_bytes_down;
    }
    if (block == 0) block = 1;
    return block;
}

int host_open(HostModel *m, const char *gguf_path, size_t ecache_bytes, int pin_frac) {
    memset(m, 0, sizeof(*m));
    snprintf(m->path, sizeof(m->path), "%s", gguf_path);
    int rc = katali_gguf_open(&m->gguf, gguf_path, 1, m->err, sizeof(m->err));
    if (rc != 0) return KATALI_ERR_IO;
    katali_arch_from_gguf(&m->gguf, &m->arch);
    rc = moe_index_build(&m->moe, &m->gguf, &m->arch);
    if (rc != KATALI_OK) { host_close(m); return rc; }
    if (ecache_bytes == 0) {
        uint64_t avail = katali_ram_avail_bytes();
        ecache_bytes = (size_t)(avail > (4ull << 30) ? (avail / 2) : (1ull << 30));
    }
    {
        size_t block = host_ecache_block_bytes(m);
        size_t cap = ecache_bytes / block;
        if (cap < 8) cap = 8;
        /* Three keys per expert (gate/up/down); allow more slots than katali2's 4096. */
        if (cap > 12288) cap = 12288;
        int want_mmap = 0;
        {
            const char *em = getenv("KATALI_ECACHE_MMAP");
            if (em && (*em == '1' || *em == 'y' || *em == 'Y' ||
                       *em == 't' || *em == 'T'))
                want_mmap = 1;
        }
        rc = ecache_init_ex(&m->ecache, cap, block, want_mmap);
        if (rc == KATALI_ERR_NOMEM && !want_mmap) {
            /* Hard Job Object / low commit: private slot malloc double-charges. */
            fprintf(stderr,
                    "phase: ecache private alloc OOM; falling back to mmap-resident\n");
            rc = ecache_init_ex(&m->ecache, cap, block, 1);
            want_mmap = 1;
        }
        if (rc != KATALI_OK) { host_close(m); return rc; }
        ecache_set_pin_frac(&m->ecache, pin_frac ? pin_frac : 25);
        ecache_bind_file(&m->ecache, m->gguf.map, m->gguf.size);
        if (!getenv("KATALI_NO_ECACHE")) {
            HostEcacheFillCtx *fctx = (HostEcacheFillCtx *)malloc(sizeof(HostEcacheFillCtx));
            if (fctx) {
                fctx->m = m;
                m->ecache_fill_ud = fctx;
                ecache_set_resolve_fn(&m->ecache, host_ecache_resolve);
                ecache_set_offset_fn(&m->ecache, host_ecache_offset);
                ecache_set_range_fill(&m->ecache, host_ecache_range_fill, fctx);
                int nw = 8;
                const char *env = getenv("KATALI_EC_WORKERS");
                if (env && *env) {
                    int v = atoi(env);
                    if (v > 0) nw = v;
                }
                if (ecache_start_workers(&m->ecache, nw, host_ecache_fill, fctx) == KATALI_OK) {
                    {
                        const char *re = getenv("KATALI_EC_RANGE_READ");
                        int ron = (re && re[0] && re[0] != '0');
                        fprintf(stderr,
                                "phase: ecache cap=%.2f GiB slots=%zu block=%.2f KiB "
                                "pin=%d%% workers=%d residency=%s range_read=%s\n",
                                (double)(cap * block) / (1024.0 * 1024.0 * 1024.0),
                                cap, (double)block / 1024.0,
                                pin_frac ? pin_frac : 25, m->ecache.n_workers,
                                m->ecache.mmap_mode ? "mmap" : "private",
                                ron ? "ON" : "OFF");
                    }

                /* Port #4: restore hot-expert freq + optional async warmup. */
                {
                    char hotp[1100];
                    host_ecache_freq_path(m, hotp, sizeof(hotp));
                    int nhot = ecache_freq_load(&m->ecache, hotp);
                    if (nhot > 0)
                        fprintf(stderr, "phase: ecache freq_load %d entries (%s)\n",
                                nhot, hotp);
                    else
                        fprintf(stderr, "phase: ecache freq_load: none yet (%s)\n",
                                hotp);
                    const char *wenv = getenv("KATALI_EC_WARMUP");
                    int do_warm = !(wenv && wenv[0] == '0');
                    if (do_warm && nhot > 0 && m->ecache.n_workers > 0) {
                        int nq = ecache_prefetch_hottest(&m->ecache, m->ecache.capacity);
                        fprintf(stderr,
                                "phase: ecache prefetch_hottest queued %d (async)\n",
                                nq);
                    }
                }
                } else {
                    free(fctx);
                    m->ecache_fill_ud = NULL;
                    fprintf(stderr, "phase: ecache workers failed; sync get only "
                            "(residency=%s)\n",
                            m->ecache.mmap_mode ? "mmap" : "private");
                }
            }
        } else {
            fprintf(stderr, "phase: ecache disabled (KATALI_NO_ECACHE)\n");
        }
    }

    if (katali_gguf_tokenizer_init(&m->tok, &m->gguf, m->err, sizeof(m->err)) == 0) {
        m->tok_ok = 1;
        if (m->arch.vocab <= 0) m->arch.vocab = m->tok.vocab_size;
    }
    katali_gguf_sampler_init(&m->samp);

    rc = host_state_alloc(m, 2048);
    if (rc != KATALI_OK) { host_close(m); return rc; }

    /* Prior-token top-k routes for speculative prefetch (port #2). */
    {
        int topk = m->arch.n_experts_active > 0 ? m->arch.n_experts_active : 8;
        int nL = m->moe.n_layers > 0 ? m->moe.n_layers : m->arch.n_layers;
        if (nL < 1) nL = 1;
        m->last_topk = topk;
        m->last_routes_valid = 0;
        m->last_eids = (int *)calloc((size_t)nL * (size_t)topk, sizeof(int));
        if (m->last_eids) {
            for (int i = 0; i < nL * topk; i++) m->last_eids[i] = -1;
        } else {
            m->last_topk = 0;
        }
    }

    /*
     * Optional CUDA tier. Detection happens once and is NEVER fatal: a machine
     * with no NVIDIA GPU, no driver, or no CUDA runtime keeps running the
     * existing CPU + system RAM + SSD path. Disable with KATALI_CUDA=0.
     * The backend DLL is loaded here, before any placement decision is made, so
     * the elastic scheduler can see the real VRAM budget. CPU/GPU dispatch is
     * wired separately and defaults to CPU.
     */
    fprintf(stderr, "phase: cuda %s (%s)\n",
            katali_cuda_available() ? "available" : "unavailable",
            katali_cuda_status());

    /*
     * Phase 8 EXPERIMENT: clock keep-warm. Off by default; set
     * KATALI_CUDA_KEEPWARM=<ms> to schedule one tiny kernel every <ms> and test
     * whether that alone prevents the memory clock collapsing to 405 MHz.
     */
    if (katali_cuda_available()) {
        const char *kw = getenv("KATALI_CUDA_KEEPWARM");
        const int kw_ms = (kw && *kw) ? atoi(kw) : 0;
        if (kw_ms > 0) {
            if (katali_cuda_keepwarm(kw_ms) == KATALI_OK)
                fprintf(stderr, "phase: cuda keepwarm ON every %d ms "
                                "(experiment)\n", kw_ms);
            else
                fprintf(stderr, "phase: cuda keepwarm FAILED to start\n");
        }
    }

    /*
     * Optional VRAM expert tier (third residency level).
     *
     * DEFAULT OFF, and that is a MEASURED decision, not caution. With the tier
     * on, the kernels are individually 6.9-10.9x faster than the CPU on
     * DRAM-bound expert traffic (see `cuda-bench`), but the integration measured
     * 2.14 tok/s decode versus 3.11 tok/s CPU-only on the 35B. The cause is CUDA
     * API latency, not compute: ~320 experts/token x ~8 launch/sync/copy calls
     * each is ~2560 round trips per token, and each WDDM round trip costs tens
     * of microseconds -> 150-250 ms/token of overhead against ~32 ms of real GPU
     * work. Until the whole layer's MoE is fused into a handful of launches
     * (one call per layer instead of one per expert), enabling this by default
     * would be a regression.
     *
     * Enable explicitly with KATALI_CUDA_MOE=1 (or set a budget with
     * KATALI_VRAM_GB) once the layer-level fusion lands.
     */
    if (katali_cuda_available() && m->arch.hidden > 0 && m->arch.moe_intermediate > 0) {
        const char *on = getenv("KATALI_CUDA_MOE");
        const char *gb = getenv("KATALI_VRAM_GB");
        int want = (on && (on[0] == '1' || on[0] == 'y' || on[0] == 'Y')) ||
                   (gb && gb[0] && gb[0] != '0');
        if (!want) {
            fprintf(stderr, "phase: vram tier OFF (opt-in via KATALI_CUDA_MOE=1). "
                            "Still slower than CPU end-to-end: the measured cause is "
                            "the GPU staying at its idle clock state because inference "
                            "is a ~6%% duty cycle - see docs/CUDA_PLAN.md 14.5\n");
        } else if (vram_cache_open(&m->vram, 0) != KATALI_OK) {
            fprintf(stderr, "phase: vram tier OFF (no usable budget; "
                            "set KATALI_VRAM_GB to override)\n");
        } else {
            size_t Hf = (size_t)m->arch.hidden * sizeof(float);
            size_t Ff = (size_t)m->arch.moe_intermediate * sizeof(float);
            int ok = 1;
            if (katali_cuda_malloc(&m->vram_x,  Hf) != KATALI_OK) ok = 0;
            if (ok && katali_cuda_malloc(&m->vram_gy, Ff) != KATALI_OK) ok = 0;
            if (ok && katali_cuda_malloc(&m->vram_uy, Ff) != KATALI_OK) ok = 0;
            if (ok && katali_cuda_malloc(&m->vram_iy, Ff) != KATALI_OK) ok = 0;
            if (ok && katali_cuda_malloc(&m->vram_dy, Hf) != KATALI_OK) ok = 0;
            if (ok) {
                m->vram_ready = 1;
                size_t used = 0, budget = 0, nent = 0;
                uint64_t hh = 0, mm = 0, ee = 0, uu = 0;
                vram_cache_stats(&m->vram, &hh, &mm, &ee, &uu, &used, &budget, &nent);
                fprintf(stderr,
                        "phase: vram tier ON budget=%.2f GiB activations=device; "
                        "expert hot-set cached in VRAM\n",
                        (double)budget / (1024.0 * 1024.0 * 1024.0));
            } else {
                katali_cuda_free(m->vram_x);  m->vram_x  = NULL;
                katali_cuda_free(m->vram_gy); m->vram_gy = NULL;
                katali_cuda_free(m->vram_uy); m->vram_uy = NULL;
                katali_cuda_free(m->vram_iy); m->vram_iy = NULL;
                katali_cuda_free(m->vram_dy); m->vram_dy = NULL;
                vram_cache_close(&m->vram);
                fprintf(stderr, "phase: vram tier OFF (device scratch alloc failed)\n");
            }
        }
    }

    m->opened = 1;
    return KATALI_OK;
}

void host_close(HostModel *m) {
    if (!m) return;
    if (m->vram_ready) {
        size_t used = 0, budget = 0, nent = 0;
        uint64_t hits = 0, misses = 0, evict = 0, ups = 0;
        vram_cache_stats(&m->vram, &hits, &misses, &evict, &ups,
                         &used, &budget, &nent);
        fprintf(stderr,
                "vram: hits=%llu misses=%llu evictions=%llu uploads=%llu "
                "resident=%zu used=%.2f/%.2f GiB gpu_fallbacks=%d\n",
                (unsigned long long)hits, (unsigned long long)misses,
                (unsigned long long)evict, (unsigned long long)ups, nent,
                (double)used / (1024.0 * 1024.0 * 1024.0),
                (double)budget / (1024.0 * 1024.0 * 1024.0),
                m->vram_fallbacks);
    }
    katali_cuda_free(m->vram_x);  m->vram_x  = NULL;
    katali_cuda_free(m->vram_gy); m->vram_gy = NULL;
    katali_cuda_free(m->vram_uy); m->vram_uy = NULL;
    katali_cuda_free(m->vram_iy); m->vram_iy = NULL;
    katali_cuda_free(m->vram_dy); m->vram_dy = NULL;
    vram_cache_close(&m->vram);
    m->vram_ready = 0;
    host_state_free(m);
    free(m->last_eids);
    m->last_eids = NULL;
    m->last_topk = 0;
    m->last_routes_valid = 0;
    if (m->tok_ok) katali_gguf_tokenizer_free(&m->tok);
    katali_gguf_sampler_free(&m->samp);
        /* Port #4: persist hot-expert freq (katali2 hot_experts.kcache). */
    if (m->opened && m->ecache.sync && m->ecache.n_workers >= 0) {
        char hotp[1100];
        host_ecache_freq_path(m, hotp, sizeof(hotp));
        int ns = ecache_freq_save(&m->ecache, hotp);
        if (ns >= 0)
            fprintf(stderr, "phase: ecache freq_save %d entries (%s)\n", ns, hotp);
    }
    ecache_stop_workers(&m->ecache);
    ecache_free(&m->ecache);
    free(m->ecache_fill_ud);
    m->ecache_fill_ud = NULL;
    moe_index_free(&m->moe);
    katali_arch_free(&m->arch);
    if (m->gguf.ok) katali_gguf_close(&m->gguf);
    memset(m, 0, sizeof(*m));
}

int host_probe_expert(HostModel *m, int layer, int expert) {
    if (!m || !m->opened) return KATALI_ERR;
    if (layer < 0 || layer >= m->moe.n_layers) return KATALI_ERR;
    if (expert < 0 || expert >= m->moe.n_experts) return KATALI_ERR;
    MoeLayerTensors *L = &m->moe.layers[layer];
    if (!L->gate_exps || !L->up_exps || !L->down_exps) {
        fprintf(stderr, "layer %d missing expert tensors\n", layer);
        return KATALI_ERR;
    }
    const char *names[3] = {"gate", "up", "down"};
    const KataliGgufTensor *ts[3] = { L->gate_exps, L->up_exps, L->down_exps };
    for (int w = 0; w < 3; w++) {
        uint64_t rel = 0, len = 0;
        if (moe_expert_span(L, w, expert, &rel, &len) != KATALI_OK) return KATALI_ERR;
        const uint8_t *src = ts[w]->data + rel;
        const uint8_t *p = ecache_get(&m->ecache, layer, expert * 3 + w, src, len);
        if (!p) {
            fprintf(stderr, "ecache_get failed layer=%d expert=%d which=%s\n",
                    layer, expert, names[w]);
            return KATALI_ERR;
        }
        printf("expert L%d E%d %s: src=%p len=%llu type=%s ecache_ok\n",
               layer, expert, names[w],
               (const void *)src, (unsigned long long)len,
               katali_ggml_type_name(ts[w]->type));
    }
    size_t used = 0, cap = 0; int n = 0;
    ecache_stats(&m->ecache, &used, &cap, &n);
    printf("ecache after probe: used=%.2f MiB resident=%d\n",
           (double)used / (1024.0 * 1024.0), n);
    return KATALI_OK;
}

static int append_id(int *ids, int at, int cap, int id) {
    if (at < 0 || at >= cap) return -1;
    ids[at] = id;
    return at + 1;
}

static int append_text(const KataliGgufTokenizer *tk, const char *s,
                       int *ids, int at, int cap) {
    int tmp[2048];
    int n = katali_gguf_tokenizer_encode(tk, s, tmp, 2048);
    if (n < 0) return -1;
    for (int i = 0; i < n; i++) {
        if (at >= cap) return -1;
        ids[at++] = tmp[i];
    }
    return at;
}


static int env_flag_on(const char *name) {
    const char *v = getenv(name);
    if (!v || !v[0]) return 0;
    if (v[0] == '0' && v[1] == '\0') return 0;
    if ((v[0] == 'n' || v[0] == 'N') &&
        (v[1] == 'o' || v[1] == 'O') && v[2] == '\0') return 0;
    if ((v[0] == 'f' || v[0] == 'F') &&
        (v[1] == 'a' || v[1] == 'A') &&
        (v[2] == 'l' || v[2] == 'L') &&
        (v[3] == 's' || v[3] == 'S') &&
        (v[4] == 'e' || v[4] == 'E') && v[5] == '\0') return 0;
    return 1;
}

/* Default OFF: Qwen3/3.5/3.6 MoE emit <think> unless disabled.
 * Set KATALI_THINK=1 (or KATALI_ENABLE_THINK=1) to re-enable thinking. */
static int want_thinking(void) {
    return env_flag_on("KATALI_THINK") || env_flag_on("KATALI_ENABLE_THINK");
}

static int format_prompt(const KataliGgufTokenizer *tk, const char *prompt,
                         int *ids, int cap) {
    if (tk->im_start_id >= 0 && tk->im_end_id >= 0) {
        int at = 0;
        at = append_id(ids, at, cap, tk->im_start_id);
        if (at < 0) return -1;
        at = append_text(tk, "user\n", ids, at, cap);
        if (at < 0) return -1;
        at = append_text(tk, prompt, ids, at, cap);
        if (at < 0) return -1;
        at = append_id(ids, at, cap, tk->im_end_id);
        if (at < 0) return -1;
        at = append_text(tk, "\n", ids, at, cap);
        if (at < 0) return -1;
        at = append_id(ids, at, cap, tk->im_start_id);
        if (at < 0) return -1;
        at = append_text(tk, "assistant\n", ids, at, cap);
        if (at < 0) return -1;
        /* Qwen hard-disable thinking (HF enable_thinking=False): after
         * <|im_start|>assistant\n prefill an empty <think>\n\n</think>\n\n
         * so generation continues with the answer. Default ON when special
         * think tokens exist. KATALI_THINK=1 skips this (thinking on).
         * KATALI_PREFILL_THINK_OPEN still forces open-only (debug).
         * Legacy KATALI_PREFILL_THINK=1 forces empty prefill even if think on. */
        if (getenv("KATALI_PREFILL_THINK_OPEN") && tk->think_open_id >= 0) {
            at = append_id(ids, at, cap, tk->think_open_id);
            if (at < 0) return -1;
            at = append_text(tk, "\n", ids, at, cap);
            if (at < 0) return -1;
        } else if ((!want_thinking() || env_flag_on("KATALI_PREFILL_THINK")) &&
                   tk->think_open_id >= 0 && tk->think_close_id >= 0) {
            at = append_id(ids, at, cap, tk->think_open_id);
            if (at < 0) return -1;
            at = append_text(tk, "\n\n", ids, at, cap);
            if (at < 0) return -1;
            at = append_id(ids, at, cap, tk->think_close_id);
            if (at < 0) return -1;
            at = append_text(tk, "\n\n", ids, at, cap);
            if (at < 0) return -1;
        }
        return at;
    }
    return katali_gguf_tokenizer_encode(tk, prompt, ids, cap);
}

int host_generate(HostModel *m, const char *prompt, int max_tokens) {
    if (!m || !m->opened || !prompt) return KATALI_ERR;
    if (!m->tok_ok) {
        fprintf(stderr, "host_generate: tokenizer unavailable\n");
        return KATALI_ERR_SUPPORT;
    }
    if (max_tokens <= 0) max_tokens = 64;
    host_state_reset(m);

    int ids[8192];
    int n;
    if (getenv("KATALI_RAW_PROMPT")) {
        n = katali_gguf_tokenizer_encode(&m->tok, prompt, ids, 8192);
    } else {
        n = format_prompt(&m->tok, prompt, ids, 8192);
    }
    if (n <= 0) {
        fprintf(stderr, "host_generate: encode failed\n");
        return KATALI_ERR;
    }
    if (n >= m->st.max_seq) n = m->st.max_seq - 1;

    if (getenv("KATALI_DEBUG")) {
        fprintf(stderr, "prompt_tokens=%d im_start=%d think_open=%d think_close=%d\n",
                n, m->tok.im_start_id, m->tok.think_open_id, m->tok.think_close_id);
        fprintf(stderr, "prompt ids:");
        for (int i = 0; i < n && i < 32; i++) fprintf(stderr, " %d", ids[i]);
        fprintf(stderr, "\n");
        for (int i = 0; i < n && i < 16; i++) {
            char tb[64]; int nb = katali_gguf_tokenizer_decode(&m->tok, ids[i], tb, sizeof(tb));
            if (nb < 0) nb = 0;
            for (int c = 0; c < nb; c++) if ((unsigned char)tb[c] < 32 || (unsigned char)tb[c] > 126) tb[c] = '?';
            tb[nb] = 0;
            fprintf(stderr, "  [%d] '%s'\n", ids[i], tb);
        }
    }
    /* Port #7: let async hot-set warmup land before prefill so generate I/O
     * does not compete with open-time prefetch_hottest; then zero counters so
     * hits/misses/bytes are the forward only (katali2 host_generate). */
    if (m->ecache.n_workers > 0 || (m->ecache.sync && m->ecache.capacity > 0)) {
        int left = ecache_wait_idle(&m->ecache, 15000);
        fprintf(stderr, "phase: ecache warmup idle leftover=%d\n", left);
        ecache_reset_stats(&m->ecache);
    }

    double prefill_t0 = katali_time_s();
    int batched = 0;
    /* Port #5: layer-major batch prefill (union experts per layer + ensure_many).
     * Falls back to per-token on decline. Decode still uses host_forward_token. */
    if (host_prefill_wanted(m, n)) {
        int brc = host_prefill_batch(m, ids, n);
        if (brc < 0) {
            fprintf(stderr, "host_generate: layer-major prefill failed\n");
            return KATALI_ERR;
        }
        if (brc == 0) batched = 1;
    }
    if (!batched) {
        for (int i = 0; i < n; i++) {
            if (host_forward_token(m, ids[i], NULL) != KATALI_OK) {
                fprintf(stderr, "host_generate: prefill failed at %d\n", i);
                return KATALI_ERR;
            }
        }
    }
    {
        double sec = katali_time_s() - prefill_t0;
        if (sec < 1e-9) sec = 1e-9;
        fprintf(stderr, "phase: prefill tokens=%d sec=%.3f tok/s=%.3f%s\n",
                n, sec, (double)n / sec, batched ? " (layer-major)" : "");
    }
    /* Post-prefill (katali2 host_qwen_prefill_batch): decay LFU, soft-pin last
     * routes under PIN_FRAC budget, drop unpinned so prompt-union experts do not
     * steal decode slots. GGUF expands each expert to 3 weight keys.
     * Skip when batch already did freq_decay+drop (avoid double decay). */
    if (!batched && m->last_eids && m->last_topk > 0 && !getenv("KATALI_NO_ECACHE")) {
        m->last_routes_valid = 1;
        int nL = m->moe.n_layers > 0 ? m->moe.n_layers : m->arch.n_layers;
        if (nL < 1) nL = 1;
        ecache_freq_decay(&m->ecache, 4);
        ecache_unpin_all(&m->ecache);
        if (ecache_pins_enabled()) {
            int lim = ecache_spec_pin_limit(&m->ecache);
            int per = (nL > 0) ? (lim / nL) : lim;
            if (per < 1 && lim > 0) per = 1;
            if (per > m->last_topk) per = m->last_topk;
            int left = lim;
            for (int L = 0; L < nL && left > 0; L++) {
                int *slot = m->last_eids + (size_t)L * (size_t)m->last_topk;
                int n_pin = per < left ? per : left;
                for (int t = 0; t < n_pin; t++) {
                    int eid = slot[t];
                    if (eid < 0) continue;
                    for (int w = 0; w < 3; w++)
                        ecache_pin(&m->ecache, L, eid * 3 + w);
                }
                left -= n_pin;
            }
            ecache_drop_unpinned(&m->ecache);
            fprintf(stderr,
                    "phase: ecache post-prefill drop_unpinned lim=%d per_layer=%d "
                    "pin_frac=%d\n",
                    lim, per, m->ecache.pin_frac);
        }
    }
    if (getenv("KATALI_DEBUG")) {
        int top_i[5]; float top_v[5];
        for (int t = 0; t < 5; t++) { top_i[t] = -1; top_v[t] = -1e30f; }
        for (int i = 0; i < m->arch.vocab; i++) {
            float v = m->st.logits[i];
            for (int t = 0; t < 5; t++) {
                if (v > top_v[t]) {
                    for (int u = 4; u > t; u--) { top_v[u] = top_v[u-1]; top_i[u] = top_i[u-1]; }
                    top_v[t] = v; top_i[t] = i; break;
                }
            }
        }
        {
            int n_nan = 0, n_inf = 0;
            float amin = 0.f, amax = 0.f;
            for (int i = 0; i < m->arch.vocab; i++) {
                float v = m->st.logits[i];
                if (v != v) n_nan++;
                else if (v > 1e30f || v < -1e30f) n_inf++;
                if (i == 0 || v < amin) amin = v;
                if (i == 0 || v > amax) amax = v;
            }
            fprintf(stderr, "after_prefill logits: nan=%d infish=%d min=%.3g max=%.3g trunk=%d/%d\n",
                    n_nan, n_inf, amin, amax, m->arch.n_trunk, m->arch.n_layers);
        }
        fprintf(stderr, "after_prefill top5:");
        for (int t = 0; t < 5; t++) {
            char tb[64]; int nb = 0;
            if (top_i[t] >= 0)
                nb = katali_gguf_tokenizer_decode(&m->tok, top_i[t], tb, sizeof(tb));
            if (nb < 0) nb = 0;
            for (int c = 0; c < nb; c++) if (tb[c] < 32 || tb[c] > 126) tb[c] = '?';
            tb[nb] = 0;
            fprintf(stderr, " [%d]=%.3f'%s'", top_i[t], top_v[t], tb);
        }
        fprintf(stderr, "\n");
    }

    int recent[256];
    int n_recent = 0;
    double decode_t0 = katali_time_s();
    int n_gen = 0;
    /* Strip residual <think>...</think> from stdout unless thinking enabled.
     * Seed suppress from prompt markers (empty prefill ends with close -> 0). */
    int think_open = m->tok.think_open_id;
    int think_close = m->tok.think_close_id;
    int show_think = want_thinking();
    int suppress = 0;
    if (!show_think && think_open >= 0) {
        for (int i = 0; i < n; i++) {
            if (ids[i] == think_open) suppress = 1;
            else if (think_close >= 0 && ids[i] == think_close) suppress = 0;
        }
    }
    /* Phase 1 telemetry: reset so the end-of-run line describes decode only.
     * With KATALI_CUDA_TELEMETRY=1 the counters are reset per token and printed
     * per token, which is what separates "one expensive token" from a pattern. */
    /* Phase 1 timeline setup (KATALI_CUDA_TIMELINE=1). gpu_k needs device-event
     * timing, which is enabled through the ABI rather than an env var because
     * the lazily-loaded DLL snapshots the CRT environment at LoadLibrary time. */
    int tl_on = 0, tl_gpu_time = 0;
    double tl_t0 = 0.0, tl_gpu0 = 0.0;
    {
        const char *e = getenv("KATALI_CUDA_TIMELINE");
        tl_on = (e && e[0] && e[0] != '0') ? 1 : 0;
        if (tl_on) {
            katali_cuda_set_gpu_timing(1);
            tl_gpu_time = 1;
            m->tl_attn = m->tl_router = m->tl_shared = m->tl_gpu = m->tl_rest = 0.0;
        }
    }
    katali_cuda_telemetry_reset();
    for (int t = 0; t < max_tokens; t++) {
        if (tl_on) {
            KataliCudaTelemetry tls;
            katali_cuda_telemetry_get(&tls);
            tl_gpu0 = tls.gpu_seconds;
            tl_t0 = katali_time_s();
            m->tl_attn = m->tl_router = m->tl_shared = m->tl_gpu = m->tl_rest = 0.0;
        }
        if (katali_cuda_telemetry_enabled()) katali_cuda_telemetry_reset();
        if (getenv("KATALI_TRACE_LOGITS") && t < 8) {
            int top_i[5]; float top_v[5];
            for (int u = 0; u < 5; u++) { top_i[u] = -1; top_v[u] = -1e30f; }
            for (int i = 0; i < m->arch.vocab; i++) {
                float v = m->st.logits[i];
                for (int u = 0; u < 5; u++) {
                    if (v > top_v[u]) {
                        for (int w = 4; w > u; w--) { top_v[w] = top_v[w-1]; top_i[w] = top_i[w-1]; }
                        top_v[u] = v; top_i[u] = i; break;
                    }
                }
            }
            fprintf(stderr, "step%d top5:", t);
            for (int u = 0; u < 5; u++) {
                char tb[64]; int nb = 0;
                if (top_i[u] >= 0) nb = katali_gguf_tokenizer_decode(&m->tok, top_i[u], tb, sizeof(tb));
                if (nb < 0) nb = 0;
                for (int c = 0; c < nb; c++) if ((unsigned char)tb[c] < 32 || (unsigned char)tb[c] > 126) tb[c] = '?';
                tb[nb] = 0;
                fprintf(stderr, " [%d]=%.3f'%s'", top_i[u], top_v[u], tb);
            }
            fprintf(stderr, "\n");
        }
        int next = katali_gguf_sample(&m->samp, m->st.logits, m->arch.vocab,
                                      recent, n_recent);
        if (next < 0) break;
        if (next == m->tok.eos_id || next == m->tok.im_end_id ||
            next == m->tok.endoftext_id) break;
        n_gen++;
        char buf[256];
        int nb = katali_gguf_tokenizer_decode(&m->tok, next, buf, sizeof(buf));
        int show_text = 1;
        if (!show_think && think_open >= 0) {
            if (next == think_open) { suppress = 1; show_text = 0; }
            else if (think_close >= 0 && next == think_close) { suppress = 0; show_text = 0; }
            else if (suppress) show_text = 0;
        }
        if (show_text && nb > 0) {
            fwrite(buf, 1, (size_t)nb, stdout);
            fflush(stdout);
        }
        if (n_recent < 256) recent[n_recent++] = next;
        else {
            memmove(recent, recent + 1, 255 * sizeof(int));
            recent[255] = next;
        }
        if (host_forward_token(m, next, NULL) != KATALI_OK) {
            fprintf(stderr, "\nhost_generate: decode step failed\n");
            return KATALI_ERR;
        }
        if (katali_cuda_telemetry_enabled()) {
            char tag[64];
            snprintf(tag, sizeof(tag), "phase: cuda-telemetry token=%d", t);
            katali_cuda_telemetry_print(tag);
        }
        /*
         * Phase 1: one compact timeline line per decode token.
         *   attn/router/shared/gpu_sub are CPU spans measured in the engine.
         *   gpu_k is device-event kernel time, so gpu_sub - gpu_k is the API +
         *   wait overhead the CPU pays for that work.
         *   idle = wall - gpu_k, and duty = gpu_k / wall.
         */
        if (tl_on) {
            const double wall = katali_time_s() - tl_t0;
            double gpu_k = 0.0;
            if (tl_gpu_time) {
                KataliCudaTelemetry tn;
                katali_cuda_telemetry_get(&tn);
                gpu_k = tn.gpu_seconds - tl_gpu0;
            }
            fprintf(stderr,
                    "tl token=%-3d wall=%.1fms attn=%.1f router=%.1f shared=%.1f "
                    "vram=%.1f xup=%.1f sub=%.1f gpu_k=%.1f wait=%.1f rest=%.1f "
                    "idle=%.1f duty=%.1f%%\n",
                    t, wall * 1000.0, m->tl_attn * 1000.0, m->tl_router * 1000.0,
                    m->tl_shared * 1000.0, m->tl_vram * 1000.0, m->tl_xup * 1000.0,
                    m->tl_sub * 1000.0, gpu_k * 1000.0, m->tl_wait * 1000.0,
                    (wall - m->tl_attn - m->tl_router - m->tl_shared - m->tl_gpu) * 1000.0,
                    (wall - gpu_k) * 1000.0,
                    wall > 0.0 ? 100.0 * gpu_k / wall : 0.0);
            m->tl_attn = m->tl_router = m->tl_shared = m->tl_gpu = m->tl_rest = 0.0;
            m->tl_wait = m->tl_vram = m->tl_xup = m->tl_sub = 0.0;
        }
    }
    fputc('\n', stdout);
    {
        double sec = katali_time_s() - decode_t0;
        if (sec < 1e-9) sec = 1e-9;
        fprintf(stderr, "speed: decode_tokens=%d decode_sec=%.3f tok/s=%.3f\n",
                n_gen, sec, (double)n_gen / sec);
        if (!katali_cuda_telemetry_enabled())
            katali_cuda_telemetry_print("phase: cuda-telemetry decode-total");
    }
    {
        size_t used = 0, cap = 0; int nres = 0;
        uint64_t hits = 0, misses = 0, ev = 0, bl = 0;
        ecache_stats(&m->ecache, &used, &cap, &nres);
        ecache_stats_ex(&m->ecache, &hits, &misses, &ev, &bl);
        fprintf(stderr,
                "ecache: hits=%llu misses=%llu evictions=%llu resident=%d "
                "used=%.2f/%.2f MiB bytes_loaded=%.2f MiB pf_hit=%llu pf_iss=%llu pf_done=%llu\n",
                (unsigned long long)hits, (unsigned long long)misses,
                (unsigned long long)ev, nres,
                (double)used / (1024.0 * 1024.0), (double)cap / (1024.0 * 1024.0),
                (double)bl / (1024.0 * 1024.0),
                (unsigned long long)m->ecache.prefetch_hits,
                (unsigned long long)m->ecache.prefetch_issued,
                (unsigned long long)m->ecache.prefetch_done);
        fprintf(stderr, "ecache: hot-experts resident(>=2 uses)=%zu/%zu\n",
                ecache_hot_resident(&m->ecache, 2), m->ecache.capacity);
    }
    return KATALI_OK;
}
