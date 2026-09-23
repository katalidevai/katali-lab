/* Katali-GGUF backend + format-independent registry — Apache-2.0 */
#include "katali_backend.h"
#include "katali_gguf_model.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ==================================================================== *
 * GGUF backend                                                          *
 * ==================================================================== */
typedef struct GgufBackendImpl {
    KataliGgufModel model;
    int opened;
    int loads;        /* model loads performed by this handle (always 1) */
    int kv_resets;    /* cumulative reset_state() calls */
} GgufBackendImpl;

static int gguf_ops_open(KataliBackend *b, const KataliBackendOptions *opts) {
    if (!b || !opts || !opts->model_path) return -1;
    GgufBackendImpl *g = (GgufBackendImpl *)calloc(1, sizeof(*g));
    if (!g) return -1;
    char err[320];
    err[0] = '\0';
    if (katali_gguf_model_open(&g->model, opts, err, sizeof(err)) != 0) {
        fprintf(stderr, "katali-gguf: open failed: %s\n", err[0] ? err : "unknown error");
        free(g);
        return -1;
    }
    g->opened = 1;
    g->loads = 1;
    b->impl = g;
    snprintf(b->name, sizeof(b->name), "gguf");
    return 0;
}

static int gguf_ops_reset(KataliBackend *b) {
    GgufBackendImpl *g = b ? (GgufBackendImpl *)b->impl : NULL;
    if (!g || !g->opened) return -1;
    katali_gguf_model_reset(&g->model);
    g->kv_resets++;
    return 0;
}

int katali_backend_last_report(const KataliBackend *b, KataliBackendReport *out) {
    const GgufBackendImpl *g = b ? (const GgufBackendImpl *)b->impl : NULL;
    if (!g || !g->opened || !out) return -1;
    KataliGgufGenReport r;
    katali_gguf_model_last_report(&g->model, &r);
    memset(out, 0, sizeof(*out));
    out->load_s = g->model.stats.open_s;
    out->tokenize_s = r.tokenize_s;
    out->prefill_s = r.prefill_s;
    out->ttft_s = r.ttft_s;
    out->decode_s = r.decode_s;
    out->total_s = r.total_s;
    out->prompt_tokens = r.prompt_tokens;
    out->generated_tokens = r.generated_tokens;
    out->decode_tps = r.decode_tps;
    out->batched_prefill = r.batched_prefill;
    out->model_loads = g->loads;
    out->kv_resets = g->kv_resets;
    out->last_reset_s = g->model.stats.last_reset_s;
    out->resident_bytes = katali_gguf_model_resident_bytes(&g->model);
    return 0;
}

static int gguf_ops_generate(KataliBackend *b, const char *prompt,
                             char *output, size_t output_cap) {
    GgufBackendImpl *g = b ? (GgufBackendImpl *)b->impl : NULL;
    if (!g || !g->opened) return -1;
    return katali_gguf_model_generate(&g->model, prompt, output, output_cap);
}

static int gguf_ops_cancel(KataliBackend *b) {
    GgufBackendImpl *g = b ? (GgufBackendImpl *)b->impl : NULL;
    if (!g || !g->opened) return -1;
    katali_gguf_model_cancel(&g->model);
    return 0;
}

static size_t gguf_ops_resident_bytes(const KataliBackend *b) {
    const GgufBackendImpl *g = b ? (const GgufBackendImpl *)b->impl : NULL;
    if (!g || !g->opened) return 0;
    return katali_gguf_model_resident_bytes(&g->model);
}

static void gguf_ops_close(KataliBackend *b) {
    if (!b) return;
    GgufBackendImpl *g = (GgufBackendImpl *)b->impl;
    if (g) {
        if (g->opened) katali_gguf_model_close(&g->model);
        free(g);
    }
    b->impl = NULL;
}

static const KataliBackendOps g_gguf_ops = {
    gguf_ops_open,
    gguf_ops_reset,
    gguf_ops_generate,
    gguf_ops_cancel,
    gguf_ops_resident_bytes,
    gguf_ops_close
};

const KataliBackendOps *katali_gguf_backend_ops(void) { return &g_gguf_ops; }

int katali_gguf_backend_describe(const KataliBackend *b, struct KataliGgufInfo *out) {
    const GgufBackendImpl *g = b ? (const GgufBackendImpl *)b->impl : NULL;
    if (!g || !g->opened || !out) return -1;
    return katali_gguf_describe(&g->model.file, out);
}

const struct KataliGgufStats *katali_gguf_backend_stats(const KataliBackend *b) {
    const GgufBackendImpl *g = b ? (const GgufBackendImpl *)b->impl : NULL;
    if (!g || !g->opened) return NULL;
    return &g->model.stats;
}

const char *katali_gguf_backend_name(void) { return "Katali GGUF"; }

const char *katali_output_kind_name(KataliOutputKind kind) {
    switch (kind) {
        case KATALI_OUT_TEXT: return "text";
        case KATALI_OUT_JSON: return "json";
        case KATALI_OUT_TOOL_CALL: return "tool_call";
        case KATALI_OUT_FINAL: return "final";
        case KATALI_OUT_CLARIFICATION: return "clarification";
        default: return "unknown";
    }
}

/* ==================================================================== *
 * Registry                                                              *
 * ==================================================================== */
typedef struct RegEntry {
    char name[32];
    const KataliBackendOps *ops;
} RegEntry;

static RegEntry g_reg[8];
static int g_reg_n = 0;

int katali_backend_register(const char *name, const KataliBackendOps *ops) {
    if (!name || !ops) return -1;
    for (int i = 0; i < g_reg_n; i++) {
        if (strcmp(g_reg[i].name, name) == 0) { g_reg[i].ops = ops; return 0; }
    }
    if (g_reg_n >= 8) return -1;
    snprintf(g_reg[g_reg_n].name, sizeof(g_reg[g_reg_n].name), "%s", name);
    g_reg[g_reg_n].ops = ops;
    g_reg_n++;
    return 0;
}

const KataliBackendOps *katali_backend_lookup(const char *name) {
    if (!name) return NULL;
    if (strcmp(name, "gguf") == 0) return &g_gguf_ops;
    for (int i = 0; i < g_reg_n; i++)
        if (strcmp(g_reg[i].name, name) == 0) return g_reg[i].ops;
    return NULL;
}

KataliBackend *katali_backend_create(const char *name,
                                     const KataliBackendOptions *opts) {
    const KataliBackendOps *ops = katali_backend_lookup(name);
    if (!ops) return NULL;
    KataliBackend *b = (KataliBackend *)calloc(1, sizeof(*b));
    if (!b) return NULL;
    b->ops = ops;
    snprintf(b->name, sizeof(b->name), "%s", name);
    if (ops->open && ops->open(b, opts) != 0) {
        if (ops->close) ops->close(b);
        free(b);
        return NULL;
    }
    return b;
}

void katali_backend_destroy(KataliBackend *b) {
    if (!b) return;
    if (b->ops && b->ops->close) b->ops->close(b);
    free(b);
}
