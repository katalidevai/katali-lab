#include "katali_dense_cuda.h"
#include "katali_cuda.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

typedef struct { const KataliGgufTensor *host; void *dev; uint64_t bytes; } DenseWeight;
static DenseWeight g_w[256];
static int g_nw;
static float *g_x, *g_y;
static uint64_t g_cap;
static int g_checked, g_enabled;

int katali_dense_cuda_enabled(void) {
    if (!g_checked) {
        g_checked = 1;
        const char *e = getenv("KATALI_CUDA");
        /* Dense Qwen3 is CPU-first; opt into CUDA explicitly with KATALI_CUDA=1. */
        g_enabled = e && (e[0] == '1' || e[0] == 'y' || e[0] == 'Y') && katali_cuda_probe() != 0; fprintf(stderr, "phase: dense cuda %s (%s)\\\\n", g_enabled ? "ON" : "OFF", katali_cuda_status());
    }
    return g_enabled;
}

static int ensure_buffers(uint64_t cols, uint64_t rows) {
    uint64_t need = cols > rows ? cols : rows;
    if (g_cap >= need && g_x && g_y) return 0;
    if (g_x) katali_cuda_free(g_x);
    if (g_y) katali_cuda_free(g_y);
    g_x = g_y = NULL; g_cap = 0;
    if (katali_cuda_malloc((void **)&g_x, need * sizeof(float)) != KATALI_OK) return -1;
    if (katali_cuda_malloc((void **)&g_y, need * sizeof(float)) != KATALI_OK) { katali_cuda_free(g_x); g_x=NULL; return -1; }
    g_cap = need;
    return 0;
}

static void *device_weight(const KataliGgufTensor *t) {
    for (int i=0; i<g_nw; i++) if (g_w[i].host == t) return g_w[i].dev;
    if (g_nw >= (int)(sizeof(g_w)/sizeof(g_w[0])) || !t || !t->data || !t->nbytes) return NULL;
    void *d = NULL;
    if (katali_cuda_malloc(&d, (size_t)t->nbytes) != KATALI_OK) return NULL;
    if (katali_cuda_upload(d, t->data, (size_t)t->nbytes) != KATALI_OK) { katali_cuda_free(d); return NULL; }
    g_w[g_nw++] = (DenseWeight){ t, d, t->nbytes };
    return d;
}

int katali_dense_cuda_matvec(const KataliGgufTensor *t, const float *x, float *y) {
    if (!katali_dense_cuda_enabled() || !t || !x || !y) return -1;
    if (!katali_cuda_supports_type(t->type)) return -1;
    uint64_t cols=t->dims[0], rows=t->dims[1];
    if (ensure_buffers(cols, rows) != 0) return -1;
    void *w=device_weight(t); if (!w) return -1;
    if (katali_cuda_upload(g_x, x, (size_t)cols*sizeof(float)) != KATALI_OK) return -1;
    if (katali_cuda_matvec(t->type, w, rows, cols, g_x, g_y) != KATALI_OK) return -1;
    if (katali_cuda_download(y, g_y, (size_t)rows*sizeof(float)) != KATALI_OK) return -1;
    return 0;
}

int katali_dense_cuda_matvec_multi(const KataliGgufTensor **ts, float **ys, int n, const float *x) {
    if (n <= 0 || n > 4 || !ts || !ys || !x || !katali_dense_cuda_enabled()) return -1;
    uint64_t cols = ts[0]->dims[0];
    for (int i=0; i<n; i++) if (!ts[i] || !ts[i]->data || ts[i]->dims[0] != cols) return -1;
    uint64_t maxrows = 0;
    for (int i=0; i<n; i++) if (ts[i]->dims[1] > maxrows) maxrows = ts[i]->dims[1];
    if (ensure_buffers(cols, maxrows) != 0) return -1;
    if (katali_cuda_upload(g_x, x, (size_t)cols*sizeof(float)) != KATALI_OK) return -1;
    for (int i=0; i<n; i++) {
        if (!katali_cuda_supports_type(ts[i]->type)) return -1;
        void *w = device_weight(ts[i]);
        if (!w || katali_cuda_matvec(ts[i]->type, w, ts[i]->dims[1], cols, g_x, g_y) != KATALI_OK) return -1;
        if (katali_cuda_download(ys[i], g_y, (size_t)ts[i]->dims[1]*sizeof(float)) != KATALI_OK) return -1;
    }
    return 0;
}

void katali_dense_cuda_shutdown(void) {
    for (int i=0; i<g_nw; i++) if (g_w[i].dev) katali_cuda_free(g_w[i].dev);
    if (g_x) katali_cuda_free(g_x);
    if (g_y) katali_cuda_free(g_y);
    memset(g_w,0,sizeof(g_w)); g_nw=0; g_x=g_y=NULL; g_cap=0; g_checked=0; g_enabled=0;
}