/* Katali-GGUF quantized tensor decoding — Apache-2.0
 *
 * The block layouts implemented here are the public GGUF / ggml quantization
 * format. All code is original. Reference decoders are the source of truth and
 * are exercised by tests/test_dtype.c.
 */
#include "katali_gguf_dtype.h"
#include "katali_gguf_fp.h"
#include "katali_gguf_threads.h"
#include "katali_gguf_simd.h"
#include "katali_gguf_prof.h"

#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <float.h>

/* ==================================================================== *
 * Kernel profiler (timing only; no effect on results)                   *
 * ==================================================================== */
#ifdef _WIN32
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0600
#endif
#include <windows.h>
static double prof_now(void) {
    static LARGE_INTEGER freq;
    static int have = 0;
    if (!have) { QueryPerformanceFrequency(&freq); have = 1; }
    LARGE_INTEGER c;
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)freq.QuadPart;
}
#else
#include <time.h>
static double prof_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
#endif

static unsigned long long g_prof_calls = 0;
static unsigned long long g_prof_bytes = 0;
static unsigned long long g_prof_flops = 0;
static double g_prof_s = 0.0;

static void role_reset(void);   /* defined with the per-role table below */

void katali_ggml_prof_reset(void) {
    g_prof_calls = 0; g_prof_bytes = 0; g_prof_flops = 0; g_prof_s = 0.0;
    role_reset();
}

void katali_ggml_prof_stats(unsigned long long *calls, double *seconds,
                            unsigned long long *bytes, unsigned long long *flops) {
    if (calls) *calls = g_prof_calls;
    if (seconds) *seconds = g_prof_s;
    if (bytes) *bytes = g_prof_bytes;
    if (flops) *flops = g_prof_flops;
}

/* --- per-(role, type) accounting ---------------------------------------
 * Only the combinations that actually occur are stored, so the table stays tiny
 * (a handful of entries for a Qwen3 model) and the lookup is a short linear scan
 * per matvec call - ~250 calls per token, which is nothing against the kernel
 * time it is measuring. */
#define ROLE_SLOT_MAX 48
typedef struct RoleSlot {
    int      role;
    uint32_t type;
    int      prefill;      /* 1 for the batched matmul path, 0 for decode */
    unsigned long long calls, bytes, flops;
    double   s;
} RoleSlot;

static RoleSlot g_role[ROLE_SLOT_MAX];
static int      g_role_n = 0;

static void role_reset(void) {
    g_role_n = 0;
    memset(g_role, 0, sizeof(g_role));
}

static void role_add_ex(int role, uint32_t type, int prefill, double seconds,
                        unsigned long long bytes, unsigned long long flops) {
    /* The same single flag that gates all profiling (KATALI_GGUF_NO_PROF=1), so a
     * pure-speed run pays nothing for this - no separate switch, consistent with
     * the rest of the profiler. */
    if (!katali_prof_enabled()) return;
    RoleSlot *s = NULL;
    for (int i = 0; i < g_role_n; i++) {
        if (g_role[i].role == role && g_role[i].type == type &&
            g_role[i].prefill == prefill) { s = &g_role[i]; break; }
    }
    if (!s) {
        if (g_role_n >= ROLE_SLOT_MAX) return;   /* never expected: few combos exist */
        s = &g_role[g_role_n++];
        s->role = role;
        s->type = type;
        s->prefill = prefill;
        s->calls = s->bytes = s->flops = 0;
        s->s = 0.0;
    }
    s->calls++;
    s->s += seconds;
    s->bytes += bytes;
    s->flops += flops;
}

const char *katali_ggml_role_name(int role) {
    switch (role) {
        case KATALI_ROLE_LM_HEAD: return "lm_head";
        case KATALI_ROLE_WQ:      return "attn_q";
        case KATALI_ROLE_WK:      return "attn_k";
        case KATALI_ROLE_WV:      return "attn_v";
        case KATALI_ROLE_WO:      return "attn_o";
        case KATALI_ROLE_GATE:    return "ffn_gate";
        case KATALI_ROLE_UP:      return "ffn_up";
        case KATALI_ROLE_DOWN:    return "ffn_down";
        default:                  return "other";
    }
}

int katali_ggml_prof_role_count(void) { return g_role_n; }

void katali_ggml_prof_role_get(int i, int *role, uint32_t *type,
                               unsigned long long *calls, double *seconds,
                               unsigned long long *bytes, unsigned long long *flops) {
    if (i < 0 || i >= g_role_n) return;
    if (role)    *role = g_role[i].role;
    if (type)    *type = g_role[i].type;
    if (calls)   *calls = g_role[i].calls;
    if (seconds) *seconds = g_role[i].s;
    if (bytes)   *bytes = g_role[i].bytes;
    if (flops)   *flops = g_role[i].flops;
}

/* 1 when slot i is the batched prefill path, 0 for the token-by-token path, so
 * the caller can report prefill and decode kernels separately. */
int katali_ggml_prof_role_is_prefill(int i) {
    if (i < 0 || i >= g_role_n) return 0;
    return g_role[i].prefill;
}

/* ==================================================================== *
 * Type table                                                            *
 * ==================================================================== */
typedef struct TypeInfo {
    uint32_t   type;
    const char *name;
    uint64_t   block;   /* elements per block */
    uint64_t   bytes;   /* bytes per block */
    int        supported;
} TypeInfo;

static const TypeInfo k_types[] = {
    { KGGML_F32,  "F32",  1,   4,  1 },
    { KGGML_F16,  "F16",  1,   2,  1 },
    { KGGML_Q4_0, "Q4_0", 32,  18, 1 },
    { KGGML_Q4_1, "Q4_1", 32,  20, 1 },
    { KGGML_Q5_0, "Q5_0", 32,  22, 1 },
    { KGGML_Q5_1, "Q5_1", 32,  24, 1 },
    { KGGML_Q8_0, "Q8_0", 32,  34, 1 },
    { KGGML_Q8_1, "Q8_1", 32,  36, 1 },
    { KGGML_Q2_K, "Q2_K", 256, 84, 1 },
    { KGGML_Q3_K, "Q3_K", 256, 110, 1 },
    { KGGML_Q4_K, "Q4_K", 256, 144, 1 },
    { KGGML_Q5_K, "Q5_K", 256, 176, 1 },
    { KGGML_Q6_K, "Q6_K", 256, 210, 1 },
    { KGGML_Q8_K, "Q8_K", 256, 292, 1 },
    { KGGML_I8,   "I8",   1,   1,  1 },
    { KGGML_I16,  "I16",  1,   2,  1 },
    { KGGML_I32,  "I32",  1,   4,  1 },
    { KGGML_I64,  "I64",  1,   8,  1 },
    { KGGML_F64,  "F64",  1,   8,  1 },
    { KGGML_BF16, "BF16", 1,   2,  1 },
    { 0, NULL, 0, 0, 0 }
};

static const TypeInfo *type_info(uint32_t type) {
    for (const TypeInfo *t = k_types; t->name; t++)
        if (t->type == type) return t;
    return NULL;
}

const char *katali_ggml_type_name(uint32_t type) {
    const TypeInfo *t = type_info(type);
    return t ? t->name : "UNKNOWN";
}

int katali_ggml_type_supported(uint32_t type) {
    const TypeInfo *t = type_info(type);
    return t && t->supported;
}

int katali_ggml_type_is_quant(uint32_t type) {
    switch (type) {
        case KGGML_Q4_0: case KGGML_Q4_1: case KGGML_Q5_0: case KGGML_Q5_1:
        case KGGML_Q8_0: case KGGML_Q8_1: case KGGML_Q2_K: case KGGML_Q3_K:
        case KGGML_Q4_K: case KGGML_Q5_K: case KGGML_Q6_K: case KGGML_Q8_K:
            return 1;
        default: return 0;
    }
}

uint64_t katali_ggml_type_block_size(uint32_t type) {
    const TypeInfo *t = type_info(type);
    return t ? t->block : 0;
}
uint64_t katali_ggml_type_block_bytes(uint32_t type) {
    const TypeInfo *t = type_info(type);
    return t ? t->bytes : 0;
}
uint64_t katali_ggml_row_bytes(uint32_t type, uint64_t n) {
    uint64_t bs = katali_ggml_type_block_size(type);
    uint64_t bb = katali_ggml_type_block_bytes(type);
    if (bs == 0 || bb == 0) return 0;
    if (n % bs != 0) return 0;
    if (n / bs > UINT64_MAX / bb) return 0;
    return (n / bs) * bb;
}

/* ==================================================================== *
 * Small helpers                                                         *
 * ==================================================================== */
static inline uint16_t ld_u16(const uint8_t *p) {
    uint16_t v; memcpy(&v, p, 2); return v;
}
static inline uint32_t ld_u32(const uint8_t *p) {
    uint32_t v; memcpy(&v, p, 4); return v;
}
static inline float ld_f32(const uint8_t *p) {
    float v; memcpy(&v, p, 4); return v;
}

/* ==================================================================== *
 * Reference dequantization (non-K blocks)                               *
 * ==================================================================== */
static int dequant_nonblock(uint32_t type, const uint8_t *s, uint64_t n, float *y) {
    switch (type) {
        case KGGML_F32:
            for (uint64_t i = 0; i < n; i++) y[i] = ld_f32(s + 4 * i);
            return 0;
        case KGGML_F16:
            for (uint64_t i = 0; i < n; i++) y[i] = katali_f16_to_f32(ld_u16(s + 2 * i));
            return 0;
        case KGGML_BF16:
            for (uint64_t i = 0; i < n; i++) y[i] = katali_bf16_to_f32(ld_u16(s + 2 * i));
            return 0;
        case KGGML_F64:
            for (uint64_t i = 0; i < n; i++) { double d; memcpy(&d, s + 8 * i, 8); y[i] = (float)d; }
            return 0;
        case KGGML_I8:  for (uint64_t i = 0; i < n; i++) y[i] = (float)(int8_t)s[i]; return 0;
        case KGGML_I16: for (uint64_t i = 0; i < n; i++) y[i] = (float)(int16_t)ld_u16(s + 2 * i); return 0;
        case KGGML_I32: for (uint64_t i = 0; i < n; i++) y[i] = (float)(int32_t)ld_u32(s + 4 * i); return 0;
        case KGGML_I64: for (uint64_t i = 0; i < n; i++) { int64_t v; memcpy(&v, s + 8 * i, 8); y[i] = (float)v; } return 0;
        default: return -1;
    }
}

static void dequant_q4_0(const uint8_t *s, uint64_t nb, float *y) {
    for (uint64_t b = 0; b < nb; b++) {
        const uint8_t *blk = s + b * 18;
        float d = katali_f16_to_f32(ld_u16(blk));
        const uint8_t *qs = blk + 2;
        for (int j = 0; j < 16; j++) {
            y[b * 32 + j]      = d * (float)((qs[j] & 0x0F) - 8);
            y[b * 32 + 16 + j] = d * (float)((qs[j] >> 4) - 8);
        }
    }
}

static void dequant_q4_1(const uint8_t *s, uint64_t nb, float *y) {
    for (uint64_t b = 0; b < nb; b++) {
        const uint8_t *blk = s + b * 20;
        float d = katali_f16_to_f32(ld_u16(blk));
        float m = katali_f16_to_f32(ld_u16(blk + 2));
        const uint8_t *qs = blk + 4;
        for (int j = 0; j < 16; j++) {
            y[b * 32 + j]      = d * (float)(qs[j] & 0x0F) + m;
            y[b * 32 + 16 + j] = d * (float)(qs[j] >> 4) + m;
        }
    }
}

static void dequant_q5_0(const uint8_t *s, uint64_t nb, float *y) {
    for (uint64_t b = 0; b < nb; b++) {
        const uint8_t *blk = s + b * 22;
        float d = katali_f16_to_f32(ld_u16(blk));
        uint32_t qh = ld_u32(blk + 2);
        const uint8_t *qs = blk + 6;
        for (int j = 0; j < 16; j++) {
            int x0 = (qs[j] & 0x0F) | (int)(((qh >> j) << 4) & 0x10);
            int x1 = (qs[j] >> 4)   | (int)((qh >> (j + 12)) & 0x10);
            y[b * 32 + j]      = d * (float)(x0 - 16);
            y[b * 32 + 16 + j] = d * (float)(x1 - 16);
        }
    }
}

static void dequant_q5_1(const uint8_t *s, uint64_t nb, float *y) {
    for (uint64_t b = 0; b < nb; b++) {
        const uint8_t *blk = s + b * 24;
        float d = katali_f16_to_f32(ld_u16(blk));
        float m = katali_f16_to_f32(ld_u16(blk + 2));
        uint32_t qh = ld_u32(blk + 4);
        const uint8_t *qs = blk + 8;
        for (int j = 0; j < 16; j++) {
            int x0 = (qs[j] & 0x0F) | (int)(((qh >> j) << 4) & 0x10);
            int x1 = (qs[j] >> 4)   | (int)((qh >> (j + 12)) & 0x10);
            y[b * 32 + j]      = d * (float)x0 + m;
            y[b * 32 + 16 + j] = d * (float)x1 + m;
        }
    }
}

static void dequant_q8_0(const uint8_t *s, uint64_t nb, float *y) {
    for (uint64_t b = 0; b < nb; b++) {
        const uint8_t *blk = s + b * 34;
        float d = katali_f16_to_f32(ld_u16(blk));
        const int8_t *qs = (const int8_t *)(blk + 2);
        for (int j = 0; j < 32; j++) y[b * 32 + j] = d * (float)qs[j];
    }
}

static void dequant_q8_1(const uint8_t *s, uint64_t nb, float *y) {
    for (uint64_t b = 0; b < nb; b++) {
        const uint8_t *blk = s + b * 36;
        float d = katali_f16_to_f32(ld_u16(blk));
        const int8_t *qs = (const int8_t *)(blk + 4);
        for (int j = 0; j < 32; j++) y[b * 32 + j] = d * (float)qs[j];
    }
}

static void dequant_q8_K(const uint8_t *s, uint64_t nb, float *y) {
    for (uint64_t b = 0; b < nb; b++) {
        const uint8_t *blk = s + b * 292;
        float d = ld_f32(blk);
        const int8_t *qs = (const int8_t *)(blk + 4);
        for (int j = 0; j < 256; j++) y[b * 256 + j] = d * (float)qs[j];
    }
}

/* ==================================================================== *
 * Reference dequantization (K-quants)                                   *
 * ==================================================================== */
/* 6-bit scale/min unpacking used by Q4_K and Q5_K. */
static void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63u;
        *m = q[j + 4] & 63u;
    } else {
        *d = (uint8_t)((q[j + 4] & 0x0Fu) | ((q[j - 4] >> 6) << 4));
        *m = (uint8_t)((q[j + 4] >> 4)   | ((q[j]     >> 6) << 4));
    }
}

static void dequant_q2_K(const uint8_t *s, uint64_t nb, float *y) {
    for (uint64_t b = 0; b < nb; b++) {
        const uint8_t *blk = s + b * 84;
        float d = katali_f16_to_f32(ld_u16(blk + 80));
        float mn = katali_f16_to_f32(ld_u16(blk + 82));
        const uint8_t *sc = blk;
        const uint8_t *q = blk + 16;
        int is = 0;
        float *out = y + b * 256;
        for (int n = 0; n < 256; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                uint8_t s0 = sc[is++];
                float dl = d * (float)(s0 & 0x0F);
                float ml = mn * (float)(s0 >> 4);
                for (int l = 0; l < 16; l++)
                    out[n + j * 32 + l] = dl * (float)((q[l] >> shift) & 3) - ml;
                uint8_t s1 = sc[is++];
                dl = d * (float)(s1 & 0x0F);
                ml = mn * (float)(s1 >> 4);
                for (int l = 0; l < 16; l++)
                    out[n + j * 32 + 16 + l] = dl * (float)((q[l + 16] >> shift) & 3) - ml;
                shift += 2;
            }
            q += 32;
        }
    }
}

static void dequant_q3_K(const uint8_t *s, uint64_t nb, float *y) {
    for (uint64_t b = 0; b < nb; b++) {
        const uint8_t *blk = s + b * 110;
        float d = katali_f16_to_f32(ld_u16(blk + 108));
        const uint8_t *hm = blk;
        const uint8_t *q = blk + 32;
        const uint8_t *raw = blk + 96;
        uint32_t aux[4];
        aux[0] = ld_u32(raw);
        aux[1] = ld_u32(raw + 4);
        aux[2] = ld_u32(raw + 8);
        int8_t scales[16];
        uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & 0x0F0F0F0Fu) | (((tmp >> 4) & 0x03030303u) << 4);
        aux[3] = ((aux[1] >> 4) & 0x0F0F0F0Fu) | (((tmp >> 6) & 0x03030303u) << 4);
        aux[0] = (aux[0] & 0x0F0F0F0Fu) | (((tmp >> 0) & 0x03030303u) << 4);
        aux[1] = (aux[1] & 0x0F0F0F0Fu) | (((tmp >> 2) & 0x03030303u) << 4);
        memcpy(scales, aux, 16);
        float *out = y + b * 256;
        int is = 0;
        uint8_t m = 1;
        for (int n = 0; n < 256; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                float dl = d * (float)(scales[is++] - 32);
                for (int l = 0; l < 16; l++)
                    out[n + j * 32 + l] =
                        dl * (float)(((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
                dl = d * (float)(scales[is++] - 32);
                for (int l = 0; l < 16; l++)
                    out[n + j * 32 + 16 + l] =
                        dl * (float)(((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4));
                shift += 2;
                m = (uint8_t)(m << 1);
            }
            q += 32;
        }
    }
}

static void dequant_q4_K(const uint8_t *s, uint64_t nb, float *y) {
    for (uint64_t b = 0; b < nb; b++) {
        const uint8_t *blk = s + b * 144;
        float d = katali_f16_to_f32(ld_u16(blk));
        float mn = katali_f16_to_f32(ld_u16(blk + 2));
        const uint8_t *sc = blk + 4;
        const uint8_t *q = blk + 16;
        float *out = y + b * 256;
        int is = 0;
        for (int n = 0; n < 256; n += 64) {
            uint8_t s0, m0, s1, m1;
            get_scale_min_k4(is + 0, sc, &s0, &m0);
            get_scale_min_k4(is + 1, sc, &s1, &m1);
            float d1 = d * s0, m1v = mn * m0;
            float d2 = d * s1, m2v = mn * m1;
            for (int l = 0; l < 32; l++)
                out[n + l] = d1 * (float)(q[l] & 0x0F) - m1v;
            for (int l = 0; l < 32; l++)
                out[n + 32 + l] = d2 * (float)(q[l] >> 4) - m2v;
            q += 32;
            is += 2;
        }
    }
}

static void dequant_q5_K(const uint8_t *s, uint64_t nb, float *y) {
    for (uint64_t b = 0; b < nb; b++) {
        const uint8_t *blk = s + b * 176;
        float d = katali_f16_to_f32(ld_u16(blk));
        float mn = katali_f16_to_f32(ld_u16(blk + 2));
        const uint8_t *sc = blk + 4;
        const uint8_t *qh = blk + 16;
        const uint8_t *ql = blk + 48;
        float *out = y + b * 256;
        int is = 0;
        uint8_t u1 = 1, u2 = 2;
        for (int n = 0; n < 256; n += 64) {
            uint8_t s0, m0, s1, m1;
            get_scale_min_k4(is + 0, sc, &s0, &m0);
            get_scale_min_k4(is + 1, sc, &s1, &m1);
            float d1 = d * s0, m1v = mn * m0;
            float d2 = d * s1, m2v = mn * m1;
            for (int l = 0; l < 32; l++) {
                int hi = (qh[l] & u1) ? 16 : 0;
                out[n + l] = d1 * (float)((ql[l] & 0x0F) + hi) - m1v;
            }
            for (int l = 0; l < 32; l++) {
                int hi = (qh[l] & u2) ? 16 : 0;
                out[n + 32 + l] = d2 * (float)((ql[l] >> 4) + hi) - m2v;
            }
            ql += 32;
            is += 2;
            u1 = (uint8_t)(u1 << 2);
            u2 = (uint8_t)(u2 << 2);
        }
    }
}

static void dequant_q6_K(const uint8_t *s, uint64_t nb, float *y) {
    for (uint64_t b = 0; b < nb; b++) {
        const uint8_t *blk = s + b * 210;
        const uint8_t *ql = blk;
        const uint8_t *qh = blk + 128;
        const int8_t *sc = (const int8_t *)(blk + 192);
        float d = katali_f16_to_f32(ld_u16(blk + 208));
        float *out = y + b * 256;
        for (int n = 0; n < 256; n += 128) {
            for (int l = 0; l < 32; l++) {
                int is = l / 16;
                int q1 = (int)((ql[l]      & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32;
                int q2 = (int)((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
                int q3 = (int)((ql[l]      >> 4)   | (((qh[l] >> 4) & 3) << 4)) - 32;
                int q4 = (int)((ql[l + 32] >> 4)   | (((qh[l] >> 6) & 3) << 4)) - 32;
                out[n + l]      = d * (float)sc[is + 0] * (float)q1;
                out[n + l + 32] = d * (float)sc[is + 2] * (float)q2;
                out[n + l + 64] = d * (float)sc[is + 4] * (float)q3;
                out[n + l + 96] = d * (float)sc[is + 6] * (float)q4;
            }
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

/* ==================================================================== *
 * Public reference dequantization                                       *
 * ==================================================================== */
int katali_ggml_dequant_ref(uint32_t type, const void *src, uint64_t n, float *out) {
    if (!src || !out) return -1;
    uint64_t bs = katali_ggml_type_block_size(type);
    if (bs == 0 || bs == 1) return dequant_nonblock(type, (const uint8_t *)src, n, out);
    if (n % bs != 0) return -1;
    uint64_t nb = n / bs;
    const uint8_t *s = (const uint8_t *)src;
    switch (type) {
        case KGGML_Q4_0: dequant_q4_0(s, nb, out); return 0;
        case KGGML_Q4_1: dequant_q4_1(s, nb, out); return 0;
        case KGGML_Q5_0: dequant_q5_0(s, nb, out); return 0;
        case KGGML_Q5_1: dequant_q5_1(s, nb, out); return 0;
        case KGGML_Q8_0: dequant_q8_0(s, nb, out); return 0;
        case KGGML_Q8_1: dequant_q8_1(s, nb, out); return 0;
        case KGGML_Q8_K: dequant_q8_K(s, nb, out); return 0;
        case KGGML_Q2_K: dequant_q2_K(s, nb, out); return 0;
        case KGGML_Q3_K: dequant_q3_K(s, nb, out); return 0;
        case KGGML_Q4_K: dequant_q4_K(s, nb, out); return 0;
        case KGGML_Q5_K: dequant_q5_K(s, nb, out); return 0;
        case KGGML_Q6_K: dequant_q6_K(s, nb, out); return 0;
        default: return -1;
    }
}

/* ==================================================================== *
 * Dot products                                                          *
 * ==================================================================== */
float katali_ggml_vec_dot_ref(uint32_t type, const void *src,
                              const float *x, uint64_t n) {
    uint64_t bs = katali_ggml_type_block_size(type);
    uint64_t bb = katali_ggml_type_block_bytes(type);
    if (bs == 0 || bb == 0 || n % bs != 0) return 0.0f;
    const uint8_t *s = (const uint8_t *)src;
    float tmp[256];
    double acc = 0.0;
    uint64_t nb = n / bs;
    for (uint64_t b = 0; b < nb; b++) {
        if (katali_ggml_dequant_ref(type, s + b * bb, bs, tmp) != 0) return 0.0f;
        const float *xb = x + b * bs;
        for (uint64_t i = 0; i < bs; i++) acc += (double)tmp[i] * (double)xb[i];
    }
    return (float)acc;
}

/* Blocked fast paths. All read exactly the bytes the reference reads, so the
 * two agree to float rounding. */
static float dot_q4_0(const uint8_t *s, const float *x, uint64_t nb) {
    double acc = 0.0;
    for (uint64_t b = 0; b < nb; b++) {
        const uint8_t *blk = s + b * 18;
        float d = katali_f16_to_f32(ld_u16(blk));
        const uint8_t *qs = blk + 2;
        const float *xb = x + b * 32;
        double sum = 0.0;
        for (int j = 0; j < 16; j++) {
            sum += (double)((qs[j] & 0x0F) - 8) * (double)xb[j];
            sum += (double)((qs[j] >> 4) - 8) * (double)xb[16 + j];
        }
        acc += (double)d * sum;
    }
    return (float)acc;
}

static float dot_q8_0(const uint8_t *s, const float *x, uint64_t nb) {
    double acc = 0.0;
    for (uint64_t b = 0; b < nb; b++) {
        const uint8_t *blk = s + b * 34;
        float d = katali_f16_to_f32(ld_u16(blk));
        const int8_t *qs = (const int8_t *)(blk + 2);
        const float *xb = x + b * 32;
        double sum = 0.0;
        for (int j = 0; j < 32; j++) sum += (double)qs[j] * (double)xb[j];
        acc += (double)d * sum;
    }
    return (float)acc;
}

static float dot_q4_K(const uint8_t *s, const float *x, uint64_t nb) {
    double acc = 0.0;
    for (uint64_t b = 0; b < nb; b++) {
        const uint8_t *blk = s + b * 144;
        float d = katali_f16_to_f32(ld_u16(blk));
        float mn = katali_f16_to_f32(ld_u16(blk + 2));
        const uint8_t *sc = blk + 4;
        const uint8_t *q = blk + 16;
        const float *xb = x + b * 256;
        int is = 0;
        for (int n = 0; n < 256; n += 64) {
            uint8_t s0, m0, s1, m1;
            get_scale_min_k4(is + 0, sc, &s0, &m0);
            get_scale_min_k4(is + 1, sc, &s1, &m1);
            float d1 = d * s0, m1v = mn * m0;
            float d2 = d * s1, m2v = mn * m1;
            double s1acc = 0.0, s2acc = 0.0, x1acc = 0.0, x2acc = 0.0;
            for (int l = 0; l < 32; l++) {
                s1acc += (double)(q[l] & 0x0F) * (double)xb[n + l];
                x1acc += (double)xb[n + l];
                s2acc += (double)(q[l] >> 4) * (double)xb[n + 32 + l];
                x2acc += (double)xb[n + 32 + l];
            }
            acc += (double)d1 * s1acc - (double)m1v * x1acc;
            acc += (double)d2 * s2acc - (double)m2v * x2acc;
            q += 32;
            is += 2;
        }
    }
    return (float)acc;
}

static float dot_f16(const uint8_t *s, const float *x, uint64_t n) {
    double acc = 0.0;
    for (uint64_t i = 0; i < n; i++)
        acc += (double)katali_f16_to_f32(ld_u16(s + 2 * i)) * (double)x[i];
    return (float)acc;
}

static float dot_bf16(const uint8_t *s, const float *x, uint64_t n) {
    double acc = 0.0;
    for (uint64_t i = 0; i < n; i++)
        acc += (double)katali_bf16_to_f32(ld_u16(s + 2 * i)) * (double)x[i];
    return (float)acc;
}

static float dot_f32(const uint8_t *s, const float *x, uint64_t n) {
    double acc = 0.0;
    for (uint64_t i = 0; i < n; i++) acc += (double)ld_f32(s + 4 * i) * (double)x[i];
    return (float)acc;
}

float katali_ggml_vec_dot_scalar(uint32_t type, const void *src,
                                 const float *x, uint64_t n) {
    uint64_t bs = katali_ggml_type_block_size(type);
    if (bs == 0) return 0.0f;
    switch (type) {
        case KGGML_F32:  return dot_f32((const uint8_t *)src, x, n);
        case KGGML_F16:  return dot_f16((const uint8_t *)src, x, n);
        case KGGML_BF16: return dot_bf16((const uint8_t *)src, x, n);
        case KGGML_Q8_0: return n % 32 == 0 ? dot_q8_0((const uint8_t *)src, x, n / 32) : 0.0f;
        case KGGML_Q4_0: return n % 32 == 0 ? dot_q4_0((const uint8_t *)src, x, n / 32) : 0.0f;
        case KGGML_Q4_K: return n % 256 == 0 ? dot_q4_K((const uint8_t *)src, x, n / 256) : 0.0f;
        default: return katali_ggml_vec_dot_ref(type, src, x, n);
    }
}

/* SIMD dispatch wrapper: tries the accelerated path, then the scalar path. */
float katali_ggml_vec_dot(uint32_t type, const void *src,
                          const float *x, uint64_t n) {
    if (!src || !x) return 0.0f;
    if (katali_ggml_simd_available()) {
        float acc;
        if (katali_ggml_vec_dot_avx2(type, src, x, n, &acc) == 0) return acc;
    }
    return katali_ggml_vec_dot_scalar(type, src, x, n);
}

/* Row decode for the prefill path: the AVX2 materializing decoder when it applies
 * and is enabled, otherwise the scalar reference. Both produce bit-identical
 * output (asserted in test_core), so the choice cannot change results.
 * KATALI_GGUF_NO_DEQUANT_AVX2=1 forces the reference for every type;
 * KATALI_GGUF_NO_DEQUANT_Q6K_AVX2=1 forces it for Q6_K only, so one decoder can
 * be A/B'd while the other stays enabled. The reference remains the source of
 * truth either way. */
static int dequant_avx2_enabled(uint32_t type) {
    static int all = -1, q6k = -1;
    if (all < 0) {
        const char *e = getenv("KATALI_GGUF_NO_DEQUANT_AVX2");
        all = (e && *e && e[0] != '0') ? 0 : 1;
        e = getenv("KATALI_GGUF_NO_DEQUANT_Q6K_AVX2");
        q6k = (e && *e && e[0] != '0') ? 0 : 1;
    }
    if (!all) return 0;
    if (type == KGGML_Q6_K) return q6k;
    return 1;
}

int katali_ggml_dequant_rows(uint32_t type, const void *src, uint64_t n, float *out) {
    if (!src || !out) return -1;
    if (dequant_avx2_enabled(type) &&
        katali_ggml_dequant_rows_avx2(type, src, n, out) == 0) {
        return 0;
    }
    return katali_ggml_dequant_ref(type, src, n, out);
}

/* ==================================================================== *
 * Multi-row F32 dot (prefill inner step)                                *
 * ==================================================================== */
void katali_ggml_f32_rows_dot(const float *const *rows, int nr,
                              const float *x, uint64_t n, float *out) {
    if (nr <= 0) return;
    if (nr == 1) { out[0] = katali_ggml_vec_dot(KGGML_F32, rows[0], x, n); return; }
    if (katali_ggml_f32_rows_dot_avx2(rows, nr, x, n, out) == 0) return;
    /* Scalar/no-SIMD fallback: per-row dots, same results as the accelerated
     * kernel (bit-identical per row by construction, both use dot_f32's order). */
    for (int r = 0; r < nr; r++)
        out[r] = katali_ggml_vec_dot(KGGML_F32, rows[r], x, n);
}

/* ==================================================================== *
 * Matrix-vector products                                                *
 * ==================================================================== */
typedef struct MatvecCtx {
    uint32_t type;
    const uint8_t *w;
    uint64_t rows, cols, row_bytes;
    const float *x;
    float *y;
    volatile int *cancel;
    /* 1 when the accelerated row-loop handles this type */
    int use_rows;
    /* Precomputed Q4_K activation sums for this call (NULL => per-row path) */
    const float *q4k_sums;
} MatvecCtx;

static void matvec_worker(int worker, int n_workers, void *arg) {
    MatvecCtx *c = (MatvecCtx *)arg;
    /* Contiguous row range per worker: one sequential weight stream each. Rows
     * have identical cost, so this is also perfectly balanced, and it is
     * numerically identical to any other partition (rows are independent). */
    const uint64_t per = (c->rows + (uint64_t)n_workers - 1) / (uint64_t)n_workers;
    uint64_t r0 = (uint64_t)worker * per;
    uint64_t r1 = r0 + per;
    if (r0 > c->rows) r0 = c->rows;
    if (r1 > c->rows) r1 = c->rows;

    if (c->use_rows &&
        katali_ggml_matvec_rows(c->type, c->w, c->rows, c->cols, c->row_bytes,
                                c->x, c->y, r0, r1, c->cancel, c->q4k_sums) == 0)
        return;
    const float *x = c->x;
    const uint64_t cols = c->cols;
    for (uint64_t r = r0; r < r1; r++) {
        if (c->cancel && *c->cancel) return;
        c->y[r] = katali_ggml_vec_dot(c->type, c->w + r * c->row_bytes, x, cols);
    }
}

/* 1 unless KATALI_GGUF_NO_ROWLOOP=1: use the accelerated per-type row loop
 * (resolved once per matvec) instead of the generic per-row dot. Kept as a
 * runtime switch so the two paths can be A/B'd in the same binary. */
static int rowloop_enabled(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("KATALI_GGUF_NO_ROWLOOP");
        v = (e && *e && e[0] != '0') ? 0 : 1;
    }
    return v;
}

/* KATALI_GGUF_SPLIT_DISPATCH=K (default 1) issues each matvec as K sequential
 * dispatches over row slices instead of one dispatch over all rows. Total work,
 * the per-row kernel and the row partitioning are unchanged, so the output is
 * bit-identical; only the number of synchronisation points per matvec changes.
 * It exists to measure how much of the caller's sync_wait is per-dispatch
 * overhead rather than worker imbalance, which is what decides whether reducing
 * the dispatch count (fusing forward-pass matvecs) can pay. */
static int split_dispatch(void) {
    static int v = -2;
    if (v == -2) {
        const char *e = getenv("KATALI_GGUF_SPLIT_DISPATCH");
        v = (e && *e) ? atoi(e) : 1;
        if (v < 1) v = 1;
        if (v > 64) v = 64;
    }
    return v;
}

void katali_ggml_matvec_role(uint32_t type, const void *w,
                             uint64_t rows, uint64_t cols,
                             const float *x, float *y,
                             int n_threads, volatile int *cancel,
                             float *q4k_scratch, uint64_t q4k_scratch_cap,
                             int role) {
    if (!w || !x || !y) return;
    uint64_t rb = katali_ggml_row_bytes(type, cols);
    if (rb == 0) { for (uint64_t r = 0; r < rows; r++) y[r] = 0.0f; return; }
    double t0 = prof_now();
    MatvecCtx c;
    c.type = type;
    c.w = (const uint8_t *)w;
    c.rows = rows;
    c.cols = cols;
    c.row_bytes = rb;
    c.x = x;
    c.y = y;
    c.cancel = cancel;
    c.use_rows = (katali_ggml_simd_available() && rowloop_enabled()) ? 1 : 0;
    /* Q4_K: the per-group sums of the activation vector do not depend on the
     * weight row, so compute them once here (O(cols)) rather than once per row
     * inside the kernel. The buffer is caller-owned and sized at model open —
     * no allocation happens here or in the token loop. If it is absent, too
     * small, or disabled (KATALI_GGUF_NO_Q4K_SUMCACHE=1) the kernel recomputes
     * per row exactly as before. */
    c.q4k_sums = NULL;
    if (type == KGGML_Q4_K && c.use_rows && q4k_scratch && q4k_scratch_cap &&
        katali_ggml_q4k_sums(x, cols, q4k_scratch, q4k_scratch_cap) == 0) {
        c.q4k_sums = q4k_scratch;
    }

    if (n_threads <= 1 || rows < 2) {
        if (c.use_rows &&
            katali_ggml_matvec_rows(type, c.w, rows, cols, rb, x, y, 0, rows,
                                    cancel, c.q4k_sums) == 0) {
            /* handled by the accelerated row loop */
        } else {
            for (uint64_t r = 0; r < rows; r++) {
                if (cancel && *cancel) break;
                y[r] = katali_ggml_vec_dot(type, c.w + r * rb, x, cols);
            }
        }
    } else {
        /* Explicit count, no global mutation: the pool is resized only when the
         * count actually changes, so repeated matvecs reuse the same workers. */
        int k = split_dispatch();
        if (k > 1 && rows >= (uint64_t)k) {
            for (int s = 0; s < k; s++) {
                uint64_t r0 = rows * (uint64_t)s / (uint64_t)k;
                uint64_t r1 = rows * (uint64_t)(s + 1) / (uint64_t)k;
                if (r0 >= r1) continue;
                MatvecCtx cs = c;
                cs.rows = r1 - r0;
                cs.w = c.w + r0 * rb;
                cs.y = c.y + r0;
                katali_gguf_parallel_run_n(n_threads, matvec_worker, &cs);
            }
        } else {
            katali_gguf_parallel_run_n(n_threads, matvec_worker, &c);
        }
    }
    double dt = prof_now() - t0;
    g_prof_s += dt;
    g_prof_calls++;
    g_prof_bytes += rows * rb;
    g_prof_flops += rows * cols * 2ull;
    role_add_ex(role, type, 0, dt, rows * rb, rows * cols * 2ull);
    katali_prof_add_n(KATALI_PHASE_MATVEC, prof_now() - t0, 1);
}

void katali_ggml_matvec(uint32_t type, const void *w,
                        uint64_t rows, uint64_t cols,
                        const float *x, float *y,
                        int n_threads, volatile int *cancel,
                        float *q4k_scratch, uint64_t q4k_scratch_cap) {
    katali_ggml_matvec_role(type, w, rows, cols, x, y, n_threads, cancel,
                            q4k_scratch, q4k_scratch_cap, KATALI_ROLE_OTHER);
}
/* ==================================================================== *
 * Fused multi-matvec (fewer dispatch round-trips)                       *
 *                                                                       *
 * Several projections share one activation vector, so they can be       *
 * computed under a single dispatch. The row space of all jobs is        *
 * concatenated and partitioned once, which pays the wake/join and tail  *
 * costs once for the group instead of once per job. Each row is still   *
 * computed by the identical per-row kernel with identical inputs, so    *
 * the output is bit-identical to the same jobs run separately.          *
 * ==================================================================== */
typedef struct FusedCtx {
    const KataliMatvecJob *jobs;
    int      n_jobs;
    uint64_t total_rows, cols;
    const float *x;
    volatile int *cancel;
    const float *q4k_sums;
    int      use_rows;
} FusedCtx;

static void fused_worker(int worker, int n_workers, void *arg) {
    FusedCtx *c = (FusedCtx *)arg;
    /* One contiguous slice of the concatenated row space. Each job's own rows
     * stay contiguous, so a job is cut at most at the two ends of this slice. */
    const uint64_t per = (c->total_rows + (uint64_t)n_workers - 1) / (uint64_t)n_workers;
    uint64_t g0 = (uint64_t)worker * per;
    uint64_t g1 = g0 + per;
    if (g0 > c->total_rows) g0 = c->total_rows;
    if (g1 > c->total_rows) g1 = c->total_rows;
    if (g0 >= g1) return;

    uint64_t js = 0; /* start of the current job's rows in the global space */
    for (int j = 0; j < c->n_jobs; j++) {
        const KataliMatvecJob *jb = &c->jobs[j];
        const uint64_t je = js + jb->rows;
        if (g1 > js && g0 < je) {
            const uint64_t a = g0 > js ? g0 : js;
            const uint64_t b = g1 < je ? g1 : je;
            const uint64_t r0 = a - js, r1 = b - js;
            const uint64_t rb = katali_ggml_row_bytes(jb->type, c->cols);
            int done = 0;
            if (c->use_rows && rb) {
                done = (katali_ggml_matvec_rows(jb->type, jb->w, jb->rows, c->cols,
                                                rb, c->x, jb->y, r0, r1, c->cancel,
                                                c->q4k_sums) == 0);
            }
            if (!done) {
                for (uint64_t r = r0; r < r1; r++) {
                    if (c->cancel && *c->cancel) return;
                    jb->y[r] = katali_ggml_vec_dot(jb->type,
                                                   (const uint8_t *)jb->w + r * rb,
                                                   c->x, c->cols);
                }
            }
        }
        js = je;
    }
}

void katali_ggml_matvec_fused(const KataliMatvecJob *jobs, int n_jobs,
                              uint64_t cols, const float *x,
                              int n_threads, volatile int *cancel,
                              float *q4k_scratch, uint64_t q4k_scratch_cap) {
    if (!jobs || n_jobs <= 0 || !x || cols == 0) return;
    double t0 = prof_now();

    FusedCtx c;
    c.jobs = jobs;
    c.n_jobs = n_jobs;
    c.total_rows = 0;
    c.cols = cols;
    c.x = x;
    c.cancel = cancel;
    c.use_rows = (katali_ggml_simd_available() && rowloop_enabled()) ? 1 : 0;
    c.q4k_sums = NULL;
    uint64_t bytes = 0, flops = 0;
    int has_q4k = 0;
    for (int j = 0; j < n_jobs; j++) {
        const KataliMatvecJob *jb = &jobs[j];
        if (!jb->w || !jb->y || jb->rows == 0) continue;
        uint64_t rb = katali_ggml_row_bytes(jb->type, cols);
        if (rb == 0) continue;
        c.total_rows += jb->rows;
        bytes += jb->rows * rb;
        flops += jb->rows * cols * 2ull;
        if (jb->type == KGGML_Q4_K) has_q4k = 1;
    }
    if (c.total_rows == 0) return;

    /* One sums buffer serves every job: the sums depend only on x and cols. */
    if (has_q4k && c.use_rows && q4k_scratch && q4k_scratch_cap &&
        katali_ggml_q4k_sums(x, cols, q4k_scratch, q4k_scratch_cap) == 0) {
        c.q4k_sums = q4k_scratch;
    }

    if (n_threads <= 1 || c.total_rows < 2) {
        fused_worker(0, 1, &c);
    } else {
        katali_gguf_parallel_run_n(n_threads, fused_worker, &c);
    }

    double dt = prof_now() - t0;
    g_prof_s += dt;
    g_prof_calls++;
    g_prof_bytes += bytes;
    g_prof_flops += flops;
    /* Bytes and flops are exact per job; the group's wall time is attributed in
     * proportion to bytes, which is the right model for a bandwidth-bound kernel
     * and keeps the per-role totals comparable with the unfused path. */
    for (int j = 0; j < n_jobs; j++) {
        const KataliMatvecJob *jb = &jobs[j];
        if (!jb->w || !jb->y || jb->rows == 0) continue;
        uint64_t jrb = katali_ggml_row_bytes(jb->type, cols);
        if (jrb == 0) continue;
        unsigned long long jbytes = jb->rows * jrb;
        double share = bytes ? dt * (double)jbytes / (double)bytes : 0.0;
        role_add_ex(jb->role, jb->type, 0, share, jbytes, jb->rows * cols * 2ull);
    }
    katali_prof_add_n(KATALI_PHASE_MATVEC, prof_now() - t0, 1);
}

/* ==================================================================== *
 * Batched matmul (prompt prefill)                                       *
 * ==================================================================== */
typedef struct MatmulCtx {
    uint32_t type;
    const uint8_t *w;
    uint64_t rows, cols, row_bytes, B;
    const float *X;
    float *Y;
    float *scratch;   /* nr cols-long decode buffers per worker */
    int nr;           /* weight rows processed per inner pass: 1, 2 or 4 */
    volatile int *cancel;
} MatmulCtx;

/* How many weight rows share one pass over the activation vector. Tuned by
 * measurement (see docs/benchmarks.md): the activation re-reads dominate this
 * kernel, so sharing them across rows is the win. KATALI_GGUF_PREFILL_ROWS=1
 * selects the previous one-row-per-pass behaviour, which is also the automatic
 * choice when SIMD is unavailable or the tensor has fewer than two rows. */
static int prefill_rows(void) {
    static int v = -2;
    if (v == -2) {
        const char *e = getenv("KATALI_GGUF_PREFILL_ROWS");
        v = (e && *e) ? atoi(e) : 4;
        if (v != 1 && v != 2 && v != 4) v = 4;
    }
    return v;
}

static void matmul_worker(int worker, int n_workers, void *arg) {
    MatmulCtx *c = (MatmulCtx *)arg;
    const int NR = c->nr;
    float *bufs = c->scratch + (size_t)worker * (size_t)NR * c->cols;
    /* F32 rows can be read in place; every other type is materialized once per
     * row by katali_ggml_dequant_rows (AVX2 row decoder when available, scalar
     * reference otherwise - bit-identical either way).
     * One decode followed by B cheap F32 dots beats unpacking the quantized row once
     * per token, which was implemented and measured as no improvement (4B,
     * 511-token prefill: 61.4 s versus 61.8 s unpaired, on a machine whose own
     * run-to-run spread on that workload is ~50%, so the alternative was not
     * pursued further). */
    const int direct = (c->type == KGGML_F32);
    const uint64_t per = (c->rows + (uint64_t)n_workers - 1) / (uint64_t)n_workers;
    uint64_t r0 = (uint64_t)worker * per;
    uint64_t r1 = r0 + per;
    if (r0 > c->rows) r0 = c->rows;
    if (r1 > c->rows) r1 = c->rows;

    uint64_t r = r0;
    while (r < r1) {
        if (c->cancel && *c->cancel) return;
        int k = (int)((r1 - r) < (uint64_t)NR ? (r1 - r) : (uint64_t)NR);
        if (k == 3) k = 1;   /* the kernels take 1, 2 or 4 rows */
        const float *rowp[4];
        for (int j = 0; j < k; j++) {
            const uint8_t *row = c->w + (r + (uint64_t)j) * c->row_bytes;
            if (direct) {
                rowp[j] = (const float *)(const void *)row;
            } else {
                float *buf = bufs + (size_t)j * c->cols;
                if (katali_ggml_dequant_rows(c->type, row, c->cols, buf) != 0) return;
                rowp[j] = buf;
            }
        }
        for (uint64_t t = 0; t < c->B; t++) {
            katali_ggml_f32_rows_dot(rowp, k, c->X + t * c->cols, c->cols,
                                     c->Y + t * c->rows + r);
        }
        r += (uint64_t)k;
    }
}

void katali_ggml_matmul_role(uint32_t type, const void *w,
                             uint64_t rows, uint64_t cols,
                             const float *X, uint64_t B, float *Y,
                             int n_threads, volatile int *cancel, int role) {
    if (!w || !X || !Y || B == 0 || rows == 0) return;
    uint64_t rb = katali_ggml_row_bytes(type, cols);
    if (rb == 0) return;

    int nt = n_threads > 0 ? n_threads : katali_gguf_get_threads();
    if (nt < 1) nt = 1;
    if (nt > (int)rows) nt = (int)rows;
    int single = (nt <= 1 || rows < 2);

    double t0 = prof_now();
    /* One decode buffer per (worker, row-in-pass); nr is 1 when there is nothing
     * to share with (single worker, fewer than two rows) or when the SIMD
     * multi-row kernels are unavailable. */
    int nr = 1;
    if (!single && (rows >= 2) && katali_ggml_simd_available()) nr = prefill_rows();
    float *scratch = (float *)malloc((size_t)(single ? 1 : nt) * (size_t)nr *
                                     cols * sizeof(float));
    if (!scratch) return;
    MatmulCtx c;
    c.type = type;
    c.w = (const uint8_t *)w;
    c.rows = rows;
    c.cols = cols;
    c.row_bytes = rb;
    c.B = B;
    c.X = X;
    c.Y = Y;
    c.scratch = scratch;
    c.nr = nr;
    c.cancel = cancel;

    if (single) matmul_worker(0, 1, &c);
    else katali_gguf_parallel_run_n(nt, matmul_worker, &c);

    free(scratch);
    double dt = prof_now() - t0;
    g_prof_s += dt;
    g_prof_calls++;
    g_prof_bytes += rows * rb;
    g_prof_flops += rows * cols * 2ull * B;
    role_add_ex(role, type, 1, dt, rows * rb, rows * cols * 2ull * B);
    katali_prof_add_n(KATALI_PHASE_MATVEC, prof_now() - t0, 1);
}

void katali_ggml_matmul(uint32_t type, const void *w,
                        uint64_t rows, uint64_t cols,
                        const float *X, uint64_t B, float *Y,
                        int n_threads, volatile int *cancel) {
    katali_ggml_matmul_role(type, w, rows, cols, X, B, Y, n_threads, cancel,
                            KATALI_ROLE_OTHER);
}



typedef struct F32MatvecCtx {
    const float *w;
    uint64_t rows, cols;
    const float *x;
    float *y;
} F32MatvecCtx;

static void f32_matvec_worker(int worker, int n_workers, void *arg) {
    F32MatvecCtx *c = (F32MatvecCtx *)arg;
    for (uint64_t r = (uint64_t)worker; r < c->rows; r += (uint64_t)n_workers) {
        const float *row = c->w + r * c->cols;
        double acc = 0.0;
        for (uint64_t i = 0; i < c->cols; i++) acc += (double)row[i] * (double)c->x[i];
        c->y[r] = (float)acc;
    }
}

void katali_f32_matvec(const float *w, uint64_t rows, uint64_t cols,
                       const float *x, float *y, int n_threads) {
    if (!w || !x || !y) return;
    F32MatvecCtx c;
    c.w = w; c.rows = rows; c.cols = cols; c.x = x; c.y = y;
    if (n_threads <= 1 || rows < 2) {
        for (uint64_t r = 0; r < rows; r++) {
            const float *row = w + r * cols;
            double acc = 0.0;
            for (uint64_t i = 0; i < cols; i++) acc += (double)row[i] * (double)x[i];
            y[r] = (float)acc;
        }
        return;
    }
    katali_gguf_parallel_run_n(n_threads, f32_matvec_worker, &c);
}
/* ==================================================================== *
 * Self-test: optimized vs reference                                     *
 * ==================================================================== */
static uint32_t sel_rng(uint32_t *s) {
    uint32_t x = *s;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    *s = x ? x : 1u;
    return *s;
}

/* Force the block's scale/min f16 (or f32) fields to finite small values so a
 * random payload cannot produce NaN and defeat the comparison. */
static void sanitize_scales(uint32_t type, uint8_t *blk, uint64_t bb) {
    uint16_t half = katali_f32_to_f16(0.05f);
    (void)bb;
    switch (type) {
        case KGGML_Q4_0: case KGGML_Q5_0: case KGGML_Q8_0: case KGGML_Q8_1:
            memcpy(blk, &half, 2); break;
        case KGGML_Q4_1: case KGGML_Q5_1: case KGGML_Q4_K: case KGGML_Q5_K:
            memcpy(blk, &half, 2); memcpy(blk + 2, &half, 2); break;
        case KGGML_Q2_K: memcpy(blk + 80, &half, 2); memcpy(blk + 82, &half, 2); break;
        case KGGML_Q3_K: memcpy(blk + 108, &half, 2); break;
        case KGGML_Q6_K: memcpy(blk + 208, &half, 2); break;
        case KGGML_Q8_K: { float f = 0.05f; memcpy(blk, &f, 4); break; }
        default: break;
    }
}

int katali_dtype_selftest(void) {
    static const uint32_t types[] = {
        KGGML_F32, KGGML_F16, KGGML_BF16, KGGML_Q4_0, KGGML_Q4_1,
        KGGML_Q5_0, KGGML_Q5_1, KGGML_Q8_0, KGGML_Q8_1, KGGML_Q2_K,
        KGGML_Q3_K, KGGML_Q4_K, KGGML_Q5_K, KGGML_Q6_K, KGGML_Q8_K,
        KGGML_I8, KGGML_I16, KGGML_I32, KGGML_I64, KGGML_F64
    };
    int failures = 0;
    for (size_t ti = 0; ti < sizeof(types) / sizeof(types[0]); ti++) {
        uint32_t type = types[ti];
        if (!katali_ggml_type_supported(type)) continue;
        uint64_t bs = katali_ggml_type_block_size(type);
        uint64_t bb = katali_ggml_type_block_bytes(type);
        uint64_t nblocks = 4;
        uint64_t n = bs * nblocks;
        uint8_t *src = (uint8_t *)malloc((size_t)(bb * nblocks));
        float *x = (float *)malloc((size_t)n * sizeof(float));
        float *dec = (float *)malloc((size_t)n * sizeof(float));
        if (!src || !x || !dec) { free(src); free(x); free(dec); return -1; }

        uint32_t rng = 0x1234567u + ti;
        for (uint64_t i = 0; i < bb * nblocks; i++) src[i] = (uint8_t)sel_rng(&rng);
        for (uint64_t b = 0; b < nblocks; b++) sanitize_scales(type, src + b * bb, bb);
        for (uint64_t i = 0; i < n; i++) x[i] = ((float)(sel_rng(&rng) % 2001) - 1000.0f) / 500.0f;

        float ref = katali_ggml_vec_dot_ref(type, src, x, n);
        float opt = katali_ggml_vec_dot(type, src, x, n);
        float diff = fabsf(ref - opt);
        float tol = 1e-4f + 1e-3f * fabsf(ref);
        int ok = (ref == opt) || (isnan(ref) && isnan(opt)) || (diff <= tol);
        if (!ok) {
            printf("  FAIL %-5s ref=%.6f opt=%.6f diff=%.6g\n",
                   katali_ggml_type_name(type), ref, opt, diff);
            failures++;
        }
        if (katali_ggml_dequant_ref(type, src, n, dec) != 0) {
            printf("  FAIL %-5s dequant returned error\n", katali_ggml_type_name(type));
            failures++;
        }
        free(src); free(x); free(dec);
    }
    /* Explicit Q4_0 layout check: d=2.0, qs[0]=0xFF => low/high nibble 15. */
    {
        uint8_t blk[18];
        uint16_t d = katali_f32_to_f16(2.0f);
        memcpy(blk, &d, 2);
        memset(blk + 2, 0, 16);
        blk[2] = 0xFF;
        float out[32];
        if (katali_ggml_dequant_ref(KGGML_Q4_0, blk, 32, out) != 0 ||
            fabsf(out[0] - 14.0f) > 1e-6f || fabsf(out[16] - 14.0f) > 1e-6f ||
            fabsf(out[1] + 16.0f) > 1e-6f) {
            printf("  FAIL Q4_0 explicit layout check\n");
            failures++;
        }
    }
    /* F16 round-trip sanity. */
    {
        const float vals[] = { 0.0f, 1.0f, -1.0f, 0.5f, 65504.0f, 0.000123f };
        for (size_t i = 0; i < sizeof(vals) / sizeof(vals[0]); i++) {
            float back = katali_f16_to_f32(katali_f32_to_f16(vals[i]));
            float rel = fabsf(back - vals[i]) / (fabsf(vals[i]) + 1e-9f);
            if (rel > 2e-3f) { printf("  FAIL F16 round-trip %g -> %g\n", vals[i], back); failures++; }
        }
    }
    if (failures == 0) printf("katali dtype selftest: PASS\n");
    else printf("katali dtype selftest: %d FAILURE(S)\n", failures);
    return failures;
}

/* ==================================================================== *
 * SIMD vs scalar vs reference                                           *
 * ==================================================================== */
static void fill_numeric(uint32_t type, uint8_t *dst, uint64_t n, uint32_t *rng) {
    for (uint64_t i = 0; i < n; i++) {
        float v = ((float)(sel_rng(rng) % 2001) - 1000.0f) / 500.0f;
        switch (type) {
            case KGGML_F32: { memcpy(dst + 4 * i, &v, 4); break; }
            case KGGML_F16: { uint16_t h = katali_f32_to_f16(v); memcpy(dst + 2 * i, &h, 2); break; }
            case KGGML_BF16: { uint16_t h = katali_f32_to_bf16(v); memcpy(dst + 2 * i, &h, 2); break; }
            default: break;
        }
    }
}

int katali_dtype_simd_selftest(void) {
    static const uint32_t types[] = {
        KGGML_F32, KGGML_F16, KGGML_BF16, KGGML_Q4_0, KGGML_Q8_0,
        KGGML_Q4_K, KGGML_Q6_K
    };
    int available = katali_ggml_simd_available();
    printf("katali SIMD path: %s (%s)\n", katali_ggml_simd_name(),
           available ? "active" : "not available on this CPU");
    int failures = 0;
    for (size_t ti = 0; ti < sizeof(types) / sizeof(types[0]); ti++) {
        /* vec_dot_avx2 is only meaningful when an accelerated path exists. */
        if (!available) break;
        uint32_t type = types[ti];
        uint64_t bs = katali_ggml_type_block_size(type);
        uint64_t bb = katali_ggml_type_block_bytes(type);
        /* Enough blocks to exercise the vector loop, not just the scalar tail.
         * (For the byte-per-element types a 4-block test is only 4 elements,
         * which historically let the 16-wide main loop go untested.) */
        uint64_t nblocks = 4;
        if (bs < 16) nblocks = 256 / bs;
        else if (bs < 256) nblocks = 1024 / bs;
        uint64_t n = bs * nblocks;
        uint8_t *src = (uint8_t *)malloc((size_t)(bb * nblocks));
        float *x = (float *)malloc((size_t)n * sizeof(float));
        if (!src || !x) { free(src); free(x); return -1; }

        uint32_t rng = 0x9E3779B9u + (uint32_t)ti;
        if (type == KGGML_F32 || type == KGGML_F16 || type == KGGML_BF16) {
            fill_numeric(type, src, n, &rng);
        } else {
            for (uint64_t i = 0; i < bb * nblocks; i++) src[i] = (uint8_t)sel_rng(&rng);
            for (uint64_t b = 0; b < nblocks; b++) sanitize_scales(type, src + b * bb, bb);
        }
        for (uint64_t i = 0; i < n; i++)
            x[i] = ((float)(sel_rng(&rng) % 2001) - 1000.0f) / 500.0f;

        float ref = katali_ggml_vec_dot_ref(type, src, x, n);
        float scalar = katali_ggml_vec_dot_scalar(type, src, x, n);
        float simd = 0.0f;
        int rc = katali_ggml_vec_dot_avx2(type, src, x, n, &simd);
        float disp = katali_ggml_vec_dot(type, src, x, n);

        float tol = 1e-4f + 1e-3f * fabsf(ref);
        if (rc != 0) {
            printf("  FAIL %-5s avx2 dispatcher rejected a supported type\n",
                   katali_ggml_type_name(type));
            failures++;
        } else {
            if (!(fabsf(simd - ref) <= tol)) {
                printf("  FAIL %-5s simd=%.6f ref=%.6f diff=%.6g\n",
                       katali_ggml_type_name(type), simd, ref,
                       fabsf(simd - ref));
                failures++;
            }
            if (!(fabsf(scalar - ref) <= tol)) {
                printf("  FAIL %-5s scalar=%.6f ref=%.6f\n",
                       katali_ggml_type_name(type), scalar, ref);
                failures++;
            }
            if (disp != simd && !(isnan(disp) && isnan(simd))) {
                printf("  FAIL %-5s dispatch=%.6f simd=%.6f\n",
                       katali_ggml_type_name(type), disp, simd);
                failures++;
            }
            /* scalar fallback must still be reachable and correct */
            if (!(fabsf(scalar - ref) <= tol)) failures++;
        }
        free(src); free(x);
    }
    if (failures == 0) printf("katali SIMD selftest: PASS\n");
    else printf("katali SIMD selftest: %d FAILURE(S)\n", failures);
    return failures;
}









