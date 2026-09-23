/* Katali-GGUF phase profiler — Apache-2.0 */
#include "katali_gguf_prof.h"

#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0600
#endif
#include <windows.h>
#else
#include <time.h>
#endif

typedef struct ProfSlot {
    double seconds;
    unsigned long long calls;
    unsigned long long bytes;
} ProfSlot;

static ProfSlot g_slot[KATALI_PHASE_COUNT];
static int g_enabled = -1;

#ifdef _WIN32
static double prof_now_win(void) {
    static LARGE_INTEGER freq;
    static int have = 0;
    if (!have) { QueryPerformanceFrequency(&freq); have = 1; }
    LARGE_INTEGER c;
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)freq.QuadPart;
}
#endif

double katali_prof_now(void) {
#ifdef _WIN32
    return prof_now_win();
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
#endif
}

int katali_prof_enabled(void) {
    if (g_enabled < 0) {
        const char *off = getenv("KATALI_GGUF_NO_PROF");
        g_enabled = (off && *off && off[0] != '0') ? 0 : 1;
    }
    return g_enabled;
}

void katali_prof_reset(void) {
    memset(g_slot, 0, sizeof(g_slot));
}

void katali_prof_add(int phase, double seconds) {
    if (phase < 0 || phase >= KATALI_PHASE_COUNT) return;
    g_slot[phase].seconds += seconds;
}

void katali_prof_inc(int phase) {
    if (phase < 0 || phase >= KATALI_PHASE_COUNT) return;
    g_slot[phase].calls++;
}

void katali_prof_add_n(int phase, double seconds, unsigned long long calls) {
    if (phase < 0 || phase >= KATALI_PHASE_COUNT) return;
    g_slot[phase].seconds += seconds;
    g_slot[phase].calls += calls;
}

double katali_prof_seconds(int phase) {
    if (phase < 0 || phase >= KATALI_PHASE_COUNT) return 0.0;
    return g_slot[phase].seconds;
}

unsigned long long katali_prof_calls(int phase) {
    if (phase < 0 || phase >= KATALI_PHASE_COUNT) return 0;
    return g_slot[phase].calls;
}

void katali_prof_add_bytes(int phase, unsigned long long bytes) {
    if (phase < 0 || phase >= KATALI_PHASE_COUNT) return;
    g_slot[phase].bytes += bytes;
}

unsigned long long katali_prof_bytes(int phase) {
    if (phase < 0 || phase >= KATALI_PHASE_COUNT) return 0;
    return g_slot[phase].bytes;
}

const char *katali_prof_name(int phase) {
    switch (phase) {
        case KATALI_PHASE_MATVEC:     return "matvec";
        case KATALI_PHASE_SYNC:       return "sync_wait";
        case KATALI_PHASE_ATTENTION:  return "attention";
        case KATALI_PHASE_RMSNORM:    return "rmsnorm";
        case KATALI_PHASE_NORMDQ:     return "norm_dequant";
        case KATALI_PHASE_ROPE:       return "rope";
        case KATALI_PHASE_KV:         return "kv_store";
        case KATALI_PHASE_ACTIVATION: return "activation";
        case KATALI_PHASE_SAMPLER:    return "sampler";
        case KATALI_PHASE_TOKENIZE:   return "tokenize";
        case KATALI_PHASE_PREFILL:    return "prefill";
        case KATALI_PHASE_ATTN_SCORES:  return "attn_scores";
        case KATALI_PHASE_ATTN_SOFTMAX: return "attn_softmax";
        case KATALI_PHASE_ATTN_ACCUM:   return "attn_accum";
        case KATALI_PHASE_EMBED:        return "embed_lookup";
        default:                      return "?";
    }
}
