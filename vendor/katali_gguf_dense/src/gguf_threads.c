/* Katali-GGUF CPU thread pool — Apache-2.0 */
#include "katali_gguf_threads.h"
#include "katali_gguf_prof.h"

#include <stdlib.h>
#include <string.h>

/* ==================================================================== *
 * Optional dispatch/synchronisation breakdown (KATALI_GGUF_SYNC_STATS)  *
 *                                                                       *
 * The accumulator is shared by both pool implementations; only the      *
 * Windows pool records timestamps (the POSIX branch is not compiled on  *
 * this target, so its dispatch counters simply stay at zero and the     *
 * breakdown is not printed). Off by default: one cached int test per    *
 * dispatch when disabled. See katali_gguf_threads.h for the field        *
 * meanings.                                                             *
 * ==================================================================== */
static KataliSyncStats g_sync;
static int g_sync_on = -1;

int katali_gguf_sync_stats_enabled(void) {
    if (g_sync_on < 0) {
        const char *e = getenv("KATALI_GGUF_SYNC_STATS");
        g_sync_on = (e && *e && e[0] != '0') ? 1 : 0;
    }
    return g_sync_on;
}

void katali_gguf_sync_stats_reset(void) {
    memset(&g_sync, 0, sizeof(g_sync));
}

void katali_gguf_sync_stats_get(KataliSyncStats *out) {
    if (out) *out = g_sync;
}

#ifdef _WIN32
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0600
#endif
#include <windows.h>

typedef struct Pool {
    int      inited;
    int      n_workers;       /* workers spawned (excludes caller) */
    int      requested;       /* configured thread count */
    HANDLE  *start_ev;
    HANDLE  *done_ev;
    HANDLE  *threads;
    KataliGgufParallelFn fn;
    void    *ctx;
    volatile LONG shutdown;
    /* KATALI_GGUF_SYNC_STATS only; NULL when disabled. end_at has one extra
     * slot for the caller, which is also a participant (worker 0). */
    int      stats;
    double  *wake_at;
    double  *end_at;
} Pool;

static Pool g_pool;

/* Dispatch mechanism note (measured, not assumed).
 *
 * Each dispatch costs the caller n_workers ResetEvent + n_workers SetEvent
 * syscalls (22 syscalls at 12 threads, ~28 us/dispatch under
 * KATALI_GGUF_SYNC_STATS). That cost is additive rather than hidden behind
 * worker compute, so cheaper alternatives were implemented, tested and
 * rejected on measurement:
 *
 *  - One manual-reset "gate" event instead of per-worker start events: unsafe.
 *    A gate left set lets a worker fall straight through its wait and run the
 *    next dispatch spuriously (observed as heap corruption).
 *  - One ReleaseSemaphore(n_workers) for waking, countdown for completion: not
 *    identity-safe. A fast worker can consume two tokens while a slow worker
 *    never wakes, so the caller sees "all workers finished" with one worker's
 *    row range never computed.
 *  - Countdown completion alone (InterlockedDecrement + one shared event,
 *    11 fewer syscalls per dispatch): same-binary alternating A/B gave -0.92%
 *    at 12 threads, 0/6 pairs favourable, with sync_wait rising 0.174 s ->
 *    0.289 s. Eleven workers hammering one cache line costs the straggler more
 *    than the eleven ResetEvent calls it removes, so it was reverted.
 *  - Auto-reset completion events with WaitForMultipleObjects(bWaitAll=TRUE):
 *    hangs. An auto-reset event is consumed as it signals, so with three or
 *    more of them the "all signalled" condition is never observed.
 *
 * The per-worker auto-reset start events and manual-reset completion events
 * below are what remains: identity-safe by construction, and not the dominant
 * term in the profile. */
static DWORD WINAPI pool_worker(LPVOID arg) {
    intptr_t idx = (intptr_t)arg;
    for (;;) {
        WaitForSingleObject(g_pool.start_ev[idx], INFINITE);
        if (g_pool.shutdown) return 0;
        if (g_pool.stats) g_pool.wake_at[idx] = katali_prof_now();
        g_pool.fn((int)idx + 1, g_pool.n_workers + 1, g_pool.ctx);
        if (g_pool.stats) g_pool.end_at[idx] = katali_prof_now();
        SetEvent(g_pool.done_ev[idx]);
    }
}

static int detected_threads(void) {
    const char *env = getenv("KATALI_GGUF_THREADS");
    if (env && *env) {
        int v = atoi(env);
        if (v > 0) return v;
    }
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    int n = (int)si.dwNumberOfProcessors;
    return n > 0 ? n : 1;
}

void katali_gguf_set_threads(int n) {
    const char *env = getenv("KATALI_GGUF_THREADS");
    if (n <= 0 && env && *env) n = atoi(env);
    if (n <= 0) n = detected_threads();
    if (n < 1) n = 1;
    if (n > 256) n = 256;
    /* Record the request only; the pool is resized on the next dispatch so a
     * caller is never forced to wait for a teardown it does not need. */
    g_pool.requested = n;
}

static void pool_start(int n_workers) {
    g_pool.n_workers = n_workers;
    g_pool.start_ev = (HANDLE *)calloc((size_t)n_workers, sizeof(HANDLE));
    g_pool.done_ev  = (HANDLE *)calloc((size_t)n_workers, sizeof(HANDLE));
    g_pool.threads  = (HANDLE *)calloc((size_t)n_workers, sizeof(HANDLE));
    if (!g_pool.start_ev || !g_pool.done_ev || !g_pool.threads) {
        g_pool.n_workers = 0;
        return;
    }
    for (int i = 0; i < n_workers; i++) {
        g_pool.start_ev[i] = CreateEventA(NULL, FALSE, FALSE, NULL); /* auto-reset */
        g_pool.done_ev[i]  = CreateEventA(NULL, TRUE,  FALSE, NULL); /* manual-reset */
        if (!g_pool.start_ev[i] || !g_pool.done_ev[i]) { g_pool.n_workers = i; return; }
        g_pool.threads[i] = CreateThread(NULL, 0, pool_worker,
                                         (LPVOID)(intptr_t)i, 0, NULL);
        if (!g_pool.threads[i]) { g_pool.n_workers = i; return; }
    }
    g_pool.inited = 1;

    /* Timestamp buffers are allocated once with the pool, never per dispatch. */
    g_pool.stats = katali_gguf_sync_stats_enabled();
    if (g_pool.stats) {
        g_pool.wake_at = (double *)calloc((size_t)n_workers + 1, sizeof(double));
        g_pool.end_at  = (double *)calloc((size_t)n_workers + 1, sizeof(double));
        if (!g_pool.wake_at || !g_pool.end_at) {
            free(g_pool.wake_at); free(g_pool.end_at);
            g_pool.wake_at = NULL; g_pool.end_at = NULL;
            g_pool.stats = 0;
        }
    }
}

int katali_gguf_get_threads(void) {
    if (g_pool.requested > 0) return g_pool.requested;
    return detected_threads();
}

void katali_gguf_parallel_run_n(int n_threads, KataliGgufParallelFn fn, void *ctx) {
    int want = n_threads > 0 ? n_threads : katali_gguf_get_threads();
    if (want <= 1) { fn(0, 1, ctx); return; }
    int n_workers = want - 1; /* caller participates as worker 0 */

    if (!g_pool.inited || g_pool.n_workers != n_workers) {
        katali_gguf_threads_shutdown();
        pool_start(n_workers);
        if (!g_pool.inited || g_pool.n_workers != n_workers) {
            fn(0, 1, ctx); /* pool creation failed: run on the caller */
            return;
        }
    }

    g_pool.fn = fn;
    g_pool.ctx = ctx;
    double t0 = g_pool.stats ? katali_prof_now() : 0.0;
    for (int i = 0; i < n_workers; i++) {
        ResetEvent(g_pool.done_ev[i]);
        SetEvent(g_pool.start_ev[i]);
    }
    double t_events = g_pool.stats ? katali_prof_now() : 0.0;
    fn(0, n_workers + 1, ctx); /* caller is worker 0 */
    double tw = katali_prof_enabled() ? katali_prof_now() : 0.0;
    double t_own = g_pool.stats ? tw : 0.0;
    WaitForMultipleObjects((DWORD)n_workers, g_pool.done_ev, TRUE, INFINITE);
    if (tw > 0.0)
        katali_prof_add_n(KATALI_PHASE_SYNC, katali_prof_now() - tw, 1);

    /* Per-dispatch breakdown: split the caller's wait into event signalling,
     * wake latency, participant compute span and completion spread. The caller
     * is a participant too, and its span deliberately includes the event loop
     * since that is time it is busy before reaching its own rows. */
    if (g_pool.stats) {
        double t_all = katali_prof_now();
        double wake_max = 0.0, work_max = 0.0, work_min = 1e30;
        double end_max = t_own, end_min = t_own;
        for (int i = 0; i < n_workers; i++) {
            double wk = g_pool.wake_at[i] - t0;
            double span = g_pool.end_at[i] - g_pool.wake_at[i];
            if (wk > wake_max) wake_max = wk;
            if (span > work_max) work_max = span;
            if (span < work_min) work_min = span;
            if (g_pool.end_at[i] > end_max) end_max = g_pool.end_at[i];
            if (g_pool.end_at[i] < end_min) end_min = g_pool.end_at[i];
        }
        double caller_span = t_own - t0;
        if (caller_span > work_max) work_max = caller_span;
        if (caller_span < work_min) work_min = caller_span;
        g_sync.dispatches++;
        g_sync.events_s     += t_events - t0;
        g_sync.wall_s       += t_all - t0;
        g_sync.wake_max_s   += wake_max;
        g_sync.work_max_s   += work_max;
        g_sync.work_min_s   += work_min;
        g_sync.end_spread_s += end_max - end_min;
        g_sync.residual_s   += t_all - end_max;
        g_sync.caller_wait_s += t_all - t_own;
    }
}

void katali_gguf_parallel_run(KataliGgufParallelFn fn, void *ctx) {
    katali_gguf_parallel_run_n(katali_gguf_get_threads(), fn, ctx);
}

void katali_gguf_threads_shutdown(void) {
    if (!g_pool.inited) return;
    g_pool.shutdown = 1;
    for (int i = 0; i < g_pool.n_workers; i++) SetEvent(g_pool.start_ev[i]);
    WaitForMultipleObjects((DWORD)g_pool.n_workers, g_pool.threads, TRUE, 5000);
    for (int i = 0; i < g_pool.n_workers; i++) {
        CloseHandle(g_pool.threads[i]);
        CloseHandle(g_pool.start_ev[i]);
        CloseHandle(g_pool.done_ev[i]);
    }
    free(g_pool.threads); free(g_pool.start_ev); free(g_pool.done_ev);
    free(g_pool.wake_at); free(g_pool.end_at);
    memset(&g_pool, 0, sizeof(g_pool));
}

#else /* POSIX fallback (kept for portability; Katali ships Windows-first) */

#include <pthread.h>
#include <unistd.h>

typedef struct Pool {
    int inited, n_workers, requested;
    pthread_t *threads;
    pthread_mutex_t mu;
    pthread_cond_t cond_work, cond_done;
    KataliGgufParallelFn fn;
    void *ctx;
    int generation, done, shutdown;
} Pool;

static Pool g_pool;

static void *pool_worker(void *arg) {
    int worker_id = (int)(intptr_t)arg + 1; /* caller is worker 0 */
    int last = 0;
    for (;;) {
        pthread_mutex_lock(&g_pool.mu);
        while (g_pool.generation == last && !g_pool.shutdown)
            pthread_cond_wait(&g_pool.cond_work, &g_pool.mu);
        if (g_pool.shutdown) { pthread_mutex_unlock(&g_pool.mu); return NULL; }
        int gen = g_pool.generation;
        KataliGgufParallelFn fn = g_pool.fn;
        void *ctx = g_pool.ctx;
        pthread_mutex_unlock(&g_pool.mu);
        fn(worker_id, g_pool.n_workers + 1, ctx);
        pthread_mutex_lock(&g_pool.mu);
        last = gen;
        g_pool.done++;
        if (g_pool.done >= g_pool.n_workers) pthread_cond_signal(&g_pool.cond_done);
        pthread_mutex_unlock(&g_pool.mu);
    }
}

static int detected_threads(void) {
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? (int)n : 1;
}

void katali_gguf_set_threads(int n) {
    if (n <= 0) n = detected_threads();
    if (n < 1) n = 1;
    if (n > 256) n = 256;
    g_pool.requested = n; /* pool resized on next dispatch */
}

int katali_gguf_get_threads(void) {
    return g_pool.requested > 0 ? g_pool.requested : detected_threads();
}

static void pool_start(int n_workers) {
    g_pool.n_workers = n_workers;
    g_pool.threads = (pthread_t *)calloc((size_t)n_workers, sizeof(pthread_t));
    if (!g_pool.threads) { g_pool.n_workers = 0; return; }
    pthread_mutex_init(&g_pool.mu, NULL);
    pthread_cond_init(&g_pool.cond_work, NULL);
    pthread_cond_init(&g_pool.cond_done, NULL);
    for (int i = 0; i < n_workers; i++)
        if (pthread_create(&g_pool.threads[i], NULL, pool_worker, (void *)(intptr_t)i) != 0) {
            g_pool.n_workers = i; break;
        }
    g_pool.inited = 1;
}

void katali_gguf_parallel_run_n(int n_threads, KataliGgufParallelFn fn, void *ctx) {
    int want = n_threads > 0 ? n_threads : katali_gguf_get_threads();
    if (want <= 1) { fn(0, 1, ctx); return; }
    int n_workers = want - 1;
    if (!g_pool.inited || g_pool.n_workers != n_workers) {
        katali_gguf_threads_shutdown();
        pool_start(n_workers);
        if (!g_pool.inited || g_pool.n_workers != n_workers) { fn(0, 1, ctx); return; }
    }
    pthread_mutex_lock(&g_pool.mu);
    g_pool.fn = fn; g_pool.ctx = ctx; g_pool.done = 0;
    g_pool.generation++;
    pthread_cond_broadcast(&g_pool.cond_work);
    pthread_mutex_unlock(&g_pool.mu);
    fn(0, n_workers + 1, ctx);
    double tw = katali_prof_enabled() ? katali_prof_now() : 0.0;
    pthread_mutex_lock(&g_pool.mu);
    while (g_pool.done < g_pool.n_workers) pthread_cond_wait(&g_pool.cond_done, &g_pool.mu);
    pthread_mutex_unlock(&g_pool.mu);
    if (tw > 0.0)
        katali_prof_add_n(KATALI_PHASE_SYNC, katali_prof_now() - tw, 1);
}

void katali_gguf_parallel_run(KataliGgufParallelFn fn, void *ctx) {
    katali_gguf_parallel_run_n(katali_gguf_get_threads(), fn, ctx);
}

void katali_gguf_threads_shutdown(void) {
    if (!g_pool.inited) return;
    pthread_mutex_lock(&g_pool.mu);
    g_pool.shutdown = 1;
    pthread_cond_broadcast(&g_pool.cond_work);
    pthread_mutex_unlock(&g_pool.mu);
    for (int i = 0; i < g_pool.n_workers; i++) pthread_join(g_pool.threads[i], NULL);
    pthread_mutex_destroy(&g_pool.mu);
    pthread_cond_destroy(&g_pool.cond_work);
    pthread_cond_destroy(&g_pool.cond_done);
    free(g_pool.threads);
    memset(&g_pool, 0, sizeof(g_pool));
}

#endif

