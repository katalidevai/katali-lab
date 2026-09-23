/* Katali-GGUF CPU thread pool — Apache-2.0
 *
 * A small persistent worker pool used to partition row-parallel work (GEMV,
 * attention heads). Each worker writes only its own output slice, so there are
 * no shared writes and no locks on the hot path. Follows the same shape as
 * katali2's katali_threads.h so the two pools can be consolidated later.
 */
#ifndef KATALI_GGUF_THREADS_H
#define KATALI_GGUF_THREADS_H

#ifdef __cplusplus
extern "C" {
#endif

/* 0 => auto (processor count). Clamped to [1, 256]. */
void katali_gguf_set_threads(int n);
int  katali_gguf_get_threads(void);

typedef void (*KataliGgufParallelFn)(int worker_id, int n_workers, void *ctx);

/* Run fn once per worker with the current thread count. When the count is 1 the
 * call is inlined on the calling thread (no pool interaction). */
void katali_gguf_parallel_run(KataliGgufParallelFn fn, void *ctx);

/* Same, but with an explicit thread count for this dispatch. The pool is
 * (re)sized only when the count actually changes, so a caller that passes the
 * same n repeatedly never pays a pool rebuild. n <= 0 means "current". */
void katali_gguf_parallel_run_n(int n_threads, KataliGgufParallelFn fn, void *ctx);

/* Release pool threads (application shutdown). Safe to call repeatedly. */
void katali_gguf_threads_shutdown(void);

/* --- optional dispatch/synchronisation breakdown -------------------------
 * Off unless KATALI_GGUF_SYNC_STATS=1 (resolved once, cached). When off a
 * dispatch pays only one cached int test, so clean benchmark runs are
 * unaffected. When on, each dispatch records per-worker wake and completion
 * timestamps so the caller's wait can be split into the parts that matter:
 *
 *   events        caller's ResetEvent/SetEvent loop (n_workers pairs)
 *   wake_max      slowest worker's latency from dispatch start to resuming
 *   work_max/min  slowest/fastest participant's compute span
 *   end_spread    max end - min end across participants (row imbalance)
 *   residual      all-done minus the last participant's end (event latency)
 *   caller_wait   the existing KATALI_PHASE_SYNC number, reproduced here
 *
 * Two QueryPerformanceCounter reads per worker per dispatch (~40 ns) against
 * dispatches that run 50-200 us, but it is still off by default. */
typedef struct KataliSyncStats {
    unsigned long long dispatches;
    double events_s;      /* summed caller event-signalling loop */
    double wall_s;        /* summed dispatch start -> all workers done */
    double wake_max_s;    /* summed per-dispatch max worker wake latency */
    double work_max_s;    /* summed per-dispatch max participant span */
    double work_min_s;    /* summed per-dispatch min participant span */
    double end_spread_s;  /* summed per-dispatch max_end - min_end */
    double residual_s;    /* summed per-dispatch all_done - max_end */
    double caller_wait_s; /* summed per-dispatch wait after the caller's rows */
} KataliSyncStats;

int  katali_gguf_sync_stats_enabled(void);
void katali_gguf_sync_stats_reset(void);
void katali_gguf_sync_stats_get(KataliSyncStats *out);

#ifdef __cplusplus
}
#endif
#endif /* KATALI_GGUF_THREADS_H */
