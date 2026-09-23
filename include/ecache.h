#ifndef KATALI_ECACHE_H
#define KATALI_ECACHE_H
#include <stddef.h>
#include <stdint.h>
#include "katali.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Elastic expert cache — port of katali2 ExpertCache (pre-RAC core).
 *
 * Key space: THREE keys per routed expert —
 *   key = expert * 3 + which   (0=gate, 1=up, 2=down)
 * Port #2: moe_ffn issues 3*topk weight keys into ensure_many.
 *
 * Residency models:
 *   owned=1 (default): GGUF mmap → private memcpy into malloc'd slot buffers
 *   owned=0 (mmap):    entry.data points INTO the GGUF mmap (no weight malloc)
 *                      KATALI_ECACHE_MMAP=1, or automatic NOMEM fallback
 *
 * Accounting: used/cap still count LOGICAL resident bytes (n_valid * block_bytes)
 * for pin/LFU/ensure policy. Under Job Object, private copies double-charge;
 * mmap mode does not malloc weight blobs — OS WS grows on fault/touch.
 *
 * Port #3b: drop_unpinned, pins_enabled, spec_pin_limit (PIN_FRAC / PIN_MAX).
 * Port #4: freq_load/save + prefetch_hottest; ensure_many queues worker fills.
 * Port #6: ECacheRangeFillFn + ensure_many coalescing (KATALI_EC_RANGE_READ=1).
 * Port #7: wait_idle + hot_resident (warmup drain + hot-set diagnostics).
 * Skip: READY/PIPE/STREAM, ensure_many_ex flags, packed one-key.
 */

typedef struct ECacheEntry {
    int      layer;
    int      expert_id; /* weight key: expert*3+which */
    uint8_t *data;
    size_t   nbytes;
    uint64_t last_use;
    int      valid;
    int      pinned;
    int      loading;
    int      owned; /* 1 = free(data) on evict/free; 0 = mmap pointer, just clear */
} ECacheEntry;

typedef struct ECache ECache;

typedef int (*ECacheFillFn)(void *userdata, int layer, int expert_id,
                            uint8_t *dest, size_t nbytes);
typedef uint64_t (*ECacheOffsetFn)(void *userdata, int layer, int expert_id);
/* Resolve weight span in-place (mmap-resident). out_ptr into GGUF map. */
typedef int (*ECacheResolveFn)(void *userdata, int layer, int expert_id,
                               const uint8_t **out_ptr, size_t *out_nbytes);
/* Contiguous range fill: one read/touch of [min(off), max(off+len)), then
 * split into dests[i] (private). Mmap mode may pass dests[i]=NULL and only
 * warm the span; caller then resolve_fn installs pointers. */
typedef int (*ECacheRangeFillFn)(void *userdata,
                                 const int *layers, const int *expert_ids,
                                 const uint64_t *offsets, const size_t *lengths,
                                 uint8_t *const *dests, int n);

struct ECache {
    ECacheEntry *entries;
    size_t capacity;
    size_t block_bytes;
    int pin_frac; /* 0..100 speculative pin budget helper */
    int mmap_mode; /* 1 = store mmap pointers (owned=0); 0 = private copy */
    uint64_t clock;
    uint64_t hits;
    uint64_t misses;
    uint64_t evictions;
    uint64_t bytes_loaded;
    uint64_t prefetch_hits;
    uint64_t prefetch_issued;
    uint64_t prefetch_done;
    void *sync;
    int n_workers;
    ECacheFillFn default_fill;
    void *default_fill_ud;
    ECacheOffsetFn offset_fn;
    ECacheResolveFn resolve_fn;
    ECacheRangeFillFn range_fill;
    void *range_fill_ud;
    /* retained for diagnostics / bind_file compat */
    const uint8_t *file_base;
    uint64_t       file_size;
};

/* katali2-style: capacity = #slots, block_bytes = max weight blob size.
 * ecache_init reads KATALI_ECACHE_MMAP (1/y/Y → mmap mode). */
KATALI_API int  ecache_init(ECache *c, size_t capacity, size_t block_bytes);
/* mmap_mode: 0=private, 1=mmap pointers, -1=honor env KATALI_ECACHE_MMAP. */
KATALI_API int  ecache_init_ex(ECache *c, size_t capacity, size_t block_bytes,
                               int mmap_mode);
KATALI_API void ecache_set_pin_frac(ECache *c, int pin_frac);
/* Enable/disable LFU frequency updates. Cache residency and LRU timestamps
 * continue to work while disabled; intended for prompt-prefill isolation. */
KATALI_API void ecache_set_freq_tracking(ECache *c, int enabled);
KATALI_API void ecache_bind_file(ECache *c, const uint8_t *base, uint64_t size);
KATALI_API void ecache_set_resolve_fn(ECache *c, ECacheResolveFn fn);
KATALI_API void ecache_free(ECache *c);

KATALI_API int  ecache_start_workers(ECache *c, int n_workers,
                                     ECacheFillFn fill, void *userdata);
KATALI_API void ecache_stop_workers(ECache *c);
KATALI_API void ecache_set_offset_fn(ECache *c, ECacheOffsetFn fn);
KATALI_API void ecache_set_range_fill(ECache *c, ECacheRangeFillFn fn,
                                      void *userdata);

/* Sync get with FillFn (katali2). In mmap mode uses resolve_fn if set. */
KATALI_API const uint8_t *ecache_get_fill(ECache *c, int layer, int expert_id,
                                          ECacheFillFn fill, void *userdata);

/*
 * Compat sync path for moe_ffn / probe.
 * Private mode: memcpy src into owned slot.
 * Mmap mode: store src pointer (owned=0); no malloc/memcpy of weights.
 * Pointer valid until eviction (mmap lifetime = model open).
 */
KATALI_API const uint8_t *ecache_get(ECache *c, int layer, int expert_id,
                                     const uint8_t *src, uint64_t len);

KATALI_API const uint8_t *ecache_peek(ECache *c, int layer, int expert_id);
KATALI_API int  ecache_resident(const ECache *c, int layer, int expert_id);

KATALI_API void ecache_pin(ECache *c, int layer, int expert_id);
KATALI_API void ecache_unpin(ECache *c, int layer, int expert_id);
KATALI_API void ecache_unpin_all(ECache *c);

/* Speculative prior-token pin set (KATALI_EC_PIN=0 disables).
 * Fills inside ensure_many always pin. */
KATALI_API int  ecache_pins_enabled(void);
/* Speculative pin budget: KATALI_EC_PIN=0 → 0; KATALI_EC_PIN_MAX=N;
 * KATALI_EC_PIN_FRAC=1..100 (else c->pin_frac, default 25% of capacity). */
KATALI_API int  ecache_spec_pin_limit(const ECache *c);
/* Halve freq-table counts `rounds` times (post-prefill LFU reset). */
KATALI_API void ecache_freq_decay(ECache *c, int rounds);
/* Invalidate resident blocks that are not soft-pinned (post-prefill). */
KATALI_API void ecache_drop_unpinned(ECache *c);

KATALI_API void ecache_prefetch(ECache *c, int layer, int expert_id);
KATALI_API int  ecache_ensure_many(ECache *c,
                                   const int *layers, const int *experts, int n,
                                   ECacheFillFn fill, void *userdata);

/* Hot-expert freq persistence (katali2 expert_cache_freq_*).
 *   freq_load: merge counts from file (entries merged, -1 if absent/bad)
 *   freq_save: atomic write (entries written, -1 error)
 *   prefetch_hottest: non-blocking queue of hottest keys (jobs queued) */
KATALI_API int  ecache_freq_load(ECache *c, const char *path);
KATALI_API int  ecache_freq_save(ECache *c, const char *path);
KATALI_API int  ecache_prefetch_hottest(ECache *c, size_t max_n);

/* Port #7: drain prefetch/ensure worker queue (+ in-flight loads).
 * Returns leftover busy count (0 = idle). timeout_ms < 0 waits forever;
 * timeout_ms == 0 is a non-blocking poll. */
KATALI_API int  ecache_wait_idle(ECache *c, int timeout_ms);
/* Resident slots whose freq count >= min_count (katali2 hot_resident). */
KATALI_API size_t ecache_hot_resident(const ECache *c, uint32_t min_count);

KATALI_API void ecache_reset_stats(ECache *c);
/* used/cap = logical resident bytes (policy); n_resident = valid slots */
KATALI_API void ecache_stats(const ECache *c, size_t *used, size_t *cap, int *n_resident);
KATALI_API void ecache_stats_ex(const ECache *c,
                                uint64_t *hits, uint64_t *misses, uint64_t *evictions,
                                uint64_t *bytes_loaded);

#ifdef __cplusplus
}
#endif
#endif
