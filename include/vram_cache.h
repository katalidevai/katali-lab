#ifndef KATALI_VRAM_CACHE_H
#define KATALI_VRAM_CACHE_H
#include <stddef.h>
#include <stdint.h>
#include "katali.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * VRAM expert tier — the third residency level (SSD -> RAM -> VRAM).
 *
 * Deliberately mirrors the existing ECache contract so it slots in beside it:
 * same (layer, key) keyspace with key = expert*3 + which, same LRU-flavoured
 * eviction, same "logical resident bytes" accounting. The difference is only
 * WHERE the bytes live: a cudaMalloc'ed device allocation instead of
 * malloc/mmap.
 *
 * Everything here is a hard no-op when CUDA is unavailable: vram_cache_open()
 * fails, vram_cache_enabled() stays 0, and every vram_cache_get() returns NULL,
 * so callers take their existing CPU path unchanged.
 *
 * Keys have different sizes (gate/up ~0.6 MB, down ~0.8 MB on the 35B), so
 * allocations are exact-sized per entry rather than fixed-slot.
 */
typedef struct VramEntry {
    uint64_t key;       /* 0 = empty slot; else (layer << 32) | wkey + 1 */
    void    *dev;
    size_t   bytes;
    uint64_t last_use;
} VramEntry;

typedef struct VramCache {
    VramEntry *tab;      /* open-addressing hash table */
    size_t     cap;      /* power of two */
    size_t     n;        /* live entries */
    size_t     budget;   /* device byte ceiling */
    size_t     used;     /* device bytes currently allocated */
    uint64_t   clock;
    uint64_t   hits, misses, evictions, uploads;
    int        enabled;
} VramCache;

/* budget_bytes == 0 selects an automatic budget from free VRAM.
 * Returns KATALI_OK when the tier is live, KATALI_ERR otherwise (never fatal). */
KATALI_API int  vram_cache_open(VramCache *vc, size_t budget_bytes);
KATALI_API void vram_cache_close(VramCache *vc);
KATALI_API int  vram_cache_enabled(const VramCache *vc);
/* KATALI_VRAM_GB / KATALI_VRAM_MB override; 0 disables the tier explicitly. */
KATALI_API size_t vram_cache_budget_from_env(uint64_t free_vram_bytes);

/* Device pointer holding `host_src[0..nbytes)`, uploading on miss and evicting
 * LRU entries when over budget. NULL means "use the CPU path". */
KATALI_API void *vram_cache_get(VramCache *vc, int layer, int wkey,
                                const uint8_t *host_src, size_t nbytes);
KATALI_API void vram_cache_stats(const VramCache *vc,
                                 uint64_t *hits, uint64_t *misses,
                                 uint64_t *evictions, uint64_t *uploads,
                                 size_t *used_bytes, size_t *budget_bytes,
                                 size_t *n_entries);

#ifdef __cplusplus
}
#endif
#endif /* KATALI_VRAM_CACHE_H */