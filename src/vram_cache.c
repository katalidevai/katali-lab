/* katali-lab — VRAM expert tier (third residency level) — Apache-2.0
 *
 * See include/vram_cache.h. Nothing here is reachable unless the optional CUDA
 * backend loaded successfully, so a machine without an NVIDIA GPU never touches
 * this code path and never allocates device memory.
 */
#include "vram_cache.h"
#include "katali_cuda.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define VRAM_DEFAULT_RESERVE (512ull << 20) /* keep 512 MiB VRAM for the display */

static size_t round_pow2(size_t n) {
    size_t p = 1;
    while (p < n) p <<= 1;
    return p;
}

size_t vram_cache_budget_from_env(uint64_t free_vram_bytes) {
    const char *gb = getenv("KATALI_VRAM_GB");
    const char *mb = getenv("KATALI_VRAM_MB");
    if (gb && *gb) {
        long v = atol(gb);
        if (v <= 0) return 0;                    /* explicit disable */
        return (size_t)v * 1024u * 1024u * 1024u;
    }
    if (mb && *mb) {
        long v = atol(mb);
        if (v <= 0) return 0;
        return (size_t)v * 1024u * 1024u;
    }
    /* Auto: whatever is free minus a reserve for the desktop/compositor. */
    if (free_vram_bytes <= VRAM_DEFAULT_RESERVE) return 0;
    return (size_t)(free_vram_bytes - VRAM_DEFAULT_RESERVE);
}

int vram_cache_open(VramCache *vc, size_t budget_bytes) {
    if (!vc) return KATALI_ERR;
    memset(vc, 0, sizeof(*vc));
    if (getenv("KATALI_NO_VRAM")) return KATALI_ERR;
    const KataliCudaDeviceInfo *d = katali_cuda_device();
    if (!d) return KATALI_ERR;
    if (budget_bytes == 0)
        budget_bytes = vram_cache_budget_from_env(d->vram_free_bytes);
    if (budget_bytes < (8u << 20)) return KATALI_ERR;   /* too small to help */
    vc->budget = budget_bytes;
    /*
     * Slot count is derived from the budget, not fixed. Entries are exact-sized
     * per expert slab (~0.6-0.8 MB on the 35B), so 256 KiB per slot is a safe
     * over-estimate of the entry count the budget can hold. A fixed 4096-slot
     * table silently capped residency at ~2.4 GiB and made every later lookup
     * fall back to the CPU.
     */
    {
        size_t est = budget_bytes / (256u * 1024u);
        if (est < 1024u) est = 1024u;
        if (est > (1u << 20)) est = 1u << 20;
        vc->cap = round_pow2(est);
    }
    vc->tab = (VramEntry *)calloc(vc->cap, sizeof(VramEntry));
    if (!vc->tab) { memset(vc, 0, sizeof(*vc)); return KATALI_ERR; }
    vc->enabled = 1;
    return KATALI_OK;
}

void vram_cache_close(VramCache *vc) {
    if (!vc || !vc->tab) return;
    for (size_t i = 0; i < vc->cap; i++)
        if (vc->tab[i].key) katali_cuda_free(vc->tab[i].dev);
    free(vc->tab);
    size_t used = vc->used, budget = vc->budget;
    uint64_t h = vc->hits, m = vc->misses, e = vc->evictions, u = vc->uploads;
    memset(vc, 0, sizeof(*vc));
    (void)used; (void)budget; (void)h; (void)m; (void)e; (void)u;
}

int vram_cache_enabled(const VramCache *vc) { return vc && vc->enabled; }

static size_t hash_slot(uint64_t k, size_t cap) {
    k ^= k >> 33; k *= 0xff51afd7ed558ccdull; k ^= k >> 33;
    return (size_t)(k & (uint64_t)(cap - 1));
}

/* Find the entry for `key`, or an empty slot to insert into. */
static VramEntry *tab_find(VramCache *vc, uint64_t key, int *found) {
    size_t s = hash_slot(key, vc->cap);
    for (size_t i = 0; i < vc->cap; i++) {
        VramEntry *e = &vc->tab[(s + i) & (vc->cap - 1)];
        if (e->key == key) { *found = 1; return e; }
        if (e->key == 0) { *found = 0; return e; }
    }
    *found = 0;
    return NULL;
}

/* Drop entries (oldest first) until `need` bytes fit inside the budget. */
static int vram_evict_for(VramCache *vc, size_t need) {
    while (vc->used + need > vc->budget) {
        size_t best = vc->cap;
        uint64_t oldest = ~0ull;
        for (size_t i = 0; i < vc->cap; i++) {
            if (!vc->tab[i].key) continue;
            if (vc->tab[i].last_use < oldest) { oldest = vc->tab[i].last_use; best = i; }
        }
        if (best == vc->cap) return 0;            /* nothing left to evict */
        VramEntry *e = &vc->tab[best];
        katali_cuda_free(e->dev);
        vc->used -= e->bytes;
        e->key = 0; e->dev = NULL; e->bytes = 0; e->last_use = 0;
        vc->evictions++;
        vc->n--;
    }
    return 1;
}

void *vram_cache_get(VramCache *vc, int layer, int wkey,
                     const uint8_t *host_src, size_t nbytes) {
    if (!vc || !vc->enabled || !host_src || nbytes == 0) return NULL;
    if (nbytes > vc->budget) return NULL;         /* never thrash on one item */
    uint64_t key = ((uint64_t)(uint32_t)layer << 32) | (uint32_t)(wkey + 1);
    int found = 0;
    VramEntry *e = tab_find(vc, key, &found);
    vc->clock++;
    if (found && e && e->dev && e->bytes == nbytes) {
        e->last_use = vc->clock;
        vc->hits++;
        return e->dev;
    }
    vc->misses++;
    if (!e) return NULL;                          /* table full: fall back */
    if (!vram_evict_for(vc, nbytes)) return NULL;
    void *dev = NULL;
    if (katali_cuda_malloc(&dev, nbytes) != KATALI_OK) return NULL;
    if (katali_cuda_upload(dev, host_src, nbytes) != KATALI_OK) {
        katali_cuda_free(dev);
        return NULL;
    }
    if (e->key == 0) vc->n++;
    e->key = key; e->dev = dev; e->bytes = nbytes; e->last_use = vc->clock;
    vc->used += nbytes;
    vc->uploads++;
    return dev;
}

void vram_cache_stats(const VramCache *vc, uint64_t *hits, uint64_t *misses,
                      uint64_t *evictions, uint64_t *uploads,
                      size_t *used_bytes, size_t *budget_bytes,
                      size_t *n_entries) {
    if (!vc) return;
    if (hits) *hits = vc->hits;
    if (misses) *misses = vc->misses;
    if (evictions) *evictions = vc->evictions;
    if (uploads) *uploads = vc->uploads;
    if (used_bytes) *used_bytes = vc->used;
    if (budget_bytes) *budget_bytes = vc->budget;
    if (n_entries) *n_entries = vc->n;
}