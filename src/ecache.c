#if !defined(_WIN32) && !defined(_POSIX_C_SOURCE)
#  define _POSIX_C_SOURCE 200809L
#endif
/* Elastic expert cache — katali2 ExpertCache port — Apache-2.0
 * Port #1: pin/unpin, LFU+freq eviction, workers, ensure_many, prefetch, stats.
 * Port #3: mmap-resident (owned=0) fill — no private weight malloc under Job Object.
 * Port #3b: drop_unpinned + speculative pin budget (PIN_FRAC).
 * Port #4: freq_load/save, prefetch_hottest; ensure_many→worker queue;
 *          Win32 worker wake fixed (katali2-faithful).
 * Port #6: range_fill + ensure_many adjacent-offset coalescing
 *          (KATALI_EC_RANGE_READ=1, default OFF).
 * Port #7: wait_idle + hot_resident.
 * No READY/PIPE/STREAM.
 */
#ifndef _WIN32_WINNT
#  define _WIN32_WINNT 0x0600
#endif
#include "ecache.h"
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
#else
#  include <pthread.h>
#  include <unistd.h>
#endif

#define EC_HASH_EMPTY  (-1)
#define EC_MAX_WORKERS 16
#define EC_QCAP        256
#define EC_FREQ_SATURATE   0xFFFFu
#define EC_FREQ_DECAY_MASK ((1u << 15) - 1)

typedef struct {
    int layer;
    int eid;
    uint32_t count;
} EcFreqSlot;

typedef struct {
    int layer;
    int expert_id;
} EcJob;

typedef struct {
#ifdef _WIN32
    CRITICAL_SECTION cs;
    CONDITION_VARIABLE filled;
    HANDLE wake;
    HANDLE workers[EC_MAX_WORKERS];
#else
    pthread_mutex_t mu;
    pthread_cond_t cv;
    pthread_cond_t filled;
    pthread_t workers[EC_MAX_WORKERS];
#endif
    int n_workers;
    int stop;
    EcJob queue[EC_QCAP];
    int qhead, qtail, qcount;
    ECache *cache;
} EcSync;

typedef struct {
    int *keys_layer;
    int *keys_eid;
    int *vals;
    size_t n;
} EcMap;

typedef struct {
    EcSync sync;
    EcMap *map;
    int *pin_layer;
    int *pin_eid;
    int pin_n;
    int pin_cap;
    EcFreqSlot *freq;
    size_t freq_cap;
    uint64_t use_count;
    int freq_tracking;
} EcRuntime;

static uint32_t ec_hash(int layer, int expert_id) {
    uint32_t x = (uint32_t)layer * 2654435761u ^ (uint32_t)expert_id * 2246822519u;
    return x;
}

static EcMap *map_new(size_t cap) {
    EcMap *m = (EcMap *)calloc(1, sizeof(EcMap));
    if (!m) return NULL;
    m->n = cap * 2;
    if (m->n < 16) m->n = 16;
    m->keys_layer = (int *)malloc(m->n * sizeof(int));
    m->keys_eid = (int *)malloc(m->n * sizeof(int));
    m->vals = (int *)malloc(m->n * sizeof(int));
    if (!m->keys_layer || !m->keys_eid || !m->vals) {
        free(m->keys_layer); free(m->keys_eid); free(m->vals); free(m);
        return NULL;
    }
    for (size_t i = 0; i < m->n; i++) {
        m->keys_layer[i] = EC_HASH_EMPTY;
        m->vals[i] = -1;
    }
    return m;
}

static void map_free(EcMap *m) {
    if (!m) return;
    free(m->keys_layer); free(m->keys_eid); free(m->vals); free(m);
}

static int map_find(EcMap *m, int layer, int eid) {
    if (!m) return -1;
    size_t i = ec_hash(layer, eid) % m->n;
    for (size_t n = 0; n < m->n; n++) {
        if (m->keys_layer[i] == EC_HASH_EMPTY) return -1;
        if (m->keys_layer[i] == layer && m->keys_eid[i] == eid) return m->vals[i];
        i = (i + 1) % m->n;
    }
    return -1;
}

static void map_put(EcMap *m, int layer, int eid, int idx) {
    if (!m) return;
    size_t i = ec_hash(layer, eid) % m->n;
    for (size_t n = 0; n < m->n; n++) {
        if (m->keys_layer[i] == EC_HASH_EMPTY ||
            (m->keys_layer[i] == layer && m->keys_eid[i] == eid)) {
            m->keys_layer[i] = layer;
            m->keys_eid[i] = eid;
            m->vals[i] = idx;
            return;
        }
        i = (i + 1) % m->n;
    }
}

static void map_del(EcMap *m, int layer, int eid) {
    if (!m) return;
    size_t i = ec_hash(layer, eid) % m->n;
    for (size_t n = 0; n < m->n; n++) {
        if (m->keys_layer[i] == EC_HASH_EMPTY) return;
        if (m->keys_layer[i] == layer && m->keys_eid[i] == eid) {
            m->keys_layer[i] = EC_HASH_EMPTY;
            m->vals[i] = -1;
            size_t j = (i + 1) % m->n;
            while (m->keys_layer[j] != EC_HASH_EMPTY) {
                int bl = m->keys_layer[j], be = m->keys_eid[j], bv = m->vals[j];
                m->keys_layer[j] = EC_HASH_EMPTY;
                m->vals[j] = -1;
                map_put(m, bl, be, bv);
                j = (j + 1) % m->n;
            }
            return;
        }
        i = (i + 1) % m->n;
    }
}

static EcRuntime *rt_of(ECache *c) {
    return (EcRuntime *)c->sync;
}

static int is_pinned_key(EcRuntime *rt, int layer, int eid) {
    for (int i = 0; i < rt->pin_n; i++)
        if (rt->pin_layer[i] == layer && rt->pin_eid[i] == eid) return 1;
    return 0;
}

static void remember_pin(EcRuntime *rt, int layer, int eid) {
    if (is_pinned_key(rt, layer, eid)) return;
    if (rt->pin_n >= rt->pin_cap) {
        int nc = rt->pin_cap ? rt->pin_cap * 2 : 64;
        int *nl = (int *)realloc(rt->pin_layer, (size_t)nc * sizeof(int));
        int *ne = (int *)realloc(rt->pin_eid, (size_t)nc * sizeof(int));
        if (!nl || !ne) { free(nl); free(ne); return; }
        rt->pin_layer = nl; rt->pin_eid = ne; rt->pin_cap = nc;
    }
    rt->pin_layer[rt->pin_n] = layer;
    rt->pin_eid[rt->pin_n] = eid;
    rt->pin_n++;
}

static void forget_pin(EcRuntime *rt, int layer, int eid) {
    for (int i = 0; i < rt->pin_n; i++) {
        if (rt->pin_layer[i] == layer && rt->pin_eid[i] == eid) {
            rt->pin_layer[i] = rt->pin_layer[rt->pin_n - 1];
            rt->pin_eid[i] = rt->pin_eid[rt->pin_n - 1];
            rt->pin_n--;
            return;
        }
    }
}

static uint32_t freq_count_of(EcRuntime *rt, int layer, int eid) {
    if (!rt || !rt->freq) return 0;
    size_t i = ec_hash(layer, eid) % rt->freq_cap;
    for (size_t n = 0; n < rt->freq_cap; n++) {
        if (rt->freq[i].layer == EC_HASH_EMPTY) break;
        if (rt->freq[i].layer == layer && rt->freq[i].eid == eid)
            return rt->freq[i].count;
        i = (i + 1) % rt->freq_cap;
    }
    return 0;
}

static void freq_decay(EcRuntime *rt) {
    for (size_t i = 0; i < rt->freq_cap; i++)
        if (rt->freq[i].layer != EC_HASH_EMPTY)
            rt->freq[i].count >>= 1;
}

static void freq_bump(EcRuntime *rt, int layer, int eid) {
    if (!rt || !rt->freq || !rt->freq_tracking || layer < 0 || eid < 0) return;
    int decay_now = 0;
    size_t i = ec_hash(layer, eid) % rt->freq_cap;
    for (size_t n = 0; n < rt->freq_cap; n++) {
        EcFreqSlot *s = &rt->freq[i];
        if (s->layer == EC_HASH_EMPTY) {
            s->layer = layer; s->eid = eid; s->count = 1;
            decay_now = 1; break;
        }
        if (s->layer == layer && s->eid == eid) {
            if (s->count < EC_FREQ_SATURATE) s->count++;
            decay_now = 1; break;
        }
        i = (i + 1) % rt->freq_cap;
    }
    if (!decay_now) {
        freq_decay(rt);
        i = ec_hash(layer, eid) % rt->freq_cap;
        for (size_t n = 0; n < rt->freq_cap; n++) {
            EcFreqSlot *s = &rt->freq[i];
            if (s->layer == EC_HASH_EMPTY) {
                s->layer = layer; s->eid = eid; s->count = 1; return;
            }
            if (s->layer == layer && s->eid == eid) {
                if (s->count < EC_FREQ_SATURATE) s->count++;
                return;
            }
            i = (i + 1) % rt->freq_cap;
        }
        return;
    }
    rt->use_count++;
    if ((rt->use_count & EC_FREQ_DECAY_MASK) == 0) freq_decay(rt);
}

/* Merge a loaded count: keep the larger of existing vs incoming (katali2). */
static void freq_merge(EcRuntime *rt, int layer, int eid, uint32_t count) {
    if (!rt || !rt->freq || layer < 0 || eid < 0 || count == 0) return;
    size_t i = ec_hash(layer, eid) % rt->freq_cap;
    for (size_t n = 0; n < rt->freq_cap; n++) {
        EcFreqSlot *s = &rt->freq[i];
        if (s->layer == EC_HASH_EMPTY) {
            s->layer = layer;
            s->eid = eid;
            s->count = count;
            return;
        }
        if (s->layer == layer && s->eid == eid) {
            if (count > s->count) s->count = count;
            return;
        }
        i = (i + 1) % rt->freq_cap;
    }
}


static void lock_rt(EcRuntime *rt) {
#ifdef _WIN32
    EnterCriticalSection(&rt->sync.cs);
#else
    pthread_mutex_lock(&rt->sync.mu);
#endif
}

static void unlock_rt(EcRuntime *rt) {
#ifdef _WIN32
    LeaveCriticalSection(&rt->sync.cs);
#else
    pthread_mutex_unlock(&rt->sync.mu);
#endif
}

static void broadcast_filled(EcRuntime *rt) {
#ifdef _WIN32
    WakeAllConditionVariable(&rt->sync.filled);
#else
    pthread_cond_broadcast(&rt->sync.filled);
#endif
}

static int find_entry(ECache *c, int layer, int expert_id) {
    EcRuntime *rt = rt_of(c);
    if (rt && rt->map) {
        int idx = map_find(rt->map, layer, expert_id);
        if (idx >= 0 && c->entries[idx].valid &&
            c->entries[idx].layer == layer &&
            c->entries[idx].expert_id == expert_id)
            return idx;
        return -1;
    }
    for (size_t i = 0; i < c->capacity; i++) {
        if (c->entries[i].valid &&
            c->entries[i].layer == layer &&
            c->entries[i].expert_id == expert_id)
            return (int)i;
    }
    return -1;
}

/* LFU+recency among unpinned; fallback to any non-loading (avoid soft-pin deadlock). */
static int find_lru(ECache *c) {
    EcRuntime *rt = rt_of(c);
    int best_u = -1, best_a = -1;
    uint32_t best_uf = UINT32_MAX, best_af = UINT32_MAX;
    uint64_t best_ut = UINT64_MAX, best_at = UINT64_MAX;
    for (size_t i = 0; i < c->capacity; i++) {
        ECacheEntry *e = &c->entries[i];
        if (!e->valid && !e->loading) return (int)i;
        if (e->loading) continue;
        uint32_t f = freq_count_of(rt, e->layer, e->expert_id);
        if (f < best_af || (f == best_af && e->last_use < best_at)) {
            best_af = f; best_at = e->last_use; best_a = (int)i;
        }
        if (!e->pinned && (f < best_uf || (f == best_uf && e->last_use < best_ut))) {
            best_uf = f; best_ut = e->last_use; best_u = (int)i;
        }
    }
    return best_u >= 0 ? best_u : best_a;
}

static int claim_slot(ECache *c, int layer, int expert_id) {
    int idx = find_entry(c, layer, expert_id);
    if (idx >= 0) {
        if (c->entries[idx].loading) return -2;
        if (c->entries[idx].valid) return idx;
    }
    idx = find_lru(c);
    if (idx < 0) return -1;
    EcRuntime *rt = rt_of(c);
    if (c->entries[idx].valid) {
        c->evictions++;
        if (rt && rt->map)
            map_del(rt->map, c->entries[idx].layer, c->entries[idx].expert_id);
        c->entries[idx].valid = 0;
        /* mmap-resident: drop pointer only; private: keep malloc buffer for reuse */
        if (!c->entries[idx].owned) {
            c->entries[idx].data = NULL;
            c->entries[idx].nbytes = 0;
        }
    }
    c->entries[idx].layer = layer;
    c->entries[idx].expert_id = expert_id;
    c->entries[idx].loading = 1;
    c->entries[idx].pinned = (rt && is_pinned_key(rt, layer, expert_id)) ? 1 : 0;
    return idx;
}

static void finish_slot(ECache *c, int idx, int ok) {
    EcRuntime *rt = rt_of(c);
    c->entries[idx].loading = 0;
    if (ok) {
        c->entries[idx].valid = 1;
        c->clock++;
        c->entries[idx].last_use = c->clock;
        c->bytes_loaded += c->block_bytes;
        if (rt && rt->map)
            map_put(rt->map, c->entries[idx].layer, c->entries[idx].expert_id, idx);
        if (rt && is_pinned_key(rt, c->entries[idx].layer, c->entries[idx].expert_id))
            c->entries[idx].pinned = 1;
    } else {
        c->entries[idx].valid = 0;
    }
    if (rt) broadcast_filled(rt);
}

static int queue_contains(EcSync *s, int layer, int eid) {
    for (int i = 0, idx = s->qhead; i < s->qcount; i++) {
        if (s->queue[idx].layer == layer && s->queue[idx].expert_id == eid) return 1;
        idx = (idx + 1) % EC_QCAP;
    }
    return 0;
}

static int queue_push(EcSync *s, int layer, int eid) {
    if (s->qcount >= EC_QCAP) return -1;
    if (queue_contains(s, layer, eid)) return 0;
    s->queue[s->qtail].layer = layer;
    s->queue[s->qtail].expert_id = eid;
    s->qtail = (s->qtail + 1) % EC_QCAP;
    s->qcount++;
    return 1;
}

static int queue_pop(EcSync *s, EcJob *out) {
    if (s->qcount <= 0) return 0;
    *out = s->queue[s->qhead];
    s->qhead = (s->qhead + 1) % EC_QCAP;
    s->qcount--;
    return 1;
}


/* Prefetch pages so mmap-resident workers warm the OS file cache before the
 * compute thread dereferences the expert. Windows' PrefetchVirtualMemory is
 * used dynamically because the project still targets older SDK declarations;
 * the page-touch loop remains the portable/failure fallback. */
static void touch_span(const uint8_t *p, size_t n) {
    if (!p || n == 0) return;
#ifdef _WIN32
    {
        typedef struct KataliMemoryRangeEntry {
            PVOID VirtualAddress;
            SIZE_T NumberOfBytes;
        } KataliMemoryRangeEntry;
        typedef BOOL (WINAPI *KataliPrefetchVirtualMemoryFn)(
            HANDLE, ULONG_PTR, KataliMemoryRangeEntry *, ULONG);
        HMODULE k32 = GetModuleHandleW(L"kernel32.dll");
        KataliPrefetchVirtualMemoryFn pf = NULL;
        if (k32)
            pf = (KataliPrefetchVirtualMemoryFn)(void *)GetProcAddress(
                k32, "PrefetchVirtualMemory");
        if (pf) {
            KataliMemoryRangeEntry range;
            range.VirtualAddress = (PVOID)(uintptr_t)p;
            range.NumberOfBytes = (SIZE_T)n;
            if (pf(GetCurrentProcess(), 1, &range, 0)) return;
        }
    }
#endif
    volatile uint8_t sink = 0;
    sink ^= p[0];
    if (n > 1) sink ^= p[n - 1];
    /* stride ~4KiB to fault interior pages without full scan cost */
    for (size_t off = 4096; off < n; off += 4096)
        sink ^= p[off];
    (void)sink;
}

/* Install residency into slot idx. Mmap mode: resolve/src pointer (owned=0).
 * Private mode: FillFn memcpy into owned buffer. Returns 0 on success. */
static int fill_slot(ECache *c, int idx, int layer, int expert_id,
                     ECacheFillFn fill, void *userdata,
                     const uint8_t *src_opt, size_t src_len) {
    ECacheEntry *e = &c->entries[idx];
    if (c->mmap_mode) {
        const uint8_t *ptr = NULL;
        size_t nbytes = 0;
        if (src_opt && src_len > 0) {
            ptr = src_opt;
            nbytes = src_len;
        } else if (c->resolve_fn) {
            if (c->resolve_fn(userdata ? userdata : c->default_fill_ud,
                              layer, expert_id, &ptr, &nbytes) != 0 || !ptr)
                return -1;
        } else if (c->offset_fn && c->file_base) {
            uint64_t off = c->offset_fn(userdata ? userdata : c->default_fill_ud,
                                        layer, expert_id);
            if (off >= c->file_size) return -1;
            ptr = c->file_base + off;
            nbytes = c->block_bytes;
            if (off + nbytes > c->file_size) nbytes = (size_t)(c->file_size - off);
        } else {
            return -1;
        }
        if (e->owned && e->data) {
            free(e->data);
            e->data = NULL;
        }
        e->data = (uint8_t *)(uintptr_t)ptr;
        e->nbytes = nbytes > 0 ? nbytes : c->block_bytes;
        e->owned = 0;
        touch_span(ptr, e->nbytes);
        return 0;
    }
    /* private-copy path */
    if (!e->data || !e->owned) {
        uint8_t *buf = (uint8_t *)malloc(c->block_bytes);
        if (!buf) return -1;
        if (e->data && e->owned) free(e->data);
        e->data = buf;
        e->owned = 1;
        e->nbytes = c->block_bytes;
    }
    if (src_opt && src_len > 0) {
        size_t n = src_len < c->block_bytes ? src_len : c->block_bytes;
        memcpy(e->data, src_opt, n);
        if (n < c->block_bytes) memset(e->data + n, 0, c->block_bytes - n);
        return 0;
    }
    if (!fill) return -1;
    return fill(userdata, layer, expert_id, e->data, c->block_bytes);
}

static void do_fill_job(ECache *c, int layer, int expert_id) {
    ECacheFillFn fill = c->default_fill;
    void *ud = c->default_fill_ud;
    /* Mmap mode may resolve without a FillFn memcpy; private needs FillFn. */
    if (!fill && !(c->mmap_mode && c->resolve_fn)) return;
    EcRuntime *rt = rt_of(c);
    lock_rt(rt);
    int existing = find_entry(c, layer, expert_id);
    if (existing >= 0 && c->entries[existing].valid && !c->entries[existing].loading) {
        unlock_rt(rt); return;
    }
    if (existing >= 0 && c->entries[existing].loading) {
        unlock_rt(rt); return;
    }
    int idx = claim_slot(c, layer, expert_id);
    if (idx == -2 || idx < 0) { unlock_rt(rt); return; }
    if (c->entries[idx].valid && !c->entries[idx].loading) {
        unlock_rt(rt); return;
    }
    unlock_rt(rt);
    int rc = fill_slot(c, idx, layer, expert_id, fill, ud, NULL, 0);
    lock_rt(rt);
    finish_slot(c, idx, rc == 0);
    /* Match katali2: bump LFU on successful prefetch fill so hot prefetches
     * are not immediately the coldest eviction candidates. */
    if (rc == 0) freq_bump(rt, layer, expert_id);
    c->prefetch_done++;
    unlock_rt(rt);
}

#ifdef _WIN32
static DWORD WINAPI ec_worker(LPVOID arg) {
    ECache *c = (ECache *)arg;
    EcRuntime *rt = rt_of(c);
    for (;;) {
        EcJob job;
        lock_rt(rt);
        /* katali2-faithful: wait on wake event (prefetch SetEvent), not the
         * filled CV (that signals wait_slot_ready waiters). */
        while (rt->sync.qcount == 0 && !rt->sync.stop) {
            LeaveCriticalSection(&rt->sync.cs);
            WaitForSingleObject(rt->sync.wake, 50);
            EnterCriticalSection(&rt->sync.cs);
        }
        if (rt->sync.stop && rt->sync.qcount == 0) { unlock_rt(rt); break; }
        if (!queue_pop(&rt->sync, &job)) { unlock_rt(rt); continue; }
        unlock_rt(rt);
        do_fill_job(c, job.layer, job.expert_id);
    }
    return 0;
}
#else
static void *ec_worker(void *arg) {
    ECache *c = (ECache *)arg;
    EcRuntime *rt = rt_of(c);
    for (;;) {
        EcJob job;
        lock_rt(rt);
        while (rt->sync.qcount == 0 && !rt->sync.stop)
            pthread_cond_wait(&rt->sync.cv, &rt->sync.mu);
        if (rt->sync.stop && rt->sync.qcount == 0) { unlock_rt(rt); break; }
        if (!queue_pop(&rt->sync, &job)) { unlock_rt(rt); continue; }
        unlock_rt(rt);
        do_fill_job(c, job.layer, job.expert_id);
    }
    return NULL;
}
#endif

static int env_wants_mmap(void) {
    const char *e = getenv("KATALI_ECACHE_MMAP");
    if (!e || !*e) return 0;
    return (*e == '1' || *e == 'y' || *e == 'Y' || *e == 't' || *e == 'T');
}

int ecache_init(ECache *c, size_t capacity, size_t block_bytes) {
    return ecache_init_ex(c, capacity, block_bytes, -1);
}

int ecache_init_ex(ECache *c, size_t capacity, size_t block_bytes, int mmap_mode) {
    memset(c, 0, sizeof(*c));
    if (capacity < 1 || block_bytes < 1) return KATALI_ERR;
    if (mmap_mode < 0) mmap_mode = env_wants_mmap() ? 1 : 0;
    c->capacity = capacity;
    c->block_bytes = block_bytes;
    c->pin_frac = 25;
    c->mmap_mode = mmap_mode ? 1 : 0;
    c->entries = (ECacheEntry *)calloc(capacity, sizeof(ECacheEntry));
    if (!c->entries) return KATALI_ERR_NOMEM;
    for (size_t i = 0; i < capacity; i++) {
        if (c->mmap_mode) {
            c->entries[i].data = NULL;
            c->entries[i].nbytes = 0;
            c->entries[i].owned = 0;
        } else {
            c->entries[i].data = (uint8_t *)malloc(block_bytes);
            if (!c->entries[i].data) { ecache_free(c); return KATALI_ERR_NOMEM; }
            c->entries[i].nbytes = block_bytes;
            c->entries[i].owned = 1;
        }
    }
    EcRuntime *rt = (EcRuntime *)calloc(1, sizeof(EcRuntime));
    if (!rt) { ecache_free(c); return KATALI_ERR_NOMEM; }
    rt->map = map_new(capacity);
    rt->freq_cap = capacity * 4;
    if (rt->freq_cap < 64) rt->freq_cap = 64;
    rt->freq = (EcFreqSlot *)calloc(rt->freq_cap, sizeof(EcFreqSlot));
    if (!rt->map || !rt->freq) {
        map_free(rt->map); free(rt->freq); free(rt);
        ecache_free(c); return KATALI_ERR_NOMEM;
    }
    for (size_t i = 0; i < rt->freq_cap; i++) rt->freq[i].layer = EC_HASH_EMPTY;
#ifdef _WIN32
    InitializeCriticalSection(&rt->sync.cs);
    InitializeConditionVariable(&rt->sync.filled);
    rt->sync.wake = CreateEvent(NULL, FALSE, FALSE, NULL);
#else
    pthread_mutex_init(&rt->sync.mu, NULL);
    pthread_cond_init(&rt->sync.cv, NULL);
    pthread_cond_init(&rt->sync.filled, NULL);
#endif
    rt->sync.cache = c;
    rt->freq_tracking = 1;
    c->sync = rt;
    return KATALI_OK;
}

void ecache_set_pin_frac(ECache *c, int pin_frac) {
    if (!c) return;
    if (pin_frac < 0) pin_frac = 0;
    if (pin_frac > 100) pin_frac = 100;
    c->pin_frac = pin_frac;
}

void ecache_set_freq_tracking(ECache *c, int enabled) {
    if (!c || !c->sync) return;
    EcRuntime *rt = rt_of(c);
    if (!rt) return;
    lock_rt(rt);
    rt->freq_tracking = enabled ? 1 : 0;
    unlock_rt(rt);
}

void ecache_bind_file(ECache *c, const uint8_t *base, uint64_t size) {
    if (!c) return;
    c->file_base = base;
    c->file_size = size;
}

void ecache_set_resolve_fn(ECache *c, ECacheResolveFn fn) {
    if (c) c->resolve_fn = fn;
}

void ecache_stop_workers(ECache *c) {
    if (!c || !c->sync) return;
    EcRuntime *rt = rt_of(c);
    if (rt->sync.n_workers <= 0) return;
    lock_rt(rt);
    rt->sync.stop = 1;
    unlock_rt(rt);
#ifdef _WIN32
    SetEvent(rt->sync.wake);
    WakeAllConditionVariable(&rt->sync.filled);
    for (int i = 0; i < rt->sync.n_workers; i++) {
        if (rt->sync.workers[i]) {
            WaitForSingleObject(rt->sync.workers[i], 5000);
            CloseHandle(rt->sync.workers[i]);
            rt->sync.workers[i] = NULL;
        }
    }
#else
    pthread_cond_broadcast(&rt->sync.cv);
    pthread_cond_broadcast(&rt->sync.filled);
    for (int i = 0; i < rt->sync.n_workers; i++)
        pthread_join(rt->sync.workers[i], NULL);
#endif
    rt->sync.n_workers = 0;
    c->n_workers = 0;
}

int ecache_start_workers(ECache *c, int n_workers,
                         ECacheFillFn fill, void *userdata) {
    if (!c || !c->sync || !fill) return KATALI_ERR;
    if (n_workers < 1) n_workers = 1;
    if (n_workers > EC_MAX_WORKERS) n_workers = EC_MAX_WORKERS;
    ecache_stop_workers(c);
    EcRuntime *rt = rt_of(c);
    c->default_fill = fill;
    c->default_fill_ud = userdata;
    rt->sync.stop = 0;
    rt->sync.qhead = rt->sync.qtail = rt->sync.qcount = 0;
    int launched = 0;
#ifdef _WIN32
    for (int i = 0; i < n_workers; i++) {
        HANDLE h = CreateThread(NULL, 0, ec_worker, c, 0, NULL);
        if (h) rt->sync.workers[launched++] = h;
    }
#else
    for (int i = 0; i < n_workers; i++) {
        if (pthread_create(&rt->sync.workers[launched], NULL, ec_worker, c) == 0)
            launched++;
    }
#endif
    rt->sync.n_workers = launched;
    c->n_workers = launched;
    return launched > 0 ? KATALI_OK : KATALI_ERR;
}

void ecache_set_offset_fn(ECache *c, ECacheOffsetFn fn) {
    if (c) c->offset_fn = fn;
}

void ecache_set_range_fill(ECache *c, ECacheRangeFillFn fn, void *userdata) {
    if (!c) return;
    c->range_fill = fn;
    c->range_fill_ud = userdata;
}

void ecache_free(ECache *c) {
    if (!c) return;
    ecache_stop_workers(c);
    if (c->default_fill_ud) {
        /* Host owns fill ctx lifetime; do not free here. */
    }
    if (c->sync) {
        EcRuntime *rt = rt_of(c);
        map_free(rt->map);
        free(rt->freq);
        free(rt->pin_layer); free(rt->pin_eid);
#ifdef _WIN32
        if (rt->sync.wake) CloseHandle(rt->sync.wake);
        DeleteCriticalSection(&rt->sync.cs);
#else
        pthread_mutex_destroy(&rt->sync.mu);
        pthread_cond_destroy(&rt->sync.cv);
        pthread_cond_destroy(&rt->sync.filled);
#endif
        free(rt);
        c->sync = NULL;
    }
    if (c->entries) {
        for (size_t i = 0; i < c->capacity; i++) {
            if (c->entries[i].owned) free(c->entries[i].data);
            c->entries[i].data = NULL;
        }
        free(c->entries);
    }
    memset(c, 0, sizeof(*c));
}

const uint8_t *ecache_peek(ECache *c, int layer, int expert_id) {
    if (!c || !c->sync) return NULL;
    EcRuntime *rt = rt_of(c);
    lock_rt(rt);
    int idx = find_entry(c, layer, expert_id);
    const uint8_t *p = NULL;
    if (idx >= 0 && c->entries[idx].valid && !c->entries[idx].loading) {
        c->clock++;
        c->entries[idx].last_use = c->clock;
        c->hits++;
        freq_bump(rt, layer, expert_id);
        p = c->entries[idx].data;
    }
    unlock_rt(rt);
    return p;
}

int ecache_resident(const ECache *c, int layer, int expert_id) {
    if (!c || !c->sync) return 0;
    EcRuntime *rt = rt_of((ECache *)c);
    lock_rt(rt);
    int idx = find_entry((ECache *)c, layer, expert_id);
    int ok = (idx >= 0 && c->entries[idx].valid && !c->entries[idx].loading);
    unlock_rt(rt);
    return ok;
}

void ecache_pin(ECache *c, int layer, int expert_id) {
    if (!c || !c->sync) return;
    EcRuntime *rt = rt_of(c);
    lock_rt(rt);
    remember_pin(rt, layer, expert_id);
    int idx = find_entry(c, layer, expert_id);
    if (idx >= 0) c->entries[idx].pinned = 1;
    unlock_rt(rt);
}

void ecache_unpin(ECache *c, int layer, int expert_id) {
    if (!c || !c->sync) return;
    EcRuntime *rt = rt_of(c);
    lock_rt(rt);
    forget_pin(rt, layer, expert_id);
    int idx = find_entry(c, layer, expert_id);
    if (idx >= 0) c->entries[idx].pinned = 0;
    unlock_rt(rt);
}

void ecache_unpin_all(ECache *c) {
    if (!c || !c->sync) return;
    EcRuntime *rt = rt_of(c);
    lock_rt(rt);
    rt->pin_n = 0;
    for (size_t i = 0; i < c->capacity; i++) c->entries[i].pinned = 0;
    unlock_rt(rt);
}

/* Speculative (prior-token) pin set: on by default, KATALI_EC_PIN=0 disables.
 * Read once — policy switch, not a live knob. */
int ecache_pins_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *env = getenv("KATALI_EC_PIN");
        cached = (env && env[0] == '0') ? 0 : 1;
    }
    return cached;
}

int ecache_spec_pin_limit(const ECache *c) {
    if (!ecache_pins_enabled()) return 0;
    const char *mx = getenv("KATALI_EC_PIN_MAX");
    if (mx && mx[0]) {
        int n = atoi(mx);
        return n < 0 ? 0 : n;
    }
    /* Default 25% of capacity (host_open sets c->pin_frac; env overrides).
     * KATALI_EC_PIN_FRAC=100 restores unlimited; =0 disables speculative pins. */
    int pct = (c && c->pin_frac > 0) ? c->pin_frac : 25;
    const char *fr = getenv("KATALI_EC_PIN_FRAC");
    if (fr && fr[0]) pct = atoi(fr);
    if (!c || c->capacity == 0) return 1 << 30;
    if (pct <= 0) return 0;
    if (pct >= 100) return (int)c->capacity * 1024; /* unlimited vs topk set */
    int n = (int)((c->capacity * (size_t)pct) / 100u);
    return n < 0 ? 0 : n;
}

void ecache_freq_decay(ECache *c, int rounds) {
    if (!c || !c->sync || rounds <= 0) return;
    EcRuntime *rt = rt_of(c);
    if (!rt || !rt->freq) return;
    if (rounds > 16) rounds = 16;
    lock_rt(rt);
    for (int r = 0; r < rounds; r++)
        freq_decay(rt);
    unlock_rt(rt);
}

void ecache_drop_unpinned(ECache *c) {
    if (!c || !c->sync) return;
    EcRuntime *rt = rt_of(c);
    lock_rt(rt);
    for (size_t i = 0; i < c->capacity; i++) {
        ECacheEntry *e = &c->entries[i];
        if (!e->valid || e->loading || e->pinned) continue;
        if (rt->map)
            map_del(rt->map, e->layer, e->expert_id);
        e->valid = 0;
        e->layer = EC_HASH_EMPTY;
        e->expert_id = -1;
        e->last_use = 0;
        /* mmap-resident: drop pointer; private: keep malloc buffer for reuse */
        if (!e->owned) {
            e->data = NULL;
            e->nbytes = 0;
        }
    }
    unlock_rt(rt);
}

void ecache_prefetch(ECache *c, int layer, int expert_id) {
    if (!c || !c->sync || c->n_workers <= 0) return;
    EcRuntime *rt = rt_of(c);
    lock_rt(rt);
    int idx = find_entry(c, layer, expert_id);
    if (idx >= 0 && (c->entries[idx].valid || c->entries[idx].loading)) {
        c->prefetch_hits++;
        unlock_rt(rt);
        return;
    }
    int pushed = queue_push(&rt->sync, layer, expert_id);
    if (pushed > 0) c->prefetch_issued++;
    unlock_rt(rt);
#ifdef _WIN32
    if (pushed > 0) SetEvent(rt->sync.wake);
#else
    if (pushed > 0) pthread_cond_signal(&rt->sync.cv);
#endif
}

static void wake_workers(ECache *c) {
    if (!c || !c->sync) return;
    EcRuntime *rt = rt_of(c);
#ifdef _WIN32
    SetEvent(rt->sync.wake);
#else
    pthread_cond_broadcast(&rt->sync.cv);
#endif
}

static void wait_slot_ready(ECache *c, int layer, int expert_id) {
    EcRuntime *rt = rt_of(c);
    int waited_ms = 0;
    for (;;) {
        lock_rt(rt);
        int idx = find_entry(c, layer, expert_id);
        int done = 0, loading = 0, queued = 0;
        if (idx >= 0) {
            if (c->entries[idx].valid && !c->entries[idx].loading) done = 1;
            if (c->entries[idx].loading) loading = 1;
        }
        queued = queue_contains(&rt->sync, layer, expert_id);
        if (done) { unlock_rt(rt); return; }
        if (!loading && !queued) { unlock_rt(rt); return; }
        if (waited_ms >= 30000) {
            if (loading && idx >= 0) finish_slot(c, idx, 0);
            unlock_rt(rt);
            return;
        }
#ifdef _WIN32
        SleepConditionVariableCS(&rt->sync.filled, &rt->sync.cs, 100);
#else
        {
            struct timespec ts;
            clock_gettime(CLOCK_REALTIME, &ts);
            ts.tv_nsec += 100000000L;
            if (ts.tv_nsec >= 1000000000L) { ts.tv_sec++; ts.tv_nsec -= 1000000000L; }
            pthread_cond_timedwait(&rt->sync.filled, &rt->sync.mu, &ts);
        }
#endif
        unlock_rt(rt);
        waited_ms += 100;
    }
}

const uint8_t *ecache_get_fill(ECache *c, int layer, int expert_id,
                               ECacheFillFn fill, void *userdata) {
    if (!c || !fill) return NULL;
    EcRuntime *rt = rt_of(c);
    if (!rt) return NULL;

    lock_rt(rt);
    int idx = find_entry(c, layer, expert_id);
    c->clock++;
    if (idx >= 0 && c->entries[idx].valid && !c->entries[idx].loading) {
        c->hits++;
        c->entries[idx].last_use = c->clock;
        freq_bump(rt, layer, expert_id);
        const uint8_t *p = c->entries[idx].data;
        unlock_rt(rt);
        return p;
    }
    if (idx >= 0 && c->entries[idx].loading) {
        unlock_rt(rt);
        wait_slot_ready(c, layer, expert_id);
        lock_rt(rt);
        idx = find_entry(c, layer, expert_id);
        if (idx >= 0 && c->entries[idx].valid) {
            c->hits++;
            c->entries[idx].last_use = c->clock;
            freq_bump(rt, layer, expert_id);
            const uint8_t *p = c->entries[idx].data;
            unlock_rt(rt);
            return p;
        }
        unlock_rt(rt);
        lock_rt(rt);
    }

    c->misses++;
    idx = claim_slot(c, layer, expert_id);
    if (idx == -2) {
        unlock_rt(rt);
        wait_slot_ready(c, layer, expert_id);
        return ecache_get_fill(c, layer, expert_id, fill, userdata);
    }
    if (idx < 0) { unlock_rt(rt); return NULL; }
    if (c->entries[idx].valid && !c->entries[idx].loading) {
        c->hits++;
        c->misses--;
        c->entries[idx].last_use = c->clock;
        freq_bump(rt, layer, expert_id);
        const uint8_t *p = c->entries[idx].data;
        unlock_rt(rt);
        return p;
    }
    unlock_rt(rt);

    int rc = fill_slot(c, idx, layer, expert_id, fill, userdata, NULL, 0);

    lock_rt(rt);
    finish_slot(c, idx, rc == 0);
    if (rc == 0) freq_bump(rt, layer, expert_id);
    const uint8_t *p = (rc == 0) ? c->entries[idx].data : NULL;
    unlock_rt(rt);
    return p;
}

typedef struct {
    const uint8_t *src;
    size_t len;
} EcacheSrcFill;

static int src_fill_fn(void *userdata, int layer, int expert_id,
                       uint8_t *dest, size_t nbytes) {
    (void)layer; (void)expert_id;
    EcacheSrcFill *s = (EcacheSrcFill *)userdata;
    if (!s || !s->src || s->len == 0 || !dest || nbytes == 0) return -1;
    size_t n = s->len < nbytes ? s->len : nbytes;
    memcpy(dest, s->src, n);
    if (n < nbytes) memset(dest + n, 0, nbytes - n);
    return 0;
}

const uint8_t *ecache_get(ECache *c, int layer, int expert_id,
                          const uint8_t *src, uint64_t len) {
    if (!c || !src || len == 0) return NULL;
    if (len > c->block_bytes) {
        /* Oversized weight: refuse rather than truncate silently. */
        fprintf(stderr, "ecache_get: len=%llu > block_bytes=%zu (L=%d key=%d)\n",
                (unsigned long long)len, c->block_bytes, layer, expert_id);
        return NULL;
    }
    if (c->mmap_mode) {
        /* Install src pointer as owned=0 resident — no private malloc/memcpy. */
        EcRuntime *rt = rt_of(c);
        if (!rt) return NULL;
        lock_rt(rt);
        int idx = find_entry(c, layer, expert_id);
        c->clock++;
        if (idx >= 0 && c->entries[idx].valid && !c->entries[idx].loading) {
            c->hits++;
            c->entries[idx].last_use = c->clock;
            freq_bump(rt, layer, expert_id);
            const uint8_t *p = c->entries[idx].data;
            unlock_rt(rt);
            return p;
        }
        if (idx >= 0 && c->entries[idx].loading) {
            unlock_rt(rt);
            wait_slot_ready(c, layer, expert_id);
            return ecache_get(c, layer, expert_id, src, len);
        }
        c->misses++;
        idx = claim_slot(c, layer, expert_id);
        if (idx == -2) {
            unlock_rt(rt);
            wait_slot_ready(c, layer, expert_id);
            return ecache_get(c, layer, expert_id, src, len);
        }
        if (idx < 0) { unlock_rt(rt); return NULL; }
        if (c->entries[idx].valid && !c->entries[idx].loading) {
            c->hits++;
            c->misses--;
            c->entries[idx].last_use = c->clock;
            freq_bump(rt, layer, expert_id);
            const uint8_t *p = c->entries[idx].data;
            unlock_rt(rt);
            return p;
        }
        unlock_rt(rt);
        int rc = fill_slot(c, idx, layer, expert_id, NULL, NULL, src, (size_t)len);
        lock_rt(rt);
        finish_slot(c, idx, rc == 0);
        if (rc == 0) freq_bump(rt, layer, expert_id);
        const uint8_t *p = (rc == 0) ? c->entries[idx].data : NULL;
        unlock_rt(rt);
        return p;
    }
    EcacheSrcFill s = { src, (size_t)len };
    return ecache_get_fill(c, layer, expert_id, src_fill_fn, &s);
}

typedef struct {
    ECache *c;
    ECacheFillFn fill;
    void *ud;
    int layer;
    int expert_id;
    int idx;
    uint64_t off;
    size_t nbytes; /* actual weight span (adjacency); may be < block_bytes */
} EcOneJob;

static int cmp_job_off(const void *a, const void *b) {
    const EcOneJob *ja = (const EcOneJob *)a;
    const EcOneJob *jb = (const EcOneJob *)b;
    if (ja->off < jb->off) return -1;
    if (ja->off > jb->off) return 1;
    return 0;
}

#ifdef _WIN32
static DWORD WINAPI ec_one_fill(LPVOID arg) {
    EcOneJob *j = (EcOneJob *)arg;
    int rc = fill_slot(j->c, j->idx, j->layer, j->expert_id, j->fill, j->ud, NULL, 0);
    EcRuntime *rt = rt_of(j->c);
    lock_rt(rt);
    finish_slot(j->c, j->idx, rc == 0);
    unlock_rt(rt);
    return 0;
}
#else
static void *ec_one_fill(void *arg) {
    EcOneJob *j = (EcOneJob *)arg;
    int rc = fill_slot(j->c, j->idx, j->layer, j->expert_id, j->fill, j->ud, NULL, 0);
    EcRuntime *rt = rt_of(j->c);
    lock_rt(rt);
    finish_slot(j->c, j->idx, rc == 0);
    unlock_rt(rt);
    return NULL;
}
#endif

int ecache_ensure_many(ECache *c,
                       const int *layers, const int *experts, int n,
                       ECacheFillFn fill, void *userdata) {
    /* Fill may be NULL in mmap mode when resolve_fn is set (worker/resolve path). */
    if (!c || !layers || !experts || n <= 0) return KATALI_ERR;
    if (!fill && !(c->mmap_mode && c->resolve_fn)) return KATALI_ERR;
    EcRuntime *rt = rt_of(c);

    /* 96: enough for 3 weight keys * topk (topk<=32). Port #2 three-key demand. */
    int Lloc[96], Eloc[96];
    int nn = n > 96 ? 96 : n;
    int u = 0;
    for (int i = 0; i < nn; i++) {
        int dup = 0;
        for (int j = 0; j < u; j++)
            if (Lloc[j] == layers[i] && Eloc[j] == experts[i]) { dup = 1; break; }
        if (!dup) { Lloc[u] = layers[i]; Eloc[u] = experts[i]; u++; }
    }
    n = u;

    for (int i = 0; i < n; i++)
        ecache_pin(c, Lloc[i], Eloc[i]);

    /* Port #6: opt-in contiguous range reads (katali2 ExpertRangeFillFn).
     * Default OFF — unchanged behavior. When ON and range_fill is wired,
     * take the sync claim path so adjacent offset runs can coalesce; skip
     * the async worker queue for this batch (same as katali2 ensure_many_ex). */
    {
        const char *range_env = getenv("KATALI_EC_RANGE_READ");
        int range_want = (c->range_fill && range_env && range_env[0] &&
                          range_env[0] != '0');
        if (range_want) {
            static int logged_range;
            if (!logged_range) {
                logged_range = 1;
                fprintf(stderr,
                        "ecache: range_read=ON (KATALI_EC_RANGE_READ) "
                        "residency=%s\n",
                        c->mmap_mode ? "mmap" : "private");
            }

            for (int i = 0; i < n; i++)
                wait_slot_ready(c, Lloc[i], Eloc[i]);

            EcOneJob jobs[96];
            int n_jobs = 0;
            for (int i = 0; i < n; i++) {
                lock_rt(rt);
                int idx = find_entry(c, Lloc[i], Eloc[i]);
                if (idx >= 0 && c->entries[idx].valid && !c->entries[idx].loading) {
                    c->entries[idx].pinned = 1;
                    c->clock++;
                    c->entries[idx].last_use = c->clock;
                    freq_bump(rt, Lloc[i], Eloc[i]);
                    unlock_rt(rt);
                    continue;
                }
                if (idx >= 0 && c->entries[idx].loading) {
                    unlock_rt(rt);
                    wait_slot_ready(c, Lloc[i], Eloc[i]);
                    i--;
                    continue;
                }
                c->misses++;
                idx = claim_slot(c, Lloc[i], Eloc[i]);
                if (idx == -2) {
                    unlock_rt(rt);
                    wait_slot_ready(c, Lloc[i], Eloc[i]);
                    i--;
                    continue;
                }
                if (idx < 0) { unlock_rt(rt); continue; }
                if (c->entries[idx].valid && !c->entries[idx].loading) {
                    c->misses--;
                    c->entries[idx].pinned = 1;
                    unlock_rt(rt);
                    continue;
                }
                jobs[n_jobs].c = c;
                jobs[n_jobs].fill = fill ? fill : c->default_fill;
                jobs[n_jobs].ud = userdata ? userdata : c->default_fill_ud;
                jobs[n_jobs].layer = Lloc[i];
                jobs[n_jobs].expert_id = Eloc[i];
                jobs[n_jobs].idx = idx;
                jobs[n_jobs].off = 0;
                jobs[n_jobs].nbytes = c->block_bytes;
                if (c->offset_fn)
                    jobs[n_jobs].off = c->offset_fn(jobs[n_jobs].ud, Lloc[i], Eloc[i]);
                /* Actual span length for adjacency (gate/up/down may differ). */
                if (c->resolve_fn) {
                    const uint8_t *rp = NULL;
                    size_t rn = 0;
                    if (c->resolve_fn(jobs[n_jobs].ud, Lloc[i], Eloc[i], &rp, &rn) == 0
                        && rn > 0)
                        jobs[n_jobs].nbytes = rn;
                }
                n_jobs++;
                unlock_rt(rt);
            }

            if (n_jobs == 0) return KATALI_OK;
            if (n_jobs > 1)
                qsort(jobs, (size_t)n_jobs, sizeof(EcOneJob), cmp_job_off);

            /* Sparse batch: no adjacent run — fall through to single fills
             * on the claimed jobs (do not serialize via empty range loop vs
             * parallel workers; still finish claimed slots here). */
            int has_range_group = 0;
            for (int i = 0; i + 1 < n_jobs; i++) {
                if (jobs[i + 1].off == jobs[i].off + (uint64_t)jobs[i].nbytes) {
                    has_range_group = 1;
                    break;
                }
            }

            int range_groups = 0, range_jobs = 0, range_max = 0;
            for (int i = 0; i < n_jobs; ) {
                int j = i + 1;
                if (has_range_group) {
                    while (j < n_jobs &&
                           jobs[j].off == jobs[j - 1].off + (uint64_t)jobs[j - 1].nbytes)
                        j++;
                }
                int cnt = j - i;
                if (has_range_group && cnt > 1) {
                    int ll[96], ee[96];
                    uint64_t oo[96];
                    size_t nnlen[96];
                    uint8_t *dd[96];
                    range_groups++;
                    range_jobs += cnt;
                    if (cnt > range_max) range_max = cnt;
                    for (int k = 0; k < cnt; k++) {
                        EcOneJob *q = &jobs[i + k];
                        ll[k] = q->layer;
                        ee[k] = q->expert_id;
                        oo[k] = q->off;
                        nnlen[k] = q->nbytes;
                        if (c->mmap_mode) {
                            /* Pointer install happens after span touch. */
                            dd[k] = NULL;
                        } else {
                            ECacheEntry *e = &c->entries[q->idx];
                            if (!e->data || !e->owned) {
                                uint8_t *buf = (uint8_t *)malloc(c->block_bytes);
                                if (!buf) {
                                    for (int t = 0; t < cnt; t++) {
                                        lock_rt(rt);
                                        finish_slot(c, jobs[i + t].idx, 0);
                                        unlock_rt(rt);
                                    }
                                    return KATALI_ERR_NOMEM;
                                }
                                if (e->data && e->owned) free(e->data);
                                e->data = buf;
                                e->owned = 1;
                                e->nbytes = c->block_bytes;
                            }
                            dd[k] = e->data;
                        }
                    }
                    int rc = c->range_fill(c->range_fill_ud, ll, ee, oo, nnlen, dd, cnt);
                    for (int k = 0; k < cnt; k++) {
                        EcOneJob *q = &jobs[i + k];
                        if (rc == 0 && c->mmap_mode) {
                            /* Span warmed; install mmap pointers per key. */
                            int fr = fill_slot(c, q->idx, q->layer, q->expert_id,
                                               q->fill, q->ud, NULL, 0);
                            lock_rt(rt);
                            finish_slot(c, q->idx, fr == 0);
                            if (fr == 0) freq_bump(rt, q->layer, q->expert_id);
                            unlock_rt(rt);
                        } else {
                            lock_rt(rt);
                            finish_slot(c, q->idx, rc == 0);
                            if (rc == 0) {
                                if (!c->mmap_mode)
                                    c->entries[q->idx].nbytes = q->nbytes;
                                freq_bump(rt, q->layer, q->expert_id);
                            }
                            unlock_rt(rt);
                        }
                    }
                } else {
                    /* Single-key fill (or sparse: every job). */
                    for (int k = i; k < j; k++) {
                        EcOneJob *q = &jobs[k];
                        int rc = fill_slot(c, q->idx, q->layer, q->expert_id,
                                           q->fill, q->ud, NULL, 0);
                        lock_rt(rt);
                        finish_slot(c, q->idx, rc == 0);
                        if (rc == 0) freq_bump(rt, q->layer, q->expert_id);
                        unlock_rt(rt);
                    }
                }
                i = j;
            }
            if (getenv("KATALI_EC_RANGE_TRACE"))
                fprintf(stderr,
                        "ecache: range groups=%d jobs=%d max=%d total=%d "
                        "has_adj=%d\n",
                        range_groups, range_jobs, range_max, n_jobs,
                        has_range_group);
            return KATALI_OK;
        }
    }

    /* Port #4: when async workers are alive, queue cold misses onto the same
     * prefetch pool katali2 uses (increments prefetch_issued/done). Sync
     * CreateThread/pthread batch remains the fallback when workers==0. */
    if (c->n_workers > 0 && c->sync) {
        int queued_any = 0;
        for (int i = 0; i < n; i++) {
            lock_rt(rt);
            int idx = find_entry(c, Lloc[i], Eloc[i]);
            if (idx >= 0 && c->entries[idx].valid && !c->entries[idx].loading) {
                c->entries[idx].pinned = 1;
                c->clock++;
                c->entries[idx].last_use = c->clock;
                freq_bump(rt, Lloc[i], Eloc[i]);
                unlock_rt(rt);
                continue;
            }
            if (idx >= 0 && c->entries[idx].loading) {
                unlock_rt(rt);
                continue; /* wait below */
            }
            if (queue_contains(&rt->sync, Lloc[i], Eloc[i])) {
                unlock_rt(rt);
                continue;
            }
            c->misses++;
            int pushed = queue_push(&rt->sync, Lloc[i], Eloc[i]);
            if (pushed > 0) {
                c->prefetch_issued++;
                queued_any = 1;
                unlock_rt(rt);
                wake_workers(c); /* one SetEvent per job (auto-reset wake) */
            } else {
                /* Queue full / dup: fall through to sync claim on this key. */
                unlock_rt(rt);
                /* Re-run single-key sync fill via claim path below by marking
                 * with a sentinel: push into sync job list. */
                lock_rt(rt);
                idx = find_entry(c, Lloc[i], Eloc[i]);
                if (idx >= 0 && (c->entries[idx].valid || c->entries[idx].loading)) {
                    unlock_rt(rt);
                    continue;
                }
                idx = claim_slot(c, Lloc[i], Eloc[i]);
                if (idx < 0 || idx == -2) { unlock_rt(rt); continue; }
                if (c->entries[idx].valid && !c->entries[idx].loading) {
                    unlock_rt(rt);
                    continue;
                }
                unlock_rt(rt);
                int rc = fill_slot(c, idx, Lloc[i], Eloc[i],
                                   fill ? fill : c->default_fill,
                                   userdata ? userdata : c->default_fill_ud,
                                   NULL, 0);
                lock_rt(rt);
                finish_slot(c, idx, rc == 0);
                if (rc == 0) {
                    freq_bump(rt, Lloc[i], Eloc[i]);
                    c->prefetch_done++; /* count as ensure fill completion */
                }
                unlock_rt(rt);
            }
        }
        if (queued_any) wake_workers(c);
        for (int i = 0; i < n; i++)
            wait_slot_ready(c, Lloc[i], Eloc[i]);
        return KATALI_OK;
    }

    /* workers==0: original sync parallel fill (katali2 ensure_many batch). */
    for (int i = 0; i < n; i++)
        wait_slot_ready(c, Lloc[i], Eloc[i]);

    EcOneJob jobs[96];
    int n_jobs = 0;
    for (int i = 0; i < n; i++) {
        lock_rt(rt);
        int idx = find_entry(c, Lloc[i], Eloc[i]);
        if (idx >= 0 && c->entries[idx].valid && !c->entries[idx].loading) {
            c->entries[idx].pinned = 1;
            c->clock++;
            c->entries[idx].last_use = c->clock;
            freq_bump(rt, Lloc[i], Eloc[i]);
            unlock_rt(rt);
            continue;
        }
        if (idx >= 0 && c->entries[idx].loading) {
            unlock_rt(rt);
            if (n_jobs == 0) { wait_slot_ready(c, Lloc[i], Eloc[i]); i--; }
            continue;
        }
        c->misses++;
        idx = claim_slot(c, Lloc[i], Eloc[i]);
        if (idx == -2) {
            unlock_rt(rt);
            if (n_jobs == 0) { wait_slot_ready(c, Lloc[i], Eloc[i]); i--; }
            continue;
        }
        if (idx < 0) { unlock_rt(rt); continue; }
        if (c->entries[idx].valid && !c->entries[idx].loading) {
            c->misses--;
            c->entries[idx].pinned = 1;
            unlock_rt(rt);
            continue;
        }
        jobs[n_jobs].c = c;
        jobs[n_jobs].fill = fill ? fill : c->default_fill;
        jobs[n_jobs].ud = userdata ? userdata : c->default_fill_ud;
        jobs[n_jobs].layer = Lloc[i];
        jobs[n_jobs].expert_id = Eloc[i];
        jobs[n_jobs].idx = idx;
        jobs[n_jobs].off = c->offset_fn ? c->offset_fn(jobs[n_jobs].ud, Lloc[i], Eloc[i]) : 0;
        jobs[n_jobs].nbytes = c->block_bytes;
        n_jobs++;
        unlock_rt(rt);
    }

    if (n_jobs == 0) return KATALI_OK;
    if (n_jobs > 1)
        qsort(jobs, (size_t)n_jobs, sizeof(EcOneJob), cmp_job_off);

#ifdef _WIN32
    HANDLE hs[64];
    int nh = 0;
    for (int i = 0; i < n_jobs; i++) {
        HANDLE h = CreateThread(NULL, 0, ec_one_fill, &jobs[i], 0, NULL);
        if (h) hs[nh++] = h;
        else ec_one_fill(&jobs[i]);
    }
    if (nh > 0) {
        WaitForMultipleObjects((DWORD)nh, hs, TRUE, INFINITE);
        for (int i = 0; i < nh; i++) CloseHandle(hs[i]);
    }
#else
    pthread_t ts[64];
    int nt = 0;
    for (int i = 0; i < n_jobs; i++) {
        if (pthread_create(&ts[nt], NULL, ec_one_fill, &jobs[i]) == 0) nt++;
        else ec_one_fill(&jobs[i]);
    }
    for (int i = 0; i < nt; i++) pthread_join(ts[i], NULL);
#endif
    return KATALI_OK;
}

#define HOTFILE_MAGIC   0x4648544Bu /* "KTHF" little-endian */
#define HOTFILE_VERSION 1u

int ecache_freq_load(ECache *c, const char *path) {
    if (!c || !path || !c->sync) return -1;
    EcRuntime *rt = rt_of(c);
    if (!rt || !rt->freq) return -1;
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    uint32_t hdr[3];
    if (fread(hdr, sizeof(uint32_t), 3, f) != 3 ||
        hdr[0] != HOTFILE_MAGIC || hdr[1] != HOTFILE_VERSION) {
        fclose(f);
        return -1;
    }
    uint32_t n = hdr[2];
    if (n > (1u << 20)) { fclose(f); return -1; }
    int merged = 0;
    lock_rt(rt);
    for (uint32_t i = 0; i < n; i++) {
        int32_t rec[3]; /* layer, eid, count */
        if (fread(rec, sizeof(int32_t), 3, f) != 3) break;
        if (rec[0] < 0 || rec[1] < 0 || rec[2] <= 0) continue;
        freq_merge(rt, rec[0], rec[1], (uint32_t)rec[2]);
        merged++;
    }
    unlock_rt(rt);
    fclose(f);
    return merged;
}

int ecache_freq_save(ECache *c, const char *path) {
    if (!c || !path || !c->sync) return -1;
    EcRuntime *rt = rt_of(c);
    if (!rt || !rt->freq) return -1;

    int32_t *rec = NULL;
    size_t n = 0;
    lock_rt(rt);
    for (size_t i = 0; i < rt->freq_cap; i++)
        if (rt->freq[i].layer != EC_HASH_EMPTY && rt->freq[i].count > 0)
            n++;
    if (n > 0) {
        rec = (int32_t *)malloc(n * 3 * sizeof(int32_t));
        if (rec) {
            size_t o = 0;
            for (size_t i = 0; i < rt->freq_cap && o < n * 3; i++) {
                if (rt->freq[i].layer == EC_HASH_EMPTY || rt->freq[i].count == 0)
                    continue;
                rec[o++] = rt->freq[i].layer;
                rec[o++] = rt->freq[i].eid;
                rec[o++] = (int32_t)rt->freq[i].count;
            }
            n = o / 3;
        } else {
            n = 0;
        }
    }
    unlock_rt(rt);

    char tmp[1400];
    snprintf(tmp, sizeof(tmp), "%s.tmp", path);
    FILE *f = fopen(tmp, "wb");
    if (!f) { free(rec); return -1; }
    uint32_t hdr[3] = { HOTFILE_MAGIC, HOTFILE_VERSION, (uint32_t)n };
    int bad = fwrite(hdr, sizeof(uint32_t), 3, f) != 3;
    if (!bad && n > 0)
        bad = fwrite(rec, sizeof(int32_t), n * 3, f) != n * 3;
    if (fclose(f) != 0) bad = 1;
    free(rec);
    if (bad) { remove(tmp); return -1; }

#ifdef _WIN32
    if (!MoveFileExA(tmp, path, MOVEFILE_REPLACE_EXISTING)) {
        remove(tmp);
        return -1;
    }
#else
    if (rename(tmp, path) != 0) { remove(tmp); return -1; }
#endif
    return (int)n;
}

typedef struct { int layer; int eid; uint32_t count; } EcHot;
static int cmp_hot_desc(const void *a, const void *b) {
    const EcHot *x = (const EcHot *)a;
    const EcHot *y = (const EcHot *)b;
    if (x->count < y->count) return 1;
    if (x->count > y->count) return -1;
    return 0;
}

int ecache_prefetch_hottest(ECache *c, size_t max_n) {
    if (!c || !c->sync || c->n_workers <= 0) return 0;
    EcRuntime *rt = rt_of(c);
    if (!rt || !rt->freq) return 0;
    if (max_n == 0) max_n = c->capacity;
    if (max_n > 4096) max_n = 4096;

    EcHot *hot = (EcHot *)malloc(rt->freq_cap * sizeof(EcHot));
    if (!hot) return 0;
    size_t n = 0;
    lock_rt(rt);
    for (size_t i = 0; i < rt->freq_cap; i++) {
        if (rt->freq[i].layer == EC_HASH_EMPTY || rt->freq[i].count == 0)
            continue;
        hot[n].layer = rt->freq[i].layer;
        hot[n].eid = rt->freq[i].eid;
        hot[n].count = rt->freq[i].count;
        n++;
    }
    unlock_rt(rt);
    if (n == 0) { free(hot); return 0; }
    qsort(hot, n, sizeof(EcHot), cmp_hot_desc);
    if (n > max_n) n = max_n;
    int queued = 0;
    for (size_t i = 0; i < n; i++) {
        ecache_prefetch(c, hot[i].layer, hot[i].eid);
        queued++;
    }
    free(hot);
    return queued;
}


size_t ecache_hot_resident(const ECache *cc, uint32_t min_count) {
    ECache *c = (ECache *)cc;
    if (!c || !c->sync) return 0;
    EcRuntime *rt = rt_of(c);
    if (!rt) return 0;
    size_t n = 0;
    lock_rt(rt);
    for (size_t i = 0; i < c->capacity; i++) {
        ECacheEntry *e = &c->entries[i];
        if (e->valid && !e->loading &&
            freq_count_of(rt, e->layer, e->expert_id) >= min_count)
            n++;
    }
    unlock_rt(rt);
    return n;
}

int ecache_wait_idle(ECache *c, int timeout_ms) {
    if (!c || !c->sync) return 0;
    EcRuntime *rt = rt_of(c);
    if (!rt) return 0;

#ifdef _WIN32
    ULONGLONG deadline = 0;
    if (timeout_ms >= 0)
        deadline = GetTickCount64() + (ULONGLONG)timeout_ms;
#else
    struct timespec deadline;
    int have_deadline = 0;
    if (timeout_ms >= 0) {
        clock_gettime(CLOCK_MONOTONIC, &deadline);
        deadline.tv_sec += timeout_ms / 1000;
        deadline.tv_nsec += (long)(timeout_ms % 1000) * 1000000L;
        if (deadline.tv_nsec >= 1000000000L) {
            deadline.tv_sec++;
            deadline.tv_nsec -= 1000000000L;
        }
        have_deadline = 1;
    }
#endif

    for (;;) {
        lock_rt(rt);
        int busy = rt->sync.qcount;
        for (size_t i = 0; i < c->capacity; i++)
            if (c->entries[i].loading) busy++;
        if (busy == 0) { unlock_rt(rt); return 0; }
        if (timeout_ms == 0) { unlock_rt(rt); return busy; }
        if (timeout_ms > 0) {
#ifdef _WIN32
            if (GetTickCount64() >= deadline) { unlock_rt(rt); return busy; }
#else
            struct timespec now;
            clock_gettime(CLOCK_MONOTONIC, &now);
            if (now.tv_sec > deadline.tv_sec ||
                (now.tv_sec == deadline.tv_sec && now.tv_nsec >= deadline.tv_nsec)) {
                unlock_rt(rt);
                return busy;
            }
            (void)have_deadline;
#endif
        }
#ifdef _WIN32
        SleepConditionVariableCS(&rt->sync.filled, &rt->sync.cs, 50);
#else
        {
            struct timespec ts;
            clock_gettime(CLOCK_REALTIME, &ts);
            ts.tv_nsec += 50000000L;
            if (ts.tv_nsec >= 1000000000L) { ts.tv_sec++; ts.tv_nsec -= 1000000000L; }
            pthread_cond_timedwait(&rt->sync.filled, &rt->sync.mu, &ts);
        }
#endif
        unlock_rt(rt);
    }
}

void ecache_reset_stats(ECache *c) {
    if (!c) return;
    c->hits = c->misses = c->evictions = c->bytes_loaded = 0;
    c->prefetch_hits = c->prefetch_issued = c->prefetch_done = 0;
}

void ecache_stats(const ECache *c, size_t *used, size_t *cap, int *n_resident) {
    size_t u = 0;
    int n = 0;
    if (c && c->entries) {
        for (size_t i = 0; i < c->capacity; i++) {
            if (c->entries[i].valid && !c->entries[i].loading) {
                n++;
                u += c->block_bytes;
            }
        }
    }
    if (used) *used = u;
    if (cap) *cap = c ? c->capacity * c->block_bytes : 0;
    if (n_resident) *n_resident = n;
}

void ecache_stats_ex(const ECache *c,
                     uint64_t *hits, uint64_t *misses, uint64_t *evictions,
                     uint64_t *bytes_loaded) {
    if (hits) *hits = c ? c->hits : 0;
    if (misses) *misses = c ? c->misses : 0;
    if (evictions) *evictions = c ? c->evictions : 0;
    if (bytes_loaded) *bytes_loaded = c ? c->bytes_loaded : 0;
}
