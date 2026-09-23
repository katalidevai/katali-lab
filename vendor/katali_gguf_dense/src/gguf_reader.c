/* Katali-GGUF reader — Apache-2.0
 *
 * Original implementation of the public GGUF container format. No third-party
 * inference or container code is used. Every file-provided offset/length is
 * validated against the real file size before use.
 */
#include "katali_gguf.h"
#include "katali_gguf_dtype.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <stdint.h>
#include <limits.h>

#ifdef _WIN32
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0600
#endif
#include <windows.h>
static double gguf_now_s(void) {
    static LARGE_INTEGER freq;
    static int have = 0;
    if (!have) { QueryPerformanceFrequency(&freq); have = 1; }
    LARGE_INTEGER c;
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)freq.QuadPart;
}
#else
#include <time.h>
static double gguf_now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
#endif

/* ==================================================================== *
 * Platform file mapping                                                 *
 * ==================================================================== */
#ifdef _WIN32
#include <windows.h>
#endif

/* 64-bit file position helpers.
 *
 * On Windows, fseek/ftell use a 32-bit long: ftell returns -1 for a position
 * beyond LONG_MAX, so a perfectly good file over 2 GiB looked "unopenable" to
 * the size check below and was never mapped. _fseeki64/_ftelli64 are the 64-bit
 * variants and are available in this toolchain (MinGW/TDM-GCC 64-bit,
 * <stdio.h>). The non-`_WIN32` build path keeps the portable calls so the file
 * still compiles elsewhere; it is not the target platform.
 *
 * `long long` is the C11 spelling of MinGW's `__int64` and is used because these
 * statements are compiled on both paths. */
static int seek64_start(FILE *fp) {
#ifdef _WIN32
    return _fseeki64(fp, 0, SEEK_SET);
#else
    return fseek(fp, 0, SEEK_SET);
#endif
}

static int seek64_end(FILE *fp) {
#ifdef _WIN32
    return _fseeki64(fp, 0, SEEK_END);
#else
    return fseek(fp, 0, SEEK_END);
#endif
}

static long long tell64(FILE *fp) {
#ifdef _WIN32
    return (long long)_ftelli64(fp);
#else
    return (long long)ftell(fp);
#endif
}

long long katali_gguf_file_size(const char *path) {
    if (!path) return -1;
    FILE *fp = fopen(path, "rb");
    if (!fp) return -1;
    if (seek64_end(fp) != 0) { fclose(fp); return -1; }
    long long sz = tell64(fp);
    fclose(fp);
    return sz;
}

static int map_file(KataliGgufFile *f, const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return -1;
    if (seek64_end(fp) != 0) { fclose(fp); return -1; }
    long long sz = tell64(fp);
    if (sz <= 0) { fclose(fp); return -1; }
    f->size = (uint64_t)sz;

#ifdef _WIN32
    {
        HANDLE fh = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL,
                                OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
        if (fh != INVALID_HANDLE_VALUE) {
            HANDLE mh = CreateFileMappingA(fh, NULL, PAGE_READONLY, 0, 0, NULL);
            if (mh) {
                const void *view = MapViewOfFile(mh, FILE_MAP_READ, 0, 0, 0);
                if (view) {
                    f->map = (const uint8_t *)view;
                    f->mapped = 1;
                    f->map_handle = mh;
                    CloseHandle(fh);
                    fclose(fp);
                    return 0;
                }
                CloseHandle(mh);
            }
            CloseHandle(fh);
        }
    }
#endif
    /* Portable fallback: read the file into the heap. The size probe above left
     * the position at EOF, so rewind before reading (without this the read
     * returns 0 bytes and a file that simply could not be mapped was reported
     * as "cannot open file"). */
    {
        if (seek64_start(fp) != 0) { fclose(fp); return -1; }
        uint8_t *buf = (uint8_t *)malloc((size_t)f->size ? (size_t)f->size : 1);
        if (!buf) { fclose(fp); return -1; }
        if (fread(buf, 1, (size_t)f->size, fp) != (size_t)f->size) {
            free(buf); fclose(fp); return -1;
        }
        fclose(fp);
        f->map = buf;
        f->mapped = 0;
        return 0;
    }
}

static void unmap_file(KataliGgufFile *f) {
    if (!f->map) return;
#ifdef _WIN32
    if (f->mapped) {
        UnmapViewOfFile((LPCVOID)f->map);
        if (f->map_handle) CloseHandle((HANDLE)f->map_handle);
        f->map_handle = NULL;
        f->map = NULL;
        return;
    }
#endif
    free((void *)f->map);
    f->map = NULL;
}

/* ==================================================================== *
 * Bounds-checked cursor                                                 *
 * ==================================================================== */
typedef struct Reader {
    const uint8_t *m;
    uint64_t size;
    uint64_t pos;
    int err;
    char msg[160];
} Reader;

static void r_fail(Reader *r, const char *what) {
    if (r->err) return;
    r->err = 1;
    snprintf(r->msg, sizeof(r->msg), "%s (at offset %llu)",
             what, (unsigned long long)r->pos);
}

static void r_need(Reader *r, uint64_t n) {
    if (r->err) return;
    if (n > r->size - r->pos) r_fail(r, "truncated file");
}

static const uint8_t *r_bytes(Reader *r, uint64_t n) {
    r_need(r, n);
    if (r->err) return NULL;
    const uint8_t *p = r->m + r->pos;
    r->pos += n;
    return p;
}

static uint8_t r_u8(Reader *r) {
    const uint8_t *p = r_bytes(r, 1);
    return p ? p[0] : 0;
}
static uint16_t r_u16(Reader *r) {
    const uint8_t *p = r_bytes(r, 2);
    return p ? (uint16_t)(p[0] | ((uint16_t)p[1] << 8)) : 0;
}
static uint32_t r_u32(Reader *r) {
    const uint8_t *p = r_bytes(r, 4);
    return p ? ((uint32_t)p[0] | ((uint32_t)p[1] << 8) |
                ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24)) : 0;
}
static uint64_t r_u64(Reader *r) {
    uint64_t lo = r_u32(r), hi = r_u32(r);
    return lo | (hi << 32);
}
static float r_f32(Reader *r) {
    uint32_t bits = r_u32(r);
    float v;
    memcpy(&v, &bits, sizeof(v));
    return v;
}
static double r_f64(Reader *r) {
    uint64_t bits = r_u64(r);
    double v;
    memcpy(&v, &bits, sizeof(v));
    return v;
}

/* Read a length-prefixed, non-NUL-terminated byte string. */
static const uint8_t *r_string(Reader *r, uint64_t *len_out) {
    uint64_t n = r_u64(r);
    if (r->err) return NULL;
    /* A string cannot be longer than the remaining file. */
    r_need(r, n);
    if (r->err) return NULL;
    const uint8_t *p = r->m + r->pos;
    r->pos += n;
    if (len_out) *len_out = n;
    return p;
}

/* ==================================================================== *
 * Metadata value parsing                                               *
 * ==================================================================== */
static uint64_t scalar_size(uint32_t t) {
    switch (t) {
        case KGGUF_UINT8: case KGGUF_INT8: case KGGUF_BOOL: return 1;
        case KGGUF_UINT16: case KGGUF_INT16: return 2;
        case KGGUF_UINT32: case KGGUF_INT32: case KGGUF_FLOAT32: return 4;
        case KGGUF_UINT64: case KGGUF_INT64: case KGGUF_FLOAT64: return 8;
        default: return 0; /* string / array / unsupported */
    }
}

static void free_kv(KataliGgufKv *kv) {
    if (!kv) return;
    free(kv->key);
    if (kv->s_owned) free((void *)kv->s);
    free(kv->ai);
    free(kv->af);
    free(kv->as);
    free(kv->as_len);
    memset(kv, 0, sizeof(*kv));
}

/* Parse one scalar (non-array) value of type t into kv. */
static void parse_scalar(Reader *r, uint32_t t, KataliGgufKv *kv) {
    switch (t) {
        case KGGUF_UINT8:  kv->u = r_u8(r);  kv->i = (int64_t)kv->u; break;
        case KGGUF_INT8:   kv->i = (int8_t)r_u8(r); kv->u = (uint64_t)kv->i; break;
        case KGGUF_UINT16: kv->u = r_u16(r); kv->i = (int64_t)kv->u; break;
        case KGGUF_INT16:  kv->i = (int16_t)r_u16(r); kv->u = (uint64_t)kv->i; break;
        case KGGUF_UINT32: kv->u = r_u32(r); kv->i = (int64_t)kv->u; break;
        case KGGUF_INT32:  kv->i = (int32_t)r_u32(r); kv->u = (uint64_t)kv->i; break;
        case KGGUF_BOOL:   kv->u = r_u8(r) ? 1u : 0u; kv->i = (int64_t)kv->u; break;
        case KGGUF_UINT64: kv->u = r_u64(r); kv->i = (int64_t)kv->u; break;
        case KGGUF_INT64:  kv->i = (int64_t)r_u64(r); kv->u = (uint64_t)kv->i; break;
        case KGGUF_FLOAT32: kv->f = (double)r_f32(r); break;
        case KGGUF_FLOAT64: kv->f = r_f64(r); break;
        case KGGUF_STRING: {
            uint64_t n = 0;
            const uint8_t *p = r_string(r, &n);
            if (r->err) break;
            char *copy = (char *)malloc((size_t)n + 1);
            if (!copy) { r_fail(r, "out of memory"); break; }
            if (n) memcpy(copy, p, (size_t)n);
            copy[n] = '\0';
            kv->s = (const uint8_t *)copy;
            kv->s_len = (size_t)n;
            kv->s_owned = 1;
            break;
        }
        default:
            r_fail(r, "unsupported metadata value type");
            break;
    }
}

static void parse_array(Reader *r, KataliGgufKv *kv) {
    uint32_t et = r_u32(r);
    uint64_t count = r_u64(r);
    if (r->err) return;
    if (et == KGGUF_ARRAY) { r_fail(r, "nested metadata arrays are unsupported"); return; }
    /* The count can never exceed the bytes left in the file. */
    if (count > r->size - r->pos) { r_fail(r, "array count exceeds file size"); return; }

    kv->arr_type = et;
    kv->n = count;

    uint64_t esz = scalar_size(et);
    if (esz != 0) {
        if (count > (r->size - r->pos) / esz) {
            r_fail(r, "array payload exceeds file size");
            return;
        }
        r_need(r, count * esz);
        if (r->err) return;
    }

    if (et == KGGUF_STRING) {
        kv->as = (const uint8_t **)calloc((size_t)(count ? count : 1), sizeof(uint8_t *));
        kv->as_len = (size_t *)calloc((size_t)(count ? count : 1), sizeof(size_t));
        if (!kv->as || !kv->as_len) { r_fail(r, "out of memory"); return; }
        for (uint64_t i = 0; i < count; i++) {
            uint64_t n = 0;
            const uint8_t *p = r_string(r, &n);
            if (r->err) return;
            kv->as[i] = p;
            kv->as_len[i] = (size_t)n;
        }
        return;
    }
    if (et == KGGUF_FLOAT32 || et == KGGUF_FLOAT64) {
        kv->af = (double *)calloc((size_t)(count ? count : 1), sizeof(double));
        if (!kv->af) { r_fail(r, "out of memory"); return; }
        for (uint64_t i = 0; i < count; i++)
            kv->af[i] = (et == KGGUF_FLOAT32) ? (double)r_f32(r) : r_f64(r);
        return;
    }
    if (esz != 0) {
        kv->ai = (int64_t *)calloc((size_t)(count ? count : 1), sizeof(int64_t));
        if (!kv->ai) { r_fail(r, "out of memory"); return; }
        for (uint64_t i = 0; i < count; i++) {
            KataliGgufKv tmp;
            memset(&tmp, 0, sizeof(tmp));
            parse_scalar(r, et, &tmp);
            if (r->err) return;
            kv->ai[i] = (et == KGGUF_FLOAT32 || et == KGGUF_FLOAT64)
                            ? (int64_t)tmp.f : tmp.i;
        }
        return;
    }
    r_fail(r, "unsupported metadata array element type");
}

/* ==================================================================== *
 * Open / parse                                                          *
 * ==================================================================== */
static uint64_t align_up(uint64_t v, uint64_t a) {
    if (a <= 1) return v;
    uint64_t rem = v % a;
    return rem ? v + (a - rem) : v;
}

int katali_gguf_open(KataliGgufFile *f, const char *path, int use_mmap,
                     char *err, size_t err_cap) {
    if (!f || !path) return -1;
    memset(f, 0, sizeof(*f));
    f->alignment = KATALI_GGUF_DEFAULT_ALIGN;
    snprintf(f->path, sizeof(f->path), "%s", path);
    double t0 = gguf_now_s();

    if (map_file(f, path) != 0) {
        if (err && err_cap) snprintf(err, err_cap, "cannot open file: %s", path);
        return -1;
    }

    if (!use_mmap && f->mapped) {
        /* Caller explicitly asked for a heap copy: remap without mmap. This is a
         * second, independent size probe and must use the same 64-bit helpers,
         * or --no-mmap on a file over 2 GiB would still fail after the primary
         * path was fixed. */
        unmap_file(f);
        f->size = 0;
        FILE *fp = fopen(path, "rb");
        if (!fp) { if (err && err_cap) snprintf(err, err_cap, "cannot open file"); return -1; }
        if (seek64_end(fp) != 0) { fclose(fp); if (err && err_cap) snprintf(err, err_cap, "cannot seek file"); return -1; }
        long long sz = tell64(fp);
        if (seek64_start(fp) != 0) { fclose(fp); if (err && err_cap) snprintf(err, err_cap, "cannot seek file"); return -1; }
        /* A heap copy needs the whole file in one address-space-sized block. */
        uint8_t *buf = NULL;
        if (sz > 0 && (uint64_t)sz <= (uint64_t)SIZE_MAX) buf = (uint8_t *)malloc((size_t)sz);
        if (!buf || sz <= 0 || fread(buf, 1, (size_t)sz, fp) != (size_t)sz) {
            free(buf); fclose(fp);
            if (err && err_cap) {
                if (sz > 0 && (uint64_t)sz > (uint64_t)SIZE_MAX)
                    snprintf(err, err_cap, "file too large for a heap copy (%lld bytes)", sz);
                else
                    snprintf(err, err_cap, "cannot read file");
            }
            return -1;
        }
        fclose(fp);
        f->map = buf;
        f->size = (uint64_t)sz;
        f->mapped = 0;
    }
    f->map_s = gguf_now_s() - t0;
    double t_parse = gguf_now_s();

    Reader r;
    memset(&r, 0, sizeof(r));
    r.m = f->map;
    r.size = f->size;

    uint32_t magic = r_u32(&r);
    if (r.err || magic != KATALI_GGUF_MAGIC) {
        if (err && err_cap) snprintf(err, err_cap, "not a GGUF file (bad magic)");
        goto fail;
    }
    f->version = r_u32(&r);
    if (r.err || f->version < KATALI_GGUF_VERSION_MIN || f->version > KATALI_GGUF_VERSION_MAX) {
        if (err && err_cap) snprintf(err, err_cap, "unsupported GGUF version %u",
                                     (unsigned)f->version);
        goto fail;
    }
    f->tensor_count = r_u64(&r);
    f->kv_count = r_u64(&r);
    if (r.err) {
        if (err && err_cap) snprintf(err, err_cap, "truncated GGUF header");
        goto fail;
    }
    if (f->tensor_count > f->size || f->kv_count > f->size) {
        if (err && err_cap) snprintf(err, err_cap, "implausible header counts");
        goto fail;
    }

    /* --- metadata --- */
    if (f->kv_count) {
        f->kv = (KataliGgufKv *)calloc((size_t)f->kv_count, sizeof(KataliGgufKv));
        if (!f->kv) { if (err && err_cap) snprintf(err, err_cap, "out of memory"); goto fail; }
    }
    for (uint64_t i = 0; i < f->kv_count; i++) {
        KataliGgufKv *kv = &f->kv[i];
        uint64_t klen = 0;
        const uint8_t *kp = r_string(&r, &klen);
        if (r.err) break;
        kv->key = (char *)malloc((size_t)klen + 1);
        if (!kv->key) { r_fail(&r, "out of memory"); break; }
        if (klen) memcpy(kv->key, kp, (size_t)klen);
        kv->key[klen] = '\0';
        kv->type = r_u32(&r);
        kv->n = 1;
        if (r.err) break;
        if (kv->type == KGGUF_ARRAY) parse_array(&r, kv);
        else parse_scalar(&r, kv->type, kv);
        if (r.err) break;
    }
    if (r.err) {
        if (err && err_cap) snprintf(err, err_cap, "%s", r.msg);
        goto fail;
    }

    /* Alignment comes from metadata when present. */
    {
        const KataliGgufKv *a = katali_gguf_kv_find(f, "general.alignment");
        if (a && a->type != KGGUF_ARRAY && a->u >= 8 && a->u <= (1u << 20)) {
            f->alignment = a->u;
        }
    }

    /* --- tensor directory --- */
    if (f->tensor_count) {
        f->tensors = (KataliGgufTensor *)calloc((size_t)f->tensor_count,
                                                sizeof(KataliGgufTensor));
        if (!f->tensors) { if (err && err_cap) snprintf(err, err_cap, "out of memory"); goto fail; }
    }
    for (uint64_t i = 0; i < f->tensor_count; i++) {
        KataliGgufTensor *t = &f->tensors[i];
        uint64_t nlen = 0;
        const uint8_t *np = r_string(&r, &nlen);
        if (r.err) break;
        t->name = (char *)malloc((size_t)nlen + 1);
        if (!t->name) { r_fail(&r, "out of memory"); break; }
        if (nlen) memcpy(t->name, np, (size_t)nlen);
        t->name[nlen] = '\0';

        t->n_dims = r_u32(&r);
        if (r.err) break;
        if (t->n_dims < 1 || t->n_dims > KATALI_GGUF_MAX_DIMS) {
            r_fail(&r, "tensor rank out of range");
            break;
        }
        uint64_t elems = 1;
        int overflow = 0;
        for (uint32_t d = 0; d < t->n_dims; d++) {
            t->dims[d] = r_u64(&r);
            if (r.err) break;
            if (t->dims[d] == 0) { overflow = 1; continue; }
            if (elems > UINT64_MAX / t->dims[d]) overflow = 1;
            else elems *= t->dims[d];
        }
        if (r.err) break;
        if (overflow) { r_fail(&r, "tensor element count overflows"); break; }
        t->n_elements = elems;
        t->type = r_u32(&r);
        t->offset = r_u64(&r);
        if (r.err) break;
    }
    if (r.err) {
        if (err && err_cap) snprintf(err, err_cap, "%s", r.msg);
        goto fail;
    }

    /* --- resolve + validate tensor data --- */
    f->data_offset = align_up(r.pos, f->alignment);
    if (f->data_offset > f->size) {
        if (err && err_cap) snprintf(err, err_cap, "tensor data offset past end of file");
        goto fail;
    }
    {
        uint64_t data_size = f->size - f->data_offset;
        for (uint64_t i = 0; i < f->tensor_count; i++) {
            KataliGgufTensor *t = &f->tensors[i];
            uint64_t nb = katali_ggml_row_bytes(t->type, t->n_elements);
            if (t->offset > data_size) {
                if (err && err_cap)
                    snprintf(err, err_cap, "tensor '%s' offset past end of file", t->name);
                goto fail;
            }
            if (nb == 0) {
                /* Unsupported or non-block-aligned type: keep the descriptor so
                 * `validate` can report it, but never expose a pointer we could
                 * not verify. */
                t->nbytes = 0;
                t->data = NULL;
                continue;
            }
            if (nb > data_size - t->offset) {
                if (err && err_cap)
                    snprintf(err, err_cap, "tensor '%s' (%llu bytes) is truncated",
                             t->name, (unsigned long long)nb);
                goto fail;
            }
            t->nbytes = nb;
            t->data = f->map + f->data_offset + t->offset;
        }
    }

    f->parse_s = gguf_now_s() - t_parse;
    f->ok = 1;
    return 0;

fail:
    katali_gguf_close(f);
    return -1;
}

void katali_gguf_close(KataliGgufFile *f) {
    if (!f) return;
    if (f->kv) {
        for (uint64_t i = 0; i < f->kv_count; i++) free_kv(&f->kv[i]);
        free(f->kv);
        f->kv = NULL;
    }
    if (f->tensors) {
        for (uint64_t i = 0; i < f->tensor_count; i++) free(f->tensors[i].name);
        free(f->tensors);
        f->tensors = NULL;
    }
    unmap_file(f);
    f->ok = 0;
}

/* ==================================================================== *
 * Metadata + tensor accessors                                           *
 * ==================================================================== */
const KataliGgufKv *katali_gguf_kv_find(const KataliGgufFile *f, const char *key) {
    if (!f || !f->kv || !key) return NULL;
    for (uint64_t i = 0; i < f->kv_count; i++)
        if (f->kv[i].key && strcmp(f->kv[i].key, key) == 0) return &f->kv[i];
    return NULL;
}

const char *katali_gguf_get_str(const KataliGgufFile *f, const char *key,
                                const char *def, size_t *len_out) {
    const KataliGgufKv *kv = katali_gguf_kv_find(f, key);
    if (len_out) *len_out = 0;
    if (!kv || kv->type != KGGUF_STRING || !kv->s) return def;
    if (len_out) *len_out = kv->s_len;
    return (const char *)kv->s;
}

int64_t katali_gguf_get_int(const KataliGgufFile *f, const char *key, int64_t def) {
    const KataliGgufKv *kv = katali_gguf_kv_find(f, key);
    if (!kv) return def;
    if (kv->type == KGGUF_ARRAY) return def;
    if (kv->type == KGGUF_FLOAT32 || kv->type == KGGUF_FLOAT64)
        return (int64_t)kv->f;
    return kv->i;
}

double katali_gguf_get_float(const KataliGgufFile *f, const char *key, double def) {
    const KataliGgufKv *kv = katali_gguf_kv_find(f, key);
    if (!kv || kv->type == KGGUF_ARRAY) return def;
    if (kv->type == KGGUF_FLOAT32 || kv->type == KGGUF_FLOAT64) return kv->f;
    return (double)kv->i;
}

int katali_gguf_get_bool(const KataliGgufFile *f, const char *key, int def) {
    const KataliGgufKv *kv = katali_gguf_kv_find(f, key);
    if (!kv || kv->type == KGGUF_ARRAY) return def;
    if (kv->type == KGGUF_BOOL) return kv->u ? 1 : 0;
    if (kv->type == KGGUF_FLOAT32 || kv->type == KGGUF_FLOAT64) return kv->f != 0.0;
    return kv->i != 0;
}

int64_t katali_gguf_array_len(const KataliGgufFile *f, const char *key,
                              uint32_t *elem_type_out) {
    const KataliGgufKv *kv = katali_gguf_kv_find(f, key);
    if (!kv || kv->type != KGGUF_ARRAY) return -1;
    if (elem_type_out) *elem_type_out = kv->arr_type;
    return (int64_t)kv->n;
}

int64_t katali_gguf_array_int(const KataliGgufFile *f, const char *key, uint64_t idx) {
    const KataliGgufKv *kv = katali_gguf_kv_find(f, key);
    if (!kv || kv->type != KGGUF_ARRAY || idx >= kv->n) return 0;
    if (kv->ai) return kv->ai[idx];
    if (kv->af) return (int64_t)kv->af[idx];
    return 0;
}

const char *katali_gguf_array_str(const KataliGgufFile *f, const char *key,
                                  uint64_t idx, size_t *len_out) {
    const KataliGgufKv *kv = katali_gguf_kv_find(f, key);
    if (len_out) *len_out = 0;
    if (!kv || kv->type != KGGUF_ARRAY || kv->arr_type != KGGUF_STRING) return NULL;
    if (!kv->as || idx >= kv->n) return NULL;
    if (len_out) *len_out = kv->as_len[idx];
    return (const char *)kv->as[idx];
}

const KataliGgufTensor *katali_gguf_find_tensor(const KataliGgufFile *f,
                                                const char *name) {
    if (!f || !f->tensors || !name) return NULL;
    for (uint64_t i = 0; i < f->tensor_count; i++)
        if (f->tensors[i].name && strcmp(f->tensors[i].name, name) == 0)
            return &f->tensors[i];
    return NULL;
}

/* ==================================================================== *
 * Diagnostics                                                           *
 * ==================================================================== */
typedef struct TypeHist {
    uint32_t type;
    uint64_t bytes;
} TypeHist;

int katali_gguf_describe(const KataliGgufFile *f, KataliGgufInfo *out) {
    if (!f || !out || !f->ok) return -1;
    memset(out, 0, sizeof(*out));
    snprintf(out->format, sizeof(out->format), "GGUF");
    out->version = f->version;
    out->tensor_count = f->tensor_count;

    const char *arch = katali_gguf_get_str(f, "general.architecture", "", NULL);
    snprintf(out->architecture, sizeof(out->architecture), "%s", arch);

    char key[128];
    int64_t v;
#define ARCH_INT(field, dst) do { \
        snprintf(key, sizeof(key), "%s.%s", arch, field); \
        v = katali_gguf_get_int(f, key, 0); \
        if (v == 0) v = katali_gguf_get_int(f, field, 0); \
        (dst) = (int)v; \
    } while (0)
    ARCH_INT("block_count", out->layers);
    ARCH_INT("context_length", out->context_length);
    ARCH_INT("embedding_length", out->hidden);
    ARCH_INT("attention.head_count", out->attention_heads);
    ARCH_INT("attention.head_count_kv", out->key_value_heads);
#undef ARCH_INT

    int64_t vocab = katali_gguf_array_len(f, "tokenizer.ggml.tokens", NULL);
    out->vocab = vocab > 0 ? (int)vocab : 0;

    /* Quantization summary: the type holding the most tensor bytes. */
    {
        TypeHist hist[16];
        int hn = 0;
        uint64_t total = 0;
        for (uint64_t i = 0; i < f->tensor_count; i++) {
            const KataliGgufTensor *t = &f->tensors[i];
            total += t->nbytes;
            int found = 0;
            for (int j = 0; j < hn; j++)
                if (hist[j].type == t->type) { hist[j].bytes += t->nbytes; found = 1; break; }
            if (!found && hn < 16) { hist[hn].type = t->type; hist[hn].bytes = t->nbytes; hn++; }
        }
        uint32_t dom = KGGML_F32;
        uint64_t best = 0;
        for (int j = 0; j < hn; j++)
            if (hist[j].bytes > best) { best = hist[j].bytes; dom = hist[j].type; }
        snprintf(out->quantization, sizeof(out->quantization), "%s",
                 katali_ggml_type_name(dom));
        out->estimated_memory = total;
    }
    return 0;
}




