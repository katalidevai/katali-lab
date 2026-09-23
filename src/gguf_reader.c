#include "gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int need(const KgufFile *f, uint64_t off, uint64_t n) {
    return off + n <= f->map.size;
}

static uint32_t rd_u32(const uint8_t *p) {
    uint32_t v; memcpy(&v, p, 4); return v;
}
static uint64_t rd_u64(const uint8_t *p) {
    uint64_t v; memcpy(&v, p, 8); return v;
}

static int read_string(KgufFile *f, uint64_t *off, char **out) {
    if (!need(f, *off, 8)) return KATALI_ERR_FORMAT;
    uint64_t n = rd_u64(f->map.data + *off); *off += 8;
    if (!need(f, *off, n)) return KATALI_ERR_FORMAT;
    char *s = (char *)malloc((size_t)n + 1);
    if (!s) return KATALI_ERR_NOMEM;
    memcpy(s, f->map.data + *off, (size_t)n);
    s[n] = 0;
    *off += n;
    *out = s;
    return KATALI_OK;
}

static int skip_value(KgufFile *f, uint64_t *off, uint32_t type);

static int read_value(KgufFile *f, uint64_t *off, KgufKv *kv) {
    /* Preserve kv->key — caller sets it before this. */
    char *kept_key = kv->key;
    memset(kv, 0, sizeof(*kv));
    kv->key = kept_key;
    if (!need(f, *off, 4)) return KATALI_ERR_FORMAT;
    kv->type = rd_u32(f->map.data + *off); *off += 4;
    kv->n = 1;
    switch (kv->type) {
    case KGGUF_UINT8: case KGGUF_INT8: case KGGUF_BOOL:
        if (!need(f, *off, 1)) return KATALI_ERR_FORMAT;
        kv->u = f->map.data[*off]; kv->i = (int8_t)f->map.data[*off]; *off += 1; break;
    case KGGUF_UINT16: case KGGUF_INT16:
        if (!need(f, *off, 2)) return KATALI_ERR_FORMAT;
        { uint16_t v; memcpy(&v, f->map.data + *off, 2); kv->u = v; kv->i = (int16_t)v; *off += 2; } break;
    case KGGUF_UINT32: case KGGUF_INT32: case KGGUF_FLOAT32:
        if (!need(f, *off, 4)) return KATALI_ERR_FORMAT;
        if (kv->type == KGGUF_FLOAT32) { float x; memcpy(&x, f->map.data + *off, 4); kv->f = x; }
        else { uint32_t v; memcpy(&v, f->map.data + *off, 4); kv->u = v; kv->i = (int32_t)v; }
        *off += 4; break;
    case KGGUF_UINT64: case KGGUF_INT64: case KGGUF_FLOAT64:
        if (!need(f, *off, 8)) return KATALI_ERR_FORMAT;
        if (kv->type == KGGUF_FLOAT64) { double x; memcpy(&x, f->map.data + *off, 8); kv->f = x; }
        else { uint64_t v = rd_u64(f->map.data + *off); kv->u = v; kv->i = (int64_t)v; }
        *off += 8; break;
    case KGGUF_STRING:
        return read_string(f, off, &kv->s);
    case KGGUF_ARRAY: {
        if (!need(f, *off, 12)) return KATALI_ERR_FORMAT;
        kv->arr_type = rd_u32(f->map.data + *off); *off += 4;
        kv->n = rd_u64(f->map.data + *off); *off += 8;
        for (uint64_t i = 0; i < kv->n; i++) {
            int rc = skip_value(f, off, kv->arr_type);
            if (rc) return rc;
        }
        break;
    }
    default:
        return KATALI_ERR_FORMAT;
    }
    return KATALI_OK;
}

static int skip_value(KgufFile *f, uint64_t *off, uint32_t type) {
    KgufKv tmp;
    uint64_t o = *off;
    /* Fabricate a typed skip by temporarily pretending type was already read. */
    if (type == KGGUF_ARRAY) {
        if (!need(f, o, 12)) return KATALI_ERR_FORMAT;
        uint32_t at = rd_u32(f->map.data + o); o += 4;
        uint64_t n = rd_u64(f->map.data + o); o += 8;
        for (uint64_t i = 0; i < n; i++) {
            int rc = skip_value(f, &o, at);
            if (rc) return rc;
        }
        *off = o;
        return KATALI_OK;
    }
    /* Reuse read_value path: push type word then read — messy; do sizes instead. */
    switch (type) {
    case KGGUF_UINT8: case KGGUF_INT8: case KGGUF_BOOL: o += 1; break;
    case KGGUF_UINT16: case KGGUF_INT16: o += 2; break;
    case KGGUF_UINT32: case KGGUF_INT32: case KGGUF_FLOAT32: o += 4; break;
    case KGGUF_UINT64: case KGGUF_INT64: case KGGUF_FLOAT64: o += 8; break;
    case KGGUF_STRING: {
        if (!need(f, o, 8)) return KATALI_ERR_FORMAT;
        uint64_t n = rd_u64(f->map.data + o); o += 8 + n; break;
    }
    default: return KATALI_ERR_FORMAT;
    }
    if (o > f->map.size) return KATALI_ERR_FORMAT;
    *off = o;
    (void)tmp;
    return KATALI_OK;
}

uint64_t kguf_type_nbytes(uint32_t t, uint64_t ne) {
    switch (t) {
    case KGGML_TYPE_F32: return ne * 4;
    case KGGML_TYPE_F16: return ne * 2;
    case KGGML_TYPE_BF16: return ne * 2;
    case KGGML_TYPE_Q4_0: return (ne / 32) * 18;
    case KGGML_TYPE_Q4_1: return (ne / 32) * 20;
    case KGGML_TYPE_Q5_0: return (ne / 32) * 22;
    case KGGML_TYPE_Q5_1: return (ne / 32) * 24;
    case KGGML_TYPE_Q8_0: return (ne / 32) * 34;
    case KGGML_TYPE_Q2_K: return (ne / 256) * 84;
    case KGGML_TYPE_Q3_K: return (ne / 256) * 110;
    case KGGML_TYPE_Q4_K: return (ne / 256) * 144;
    case KGGML_TYPE_Q5_K: return (ne / 256) * 176;
    case KGGML_TYPE_Q6_K: return (ne / 256) * 210;
    case KGGML_TYPE_Q8_K: return (ne / 256) * 292;
    default: return 0;
    }
}

static uint64_t ggml_type_size(uint32_t t, uint64_t ne) {
    return kguf_type_nbytes(t, ne);
}

const char *kguf_type_name(uint32_t t) {
    switch (t) {
    case KGGML_TYPE_F32: return "F32";
    case KGGML_TYPE_F16: return "F16";
    case KGGML_TYPE_BF16: return "BF16";
    case KGGML_TYPE_Q4_0: return "Q4_0";
    case KGGML_TYPE_Q4_1: return "Q4_1";
    case KGGML_TYPE_Q5_0: return "Q5_0";
    case KGGML_TYPE_Q5_1: return "Q5_1";
    case KGGML_TYPE_Q8_0: return "Q8_0";
    case KGGML_TYPE_Q2_K: return "Q2_K";
    case KGGML_TYPE_Q3_K: return "Q3_K";
    case KGGML_TYPE_Q4_K: return "Q4_K";
    case KGGML_TYPE_Q5_K: return "Q5_K";
    case KGGML_TYPE_Q6_K: return "Q6_K";
    case KGGML_TYPE_Q8_K: return "Q8_K";
    default: return "UNK";
    }
}

int kguf_open(const char *path, KgufFile *out) {
    memset(out, 0, sizeof(*out));
    snprintf(out->path, sizeof(out->path), "%s", path);
    if (katali_map_file(path, &out->map) != KATALI_OK) return KATALI_ERR_IO;
    if (out->map.size < 24) { kguf_close(out); return KATALI_ERR_FORMAT; }
    const uint8_t *p = out->map.data;
    if (rd_u32(p) != KGGUF_MAGIC) { kguf_close(out); return KATALI_ERR_FORMAT; }
    out->version = rd_u32(p + 4);
    out->tensor_count = rd_u64(p + 8);
    out->kv_count = rd_u64(p + 16);
    out->alignment = 32;
    uint64_t off = 24;

    out->kv = (KgufKv *)calloc((size_t)out->kv_count, sizeof(KgufKv));
    if (!out->kv && out->kv_count) { kguf_close(out); return KATALI_ERR_NOMEM; }
    for (uint64_t i = 0; i < out->kv_count; i++) {
        if (read_string(out, &off, &out->kv[i].key) != KATALI_OK) { kguf_close(out); return KATALI_ERR_FORMAT; }
        if (read_value(out, &off, &out->kv[i]) != KATALI_OK) { kguf_close(out); return KATALI_ERR_FORMAT; }
        if (out->kv[i].key && strcmp(out->kv[i].key, "general.alignment") == 0 &&
            (out->kv[i].type == KGGUF_UINT32 || out->kv[i].type == KGGUF_UINT64)) {
            out->alignment = out->kv[i].u ? out->kv[i].u : 32;
        }
    }

    out->tensors = (KgufTensor *)calloc((size_t)out->tensor_count, sizeof(KgufTensor));
    if (!out->tensors && out->tensor_count) { kguf_close(out); return KATALI_ERR_NOMEM; }
    for (uint64_t i = 0; i < out->tensor_count; i++) {
        KgufTensor *t = &out->tensors[i];
        if (read_string(out, &off, &t->name) != KATALI_OK) { kguf_close(out); return KATALI_ERR_FORMAT; }
        if (!need(out, off, 4)) { kguf_close(out); return KATALI_ERR_FORMAT; }
        t->n_dims = rd_u32(out->map.data + off); off += 4;
        if (t->n_dims > 4) { kguf_close(out); return KATALI_ERR_FORMAT; }
        for (uint32_t d = 0; d < t->n_dims; d++) {
            if (!need(out, off, 8)) { kguf_close(out); return KATALI_ERR_FORMAT; }
            t->dims[d] = rd_u64(out->map.data + off); off += 8;
        }
        if (!need(out, off, 4 + 8)) { kguf_close(out); return KATALI_ERR_FORMAT; }
        t->type = rd_u32(out->map.data + off); off += 4;
        t->offset = rd_u64(out->map.data + off); off += 8;
        uint64_t ne = 1;
        for (uint32_t d = 0; d < t->n_dims; d++) ne *= t->dims[d];
        t->nbytes = ggml_type_size(t->type, ne);
    }
    /* Align data section */
    uint64_t align = out->alignment ? out->alignment : 32;
    out->data_offset = (off + (align - 1)) / align * align;
    for (uint64_t i = 0; i < out->tensor_count; i++) {
        KgufTensor *t = &out->tensors[i];
        uint64_t abs = out->data_offset + t->offset;
        if (t->nbytes && abs + t->nbytes <= out->map.size) {
            t->data = out->map.data + abs;
        } else {
            t->data = NULL;
        }
    }
    out->ok = 1;
    return KATALI_OK;
}

void kguf_close(KgufFile *f) {
    if (!f) return;
    if (f->kv) {
        for (uint64_t i = 0; i < f->kv_count; i++) {
            free(f->kv[i].key);
            free(f->kv[i].s);
        }
        free(f->kv);
    }
    if (f->tensors) {
        for (uint64_t i = 0; i < f->tensor_count; i++) free(f->tensors[i].name);
        free(f->tensors);
    }
    katali_unmap(&f->map);
    memset(f, 0, sizeof(*f));
}

const KgufKv *kguf_find_kv(const KgufFile *f, const char *key) {
    if (!f || !key) return NULL;
    for (uint64_t i = 0; i < f->kv_count; i++) {
        if (f->kv[i].key && strcmp(f->kv[i].key, key) == 0) return &f->kv[i];
    }
    return NULL;
}

const KgufTensor *kguf_find_tensor(const KgufFile *f, const char *name) {
    if (!f || !name) return NULL;
    for (uint64_t i = 0; i < f->tensor_count; i++) {
        if (f->tensors[i].name && strcmp(f->tensors[i].name, name) == 0)
            return &f->tensors[i];
    }
    return NULL;
}

void kguf_print_info(const KgufFile *f) {
    if (!f || !f->ok) { printf("gguf: not open\n"); return; }
    printf("file: %s\n", f->path);
    printf("version: %u  tensors: %llu  kv: %llu  size: %.2f GiB\n",
           f->version,
           (unsigned long long)f->tensor_count,
           (unsigned long long)f->kv_count,
           (double)f->map.size / (1024.0 * 1024.0 * 1024.0));
    const KgufKv *arch = kguf_find_kv(f, "general.architecture");
    if (arch && arch->s) printf("architecture: %s\n", arch->s);
    const KgufKv *name = kguf_find_kv(f, "general.name");
    if (name && name->s) printf("name: %s\n", name->s);
    int n_q4k = 0, n_q5k = 0, n_q6k = 0, n_q8 = 0, n_f = 0, n_other = 0;
    for (uint64_t i = 0; i < f->tensor_count; i++) {
        switch (f->tensors[i].type) {
        case KGGML_TYPE_Q4_K: n_q4k++; break;
        case KGGML_TYPE_Q5_K: n_q5k++; break;
        case KGGML_TYPE_Q6_K: n_q6k++; break;
        case KGGML_TYPE_Q8_0: case KGGML_TYPE_Q8_K: n_q8++; break;
        case KGGML_TYPE_F32: case KGGML_TYPE_F16: case KGGML_TYPE_BF16: n_f++; break;
        default: n_other++; break;
        }
    }
    printf("tensors: Q4_K=%d Q5_K=%d Q6_K=%d Q8*=%d float*=%d other=%d\n",
           n_q4k, n_q5k, n_q6k, n_q8, n_f, n_other);
}

void kguf_print_tensors(const KgufFile *f, const char *substr) {
    if (!f || !f->ok) return;
    for (uint64_t i = 0; i < f->tensor_count; i++) {
        const KgufTensor *t = &f->tensors[i];
        if (substr && substr[0] && (!t->name || !strstr(t->name, substr))) continue;
        printf("%s  %s  dims=[", t->name ? t->name : "?", kguf_type_name(t->type));
        for (uint32_t d = 0; d < t->n_dims; d++) {
            if (d) printf(",");
            printf("%llu", (unsigned long long)t->dims[d]);
        }
        printf("]  nbytes=%llu\n", (unsigned long long)t->nbytes);
    }
}
