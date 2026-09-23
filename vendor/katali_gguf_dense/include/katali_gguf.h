/* Katali-GGUF native GGUF reader — Apache-2.0
 *
 * Original Katali implementation of the public GGUF container format. This is
 * a file-format reader only: it contains no inference runtime and has no
 * dependency on any third-party inference engine.
 *
 * Security posture: every offset and length that comes from the file is
 * validated against the actual file size before it is used. The reader never
 * trusts the tensor directory.
 */
#ifndef KATALI_GGUF_H
#define KATALI_GGUF_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define KATALI_GGUF_MAGIC          0x46554747u /* 'G','G','U','F' little-endian */
#define KATALI_GGUF_VERSION_MIN    1u
#define KATALI_GGUF_VERSION_MAX    3u
#define KATALI_GGUF_MAX_DIMS       4
#define KATALI_GGUF_DEFAULT_ALIGN  32u

/* GGUF metadata value types (public specification). */
enum {
    KGGUF_UINT8   = 0,
    KGGUF_INT8    = 1,
    KGGUF_UINT16  = 2,
    KGGUF_INT16   = 3,
    KGGUF_UINT32  = 4,
    KGGUF_INT32   = 5,
    KGGUF_FLOAT32 = 6,
    KGGUF_BOOL    = 7,
    KGGUF_STRING  = 8,
    KGGUF_ARRAY   = 9,
    KGGUF_UINT64  = 10,
    KGGUF_INT64   = 11,
    KGGUF_FLOAT64 = 12
};

typedef struct KataliGgufKv {
    char    *key;        /* owned, NUL-terminated */
    uint32_t type;       /* KGGUF_* */
    uint32_t arr_type;   /* element type when type == KGGUF_ARRAY */
    uint64_t n;          /* element count (1 for scalars) */

    int64_t  i;          /* scalar integer value */
    uint64_t u;          /* scalar unsigned value */
    double   f;          /* scalar float value */
    const uint8_t *s;    /* scalar string bytes */
    size_t   s_len;
    int      s_owned;    /* 1 when s points into a heap copy */

    /* ARRAY payloads (parallel storage, only one is non-NULL) */
    int64_t        *ai;      /* integer/bool elements */
    double         *af;      /* floating elements */
    const uint8_t **as;      /* string elements (point into the mapping) */
    size_t         *as_len;
} KataliGgufKv;

typedef struct KataliGgufTensor {
    char     *name;      /* owned */
    uint32_t  n_dims;
    uint64_t  dims[KATALI_GGUF_MAX_DIMS]; /* GGUF order: dims[0] is contiguous */
    uint32_t  type;      /* ggml tensor type id */
    uint64_t  offset;    /* byte offset relative to the tensor data section */
    const uint8_t *data; /* resolved, bounds-checked pointer */
    uint64_t  n_elements;
    uint64_t  nbytes;
} KataliGgufTensor;

typedef struct KataliGgufFile {
    char      path[1024];
    const uint8_t *map;   /* whole file, read-only */
    uint64_t  size;
    int       mapped;     /* 1 = memory-mapped, 0 = heap read fallback */
    void     *map_handle; /* platform opaque file mapping */

    uint32_t  version;
    uint64_t  tensor_count;
    uint64_t  kv_count;
    uint64_t  alignment;
    uint64_t  data_offset; /* absolute file offset of the data section */

    KataliGgufKv     *kv;
    KataliGgufTensor *tensors;
    int       ok;

    /* Timings captured during open (seconds). */
    double    map_s;
    double    parse_s;
} KataliGgufFile;

/* Summary printed by `katali-gguf inspect`. */
typedef struct KataliGgufInfo {
    char     format[16];
    uint32_t version;
    char     architecture[64];
    int      vocab;
    int      layers;
    int      hidden;
    int      attention_heads;
    int      key_value_heads;
    int      context_length;
    char     quantization[48];
    uint64_t tensor_count;
    uint64_t estimated_memory;
} KataliGgufInfo;

/* --- lifecycle ----------------------------------------------------------- */

/* Parse a GGUF file. `use_mmap` != 0 maps the file, otherwise it is read into
 * the heap. On failure returns -1 and writes a human-readable reason to err. */
int  katali_gguf_open(KataliGgufFile *f, const char *path, int use_mmap,
                      char *err, size_t err_cap);
void katali_gguf_close(KataliGgufFile *f);

/* Byte size of a file, or -1 when it cannot be determined. Uses 64-bit
 * seek/tell so files larger than 2 GiB report their real size: a 32-bit
 * `ftell()` returns -1 beyond LONG_MAX, which made such files look unopenable.
 * Both open paths use this, so the memory-mapped and heap-copy paths cannot
 * disagree. Exposed so the regression test can assert it directly. */
long long katali_gguf_file_size(const char *path);

/* --- metadata ------------------------------------------------------------ */

const KataliGgufKv *katali_gguf_kv_find(const KataliGgufFile *f, const char *key);
const char *katali_gguf_get_str(const KataliGgufFile *f, const char *key,
                                const char *def, size_t *len_out);
int64_t     katali_gguf_get_int(const KataliGgufFile *f, const char *key,
                                int64_t def);
double      katali_gguf_get_float(const KataliGgufFile *f, const char *key,
                                  double def);
int         katali_gguf_get_bool(const KataliGgufFile *f, const char *key,
                                 int def);
/* Array helpers. Return the element count, or -1 when missing/type mismatch. */
int64_t     katali_gguf_array_len(const KataliGgufFile *f, const char *key,
                                  uint32_t *elem_type_out);
int64_t     katali_gguf_array_int(const KataliGgufFile *f, const char *key,
                                  uint64_t idx);
const char *katali_gguf_array_str(const KataliGgufFile *f, const char *key,
                                  uint64_t idx, size_t *len_out);

/* --- tensors ------------------------------------------------------------- */

const KataliGgufTensor *katali_gguf_find_tensor(const KataliGgufFile *f,
                                                const char *name);
/* Fill the diagnostics summary used by the CLI and GUI metadata panel. */
int  katali_gguf_describe(const KataliGgufFile *f, KataliGgufInfo *out);

#ifdef __cplusplus
}
#endif
#endif /* KATALI_GGUF_H */
