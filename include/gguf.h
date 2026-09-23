#ifndef KATALI_LAB_GGUF_H
#define KATALI_LAB_GGUF_H
#include <stddef.h>
#include <stdint.h>
#include "katali.h"
#include "platform.h"

#ifdef __cplusplus
extern "C" {
#endif

#define KGGUF_MAGIC 0x46554747u

enum {
    KGGUF_UINT8=0, KGGUF_INT8=1, KGGUF_UINT16=2, KGGUF_INT16=3,
    KGGUF_UINT32=4, KGGUF_INT32=5, KGGUF_FLOAT32=6, KGGUF_BOOL=7,
    KGGUF_STRING=8, KGGUF_ARRAY=9, KGGUF_UINT64=10, KGGUF_INT64=11,
    KGGUF_FLOAT64=12
};

enum {
    KGGML_TYPE_F32 = 0,
    KGGML_TYPE_F16 = 1,
    KGGML_TYPE_Q4_0 = 2,
    KGGML_TYPE_Q4_1 = 3,
    KGGML_TYPE_Q5_0 = 6,
    KGGML_TYPE_Q5_1 = 7,
    KGGML_TYPE_Q8_0 = 8,
    KGGML_TYPE_Q2_K = 10,
    KGGML_TYPE_Q3_K = 11,
    KGGML_TYPE_Q4_K = 12,
    KGGML_TYPE_Q5_K = 13,
    KGGML_TYPE_Q6_K = 14,
    KGGML_TYPE_Q8_K = 15,
    KGGML_TYPE_BF16 = 30
};

typedef struct KgufKv {
    char    *key;
    uint32_t type;
    uint32_t arr_type;
    uint64_t n;
    int64_t  i;
    uint64_t u;
    double   f;
    char    *s;
} KgufKv;

typedef struct KgufTensor {
    char     *name;
    uint32_t  n_dims;
    uint64_t  dims[4];
    uint32_t  type;
    uint64_t  offset;
    const uint8_t *data;
    uint64_t  nbytes;
} KgufTensor;

typedef struct KgufFile {
    KataliMap map;
    char      path[1024];
    uint32_t  version;
    uint64_t  tensor_count;
    uint64_t  kv_count;
    uint64_t  alignment;
    uint64_t  data_offset;
    KgufKv   *kv;
    KgufTensor *tensors;
    int       ok;
} KgufFile;

KATALI_API int  kguf_open(const char *path, KgufFile *out);
KATALI_API void kguf_close(KgufFile *f);
KATALI_API const KgufKv *kguf_find_kv(const KgufFile *f, const char *key);
KATALI_API const KgufTensor *kguf_find_tensor(const KgufFile *f, const char *name);
KATALI_API uint64_t kguf_type_nbytes(uint32_t type, uint64_t nelements);
KATALI_API const char *kguf_type_name(uint32_t type);
KATALI_API void kguf_print_info(const KgufFile *f);
KATALI_API void kguf_print_tensors(const KgufFile *f, const char *substr);

#ifdef __cplusplus
}
#endif
#endif
