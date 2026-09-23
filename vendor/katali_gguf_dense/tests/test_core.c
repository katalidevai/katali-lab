/* Katali-GGUF model-free tests — Apache-2.0
 *
 * Builds synthetic GGUF files in memory so parser, tensor validation, dtype
 * decoding, kernels, the backend registry, and the full Qwen forward path can
 * all be tested without any model file on disk.
 */
#include "katali_gguf.h"
#include "katali_gguf_dtype.h"
#include "katali_gguf_fp.h"
#include "katali_gguf_kernels.h"
#include "katali_gguf_simd.h"
#include "katali_gguf_model.h"
#include "katali_gguf_tokenizer.h"
#include "katali_gguf_sampler.h"
#include "katali_backend.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#endif

static int g_fail = 0;
static int g_pass = 0;

#define CHECK(cond, msg) do { \
        if (cond) { g_pass++; } \
        else { g_fail++; printf("  FAIL: %s\n", msg); } \
    } while (0)

/* ==================================================================== *
 * Growable byte buffer                                                  *
 * ==================================================================== */
typedef struct Buf {
    uint8_t *p;
    size_t   len, cap;
} Buf;

static void buf_need(Buf *b, size_t n) {
    if (b->len + n <= b->cap) return;
    size_t cap = b->cap ? b->cap : 256;
    while (cap < b->len + n) cap *= 2;
    b->p = (uint8_t *)realloc(b->p, cap);
    b->cap = cap;
}

static void b_bytes(Buf *b, const void *p, size_t n) {
    buf_need(b, n);
    memcpy(b->p + b->len, p, n);
    b->len += n;
}
static void b_u8(Buf *b, uint8_t v) { b_bytes(b, &v, 1); }
static void b_u32(Buf *b, uint32_t v) { b_bytes(b, &v, 4); }
static void b_u64(Buf *b, uint64_t v) { b_bytes(b, &v, 8); }
static void b_f32(Buf *b, float v) { b_bytes(b, &v, 4); }
static void b_i32(Buf *b, int32_t v) { b_bytes(b, &v, 4); }
static void b_pad(Buf *b, size_t align) {
    while (b->len % align) b_u8(b, 0);
}

/* GGUF string (u64 length + bytes, no NUL). */
static void b_str(Buf *b, const char *s) {
    uint64_t n = strlen(s);
    b_u64(b, n);
    b_bytes(b, s, (size_t)n);
}

/* metadata key/value helpers */
static void kv_key(Buf *b, const char *k) { b_str(b, k); }
static void kv_str(Buf *b, const char *k, const char *v) {
    kv_key(b, k); b_u32(b, KGGUF_STRING); b_str(b, v);
}
static void kv_u32(Buf *b, const char *k, uint32_t v) {
    kv_key(b, k); b_u32(b, KGGUF_UINT32); b_u32(b, v);
}
static void kv_f32(Buf *b, const char *k, float v) {
    kv_key(b, k); b_u32(b, KGGUF_FLOAT32); b_f32(b, v);
}
static void kv_bool(Buf *b, const char *k, int v) {
    kv_key(b, k); b_u32(b, KGGUF_BOOL); b_u8(b, (uint8_t)(v ? 1 : 0));
}
static void kv_arr_begin(Buf *b, const char *k, uint32_t elem_type, uint64_t n) {
    kv_key(b, k); b_u32(b, KGGUF_ARRAY); b_u32(b, elem_type); b_u64(b, n);
}
static void kv_arr_str(Buf *b, const char *k, const char *const *vals, int n) {
    kv_arr_begin(b, k, KGGUF_STRING, (uint64_t)n);
    for (int i = 0; i < n; i++) b_str(b, vals[i]);
}
static void kv_arr_i32(Buf *b, const char *k, const int32_t *vals, int n) {
    kv_arr_begin(b, k, KGGUF_INT32, (uint64_t)n);
    for (int i = 0; i < n; i++) b_i32(b, vals[i]);
}

/* tensor directory entry */
static void tensor_entry(Buf *b, const char *name, uint32_t type,
                         const uint64_t *dims, uint32_t ndims, uint64_t offset) {
    b_str(b, name);
    b_u32(b, ndims);
    for (uint32_t i = 0; i < ndims; i++) b_u64(b, dims[i]);
    b_u32(b, type);
    b_u64(b, offset);
}

static int write_file(const char *path, const Buf *b) {
    FILE *fp = fopen(path, "wb");
    if (!fp) return -1;
    size_t w = fwrite(b->p, 1, b->len, fp);
    fclose(fp);
    return w == b->len ? 0 : -1;
}

/* ==================================================================== *
 * GPT-2 byte-level alphabet (test-local copy)                           *
 * ==================================================================== */
static int byte_is_direct(int b) {
    return (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
}
static int test_b2u(int b) {
    if (byte_is_direct(b)) return b;
    int n = 0;
    for (int x = 0; x < 256; x++) {
        if (!byte_is_direct(x)) { if (x == b) return 256 + n; n++; }
    }
    return b;
}
static int test_utf8(int cp, char *out) {
    if (cp < 0x80) { out[0] = (char)cp; return 1; }
    if (cp < 0x800) {
        out[0] = (char)(0xC0 | (cp >> 6));
        out[1] = (char)(0x80 | (cp & 0x3F));
        return 2;
    }
    out[0] = (char)(0xE0 | (cp >> 12));
    out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
    out[2] = (char)(0x80 | (cp & 0x3F));
    return 3;
}

/* ==================================================================== *
 * Synthetic GGUF builder + parser tests                                 *
 * ==================================================================== */
static void build_min_gguf(Buf *b, uint32_t version, uint32_t tensor_type,
                           uint64_t tensor_offset, int with_data) {
    memset(b, 0, sizeof(*b));
    uint64_t dims[2] = { 4, 2 };
    char sym_a[8], sym_b[8], sym_c[8], sym_sp[8];
    int la = test_utf8(test_b2u('a'), sym_a); sym_a[la] = 0;
    int lb = test_utf8(test_b2u('b'), sym_b); sym_b[lb] = 0;
    int lc = test_utf8(test_b2u('c'), sym_c); sym_c[lc] = 0;
    int ls = test_utf8(test_b2u(' '), sym_sp); sym_sp[ls] = 0;
    const char *toks[4] = { sym_a, sym_b, sym_c, sym_sp };
    int32_t ttypes[4] = { 1, 1, 1, 1 };
    int32_t nums[3] = { 10, 20, 30 };

    b_u32(b, KATALI_GGUF_MAGIC);
    b_u32(b, version);
    b_u64(b, 1);   /* tensor_count */
    b_u64(b, 7);   /* kv_count */
    kv_str(b, "general.architecture", "qwen3");
    kv_u32(b, "qwen3.block_count", 1);
    kv_f32(b, "qwen3.rope.freq_base", 10000.0f);
    kv_bool(b, "general.flag", 1);
    kv_arr_str(b, "tokenizer.ggml.tokens", toks, 4);
    kv_arr_i32(b, "tokenizer.ggml.token_type", ttypes, 4);
    kv_arr_i32(b, "test.nums", nums, 3);
    tensor_entry(b, "t0", tensor_type, dims, 2, tensor_offset);
    b_pad(b, 32);
    if (with_data) {
        for (int i = 0; i < 8; i++) b_f32(b, (float)(i + 1));
    }
}

static void test_parser_ok(void) {
    printf("[parser: valid file]\n");
    Buf b;
    build_min_gguf(&b, 3, KGGML_F32, 0, 1);
    const char *path = ".tmp/test_min.gguf";
    CHECK(write_file(path, &b) == 0, "write synthetic gguf");
    free(b.p);

    KataliGgufFile f;
    char err[256];
    CHECK(katali_gguf_open(&f, path, 1, err, sizeof(err)) == 0, "open valid gguf");
    CHECK(f.version == 3, "version parsed");
    CHECK(f.tensor_count == 1, "tensor count parsed");
    CHECK(f.kv_count == 7, "kv count parsed");
    CHECK(strcmp(katali_gguf_get_str(&f, "general.architecture", "", NULL), "qwen3") == 0,
          "string metadata");
    CHECK(katali_gguf_get_int(&f, "qwen3.block_count", -1) == 1, "int metadata");
    CHECK(fabs(katali_gguf_get_float(&f, "qwen3.rope.freq_base", 0.0) - 10000.0) < 1e-3,
          "float metadata");
    CHECK(katali_gguf_get_bool(&f, "general.flag", 0) == 1, "bool metadata");

    uint32_t et = 0;
    CHECK(katali_gguf_array_len(&f, "tokenizer.ggml.tokens", &et) == 4 &&
          et == KGGUF_STRING, "string array length/type");
    size_t sl = 0;
    const char *t0 = katali_gguf_array_str(&f, "tokenizer.ggml.tokens", 0, &sl);
    CHECK(t0 && sl == 1 && t0[0] == 'a', "string array element");
    CHECK(katali_gguf_array_len(&f, "test.nums", NULL) == 3, "int array length");
    CHECK(katali_gguf_array_int(&f, "test.nums", 1) == 20, "int array element");

    const KataliGgufTensor *t = katali_gguf_find_tensor(&f, "t0");
    CHECK(t != NULL, "tensor found");
    CHECK(t && t->n_dims == 2 && t->dims[0] == 4 && t->dims[1] == 2, "tensor dims");
    CHECK(t && t->n_elements == 8, "tensor element count");
    CHECK(t && t->nbytes == 32, "tensor byte size");
    CHECK(t && t->data != NULL, "tensor data resolved");

    KataliGgufInfo in;
    CHECK(katali_gguf_describe(&f, &in) == 0, "describe");
    CHECK(strcmp(in.architecture, "qwen3") == 0, "describe architecture");
    CHECK(in.layers == 1, "describe layers");

    katali_gguf_close(&f);
}

static void test_parser_reject(void) {
    printf("[parser: rejection cases]\n");
    const char *path = ".tmp/test_bad.gguf";
    KataliGgufFile f;
    char err[256];
    Buf b;

    /* bad magic */
    build_min_gguf(&b, 3, KGGML_F32, 0, 1);
    b.p[0] = 'X';
    write_file(path, &b);
    CHECK(katali_gguf_open(&f, path, 1, err, sizeof(err)) != 0, "reject bad magic");
    free(b.p);

    /* unsupported version */
    build_min_gguf(&b, 99, KGGML_F32, 0, 1);
    write_file(path, &b);
    CHECK(katali_gguf_open(&f, path, 1, err, sizeof(err)) != 0, "reject bad version");
    free(b.p);

    /* truncated file */
    build_min_gguf(&b, 3, KGGML_F32, 0, 1);
    b.len -= 8;
    write_file(path, &b);
    CHECK(katali_gguf_open(&f, path, 1, err, sizeof(err)) != 0, "reject truncated file");
    free(b.p);

    /* tensor offset far past the end */
    build_min_gguf(&b, 3, KGGML_F32, (uint64_t)1 << 40, 1);
    write_file(path, &b);
    CHECK(katali_gguf_open(&f, path, 1, err, sizeof(err)) != 0, "reject out-of-range tensor offset");
    free(b.p);

    /* unsupported tensor type: parses, but exposes no data pointer */
    build_min_gguf(&b, 3, 200, 0, 1);
    write_file(path, &b);
    if (katali_gguf_open(&f, path, 1, err, sizeof(err)) == 0) {
        const KataliGgufTensor *t = katali_gguf_find_tensor(&f, "t0");
        CHECK(t && t->data == NULL && t->nbytes == 0, "unsupported type exposes no data");
        CHECK(!katali_ggml_type_supported(200), "type 200 reported unsupported");
        katali_gguf_close(&f);
    } else {
        CHECK(0, "unsupported type should still parse the directory");
    }
    free(b.p);

    /* nested array must be rejected */
    memset(&b, 0, sizeof(b));
    b_u32(&b, KATALI_GGUF_MAGIC);
    b_u32(&b, 3);
    b_u64(&b, 0);
    b_u64(&b, 1);
    kv_key(&b, "bad");
    b_u32(&b, KGGUF_ARRAY);
    b_u32(&b, KGGUF_ARRAY);
    b_u64(&b, 1);
    write_file(path, &b);
    CHECK(katali_gguf_open(&f, path, 1, err, sizeof(err)) != 0, "reject nested metadata array");
    free(b.p);
}

/* ==================================================================== *
 * Tiny synthetic Qwen3 model (full forward path, no real weights)       *
 * ==================================================================== */
#define TINY_HIDDEN 8
#define TINY_LAYERS 1
#define TINY_HEADS  2
#define TINY_KV     1
#define TINY_HD     4
#define TINY_FFN    16
#define TINY_VOCAB  256
#define TINY_CTX    64

typedef struct TinyTensor {
    const char *name;
    uint64_t dims[2];
    uint32_t ndims;
} TinyTensor;

static const TinyTensor tiny_tensors[] = {
    { "token_embd.weight",         { TINY_HIDDEN, TINY_VOCAB }, 2 },
    { "output_norm.weight",        { TINY_HIDDEN, 1 },          1 },
    { "blk.0.attn_norm.weight",    { TINY_HIDDEN, 1 },          1 },
    { "blk.0.ffn_norm.weight",     { TINY_HIDDEN, 1 },          1 },
    { "blk.0.attn_q.weight",       { TINY_HIDDEN, TINY_HEADS * TINY_HD }, 2 },
    { "blk.0.attn_k.weight",       { TINY_HIDDEN, TINY_KV * TINY_HD },    2 },
    { "blk.0.attn_v.weight",       { TINY_HIDDEN, TINY_KV * TINY_HD },    2 },
    { "blk.0.attn_output.weight",  { TINY_HEADS * TINY_HD, TINY_HIDDEN }, 2 },
    { "blk.0.ffn_gate.weight",     { TINY_HIDDEN, TINY_FFN },    2 },
    { "blk.0.ffn_up.weight",       { TINY_HIDDEN, TINY_FFN },    2 },
    { "blk.0.ffn_down.weight",     { TINY_FFN, TINY_HIDDEN },    2 }
};
#define TINY_NTENSORS ((int)(sizeof(tiny_tensors) / sizeof(tiny_tensors[0])))

static uint64_t tiny_elems(const TinyTensor *t) {
    uint64_t n = 1;
    for (uint32_t i = 0; i < t->ndims; i++) n *= t->dims[i];
    return n;
}

static void build_tiny_model(Buf *b) {
    memset(b, 0, sizeof(*b));
    b_u32(b, KATALI_GGUF_MAGIC);
    b_u32(b, 3);
    b_u64(b, TINY_NTENSORS);
    b_u64(b, 13);

    kv_str(b, "general.architecture", "qwen3");
    kv_u32(b, "qwen3.block_count", TINY_LAYERS);
    kv_u32(b, "qwen3.embedding_length", TINY_HIDDEN);
    kv_u32(b, "qwen3.attention.head_count", TINY_HEADS);
    kv_u32(b, "qwen3.attention.head_count_kv", TINY_KV);
    kv_u32(b, "qwen3.attention.key_length", TINY_HD);
    kv_u32(b, "qwen3.feed_forward_length", TINY_FFN);
    kv_u32(b, "qwen3.context_length", TINY_CTX);
    kv_f32(b, "qwen3.attention.layer_norm_rms_epsilon", 1e-5f);
    kv_f32(b, "qwen3.rope.freq_base", 10000.0f);

    /* byte-level vocabulary covering all 256 raw bytes */
    {
        char symbols[TINY_VOCAB][8];
        const char *ptrs[TINY_VOCAB];
        int32_t types[TINY_VOCAB];
        for (int i = 0; i < TINY_VOCAB; i++) {
            int n = test_utf8(test_b2u(i), symbols[i]);
            symbols[i][n] = 0;
            ptrs[i] = symbols[i];
            types[i] = 1;
        }
        kv_arr_str(b, "tokenizer.ggml.tokens", ptrs, TINY_VOCAB);
        kv_arr_i32(b, "tokenizer.ggml.token_type", types, TINY_VOCAB);
    }
    kv_u32(b, "tokenizer.ggml.eos_token_id", 0);

    /* tensor directory with cumulative offsets */
    uint64_t offset = 0;
    for (int i = 0; i < TINY_NTENSORS; i++) {
        tensor_entry(b, tiny_tensors[i].name, KGGML_F32, tiny_tensors[i].dims,
                     tiny_tensors[i].ndims, offset);
        offset += tiny_elems(&tiny_tensors[i]) * 4;
    }
    b_pad(b, 32);
    /* data */
    for (int i = 0; i < TINY_NTENSORS; i++) {
        uint64_t n = tiny_elems(&tiny_tensors[i]);
        for (uint64_t k = 0; k < n; k++) {
            float v = (float)(((k * 7 + (uint64_t)i * 3) % 17) - 8) * 0.02f;
            b_f32(b, v);
        }
    }
}

/* ==================================================================== *
 * Full forward-path tests on the tiny model                             *
 * ==================================================================== */
static void test_tiny_model(void) {
    printf("[model: tiny synthetic qwen3 forward path]\n");
    Buf b;
    build_tiny_model(&b);
    const char *path = ".tmp/test_tiny.gguf";
    CHECK(write_file(path, &b) == 0, "write tiny model gguf");
    free(b.p);

    KataliBackendOptions opts;
    memset(&opts, 0, sizeof(opts));
    opts.model_path = path;
    opts.n_threads = 2;
    opts.context_length = TINY_CTX;
    opts.max_tokens = 8;
    opts.use_mmap = 1;

    KataliGgufModel m;
    char err[320];
    int rc = katali_gguf_model_open(&m, &opts, err, sizeof(err));
    CHECK(rc == 0, "tiny model opens");
    if (rc != 0) { printf("    (%s)\n", err); return; }
    CHECK(m.n_layers == TINY_LAYERS, "tiny layers");
    CHECK(m.hidden == TINY_HIDDEN, "tiny hidden");
    CHECK(m.head_dim == TINY_HD && m.n_heads == TINY_HEADS && m.n_kv_heads == TINY_KV,
          "tiny attention dims");
    CHECK(m.q_dim == TINY_HEADS * TINY_HD && m.kv_dim == TINY_KV * TINY_HD,
          "tiny q/kv dims");
    CHECK(m.tied == 1, "output.weight tied to embedding when absent");
    CHECK(m.has_qk_norm == 0 && m.has_bias == 0, "qk-norm/bias detected as absent");
    CHECK(katali_gguf_model_resident_bytes(&m) > 0, "resident bytes nonzero");

    char out1[512], out2[512];
    int n1 = katali_gguf_model_generate(&m, "abc", out1, sizeof(out1));
    CHECK(n1 >= 0, "tiny generate returns >= 0");
    CHECK(m.kv_len > 0, "KV length advanced by prefill");
    CHECK(out1[0] != '\0' || n1 == 0, "output is a valid C string");

    katali_gguf_model_reset(&m);
    CHECK(m.kv_len == 0, "reset clears KV length");

    int n2 = katali_gguf_model_generate(&m, "abc", out2, sizeof(out2));
    CHECK(n2 >= 0, "second generate after reset");
    CHECK(n2 == n1 && strcmp(out1, out2) == 0,
          "greedy output identical after state reset (no stale KV)");

    /* Cancellation: a pre-set cancel makes generate return without producing
     * tokens, and reset_state clears it for the next request. */
    katali_gguf_model_cancel(&m);
    char out3[512];
    int n3 = katali_gguf_model_generate(&m, "abc", out3, sizeof(out3));
    CHECK(n3 == 0, "cancelled generate yields no tokens");
    katali_gguf_model_reset(&m);
    char out4[512];
    int n4 = katali_gguf_model_generate(&m, "abc", out4, sizeof(out4));
    CHECK(n4 == n1, "generate works again after cancel + reset");

    katali_gguf_model_close(&m);

    /* A non-qwen architecture must be refused loudly. */
    {
        Buf b2;
        build_tiny_model(&b2);
        /* patch architecture string "qwen3" -> "llama" in place */
        for (size_t i = 0; i + 5 < b2.len; i++) {
            if (memcmp(b2.p + i, "qwen3", 5) == 0) { memcpy(b2.p + i, "llama", 5); break; }
        }
        const char *p2 = ".tmp/test_other.gguf";
        write_file(p2, &b2);
        free(b2.p);
        KataliBackendOptions o2 = opts;
        o2.model_path = p2;
        KataliGgufModel m2;
        CHECK(katali_gguf_model_open(&m2, &o2, err, sizeof(err)) != 0,
              "non-qwen architecture refused");
    }
}

static void test_backend(void) {
    printf("[backend: registry + interface]\n");
    CHECK(katali_backend_lookup("gguf") != NULL, "gguf backend registered");
    CHECK(katali_backend_lookup("nope") == NULL, "unknown backend returns NULL");

    KataliBackendOptions opts;
    memset(&opts, 0, sizeof(opts));
    opts.model_path = ".tmp/test_tiny.gguf";
    opts.context_length = TINY_CTX;
    opts.max_tokens = 4;
    opts.use_mmap = 1;

    KataliBackend *bk = katali_backend_create("gguf", &opts);
    CHECK(bk != NULL, "backend_create opens gguf");
    if (bk) {
        CHECK(strcmp(bk->name, "gguf") == 0, "backend name reported");
        char out[256];
        int n = bk->ops->generate(bk, "abc", out, sizeof(out));
        CHECK(n >= 0, "backend generate");
        CHECK(bk->ops->resident_bytes(bk) > 0, "backend resident_bytes");
        CHECK(bk->ops->reset_state(bk) == 0, "backend reset_state");
        CHECK(bk->ops->cancel(bk) == 0, "backend cancel");
        int n2 = bk->ops->generate(bk, "abc", out, sizeof(out));
        CHECK(n2 == 0, "backend generate after cancel yields no tokens");
        katali_backend_destroy(bk);
    }
}

static void test_sampler(void) {
    printf("[sampler]\n");
    KataliGgufSampler s;
    katali_gguf_sampler_init(&s);
    float logits[5] = { -1.0f, 3.0f, 0.5f, 2.0f, -2.0f };
    CHECK(katali_gguf_sample(&s, logits, 5, NULL, 0) == 1, "greedy picks argmax");
    s.temperature = 1.0f;
    s.seed = 12345u;
    int seen = 0;
    for (int i = 0; i < 50; i++) {
        int id = katali_gguf_sample(&s, logits, 5, NULL, 0);
        if (id < 0 || id >= 5) { seen = -1; break; }
    }
    CHECK(seen == 0, "temperature sampling stays in range");
    katali_gguf_sampler_free(&s);
}

/* ==================================================================== *
 * Batched matmul vs reference / per-token matvec                        *
 * ==================================================================== */
static uint32_t tc_rng_state = 0x1234567u;
static uint32_t tc_rng(void) {
    uint32_t x = tc_rng_state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    tc_rng_state = x ? x : 1u;
    return tc_rng_state;
}

static void fill_matrix_bytes(uint32_t type, uint8_t *dst, uint64_t rows,
                              uint64_t cols) {
    uint64_t bs = katali_ggml_type_block_size(type);
    uint64_t bb = katali_ggml_type_block_bytes(type);
    uint64_t total = rows * cols;
    uint64_t nblocks = total / bs;
    if (type == KGGML_F32) {
        for (uint64_t i = 0; i < total; i++) {
            float v = ((float)(tc_rng() % 2001) - 1000.0f) / 500.0f;
            memcpy(dst + 4 * i, &v, 4);
        }
        return;
    }
    if (type == KGGML_F16) {
        for (uint64_t i = 0; i < total; i++) {
            float v = ((float)(tc_rng() % 2001) - 1000.0f) / 500.0f;
            uint16_t h = katali_f32_to_f16(v);
            memcpy(dst + 2 * i, &h, 2);
        }
        return;
    }
    for (uint64_t i = 0; i < nblocks * bb; i++) dst[i] = (uint8_t)tc_rng();
    uint16_t half = katali_f32_to_f16(0.05f);
    for (uint64_t b = 0; b < nblocks; b++) {
        uint8_t *blk = dst + b * bb;
        switch (type) {
            case KGGML_Q8_0: memcpy(blk, &half, 2); break;
            case KGGML_Q4_K: memcpy(blk, &half, 2); memcpy(blk + 2, &half, 2); break;
            case KGGML_Q6_K: memcpy(blk + 208, &half, 2); break;
            default: break;
        }
    }
}

static void test_matmul(void) {
    printf("[matmul: batched vs per-token matvec and reference]\n");
    static const uint32_t types[] = { KGGML_F32, KGGML_F16, KGGML_Q8_0,
                                      KGGML_Q4_K, KGGML_Q6_K };
    const uint64_t rows = 16, cols = 256, B = 4;
    for (size_t ti = 0; ti < sizeof(types) / sizeof(types[0]); ti++) {
        uint32_t type = types[ti];
        uint64_t rb = katali_ggml_row_bytes(type, cols);
        if (rb == 0) { CHECK(0, "matmul test row bytes"); continue; }
        uint8_t *w = (uint8_t *)malloc((size_t)(rb * rows));
        float *X = (float *)malloc((size_t)(B * cols) * sizeof(float));
        float *Y = (float *)malloc((size_t)(B * rows) * sizeof(float));
        float *yt = (float *)malloc((size_t)rows * sizeof(float));
        if (!w || !X || !Y || !yt) { free(w); free(X); free(Y); free(yt); return; }

        fill_matrix_bytes(type, w, rows, cols);
        for (uint64_t i = 0; i < B * cols; i++)
            X[i] = ((float)(tc_rng() % 2001) - 1000.0f) / 500.0f;

        katali_ggml_matmul(type, w, rows, cols, X, B, Y, 2, NULL);

        int bad_ref = 0, bad_mv = 0;
        for (uint64_t t = 0; t < B; t++) {
            katali_ggml_matvec(type, w, rows, cols, X + t * cols, yt, 2, NULL, NULL, 0);
            for (uint64_t r = 0; r < rows; r++) {
                float ref = katali_ggml_vec_dot_ref(type, w + r * rb,
                                                    X + t * cols, cols);
                float got = Y[t * rows + r];
                float tol = 1e-4f + 1e-3f * fabsf(ref);
                if (!(fabsf(got - ref) <= tol) && !(isnan(got) && isnan(ref)))
                    bad_ref++;
                if (fabsf(got - yt[r]) > 1e-3f + 1e-3f * fabsf(yt[r])) bad_mv++;
            }
        }
        if (bad_ref) printf("  FAIL %-5s matmul vs reference (%d values)\n",
                            katali_ggml_type_name(type), bad_ref);
        CHECK(bad_ref == 0, "batched matmul matches reference");
        if (bad_mv) printf("  FAIL %-5s matmul vs matvec (%d values)\n",
                           katali_ggml_type_name(type), bad_mv);
        CHECK(bad_mv == 0, "batched matmul matches per-token matvec");
        free(w); free(X); free(Y); free(yt);
    }
}

/* ==================================================================== *
 * Batched prefill must reproduce the sequential path                    *
 * ==================================================================== */
static void test_batched_prefill(void) {
    printf("[prefill: batched vs token-at-a-time]\n");
    const char *path = ".tmp/test_tiny.gguf";
    KataliBackendOptions opts;
    memset(&opts, 0, sizeof(opts));
    opts.model_path = path;
    opts.n_threads = 2;
    opts.context_length = TINY_CTX;
    opts.max_tokens = 0; /* prefill only: logits stay comparable */
    opts.use_mmap = 1;

    KataliGgufModel m;
    char err[320];
    if (katali_gguf_model_open(&m, &opts, err, sizeof(err)) != 0) {
        CHECK(0, "tiny model open for prefill test");
        return;
    }
    m.max_tokens = 0;
    const int V = m.vocab;
    float *logits_seq = (float *)malloc((size_t)V * sizeof(float));
    float *logits_bat = (float *)malloc((size_t)V * sizeof(float));
    if (!logits_seq || !logits_bat) {
        free(logits_seq); free(logits_bat);
        katali_gguf_model_close(&m);
        return;
    }
    char out[256];
    /* 20 characters -> 20 byte-level tokens with the tiny vocabulary, which is
     * above KATALI_PREFILL_BATCH_MIN (8). */
    const char *long_prompt = "abcdefghijklmnopqrst";

    katali_gguf_model_set_batch_prefill(&m, 0);
    int ns = katali_gguf_model_generate(&m, long_prompt, out, sizeof(out));
    int seq_tokens = m.last.prompt_tokens, seq_kv = m.kv_len, seq_b = m.last.batched_prefill;
    memcpy(logits_seq, m.logits, (size_t)V * sizeof(float));

    katali_gguf_model_reset(&m);
    katali_gguf_model_set_batch_prefill(&m, 1);
    int nb = katali_gguf_model_generate(&m, long_prompt, out, sizeof(out));
    int bat_tokens = m.last.prompt_tokens, bat_kv = m.kv_len, bat_b = m.last.batched_prefill;
    memcpy(logits_bat, m.logits, (size_t)V * sizeof(float));

    CHECK(ns >= 0 && nb >= 0, "both prefill paths succeed");
    CHECK(seq_b == 0, "sequential path reports batch=0");
    CHECK(bat_b == 1, "batched path reports batch=1");
    CHECK(seq_tokens == bat_tokens && seq_tokens > 8,
          "same prompt token count on both paths");
    CHECK(seq_kv == bat_kv && seq_kv == seq_tokens,
          "KV length equals prompt length on both paths");

    float maxdiff = 0.0f;
    int bad = 0;
    int arg_seq = katali_gguf_argmax(logits_seq, (size_t)V);
    int arg_bat = katali_gguf_argmax(logits_bat, (size_t)V);
    for (int i = 0; i < V; i++) {
        float d = fabsf(logits_bat[i] - logits_seq[i]);
        float tol = 1e-3f + 2e-3f * fabsf(logits_seq[i]);
        if (d > maxdiff) maxdiff = d;
        if (d > tol) bad++;
    }
    printf("  prefill logits max|diff|=%.3g differing=%d/%d argmax seq=%d bat=%d\n",
           maxdiff, bad, V, arg_seq, arg_bat);
    CHECK(bad == 0, "batched prefill logits match sequential within tolerance");
    CHECK(arg_seq == arg_bat, "greedy argmax identical for both prefill paths");

    /* A short prompt must stay on the sequential path. */
    katali_gguf_model_reset(&m);
    katali_gguf_model_set_batch_prefill(&m, 1);
    katali_gguf_model_generate(&m, "abc", out, sizeof(out));
    CHECK(m.last.prompt_tokens < 8 && m.last.batched_prefill == 0,
          "short prompt uses the sequential path");

    free(logits_seq); free(logits_bat);
    katali_gguf_model_close(&m);
}

/* ==================================================================== *
 * Q4_K hoisted activation sums vs three references                      *
 * ==================================================================== */
static void test_q4k_sum_hoist(void) {
    printf("[q4k: hoisted activation sums vs three references]\n");
    if (!katali_ggml_simd_available()) {
        printf("  (SIMD unavailable on this CPU; hoisted path not testable.\n"
               "   The per-row fallback is covered by the dtype selftest.)\n");
        CHECK(1, "q4k hoist skipped (no SIMD)");
        return;
    }
    const uint64_t rows = 24, cols = 1024; /* 4 Q4_K blocks per row */
    const uint64_t sums_cap = KATALI_Q4K_SUMS_FLOATS(cols);
    uint64_t rb = katali_ggml_row_bytes(KGGML_Q4_K, cols);
    if (rb == 0 || sums_cap == 0) { CHECK(0, "q4k hoist setup"); return; }

    uint8_t *w = (uint8_t *)malloc((size_t)(rb * rows));
    float *x = (float *)malloc((size_t)cols * sizeof(float));
    float *sums = (float *)malloc((size_t)sums_cap * sizeof(float));
    float *sums_zero = (float *)malloc((size_t)sums_cap * sizeof(float));
    if (!w || !x || !sums || !sums_zero) {
        free(w); free(x); free(sums); free(sums_zero);
        CHECK(0, "q4k hoist buffers");
        return;
    }
    fill_matrix_bytes(KGGML_Q4_K, w, rows, cols);
    for (uint64_t i = 0; i < cols; i++)
        x[i] = ((float)(tc_rng() % 2001) - 1000.0f) / 500.0f;

    int rc = katali_ggml_q4k_sums(x, cols, sums, sums_cap);
    if (rc != 0) {
        printf("  (sum hoisting disabled by KATALI_GGUF_NO_Q4K_SUMCACHE=1; skipped)\n");
        CHECK(1, "q4k sum hoisting disabled via env (expected skip)");
        free(w); free(x); free(sums); free(sums_zero);
        return;
    }
    CHECK(rc == 0, "q4k activation sums computed");
    memset(sums_zero, 0, (size_t)sums_cap * sizeof(float));

    int exact_old = 0, exact_scalar = 0, exact_ref = 0;
    int rows_used = 0, differ_with_zero_sums = 0;
    double max_abs_old = 0.0, max_rel_old = 0.0;
    double max_abs_sc = 0.0, max_rel_sc = 0.0;
    double max_abs_rf = 0.0, max_rel_rf = 0.0;
    int over_tol_sc = 0, over_tol_rf = 0;

    for (uint64_t r = 0; r < rows; r++) {
        const uint8_t *row = w + r * rb;
        float v_new = 0.0f, v_old = 0.0f, v_zero = 0.0f;
        if (katali_ggml_q4k_dot_sums(row, x, cols, sums, &v_new) != 0) continue;
        if (katali_ggml_vec_dot_avx2(KGGML_Q4_K, row, x, cols, &v_old) != 0) continue;
        float v_scalar = katali_ggml_vec_dot_scalar(KGGML_Q4_K, row, x, cols);
        float v_ref = katali_ggml_vec_dot_ref(KGGML_Q4_K, row, x, cols);
        (void)katali_ggml_q4k_dot_sums(row, x, cols, sums_zero, &v_zero);
        rows_used++;

        /* The hoist must be a pure re-association-free move of the sum
         * computation: bit-identical to the unmodified AVX2 kernel. */
        if (memcmp(&v_new, &v_old, sizeof(float)) == 0) exact_old++;
        else {
            double d = fabs((double)v_new - (double)v_old);
            double rel = d / (fabs((double)v_old) + 1e-12);
            if (d > max_abs_old) max_abs_old = d;
            if (rel > max_rel_old) max_rel_old = rel;
        }
        if (memcmp(&v_new, &v_scalar, sizeof(float)) == 0) exact_scalar++;
        else {
            double d = fabs((double)v_new - (double)v_scalar);
            double rel = d / (fabs((double)v_scalar) + 1e-12);
            if (d > max_abs_sc) max_abs_sc = d;
            if (rel > max_rel_sc) max_rel_sc = rel;
        }
        if (memcmp(&v_new, &v_ref, sizeof(float)) == 0) exact_ref++;
        else {
            double d = fabs((double)v_new - (double)v_ref);
            double rel = d / (fabs((double)v_ref) + 1e-12);
            if (d > max_abs_rf) max_abs_rf = d;
            if (rel > max_rel_rf) max_rel_rf = rel;
        }
        float tol = 1e-4f + 1e-3f * fabsf(v_ref);
        if (fabsf(v_new - v_scalar) > tol) over_tol_sc++;
        if (fabsf(v_new - v_ref) > tol) over_tol_rf++;
        if (v_zero != v_new) differ_with_zero_sums++;
    }

    printf("  rows=%d/%llu\n", rows_used, (unsigned long long)rows);
    printf("  vs unmodified AVX2 : exact=%d/%d  max_abs=%.3g max_rel=%.3g\n",
           exact_old, rows_used, max_abs_old, max_rel_old);
    printf("  vs scalar dot_q4_K : exact=%d/%d  max_abs=%.3g max_rel=%.3g\n",
           exact_scalar, rows_used, max_abs_sc, max_rel_sc);
    printf("  vs generic ref     : exact=%d/%d  max_abs=%.3g max_rel=%.3g\n",
           exact_ref, rows_used, max_abs_rf, max_rel_rf);

    CHECK(rows_used == (int)rows, "q4k hoist rows evaluated");
    /* Bit-identical to the kernel it hoists from. */
    CHECK(exact_old == rows_used, "hoisted Q4_K is bit-identical to unmodified AVX2 kernel");
    /* Scalar/reference use a different accumulation order, so compare within
     * the same tolerance the dtype selftest already uses. */
    CHECK(over_tol_sc == 0, "hoisted Q4_K matches scalar dot_q4_K within 1e-4+1e-3|ref|");
    CHECK(over_tol_rf == 0, "hoisted Q4_K matches generic reference within 1e-4+1e-3|ref|");
    /* Guard against the sums being computed but silently unused. */
    CHECK(differ_with_zero_sums == rows_used,
          "zeroed sums change the result (sums are actually consumed)");

    free(w); free(x); free(sums); free(sums_zero);
}

/* ==================================================================== *
 * Multi-row F32 dot (the prefill matmul's inner step)                    *
 * ==================================================================== */
static void test_f32_rows_dot(void) {
    printf("[f32: multi-row dot is bit-identical to the single-row kernel]\n");
    const uint64_t widths[3] = { 16, 20, 1024 };   /* 20 exercises the scalar tail */
    for (int ci = 0; ci < 3; ci++) {
        const uint64_t n = widths[ci];
        float *x = (float *)malloc((size_t)n * sizeof(float));
        float *r[4];
        for (int j = 0; j < 4; j++) r[j] = (float *)malloc((size_t)n * sizeof(float));
        if (!x || !r[0] || !r[1] || !r[2] || !r[3]) { CHECK(0, "f32 rows buffers"); return; }
        for (uint64_t i = 0; i < n; i++) {
            x[i] = (float)((int)(i % 17) - 8) * 0.25f;
            for (int j = 0; j < 4; j++)
                r[j][i] = (float)((int)((i * (uint64_t)(j + 3)) % 23) - 11) * 0.125f;
        }
        int ok = 1;
        const int nrs[3] = { 1, 2, 4 };
        for (int k = 0; k < 3; k++) {
            int nr = nrs[k];
            float out[4] = { 0, 0, 0, 0 };
            const float *rp[4] = { r[0], r[1], r[2], r[3] };
            katali_ggml_f32_rows_dot(rp, nr, x, n, out);
            for (int j = 0; j < nr; j++) {
                float ref = katali_ggml_vec_dot(KGGML_F32, r[j], x, n);
                if (out[j] != ref) ok = 0;
            }
        }
        CHECK(ok, "f32 rows dot bit-identical (1, 2 and 4 rows)");
        free(x);
        for (int j = 0; j < 4; j++) free(r[j]);
    }
}

/* ==================================================================== *
 * Prefill row decode: AVX2 materializing decoder vs the reference       *
 * ==================================================================== */
static void test_dequant_rows(void) {
    printf("[prefill: AVX2 row decode vs the scalar reference]\n");
    if (!katali_ggml_simd_available()) {
        printf("  (SIMD unavailable on this CPU; AVX2 row decoder not testable.\n"
               "   The wrapper falls back to the reference, which the batched\n"
               "   matmul and prefill tests already cover.)\n");
        CHECK(1, "dequant rows skipped (no SIMD)");
        return;
    }
    /* Block multiples the format allows, including a real 4B row width. */
    static const uint64_t widths[4] = { 256, 512, 1024, 2560 };
    const uint64_t rows = 5;                 /* odd, >1 and >4 */
    int all_exact = 1, all_accepted = 1, ref_ok = 1;
    for (int wi = 0; wi < 4; wi++) {
        const uint64_t cols = widths[wi];
        const uint64_t rb = katali_ggml_row_bytes(KGGML_Q4_K, cols);
        if (rb == 0) { CHECK(0, "dequant rows setup"); return; }
        uint8_t *w = (uint8_t *)malloc((size_t)(rb * rows));
        float *a = (float *)malloc((size_t)cols * sizeof(float));
        float *b = (float *)malloc((size_t)cols * sizeof(float));
        if (!w || !a || !b) { free(w); free(a); free(b); CHECK(0, "dequant rows buffers"); return; }
        fill_matrix_bytes(KGGML_Q4_K, w, rows, cols);
        unsigned long long exact = 0, total = 0;
        double maxabs = 0.0;
        for (uint64_t r = 0; r < rows; r++) {
            const uint8_t *row = w + r * rb;
            if (katali_ggml_dequant_ref(KGGML_Q4_K, row, cols, a) != 0) ref_ok = 0;
            if (katali_ggml_dequant_rows_avx2(KGGML_Q4_K, row, cols, b) != 0) {
                all_accepted = 0;
                continue;
            }
            for (uint64_t i = 0; i < cols; i++) {
                double d = (double)a[i] - (double)b[i];
                if (d < 0) d = -d;
                if (d > maxabs) maxabs = d;
                if (a[i] == b[i]) exact++;
                total++;
            }
        }
        printf("  Q4_K cols=%-5llu rows=%llu: exact %llu/%llu  maxabs %.3g\n",
               (unsigned long long)cols, (unsigned long long)rows, exact, total, maxabs);
        if (exact != total) all_exact = 0;
        free(w); free(a); free(b);
    }
    CHECK(ref_ok, "scalar reference accepts the tested Q4_K widths");
    CHECK(all_accepted, "AVX2 row decode accepts every tested Q4_K width");
    CHECK(all_exact, "AVX2 row decode is bit-identical to the reference (Q4_K)");

    /* Q6_K: same contract, different block layout (210 bytes, int8 scales, signed
     * 6-bit values), so it is tested separately rather than assumed. */
    {
        static const uint64_t w6[5] = { 256, 512, 1024, 3072, 4096 };
        int exact6 = 1, accepted6 = 1, ref6_ok = 1;
        for (int wi = 0; wi < 5; wi++) {
            const uint64_t cols = w6[wi];
            const uint64_t rb = katali_ggml_row_bytes(KGGML_Q6_K, cols);
            if (rb == 0) { CHECK(0, "Q6_K dequant rows setup"); return; }
            uint8_t *w = (uint8_t *)malloc((size_t)(rb * rows));
            float *a = (float *)malloc((size_t)cols * sizeof(float));
            float *b = (float *)malloc((size_t)cols * sizeof(float));
            if (!w || !a || !b) { free(w); free(a); free(b); CHECK(0, "Q6_K dequant rows buffers"); return; }
            fill_matrix_bytes(KGGML_Q6_K, w, rows, cols);
            unsigned long long exact = 0, total = 0;
            double maxabs = 0.0, maxrel = 0.0;
            for (uint64_t r = 0; r < rows; r++) {
                const uint8_t *row = w + r * rb;
                if (katali_ggml_dequant_ref(KGGML_Q6_K, row, cols, a) != 0) ref6_ok = 0;
                if (katali_ggml_dequant_rows_avx2(KGGML_Q6_K, row, cols, b) != 0) {
                    accepted6 = 0;
                    continue;
                }
                for (uint64_t i = 0; i < cols; i++) {
                    double d = (double)a[i] - (double)b[i];
                    if (d < 0) d = -d;
                    if (d > maxabs) maxabs = d;
                    double ar = a[i] < 0 ? -a[i] : a[i];
                    if (ar > 1e-30 && d / ar > maxrel) maxrel = d / ar;
                    if (a[i] == b[i]) exact++;
                    total++;
                }
            }
            printf("  Q6_K cols=%-5llu rows=%llu: exact %llu/%llu  maxabs %.3g  maxrel %.3g\n",
                   (unsigned long long)cols, (unsigned long long)rows, exact, total,
                   maxabs, maxrel);
            if (exact != total) exact6 = 0;
            free(w); free(a); free(b);
        }
        CHECK(ref6_ok, "scalar reference accepts the tested Q6_K widths");
        CHECK(accepted6, "AVX2 row decode accepts every tested Q6_K width");
        CHECK(exact6, "AVX2 row decode is bit-identical to the reference (Q6_K)");
    }

    /* The wrapper must agree with the reference for types with no AVX2 decoder,
     * and must reject widths the format cannot represent (so the caller falls
     * back rather than producing garbage). */
    {
        const uint64_t cols = 3072, rb = katali_ggml_row_bytes(KGGML_Q6_K, cols);
        uint8_t *w = (uint8_t *)malloc((size_t)rb);
        float *a = (float *)malloc((size_t)cols * sizeof(float));
        float *b = (float *)malloc((size_t)cols * sizeof(float));
        if (w && a && b) {
            fill_matrix_bytes(KGGML_Q6_K, w, 1, cols);
            katali_ggml_dequant_ref(KGGML_Q6_K, w, cols, a);
            int rc = katali_ggml_dequant_rows(KGGML_Q6_K, w, cols, b);
            int same = (rc == 0);
            for (uint64_t i = 0; i < cols && same; i++) if (a[i] != b[i]) same = 0;
            CHECK(same, "row-decode wrapper matches the reference (Q6_K)");
            /* A type with no AVX2 decoder at all still goes through the reference. */
            katali_ggml_dequant_ref(KGGML_Q8_0, w, 32, a);
            int rc8 = katali_ggml_dequant_rows(KGGML_Q8_0, w, 32, b);
            int same8 = (rc8 == 0) && (a[0] == b[0]);
            CHECK(same8, "row-decode wrapper falls back to the reference (Q8_0)");
        } else {
            CHECK(0, "wrapper buffers");
        }
        free(w); free(a); free(b);
        uint8_t dummy[512];
        float out[300];
        memset(dummy, 0x11, sizeof(dummy));
        CHECK(katali_ggml_dequant_rows_avx2(KGGML_Q4_K, dummy, 300, out) != 0,
              "AVX2 row decode refuses a non-block width (Q4_K)");
        CHECK(katali_ggml_dequant_rows_avx2(KGGML_Q6_K, dummy, 300, out) != 0,
              "AVX2 row decode refuses a non-block width (Q6_K)");
    }
}

/* ==================================================================== *
 * Q4_K dual-row kernel and row-loop dispatch                            *
 * ==================================================================== */
static void test_q4k_dualrow(void) {
    printf("[q4k: dual-row kernel and row-loop dispatch]\n");
    if (!katali_ggml_simd_available()) {
        printf("  (SIMD unavailable on this CPU; dual-row path not testable)\n");
        CHECK(1, "q4k dual-row skipped (no SIMD)");
        return;
    }
    const uint64_t cols = 1024;
    const uint64_t sums_cap = KATALI_Q4K_SUMS_FLOATS(cols);
    const uint64_t rb = katali_ggml_row_bytes(KGGML_Q4_K, cols);
    const char *nodual = getenv("KATALI_GGUF_NO_Q4K_DUALROW");
    printf("  row-loop mode: %s\n",
           (nodual && *nodual && nodual[0] != '0') ? "single-row (forced by env)"
                                                   : "dual-row (default)");
    if (rb == 0 || sums_cap == 0) { CHECK(0, "q4k dual-row setup"); return; }

    const uint64_t rows = 25; /* 12 pairs + 1 remainder when dual-row is active */
    uint8_t *w = (uint8_t *)malloc((size_t)(rb * rows));
    float *x = (float *)malloc((size_t)cols * sizeof(float));
    float *y = (float *)malloc((size_t)rows * sizeof(float));
    float *sums = (float *)malloc((size_t)sums_cap * sizeof(float));
    if (!w || !x || !y || !sums) {
        free(w); free(x); free(y); free(sums);
        CHECK(0, "q4k dual-row buffers");
        return;
    }
    fill_matrix_bytes(KGGML_Q4_K, w, rows, cols);
    for (uint64_t i = 0; i < cols; i++)
        x[i] = ((float)(tc_rng() % 2001) - 1000.0f) / 500.0f;
    if (katali_ggml_q4k_sums(x, cols, sums, sums_cap) != 0) {
        printf("  (sum hoisting disabled; dual-row needs the sums -- skipped)\n");
        CHECK(1, "q4k dual-row skipped (sum cache disabled by env)");
        free(w); free(x); free(y); free(sums);
        return;
    }

    /* 1. The pair kernel must reproduce two independent single-row results
     *    exactly: pairing must not change either row's arithmetic. */
    int pairs = 0, pair_exact = 0;
    for (uint64_t r = 0; r + 1 < rows; r += 2) {
        float a1 = 0, a2 = 0, b1 = 0, b2 = 0;
        if (katali_ggml_q4k_dot_sums2(w + r * rb, w + (r + 1) * rb, x, cols, sums,
                                      &a1, &a2) != 0) continue;
        if (katali_ggml_q4k_dot_sums(w + r * rb, x, cols, sums, &b1) != 0) continue;
        if (katali_ggml_q4k_dot_sums(w + (r + 1) * rb, x, cols, sums, &b2) != 0) continue;
        pairs++;
        if (memcmp(&a1, &b1, sizeof(float)) == 0 &&
            memcmp(&a2, &b2, sizeof(float)) == 0) pair_exact++;
    }
    CHECK(pairs == 12, "dual-row pair kernel evaluated on 12 pairs");
    CHECK(pair_exact == pairs,
          "dual-row pair output bit-identical to two single-row calls");

    /* 2. The row loop must match the single-row path row by row, for an even
     *    row count and for the odd count that leaves a single-row remainder. */
    for (int pass = 0; pass < 2; pass++) {
        uint64_t n = pass ? rows : rows - 1; /* 25 = 12 pairs + 1, 24 = 12 pairs */
        katali_ggml_matvec(KGGML_Q4_K, w, n, cols, x, y, 1, NULL, sums, sums_cap);
        int exact = 0;
        double max_abs = 0.0, max_rel = 0.0;
        float worst = 0.0f;
        for (uint64_t r = 0; r < n; r++) {
            float single = 0.0f;
            if (katali_ggml_q4k_dot_sums(w + r * rb, x, cols, sums, &single) != 0) continue;
            if (memcmp(&y[r], &single, sizeof(float)) == 0) exact++;
            else {
                double d = fabs((double)y[r] - (double)single);
                double rel = d / (fabs((double)single) + 1e-12);
                if (d > max_abs) { max_abs = d; worst = y[r]; }
                if (rel > max_rel) max_rel = rel;
            }
        }
        printf("  matvec rows=%llu%s : bit-identical %d/%llu  max_abs=%.3g max_rel=%.3g\n",
               (unsigned long long)n, pass ? " (odd, 1 leftover)" : " (even)",
               exact, (unsigned long long)n, max_abs, max_rel);
        (void)worst;
        CHECK(exact == (int)n, pass
              ? "25-row (odd) row loop bit-identical per row"
              : "24-row (even) row loop bit-identical per row");
    }

    /* 3. Sanity against the scalar and generic reference paths, for the record. */
    double max_abs = 0.0, max_rel = 0.0;
    int over_tol = 0;
    for (uint64_t r = 0; r < rows; r++) {
        float v_ref = katali_ggml_vec_dot_ref(KGGML_Q4_K, w + r * rb, x, cols);
        float v_scalar = katali_ggml_vec_dot_scalar(KGGML_Q4_K, w + r * rb, x, cols);
        double d = fabs((double)y[r] - (double)v_ref);
        double rel = d / (fabs((double)v_ref) + 1e-12);
        if (d > max_abs) max_abs = d;
        if (rel > max_rel) max_rel = rel;
        float tol = 1e-4f + 1e-3f * fabsf(v_ref);
        if (fabsf(y[r] - v_ref) > tol || fabsf(y[r] - v_scalar) > tol) over_tol++;
    }
    printf("  vs scalar/reference : max_abs=%.3g max_rel=%.3g\n", max_abs, max_rel);
    CHECK(over_tol == 0, "dual-row row loop matches scalar/reference within 1e-4+1e-3|ref|");

    free(w); free(x); free(y); free(sums);
}

/* ==================================================================== *
 * 64-bit large-file support                                             *
 *                                                                       *
 * The regression: on Windows fseek/ftell use a 32-bit long, so ftell     *
 * returns -1 for a position beyond LONG_MAX and a valid file larger than *
 * 2 GiB was reported as "cannot open file" before it was ever mapped.    *
 * Both open paths run their own size probe - the memory-mapped one and   *
 * the explicit heap copy used by --no-mmap - so both are asserted here.  *
 *                                                                       *
 * The test file is sparse: a normal GGUF header is written, then the file *
 * is extended by one byte just past the 2 GiB boundary. NTFS reports the *
 * unwritten region as zeros without allocating clusters, so this costs a  *
 * few KiB of disk rather than 2 GB.                                      *
 * ==================================================================== */
static int make_sparse_gguf(const char *path, uint64_t total) {
    Buf b;
    build_min_gguf(&b, 3, KGGML_F32, 0, 1);
    int rc = write_file(path, &b);
    free(b.p);
    if (rc != 0) return -1;
    FILE *fp = fopen(path, "r+b");
    if (!fp) return -1;
#ifdef _WIN32
    if (_fseeki64(fp, (long long)(total - 1), SEEK_SET) != 0) { fclose(fp); return -1; }
#else
    if (fseeko(fp, (off_t)(total - 1), SEEK_SET) != 0) { fclose(fp); return -1; }
#endif
    if (fputc(0, fp) == EOF) { fclose(fp); return -1; }
    return fclose(fp) == 0 ? 0 : -1;
}

static void test_large_file_64bit(void) {
    printf("[reader: files past 2 GiB use 64-bit sizes]\n");
    const char *path = ".tmp/test_large.gguf";
    const uint64_t total = (uint64_t)2 * 1024 * 1024 * 1024 + 65536; /* just past LONG_MAX */

    CHECK(make_sparse_gguf(path, total) == 0, "create sparse GGUF just past 2 GiB");

    /* The exact bug: the size probe must not be a 32-bit value. */
    long long probe = katali_gguf_file_size(path);
    if (probe != (long long)total)
        printf("    probe returned %lld, expected %lld\n", probe, (long long)total);
    CHECK(probe == (long long)total, "size probe reports the real 64-bit size");

    KataliGgufFile f;
    char err[256];

    /* Primary path: memory-mapped. */
    err[0] = 0;
    CHECK(katali_gguf_open(&f, path, 1, err, sizeof(err)) == 0,
          "open GGUF larger than 2 GiB (mmap)");
    if (!f.ok) printf("    open error: %s\n", err);
    CHECK(f.size == total, "mmap open reports the 64-bit size");
    CHECK(f.mapped == 1, "mmap open really mapped the file");
    CHECK(f.tensor_count == 1, "metadata parsed from the >2 GiB file");
    katali_gguf_close(&f);

    /* Second call site: the explicit heap copy (what --no-mmap uses). This is
     * the one that is easy to forget, so it is asserted to agree with mmap. */
    memset(&f, 0, sizeof(f));
    err[0] = 0;
    CHECK(katali_gguf_open(&f, path, 0, err, sizeof(err)) == 0,
          "open GGUF larger than 2 GiB (heap copy / --no-mmap path)");
    CHECK(f.size == total, "heap open reports the 64-bit size");
    CHECK(f.mapped == 0, "heap open really copied the file");
    katali_gguf_close(&f);

    remove(path);
}

/* ==================================================================== *
 * Parallel attention: per-worker score buffers, bit-identity            *
 *                                                                       *
 * Heads are independent and both entry points run the same per-head      *
 * body, so the parallel result must be *identical*, not merely close.    *
 * kv_len is above the KATALI_GGUF_ATTN_MIN_KV threshold (256) so the     *
 * parallel path really runs; a shared score buffer would corrupt this.   *
 * ==================================================================== */
static void test_attention_parallel(void) {
    printf("[attention: parallel vs serial]\n");
    enum { AQ = 16, AKV = 8, AHD = 128, ACAP = 320, ALEN = 300, ANT = 4 };
    size_t nq = (size_t)AQ * AHD, nkv = (size_t)AKV * ACAP * AHD;
    float *q = (float *)malloc(nq * sizeof(float));
    float *kk = (float *)malloc(nkv * sizeof(float));
    float *vv = (float *)malloc(nkv * sizeof(float));
    float *out_ser = (float *)malloc(nq * sizeof(float));
    float *out_par = (float *)malloc(nq * sizeof(float));
    float *scores = (float *)malloc(ALEN * sizeof(float));
    float *pool = (float *)malloc((size_t)ANT * ALEN * sizeof(float));
    CHECK(q && kk && vv && out_ser && out_par && scores && pool, "allocate attention test buffers");
    if (!(q && kk && vv && out_ser && out_par && scores && pool)) {
        free(q); free(kk); free(vv); free(out_ser); free(out_par); free(scores); free(pool);
        return;
    }
    unsigned s = 987654321u;
    for (size_t i = 0; i < nq; i++)  { s = s * 1103515245u + 12345u; q[i]  = ((float)((s >> 9) & 0x3FF) / 512.0f) - 1.0f; }
    for (size_t i = 0; i < nkv; i++) {
        s = s * 1103515245u + 12345u; kk[i] = ((float)((s >> 9) & 0x3FF) / 512.0f) - 1.0f;
        s = s * 1103515245u + 12345u; vv[i] = ((float)((s >> 11) & 0x3FF) / 512.0f) - 1.0f;
    }
    memset(out_ser, 0, nq * sizeof(float));
    memset(out_par, 0, nq * sizeof(float));

    katali_gguf_gqa_decode_scratch(q, kk, vv, ALEN, ACAP, AQ, AKV, AHD, out_ser, scores);
    katali_gguf_gqa_decode_parallel(q, kk, vv, ALEN, ACAP, AQ, AKV, AHD, out_par,
                                    scores, pool, ALEN, ANT, NULL);

    size_t diff = 0, first = 0;
    for (size_t i = 0; i < nq; i++) {
        if (memcmp(&out_ser[i], &out_par[i], sizeof(float)) != 0) {
            if (!diff) first = i;
            diff++;
        }
    }
    if (diff) printf("    %llu of %llu outputs differ (first at %llu)\n",
                     (unsigned long long)diff, (unsigned long long)nq,
                     (unsigned long long)first);
    CHECK(diff == 0, "parallel attention is bit-identical to serial");

    /* Cancelled before the dispatch: outputs stay as the memset left them and
     * nothing is written by the workers. That contract belongs to the parallel
     * path only: when the toggle forces the serial fallback, the serial path
     * ignores `cancel` exactly as it always has (the model checks cancellation at
     * layer granularity), so the assertion does not apply. */
    const char *no_par = getenv("KATALI_GGUF_NO_ATTN_PARALLEL");
    if (no_par && *no_par && no_par[0] != '0') {
        printf("    (parallel attention disabled by env: cancellation check skipped)\n");
    } else {
        volatile int cancel = 1;
        memset(out_par, 0, nq * sizeof(float));
        katali_gguf_gqa_decode_parallel(q, kk, vv, ALEN, ACAP, AQ, AKV, AHD, out_par,
                                        scores, pool, ALEN, ANT, &cancel);
        int nonzero = 0;
        for (size_t i = 0; i < nq; i++) if (out_par[i] != 0.0f) { nonzero = 1; break; }
        CHECK(nonzero == 0, "cancelled parallel attention writes nothing");
    }

    free(q); free(kk); free(vv); free(out_ser); free(out_par); free(scores); free(pool);
}

int main(void) {
#ifdef _WIN32
    _mkdir(".tmp");
#else
    mkdir(".tmp", 0755);
#endif
    printf("Katali-GGUF core tests\n");
    test_parser_ok();
    test_parser_reject();
    test_large_file_64bit();
    test_attention_parallel();
    printf("[dtype decoders]\n");
    CHECK(katali_dtype_selftest() == 0, "dtype reference vs optimized selftest");
    printf("[dtype SIMD vs scalar vs reference]\n");
    CHECK(katali_dtype_simd_selftest() == 0, "SIMD acceleration selftest");
    printf("[kernels]\n");
    CHECK(katali_gguf_kernels_selftest() == 0, "kernel selftest");
    test_matmul();
    test_q4k_sum_hoist();
    test_q4k_dualrow();
    test_f32_rows_dot();
    test_dequant_rows();
    test_sampler();
    test_tiny_model();
    test_batched_prefill();
    test_backend();

    printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}



