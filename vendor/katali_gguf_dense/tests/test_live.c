/* Katali-GGUF live tests — Apache-2.0
 *
 * Requires a real Qwen3 GGUF. Pass the path as argv[1] or set
 * KATALI_GGUF_MODEL. With no model the tests SKIP (exit 0).
 */
#include "katali_gguf.h"
#include "katali_gguf_model.h"
#include "katali_backend.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_fail = 0;
static int g_pass = 0;
#define CHECK(cond, msg) do { \
        if (cond) { g_pass++; printf("  PASS  %s\n", msg); } \
        else { g_fail++; printf("  FAIL  %s\n", msg); } \
    } while (0)

static void banner(const char *t) { printf("\n== %s ==\n", t); }

static void make_options(KataliBackendOptions *o, const char *path, int max_tokens) {
    memset(o, 0, sizeof(*o));
    o->model_path = path;
    o->n_threads = 8;
    o->context_length = 512;
    o->max_tokens = max_tokens;
    o->use_mmap = 1;
    o->think = 0;
}

static int write_garbage(const char *path, int truncated) {
    FILE *fp = fopen(path, "wb");
    if (!fp) return -1;
    if (!truncated) {
        unsigned char hdr[16] = { 'G','G','U','F', 3,0,0,0, 0xFF,0xFF,0,0,0,0,0,0 };
        fwrite(hdr, 1, sizeof(hdr), fp);
    } else {
        unsigned char hdr[8] = { 'G','G','U','F', 3,0,0,0 };
        fwrite(hdr, 1, sizeof(hdr), fp);
    }
    fclose(fp);
    return 0;
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : getenv("KATALI_GGUF_MODEL");
    if (!path) {
        printf("SKIP: no model path (argv[1] or KATALI_GGUF_MODEL)\n");
        return 0;
    }
    printf("Katali-GGUF live tests on %s\n", path);

    banner("inspect metadata");
    KataliGgufFile f;
    char err[320];
    if (katali_gguf_open(&f, path, 1, err, sizeof(err)) != 0) {
        printf("  FAIL  cannot open: %s\n", err);
        return 1;
    }
    KataliGgufInfo in;
    katali_gguf_describe(&f, &in);
    printf("  format=%s version=%u arch=%s vocab=%d layers=%d hidden=%d "
           "heads=%d kv_heads=%d ctx=%d quant=%s tensors=%llu\n",
           in.format, in.version, in.architecture, in.vocab, in.layers, in.hidden,
           in.attention_heads, in.key_value_heads, in.context_length, in.quantization,
           (unsigned long long)in.tensor_count);
    CHECK(strcmp(in.architecture, "qwen3") == 0 ||
          strcmp(in.architecture, "qwen2") == 0, "architecture is qwen2/qwen3");
    CHECK(in.vocab > 0 && in.layers > 0 && in.hidden > 0, "core metadata present");
    katali_gguf_close(&f);

    banner("open model once (persistent)");
    KataliBackendOptions opts;
    make_options(&opts, path, 64);
    KataliGgufModel m;
    if (katali_gguf_model_open(&m, &opts, err, sizeof(err)) != 0) {
        printf("  FAIL  model open: %s\n", err);
        return 1;
    }
    printf("  loaded in %.3fs (map %.3fs, parse %.3fs), resident %.1f MB\n",
           m.stats.open_s, m.stats.map_s, m.stats.parse_s,
           (double)katali_gguf_model_resident_bytes(&m) / (1024.0 * 1024.0));
    CHECK(m.n_layers > 0 && m.vocab > 0, "model opened with valid dimensions");
    CHECK(katali_gguf_model_resident_bytes(&m) > 0, "resident memory measured");

    char out[4096];
    int n = 0;

    banner("answer: Hi");
    n = katali_gguf_model_generate(&m, "Hi", out, sizeof(out));
    printf("  answer: %s\n", out);
    CHECK(n > 0 && out[0] != '\0', "generated a non-empty answer");

    banner("answer: factual");
    katali_gguf_model_reset(&m);
    n = katali_gguf_model_generate(&m, "What is the capital of France?", out, sizeof(out));
    printf("  answer: %s\n", out);
    CHECK(n > 0, "factual question answered");
    CHECK(strstr(out, "Paris") != NULL, "factual answer contains Paris");

    banner("answer: math");
    katali_gguf_model_reset(&m);
    n = katali_gguf_model_generate(&m, "What is 15% of 240?", out, sizeof(out));
    printf("  answer: %s\n", out);
    CHECK(n > 0, "math question answered");
    CHECK(strstr(out, "36") != NULL || strstr(out, "0.15") != NULL ||
          strstr(out, "15") != NULL, "math answer is on-topic");

    banner("answer: code");
    katali_gguf_model_reset(&m);
    n = katali_gguf_model_generate(&m, "Write a short Python function that adds two numbers.",
                                   out, sizeof(out));
    printf("  answer: %s\n", out);
    CHECK(n > 0, "code question answered");
    CHECK(strstr(out, "def") != NULL, "code answer contains a Python definition");

    banner("two consecutive questions + state reset");
    char a1[2048], a2[2048], b1[2048];
    katali_gguf_model_reset(&m);
    int na1 = katali_gguf_model_generate(&m, "What is 2+2?", a1, sizeof(a1));
    katali_gguf_model_reset(&m);
    int na2 = katali_gguf_model_generate(&m, "What is 2+2?", a2, sizeof(a2));
    katali_gguf_model_reset(&m);
    int nb1 = katali_gguf_model_generate(&m, "What color is the sky on a clear day?",
                                         b1, sizeof(b1));
    printf("  repeat A: %s\n", a1);
    printf("  repeat B: %s\n", b1);
    CHECK(na1 > 0 && na2 > 0 && nb1 > 0, "all consecutive questions answered");
    CHECK(strcmp(a1, a2) == 0, "greedy repeat is identical (KV state reset)");
    CHECK(strstr(a1, "4") != NULL, "2+2 answer contains 4");
    CHECK(strcmp(a1, b1) != 0, "independent questions produce independent answers");

    banner("cancellation");
    katali_gguf_model_cancel(&m);
    char outc[256];
    int nc = katali_gguf_model_generate(&m, "Say something long.", outc, sizeof(outc));
    CHECK(nc == 0, "cancelled generation produces no tokens");
    katali_gguf_model_reset(&m);
    int nr = katali_gguf_model_generate(&m, "Hi", out, sizeof(out));
    CHECK(nr > 0, "generation recovers after cancel + reset");

    katali_gguf_model_close(&m);

    banner("backend interface (persistent handle)");
    {
        KataliBackend *b = katali_backend_create("gguf", &opts);
        CHECK(b != NULL, "backend_create('gguf')");
        if (b) {
            char bo[512];
            int bn = b->ops->generate(b, "What is 2+2?", bo, sizeof(bo));
            CHECK(bn > 0 && strstr(bo, "4") != NULL, "backend generate answers 2+2");
            b->ops->reset_state(b);
            int bn2 = b->ops->generate(b, "What is 2+2?", bo, sizeof(bo));
            CHECK(bn2 == bn, "backend reuse after reset_state");
            CHECK(b->ops->resident_bytes(b) > 0, "backend resident_bytes");
            katali_backend_destroy(b);
        }
    }

    banner("malformed / truncated model file");
    const char *bad1 = ".tmp/live_garbage.gguf";
    const char *bad2 = ".tmp/live_truncated.gguf";
    write_garbage(bad1, 0);
    write_garbage(bad2, 1);
    KataliGgufFile bf;
    CHECK(katali_gguf_open(&bf, bad1, 1, err, sizeof(err)) != 0,
          "implausible tensor count rejected");
    CHECK(katali_gguf_open(&bf, bad2, 1, err, sizeof(err)) != 0,
          "truncated file rejected");
    remove(bad1);
    remove(bad2);

    printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}

