/* Katali-GGUF tokenizer — Apache-2.0 */
#include "katali_gguf_tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ==================================================================== *
 * GPT-2 byte <-> unicode alphabet (public GPT-2/RoBERTa convention)     *
 * ==================================================================== */
static int byte_is_direct(int b) {
    return (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
}

static int gpt2_b2u(int b) {
    if (byte_is_direct(b)) return b;
    int n = 0;
    for (int x = 0; x < 256; x++) {
        if (!byte_is_direct(x)) {
            if (x == b) return 256 + n;
            n++;
        }
    }
    return b;
}

static int gpt2_u2b(int c) {
    if (byte_is_direct(c)) return c;
    if (c >= 256) {
        int n = c - 256, cnt = 0;
        for (int x = 0; x < 256; x++) {
            if (!byte_is_direct(x)) {
                if (cnt == n) return x;
                cnt++;
            }
        }
    }
    return -1;
}

static int utf8_encode(int cp, char *out) {
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

static int utf8_decode(const char *s, int *cp) {
    const unsigned char *u = (const unsigned char *)s;
    if (u[0] < 0x80) { *cp = u[0]; return 1; }
    if ((u[0] & 0xE0) == 0xC0 && (u[1] & 0xC0) == 0x80) {
        *cp = ((u[0] & 0x1F) << 6) | (u[1] & 0x3F);
        return 2;
    }
    if ((u[0] & 0xF0) == 0xE0 && (u[1] & 0xC0) == 0x80 && (u[2] & 0xC0) == 0x80) {
        *cp = ((u[0] & 0x0F) << 12) | ((u[1] & 0x3F) << 6) | (u[2] & 0x3F);
        return 3;
    }
    if ((u[0] & 0xF8) == 0xF0) { /* 4-byte */
        *cp = ((u[0] & 0x07) << 18) | ((u[1] & 0x3F) << 12) |
              ((u[2] & 0x3F) << 6) | (u[3] & 0x3F);
        return 4;
    }
    *cp = u[0];
    return 1;
}

/* ==================================================================== *
 * Hashes                                                                *
 * ==================================================================== */
static uint64_t fnv1a(const uint8_t *s, size_t n) {
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; i++) { h ^= s[i]; h *= 1099511628211ULL; }
    return h;
}

static int next_pow2(int n) {
    int p = 16;
    while (p < n) p <<= 1;
    return p;
}

/* vocab hash */
static int vocab_lookup(const KataliGgufTokenizer *t, const char *s, size_t n,
                        int *slot_out) {
    uint64_t h = fnv1a((const uint8_t *)s, n);
    int cap = t->hash_cap;
    int i = (int)(h & (uint64_t)(cap - 1));
    for (int guard = 0; guard < cap; guard++) {
        int head = t->hash_head[i];
        if (head == 0) { if (slot_out) *slot_out = i; return -1; }
        int id = head - 1;
        if (t->tok_len[id] == n && (n == 0 || memcmp(t->tok_bytes[id], s, n) == 0))
            return id;
        i = (i + 1) & (cap - 1);
    }
    if (slot_out) *slot_out = -1;
    return -1;
}

static void vocab_insert(KataliGgufTokenizer *t, int id) {
    uint64_t h = fnv1a(t->tok_bytes[id], t->tok_len[id]);
    int cap = t->hash_cap;
    int i = (int)(h & (uint64_t)(cap - 1));
    for (int guard = 0; guard < cap; guard++) {
        if (t->hash_head[i] == 0) {
            t->hash_head[i] = id + 1;
            t->hash_next[id] = -1;
            return;
        }
        i = (i + 1) & (cap - 1);
    }
}

/* merge hash: key = (left_id << 32) | right_id */
static uint64_t merge_hash_key(uint64_t k) {
    k ^= k >> 33; k *= 0xff51afd7ed558ccdULL;
    k ^= k >> 33; k *= 0xc4ceb9fe1a85ec53ULL;
    k ^= k >> 33;
    return k;
}

static int merge_find(const KataliGgufTokenizer *t, int left, int right) {
    if (!t->merge_head || t->merge_cap <= 0) return -1;
    uint64_t key = ((uint64_t)(uint32_t)left << 32) | (uint32_t)right;
    int i = (int)(merge_hash_key(key) & (uint64_t)(t->merge_cap - 1));
    for (int guard = 0; guard < t->merge_cap; guard++) {
        int head = t->merge_head[i];
        if (head == 0) return -1;
        if (t->merge_key[i] == key) return head - 1;
        i = (i + 1) & (t->merge_cap - 1);
    }
    return -1;
}

static void merge_insert(KataliGgufTokenizer *t, int left, int right, int rank) {
    uint64_t key = ((uint64_t)(uint32_t)left << 32) | (uint32_t)right;
    int i = (int)(merge_hash_key(key) & (uint64_t)(t->merge_cap - 1));
    for (int guard = 0; guard < t->merge_cap; guard++) {
        if (t->merge_head[i] == 0) {
            t->merge_head[i] = rank + 1;
            t->merge_key[i] = key;
            return;
        }
        if (t->merge_key[i] == key) return; /* first rank wins */
        i = (i + 1) & (t->merge_cap - 1);
    }
}

/* ==================================================================== *
 * Init / free                                                           *
 * ==================================================================== */
int katali_gguf_tokenizer_init(KataliGgufTokenizer *t, const KataliGgufFile *f,
                               char *err, size_t err_cap) {
    if (!t || !f) return -1;
    memset(t, 0, sizeof(*t));
    for (int i = 0; i < 256; i++) t->byte_id[i] = -1;

    int64_t vocab = katali_gguf_array_len(f, "tokenizer.ggml.tokens", NULL);
    if (vocab <= 0) {
        if (err && err_cap)
            snprintf(err, err_cap, "tokenizer.ggml.tokens is missing from the GGUF metadata");
        return -1;
    }
    t->vocab_size = (int)vocab;

    int64_t scores_n = katali_gguf_array_len(f, "tokenizer.ggml.scores", NULL);
    t->tok_bytes = (const uint8_t **)calloc((size_t)vocab, sizeof(uint8_t *));
    t->tok_len   = (size_t *)calloc((size_t)vocab, sizeof(size_t));
    t->tok_type  = (int *)calloc((size_t)vocab, sizeof(int));
    t->hash_next = (int *)calloc((size_t)vocab, sizeof(int));
    t->hash_cap  = next_pow2((int)vocab * 2 + 8);
    t->hash_head = (int *)calloc((size_t)t->hash_cap, sizeof(int));
    if (!t->tok_bytes || !t->tok_len || !t->tok_type || !t->hash_next || !t->hash_head) {
        if (err && err_cap) snprintf(err, err_cap, "out of memory building tokenizer");
        katali_gguf_tokenizer_free(t);
        return -1;
    }
    for (int i = 0; i < t->vocab_size; i++) {
        size_t len = 0;
        const char *s = katali_gguf_array_str(f, "tokenizer.ggml.tokens", (uint64_t)i, &len);
        t->tok_bytes[i] = (const uint8_t *)s;
        t->tok_len[i] = len;
        t->tok_type[i] = (int)katali_gguf_array_int(f, "tokenizer.ggml.token_type", (uint64_t)i);
        if (!s) { /* tolerate holes; they simply never match */
            t->tok_bytes[i] = (const uint8_t *)"";
            t->tok_len[i] = 0;
        }
        vocab_insert(t, i);
    }
    (void)scores_n; /* scores are read on demand; not needed for encode/decode */

    /* Byte -> symbol id table. */
    for (int b = 0; b < 256; b++) {
        char sym[8];
        int n = utf8_encode(gpt2_b2u(b), sym);
        t->byte_id[b] = vocab_lookup(t, sym, (size_t)n, NULL);
    }

    /* Merge rules -> (left_id,right_id) rank table. */
    int64_t merges = katali_gguf_array_len(f, "tokenizer.ggml.merges", NULL);
    if (merges > 0) {
        t->n_merges = (int)merges;
        t->merge_cap = next_pow2((int)merges * 2 + 8);
        t->merge_head = (int *)calloc((size_t)t->merge_cap, sizeof(int));
        t->merge_key  = (uint64_t *)calloc((size_t)t->merge_cap, sizeof(uint64_t));
        t->merge_result = (int *)malloc((size_t)merges * sizeof(int));
        if (!t->merge_head || !t->merge_key || !t->merge_result) {
            if (err && err_cap) snprintf(err, err_cap, "out of memory building merges");
            katali_gguf_tokenizer_free(t);
            return -1;
        }
        for (int i = 0; i < t->n_merges; i++) t->merge_result[i] = -1;
        for (int i = 0; i < t->n_merges; i++) {
            size_t len = 0;
            const char *m = katali_gguf_array_str(f, "tokenizer.ggml.merges", (uint64_t)i, &len);
            if (!m) continue;
            const char *sp = memchr(m, ' ', len);
            if (!sp) continue;
            size_t ll = (size_t)(sp - m);
            size_t rl = len - ll - 1;
            int left = vocab_lookup(t, m, ll, NULL);
            int right = vocab_lookup(t, sp + 1, rl, NULL);
            if (left < 0 || right < 0) continue;
            merge_insert(t, left, right, i);
            /* The merged token is the concatenation of the two strings; resolve
             * its vocabulary id once so the merge loop never builds strings. */
            {
                size_t tl = t->tok_len[left] + t->tok_len[right];
                char *cat = (char *)malloc(tl ? tl : 1);
                if (cat) {
                    memcpy(cat, t->tok_bytes[left], t->tok_len[left]);
                    memcpy(cat + t->tok_len[left], t->tok_bytes[right], t->tok_len[right]);
                    t->merge_result[i] = vocab_lookup(t, cat, tl, NULL);
                    free(cat);
                }
            }
        }
    }

    /* Special tokens: ids come from metadata where present, else from the
     * vocabulary strings. Never hard-coded. */
    t->bos_id = (int)katali_gguf_get_int(f, "tokenizer.ggml.bos_token_id", -1);
    t->eos_id = (int)katali_gguf_get_int(f, "tokenizer.ggml.eos_token_id", -1);
    t->pad_id = (int)katali_gguf_get_int(f, "tokenizer.ggml.padding_token_id", -1);
    t->unk_id = (int)katali_gguf_get_int(f, "tokenizer.ggml.unknown_token_id", -1);
    t->add_bos = katali_gguf_get_bool(f, "tokenizer.ggml.add_bos_token", 0);
    {
        const char *pre = katali_gguf_get_str(f, "tokenizer.ggml.pre", "", NULL);
        t->pre_minicpm5 = pre && strcmp(pre, "minicpm5") == 0;
    }

    t->im_start_id   = katali_gguf_tokenizer_find(t, "<|im_start|>", strlen("<|im_start|>"));
    t->im_end_id     = katali_gguf_tokenizer_find(t, "<|im_end|>", strlen("<|im_end|>"));
    t->endoftext_id  = katali_gguf_tokenizer_find(t, "<|endoftext|>", strlen("<|endoftext|>"));
    /* Thinking markers differ between tokenizer revisions: Qwen3 ships a
     * space-prefixed form while some quantizers emit an angle-bracket form.
     * Both are resolved from this file's vocabulary; nothing is hard-coded. */
    t->think_open_id  = katali_gguf_tokenizer_find(t, "<think>", strlen("<think>"));
    t->think_close_id = katali_gguf_tokenizer_find(t, "</think>", strlen("</think>"));
    if (t->think_open_id < 0)
        t->think_open_id = katali_gguf_tokenizer_find(t, " thinking", strlen(" thinking"));
    if (t->think_close_id < 0)
        t->think_close_id = katali_gguf_tokenizer_find(t, " thinking", strlen(" thinking"));
    /* Re-resolve the thinking markers from explicit byte sequences so the exact
     * spelling cannot be lost to source-encoding normalisation. */
    {
        static const unsigned char open_c[][13] = {
            { 0x20,'t','h','i','n','k',0 },                          /* " think"    */
            { '<','t','h','i','n','k','>',0 },                        /* "<think>"   */
            { 0xEF,0xBD,0x9C,'t','h','i','n','k',0xEF,0xBD,0x9C,0 }  /* bar form    */
        };
        static const unsigned char close_c[][14] = {
            { 0x20,'/','t','h','i','n','k',0 },                       /* " /think"   */
            { '<','/','t','h','i','n','k','>',0 },                     /* "</think>"  */
            { 0x20,'/','t','h','i','n','k','i','n','g',0 },            /* " /thinking"*/
            { 0xEF,0xBD,0x9C,'/','t','h','i','n','k',0xEF,0xBD,0x9C,0 }/* bar form    */
        };
        int found = -1;
        for (size_t ci = 0; ci < sizeof(open_c) / sizeof(open_c[0]) && found < 0; ci++)
            found = katali_gguf_tokenizer_find(t, (const char *)open_c[ci],
                                               strlen((const char *)open_c[ci]));
        if (found >= 0) t->think_open_id = found;
        found = -1;
        for (size_t ci = 0; ci < sizeof(close_c) / sizeof(close_c[0]) && found < 0; ci++)
            found = katali_gguf_tokenizer_find(t, (const char *)close_c[ci],
                                               strlen((const char *)close_c[ci]));
        if (found >= 0) t->think_close_id = found;
    }

    if (t->eos_id < 0) t->eos_id = t->im_end_id;
    if (t->eos_id < 0) t->eos_id = t->endoftext_id;

    {
        size_t clen = 0;
        const char *ct = katali_gguf_get_str(f, "tokenizer.chat_template", NULL, &clen);
        if (ct && clen) {
            t->chat_template = ct;
            t->chat_template_len = clen;
            t->has_chat_template = 1;
        }
    }
    return 0;
}

void katali_gguf_tokenizer_free(KataliGgufTokenizer *t) {
    if (!t) return;
    free(t->tok_bytes);
    free(t->tok_len);
    free(t->tok_type);
    free(t->hash_head);
    free(t->hash_next);
    free(t->merge_head);
    free(t->merge_key);
    free(t->merge_next);
    free(t->merge_result);
    memset(t, 0, sizeof(*t));
}

int katali_gguf_tokenizer_find(const KataliGgufTokenizer *t,
                               const char *s, size_t len) {
    if (!t || !s) return -1;
    return vocab_lookup(t, s, len, NULL);
}

/* ==================================================================== *
 * Pretokenization (GPT-2 style split, ASCII-exact)                       *
 * ==================================================================== */
static int cp_at(const char *s, size_t i, size_t n, int *cp, int *w) {
    if (i >= n) return 0;
    unsigned char c = (unsigned char)s[i];
    int width = 1;
    if (c >= 0x80) width = utf8_decode(s + i, cp); else *cp = c;
    if (i + (size_t)width > n) width = 1;
    *w = width;
    return 1;
}

static int cp_is_letter(int cp) {
    if ((cp >= 'a' && cp <= 'z') || (cp >= 'A' && cp <= 'Z')) return 1;
    return cp >= 0x80; /* non-ASCII treated as letters (documented approximation) */
}
static int cp_is_digit(int cp) { return cp >= '0' && cp <= '9'; }
static int cp_is_space(int cp) {
    return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' || cp == '\f' || cp == '\v';
}
static int cp_is_punct(int cp) {
    return !cp_is_space(cp) && !cp_is_letter(cp) && !cp_is_digit(cp) &&
           cp != '\r' && cp != '\n';
}

static int match_contraction(const char *s, size_t i, size_t n) {
    if (s[i] != '\'') return 0;
    if (i + 1 >= n) return 0;
    char c1 = s[i + 1];
    if (c1 == 's' || c1 == 'S' || c1 == 't' || c1 == 'T' || c1 == 'm' || c1 == 'M' ||
        c1 == 'd' || c1 == 'D') return 2;
    if (i + 2 < n) {
        char c2 = s[i + 2];
        if ((c1 == 'r' || c1 == 'R') && (c2 == 'e' || c2 == 'E')) return 3;
        if ((c1 == 'v' || c1 == 'V') && (c2 == 'e' || c2 == 'E')) return 3;
        if ((c1 == 'l' || c1 == 'L') && (c2 == 'l' || c2 == 'L')) return 3;
    }
    return 0;
}

/* Return the byte length of the next pretokenization span. */
static size_t pretok_span(const KataliGgufTokenizer *t, const char *s, size_t i, size_t n) {
    int cp, w;
    if (!cp_at(s, i, n, &cp, &w)) return 1;

    int c = match_contraction(s, i, n);
    if (c) return (size_t)c;

    if (cp_is_letter(cp)) {
        size_t j = i;
        while (j < n) { int c2, w2; cp_at(s, j, n, &c2, &w2); if (!cp_is_letter(c2)) break; j += (size_t)w2; }
        return j - i;
    }
    /* letters with an optional single leading non-space/non-letter/non-digit */
    if (cp != '\r' && cp != '\n' && !cp_is_space(cp) && !cp_is_digit(cp) && i + 1 < n) {
        int c2, w2;
        cp_at(s, i + 1, n, &c2, &w2);
        if (cp_is_letter(c2)) {
            size_t j = i + 1;
            while (j < n) { int c3, w3; cp_at(s, j, n, &c3, &w3); if (!cp_is_letter(c3)) break; j += (size_t)w3; }
            return j - i;
        }
    }
    if (cp_is_digit(cp)) return 1;

    /* optional space followed by a punctuation run, then trailing newlines */
    if (cp == ' ' && i + 1 < n) {
        int c2, w2;
        cp_at(s, i + 1, n, &c2, &w2);
        if (cp_is_punct(c2)) {
            size_t j = i + 1;
            while (j < n) { int c3, w3; cp_at(s, j, n, &c3, &w3); if (!cp_is_punct(c3)) break; j += (size_t)w3; }
            while (j < n && (s[j] == '\r' || s[j] == '\n')) j++;
            return j - i;
        }
    }
    if (cp_is_punct(cp)) {
        size_t j = i;
        while (j < n) { int c2, w2; cp_at(s, j, n, &c2, &w2); if (!cp_is_punct(c2)) break; j += (size_t)w2; }
        while (j < n && (s[j] == '\r' || s[j] == '\n')) j++;
        return j - i;
    }
    if (cp_is_space(cp)) {
        /* MiniCPM5 uses the GPT-2-style pre-tokenizer where a leading space
         * belongs to the following word (" is" -> "Ġis"). The generic path
         * intentionally preserves its existing behavior for Qwen models. */
        if (t && t->pre_minicpm5 && cp == ' ' && i + 1 < n) {
            int c2, w2;
            cp_at(s, i + 1, n, &c2, &w2);
            if (cp_is_letter(c2)) {
                size_t j = i + 1 + (size_t)w2;
                while (j < n) {
                    int c3, w3;
                    cp_at(s, j, n, &c3, &w3);
                    if (!cp_is_letter(c3)) break;
                    j += (size_t)w3;
                }
                return j - i;
            }
        }
        size_t j = i;
        while (j < n) { int c2, w2; cp_at(s, j, n, &c2, &w2); if (!cp_is_space(c2)) break; j += (size_t)w2; }
        /* leave a single trailing space to attach to the next word */
        if (j < n && j - i > 1) j -= 1;
        return j - i;
    }
    return (size_t)w;
}

/* ==================================================================== *
 * BPE                                                                   *
 * ==================================================================== */
#include <limits.h>

static int bpe_merge_span(const KataliGgufTokenizer *t, int *ids, int m) {
    if (m < 2 || !t->merge_head) return m;
    for (;;) {
        int best_rank = INT_MAX, best_j = -1;
        for (int j = 0; j + 1 < m; j++) {
            int r = merge_find(t, ids[j], ids[j + 1]);
            if (r >= 0 && r < best_rank) { best_rank = r; best_j = j; }
        }
        if (best_j < 0) break;
        int merged = t->merge_result[best_rank];
        if (merged < 0) merged = ids[best_j];
        ids[best_j] = merged;
        memmove(&ids[best_j + 1], &ids[best_j + 2], (size_t)(m - best_j - 2) * sizeof(int));
        m--;
    }
    return m;
}

int katali_gguf_tokenizer_encode(const KataliGgufTokenizer *t, const char *text,
                                 int *ids, int max_ids) {
    if (!t || !text || !ids) return -1;
    size_t n = strlen(text);
    int out_n = 0;
    size_t i = 0;
    int span_ids[1024];
    while (i < n) {
        size_t span = pretok_span(t, text, i, n);
        if (span == 0) span = 1;
        if (span > (size_t)n - i) span = n - i;
        /* bytes -> symbol ids */
        int m = 0;
        int overflow = 0;
        for (size_t b = i; b < i + span; b++) {
            int id = t->byte_id[(unsigned char)text[b]];
            if (id < 0) return -1; /* vocabulary cannot represent this byte */
            if (m >= (int)(sizeof(span_ids) / sizeof(span_ids[0]))) { overflow = 1; break; }
            span_ids[m++] = id;
        }
        if (overflow) return -1;
        m = bpe_merge_span(t, span_ids, m);
        for (int k = 0; k < m; k++) {
            if (out_n >= max_ids) return out_n;
            ids[out_n++] = span_ids[k];
        }
        i += span;
    }
    return out_n;
}

int katali_gguf_tokenizer_decode(const KataliGgufTokenizer *t, int id,
                                 char *out, size_t cap) {
    if (!t || !out || id < 0 || id >= t->vocab_size) return 0;
    if (t->tok_type && t->tok_type[id] == 3 /* CONTROL */) return 0;
    const char *s = (const char *)t->tok_bytes[id];
    size_t n = t->tok_len[id];
    size_t written = 0;
    size_t i = 0;
    while (i < n) {
        int cp, w;
        cp_at(s, i, n, &cp, &w);
        int b = gpt2_u2b(cp);
        if (b >= 0 && written < cap) out[written++] = (char)b;
        i += (size_t)w;
    }
    return (int)written;
}



