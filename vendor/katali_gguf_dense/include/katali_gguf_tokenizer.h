/* Katali-GGUF tokenizer — Apache-2.0
 *
 * Reads the vocabulary, merge rules and special-token metadata that GGUF
 * carries. No token ids are hard-coded: special tokens are resolved by looking
 * their strings up in the file's own vocabulary.
 */
#ifndef KATALI_GGUF_TOKENIZER_H
#define KATALI_GGUF_TOKENIZER_H

#include <stddef.h>
#include <stdint.h>
#include "katali_gguf.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct KataliGgufTokenizer {
    int  vocab_size;
    const uint8_t **tok_bytes; /* borrowed from the mapping */
    size_t *tok_len;
    int   *tok_type;           /* tokenizer.ggml.token_type, or NULL */
    int   *hash_head;          /* vocab hash: id+1, 0 = empty */
    int   *hash_next;
    int    hash_cap;

    int   *merge_head;         /* pair hash: rank+1 */
    uint64_t *merge_key;
    int   *merge_next;
    int   *merge_result;       /* rank -> id of the merged token */
    int    merge_cap;
    int    n_merges;

    int    byte_id[256];       /* vocab id of each raw byte's symbol, -1 none */

    int    bos_id, eos_id, pad_id, unk_id;
    int    im_start_id, im_end_id, endoftext_id;
    int    think_open_id, think_close_id;
    int    add_bos;
    int    pre_minicpm5;

    const char *chat_template;  /* borrowed */
    size_t chat_template_len;
    int    has_chat_template;
} KataliGgufTokenizer;

/* Returns 0 on success; -1 with a message in err when required metadata is
 * missing (there is no silent fallback). */
int  katali_gguf_tokenizer_init(KataliGgufTokenizer *t, const KataliGgufFile *f,
                                char *err, size_t err_cap);
void katali_gguf_tokenizer_free(KataliGgufTokenizer *t);

/* Look up an exact token string ("<|im_start|>"); returns id or -1. */
int  katali_gguf_tokenizer_find(const KataliGgufTokenizer *t,
                                const char *s, size_t len);

/* Encode UTF-8 text to ids. Returns the number written, or <0 on error. */
int  katali_gguf_tokenizer_encode(const KataliGgufTokenizer *t, const char *text,
                                  int *ids, int max_ids);

/* Append one decoded token's raw bytes. Returns bytes written (0 for control). */
int  katali_gguf_tokenizer_decode(const KataliGgufTokenizer *t, int id,
                                  char *out, size_t cap);

#ifdef __cplusplus
}
#endif
#endif /* KATALI_GGUF_TOKENIZER_H */
