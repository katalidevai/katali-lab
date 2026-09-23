/* Katali-GGUF sampler — Apache-2.0 */
#ifndef KATALI_GGUF_SAMPLER_H
#define KATALI_GGUF_SAMPLER_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct KataliGgufSampler {
    float    temperature;         /* <= 0 => greedy */
    int      top_k;               /* 0 = off */
    float    top_p;               /* >= 1.0 = off */
    float    repetition_penalty;  /* <= 1.0 = off */
    unsigned seed;
    float   *work;                /* scratch, lazily allocated */
    int     *order;               /* scratch, lazily allocated */
    int      work_cap;
} KataliGgufSampler;

void katali_gguf_sampler_init(KataliGgufSampler *s);
void katali_gguf_sampler_free(KataliGgufSampler *s);

/* Pick one token from logits[n]. `recent`/`n_recent` may be NULL/0. Does not
 * modify `logits`. */
int katali_gguf_sample(KataliGgufSampler *s, const float *logits, int n,
                       const int *recent, int n_recent);

#ifdef __cplusplus
}
#endif
#endif /* KATALI_GGUF_SAMPLER_H */
