#ifndef KATALI_Q4_H
#define KATALI_Q4_H
#include <stddef.h>
#include <stdint.h>
#include "katali.h"

#ifdef __cplusplus
extern "C" {
#endif

/* GGUF Q4_0 block: 32 weights + f16 scale (18 bytes). */
#define KQ4_0_BLOCK_SIZE 32
#define KQ4_0_BYTES      18

KATALI_API size_t q4_0_nbytes(size_t nelements);
KATALI_API float  q4_f16_to_f32(uint16_t h);
KATALI_API void   q4_0_dequant_row(const uint8_t *block, float *out32);
/* y[n] = W[n,k] @ x[k] for row-major Q4_0 matrix (GGUF layout). */
KATALI_API void   q4_0_gemv(const uint8_t *w, size_t n, size_t k,
                            const float *x, float *y);
KATALI_API int    q4_selftest(void);

#ifdef __cplusplus
}
#endif
#endif
