#ifndef KATALI_OPS_H
#define KATALI_OPS_H
#include <stddef.h>
#include "katali.h"

#ifdef __cplusplus
extern "C" {
#endif

KATALI_API void katali_rms_norm(const float *x, const float *w, float *y, size_t n, float eps);
KATALI_API void katali_silu_mul(const float *gate, const float *up, float *out, size_t n);

#ifdef __cplusplus
}
#endif
#endif
