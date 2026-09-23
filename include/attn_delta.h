#ifndef KATALI_ATTN_DELTA_H
#define KATALI_ATTN_DELTA_H
#include "host.h"
#ifdef __cplusplus
extern "C" {
#endif
KATALI_API int attn_delta_forward(HostModel *m, int layer, const float *xb, float *y);
#ifdef __cplusplus
}
#endif
#endif
