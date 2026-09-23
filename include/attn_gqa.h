#ifndef KATALI_ATTN_GQA_H
#define KATALI_ATTN_GQA_H
#include "host.h"
#ifdef __cplusplus
extern "C" {
#endif
KATALI_API int attn_gqa_forward(HostModel *m, int layer, const float *xb, float *y);
#ifdef __cplusplus
}
#endif
#endif
