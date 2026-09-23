#ifndef KATALI_DENSE_CUDA_H
#define KATALI_DENSE_CUDA_H
#include "katali_gguf.h"
int katali_dense_cuda_enabled(void);
int katali_dense_cuda_matvec(const KataliGgufTensor *t, const float *x, float *y);
int katali_dense_cuda_matvec_multi(const KataliGgufTensor **ts, float **ys, int n, const float *x);
void katali_dense_cuda_shutdown(void);
#endif


