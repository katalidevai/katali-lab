#include "ops.h"
#include <math.h>

void katali_rms_norm(const float *x, const float *w, float *y, size_t n, float eps) {
    double ss = 0.0;
    for (size_t i = 0; i < n; i++) ss += (double)x[i] * (double)x[i];
    float scale = 1.f / sqrtf((float)(ss / (double)n + (double)eps));
    for (size_t i = 0; i < n; i++) y[i] = x[i] * scale * w[i];
}

void katali_silu_mul(const float *gate, const float *up, float *out, size_t n) {
    for (size_t i = 0; i < n; i++) {
        float g = gate[i];
        float s = g / (1.f + expf(-g));
        out[i] = s * up[i];
    }
}
