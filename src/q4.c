#include "q4.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

float q4_f16_to_f32(uint16_t h) {
    uint32_t s = (h >> 15) & 1;
    uint32_t e = (h >> 10) & 0x1f;
    uint32_t f = h & 0x3ff;
    uint32_t out;
    if (e == 0) {
        if (f == 0) out = s << 31;
        else {
            e = 127 - 15 + 1;
            while ((f & 0x400) == 0) { f <<= 1; e--; }
            f &= 0x3ff;
            out = (s << 31) | (e << 23) | (f << 13);
        }
    } else if (e == 31) {
        out = (s << 31) | 0x7f800000u | (f << 13);
    } else {
        out = (s << 31) | ((e + (127 - 15)) << 23) | (f << 13);
    }
    float r; memcpy(&r, &out, 4); return r;
}

size_t q4_0_nbytes(size_t nelements) {
    return (nelements / KQ4_0_BLOCK_SIZE) * KQ4_0_BYTES;
}

void q4_0_dequant_row(const uint8_t *block, float *out32) {
    uint16_t dbits; memcpy(&dbits, block, 2);
    float d = q4_f16_to_f32(dbits);
    const uint8_t *qs = block + 2;
    for (int j = 0; j < 16; j++) {
        uint8_t byte = qs[j];
        out32[j] = ((int)(byte & 0x0f) - 8) * d;
        out32[j + 16] = ((int)(byte >> 4) - 8) * d;
    }
}

void q4_0_gemv(const uint8_t *w, size_t n, size_t k, const float *x, float *y) {
    const size_t blocks_per_row = k / KQ4_0_BLOCK_SIZE;
    float tmp[32];
    for (size_t row = 0; row < n; row++) {
        const uint8_t *rowp = w + row * blocks_per_row * KQ4_0_BYTES;
        float acc = 0.f;
        for (size_t b = 0; b < blocks_per_row; b++) {
            q4_0_dequant_row(rowp + b * KQ4_0_BYTES, tmp);
            const float *xb = x + b * KQ4_0_BLOCK_SIZE;
            for (int j = 0; j < 32; j++) acc += tmp[j] * xb[j];
        }
        y[row] = acc;
    }
}

int q4_selftest(void) {
    /* Known Q4_0 block: scale=1.0f16, nibbles encode -8..7 pattern. */
    uint8_t block[18];
    memset(block, 0, sizeof(block));
    /* f16 1.0 = 0x3c00 */
    block[0] = 0x00; block[1] = 0x3c;
    for (int j = 0; j < 16; j++) block[2 + j] = (uint8_t)((j << 4) | j);
    float y[32];
    q4_0_dequant_row(block, y);
    if (fabsf(y[0] - (-8.f)) > 1e-3f) {
        fprintf(stderr, "q4_selftest dequant y0=%g\n", y[0]);
        return KATALI_ERR;
    }
    if (fabsf(y[16] - (-8.f)) > 1e-3f) {
        fprintf(stderr, "q4_selftest dequant y16=%g\n", y[16]);
        return KATALI_ERR;
    }
    float x[32]; for (int i = 0; i < 32; i++) x[i] = 1.f;
    float out[1];
    q4_0_gemv(block, 1, 32, x, out);
    if (!isfinite(out[0])) {
        fprintf(stderr, "q4_selftest gemv non-finite\n");
        return KATALI_ERR;
    }
    return KATALI_OK;
}
