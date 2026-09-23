/* Katali-GGUF half/bfloat helpers — Apache-2.0
 * Original code; no third-party headers. */
#ifndef KATALI_GGUF_FP_H
#define KATALI_GGUF_FP_H

#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/* IEEE-754 binary16 -> binary32 (exact for finite values). */
static inline float katali_f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign; /* +/- zero */
        } else {
            /* subnormal: normalise into binary32 */
            int e = -1;
            do { mant <<= 1; e++; } while ((mant & 0x400u) == 0);
            mant &= 0x3FFu;
            bits = sign | ((uint32_t)(127 - 15 - e) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (mant << 13); /* inf / nan */
    } else {
        bits = sign | ((exp - 15u + 127u) << 23) | (mant << 13);
    }
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

static inline uint16_t katali_f32_to_f16(float v) {
    uint32_t bits;
    memcpy(&bits, &v, sizeof(bits));
    uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t  exp  = (int32_t)((bits >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = bits & 0x7FFFFFu;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign; /* underflow to zero */
        mant |= 0x800000u;
        uint32_t shift = (uint32_t)(14 - exp);
        uint16_t sub = (uint16_t)(mant >> shift);
        if ((mant >> (shift - 1)) & 1u) sub++; /* round to nearest */
        return (uint16_t)(sign | sub);
    }
    if (exp >= 31) return (uint16_t)(sign | 0x7C00u); /* inf/nan */
    uint16_t out = (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
    if (mant & 0x1000u) out++; /* round to nearest */
    return out;
}

/* bfloat16 is the upper 16 bits of a binary32 (truncation, as the format is
 * defined). */
static inline float katali_bf16_to_f32(uint16_t b) {
    uint32_t bits = (uint32_t)b << 16;
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

static inline uint16_t katali_f32_to_bf16(float v) {
    uint32_t bits;
    memcpy(&bits, &v, sizeof(bits));
    /* round-to-nearest-even */
    uint32_t lsb = (bits >> 16) & 1u;
    bits += 0x7FFFu + lsb;
    return (uint16_t)(bits >> 16);
}

#ifdef __cplusplus
}
#endif
#endif /* KATALI_GGUF_FP_H */
