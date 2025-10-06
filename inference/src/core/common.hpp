#pragma once

#include <cstdint>

namespace squash {

using bf16 = int16_t;
using uint = uint32_t;
using ulong = uint64_t;

float bf16ToFloat(bf16 value);
bf16 floatToBf16(float value);

}  // namespace squash

/////////////////////////////////////////////////////////////////////////////
// Impl

namespace squash {

inline float bf16ToFloat(bf16 value) {
    union {
        float f;
        int16_t i[2];
    } u;
    u.i[0] = 0;
    u.i[1] = value;
    return u.f;
}

inline bf16 floatToBf16(float value) {
    // Round to nearest, ties to even
    uint32_t u = reinterpret_cast<uint32_t&>(value);
    u += 0x7FFFu + ((u >> 16) & 1u);
    return static_cast<bf16>(u >> 16);
}

}  // namespace squash
