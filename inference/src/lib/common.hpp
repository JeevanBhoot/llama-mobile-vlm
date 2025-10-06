#pragma once

#include <cstdint>

namespace squash {

using bf16 = int16_t;
using uint = uint32_t;
using ulong = uint64_t;

float bf16ToFloat(bf16 value);

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
}  // namespace squash
