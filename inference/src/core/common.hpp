#pragma once

#include <cstdint>

namespace squash {

struct bf16 {
    int16_t _intValue;
};
static_assert(sizeof(bf16) == 2, "bf16 must be 2 bytes");
using uint = uint32_t;
using ulong = uint64_t;

float bf16ToFloat(bf16 value);
bf16 floatToBf16(float value);

template <class To, class From>
To cast(From value);

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
    u.i[1] = value._intValue;
    return u.f;
}

inline bf16 floatToBf16(float value) {
    // Round to nearest, ties to even
    uint32_t u = reinterpret_cast<uint32_t&>(value);
    u += 0x7FFFu + ((u >> 16) & 1u);
    return bf16{static_cast<int16_t>(u >> 16)};
}

template <>
inline float cast<float, float>(float value) {
    return value;
}

template <>
inline bf16 cast<bf16, bf16>(bf16 value) {
    return value;
}

template <>
inline bf16 cast<bf16, float>(float value) {
    return floatToBf16(value);
}

template <>
inline float cast<float, bf16>(bf16 value) {
    return bf16ToFloat(value);
}

}  // namespace squash
