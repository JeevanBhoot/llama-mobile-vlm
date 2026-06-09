// Copyright (c) 2026 Graphcore Ltd. All rights reserved.

#pragma once

#include <cstdint>

namespace squash {

struct bf16 {
    int16_t _intValue;

    bf16() = default;
    explicit bf16(float value);
    explicit operator float() const;
    static bf16 reinterpret(int16_t value);
};

using uint = uint32_t;
using ulong = uint64_t;

}  // namespace squash

/////////////////////////////////////////////////////////////////////////////
// Impl

namespace squash {

static_assert(sizeof(bf16) == 2, "bf16 must be 2 bytes");

inline bf16::bf16(float value) {
    // Round to nearest, ties to even
    uint32_t u = reinterpret_cast<uint32_t&>(value);
    u += 0x7FFFu + ((u >> 16) & 1u);
    _intValue = static_cast<int16_t>(u >> 16);
}

inline bf16::operator float() const {
    union {
        float f;
        int16_t i[2];
    } u;
    u.i[0] = 0;
    u.i[1] = _intValue;
    return u.f;
}

inline bf16 bf16::reinterpret(int16_t value) {
    bf16 v;
    v._intValue = value;
    return v;
}

}  // namespace squash
