#pragma once

#include <catch_amalgamated.hpp>
#include "lib/squash.hpp"

namespace squash {

template <class To, class From>
To cast(From value);

template <>
inline float cast<float, float>(float value) {
    return value;
}

template <>
inline bf16 cast<bf16, float>(float value) {
    union {
        float f;
        int16_t i[2];
    } u;
    u.f = value;
    return u.i[1];
}

}  // namespace squash
