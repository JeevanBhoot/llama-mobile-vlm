// Copyright (c) 2026 Graphcore Ltd. All rights reserved.

#include "tests/common.hpp"

using namespace squash;
using Approx = Catch::Approx;

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)
#include <arm_neon.h>

std::vector<bf16> vtobf16(const std::vector<float>& input) {
    std::vector<bf16> output(input.size());
    std::transform(input.begin(), input.end(), output.begin(), [](float v) { return bf16(v); });
    return output;
}

TEST_CASE("ops_impl::arm::bf16") {
    auto a = vtobf16({0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f});
    auto b = vtobf16({1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f});

    // bfmmla 1*2x4x2
    float32x4_t acc = vmovq_n_f32(0.0f);
    acc = vbfmmlaq_f32(acc, vld1q_bf16(reinterpret_cast<const __bf16*>(a.data())),
                       vld1q_bf16(reinterpret_cast<const __bf16*>(b.data())));
    REQUIRE(vgetq_lane_f32(acc, 0) == Approx(1.0f));   // a[0:4] . b[0:4]
    REQUIRE(vgetq_lane_f32(acc, 1) == Approx(5.0f));   // a[0:4] . b[4:8]
    REQUIRE(vgetq_lane_f32(acc, 2) == Approx(9.0f));   // a[4:8] . b[0:4]
    REQUIRE(vgetq_lane_f32(acc, 3) == Approx(13.0f));  // a[4:8] . b[4:8]

    // bfdot 4*1x2x1
    acc = vmovq_n_f32(0.0f);
    acc = vbfdotq_f32(acc, vld1q_bf16(reinterpret_cast<const __bf16*>(a.data())),
                      vld1q_bf16(reinterpret_cast<const __bf16*>(b.data())));
    REQUIRE(vgetq_lane_f32(acc, 0) == Approx(1.0f));   // a[0:2] . b[0:2]
    REQUIRE(vgetq_lane_f32(acc, 1) == Approx(0.0f));   // a[2:4] . b[2:4]
    REQUIRE(vgetq_lane_f32(acc, 2) == Approx(0.0f));   // a[4:6] . b[4:6]
    REQUIRE(vgetq_lane_f32(acc, 3) == Approx(13.0f));  // a[6:8] . b[6:8]
}

#endif  // __ARM_NEON && __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
