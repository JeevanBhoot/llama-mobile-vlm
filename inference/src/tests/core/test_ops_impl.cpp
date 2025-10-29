#include "tests/common.hpp"

using namespace squash;
using Approx = Catch::Approx;

#ifdef __ARM_NEON
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
    REQUIRE(vgetq_lane_f32(acc, 0) == Approx(1.0f));
    REQUIRE(vgetq_lane_f32(acc, 1) == Approx(5.0f));
    REQUIRE(vgetq_lane_f32(acc, 2) == Approx(9.0f));
    REQUIRE(vgetq_lane_f32(acc, 3) == Approx(13.0f));

    // bfdot 4*1x2x1
    acc = vmovq_n_f32(0.0f);
    acc = vbfdotq_f32(acc, vld1q_bf16(reinterpret_cast<const __bf16*>(a.data())),
                      vld1q_bf16(reinterpret_cast<const __bf16*>(b.data())));
    REQUIRE(vgetq_lane_f32(acc, 0) == Approx(1.0f));
    REQUIRE(vgetq_lane_f32(acc, 1) == Approx(0.0f));
    REQUIRE(vgetq_lane_f32(acc, 2) == Approx(0.0f));
    REQUIRE(vgetq_lane_f32(acc, 3) == Approx(13.0f));
}

#endif  // __ARM_NEON
