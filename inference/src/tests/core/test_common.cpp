#include "core/common.hpp"
#include "tests/common.hpp"

using namespace squash;

TEST_CASE("squash::bf16", "[squash]") {
    REQUIRE(float(bf16::reinterpret(0x4147)) == 12.4375f);
    REQUIRE(float(bf16::reinterpret(-0x3f80)) == -4.0f);
    REQUIRE(bf16(12.4375f)._intValue == 0x4147);
    REQUIRE(bf16(-4.0f)._intValue == -0x3f80);

    // Rounding
    REQUIRE(float(bf16(2.75f)) == 2.75f);
    REQUIRE(float(bf16(2.749f)) == 2.75f);
    REQUIRE(float(bf16(2.751f)) == 2.75f);

    // Ties to even
    REQUIRE(float(bf16(1.00390625f)) == 1.0f);
    REQUIRE(float(bf16(1.01171875f)) == 1.015625f);
}
