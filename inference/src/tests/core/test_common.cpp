#include "core/common.hpp"
#include "tests/tests.hpp"

using namespace squash;

TEST_CASE("squash::bf16ToFloat", "[squash]") {
    REQUIRE(bf16ToFloat(0x4147) == 12.4375f);
    REQUIRE(bf16ToFloat(-0x3f80) == -4.0f);
    REQUIRE(floatToBf16(12.4375f) == 0x4147);
    REQUIRE(floatToBf16(-4.0f) == -0x3f80);

    // Rounding
    REQUIRE(bf16ToFloat(floatToBf16(2.75f)) == 2.75f);
    REQUIRE(bf16ToFloat(floatToBf16(2.749f)) == 2.75f);
    REQUIRE(bf16ToFloat(floatToBf16(2.751f)) == 2.75f);

    // Ties to even
    REQUIRE(bf16ToFloat(floatToBf16(1.00390625f)) == 1.0f);
    REQUIRE(bf16ToFloat(floatToBf16(1.01171875f)) == 1.015625f);
}
