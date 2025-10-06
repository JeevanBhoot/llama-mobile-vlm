#include "core/common.hpp"
#include "tests/tests.hpp"

using namespace squash;

TEST_CASE("squash::bf16ToFloat", "[squash]") {
    REQUIRE(bf16ToFloat(0x4147) == 12.4375f);
    REQUIRE(bf16ToFloat(-0x3f80) == -4.0f);
}
