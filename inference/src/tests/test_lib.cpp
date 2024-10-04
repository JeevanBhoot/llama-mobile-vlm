#include <catch_amalgamated.hpp>

#include "squash.hpp"

TEST_CASE("Meaning is correct", "[lib]") {
    REQUIRE(squash::meaning() == 42);
}
