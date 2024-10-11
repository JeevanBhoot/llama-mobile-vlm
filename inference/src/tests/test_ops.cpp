#include <catch_amalgamated.hpp>

#include "ops.hpp"

using namespace squash;
namespace M = Catch::Matchers;

TEST_CASE("squash::ops::sample") {
    std::vector<float> ps({0.25f, 0.125f, 0.5f, 0.125f});
    std::vector<float> logits;
    std::transform(ps.begin(), ps.end(), std::back_inserter(logits),
                   [](float v) { return 10 + std::log(v); });

    std::default_random_engine rng(1234);
    const auto sampleN = 1000u;
    auto sampleMany = [&](float temperature, uint topK, float topP) {
        std::vector<uint> results(sampleN);
        for (auto& r : results) {
            r = ops::sample(logits.data(), uint(logits.size()), temperature, topK, topP, rng);
        }
        return results;
    };

    // DUMPSQ(dump(sampleMany(/*temperature*/ 1, /*topK*/ 4u, /*topP*/ 1)));

    // Greedy (zero temperature)
    REQUIRE_THAT(sampleMany(/*temperature*/ 0, /*topK*/ 4u, /*topP*/ 1),
                 M::Equals(std::vector<uint>(sampleN, 2u)));
    // Greedy (topK = 1)
    REQUIRE_THAT(sampleMany(/*temperature*/ 1, /*topK*/ 1u, /*topP*/ 1),
                 M::Equals(std::vector<uint>(sampleN, 2u)));
    // Greedy (topP = 0)
    REQUIRE_THAT(sampleMany(/*temperature*/ 1, /*topK*/ 4u, /*topP*/ 0),
                 M::Equals(std::vector<uint>(sampleN, 2u)));

    // Only take top-2
    REQUIRE_THAT(sampleMany(/*temperature*/ 1, /*topK*/ 2u, /*topP*/ 1),
                 M::Contains(0u) && !M::Contains(1u) && M::Contains(2u) && !M::Contains(3u));
    REQUIRE_THAT(sampleMany(/*temperature*/ 1, /*topK*/ 3u, /*topP*/ 0.7f),
                 M::Contains(0u) && !M::Contains(1u) && M::Contains(2u) && !M::Contains(3u));

    // Full sampling
    std::vector<uint> counts(logits.size());
    for (auto i : sampleMany(/*temperature*/ 1, /*topK*/ 4u, /*topP*/ 1)) {
        counts[i]++;
    }
    auto InRange = [](uint low, uint high) {
        return M::Predicate<uint>([low, high](uint x) { return low <= x && x <= high; });
    };
    // for p in [0.5, 0.25, 0.125]:
    //     print(p, scipy.stats.binom.ppf([0.001, 0.999], 1000, p))
    REQUIRE_THAT(counts[0], InRange(208, 293));
    REQUIRE_THAT(counts[1], InRange(94, 158));
    REQUIRE_THAT(counts[2], InRange(451, 549));
    REQUIRE_THAT(counts[3], InRange(94, 158));
}
