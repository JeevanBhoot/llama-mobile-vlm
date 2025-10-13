#include "tests/common.hpp"

using namespace squash::tensor;
namespace M = Catch::Matchers;

TEST_CASE("squash::tensor::creation") {
    // create()
    auto x = create(std::vector<float>({1, 2, 3, 4, 5, 6}));
    REQUIRE(x.shape == Shape({6}));
    REQUIRE(data<float>(x)[5] == 6);

    // zeros()
    x = zeros<float>({3, 5});
    REQUIRE(x.shape == Shape({3, 5}));
    for (auto i = 0u; i < 15; ++i) {
        REQUIRE(data<float>(x)[i] == 0.0f);
    }

    // randn()
    std::default_random_engine rng(123);
    x = randn<float>({55}, rng, 1.0f);
    REQUIRE(x.shape == Shape({55}));
}

TEST_CASE("squash::tensor::basic") {
    auto x = create(std::vector<float>({1, 2, 3, 4, 5, 6}));

    // reshape()
    x = reshape(std::move(x), {3, 2});
    REQUIRE(x.shape == Shape({3, 2}));

    // unsqueeze()
    REQUIRE(unsqueeze(x, {0}).shape == Shape({1, 3, 2}));
    REQUIRE(unsqueeze(x, {1, 1}).shape == Shape({3, 1, 1, 2}));
    REQUIRE(unsqueeze(x, {1, 2}).shape == Shape({3, 1, 1, 2}));
    REQUIRE(unsqueeze(x, {2, 1}).shape == Shape({3, 1, 2, 1}));

    // Helper function
    auto _t = [](const std::vector<float>& v, const Shape& shape) {
        return reshape(create(v), shape);
    };

    // indexLeading()
    REQUIRE_TENSOR_APPROX_EQUALS(indexLeading(x, {1}), _t({3, 4}, {2}), 0.0);

    // slice0()
    REQUIRE_TENSOR_APPROX_EQUALS(slice0(x, 1, 3), _t({3, 4, 5, 6}, {2, 2}), 0.0);

    // clone(), assign()
    auto y = clone(x);
    REQUIRE_TENSOR_APPROX_EQUALS(y, x, 0.0);
    assign(indexLeading(y, {0}), _t({10, 20}, {2}));
    assign(indexLeading(y, {2}), _t({500, 600}, {2}));
    REQUIRE_TENSOR_APPROX_EQUALS(y, _t({10, 20, 3, 4, 500, 600}, {3, 2}), 0.0);
    REQUIRE_TENSOR_APPROX_EQUALS(x, _t({1, 2, 3, 4, 5, 6}, {3, 2}), 0.0);
}

TEST_CASE("squash::tensor::sample") {
    std::vector<float> ps({0.25f, 0.125f, 0.5f, 0.125f});
    std::vector<float> logitsData;
    std::transform(ps.begin(), ps.end(), std::back_inserter(logitsData),
                   [](float v) { return 10 + std::log(v); });
    auto logits = create(logitsData);

    std::default_random_engine rng(1234);
    const auto sampleN = 1000u;
    auto sampleMany = [&](float temperature, uint topK, float topP) {
        std::vector<uint> results(sampleN);
        for (auto& r : results) {
            r = sample(logits, temperature, topK, topP, rng);
        }
        return results;
    };

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
    std::vector<uint> counts(logits.shape[0]);
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
