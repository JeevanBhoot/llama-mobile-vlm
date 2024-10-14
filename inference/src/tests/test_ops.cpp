#include "ops.hpp"
#include "tests.hpp"

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

namespace {
using random_engine = std::default_random_engine;

template <class T>
Buffer zeros(uint size) {
    Buffer data(size * sizeof(T));
    std::fill_n(data.get<T>(), size, convertTruncate<T>(0.0f));
    return data;
}

template <class T>
Buffer randn(uint size, float stddev, random_engine& rng) {
    Buffer data(size * sizeof(T));
    auto dist = std::normal_distribution<float>(0, stddev);
    std::generate_n(data.get<T>(), size, [&] { return convertTruncate<T>(dist(rng)); });
    return data;
}

namespace benchmarking {

struct Benchmark;

struct Recorder {
    Benchmark& benchmark;
    Timer timer;
    explicit Recorder(Benchmark&);
    ~Recorder();
};

struct Measurement {
    double mean;
    double error;
    uint count;
};

struct Benchmark {
    std::vector<double> times;

    Recorder record() { return Recorder(*this); }

    Measurement result() const {
        // Skip the first 25% of results
        auto begin = times.begin() + uint(0.25 * double(times.size()));
        auto n = uint(std::distance(begin, times.end()));
        auto mean = std::accumulate(begin, times.end(), 0.0) / double(n);
        auto meanSq =
            std::accumulate(begin, times.end(), 0.0,
                            [](double sumSq, double time) { return sumSq + time * time; }) /
            double(n);
        return {mean, std::sqrt((meanSq - mean * mean) / double(n)), n};
    }
};

Recorder::Recorder(Benchmark& benchmark) : benchmark(benchmark) {}
Recorder::~Recorder() {
    benchmark.times.push_back(timer.elapsed());
}
Measurement operator*(double lhs, const Measurement& rhs) {
    return {lhs * rhs.mean, lhs * rhs.error, rhs.count};
}
Measurement operator/(double lhs, const Measurement& rhs) {
    // Very rough - maybe correct for small values of error
    return {lhs / rhs.mean, rhs.error * (lhs / rhs.mean), rhs.count};
}
std::ostream& operator<<(std::ostream& out, const Measurement& m) {
    return out << m.mean << " ± " << 2 * m.error;
}

}  // namespace benchmarking
}  // namespace

TEST_CASE("benchmark-ops-MLP", "[squash][benchmark]") {
    random_engine rng(100);
    uint batchSize = 1;
    uint dModel = 2048;
    uint dFFN = 8192;

    auto inputs = randn<float>(batchSize * dModel, 1, rng);
    auto wUp = randn<bf16>(dModel * dFFN, 0.02f, rng);
    auto wGate = wUp.copy(dModel * dFFN);  // save RNG time
    auto wDown = wUp.copy(dModel * dFFN);

    benchmarking::Benchmark benchmark;
    for (auto rep = 0u; rep < 10u; ++rep) {
        auto timer = benchmark.record();
        Buffer up(batchSize * dFFN * sizeof(float));
        Buffer gate(batchSize * dFFN * sizeof(float));
        auto outputs = zeros<float>(batchSize * dModel);
        ops::matmulT(inputs.get<float>(), wUp.get<bf16>(), batchSize, dModel, dFFN,
                     up.get<float>());
        ops::matmulT(inputs.get<float>(), wGate.get<bf16>(), batchSize, dModel, dFFN,
                     gate.get<float>());
        ops::swiGluInPlace(up.get<float>(), gate.get<float>(), batchSize * dFFN);
        ops::matmulT(up.get<float>(), wDown.get<bf16>(), batchSize, dFFN, dModel,
                     outputs.get<float>());
    }
    auto result = benchmark.result();
    auto flopCount = 2 * 3 * dModel * dFFN;
    std::cerr << 1e3 * result << " ms  |  " << (1e-9 * flopCount) / result << " GFLOP/s\n";
}
