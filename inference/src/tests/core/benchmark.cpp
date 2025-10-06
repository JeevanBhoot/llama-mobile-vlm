#include "core/ops.hpp"
#include "tests/tests.hpp"

using namespace squash;

namespace {
using random_engine = std::default_random_engine;

template <class T>
Buffer zeros(uint size) {
    Buffer data(size * sizeof(T));
    std::fill_n(data.get<T>(), size, cast<T>(0.0f));
    return data;
}

template <class T>
Buffer randn(uint size, float stddev, random_engine& rng) {
    Buffer data(size * sizeof(T));
    auto dist = std::normal_distribution<float>(0, stddev);
    std::generate_n(data.get<T>(), size, [&] { return cast<T>(dist(rng)); });
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
    // Very rough - possibly correct for small values of error
    return {lhs / rhs.mean, rhs.error * (lhs / rhs.mean), rhs.count};
}
std::ostream& operator<<(std::ostream& out, const Measurement& m) {
    return out << m.mean << " ± " << 2 * m.error;
}

}  // namespace benchmarking
}  // namespace

TEMPLATE_TEST_CASE("benchmark-ops-MLP", "[squash][benchmark]", bf16) {
    selectOmpNumThreads();
    random_engine rng(100);
    uint batchSize = 1;
    uint dModel = 2048;
    uint dFFN = 8192;

    auto inputs = randn<float>(batchSize * dModel, 1, rng);
    auto wUp = randn<TestType>(dModel * dFFN, 0.02f, rng);
    auto wGate = wUp.copy(dModel * dFFN * sizeof(TestType));  // save RNG time
    auto wDown = wUp.copy(dModel * dFFN * sizeof(TestType));

    benchmarking::Benchmark benchmark;
    for (auto rep = 0u; rep < 10u; ++rep) {
        auto timer = benchmark.record();
        Buffer up(batchSize * dFFN * sizeof(float));
        Buffer gate(batchSize * dFFN * sizeof(float));
        auto outputs = zeros<float>(batchSize * dModel);
        ops::matmulT(inputs.get<float>(), wUp.template get<TestType>(), batchSize, dModel, dFFN,
                     up.get<float>());
        ops::matmulT(inputs.get<float>(), wGate.template get<TestType>(), batchSize, dModel, dFFN,
                     gate.get<float>());
        ops::swiGluInPlace(up.get<float>(), gate.get<float>(), batchSize * dFFN);
        ops::matmulT(up.get<float>(), wDown.template get<TestType>(), batchSize, dFFN, dModel,
                     outputs.get<float>());
    }
    auto result = benchmark.result();
    auto flopCount = 2 * 3 * dModel * dFFN;
    std::cerr << 1e3 * result << " ms  |  " << (1e-9 * flopCount) / result << " GFLOP/s\n";
}
