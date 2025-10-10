#include "core/tensor.hpp"
#include "lib/squash.hpp"

using namespace squash;

namespace {
using random_engine = std::default_random_engine;

template <class T>
tensor::Buffer zeros(uint size) {
    tensor::Buffer data(size * sizeof(T));
    std::fill_n(data.get<T>(), size, cast<T>(0.0f));
    return data;
}

template <class T>
tensor::Buffer randn(uint size, float stddev, random_engine& rng) {
    tensor::Buffer data(size * sizeof(T));
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

void benchmarkMlp() {
    random_engine rng(100);
    uint batchSize = 1;
    uint dModel = 2048;
    uint dFFN = 8192;

    auto inputs = tensor::randn<float>({batchSize, dModel}, rng, 1.0f);
    auto wUp = tensor::randn<bf16>({dFFN, dModel}, rng, 0.02f);
    auto wGate = tensor::clone(wUp);  // save RNG time
    auto wDown = tensor::reshape(tensor::clone(wUp), {dModel, dFFN});

    benchmarking::Benchmark benchmark;
    for (auto rep = 0u; rep < 10u; ++rep) {
        auto timer = benchmark.record();

        auto up = tensor::projection(wUp, inputs);
        auto gate = tensor::projection(wGate, inputs);
        up = tensor::swiGlu(std::move(up), gate);
        auto outputs = tensor::projection(wDown, up);
    }

    auto result = benchmark.result();
    auto flopCount = 2 * 3 * dModel * dFFN;
    std::cerr << "MLP (tensor): " << 1e3 * result << " ms  |  " << (1e-9 * flopCount) / result
              << " GFLOP/s\n";
}

}  // namespace

int main() {
    selectOmpNumThreads();
    benchmarkMlp();
    return 0;
}
