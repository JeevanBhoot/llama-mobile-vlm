#include <omp.h>
#include <cxxopts.hpp>
#include <json.hpp>
#include <thread>

#include "core/tensor.hpp"
#include "lib/squash.hpp"

using namespace squash;
using json = nlohmann::json;

namespace {

////////////////////////////////////////////////////////////////////
// Benchmarking Framework

namespace benchmarking {

struct Benchmark {
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

    const double SkipFirstFraction = 0.25;
    std::vector<double> times;

    Recorder record() { return Recorder(*this); }

    Measurement result() const {
        auto begin = times.begin() + uint(SkipFirstFraction * double(times.size()));
        auto n = uint(std::distance(begin, times.end()));
        auto mean = std::accumulate(begin, times.end(), 0.0) / double(n);
        auto meanSq =
            std::accumulate(begin, times.end(), 0.0,
                            [](double sumSq, double time) { return sumSq + time * time; }) /
            double(n);
        return {mean, std::sqrt((meanSq - mean * mean) / double(n)), n};
    }
};

Benchmark::Recorder::Recorder(Benchmark& benchmark) : benchmark(benchmark) {}
Benchmark::Recorder::~Recorder() {
    benchmark.times.push_back(timer.elapsed());
}
Benchmark::Measurement operator*(double lhs, const Benchmark::Measurement& rhs) {
    return {lhs * rhs.mean, lhs * rhs.error, rhs.count};
}
Benchmark::Measurement operator/(double lhs, const Benchmark::Measurement& rhs) {
    // Rough approximation - propagate relative error
    double value = lhs / rhs.mean;
    double relError = rhs.error / rhs.mean;
    return {value, std::abs(value) * relError, rhs.count};
}
std::ostream& operator<<(std::ostream& out, const Benchmark::Measurement& m) {
    return out << m.mean << " ± " << 2 * m.error;
}

struct Report {
    std::string name;
    bool jsonOutput;

    void operator()(json result) const {
        if (jsonOutput) {
            result["benchmark"] = name;
            std::cout << result.dump() << "\n";
        }
    }
};

struct BenchmarkRegistry {
    using Fn = void (*)(const Report&);
    std::vector<std::tuple<std::string, Fn>> benchmarks;

    void registerBenchmark(const std::string& name, Fn fn) { benchmarks.push_back({name, fn}); }

    void run(const std::string& prefix, bool jsonOutput) {
        auto nRun = 0u;
        for (const auto& [name, fn] : benchmarks) {
            if ((prefix.empty() && name.at(0) != '_') ||
                (!prefix.empty() && name.find(prefix) == 0)) {
                std::cerr << "-- Running benchmark: " << name << "\n";
                fn(Report{name, jsonOutput});
                ++nRun;
            }
        }
        if (nRun == 0) {
            std::cerr << "No benchmarks matched '" << prefix << "'\n";
        }
    }

    static BenchmarkRegistry& instance() {
        static BenchmarkRegistry registry;
        return registry;
    }

    struct Register {
        Register(const std::string& name, Fn fn) {
            BenchmarkRegistry::instance().registerBenchmark(name, fn);
        }
    };

   private:
    BenchmarkRegistry() = default;
    BenchmarkRegistry(const BenchmarkRegistry&) = delete;
    BenchmarkRegistry& operator=(const BenchmarkRegistry&) = delete;
};

// Usage: REGISTER_BENCHMARK(name)(const benchmarking::Report& report) { ... }
// If the name starts with '_', it won't be run unless explicitly requested.
#define REGISTER_BENCHMARK(name)                                                 \
    static void _benchmark_fn_##name(const benchmarking::Report& report);        \
    static benchmarking::BenchmarkRegistry::Register _benchmark_register_##name( \
        #name, &_benchmark_fn_##name);                                           \
    static void _benchmark_fn_##name

}  // namespace benchmarking

////////////////////////////////////////////////////////////////////
// Benchmarks

REGISTER_BENCHMARK(mlp)(const benchmarking::Report& report) {
    selectOmpNumThreads();
    std::default_random_engine rng(0xd87d1b26218ca5d7);
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
    std::cerr << report.name << ": " << 1e3 * result << " ms  |  " << (1e-9 * flopCount) / result
              << " GFLOP/s\n";
}

__attribute__((noinline)) void _copy(float* dest, const float* src, size_t n) {
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        dest[i] = src[i];
    }
}

REGISTER_BENCHMARK(_memory_bandwidth_omp)(const benchmarking::Report& report) {
    // Sweep: reps, nthreads, elements
    uint reps = 100u;
    auto maxThreads = std::thread::hardware_concurrency();
    std::vector<uint> nthreadsRange;
    // for (auto t = 1u; t < maxThreads; t *= 4u) {
    //     nthreadsRange.push_back(t);
    // }
    nthreadsRange.push_back(maxThreads);
    std::vector<ulong> elementsRange = {1ul << 24, 1ul << 26, 1ul << 28, 1ul << 30};

    for (auto elements : elementsRange) {
        for (auto threads : nthreadsRange) {
            omp_set_num_threads(int(threads));

            benchmarking::Benchmark benchmark;

            std::vector<float> src(elements, 0.0f), dest(elements, 0.0f);
            for (auto rep = 0u; rep < reps; ++rep) {
                auto timer = benchmark.record();
                _copy(dest.data(), src.data(), elements);
            }

            auto result = benchmark.result();
            auto byteCount = sizeof(float) * 2ul * elements;
            auto gibCount = static_cast<double>(byteCount) / double(1u << 30);
            report(json({
                {"benchmark", "memory_bandwidth_omp"},
                {"nthreads", threads},
                {"elements", elements},
                {"time_ms", 1e3 * result.mean},
                {"byte_count", byteCount},
                {"transfer_gib_s", gibCount / result.mean},
                {"time", benchmark.times},
            }));
            std::cerr << report.name << ": " << gibCount / result << " GiB/s for " << gibCount
                      << " GiB, with " << threads << " threads\n";
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    cxxopts::Options options("benchmark", "Benchmarks");
    options.add_options()                                                                       //
        ("prefix", "Prefix to select tests", cxxopts::value<std::string>()->default_value(""))  //
        ("json", "Output JSON to stdout", cxxopts::value<bool>()->default_value("false"))       //
        ("help", "Print help")                                                                  //
        ;
    options.parse_positional({"prefix"});
    options.positional_help("prefix");
    auto args = options.parse(argc, argv);
    if (args.count("help")) {
        std::cerr << options.help() << std::endl;
        return 0;
    }
    benchmarking::BenchmarkRegistry::instance().run(args["prefix"].as<std::string>(),
                                                    args["json"].as<bool>());
    return 0;
}
