#pragma once

#include <cmath>
#include <functional>
#include <iostream>
#include <json.hpp>
#include <numeric>
#include <string>
#include <tuple>
#include <vector>

#include "lib/squash.hpp"

namespace squash::benchmarking {

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
    Recorder record();
    Measurement result() const;
};

Benchmark::Measurement operator*(double, const Benchmark::Measurement&);
Benchmark::Measurement operator/(double, const Benchmark::Measurement&);
std::ostream& operator<<(std::ostream&, const Benchmark::Measurement&);

struct Report {
    std::string name;
    bool jsonOutput;
    void operator()(nlohmann::json::initializer_list_t result) const;
};
std::ostream& operator<<(std::ostream&, const Report&);

struct Registry {
    using Fn = void (*)(const Report&);
    std::vector<std::tuple<std::string, Fn>> benchmarks;

    static void run(const std::string& prefix, bool jsonOutput);
    struct Register {
        Register(const std::string& name, Fn fn);
    };
    static Registry& instance();

   private:
    Registry() = default;
    Registry(const Registry&) = delete;
    Registry& operator=(const Registry&) = delete;
};

// Usage: REGISTER_BENCHMARK(name)(const squash::benchmarking::Report& report) { ... }
#define REGISTER_BENCHMARK(name)                                                  \
    static void _benchmark_fn_##name(const squash::benchmarking::Report& report); \
    static squash::benchmarking::Registry::Register _benchmark_register_##name(   \
        #name, &_benchmark_fn_##name);                                            \
    static void _benchmark_fn_##name

}  // namespace squash::benchmarking
