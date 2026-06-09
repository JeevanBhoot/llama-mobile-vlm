// Copyright (c) 2026 Graphcore Ltd. All rights reserved.

#include "benchmark/benchmark.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>

namespace squash::benchmarking {

Benchmark::Recorder Benchmark::record() {
    return Recorder(*this);
}

Benchmark::Measurement Benchmark::result(double skipFirstFraction) const {
    auto begin = times.begin() + uint(skipFirstFraction * double(times.size()));
    auto n = uint(std::distance(begin, times.end()));
    auto mean = std::accumulate(begin, times.end(), 0.0) / double(n);
    auto meanSq = std::accumulate(begin, times.end(), 0.0,
                                  [](double sumSq, double time) { return sumSq + time * time; }) /
                  double(n);
    if (mean < 1e-6) {
        std::cerr << "WARNING: benchmark measurement mean time is very small (" << mean * 1e6
                  << " us), results may be inaccurate due to timing overhead.\n";
    }
    return {mean, std::sqrt((meanSq - mean * mean) / double(n)), n};
}

Benchmark::Recorder::Recorder(Benchmark& benchmark) : benchmark(benchmark) {}

Benchmark::Recorder::~Recorder() {
    benchmark.times.push_back(timer.elapsed());
}

Benchmark::Measurement operator*(double lhs, const Benchmark::Measurement& rhs) {
    return {lhs * rhs.mean, lhs * rhs.error, rhs.count};
}

Benchmark::Measurement operator/(double lhs, const Benchmark::Measurement& rhs) {
    double value = lhs / rhs.mean;
    double relError = rhs.error / rhs.mean;
    return {value, std::abs(value) * relError, rhs.count};
}

std::ostream& operator<<(std::ostream& out, const Benchmark::Measurement& m) {
    return out << m.mean << " ± " << 2 * m.error;
}

Report Report::operator[](const std::string& child) const {
    return {name + "." + child, jsonOutput, shuffle};
}

void Report::operator()(nlohmann::json json) const {
    if (jsonOutput) {
        json["benchmark"] = name;
        std::cout << json.dump() << "\n";
    }
}

void Report::operator()(nlohmann::json::initializer_list_t result) const {
    operator()(nlohmann::json(result));
}

std::ostream& operator<<(std::ostream& out, const Report& report) {
    return out << report.name << ": ";
}

void Registry::run(const std::vector<std::string>& prefixes,
                   bool jsonOutput,
                   uint repeat,
                   bool shuffle) {
    const auto matches = [&](const std::string& name) {
        if (prefixes.empty()) {
            return name.at(0) != '_';
        }
        return std::any_of(prefixes.begin(), prefixes.end(),
                           [&](const std::string& prefix) { return name.find(prefix) == 0; });
    };

    std::vector<std::tuple<std::string, Fn>> benchmarks;
    std::copy_if(instance().benchmarks.begin(), instance().benchmarks.end(),
                 std::back_inserter(benchmarks),
                 [&](const auto& item) { return matches(std::get<0>(item)); });

    std::random_device randomDevice;
    std::mt19937_64 rng(randomDevice());
    auto nRun = 0u;
    for (uint r = 0; r < repeat; ++r) {
        if (shuffle) {
            std::shuffle(benchmarks.begin(), benchmarks.end(), rng);
        }
        for (const auto& [name, fn] : benchmarks) {
            std::cerr << "-- Running benchmark: " << name << "\n";
            fn(Report{name, jsonOutput, shuffle});
            ++nRun;
        }
    }
    if (nRun == 0) {
        std::cerr << "No benchmarks matched " << dump(prefixes) << "\n";
    }
}

Registry::Register::Register(const std::string& name, Registry::Fn fn) {
    instance().benchmarks.push_back({name, fn});
}

Registry& Registry::instance() {
    static Registry registry;
    return registry;
}

}  // namespace squash::benchmarking
