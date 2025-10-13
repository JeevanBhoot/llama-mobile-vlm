#include <omp.h>
#include <thread>

#include "benchmark/benchmark.hpp"

using namespace squash;

namespace {

__attribute__((noinline)) void _copy(float* dest, const float* src, size_t n) {
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        dest[i] = src[i];
    }
}

REGISTER_BENCHMARK(_memory_bandwidth)(const benchmarking::Report& report) {
    // Sweep: {elements, threads}
    uint reps = 100u;
    auto maxThreads = std::thread::hardware_concurrency();
    std::vector<uint> nthreadsRange;
    for (auto t = 1u; t < maxThreads; t *= 4u) {
        nthreadsRange.push_back(t);
    }
    nthreadsRange.push_back(maxThreads);
    std::vector<ulong> elementsRange;  // = {1ul << 24, 1ul << 26, 1ul << 28, 1ul << 30};
    for (auto p = 24u; p <= 30u; p += 1u) {
        elementsRange.push_back(1ul << p);
    }

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
            auto byteCount = sizeof(float) * 2ull * elements;
            auto gibCount = static_cast<double>(byteCount) / double(1u << 30);
            report({
                {"nthreads", threads},
                {"elements", elements},
                {"time_ms", 1e3 * result.mean},
                {"byte_count", byteCount},
                {"transfer_gib_s", gibCount / result.mean},
                {"time", benchmark.times},
            });
            std::cerr << report << gibCount / result << " GiB/s for " << gibCount << " GiB, with "
                      << threads << " threads\n";
        }
    }
}

}  // anonymous namespace
