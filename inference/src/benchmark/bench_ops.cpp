
#include "benchmark/benchmark.hpp"
#include "core/ops.hpp"

#include <iostream>
#include <vector>

using namespace squash;

namespace {

REGISTER_BENCHMARK(ops_matmulT)(const benchmarking::Report& report) {
    constexpr uint dM = 1, dK = 4096, dN = 14336;

    std::vector<bf16> lhs(dM * dK), rhs(dN * dK), out(dM * dN);
    ops::randn(lhs.data(), lhs.size(), 1.0f, 0x19b33f4a1f2f9d1e);
    ops::randn(rhs.data(), rhs.size(), 0.02f, 0x8e70f4d6a9d31c52);

    benchmarking::Benchmark benchmark;
    for (auto rep = 0u; rep < 100u; ++rep) {
        auto timer = benchmark.record();
        ops::matmulT(lhs.data(), rhs.data(), dM, dK, dN, out.data());
    }

    auto result = benchmark.result();
    auto gmacPerSecond = (double(dM) * double(dK) * double(dN) / 1e9) / result.mean;
    report({{"d_m", dM},
            {"d_k", dK},
            {"d_n", dN},
            {"time_ms", 1e3 * result.mean},
            {"compute_gmac_s", gmacPerSecond}});
    std::cerr << std::right << std::setw(45) << report << 1e3 * result.mean << " ms, "
              << gmacPerSecond << " GMAC/s\n";
}

}  // namespace
