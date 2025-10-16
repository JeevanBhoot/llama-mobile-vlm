#include "benchmark/benchmark.hpp"
#include "core/tensor.hpp"

using namespace squash;

namespace {

REGISTER_BENCHMARK(mlp)(const benchmarking::Report& report) {
    selectOmpNumThreads();
    std::default_random_engine rng(0xd87d1b26218ca5d7);
    uint batchSize = 1;
    uint dModel = 2048;
    uint dFFN = 8192;

    auto timer = Timer();
    auto inputs = tensor::randn({batchSize, dModel}, 1.0f, rng());
    auto wUp = tensor::randn({dFFN, dModel}, 0.02f, rng());
    auto wGate = tensor::randn({dFFN, dModel}, 0.02f, rng());
    auto wDown = tensor::randn({dModel, dFFN}, 0.02f, rng());

    benchmarking::Benchmark benchmark;
    for (auto rep = 0u; rep < 100u; ++rep) {
        auto timer = benchmark.record();

        auto up = tensor::projection(wUp, inputs);
        auto gate = tensor::projection(wGate, inputs);
        up = tensor::swiGlu(std::move(up), gate);
        auto outputs = tensor::projection(wDown, up);
    }

    auto result = benchmark.result();
    auto macCount = 3ull * static_cast<ulong>(dModel) * static_cast<ulong>(dFFN);
    auto gmacCount = static_cast<double>(macCount) / 1e9;
    auto byteCount =
        sizeof(float) * batchSize * (3 * dModel + 5 * dFFN) + sizeof(bf16) * (3 * dFFN * dModel);
    auto gibCount = static_cast<double>(byteCount) / double(1u << 30);
    report({
        {"batch_size", batchSize},
        {"d_model", dModel},
        {"d_ffn", dFFN},
        {"time_ms", 1e3 * result.mean},
        {"mac_count", macCount},
        {"compute_gmac_s", gmacCount / result.mean},
        {"transfer_gib_s", gibCount / result.mean},
        {"time", benchmark.times},
    });
    std::cerr << report << gmacCount / result << " GMAC/s, " << gibCount / result << " GiB/s\n";
}

}  // namespace
