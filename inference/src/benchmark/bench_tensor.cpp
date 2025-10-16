#include "benchmark/benchmark.hpp"
#include "core/tensor.hpp"

#include <omp.h>

using namespace squash;

namespace {

void flushCache() {
    // Allocate a buffer larger than the largest cache
    const size_t cacheFlushSize = 512 * 1024 * 1024;
    static std::vector<char> cacheFlushBuffer(cacheFlushSize);
    for (auto n = 0; n < 10; ++n) {
        // Read-modify-write seems better than write-only for flushing caches
#pragma omp parallel for schedule(static)
        for (auto i = 0ull; i < cacheFlushBuffer.size(); ++i) {
            cacheFlushBuffer[i] += 1;
        }
    }
}

struct ComputeAndTransferBenchmark {
    ulong macCount;
    ulong byteCount;
    benchmarking::Benchmark benchmark = {};

    benchmarking::Benchmark::Recorder record() {
        flushCache();
        return benchmark.record();
    }

    void dump(const benchmarking::Report& report, nlohmann::json details) {
        auto result = benchmark.result();
        auto gmacCount = static_cast<double>(macCount) / 1e9;
        auto gibCount = static_cast<double>(byteCount) / double(1u << 30);
        details["mac_count"] = macCount;
        details["byte_count"] = byteCount;
        details["time_ms"] = 1e3 * result.mean;
        details["compute_gmac_s"] = gmacCount / result.mean;
        details["transfer_gib_s"] = gibCount / result.mean;
        details["time"] = benchmark.times;
        report(details);

        std::cerr << std::right << std::setw(40) << report << 1e3 * result.mean << " ms, "
                  << gmacCount / result.mean << " GMAC/s, " << gibCount / result.mean << " GiB/s\n";
    }
};

REGISTER_BENCHMARK(_tensor_copy)(const benchmarking::Report& report) {
    selectOmpNumThreads();

    for (auto i = 0; i < 1; ++i) {  // (use loop for temporary testing)
        // Check cache flushing works: copy speed should not exceed main memory bandwidth
        // Note - 64 MiB fits in the largest cache of a Graviton 4 CPU
        const size_t nelement = 64 * 1024 * 1024;
        auto x = tensor::randn({nelement}, 1.0f, 0x6ab49512d9d3de72);
        auto y = tensor::clone(x);

        ComputeAndTransferBenchmark benchmark{.macCount = 0,
                                              .byteCount = 2 * sizeof(bf16) * nelement};
        for (auto rep = 0u; rep < 100u; ++rep) {
            auto timer = benchmark.record();
            tensor::assign(y, x);
        }
        benchmark.dump(report, {{"nelement", nelement}});
    }
}

REGISTER_BENCHMARK(_tensor_projection)(const benchmarking::Report& report) {
    selectOmpNumThreads();
    std::vector<std::tuple<uint, uint, uint, std::string>> cases = {
        // Sizes for 11B (batchSize, dIn, dOut)
        {1, 4096, 14336, "text.mlp.up"},  //
        {1, 14336, 4096, "text.mlp.down"},
        {1, 4096, 4096, "text.attn.[q,o]"},
        {1, 4096, 1024, "text.attn.[k,v]"},
        {1, 4096, 128256, "text.output"},
        {1601, 1280, 5120, "vision.mlp.up"},
        {1601, 5120, 1280, "vision.mlp.down"},
        {1601, 1280, 1280, "vision.attn.[q,k,v,o]"},
    };
    for (auto [batchSize, dIn, dOut, name] : cases) {
        auto weight = tensor::randn({dOut, dIn}, 1.0f, 0x23f3ac651617c540);
        auto x = tensor::randn({batchSize, dIn}, 0.02f, 0x6f5d77f384975947);

        ComputeAndTransferBenchmark benchmark{
            .macCount = ulong(batchSize) * ulong(dIn * dOut),
            .byteCount = sizeof(bf16) * (batchSize * dIn + batchSize * dOut + dOut * dIn),
        };
        auto reps = std::clamp(uint(1e11 / double(benchmark.macCount)), 20u, 200u);

        for (auto rep = 0u; rep < reps; ++rep) {
            auto timer = benchmark.record();
            tensor::projection(weight, x);
        }
        benchmark.dump(report[name], {{"batch_size", batchSize}, {"d_in", dIn}, {"d_out", dOut}});
    }
}

REGISTER_BENCHMARK(_tensor_attention)(const benchmarking::Report& report) {
    selectOmpNumThreads();
    std::vector<std::tuple<uint, uint, uint, uint, uint, bool, std::string>> cases = {
        {1, 128, 8, 4, 128, true, "text.generate.self"},
        {1, 1601, 8, 4, 128, false, "text.generate.cross"},
        {128, 128, 8, 4, 128, true, "text.prefill.self"},
        {128, 1601, 8, 4, 128, false, "text.prefill.cross"},
        {1601, 1601, 16, 1, 80, false, "vision"},
    };
    for (auto [seq_q, seq_kv, heads_kv, heads_q, head_dim, causal, name] : cases) {
        auto query = tensor::randn({seq_q, heads_kv, heads_q, head_dim}, 1.0f, 0x5df167e9a6cef17c);
        auto key = tensor::randn({seq_kv, heads_kv, head_dim}, 1.0f, 0x2585ce8b37872a48);
        auto value = tensor::randn({seq_kv, heads_kv, head_dim}, 1.0f, 0x88851b4ead761eb5);

        ComputeAndTransferBenchmark benchmark{
            .macCount =
                2 * ulong(head_dim * heads_kv * heads_q) * ulong(seq_q * seq_kv) / (1 + causal),
            .byteCount = sizeof(bf16) * head_dim * heads_kv * (2 * seq_q * heads_q + 2 * seq_kv),
        };
        for (auto rep = 0u; rep < 20u; ++rep) {
            auto timer = benchmark.record();
            query = tensor::attention(std::move(query), key, value, causal);
        }
        benchmark.dump(report[name], {{"seq_q", seq_q},
                                      {"seq_kv", seq_kv},
                                      {"heads_kv", heads_kv},
                                      {"heads_q", heads_q},
                                      {"head_dim", head_dim},
                                      {"causal", causal}});
    }
}

REGISTER_BENCHMARK(tensor_mlp)(const benchmarking::Report& report) {
    selectOmpNumThreads();
    std::default_random_engine rng(0xd87d1b26218ca5d7);
    uint batchSize = 1;
    uint dModel = 4096;
    uint dFFN = 14336;

    auto inputs = tensor::randn({batchSize, dModel}, 1.0f, rng());
    auto wUp = tensor::randn({dFFN, dModel}, 0.02f, rng());
    auto wGate = tensor::randn({dFFN, dModel}, 0.02f, rng());
    auto wDown = tensor::randn({dModel, dFFN}, 0.02f, rng());

    ComputeAndTransferBenchmark benchmark{
        .macCount = 3ull * ulong(batchSize) * ulong(dModel) * ulong(dFFN),
        .byteCount = sizeof(bf16) * (batchSize * (3 * dModel + 5 * dFFN) + (3 * dFFN * dModel)),
    };

    for (auto rep = 0u; rep < 100u; ++rep) {
        auto timer = benchmark.record();

        auto up = tensor::projection(wUp, inputs);
        auto gate = tensor::projection(wGate, inputs);
        up = tensor::swiGlu(std::move(up), gate);
        auto outputs = tensor::projection(wDown, up);
    }
    benchmark.dump(report, {{"batch_size", batchSize}, {"d_model", dModel}, {"d_ffn", dFFN}});
}

}  // namespace
