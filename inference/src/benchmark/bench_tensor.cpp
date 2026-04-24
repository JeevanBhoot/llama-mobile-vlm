#include "benchmark/benchmark.hpp"
#include "core/tensor.hpp"

#include <omp.h>

using namespace squash;

namespace {

struct ComputeAndTransferBenchmark {
    ulong macCount;
    ulong byteCount;
    benchmarking::Benchmark benchmark = {};

    benchmarking::Benchmark::Recorder record() { return benchmark.record(); }

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

        std::cerr << std::right << std::setw(45) << report << 1e3 * result.mean << " ms, "
                  << gmacCount / result.mean << " GMAC/s, " << gibCount / result.mean << " GiB/s\n";
    }
};

std::vector<tensor::Tensor> cloneN(const tensor::Tensor& tensor, uint n) {
    std::vector<tensor::Tensor> clones;
    clones.reserve(n);
    for (auto i = 0u; i < n; ++i) {
        clones.push_back(clone(tensor));
    }
    return clones;
}

REGISTER_BENCHMARK(_tensor_copy)(const benchmarking::Report& report) {
    selectOmpNumThreads();

    // Use a large tensor to avoid the last-level cache
    const size_t nelement = (1ull << 30) / (2 * sizeof(bf16));  // ~1 GiB (R+W)
    auto x = tensor::randn({nelement}, 1.0f, 0x6ab49512d9d3de72);
    auto y = tensor::clone(x);

    ComputeAndTransferBenchmark benchmark{.macCount = 0, .byteCount = 2 * sizeof(bf16) * nelement};
    for (auto rep = 0u; rep < 10u; ++rep) {
        auto timer = benchmark.record();
        tensor::assign(y, x);
    }
    benchmark.dump(report, {{"nelement", nelement}});
}

// ### _tensor_proj benchmarks

tensor::Tensor randnBf16Tensor(uint dOut, uint dIn, ulong seed) {
    return tensor::randn({dOut, dIn}, 1.0f, seed);
}

tensor::Tensor randnChannelInt8Tensor(uint dOut, uint dIn, ulong seed) {
    const auto scaleOffset = tensor::align(ulong(dOut) * ulong(dIn) * sizeof(int8_t));
    auto buffer = tensor::Buffer(scaleOffset + sizeof(bf16) * ulong(dOut));

    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int8_t> valueDist(-128, 127);
    std::uniform_real_distribution<float> uniformDist(0.0f, 1.0f);
    auto* data = reinterpret_cast<int8_t*>(buffer.get<char>());
    auto* scale = reinterpret_cast<bf16*>(buffer.get<char>() + scaleOffset);
#pragma omp parallel for
    for (auto n = 0u; n < dOut; ++n) {
        scale[n] = bf16(0.02f + 0.3f * uniformDist(rng));
        for (auto k = 0u; k < dIn; ++k) {
            data[n * dIn + k] = valueDist(rng);
        }
    }
    return tensor::Tensor{{.data = tensor::_data::ChannelInt8(data, scale), .shape = {dOut, dIn}},
                          std::move(buffer)};
}

tensor::Tensor randnChannelS3D8Tensor(uint dOut, uint dIn, ulong seed) {
    const auto packedRows = (dOut + 2) / 3;
    const auto lutOffset = tensor::align(ulong(packedRows) * ulong(dIn) * sizeof(uint8_t));
    const auto scaleOffset = tensor::align(lutOffset + 3u * 64u * sizeof(int8_t));

    auto buffer = tensor::Buffer(scaleOffset + sizeof(bf16) * ulong(dOut));
    auto* data = reinterpret_cast<uint8_t*>(buffer.get<char>());
    auto* lut = reinterpret_cast<int8_t*>(buffer.get<char>() + lutOffset);
    auto* scale = reinterpret_cast<bf16*>(buffer.get<char>() + scaleOffset);

    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int8_t> centroidDist(-128, 127);
    std::uniform_int_distribution<int> indexDist(0, 31);
    std::uniform_int_distribution<int8_t> signDist(0, 1);
    std::uniform_real_distribution<float> uniformDist(0.0f, 1.0f);
    std::vector<int8_t> centroids(32u * 3u);
    for (auto& centroid : centroids) {
        centroid = centroidDist(rng);
    }
    tensor::_data::ChannelS3D8::expandLut(centroids.data(), lut);
#pragma omp parallel for
    for (auto row = 0u; row < packedRows; ++row) {
        for (auto k = 0u; k < dIn; ++k) {
            auto idx = indexDist(rng);
            auto sign0 = signDist(rng), sign1 = signDist(rng), sign2 = signDist(rng);
            data[row * dIn + k] =
                static_cast<uint8_t>((idx << 1) | sign0 | (sign1 << 6) | ((sign0 ^ sign2) << 7));
        }
    }
#pragma omp parallel for
    for (auto n = 0u; n < dOut; ++n) {
        scale[n] = bf16(0.02f + 0.3f * uniformDist(rng));
    }
    return tensor::Tensor{
        {.data = tensor::_data::ChannelS3D8(data, lut, scale), .shape = {dOut, dIn}},
        std::move(buffer)};
}

template <class MakeInput, class MakeWeight>
void benchmarkMatmulT(const benchmarking::Report& report,
                      MakeInput&& makeInput,
                      MakeWeight&& makeWeight,
                      ulong seed) {
    selectOmpNumThreads();
    std::vector<std::tuple<uint, uint, uint, std::string>> cases = {
        // Sizes for 11B (batchSize, dIn, dOut) == (dM, dK, dN)
        {1, 4096, 14336, "text.generate.mlp.up"},     //
        {1, 14336, 4096, "text.generate.mlp.down"},   //
        {1, 4096, 4096, "text.generate.attn.[q,o]"},  //
        {1, 4096, 1024, "text.generate.attn.[k,v]"},  //
        {1, 4096, 128256, "text.generate.predict"},   //
        //
        {128, 4096, 14336, "text.prefill.mlp.up"},     //
        {128, 14336, 4096, "text.prefill.mlp.down"},   //
        {128, 4096, 4096, "text.prefill.attn.[q,o]"},  //
        {128, 4096, 1024, "text.prefill.attn.[k,v]"},  //
        //
        {1601, 1280, 5120, "vision.mlp.up"},          //
        {1601, 5120, 1280, "vision.mlp.down"},        //
        {1601, 1280, 1280, "vision.attn.[q,k,v,o]"},  //
    };
    for (const auto& [batchSize, dIn, dOut, name] : cases) {
        auto caseSeed = seed ^ std::hash<std::string>{}(name);
        auto weight = makeWeight(dOut, dIn, caseSeed ^ 0x7a9dc59745b7b3db);
        auto x = makeInput(batchSize, dIn, caseSeed ^ 0xe3e1ecf114d26aa1);

        ComputeAndTransferBenchmark benchmark{
            .macCount = ulong(batchSize) * ulong(dIn) * ulong(dOut),
            .byteCount =
                countBytes(x) + countBytes(weight) + sizeof(bf16) * ulong(batchSize) * ulong(dOut),
        };
        auto reps = std::clamp(uint(1e12 / double(benchmark.macCount)), 20u, 2000u);
        auto copies = std::min(reps, uint((1ull << 30) / double(benchmark.byteCount)));  // ~1 GiB
        auto weights = cloneN(weight, copies);
        auto xs = cloneN(x, copies);
        for (auto i = 0u; i < reps; ++i) {
            auto timer = benchmark.record();
            tensor::matmulT(xs[i % copies], weights[i % copies]);
        }
        benchmark.dump(report[name], {{"batch_size", batchSize}, {"d_in", dIn}, {"d_out", dOut}});
    }
}

REGISTER_BENCHMARK(_tensor_matmulT_bf16)(const benchmarking::Report& report) {
    benchmarkMatmulT(report, randnBf16Tensor, randnBf16Tensor, 0x23f3ac651617c540);
}
REGISTER_BENCHMARK(_tensor_matmulT_int8)(const benchmarking::Report& report) {
    benchmarkMatmulT(report, randnChannelInt8Tensor, randnChannelInt8Tensor, 0xeb4852bba3aeb1d0);
}
REGISTER_BENCHMARK(_tensor_matmulT_s3d8)(const benchmarking::Report& report) {
    benchmarkMatmulT(report, randnChannelInt8Tensor, randnChannelS3D8Tensor, 0x24bec62971dd9ca1);
}

// ### other benchmarks

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
        auto reps = 20u;
        auto copies = std::min(reps, uint((1ull << 30) / double(benchmark.byteCount)));  // ~1 GiB
        auto queries = cloneN(query, copies);
        auto keys = cloneN(key, copies);
        auto values = cloneN(value, copies);
        for (auto i = 0u; i < reps; ++i) {
            auto timer = benchmark.record();
            queries[i % copies] = tensor::attention(std::move(queries[i % copies]),
                                                    keys[i % copies], values[i % copies], causal);
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
    uint copies = 10;

    auto input = tensor::randn({batchSize, dModel}, 1.0f, rng());
    auto wUp = tensor::randn({dFFN, dModel}, 0.02f, rng());
    auto wGate = tensor::randn({dFFN, dModel}, 0.02f, rng());
    auto wDown = tensor::randn({dModel, dFFN}, 0.02f, rng());

    ComputeAndTransferBenchmark benchmark{
        .macCount = 3ull * ulong(batchSize) * ulong(dModel) * ulong(dFFN),
        .byteCount = sizeof(bf16) * (batchSize * (3 * dModel + 5 * dFFN) + (3 * dFFN * dModel)),
    };
    auto inputs = cloneN(input, copies);
    auto wUps = cloneN(wUp, copies);
    auto wGates = cloneN(wGate, copies);
    auto wDowns = cloneN(wDown, copies);
    for (auto i = 0u; i < 100u; ++i) {
        auto timer = benchmark.record();
        auto& x = inputs[i % copies];
        auto up = tensor::matmulT(x, wUps[i % copies]);
        auto gate = tensor::matmulT(x, wGates[i % copies]);
        up = tensor::swiGlu(std::move(up), gate);
        auto outputs = tensor::matmulT(up, wDowns[i % copies]);
    }
    benchmark.dump(report, {{"batch_size", batchSize}, {"d_model", dModel}, {"d_ffn", dFFN}});
}

}  // namespace
