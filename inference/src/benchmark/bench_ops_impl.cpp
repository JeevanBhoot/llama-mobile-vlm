// Copyright (c) 2026 Graphcore Ltd. All rights reserved.

#include <omp.h>
#include <functional>
#include <thread>

#include "benchmark/benchmark.hpp"

using namespace squash;

namespace {

REGISTER_BENCHMARK(_loop)(const benchmarking::Report& report) {
    uint reps = 1000000u;
    benchmarking::Benchmark benchmark;
    for (auto rep = 0u; rep < reps; ++rep) {
        benchmark.record();
    }
    auto result = benchmark.result();
    report({
        {"reps", reps},
        {"time_ms", 1e3 * result.mean},
        {"time", benchmark.times},
    });
    std::cerr << report << 1e6 * result << " us\n";
}

///////////////////////////////////////////////////////////////////////////////
// Memory bandwidth

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
    std::vector<ulong> elementsRange;
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

///////////////////////////////////////////////////////////////////////////////
// Dot product

#if defined(__clang__) && (defined(__x86_64__) || defined(__i386__))
#define SQUASH_BENCH_HAS_FLOAT_CONTROL 1
#else
#define SQUASH_BENCH_HAS_FLOAT_CONTROL 0
#endif

template <uint M, uint K, uint N, bool TransposeB, bool FastMath>
__attribute__((noinline)) void dotProductImpl(const float* __restrict__ a,
                                              const float* __restrict__ b,
                                              float* __restrict__ out) {
    static_assert(!FastMath || SQUASH_BENCH_HAS_FLOAT_CONTROL,
                  "FastMath requires a supported float_control pragma");
    for (auto iM = 0u; iM < M; ++iM) {
        // #pragma unroll
        for (auto iN = 0u; iN < N; ++iN) {
            auto sum = 0.0f;
            for (auto iK = 0u; iK < K; ++iK) {
                auto aIdx = iM * K + iK;
                auto bIdx = TransposeB ? (iN * K + iK) : (iK * N + iN);
                if (FastMath) {
#if SQUASH_BENCH_HAS_FLOAT_CONTROL
#pragma float_control(precise, off)
#endif
                    sum += a[aIdx] * b[bIdx];
                } else {
                    sum += a[aIdx] * b[bIdx];
                }
            }
            out[iM * N + iN] = sum;
        }
    }
}

template <uint M, uint K, uint N, bool TransposeB, bool FastMath>
void runDotProduct(const benchmarking::Report& report) {
    auto outerReps = 100u;
    auto innerReps = 1000u;

    benchmarking::Benchmark benchmark;
    std::vector<float> a(M * K, 1.0f), b(K * N, 2.0f), out(M * N, 0.0f);
    for (auto rep = 0u; rep < outerReps; ++rep) {
        auto timer = benchmark.record();
        for (auto i = 0u; i < innerReps; ++i) {
            dotProductImpl<M, K, N, TransposeB, FastMath>(a.data(), b.data(), out.data());
        }
    }

    auto result = benchmark.result();
    auto macCount = innerReps * M * K * N;
    report({
        {"m", M},
        {"k", K},
        {"n", N},
        {"transpose_b", TransposeB},
        {"fast_math", FastMath},
        {"inner_reps", innerReps},
        {"time_ms", 1e3 * result.mean},
        {"mac_count", macCount},
        {"gmac_s", static_cast<double>(macCount) / 1e9 / result.mean},
        {"time", benchmark.times},
    });

    std::cerr << report << (TransposeB ? "B.T" : "   ") << " " << (FastMath ? "fast   " : "default")
              << " : " << static_cast<double>(macCount) / 1e9 / result
              << " GMAC/s for (m, k, n) = (" << M << ", " << K << ", " << N << ") in "
              << 1e3 * result << " ms\n";
}

REGISTER_BENCHMARK(_dot_product_small)(const benchmarking::Report& report) {
    runDotProduct<16, 16, 16, false, false>(report);
#if SQUASH_BENCH_HAS_FLOAT_CONTROL
    runDotProduct<16, 16, 16, false, true>(report);
#endif
    runDotProduct<16, 16, 16, true, false>(report);
#if SQUASH_BENCH_HAS_FLOAT_CONTROL
    runDotProduct<16, 16, 16, true, true>(report);
#endif
}

#undef SQUASH_BENCH_HAS_FLOAT_CONTROL

///////////////////////////////////////////////////////////////////////////////
// Dot product instructions

#ifdef __ARM_NEON

REGISTER_BENCHMARK(_dot_inst_throughput)(const benchmarking::Report& report) {
    auto outerReps = 100u;
    auto innerReps = 1u << 18;

    auto maxThreads = std::thread::hardware_concurrency();
    std::vector<uint> nthreadsRange;
    for (auto t = 1u; t < maxThreads; t *= 4u) {
        nthreadsRange.push_back(t);
    }
    nthreadsRange.push_back(maxThreads);

    struct OpTest {
        std::string instruction;
        uint macsPerLoop;
        std::function<void()> fn;
    };

    std::vector<OpTest> tests;
    tests.push_back({"fmla",
                     16 * 4,  // #instructions * 4-wide fmla (4*1x1x1)
                     [innerReps]() {
                         for (auto i = 0u; i < innerReps; ++i) {
                             asm volatile(
                                 "fmla v0.4s, v30.4s, v31.4s\n"
                                 "fmla v1.4s, v30.4s, v31.4s\n"
                                 "fmla v2.4s, v30.4s, v31.4s\n"
                                 "fmla v3.4s, v30.4s, v31.4s\n"
                                 //
                                 "fmla v4.4s, v30.4s, v31.4s\n"
                                 "fmla v5.4s, v30.4s, v31.4s\n"
                                 "fmla v6.4s, v30.4s, v31.4s\n"
                                 "fmla v7.4s, v30.4s, v31.4s\n"
                                 //
                                 "fmla v8.4s, v30.4s, v31.4s\n"
                                 "fmla v9.4s, v30.4s, v31.4s\n"
                                 "fmla v10.4s, v30.4s, v31.4s\n"
                                 "fmla v11.4s, v30.4s, v31.4s\n"
                                 //
                                 "fmla v12.4s, v30.4s, v31.4s\n"
                                 "fmla v13.4s, v30.4s, v31.4s\n"
                                 "fmla v14.4s, v30.4s, v31.4s\n"
                                 "fmla v15.4s, v30.4s, v31.4s\n"
                                 //
                                 :
                                 :
                                 : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
                                   "v10", "v11", "v12", "v13", "v14", "v15", "v30", "v31");
                         }
                     }});

#if defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)
    tests.push_back({"bfdot",
                     16 * 8,  // #instructions * 16 MACs per bfdot (4*1x2x1)
                     [innerReps]() {
                         for (auto i = 0u; i < innerReps; ++i) {
                             asm volatile(
                                 "bfdot v0.4s, v30.8h, v31.8h\n"
                                 "bfdot v1.4s, v30.8h, v31.8h\n"
                                 "bfdot v2.4s, v30.8h, v31.8h\n"
                                 "bfdot v3.4s, v30.8h, v31.8h\n"
                                 //
                                 "bfdot v4.4s, v30.8h, v31.8h\n"
                                 "bfdot v5.4s, v30.8h, v31.8h\n"
                                 "bfdot v6.4s, v30.8h, v31.8h\n"
                                 "bfdot v7.4s, v30.8h, v31.8h\n"
                                 //
                                 "bfdot v8.4s, v30.8h, v31.8h\n"
                                 "bfdot v9.4s, v30.8h, v31.8h\n"
                                 "bfdot v10.4s, v30.8h, v31.8h\n"
                                 "bfdot v11.4s, v30.8h, v31.8h\n"
                                 //
                                 "bfdot v12.4s, v30.8h, v31.8h\n"
                                 "bfdot v13.4s, v30.8h, v31.8h\n"
                                 "bfdot v14.4s, v30.8h, v31.8h\n"
                                 "bfdot v15.4s, v30.8h, v31.8h\n"
                                 :
                                 :
                                 : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
                                   "v10", "v11", "v12", "v13", "v14", "v15", "v30", "v31");
                         }
                     }});

    tests.push_back({"bfmmla",
                     16 * 16,  // #instructions * 16 MACs per bfmmla (1*2x4x2)
                     [innerReps]() {
                         for (auto i = 0u; i < innerReps; ++i) {
                             asm volatile(
                                 "bfmmla v0.4s, v30.8h, v31.8h\n"
                                 "bfmmla v1.4s, v30.8h, v31.8h\n"
                                 "bfmmla v2.4s, v30.8h, v31.8h\n"
                                 "bfmmla v3.4s, v30.8h, v31.8h\n"
                                 //
                                 "bfmmla v4.4s, v30.8h, v31.8h\n"
                                 "bfmmla v5.4s, v30.8h, v31.8h\n"
                                 "bfmmla v6.4s, v30.8h, v31.8h\n"
                                 "bfmmla v7.4s, v30.8h, v31.8h\n"
                                 //
                                 "bfmmla v8.4s, v30.8h, v31.8h\n"
                                 "bfmmla v9.4s, v30.8h, v31.8h\n"
                                 "bfmmla v10.4s, v30.8h, v31.8h\n"
                                 "bfmmla v11.4s, v30.8h, v31.8h\n"
                                 //
                                 "bfmmla v12.4s, v30.8h, v31.8h\n"
                                 "bfmmla v13.4s, v30.8h, v31.8h\n"
                                 "bfmmla v14.4s, v30.8h, v31.8h\n"
                                 "bfmmla v15.4s, v30.8h, v31.8h\n"
                                 :
                                 :
                                 : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
                                   "v10", "v11", "v12", "v13", "v14", "v15", "v30", "v31");
                         }
                     }});
#endif

#if defined(__ARM_FEATURE_DOTPROD)
    tests.push_back({"sdot",
                     16 * 16,  // #instructions * 16 MACs per sdot (4*1x4x1)
                     [innerReps]() {
                         for (auto i = 0u; i < innerReps; ++i) {
                             asm volatile(
                                 "sdot v0.4s, v30.16b, v31.16b\n"
                                 "sdot v1.4s, v30.16b, v31.16b\n"
                                 "sdot v2.4s, v30.16b, v31.16b\n"
                                 "sdot v3.4s, v30.16b, v31.16b\n"
                                 //
                                 "sdot v4.4s, v30.16b, v31.16b\n"
                                 "sdot v5.4s, v30.16b, v31.16b\n"
                                 "sdot v6.4s, v30.16b, v31.16b\n"
                                 "sdot v7.4s, v30.16b, v31.16b\n"
                                 //
                                 "sdot v8.4s, v30.16b, v31.16b\n"
                                 "sdot v9.4s, v30.16b, v31.16b\n"
                                 "sdot v10.4s, v30.16b, v31.16b\n"
                                 "sdot v11.4s, v30.16b, v31.16b\n"
                                 //
                                 "sdot v12.4s, v30.16b, v31.16b\n"
                                 "sdot v13.4s, v30.16b, v31.16b\n"
                                 "sdot v14.4s, v30.16b, v31.16b\n"
                                 "sdot v15.4s, v30.16b, v31.16b\n"
                                 :
                                 :
                                 : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
                                   "v10", "v11", "v12", "v13", "v14", "v15", "v30", "v31");
                         }
                     }});
#endif

#if defined(__ARM_FEATURE_MATMUL_INT8)
    tests.push_back({"smmla",
                     16 * 32,  // #instructions * 32 MACs per smmla (1*2x8x2)
                     [innerReps]() {
                         for (auto i = 0u; i < innerReps; ++i) {
                             asm volatile(
                                 "smmla v0.4s, v30.16b, v31.16b\n"
                                 "smmla v1.4s, v30.16b, v31.16b\n"
                                 "smmla v2.4s, v30.16b, v31.16b\n"
                                 "smmla v3.4s, v30.16b, v31.16b\n"
                                 //
                                 "smmla v4.4s, v30.16b, v31.16b\n"
                                 "smmla v5.4s, v30.16b, v31.16b\n"
                                 "smmla v6.4s, v30.16b, v31.16b\n"
                                 "smmla v7.4s, v30.16b, v31.16b\n"
                                 //
                                 "smmla v8.4s, v30.16b, v31.16b\n"
                                 "smmla v9.4s, v30.16b, v31.16b\n"
                                 "smmla v10.4s, v30.16b, v31.16b\n"
                                 "smmla v11.4s, v30.16b, v31.16b\n"
                                 //
                                 "smmla v12.4s, v30.16b, v31.16b\n"
                                 "smmla v13.4s, v30.16b, v31.16b\n"
                                 "smmla v14.4s, v30.16b, v31.16b\n"
                                 "smmla v15.4s, v30.16b, v31.16b\n"
                                 :
                                 :
                                 : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
                                   "v10", "v11", "v12", "v13", "v14", "v15", "v30", "v31");
                         }
                     }});
#endif

    for (auto& test : tests) {
        for (auto threads : nthreadsRange) {
            benchmarking::Benchmark benchmark;
            for (auto rep = 0u; rep < outerReps; ++rep) {
                auto timer = benchmark.record();
#pragma omp parallel for num_threads(threads) schedule(static)
                for (auto i = 0u; i < threads; ++i) {
                    test.fn();
                }
            }

            auto result = benchmark.result();
            auto macs = threads * innerReps * static_cast<ulong>(test.macsPerLoop);
            report({
                {"instruction", test.instruction},
                {"threads", threads},
                {"inner_reps", innerReps},
                {"time_ms", 1e3 * result.mean},
                {"mac_count", macs},
                {"gmac_s", static_cast<double>(macs) / 1e9 / result.mean},
                {"time", benchmark.times},
            });
            std::cerr << report << test.instruction << "  "
                      << static_cast<double>(macs) / 1e9 / result << " GMAC/s, with " << threads
                      << " threads\n";
        }
    }
}

REGISTER_BENCHMARK(_tbl_inst_throughput)(const benchmarking::Report& report) {
    auto outerReps = 20u;
    auto innerReps = 1u << 22;
    auto lookupsPerLoop = 16u * 16u;  // #instructions * 16 byte lookups per TBL.

    auto maxThreads = std::thread::hardware_concurrency();
    std::vector<uint> nthreadsRange;
    for (auto t = 1u; t < maxThreads; t *= 4u) {
        nthreadsRange.push_back(t);
    }
    nthreadsRange.push_back(maxThreads);

    for (auto threads : nthreadsRange) {
        benchmarking::Benchmark benchmark;
        for (auto rep = 0u; rep < outerReps; ++rep) {
            auto timer = benchmark.record();
#pragma omp parallel for num_threads(threads) schedule(static)
            for (auto i = 0u; i < threads; ++i) {
                for (auto j = 0u; j < innerReps; ++j) {
                    asm volatile(
                        "tbl v0.16b, {v16.16b, v17.16b, v18.16b, v19.16b}, v20.16b\n"
                        "tbl v1.16b, {v16.16b, v17.16b, v18.16b, v19.16b}, v21.16b\n"
                        "tbl v2.16b, {v16.16b, v17.16b, v18.16b, v19.16b}, v22.16b\n"
                        "tbl v3.16b, {v16.16b, v17.16b, v18.16b, v19.16b}, v23.16b\n"
                        //
                        "tbl v4.16b, {v16.16b, v17.16b, v18.16b, v19.16b}, v20.16b\n"
                        "tbl v5.16b, {v16.16b, v17.16b, v18.16b, v19.16b}, v21.16b\n"
                        "tbl v6.16b, {v16.16b, v17.16b, v18.16b, v19.16b}, v22.16b\n"
                        "tbl v7.16b, {v16.16b, v17.16b, v18.16b, v19.16b}, v23.16b\n"
                        //
                        "tbl v8.16b, {v16.16b, v17.16b, v18.16b, v19.16b}, v20.16b\n"
                        "tbl v9.16b, {v16.16b, v17.16b, v18.16b, v19.16b}, v21.16b\n"
                        "tbl v10.16b, {v16.16b, v17.16b, v18.16b, v19.16b}, v22.16b\n"
                        "tbl v11.16b, {v16.16b, v17.16b, v18.16b, v19.16b}, v23.16b\n"
                        //
                        "tbl v12.16b, {v16.16b, v17.16b, v18.16b, v19.16b}, v20.16b\n"
                        "tbl v13.16b, {v16.16b, v17.16b, v18.16b, v19.16b}, v21.16b\n"
                        "tbl v14.16b, {v16.16b, v17.16b, v18.16b, v19.16b}, v22.16b\n"
                        "tbl v15.16b, {v16.16b, v17.16b, v18.16b, v19.16b}, v23.16b\n"
                        :
                        :
                        : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
                          "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17",
                          "v18", "v19", "v20", "v21", "v22", "v23");
                }
            }
        }

        auto result = benchmark.result();
        auto lookups = threads * innerReps * static_cast<ulong>(lookupsPerLoop);
        report({
            {"instruction", "tbl_4reg"},
            {"threads", threads},
            {"inner_reps", innerReps},
            {"time_ms", 1e3 * result.mean},
            {"lookup_count", lookups},
            {"glookup_s", static_cast<double>(lookups) / 1e9 / result.mean},
            {"time", benchmark.times},
        });
        std::cerr << report << "tbl_4reg  " << static_cast<double>(lookups) / 1e9 / result
                  << " Glookup/s, with " << threads << " threads\n";
    }
}

#endif  // __ARM_NEON

}  // namespace
