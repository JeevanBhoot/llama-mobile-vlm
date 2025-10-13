#pragma once

#include "core/tensor.hpp"
#include "lib/squash.hpp"

#include <catch_amalgamated.hpp>

#define REQUIRE_TENSOR_APPROX_EQUALS(actual, expected, rmsTol) \
    squash::assertTensorApproxEquals(actual, expected, rmsTol, __FILE__, __LINE__)

///////////////////////////////////////////////////////////////////////////////
// Impl

namespace squash {

inline void assertTensorApproxEquals(const tensor::TensorV& actual,
                                     const tensor::TensorV& expected,
                                     double rmsTol,
                                     const std::string& file,
                                     const ulong line) {
    using namespace squash::tensor;
    if (actual.shape != expected.shape) {
        std::ostringstream err;
        err << file << ":" << line << " tensor shape mismatch, expected: " << expected.shape
            << ", actual: " << actual.shape << "\n";
        throw std::runtime_error(err.str());
    }
    auto pActual = data<float>(actual);
    auto pExpected = data<float>(expected);
    auto nElement = prod(actual.shape);

    // Calculate RMS
    auto rms = 0.0;
    for (auto i = 0u; i < nElement; ++i) {
        rms += static_cast<double>(pActual[i] * pActual[i]);
    }
    rms = std::sqrt(rms / nElement);

    // Find indices where the actual differs from expected by more than rmsTol * rms
    std::vector<ulong> badIndices;
    for (auto i = 0u; i < nElement; ++i) {
        auto error = std::abs(static_cast<double>(pActual[i] - pExpected[i])) / rms;
        if (error > rmsTol * rms) {
            badIndices.push_back(i);
        }
    }

    // Report an error
    if (badIndices.size()) {
        std::ostringstream err;
        err << file << ":" << line << " actual != expected, " << badIndices.size() << "/"
            << nElement << " elements differ by more than " << rmsTol << " * (RMS=" << rms << ")\n";
        auto nPrint = std::min<ulong>(10, badIndices.size());
        err << "First " << nPrint << " bad indices:\n";
        for (auto i = 0u; i < nPrint; ++i) {
            auto idx = badIndices[i];
            auto error = std::abs(static_cast<double>(pActual[idx] - pExpected[idx])) / rms;
            err << "  [" << idx << "] expected: " << pExpected[idx] << ",  actual: " << pActual[idx]
                << ",  error/RMS: " << error << "\n";
        }
        throw std::runtime_error(err.str());
    }
}

}  // namespace squash