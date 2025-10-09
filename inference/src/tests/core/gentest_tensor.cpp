#define JSON_USE_IMPLICIT_CONVERSIONS 0

#include "core/tensor.hpp"
#include "tests/tests.hpp"

#include <fstream>
#include <json.hpp>

using namespace squash::tensor;
using json = nlohmann::json;

namespace {

struct TestCase {
    std::string op;
    json data;
    Buffer& payload;

    TensorV tensor(const std::string& name) const {
        auto info = data.at(name);
        REQUIRE(info.at("type").template get<std::string>() == "tensor");
        return TensorV{
            .data = _data::Flat<float>(
                reinterpret_cast<float*>(payload.get() + info.at("offset").template get<ulong>())),
            .shape = info.at("shape").template get<Shape>(),
        };
    }
};

void assertTensorApproxEquals(const TensorV& actual,
                              const TensorV& expected,
                              double rmsTol,
                              const std::string& file,
                              const ulong line) {
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
#define REQUIRE_TENSOR_APPROX_EQUALS(actual, expected, rmsTol) \
    assertTensorApproxEquals(actual, expected, rmsTol, __FILE__, __LINE__)

void runTest(const TestCase& test) {
    if (test.op == "projection") {
        auto output = projection(castBf16(test.tensor("weight")), test.tensor("x"));
        REQUIRE_TENSOR_APPROX_EQUALS(output, test.tensor("output"), 0.01);

    } else {
        FAIL("Unknown op: " + test.op);
    }
}

}  // namespace

TEST_CASE("squash::tensor::generated") {
    const char* dataPath = std::getenv("TESTDATA");
    REQUIRE(dataPath != nullptr);
    std::ifstream in(dataPath, std::ios::binary);

    // Read data file into `meta` and `payload`
    std::string magic(8, '\0');
    in.read(magic.data(), 8);
    REQUIRE(magic == "TESTDATA");
    uint64_t metaBytes;
    in.read(reinterpret_cast<char*>(&metaBytes), sizeof(metaBytes));
    std::string metaStr(metaBytes, '\0');
    in.read(metaStr.data(), static_cast<std::streamsize>(metaBytes));
    auto meta = json::parse(metaStr);
    uint64_t payloadBytes;
    in.read(reinterpret_cast<char*>(&payloadBytes), sizeof(payloadBytes));
    Buffer payload(payloadBytes);
    in.read(payload.get(), static_cast<std::streamsize>(payloadBytes));
    REQUIRE(in.good());
    in.get();
    REQUIRE(in.eof());

    // Run tests
    for (const auto& test : meta.at("tests")) {
        auto op = test.at("op").template get<std::string>();
        auto name = test.at("name").template get<std::string>();
        INFO("test: " + op + "::" + name);
        runTest(TestCase{op, test.at("data"), payload});
    }
}
