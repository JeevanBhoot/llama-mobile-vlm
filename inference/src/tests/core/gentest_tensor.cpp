#include "core/tensor.hpp"
#include "tests/common.hpp"

#include <fstream>
#define JSON_USE_IMPLICIT_CONVERSIONS 0
#include <json.hpp>

using namespace squash;
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
        auto dtype = info.at("dtype").template get<std::string>();
        auto shape = info.at("shape").template get<Shape>();
        auto offset = info.at("offset").template get<squash::ulong>();
        if (dtype == "float32") {
            return TensorV{
                .data = _data::Flat<float>(reinterpret_cast<float*>(payload.get() + offset)),
                .shape = std::move(shape),
            };
        } else if (dtype == "bfloat16") {
            return TensorV{
                .data = _data::Flat<bf16>(reinterpret_cast<bf16*>(payload.get() + offset)),
                .shape = std::move(shape),
            };
        } else if (dtype == "int8") {
            auto scale = info.at("scale");
            REQUIRE(scale.at("dtype").template get<std::string>() == "bfloat16");
            auto scaleShape = scale.at("shape").template get<Shape>();
            auto scaleOffset = scale.at("offset").template get<squash::ulong>();
            REQUIRE(shape.size() == 2);
            REQUIRE(scaleShape == Shape({shape[0]}));
            return TensorV{
                .data = _data::ChannelInt8(
                    reinterpret_cast<int8_t*>(payload.get() + offset),
                    reinterpret_cast<squash::bf16*>(payload.get() + scaleOffset)),
                .shape = std::move(shape),
            };
        } else {
            std::ostringstream err;
            err << "Unsupported tensor dtype: " << dtype;
            throw std::runtime_error(err.str());
        }
    }

    TensorV tensor_channel_int8(const std::string& name) const {
        auto info = data.at(name);
        REQUIRE(info.at("type").template get<std::string>() == "tensor_channel_int8");
        auto shape = info.at("shape").template get<Shape>();
        auto scaleShape = info.at("scale_shape").template get<Shape>();
        REQUIRE(shape.size() == 2);
        REQUIRE(scaleShape == Shape({shape[0]}));
        return TensorV{
            .data = _data::ChannelInt8(
                reinterpret_cast<int8_t*>(payload.get() +
                                          info.at("offset").template get<squash::ulong>()),
                reinterpret_cast<squash::bf16*>(
                    payload.get() + info.at("scale_offset").template get<squash::ulong>())),
            .shape = std::move(shape),
        };
    }

    Tensor tensor_bf16(const std::string& name) const { return castBf16(tensor(name)); }

    template <typename T>
    T scalar(const std::string& name) const {
        auto info = data.at(name);
        REQUIRE(info.at("type").template get<std::string>() == "scalar");
        return info.at("data").template get<T>();
    }

    template <typename T>
    std::vector<T> list(const std::string& name) const {
        auto info = data.at(name);
        REQUIRE(info.at("type").template get<std::string>() == "list");
        return info.at("data").template get<std::vector<T>>();
    }
};

void runTest(const TestCase& test) {
    const auto DefaultTol = 0.05;
    if (false) {
        // dummy

    } else if (test.op == "cast") {
        REQUIRE_TENSOR_APPROX_EQUALS(castBf16(test.tensor("float")), test.tensor("bf16"), 0.0);

    } else if (test.op == "concat2") {
        auto output =
            concat({test.tensor_bf16("t0"), test.tensor_bf16("t1")}, test.scalar<uint>("dim"));
        REQUIRE_TENSOR_APPROX_EQUALS(output, test.tensor_bf16("output"), DefaultTol);

    } else if (test.op == "tile") {
        auto output = tile(test.tensor_bf16("tensor"), test.list<uint>("reps"));
        REQUIRE_TENSOR_APPROX_EQUALS(output, test.tensor_bf16("output"), DefaultTol);

    } else if (test.op == "add") {
        auto output = add(test.tensor_bf16("x"), test.tensor_bf16("y"));
        REQUIRE_TENSOR_APPROX_EQUALS(output, test.tensor_bf16("output"), DefaultTol);

    } else if (test.op == "broadcastAdd") {
        auto output = broadcastAdd(test.tensor_bf16("x"), test.tensor_bf16("y"));
        REQUIRE_TENSOR_APPROX_EQUALS(output, test.tensor_bf16("output"), DefaultTol);

    } else if (test.op == "gelu") {
        auto output = gelu(test.tensor_bf16("x"));
        REQUIRE_TENSOR_APPROX_EQUALS(output, test.tensor_bf16("output"), DefaultTol);

    } else if (test.op == "swiGlu") {
        auto output = swiGlu(test.tensor_bf16("x"), test.tensor_bf16("gate"));
        REQUIRE_TENSOR_APPROX_EQUALS(output, test.tensor_bf16("output"), DefaultTol);

    } else if (test.op == "rmsNorm") {
        auto output = rmsNorm(test.tensor_bf16("weight"), test.tensor_bf16("x"),
                              test.scalar<float>("epsilon"));
        REQUIRE_TENSOR_APPROX_EQUALS(output, test.tensor_bf16("output"), DefaultTol);

    } else if (test.op == "layerNorm") {
        auto output = layerNorm(test.tensor_bf16("weight"), test.tensor_bf16("bias"),
                                test.tensor_bf16("x"), test.scalar<float>("epsilon"));
        REQUIRE_TENSOR_APPROX_EQUALS(output, test.tensor_bf16("output"), 0.1);  // extreme value

    } else if (test.op == "embeddingLookup") {
        auto output = embeddingLookup(test.tensor("weight"), test.list<uint>("tokens"));
        REQUIRE_TENSOR_APPROX_EQUALS(output, test.tensor("output"), DefaultTol);

    } else if (test.op == "projection") {
        auto output = projection(test.tensor("weight"), test.tensor("x"));
        REQUIRE_TENSOR_APPROX_EQUALS(output, test.tensor("output"), DefaultTol);

    } else if (test.op == "rotate") {
        auto output =
            rotate(test.tensor_bf16("x"), test.list<float>("freq"), test.scalar<uint>("offset"));
        REQUIRE_TENSOR_APPROX_EQUALS(output, test.tensor_bf16("output"), DefaultTol);

    } else if (test.op == "attention") {
        auto output = attention(test.tensor_bf16("query"), test.tensor_bf16("key"),
                                test.tensor_bf16("value"), test.scalar<bool>("causal"));
        REQUIRE_TENSOR_APPROX_EQUALS(output, test.tensor_bf16("output"), DefaultTol);

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
