#include <catch_amalgamated.hpp>
#include <iostream>
#include <random>

#include "squash.hpp"

using namespace squash;

TEST_CASE("squash::bf16ToFloat", "[squash]") {
    REQUIRE(bf16ToFloat(0x4147) == 12.4375f);
    REQUIRE(bf16ToFloat(-0x3f80) == -4.0f);
}

namespace {
std::vector<float> ropeAngularFrequency(uint dHead) {
    std::vector<float> freq;
    for (auto i = 0u; i < dHead; i += 2) {
        freq.push_back(std::pow(1e4f, -float(i) / float(dHead)));
    }
    return freq;
}
inline bf16 floatToBf16_truncate(float value) {
    union {
        float f;
        int16_t i[2];
    } u;
    u.f = value;
    return u.i[1];
}
}  // namespace

TEST_CASE("squash::Generator", "[squash]") {
    constexpr auto alignment = 32u;
    Model m{
        // Metadata
        .source = "test",
        .created = "",
        .alignment = alignment,
        // Config
        .dLayers = 3,
        .dVocab = 256,
        .dModel = 128,
        .dMLP = 512,
        .dAttentionHead = 64,
        .dAttentionQ = 2,
        .dAttentionKV = 2,
        .dSequenceMax = 1024,
        .normEpsilon = 1e-5f,
        .ropeAngularFrequency = ropeAngularFrequency(64),
        // Parameters
        .embedTokens = {},
        .layers = {},
        .finalNorm = {},
        // Data
        ._data = Buffer(0, alignment),
    };

    // Create buffer
    auto totalHeads = m.dAttentionKV * m.dAttentionQ;
    auto nParameters = m.dVocab * m.dModel                                                // embed
                       + m.dLayers * (m.dModel                                            // norm
                                      + 2 * m.dModel * totalHeads * m.dAttentionHead      // q+o
                                      + 2 * m.dModel * m.dAttentionKV * m.dAttentionHead  // k+v
                                      + m.dModel                                          // norm
                                      + 3 * m.dModel * m.dMLP                             // mlp
                                      )                                                   //
                       + m.dModel;                                                        // norm
    m._data = Buffer(sizeof(bf16) * nParameters, alignment);
    auto buffer = reinterpret_cast<bf16*>(m._data.get());
    std::default_random_engine rng(12345u);
    for (auto i = 0u; i < nParameters; ++i) {
        buffer[i] = floatToBf16_truncate(std::normal_distribution<float>(0, 0.02f)(rng));
    }

    // Create tensor views
    auto ptr = buffer;
    auto allocate = [&](const std::vector<uint>& shape) {
        auto t = TensorV{ptr, shape};
        ptr += prod(shape);
        return t;
    };
    m.embedTokens = allocate({m.dVocab, m.dModel});
    for (auto n = 0u; n < m.dLayers; ++n) {
        m.layers.push_back(
            {{
                 .norm = allocate({m.dModel}),                                      //
                 .query = allocate({totalHeads * m.dAttentionHead, m.dModel}),      //
                 .key = allocate({m.dAttentionKV * m.dAttentionHead, m.dModel}),    //
                 .value = allocate({m.dAttentionKV * m.dAttentionHead, m.dModel}),  //
                 .output = allocate({m.dModel, totalHeads * m.dAttentionHead}),     //
             },
             {
                 .norm = allocate({m.dModel}),          //
                 .up = allocate({m.dMLP, m.dModel}),    //
                 .gate = allocate({m.dMLP, m.dModel}),  //
                 .down = allocate({m.dModel, m.dMLP}),  //
             }});
    }
    m.finalNorm = allocate({m.dModel});
    if (ptr - buffer != nParameters) {
        throw std::logic_error("Wrong number of parameters");
    }

    // Generate from the model
    auto generationCount = 10u;
    Generator generator(m);
    std::vector<uint> tokens;
    tokens.push_back(generator.prefill({10, 20, 30}, generationCount));
    for (auto i = 0u; i < generationCount; ++i) {
        tokens.push_back(generator.generate());
    }
    std::copy(tokens.begin(), tokens.end(), std::ostream_iterator<uint>(std::cerr, ", "));
    std::cerr << "\n";
}
