#pragma once

#include <catch_amalgamated.hpp>
#include "lib/squash.hpp"

namespace squash {

template <class To, class From>
To cast(From value);

template <>
inline float cast<float, float>(float value) {
    return value;
}
template <>
inline bf16 cast<bf16, float>(float value) {
    return floatToBf16(value);
}
template <>
inline float cast<float, bf16>(bf16 value) {
    return bf16ToFloat(value);
}

struct TestModelConfig {
    struct Text {
        uint dLayers;
        uint dVocab;
        uint dModel;
        uint dMLP;
        uint dAttentionHead;
        uint dAttentionQ;
        uint dAttentionKV;
        bool tiedEmbeddings;
        std::vector<uint> crossAttentionLayers;
    };
    struct Vision {
        uint dImage;
        uint dPatch;
        uint dLayers0;
        uint dLayers1;
        uint dModel;
        uint dMlp;
        uint dAttentionHead;
        uint dAttentionQkv;
        std::vector<uint> outputTaps;
    };
    Text text;
    std::optional<Vision> vision;

    static TestModelConfig smallLLM();
    static TestModelConfig smallVLM();
    static TestModelConfig llama1B_LLM();
    static TestModelConfig llama11B_VLM();
};

Model createTestModel(const TestModelConfig& config);

}  // namespace squash
