// Copyright (c) 2026 Graphcore Ltd. All rights reserved.

#pragma once

#include "lib/squash.hpp"

namespace squash {

struct Dummy {
    struct TextConfig {
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
    struct VisionConfig {
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
    struct Config {
        ulong seed;
        TextConfig text;
        std::optional<VisionConfig> vision;
    };

    static Config smallLLM();
    static Config smallVLM();
    static Config llama1B_LLM();
    static Config llama11B_VLM();

    static Model createModel(const Config& config);
    static Image createImage(uint dImage, ulong seed);
};

}  // namespace squash
