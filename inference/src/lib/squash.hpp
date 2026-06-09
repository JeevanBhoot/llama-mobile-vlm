// Copyright (c) 2026 Graphcore Ltd. All rights reserved.

#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
#include <iostream>
#include <optional>
#include <random>
#include <vector>

#include "core/common.hpp"
#include "core/tensor.hpp"
#include "core/tokenizer.hpp"

namespace squash {

/// Common ///

struct Timer {
    typedef std::chrono::high_resolution_clock clock;
    clock::time_point start;
    Timer();
    double elapsed() const;
};

template <class T>
struct Dump {
    const T& sequence;
};
template <class T>
Dump<T> dump(const T&);
template <class T>
std::ostream& operator<<(std::ostream&, const Dump<T>&);

struct Image {
    uint height;
    uint width;
    std::vector<uint8_t> data;  // [y*(width*3) + x*3 + c]

    Image(uint height, uint width, std::vector<uint8_t>&& data);
    Image(const Image&) = delete;
    Image& operator=(const Image&) = delete;
    Image(Image&&) = default;
};
Image loadImage(const std::string&);
Image resizeImage(const Image&, uint height, uint width);

void selectOmpNumThreads();

using ProgressCallback = std::function<void(double)>;

/// Model ///
// Holds all shape and parameter data (views onto an underlying buffer)

struct TextModel {
    struct AttentionLayer {
        tensor::TensorV norm;
        tensor::TensorV query;
        tensor::TensorV key;
        tensor::TensorV value;
        tensor::TensorV output;
        std::optional<tensor::TensorV> query_norm;
        std::optional<tensor::TensorV> key_norm;
    };
    struct MLPLayer {
        tensor::TensorV norm;
        tensor::TensorV up;
        tensor::TensorV gate;
        tensor::TensorV down;
    };
    struct Layer {
        AttentionLayer attention;
        MLPLayer mlp;
    };

    // Config
    uint dLayers;
    uint dVocab;
    uint dModel;
    uint dMLP;
    uint dAttentionHead;
    uint dAttentionQ;
    uint dAttentionKV;
    uint dSequenceMax;
    float normEpsilon;
    std::vector<float> ropeAngularFrequency;
    bool tiedEmbeddings;
    std::vector<uint> crossAttentionLayers;

    // Parameters
    tensor::TensorV embedTokens;
    std::vector<Layer> layers;
    tensor::TensorV finalNorm;
    tensor::TensorV predictTokens;

    // Vocab
    Tokenizer tokenizer;
    uint beginOfTextID;
    uint endOfTextID;
    std::optional<uint> imageID;
};

struct VisionModel {
    struct Affine {
        tensor::TensorV weight;
        tensor::TensorV bias;
    };
    struct AttentionLayer {
        Affine norm;
        tensor::TensorV query;
        tensor::TensorV key;
        tensor::TensorV value;
        tensor::TensorV output;
    };
    struct MLPLayer {
        Affine norm;
        Affine up;
        Affine down;
    };
    struct Layer {
        AttentionLayer attention;
        MLPLayer mlp;
    };

    // Config
    std::vector<float> imageMean;
    std::vector<float> imageStd;
    uint dImage;
    uint dPatch;
    uint dLayers0;
    uint dLayers1;
    uint dModel;
    uint dMlp;
    uint dAttentionHead;
    uint dAttentionQkv;
    float normEpsilon;
    std::vector<uint> outputTaps;

    // Parameters
    tensor::TensorV patchEmbedding;
    tensor::TensorV positionalEmbedding;
    tensor::TensorV classEmbedding;
    Affine layerNormPre;
    std::vector<Layer> layers0;
    Affine layerNormPost;
    tensor::TensorV tileEmbeddingPost;
    std::vector<Layer> layers1;
    Affine multiModalProjector;
};

struct Model {
    TextModel textModel;
    std::optional<VisionModel> visionModel;

    // Metadata & data buffer
    std::string source;
    std::string created;
    ulong alignment;
    tensor::Buffer _data;
};

Model loadSquashedTensors(std::istream&, const ProgressCallback& progress = {});
namespace impl {
std::string regexUnicodeToModifiedECMA(const std::string&);
}  // namespace impl

/// Generator ///

// A Generator holds a KV cache and executes batch=1 inference
// It references Model, which must outlive it
struct Generator {
    struct KVCache {
        struct Entry {
            tensor::Tensor key;
            tensor::Tensor value;
        };
        std::vector<Entry> entries;
        uint dSequence;
        uint dSequenceMax;
    };

    struct Options {
        uint maxGeneratedTokens;
        std::optional<uint> seed;

        // Sample according to logits, with temperature (0 = greedy), and
        // additional shaping - only sample the top min(topK, n_topP) tokens
        float temperature;
        uint topK;
        float topP;

        static Options greedy(uint maxGeneratedTokens);
    };

    Model& model;
    std::default_random_engine rng;
    KVCache kvCache;
    std::optional<KVCache> crossAttentionCache;
    Options options;
    uint prevToken;

    explicit Generator(Model&);
    std::vector<std::string> prefill(const std::string& prefix,
                                     const std::optional<Image>& image,
                                     const Options& options,
                                     const ProgressCallback& progress = {});
    // Returns an empty token for endOfText or reaching maxGeneratedTokens
    std::string generate();
};

}  // namespace squash

///////////////////////////////////////////////////////////////////////////////
// Impl

#define DUMPSQ(obj) std::cerr << __FILE__ << ":" << __LINE__ << " " << obj << std::endl;

namespace squash {

template <class T>
Dump<T> dump(const T& sequence) {
    return {sequence};
}

template <class T>
std::ostream& operator<<(std::ostream& out, const Dump<T>& x) {
    out << "{";
    for (auto i = 0u; i < x.sequence.size(); ++i) {
        if (i) out << ", ";
        out << x.sequence[i];
    }
    return out << "}";
}

}  // namespace squash
