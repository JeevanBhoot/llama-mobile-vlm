#pragma once

#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <variant>
#include <vector>

#include "lib/tokenizer.hpp"

namespace squash {

/// Common ///

using uint = uint32_t;
using ulong = uint64_t;
using bf16 = int16_t;

constexpr ulong DefaultAlignment = 32u;

float bf16ToFloat(bf16 value);
uint prod(const std::vector<uint>&);

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

/// Tensor ///

struct Buffer {
    Buffer(ulong size, ulong alignment = DefaultAlignment);
    Buffer(Buffer&&);
    Buffer& operator=(Buffer&&);
    ~Buffer();
    void reset();
    template <class T = char>
    T* get() const;
    Buffer copy(ulong size, ulong alignment = DefaultAlignment) const;

   private:
    char* _data;
};

namespace tensor_data {
template <class T>
struct Flat {
    T* data;
    Flat(T* data = nullptr) : data(data) {}
};
}  // namespace tensor_data

// TensorV is a non-owning Tensor view
struct TensorV {
    std::variant<tensor_data::Flat<bf16>, tensor_data::Flat<float>> data;
    std::vector<uint> shape;
};

struct Tensor : TensorV {
    Buffer _data;
};

std::ostream& operator<<(std::ostream&, const TensorV&);
void saveNpy(std::ostream&, const TensorV&);

/// Model ///
// Holds all shape and parameter data (views onto an underlying buffer)

struct TextModel {
    struct AttentionLayer {
        TensorV norm;
        TensorV query;
        TensorV key;
        TensorV value;
        TensorV output;
    };
    struct MLPLayer {
        TensorV norm;
        TensorV up;
        TensorV gate;
        TensorV down;
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
    TensorV embedTokens;
    std::vector<Layer> layers;
    TensorV finalNorm;
    TensorV predictTokens;

    // Vocab
    Tokenizer tokenizer;
    uint beginOfTextID;
    uint endOfTextID;
};

struct VisionModel {
    std::vector<float> imageMean;
    std::vector<float> imageStd;
    uint dImage;
    uint dPatch;
};

struct Model {
    TextModel textModel;
    std::optional<VisionModel> visionModel;

    // Metadata & data buffer
    std::string source;
    std::string created;
    ulong alignment;
    Buffer _data;
};

Model loadSquashedTensors(std::istream&);
namespace impl {
std::string regexUnicodeToModifiedECMA(const std::string&);
}  // namespace impl

/// Generator ///

// A Generator holds a KV cache and executes batch=1 inference
// It references Model, which must outlive it
struct Generator {
    struct KVCache {
        struct Entry {
            Tensor key;
            Tensor value;
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
    Options options;
    uint prevToken;

    explicit Generator(Model&);
    std::vector<std::string> prefill(const std::string& prefix,
                                     const std::optional<Image>& image,
                                     const Options& options);
    // Returns an empty token for endOfText or reaching maxGeneratedTokens
    std::string generate();
};

}  // namespace squash

#include "squash_impl.ipp"
