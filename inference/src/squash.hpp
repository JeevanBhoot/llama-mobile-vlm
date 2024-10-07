#pragma once

#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <numeric>
#include <variant>
#include <vector>

namespace squash {

/// Common ///

using uint = uint32_t;
using ulong = uint64_t;
using bf16 = int16_t;

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

/// Tensor ///

struct Buffer {
    Buffer(ulong size, ulong alignment);
    Buffer(Buffer&&);
    Buffer& operator=(Buffer&&);
    ~Buffer();
    char* get() const;
    void reset();

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

/// Model ///

// The model holds all shape and parameter data (views onto an underlying buffer)
struct Model {
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

    // Metadata
    std::string source;
    std::string created;
    ulong alignment;

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

    // Parameters
    TensorV embedTokens;
    std::vector<Layer> layers;
    TensorV finalNorm;

    // Data
    Buffer _data;
};

Model sqt_load(std::istream&);

/// Generator ///

// The generator holds a KV cache and executes batch=1 inference
// Note that it references Model, which must outlive it
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
    Model& model;
    KVCache kvCache;
    uint prevToken;

    explicit Generator(Model&);
    uint prefill(const std::vector<uint>& prefix, uint maxGeneratedTokens);
    uint generate();
};

}  // namespace squash

#include "squash.impl.hpp"
