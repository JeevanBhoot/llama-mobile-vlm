#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <variant>
#include <vector>

// Development only
#define SQDUMP(obj) std::cerr << __FILE__ << ":" << __LINE__ << " " << obj << std::endl;

namespace squash {

// Common

using uint = uint32_t;
using ulong = uint64_t;
using bf16 = int16_t;

float bf16ToFloat(bf16 value);

namespace tensor_data {

struct BF16 {
    bf16* data;
    BF16(bf16* data = nullptr) : data(data) {}
};

}  // namespace tensor_data

// TensorV is a non-owning "tensor view"
struct TensorV {
    std::variant<tensor_data::BF16> data;
    std::vector<uint> shape;
};

struct Timer {
    typedef std::chrono::high_resolution_clock clock;
    clock::time_point start;
    Timer();
    double elapsed() const;
};

// Model

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
    uint alignment;

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
    std::unique_ptr<char[]> _parameterData;
};

Model sqt_load(std::istream&);

// Temporary

int meaning();

///////////////////////////////////////////////////////////////////////////////
// Implementations

inline float bf16ToFloat(bf16 value) {
    union {
        float f;
        int16_t i[2];
    } u;
    u.i[0] = 0;
    u.i[1] = value;
    return u.f;
}

inline Timer::Timer() : start(clock::now()) {}

inline double Timer::elapsed() const {
    return std::chrono::duration_cast<std::chrono::duration<double>>(clock::now() - start).count();
}

}  // namespace squash
