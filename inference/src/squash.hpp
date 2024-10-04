#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <numeric>
#include <variant>
#include <vector>

// Development only
#define SQDUMP(obj) std::cerr << __FILE__ << ":" << __LINE__ << " " << obj << std::endl;

namespace squash {

/// Common ///

using uint = uint32_t;
using ulong = uint64_t;
using bf16 = int16_t;

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

float bf16ToFloat(bf16 value);
uint prod(const std::vector<uint>&);

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

struct Timer {
    typedef std::chrono::high_resolution_clock clock;
    clock::time_point start;
    Timer();
    double elapsed() const;
};

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
    Buffer _data;
};

Model sqt_load(std::istream&);

/// Generator ///

// The generator holds a KV cache and executes batch=1 inference
// Note that it references Model, which must outlive it
struct Generator {
    struct Cache {
        Tensor key;
        Tensor value;
    };
    Model& model;
    std::vector<Cache> cache;

    explicit Generator(Model&);
    uint prefill(const std::vector<uint>& prefix, uint maxGeneratedTokens);
    uint generate();
};

///////////////////////////////////////////////////////////////////////////////
/// Implementations ///

inline Buffer::Buffer(ulong size, ulong alignment)
    : _data(reinterpret_cast<char*>(std::aligned_alloc(alignment, size))) {}
inline Buffer::Buffer(Buffer&& other) : _data(other._data) {
    other._data = nullptr;
}
inline Buffer& Buffer::operator=(Buffer&& other) {
    reset();
    this->_data = other._data;
    other._data = nullptr;
    return *this;
}
inline Buffer::~Buffer() {
    reset();
}
inline void Buffer::reset() {
    if (_data) {
        std::free(_data);
        _data = nullptr;
    }
}
inline char* Buffer::get() const {
    return _data;
}

inline float bf16ToFloat(bf16 value) {
    union {
        float f;
        int16_t i[2];
    } u;
    u.i[0] = 0;
    u.i[1] = value;
    return u.f;
}

inline uint prod(const std::vector<uint>& x) {
    return std::accumulate(x.begin(), x.end(), 1u, std::multiplies<uint>());
}

inline Timer::Timer() : start(clock::now()) {}

inline double Timer::elapsed() const {
    return std::chrono::duration_cast<std::chrono::duration<double>>(clock::now() - start).count();
}

}  // namespace squash
