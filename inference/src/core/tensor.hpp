#pragma once

#include <iostream>
#include <random>
#include <variant>
#include <vector>

#include "common.hpp"

namespace squash::tensor {

using Shape = std::vector<uint>;
constexpr ulong DefaultAlignment = 32u;

// An owning memory buffer, with alignment
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

namespace _data {
template <class T>
struct Flat {
    T* data;
    Flat(T* data = nullptr) : data(data) {}
};
}  // namespace _data

// A non-owning Tensor view
struct TensorV {
    using DataT = std::variant<_data::Flat<bf16>, _data::Flat<float>>;
    DataT data;
    Shape shape;
};

// A self-owning tensor
struct Tensor : TensorV {
    Buffer _data;
};

uint prod(const Shape&);
std::ostream& operator<<(std::ostream&, const Shape&);
std::ostream& operator<<(std::ostream&, const TensorV&);
void saveNpy(std::ostream&, const TensorV&);
void saveNpy(const std::string& path, const TensorV&);

// Tensor operations

const bf16* getBf16(const TensorV& tensor);  // TODO - remove from API?
const float* getFloat(const TensorV& tensor);
float* getFloat(TensorV& tensor);

Tensor empty(std::vector<uint>&& shape);

TensorV reshape(const TensorV& tensor, const std::vector<uint>& shape);
Tensor reshape(Tensor&& tensor, const std::vector<uint>& shape);
Shape strides(const TensorV& tensor);

TensorV sliceLeading(const TensorV& tensor, const std::vector<uint>& indices);
TensorV unsqueeze(const TensorV& tensor, const std::vector<uint>& indices);

Tensor castFloat(const TensorV& x);
Tensor embeddingLookup(const TensorV& weight, const std::vector<uint>& tokens);
Tensor concat(const std::vector<TensorV>& tensors, uint dim);
Tensor tile(const TensorV& tensor, const std::vector<uint>& reps);

void addInPlace(TensorV& x, const TensorV& y);
void broadcastAddInPlace(TensorV& x, const TensorV& y);
void geluInPlace(TensorV& tensor);
void swiGluInPlace(TensorV& upOut, const TensorV& gate);

Tensor rmsNorm(const TensorV& weight, const TensorV& x, float epsilon);
Tensor layerNorm(const TensorV& weight, const TensorV& bias, const TensorV& x, float epsilon);
Tensor projection(const TensorV& weight, const TensorV& x);

uint sample(const TensorV& logits,
            float temperature,
            uint topK,
            float topP,
            std::default_random_engine& rng);

}  // namespace squash::tensor

///////////////////////////////////////////////////////////////////////////////
// Impl

namespace squash::tensor {

template <class T>
inline T* Buffer::get() const {
    return reinterpret_cast<T*>(_data);
}

template <class T>
Tensor empty(std::vector<uint>&& shape) {
    auto buffer = Buffer(sizeof(T) * prod(shape));
    return Tensor{{.data = _data::Flat<float>(reinterpret_cast<float*>(buffer.get())),
                   .shape = std::move(shape)},
                  std::move(buffer)};
}

}  // namespace squash::tensor
