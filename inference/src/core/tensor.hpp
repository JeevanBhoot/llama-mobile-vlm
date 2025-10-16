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

// An owning tensor
struct Tensor : TensorV {
    Buffer _data;
};

uint prod(const Shape&);
std::ostream& operator<<(std::ostream&, const Shape&);
std::ostream& operator<<(std::ostream&, const TensorV&);
void saveNpy(std::ostream&, const TensorV&);
void saveNpy(const std::string& path, const TensorV&);

template <class T>
T* data(const TensorV& tensor);
Shape strides(const TensorV& tensor);

// Tensor operations

// Views
TensorV reshape(const TensorV& tensor, const Shape& shape);
Tensor reshape(Tensor&& tensor, const Shape& shape);
TensorV indexLeading(const TensorV& tensor, const std::vector<uint>& indices);
TensorV slice0(const TensorV& tensor, uint start, uint end);
TensorV unsqueeze(const TensorV& tensor, const std::vector<uint>& indices);

// Creation
template <class T>
Tensor empty(Shape shape);
template <class T>
Tensor create(const std::vector<T>& data);
template <class T>
Tensor zeros(Shape shape);
Tensor randn(Shape shape, float stddev, ulong seed);

// Data movement and type conversion
Tensor clone(const TensorV& tensor);
void assign(const TensorV& tensor, const TensorV& src);
Tensor castFloat(const TensorV& tensor);
Tensor castBf16(const TensorV& tensor);
Tensor concat(const std::vector<TensorV>& tensors, uint dim);
Tensor tile(const TensorV& tensor, const std::vector<uint>& reps);

// Maths/NN ops
Tensor add(Tensor&& x, const TensorV& y);
Tensor broadcastAdd(Tensor&& x, const TensorV& y);
Tensor gelu(Tensor&& x);
Tensor swiGlu(Tensor&& x, const TensorV& gate);
Tensor rmsNorm(const TensorV& weight, const TensorV& x, float epsilon);
Tensor layerNorm(const TensorV& weight, const TensorV& bias, const TensorV& x, float epsilon);
Tensor embeddingLookup(const TensorV& weight, const std::vector<uint>& tokens);

// weight :: (dOut, dIn)
// x      :: (batch, dIn)
Tensor projection(const TensorV& weight, const TensorV& x);

// tensor :: (dS, ..., dim)
// freq   :: (dim/2)
Tensor rotate(Tensor&& x, const std::vector<float>& freq, uint offset);

// query,out :: (dSq, dHkv, dHq, dim)
// key,value :: (dSkv, dHkv, dim)
Tensor attention(Tensor&& query, const TensorV& key, const TensorV& value, bool causal);

// Special ops
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
T* data(const TensorV& tensor) {
    return std::get<_data::Flat<T>>(tensor.data).data;
}

template <class T>
Tensor empty(Shape shape) {
    auto buffer = Buffer(sizeof(T) * prod(shape));
    return Tensor{{
                      .data = _data::Flat<T>(reinterpret_cast<T*>(buffer.get())),
                      .shape = std::move(shape),
                  },
                  std::move(buffer)};
}

template <class T>
Tensor create(const std::vector<T>& data_) {
    auto t = empty<T>({static_cast<uint>(data_.size())});
    std::copy(data_.begin(), data_.end(), data<T>(t));
    return t;
}

template <class T>
Tensor zeros(Shape shape) {
    auto t = empty<T>(std::move(shape));
    std::fill_n(data<T>(t), prod(t.shape), T(0.0f));
    return t;
}

}  // namespace squash::tensor
