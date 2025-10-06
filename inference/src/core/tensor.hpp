#pragma once

#include <iostream>
#include <variant>
#include <vector>

#include "common.hpp"

namespace squash::tensor {

using Shape = std::vector<uint>;
constexpr ulong DefaultAlignment = 32u;

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

// TensorV is a non-owning Tensor view
struct TensorV {
    using DataT = std::variant<_data::Flat<bf16>, _data::Flat<float>>;
    DataT data;
    Shape shape;
};

struct Tensor : TensorV {
    Buffer _data;
};

uint prod(const Shape&);
std::ostream& operator<<(std::ostream&, const Shape&);
std::ostream& operator<<(std::ostream&, const TensorV&);
void saveNpy(std::ostream&, const TensorV&);
void saveNpy(const std::string& path, const TensorV&);

}  // namespace squash::tensor

///////////////////////////////////////////////////////////////////////////////
// Impl

namespace squash::tensor {

template <class T>
inline T* Buffer::get() const {
    return reinterpret_cast<T*>(_data);
}

}  // namespace squash::tensor
