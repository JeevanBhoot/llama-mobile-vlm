#include "tensor.hpp"

#include <fstream>
#include <numeric>
#include <sstream>

namespace squash::tensor {

uint prod(const Shape& x) {
    return std::accumulate(x.begin(), x.end(), 1u, std::multiplies<uint>());
}

std::ostream& operator<<(std::ostream& out, const Shape& shape) {
    out << "(";
    for (size_t i = 0; i < shape.size(); i++) {
        if (i) {
            out << ", ";
        }
        out << shape[i];
    }
    if (shape.size() == 1) {
        out << ",";
    }
    return out << ")";
}

/// Buffer ///

Buffer::Buffer(ulong size, ulong alignment)
    : _data(reinterpret_cast<char*>(std::aligned_alloc(alignment, size))) {
    if (!_data) {
        throw std::runtime_error("Allocation failed");
    }
}

Buffer::Buffer(Buffer&& other) : _data(other._data) {
    other._data = nullptr;
}

Buffer& Buffer::operator=(Buffer&& other) {
    reset();
    this->_data = other._data;
    other._data = nullptr;
    return *this;
}

Buffer::~Buffer() {
    reset();
}

void Buffer::reset() {
    if (_data) {
        std::free(_data);
        _data = nullptr;
    }
}

Buffer Buffer::copy(ulong size, ulong alignment) const {
    Buffer result(size, alignment);
    std::copy(_data, _data + size, result._data);
    return result;
}

/// Tensor ///

namespace {
constexpr ulong MaxPrint = 8u;

float toFloat(float v) {
    return v;
}
float toFloat(bf16 v) {
    return bf16ToFloat(v);
}
template <class T>
void printFlatTensorData(std::ostream& out, const _data::Flat<T>& data, ulong nElements) {
    if (nElements <= MaxPrint) {
        for (auto i = 0u; i < nElements; ++i) {
            if (i) out << ", ";
            out << toFloat(data.data[i]);
        }
    } else {
        for (auto i = 0u; i < MaxPrint / 2; ++i) {
            if (i) out << ", ";
            out << toFloat(data.data[i]);
        }
        out << " ... ";
        auto start2 = nElements - MaxPrint / 2;
        for (auto i = start2; i < nElements; ++i) {
            if (start2 < i) out << ", ";
            out << toFloat(data.data[i]);
        }
    }
}

}  // namespace

std::ostream& operator<<(std::ostream& out, const TensorV& tensor) {
    out << "Tensor{" << tensor.shape << ": ";
    std::visit(
        [&](auto&& data) {
            auto nElements = prod(tensor.shape);
            using T = std::decay_t<decltype(data)>;
            if constexpr (std::is_same_v<T, _data::Flat<float>> ||
                          std::is_same_v<T, _data::Flat<bf16>>) {
                printFlatTensorData(out, data, nElements);
            } else {
                out << "((unknown))";
            }
        },
        tensor.data);
    return out << "}";
}

void saveNpy(std::ostream& out, const TensorV& tensor) {
    // Magic + Version
    std::string magic = "\x93NUMPY";
    out.write(magic.c_str(), static_cast<std::streamsize>(magic.size()));
    out.put(1);  // version.major
    out.put(0);  // version.minor

    // Header
    std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': (";
    for (size_t i = 0; i < tensor.shape.size(); i++) {
        if (i) header += ", ";
        header += std::to_string(tensor.shape[i]);
    }
    if (tensor.shape.size() == 1) header += ",";  // tuple syntax for 1D
    header += "), }";

    // Pad with spaces so that total_size is a multiple of 16
    // size: magic + version + header_size (u16) + header + newline
    size_t headerPad = 16 - ((magic.size() + 2 + 2 + header.size() + 1) % 16);
    header.append(headerPad, ' ');
    header.push_back('\n');
    uint16_t headerSize = static_cast<uint16_t>(header.size());
    out.write(reinterpret_cast<char*>(&headerSize), sizeof(headerSize));
    out.write(header.c_str(), static_cast<std::streamsize>(header.size()));

    // Data
    auto nElement = prod(tensor.shape);
    std::visit(
        [&out, nElement](auto&& data) {
            using T = std::decay_t<decltype(data)>;
            if constexpr (std::is_same_v<T, _data::Flat<float>>) {
                out.write(reinterpret_cast<const char*>(data.data),
                          static_cast<std::streamsize>(nElement * sizeof(float)));
            } else if constexpr (std::is_same_v<T, _data::Flat<bf16>>) {
                std::vector<float> fData(nElement);
                std::transform(data.data, data.data + nElement, fData.data(), bf16ToFloat);
                out.write(reinterpret_cast<const char*>(fData.data()),
                          static_cast<std::streamsize>(nElement * sizeof(float)));
            } else {
                std::ostringstream err;
                err << "saveNpy: Unexpected TensorV.data type: " << typeid(T).name() << "\n";
                throw std::runtime_error(err.str());
            }
        },
        tensor.data);
}

void saveNpy(const std::string& path, const TensorV& tensor) {
    std::ofstream out(path, std::ios::binary);
    saveNpy(out, tensor);
    if (!out) {
        std::ostringstream err;
        err << "saveNpy: error writing file " << path;
        throw std::runtime_error(err.str());
    }
}

}  // namespace squash::tensor
