#include "tensor.hpp"
#include "ops.hpp"

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

// Operations

TensorV reshape(const TensorV& tensor, const std::vector<uint>& shape) {
    if (prod(tensor.shape) != prod(shape)) {
        std::ostringstream msg;
        msg << "Cannot reshape " << tensor.shape << " to " << shape;
        throw std::invalid_argument(msg.str());
    }
    return {tensor.data, shape};
}

Tensor reshape(Tensor&& tensor, const std::vector<uint>& shape) {
    return {reshape(tensor, shape), std::move(tensor._data)};
}

std::vector<uint> strides(const TensorV& tensor) {
    std::vector<uint> strides(tensor.shape.size(), 1u);
    for (auto i = strides.size() - 1; i != 0; --i) {
        strides[i - 1] = strides[i] * tensor.shape[i];
    }
    return strides;
}

TensorV sliceLeading(const TensorV& tensor, const std::vector<uint>& indices) {
    auto stride = strides(tensor);
    auto offset = 0u;
    for (auto i = 0u; i < indices.size(); ++i) {
        offset += stride[i] * indices[i];
    }
    auto data = std::visit(
        [offset](auto& d) { return TensorV::DataT(std::decay_t<decltype(d)>(d.data + offset)); },
        tensor.data);
    return TensorV{
        data, {tensor.shape.begin() + static_cast<ptrdiff_t>(indices.size()), tensor.shape.end()}};
}

TensorV unsqueeze(const TensorV& tensor, const std::vector<uint>& indices) {
    auto shape = tensor.shape;
    for (auto i : indices) {
        shape.insert(shape.begin() + i, 1u);
    }
    return TensorV{tensor.data, shape};
}

const bf16* getBf16(const TensorV& tensor) {
    return std::get<_data::Flat<bf16>>(tensor.data).data;
}

const float* getFloat(const TensorV& tensor) {
    return std::get<_data::Flat<float>>(tensor.data).data;
}

float* getFloat(TensorV& tensor) {
    return std::get<_data::Flat<float>>(tensor.data).data;
}

Tensor embeddingLookup(const TensorV& weight, const std::vector<uint>& tokens) {
    auto out = empty<float>({uint(tokens.size()), weight.shape[1]});
    ops::gather(getBf16(weight), tokens.data(), uint(tokens.size()), weight.shape[1],
                getFloat(out));
    return out;
}

Tensor castFloat(const TensorV& x) {
    auto out = empty<float>({x.shape.begin(), x.shape.end()});
    ops::castFloat(getBf16(x), getFloat(out), prod(out.shape));
    return out;
}

Tensor concat(const std::vector<TensorV>& tensors, uint dim) {
    // Compute summary dimensions
    auto dConcat = 0u;
    for (auto i = 0u; i < tensors.size(); ++i) {
        if ([&] {
                if (tensors[i].shape.size() != tensors[0].shape.size()) {
                    return true;
                }
                for (auto j = 0u; j < tensors[0].shape.size(); ++j) {
                    if (j != dim && tensors[i].shape[j] != tensors[0].shape[j]) {
                        return true;
                    }
                }
                return false;
            }()) {
            std::ostringstream msg;
            msg << "concat: bad shape at index " << i << ", expected shape " << tensors[i].shape
                << " to match the first tensor, " << tensors[0].shape
                << ", except at concatenation dimension " << i;
            throw std::invalid_argument(msg.str());
        }
        dConcat += tensors[i].shape[dim];
    }
    auto dLeading = 1u;
    for (auto i = 0u; i < dim; ++i) {
        dLeading *= tensors[0].shape[i];
    }
    auto dTrailing = 1u;
    for (auto i = dim + 1; i < tensors[0].shape.size(); ++i) {
        dTrailing *= tensors[0].shape[i];
    }

    // Allocate result & concatenate via strided-copy
    auto shape = tensors[0].shape;
    shape[dim] = dConcat;
    auto out = empty<float>(std::move(shape));
    auto dimIndex = 0u;
    for (auto& t : tensors) {
        auto dChunk = dTrailing * t.shape[dim];
        ops::copyStrided(getFloat(t), dLeading, dChunk, dChunk, dTrailing * dConcat,
                         getFloat(out) + dimIndex * dTrailing);
        dimIndex += t.shape[dim];
    }
    return out;
}

Tensor tile(const TensorV& tensor, const std::vector<uint>& reps) {
    auto shape = reps;
    shape.insert(shape.end(), tensor.shape.begin(), tensor.shape.end());
    auto out = empty<float>(std::move(shape));
    auto dim = prod(tensor.shape);
    for (auto i = 0u; i < prod(reps); ++i) {
        ops::copy(getFloat(tensor), dim, getFloat(out) + i * dim);
    }
    return out;
}

void addInPlace(TensorV& x, const TensorV& y) {
    ops::addInPlace(getFloat(x), getFloat(y), prod(x.shape));
}

void broadcastAddInPlace(TensorV& x, const TensorV& y) {
    if (y.shape.size() != 1 || y.shape[0] != x.shape.back()) {
        std::ostringstream err;
        err << "broadcastAddInPlace bad shapes " << x.shape << " and " << y.shape;
        throw std::invalid_argument(err.str());
    }
    ops::broadcastAddInPlace(getFloat(x), getBf16(y), prod(x.shape) / y.shape[0], y.shape[0]);
}

void geluInPlace(TensorV& tensor) {
    ops::geluInPlace(getFloat(tensor), prod(tensor.shape));
}

void swiGluInPlace(TensorV& upOut, const TensorV& gate) {
    ops::swiGluInPlace(getFloat(upOut), getFloat(gate), prod(upOut.shape));
}

Tensor rmsNorm(const TensorV& weight, const TensorV& x, float epsilon) {
    if (weight.shape.size() != 1 || weight.shape[0] != x.shape.back()) {
        std::ostringstream err;
        err << "rmsNorm bad shapes " << weight.shape << " and " << x.shape;
        throw std::invalid_argument(err.str());
    }
    auto out = empty<float>({x.shape.begin(), x.shape.end()});
    ops::rmsNorm(getBf16(weight), getFloat(x), x.shape[0], x.shape[1], epsilon, getFloat(out));
    return out;
}

Tensor layerNorm(const TensorV& weight, const TensorV& bias, const TensorV& x, float epsilon) {
    auto out = empty<float>({x.shape.begin(), x.shape.end()});
    ops::layerNorm(getBf16(weight), getBf16(bias), getFloat(x), x.shape[0], x.shape[1], epsilon,
                   getFloat(out));
    return out;
}

Tensor projection(const TensorV& weight, const TensorV& x) {
    auto out = empty<float>({x.shape[0], weight.shape[0]});
    ops::matmulT(getFloat(x), getBf16(weight), x.shape[0], x.shape[1], weight.shape[0],
                 getFloat(out));
    return out;
}

uint sample(const TensorV& logits,
            float temperature,
            uint topK,
            float topP,
            std::default_random_engine& rng) {
    if (logits.shape.size() != 1) {
        std::ostringstream err;
        err << "sample: expected 1D logits, got shape " << logits.shape;
        throw std::invalid_argument(err.str());
    }
    return ops::sample(getFloat(logits), logits.shape[0], temperature, topK, topP, rng);
}

}  // namespace squash::tensor
