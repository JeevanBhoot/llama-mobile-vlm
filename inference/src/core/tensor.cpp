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

TensorV reshape(const TensorV& tensor, const Shape& shape) {
    if (prod(tensor.shape) != prod(shape)) {
        std::ostringstream msg;
        msg << "Cannot reshape " << tensor.shape << " to " << shape;
        throw std::invalid_argument(msg.str());
    }
    return {tensor.data, shape};
}

Tensor reshape(Tensor&& tensor, const Shape& shape) {
    return {reshape(tensor, shape), std::move(tensor._data)};
}

std::vector<uint> strides(const TensorV& tensor) {
    std::vector<uint> strides(tensor.shape.size(), 1u);
    for (auto i = strides.size() - 1; i != 0; --i) {
        strides[i - 1] = strides[i] * tensor.shape[i];
    }
    return strides;
}

TensorV indexLeading(const TensorV& tensor, const std::vector<uint>& indices) {
    auto offset = 0u;
    auto stride = strides(tensor);
    for (auto i = 0u; i < indices.size(); ++i) {
        if (indices[i] >= tensor.shape[i]) {
            std::ostringstream err;
            err << "indexLeading: bad index " << indices[i] << " for dimension " << i << " of size "
                << tensor.shape[i];
            throw std::invalid_argument(err.str());
        }
        offset += stride[i] * indices[i];
    }
    auto data = std::visit(
        [offset](auto& d) { return TensorV::DataT(std::decay_t<decltype(d)>(d.data + offset)); },
        tensor.data);
    return TensorV{
        data, {tensor.shape.begin() + static_cast<ptrdiff_t>(indices.size()), tensor.shape.end()}};
}

TensorV slice0(const TensorV& tensor, uint start, uint end) {
    if (tensor.shape.size() == 0 || start > end || end > tensor.shape[0]) {
        std::ostringstream err;
        err << "slice0: bad indices start: " << start << ", end: " << end << " for shape "
            << tensor.shape;
        throw std::invalid_argument(err.str());
    }
    auto offset = start * (prod(tensor.shape) / tensor.shape[0]);
    auto data = std::visit(
        [offset](auto& d) { return TensorV::DataT(std::decay_t<decltype(d)>(d.data + offset)); },
        tensor.data);
    auto shape = tensor.shape;
    shape[0] = end - start;
    return TensorV{data, shape};
}

TensorV unsqueeze(const TensorV& tensor, const std::vector<uint>& indices) {
    auto shape = tensor.shape;
    for (auto i : indices) {
        shape.insert(shape.begin() + i, 1u);
    }
    return TensorV{tensor.data, shape};
}

bf16* getBf16(const TensorV& tensor) {
    return std::get<_data::Flat<bf16>>(tensor.data).data;
}

float* getFloat(const TensorV& tensor) {
    return std::get<_data::Flat<float>>(tensor.data).data;
}

Tensor embeddingLookup(const TensorV& weight, const std::vector<uint>& tokens) {
    auto out = empty<float>({uint(tokens.size()), weight.shape[1]});
    ops::gather(getBf16(weight), tokens.data(), uint(tokens.size()), weight.shape[1],
                getFloat(out));
    return out;
}

void assign(const TensorV& tensor, const TensorV& src) {
    if (tensor.shape != src.shape) {
        std::ostringstream err;
        err << "assign: shapes don't match, tensor.shape: " << tensor.shape
            << ", src.shape: " << src.shape;
        throw std::invalid_argument(err.str());
    }
    ops::copy(getFloat(src), prod(src.shape), getFloat(tensor));
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

Tensor add(Tensor&& x, const TensorV& y) {
    if (x.shape != y.shape) {
        std::ostringstream err;
        err << "add: shapes don't match, x.shape: " << x.shape << " and y.shape: " << y.shape;
        throw std::invalid_argument(err.str());
    }
    ops::addInPlace(getFloat(x), getFloat(y), prod(x.shape));
    return std::move(x);
}

Tensor broadcastAdd(Tensor&& x, const TensorV& y) {
    if (y.shape.size() != 1 || y.shape[0] != x.shape.back()) {
        std::ostringstream err;
        err << "broadcastAdd: bad shapes " << x.shape << " and " << y.shape;
        throw std::invalid_argument(err.str());
    }
    ops::broadcastAddInPlace(getFloat(x), getBf16(y), prod(x.shape) / y.shape[0], y.shape[0]);
    return std::move(x);
}

Tensor gelu(Tensor&& tensor) {
    ops::geluInPlace(getFloat(tensor), prod(tensor.shape));
    return std::move(tensor);
}

Tensor swiGlu(Tensor&& up, const TensorV& gate) {
    if (up.shape != gate.shape) {
        std::ostringstream err;
        err << "swiGlu: bad shapes up: " << up.shape << ", gate: " << gate.shape;
        throw std::invalid_argument(err.str());
    }
    ops::swiGluInPlace(getFloat(up), getFloat(gate), prod(up.shape));
    return std::move(up);
}

Tensor rmsNorm(const TensorV& weight, const TensorV& x, float epsilon) {
    if (weight.shape.size() != 1 || weight.shape[0] != x.shape.back()) {
        std::ostringstream err;
        err << "rmsNorm: bad shapes weight: " << weight.shape << ", x: " << x.shape;
        throw std::invalid_argument(err.str());
    }
    auto out = empty<float>({x.shape.begin(), x.shape.end()});
    ops::rmsNorm(getBf16(weight), getFloat(x), prod(x.shape) / x.shape.back(), x.shape.back(),
                 epsilon, getFloat(out));
    return out;
}

Tensor layerNorm(const TensorV& weight, const TensorV& bias, const TensorV& x, float epsilon) {
    if (weight.shape.size() != 1 || weight.shape[0] != x.shape.back() ||
        bias.shape != weight.shape) {
        std::ostringstream err;
        err << "layerNorm: bad shapes weight: " << weight.shape << ", bias: " << bias.shape
            << ", x: " << x.shape;
        throw std::invalid_argument(err.str());
    }
    auto out = empty<float>({x.shape.begin(), x.shape.end()});
    ops::layerNorm(getBf16(weight), getBf16(bias), getFloat(x), prod(x.shape) / x.shape.back(),
                   x.shape.back(), epsilon, getFloat(out));
    return out;
}

Tensor projection(const TensorV& weight, const TensorV& x) {
    if (weight.shape.size() != 2 || x.shape.size() != 2 || weight.shape[1] != x.shape[1]) {
        std::ostringstream err;
        err << "projection: bad shapes " << weight.shape << " and " << x.shape
            << ", expected (dOut, dIn) and (batch, dIn)";
        throw std::invalid_argument(err.str());
    }
    auto out = empty<float>({x.shape[0], weight.shape[0]});
    ops::matmulT(getFloat(x), getBf16(weight), x.shape[0], x.shape[1], weight.shape[0],
                 getFloat(out));
    return out;
}

Tensor rotate(Tensor&& tensor, const std::vector<float>& freq, uint offset) {
    if (tensor.shape.size() < 2 || tensor.shape.back() != freq.size() * 2) {
        std::ostringstream err;
        err << "rotate: bad shape: " << tensor.shape << " or freq.size " << freq.size()
            << ", expected shape: (dS, ..., dim) and freq.size: (dim/2)";
        throw std::invalid_argument(err.str());
    }
    auto dH = prod(tensor.shape) / (tensor.shape[0] * tensor.shape.back());
    ops::rotateInPlace(getFloat(tensor), freq.data(), offset, /*dS*/ tensor.shape[0],
                       /*dH*/ dH, /*dim*/ tensor.shape.back());
    return std::move(tensor);
}

Tensor attention(Tensor&& query, const TensorV& key, const TensorV& value, bool causal) {
    // query,out :: (dSq, dHkv, dHq, dim)
    // key,value :: (dSkv, dHkv, dim)
    if (query.shape.size() != 4 || key.shape.size() != 3 || (query.shape[1] != key.shape[1]) ||
        (query.shape[3] != key.shape[2]) || key.shape != value.shape) {
        std::ostringstream err;
        err << "attention: bad shapes, query: " << query.shape << ", key: " << key.shape
            << ", value: " << value.shape
            << "; expected query: (dSq, dHkv, dHq, dim) and key,value: (dSkv, dHkv, dim)";
        throw std::invalid_argument(err.str());
    }
    ops::attentionInPlace(getFloat(query), getFloat(key), getFloat(value), /*dSq*/ query.shape[0],
                          /*dSkv*/ key.shape[0], /*dHq*/ query.shape[2],
                          /*dHkv*/ query.shape[1], /*dim*/ query.shape[3], causal);
    return std::move(query);
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
