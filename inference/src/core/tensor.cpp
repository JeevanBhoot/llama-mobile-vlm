#include "tensor.hpp"
#include "ops.hpp"

#include <fstream>
#include <numeric>
#include <regex>
#include <sstream>
#include <unordered_map>

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

// Note: round up allocated size to a multiple of `alignment`, required on Android
Buffer::Buffer(ulong size, ulong alignment)
    : _data(reinterpret_cast<char*>(
          std::aligned_alloc(alignment, (size + alignment - 1) / alignment * alignment))) {
    if (!_data) {
        std::ostringstream err;
        err << "Buffer allocation failed, size: " << size << ", alignment: " << alignment;
        throw std::runtime_error(err.str());
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

template <class T>
void printFlatTensorData(std::ostream& out, const _data::Flat<T>& data, ulong nElements) {
    if (nElements <= MaxPrint) {
        for (auto i = 0u; i < nElements; ++i) {
            if (i) out << ", ";
            out << float(data.data[i]);
        }
    } else {
        for (auto i = 0u; i < MaxPrint / 2; ++i) {
            if (i) out << ", ";
            out << float(data.data[i]);
        }
        out << " ... ";
        auto start2 = nElements - MaxPrint / 2;
        for (auto i = start2; i < nElements; ++i) {
            if (start2 < i) out << ", ";
            out << float(data.data[i]);
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
                std::transform(data.data, data.data + nElement, fData.data(),
                               [](bf16 x) { return float(x); });
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

// Basic & restrictive implementation
Tensor loadNpy(std::istream& in) {
    // Read magic + version
    char magic[6];
    char versionMajor, versionMinor;
    in.read(magic, sizeof(magic)).get(versionMajor).get(versionMinor);
    if (std::string(magic, sizeof(magic)) != "\x93NUMPY" || versionMajor != 1 ||
        versionMinor != 0) {
        throw std::runtime_error("loadNpy: invalid npy file (bad magic or version)");
    }

    // Read header
    uint16_t headerSize;
    in.read(reinterpret_cast<char*>(&headerSize), sizeof(headerSize));
    std::string header(headerSize, '\0');
    in.read(header.data(), static_cast<std::streamsize>(headerSize));

    // Parse header
    std::regex re("'(\\w+?)': (\\(.+?\\)|.+?),");
    std::sregex_iterator iter(header.begin(), header.end(), re);
    std::unordered_map<std::string, std::string> headerMap;
    for (; iter != std::sregex_iterator(); ++iter) {
        std::smatch match = *iter;
        headerMap[match[1]] = match[2];
    }
    if (headerMap["descr"] != "'<f4'") {
        throw std::runtime_error("loadNpy: only '<f4' dtype is supported");
    }
    if (headerMap["fortran_order"] != "False") {
        throw std::runtime_error("loadNpy: fortran_order=True is not supported");
    }
    // Parse shape
    Shape shape;
    std::string shapeStr = headerMap["shape"].substr(1, headerMap["shape"].size() - 2);
    std::string dimStr;
    std::stringstream ss(shapeStr);
    while (std::getline(ss, dimStr, ',')) {
        if (!dimStr.empty()) {
            shape.push_back(static_cast<uint>(std::stoul(dimStr)));
        }
    }

    // Read data
    auto result = empty<float>(shape);
    in.read(reinterpret_cast<char*>(data<float>(result)),
            static_cast<std::streamsize>(prod(shape) * sizeof(float)));

    if (!in.good()) {
        throw std::runtime_error("loadNpy: error reading data");
    }
    return result;
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

Tensor loadNpy(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::ostringstream err;
        err << "loadNpy: error opening file " << path;
        throw std::runtime_error(err.str());
    }
    return loadNpy(in);
}

std::vector<uint> strides(const TensorV& tensor) {
    std::vector<uint> strides(tensor.shape.size(), 1u);
    for (auto i = strides.size() - 1; i != 0; --i) {
        strides[i - 1] = strides[i] * tensor.shape[i];
    }
    return strides;
}

// Operations

TensorV reshape(const TensorV& tensor, const Shape& shape) {
    if (!((std::holds_alternative<_data::Flat<float>>(tensor.data) ||
           std::holds_alternative<_data::Flat<bf16>>(tensor.data)))) {
        throw std::invalid_argument("reshape requires Flat tensor data");
    }
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
        [offset](auto& d) -> TensorV::DataT {
            using T = std::decay_t<decltype(d)>;
            if constexpr (std::is_same_v<T, _data::ChannelInt8>) {
                throw std::invalid_argument("indexLeading does not support ChannelInt8 tensors");
            } else {
                return T(d.data + offset);
            }
        },
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
        [offset](auto& d) -> TensorV::DataT {
            using T = std::decay_t<decltype(d)>;
            if constexpr (std::is_same_v<T, _data::ChannelInt8>) {
                throw std::invalid_argument("slice0 does not support ChannelInt8 tensors");
            } else {
                return T(d.data + offset);
            }
        },
        tensor.data);
    auto shape = tensor.shape;
    shape[0] = end - start;
    return TensorV{data, shape};
}

TensorV unsqueeze(const TensorV& tensor, const std::vector<uint>& indices) {
    if (!((std::holds_alternative<_data::Flat<float>>(tensor.data) ||
           std::holds_alternative<_data::Flat<bf16>>(tensor.data)))) {
        throw std::invalid_argument("unsqueeze requires Flat tensor data");
    }
    auto shape = tensor.shape;
    for (auto i : indices) {
        shape.insert(shape.begin() + i, 1u);
    }
    return TensorV{tensor.data, shape};
}

Tensor randn(Shape shape, float stddev, ulong seed) {
    auto t = empty<bf16>(std::move(shape));
    ops::randn(data<bf16>(t), prod(t.shape), stddev, seed);
    return t;
}

Tensor clone(const TensorV& tensor) {
    return std::visit(
        [&](auto& d) -> Tensor {
            if constexpr (std::is_same_v<std::decay_t<decltype(d)>, _data::Flat<float>>) {
                auto out = empty<float>(tensor.shape);
                ops::copy(d.data, prod(tensor.shape), data<float>(out));
                return out;

            } else if constexpr (std::is_same_v<std::decay_t<decltype(d)>, _data::Flat<bf16>>) {
                auto out = empty<bf16>(tensor.shape);
                ops::copy(d.data, prod(tensor.shape), data<bf16>(out));
                return out;

            } else {
                std::ostringstream err;
                err << "clone: Unexpected TensorV::data type: " << typeid(d).name() << "\n";
                throw std::runtime_error(err.str());
            }
        },
        tensor.data);
}

void assign(const TensorV& tensor, const TensorV& src) {
    if (tensor.shape != src.shape) {
        std::ostringstream err;
        err << "assign: shapes don't match, tensor.shape: " << tensor.shape
            << ", src.shape: " << src.shape;
        throw std::invalid_argument(err.str());
    }
    ops::copy(data<bf16>(src), prod(src.shape), data<bf16>(tensor));
}

Tensor castFloat(const TensorV& tensor) {
    if (std::holds_alternative<_data::Flat<float>>(tensor.data)) {
        return clone(tensor);
    }
    auto out = empty<float>({tensor.shape.begin(), tensor.shape.end()});
    ops::castFloat(data<bf16>(tensor), data<float>(out), prod(out.shape));
    return out;
}

Tensor castBf16(const TensorV& tensor) {
    if (std::holds_alternative<_data::Flat<bf16>>(tensor.data)) {
        return clone(tensor);
    }
    auto out = empty<bf16>({tensor.shape.begin(), tensor.shape.end()});
    ops::castBf16(data<float>(tensor), data<bf16>(out), prod(out.shape));
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
    auto out = empty<bf16>(std::move(shape));
    auto dimIndex = 0u;
    for (auto& t : tensors) {
        auto dChunk = dTrailing * t.shape[dim];
        ops::copyStrided(data<bf16>(t), dLeading, dChunk, dChunk, dTrailing * dConcat,
                         data<bf16>(out) + dimIndex * dTrailing);
        dimIndex += t.shape[dim];
    }
    return out;
}

Tensor tile(const TensorV& tensor, const std::vector<uint>& reps) {
    auto shape = reps;
    shape.insert(shape.end(), tensor.shape.begin(), tensor.shape.end());
    auto out = empty<bf16>(std::move(shape));
    auto dim = prod(tensor.shape);
    for (auto i = 0u; i < prod(reps); ++i) {
        ops::copy(data<bf16>(tensor), dim, data<bf16>(out) + i * dim);
    }
    return out;
}

Tensor add(Tensor&& x, const TensorV& y) {
    if (x.shape != y.shape) {
        std::ostringstream err;
        err << "add: shapes don't match, x.shape: " << x.shape << " and y.shape: " << y.shape;
        throw std::invalid_argument(err.str());
    }
    ops::addInPlace(data<bf16>(x), data<bf16>(y), prod(x.shape));
    return std::move(x);
}

Tensor broadcastAdd(Tensor&& x, const TensorV& y) {
    if (y.shape.size() != 1 || y.shape[0] != x.shape.back()) {
        std::ostringstream err;
        err << "broadcastAdd: bad shapes " << x.shape << " and " << y.shape;
        throw std::invalid_argument(err.str());
    }
    ops::broadcastAddInPlace(data<bf16>(x), data<bf16>(y), prod(x.shape) / y.shape[0], y.shape[0]);
    return std::move(x);
}

Tensor gelu(Tensor&& x) {
    ops::geluInPlace(data<bf16>(x), prod(x.shape));
    return std::move(x);
}

Tensor swiGlu(Tensor&& x, const TensorV& gate) {
    if (x.shape != gate.shape) {
        std::ostringstream err;
        err << "swiGlu: bad shapes x: " << x.shape << ", gate: " << gate.shape;
        throw std::invalid_argument(err.str());
    }
    ops::swiGluInPlace(data<bf16>(x), data<bf16>(gate), prod(x.shape));
    return std::move(x);
}

Tensor rmsNorm(const TensorV& weight, const TensorV& x, float epsilon) {
    if (weight.shape.size() != 1 || weight.shape[0] != x.shape.back()) {
        std::ostringstream err;
        err << "rmsNorm: bad shapes weight: " << weight.shape << ", x: " << x.shape;
        throw std::invalid_argument(err.str());
    }
    auto out = empty<bf16>({x.shape.begin(), x.shape.end()});
    ops::rmsNorm(data<bf16>(weight), data<bf16>(x), prod(x.shape) / x.shape.back(), x.shape.back(),
                 epsilon, data<bf16>(out));
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
    auto out = empty<bf16>({x.shape.begin(), x.shape.end()});
    ops::layerNorm(data<bf16>(weight), data<bf16>(bias), data<bf16>(x),
                   prod(x.shape) / x.shape.back(), x.shape.back(), epsilon, data<bf16>(out));
    return out;
}

Tensor embeddingLookup(const TensorV& weight, const std::vector<uint>& tokens) {
    if (weight.shape.size() != 2) {
        std::ostringstream err;
        err << "embeddingLookup: expected 2D weight, weight.shape: " << weight.shape;
        throw std::invalid_argument(err.str());
    }
    auto out = empty<bf16>({uint(tokens.size()), weight.shape[1]});
    if (std::holds_alternative<_data::ChannelInt8>(weight.data)) {
        auto weight_ = std::get<_data::ChannelInt8>(weight.data);
        ops::gather(weight_.data, weight_.scale, tokens.data(), uint(tokens.size()),
                    weight.shape[1], data<bf16>(out));
    } else {
        ops::gather(data<bf16>(weight), tokens.data(), uint(tokens.size()), weight.shape[1],
                    data<bf16>(out));
    }
    return out;
}

Tensor projection(const TensorV& weight, const TensorV& x) {
    if (weight.shape.size() != 2 || x.shape.size() != 2 || weight.shape[1] != x.shape[1]) {
        std::ostringstream err;
        err << "projection: bad shapes " << weight.shape << " and " << x.shape
            << ", expected (dOut, dIn) and (batch, dIn)";
        throw std::invalid_argument(err.str());
    }
    auto out = empty<bf16>({x.shape[0], weight.shape[0]});
    if (std::holds_alternative<_data::ChannelInt8>(weight.data)) {
        auto weight_ = std::get<_data::ChannelInt8>(weight.data);
        ops::matmulT(data<bf16>(x), weight_.data, weight_.scale, x.shape[0], x.shape[1],
                     weight.shape[0], data<bf16>(out));
    } else {
        ops::matmulT(data<bf16>(x), data<bf16>(weight), x.shape[0], x.shape[1], weight.shape[0],
                     data<bf16>(out));
    }
    return out;
}

Tensor rotate(Tensor&& x, const std::vector<float>& freq, uint offset) {
    if (x.shape.size() < 2 || x.shape.back() != freq.size() * 2) {
        std::ostringstream err;
        err << "rotate: bad shape: " << x.shape << " or freq.size " << freq.size()
            << ", expected shape: (dS, ..., dim) and freq.size: (dim/2)";
        throw std::invalid_argument(err.str());
    }
    auto dH = prod(x.shape) / (x.shape[0] * x.shape.back());
    ops::rotateInPlace(data<bf16>(x), freq.data(), offset, /*dS*/ x.shape[0],
                       /*dH*/ dH, /*dim*/ x.shape.back());
    return std::move(x);
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
    ops::attentionInPlace(data<bf16>(query), data<bf16>(key), data<bf16>(value),
                          /*dSq*/ query.shape[0],
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
        err << "sample: expected logits to be 1D, logits.shape: " << logits.shape;
        throw std::invalid_argument(err.str());
    }
    return ops::sample(data<bf16>(logits), logits.shape[0], temperature, topK, topP, rng);
}

}  // namespace squash::tensor
