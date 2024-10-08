#include "squash.hpp"

namespace squash {

namespace {
constexpr ulong MaxPrint = 8u;

float toFloat(float v) {
    return v;
}

float toFloat(bf16 v) {
    return bf16ToFloat(v);
}

template <class T>
void printFlatTensorData(std::ostream& out, const tensor_data::Flat<T>& data, ulong nElements) {
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
    out << "Tensor{(" << dump(tensor.shape) << "): ";
    std::visit(
        [&](auto&& data) {
            auto nElements = prod(tensor.shape);
            using T = std::decay_t<decltype(data)>;
            if constexpr (std::is_same_v<T, tensor_data::Flat<float>> ||
                          std::is_same_v<T, tensor_data::Flat<bf16>>) {
                printFlatTensorData(out, data, nElements);
            } else {
                out << "((unknown))";
            }
        },
        tensor.data);
    return out << "}";
}

}  // namespace squash
