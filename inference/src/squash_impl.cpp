#include "squash.hpp"

#include <omp.h>
#include <thread>

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

void selectOmpNumThreads() {
    auto nthreads = std::thread::hardware_concurrency();
#if defined(ANDROID) && defined(__aarch64__)
    // Assume BIG.little on Android/ARM, e.g. 9-core = 5 threads
    nthreads = uint(std::ceil(double(nthreads) / 2));
#endif
    omp_set_num_threads(int(nthreads));
}

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
