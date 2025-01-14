#include "squash.hpp"

#include <omp.h>
#include <thread>
#include "stb_image.h"
#include "stb_image_resize2.h"

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

Image::Image(uint height, uint width, std::vector<uint8_t>&& data_)
    : height(height), width(width), data(std::move(data_)) {
    if (data.size() != height * width * 3) {
        std::ostringstream err;
        err << "Image data is the wrong size: expected a " << height << "x" << width
            << " image to have size " << height * width * 3 << ", actual size " << data.size();
        throw std::runtime_error(err.str());
    }
}

Image loadImage(const std::string& path) {
    int width, height, channels;
    unsigned char* data = stbi_load(path.c_str(), &width, &height, &channels, 3);
    if (!data) {
        std::ostringstream err;
        err << "Cannot load file " << path << " -- " << stbi_failure_reason();
        throw std::runtime_error(err.str());
    }
    Image result(static_cast<uint>(height), static_cast<uint>(width),
                 {data, data + height * width * 3});
    stbi_image_free(data);
    return result;
}

Image resizeImage(const Image& image, uint height, uint width) {
    float scale = std::min(static_cast<float>(height) / static_cast<float>(image.height),
                           static_cast<float>(width) / static_cast<float>(image.width));
    int newHeight = static_cast<int>(scale * static_cast<float>(image.height));
    int newWidth = static_cast<int>(scale * static_cast<float>(image.width));
    Image result(height, width, std::vector<uint8_t>(height * width * 3, 0));
    stbir_resize_uint8_srgb(image.data.data(), static_cast<int>(image.width),
                            static_cast<int>(image.height), 0, result.data.data(), newWidth,
                            newHeight, static_cast<int>(result.width * 3), STBIR_RGB);
    return result;
}

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
