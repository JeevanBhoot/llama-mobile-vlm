#include "squash.hpp"

#include <omp.h>
#include <thread>
#include "stb_image.h"
#include "stb_image_resize2.h"

namespace squash {

void selectOmpNumThreads() {
    auto nthreads = std::thread::hardware_concurrency();
#if defined(ANDROID) && defined(__aarch64__)
    // Assume BIG.little on Android/ARM, e.g. 9-core = 5 threads
    nthreads = uint(std::ceil(double(nthreads) / 2));
#endif
    omp_set_num_threads(int(nthreads));
}

/// Timer ///

Timer::Timer() : start(clock::now()) {}

double Timer::elapsed() const {
    return std::chrono::duration_cast<std::chrono::duration<double>>(clock::now() - start).count();
}

/// Image ///

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
    stbir_resize_uint8_linear(image.data.data(), static_cast<int>(image.width),
                              static_cast<int>(image.height), 0, result.data.data(), newWidth,
                              newHeight, static_cast<int>(result.width * 3), STBIR_RGB);
    return result;
}

}  // namespace squash
