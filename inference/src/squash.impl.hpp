#include "squash.hpp"

// Development only
#define SQDUMP(obj) std::cerr << __FILE__ << ":" << __LINE__ << " " << obj << std::endl;

namespace squash {

/// Common ///

inline float bf16ToFloat(bf16 value) {
    union {
        float f;
        int16_t i[2];
    } u;
    u.i[0] = 0;
    u.i[1] = value;
    return u.f;
}

inline uint prod(const std::vector<uint>& x) {
    return std::accumulate(x.begin(), x.end(), 1u, std::multiplies<uint>());
}

/// Buffer ///

inline Buffer::Buffer(ulong size, ulong alignment)
    : _data(reinterpret_cast<char*>(std::aligned_alloc(alignment, size))) {
    if (!_data) {
        throw std::runtime_error("Allocation failed");
    }
}

inline Buffer::Buffer(Buffer&& other) : _data(other._data) {
    other._data = nullptr;
}

inline Buffer& Buffer::operator=(Buffer&& other) {
    reset();
    this->_data = other._data;
    other._data = nullptr;
    return *this;
}

inline Buffer::~Buffer() {
    reset();
}

inline void Buffer::reset() {
    if (_data) {
        std::free(_data);
        _data = nullptr;
    }
}

inline char* Buffer::get() const {
    return _data;
}

/// Timer ///

inline Timer::Timer() : start(clock::now()) {}

inline double Timer::elapsed() const {
    return std::chrono::duration_cast<std::chrono::duration<double>>(clock::now() - start).count();
}

}  // namespace squash
