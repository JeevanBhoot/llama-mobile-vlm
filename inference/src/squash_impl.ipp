#include "squash.hpp"

// Development only
#define DUMPSQ(obj) std::cerr << __FILE__ << ":" << __LINE__ << " " << obj << std::endl;

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

inline Timer::Timer() : start(clock::now()) {}

inline double Timer::elapsed() const {
    return std::chrono::duration_cast<std::chrono::duration<double>>(clock::now() - start).count();
}

template <class T>
Dump<T> dump(const T& sequence) {
    return {sequence};
}

template <class T>
std::ostream& operator<<(std::ostream& out, const Dump<T>& x) {
    for (auto i = 0u; i < x.sequence.size(); ++i) {
        if (i) out << ", ";
        out << x.sequence[i];
    }
    return out;
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

template <class T>
inline T* Buffer::get() const {
    return reinterpret_cast<T*>(_data);
}

inline Buffer Buffer::copy(ulong size, ulong alignment) const {
    Buffer result(size, alignment);
    std::copy(_data, _data + size, result._data);
    return result;
}

}  // namespace squash
