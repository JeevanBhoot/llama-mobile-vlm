#include "ops.hpp"

namespace squash::ops {
void gather(const bf16* __restrict__ weight,
            const uint* __restrict__ indices,
            uint nIndices,
            uint nValues,
            float* __restrict__ out) {
    for (auto n = 0u; n < nIndices; ++n) {
        for (auto i = 0u; i < nValues; ++i) {
            out[n * nValues + i] = bf16ToFloat(weight[indices[n] * nValues + i]);
        }
    }
}
}  // namespace squash::ops