#pragma once

#include "squash.hpp"

namespace squash::ops {
void gather(const bf16* weight, const uint* indices, uint nIndices, uint nValues, float* out);
}  // namespace squash::ops
