#pragma once

#include "squash.hpp"

namespace squash::ops {

void copy(const float* src, uint n, float* dest);
void addInPlace(float* x, const float* y, uint n);
void rotateInPlace(float* x, const float* freq, uint offsetS, uint dS, uint dH, uint dim);
void softmaxInPlace(float* x, uint batch, uint dim);
void selfAttentionInPlace(float* queryOut,
                          const float* key,
                          const float* value,
                          uint dSq,   // sequence length (query)
                          uint dSkv,  // sequence length (key-value)
                          uint dHq,   // heads (query)
                          uint dHkv,  // heads (key-value)
                          uint dim);  // head dimension
void swiGluInPlace(float* x, const float* gate, uint n);

void gather(const bf16* weight, const uint* indices, uint nIndices, uint dim, float* out);
void rmsNorm(const bf16* weight, const float* x, uint batch, uint dim, float epsilon, float* out);
// lhs {dM, dK}, rhs {dN, dK}, out {dM, dN}
void matmulT(const float* lhs, const bf16* rhs, uint dM, uint dK, uint dN, float* out);

}  // namespace squash::ops
