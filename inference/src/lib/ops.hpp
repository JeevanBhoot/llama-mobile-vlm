#pragma once

#include <random>
#include "squash.hpp"

namespace squash::ops {

// float

void copy(const float* src, uint n, float* dest);
// Copies dest[i*sDest + j] = src[i*sSrc + j], for i in [0, n), j in [0, d)
void copyStrided(const float* src, uint n, uint d, uint sSrc, uint sDest, float* dest);
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
                          uint dim,   // head dimension
                          bool causal);

void swiGluInPlace(float* x, const float* gate, uint n);
void geluInPlace(float* x, uint n);
uint sample(const float* logits,
            uint n,
            float temperature,
            uint topK,
            float topP,
            std::default_random_engine&);

// bf16

void gather(const bf16* weight, const uint* indices, uint nIndices, uint dim, float* out);
void rmsNorm(const bf16* weight, const float* x, uint batch, uint dim, float epsilon, float* out);
void layerNorm(const bf16* weight,
               const bf16* bias,
               const float* x,
               uint batch,
               uint dim,
               float epsilon,
               float* out);
// lhs {dM, dK}, rhs {dN, dK}, out {dM, dN}
void matmulT(const float* lhs, const bf16* rhs, uint dM, uint dK, uint dN, float* out);
void broadcastAddInPlace(float* x, const bf16* y, uint n, uint d);

// conversion

void castFloat(const bf16* in, float* out, uint n);

}  // namespace squash::ops
