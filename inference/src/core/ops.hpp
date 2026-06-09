// Copyright (c) 2026 Graphcore Ltd. All rights reserved.

#pragma once

#include <random>
#include "common.hpp"

namespace squash::ops {

// Data copy/convert

void copy(const float* src, uint n, float* dest);
void copy(const bf16* src, uint n, bf16* dest);
void copy(const int8_t* src_data,
          const bf16* src_scale,
          uint dN,
          uint dK,
          int8_t* dest_data,
          bf16* dest_scale);
// Copies dest[i*sDest + j] = src[i*sSrc + j], for i in [0, n), j in [0, d)
void copyStrided(const bf16* src, uint n, uint d, uint sSrc, uint sDest, bf16* dest);

void castFloat(const bf16* in, float* out, uint n);
void castBf16(const float* in, bf16* out, uint n);
void castChannelInt8(const bf16* in, uint dN, uint dK, int8_t* out_data, bf16* out_scale);
void castChannelInt8(const uint8_t* in_data,
                     const int8_t* in_lut,
                     const bf16* in_scale,
                     uint dN,
                     uint dK,
                     int8_t* out_data,
                     bf16* out_scale);

// Matmuls

// lhs {dM, dK}, rhs {dN, dK}, out {dM, dN}
void matmulT(const bf16* lhs, const bf16* rhs, uint dM, uint dK, uint dN, bf16* out);
// lhs {dM, dK}, lhsScale {dM}, rhs {dN, dK}, rhsScale {dN}, out {dM, dN}
void matmulT(const int8_t* lhs,
             const bf16* lhsScale,
             const int8_t* rhs,
             const bf16* rhsScale,
             uint dM,
             uint dK,
             uint dN,
             bf16* out);
// lhs {dM, dK}, lhsScale {dM}, rhs {dN/3, dK}, rhsLut {3, 64}, rhsScale {dN}, out {dM, dN}
void matmulT(const int8_t* lhs,
             const bf16* lhsScale,
             const uint8_t* rhs,
             const int8_t* rhsLut,
             const bf16* rhsScale,
             uint dM,
             uint dK,
             uint dN,
             bf16* out);

// Gathers

// weight {N, dim}, indices {nIndices}, out {nIndices, dim}
void gather(const bf16* weight, const uint* indices, uint nIndices, uint dim, bf16* out);
// weight {N, dim}, weightScale {N}, indices {nIndices}, out {nIndices, dim}
void gather(const int8_t* weight,
            const bf16* weightScale,
            const uint* indices,
            uint nIndices,
            uint dim,
            bf16* out);
// weight {N/3, dim}, weightLut {3, 64}, weightScale {N}, indices {nIndices}, out {nIndices, dim}
void gather(const uint8_t* weight,
            const int8_t* weightLut,
            const bf16* weightScale,
            const uint* indices,
            uint nIndices,
            uint dim,
            bf16* out);

// Maths/NN ops

void addInPlace(bf16* x, const bf16* y, uint n);
void broadcastAddInPlace(bf16* x, const bf16* y, uint n, uint d);
void geluInPlace(bf16* x, uint n);
void swiGluInPlace(bf16* x, const bf16* gate, uint n);
void rmsNorm(const bf16* weight, const bf16* x, uint batch, uint dim, float epsilon, bf16* out);
void layerNorm(const bf16* weight,
               const bf16* bias,
               const bf16* x,
               uint batch,
               uint dim,
               float epsilon,
               bf16* out);
void rotateInPlace(bf16* x, const float* freq, uint offsetS, uint dS, uint dH, uint dim);
void attentionInPlace(bf16* queryOut,
                      const bf16* key,
                      const bf16* value,
                      uint dSq,   // sequence length (query)
                      uint dSkv,  // sequence length (key-value)
                      uint dHq,   // heads (query)
                      uint dHkv,  // heads (key-value)
                      uint dim,   // head dimension
                      bool causal);

// Sampling ops

void randn(bf16* out, ulong n, float stddev, ulong seed);
uint sample(const bf16* logits,
            uint n,
            float temperature,
            uint topK,
            float topP,
            std::default_random_engine&);

}  // namespace squash::ops
