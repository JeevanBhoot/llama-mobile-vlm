#include "ops.hpp"

#include <omp.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <numbers>
#include <tuple>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif  // __ARM_NEON

namespace squash::ops {

// Float type

void copy(const float* src, uint n, float* dest) {
#pragma omp parallel for
    for (auto i = 0u; i < n; ++i) {
        dest[i] = src[i];
    }
}

void castFloat(const bf16* in, float* out, uint n) {
    for (uint i = 0; i < n; ++i) {
        out[i] = float(in[i]);
    }
}

void castBf16(const float* in, bf16* out, uint n) {
    for (uint i = 0; i < n; ++i) {
        out[i] = bf16(in[i]);
    }
}

// Data movement and type conversion

void copy(const bf16* src, uint n, bf16* dest) {
#pragma omp parallel for
    for (auto i = 0u; i < n; ++i) {
        dest[i] = src[i];
    }
}

void copyStrided(const bf16* src, uint n, uint d, uint sSrc, uint sDest, bf16* dest) {
    for (uint i = 0; i < n; ++i) {
        for (uint j = 0; j < d; ++j) {
            dest[i * sDest + j] = src[i * sSrc + j];
        }
    }
}

// Maths/NN ops

void addInPlace(bf16* __restrict__ x, const bf16* __restrict__ y, const uint n) {
    for (auto i = 0u; i < n; ++i) {
        x[i] = bf16(float(x[i]) + float(y[i]));
    }
}

void broadcastAddInPlace(bf16* __restrict__ x, const bf16* __restrict__ y, uint n, uint d) {
    for (auto i = 0u; i < n; ++i) {
        for (auto j = 0u; j < d; ++j) {
            auto idx = i * d + j;
            x[idx] = bf16(float(x[idx]) + float(y[j]));
        }
    }
}

void geluInPlace(bf16* x, uint n) {
    // As per torch's 'approximate' GELU
    const float c0 = std::sqrt(2.0f / std::numbers::pi_v<float>);
    const float c1 = 0.044715f;
    for (auto i = 0u; i < n; ++i) {
        float xi = float(x[i]);
        float z = std::tanh(c0 * (xi + c1 * xi * xi * xi));
        x[i] = bf16(0.5f * xi * (1.0f + z));
    }
}

void swiGluInPlace(bf16* __restrict__ x, const bf16* __restrict__ gate, const uint n) {
    for (auto i = 0u; i < n; ++i) {
        auto gi = float(gate[i]);
        x[i] = bf16(float(x[i]) * gi / (1 + std::exp(-gi)));
    }
}

void rmsNorm(const bf16* __restrict__ weight,
             const bf16* __restrict__ x,
             const uint batch,
             const uint dim,
             float epsilon,
             bf16* __restrict__ out) {
    for (auto n = 0u; n < batch; ++n) {
        auto xn = x + n * dim;
        float sumSq = 0;
        for (auto i = 0u; i < dim; ++i) {
            auto xi = float(xn[i]);
            sumSq += xi * xi;
        }
        float scale = 1 / std::sqrt(sumSq / float(dim) + epsilon);
        for (auto i = 0u; i < dim; ++i) {
            out[n * dim + i] = bf16(float(xn[i]) * scale * float(weight[i]));
        }
    }
}

void layerNorm(const bf16* __restrict__ weight,
               const bf16* __restrict__ bias,
               const bf16* __restrict__ x,
               uint batch,
               uint dim,
               float epsilon,
               bf16* __restrict__ out) {
    for (auto n = 0u; n < batch; ++n) {
        auto xn = x + n * dim;
        float sum = 0, sumSq = 0;
        for (auto i = 0u; i < dim; ++i) {
            auto xi = float(xn[i]);
            sum += xi;
            sumSq += xi * xi;
        }
        float mean = sum / float(dim);
        float scale = 1 / std::sqrt(sumSq / float(dim) - mean * mean + epsilon);
        for (auto i = 0u; i < dim; ++i) {
            float normed = (float(xn[i]) - mean) * scale;
            out[n * dim + i] = bf16(normed * float(weight[i]) + float(bias[i]));
        }
    }
}

void gather(const bf16* __restrict__ weight,
            const uint* __restrict__ indices,
            const uint nIndices,
            const uint dim,
            bf16* __restrict__ out) {
    for (auto n = 0u; n < nIndices; ++n) {
        for (auto i = 0u; i < dim; ++i) {
            out[n * dim + i] = weight[indices[n] * dim + i];
        }
    }
}

void gather(const int8_t* __restrict__ weight,
            const bf16* __restrict__ weightScale,
            const uint* __restrict__ indices,
            const uint nIndices,
            const uint dim,
            bf16* __restrict__ out) {
#pragma omp parallel for
    for (auto n = 0u; n < nIndices; ++n) {
        auto row = indices[n];
        auto rowScale = float(weightScale[row]);
        for (auto i = 0u; i < dim; ++i) {
            out[n * dim + i] = bf16(float(weight[row * dim + i]) * rowScale);
        }
    }
}

namespace {

int8_t _decode_s3d8(const uint8_t pack, const int8_t* __restrict__ lut, const uint vIdx) {
    switch (vIdx) {
        case 0:
            return lut[0 * 64u + (pack & 0x3Fu)];
        case 1:
            return lut[1 * 64u + ((pack >> 1) & 0x3Fu)];
        case 2:
            return lut[2 * 64u + ((pack & 0x3Fu) ^ (pack >> 7))];
        default:
            return 0;
    }
}

}  // namespace

void gather(const uint8_t* __restrict__ weight,
            const int8_t* __restrict__ weightLut,
            const bf16* __restrict__ weightScale,
            const uint* __restrict__ indices,
            const uint nIndices,
            const uint dim,
            bf16* __restrict__ out) {
#pragma omp parallel for
    for (auto n = 0u; n < nIndices; ++n) {
        const auto row = indices[n];
        auto rowScale = float(weightScale[row]);
        for (auto k = 0u; k < dim; ++k) {
            auto v = _decode_s3d8(weight[(row / 3) * dim + k], weightLut, row % 3);
            out[n * dim + k] = bf16(float(v) * rowScale);
        }
    }
}

namespace {

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)

float _dot_product_bf16(const bf16* __restrict__ a, const bf16* __restrict__ b, const uint dK) {
    float32x4_t acc = vmovq_n_f32(0.0f);
    // Main loop, process 8 elements per iteration
    const auto kStop = (dK / 8) * 8;
    for (auto k = 0u; k < kStop; k += 8) {
        auto ai = vld1q_bf16(reinterpret_cast<const __bf16*>(a + k));
        auto bi = vld1q_bf16(reinterpret_cast<const __bf16*>(b + k));
        acc = vbfdotq_f32(acc, ai, bi);
    }
    // Handle remainder when dK is not a multiple of 8
    float result = vaddvq_f32(acc);
    for (auto k = kStop; k < dK; ++k) {
        result += float(a[k]) * float(b[k]);
    }
    return result;
}

template <uint BlockM, uint BlockN>
void _matmulT_chunk_bfmmla(const __bf16* __restrict__ a,
                           const __bf16* __restrict__ b,
                           const uint dK,
                           const uint dN,
                           __bf16* __restrict__ out) {
    // Note: we expect all BlockM, BlockN loops to be unrolled
    static_assert(BlockM % 2 == 0 && BlockN % 2 == 0, "BlockM and BlockN must be even");

    // Each accumulator holds a 2x2 result, accumulated over the full `k` dimension
    float32x4_t accs[(BlockM / 2) * (BlockN / 2)];
    for (auto i = 0u; i < (BlockM / 2) * (BlockN / 2); ++i) {
        accs[i] = vmovq_n_f32(0.0f);
    }
    // Main loop, process `(m, k, n) = (BlockM, 8, BlockN)` elements per iteration
    const auto kStop = (dK / 8) * 8;
    for (auto k = 0u; k < kStop; k += 8) {
        bfloat16x8_t aa[BlockM], bb[BlockN];
        for (auto m = 0u; m < BlockM; ++m) {
            aa[m] = vld1q_bf16(&a[m * dK + k]);
        }
        for (auto n = 0u; n < BlockN; ++n) {
            bb[n] = vld1q_bf16(&b[n * dK + k]);
        }
        for (auto m = 0u; m < (BlockM / 2); ++m) {
            for (auto n = 0u; n < (BlockN / 2); ++n) {
                auto& acc = accs[m * (BlockN / 2) + n];
                acc = vbfmmlaq_f32(
                    acc, vcombine_bf16(vget_low_bf16(aa[2 * m]), vget_low_bf16(aa[2 * m + 1])),
                    vcombine_bf16(vget_low_bf16(bb[2 * n]), vget_low_bf16(bb[2 * n + 1])));
                acc = vbfmmlaq_f32(
                    acc, vcombine_bf16(vget_high_bf16(aa[2 * m]), vget_high_bf16(aa[2 * m + 1])),
                    vcombine_bf16(vget_high_bf16(bb[2 * n]), vget_high_bf16(bb[2 * n + 1])));
            }
        }
    }
    // Handle remainder when dK is not a multiple of 8
    for (auto k = kStop; k < dK; ++k) {
        for (auto m = 0u; m < (BlockM / 2); ++m) {
            for (auto n = 0u; n < (BlockN / 2); ++n) {
                auto& acc = accs[m * (BlockN / 2) + n];
                float a0 = vcvtah_f32_bf16(a[(2 * m + 0) * dK + k]);
                float a1 = vcvtah_f32_bf16(a[(2 * m + 1) * dK + k]);
                float b0 = vcvtah_f32_bf16(b[(2 * n + 0) * dK + k]);
                float b1 = vcvtah_f32_bf16(b[(2 * n + 1) * dK + k]);
                acc = vmlaq_f32(acc, float32x4_t{a0, a0, a1, a1}, float32x4_t{b0, b1, b0, b1});
            }
        }
    }
    // Store out results, a BlockM x BlockN matrix
    for (auto m = 0u; m < (BlockM / 2); ++m) {
        for (auto n = 0u; n < (BlockN / 2); ++n) {
            auto acc_bf16 = vcvt_bf16_f32(accs[m * (BlockN / 2) + n]);
            vst1_lane_bf16(&out[(2 * m + 0) * dN + (2 * n + 0)], acc_bf16, 0);
            vst1_lane_bf16(&out[(2 * m + 0) * dN + (2 * n + 1)], acc_bf16, 1);
            vst1_lane_bf16(&out[(2 * m + 1) * dN + (2 * n + 0)], acc_bf16, 2);
            vst1_lane_bf16(&out[(2 * m + 1) * dN + (2 * n + 1)], acc_bf16, 3);
        }
    }
}

void _matmulT(const bf16* __restrict__ a,
              const bf16* __restrict__ b,
              const uint dM,
              const uint dK,
              const uint dN,
              bf16* __restrict__ out) {
    // Matrix-vector case
    if (dM == 1 || dN == 1) {
#pragma omp parallel for
        for (auto i = 0u; i < dM * dN; ++i) {
            auto m = i / dN;
            auto n = i % dN;
            out[m * dN + n] = bf16(_dot_product_bf16(&a[m * dK], &b[n * dK], dK));
        }
        return;
    }

    constexpr auto G0 = 16u;  // block size
    constexpr auto G1 = 8u;   // inner block size

    const auto blocksM = (dM + G0 - 1) / G0;
    const auto blocksN = (dN + G0 - 1) / G0;

#pragma omp parallel for
    for (auto i0 = 0u; i0 < blocksM * blocksN; ++i0) {
        auto m0 = G0 * (i0 / blocksN);
        auto m1 = std::min(m0 + G0, dM);
        auto n0 = G0 * (i0 % blocksN);
        auto n1 = std::min(n0 + G0, dN);

        // Main loop
        auto mStop = (m1 / G1) * G1;
        auto nStop = (n1 / G1) * G1;
        for (auto n = n0; n < nStop; n += G1) {
            for (auto m = m0; m < mStop; m += G1) {
                _matmulT_chunk_bfmmla<G1, G1>(reinterpret_cast<const __bf16*>(&a[m * dK]),
                                              reinterpret_cast<const __bf16*>(&b[n * dK]), dK, dN,
                                              reinterpret_cast<__bf16*>(&out[m * dN + n]));
            }
        }
        // Handle remainder when dN is not a multiple of G1, `out[m0:m1, nStop:n1]`
        for (auto n = nStop; n < n1; ++n) {
            for (auto m = m0; m < m1; ++m) {
                out[m * dN + n] = bf16(_dot_product_bf16(&a[m * dK], &b[n * dK], dK));
            }
        }
        // Handle remainder when dM is not a multiple of G1, `out[mStop:m1, n0:nStop]`
        // (note: excludes the bottom-right corner which is handled in the loop above)
        for (auto m = mStop; m < m1; ++m) {
            for (auto n = n0; n < nStop; ++n) {
                out[m * dN + n] = bf16(_dot_product_bf16(&a[m * dK], &b[n * dK], dK));
            }
        }
    }
}

#else  // !(__ARM_NEON && __ARM_FEATURE_BF16_VECTOR_ARITHMETIC)

float _dot_product_bf16(const bf16* __restrict__ a, const bf16* __restrict__ b, const uint n) {
    float result = 0;
    for (auto i = 0u; i < n; ++i) {
        result += float(a[i]) * float(b[i]);
    }
    return result;
}

void _matmulT(const bf16* __restrict__ lhs,
              const bf16* __restrict__ rhs,
              const uint dM,
              const uint dK,
              const uint dN,
              bf16* __restrict__ out) {
#pragma omp parallel for
    for (auto n = 0u; n < dN; ++n) {
        for (auto m = 0u; m < dM; ++m) {
            out[m * dN + n] = bf16(_dot_product_bf16(&lhs[m * dK], &rhs[n * dK], dK));
        }
    }
}

#endif  // __ARM_NEON && __ARM_FEATURE_BF16_VECTOR_ARITHMETIC

float _dot_product_bf16_int8(const bf16* __restrict__ a,
                             const int8_t* __restrict__ b,
                             const uint n) {
    float result = 0;
#pragma omp simd reduction(+ : result)
    for (auto i = 0u; i < n; ++i) {
        result += float(a[i]) * float(b[i]);
    }
    return result;
}

float _dot_product_bf16_s3d8(const bf16* __restrict__ a,
                             const uint8_t* __restrict__ b,
                             const int8_t* __restrict__ bLut,
                             const uint n,
                             const uint dK) {
    float result = 0;
    for (auto k = 0u; k < dK; ++k) {
        result += float(a[k]) * float(_decode_s3d8(b[k], bLut, n % 3));
    }
    return result;
}

}  // namespace

void matmulT(const bf16* __restrict__ lhs,
             const bf16* __restrict__ rhs,
             const uint dM,
             const uint dK,
             const uint dN,
             bf16* __restrict__ out) {
    _matmulT(lhs, rhs, dM, dK, dN, out);
}

void matmulT(const bf16* __restrict__ lhs,
             const int8_t* __restrict__ rhs,
             const bf16* __restrict__ rhsScale,
             const uint dM,
             const uint dK,
             const uint dN,
             bf16* __restrict__ out) {
#pragma omp parallel for
    for (auto n = 0u; n < dN; ++n) {
        auto scale = float(rhsScale[n]);
        for (auto m = 0u; m < dM; ++m) {
            auto dot = _dot_product_bf16_int8(&lhs[m * dK], &rhs[n * dK], dK);
            out[m * dN + n] = bf16(dot * scale);
        }
    }
}

void matmulT(const bf16* __restrict__ lhs,
             const uint8_t* __restrict__ rhs,
             const int8_t* __restrict__ rhsLut,
             const bf16* __restrict__ rhsScale,
             const uint dM,
             const uint dK,
             const uint dN,
             bf16* __restrict__ out) {
#pragma omp parallel for
    for (auto n = 0u; n < dN; ++n) {
        const auto scale = float(rhsScale[n]);
        for (auto m = 0u; m < dM; ++m) {
            auto dot = _dot_product_bf16_s3d8(&lhs[m * dK], &rhs[(n / 3) * dK], rhsLut, n, dK);
            out[m * dN + n] = bf16(dot * scale);
        }
    }
}

void rotateInPlace(bf16* __restrict__ x,
                   const float* __restrict__ freq,
                   const uint offsetS,
                   const uint dS,
                   const uint dH,
                   const uint dim) {
    for (auto s = 0u; s < dS; ++s) {
        for (auto h = 0u; h < dH; ++h) {
            for (auto i = 0u; i < dim / 2; ++i) {
                auto idxRe = s * (dH * dim) + h * (dim) + i;
                auto idxIm = idxRe + dim / 2;
                auto re = float(x[idxRe]);
                auto im = float(x[idxIm]);
                auto cos = std::cos(freq[i] * float(s + offsetS));
                auto sin = std::sin(freq[i] * float(s + offsetS));
                x[idxRe] = bf16(cos * re - sin * im);
                x[idxIm] = bf16(cos * im + sin * re);
            }
        }
    }
}

namespace {
void softmaxInPlace(float* __restrict__ x, const uint batch, const uint dim) {
    for (auto n = 0u; n < batch; ++n) {
        auto xn = x + n * dim;
        auto max = *std::max_element(xn, xn + dim);
        float sum = 0;
        for (auto i = 0u; i < dim; ++i) {
            xn[i] = std::exp(xn[i] - max);
            sum += xn[i];
        }
        for (auto i = 0u; i < dim; ++i) {
            xn[i] /= sum;
        }
    }
}
}  // namespace

void attentionInPlace(bf16* __restrict__ queryOut,
                      const bf16* __restrict__ key,
                      const bf16* __restrict__ value,
                      const uint dSq,
                      const uint dSkv,
                      const uint dHq,
                      const uint dHkv,
                      const uint dim,
                      const bool causal) {
#pragma omp parallel
    {
        std::unique_ptr<float[]> scores(new float[dSkv]);
        std::unique_ptr<float[]> outTmp(new float[dim]);
#pragma omp for
        for (auto n = 0u; n < dHkv * dSq * dHq; ++n) {
            auto hKv = n / (dSq * dHq);
            auto sQ = (n / dHq) % dSq;
            auto hQ = n % dHq;
            auto dSkv_row = causal ? (dSkv + 1 + sQ - dSq) : dSkv;
            // q @ k.T / sqrt(dim)
            for (auto sKv = 0u; sKv < dSkv_row; ++sKv) {
                float dot = _dot_product_bf16(
                    &queryOut[sQ * (dHkv * dHq * dim) + hKv * (dHq * dim) + hQ * (dim)],
                    &key[sKv * (dHkv * dim) + hKv * (dim)], dim);
                scores[sKv] = dot / std::sqrt(float(dim));
            }
            softmaxInPlace(scores.get(), 1u, dSkv_row);
            // s @ v
            std::fill_n(outTmp.get(), dim, 0.0f);
            for (auto sKv = 0u; sKv < dSkv_row; ++sKv) {
                for (auto i = 0u; i < dim; ++i) {
                    outTmp[i] += scores[sKv] * float(value[sKv * (dHkv * dim) + hKv * (dim) + i]);
                }
            }
            for (auto i = 0u; i < dim; ++i) {
                queryOut[sQ * (dHkv * dHq * dim) + hKv * (dHq * dim) + hQ * (dim) + i] =
                    bf16(outTmp[i]);
            }
        }
    }
}

// Special ops

void randn(bf16* out, ulong n, float stddev, ulong seed) {
#pragma omp parallel
    {
        std::default_random_engine rng(seed + static_cast<ulong>(omp_get_thread_num()));
#pragma omp for schedule(static)
        for (auto i = 0ul; i < n; ++i) {
            out[i] = bf16(std::normal_distribution<float>(0, stddev)(rng));
        }
    }
}

uint sample(const bf16* logits,
            uint n,
            float temperature,
            uint topK,
            float topP,
            std::default_random_engine& rng) {
    // Compute the safe log-softmax normaliser
    std::vector<std::tuple<float, uint>> logitsAndIndices;
    logitsAndIndices.reserve(n);
    auto maxLogit = float(logits[0]);
    for (auto i = 1u; i < n; ++i) {
        maxLogit = std::max(maxLogit, float(logits[i]));
    }
    auto sumExp = 0.f;
    for (auto i = 0u; i < n; ++i) {
        auto x = float(logits[i]) - maxLogit;
        logitsAndIndices.push_back({x, i});
        sumExp += std::exp(x);
    }

    // Sort the logits & indices in descending order
    std::sort(logitsAndIndices.begin(), logitsAndIndices.end(),
              [](const auto& lhs, const auto& rhs) { return std::get<0>(lhs) > std::get<0>(rhs); });

    // Calculate how many alternatives we're actually going to sample from
    auto topKandTopP = 1u;
    auto cumulativeP = std::exp(std::get<0>(logitsAndIndices[0]));
    while (topKandTopP < std::min(topK, n) & cumulativeP < topP * sumExp) {
        cumulativeP += std::exp(std::get<0>(logitsAndIndices[topKandTopP++]));
    }

    // Sample using Gumbel-max
    for (auto i = 0u; i < topKandTopP; ++i) {
        auto noise = std::log(-std::log(std::uniform_real_distribution<float>()(rng)));
        std::get<0>(logitsAndIndices[i]) -= temperature * noise;
    }
    return std::get<1>(
        *std::max_element(logitsAndIndices.begin(), logitsAndIndices.begin() + topKandTopP));
}

}  // namespace squash::ops
