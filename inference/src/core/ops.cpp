#include "ops.hpp"

#include <omp.h>
#include <algorithm>
#include <cmath>
#include <iostream>

#ifdef __ARM_NEON
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
    const float c0 = std::sqrtf(2.0f / M_PIf);
    const float c1 = 0.044715f;
    for (auto i = 0u; i < n; ++i) {
        float xi = float(x[i]);
        float z = std::tanhf(c0 * (xi + c1 * xi * xi * xi));
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

namespace {

#if defined(__ARM_NEON)

bf16 dot_product_bf16(const bf16* __restrict__ a, const bf16* __restrict__ b, const uint n) {
    float32x4_t acc = vmovq_n_f32(0.0f);
    for (auto i = 0u; i < n / 8; ++i) {
        acc = vbfdotq_f32(acc, vld1q_bf16(reinterpret_cast<const __bf16*>(a + i * 8)),
                          vld1q_bf16(reinterpret_cast<const __bf16*>(b + i * 8)));
    }
    float result = vaddvq_f32(acc);
    for (auto i = (n / 8) * 8; i < n; ++i) {
        result += float(a[i]) * float(b[i]);
    }
    return bf16(result);
}

#else  // !__ARM_NEON

bf16 dot_product_bf16(const bf16* __restrict__ a, const bf16* __restrict__ b, const uint n) {
    float result = 0;
    for (auto i = 0u; i < n; ++i) {
        result += float(a[i]) * float(b[i]);
    }
    return bf16(result);
}

#endif  // __ARM_NEON

}  // namespace

void matmulT(const bf16* __restrict__ lhs,
             const bf16* __restrict__ rhs,
             const uint dM,
             const uint dK,
             const uint dN,
             bf16* __restrict__ out) {
#pragma omp parallel for
    for (auto n = 0u; n < dN; ++n) {
        for (auto m = 0u; m < dM; ++m) {
            out[m * dN + n] = dot_product_bf16(&lhs[m * dK], &rhs[n * dK], dK);
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
#pragma omp for
        for (auto n = 0u; n < dHkv * dSq * dHq; ++n) {
            auto hKv = n / (dSq * dHq);
            auto sQ = (n / dHq) % dSq;
            auto hQ = n % dHq;
            auto dSkv_row = causal ? (dSkv + 1 + sQ - dSq) : dSkv;
            // q @ k.T / sqrt(dim)
            for (auto sKv = 0u; sKv < dSkv_row; ++sKv) {
                float dot = 0;
                for (auto i = 0u; i < dim; ++i) {
                    dot += float(queryOut[sQ * (dHkv * dHq * dim) + hKv * (dHq * dim) + hQ * (dim) +
                                          i]) *
                           float(key[sKv * (dHkv * dim) + hKv * (dim) + i]);
                }
                scores[sKv] = dot / std::sqrt(float(dim));
            }
            softmaxInPlace(scores.get(), 1u, dSkv_row);
            // s @ v
            for (auto i = 0u; i < dim; ++i) {
                float dot = 0;
                for (auto sKv = 0u; sKv < dSkv_row; ++sKv) {
                    dot += scores[sKv] * float(value[sKv * (dHkv * dim) + hKv * (dim) + i]);
                }
                queryOut[sQ * (dHkv * dHq * dim) + hKv * (dHq * dim) + hQ * (dim) + i] = bf16(dot);
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
