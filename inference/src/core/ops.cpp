#include "ops.hpp"

#include <algorithm>
#include <cmath>

namespace squash::ops {

void copy(const float* src, uint n, float* dest) {
    std::copy_n(src, n, dest);
}

void copyStrided(const float* src, uint n, uint d, uint sSrc, uint sDest, float* dest) {
    for (uint i = 0; i < n; ++i) {
        for (uint j = 0; j < d; ++j) {
            dest[i * sDest + j] = src[i * sSrc + j];
        }
    }
}

void addInPlace(float* __restrict__ x, const float* __restrict__ y, const uint n) {
    for (auto i = 0u; i < n; ++i) {
        x[i] += y[i];
    }
}

void rotateInPlace(float* __restrict__ x,
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
                auto re = x[idxRe];
                auto im = x[idxIm];
                auto cos = std::cos(freq[i] * float(s + offsetS));
                auto sin = std::sin(freq[i] * float(s + offsetS));
                x[idxRe] = cos * re - sin * im;
                x[idxIm] = cos * im + sin * re;
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

void attentionInPlace(float* __restrict__ queryOut,
                      const float* __restrict__ key,
                      const float* __restrict__ value,
                      const uint dSq,
                      const uint dSkv,
                      const uint dHq,
                      const uint dHkv,
                      const uint dim,
                      const bool causal) {
    std::unique_ptr<float[]> scores(new float[dSkv]);
    for (auto hKv = 0u; hKv < dHkv; ++hKv) {
        for (auto sQ = 0u; sQ < dSq; ++sQ) {
            for (auto hQ = 0u; hQ < dHq; ++hQ) {
                auto dSkv_row = causal ? (dSkv + 1 + sQ - dSq) : dSkv;
                // q @ k.T / sqrt(dim)
                for (auto sKv = 0u; sKv < dSkv_row; ++sKv) {
                    float dot = 0;
                    for (auto i = 0u; i < dim; ++i) {
                        dot +=
                            queryOut[sQ * (dHkv * dHq * dim) + hKv * (dHq * dim) + hQ * (dim) + i] *
                            key[sKv * (dHkv * dim) + hKv * (dim) + i];
                    }
                    scores[sKv] = dot / std::sqrt(float(dim));
                }
                softmaxInPlace(scores.get(), 1u, dSkv_row);
                // s @ v
                for (auto i = 0u; i < dim; ++i) {
                    float dot = 0;
                    for (auto sKv = 0u; sKv < dSkv_row; ++sKv) {
                        dot += scores[sKv] * value[sKv * (dHkv * dim) + hKv * (dim) + i];
                    }
                    queryOut[sQ * (dHkv * dHq * dim) + hKv * (dHq * dim) + hQ * (dim) + i] = dot;
                }
            }
        }
    }
}

void swiGluInPlace(float* __restrict__ x, const float* __restrict__ gate, const uint n) {
    for (auto i = 0u; i < n; ++i) {
        x[i] *= gate[i] / (1 + std::exp(-gate[i]));
    }
}

void geluInPlace(float* x, uint n) {
    // As per torch's 'approximate' GELU
    const float c0 = std::sqrtf(2.0f / M_PIf);
    const float c1 = 0.044715f;
    for (auto i = 0u; i < n; ++i) {
        float z = std::tanhf(c0 * (x[i] + c1 * x[i] * x[i] * x[i]));
        x[i] = 0.5f * x[i] * (1.0f + z);
    }
}

void gather(const bf16* __restrict__ weight,
            const uint* __restrict__ indices,
            const uint nIndices,
            const uint dim,
            float* __restrict__ out) {
    for (auto n = 0u; n < nIndices; ++n) {
        for (auto i = 0u; i < dim; ++i) {
            out[n * dim + i] = bf16ToFloat(weight[indices[n] * dim + i]);
        }
    }
}

void rmsNorm(const bf16* __restrict__ weight,
             const float* __restrict__ x,
             const uint batch,
             const uint dim,
             float epsilon,
             float* __restrict__ out) {
    for (auto n = 0u; n < batch; ++n) {
        auto xn = x + n * dim;
        float sumSq = 0;
        for (auto i = 0u; i < dim; ++i) {
            sumSq += xn[i] * xn[i];
        }
        float scale = 1 / std::sqrt(sumSq / float(dim) + epsilon);
        for (auto i = 0u; i < dim; ++i) {
            out[n * dim + i] = xn[i] * scale * bf16ToFloat(weight[i]);
        }
    }
}

void layerNorm(const bf16* __restrict__ weight,
               const bf16* __restrict__ bias,
               const float* x,
               uint batch,
               uint dim,
               float epsilon,
               float* __restrict__ out) {
    for (auto n = 0u; n < batch; ++n) {
        auto xn = x + n * dim;
        float sum = 0, sumSq = 0;
        for (auto i = 0u; i < dim; ++i) {
            sum += xn[i];
            sumSq += xn[i] * xn[i];
        }
        float mean = sum / float(dim);
        float scale = 1 / std::sqrt(sumSq / float(dim) - mean * mean + epsilon);
        for (auto i = 0u; i < dim; ++i) {
            float normed = (xn[i] - mean) * scale;
            out[n * dim + i] = normed * bf16ToFloat(weight[i]) + bf16ToFloat(bias[i]);
        }
    }
}

void matmulT(const float* __restrict__ lhs,
             const bf16* __restrict__ rhs,
             const uint dM,
             const uint dK,
             const uint dN,
             float* __restrict__ out) {
#pragma omp parallel for
    for (auto n = 0u; n < dN; ++n) {
        for (auto m = 0u; m < dM; ++m) {
            float dot = 0;
            for (auto k = 0u; k < dK; ++k) {
                dot += lhs[m * dK + k] * bf16ToFloat(rhs[n * dK + k]);
            }
            out[m * dN + n] = dot;
        }
    }
}

void broadcastAddInPlace(float* __restrict__ x, const bf16* __restrict__ y, uint n, uint d) {
    for (auto i = 0u; i < n; ++i) {
        for (auto j = 0u; j < d; ++j) {
            x[i * d + j] += bf16ToFloat(y[j]);
        }
    }
}

uint sample(const float* logits,
            uint n,
            float temperature,
            uint topK,
            float topP,
            std::default_random_engine& rng) {
    // Compute the safe log-softmax normaliser
    std::vector<std::tuple<float, uint>> logitsAndIndices;
    logitsAndIndices.reserve(n);
    auto maxLogit = *std::max_element(logits, logits + n);
    auto sumExp = 0.f;
    for (auto i = 0u; i < n; ++i) {
        auto x = logits[i] - maxLogit;
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

void castFloat(const bf16* in, float* out, uint n) {
    for (uint i = 0; i < n; ++i) {
        out[i] = bf16ToFloat(in[i]);
    }
}

void castBf16(const float* in, bf16* out, uint n) {
    for (uint i = 0; i < n; ++i) {
        out[i] = floatToBf16(in[i]);
    }
}

}  // namespace squash::ops
