#include "ops.hpp"

#include <algorithm>
#include <cmath>

namespace squash::ops {

// Float type

void copy(const float* src, uint n, float* dest) {
    std::copy_n(src, n, dest);
}

void castFloat(const bf16* in, float* out, uint n) {
    for (uint i = 0; i < n; ++i) {
        out[i] = cast<float>(in[i]);
    }
}

void castBf16(const float* in, bf16* out, uint n) {
    for (uint i = 0; i < n; ++i) {
        out[i] = cast<bf16>(in[i]);
    }
}

// Data movement and type conversion

void copy(const bf16* src, uint n, bf16* dest) {
    std::copy_n(src, n, dest);
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
        x[i] = cast<bf16>(cast<float>(x[i]) + cast<float>(y[i]));
    }
}

void broadcastAddInPlace(bf16* __restrict__ x, const bf16* __restrict__ y, uint n, uint d) {
    for (auto i = 0u; i < n; ++i) {
        for (auto j = 0u; j < d; ++j) {
            auto idx = i * d + j;
            x[idx] = cast<bf16>(cast<float>(x[idx]) + cast<float>(y[j]));
        }
    }
}

void geluInPlace(bf16* x, uint n) {
    // As per torch's 'approximate' GELU
    const float c0 = std::sqrtf(2.0f / M_PIf);
    const float c1 = 0.044715f;
    for (auto i = 0u; i < n; ++i) {
        float xi = cast<float>(x[i]);
        float z = std::tanhf(c0 * (xi + c1 * xi * xi * xi));
        x[i] = cast<bf16>(0.5f * xi * (1.0f + z));
    }
}

void swiGluInPlace(bf16* __restrict__ x, const bf16* __restrict__ gate, const uint n) {
    for (auto i = 0u; i < n; ++i) {
        auto gi = cast<float>(gate[i]);
        auto xi = cast<float>(x[i]);
        x[i] = cast<bf16>(xi * gi / (1 + std::exp(-gi)));
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
            auto xi = cast<float>(xn[i]);
            sumSq += xi * xi;
        }
        float scale = 1 / std::sqrt(sumSq / float(dim) + epsilon);
        for (auto i = 0u; i < dim; ++i) {
            out[n * dim + i] = cast<bf16>(cast<float>(xn[i]) * scale * cast<float>(weight[i]));
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
            auto xi = cast<float>(xn[i]);
            sum += xi;
            sumSq += xi * xi;
        }
        float mean = sum / float(dim);
        float scale = 1 / std::sqrt(sumSq / float(dim) - mean * mean + epsilon);
        for (auto i = 0u; i < dim; ++i) {
            float normed = (cast<float>(xn[i]) - mean) * scale;
            out[n * dim + i] = cast<bf16>(normed * cast<float>(weight[i]) + cast<float>(bias[i]));
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

void matmulT(const bf16* __restrict__ lhs,
             const bf16* __restrict__ rhs,
             const uint dM,
             const uint dK,
             const uint dN,
             bf16* __restrict__ out) {
#pragma omp parallel for
    for (auto n = 0u; n < dN; ++n) {
        for (auto m = 0u; m < dM; ++m) {
            float dot = 0;
            for (auto k = 0u; k < dK; ++k) {
                dot += cast<float>(lhs[m * dK + k]) * cast<float>(rhs[n * dK + k]);
            }
            out[m * dN + n] = cast<bf16>(dot);
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
                auto re = cast<float>(x[idxRe]);
                auto im = cast<float>(x[idxIm]);
                auto cos = std::cos(freq[i] * float(s + offsetS));
                auto sin = std::sin(freq[i] * float(s + offsetS));
                x[idxRe] = cast<bf16>(cos * re - sin * im);
                x[idxIm] = cast<bf16>(cos * im + sin * re);
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
    std::unique_ptr<float[]> scores(new float[dSkv]);
    for (auto hKv = 0u; hKv < dHkv; ++hKv) {
        for (auto sQ = 0u; sQ < dSq; ++sQ) {
            for (auto hQ = 0u; hQ < dHq; ++hQ) {
                auto dSkv_row = causal ? (dSkv + 1 + sQ - dSq) : dSkv;
                // q @ k.T / sqrt(dim)
                for (auto sKv = 0u; sKv < dSkv_row; ++sKv) {
                    float dot = 0;
                    for (auto i = 0u; i < dim; ++i) {
                        dot += cast<float>(queryOut[sQ * (dHkv * dHq * dim) + hKv * (dHq * dim) +
                                                    hQ * (dim) + i]) *
                               cast<float>(key[sKv * (dHkv * dim) + hKv * (dim) + i]);
                    }
                    scores[sKv] = dot / std::sqrt(float(dim));
                }
                softmaxInPlace(scores.get(), 1u, dSkv_row);
                // s @ v
                for (auto i = 0u; i < dim; ++i) {
                    float dot = 0;
                    for (auto sKv = 0u; sKv < dSkv_row; ++sKv) {
                        dot +=
                            scores[sKv] * cast<float>(value[sKv * (dHkv * dim) + hKv * (dim) + i]);
                    }
                    queryOut[sQ * (dHkv * dHq * dim) + hKv * (dHq * dim) + hQ * (dim) + i] =
                        cast<bf16>(dot);
                }
            }
        }
    }
}

// Special ops

uint sample(const bf16* logits,
            uint n,
            float temperature,
            uint topK,
            float topP,
            std::default_random_engine& rng) {
    // Compute the safe log-softmax normaliser
    std::vector<std::tuple<float, uint>> logitsAndIndices;
    logitsAndIndices.reserve(n);
    auto maxLogit = cast<float>(logits[0]);
    for (auto i = 1u; i < n; ++i) {
        maxLogit = std::max(maxLogit, cast<float>(logits[i]));
    }
    auto sumExp = 0.f;
    for (auto i = 0u; i < n; ++i) {
        auto x = cast<float>(logits[i]) - maxLogit;
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
