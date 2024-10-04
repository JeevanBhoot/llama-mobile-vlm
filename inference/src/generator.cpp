#include "ops.hpp"
#include "squash.hpp"

#include <algorithm>
#include <iostream>

namespace squash {

namespace {

constexpr ulong ActivationAlignment = 32u;

template <class T>
Tensor allocateTensor(std::vector<uint>&& shape) {
    auto buffer = Buffer(sizeof(T) * prod(shape), ActivationAlignment);
    return Tensor{{.data = tensor_data::Flat<float>(reinterpret_cast<float*>(buffer.get())),
                   .shape = std::move(shape)},
                  std::move(buffer)};
}
const bf16* getBf16(const TensorV& tensor) {
    return std::get<tensor_data::Flat<bf16>>(tensor.data).data;
}
const float* getFloat(const TensorV& tensor) {
    return std::get<tensor_data::Flat<float>>(tensor.data).data;
}
float* getFloat(TensorV& tensor) {
    return std::get<tensor_data::Flat<float>>(tensor.data).data;
}

Tensor embeddingLookup(const TensorV& weight, const std::vector<uint>& tokens) {
    auto out = allocateTensor<float>({uint(tokens.size()), weight.shape[1]});
    ops::gather(getBf16(weight), tokens.data(), uint(tokens.size()), weight.shape[1],
                getFloat(out));
    return out;
}

void addInPlace(TensorV& x, const TensorV& y) {
    ops::addInPlace(getFloat(x), getFloat(y), prod(x.shape));
}

Tensor rmsNorm(const TensorV& weight, const TensorV& x, float epsilon) {
    auto out = allocateTensor<float>({x.shape.begin(), x.shape.end()});
    ops::rmsNorm(getBf16(weight), getFloat(x), x.shape[0], x.shape[1], epsilon, getFloat(out));
    return out;
}

Tensor projection(const TensorV& weight, const TensorV& x) {
    auto out = allocateTensor<float>({x.shape[0], weight.shape[0]});
    ops::matmulT(getFloat(x), getBf16(weight), x.shape[0], x.shape[1], weight.shape[0],
                 getFloat(out));
    return out;
}

Tensor attention(const Model& model, const Model::AttentionLayer& layer, const TensorV& x) {
    auto z = rmsNorm(layer.norm, x, model.normEpsilon);
    auto query = projection(layer.query, z);
    auto key = projection(layer.key, z);
    auto value = projection(layer.value, z);
    ops::rotateInPlace(getFloat(query), model.ropeAngularFrequency.data(), 0, query.shape[0],
                       model.dAttentionKV * model.dAttentionQ, model.dAttentionHead);
    ops::rotateInPlace(getFloat(key), model.ropeAngularFrequency.data(), 0, key.shape[0],
                       model.dAttentionKV, model.dAttentionHead);
    ops::selfAttentionInPlace(getFloat(query), getFloat(key), getFloat(value), query.shape[0],
                              key.shape[0], model.dAttentionKV, model.dAttentionQ,
                              model.dAttentionHead);
    return projection(layer.output, query);
}

Tensor mlp(const Model& model, const Model::MLPLayer& layer, const TensorV& x) {
    auto z = rmsNorm(layer.norm, x, model.normEpsilon);
    auto up = projection(layer.up, z);
    auto gate = projection(layer.gate, z);
    ops::swiGluInPlace(getFloat(up), getFloat(gate), prod(up.shape));
    return projection(layer.down, up);
}

uint nextToken(const TensorV& logits) {
    auto start = getFloat(logits) + (logits.shape[0] - 1) * logits.shape[1];
    return uint(std::max_element(start, start + logits.shape[1]) - start);
}

}  // namespace

Generator::Generator(Model& model) : model(model) {}

uint Generator::prefill(const std::vector<uint>& prefix, uint /*maxGeneratedTokens*/) {
    auto x = embeddingLookup(model.embedTokens, prefix);
    for (auto& layer : model.layers) {
        auto a = attention(model, layer.attention, x);
        addInPlace(x, a);
        addInPlace(x, mlp(model, layer.mlp, x));
    }
    x = rmsNorm(model.finalNorm, x, model.normEpsilon);
    x = projection(model.embedTokens, x);
    return nextToken(x);
}

uint Generator::generate() {
    return 0;  // TODO
}

}  // namespace squash
