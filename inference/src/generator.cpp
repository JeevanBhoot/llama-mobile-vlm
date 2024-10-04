#include "ops.hpp"
#include "squash.hpp"

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

Tensor embeddingLookup(const TensorV& weight, const std::vector<uint>& tokens) {
    auto out = allocateTensor<float>({uint(tokens.size()), weight.shape[1]});
    ops::gather(std::get<tensor_data::Flat<bf16>>(weight.data).data, tokens.data(),
                uint(tokens.size()), weight.shape[1],
                std::get<tensor_data::Flat<float>>(out.data).data);
    return out;
}
}  // namespace

Generator::Generator(Model& model) : model(model) {}

uint Generator::prefill(const std::vector<uint>& prefix, uint /*maxGeneratedTokens*/) {
    auto t = embeddingLookup(model.embedTokens, prefix);
    SQDUMP(t);
    return 0;
}

uint Generator::generate() {
    return 0;
}

}  // namespace squash
