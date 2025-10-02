#include "lib/ops.hpp"
#include "squash.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>

namespace squash {

namespace {

// Tensors

template <class T>
Tensor allocateTensor(std::vector<uint>&& shape) {
    auto buffer = Buffer(sizeof(T) * prod(shape));
    return Tensor{{.data = tensor_data::Flat<float>(reinterpret_cast<float*>(buffer.get())),
                   .shape = std::move(shape)},
                  std::move(buffer)};
}

TensorV reshape(const TensorV& tensor, const std::vector<uint>& shape) {
    if (prod(tensor.shape) != prod(shape)) {
        std::ostringstream msg;
        msg << "Cannot reshape " << dump(tensor.shape) << " to " << dump(shape);
        throw std::invalid_argument(msg.str());
    }
    return {tensor.data, shape};
}

std::vector<uint> strides(const TensorV& tensor) {
    std::vector<uint> strides(tensor.shape.size(), 1u);
    for (auto i = strides.size() - 1; i != 0; --i) {
        strides[i - 1] = strides[i] * tensor.shape[i];
    }
    return strides;
}

TensorV sliceLeading(const TensorV& tensor, const std::vector<uint>& indices) {
    auto stride = strides(tensor);
    auto offset = 0u;
    for (auto i = 0u; i < indices.size(); ++i) {
        offset += stride[i] * indices[i];
    }
    auto data = std::visit(
        [offset](auto& d) { return TensorV::DataT(std::decay_t<decltype(d)>(d.data + offset)); },
        tensor.data);
    return TensorV{
        data, {tensor.shape.begin() + static_cast<ptrdiff_t>(indices.size()), tensor.shape.end()}};
}

TensorV unsqueeze(const TensorV& tensor, const std::vector<uint>& indices) {
    auto shape = tensor.shape;
    for (auto i : indices) {
        shape.insert(shape.begin() + i, 1u);
    }
    return TensorV{tensor.data, shape};
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

Tensor castFloat(const TensorV& x) {
    auto out = allocateTensor<float>({x.shape.begin(), x.shape.end()});
    ops::castFloat(getBf16(x), getFloat(out), prod(out.shape));
    return out;
}

Tensor concat(const std::vector<TensorV>& tensors, uint dim) {
    // Compute summary dimensions
    auto dConcat = 0u;
    for (auto i = 0u; i < tensors.size(); ++i) {
        if ([&] {
                if (tensors[i].shape.size() != tensors[0].shape.size()) {
                    return true;
                }
                for (auto j = 0u; j < tensors[0].shape.size(); ++j) {
                    if (j != dim && tensors[i].shape[j] != tensors[0].shape[j]) {
                        return true;
                    }
                }
                return false;
            }()) {
            std::ostringstream msg;
            msg << "concat: bad shape at index " << i << ", expected shape "
                << dump(tensors[i].shape) << " to match the first tensor, "
                << dump(tensors[0].shape) << ", except at concatenation dimension " << i;
            throw std::invalid_argument(msg.str());
        }
        dConcat += tensors[i].shape[dim];
    }
    auto dLeading = 1u;
    for (auto i = 0u; i < dim; ++i) {
        dLeading *= tensors[0].shape[i];
    }
    auto dTrailing = 1u;
    for (auto i = dim + 1; i < tensors[0].shape.size(); ++i) {
        dTrailing *= tensors[0].shape[i];
    }

    // Allocate result & concatenate via strided-copy
    auto shape = tensors[0].shape;
    shape[dim] = dConcat;
    auto out = allocateTensor<float>(std::move(shape));
    auto dimIndex = 0u;
    for (auto& t : tensors) {
        auto dChunk = dTrailing * t.shape[dim];
        ops::copyStrided(getFloat(t), dLeading, dChunk, dChunk, dTrailing * dConcat,
                         getFloat(out) + dimIndex * dTrailing);
        dimIndex += t.shape[dim];
    }
    return out;
}

void addInPlace(TensorV& x, const TensorV& y) {
    ops::addInPlace(getFloat(x), getFloat(y), prod(x.shape));
}

void broadcastAddInPlace(TensorV& x, const TensorV& y) {
    if (y.shape.size() != 1 || y.shape[0] != x.shape.back()) {
        std::ostringstream err;
        err << "broadcastAddInPlace bad shapes " << dump(x.shape) << ", " << dump(y.shape);
        throw std::invalid_argument(err.str());
    }
    ops::broadcastAddInPlace(getFloat(x), getBf16(y), prod(x.shape) / y.shape[0], y.shape[0]);
}

Tensor rmsNorm(const TensorV& weight, const TensorV& x, float epsilon) {
    auto out = allocateTensor<float>({x.shape.begin(), x.shape.end()});
    ops::rmsNorm(getBf16(weight), getFloat(x), x.shape[0], x.shape[1], epsilon, getFloat(out));
    return out;
}

Tensor layerNorm(const TensorV& weight, const TensorV& bias, const TensorV& x, float epsilon) {
    auto out = allocateTensor<float>({x.shape.begin(), x.shape.end()});
    ops::layerNorm(getBf16(weight), getBf16(bias), getFloat(x), x.shape[0], x.shape[1], epsilon,
                   getFloat(out));
    return out;
}

Tensor projection(const TensorV& weight, const TensorV& x) {
    auto out = allocateTensor<float>({x.shape[0], weight.shape[0]});
    ops::matmulT(getFloat(x), getBf16(weight), x.shape[0], x.shape[1], weight.shape[0],
                 getFloat(out));
    return out;
}

Tensor attention(const TextModel& model,
                 const TextModel::AttentionLayer& layer,
                 const TensorV& x,
                 uint tokenCount,
                 Generator::KVCache::Entry& cache) {
    auto z = rmsNorm(layer.norm, x, model.normEpsilon);
    auto query = projection(layer.query, z);
    auto key = projection(layer.key, z);
    auto value = projection(layer.value, z);
    ops::rotateInPlace(getFloat(query), model.ropeAngularFrequency.data(), tokenCount,
                       query.shape[0], model.dAttentionKV * model.dAttentionQ,
                       model.dAttentionHead);
    ops::rotateInPlace(getFloat(key), model.ropeAngularFrequency.data(), tokenCount, key.shape[0],
                       model.dAttentionKV, model.dAttentionHead);
    ops::copy(getFloat(key), prod(key.shape), getFloat(cache.key) + tokenCount * key.shape[1]);
    ops::copy(getFloat(value), prod(value.shape),
              getFloat(cache.value) + tokenCount * key.shape[1]);
    ops::selfAttentionInPlace(getFloat(query), getFloat(cache.key), getFloat(cache.value),
                              /*dSq*/ query.shape[0], /*dSkv*/ key.shape[0] + tokenCount,
                              /*dHq*/ model.dAttentionQ, /*dHkv*/ model.dAttentionKV,
                              /*dim*/ model.dAttentionHead, /*causal*/ true);
    return projection(layer.output, query);
}

Tensor mlp(const TextModel& model, const TextModel::MLPLayer& layer, const TensorV& x) {
    auto z = rmsNorm(layer.norm, x, model.normEpsilon);
    auto up = projection(layer.up, z);
    auto gate = projection(layer.gate, z);
    ops::swiGluInPlace(getFloat(up), getFloat(gate), prod(up.shape));
    return projection(layer.down, up);
}

Tensor visionAttention(const VisionModel& model,
                       const VisionModel::AttentionLayer& layer,
                       const TensorV& x) {
    auto z = layerNorm(layer.norm.weight, layer.norm.bias, x, model.normEpsilon);
    auto query = projection(layer.query, z);
    auto key = projection(layer.key, z);
    auto value = projection(layer.value, z);
    auto dS = query.shape[0];
    ops::selfAttentionInPlace(getFloat(query), getFloat(key), getFloat(value), dS, dS, /*dHq*/ 1u,
                              /*dHkv*/ model.dAttentionQkv, /*dim*/ model.dAttentionHead,
                              /*causal*/ false);
    return projection(layer.output, query);
}

Tensor visionMlp(const VisionModel& model, const VisionModel::MLPLayer layer, const TensorV& x) {
    auto z = layerNorm(layer.norm.weight, layer.norm.bias, x, model.normEpsilon);
    z = projection(layer.up.weight, z);
    broadcastAddInPlace(z, layer.up.bias);
    ops::geluInPlace(getFloat(z), prod(z.shape));
    z = projection(layer.down.weight, z);
    broadcastAddInPlace(z, layer.down.bias);
    return z;
}

// High-level

// Resize, normalise colour channels, and chop into patches
// returns: (yPatchIdx * xPatchIdx, channel * yIdx * xIdx)
Tensor preprocess(const VisionModel& model, const Image& image) {
    const auto resized = resizeImage(image, model.dImage, model.dImage);
    const auto nPatch = (model.dImage / model.dPatch);
    const auto dChannel = 3;
    const auto dPatch = model.dPatch;

    auto result = allocateTensor<float>({nPatch * nPatch, dChannel * dPatch * dPatch});
    auto ptr = getFloat(result);
    const auto nStride = dChannel * dPatch * dPatch;
    const auto cStride = dPatch * dPatch;
    for (auto n = 0u; n < nPatch * nPatch; ++n) {
        for (auto i = 0u; i < dPatch * dPatch; ++i) {
            for (auto c = 0u; c < dChannel; ++c) {
                auto x = (n % nPatch) * dPatch + (i % dPatch);
                auto y = (n / nPatch) * dPatch + (i / dPatch);
                auto px = resized.data[y * (resized.width * dChannel) + x * dChannel + c];
                ptr[n * nStride + c * cStride + i] =
                    (px / 255.0f - model.imageMean[c]) / model.imageStd[c];
            }
        }
    }
    return result;
}

uint nextToken(Generator& g, const TensorV& logits) {
    auto data = getFloat(logits) + (logits.shape[0] - 1) * logits.shape[1];
    return ops::sample(data, logits.shape[1], g.options.temperature, g.options.topK, g.options.topP,
                       g.rng);
}

void resetCache(Generator& g, uint dSequenceMax) {
    g.kvCache.dSequence = 0;
    g.kvCache.dSequenceMax = dSequenceMax;
    g.kvCache.entries.clear();
    const auto& textModel = g.model.textModel;
    for (auto i = 0u; i < textModel.dLayers; ++i) {
        g.kvCache.entries.push_back({allocateTensor<float>({dSequenceMax, textModel.dAttentionKV,
                                                            textModel.dAttentionHead}),
                                     allocateTensor<float>({dSequenceMax, textModel.dAttentionKV,
                                                            textModel.dAttentionHead})});
    }
}

void forward(Generator& g, const std::vector<uint>& tokens) {
    if (g.kvCache.dSequenceMax < g.kvCache.dSequence + tokens.size()) {
        std::ostringstream err;
        err << "Over-full KV cache of maximum size " << g.kvCache.dSequenceMax << " tokens";
        throw std::runtime_error(err.str());
    }
    auto& model = g.model.textModel;
    auto x = embeddingLookup(model.embedTokens, tokens);
    for (auto i = 0u; i < model.dLayers; ++i) {
        if (std::find(model.crossAttentionLayers.begin(), model.crossAttentionLayers.end(), i) !=
            model.crossAttentionLayers.end()) {
            continue;  // TODO - cross attention
        }
        addInPlace(x, attention(model, model.layers[i].attention, x, g.kvCache.dSequence,
                                g.kvCache.entries[i]));
        addInPlace(x, mlp(model, model.layers[i].mlp, x));
    }
    x = rmsNorm(model.finalNorm, x, model.normEpsilon);
    x = projection(model.predictTokens, x);
    g.kvCache.dSequence += uint(tokens.size());
    g.prevToken = nextToken(g, x);
}

void forwardImage(Generator& g, const TensorV& image) {
    auto& model = *g.model.visionModel;

    // Embeddings
    auto x = projection(
        reshape(model.patchEmbedding, {model.dModel, 3 * model.dPatch * model.dPatch}), image);
    // Lookup {aspectRatioID = 0, tileIndex = 0}
    addInPlace(x, castFloat(sliceLeading(model.positionalEmbedding, {0, 0})));
    x = concat({unsqueeze(castFloat(sliceLeading(model.classEmbedding, {0, 0})), {0}), x}, 0);
    x = layerNorm(model.layerNormPre.weight, model.layerNormPre.bias, x, model.normEpsilon);

    // Transformer stack
    addInPlace(x, visionAttention(model, model.layers0[0].attention, x));
    addInPlace(x, visionMlp(model, model.layers0[0].mlp, x));

    std::ofstream f("tmp/x.npy", std::ios_base::binary);
    saveNpy(f, x);
}

}  // namespace

Generator::Options Generator::Options::greedy(uint maxGeneratedTokens) {
    return {.maxGeneratedTokens = maxGeneratedTokens,
            .seed = std::nullopt,
            .temperature = 0,
            .topK = 1,
            .topP = 0};
}

Generator::Generator(Model& model) : model(model) {}

std::vector<std::string> Generator::prefill(const std::string& prefix,
                                            const std::optional<Image>& image,
                                            const Options& options) {
    // Set generation state
    this->options = options;
    if (options.seed.has_value()) {
        this->rng.seed(*options.seed);
    } else {
        std::random_device d;
        this->rng.seed(d());
    }

    // Handle image
    if (image) {
        if (!model.visionModel) {
            std::ostringstream err;
            err << "Passed an image to " << model.source << ", which does not support vision input";
            throw std::runtime_error(err.str());
        }
        forwardImage(*this, preprocess(*model.visionModel, *image));
    }

    // Handle text
    auto tokens = model.textModel.tokenizer.encode(prefix);
    tokens.insert(tokens.begin(), model.textModel.beginOfTextID);
    resetCache(*this, uint(tokens.size() + options.maxGeneratedTokens));
    forward(*this, tokens);

    // Return tokens, including prompt
    if (this->prevToken != model.textModel.endOfTextID) {
        tokens.push_back(this->prevToken);
    }
    std::vector<std::string> stringTokens;
    stringTokens.reserve(tokens.size() - 1);
    std::transform(tokens.begin() + 1, tokens.end(), std::back_inserter(stringTokens),
                   [&](uint t) { return model.textModel.tokenizer.decode({t}); });
    return stringTokens;
}

std::string Generator::generate() {
    if (kvCache.dSequence == kvCache.dSequenceMax) {
        return "";
    }
    forward(*this, {prevToken});
    return (prevToken == model.textModel.endOfTextID)
               ? ""
               : model.textModel.tokenizer.decode({prevToken});
}

}  // namespace squash
