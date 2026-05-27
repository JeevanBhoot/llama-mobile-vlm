#include "squash.hpp"

#include <algorithm>
#include <iostream>
#include <sstream>

using namespace squash::tensor;

namespace squash {

namespace {

constexpr auto S3D8DequantizeBatchThreshold = 16u;

Tensor projection(const TensorV& weight, const TensorV& x) {
    if (std::holds_alternative<_data::ChannelS3D8>(weight.data) &&
        x.shape[0] > S3D8DequantizeBatchThreshold) {
        auto weightInt8 = castChannelInt8(weight);
        return tensor::matmulT(castChannelInt8(x), weightInt8);
    }
    if (std::holds_alternative<_data::ChannelInt8>(weight.data) ||
        std::holds_alternative<_data::ChannelS3D8>(weight.data)) {
        return tensor::matmulT(castChannelInt8(x), weight);
    }
    return tensor::matmulT(x, weight);
}

const ProgressCallback NoProgressCallback;
struct ProgressTracker {
    uint currentStep;
    uint totalSteps;
    const ProgressCallback& callback;

    ProgressTracker(uint totalSteps, const ProgressCallback& callback)
        : currentStep(0), totalSteps(totalSteps), callback(callback) {
        report();
    }

    void report() const {
        if (callback) {
            auto value = totalSteps == 0
                             ? 1.0
                             : static_cast<double>(currentStep) / static_cast<double>(totalSteps);
            callback(std::clamp(value, 0.0, 1.0));
        }
    }

    void step() {
        currentStep++;
        report();
    }
};

// Layers

Tensor attention(const TextModel& model,
                 const TextModel::AttentionLayer& layer,
                 const TensorV& x,
                 uint tokenCount,
                 Generator::KVCache::Entry& cache) {
    if (layer.query_norm || layer.key_norm) {
        throw std::invalid_argument("attention() does not support query_norm or key_norm");
    }
    auto sKV = tokenCount + x.shape[0];

    auto z = rmsNorm(layer.norm, x, model.normEpsilon);
    auto query = reshape(projection(layer.query, z),  //
                         {x.shape[0], model.dAttentionKV, model.dAttentionQ, model.dAttentionHead});
    auto key = reshape(projection(layer.key, z),  //
                       {x.shape[0], model.dAttentionKV, model.dAttentionHead});
    auto value = reshape(projection(layer.value, z), key.shape);
    query = rotate(std::move(query), model.ropeAngularFrequency, tokenCount);
    key = rotate(std::move(key), model.ropeAngularFrequency, tokenCount);
    assign(slice0(cache.key, tokenCount, sKV), key);
    assign(slice0(cache.value, tokenCount, sKV), value);

    auto mix = reshape(attention(std::move(query), slice0(cache.key, 0, sKV),
                                 slice0(cache.value, 0, sKV), /*causal*/ true),
                       {x.shape[0], model.dAttentionKV * model.dAttentionQ * model.dAttentionHead});
    return projection(layer.output, mix);
}

Tensor crossAttention(const TextModel& model,
                      const TextModel::AttentionLayer& layer,
                      const TensorV& x,
                      Generator::KVCache::Entry& cache) {
    auto z = rmsNorm(layer.norm, x, model.normEpsilon);
    auto query = reshape(projection(layer.query, z),  //
                         {x.shape[0], model.dAttentionKV, model.dAttentionQ, model.dAttentionHead});
    query = rmsNorm(*layer.query_norm, query, model.normEpsilon);
    auto mix = reshape(attention(std::move(query), cache.key, cache.value, /*causal*/ false),
                       {x.shape[0], model.dAttentionKV * model.dAttentionQ * model.dAttentionHead});
    return projection(layer.output, mix);
}

Tensor mlp(const TextModel& model, const TextModel::MLPLayer& layer, const TensorV& x) {
    auto z = rmsNorm(layer.norm, x, model.normEpsilon);
    auto up = projection(layer.up, z);
    auto gate = projection(layer.gate, z);
    z = swiGlu(std::move(up), gate);
    return projection(layer.down, z);
}

Tensor visionAttention(const VisionModel& model,
                       const VisionModel::AttentionLayer& layer,
                       const TensorV& x) {
    auto z = layerNorm(layer.norm.weight, layer.norm.bias, x, model.normEpsilon);
    auto query = reshape(projection(layer.query, z),  //
                         {x.shape[0], model.dAttentionQkv, 1, model.dAttentionHead});
    auto key = reshape(projection(layer.key, z),  //
                       {x.shape[0], model.dAttentionQkv, model.dAttentionHead});
    auto value = reshape(projection(layer.value, z), key.shape);
    auto mix = reshape(attention(std::move(query), key, value, /*causal*/ false),
                       {x.shape[0], model.dAttentionQkv * model.dAttentionHead});
    return projection(layer.output, mix);
}

Tensor visionMlp(const VisionModel& model, const VisionModel::MLPLayer layer, const TensorV& x) {
    auto z = layerNorm(layer.norm.weight, layer.norm.bias, x, model.normEpsilon);
    z = projection(layer.up.weight, z);
    z = broadcastAdd(std::move(z), layer.up.bias);
    z = gelu(std::move(z));
    z = projection(layer.down.weight, z);
    z = broadcastAdd(std::move(z), layer.down.bias);
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

    auto result = empty<bf16>({nPatch * nPatch, dChannel * dPatch * dPatch});
    auto ptr = data<bf16>(result);
    const auto nStride = dChannel * dPatch * dPatch;
    const auto cStride = dPatch * dPatch;
    for (auto n = 0u; n < nPatch * nPatch; ++n) {
        for (auto i = 0u; i < dPatch * dPatch; ++i) {
            for (auto c = 0u; c < dChannel; ++c) {
                auto x = (n % nPatch) * dPatch + (i % dPatch);
                auto y = (n / nPatch) * dPatch + (i / dPatch);
                auto px = resized.data[y * (resized.width * dChannel) + x * dChannel + c];
                ptr[n * nStride + c * cStride + i] =
                    bf16((px / 255.0f - model.imageMean[c]) / model.imageStd[c]);
            }
        }
    }
    return result;
}

uint nextToken(Generator& g, const TensorV& logits) {
    return sample(indexLeading(logits, {logits.shape[0] - 1}), g.options.temperature,
                  g.options.topK, g.options.topP, g.rng);
}

void resetCache(Generator& g, uint dSequenceMax) {
    g.kvCache.dSequence = 0;
    g.kvCache.dSequenceMax = dSequenceMax;
    g.kvCache.entries.clear();
    const auto& textModel = g.model.textModel;
    for (auto i = 0u; i < textModel.dLayers; ++i) {
        g.kvCache.entries.push_back(
            {empty<bf16>({dSequenceMax, textModel.dAttentionKV, textModel.dAttentionHead}),
             empty<bf16>({dSequenceMax, textModel.dAttentionKV, textModel.dAttentionHead})});
    }
}

void forward(Generator& g, const std::vector<uint>& tokens, ProgressTracker& progress) {
    if (g.kvCache.dSequenceMax < g.kvCache.dSequence + tokens.size()) {
        std::ostringstream err;
        err << "Over-full KV cache of maximum size " << g.kvCache.dSequenceMax << " tokens";
        throw std::runtime_error(err.str());
    }
    auto& model = g.model.textModel;
    auto x = embeddingLookup(model.embedTokens, tokens);
    for (auto i = 0u; i < model.dLayers; ++i) {
        auto xattn =
            std::find(model.crossAttentionLayers.begin(), model.crossAttentionLayers.end(), i);
        if (xattn != model.crossAttentionLayers.end()) {
            auto xi = static_cast<size_t>(xattn - model.crossAttentionLayers.begin());
            if (g.crossAttentionCache) {
                x = add(std::move(x), crossAttention(model, model.layers[i].attention, x,
                                                     g.crossAttentionCache->entries[xi]));
            }
        } else {
            x = add(std::move(x), attention(model, model.layers[i].attention, x,
                                            g.kvCache.dSequence, g.kvCache.entries[i]));
        }
        x = add(std::move(x), mlp(model, model.layers[i].mlp, x));
        progress.step();
    }
    x = clone(slice0(x, x.shape[0] - 1, x.shape[0]));  // for efficiency, only keep last token
    x = rmsNorm(model.finalNorm, x, model.normEpsilon);
    x = projection(model.predictTokens, x);
    g.kvCache.dSequence += uint(tokens.size());
    g.prevToken = nextToken(g, x);
}

Tensor forwardImage(Generator& g, const TensorV& image, ProgressTracker& progress) {
    auto& model = *g.model.visionModel;

    // Embeddings
    auto x = projection(
        reshape(model.patchEmbedding, {model.dModel, 3 * model.dPatch * model.dPatch}), image);
    // Lookup {aspectRatioID = 0, tileIndex = 0}
    x = add(std::move(x), indexLeading(model.positionalEmbedding, {0, 0}));
    x = concat({unsqueeze(indexLeading(model.classEmbedding, {0, 0}), {0}), x}, 0);
    x = layerNorm(model.layerNormPre.weight, model.layerNormPre.bias, x, model.normEpsilon);

    auto out = tile(model.multiModalProjector.bias, {x.shape[0]});

    // First transformer stack
    for (auto i = 0u; i < model.dLayers0; ++i) {
        auto& layer = model.layers0[i];
        x = add(std::move(x), visionAttention(model, layer.attention, x));
        x = add(std::move(x), visionMlp(model, layer.mlp, x));
        auto tap = std::find(model.outputTaps.begin(), model.outputTaps.end(), i);
        if (tap != model.outputTaps.end()) {
            auto idx = static_cast<uint>(tap - model.outputTaps.begin());
            out = add(std::move(out),
                      projection(indexLeading(model.multiModalProjector.weight, {idx}), x));
        }
        progress.step();
    }
    x = layerNorm(model.layerNormPost.weight, model.layerNormPost.bias, x, model.normEpsilon);
    // Lookup {aspectRatioID = 0, tileIndex = 0}
    x = broadcastAdd(std::move(x), indexLeading(model.tileEmbeddingPost, {0, 0}));

    // Second transformer stack
    for (auto i = 0u; i < model.dLayers1; ++i) {
        auto& layer = model.layers1[i];
        x = add(std::move(x), visionAttention(model, layer.attention, x));
        x = add(std::move(x), visionMlp(model, layer.mlp, x));
        progress.step();
    }
    out = add(std::move(out), projection(indexLeading(model.multiModalProjector.weight,
                                                      {static_cast<uint>(model.outputTaps.size())}),
                                         x));
    return out;
}

Generator::KVCache prepareCrossAttentionCache(const TextModel& model,
                                              const TensorV& imageOut,
                                              ProgressTracker& progress) {
    auto kvShape = std::vector<uint>{imageOut.shape[0], model.dAttentionKV, model.dAttentionHead};
    Generator::KVCache cache;
    cache.dSequence = cache.dSequenceMax = imageOut.shape[0];
    cache.entries.reserve(model.crossAttentionLayers.size());
    for (auto i : model.crossAttentionLayers) {
        const auto& layer = model.layers[i];
        auto key = reshape(projection(layer.attention.key, imageOut), kvShape);
        key = rmsNorm(*layer.attention.key_norm, key, model.normEpsilon);
        auto value = reshape(projection(layer.attention.value, imageOut), kvShape);
        cache.entries.push_back({std::move(key), std::move(value)});
        progress.step();
    }
    return cache;
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
                                            const Options& options,
                                            const ProgressCallback& progressCallback) {
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
    }

    auto progressSteps = model.textModel.dLayers;
    if (image) {
        progressSteps += model.visionModel->dLayers0;
        progressSteps += model.visionModel->dLayers1;
        progressSteps += uint(model.textModel.crossAttentionLayers.size());
    }
    ProgressTracker progress(progressSteps, progressCallback);

    // Handle image
    if (image) {
        auto imageOut = forwardImage(*this, preprocess(*model.visionModel, *image), progress);
        crossAttentionCache = prepareCrossAttentionCache(model.textModel, imageOut, progress);
    } else {
        crossAttentionCache.reset();
    }

    // Handle text
    std::vector<uint> tokens = {model.textModel.beginOfTextID};
    if (image) {
        tokens.push_back(*model.textModel.imageID);
    }
    auto nSpecial = tokens.size();
    auto encoded = model.textModel.tokenizer.encode(prefix);
    tokens.insert(tokens.end(), encoded.begin(), encoded.end());
    resetCache(*this, uint(tokens.size() + options.maxGeneratedTokens));
    forward(*this, tokens, progress);

    // Return tokens, including prompt
    if (this->prevToken != model.textModel.endOfTextID) {
        tokens.push_back(this->prevToken);
    }
    std::vector<std::string> stringTokens;
    stringTokens.reserve(tokens.size() - 1);
    std::transform(tokens.begin() + static_cast<long>(nSpecial), tokens.end(),
                   std::back_inserter(stringTokens),
                   [&](uint t) { return model.textModel.tokenizer.decode({t}); });
    return stringTokens;
}

std::string Generator::generate() {
    if (kvCache.dSequence == kvCache.dSequenceMax) {
        return "";
    }
    ProgressTracker progress(model.textModel.dLayers, NoProgressCallback);
    forward(*this, {prevToken}, progress);
    return (prevToken == model.textModel.endOfTextID)
               ? ""
               : model.textModel.tokenizer.decode({prevToken});
}

}  // namespace squash
