#include "bench_model.hpp"

#include <sstream>

using namespace squash;

namespace {

std::vector<float> ropeAngularFrequency(uint dHead) {
    std::vector<float> freq;
    for (auto i = 0u; i < dHead; i += 2) {
        freq.push_back(std::pow(1e4f, -float(i) / float(dHead)));
    }
    return freq;
}

ulong countParameters(const Model& model) {
    auto& tm = model.textModel;
    auto tEmbed = tm.dVocab * tm.dModel;
    auto tAttn = tm.dModel                                                               // norm
                 + 2 * tm.dModel * tm.dAttentionKV * tm.dAttentionQ * tm.dAttentionHead  // q+o
                 + 2 * tm.dModel * tm.dAttentionKV * tm.dAttentionHead;                  // k+v
    auto tCrossAttnNorm = 2 * tm.dAttentionHead;
    auto tMlp = tm.dModel + 3 * tm.dModel * tm.dMLP;
    auto tOutput = tm.dModel + (!tm.tiedEmbeddings) * tm.dVocab * tm.dModel;
    auto total = tEmbed + tm.dLayers * (tAttn + tMlp) + tOutput +
                 tCrossAttnNorm * tm.crossAttentionLayers.size();

    if (model.visionModel) {
        auto& vm = *model.visionModel;
        auto nPatch = (vm.dImage / vm.dPatch) * (vm.dImage / vm.dPatch);
        auto vPatchEmbed = 3 * vm.dPatch * vm.dPatch * vm.dModel;
        auto vPosEmbed = 8 * 4 * nPatch * vm.dModel;  // 8 aspect ratios, 4 tile indices
        auto vClassEmbed = 8 * 4 * vm.dModel;
        auto vNorms = 2 * 2 * vm.dModel;  // {pre, post}, {weight, bias}
        auto vAttn = 2 * vm.dModel + 4 * vm.dModel * vm.dAttentionQkv * vm.dAttentionHead;
        auto vMlp = 2 * vm.dModel              // norm
                    + 2 * vm.dModel * vm.dMlp  // weights
                    + vm.dMlp + vm.dModel;     // biases
        auto vTileEmbedPost = 8 * 4 * vm.dModel;
        auto vOutput =
            (static_cast<ulong>(vm.outputTaps.size()) + 1) * vm.dModel * tm.dModel + tm.dModel;

        total += vPatchEmbed + vPosEmbed + vClassEmbed + vNorms + vTileEmbedPost + vOutput +
                 (vm.dLayers0 + vm.dLayers1) * (vAttn + vMlp);
    }
    return total;
}

}  // namespace

namespace squash {

Dummy::Config Dummy::smallLLM() {
    return {.seed = 0xf3355e3e3a92fe03,
            .text =
                {
                    .dLayers = 3,
                    .dVocab = 384 + 2,
                    .dModel = 128,
                    .dMLP = 512,
                    .dAttentionHead = 64,
                    .dAttentionQ = 5,
                    .dAttentionKV = 7,
                    .tiedEmbeddings = true,
                    .crossAttentionLayers = {},
                },
            .vision = std::nullopt};
}

Dummy::Config Dummy::smallVLM() {
    auto c = smallLLM();
    c.vision = {
        .dImage = 70,
        .dPatch = 10,
        .dLayers0 = 4,
        .dLayers1 = 3,
        .dModel = 96,
        .dMlp = 256,
        .dAttentionHead = 32,
        .dAttentionQkv = 4,
        .outputTaps = {1, 3},
    };
    c.text.crossAttentionLayers = {1, 2};
    return c;
}

Model Dummy::createModel(const Dummy::Config& c) {
    // Create model
    std::vector<std::string> vocab;
    for (auto i = 0u; i < c.text.dVocab - 2 - c.vision.has_value(); ++i) {
        std::ostringstream token;
        token << "_" << i;
        vocab.push_back(token.str());
    }
    Model m{
        .textModel{
            // Config
            .dLayers = c.text.dLayers,
            .dVocab = c.text.dVocab,
            .dModel = c.text.dModel,
            .dMLP = c.text.dMLP,
            .dAttentionHead = c.text.dAttentionHead,
            .dAttentionQ = c.text.dAttentionQ,
            .dAttentionKV = c.text.dAttentionKV,
            .dSequenceMax = 1024,
            .normEpsilon = 1e-5f,
            .ropeAngularFrequency = ropeAngularFrequency(c.text.dAttentionHead),
            .tiedEmbeddings = c.text.tiedEmbeddings,
            .crossAttentionLayers = c.text.crossAttentionLayers,
            // Parameters
            .embedTokens = {},
            .layers = {},
            .finalNorm = {},
            .predictTokens = {},
            // Vocab
            .tokenizer = Tokenizer(std::regex("_[0-9]+"), {}, std::vector<std::string>(vocab)),
            .beginOfTextID = uint(vocab.size()),
            .endOfTextID = uint(vocab.size() + 1),
            .imageID =
                c.vision.has_value() ? std::make_optional(uint(vocab.size() + 2)) : std::nullopt,
        },
        .visionModel = {},
        // Metadata
        .source = "test",
        .created = "",
        .alignment = tensor::DefaultAlignment,
        ._data = tensor::Buffer(0),
    };
    if (c.vision) {
        m.visionModel = {
            // Config
            .imageMean = std::vector<float>(3, 0.5f),
            .imageStd = std::vector<float>(3, 0.5f),
            .dImage = c.vision->dImage,
            .dPatch = c.vision->dPatch,
            .dLayers0 = c.vision->dLayers0,
            .dLayers1 = c.vision->dLayers1,
            .dModel = c.vision->dModel,
            .dMlp = c.vision->dMlp,
            .dAttentionHead = c.vision->dAttentionHead,
            .dAttentionQkv = c.vision->dAttentionQkv,
            .normEpsilon = 1e-5f,
            .outputTaps = c.vision->outputTaps,
            // Parameters
            .patchEmbedding = {},
            .positionalEmbedding = {},
            .classEmbedding = {},
            .layerNormPre = {},
            .layers0 = {},
            .layerNormPost = {},
            .tileEmbeddingPost = {},
            .layers1 = {},
            .multiModalProjector = {},
        };
    }

    // Create buffer
    auto nParameters = countParameters(m);
    m._data = tensor::Buffer(sizeof(bf16) * nParameters);
    auto buffer = reinterpret_cast<bf16*>(m._data.get());
    std::default_random_engine rng(c.seed);
    for (auto i = 0u; i < nParameters; ++i) {
        buffer[i] = floatToBf16(std::normal_distribution<float>(0, 0.02f)(rng));
    }

    // Create tensor views
    auto ptr = buffer;
    auto allocate = [&](const tensor::Shape& shape) {
        auto t = tensor::TensorV{ptr, shape};
        ptr += tensor::prod(shape);
        if (static_cast<ulong>(ptr - buffer) > nParameters) {
            throw std::logic_error("Parameter buffer overflow - check nParameters");
        }
        return t;
    };

    // Text model
    auto& tm = m.textModel;
    tm.embedTokens = allocate({tm.dVocab, tm.dModel});
    for (auto n = 0u; n < tm.dLayers; ++n) {
        auto crossAttention =
            std::find(tm.crossAttentionLayers.begin(), tm.crossAttentionLayers.end(), n) !=
            tm.crossAttentionLayers.end();
        tm.layers.push_back(
            {{
                 .norm = allocate({tm.dModel}),
                 .query =
                     allocate({tm.dAttentionKV * tm.dAttentionQ * tm.dAttentionHead, tm.dModel}),
                 .key = allocate({tm.dAttentionKV * tm.dAttentionHead, tm.dModel}),
                 .value = allocate({tm.dAttentionKV * tm.dAttentionHead, tm.dModel}),
                 .output =
                     allocate({tm.dModel, tm.dAttentionKV * tm.dAttentionQ * tm.dAttentionHead}),
                 .query_norm = crossAttention ? std::make_optional(allocate({tm.dAttentionHead}))
                                              : std::nullopt,
                 .key_norm = crossAttention ? std::make_optional(allocate({tm.dAttentionHead}))
                                            : std::nullopt,
             },
             {
                 .norm = allocate({tm.dModel}),
                 .up = allocate({tm.dMLP, tm.dModel}),
                 .gate = allocate({tm.dMLP, tm.dModel}),
                 .down = allocate({tm.dModel, tm.dMLP}),
             }});
    }
    tm.finalNorm = allocate({tm.dModel});
    if (tm.tiedEmbeddings) {
        tm.predictTokens = tm.embedTokens;
    } else {
        tm.predictTokens = allocate({tm.dVocab, tm.dModel});
    }

    // Vision model
    if (m.visionModel) {
        auto& vm = *m.visionModel;
        auto nPatch = vm.dImage / vm.dPatch;
        vm.patchEmbedding = allocate({vm.dModel, 3, vm.dPatch, vm.dPatch});
        vm.positionalEmbedding = allocate({8, 4, nPatch * nPatch, vm.dModel});
        vm.classEmbedding = allocate({8, 4, vm.dModel});
        vm.layerNormPre = {allocate({vm.dModel}), allocate({vm.dModel})};
        auto allocateLayer = [&] {
            return VisionModel::Layer{
                {
                    .norm = {.weight = allocate({vm.dModel}), .bias = allocate({vm.dModel})},
                    .query = allocate({vm.dAttentionQkv * vm.dAttentionHead, vm.dModel}),
                    .key = allocate({vm.dAttentionQkv * vm.dAttentionHead, vm.dModel}),
                    .value = allocate({vm.dAttentionQkv * vm.dAttentionHead, vm.dModel}),
                    .output = allocate({vm.dModel, vm.dAttentionQkv * vm.dAttentionHead}),
                },
                {
                    .norm = {.weight = allocate({vm.dModel}), .bias = allocate({vm.dModel})},
                    .up = {.weight = allocate({vm.dMlp, vm.dModel}), .bias = allocate({vm.dMlp})},
                    .down = {.weight = allocate({vm.dModel, vm.dMlp}),
                             .bias = allocate({vm.dModel})},
                }};
        };
        for (auto n = 0u; n < vm.dLayers0; ++n) {
            vm.layers0.push_back(allocateLayer());
        }
        vm.layerNormPost = {allocate({vm.dModel}), allocate({vm.dModel})};
        vm.tileEmbeddingPost = allocate({8, 4, vm.dModel});
        for (auto n = 0u; n < vm.dLayers1; ++n) {
            vm.layers1.push_back(allocateLayer());
        }
        // Note: output dimension is from the text model (tm.dModel)
        vm.multiModalProjector = {
            allocate({static_cast<uint>(vm.outputTaps.size()) + 1, tm.dModel, vm.dModel}),
            allocate({tm.dModel})};
    }

    // Final check
    if (static_cast<ulong>(ptr - buffer) != nParameters) {
        std::ostringstream err;
        err << "Parameter count mismatch, expected: " << nParameters
            << ", actual: " << (ptr - buffer) << "\n";
        throw std::logic_error(err.str());
    }
    return m;
}

Image Dummy::createImage(uint dImage, ulong seed) {
    squash::Image image(dImage, dImage, std::vector<uint8_t>(dImage * dImage * 3, 0));
    std::default_random_engine rng(seed);
    for (auto& v : image.data) {
        v = std::uniform_int_distribution<uint8_t>(0, 255)(rng);
    }
    return image;
}

}  // namespace squash
