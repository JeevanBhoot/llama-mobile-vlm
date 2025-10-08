#include <random>
#include <regex>

#include "tests/tests.hpp"

using namespace squash;

TEST_CASE("squash::impl::regexUnicodeToModifiedECMA", "[lib]") {
    REQUIRE(impl::regexUnicodeToModifiedECMA(R"((?i:foo)|\p{L}+|[-\p{N}_#]+)") ==
            R"((foo)|[[:alpha:]]+|[-[:digit:]_#]+)");
    // Escaped \[ and \] shouldn't count as nesting
    REQUIRE(impl::regexUnicodeToModifiedECMA(R"(\[\p{L}\])") == R"(\[[[:alpha:]]\])");

    // This is the pattern we expect for Llama 3.2
    std::regex pattern(impl::regexUnicodeToModifiedECMA(
        R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3})"
        R"(| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)"));

    std::string s = "This ISN'T my number:     1234567890.\n\nSuperfluously λ \x7f!!!!!";
    std::vector<std::string> expected({"This", " ISN", "'T", " my", " number", ":", "    ", " ",
                                       "123", "456", "789", "0", ".\n\n", "Superfluously", " λ",
                                       " \x7f!!!!!"});
    std::vector<std::string> parts(std::sregex_token_iterator(s.begin(), s.end(), pattern),
                                   std::sregex_token_iterator());
    REQUIRE_THAT(parts, Catch::Matchers::Equals(expected));
}

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

TestModelConfig TestModelConfig::smallLLM() {
    return {.text{
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

TestModelConfig TestModelConfig::smallVLM() {
    auto c = smallLLM();
    c.vision = TestModelConfig::Vision{
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

Model createTestModel(const TestModelConfig& c) {
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
    std::default_random_engine rng(12345u);
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
    REQUIRE(static_cast<ulong>(ptr - buffer) == nParameters);
    return m;
}

}  // namespace squash

namespace {

std::string generate(Model& model,
                     const std::string& prompt,
                     const std::optional<squash::Image>& image,
                     unsigned count) {
    Generator generator(model);
    auto text = prompt;
    auto prefillOut = generator.prefill(prompt, image, Generator::Options::greedy(count));
    REQUIRE(prefillOut.size() == 4u);
    text += prefillOut.back();
    for (auto i = 0u; i < count; ++i) {
        text += generator.generate();
    }
    REQUIRE(generator.generate() == "");
    return text;
}

}  // namespace

TEST_CASE("smallLLM", "[lib]") {
    for (bool tiedEmbeddings : {false, true}) {
        auto config = TestModelConfig::smallLLM();
        config.text.tiedEmbeddings = tiedEmbeddings;
        auto model = createTestModel(config);

        // Empirical match - sensitive to seed
        const auto expected =
            tiedEmbeddings ? "_10_20_30_21_113_21_113_21_113" : "_10_20_30_126_354_54_343_170_255";
        REQUIRE(generate(model, "_10_20_30", std::nullopt, 5) == expected);
    }
}

TEST_CASE("smallVLM", "[lib]") {
    auto config = TestModelConfig::smallVLM();
    config.vision->dLayers0 = 0;
    config.vision->dLayers1 = 0;
    auto model = createTestModel(config);

    auto dImage = model.visionModel->dImage;
    squash::Image image(dImage, dImage, std::vector<uint8_t>(dImage * dImage * 3, 0));
    std::default_random_engine rng(102030u);
    for (auto& v : image.data) {
        v = std::uniform_int_distribution<uint8_t>(0, 255)(rng);
    }

    // Empirical match - sensitive to seed
    REQUIRE(generate(model, "_10_20_30", std::move(image), 5) == "_10_20_30_7_363_7_363_7_363");
}
