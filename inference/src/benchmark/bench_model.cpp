#include "bench_model.hpp"
#include "benchmark.hpp"
#include "core/ops.hpp"

#include <omp.h>
#include <sstream>

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

Dummy::Config Dummy::llama1B_LLM() {
    // From https://huggingface.co/meta-llama/Llama-3.2-1B/blob/main/config.json
    return {.seed = 0x12e2c22f229c5fb8,
            .text =
                {
                    .dLayers = 16,
                    .dVocab = 128256,
                    .dModel = 2048,
                    .dMLP = 8192,
                    .dAttentionHead = 64,
                    .dAttentionQ = 4,
                    .dAttentionKV = 8,
                    .tiedEmbeddings = true,
                    .crossAttentionLayers = {},
                },
            .vision = std::nullopt};
}

Dummy::Config Dummy::llama11B_VLM() {
    // From https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct/blob/main/config.json
    return {.seed = 0xdc8c23c61606b2d5,
            .text =
                {
                    .dLayers = 40,
                    .dVocab = 128256,
                    .dModel = 4096,
                    .dMLP = 14336,
                    .dAttentionHead = 128,  // hidden_size / num_attention_heads
                    .dAttentionQ = 4,
                    .dAttentionKV = 8,
                    .tiedEmbeddings = false,
                    .crossAttentionLayers = {3, 8, 13, 18, 23, 28, 33, 38},
                },
            .vision = {{
                .dImage = 560,
                .dPatch = 14,
                .dLayers0 = 32,
                .dLayers1 = 8,
                .dModel = 1280,
                .dMlp = 5120,
                .dAttentionHead = 80,  // hidden_size / attention_heads
                .dAttentionQkv = 16,
                .outputTaps = {3, 7, 15, 23, 30},
            }}};
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
    auto total = tEmbed + tm.dLayers * uint64_t(tAttn + tMlp) + tOutput +
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
                 (vm.dLayers0 + vm.dLayers1) * uint64_t(vAttn + vMlp);
    }
    return total;
}

}  // namespace

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
    ops::randn(buffer, nParameters, 0.02f, c.seed);

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

namespace {

std::string randomPrompt(uint nToken, uint dVocab, ulong seed) {
    std::default_random_engine rng(seed);
    std::ostringstream prompt;
    for (auto i = 0u; i < nToken; ++i) {
        prompt << "_" << std::uniform_int_distribution<uint>(0, dVocab - 3)(rng);
    }
    return prompt.str();
}

void runModelBenchmark(const Dummy::Config& config,
                       uint nPrompt,
                       uint nGenerate,
                       const benchmarking::Report& report) {
    // Create
    Timer createTimer;
    auto model = Dummy::createModel(config);
    auto createTime = createTimer.elapsed();
    report({{"phase", "create"}, {"n_parameters", countParameters(model)}, {"time", createTime}});
    std::cerr << report << "create " << createTime << " s\n";

    // Prefill
    Generator generator(model);
    auto prompt = randomPrompt(nPrompt, model.textModel.dVocab, 0xe28051a583870442);
    auto image =
        config.vision
            ? std::make_optional(Dummy::createImage(config.vision->dImage, 0x9b4f4e68f30f5113))
            : std::nullopt;
    Timer prefillTimer;
    auto out = generator.prefill(prompt, image, Generator::Options::greedy(nGenerate));
    auto prefillTime = prefillTimer.elapsed();
    report({{"phase", "prefill"}, {"prompt_tokens", nPrompt}, {"time", prefillTime}});
    std::cerr << report << "prefill " << prefillTime << " s (" << nPrompt / prefillTime
              << " token/s)\n";

    // Generate
    benchmarking::Benchmark generateBenchmark;
    for (auto i = 0u; i < nGenerate; ++i) {
        auto timer = generateBenchmark.record();
        out.push_back(generator.generate());
    }
    auto result = generateBenchmark.result(0);
    report({{"phase", "generate"},
            {"tokens_per_second", 1 / result.mean},
            {"time", generateBenchmark.times}});
    std::cerr << report << "generate " << 1 / result << " token/s\n";
}

REGISTER_BENCHMARK(text_model_1B)(const benchmarking::Report& report) {
    runModelBenchmark(Dummy::llama1B_LLM(), /*nPrompt*/ 64u, /*nGenerate*/ 64u, report);
}

REGISTER_BENCHMARK(_vision_model_11B)(const benchmarking::Report& report) {
    auto config = Dummy::llama11B_VLM();
    // Reduce image layers for faster benchmark
    // config.vision->dLayers0 = 2;
    // config.vision->dLayers1 = 2;
    // config.vision->outputTaps = {0, 0, 0, 1, 1};  // keep 5
    runModelBenchmark(config, /*nPrompt*/ 64u, /*nGenerate*/ 64u, report);
}

}  // namespace
}  // namespace squash
