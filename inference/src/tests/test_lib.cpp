#include <random>
#include <regex>

#include "tests/tests.hpp"

using namespace squash;

TEST_CASE("squash::impl::regexUnicodeToModifiedECMA", "[squash]") {
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
}  // namespace

TEST_CASE("squash::TextGenerator", "[squash]") {
    // Create vocab
    std::vector<std::string> vocab;
    for (auto i = 0u; i < 256 - 2; ++i) {
        std::ostringstream token;
        token << "_" << i;
        vocab.push_back(token.str());
    }
    Model m{
        .textModel{
            // Config
            .dLayers = 3,
            .dVocab = uint(vocab.size() + 2),
            .dModel = 128,
            .dMLP = 512,
            .dAttentionHead = 64,
            .dAttentionQ = 2,
            .dAttentionKV = 2,
            .dSequenceMax = 1024,
            .normEpsilon = 1e-5f,
            .ropeAngularFrequency = ropeAngularFrequency(64),
            .tiedEmbeddings = true,
            .crossAttentionLayers = {},
            // Parameters
            .embedTokens = {},
            .layers = {},
            .finalNorm = {},
            .predictTokens = {},
            // Vocab
            .tokenizer = Tokenizer(std::regex("_[0-9]+"), {}, std::vector<std::string>(vocab)),
            .beginOfTextID = uint(vocab.size()),
            .endOfTextID = uint(vocab.size() + 1),
            .imageID = uint(vocab.size() + 2),
        },
        .visionModel = {},
        // Metadata
        .source = "test",
        .created = "",
        .alignment = DefaultAlignment,
        ._data = Buffer(0),
    };

    // Create buffer
    auto& tm = m.textModel;
    auto totalHeads = tm.dAttentionKV * tm.dAttentionQ;
    auto nParameters = tm.dVocab * tm.dModel                                             // embed
                       + tm.dLayers * (tm.dModel                                         // norm
                                       + 2 * tm.dModel * totalHeads * tm.dAttentionHead  // q+o
                                       + 2 * tm.dModel * tm.dAttentionKV * tm.dAttentionHead  // k+v
                                       + tm.dModel                // norm
                                       + 3 * tm.dModel * tm.dMLP  // mlp
                                       )                          //
                       + tm.dModel;                               // norm
    m._data = Buffer(sizeof(bf16) * nParameters);
    auto buffer = reinterpret_cast<bf16*>(m._data.get());
    std::default_random_engine rng(12345u);
    for (auto i = 0u; i < nParameters; ++i) {
        buffer[i] = convertTruncate<bf16>(std::normal_distribution<float>(0, 0.02f)(rng));
    }

    // Create tensor views
    auto ptr = buffer;
    auto allocate = [&](const std::vector<uint>& shape) {
        auto t = TensorV{ptr, shape};
        ptr += prod(shape);
        return t;
    };
    tm.embedTokens = allocate({tm.dVocab, tm.dModel});
    tm.predictTokens = tm.embedTokens;
    for (auto n = 0u; n < tm.dLayers; ++n) {
        tm.layers.push_back(
            {{
                 .norm = allocate({tm.dModel}),
                 .query = allocate({totalHeads * tm.dAttentionHead, tm.dModel}),
                 .key = allocate({tm.dAttentionKV * tm.dAttentionHead, tm.dModel}),
                 .value = allocate({tm.dAttentionKV * tm.dAttentionHead, tm.dModel}),
                 .output = allocate({tm.dModel, totalHeads * tm.dAttentionHead}),
                 .query_norm = {},
                 .key_norm = {},
             },
             {
                 .norm = allocate({tm.dModel}),
                 .up = allocate({tm.dMLP, tm.dModel}),
                 .gate = allocate({tm.dMLP, tm.dModel}),
                 .down = allocate({tm.dModel, tm.dMLP}),
             }});
    }
    tm.finalNorm = allocate({tm.dModel});
    if (ptr - buffer != nParameters) {
        throw std::logic_error("Wrong number of parameters");
    }

    // Generate from the model
    auto generationCount = 5u;
    Generator generator(m);
    std::string text = "_10_20_30";
    auto prefillOut = generator.prefill(text, {}, Generator::Options::greedy(generationCount));
    REQUIRE(prefillOut.size() == 4u);
    text += prefillOut.back();
    for (auto i = 0u; i < generationCount; ++i) {
        text += generator.generate();
    }
    REQUIRE(generator.generate() == "");
    REQUIRE(text == "_10_20_30_147_30_147_30_147_30");
}
