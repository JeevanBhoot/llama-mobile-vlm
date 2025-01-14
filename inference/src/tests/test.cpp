#include <random>
#include <regex>

#include "tests.hpp"

using namespace squash;

TEST_CASE("squash::bf16ToFloat", "[squash]") {
    REQUIRE(bf16ToFloat(0x4147) == 12.4375f);
    REQUIRE(bf16ToFloat(-0x3f80) == -4.0f);
}

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
        // Metadata
        .source = "test",
        .created = "",
        .alignment = DefaultAlignment,
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
        // Parameters
        .embedTokens = {},
        .layers = {},
        .finalNorm = {},
        // Vocab
        .tokenizer = Tokenizer(std::regex("_[0-9]+"), {}, std::vector<std::string>(vocab)),
        .beginOfTextID = uint(vocab.size()),
        .endOfTextID = uint(vocab.size() + 1),
        // Data
        ._data = Buffer(0),
    };

    // Create buffer
    auto totalHeads = m.dAttentionKV * m.dAttentionQ;
    auto nParameters = m.dVocab * m.dModel                                                // embed
                       + m.dLayers * (m.dModel                                            // norm
                                      + 2 * m.dModel * totalHeads * m.dAttentionHead      // q+o
                                      + 2 * m.dModel * m.dAttentionKV * m.dAttentionHead  // k+v
                                      + m.dModel                                          // norm
                                      + 3 * m.dModel * m.dMLP                             // mlp
                                      )                                                   //
                       + m.dModel;                                                        // norm
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
    m.embedTokens = allocate({m.dVocab, m.dModel});
    for (auto n = 0u; n < m.dLayers; ++n) {
        m.layers.push_back(
            {{
                 .norm = allocate({m.dModel}),                                      //
                 .query = allocate({totalHeads * m.dAttentionHead, m.dModel}),      //
                 .key = allocate({m.dAttentionKV * m.dAttentionHead, m.dModel}),    //
                 .value = allocate({m.dAttentionKV * m.dAttentionHead, m.dModel}),  //
                 .output = allocate({m.dModel, totalHeads * m.dAttentionHead}),     //
             },
             {
                 .norm = allocate({m.dModel}),          //
                 .up = allocate({m.dMLP, m.dModel}),    //
                 .gate = allocate({m.dMLP, m.dModel}),  //
                 .down = allocate({m.dModel, m.dMLP}),  //
             }});
    }
    m.finalNorm = allocate({m.dModel});
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
#ifdef ANDROID
    REQUIRE(text == "_10_20_30_147_30_147_30_147_30");
#else
    REQUIRE(text == "_10_20_30_166_90_83_90_83_90");
#endif  // !ANDROID
}
