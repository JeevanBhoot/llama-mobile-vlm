// Copyright (c) 2026 Graphcore Ltd. All rights reserved.

#include <random>
#include <regex>

#include "benchmark/bench_model.hpp"
#include "tests/common.hpp"

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
        INFO("tiedEmbeddings = " << tiedEmbeddings);
        auto config = Dummy::smallLLM();
        config.text.tiedEmbeddings = tiedEmbeddings;
        auto model = Dummy::createModel(config);

        REQUIRE(generate(model, "_10_20_30", std::nullopt, 5).starts_with("_10_20_30"));
    }
}

TEST_CASE("smallVLM", "[lib]") {
    auto config = Dummy::smallVLM();
    auto model = Dummy::createModel(config);
    auto image = Dummy::createImage(config.vision->dImage, 0x10203040);

    REQUIRE(generate(model, "_10_20_30", std::move(image), 5).starts_with("_10_20_30"));
}
