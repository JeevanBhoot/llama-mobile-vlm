#include "core/tokenizer.hpp"
#include "tests/common.hpp"

using namespace squash;

TEST_CASE("squash::impl::encodeUTF8_2B", "[squash]") {
    auto checkEncodeDecode = [](uint16_t codepoint, const std::string& expected) {
        std::string encoded;
        impl::encodeUTF8_2B(codepoint, encoded);
        REQUIRE(encoded == expected);
        auto it = encoded.cbegin();
        auto decoded = impl::decodeUTF8_2B(it);
        REQUIRE(decoded == codepoint);
        REQUIRE(it == encoded.end());
    };
    checkEncodeDecode(0x20, " ");
    checkEncodeDecode(0x7f, "\x7f");
    checkEncodeDecode(0x80, "\xc2\x80");
    checkEncodeDecode(0x143, "\xc5\x83");
}

TEST_CASE("squash::impl::encodeBytesForBPE", "[squash]") {
    auto checkEncodeDecode = [](const std::string& bytes, const std::string& expected) {
        auto chars = impl::encodeBytesForBPE(bytes);
        REQUIRE(chars == expected);
        auto reconstructed = impl::decodeBytesForBPE(chars);
        REQUIRE(reconstructed == bytes);
    };
    // Plain encoding
    checkEncodeDecode("A", "A");
    checkEncodeDecode("λ", "\xc3\x8e\xc2\xbb");
    // Mapping via each rule
    checkEncodeDecode(" ", "Ġ");
    checkEncodeDecode("\xa0", "\xc5\x82");
    checkEncodeDecode("\xad", "\xc5\x83");
    // Multiple characters
    checkEncodeDecode("I am_λ ", "IĠam_\xc3\x8e\xc2\xbbĠ");
}

TEST_CASE("squash::Tokenizer", "[squash]") {
    std::regex preTokenizer(" |[[:alpha:]]+");
    std::vector<std::string> vocab({"Ġ", "a", "c", "h", "t", "s",  //
                                    "at", "ha", "cat", "cats", "tat"});
    std::vector<std::string> mergeRules({"at", "ha", "cats", "cat"});

    // - "hat" tests precedence "at" before "ha"
    // - " " tests byte encoding mapping
    // - "catcats" tests multiple merges
    // - "tat" tests whole-token matching, without using merge rules
    std::string original = "a hat haa catcats tat";
    std::vector<std::string> expected(
        {"a", "Ġ", "h", "at", "Ġ", "ha", "a", "Ġ", "cat", "cats", "Ġ", "tat"});

    Tokenizer tokenizer(preTokenizer, mergeRules, std::vector<std::string>(vocab));
    auto tokens = tokenizer.encode(original);
    for (auto i = 0u; i < std::min(expected.size(), tokens.size()); ++i) {
        REQUIRE(vocab.at(tokens[i]) == expected[i]);
    }
    REQUIRE(tokens.size() == expected.size());

    auto decoded = tokenizer.decode(tokens);
    REQUIRE(decoded == original);
}
