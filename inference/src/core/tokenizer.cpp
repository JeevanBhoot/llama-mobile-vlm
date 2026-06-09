// Copyright (c) 2026 Graphcore Ltd. All rights reserved.

#include "tokenizer.hpp"

#include <limits>
#include <regex>
#include <sstream>
#include <unordered_map>

namespace squash {

namespace impl {

void encodeUTF8_2B(uint16_t codepoint, std::string& s) {
    if (codepoint <= 0x7f) {
        s.push_back(char(codepoint));
    } else if (codepoint <= 0x7ff) {
        s.append({char(0xc0 + (codepoint >> 6)), char(0x80 + (codepoint & 0x3f))});
    } else {
        std::ostringstream err;
        err << "Cannot encode codepoint 0x" << std::hex << codepoint
            << ", out of supported range <= 0x7ff";
        throw std::runtime_error(err.str());
    }
}

uint16_t decodeUTF8_2B(std::string::const_iterator& it) {
    auto ch = uint8_t(*(it++));
    if ((ch & 0x80) == 0) {  // 1-byte
        return ch & 0x7f;
    } else if ((ch & 0xe0) == 0xc0) {  // 2-byte
        auto ch2 = uint8_t(*(it++));
        if ((ch2 & 0xc0) != 0x80) {  // bad
            std::ostringstream err;
            err << "Bad UTF8 continuation byte " << std::hex << ch2;
            throw std::runtime_error(err.str());
        }
        return uint16_t((uint16_t(ch & 0x1f) << 6) + (ch2 & 0x3f));
    } else {  // 3+ bytes
        std::ostringstream err;
        err << "Cannot decode UTF8 character beginning with " << std::hex << ch
            << " (more than 2 bytes)";
        throw std::runtime_error(err.str());
    }
    return 0;
}

// Map to printable (these rules come from GPT-2)
//   0x00..0x20 => 0x100..0x120
//   0x7f..0xa0 => 0x121..0x142
//   0xad       => 0x143
std::string encodeBytesForBPE(const std::string& s) {
    std::string out;
    for (auto i = 0u; i < s.size(); ++i) {
        auto b = static_cast<uint8_t>(s[i]);
        if (0x00 <= b && b <= 0x20) {
            encodeUTF8_2B(uint16_t(b) - 0x00 + 0x100, out);
        } else if (0x7f <= b && b <= 0xa0) {
            encodeUTF8_2B(uint16_t(b) - 0x7f + 0x121, out);
        } else if (0xad == b) {
            encodeUTF8_2B(uint16_t(0x143), out);
        } else {
            encodeUTF8_2B(uint16_t(b), out);
        }
    }
    return out;
}

std::string decodeBytesForBPE(const std::string& s) {
    std::string out;
    for (auto it = s.cbegin(); it != s.cend();) {
        auto c = decodeUTF8_2B(it);
        if (0x100 <= c && c <= 0x120) {
            out.push_back(char(c - (0x100 - 0x00)));
        } else if (0x121 <= c && c <= 0x142) {
            out.push_back(char(c - (0x121 - 0x7f)));
        } else if (0x143 == c) {
            out.push_back(char(0xad));
        } else if (0x100 <= c) {
            std::ostringstream err;
            err << "Unexpected byte encoding, U+" << std::hex << c;
            throw std::runtime_error(err.str());
        } else {
            out.push_back(char(c));
        }
    }
    return out;
}

}  // namespace impl

namespace {
// Call `func` for each token decoded from `word`
template <class F>
void tokenizeBPE(const std::unordered_map<std::string, uint>& mergeToRank,
                 const std::string& word,
                 F&& func) {
    // Spans represent merged character ranges of `word`,
    // specifically spans[i] :: `word[i .. spans[i].next]`.
    //
    // They form a doubly-linked list from i -> spans[i].next, so we iterate via:
    //   for (auto i = 0u; i != spans[i].next; i = spans[i].next)
    //
    struct TokenSpan {
        uint prev;
        uint next;
        uint mergeRank;
    };
    constexpr auto MaxRank = std::numeric_limits<uint>::max();
    std::vector<TokenSpan> spans;

    // Split into single-character spans
    spans.reserve(word.size() + 1);
    for (auto i = 0u; i < word.size(); ++i) {
        // After encodeBytesForBPE, every character should be 1-2 bytes
        // std::min unnecessary given valid utf8, but defensive against buffer overflow
        auto charLength = std::min(1u + (uint8_t(word[i]) >= 0xc0), uint(word.size()) - i);
        spans.push_back({0u, i + charLength, MaxRank});
    }
    spans.push_back({0u, uint(word.size()), MaxRank});
    for (auto i = 0u; i != spans[i].next; i = spans[i].next) {
        spans[spans[i].next].prev = i;
    }

    // Update the `index` span's mergeRank
    auto updateNextMerge = [&](uint index) {
        auto& span = spans[index];
        auto nextNext = spans[span.next].next;
        if (span.next == nextNext) {
            span.mergeRank = MaxRank;  // span already reaches the end
        } else {
            auto it = mergeToRank.find(word.substr(index, nextNext - index));
            span.mergeRank = (it != mergeToRank.end()) ? it->second : MaxRank;
        }
    };

    // Find next merges
    struct {
        uint rank;
        uint index;
    } nextMerge = {MaxRank, 0u};
    for (auto i = 0u; i != spans[i].next; i = spans[i].next) {
        updateNextMerge(i);
        if (spans[i].mergeRank < nextMerge.rank) {
            nextMerge = {spans[i].mergeRank, i};
        }
    }

    // Iteratively merge
    while (nextMerge.rank != MaxRank) {
        // Apply next merge & update affected merges
        auto index = nextMerge.index;
        spans[index].next = spans[spans[index].next].next;
        spans[spans[index].next].prev = index;
        updateNextMerge(index);
        if (spans[index].prev != index) {
            updateNextMerge(spans[index].prev);
        }

        // Find the next merge
        nextMerge = {MaxRank, 0u};
        for (auto i = 0u; i != spans[i].next; i = spans[i].next) {
            if (spans[i].mergeRank < nextMerge.rank) {
                nextMerge = {spans[i].mergeRank, i};
            }
        }
    }

    // Emit tokens
    for (auto i = 0u; i != spans[i].next; i = spans[i].next) {
        func(word.substr(i, spans[i].next - i));
    }
}
}  // namespace

struct Tokenizer::Impl {
    std::regex preTokenizer;
    // Map a merge rule, with no separator, to the rank of the rule in the list
    // (earlier rules must be applied first)
    std::unordered_map<std::string, uint> mergeToRank;
    const std::vector<std::string> idToToken;
    std::unordered_map<std::reference_wrapper<const std::string>,
                       uint,
                       std::hash<std::string>,
                       std::equal_to<std::string>>
        tokenToID;

    Impl(const std::regex& preTokenizer,
         const std::vector<std::string>& merges,
         std::vector<std::string>&& vocab)
        : preTokenizer(preTokenizer), idToToken(std::move(vocab)) {
        for (auto i = 0u; i < merges.size(); ++i) {
            mergeToRank[merges[i]] = i;
        }
        for (auto i = 0u; i < idToToken.size(); ++i) {
            tokenToID[idToToken[i]] = i;
        }
    }

    std::vector<uint> encode(const std::string& s) const {
        std::vector<uint> tokens;
        for (auto it = std::sregex_token_iterator(s.begin(), s.end(), preTokenizer);
             it != std::sregex_token_iterator(); ++it) {
            auto encoded = impl::encodeBytesForBPE(*it);
            auto match = tokenToID.find(encoded);
            if (match != tokenToID.end()) {
                tokens.push_back(match->second);
            } else {
                tokenizeBPE(mergeToRank, impl::encodeBytesForBPE(*it),
                            [&](const std::string& token) {
                                auto match = tokenToID.find(token);
                                if (match == tokenToID.end()) {
                                    std::ostringstream err;
                                    err << "Could not find token \"" << token << "\"";
                                    throw std::runtime_error(err.str());
                                }
                                tokens.push_back(match->second);
                            });
            }
        }
        return tokens;
    }

    std::string decode(const std::vector<uint>& tokens) const {
        std::string out;
        for (auto token : tokens) {
            if (idToToken.size() <= token) {
                std::ostringstream err;
                err << "Token " << token << " is out of vocab range (" << idToToken.size() << ")";
                throw std::runtime_error(err.str());
            }
            out.append(impl::decodeBytesForBPE(idToToken[token]));
        }
        return out;
    }
};

/// Tokenizer API ///

Tokenizer::Tokenizer(const std::regex& preTokenizer,
                     const std::vector<std::string>& merges,
                     std::vector<std::string>&& vocab)
    : _impl(new Impl(preTokenizer, merges, std::move(vocab))) {}

Tokenizer::Tokenizer(Tokenizer&& other) : _impl(std::move(other._impl)) {}

Tokenizer::~Tokenizer() {}

std::vector<unsigned> Tokenizer::encode(const std::string& text) const {
    return _impl->encode(text);
}

std::string Tokenizer::decode(const std::vector<unsigned>& tokens) const {
    return _impl->decode(tokens);
}

}  // namespace squash
