#pragma once

#include <memory>
#include <regex>
#include <string>
#include <vector>

namespace squash {

struct Tokenizer {
    // Following TikToken, each merge is a single string representing
    // bytes to be merged (there is no space delimiter)
    Tokenizer(const std::regex& preTokenizer,
              const std::vector<std::string>& merges,
              std::vector<std::string>&& vocab);
    Tokenizer(Tokenizer&&);
    ~Tokenizer();

    std::vector<unsigned> encode(const std::string&) const;
    std::string decode(const std::vector<unsigned>&) const;

   private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

/// Implementation (exposed for testing) ///

namespace impl {
void encodeUTF8_2B(uint16_t, std::string&);
uint16_t decodeUTF8_2B(std::string::const_iterator&);
std::string encodeBytesForBPE(const std::string&);
std::string decodeBytesForBPE(const std::string&);
}  // namespace impl

}  // namespace squash
