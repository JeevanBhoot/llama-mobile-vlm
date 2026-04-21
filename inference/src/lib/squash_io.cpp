#include <sys/sysinfo.h>
#include <iostream>
#define JSON_USE_IMPLICIT_CONVERSIONS 0
#include <json.hpp>
#include <regex>
#include <sstream>
#include <unordered_map>

#include "squash.hpp"

namespace squash {

namespace impl {
// Convert a subset of unicode/Python regexes to C++ (modified ECMAScript) compatible
// regexes. In particular, remove "?i:", and convert \p{N} -> [:digit:],
// \p{L} -> [:alpha:].
std::string regexUnicodeToModifiedECMA(const std::string& original) {
    std::string modified = std::regex_replace(original, std::regex(R"(\?i:)"), "");

    // Add "fat" character classes "[[:NAME:]]", which we will slim down later
    modified = std::regex_replace(modified, std::regex(R"(\\p\{N\})"), "[[:digit:]]");
    modified = std::regex_replace(modified, std::regex(R"(\\p\{L\})"), "[[:alpha:]]");

    // Scan through the pattern, converting [[:NAME:]] -> [:NAME:] whenever it's
    // already contained in a character class
    auto i = 0u;
    auto nesting = 0u;
    while (i < modified.size()) {
        auto next2 = modified.substr(i, 2);
        auto next3 = modified.substr(i, 3);
        if (next2 == "\\[" || next2 == "\\]") {
            i += 2;
        } else if (next3 == "[[:" || next3 == ":]]") {
            if (nesting) {
                modified.erase(i + 1, 1);
                i += 2;
            } else {
                i += 3;
            }
        } else {
            nesting += (modified[i] == '[');
            nesting -= (modified[i] == ']');
            i++;
        }
    }
    return modified;
}
}  // namespace impl

namespace {
constexpr auto Magic = 0x7471732eu;
constexpr auto Version = 2u;
constexpr auto MaxRamProportion = 0.75f;
constexpr auto BufferChunkSize = 4096u;

using json = nlohmann::json;

ulong align(ulong index, ulong alignment) {
    return alignment * ((index + alignment - 1) / alignment);
}

ulong maxTensorEndOffset(const json& entry, ulong alignment) {
    auto maxEnd = ulong(0);
    for (const auto& p : entry.items()) {
        const auto& tensor = p.value();
        for (const auto* t : {&tensor, tensor.contains("scale") ? &tensor.at("scale") : nullptr,
                              tensor.contains("table") ? &tensor.at("table") : nullptr}) {
            if (t && t->contains("data_offsets")) {
                auto end = (*t).at("data_offsets")[1].template get<ulong>();
                maxEnd = std::max(maxEnd, align(end, alignment));
            }
        }
    }
    return maxEnd;
}

ulong s3d8LutStorageSize(const json& entry, ulong alignment) {
    ulong total = 0;
    for (const auto& p : entry.items()) {
        if (p.value().at("dtype").template get<std::string>() == "S3D8") {
            total += align(3 * 64 * sizeof(int8_t), alignment);
        }
    }
    return total;
}

void checkRAM(ulong bufferSize) {
    struct sysinfo info;
    if (sysinfo(&info)) {
        throw std::runtime_error("Could not get sysinfo (to get total RAM)");
    }
    auto bufferSizeGB = static_cast<float>(bufferSize) / 1e9f;
    auto ramGB = static_cast<float>(info.totalram) / 1e9f;
    if (ramGB * MaxRamProportion < bufferSizeGB) {
        std::ostringstream err;
        err << "Cannot load model - would consume " << bufferSizeGB
            << " GB of RAM for parameters, which is more than " << MaxRamProportion << " * "
            << ramGB << " GB available";
        throw std::runtime_error(err.str());
    }
}

Tokenizer loadTokenizer(const json& j) {
    std::regex preTokenizer(
        impl::regexUnicodeToModifiedECMA(j.at("pre_tokenizer").template get<std::string>()));
    auto merges = j.at("merges").template get<std::vector<std::string>>();
    auto vocab = j.at("vocab").template get<std::vector<std::string>>();
    return Tokenizer(preTokenizer, merges, std::move(vocab));
}

struct ParamBuffer {
    ulong alignment;
    const tensor::Buffer& buffer;
    ulong nextS3D8LutOffset;
    std::unordered_map<std::string, int8_t*> s3d8Luts;
};

struct TensorLoader {
    const json& header;
    ParamBuffer& params;
    std::string prefix;

    std::string fullName(const std::string& name) const {
        return prefix.empty() ? name : (prefix + "." + name);
    }

    std::optional<tensor::TensorV> get(const std::string& name) const {
        if (header.contains(fullName(name))) {
            return std::make_optional((*this)(name));
        }
        return std::nullopt;
    }

    // Create a scoped TensorLoader with `name` appended to the `prefix`
    TensorLoader operator[](const std::string& name) const {
        return TensorLoader{header, params, this->fullName(name)};
    }

    int8_t* loadLut(const json& entry, const std::string& fullName) const {
        if (auto dtype = entry.at("dtype").template get<std::string>(); dtype != "INT8") {
            std::ostringstream err;
            err << "S3D8 Tensor " << fullName << " expected INT8 table, actual dtype = " << dtype;
            throw std::runtime_error(err.str());
        }
        if (auto shape = entry.at("shape").template get<std::vector<uint>>();
            shape != std::vector<uint>({32u, 3u})) {
            std::ostringstream err;
            err << "S3D8 Tensor " << fullName << " has bad table shape = " << dump(shape)
                << ", expected {32, 3}";
            throw std::runtime_error(err.str());
        }
        if (auto it = params.s3d8Luts.find(fullName); it != params.s3d8Luts.end()) {
            return it->second;
        }
        auto* lut = reinterpret_cast<int8_t*>(params.buffer.get() + params.nextS3D8LutOffset);
        const auto offset = entry.at("data_offsets")[0].template get<ulong>();
        tensor::_data::ChannelS3D8::expandLut(
            reinterpret_cast<int8_t*>(params.buffer.get() + offset), lut);
        params.s3d8Luts.emplace(fullName, lut);
        params.nextS3D8LutOffset += align(3 * 64 * sizeof(int8_t), params.alignment);
        return lut;
    }

    bf16* loadScale(const json& entry,
                    const std::string& fullName,
                    const std::vector<uint>& shape) const {
        auto scaleTensor = loadTensor(entry, fullName + "#scale");
        if (scaleTensor.shape != std::vector<uint>({shape[0], 1u})) {
            std::ostringstream err;
            err << "Tensor " << fullName << " has bad scale shape = " << dump(scaleTensor.shape)
                << ", expected {" << shape[0] << ", 1}";
            throw std::runtime_error(err.str());
        }
        if (!std::holds_alternative<tensor::_data::Flat<bf16>>(scaleTensor.data)) {
            std::ostringstream err;
            err << "Tensor " << fullName
                << " expected BF16 scale, actual type_index = " << scaleTensor.data.index();
            throw std::runtime_error(err.str());
        }
        return std::get<tensor::_data::Flat<bf16>>(scaleTensor.data).data;
    }

    // Load the given named tensor (full name "{prefix}.{name}")
    tensor::TensorV loadTensor(const json& entry, const std::string& fullName) const {
        auto dtype = entry.at("dtype").template get<std::string>();
        auto offset = entry.at("data_offsets")[0].template get<ulong>();
        if (offset % params.alignment) {
            std::ostringstream err;
            err << "Tensor " << fullName << " is misaligned, offset = " << offset;
            throw std::runtime_error(err.str());
        }
        auto shape = entry.at("shape").template get<std::vector<uint>>();
        if (dtype == "BF16") {
            return {
                tensor::_data::Flat(reinterpret_cast<bf16*>(params.buffer.get() + offset)),
                std::move(shape),
            };
        }
        if (dtype == "INT8") {
            return {
                tensor::_data::ChannelInt8(reinterpret_cast<int8_t*>(params.buffer.get() + offset),
                                           loadScale(entry.at("scale"), fullName, shape)),
                std::move(shape),
            };
        }
        if (dtype == "S3D8") {
            return {
                tensor::_data::ChannelS3D8(reinterpret_cast<uint8_t*>(params.buffer.get() + offset),
                                           loadLut(entry.at("table"), fullName),
                                           loadScale(entry.at("scale"), fullName, shape)),
                std::move(shape),
            };
        }
        std::ostringstream err;
        err << "Tensor " << fullName << " has unsupported dtype = " << dtype;
        throw std::runtime_error(err.str());
    }

    // Load the given named tensor (full name "{prefix}.{name}")
    tensor::TensorV operator()(const std::string& name) const {
        auto fullName = this->fullName(name);
        return loadTensor(header.at(fullName), fullName);
    }
};

TextModel loadTextModel(const json& header, const json& metadata, ParamBuffer& params) {
    TensorLoader model{header, params, "text_model"};
    auto& c = metadata.at("config").at("text");
    auto& v = metadata.at("vocab");
    auto dLayers = c.at("d_layers").template get<uint>();
    std::vector<TextModel::Layer> layers;
    for (auto n = 0u; n < dLayers; ++n) {
        auto layer = model["layers." + std::to_string(n)];
        auto attn = layer["attn"];
        auto mlp = layer["mlp"];
        layers.push_back(  //
            {{
                 .norm = attn("norm.weight"),
                 .query = attn("q_proj.weight"),
                 .key = attn("k_proj.weight"),
                 .value = attn("v_proj.weight"),
                 .output = attn("o_proj.weight"),
                 .query_norm = attn.get("q_norm.weight"),
                 .key_norm = attn.get("k_norm.weight"),
             },
             {
                 .norm = mlp("norm.weight"),
                 .up = mlp("up_proj.weight"),
                 .gate = mlp("gate_proj.weight"),
                 .down = mlp("down_proj.weight"),
             }});
    }
    auto tiedEmbeddings = c.at("tied_embeddings").template get<bool>();
    return TextModel{
        // Config
        .dLayers = c.at("d_layers").template get<uint>(),
        .dVocab = c.at("d_vocab").template get<uint>(),
        .dModel = c.at("d_model").template get<uint>(),
        .dMLP = c.at("d_mlp").template get<uint>(),
        .dAttentionHead = c.at("d_attention_head").template get<uint>(),
        .dAttentionQ = c.at("d_attention_q").template get<uint>(),
        .dAttentionKV = c.at("d_attention_kv").template get<uint>(),
        .dSequenceMax = c.at("d_sequence_max").template get<uint>(),
        .normEpsilon = c.at("norm_epsilon").template get<float>(),
        .ropeAngularFrequency = c.at("rope_angular_frequency").template get<std::vector<float>>(),
        .tiedEmbeddings = tiedEmbeddings,
        .crossAttentionLayers = c.at("cross_attention_layers").template get<std::vector<uint>>(),

        // Parameters
        .embedTokens = model("embed_tokens.weight"),
        .layers = layers,
        .finalNorm = model("norm.weight"),
        .predictTokens = model(tiedEmbeddings ? "embed_tokens.weight" : "lm_head.weight"),

        // Vocab
        .tokenizer = loadTokenizer(v),
        .beginOfTextID = v.at("begin_of_text_id").template get<uint>(),
        .endOfTextID = v.at("end_of_text_id").template get<uint>(),
        .imageID = v.contains("image_id") && !v.at("image_id").is_null()
                       ? std::optional<uint>(v.at("image_id").template get<uint>())
                       : std::nullopt,
    };
}

VisionModel loadVisionModel(const json& header, const json& metadata, ParamBuffer& params) {
    auto loadAffine = [](const TensorLoader& m) {
        return VisionModel::Affine{.weight = m("weight"), .bias = m("bias")};
    };
    auto loadLayers = [loadAffine](const TensorLoader& stack, uint n) {
        std::vector<VisionModel::Layer> layers;
        for (auto i = 0u; i < n; ++i) {
            auto layer = stack[std::to_string(i)];
            auto attn = layer["attn"];
            auto mlp = layer["mlp"];
            layers.push_back({{
                                  .norm = loadAffine(attn["norm"]),
                                  .query = attn("q_proj.weight"),
                                  .key = attn("k_proj.weight"),
                                  .value = attn("v_proj.weight"),
                                  .output = attn("o_proj.weight"),
                              },
                              {
                                  .norm = loadAffine(mlp["norm"]),
                                  .up = loadAffine(mlp["up_proj"]),
                                  .down = loadAffine(mlp["down_proj"]),
                              }});
        }
        return layers;
    };

    TensorLoader model{header, params, "vision_model"};
    auto& c = metadata.at("config").at("vision");
    auto dLayers0 = c.at("d_layers0").template get<uint>();
    auto dLayers1 = c.at("d_layers1").template get<uint>();
    std::vector<VisionModel::Layer> layers0 = loadLayers(model["layers0"], dLayers0);
    std::vector<VisionModel::Layer> layers1 = loadLayers(model["layers1"], dLayers1);
    return VisionModel{
        // Config
        .imageMean = c.at("image_mean").template get<std::vector<float>>(),
        .imageStd = c.at("image_std").template get<std::vector<float>>(),
        .dImage = c.at("d_image").template get<uint>(),
        .dPatch = c.at("d_patch").template get<uint>(),
        .dLayers0 = c.at("d_layers0").template get<uint>(),
        .dLayers1 = c.at("d_layers1").template get<uint>(),
        .dModel = c.at("d_model").template get<uint>(),
        .dMlp = c.at("d_mlp").template get<uint>(),
        .dAttentionHead = c.at("d_attention_head").template get<uint>(),
        .dAttentionQkv = c.at("d_attention_qkv").template get<uint>(),
        .normEpsilon = c.at("norm_epsilon").template get<float>(),
        .outputTaps = c.at("output_taps").template get<std::vector<uint>>(),

        // Parameters
        .patchEmbedding = model("patch_embedding.weight"),
        .positionalEmbedding = model("positional_embedding.weight"),
        .classEmbedding = model("class_embedding.weight"),
        .layerNormPre = loadAffine(model["layernorm_pre"]),
        .layers0 = layers0,
        .layerNormPost = loadAffine(model["layernorm_post"]),
        .tileEmbeddingPost = model("tile_embedding_post.weight"),
        .layers1 = layers1,
        .multiModalProjector = loadAffine(model["multi_modal_projector"]),
    };
}

}  // namespace

Model loadSquashedTensors(std::istream& in) {
    // Preamble
    char preamble[16];
    in.read(preamble, 16);
    auto magic = *reinterpret_cast<uint32_t*>(preamble);
    auto version = *reinterpret_cast<uint32_t*>(preamble + 4);
    auto headerSize = *reinterpret_cast<uint64_t*>(preamble + 8);
    if (!in.good()) {
        std::ostringstream err;
        err << "File not found, or too short (length = " << in.tellg() << " bytes)";
        throw std::runtime_error(err.str());
    }
    if (magic != Magic || version != Version) {
        std::ostringstream err;
        err << "Bad magic " << std::hex << magic << " (expected " << std::hex << Magic
            << ") or version " << std::hex << version << " (expected " << std::hex << Version
            << ")";
        throw std::runtime_error(err.str());
    }

    // Header
    std::string headerStr(headerSize, ' ');
    in.read(headerStr.data(), static_cast<long>(headerSize));
    auto header = json::parse(headerStr);
    auto metadata = header.at("__metadata__");
    header.erase("__metadata__");
    auto alignment = metadata.at("alignment").template get<ulong>();

    // Buffer
    auto bufferLength = maxTensorEndOffset(header, alignment);
    auto s3d8LutBytes = s3d8LutStorageSize(header, alignment);
    checkRAM(bufferLength + s3d8LutBytes);
    tensor::Buffer _data(bufferLength + s3d8LutBytes, alignment);
    for (auto i = ulong(0); i < bufferLength; i += BufferChunkSize) {
        in.read(_data.get() + i,
                static_cast<std::streamsize>(std::min(i + BufferChunkSize, bufferLength) - i));
    }
    ParamBuffer params{
        .alignment = alignment,
        .buffer = _data,
        .nextS3D8LutOffset = bufferLength,
        .s3d8Luts = {},
    };
    return Model{
        .textModel = loadTextModel(header, metadata, params),
        .visionModel = metadata.at("config").at("vision").is_null()
                           ? std::optional<VisionModel>{}
                           : loadVisionModel(header, metadata, params),

        // Metadata & data buffer
        .source = metadata.at("source").template get<std::string>(),
        .created = metadata.at("created").template get<std::string>(),
        .alignment = metadata.at("alignment").template get<ulong>(),
        ._data = std::move(_data),
    };
};

}  // namespace squash
