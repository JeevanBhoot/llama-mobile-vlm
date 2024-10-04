#define JSON_USE_IMPLICIT_CONVERSIONS 0

#include <sys/sysinfo.h>
#include <iostream>
#include <json.hpp>
#include <sstream>

#include "squash.hpp"

namespace squash {

namespace {
constexpr auto Magic = 0x7471732eu;
constexpr auto Version = 0u;
constexpr auto MaxRamProportion = 0.75f;
constexpr auto BufferChunkSize = 4096u;

using json = nlohmann::json;

ulong align(ulong index, uint alignment) {
    return alignment * (index + alignment - 1) / alignment;
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

}  // namespace

Model sqt_load(std::istream& in) {
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
    auto metadata = header["__metadata__"];
    header.erase("__metadata__");
    auto alignment = metadata["alignment"].template get<uint>();

    // Buffer
    auto bufferLength = ulong(0);
    for (auto& x : header) {
        bufferLength =
            std::max(bufferLength, align(x["data_offsets"][1].template get<ulong>(), alignment));
    }
    checkRAM(bufferLength);
    Buffer _data(bufferLength, alignment);
    for (auto i = ulong(0); i < bufferLength; i += BufferChunkSize) {
        in.read(_data.get() + i,
                static_cast<std::streamsize>(std::min(i + BufferChunkSize, bufferLength) - i));
    }

    // Tensors & model config
    auto loadTensorV = [&](const std::string& name) -> TensorV {
        auto fullName = "model." + name + ".weight";
        auto& entry = header[fullName];
        auto dtype = entry["dtype"].template get<std::string>();
        if (dtype != "BF16") {
            std::ostringstream err;
            err << "Tensor " << fullName << " has unsupported dtype = " << dtype;
            throw std::runtime_error(err.str());
        }
        auto offset = entry["data_offsets"][0].template get<ulong>();
        if (offset % alignment) {
            std::ostringstream err;
            err << "Tensor " << fullName << " is misaligned, offset = " << offset;
            throw std::runtime_error(err.str());
        }
        return TensorV{
            tensor_data::Flat(reinterpret_cast<bf16*>(_data.get() + offset)),
            entry["shape"].template get<std::vector<uint>>(),
        };
    };
    auto& c = metadata["config"];
    auto dLayers = c["d_layers"].template get<uint>();
    std::vector<Model::Layer> layers;
    for (auto n = 0u; n < dLayers; ++n) {
        auto base = "layers." + std::to_string(n);
        auto attn = base + ".self_attn";
        auto mlp = base + ".mlp";
        layers.push_back(  //
            {{
                 .norm = loadTensorV(base + ".input_layernorm"),
                 .query = loadTensorV(attn + ".q_proj"),
                 .key = loadTensorV(attn + ".k_proj"),
                 .value = loadTensorV(attn + ".v_proj"),
                 .output = loadTensorV(attn + ".o_proj"),
             },
             {
                 .norm = loadTensorV(base + ".post_attention_layernorm"),
                 .up = loadTensorV(mlp + ".up_proj"),
                 .gate = loadTensorV(mlp + ".gate_proj"),
                 .down = loadTensorV(mlp + ".down_proj"),
             }});
    }
    return Model{
        // Metadata
        .source = metadata["source"].template get<std::string>(),
        .created = metadata["created"].template get<std::string>(),
        .alignment = alignment,

        // Config
        .dLayers = dLayers,
        .dVocab = c["d_vocab"].template get<uint>(),
        .dModel = c["d_model"].template get<uint>(),
        .dMLP = c["d_mlp"].template get<uint>(),
        .dAttentionHead = c["d_attention_head"].template get<uint>(),
        .dAttentionQ = c["d_attention_q"].template get<uint>(),
        .dAttentionKV = c["d_attention_kv"].template get<uint>(),
        .dSequenceMax = c["d_sequence_max"].template get<uint>(),
        .normEpsilon = c["norm_epsilon"].template get<float>(),
        .ropeAngularFrequency = c["rope_angular_frequency"].template get<std::vector<float>>(),

        // Parameters
        .embedTokens = loadTensorV("embed_tokens"),
        .layers = layers,
        .finalNorm = loadTensorV("norm"),

        // Data
        ._data = std::move(_data),
    };
};

}  // namespace squash
