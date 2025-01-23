"""Model serialization in the squashedtensors '.sqt' format."""

import contextlib
import datetime
import io
import json
import re
import struct
import tempfile
import warnings
from pathlib import Path
from typing import IO, Any, Union

import torch
import transformers
from torch import Tensor, nn
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.models.mllama.modeling_mllama import MllamaForConditionalGeneration
from transformers.models.mllama.image_processing_mllama import MllamaImageProcessor


def encode_bf16(tensor: Tensor) -> tuple[str, Tensor]:
    return ("BF16", tensor.contiguous().flatten().to(torch.bfloat16).view(torch.uint8))


def align(n: int, alignment: int) -> int:
    return n + (-n % alignment)


def rope_angular_frequency(config: transformers.PretrainedConfig) -> Tensor:
    head_dim = getattr(
        config, "head_dim", config.hidden_size // config.num_attention_heads
    )
    freq = config.rope_theta ** -(
        torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim
    )
    s = config.rope_scaling
    z = (
        s["original_max_position_embeddings"] * freq / (2 * torch.pi)
        - s["low_freq_factor"]
    ) / (s["high_freq_factor"] - s["low_freq_factor"])
    freq *= torch.lerp(torch.tensor(1 / s["factor"]), torch.tensor(1.0), z.clip(0, 1))
    return freq


def get_vocab_dict(tokenizer: transformers.PreTrainedTokenizerFast) -> dict[str, Any]:
    # Persist-to-file & load to get acccess to the tokenizer internals
    with tempfile.TemporaryDirectory() as tmp:
        tokenizer.backend_tokenizer.save(tmp + "/tokenizer.json")
        with open(tmp + "/tokenizer.json") as f:
            data = json.load(f)

    # Checks, pre-tokenizer regex, special IDs
    assert data["model"]["type"] == "BPE"
    if not data["model"]["ignore_merges"]:
        warnings.warn(
            "The tokenizer does not set `ignore_merges` (which matches full tokens first)"
            ", but our codebase always behaves as if `ignore_merges` is `True`"
        )
    pre_split, pre_byte = data["pre_tokenizer"]["pretokenizers"]
    pre_regex = pre_split["pattern"]["Regex"]
    assert pre_byte["type"] == "ByteLevel"
    (begin_of_text_id,) = (
        t["id"] for t in data["added_tokens"] if t["content"] == "<|begin_of_text|>"
    )
    (end_of_text_id,) = (
        t["id"] for t in data["added_tokens"] if t["content"] == "<|end_of_text|>"
    )

    # Concatenate merges to single strings & de-duplicate
    merge_set = set([])
    merges = []
    for a, b in data["model"]["merges"]:
        merge = a + b
        if merge not in merge_set:
            merge_set.add(merge)
            merges.append(merge)

    # Convert vocab from a dict to a list
    vocab = [None] * len(data["model"]["vocab"])
    for token, id in data["model"]["vocab"].items():
        vocab[id] = token
    assert all(token is not None for token in vocab)

    return dict(
        begin_of_text_id=begin_of_text_id,
        end_of_text_id=end_of_text_id,
        pre_tokenizer=pre_regex,
        merges=merges,
        vocab=vocab,
    )


def prepare_parameters(
    model: LlamaForCausalLM | MllamaForConditionalGeneration,
) -> dict[str, nn.Parameter]:
    """Rename parameters for uniform storage between models."""
    params = {}
    for name, parameter in model.named_parameters():
        name = re.sub(r"^model.", "text_model.", name)
        name = re.sub(r"^language_model.model.", "text_model.", name)
        name = re.sub(r"cross_attn|self_attn", "attn", name)
        params[name] = parameter
    # TODO -- need to do a few more things:
    #  - merge vision embedding params
    #  - merge cross attention gates
    return params


def get_config_dict(
    config: transformers.PretrainedConfig, image_processor: MllamaImageProcessor | None
) -> dict[str, Any]:
    if hasattr(config, "vision_config"):
        text, vision = config.text_config, config.vision_config
    else:
        text, vision = config, None
    return dict(
        text=dict(
            d_layers=text.num_hidden_layers,
            d_vocab=text.vocab_size,
            d_model=text.hidden_size,
            d_mlp=text.intermediate_size,
            d_attention_head=getattr(
                text, "head_dim", text.hidden_size // text.num_attention_heads
            ),
            d_attention_q=text.num_attention_heads // text.num_key_value_heads,
            d_attention_kv=text.num_key_value_heads,
            d_sequence_max=text.max_position_embeddings,
            norm_epsilon=text.rms_norm_eps,
            rope_angular_frequency=rope_angular_frequency(text).tolist(),
        ),
        vision=(
            dict(
                # Input
                image_mean=image_processor.image_mean,
                image_std=image_processor.image_std,
                d_image=vision.image_size,
                d_patch=vision.patch_size,
                # Core
                d_layers0=vision.num_hidden_layers,
                d_layers1=vision.num_global_layers,
                d_model=vision.hidden_size,
                d_mlp=vision.intermediate_size,
                d_attention_head=vision.hidden_size // vision.attention_heads,
                d_attention_qkv=vision.attention_heads,
                norm_epsilon=vision.norm_eps,
                # Output
                output_taps=vision.intermediate_layers_indices,
            )
            if vision
            else None
        ),
    )


def save(
    model: LlamaForCausalLM,
    tokenizer: transformers.PreTrainedTokenizerFast,
    image_processor: MllamaImageProcessor | None,
    file_or_path: Union[str, Path, IO[bytes]],
    alignment: int = 32,
) -> None:
    """Save a Llama model to '.sqt' format."""
    header_dict = dict(
        __metadata__=dict(
            source=model.config._name_or_path,
            created=datetime.datetime.now().isoformat(timespec="seconds"),
            alignment=alignment,
            config=get_config_dict(model.config, image_processor),
            vocab=get_vocab_dict(tokenizer),
        )
    )
    params = prepare_parameters(model)

    # Measure serialized tensors, but discard them (for sake of memory usage)
    last_offset = 0
    for name, tensor in params.items():
        dtype, data = encode_bf16(tensor)
        header_dict[name] = dict(
            dtype=dtype,
            shape=list(tensor.shape),
            data_offsets=[last_offset, last_offset + len(data)],
        )
        last_offset += align(len(data), alignment)

    with contextlib.ExitStack() as stack:
        f = (
            file_or_path
            if isinstance(file_or_path, io.IOBase)
            else stack.enter_context(open(file_or_path, "wb"))
        )

        f.write(b".sqt")  # magic
        f.write(struct.pack("<I", 0))  # version
        header = json.dumps(header_dict, separators=(",", ":")).encode("utf8")
        nheader_pad = -(16 + len(header)) % alignment
        f.write(struct.pack("<Q", len(header) + nheader_pad))  # header length
        f.write(header + nheader_pad * b" ")  # header

        # buffer
        buffer_start = f.tell()
        assert buffer_start % alignment == 0
        for name, tensor in params.items():
            begin, end = header_dict[name]["data_offsets"]
            _, data = encode_bf16(tensor)
            assert f.tell() == buffer_start + begin
            f.write(data.numpy().tobytes())
            assert f.tell() == buffer_start + end
            f.write((align(len(data), alignment) - len(data)) * b"\0")
