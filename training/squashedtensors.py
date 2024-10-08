"""Model serialization in the squashedtensors '.sqt' format."""

import contextlib
import datetime
import io
import json
import struct
import tempfile
from pathlib import Path
from typing import IO, Any, Dict, Tuple, Union

import torch
import transformers
from torch import Tensor
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM


def encode_bf16(tensor: Tensor) -> Tuple[str, Tensor]:
    return ("BF16", tensor.contiguous().flatten().to(torch.bfloat16).view(torch.uint8))


def align(n: int, alignment: int) -> int:
    return n + (-n % alignment)


def rope_angular_frequency(config: LlamaConfig) -> Tensor:
    freq = config.rope_theta ** -(
        torch.arange(0, config.head_dim, 2, dtype=torch.float) / config.head_dim
    )
    s = config.rope_scaling
    z = (
        s["original_max_position_embeddings"] * freq / (2 * torch.pi)
        - s["low_freq_factor"]
    ) / (s["high_freq_factor"] - s["low_freq_factor"])
    freq *= torch.lerp(torch.tensor(1 / s["factor"]), torch.tensor(1.0), z.clip(0, 1))
    return freq


def get_vocab_dict(tokenizer: transformers.PreTrainedTokenizerFast) -> Dict[str, Any]:
    # Persist-to-file & load to get acccess to the tokenizer internals
    with tempfile.TemporaryDirectory() as tmp:
        tokenizer.backend_tokenizer.save(tmp + "/tokenizer.json")
        with open(tmp + "/tokenizer.json") as f:
            data = json.load(f)

    # Checks, pre-tokenizer regex, special IDs
    assert data["model"]["type"] == "BPE"
    assert data["model"]["ignore_merges"]
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


def save(
    model: LlamaForCausalLM,
    tokenizer: transformers.PreTrainedTokenizerFast,
    file_or_path: Union[str, Path, IO[bytes]],
    alignment: int = 32,
) -> None:
    """Save a Llama model to '.sqt' format."""
    header_dict = dict(
        __metadata__=dict(
            source=model.config._name_or_path,
            created=datetime.datetime.now().isoformat(timespec="seconds"),
            alignment=alignment,
            config=dict(
                d_layers=model.config.num_hidden_layers,
                d_vocab=model.config.vocab_size,
                d_model=model.config.hidden_size,
                d_mlp=model.config.intermediate_size,
                d_attention_head=model.config.head_dim,
                d_attention_q=model.config.num_attention_heads
                // model.config.num_key_value_heads,
                d_attention_kv=model.config.num_key_value_heads,
                d_sequence_max=model.config.max_position_embeddings,
                norm_epsilon=model.config.rms_norm_eps,
                rope_angular_frequency=rope_angular_frequency(model.config).tolist(),
            ),
            vocab=get_vocab_dict(tokenizer),
        )
    )
    # Measure serialized tensors, but discard them (for sake of memory usage)
    last_offset = 0
    for name, tensor in model.named_parameters():
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
        for name, tensor in model.named_parameters():
            begin, end = header_dict[name]["data_offsets"]
            _, data = encode_bf16(tensor)
            assert f.tell() == buffer_start + begin
            f.write(data.numpy().tobytes())
            assert f.tell() == buffer_start + end
            f.write((align(len(data), alignment) - len(data)) * b"\0")
