# Copyright (c) 2026 Graphcore Ltd. All rights reserved.

import io
import json
import struct
from pathlib import Path

import torch
import weight_formats.quantisation as Q
import weight_formats.quantisation_training as QT
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM

import squashedtensors

INT_FMT = Q.LinearScalingFormat(
    Q.IntFormat(8),
    scale_format=Q.BFLOAT16,
    block_shape=(1, None),
    scaling="absmax",
)

S3D8_FMT = Q.LinearScalingFormat(
    Q.Sign3D8Format(
        Q.VectorLUTFormat.create(
            torch.cartesian_prod(
                torch.linspace(0, 127, 4),
                torch.linspace(0, 127, 4),
                torch.linspace(0, 127, 2),
            ),
            Q.TorchFormat(torch.int8),
            "S3D8",
            range=(0, 127),
        )
    ),
    scale_format=Q.BFLOAT16,
    block_shape=(1, None),
    scaling="absmax",
)


def make_qat_model(scaling_mode: str, fmt: Q.TensorFormat) -> LlamaForCausalLM:
    model = LlamaForCausalLM(
        LlamaConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            vocab_size=32,
            max_position_embeddings=64,
        )
    )
    QT.convert(
        model,
        fmt_spec=fmt,
        scaling_mode=scaling_mode,
        clip_gradient=False,
        error_weight=None,
        activation_fmt=None,
        progress=False,
    )
    for _, parameter in QT.get_named_parameters(model, "centroids"):
        parameter.requires_grad_(False)
    return model


def read_header(buffer: io.BytesIO) -> tuple[int, dict[str, object]]:
    buffer.seek(0)
    assert buffer.read(4) == b".sqt"
    version = struct.unpack("<I", buffer.read(4))[0]
    header_size = struct.unpack("<Q", buffer.read(8))[0]
    header = json.loads(buffer.read(header_size).decode("utf8").rstrip(" "))
    return version, header


def test_get_vocab_dict_includes_added_tokens_and_stop_ids() -> None:
    class FakeBackendTokenizer:
        def save(self, path: str) -> None:
            data = {
                "model": {
                    "type": "BPE",
                    "ignore_merges": True,
                    "vocab": {"a": 0, "b": 1},
                    "merges": [],
                },
                "pre_tokenizer": {
                    "pretokenizers": [
                        {"pattern": {"Regex": "."}},
                        {"type": "ByteLevel"},
                    ]
                },
                "added_tokens": [
                    {"id": 2, "content": "<|begin_of_text|>"},
                    {"id": 3, "content": "<|end_of_text|>"},
                    {"id": 4, "content": "<|start_header_id|>"},
                    {"id": 5, "content": "<|end_header_id|>"},
                    {"id": 6, "content": "<|eot_id|>"},
                    {"id": 7, "content": "<|eom_id|>"},
                ],
            }
            Path(path).write_text(json.dumps(data))

    class FakeTokenizer:
        backend_tokenizer = FakeBackendTokenizer()
        chat_template = "{{ bos_token }}"
        eos_token_id = 6

    config = LlamaConfig(vocab_size=8, eos_token_id=[3, 6])

    vocab = squashedtensors.get_vocab_dict(FakeTokenizer(), config)

    assert vocab["chat_template"] == "{{ bos_token }}"
    assert vocab["vocab"] == [
        "a",
        "b",
        "<|begin_of_text|>",
        "<|end_of_text|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|eot_id|>",
        "<|eom_id|>",
    ]
    assert vocab["begin_of_text_id"] == 2
    assert vocab["start_header_id"] == 4
    assert vocab["end_header_id"] == 5
    assert vocab["eot_id"] == 6
    assert vocab["image_id"] is None
    assert vocab["stop_token_ids"] == [3, 6, 7]


@torch.inference_mode()
def test_save_channel_int8_qat_dynamic(monkeypatch) -> None:
    monkeypatch.setattr(squashedtensors, "get_vocab_dict", lambda *_: {})
    buffer = io.BytesIO()
    squashedtensors.save(make_qat_model("dynamic", INT_FMT), None, None, buffer)

    version, header = read_header(buffer)
    entry = header["text_model.layers.0.attn.q_proj.weight"]

    assert version == 3
    assert entry["dtype"] == "INT8"
    assert entry["shape"] == [16, 16]
    assert entry["scale"]["dtype"] == "BF16"
    assert entry["scale"]["shape"] == [16, 1]
    assert all(not key.endswith((".master", ".centroids", ".scale")) for key in header)


@torch.inference_mode()
def test_save_comment_metadata(monkeypatch) -> None:
    monkeypatch.setattr(squashedtensors, "get_vocab_dict", lambda *_: {})
    buffer = io.BytesIO()
    squashedtensors.save(
        make_qat_model("dynamic", INT_FMT),
        None,
        None,
        buffer,
        comment="calibration sweep 17",
    )

    _, header = read_header(buffer)

    assert header["__metadata__"]["comment"] == "calibration sweep 17"


@torch.inference_mode()
def test_save_channel_int8_qat_parameter(monkeypatch) -> None:
    monkeypatch.setattr(squashedtensors, "get_vocab_dict", lambda *_: {})
    buffer = io.BytesIO()
    squashedtensors.save(make_qat_model("parameter", INT_FMT), None, None, buffer)

    _, header = read_header(buffer)
    entry = header["text_model.layers.0.attn.q_proj.weight"]

    assert entry["dtype"] == "INT8"
    assert entry["data_offsets"][1] - entry["data_offsets"][0] == 16 * 16
    assert (
        entry["scale"]["data_offsets"][1] - entry["scale"]["data_offsets"][0] == 16 * 2
    )


@torch.inference_mode()
def test_save_channel_s3d8_qat_dynamic(monkeypatch) -> None:
    monkeypatch.setattr(squashedtensors, "get_vocab_dict", lambda *_: {})
    buffer = io.BytesIO()
    squashedtensors.save(make_qat_model("dynamic", S3D8_FMT), None, None, buffer)

    _, header = read_header(buffer)
    entry = header["text_model.layers.0.attn.q_proj.weight"]

    assert entry["dtype"] == "S3D8"
    assert entry["shape"] == [16, 16]
    assert entry["data_offsets"][1] - entry["data_offsets"][0] == 16 * 6
    assert entry["table"]["dtype"] == "INT8"
    assert entry["table"]["shape"] == [32, 3]
    assert (
        entry["table"]["data_offsets"][1] - entry["table"]["data_offsets"][0] == 32 * 3
    )
    assert entry["scale"]["dtype"] == "BF16"
    assert entry["scale"]["shape"] == [16, 1]
    assert all(not key.endswith((".master", ".centroids", ".scale")) for key in header)
