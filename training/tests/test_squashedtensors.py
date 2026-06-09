# Copyright (c) 2026 Graphcore Ltd. All rights reserved.

import io
import json
import struct

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


@torch.inference_mode()
def test_save_channel_int8_qat_dynamic(monkeypatch) -> None:
    monkeypatch.setattr(squashedtensors, "get_vocab_dict", lambda _: {})
    buffer = io.BytesIO()
    squashedtensors.save(make_qat_model("dynamic", INT_FMT), None, None, buffer)

    version, header = read_header(buffer)
    entry = header["text_model.layers.0.attn.q_proj.weight"]

    assert version == 2
    assert entry["dtype"] == "INT8"
    assert entry["shape"] == [16, 16]
    assert entry["scale"]["dtype"] == "BF16"
    assert entry["scale"]["shape"] == [16, 1]
    assert all(not key.endswith((".master", ".centroids", ".scale")) for key in header)


@torch.inference_mode()
def test_save_channel_int8_qat_parameter(monkeypatch) -> None:
    monkeypatch.setattr(squashedtensors, "get_vocab_dict", lambda _: {})
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
    monkeypatch.setattr(squashedtensors, "get_vocab_dict", lambda _: {})
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
