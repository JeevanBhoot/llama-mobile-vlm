# Copyright (c) 2026 Graphcore Ltd. All rights reserved.

import pytest
import safetensors.torch
import torch
import weight_formats.quantisation_training as QT
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM

import direct_cast


def tiny_llama() -> LlamaForCausalLM:
    return LlamaForCausalLM(
        LlamaConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            vocab_size=32,
            max_position_embeddings=64,
            tie_word_embeddings=True,
        )
    )


@pytest.mark.parametrize(
    ("fmt", "wrapper_type"),
    [("int8", QT.Weight), ("s3d8", QT.Sign3D8Weight)],
)
def test_direct_cast_checkpoint_load_convert_roundtrip(
    monkeypatch, tmp_path, fmt, wrapper_type
) -> None:
    monkeypatch.setattr(direct_cast, "load_model", lambda *_, **__: tiny_llama())
    output_path = tmp_path / f"direct-{fmt}.safetensors"

    direct_cast.direct_cast(
        "unused-model-id",
        output_path,
        fmt,
        dtype=torch.bfloat16,
    )

    state = safetensors.torch.load_file(output_path)
    assert "_quantisation_meta" in state

    model = tiny_llama()
    QT.load_convert(model, state)

    assert any(isinstance(m, wrapper_type) for m in model.modules())
