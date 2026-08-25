# Copyright (c) 2026 Graphcore Ltd. All rights reserved.

import json
import math
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

import pytest
import safetensors.torch
import torch
import weight_formats.quantisation_training as QT


def fake_vqa_module(evaluate=None):
    class FakeVQA:
        QUESTION_TEMPLATE = "Question: {question}"
        data = mock.Mock(return_value=["a"])

        @classmethod
        def prepare_batch(cls, batch, system_template):
            return SimpleNamespace(
                images=batch["image"],
                prompts=[
                    system_template.format(prompt=f"Question: {q}")
                    for q in batch["question"]
                ],
                answers=batch.get("answers", []),
                ids=batch.get("question_id", []),
            )

    task = SimpleNamespace(
        METRICS=("accuracy",),
        RELAXED_METRICS=("accuracy_relaxed",),
        data=mock.Mock(return_value=["a"]),
    )
    fake_vqa = ModuleType("eval.vqa")
    fake_vqa.TASKS = {"vqa": task}
    fake_vqa.VQA = FakeVQA
    fake_vqa.LLAMA_PROMPT_TEMPLATES = {"instruct": "<s>{prompt}</s>"}
    fake_vqa.evaluate = evaluate or mock.Mock()
    return fake_vqa


sys.modules["eval.vqa"] = fake_vqa_module()

import gptq.common as common
import gptq.local as local


def test_local_cli_supports_ordinary_int2() -> None:
    args = local.build_parser().parse_args(["quantize", "--bits", "2"])

    assert args.bits == 2


def test_default_output_dir_for_int_codebook_includes_sweep_params() -> None:
    assert local.default_output_dir(
        4,
        "int-codebook",
        codepoints=7,
        group_size=128,
    ) == Path("out/gptq/llama-3.2-vision-local-gptq-int-k7-g128-c4")


def test_default_output_dir_for_affine_codebook_is_distinct() -> None:
    assert local.default_output_dir(
        None,
        "int-codebook-affine",
        codepoints=6,
        group_size=128,
    ) == Path(
        "out/gptq/llama-3.2-vision-local-gptq-int-affine-k6-g128-c4"
    )


def test_default_output_dir_identifies_full_multimodal_calibration() -> None:
    assert local.default_output_dir(
        4,
        target_scope="full-multimodal",
        calibration_source="synthetic",
    ) == Path("out/gptq/llama-3.2-vision-local-gptq-int4-synthetic-full")


class TinySelfAttention(torch.nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.k_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.q_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x):
        x = self.k_proj(x) + self.v_proj(x) + self.q_proj(x)
        return self.o_proj(x)


class TinyMlp(torch.nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.up_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.gate_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.down_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(self.up_proj(x) + self.gate_proj(x))


class TinyDecoderLayer(torch.nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.self_attn = TinySelfAttention(hidden_size)
        self.mlp = TinyMlp(hidden_size)

    def forward(self, hidden_states, **kwargs):
        return (hidden_states + self.self_attn(hidden_states) + self.mlp(hidden_states),)


class TinyRotaryEmbedding(torch.nn.Module):
    def forward(self, hidden_states, position_ids):
        return (
            torch.zeros_like(hidden_states),
            torch.ones_like(hidden_states),
        )


class MllamaCrossAttentionDecoderLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.called = False
        self.cross_attn = torch.nn.Linear(8, 8, bias=False)

    def forward(self, hidden_states, **kwargs):
        self.called = True
        raise ValueError("cross attention should be skipped for text-only calibration")


class TinyLanguageModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(use_cache=True)
        self.embed_tokens = torch.nn.Embedding(32, 8)
        self.rotary_emb = TinyRotaryEmbedding()
        self.cross_attention_layers = [1]
        self.cross_layer = MllamaCrossAttentionDecoderLayer()
        self.layers = torch.nn.ModuleList(
            [
                TinyDecoderLayer(8),
                self.cross_layer,
                TinyDecoderLayer(8),
            ]
        )


class TinyMllamaModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = TinyLanguageModel()


class TinyMllama(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(use_cache=True)
        self.model = TinyMllamaModel()
        self.layers = self.model.language_model.layers
        self.forward_called = False

    def forward(self, input_ids, **kwargs):
        self.forward_called = True
        hidden_states = self.model.language_model.embed_tokens(input_ids)
        for layer_index, layer in enumerate(self.layers):
            if local.is_mllama_cross_attention_layer(
                layer,
                layer_index=layer_index,
                cross_attention_layers=frozenset(
                    self.model.language_model.cross_attention_layers
                ),
            ):
                continue
            hidden_states = layer(hidden_states)[0]
        return SimpleNamespace(logits=hidden_states)

    def save_pretrained(self, output_dir, safe_serialization=True):
        Path(output_dir, "model.safetensors").write_text("weights")


class TinyIdentityPosition(torch.nn.Module):
    def forward(self, hidden_state, aspect_ratio_ids):
        return hidden_state


class TinyVisionAttention(torch.nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.q_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_state, attention_mask=None):
        hidden_state = (
            self.q_proj(hidden_state)
            + self.k_proj(hidden_state)
            + self.v_proj(hidden_state)
        )
        return self.o_proj(hidden_state), None


class TinyVisionMlp(torch.nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.fc1 = torch.nn.Linear(hidden_size, hidden_size * 2)
        self.fc2 = torch.nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, hidden_state):
        return self.fc2(torch.relu(self.fc1(hidden_state)))


class TinyVisionLayer(torch.nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.self_attn = TinyVisionAttention(hidden_size)
        self.mlp = TinyVisionMlp(hidden_size)

    def forward(self, hidden_state, attention_mask=None):
        hidden_state = hidden_state + self.self_attn(hidden_state, attention_mask)[0]
        return hidden_state + self.mlp(hidden_state)


class TinyVisionEncoder(torch.nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.layers = torch.nn.ModuleList([TinyVisionLayer(hidden_size)])


class TinyVisionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embedding = torch.nn.Conv2d(1, 4, kernel_size=1, bias=False)
        self.pre_tile_positional_embedding = TinyIdentityPosition()
        self.gated_positional_embedding = TinyIdentityPosition()
        self.post_tile_positional_embedding = TinyIdentityPosition()
        self.layernorm_pre = torch.nn.LayerNorm(4)
        self.layernorm_post = torch.nn.LayerNorm(4)
        self.transformer = TinyVisionEncoder(4)
        self.global_transformer = TinyVisionEncoder(4)
        self.intermediate_layers_indices = [0]
        self.num_patches = 5

    def apply_class_embedding(self, hidden_state):
        batch, _, hidden = hidden_state.shape
        cls = torch.zeros(batch, 1, hidden, device=hidden_state.device)
        return torch.cat([cls, hidden_state], dim=1)


class TinyCrossAttention(torch.nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.q_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states, cross_attention_states=None, **kwargs):
        key = self.k_proj(cross_attention_states).mean(dim=1, keepdim=True)
        value = self.v_proj(cross_attention_states).mean(dim=1, keepdim=True)
        hidden_states = self.q_proj(hidden_states) + key + value
        return self.o_proj(hidden_states), None


class TinyCrossAttentionDecoderLayer(torch.nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.called = False
        self.cross_attn = TinyCrossAttention(hidden_size)
        self.mlp = TinyMlp(hidden_size)

    def forward(self, hidden_states, cross_attention_states=None, **kwargs):
        self.called = True
        hidden_states = hidden_states + self.cross_attn(
            hidden_states,
            cross_attention_states=cross_attention_states,
        )[0]
        return hidden_states + self.mlp(hidden_states)


class TinyFullLanguageModel(TinyLanguageModel):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.config = SimpleNamespace(use_cache=True)
        self.embed_tokens = torch.nn.Embedding(32, 8)
        self.rotary_emb = TinyRotaryEmbedding()
        self.norm = torch.nn.LayerNorm(8)
        self.cross_attention_layers = [1]
        self.layers = torch.nn.ModuleList(
            [
                TinyDecoderLayer(8),
                TinyCrossAttentionDecoderLayer(8),
            ]
        )


class TinyFullMllamaModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_model = TinyVisionModel()
        self.multi_modal_projector = torch.nn.Linear(8, 8)
        self.language_model = TinyFullLanguageModel()


class TinyFullMllama(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = TinyFullMllamaModel()
        self.lm_head = torch.nn.Linear(8, 32, bias=False)

    def save_pretrained(self, output_dir, safe_serialization=True):
        Path(output_dir, "model.safetensors").write_text("weights")


class DummyProcessor:
    tokenizer = object()

    def save_pretrained(self, output_dir):
        Path(output_dir, "processor_config.json").write_text("{}")


def test_collect_first_layer_inputs_uses_mllama_direct_text_path() -> None:
    model = TinyMllama()
    stack = local.mllama_text_layers(model)
    input_ids = torch.tensor([[1, 2, 3]])
    attention_mask = torch.ones_like(input_ids)

    captured = local.collect_first_layer_inputs(
        stack,
        [{"input_ids": input_ids, "attention_mask": attention_mask}],
        device="cpu",
    )

    assert model.forward_called is False
    assert len(captured) == 1
    torch.testing.assert_close(
        captured.args[0][0],
        model.model.language_model.embed_tokens(input_ids).detach(),
    )
    assert captured.kwargs[0]["attention_mask"] is None
    torch.testing.assert_close(
        captured.kwargs[0]["position_ids"],
        torch.tensor([[0, 1, 2]]),
    )
    assert captured.kwargs[0]["use_cache"] is False
    assert "position_embeddings" in captured.kwargs[0]


def test_cross_attention_scope_is_structural() -> None:
    class RenamedCrossAttentionLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.cross_attn = torch.nn.Linear(8, 8, bias=False)
            self.called = False

        def forward(self, hidden_states, **kwargs):
            self.called = True
            raise AssertionError("cross attention should not run")

    layer = RenamedCrossAttentionLayer()
    inputs = local.LayerInputs(
        args=[[torch.randn(1, 3, 8)]],
        kwargs=[{"position_ids": torch.tensor([[0, 1, 2]])}],
    )

    next_inputs, logs = local.quantize_layer(
        layer,
        inputs,
        local.GPTQConfig(bits=4),
        device="cpu",
        verbose=False,
    )

    assert logs == []
    assert layer.called is False
    torch.testing.assert_close(next_inputs.args[0][0], inputs.args[0][0])


def test_self_attention_scope_requires_all_gptqmodel_targets() -> None:
    layer = TinyDecoderLayer(8)
    del layer.self_attn.o_proj
    inputs = local.LayerInputs(
        args=[[torch.randn(1, 3, 8)]],
        kwargs=[{"position_ids": torch.tensor([[0, 1, 2]])}],
    )

    with pytest.raises(ValueError, match="missing linear modules"):
        local.quantize_layer(
            layer,
            inputs,
            local.GPTQConfig(bits=4),
            device="cpu",
            verbose=False,
        )


def test_quantize_writes_dense_artifact(monkeypatch, tmp_path) -> None:
    torch.manual_seed(625464)
    model = TinyMllama()
    monkeypatch.setattr(local, "load_model", lambda *_, **__: model)
    monkeypatch.setattr(
        local.transformers.AutoProcessor,
        "from_pretrained",
        mock.Mock(return_value=DummyProcessor()),
    )
    monkeypatch.setattr(local, "load_c4_calibration", lambda **_: ["a", "b"])
    monkeypatch.setattr(
        local,
        "tokenise_calibration",
        lambda *_, **__: [
            {"input_ids": torch.tensor([1, 2, 3, 4])},
            {"input_ids": torch.tensor([4, 3, 2, 1])},
        ],
    )

    metadata = local.quantize(
        model_name="model",
        output_dir=tmp_path,
        bits=4,
        group_size=128,
        batch_size=2,
        calibration_samples=2,
        calibration_data_min_length=0,
        device="cpu",
        torch_dtype="float32",
        verbose=False,
    )

    assert metadata["artifact_type"] == "dense_dequantized_local_gptq"
    assert metadata["n_quantized_modules"] == 14
    assert metadata["quantization_batch_size"] == 2
    assert metadata["calibration_prepared_examples"] == 2
    assert metadata["calibration_batches"] == 1
    assert metadata["target_scope"] == "mllama_text_self_attention_decoder_layers"
    assert metadata["skipped_cross_attention_layers"] == [
        {
            "layer": 1,
            "full_name": "model.language_model.layers.1",
            "class": "MllamaCrossAttentionDecoderLayer",
            "reason": "mllama_text_only_cross_attention",
        }
    ]
    assert model.model.language_model.config.use_cache is True
    assert not model.model.language_model.cross_layer.called
    assert all(
        log["full_name"].startswith(
            (
                "model.language_model.layers.0.",
                "model.language_model.layers.2.",
            )
        )
        for log in metadata["quantization_log"]
    )
    storage = metadata["estimated_packed_storage"]
    assert storage["quantized_tensor_count"] == 14
    assert storage["unmatched_quantized_tensor_names"] == []
    assert storage["estimated_packed_without_g_idx_bytes"] < storage[
        "dense_state_dict_bytes"
    ]
    assert storage["estimated_packed_with_g_idx_bytes"] >= storage[
        "estimated_packed_without_g_idx_bytes"
    ]
    assert (tmp_path / "model.safetensors").exists()
    assert (tmp_path / "processor_config.json").exists()
    assert (tmp_path / local.METADATA_FILENAME).exists()


def test_quantize_int_codebook_writes_scale_only_metadata(
    monkeypatch, tmp_path
) -> None:
    torch.manual_seed(625464)
    model = TinyMllama()
    monkeypatch.setattr(local, "load_model", lambda *_, **__: model)
    monkeypatch.setattr(
        local.transformers.AutoProcessor,
        "from_pretrained",
        mock.Mock(return_value=DummyProcessor()),
    )
    monkeypatch.setattr(local, "load_c4_calibration", lambda **_: ["a", "b"])
    monkeypatch.setattr(
        local,
        "tokenise_calibration",
        lambda *_, **__: [
            {"input_ids": torch.tensor([1, 2, 3, 4])},
            {"input_ids": torch.tensor([4, 3, 2, 1])},
        ],
    )

    metadata = local.quantize(
        model_name="model",
        output_dir=tmp_path,
        bits=4,
        quantization_format="int-codebook",
        codepoints=6,
        group_size=4,
        batch_size=2,
        calibration_samples=2,
        calibration_data_min_length=0,
        device="cpu",
        torch_dtype="float32",
        verbose=False,
    )

    assert metadata["artifact_type"] == "dense_dequantized_local_gptq_int_codebook"
    assert metadata["format"] == "int-codebook"
    assert metadata["bits"] is None
    assert metadata["codepoints"] == 6
    assert metadata["effective_weight_bits"] == pytest.approx(math.log2(6))
    assert metadata["quantizer_mode"] == "scale_only_absmax_int_codebook"
    assert metadata["scale_dtype"] == "bfloat16"
    assert metadata["n_quantized_modules"] == 14

    first_log = metadata["quantization_log"][0]
    assert first_log["quantization_format"] == "int-codebook"
    assert first_log["codepoints"] == 6
    assert first_log["quantizer_mode"] == "scale_only_absmax_int_codebook"
    assert "zero_shape" not in first_log
    assert first_log["g_idx_shape"] == [8]

    storage = metadata["estimated_packed_storage"]
    assert storage["assumptions"]["quantized_weight_bits"] == pytest.approx(
        math.log2(6)
    )
    expected_scale_values = sum(
        math.prod(log["scale_shape"]) for log in metadata["quantization_log"]
    )
    assert storage["scale_zero_bytes"] == expected_scale_values * 2
    assert storage["g_idx_bytes"] > 0
    assert storage["centroid_bytes"] == 0
    assert (tmp_path / "model.safetensors").exists()
    assert (tmp_path / "processor_config.json").exists()
    assert (tmp_path / local.METADATA_FILENAME).exists()


def test_quantize_affine_codebook_omits_implicit_zero_from_storage(
    monkeypatch, tmp_path
) -> None:
    torch.manual_seed(625464)
    model = TinyMllama()
    monkeypatch.setattr(local, "load_model", lambda *_, **__: model)
    monkeypatch.setattr(
        local.transformers.AutoProcessor,
        "from_pretrained",
        mock.Mock(return_value=DummyProcessor()),
    )
    monkeypatch.setattr(local, "load_c4_calibration", lambda **_: ["a", "b"])
    monkeypatch.setattr(
        local,
        "tokenise_calibration",
        lambda *_, **__: [
            {"input_ids": torch.tensor([1, 2, 3, 4])},
            {"input_ids": torch.tensor([4, 3, 2, 1])},
        ],
    )

    metadata = local.quantize(
        model_name="model",
        output_dir=tmp_path,
        bits=None,
        quantization_format="int-codebook-affine",
        codepoints=6,
        group_size=4,
        batch_size=2,
        calibration_samples=2,
        calibration_data_min_length=0,
        device="cpu",
        torch_dtype="float32",
        storage_scale_zero_dtype="float16",
        verbose=False,
    )

    assert (
        metadata["artifact_type"]
        == "dense_dequantized_local_gptq_int_codebook_affine"
    )
    assert metadata["format"] == "int-codebook-affine"
    assert metadata["bits"] is None
    assert metadata["codepoints"] == 6
    assert metadata["effective_weight_bits"] == pytest.approx(math.log2(6))
    assert metadata["quantizer_mode"] == "uniform_affine_absmax_int_codebook"
    assert metadata["scale_dtype"] == "float16"
    assert metadata["gptq"]["bits"] is None
    assert metadata["gptq"]["sym"] is True
    assert metadata["gptq"]["scale_zero_dtype"] == "float16"

    first_log = metadata["quantization_log"][0]
    assert first_log["quantization_format"] == "int-codebook-affine"
    assert first_log["codepoints"] == 6
    assert first_log["zero_shape"] == first_log["scale_shape"]
    assert first_log["g_idx_shape"] == [8]

    storage = metadata["estimated_packed_storage"]
    assert storage["radix_chunk_bytes"] == 1
    assert storage["radix_symbols_per_chunk"] == 3
    assert storage["radix_bits_per_weight"] == pytest.approx(8 / 3)
    expected_scale_values = sum(
        math.prod(log["scale_shape"]) for log in metadata["quantization_log"]
    )
    assert storage["scale_zero_bytes"] == expected_scale_values * 2
    assert (
        storage["assumptions"]["zero_point_storage"]
        == "implicit floor(K / 2); no independent storage"
    )
    assert storage["estimated_realizable_packed_with_g_idx_bytes"] >= storage[
        "estimated_packed_with_g_idx_bytes"
    ]


def test_radix_packing_selects_three_k6_symbols_per_byte() -> None:
    packing = local.radix_packing_parameters(6)

    assert packing == {
        "chunk_bytes": 1,
        "symbols_per_chunk": 3,
        "bits_per_weight": pytest.approx(8 / 3),
    }


def test_radix_storage_accounts_for_each_tensor_tail() -> None:
    class TwoWeights(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.first = torch.nn.Linear(4, 1, bias=False)
            self.second = torch.nn.Linear(4, 1, bias=False)

    logs = [
        {
            "full_name": name,
            "shape": [1, 4],
            "scale_shape": [1, 1],
            "zero_shape": [1, 1],
            "g_idx_shape": [4],
        }
        for name in ("first", "second")
    ]

    storage = local.estimate_packed_storage(
        TwoWeights(),
        layer_logs=logs,
        bits=math.log2(6),
        scale_zero_dtype="bfloat16",
        quantization_format="int-codebook-affine",
        codepoints=6,
        sym=False,
    )

    # Each four-symbol tensor needs two one-byte chunks; combining tensor tails
    # would incorrectly report three bytes.
    assert storage["radix_packed_weight_bytes"] == 4
    assert storage["ideal_packed_weight_bytes"] == 4
    assert storage["scale_zero_bytes"] == 8
    assert storage["g_idx_bytes"] == 32
    assert storage["full_model_effective_bits_with_g_idx"] == 44.0


def test_affine_codebook_cli_validation_and_bits_none(monkeypatch) -> None:
    with pytest.raises(SystemExit):
        local.main(
            [
                "quantize",
                "--format",
                "int-codebook-affine",
                "--codepoints",
                "6",
                "--bits",
                "3",
            ]
        )
    with pytest.raises(SystemExit):
        local.main(["quantize", "--format", "int-codebook-affine"])

    captured = {}

    def fake_quantize(**kwargs):
        captured.update(kwargs)
        return {
            "estimated_packed_storage": {
                "estimated_realizable_packed_with_g_idx_bytes": 0,
                "full_model_effective_bits_with_g_idx": 0.0,
            },
            "artifact_size_bytes": 0,
        }

    monkeypatch.setattr(local, "quantize", fake_quantize)
    local.main(
        [
            "quantize",
            "--format",
            "int-codebook-affine",
            "--codepoints",
            "6",
            "--quiet",
        ]
    )

    assert captured["bits"] is None
    assert captured["quantization_format"] == "int-codebook-affine"
    assert captured["storage_scale_zero_dtype"] == "bfloat16"
    assert captured["output_dir"] == Path(
        "out/gptq/llama-3.2-vision-local-gptq-int-affine-k6-g128-c4"
    )


def test_tokenise_calibration_returns_plain_dicts() -> None:
    class DummyTokenizer:
        def __call__(self, text, **kwargs):
            assert text == "sample"
            assert kwargs["return_tensors"] == "pt"
            return common.transformers.BatchEncoding(
                {
                    "input_ids": torch.tensor([[1, 2, 3]]),
                    "attention_mask": torch.tensor([[1, 1, 1]]),
                }
            )

    result = common.tokenise_calibration(
        ["sample"],
        tokenizer=DummyTokenizer(),
        max_tokens=16,
    )

    assert type(result[0]) is dict
    torch.testing.assert_close(result[0]["input_ids"], torch.tensor([[1, 2, 3]]))


def test_vqav2_calibration_uses_processor_images_and_prompts(monkeypatch) -> None:
    fake_vqa = fake_vqa_module()
    fake_vqa.VQA.data = mock.Mock(
        return_value=[
            {
                "question_id": 1,
                "image": "image-a",
                "question": "what is shown?",
                "answers": [],
            }
        ]
    )
    monkeypatch.setattr(local, "vqa", fake_vqa)

    class RecordingProcessor:
        def __init__(self):
            self.calls = []

        def __call__(self, images, prompts, **kwargs):
            self.calls.append((images, prompts, kwargs))
            return {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.ones(1, 3, dtype=torch.long),
                "pixel_values": torch.randn(1, 1, 1, 1, 2, 2),
                "aspect_ratio_ids": torch.ones(1, 1, dtype=torch.long),
                "aspect_ratio_mask": torch.ones(1, 1, 1),
                "cross_attention_mask": torch.ones(1, 3, 1, 1),
            }

    processor = RecordingProcessor()

    batches = local.load_vqav2_calibration_batches(
        processor,
        n_samples=1,
        batch_size=1,
        split="validation",
        max_tokens=16,
        load_from_s3=False,
    )

    fake_vqa.VQA.data.assert_called_once_with(
        split="validation",
        limit=1,
        load_from_s3=False,
    )
    images, prompts, kwargs = processor.calls[0]
    assert images == [["image-a"]]
    assert prompts == ["<s>Question: what is shown?</s>"]
    assert kwargs["padding"] is True
    assert kwargs["truncation"] is True
    assert kwargs["max_length"] == 16
    assert batches[0]["input_ids"].shape == (1, 3)


def test_vqav2_train_calibration_uses_small_train_dataset(monkeypatch) -> None:
    data = local.datasets.Dataset.from_dict(
        {
            "question_id": [1, 2],
            "image": ["image-a", "image-b"],
            "question": ["question-a", "question-b"],
            "answers": [["a"], ["b"]],
        }
    )
    load_dataset = mock.Mock(return_value=data)
    monkeypatch.setattr(local.datasets, "load_dataset", load_dataset)

    class RecordingProcessor:
        def __init__(self):
            self.prompts = []

        def __call__(self, images, prompts, **kwargs):
            self.prompts.extend(prompts)
            return {"input_ids": torch.ones(len(prompts), 2, dtype=torch.long)}

    processor = RecordingProcessor()
    batches = local.load_vqav2_calibration_batches(
        processor,
        n_samples=2,
        batch_size=1,
        split="train",
        max_tokens=16,
        load_from_s3=False,
    )

    load_dataset.assert_called_once_with(
        local.VQAV2_TRAIN_CALIBRATION_DATASET,
        split="train",
    )
    assert len(batches) == 2
    assert all("Question: question-" in prompt for prompt in processor.prompts)


def test_public_synthetic_s3_path_resolves_data_root() -> None:
    path = (
        "s3://graphcore-research-public/2026-llama-mobile/data/generation/"
        "llama-3.2-11b-vision-instruct/imagenet-train/new-prompts-1280k/"
    )

    assert local._parse_public_synthetic_s3_path(path) == (
        "graphcore-research-public",
        "2026-llama-mobile/data",
        "generation/llama-3.2-11b-vision-instruct/imagenet-train/"
        "new-prompts-1280k",
    )


def test_public_synthetic_s3_path_rejects_raw_dataset() -> None:
    with pytest.raises(ValueError, match="generated rollout"):
        local._parse_public_synthetic_s3_path(
            "s3://bucket/project/data/datasets/imagenet-train/"
        )


def test_stage_public_synthetic_data_downloads_rollout_and_images(
    monkeypatch,
    tmp_path,
) -> None:
    import train_data
    import utility

    monkeypatch.setattr(utility, "LOCAL_DATA_PATH", str(tmp_path))
    monkeypatch.setattr(
        train_data,
        "load_config",
        lambda _: SimpleNamespace(dataset_name="imagenet", split="train"),
    )
    downloads = []

    def record_download(bucket, prefix, destination, **kwargs):
        downloads.append((bucket, prefix, destination, kwargs))
        if prefix.endswith("datasets/imagenet-train") and not (
            destination / "state.json"
        ).exists():
            destination.mkdir(parents=True, exist_ok=True)
            (destination / "state.json").write_text(
                json.dumps(
                    {
                        "_data_files": [
                            {"filename": f"data-{index:05d}-of-00010.arrow"}
                            for index in range(10)
                        ]
                    }
                )
            )

    monkeypatch.setattr(
        local,
        "_download_public_s3_prefix",
        record_download,
    )

    relative = local._stage_public_synthetic_data(
        "s3://public/project/data/generation/model/imagenet-train/run"
    )

    assert relative == "generation/model/imagenet-train/run"
    assert [download[1] for download in downloads] == [
        "project/data/generation/model/imagenet-train/run",
        "project/data/datasets/imagenet-train",
        "project/data/datasets/imagenet-train",
    ]
    manifest = json.loads(
        (tmp_path / relative / local.PUBLIC_SYNTHETIC_MANIFEST).read_text()
    )
    assert manifest["image_shards"] == 8
    assert len(manifest["image_files"]) == 8


def test_load_public_synthetic_calibration_uses_staged_subset(tmp_path) -> None:
    image_path = tmp_path / "images"
    local.datasets.Dataset.from_dict(
        {"index": [10, 20], "image": ["image-10", "image-20"]}
    ).save_to_disk(image_path)
    image_files = [str(path) for path in image_path.glob("*.arrow")]

    rollout_path = tmp_path / "rollout"
    (rollout_path / "out").mkdir(parents=True)
    (rollout_path / local.PUBLIC_SYNTHETIC_MANIFEST).write_text(
        json.dumps({"image_files": image_files})
    )
    (rollout_path / "out" / "out.jsonl").write_text(
        "\n".join(
            [
                json.dumps([10, "<|begin_of_text|>output-10"]),
                json.dumps([20, "<|begin_of_text|>output-20"]),
            ]
        )
        + "\n"
    )

    class RecordingProcessor:
        def __init__(self):
            self.texts = []

        def __call__(self, images, texts, **kwargs):
            self.texts.extend(texts)
            return {"input_ids": torch.ones(len(texts), 2, dtype=torch.long)}

    processor = RecordingProcessor()
    batches = local._load_public_synthetic_calibration_batches(
        processor,
        local_paths=[rollout_path],
        n_samples=2,
        batch_size=1,
        max_tokens=16,
    )

    assert len(batches) == 2
    assert set(processor.texts) == {"output-10", "output-20"}


def _tiny_multimodal_batch() -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
        "pixel_values": torch.randn(1, 1, 1, 1, 2, 2),
        "aspect_ratio_ids": torch.ones(1, 1, dtype=torch.long),
        "aspect_ratio_mask": torch.ones(1, 1, 1),
        "cross_attention_mask": torch.ones(1, 4, 1, 1),
    }


def test_deferred_vision_mask_is_exact_and_reuses_bounded_cache() -> None:
    local.clear_vision_attention_mask_cache()
    aspect_ratio_mask = torch.tensor([[1, 0]])
    deferred = local.DeferredVisionAttentionMask(
        aspect_ratio_mask=aspect_ratio_mask,
        num_patches=3,
        target_length=4,
        dtype=torch.float32,
    )

    expected = local._prepare_aspect_ratio_attention_mask(
        aspect_ratio_mask,
        num_patches=3,
        target_length=4,
        dtype=torch.float32,
    )
    first = deferred.to("cpu")
    second = deferred.to("cpu")

    torch.testing.assert_close(first, expected)
    assert first.data_ptr() == second.data_ptr()
    assert len(local._VISION_ATTENTION_MASK_CACHE) == 1


def test_projector_input_stream_matches_materialized_and_can_release_sources() -> None:
    context = local.VisionBatchContext(
        batch_size=1,
        num_concurrent_media=1,
        num_tiles=1,
        num_patches=2,
        num_padding_patches=0,
        dim=2,
        aspect_ratio_ids=torch.ones(1, 1, dtype=torch.long),
        attention_mask=local.DeferredVisionAttentionMask(
            aspect_ratio_mask=torch.ones(1, 1),
            num_patches=2,
            target_length=2,
            dtype=torch.float32,
        ),
    )
    global_outputs = local.LayerInputs(
        args=[[torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])]],
        kwargs=[{}],
    )
    local_capture = local.LayerInputs(
        args=[[torch.tensor([[[5.0, 6.0], [7.0, 8.0]]])]],
        kwargs=[{}],
    )
    captures = {0: local_capture}
    materialized = local.build_projector_inputs(
        global_outputs,
        captures,
        [context],
        [0],
        device="cpu",
    )

    streamed = list(
        local.iter_projector_inputs(
            global_outputs,
            captures,
            [context],
            [0],
            device="cpu",
            consume=True,
        )
    )

    torch.testing.assert_close(streamed[0][0][0], materialized.args[0][0])
    assert global_outputs.args[0] == []
    assert local_capture.args[0] == []


def test_quantize_full_multimodal_targets_vision_cross_projector_and_lm_head(
    monkeypatch,
    tmp_path,
) -> None:
    torch.manual_seed(625464)
    model = TinyFullMllama()
    monkeypatch.setattr(local, "load_model", lambda *_, **__: model)
    monkeypatch.setattr(
        local.transformers.AutoProcessor,
        "from_pretrained",
        mock.Mock(return_value=DummyProcessor()),
    )
    monkeypatch.setattr(
        local,
        "load_vqav2_calibration_batches",
        mock.Mock(return_value=[_tiny_multimodal_batch()]),
    )

    metadata = local.quantize(
        model_name="model",
        output_dir=tmp_path,
        bits=4,
        target_scope="full-multimodal",
        calibration_samples=1,
        calibration_max_tokens=16,
        device="cpu",
        torch_dtype="float32",
        verbose=False,
    )

    full_names = {log["full_name"] for log in metadata["quantization_log"]}
    assert metadata["target_scope_option"] == "full-multimodal"
    assert metadata["target_scope"] == "mllama_full_multimodal_heavy_linears"
    assert metadata["calibration_source"] == "vqav2"
    assert metadata["calibration"]["split"] == "train"
    assert (
        metadata["calibration"]["dataset"]
        == local.VQAV2_TRAIN_CALIBRATION_DATASET
    )
    assert "prototype_warning" not in metadata["calibration"]
    assert metadata["skipped_cross_attention_layers"] == []
    assert model.model.language_model.layers[1].called
    assert "model.vision_model.transformer.layers.0.self_attn.q_proj" in full_names
    assert "model.vision_model.global_transformer.layers.0.mlp.fc2" in full_names
    assert "model.language_model.layers.1.cross_attn.k_proj" in full_names
    assert "model.multi_modal_projector" in full_names
    assert "lm_head" in full_names
    assert metadata["estimated_packed_storage"]["unmatched_quantized_tensor_names"] == []


def test_s3d8_lm_head_offload_moves_only_completed_modules(monkeypatch) -> None:
    model = TinyFullMllama()
    vision_stack = local.mllama_vision_layers(model)
    text_stack = local.mllama_text_layers(model)
    projector = model.model.multi_modal_projector
    lm_head = model.lm_head

    language_to = mock.Mock(return_value=text_stack.language_model)
    vision_to = mock.Mock(return_value=vision_stack.vision_model)
    projector_to = mock.Mock(return_value=projector)
    lm_head_to = mock.Mock(return_value=lm_head)
    monkeypatch.setattr(text_stack.language_model, "to", language_to)
    monkeypatch.setattr(vision_stack.vision_model, "to", vision_to)
    monkeypatch.setattr(projector, "to", projector_to)
    monkeypatch.setattr(lm_head, "to", lm_head_to)
    empty_cache = mock.Mock()
    monkeypatch.setattr(local.torch.cuda, "empty_cache", empty_cache)

    local.offload_completed_modules_for_lm_head(
        vision_stack,
        text_stack,
        projector,
        lm_head,
        "cuda:0",
    )

    language_to.assert_called_once_with("cpu")
    vision_to.assert_called_once_with("cpu")
    projector_to.assert_called_once_with("cpu")
    lm_head_to.assert_called_once_with("cuda:0")
    empty_cache.assert_called_once_with()


def test_quantize_full_multimodal_s3d8_checkpoint_includes_full_targets(
    monkeypatch,
    tmp_path,
) -> None:
    torch.manual_seed(625464)
    model = TinyFullMllama()
    monkeypatch.setattr(local, "load_model", lambda *_, **__: model)
    monkeypatch.setattr(
        local.transformers.AutoProcessor,
        "from_pretrained",
        mock.Mock(return_value=DummyProcessor()),
    )
    monkeypatch.setattr(
        local,
        "load_vqav2_calibration_batches",
        mock.Mock(return_value=[_tiny_multimodal_batch()]),
    )

    metadata = local.quantize(
        model_name="model",
        output_dir=tmp_path,
        bits=4,
        quantization_format="s3d8",
        target_scope="full-multimodal",
        calibration_samples=1,
        calibration_max_tokens=16,
        device="cpu",
        torch_dtype="float32",
        verbose=False,
    )

    checkpoint_path = tmp_path / local.S3D8_CHECKPOINT_FILENAME
    assert metadata["quantized_checkpoint_path"] == str(checkpoint_path)
    assert checkpoint_path.exists()
    state = safetensors.torch.load_file(checkpoint_path)
    loaded = TinyFullMllama()
    QT.load_convert(loaded, state)
    assert isinstance(
        loaded.model.vision_model.transformer.layers[0].self_attn.q_proj.weight,
        QT.Sign3D8Weight,
    )
    assert isinstance(
        loaded.model.language_model.layers[1].cross_attn.k_proj.weight,
        QT.Sign3D8Weight,
    )
    assert isinstance(loaded.model.multi_modal_projector.weight, QT.Sign3D8Weight)
    assert isinstance(loaded.lm_head.weight, QT.Sign3D8Weight)


def test_quantize_s3d8_writes_dense_and_quantized_artifacts(
    monkeypatch, tmp_path
) -> None:
    torch.manual_seed(625464)
    model = TinyMllama()
    monkeypatch.setattr(local, "load_model", lambda *_, **__: model)
    monkeypatch.setattr(
        local.transformers.AutoProcessor,
        "from_pretrained",
        mock.Mock(return_value=DummyProcessor()),
    )
    monkeypatch.setattr(local, "load_c4_calibration", lambda **_: ["a", "b"])
    monkeypatch.setattr(
        local,
        "tokenise_calibration",
        lambda *_, **__: [
            {"input_ids": torch.tensor([1, 2, 3, 4])},
            {"input_ids": torch.tensor([4, 3, 2, 1])},
        ],
    )

    metadata = local.quantize(
        model_name="model",
        output_dir=tmp_path,
        bits=4,
        quantization_format="s3d8",
        group_size=128,
        batch_size=2,
        calibration_samples=2,
        calibration_data_min_length=0,
        device="cpu",
        torch_dtype="float32",
        verbose=False,
    )

    checkpoint_path = tmp_path / local.S3D8_CHECKPOINT_FILENAME
    assert metadata["artifact_type"] == "dense_dequantized_local_gptq_s3d8"
    assert metadata["format"] == "s3d8"
    assert metadata["bits"] is None
    assert metadata["effective_weight_bits"] == 8 / 3
    assert metadata["quantized_checkpoint_path"] == str(checkpoint_path)
    assert metadata["n_quantized_modules"] == 14
    assert checkpoint_path.exists()
    assert (tmp_path / "model.safetensors").exists()
    assert (tmp_path / "processor_config.json").exists()

    first_log = metadata["quantization_log"][0]
    assert first_log["quantization_format"] == "s3d8"
    assert first_log["centroid_shape"] == [32, 3]
    assert "zero_shape" not in first_log
    assert "g_idx_shape" not in first_log
    storage = metadata["estimated_packed_storage"]
    assert storage["quantized_tensor_count"] == 14
    assert storage["g_idx_bytes"] == 0
    assert storage["centroid_bytes"] > 0

    state = safetensors.torch.load_file(checkpoint_path)
    assert "_quantisation_meta" in state
    loaded = TinyMllama()
    QT.load_convert(loaded, state)
    target = loaded.model.language_model.layers[0].self_attn.q_proj.weight
    skipped = loaded.model.language_model.layers[1].cross_attn.weight
    assert isinstance(target, QT.Sign3D8Weight)
    assert isinstance(skipped, QT.UnquantisedWeight)


def test_evaluate_writes_outputs(monkeypatch, tmp_path) -> None:
    fake_vqa = fake_vqa_module(
        evaluate=mock.Mock(return_value=[{"id": 1, "output": "yes", "accuracy": 1.0}])
    )
    model = mock.Mock()
    model.eval = mock.Mock()
    monkeypatch.setattr(
        local.transformers.MllamaForConditionalGeneration,
        "from_pretrained",
        mock.Mock(return_value=model),
    )
    monkeypatch.setattr(local.transformers.AutoProcessor, "from_pretrained", mock.Mock())
    monkeypatch.setattr(local, "vqa", fake_vqa)
    (tmp_path / "model.safetensors").write_text("weights")

    summary = local.evaluate(
        model_dir=tmp_path,
        output_dir=tmp_path / "eval",
        tasks=["vqa"],
        n_examples=1,
        device="cpu",
        torch_dtype="float32",
    )

    fake_vqa.TASKS["vqa"].data.assert_called_once_with(
        limit=1,
        load_from_s3=False,
    )
    assert summary["tasks"]["vqa"]["accuracy"] == 1.0
    assert summary["load_vqa_from_s3"] is False
    assert summary["vqa_s3_path"] is None
    assert summary["vqa_s3_local_path"] is None
    assert (tmp_path / "eval" / "summary.json").exists()
    assert (tmp_path / "eval" / "summary.partial.json").exists()
    assert (tmp_path / "eval" / "vqa.jsonl").exists()


def test_evaluate_streams_jsonl_before_task_completes(monkeypatch, tmp_path) -> None:
    def evaluate_then_fail(**kwargs):
        yield {"id": 1, "output": "yes", "accuracy": 1.0}
        raise RuntimeError("evaluation stopped")

    fake_vqa = fake_vqa_module(evaluate=evaluate_then_fail)
    model = mock.Mock()
    model.eval = mock.Mock()
    monkeypatch.setattr(
        local.transformers.MllamaForConditionalGeneration,
        "from_pretrained",
        mock.Mock(return_value=model),
    )
    monkeypatch.setattr(local.transformers.AutoProcessor, "from_pretrained", mock.Mock())
    monkeypatch.setattr(local, "vqa", fake_vqa)
    (tmp_path / "model.safetensors").write_text("weights")

    with pytest.raises(RuntimeError, match="evaluation stopped"):
        local.evaluate(
            model_dir=tmp_path,
            output_dir=tmp_path / "eval",
            tasks=["vqa"],
            n_examples=1,
            device="cpu",
            torch_dtype="float32",
        )

    assert (tmp_path / "eval" / "vqa.jsonl").read_text().splitlines() == [
        '{"id": 1, "output": "yes", "accuracy": 1.0}'
    ]


def test_evaluate_resume_appends_missing_examples(monkeypatch, tmp_path) -> None:
    fake_vqa = fake_vqa_module(
        evaluate=mock.Mock(return_value=[{"id": 2, "output": "no", "accuracy": 0.0}])
    )
    fake_vqa.TASKS["vqa"].data.return_value = ["done", "remaining"]
    model = mock.Mock()
    model.eval = mock.Mock()
    monkeypatch.setattr(
        local.transformers.MllamaForConditionalGeneration,
        "from_pretrained",
        mock.Mock(return_value=model),
    )
    monkeypatch.setattr(local.transformers.AutoProcessor, "from_pretrained", mock.Mock())
    monkeypatch.setattr(local, "vqa", fake_vqa)
    (tmp_path / "model.safetensors").write_text("weights")
    output_dir = tmp_path / "eval"
    output_dir.mkdir()
    (output_dir / "vqa.jsonl").write_text(
        '{"id": 1, "output": "yes", "accuracy": 1.0}\n'
    )

    summary = local.evaluate(
        model_dir=tmp_path,
        output_dir=output_dir,
        tasks=["vqa"],
        n_examples=2,
        device="cpu",
        torch_dtype="float32",
        resume=True,
    )

    assert fake_vqa.evaluate.call_args.kwargs["data"] == ["remaining"]
    assert summary["tasks"]["vqa"]["n_examples"] == 2
    assert (output_dir / "vqa.jsonl").read_text().splitlines() == [
        '{"id": 1, "output": "yes", "accuracy": 1.0}',
        '{"id": 2, "output": "no", "accuracy": 0.0}',
    ]


def test_evaluate_resume_uses_missing_dataset_ids(monkeypatch, tmp_path) -> None:
    fake_vqa = fake_vqa_module(
        evaluate=mock.Mock(return_value=[{"id": 2, "output": "no", "accuracy": 0.0}])
    )

    class DatasetWithIds:
        column_names = ["question_id", "question"]

        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, key):
            if isinstance(key, str):
                return [row[key] for row in self.rows]
            return self.rows[key]

        def select(self, indices):
            return DatasetWithIds([self.rows[index] for index in indices])

    data = DatasetWithIds(
        [
            {"question_id": 1, "question": "done"},
            {"question_id": 2, "question": "remaining"},
            {"question_id": 3, "question": "done"},
        ]
    )
    fake_vqa.TASKS["vqa"].data.return_value = data
    model = mock.Mock()
    model.eval = mock.Mock()
    monkeypatch.setattr(
        local.transformers.MllamaForConditionalGeneration,
        "from_pretrained",
        mock.Mock(return_value=model),
    )
    monkeypatch.setattr(local.transformers.AutoProcessor, "from_pretrained", mock.Mock())
    monkeypatch.setattr(local, "vqa", fake_vqa)
    (tmp_path / "model.safetensors").write_text("weights")
    output_dir = tmp_path / "eval"
    output_dir.mkdir()
    (output_dir / "vqa.jsonl").write_text(
        "\n".join(
            [
                '{"id": 3, "output": "maybe", "accuracy": 1.0}',
                '{"id": 1, "output": "yes", "accuracy": 1.0}',
            ]
        )
        + "\n"
    )

    summary = local.evaluate(
        model_dir=tmp_path,
        output_dir=output_dir,
        tasks=["vqa"],
        n_examples=3,
        device="cpu",
        torch_dtype="float32",
        resume=True,
    )

    evaluated_data = fake_vqa.evaluate.call_args.kwargs["data"]
    assert evaluated_data["question_id"] == [2]
    assert summary["tasks"]["vqa"]["n_examples"] == 3
    assert summary["tasks"]["vqa"]["accuracy"] == pytest.approx(2 / 3)
    assert (output_dir / "vqa.jsonl").read_text().splitlines() == [
        '{"id": 3, "output": "maybe", "accuracy": 1.0}',
        '{"id": 1, "output": "yes", "accuracy": 1.0}',
        '{"id": 2, "output": "no", "accuracy": 0.0}',
    ]


def test_select_remaining_eval_data_all_examples_completed() -> None:
    data = common.datasets.Dataset.from_dict({"value": [1, 2]})

    remaining = common.select_remaining_eval_data(data, completed_count=2)

    assert len(remaining) == 0


def test_evaluate_refuses_existing_output_without_resume_or_overwrite(
    monkeypatch, tmp_path
) -> None:
    fake_vqa = fake_vqa_module()
    model = mock.Mock()
    model.eval = mock.Mock()
    monkeypatch.setattr(
        local.transformers.MllamaForConditionalGeneration,
        "from_pretrained",
        mock.Mock(return_value=model),
    )
    monkeypatch.setattr(local.transformers.AutoProcessor, "from_pretrained", mock.Mock())
    monkeypatch.setattr(local, "vqa", fake_vqa)
    (tmp_path / "model.safetensors").write_text("weights")
    output_dir = tmp_path / "eval"
    output_dir.mkdir()
    (output_dir / "vqa.jsonl").write_text(
        '{"id": 1, "output": "yes", "accuracy": 1.0}\n'
    )

    with pytest.raises(FileExistsError, match="already exists"):
        local.evaluate(
            model_dir=tmp_path,
            output_dir=output_dir,
            tasks=["vqa"],
            n_examples=1,
            device="cpu",
            torch_dtype="float32",
        )


def test_load_eval_data_syncs_explicit_vqa_s3_path(monkeypatch, tmp_path) -> None:
    selected_columns = mock.Mock()
    shuffled = mock.Mock()
    dataset = mock.Mock()
    dataset.select_columns.return_value = selected_columns
    selected_columns.shuffle.return_value = shuffled
    shuffled.select.return_value = ["row"]

    run = mock.Mock()
    load_from_disk = mock.Mock(return_value=dataset)
    monkeypatch.setattr(common.subprocess, "run", run)
    monkeypatch.setattr(common.datasets, "load_from_disk", load_from_disk)

    local_path = tmp_path / "vqav2-validation"
    result = common.load_eval_data(
        fake_vqa_module().TASKS,
        "vqa",
        n_examples=3,
        vqa_s3_path="s3://bucket/path/vqav2-validation",
        vqa_s3_local_path=local_path,
    )

    run.assert_called_once_with(
        [
            "aws",
            "s3",
            "sync",
            "--no-sign-request",
            "s3://bucket/path/vqav2-validation",
            str(local_path),
        ],
        check=True,
    )
    load_from_disk.assert_called_once_with(str(local_path))
    dataset.select_columns.assert_called_once_with(
        ["question_id", "image_id", "question", "image", "answers"]
    )
    selected_columns.shuffle.assert_called_once_with(625464)
    shuffled.select.assert_called_once_with(range(3))
    assert result == ["row"]
