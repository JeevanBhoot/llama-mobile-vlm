# Copyright (c) 2026 Graphcore Ltd. All rights reserved.

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

import pytest
import torch


def fake_vqa_module(evaluate=None):
    task = SimpleNamespace(
        METRICS=("accuracy",),
        RELAXED_METRICS=("accuracy_relaxed",),
        data=mock.Mock(return_value=["a"]),
    )
    fake_vqa = ModuleType("eval.vqa")
    fake_vqa.TASKS = {"vqa": task}
    fake_vqa.evaluate = evaluate or mock.Mock()
    return fake_vqa


sys.modules["eval.vqa"] = fake_vqa_module()

import gptq.common as common
import gptq.local as local


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
