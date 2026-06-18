# Copyright (c) 2026 Graphcore Ltd. All rights reserved.

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

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


class TinyMllama(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(use_cache=True)
        self.embed_tokens = torch.nn.Embedding(32, 8)
        self.language_model = SimpleNamespace(
            model=SimpleNamespace(layers=torch.nn.ModuleList([TinyDecoderLayer(8)]))
        )
        self.layers = self.language_model.model.layers

    def forward(self, input_ids, **kwargs):
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states)[0]
        return SimpleNamespace(logits=hidden_states)

    def save_pretrained(self, output_dir, safe_serialization=True):
        Path(output_dir, "model.safetensors").write_text("weights")


class DummyProcessor:
    tokenizer = object()

    def save_pretrained(self, output_dir):
        Path(output_dir, "processor_config.json").write_text("{}")


def test_quantize_writes_dense_artifact(monkeypatch, tmp_path) -> None:
    torch.manual_seed(625464)
    monkeypatch.setattr(local, "load_model", lambda *_, **__: TinyMllama())
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
        calibration_samples=2,
        device="cpu",
        torch_dtype="float32",
    )

    assert metadata["artifact_type"] == "dense_dequantized_local_gptq"
    assert metadata["n_quantized_modules"] == 7
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
        torch_dtype="float32",
    )

    assert summary["tasks"]["vqa"]["accuracy"] == 1.0
    assert (tmp_path / "eval" / "summary.json").exists()
    assert (tmp_path / "eval" / "vqa.jsonl").exists()
