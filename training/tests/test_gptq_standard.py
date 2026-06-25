# Copyright (c) 2026 Graphcore Ltd. All rights reserved.

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

import pytest


def fake_vqa_module(evaluate=None):
    task = SimpleNamespace(
        METRICS=("accuracy",),
        RELAXED_METRICS=("accuracy_relaxed",),
        data=mock.Mock(return_value=["a"]),
    )
    fake_vqa = ModuleType("eval.vqa")
    fake_vqa.TASKS = {"vqa": task, "ai2d": task}
    fake_vqa.evaluate = evaluate or mock.Mock()
    return fake_vqa


sys.modules["eval.vqa"] = fake_vqa_module()

import gptq.standard as standard


def test_default_output_dir() -> None:
    assert standard.default_output_dir(4) == Path(
        "out/gptq/llama-3.2-vision-gptq-int4-c4"
    )


def test_summarise_results() -> None:
    fake_vqa = fake_vqa_module()
    task_results = {
        "vqa": [
            {"accuracy": 0.0, "accuracy_relaxed": 1.0},
            {"accuracy": 1.0, "accuracy_relaxed": 1.0},
        ],
        "ai2d": [
            {"accuracy": 1.0, "accuracy_relaxed": 1.0},
            {"accuracy": 1.0, "accuracy_relaxed": 1.0},
        ],
    }

    summary = standard.summarise_results(
        task_results,
        task_definitions=fake_vqa.TASKS,
        include_relaxed_metrics=True,
    )

    assert summary["tasks"]["vqa"]["accuracy"] == 0.5
    assert summary["tasks"]["vqa"]["accuracy_relaxed"] == 1.0
    assert summary["tasks"]["ai2d"]["accuracy"] == 1.0
    assert summary["avg_primary"] == 0.75


def test_summarise_results_rejects_empty_task() -> None:
    fake_vqa = fake_vqa_module()
    with pytest.raises(ValueError, match="No results"):
        standard.summarise_results({"vqa": []}, task_definitions=fake_vqa.TASKS)


def test_quantize_writes_metadata(monkeypatch, tmp_path) -> None:
    class DummyQuantizeConfig:
        def __init__(self, **kwargs):
            vars(self).update(kwargs)

    class DummyGPTQModel:
        tokenizer = mock.Mock()

        @classmethod
        def load(cls, *args, **kwargs):
            cls.load_args = args
            cls.load_kwargs = kwargs
            return cls()

        def quantize(self, calibration_dataset, **kwargs):
            self.calibration_dataset = calibration_dataset
            self.quantize_kwargs = kwargs

        def save(self, output_dir):
            Path(output_dir, "model.safetensors").write_text("weights")

    monkeypatch.setattr(standard, "_get_gptq_model_cls", lambda: DummyGPTQModel)
    monkeypatch.setattr(
        standard,
        "_get_quantize_config_cls",
        lambda: DummyQuantizeConfig,
    )
    monkeypatch.setattr(standard, "load_c4_calibration", lambda **_: ["sample"])
    monkeypatch.setattr(
        standard,
        "tokenise_calibration",
        lambda *_, **__: [{"input_ids": 1}],
    )

    metadata = standard.quantize(
        model_name="model",
        output_dir=tmp_path,
        bits=4,
        calibration_samples=1,
    )

    assert metadata["model_name"] == "model"
    assert metadata["bits"] == 4
    assert metadata["artifact_size_bytes"] > 0
    assert (tmp_path / standard.METADATA_FILENAME).exists()


def test_evaluate_writes_outputs(monkeypatch, tmp_path) -> None:
    fake_vqa = fake_vqa_module(
        evaluate=mock.Mock(return_value=[{"id": 1, "output": "yes", "accuracy": 1.0}])
    )
    qmodel = mock.Mock()
    qmodel.model = mock.Mock()
    qmodel.model.eval = mock.Mock()
    gptq_model_cls = mock.Mock()
    gptq_model_cls.load.return_value = qmodel
    monkeypatch.setattr(standard, "_get_gptq_model_cls", lambda: gptq_model_cls)
    monkeypatch.setattr(
        standard.transformers.AutoProcessor, "from_pretrained", mock.Mock()
    )
    monkeypatch.setattr(standard, "vqa", fake_vqa)

    summary = standard.evaluate(
        model_dir=tmp_path,
        output_dir=tmp_path / "eval",
        tasks=["vqa"],
        n_examples=1,
        device="cpu",
    )

    assert summary["tasks"]["vqa"]["accuracy"] == 1.0
    qmodel.to.assert_called_once_with("cpu")
    assert (tmp_path / "eval" / "summary.json").exists()
    assert (tmp_path / "eval" / "summary.partial.json").exists()
    assert (tmp_path / "eval" / "vqa.jsonl").exists()
