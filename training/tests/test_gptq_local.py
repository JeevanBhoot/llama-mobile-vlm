# Copyright (c) 2026 Graphcore Ltd. All rights reserved.

import json
import math
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import datasets
import pytest
import safetensors.torch
import torch
import weight_formats.quantisation_training as QT
from transformers import (
    MllamaConfig,
    MllamaForConditionalGeneration,
    MllamaTextConfig,
    MllamaVisionConfig,
)

import gptq.local as local


def tiny_mllama() -> MllamaForConditionalGeneration:
    vision = MllamaVisionConfig(
        hidden_size=8,
        num_hidden_layers=1,
        num_global_layers=1,
        attention_heads=2,
        num_channels=1,
        intermediate_size=16,
        vision_output_dim=16,
        image_size=2,
        patch_size=1,
        max_num_tiles=1,
        supported_aspect_ratios=[[1, 1]],
        intermediate_layers_indices=[0],
    )
    text = MllamaTextConfig(
        vocab_size=32,
        hidden_size=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        intermediate_size=16,
        max_position_embeddings=16,
        cross_attention_layers=[1],
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    config = MllamaConfig(
        vision_config=vision,
        text_config=text,
        image_token_index=31,
    )
    return MllamaForConditionalGeneration(config)


class DummyProcessor:
    tokenizer = object()

    def save_pretrained(self, output_dir: Path) -> None:
        Path(output_dir, "processor_config.json").write_text("{}")


def multimodal_batch() -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor([[1, 31, 3, 4]]),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
        "pixel_values": torch.randn(1, 1, 1, 1, 2, 2),
        "aspect_ratio_ids": torch.ones(1, 1, dtype=torch.long),
        "aspect_ratio_mask": torch.ones(1, 1, 1),
        "cross_attention_mask": torch.ones(1, 4, 1, 1),
    }


def test_module_cli_appendix_defaults(monkeypatch) -> None:
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
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gptq",
            "quantize",
            "--format",
            "s3d8",
            "--target-scope",
            "full-multimodal",
            "--calibration-source",
            "synthetic",
            "--calibration-data-path",
            "samples",
            "--calibration-samples",
            "512",
            "--calibration-max-tokens",
            "1024",
            "--batch-size",
            "1",
            "--group-size",
            "128",
            "--blocksize",
            "128",
            "--damp-percent",
            "0.05",
            "--damp-auto-increment",
            "0.01",
            "--torch-dtype",
            "bfloat16",
            "--quiet",
        ],
    )

    runpy.run_module("gptq", run_name="__main__")

    assert captured["output_dir"] == Path(
        "out/gptq/llama-3.2-vision-local-gptq-s3d8-synthetic-full"
    )
    assert {
        key: captured[key]
        for key in (
            "target_scope",
            "calibration_source",
            "calibration_samples",
            "calibration_max_tokens",
            "batch_size",
            "group_size",
            "blocksize",
            "damp_percent",
            "damp_auto_increment",
            "desc_act",
            "act_group_aware",
            "sym",
            "mse",
            "torch_dtype",
        )
    } == {
        "target_scope": "full-multimodal",
        "calibration_source": "synthetic",
        "calibration_samples": 512,
        "calibration_max_tokens": 1024,
        "batch_size": 1,
        "group_size": 128,
        "blocksize": 128,
        "damp_percent": 0.05,
        "damp_auto_increment": 0.01,
        "desc_act": False,
        "act_group_aware": True,
        "sym": True,
        "mse": 0.0,
        "torch_dtype": "bfloat16",
    }


def test_synthetic_pairs_images_and_text(monkeypatch) -> None:
    import train_data

    datums = [
        SimpleNamespace(image="image-a", out="<|begin_of_text|>prompt-a"),
        SimpleNamespace(image="image-b", out="<|begin_of_text|>prompt-b"),
    ]
    dataset = mock.Mock()
    dataset.get_datums.return_value = iter(datums)
    monkeypatch.setattr(train_data, "Dataset", mock.Mock(return_value=dataset))
    processor = mock.Mock(return_value={"input_ids": torch.ones(2, 2)})

    batches = local.load_synthetic_calibration_batches(
        processor,
        calibration_data_paths=[Path("samples")],
        n_samples=2,
        batch_size=2,
        max_tokens=16,
    )

    assert len(batches) == 1
    assert processor.call_args.args == (
        [["image-a"], ["image-b"]],
        ["prompt-a", "prompt-b"],
    )


def test_full_mllama_checkpoint_roundtrip(monkeypatch, tmp_path) -> None:
    torch.manual_seed(625464)
    model = tiny_mllama()
    eligible = {
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
    }
    monkeypatch.setattr(local, "load_model", lambda *_, **__: model)
    monkeypatch.setattr(
        local.transformers.AutoProcessor,
        "from_pretrained",
        mock.Mock(return_value=DummyProcessor()),
    )
    monkeypatch.setattr(
        local,
        "load_synthetic_calibration_batches",
        mock.Mock(return_value=[multimodal_batch()]),
    )

    metadata = local.quantize(
        model_name="unused",
        output_dir=tmp_path,
        bits=None,
        quantization_format="s3d8",
        target_scope="full-multimodal",
        calibration_source="synthetic",
        calibration_data_paths=["samples"],
        calibration_samples=1,
        calibration_max_tokens=1024,
        batch_size=1,
        group_size=128,
        blocksize=128,
        device="cpu",
        torch_dtype="float32",
        verbose=False,
    )

    assert {log["full_name"] for log in metadata["quantization_log"]} == eligible
    assert metadata["n_quantized_modules"] == len(eligible)
    assert metadata["target_scope_option"] == "full-multimodal"
    assert metadata["calibration_source"] == "synthetic"
    assert metadata["quantization_batch_size"] == 1
    assert metadata["calibration_max_tokens"] == 1024
    assert metadata["calibration_prepared_examples"] == 1
    assert metadata["scale_dtype"] == "bfloat16"
    assert {
        key: metadata["gptq"][key]
        for key in (
            "group_size",
            "blocksize",
            "damp_percent",
            "damp_auto_increment",
            "desc_act",
            "act_group_aware",
            "sym",
            "mse",
        )
    } == {
        "group_size": 128,
        "blocksize": 128,
        "damp_percent": 0.05,
        "damp_auto_increment": 0.01,
        "desc_act": False,
        "act_group_aware": True,
        "sym": True,
        "mse": 0.0,
    }

    dense_model = MllamaForConditionalGeneration.from_pretrained(tmp_path)
    for name in eligible:
        dense_weight = dense_model.get_submodule(name).weight
        saved_weight = model.get_submodule(name).weight
        assert isinstance(saved_weight, QT.Sign3D8Weight)
        torch.testing.assert_close(dense_weight, saved_weight.master)

    format_state = safetensors.torch.load_file(
        tmp_path / local.S3D8_CHECKPOINT_FILENAME
    )
    restored = tiny_mllama()
    QT.load_convert(restored, format_state)
    for name in eligible:
        saved_weight = model.get_submodule(name).weight
        restored_weight = restored.get_submodule(name).weight
        assert isinstance(saved_weight, QT.Sign3D8Weight)
        assert isinstance(restored_weight, QT.Sign3D8Weight)
        torch.testing.assert_close(restored_weight.scale, saved_weight.scale)
        torch.testing.assert_close(restored_weight.centroids, saved_weight.centroids)


def test_packed_storage_keeps_tensor_tails() -> None:
    model = torch.nn.ModuleDict(
        {
            "first": torch.nn.Linear(4, 1, bias=False),
            "second": torch.nn.Linear(4, 1, bias=False),
        }
    )
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
        model,
        layer_logs=logs,
        bits=math.log2(6),
        scale_zero_dtype="bfloat16",
        quantization_format="int-codebook-affine",
        codepoints=6,
        sym=True,
    )

    assert storage["radix_packed_weight_bytes"] == 4


def test_evaluation_resume_uses_ids(monkeypatch, tmp_path) -> None:
    data = datasets.Dataset.from_dict(
        {
            "question_id": [1, 2, 3],
            "question": ["done", "remaining", "done"],
        }
    )
    task = SimpleNamespace(
        METRICS=("accuracy",),
        RELAXED_METRICS=(),
        data=mock.Mock(return_value=data),
    )
    evaluate = mock.Mock(
        return_value=[{"id": 2, "output": "no", "accuracy": 0.0}]
    )
    monkeypatch.setattr(
        local,
        "vqa",
        SimpleNamespace(TASKS={"vqa": task}, evaluate=evaluate),
    )
    monkeypatch.setattr(
        local.transformers.MllamaForConditionalGeneration,
        "from_pretrained",
        mock.Mock(return_value=torch.nn.Module()),
    )
    monkeypatch.setattr(
        local.transformers.AutoProcessor,
        "from_pretrained",
        mock.Mock(return_value=object()),
    )
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model.safetensors").touch()
    output_dir = tmp_path / "evaluation"
    output_dir.mkdir()
    (output_dir / "vqa.jsonl").write_text(
        '{"id": 3, "output": "maybe", "accuracy": 1.0}\n'
        '{"id": 1, "output": "yes", "accuracy": 1.0}\n'
    )

    summary = local.evaluate(
        model_dir=model_dir,
        output_dir=output_dir,
        tasks=["vqa"],
        n_examples=3,
        device="cpu",
        torch_dtype="float32",
        resume=True,
    )

    assert evaluate.call_args.kwargs["data"]["question_id"] == [2]
    assert summary["tasks"]["vqa"] == {
        "n_examples": 3,
        "accuracy": pytest.approx(2 / 3),
    }
    written_summary = json.loads((output_dir / "summary.json").read_text())
    assert written_summary["avg_primary"] == pytest.approx(2 / 3)
