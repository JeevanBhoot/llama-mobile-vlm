# Copyright (c) 2026 Graphcore Ltd. All rights reserved.

import unittest.mock as um
from dataclasses import asdict
from typing import Any, Iterable

import weight_formats.quantisation as Q
from transformers import AutoModelForCausalLM

import experiments as E


def test_run_experiment() -> None:
    xp = E.Experiment(
        name="test",
        model="meta-llama/Llama-3.2-11B-Vision-Instruct",
        task=E.Task.vqa(n_examples=2),
        quantisation=Q.LinearScalingFormat(
            Q.parse("E0M2"),
            Q.parse("BFLOAT16"),
            block_shape=(None, None),
            scaling="absmax",
        ),
        execution=E.Execution(device="cuda", batch_size=8, wandb=False),
        notes="testing",
    )

    def mock_vqa_evaluate(**kwargs: dict[str, Any]) -> Iterable[dict[str, Any]]:
        for i in range(len(kwargs["data"])):
            yield dict(
                id=i,
                output=f"Output {i}",
                accuracy=[0.5, 1.0][i],
            )

    def mock_from_pretrained(*args, **kwargs):
        return AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-70m")

    with um.patch("eval.vqa.evaluate", mock_vqa_evaluate), um.patch(
        "transformers.MllamaForConditionalGeneration.from_pretrained",
        mock_from_pretrained,
    ):
        out = E.run_experiment(xp)

    for k, v in asdict(xp).items():
        assert out[k] == v

    assert all(k in out for k in ["n_params", "n_bytes", "duration"])
    assert len(out["results"]) == 2
    assert out["n_examples"] == 2
    assert out["accuracy"] == 0.75
