# Copyright (c) 2026 Graphcore Ltd. All rights reserved.

import unittest.mock as um
from typing import Any

import PIL
import pytest
import torch
import transformers

import train_data
from eval import outcompare


def test_get_out_wo_prompt() -> None:
    # NOTE: prompt is just str, out is str after tokenisation + generation
    processor = transformers.AutoProcessor.from_pretrained(
        "meta-llama/Llama-3.2-11B-Vision-Instruct"
    )
    datum = train_data.Datum(
        index=0,
        image=PIL.Image.new("RGB", (32, 32)),
        prompt="<|image|>Question:",
        out="Answer",
    )
    with pytest.raises(AssertionError):
        _ = outcompare._get_out_wo_prompt(datum, processor)
    datum.out = "<|begin_of_text|><|image|><|begin_of_text|>Question:Answer"
    assert outcompare._get_out_wo_prompt(datum, processor) == "Answer"


def test_evaluate() -> None:
    model = um.Mock()
    processor = transformers.AutoProcessor.from_pretrained(
        "meta-llama/Llama-3.2-11B-Vision-Instruct"
    )

    def mock_generate(**inp: dict[str, Any]):
        return torch.cat(
            [
                inp["input_ids"],
                processor.tokenizer.encode("AnsWER", return_tensors="pt")[:, 1:],
            ],
            dim=1,
        )

    model.generate = um.Mock(side_effect=mock_generate)
    model.device = "cpu"

    example = dict(
        index=0,
        image=PIL.Image.new("RGB", (32, 32)),
        prompt="<|image|>Question:",
        out="<|begin_of_text|><|image|><|begin_of_text|>Question:Answer",
    )

    # Hacky
    ds = train_data.Dataset.__new__(train_data.Dataset)
    ds.data = [example]
    config = train_data.GenerationConfig.default()
    config.sampling_settings.do_sample = False
    ds._configs = [config.to_dict()]

    expected = [dict(index=0, match_length_char=3, ref_char_length=6)]
    for o, expected_o in zip(
        outcompare.evaluate(model, processor, ds, char_limit=256, batch_size=1),
        expected,
    ):
        assert o == expected_o
