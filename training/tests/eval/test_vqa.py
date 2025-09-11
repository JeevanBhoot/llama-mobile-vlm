import unittest.mock as um
from typing import Any

import torch
import transformers
from PIL import Image

from eval import vqa


def test_vqa() -> None:
    example = next(vqa.VQA.get_examples())
    assert set(example.keys()) == {
        "question_id",
        "image_id",
        "question",
        "image",
        "answers",
        "prompt",
    }


def test_process_text() -> None:
    text = "\n\n  THE MAN is  \t  1.99m tall;;ALMOST two meters. \n \n"
    expected = "man is 1.99m tall almost 2 meters"
    assert expected == vqa.process_text(text)

    text = "London, has,population,of 8,866,180."
    expected = "london has population of 8866180"
    assert expected == vqa.process_text(text)


def test_evaluate_prediction() -> None:
    answers = ["Three musketeers", "3 Musketeers"] + ["4 swordsmen"] * 8
    out = "  The three MUSKETEERS! <|eot_id|>"
    assert vqa.evaluate_prediction(out, answers) == 2 / 3


def test_evaluate() -> None:
    model = um.Mock()
    processor = transformers.AutoProcessor.from_pretrained(
        "meta-llama/Llama-3.2-11B-Vision-Instruct"
    )

    def mock_generate(**inp: dict[str, Any]):
        return torch.cat(
            [
                inp["input_ids"],
                processor.tokenizer.encode(" bicycle", return_tensors="pt")[:, 1:],
            ],
            dim=1,
        )

    model.generate = um.Mock(side_effect=mock_generate)
    model.device = "cpu"

    examples = (
        dict(
            question_id=i,
            prompt="<|image|> What?",
            image=Image.fromarray(
                torch.randint(0, 256, (224, 224, 3), dtype=torch.uint8).numpy()
            ),
            answers=[{"answer": "car"}] + [{"answer": "bicycle"}] * i,
        )
        for i in range(5)
    )
    expected = (
        dict(id=i, output=" bicycle", accuracy=[0.0, 1 / 3, 2 / 3, 1.0][i])
        for i in range(4)
    )
    for o, expected_o in zip(
        vqa.evaluate(model, processor, examples, batch_size=1, n_examples=4), expected
    ):
        assert o == expected_o
