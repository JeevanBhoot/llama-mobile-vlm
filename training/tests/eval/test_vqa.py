import unittest.mock as um
from typing import Any

import torch
import transformers

from eval import vqa


def test_vqa() -> None:
    # NOTE: Slow first time as it fetches the dataset

    data = vqa.VQA.data(limit=10)
    assert len(data) == 10
    assert set(data.column_names) == {
        "question_id",
        "image_id",
        "question",
        "image",
        "answers",
    }


def test_process_text() -> None:
    text = "\n\n  THE MAN is  \t  1.99m tall;;ALMOST two meters. \n \n"
    expected = "man is 1.99m tall almost 2 meters"
    assert expected == vqa._process_text(text)

    text = "London, has,population,of 8,866,180."
    expected = "london has population of 8866180"
    assert expected == vqa._process_text(text)


def test_vqa_evaluate_prediction() -> None:
    answers = ["Three musketeers", "3 Musketeers"] + ["4 swordsmen"] * 8
    out = "  The three MUSKETEERS! 4 swordsmen <|eot_id|>"
    results = vqa.VQA.evaluate_prediction(out, answers)
    assert results == dict(accuracy=2 / 3, accuracy_easy=1.0)


def test_chartqa_get_answer() -> None:
    texts = [
        "Answer: 1",
        "FINAL ANSWER: 2.<|eot_id|>",
        "**Answer**: 3. Then 4.",
        "*Answer*: 5!",
    ]
    out = [vqa.ChartQA._get_answer(text) for text in texts]
    expected = ["1", "2.", "3. Then 4.", "5!"]
    assert out == expected


def test_chartqa_parse_numeric() -> None:
    texts = [
        "Some text.",
        "Five",
        "3.1415",
        "135",
        "The number is 10, I think.",
        "$53.4",
        "13.5%",
    ]
    out = [vqa.ChartQA._parse_numeric(text) for text in texts]
    expected = [None, 5.0, 3.1415, 135.0, 10.0, 53.4, 13.5]
    assert out == expected


def test_chartqa_evaluate_prediction() -> None:
    base = "Some text. Some more text... {}"
    labels = ["green", "100", "100", "green", "100", "100"]
    hits = [
        base.format("Answer: Green."),
        base.format("FINAL ANSWER: 104.5"),
        # NOTE: Slightly more lenient for numeric answers
        base.format("Answer: 100. But there is more text after."),
    ]

    misses = [
        base.format("Answer: Green. But there is more text after."),
        base.format("Answer: 105.1"),
        base.format("The answer is 100."),
    ]
    out = [vqa.ChartQA.evaluate_prediction(x, l) for x, l in zip(hits + misses, labels)]
    expected = [{"accuracy": acc} for acc in [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]]
    assert out == expected


def test_ai2d_get_answer() -> None:
    texts = [
        "Answer: 1",
        "FINAL ANSWER: 2.<|eot_id|>",
        "**Answer**: 3. Then 4.",
        "*Answer*: 5!",
        "Correct option is: This",
    ]
    out = [vqa.AI2D._get_answer(text) for text in texts]
    expected = ["1", "2.", "3. Then 4.", "5!", "This"]
    assert out == expected


def test_ai2d_parse_numeric() -> None:
    texts = ["0)", "1)", "Answer is 2)", "3), I think.", "4)", "5)"]
    out = [vqa.AI2D._parse_numeric(text) for text in texts]
    expected = [None, 1, 2, 3, 4, None]
    assert out == expected


def test_ai2d_evaluate_prediction() -> None:
    base = "Some text. Some more text... {}"
    labels = ["0"] * 6  # labels go 0, 1, ... (instead of 1, 2, ...)
    hits = [
        base.format("Answer: 1)"),
        base.format("Correct option: 1)"),
        base.format("Answer is: 1). But there is more text after."),
    ]

    misses = [
        base.format("Answer: 1."),  # looking for N)
        base.format("Answer: 2)"),  # incorrect
        base.format("The correct option is 1)."),  # looking for answer/option:
    ]
    out = [vqa.AI2D.evaluate_prediction(x, l) for x, l in zip(hits + misses, labels)]
    expected = [{"accuracy": acc} for acc in [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]]
    print(out)
    print(expected)
    assert out == expected


def test_docvqa_evaluate_prediction() -> None:
    answers = ["ANSWER", "Something else"]
    texts = ["xxxxer", "    Bnswer    "]
    expected = [{"anls": x} for x in [0.0, 5 / 6]]
    for t, e in zip(texts, expected):
        assert vqa.DocVQA.evaluate_prediction(t, answers) == e


def test_evaluate() -> None:
    model = um.Mock()
    processor = transformers.AutoProcessor.from_pretrained(
        "meta-llama/Llama-3.2-11B-Vision-Instruct"
    )

    def mock_generate(**inp: dict[str, Any]):
        return torch.cat(
            [
                inp["input_ids"],
                processor.tokenizer.encode(" end", return_tensors="pt")[:, 1:],
            ],
            dim=1,
        )

    model.generate = um.Mock(side_effect=mock_generate)
    model.device = "cpu"

    limit = 2

    # NOTE: Might be slow as it downloads each dataset
    for task_name, task in vqa.TASKS.items():
        data = task.data(limit=limit)
        out = vqa.evaluate(
            model, processor, task_name=task_name, data=data, batch_size=1
        )
        expected_keys = {"id", "output", *task.METRICS}
        for x, y in zip(data, out):
            assert set(y.keys()) == expected_keys
            ids = [
                x.get(k)
                for k in ["id", "question_id", "questionId"]
                if x.get(k) is not None
            ]
            assert len(ids) == 1
            assert y["id"] == ids[0]
            assert y["output"] == " end"
