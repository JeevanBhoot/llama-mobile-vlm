"""
Evaluate Visual question-answering tasks

- VQAv2
    - https://visualqa.org/evaluation.html
    - Baseline accuracy: 75.6% (val, 1k sample), Meta: 75.2% (test)
    - NOTE: Using validation, not test set (HF missing ground truth on test)

- ChartQA
    - https://arxiv.org/abs/2203.10244
    - Baseline accuracy: 75.9% (test, 1k sample), Meta: 83.4% (test)
    - NOTE: Uses "relaxed accuracy": numeric answers should allow 5% tolerance

- DocVQA
    - https://arxiv.org/abs/2007.00398
    - Baseline accuracy: 84.1% (val, 1k sample), Meta: 88.4% (test)
    - NOTE: Uses Average Normalized Levenshtein Similarity (ANLS) metric
    - NOTE: InfographicVQA (https://arxiv.org/abs/2104.12756) defines 0.5 ANLS threshold
    which is used in this repo as well [threshold=1.0 gives 85% baseline accuracy]
    - NOTE: Using validation, not test set (HF missing ground truth on test)

- AI2D
    - https://prior.allenai.org/projects/diagram-understanding
    - Baseline accuracy: 63.5% (test, 1k sample), Meta: 91.1% (test)
        - TODO: Improve baseline accuracy
    - NOTE: Meta's proposed prompt is incomplete (missing MC options)

Prompt formatting inherited from:
https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/eval_details.md

Baseline: Llama-3.2-11B-Vision-Instruct

Tasks not supported: MMMU(-Pro), MathVista [NOTE: max_generation_length = 2048]
"""

import math
import subprocess
from dataclasses import dataclass
from typing import Any, Iterable, Literal, TypeAlias

import datasets
import regex as re
import torch
from PIL.ImageFile import ImageFile
from torchaudio.functional import edit_distance
from tqdm import tqdm
from transformers import MllamaForConditionalGeneration, MllamaProcessor

from utility import (
    LLAMA_PROMPT_TEMPLATES,
    LOCAL_DATA_PATH,
    S3_DATA_PATH,
    set_padding_side_left,
)

WORD_TO_NUM = {
    "none": "0",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
}

PUNCTUATION = [
    ";",
    r"/",
    "[",
    "]",
    '"',
    "{",
    "}",
    "(",
    ")",
    "=",
    "+",
    "\\",
    "_",
    "-",
    ">",
    "<",
    "@",
    "`",
    ",",
    "?",
    "!",
]


def _process_text(text: str) -> str:
    """
    VQA rules for processing text (see https://visualqa.org/evaluation.html)
        * Missing apostrophe addition: havent -> haven't etc.
    """

    # Replace \n, \t etc. with a whitespace
    text = re.sub(r"\s", " ", text)

    # Make lowercase
    text = text.lower()

    # Remove periods, except when decimal
    text = re.sub(r"\.(?!\d)", "", text)

    # Convert number words to digits
    text = re.sub(
        rf"\b({'|'.join(WORD_TO_NUM.keys())})\b",
        lambda match: WORD_TO_NUM[match.group()],
        text,
    )

    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", "", text)

    # Replace punctuation
    # - Delete commas from numbers
    text = re.sub(r"(?<=\d)(\,)(?=\d)", "", text)
    # - For others, replace with ' '
    pattern = rf"[{''.join(re.escape(char) for char in PUNCTUATION)}]"
    text = re.sub(pattern, " ", text)

    # Delete repeated whitespaces
    text = re.sub(r"\s+", " ", text)

    # Delete leading or trailing whitespaces
    text = text.strip()

    return text


@dataclass
class Batch:
    ids: list[int]
    images: list[ImageFile]
    prompts: list[str]
    answers: list[Any]


METRIC: TypeAlias = Literal["accuracy", "accuracy_easy", "anls"]


class Task:
    QUESTION_TEMPLATE: str
    MAX_NEW_TOKENS: int
    METRICS: list[METRIC]

    @classmethod
    def data(
        cls, split: str, limit: int | None, shuffle_seed: int | None, **kwargs: Any
    ) -> datasets.Dataset:
        raise NotImplementedError

    @classmethod
    def prepare_batch(cls, batch: dict[str, Any], system_template: str) -> Batch:
        raise NotImplementedError

    @classmethod
    def evaluate_prediction(cls, out: str, answer: Any) -> dict[str, float]:
        raise NotImplementedError


class VQA(Task):
    QUESTION_TEMPLATE = (
        "Look at the image carefully and answer this visual question. "
        "For yes/no questions, just respond Yes or No. "
        "If the answer is numeric, just respond with the number and nothing else. "
        "If the answer has multiple words, "
        "just respond with the words and absolutely nothing else. "
        "Never respond in a sentence or a phrase.\n "
        "Respond with as few words as possible.\n Question: {question}"
    )
    MAX_NEW_TOKENS = 25
    METRICS = ["accuracy", "accuracy_easy"]

    @classmethod
    def data(
        cls,
        split: str = "validation",
        limit: int | None = 4096,
        shuffle_seed: int | None = 625464,
        load_from_s3: bool = True,
    ) -> datasets.Dataset:
        cols = ["question_id", "image_id", "question", "image", "answers"]

        if load_from_s3:
            dir_name = f"vqav2-{split}"
            local_path = f"{LOCAL_DATA_PATH}/{dir_name}"
            s3_path = f"{S3_DATA_PATH}/{dir_name}"
            subprocess.run(["aws", "s3", "sync", s3_path, local_path], check=True)
            ds = datasets.load_from_disk(local_path)
        else:
            ds = datasets.load_dataset("lmms-lab/VQAv2", split=split)

        ds = ds.select_columns(cols)

        # NOTE: Images re-appear in consequtive questions, best to shuffle
        if shuffle_seed:
            ds = ds.shuffle(shuffle_seed)

        if limit is not None:
            ds = ds.select(range(limit))

        return ds

    @classmethod
    def prepare_batch(cls, batch: dict[str, Any], system_template: str) -> Batch:
        return Batch(
            ids=batch["question_id"],
            images=batch["image"],
            prompts=[
                system_template.format(prompt=cls.QUESTION_TEMPLATE.format(question=q))
                for q in batch["question"]
            ],
            answers=[[x["answer"] for x in a] for a in batch["answers"]],
        )

    @classmethod
    def evaluate_prediction(cls, out: str, answers: list[str]) -> dict[str, float]:
        out_norm = _process_text(out)
        answers_norm = [_process_text(answer) for answer in answers]

        # Accuracy = min{n_matches / 3, 1} (taken from instructions)
        n_matches = sum(
            [out_norm.startswith(answer) for answer in answers_norm if answer]
        )
        n_matches_easy = sum(
            [
                bool(re.search(rf"\b{re.escape(answer)}\b", out_norm))
                for answer in answers_norm
                if answer
            ]
        )
        return dict(
            accuracy=min(n_matches / 3, 1.0), accuracy_easy=min(n_matches_easy / 3, 1.0)
        )


class ChartQA(Task):
    QUESTION_TEMPLATE = (
        "You are provided a chart image and will be asked a question. "
        "You have to think through your answer and provide a step-by-step solution. "
        "Once you have the solution, write the final answer in at most a few words "
        'at the end with the phrase "FINAL ANSWER:". '
        "The question is: {question}<cot_start>Let's think step by step."
    )
    MAX_NEW_TOKENS = 512
    METRICS = ["accuracy"]

    @classmethod
    def data(
        cls,
        split: str = "test",
        limit: int | None = None,
        shuffle_seed: int | None = 77524,
    ) -> datasets.Dataset:
        cols = ["image", "query", "label"]

        ds = datasets.load_dataset("HuggingFaceM4/ChartQA", split=split)

        ds = ds.select_columns(cols)

        # Add an ID column
        ds = ds.add_column("id", column=range(len(ds)))

        # NOTE: Images re-appear consequtive questions, best to shuffle
        if shuffle_seed:
            ds = ds.shuffle(shuffle_seed)

        if limit is not None:
            ds = ds.select(range(limit))

        return ds

    @classmethod
    def prepare_batch(cls, batch: dict[str, Any], system_template: str) -> Batch:
        return Batch(
            ids=batch["id"],
            images=batch["image"],
            prompts=[
                system_template.format(prompt=cls.QUESTION_TEMPLATE.format(question=q))
                for q in batch["query"]
            ],
            # NOTE: There is only one answer per question
            answers=[x[0] for x in batch["label"]],
        )

    @classmethod
    def _get_answer(cls, text: str) -> str:
        # Extract text following "Answer: " / "Final answer: "
        #   - Allows for markdown '*' decorators
        match = re.search(r"(?i)\banswer[\*]*:[\*]*([\s\S]*)", text)
        if match:
            return match.group(1).strip().replace("<|eot_id|>", "")
        return ""

    @classmethod
    def _parse_numeric(cls, text: str) -> float | None:
        # Convert number words to digits
        text = text.lower()
        text = re.sub(
            rf"\b({'|'.join(WORD_TO_NUM.keys())})\b",
            lambda match: WORD_TO_NUM[match.group()],
            text,
        )

        # Extract just the numerical part (if it exists)
        match = re.search(r"[-]?\d*\.?\d+", text)
        if match:
            return float(match.group())
        return None

    @classmethod
    def evaluate_prediction(cls, out: str, label: str) -> dict[str, float]:
        answer = cls._get_answer(out)

        # If numeric, allow for 5% tolerance
        num_answer = cls._parse_numeric(answer)
        if num_answer is not None:
            try:
                num_label = float(label)
                match = 0.95 * num_label < num_answer < 1.05 * num_label
                return dict(accuracy=float(match))
            except ValueError:
                pass

        # Ignore case, punctuation, etc.
        return dict(accuracy=float(_process_text(answer) == _process_text(label)))


class DocVQA(Task):
    QUESTION_TEMPLATE = (
        "Read the text in the image carefully and answer the question "
        "with the text as seen exactly in the image. For yes/no questions, just "
        "respond Yes or No. If the answer is numeric, just respond with the number "
        "and nothing else. If the answer has multiple words, just respond with the "
        "words and absolutely nothing else. Never respond in a sentence or a phrase.\n "
        "Question: {question}"
    )
    MAX_NEW_TOKENS = 512
    METRICS = ["anls"]

    @classmethod
    def data(
        cls,
        split: str = "validation",
        limit: int | None = None,
        shuffle_seed: int | None = 37374,
    ) -> datasets.Dataset:
        ds = datasets.load_dataset("lmms-lab/DocVQA", name="DocVQA", split=split)

        if shuffle_seed:
            ds = ds.shuffle(shuffle_seed)

        if limit is not None:
            ds = ds.select(range(limit))

        return ds

    @classmethod
    def prepare_batch(cls, batch: dict[str, Any], system_template: str) -> Batch:
        return Batch(
            ids=batch["questionId"],
            images=batch["image"],
            prompts=[
                system_template.format(prompt=cls.QUESTION_TEMPLATE.format(question=q))
                for q in batch["question"]
            ],
            answers=batch["answers"],
        )

    @classmethod
    def _preprocess(cls, text: str) -> str:
        # Minimal preprocessing (case-insensitive per instructions)
        # TODO: eot token processing should probably be outside
        return text.lower().strip().replace("<|eot_id|>", "")

    @classmethod
    def evaluate_prediction(
        cls, out: str, answers: list[str], threshold: float = 0.5
    ) -> dict[str, float]:
        out = cls._preprocess(out)
        results = []
        for ans in answers:
            ans = cls._preprocess(ans)
            norm_dist = edit_distance(out, ans) / max(len(out), len(ans))
            # If normalised distance >= threshold, assume answer is wrong and set to 1
            norm_dist = norm_dist if norm_dist < threshold else 1
            results.append(1 - norm_dist)
        return dict(anls=max(results))  # report the best match


class AI2D(Task):
    # NOTE: Meta's template is missing options
    _META_TEMPLATE = (
        "Look at the scientific diagram carefully and answer the following "
        "question: {question}\n "
        "Think step by step and finally respond to the question with only the "
        'correct option number as "FINAL ANSWER". Let\'s think step by step.'
    )
    QUESTION_TEMPLATE = (
        "Look at the scientific diagram carefully and answer the following "
        "question: {question}\n "
        "Think step by step and finally respond to the question with only the "
        'correct option number as "FINAL ANSWER". Options:\n{options}'
        "Let's think step by step."
    )
    MAX_NEW_TOKENS = 400
    METRICS = ["accuracy"]

    @classmethod
    def _format_options(cls, options: list[str]) -> str:
        return "".join(f"{i}) {option}\n" for i, option in enumerate(options, start=1))

    @classmethod
    def data(
        cls,
        split: str = "test",
        limit: int | None = None,
        shuffle_seed: int | None = 85393,
    ) -> datasets.Dataset:
        ds = datasets.load_dataset("lmms-lab/ai2d", split=split)

        # Add an ID column
        ds = ds.add_column("id", column=range(len(ds)))

        if shuffle_seed:
            ds = ds.shuffle(shuffle_seed)

        if limit is not None:
            ds = ds.select(range(limit))

        return ds

    @classmethod
    def prepare_batch(cls, batch: dict[str, Any], system_template: str) -> Batch:
        return Batch(
            ids=batch["id"],
            images=batch["image"],
            prompts=[
                system_template.format(
                    prompt=cls.QUESTION_TEMPLATE.format(
                        question=q, options=cls._format_options(options)
                    )
                )
                for q, options in zip(batch["question"], batch["options"])
            ],
            answers=batch["answer"],
        )

    @classmethod
    def _get_answer(cls, text: str) -> str:
        # Extract text following "Answer: " / "Final answer: " / "Option: "
        #   - Allows for markdown '*' decorators
        #   - Allows also "correct answer/option is: "
        match = re.search(
            r"(?i)\b(?:answer|option)(?:\s+is)?[\*]*:[\*]*([\s\S]*)", text
        )
        if match:
            return match.group(1).strip().replace("<|eot_id|>", "")
        return ""

    @classmethod
    def _parse_numeric(cls, text: str) -> int | None:
        # In the answer, find "1)", "2)", etc. and return the index
        match = re.search(r"([1-4])\)", text)
        if match:
            return int(match.group(1))
        return None

    @classmethod
    def evaluate_prediction(cls, out: str, answer: str) -> dict[str, float]:
        num_answer = int(answer)
        num_out = cls._parse_numeric(cls._get_answer(out))
        # Ground truth starts from 0
        num_out = num_out - 1 if num_out is not None else None
        return dict(accuracy=float(num_out == num_answer))


TASKS: dict[str, Task] = dict(vqa=VQA, chartqa=ChartQA, docvqa=DocVQA, ai2d=AI2D)


def evaluate(
    model: MllamaForConditionalGeneration,
    processor: MllamaProcessor,
    task_name: str,
    data: datasets.Dataset,
    batch_size: int,
    system_template: str = LLAMA_PROMPT_TEMPLATES["instruct"],
    disable_progress: bool = False,
) -> Iterable[dict]:
    task = TASKS[task_name]

    n_batches = math.ceil(len(data) / batch_size)

    for batch in tqdm(
        data.iter(batch_size),
        desc=f"Evaluating {task_name}",
        total=n_batches,
        disable=disable_progress,
    ):
        p_batch = task.prepare_batch(batch, system_template=system_template)

        # Set to left-padding so we can call model.generate() when batched
        with set_padding_side_left(processor.tokenizer):
            inp = processor(
                [[img] for img in p_batch.images],
                p_batch.prompts,
                return_tensors="pt",
                padding=True,
            ).to(model.device)

        with torch.no_grad():
            out = model.generate(
                **inp,
                max_new_tokens=task.MAX_NEW_TOKENS,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        # Trim the input tokens
        n_inp_toks = inp["input_ids"].shape[-1]
        gen_tok_ids = [x[n_inp_toks:] for x in out]

        pad_token = processor.tokenizer.special_tokens_map["pad_token"]
        for id, out_ids, answers in zip(p_batch.ids, gen_tok_ids, p_batch.answers):
            out = processor.decode(out_ids).replace(pad_token, "")
            yield dict(
                id=id,
                output=out,
                **task.evaluate_prediction(out, answers),
            )
