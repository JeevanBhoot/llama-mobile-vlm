"""
Evaluate Visual Question Answering tasks
"""

import subprocess
from typing import Iterable, Optional

import datasets
import regex as re
import torch
from tqdm import tqdm
from transformers import MllamaForConditionalGeneration, MllamaProcessor

from utility import (
    LLAMA_PROMPT_TEMPLATES,
    LOCAL_DATA_PATH,
    S3_DATA_PATH,
    set_padding_side_left,
)


class VQA:
    QUESTION_TEMPLATE = (
        "Look at the image carefully and answer this visual question. "
        "For yes/no questions, just respond Yes or No. "
        "If the answer is numeric, just respond with the number and nothing else. "
        "If the answer has multiple words, "
        "just respond with the words and absolutely nothing else. "
        "Never respond in a sentence or a phrase.\n "
        "Respond with as few words as possible.\n Question: {question}"
    )

    @classmethod
    def data(
        cls,
        split: str = "validation",
        limit: Optional[int] = 4096,
        shuffle_seed: Optional[int] = 625464,
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
        # NOTE: Images re-appear consequtive questions, best to shuffle
        if shuffle_seed:
            ds = ds.shuffle(shuffle_seed)

        if limit is not None:
            ds = ds.select(range(limit))

        # Add prompt column
        def _add_prompts(batch):
            qs = batch["question"]
            return {"prompt": [cls.QUESTION_TEMPLATE.format(question=q) for q in qs]}

        ds = ds.map(
            _add_prompts,
            batched=True,
            batch_size=1000,
        )

        return ds


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


"""NOTE: VQA rules for processing text (see https://visualqa.org/evaluation.html)
    * Missing apostrophe addition: havent -> haven't etc."""


def process_text(text: str) -> str:
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


# Accuracy = min{n_matches / 3, 1} (taken from instructions)
def evaluate_prediction(out: str, answers: list[str]) -> dict[str, float]:
    out_norm = process_text(out)
    answers_norm = [process_text(answer) for answer in answers]
    n_matches = sum([out_norm.startswith(answer) for answer in answers_norm if answer])
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


def evaluate(
    model: MllamaForConditionalGeneration,
    processor: MllamaProcessor,
    data: datasets.Dataset,
    batch_size: int,
    max_new_tokens: int = 25,
    system_template: str = LLAMA_PROMPT_TEMPLATES["instruct"],
    disable_progress: bool = False,
) -> Iterable[dict]:
    n_batches = len(data) // batch_size

    for batch in tqdm(
        data.iter(batch_size),
        desc="Evaluating VQA task",
        total=n_batches,
        disable=disable_progress,
    ):
        imgs = [[img] for img in batch["image"]]
        prompts = [system_template.format(prompt=prompt) for prompt in batch["prompt"]]

        # Set to left-padding so we can call model.generate() when batched
        with set_padding_side_left(processor.tokenizer):
            inp = processor(imgs, prompts, return_tensors="pt", padding=True).to(
                model.device
            )

        with torch.no_grad():
            out = model.generate(
                **inp,
                max_new_tokens=max_new_tokens,
                pad_token_id=processor.tokenizer.eos_token_id,  # <eot_id>
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        # Trim the input tokens
        n_inp_toks = inp["input_ids"].shape[-1]
        gen_tok_ids = [x[n_inp_toks:] for x in out]

        for answers, id, out_ids in zip(
            batch["answers"], batch["question_id"], gen_tok_ids
        ):
            out = processor.decode(out_ids)
            answers = [x["answer"] for x in answers]

            yield dict(id=id, output=out, **evaluate_prediction(out, answers))
