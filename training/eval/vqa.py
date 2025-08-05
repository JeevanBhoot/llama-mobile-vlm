# Evaluate standard visual QA tasks (currently just VQAv2)

from itertools import islice
from typing import Iterable, Iterator, Optional

import datasets
import regex as re
import torch
import transformers

from utility import batches, set_padding_side_left


class VQA:
    @staticmethod
    def get_default_prompt_template() -> str:
        return (
            "<|image|> Look at the image carefully and answer this visual question. "
            "For yes/no questions, just respond Yes or No. "
            "If the answer is numeric, just respond with the number and nothing else. "
            "If the answer has multiple words, "
            "just respond with the words and absolutely nothing else. "
            "Never respond in a sentence or a phrase.\n "
            "Respond with as few words as possible.\n Question: {question}"
        )

    @classmethod
    def make_example(cls, row: dict) -> dict:
        return dict(
            **{
                k: row[k]
                for k in ["question_id", "image_id", "question", "image", "answers"]
            },
            prompt=cls.get_default_prompt_template().format(question=row["question"]),
        )

    @classmethod
    def get_examples(
        cls,
        split: str = "validation",
        shuffle_seed: int = 625464,
    ) -> Iterator[dict]:
        ds = datasets.load_dataset("lmms-lab/VQAv2", split=split).shuffle(shuffle_seed)
        for row in ds:
            yield cls.make_example(row)


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
def evaluate_prediction(out: str, answers: list[str]) -> float:
    out_norm = process_text(out)
    answers_norm = [process_text(answer) for answer in answers]
    n_matches = sum([out_norm.startswith(answer) for answer in answers_norm if answer])
    acc = min(n_matches / 3, 1.0)
    return acc


EMPTY_TEMPLATE = "{prompt}"
CHAT_TEMPLATE = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{prompt}"
    "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
)


# TODO: add progress
def evaluate(
    model: transformers.PreTrainedModel,
    processor: transformers.ProcessorMixin,
    examples: Iterable[dict],
    batch_size: int,
    n_examples: Optional[int] = None,
    max_new_tokens: int = 25,
    system_template: str = CHAT_TEMPLATE,
) -> Iterable[dict]:
    examples = islice(examples, n_examples)

    for batch in batches(examples, batch_size):
        # TODO: Move tokenisation + generation to a separate adapter?

        # Tokenise the batch
        imgs = [[x["image"]] for x in batch]  # NOTE: Each example gets a list of images
        prompts = [system_template.format(prompt=x["prompt"]) for x in batch]

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

        for example, out_ids in zip(batch, gen_tok_ids):
            out = processor.decode(out_ids)
            answers = [x["answer"] for x in example["answers"]]

            yield dict(
                id=example["question_id"],
                output=out,
                accuracy=evaluate_prediction(out, answers),
            )
