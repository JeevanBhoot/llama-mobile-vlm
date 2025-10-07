"""
Compare greedy output generation vs. the reference model.
Reports number of characters until divergence.
"""

import os
from typing import Any, Iterable

import torch
import tqdm
from transformers import MllamaForConditionalGeneration, MllamaProcessor

import train_data
from utility import batches, set_padding_side_left


def _get_out_wo_prompt(datum: train_data.Datum, processor: MllamaProcessor) -> str:
    ids = processor([datum.image], [datum.prompt])["input_ids"][0]
    prompt_post = processor.tokenizer.decode(ids)
    assert datum.out.find(prompt_post) == 0
    return datum.out[len(prompt_post) :]  # noqa: E203


def _check_dataset_configs(configs: dict[str, Any]) -> bool:
    """Check if all examples were generated greedily"""
    return all(not config["sampling_settings"]["do_sample"] for config in configs)


def evaluate(
    model: MllamaForConditionalGeneration,
    processor: MllamaProcessor,
    dataset: train_data.Dataset,
    char_limit: int,
    batch_size: int,
    max_new_tokens: int = 128,
    disable_progress: bool = False,
) -> Iterable[dict[str, Any]]:
    assert _check_dataset_configs(
        dataset._configs
    ), "Dataset must be generated greedily"

    n_examples = len(dataset)
    if n_examples % batch_size != 0:
        print(
            f"Warning: Number of examples {n_examples}"
            " is not divisible by {batch_size}. Dropping last batch."
        )
    n_batches = n_examples // batch_size

    for batch in tqdm.tqdm(
        batches(dataset.get_datums(), batch_size, drop_last=True),
        desc="Evaluating char distance",
        total=n_batches,
        disable=disable_progress,
    ):
        imgs = [[x.image] for x in batch]
        prompts = [x.prompt for x in batch]
        ref_texts = [_get_out_wo_prompt(x, processor)[:char_limit] for x in batch]

        with set_padding_side_left(processor.tokenizer):
            inp = processor(imgs, prompts, return_tensors="pt", padding=True).to(
                model.device
            )

        # NOTE: Could be different from the original number of tokens
        ref_tok_lens = [len(processor.tokenizer.encode(ref)) for ref in ref_texts]

        with torch.no_grad():
            out = model.generate(
                **inp,
                max_new_tokens=min(max(ref_tok_lens), max_new_tokens),
                do_sample=False,
                temperature=None,
                top_p=None,
            )
        # Trim the input tokens
        n_inp_toks = inp["input_ids"].shape[-1]
        gen_texts = [processor.tokenizer.decode(x[n_inp_toks:]) for x in out]

        for i, (ref_text, gen_text) in enumerate(zip(ref_texts, gen_texts)):
            yield dict(
                index=batch[i].index,
                match_length_char=len(os.path.commonprefix([ref_text, gen_text])),
                ref_char_length=len(ref_text),
            )
