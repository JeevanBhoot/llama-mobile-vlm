"""A basic evaluation harness for language model adaptation.

The idea is to compare outputs against a reference model.
"""

import itertools as it
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import datasets
import torch
import torchaudio
import tqdm
import transformers
from torch import Tensor

from training.utility import batches

DEFAULT_DTYPE = dict(cpu=torch.float32, cuda=torch.bfloat16)


@dataclass
class Datum:
    inp: Dict[str, Tensor]
    completion: Tensor
    entropy: Tensor


@dataclass
class EvalDataset:
    model: str
    dtype: torch.dtype
    dataset: str
    data: List[Datum]
    batch_size: int  # Keep track of batch_size due to bfloat16
    suppress_eos_token: bool

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(asdict(self), path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "EvalDataset":
        d = torch.load(path, weights_only=True)
        d["data"] = [Datum(**x) for x in d["data"]]
        return cls(**d)


def _dataset_row_to_input(
    row: Dict[str, Any], dataset_name: str, processor: transformers.ProcessorMixin
) -> transformers.BatchFeature:
    """Tokenise image + text pairs"""
    if dataset_name == "LMMs-Lab-Dev/vqav2_fewshot_val::validation":
        assert hasattr(
            processor, "image_processor"
        ), "Using multimodal dataset with a non-multimodal model"
        img = row["image"]
        # LLama requires manually adding image token, PaliGemma does not
        prompt = processor.image_token if hasattr(processor, "image_token") else ""
        prompt += row["question"]
        inp = processor(img, prompt, return_tensors="pt")
        return inp
    elif dataset_name == "wikitext:wikitext-103-raw-v1:validation":
        prompt = row["text"]
        inp = processor(prompt, return_tensors="pt")
        return inp
    else:
        raise ValueError(f"{dataset_name} is not supported")


def _get_inputs(
    dataset_name: str,
    processor: transformers.ProcessorMixin,
    prompt_length: int,
    seed: int = 283492,
) -> Iterable[Dict[str, Tensor]]:
    """Extract a list of processed multimodal inputs from a dataset.

    dataset_name -- path:name:split e.g. "wikitext:wikitext-103-raw-v1:validation"
    """
    path, name, split = dataset_name.split(":")
    dataset = datasets.load_dataset(path, name=name, split=split)
    for row in dataset.shuffle(seed=seed):
        inp = _dataset_row_to_input(row, dataset_name, processor)

        # Vision processors add a constant number of image tokens to prompt
        # (1 for Llama, 256 for PaliGemma) -> don't count these for prompt_length
        total_length = prompt_length
        if hasattr(processor, "image_token_id"):
            total_length += (inp.input_ids == processor.image_token_id).sum()
        if inp.input_ids.shape[1] > total_length:
            yield {
                k: (
                    v[:, :total_length]
                    if k in ["input_ids", "attention_mask", "cross_attention_mask"]
                    else v
                )
                for k, v in inp.items()
            }


def _batched_causal_cross_entropy(
    logits: Tensor, tokens: Tensor, ignore_index: int = -100
) -> Tensor:
    """Compute the X-Ent for each element of a batch of shape (..., sequence_length)."""
    next_tokens = tokens[..., 1:]
    # NOTE: Ignore image tokens
    return torch.nn.functional.cross_entropy(
        logits[..., :-1, :].flatten(end_dim=-2),
        next_tokens.flatten(),
        reduction="none",
        ignore_index=ignore_index,
    ).reshape(next_tokens.shape)


def _complete(
    model: transformers.PreTrainedModel,
    inp: Dict[str, Tensor],
    length: int,
    suppress_eos_token: bool,
) -> Tensor:
    pad_token_id = (
        model.config.text_config.eos_token_id
        if hasattr(model.config, "text_config")
        else model.config.eos_token_id
    )
    if isinstance(pad_token_id, list):
        pad_token_id = pad_token_id[0]
    completion: Tensor = model.generate(
        **inp,
        max_new_tokens=length,
        pad_token_id=pad_token_id,
        do_sample=False,
        temperature=None,
        top_p=None,
        suppress_tokens=[model.config.eos_token_id] if suppress_eos_token else None,
    )
    return completion


def _generate_datums(
    model: transformers.PreTrainedModel,
    inputs: Iterable[Dict[str, Tensor]],
    completion_length: int,
    suppress_eos_token: bool,
    batch_size: int,
) -> Iterable[Datum]:
    with torch.no_grad():
        for batch in batches(inputs, batch_size):
            inp = {
                k: torch.cat([x[k] for x in batch]).to(model.device)
                for k in batch[0].keys()
            }
            completions = _complete(model, inp, completion_length, suppress_eos_token)
            vision_kwargs = {}
            for k in [
                "pixel_values",
                "aspect_ratio_ids",
                "aspect_ratio_mask",
                "cross_attention_mask",
            ]:
                if k in inp:
                    if k == "cross_attention_mask":
                        mask, shape = inp[k], inp[k].shape
                        vision_kwargs[k] = torch.cat(
                            [
                                mask,
                                mask[:, -1:, ...].expand(
                                    shape[0], completion_length, *shape[2:]
                                ),
                            ],
                            dim=1,
                        )
                    else:
                        vision_kwargs[k] = inp[k]

            entropies = _batched_causal_cross_entropy(
                model(completions, **vision_kwargs).logits,
                completions,
                ignore_index=(
                    model.config.image_token_index
                    if hasattr(model.config, "image_token_index")
                    else -100
                ),
            )
            prefill_length = inp["input_ids"].shape[1]
            # NOTE: no batch dimension here
            for idx, (completion, entropy) in enumerate(
                zip(completions[:, prefill_length:], entropies[:, prefill_length - 1 :])
            ):
                yield Datum(
                    {k: v[idx].detach().cpu() for k, v in inp.items()},
                    completion.detach().cpu(),
                    entropy.detach().cpu(),
                )


def generate_dataset(
    model_name: str,
    prompt_length: int,
    completion_length: int,
    batch_size: int,
    suppress_eos_token: bool = False,
    dataset: str = "LMMs-Lab-Dev/vqav2_fewshot_val::validation",
    device: Optional[str] = None,
    dtype: Optional[str] = None,
    limit: Optional[int] = None,
    progress: bool = True,
) -> EvalDataset:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if dtype is None:
        dtype = DEFAULT_DTYPE[device]

    if "Llama" in model_name and "Vision" in model_name:
        model = transformers.MllamaForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=dtype
        )
    elif "paligemma" in model_name:
        model = transformers.PaliGemmaForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=dtype
        )
    else:
        print(
            f"Warning: Using AutoModelForCausalLM for {model_name}",
            file=sys.stderr,
        )
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True, torch_dtype=dtype
        )
    model.to(device)
    processor = transformers.AutoProcessor.from_pretrained(model_name)
    inputs = list(it.islice(_get_inputs(dataset, processor, prompt_length), limit))
    data = _generate_datums(
        model, inputs, completion_length, suppress_eos_token, batch_size
    )
    if progress:
        data = tqdm.tqdm(data, total=len(inputs), desc=model_name)
    return EvalDataset(
        model=model_name,
        dtype=dtype,
        dataset=dataset,
        data=list(data),
        batch_size=batch_size,
        suppress_eos_token=suppress_eos_token,
    )


METRICS = ["exact_match_length", "edit_distance_L16", "entropy_rmse"]


def _evaluate_rmse(
    model: transformers.PreTrainedModel, data: Iterable[Datum], batch_size: int
) -> Iterable[float]:
    for batch in tqdm.tqdm(
        batches(data, batch_size), desc="Evaluting cross-entropy MSE"
    ):
        tokens = torch.stack(
            [torch.cat([d.inp["input_ids"], d.completion]) for d in batch]
        ).to(model.device)
        vision_kwargs = {}
        for k in [
            "pixel_values",
            "aspect_ratio_ids",
            "aspect_ratio_mask",
            "cross_attention_mask",
        ]:
            if k in batch[0].inp:
                vision_kwargs[k] = torch.stack([d.inp[k] for d in batch]).to(
                    model.device
                )
                if k == "cross_attention_mask":
                    mask, shape = vision_kwargs[k], vision_kwargs[k].shape
                    vision_kwargs[k] = torch.cat(
                        [
                            mask,
                            mask[:, -1:, ...].expand(
                                shape[0], batch[0].completion.shape[0], *shape[2:]
                            ),
                        ],
                        dim=1,
                    )

        entropy = _batched_causal_cross_entropy(
            model(tokens, **vision_kwargs).logits,
            tokens,
            ignore_index=(
                model.config.image_token_index
                if hasattr(model.config, "image_token_index")
                else -100
            ),
        )
        target_entropy = torch.stack([d.entropy for d in batch]).to(model.device)
        mse = torch.nn.functional.mse_loss(
            target_entropy,
            entropy[..., -target_entropy.shape[-1] :],
            reduction="none",
        )
        yield from torch.sqrt(torch.mean(mse, dim=1)).tolist()


def _evaluate_exact_match(
    model: transformers.PreTrainedModel,
    data: Iterable[Datum],
    suppress_eos_token: bool,
    batch_size: int,
    edit_distance_length: int,
) -> Iterable[Tuple[int, int]]:
    for batch in tqdm.tqdm(batches(data, batch_size), desc="Evaluating exact match"):
        length = len(batch[0].completion)
        inp = {
            k: torch.stack([d.inp[k] for d in batch]).to(model.device)
            for k in batch[0].inp.keys()
        }
        completion = _complete(model, inp, length, suppress_eos_token)[:, -length:]
        target_completion = torch.stack([d.completion for d in batch]).to(model.device)
        for expected, actual in zip(target_completion, completion):
            yield (
                int(torch.cummin(expected[: len(actual)] == actual, 0).values.sum()),
                torchaudio.functional.edit_distance(
                    expected[:edit_distance_length], actual[:edit_distance_length]
                ),
            )


def _mean_and_stderr(samples: Tensor, name: str) -> Dict[str, float]:
    fsamples = samples.float()
    return {
        name: float(fsamples.mean()),
        f"{name}_stderr": float(fsamples.std() / fsamples.nelement() ** 0.5),
    }


def evaluate(
    model: transformers.PreTrainedModel,
    dataset: EvalDataset,
    batch_size: int,
    limit: Optional[int] = None,
    metrics: List[str] = METRICS,
) -> Dict[str, float]:
    """Evaluate a model adaptation against a reference (the original model)."""

    if any(m not in METRICS for m in metrics):
        raise ValueError(
            f"Unknown metrics {[m for m in metrics if m not in METRICS]}"
            f" (expected {METRICS})"
        )
    if model.config._name_or_path != dataset.model:
        print(
            f"Warning: evaluating the model {model.config._name_or_path!r} against a"
            f" different reference {dataset.model!r}",
            file=sys.stderr,
        )

    results = {}
    with torch.no_grad():
        if "entropy_rmse" in metrics:
            rmse = torch.tensor(
                list(_evaluate_rmse(model, dataset.data[:limit], batch_size))
            )
            results.update(_mean_and_stderr(rmse, "entropy_rmse"))
        if "exact_match_length" in metrics or "edit_distance_L16" in metrics:
            stats = torch.tensor(
                list(
                    _evaluate_exact_match(
                        model,
                        dataset.data[:limit],
                        dataset.suppress_eos_token,
                        batch_size,
                        edit_distance_length=16,
                    )
                )
            )
            if "exact_match_length" in metrics:
                results.update(_mean_and_stderr(stats[:, 0], "exact_match_length"))
            if "edit_distance_L16" in metrics:
                results.update(_mean_and_stderr(stats[:, 1], "edit_distance_L16"))
    return results
