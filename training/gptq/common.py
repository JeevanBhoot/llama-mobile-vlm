# Copyright (c) 2026 Graphcore Ltd. All rights reserved.

"""Shared utilities for GPTQ baseline scripts."""

import json
import random
import subprocess
from pathlib import Path
from typing import Any, Iterable

import datasets
import torch
import transformers

DEFAULT_EVAL_DEVICE = "cuda"
DEFAULT_CALIBRATION_SAMPLES = 512
DEFAULT_CALIBRATION_MAX_TOKENS = 1024
DEFAULT_CALIBRATION_SORT = "desc"
DEFAULT_CALIBRATION_DATA_MIN_LENGTH = 10
SUPPORTED_CALIBRATION_SORTS = ("asc", "desc", "shuffle")


def load_c4_calibration(
    n_samples: int,
    data_files: str,
    split: str,
) -> list[str]:
    ds = datasets.load_dataset(
        "allenai/c4",
        data_files=data_files,
        split=split,
    )
    ds = ds.select(range(n_samples))
    return list(ds["text"])


def tokenise_calibration(
    texts: Iterable[str],
    tokenizer: transformers.PreTrainedTokenizerBase,
    max_tokens: int,
) -> list[dict[str, Any]]:
    return [
        dict(
            tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_tokens,
            )
        )
        for text in texts
    ]


def _calibration_length(example: dict[str, Any]) -> int:
    input_ids = example["input_ids"]
    if torch.is_tensor(input_ids):
        if input_ids.ndim == 0:
            return 1
        if input_ids.ndim == 1:
            return int(input_ids.shape[0])
        return int(input_ids.shape[1])

    if len(input_ids) == 0:
        return 0
    if isinstance(input_ids[0], (list, tuple)):
        return len(input_ids[0])
    return len(input_ids)


def prepare_tokenized_calibration(
    calibration_dataset: Iterable[dict[str, Any]],
    calibration_sort: str,
    calibration_data_min_length: int,
) -> list[dict[str, Any]]:
    prepared = [
        example
        for example in calibration_dataset
        if _calibration_length(example) > calibration_data_min_length
    ]

    if calibration_sort == "asc":
        return sorted(prepared, key=_calibration_length)
    if calibration_sort == "desc":
        return sorted(prepared, key=_calibration_length, reverse=True)
    if calibration_sort == "shuffle":
        shuffled = prepared[:]
        random.shuffle(shuffled)
        return shuffled
    raise ValueError(
        f"Unsupported calibration_sort={calibration_sort!r}; expected "
        "'asc', 'desc', or 'shuffle'"
    )


def _squeeze_tokenized_value(value: Any) -> Any:
    if torch.is_tensor(value) and value.ndim == 2 and value.shape[0] == 1:
        return value.squeeze(0)
    if (
        isinstance(value, list)
        and len(value) == 1
        and isinstance(value[0], (list, tuple))
    ):
        return value[0]
    return value


def _manual_pad_tokenized_rows(
    rows: list[dict[str, Any]],
    tokenizer: transformers.PreTrainedTokenizerBase,
) -> dict[str, torch.Tensor]:
    padding_side = getattr(tokenizer, "padding_side", "right")
    pad_token_id = getattr(tokenizer, "pad_token_id", 0)
    if pad_token_id is None:
        pad_token_id = 0

    batched = {}
    for key in rows[0]:
        tensors = []
        for row in rows:
            value = row[key]
            tensor = value if torch.is_tensor(value) else torch.tensor(value)
            if tensor.ndim == 0:
                tensor = tensor.unsqueeze(0)
            if tensor.ndim != 1:
                raise ValueError(
                    f"Cannot batch calibration field {key!r} with shape "
                    f"{tuple(tensor.shape)}"
                )
            tensors.append(tensor)

        max_len = max(int(tensor.shape[0]) for tensor in tensors)
        fill_value = 0 if key == "attention_mask" else pad_token_id
        padded = []
        for tensor in tensors:
            pad_len = max_len - int(tensor.shape[0])
            if pad_len:
                pad = torch.full(
                    (pad_len,),
                    fill_value,
                    dtype=tensor.dtype,
                    device=tensor.device,
                )
                tensor = (
                    torch.cat([pad, tensor])
                    if padding_side == "left"
                    else torch.cat([tensor, pad])
                )
            padded.append(tensor)
        batched[key] = torch.stack(padded)

    return batched


def batch_tokenized_calibration(
    calibration_dataset: list[dict[str, Any]],
    tokenizer: transformers.PreTrainedTokenizerBase,
    batch_size: int,
) -> list[dict[str, Any]]:
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    batches = []
    for start in range(0, len(calibration_dataset), batch_size):
        rows = [
            {key: _squeeze_tokenized_value(value) for key, value in example.items()}
            for example in calibration_dataset[start : start + batch_size]
        ]
        if not rows:
            continue

        pad = getattr(tokenizer, "pad", None)
        if callable(pad):
            try:
                batches.append(pad(rows, return_tensors="pt"))
                continue
            except (TypeError, ValueError):
                pass

        batches.append(_manual_pad_tokenized_rows(rows, tokenizer))

    return batches


def load_eval_data(
    task_definitions: dict[str, Any],
    task_name: str,
    n_examples: int,
    load_vqa_from_s3: bool = False,
    vqa_s3_path: str | None = None,
    vqa_s3_local_path: Path | None = None,
) -> datasets.Dataset:
    if task_name == "vqa" and vqa_s3_path is not None:
        local_path = vqa_s3_local_path
        if local_path is None:
            local_path = Path("data/datasets") / Path(vqa_s3_path.rstrip("/")).name
        subprocess.run(
            ["aws", "s3", "sync", "--no-sign-request", vqa_s3_path, str(local_path)],
            check=True,
        )
        ds = datasets.load_from_disk(str(local_path))
        ds = ds.select_columns(
            ["question_id", "image_id", "question", "image", "answers"]
        )
        ds = ds.shuffle(625464)
        return ds.select(range(n_examples))

    kwargs: dict[str, Any] = {"limit": n_examples}
    if task_name == "vqa":
        kwargs["load_from_s3"] = load_vqa_from_s3
    return task_definitions[task_name].data(**kwargs)


def resolve_eval_device(device: str | None) -> str:
    resolved = device or DEFAULT_EVAL_DEVICE
    if str(resolved).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"Requested evaluation device {resolved!r}, but CUDA is not available. "
            "Use --device cpu only for intentional CPU evaluation."
        )
    return resolved


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open() as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSONL in {path} at line {line_number}"
                    ) from error
    return records


def prepare_jsonl_output(
    path: Path,
    *,
    resume: bool,
    overwrite: bool,
) -> tuple[list[dict[str, Any]], str]:
    if resume and overwrite:
        raise ValueError("--resume and --overwrite cannot be used together")
    if path.exists():
        if resume:
            return read_jsonl(path), "a"
        if overwrite:
            return [], "w"
        raise FileExistsError(
            f"{path} already exists. Use --resume to append missing examples or "
            "--overwrite to rerun the task from scratch."
        )
    return [], "w"


def select_remaining_eval_data(data: Any, completed_count: int) -> Any:
    if completed_count <= 0:
        return data
    if completed_count > len(data):
        raise ValueError(
            f"Existing output has {completed_count} records, but the evaluation "
            f"dataset only has {len(data)} examples"
        )
    if hasattr(data, "select"):
        return data.select(range(completed_count, len(data)))
    return data[completed_count:]


def directory_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for record in records:
            print(json.dumps(record), file=f)


def write_jsonl_stream(
    path: Path,
    records: Iterable[dict[str, Any]],
    *,
    existing_records: Iterable[dict[str, Any]] = (),
    mode: str = "w",
) -> list[dict[str, Any]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    written = list(existing_records)
    with path.open(mode) as f:
        for record in records:
            print(json.dumps(record), file=f, flush=True)
            written.append(record)
    return written


def summarise_results(
    task_results: dict[str, list[dict[str, Any]]],
    task_definitions: dict[str, Any],
    include_relaxed_metrics: bool = False,
) -> dict[str, Any]:
    summary: dict[str, Any] = {"tasks": {}}
    primary_scores = []
    for task_name, results in task_results.items():
        if not results:
            raise ValueError(f"No results for task {task_name!r}")

        metrics = list(task_definitions[task_name].METRICS)
        if include_relaxed_metrics:
            metrics += task_definitions[task_name].RELAXED_METRICS

        task_summary = {"n_examples": len(results)}
        for metric in metrics:
            task_summary[metric] = sum(float(x[metric]) for x in results) / len(
                results
            )

        primary_scores.append(float(task_summary[metrics[0]]))
        summary["tasks"][task_name] = task_summary

    summary["avg_primary"] = sum(primary_scores) / len(primary_scores)
    return summary
