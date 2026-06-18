# Copyright (c) 2026 Graphcore Ltd. All rights reserved.

"""Shared utilities for GPTQ baseline scripts."""

import json
from pathlib import Path
from typing import Any, Iterable

import datasets
import transformers


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
        tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_tokens,
        )
        for text in texts
    ]


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
