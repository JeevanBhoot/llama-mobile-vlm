# Copyright (c) 2026 Graphcore Ltd. All rights reserved.

"""Standard GPTQ baselines using GPTQModel and C4 calibration."""

import argparse
import json
import datasets
import transformers
from pathlib import Path
from typing import Any, Iterable
from gptqmodel import GPTQModel, QuantizeConfig

from eval import vqa

DEFAULT_MODEL_NAME = "meta-llama/Llama-3.2-11B-Vision-Instruct"
DEFAULT_C4_DATA_FILES = "en/c4-train.00001-of-01024.json.gz"
DEFAULT_C4_SPLIT = "train"
DEFAULT_CALIBRATION_SAMPLES = 1024
DEFAULT_CALIBRATION_MAX_TOKENS = 2048
DEFAULT_GROUP_SIZE = 128
DEFAULT_TORCH_DTYPE = "bfloat16"
DEFAULT_TASKS = ("vqa", "chartqa", "docvqa", "ai2d")
SUPPORTED_BITS = (3, 4)
METADATA_FILENAME = "metadata.json"


def default_output_dir(bits: int) -> Path:
    return Path(f"out/gptq/llama-3.2-vision-gptq-int{bits}-c4")


def load_c4_calibration(
    n_samples: int = DEFAULT_CALIBRATION_SAMPLES,
    data_files: str = DEFAULT_C4_DATA_FILES,
    split: str = DEFAULT_C4_SPLIT,
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
    include_relaxed_metrics: bool = False,
) -> dict[str, Any]:
    summary: dict[str, Any] = {"tasks": {}}
    primary_scores = []
    for task_name, results in task_results.items():
        if not results:
            raise ValueError(f"No results for task {task_name!r}")

        metrics = list(vqa.TASKS[task_name].METRICS)
        if include_relaxed_metrics:
            metrics += vqa.TASKS[task_name].RELAXED_METRICS

        task_summary = {"n_examples": len(results)}
        for metric in metrics:
            task_summary[metric] = sum(float(x[metric]) for x in results) / len(
                results
            )

        primary_scores.append(float(task_summary[metrics[0]]))
        summary["tasks"][task_name] = task_summary

    summary["avg_primary"] = sum(primary_scores) / len(primary_scores)
    return summary


def quantize(
    model_name: str,
    output_dir: Path,
    bits: int,
    group_size: int = DEFAULT_GROUP_SIZE,
    batch_size: int = 1,
    calibration_samples: int = DEFAULT_CALIBRATION_SAMPLES,
    calibration_max_tokens: int = DEFAULT_CALIBRATION_MAX_TOKENS,
    calibration_gpu_cache: bool = False,
    buffered_fwd: bool = False,
    c4_data_files: str = DEFAULT_C4_DATA_FILES,
    c4_split: str = DEFAULT_C4_SPLIT,
    device: str | None = None,
    torch_dtype: str = DEFAULT_TORCH_DTYPE,
) -> dict[str, Any]:
    calibration_texts = load_c4_calibration(
        n_samples=calibration_samples,
        data_files=c4_data_files,
        split=c4_split,
    )
    quant_config = QuantizeConfig(bits=bits, group_size=group_size, device=device)

    model = GPTQModel.load(model_name, quant_config, torch_dtype=torch_dtype)
    calibration_dataset = tokenise_calibration(
        calibration_texts,
        tokenizer=model.tokenizer,
        max_tokens=calibration_max_tokens,
    )
    model.quantize(
        calibration_dataset,
        batch_size=batch_size,
        calibration_enable_gpu_cache=calibration_gpu_cache,
        buffered_fwd=buffered_fwd,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(output_dir))

    metadata = {
        "model_name": model_name,
        "output_dir": str(output_dir),
        "bits": bits,
        "group_size": group_size,
        "quantization_batch_size": batch_size,
        "calibration_max_tokens": calibration_max_tokens,
        "calibration_gpu_cache": calibration_gpu_cache,
        "buffered_fwd": buffered_fwd,
        "torch_dtype": torch_dtype,
        "calibration": {
            "dataset": "allenai/c4",
            "data_files": c4_data_files,
            "split": c4_split,
            "n_samples": calibration_samples,
            "field": "text",
        },
        "device": device,
        "artifact_size_bytes": directory_size(output_dir),
    }
    write_json(output_dir / METADATA_FILENAME, metadata)
    return metadata


def evaluate(
    model_dir: Path,
    output_dir: Path,
    processor_name: str = DEFAULT_MODEL_NAME,
    tasks: Iterable[str] = DEFAULT_TASKS,
    n_examples: int = 1024,
    batch_size: int = 1,
    include_relaxed_metrics: bool = False,
    device: str | None = None,
) -> dict[str, Any]:
    qmodel = GPTQModel.load(str(model_dir))
    if device is not None:
        qmodel.to(device)
    model = qmodel.model
    model.eval()
    processor = transformers.AutoProcessor.from_pretrained(processor_name)
    artifact_size_bytes = directory_size(model_dir)

    task_results = {}
    for task_name in tasks:
        data = vqa.TASKS[task_name].data(limit=n_examples)
        results = list(
            vqa.evaluate(
                model=model,
                processor=processor,
                task_name=task_name,
                data=data,
                batch_size=batch_size,
                include_relaxed_metrics=include_relaxed_metrics,
            )
        )
        task_results[task_name] = results
        write_jsonl(output_dir / f"{task_name}.jsonl", results)

    summary = summarise_results(
        task_results, include_relaxed_metrics=include_relaxed_metrics
    )
    summary.update(
        {
            "model_dir": str(model_dir),
            "processor_name": processor_name,
            "n_examples_per_task": n_examples,
            "evaluation_batch_size": batch_size,
            "include_relaxed_metrics": include_relaxed_metrics,
            "artifact_size_bytes": artifact_size_bytes,
        }
    )
    write_json(output_dir / "summary.json", summary)
    return summary


def _add_quantize_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--bits",
        type=int,
        choices=SUPPORTED_BITS,
        required=True,
        help="GPTQ weight bit width",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Hugging Face model path or local checkpoint path",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for the saved GPTQModel artifact",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=DEFAULT_GROUP_SIZE,
        help="Group size for scaling factors in the quantization format",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Calibration batch size",
    )
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=DEFAULT_CALIBRATION_SAMPLES,
        help="Number of C4 calibration samples",
    )
    parser.add_argument(
        "--calibration-max-tokens",
        type=int,
        default=2048,
        help="Maximum tokens per calibration sample",
    )
    parser.add_argument(
        "--calibration-gpu-cache",
        action="store_true",
        help="Cache calibration activations on GPU",
    )
    parser.add_argument(
        "--buffered-fwd",
        action="store_true",
        help="Buffer forward inputs on CPU to reduce VRAM use",
    )
    parser.add_argument(
        "--c4-data-files",
        default=DEFAULT_C4_DATA_FILES,
        help="C4 data file passed to datasets.load_dataset",
    )
    parser.add_argument(
        "--c4-split",
        default=DEFAULT_C4_SPLIT,
        help="C4 split passed to datasets.load_dataset",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device used for GPTQ quantization",
    )
    parser.add_argument(
        "--torch-dtype",
        default=DEFAULT_TORCH_DTYPE,
        help="Torch dtype for loading the base model",
    )


def _add_evaluate_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "model_dir",
        type=Path,
        help="Directory containing a saved GPTQModel artifact",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for summary.json and per-task JSONL outputs",
    )
    _add_eval_settings_args(parser, batch_size_arg="--batch-size")
    parser.add_argument(
        "--device",
        default=None,
        help="Device used for evaluation",
    )


def _add_eval_settings_args(
    parser: argparse.ArgumentParser, batch_size_arg: str
) -> None:
    parser.add_argument(
        "--processor-name",
        default=DEFAULT_MODEL_NAME,
        help="Hugging Face processor path or local processor path",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=tuple(vqa.TASKS),
        default=list(DEFAULT_TASKS),
        help="Evaluation tasks to run",
    )
    parser.add_argument(
        "--n-examples",
        type=int,
        default=1024,
        help="Number of examples per evaluation task",
    )
    parser.add_argument(
        batch_size_arg,
        type=int,
        default=1,
        help="Evaluation batch size",
    )
    parser.add_argument(
        "--include-relaxed-metrics",
        action="store_true",
        help="Report relaxed task metrics where available",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run standard GPTQ INT4/INT3 C4 baselines with GPTQModel"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    quantize_parser = subparsers.add_parser("quantize")
    _add_quantize_args(quantize_parser)

    evaluate_parser = subparsers.add_parser("evaluate")
    _add_evaluate_args(evaluate_parser)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "quantize":
        output_dir = args.output_dir or default_output_dir(args.bits)
        quantize(
            model_name=args.model_name,
            output_dir=output_dir,
            bits=args.bits,
            group_size=args.group_size,
            batch_size=args.batch_size,
            calibration_samples=args.calibration_samples,
            calibration_max_tokens=args.calibration_max_tokens,
            calibration_gpu_cache=args.calibration_gpu_cache,
            buffered_fwd=args.buffered_fwd,
            c4_data_files=args.c4_data_files,
            c4_split=args.c4_split,
            device=args.device,
            torch_dtype=args.torch_dtype,
        )
    elif args.command == "evaluate":
        output_dir = args.output_dir or args.model_dir / "evaluation"
        evaluate(
            model_dir=args.model_dir,
            output_dir=output_dir,
            processor_name=args.processor_name,
            tasks=args.tasks,
            n_examples=args.n_examples,
            batch_size=args.batch_size,
            include_relaxed_metrics=args.include_relaxed_metrics,
            device=args.device,
        )
    else:
        raise ValueError(f"Unsupported command {args.command!r}")


if __name__ == "__main__":
    main()
