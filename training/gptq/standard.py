# Copyright (c) 2026 Graphcore Ltd. All rights reserved.

"""Standard GPTQ baselines using GPTQModel and C4 calibration."""

import argparse
import json
import subprocess
import datasets
import transformers
from pathlib import Path
from typing import Any, Iterable
from gptqmodel import GPTQModel, QuantizeConfig

from eval import vqa

DEFAULT_MODEL_NAME = "meta-llama/Llama-3.2-11B-Vision-Instruct"
DEFAULT_C4_DATA_FILES = "en/c4-train.00001-of-01024.json.gz"
DEFAULT_C4_SPLIT = "train"
DEFAULT_CALIBRATION_SAMPLES = 512
DEFAULT_CALIBRATION_MAX_TOKENS = 1024
DEFAULT_GROUP_SIZE = 128
DEFAULT_DTYPE = "bfloat16"
DEFAULT_TASKS = ("vqa", "chartqa", "docvqa", "ai2d")
SUPPORTED_BITS = (2, 3, 4, 5, 6, 8)
SUPPORTED_FORMATS = ("gptq", "gptq_v2", "marlin", "bitblas")
SUPPORTED_CALIBRATION_SORTS = ("asc", "desc", "shuffle")
SUPPORTED_GC_MODES = ("interval", "on_stage_end")
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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    rewrite = False
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                rewrite = True
                break
    if rewrite:
        write_jsonl(path, records)
    return records


def normalise_id(value: Any) -> str:
    return str(value)


def eval_id_column(data: datasets.Dataset) -> str:
    for column_name in ("id", "question_id", "questionId"):
        if column_name in data.column_names:
            return column_name
    raise ValueError(
        "Could not find an evaluation id column. Expected one of "
        "'id', 'question_id', or 'questionId'."
    )


def select_missing_eval_examples(
    data: datasets.Dataset, records: list[dict[str, Any]]
) -> datasets.Dataset:
    seen_ids = {normalise_id(record["id"]) for record in records}
    if not seen_ids:
        return data

    id_column = eval_id_column(data)
    missing_indices = [
        idx
        for idx, id in enumerate(data[id_column])
        if normalise_id(id) not in seen_ids
    ]
    return data.select(missing_indices)


def task_records_for_data(
    data: datasets.Dataset, records: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    records_by_id = {normalise_id(record["id"]): record for record in records}
    id_column = eval_id_column(data)
    out = []
    missing_ids = []
    for id in data[id_column]:
        key = normalise_id(id)
        if key in records_by_id:
            out.append(records_by_id[key])
        else:
            missing_ids.append(id)

    if missing_ids:
        raise ValueError(
            f"Missing {len(missing_ids)} cached evaluation records; "
            f"first missing id: {missing_ids[0]!r}"
        )
    return out


def append_eval_results(
    path: Path,
    results: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    written = []
    with path.open("a") as f:
        for record in results:
            print(json.dumps(record), file=f, flush=True)
            written.append(record)
    return written


def config_to_metadata(config: QuantizeConfig) -> dict[str, Any]:
    if hasattr(config, "to_dict"):
        return config.to_dict()
    return {
        key: serialise_metadata_value(value)
        for key, value in vars(config).items()
        if not key.startswith("_")
    }


def serialise_metadata_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): serialise_metadata_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialise_metadata_value(x) for x in value]
    if hasattr(value, "value"):
        return value.value
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def make_quantize_config(
    bits: int,
    group_size: int,
    format: str,
    device: str | None,
    pack_dtype: str | None,
    desc_act: bool | None,
    sym: bool,
    act_group_aware: bool | None,
    static_groups: bool,
    damp_percent: float | None,
    damp_auto_increment: float | None,
    mse: float,
    offload_to_disk: bool,
    offload_to_disk_path: str | None,
    calibration_data_device: str | None,
    gc_mode: str,
    wait_for_submodule_finalizers: bool,
) -> QuantizeConfig:
    kwargs: dict[str, Any] = {
        "bits": bits,
        "group_size": group_size,
        "format": format,
        "device": device,
        "pack_dtype": pack_dtype,
        "desc_act": desc_act,
        "sym": sym,
        "act_group_aware": act_group_aware,
        "static_groups": static_groups,
        "damp_percent": damp_percent,
        "damp_auto_increment": damp_auto_increment,
        "mse": mse,
        "offload_to_disk": offload_to_disk,
        "offload_to_disk_path": offload_to_disk_path,
        "calibration_data_device": calibration_data_device,
        "gc_mode": gc_mode,
        "wait_for_submodule_finalizers": wait_for_submodule_finalizers,
    }
    return QuantizeConfig(**{k: v for k, v in kwargs.items() if v is not None})


def make_quantize_call_kwargs(
    calibration_concat_size: int | None,
    calibration_sort: str,
    batch_size: int,
    backend: str,
    calibration_data_min_length: int,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "calibration_concat_size": calibration_concat_size,
        "calibration_sort": calibration_sort,
        "batch_size": batch_size,
        "backend": backend,
        "calibration_data_min_length": calibration_data_min_length,
    }
    return {k: v for k, v in kwargs.items() if v is not None}


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


def load_eval_data(
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
    return vqa.TASKS[task_name].data(**kwargs)


def quantize(
    model_name: str,
    output_dir: Path,
    calibration_samples: int = DEFAULT_CALIBRATION_SAMPLES,
    calibration_max_tokens: int = DEFAULT_CALIBRATION_MAX_TOKENS,
    c4_data_files: str = DEFAULT_C4_DATA_FILES,
    c4_split: str = DEFAULT_C4_SPLIT,
    dtype: str = DEFAULT_DTYPE,
    bits: int = 4,
    group_size: int = DEFAULT_GROUP_SIZE,
    format: str = "gptq",
    pack_dtype: str | None = None,
    desc_act: bool | None = None,
    sym: bool = True,
    act_group_aware: bool | None = None,
    static_groups: bool = False,
    damp_percent: float | None = None,
    damp_auto_increment: float | None = None,
    mse: float = 0.0,
    device: str | None = None,
    backend: str = "auto",
    offload_to_disk: bool = True,
    offload_to_disk_path: str | None = None,
    calibration_data_device: str | None = None,
    gc_mode: str = "interval",
    wait_for_submodule_finalizers: bool = False,
    calibration_concat_size: int | None = None,
    calibration_sort: str = "desc",
    batch_size: int = 1,
    calibration_data_min_length: int = 10,
) -> dict[str, Any]:
    calibration_texts = load_c4_calibration(
        n_samples=calibration_samples,
        data_files=c4_data_files,
        split=c4_split,
    )
    quant_config = make_quantize_config(
        bits=bits,
        group_size=group_size,
        format=format,
        device=device,
        pack_dtype=pack_dtype,
        desc_act=desc_act,
        sym=sym,
        act_group_aware=act_group_aware,
        static_groups=static_groups,
        damp_percent=damp_percent,
        damp_auto_increment=damp_auto_increment,
        mse=mse,
        offload_to_disk=offload_to_disk,
        offload_to_disk_path=offload_to_disk_path,
        calibration_data_device=calibration_data_device,
        gc_mode=gc_mode,
        wait_for_submodule_finalizers=wait_for_submodule_finalizers,
    )
    quantize_call_kwargs = make_quantize_call_kwargs(
        calibration_concat_size=calibration_concat_size,
        calibration_sort=calibration_sort,
        batch_size=batch_size,
        backend=backend,
        calibration_data_min_length=calibration_data_min_length,
    )

    model = GPTQModel.load(model_name, quant_config, dtype=dtype)
    calibration_dataset = tokenise_calibration(
        calibration_texts,
        tokenizer=model.tokenizer,
        max_tokens=calibration_max_tokens,
    )
    model.quantize(
        calibration_dataset,
        **quantize_call_kwargs,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(output_dir))

    metadata = {
        "model_name": model_name,
        "output_dir": str(output_dir),
        "quantization_batch_size": batch_size,
        "calibration_max_tokens": calibration_max_tokens,
        "dtype": dtype,
        "quantize_config": config_to_metadata(quant_config),
        "quantize_args": serialise_metadata_value(quantize_call_kwargs),
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
    load_vqa_from_s3: bool = False,
    vqa_s3_path: str | None = None,
    vqa_s3_local_path: Path | None = None,
    device: str | None = None,
    backend: str = "auto",
    dtype: str | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_size_bytes = directory_size(model_dir)

    task_states = []
    needs_evaluation = False
    for task_name in tasks:
        task_path = output_dir / f"{task_name}.jsonl"
        data = load_eval_data(
            task_name,
            n_examples=n_examples,
            load_vqa_from_s3=load_vqa_from_s3,
            vqa_s3_path=vqa_s3_path,
            vqa_s3_local_path=vqa_s3_local_path,
        )
        existing_results = load_jsonl(task_path) if task_path.exists() else []
        data_to_evaluate = select_missing_eval_examples(data, existing_results)
        needs_evaluation = needs_evaluation or bool(len(data_to_evaluate))
        task_states.append(
            (task_name, task_path, data, existing_results, data_to_evaluate)
        )

    model = None
    processor = None
    if needs_evaluation:
        load_kwargs: dict[str, Any] = {"backend": backend}
        if dtype is not None:
            load_kwargs["dtype"] = dtype
        qmodel = GPTQModel.load(str(model_dir), **load_kwargs)
        if device is not None:
            qmodel.to(device)
        model = qmodel.model
        model.eval()
        processor = transformers.AutoProcessor.from_pretrained(processor_name)

    task_results = {}
    for task_name, task_path, data, existing_results, data_to_evaluate in task_states:
        if len(data_to_evaluate):
            assert model is not None
            assert processor is not None
            new_results = append_eval_results(
                task_path,
                vqa.evaluate(
                    model=model,
                    processor=processor,
                    task_name=task_name,
                    data=data_to_evaluate,
                    batch_size=batch_size,
                    include_relaxed_metrics=include_relaxed_metrics,
                ),
            )
            existing_results.extend(new_results)

        task_results[task_name] = task_records_for_data(data, existing_results)

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
            "load_vqa_from_s3": load_vqa_from_s3,
            "vqa_s3_path": vqa_s3_path,
            "vqa_s3_local_path": str(vqa_s3_local_path)
            if vqa_s3_local_path
            else None,
            "backend": backend,
            "dtype": dtype,
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
        "--calibration-samples",
        type=int,
        default=DEFAULT_CALIBRATION_SAMPLES,
        help="Number of C4 calibration samples",
    )
    parser.add_argument(
        "--calibration-max-tokens",
        type=int,
        default=DEFAULT_CALIBRATION_MAX_TOKENS,
        help="Maximum tokens per calibration sample",
    )
    parser.add_argument(
        "--dtype",
        default=DEFAULT_DTYPE,
        help="dtype passed to GPTQModel.load",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=DEFAULT_GROUP_SIZE,
        help="Quantization group size for per-group scales",
    )
    parser.add_argument(
        "--format",
        choices=SUPPORTED_FORMATS,
        default="gptq",
        help="GPTQ checkpoint format",
    )
    parser.add_argument(
        "--pack-dtype",
        default=None,
        help="Packed weight dtype, for example int32, int16, or int8",
    )
    parser.add_argument(
        "--sym",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use symmetric quantization",
    )
    parser.add_argument(
        "--desc-act",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Activation-order quantization; GPTQModel default is used when omitted",
    )
    parser.add_argument(
        "--static-groups",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use GPTQ static groups",
    )
    parser.add_argument(
        "--act-group-aware",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use GPTQModel act_group_aware quality recovery",
    )
    parser.add_argument(
        "--damp-percent",
        type=float,
        default=None,
        help="GPTQ dampening percentage",
    )
    parser.add_argument(
        "--damp-auto-increment",
        type=float,
        default=None,
        help="GPTQ dampening auto-increment",
    )
    parser.add_argument(
        "--mse",
        type=float,
        default=0.0,
        help="GPTQ mse grid-search strength",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device used for GPTQ quantization",
    )
    parser.add_argument(
        "--backend",
        default="auto",
        help="GPTQModel quantization backend",
    )
    parser.add_argument(
        "--calibration-concat-size",
        type=int,
        default=None,
        help="Concatenate calibration tokens into fixed-size chunks",
    )
    parser.add_argument(
        "--calibration-sort",
        choices=SUPPORTED_CALIBRATION_SORTS,
        default="desc",
        help="Calibration sample ordering",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Calibration batch size",
    )
    parser.add_argument(
        "--calibration-data-min-length",
        type=int,
        default=10,
        help="Drop calibration samples shorter than this many tokens",
    )
    parser.add_argument(
        "--calibration-data-device",
        default=None,
        help="Device for captured calibration data, e.g. cpu, cuda:1, or balanced",
    )
    parser.add_argument(
        "--offload-to-disk",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Offload completed module state to disk during quantization",
    )
    parser.add_argument(
        "--offload-to-disk-path",
        default=None,
        help="Directory for GPTQModel disk offload",
    )
    parser.add_argument(
        "--gc-mode",
        choices=SUPPORTED_GC_MODES,
        default="interval",
        help="GPTQModel garbage-collection mode",
    )
    parser.add_argument(
        "--wait-for-submodule-finalizers",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Wait for packing/offload finalizers before moving to the next layer",
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
    parser.add_argument(
        "--backend",
        default="auto",
        help="GPTQModel backend used when loading the quantized artifact",
    )
    parser.add_argument(
        "--dtype",
        default=None,
        help="dtype passed to GPTQModel.load during evaluation",
    )
    parser.add_argument(
        "--load-vqa-from-s3",
        action="store_true",
        help="Load VQAv2 from the legacy S3 cache instead of Hugging Face",
    )
    parser.add_argument(
        "--vqa-s3-path",
        default=None,
        help="Temporary VQAv2 S3 dataset path to sync with --no-sign-request",
    )
    parser.add_argument(
        "--vqa-s3-local-path",
        type=Path,
        default=None,
        help="Optional local destination for --vqa-s3-path",
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
            calibration_samples=args.calibration_samples,
            calibration_max_tokens=args.calibration_max_tokens,
            c4_data_files=args.c4_data_files,
            c4_split=args.c4_split,
            dtype=args.dtype,
            bits=args.bits,
            group_size=args.group_size,
            format=args.format,
            pack_dtype=args.pack_dtype,
            desc_act=args.desc_act,
            sym=args.sym,
            static_groups=args.static_groups,
            act_group_aware=args.act_group_aware,
            damp_percent=args.damp_percent,
            damp_auto_increment=args.damp_auto_increment,
            mse=args.mse,
            device=args.device,
            backend=args.backend,
            calibration_concat_size=args.calibration_concat_size,
            calibration_sort=args.calibration_sort,
            batch_size=args.batch_size,
            calibration_data_min_length=args.calibration_data_min_length,
            calibration_data_device=args.calibration_data_device,
            offload_to_disk=args.offload_to_disk,
            offload_to_disk_path=args.offload_to_disk_path,
            gc_mode=args.gc_mode,
            wait_for_submodule_finalizers=args.wait_for_submodule_finalizers,
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
            load_vqa_from_s3=args.load_vqa_from_s3,
            vqa_s3_path=args.vqa_s3_path,
            vqa_s3_local_path=args.vqa_s3_local_path,
            device=args.device,
            backend=args.backend,
            dtype=args.dtype,
        )
    else:
        raise ValueError(f"Unsupported command {args.command!r}")


if __name__ == "__main__":
    main()
