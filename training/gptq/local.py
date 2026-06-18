# Copyright (c) 2026 Graphcore Ltd. All rights reserved.

"""Local dense GPTQ baseline for parity and S3D8 follow-up work."""

import argparse
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import transformers

from eval import vqa
from gptq.algorithm import GPTQConfig, GPTQLinearQuantizer
from gptq.common import (
    directory_size,
    load_c4_calibration,
    summarise_results,
    tokenise_calibration,
    write_json,
    write_jsonl,
)

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

TARGET_MODULE_GROUPS = (
    ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
    ("self_attn.o_proj",),
    ("mlp.gate_proj", "mlp.up_proj"),
    ("mlp.down_proj",),
)

DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


class _StopForward(Exception):
    pass


@dataclass
class LayerInputs:
    args: list[list[Any]]
    kwargs: list[dict[str, Any]]

    def __len__(self) -> int:
        return len(self.args)


def default_output_dir(bits: int) -> Path:
    return Path(f"out/gptq/llama-3.2-vision-local-gptq-int{bits}-c4")


def _move_to_device(x: Any, device: torch.device | str) -> Any:
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, dict):
        return {k: _move_to_device(v, device) for k, v in x.items()}
    if isinstance(x, list):
        return [_move_to_device(v, device) for v in x]
    if isinstance(x, tuple):
        return tuple(_move_to_device(v, device) for v in x)
    return x


def _detach_to_device(x: Any, device: torch.device | str) -> Any:
    if torch.is_tensor(x):
        return x.detach().to(device)
    if isinstance(x, dict):
        return {k: _detach_to_device(v, device) for k, v in x.items()}
    if isinstance(x, list):
        return [_detach_to_device(v, device) for v in x]
    if isinstance(x, tuple):
        return tuple(_detach_to_device(v, device) for v in x)
    return x


def mllama_text_layers(model: torch.nn.Module) -> torch.nn.ModuleList:
    return model.language_model.model.layers


def collect_first_layer_inputs(
    model: torch.nn.Module,
    calibration_dataset: Iterable[dict[str, Any]],
    device: str,
    calibration_gpu_cache: bool = False,
) -> LayerInputs:
    layers = mllama_text_layers(model)
    storage_device = device if calibration_gpu_cache else "cpu"
    cached_args = []
    cached_kwargs = []

    def store_input_hook(_, args, kwargs):
        cached_args.append([_detach_to_device(arg, storage_device) for arg in args])
        cached_kwargs.append(
            {k: _detach_to_device(v, storage_device) for k, v in kwargs.items()}
        )
        raise _StopForward

    handle = layers[0].register_forward_pre_hook(store_input_hook, with_kwargs=True)
    model.eval()
    try:
        with torch.inference_mode():
            for example in calibration_dataset:
                batch = {}
                for key, value in example.items():
                    if torch.is_tensor(value) and value.ndim == 1:
                        batch[key] = value.unsqueeze(0)
                    else:
                        batch[key] = value
                batch = _move_to_device(batch, device)
                try:
                    model(**batch)
                except _StopForward:
                    pass
    finally:
        handle.remove()

    return LayerInputs(cached_args, cached_kwargs)


def _run_layer(
    layer: torch.nn.Module,
    args: list[Any],
    kwargs: dict[str, Any],
    device: str,
) -> Any:
    args = [_move_to_device(arg, device) for arg in args]
    kwargs = {k: _move_to_device(v, device) for k, v in kwargs.items()}
    return layer(*args, **kwargs)


def _quantize_subset(
    layer: torch.nn.Module,
    subset: dict[str, torch.nn.Linear],
    inputs: LayerInputs,
    config: GPTQConfig,
    device: str,
) -> list[dict[str, Any]]:
    quantizers = {
        name: GPTQLinearQuantizer(module=module, config=config)
        for name, module in subset.items()
    }
    handles = []
    for name, module in subset.items():
        handles.append(
            module.register_forward_hook(
                lambda _, inp, __, name=name: quantizers[name].add_batch(inp[0].data)
            )
        )

    try:
        with torch.inference_mode():
            for args, kwargs in zip(inputs.args, inputs.kwargs):
                _run_layer(layer, args, kwargs, device)
    finally:
        for handle in handles:
            handle.remove()

    logs = []
    for name, module in subset.items():
        result = quantizers[name].quantize()
        module.weight.data = result.weight.to(module.weight.device)
        logs.append(
            {
                "module": name,
                "shape": list(module.weight.shape),
                "avg_loss": result.avg_loss,
                "duration": result.duration,
                "damp_percent": result.damp_percent,
                "scale_shape": list(result.scale.shape),
                "zero_shape": list(result.zero.shape),
                "g_idx_shape": list(result.g_idx.shape),
            }
        )

    return logs


def quantize_layer(
    layer: torch.nn.Module,
    inputs: LayerInputs,
    config: GPTQConfig,
    device: str,
) -> tuple[LayerInputs, list[dict[str, Any]]]:
    layer_modules = dict(layer.named_modules())
    logs = []

    if layer.__class__.__name__.lower() != "mllamacrossattentiondecoderlayer":
        for group in TARGET_MODULE_GROUPS:
            subset = {
                name: layer_modules[name]
                for name in group
                if name in layer_modules
                and isinstance(layer_modules[name], torch.nn.Linear)
            }
            if subset:
                logs.extend(_quantize_subset(layer, subset, inputs, config, device))

    next_args = []
    next_kwargs = []
    with torch.inference_mode():
        for args, kwargs in zip(inputs.args, inputs.kwargs):
            output = _run_layer(layer, args, kwargs, device)
            if isinstance(output, (tuple, list)):
                output = output[0]
            hidden_states = _detach_to_device(output, "cpu")
            next_args.append([hidden_states])
            next_kwargs.append(
                {k: _detach_to_device(v, "cpu") for k, v in kwargs.items()}
            )

    return LayerInputs(next_args, next_kwargs), logs


def load_model(
    model_name_or_path: str, dtype: torch.dtype, device: str
) -> torch.nn.Module:
    model = transformers.MllamaForConditionalGeneration.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
    )
    if device != "cpu":
        model.to(device)
    return model


def quantize(
    model_name: str,
    output_dir: Path,
    bits: int,
    group_size: int = DEFAULT_GROUP_SIZE,
    batch_size: int = 1,
    calibration_samples: int = DEFAULT_CALIBRATION_SAMPLES,
    calibration_max_tokens: int = DEFAULT_CALIBRATION_MAX_TOKENS,
    calibration_gpu_cache: bool = False,
    c4_data_files: str = DEFAULT_C4_DATA_FILES,
    c4_split: str = DEFAULT_C4_SPLIT,
    device: str = "cuda",
    torch_dtype: str = DEFAULT_TORCH_DTYPE,
    desc_act: bool = False,
    act_group_aware: bool = True,
    static_groups: bool = False,
    sym: bool = True,
    damp_percent: float = 0.05,
    damp_auto_increment: float = 0.01,
    blocksize: int = 128,
    mse: float = 0.0,
) -> dict[str, Any]:
    if bits not in SUPPORTED_BITS:
        raise ValueError(f"Unsupported bits={bits}, expected one of {SUPPORTED_BITS}")

    dtype = DTYPES[torch_dtype]
    model = load_model(model_name, dtype=dtype, device=device)
    processor = transformers.AutoProcessor.from_pretrained(model_name)
    tokenizer = processor.tokenizer

    calibration_texts = load_c4_calibration(
        n_samples=calibration_samples,
        data_files=c4_data_files,
        split=c4_split,
    )
    calibration_dataset = tokenise_calibration(
        calibration_texts,
        tokenizer=tokenizer,
        max_tokens=calibration_max_tokens,
    )

    config = GPTQConfig(
        bits=bits,
        group_size=group_size,
        blocksize=blocksize,
        damp_percent=damp_percent,
        damp_auto_increment=damp_auto_increment,
        desc_act=desc_act,
        act_group_aware=act_group_aware,
        static_groups=static_groups,
        sym=sym,
        mse=mse,
    )

    forward_pass_use_cache = getattr(model.config, "use_cache", False)
    model.config.use_cache = False
    try:
        layer_inputs = collect_first_layer_inputs(
            model,
            calibration_dataset,
            device=device,
            calibration_gpu_cache=calibration_gpu_cache,
        )

        layer_logs = []
        for layer_index, layer in enumerate(mllama_text_layers(model)):
            layer_inputs, logs = quantize_layer(layer, layer_inputs, config, device)
            for log in logs:
                log["layer"] = layer_index
                log["full_name"] = (
                    f"language_model.model.layers.{layer_index}.{log['module']}"
                )
            layer_logs.extend(logs)
    finally:
        model.config.use_cache = forward_pass_use_cache
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    processor.save_pretrained(output_dir)
    if hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(output_dir)

    metadata = {
        "model_name": model_name,
        "output_dir": str(output_dir),
        "artifact_type": "dense_dequantized_local_gptq",
        "bits": bits,
        "group_size": group_size,
        "quantization_batch_size": batch_size,
        "calibration_max_tokens": calibration_max_tokens,
        "calibration_gpu_cache": calibration_gpu_cache,
        "torch_dtype": torch_dtype,
        "gptq": dataclasses.asdict(config),
        "calibration": {
            "dataset": "allenai/c4",
            "data_files": c4_data_files,
            "split": c4_split,
            "n_samples": calibration_samples,
            "field": "text",
        },
        "device": device,
        "target_module_groups": TARGET_MODULE_GROUPS,
        "n_quantized_modules": len(layer_logs),
        "quantization_log": layer_logs,
        "artifact_size_bytes": directory_size(output_dir),
    }
    write_json(output_dir / METADATA_FILENAME, metadata)
    return metadata


def evaluate(
    model_dir: Path,
    output_dir: Path,
    processor_name: str | None = None,
    tasks: Iterable[str] = DEFAULT_TASKS,
    n_examples: int = 1024,
    batch_size: int = 1,
    include_relaxed_metrics: bool = False,
    device: str | None = None,
    torch_dtype: str = DEFAULT_TORCH_DTYPE,
) -> dict[str, Any]:
    dtype = DTYPES[torch_dtype]
    model = transformers.MllamaForConditionalGeneration.from_pretrained(
        model_dir,
        torch_dtype=dtype,
    )
    if device is not None:
        model.to(device)
    model.eval()

    processor_path = processor_name or str(model_dir)
    processor = transformers.AutoProcessor.from_pretrained(processor_path)
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
        task_results,
        task_definitions=vqa.TASKS,
        include_relaxed_metrics=include_relaxed_metrics,
    )
    summary.update(
        {
            "model_dir": str(model_dir),
            "processor_name": processor_path,
            "n_examples_per_task": n_examples,
            "evaluation_batch_size": batch_size,
            "include_relaxed_metrics": include_relaxed_metrics,
            "torch_dtype": torch_dtype,
            "artifact_size_bytes": artifact_size_bytes,
        }
    )
    write_json(output_dir / "summary.json", summary)
    return summary


def _add_gptq_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--bits", type=int, choices=SUPPORTED_BITS, required=True)
    parser.add_argument("--group-size", type=int, default=DEFAULT_GROUP_SIZE)
    parser.add_argument("--blocksize", type=int, default=128)
    parser.add_argument("--damp-percent", type=float, default=0.05)
    parser.add_argument("--damp-auto-increment", type=float, default=0.01)
    parser.add_argument(
        "--desc-act", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--act-group-aware", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--static-groups", action="store_true")
    parser.add_argument("--sym", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mse", type=float, default=0.0)


def _add_quantize_args(parser: argparse.ArgumentParser) -> None:
    _add_gptq_args(parser)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--calibration-samples", type=int, default=DEFAULT_CALIBRATION_SAMPLES
    )
    parser.add_argument(
        "--calibration-max-tokens", type=int, default=DEFAULT_CALIBRATION_MAX_TOKENS
    )
    parser.add_argument("--calibration-gpu-cache", action="store_true")
    parser.add_argument("--c4-data-files", default=DEFAULT_C4_DATA_FILES)
    parser.add_argument("--c4-split", default=DEFAULT_C4_SPLIT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--torch-dtype", choices=tuple(DTYPES), default=DEFAULT_TORCH_DTYPE
    )


def _add_evaluate_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--processor-name", default=None)
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--n-examples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--include-relaxed-metrics", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--torch-dtype",
        choices=tuple(DTYPES),
        default=DEFAULT_TORCH_DTYPE,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run local dense GPTQ INT4/INT3 C4 baselines"
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
            c4_data_files=args.c4_data_files,
            c4_split=args.c4_split,
            device=args.device,
            torch_dtype=args.torch_dtype,
            desc_act=args.desc_act,
            act_group_aware=args.act_group_aware,
            static_groups=args.static_groups,
            sym=args.sym,
            damp_percent=args.damp_percent,
            damp_auto_increment=args.damp_auto_increment,
            blocksize=args.blocksize,
            mse=args.mse,
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
            torch_dtype=args.torch_dtype,
        )
    else:
        raise ValueError(f"Unsupported command {args.command!r}")


if __name__ == "__main__":
    main()
