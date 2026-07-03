# Copyright (c) 2026 Graphcore Ltd. All rights reserved.

"""Local dense GPTQ baseline."""

import argparse
import dataclasses
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import safetensors.torch
import torch
import transformers
import weight_formats.quantisation as Q
import weight_formats.quantisation_training as QT

from eval import vqa
from gptq.algorithm import GPTQConfig, GPTQLinearQuantizer
from gptq.common import (
    DEFAULT_CALIBRATION_DATA_MIN_LENGTH,
    DEFAULT_CALIBRATION_MAX_TOKENS,
    DEFAULT_CALIBRATION_SAMPLES,
    DEFAULT_CALIBRATION_SORT,
    DEFAULT_EVAL_DEVICE,
    SUPPORTED_CALIBRATION_SORTS,
    batch_tokenized_calibration,
    directory_size,
    eval_records_for_data,
    load_c4_calibration,
    load_eval_data,
    prepare_jsonl_output,
    prepare_tokenized_calibration,
    resolve_eval_device,
    select_eval_data_to_run,
    summarise_results,
    tokenise_calibration,
    write_json,
    write_jsonl_stream,
)
from quant_formats import checkpoint_state

DEFAULT_MODEL_NAME = "meta-llama/Llama-3.2-11B-Vision-Instruct"
DEFAULT_C4_DATA_FILES = "en/c4-train.00001-of-01024.json.gz"
DEFAULT_C4_SPLIT = "train"
DEFAULT_GROUP_SIZE = 128
DEFAULT_TORCH_DTYPE = "bfloat16"
DEFAULT_TASKS = ("vqa", "chartqa", "docvqa", "ai2d")
SUPPORTED_BITS = (3, 4)
SUPPORTED_FORMATS = ("int", "int-codebook", "s3d8")
METADATA_FILENAME = "metadata.json"
DEFAULT_STORAGE_SCALE_ZERO_DTYPE = "bfloat16"
S3D8_CHECKPOINT_FILENAME = "gptq-s3d8.safetensors"

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
DTYPE_STORAGE_BYTES = {
    "bfloat16": 2,
    "float16": 2,
    "float32": 4,
}


@dataclass
class LayerInputs:
    args: list[list[Any]]
    kwargs: list[dict[str, Any]]

    def __len__(self) -> int:
        return len(self.args)


@dataclass(frozen=True)
class TextLayerStack:
    prefix: str
    language_model: torch.nn.Module
    layers: torch.nn.ModuleList
    cross_attention_layers: frozenset[int]


def default_output_dir(
    bits: int,
    quantization_format: str = "int",
    codepoints: int | None = None,
    group_size: int = DEFAULT_GROUP_SIZE,
) -> Path:
    if quantization_format == "s3d8":
        return Path("out/gptq/llama-3.2-vision-local-gptq-s3d8-c4")
    if quantization_format == "int-codebook":
        if codepoints is None:
            raise ValueError("int-codebook output directories require codepoints")
        return Path(
            "out/gptq/"
            f"llama-3.2-vision-local-gptq-int-k{codepoints}-g{group_size}-c4"
        )
    return Path(f"out/gptq/llama-3.2-vision-local-gptq-int{bits}-c4")


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {seconds:.0f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m"


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


def mllama_text_layers(model: torch.nn.Module) -> TextLayerStack:
    for prefix in ("model.language_model.layers", "language_model.model.layers"):
        module = model
        try:
            parts = prefix.split(".")
            for attr in parts[:-1]:
                module = getattr(module, attr)
            layers = getattr(module, parts[-1])
        except AttributeError:
            continue

        if isinstance(layers, torch.nn.ModuleList):
            return TextLayerStack(
                prefix=prefix,
                language_model=module,
                layers=layers,
                cross_attention_layers=frozenset(
                    getattr(module, "cross_attention_layers", ())
                ),
            )

    raise AttributeError(
        "Unable to resolve Mllama text decoder layers. Tried "
        "model.language_model.layers and language_model.model.layers."
    )


def _prepare_first_layer_attention_mask(attention_mask: Any) -> Any:
    if attention_mask is None or not torch.is_tensor(attention_mask):
        return attention_mask
    if (
        attention_mask.ndim <= 2
        and bool(attention_mask.to(dtype=torch.bool).all().item())
    ):
        return None
    return attention_mask


def _mllama_first_layer_kwargs(
    language_model: torch.nn.Module,
    batch: dict[str, Any],
    use_cache: bool,
) -> tuple[list[Any], dict[str, Any]]:
    input_ids = batch.get("input_ids")
    if input_ids is None:
        raise ValueError("Mllama local GPTQ calibration requires input_ids")

    attention_mask = batch.get("attention_mask")
    position_ids = batch.get("position_ids")
    past_key_values = batch.get("past_key_values")

    embedding_weight = getattr(language_model.embed_tokens, "weight", None)
    if torch.is_tensor(embedding_weight) and input_ids.device != embedding_weight.device:
        input_ids = input_ids.to(device=embedding_weight.device)

    inputs_embeds = language_model.embed_tokens(input_ids)
    if getattr(inputs_embeds, "is_meta", False):
        raise RuntimeError("Mllama input capture produced meta inputs_embeds")

    if position_ids is None:
        past_seen_tokens = (
            past_key_values.get_seq_length() if past_key_values is not None else 0
        )
        position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
        position_ids = position_ids + past_seen_tokens
        position_ids = position_ids.unsqueeze(0)
    elif position_ids.device != inputs_embeds.device:
        position_ids = position_ids.to(device=inputs_embeds.device)

    position_embeddings = language_model.rotary_emb(
        inputs_embeds,
        position_ids=position_ids,
    )

    kwargs = {
        "attention_mask": _prepare_first_layer_attention_mask(attention_mask),
        "position_ids": position_ids,
        "past_key_values": past_key_values,
        "use_cache": use_cache,
        "position_embeddings": position_embeddings,
    }
    return [inputs_embeds], kwargs


def collect_first_layer_inputs(
    text_layer_stack: TextLayerStack,
    calibration_dataset: Iterable[dict[str, Any]],
    device: str,
    calibration_gpu_cache: bool = False,
    use_cache: bool = False,
) -> LayerInputs:
    storage_device = device if calibration_gpu_cache else "cpu"
    cached_args = []
    cached_kwargs = []
    text_layer_stack.language_model.eval()
    with torch.inference_mode():
        for example in calibration_dataset:
            batch = {}
            for key, value in example.items():
                if torch.is_tensor(value) and value.ndim == 1:
                    batch[key] = value.unsqueeze(0)
                else:
                    batch[key] = value
            batch = _move_to_device(batch, device)
            args, kwargs = _mllama_first_layer_kwargs(
                text_layer_stack.language_model,
                batch,
                use_cache=use_cache,
            )
            cached_args.append([_detach_to_device(arg, storage_device) for arg in args])
            cached_kwargs.append(
                {k: _detach_to_device(v, storage_device) for k, v in kwargs.items()}
            )

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


def is_mllama_cross_attention_layer(
    layer: torch.nn.Module,
    layer_index: int | None = None,
    cross_attention_layers: frozenset[int] = frozenset(),
) -> bool:
    if layer_index is not None and layer_index in cross_attention_layers:
        return True
    if hasattr(layer, "cross_attn"):
        return True
    return layer.__class__.__name__.lower() == "mllamacrossattentiondecoderlayer"


def mllama_layer_kind(
    layer: torch.nn.Module,
    layer_index: int | None = None,
    cross_attention_layers: frozenset[int] = frozenset(),
) -> str:
    if is_mllama_cross_attention_layer(
        layer,
        layer_index=layer_index,
        cross_attention_layers=cross_attention_layers,
    ):
        return "cross_attention"
    if hasattr(layer, "self_attn"):
        return "self_attention"
    return "unknown"


def validate_target_layer_modules(
    layer: torch.nn.Module,
    layer_index: int | None = None,
) -> None:
    if not hasattr(layer, "self_attn"):
        label = f"layer {layer_index}" if layer_index is not None else "layer"
        raise ValueError(
            f"Unexpected Mllama text target scope for {label}: "
            f"{layer.__class__.__name__} is neither a self-attention decoder "
            "layer nor a recognized cross-attention decoder layer."
        )

    layer_modules = dict(layer.named_modules())
    missing = [
        name
        for group in TARGET_MODULE_GROUPS
        for name in group
        if name not in layer_modules
        or not isinstance(layer_modules[name], torch.nn.Linear)
    ]
    if missing:
        label = f"layer {layer_index}" if layer_index is not None else "layer"
        raise ValueError(
            f"Unexpected Mllama text target scope for {label}: missing linear "
            f"modules {missing}. Cross-attention layers should be listed in "
            "language_model.cross_attention_layers or use class "
            "MllamaCrossAttentionDecoderLayer."
        )


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
        module.weight.data = result.weight.to(module.weight.device).clone()
        log = {
            "module": name,
            "shape": list(module.weight.shape),
            "avg_loss": result.avg_loss,
            "duration": result.duration,
            "damp_percent": result.damp_percent,
            "quantization_format": result.quantization_format,
            "effective_bits": result.effective_bits or config.bits,
            "scale_shape": list(result.scale.shape),
        }
        if result.quantization_format == "int":
            log.update(
                {
                    "zero_shape": list(result.zero.shape),
                    "g_idx_shape": list(result.g_idx.shape),
                }
            )
        elif result.quantization_format == "int-codebook":
            log.update(
                {
                    "codepoints": config.codepoints,
                    "quantizer_mode": "scale_only_absmax_int_codebook",
                    "g_idx_shape": list(result.g_idx.shape),
                }
            )
        elif result.quantization_format == "s3d8":
            log.update(
                {
                    "centroid_shape": list(result.centroid_shape or ()),
                    "_fitted_format": result.fitted_format,
                    "_scale": result.scale.detach().cpu().clone(),
                }
            )
        else:
            raise ValueError(
                f"Unsupported quantization format {result.quantization_format!r}"
            )
        logs.append(log)

    return logs


def quantize_layer(
    layer: torch.nn.Module,
    inputs: LayerInputs,
    config: GPTQConfig,
    device: str,
    layer_index: int | None = None,
    cross_attention_layers: frozenset[int] = frozenset(),
    progress_name: str | None = None,
    verbose: bool = True,
) -> tuple[LayerInputs, list[dict[str, Any]]]:
    layer_kind = mllama_layer_kind(
        layer,
        layer_index=layer_index,
        cross_attention_layers=cross_attention_layers,
    )
    if layer_kind == "cross_attention":
        if verbose and progress_name is not None:
            print(f"{progress_name}: skipped cross-attention layer", flush=True)
        return replay_text_only_cross_attention_layer(inputs), []
    validate_target_layer_modules(layer, layer_index=layer_index)
    layer_modules = dict(layer.named_modules())
    logs = []

    for group in TARGET_MODULE_GROUPS:
        subset = {
            name: layer_modules[name]
            for name in group
            if name in layer_modules
            and isinstance(layer_modules[name], torch.nn.Linear)
        }
        if subset:
            if verbose and progress_name is not None:
                print(
                    f"{progress_name}: quantizing {', '.join(subset)}",
                    flush=True,
                )
            group_start = time.perf_counter()
            group_logs = _quantize_subset(layer, subset, inputs, config, device)
            if verbose and progress_name is not None:
                for log in group_logs:
                    print(
                        f"{progress_name}: {log['module']} "
                        f"loss={log['avg_loss']:.6g} "
                        f"damp={log['damp_percent']:.5g} "
                        f"time={format_duration(log['duration'])}",
                        flush=True,
                    )
                print(
                    f"{progress_name}: group done in "
                    f"{format_duration(time.perf_counter() - group_start)}",
                    flush=True,
                )
            logs.extend(group_logs)

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


def replay_text_only_cross_attention_layer(
    inputs: LayerInputs,
) -> LayerInputs:
    next_args = []
    next_kwargs = []
    for args, kwargs in zip(inputs.args, inputs.kwargs):
        next_args.append([_detach_to_device(args[0], "cpu")])
        next_kwargs.append(
            {k: _detach_to_device(v, "cpu") for k, v in kwargs.items()}
        )
    return LayerInputs(next_args, next_kwargs)


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


def estimate_packed_storage(
    model: torch.nn.Module,
    layer_logs: list[dict[str, Any]],
    bits: float,
    scale_zero_dtype: str,
    quantization_format: str = "int",
) -> dict[str, Any]:
    if scale_zero_dtype not in DTYPE_STORAGE_BYTES:
        raise ValueError(
            f"Unsupported storage scale/zero dtype {scale_zero_dtype!r}; "
            f"expected one of {tuple(DTYPE_STORAGE_BYTES)}"
        )

    scale_zero_bytes_per_value = DTYPE_STORAGE_BYTES[scale_zero_dtype]
    logs_by_weight_name = {
        f"{log['full_name']}.weight": log
        for log in layer_logs
    }
    unmatched_quantized_names = set(logs_by_weight_name)

    dense_state_dict_bytes = 0
    unquantized_dense_bytes = 0
    unquantized_tensor_count = 0
    unquantized_value_count = 0
    quantized_dense_bytes = 0
    quantized_value_count = 0
    packed_weight_bytes = 0
    scale_zero_bytes = 0
    centroid_bytes = 0
    g_idx_bytes = 0

    for name, tensor in model.state_dict().items():
        n_values = tensor.numel()
        dense_bytes = n_values * tensor.element_size()
        dense_state_dict_bytes += dense_bytes

        log = logs_by_weight_name.get(name)
        if log is None:
            unquantized_dense_bytes += dense_bytes
            unquantized_tensor_count += 1
            unquantized_value_count += n_values
            continue

        unmatched_quantized_names.discard(name)
        quantized_dense_bytes += dense_bytes
        quantized_value_count += n_values
        if quantization_format == "s3d8":
            if len(log["shape"]) != 2:
                raise ValueError(f"S3D8 storage estimate expects a 2D tensor: {log}")
            rows, cols = log["shape"]
            packed_rows = rows + (-rows % 3)
            packed_cols = cols + (-cols % 16)
            packed_weight_bytes += (packed_rows // 3) * packed_cols
            centroid_bytes += math.prod(log["centroid_shape"])
            scale_zero_values = math.prod(log["scale_shape"])
        elif quantization_format == "int-codebook":
            packed_weight_bytes += math.ceil(n_values * bits / 8)
            scale_zero_values = math.prod(log["scale_shape"])
            g_idx_bytes += math.prod(log["g_idx_shape"]) * 4
        else:
            packed_weight_bytes += math.ceil(n_values * bits / 8)
            scale_zero_values = math.prod(log["scale_shape"]) + math.prod(
                log["zero_shape"]
            )
            g_idx_bytes += math.prod(log["g_idx_shape"]) * 4
        scale_zero_bytes += scale_zero_values * scale_zero_bytes_per_value

    packed_without_g_idx = (
        unquantized_dense_bytes
        + packed_weight_bytes
        + scale_zero_bytes
        + centroid_bytes
    )
    packed_with_g_idx = packed_without_g_idx + g_idx_bytes

    total_values = unquantized_value_count + quantized_value_count
    quantized_bytes_without_g_idx = (
        packed_weight_bytes + scale_zero_bytes + centroid_bytes
    )
    quantized_bytes_with_g_idx = quantized_bytes_without_g_idx + g_idx_bytes

    return {
        "assumptions": {
            "scope": "state_dict tensors only; tokenizer/config files are excluded",
            "quantized_weight_bits": bits,
            "quantization_format": quantization_format,
            "scale_zero_dtype": scale_zero_dtype,
            "g_idx_dtype": "int32",
            "unquantized_tensors": "stored densely at their current dtype",
            "packed_weight_padding": (
                "S3D8 rows padded to multiple of 3 and columns to multiple of 16"
                if quantization_format == "s3d8"
                else (
                    "ideal fractional code packing rounded up to whole bytes per tensor"
                    if quantization_format == "int-codebook"
                    else "rounded up to whole bytes per tensor"
                )
            ),
        },
        "dense_state_dict_bytes": dense_state_dict_bytes,
        "estimated_packed_without_g_idx_bytes": packed_without_g_idx,
        "estimated_packed_with_g_idx_bytes": packed_with_g_idx,
        "overall_bits_per_value_without_g_idx": (
            packed_without_g_idx * 8 / total_values if total_values else 0.0
        ),
        "overall_bits_per_value_with_g_idx": (
            packed_with_g_idx * 8 / total_values if total_values else 0.0
        ),
        "quantized_effective_bits_per_value_without_g_idx": (
            quantized_bytes_without_g_idx * 8 / quantized_value_count
            if quantized_value_count
            else 0.0
        ),
        "quantized_effective_bits_per_value_with_g_idx": (
            quantized_bytes_with_g_idx * 8 / quantized_value_count
            if quantized_value_count
            else 0.0
        ),
        "total_state_dict_values": total_values,
        "quantized_values": quantized_value_count,
        "unquantized_values": unquantized_value_count,
        "quantized_dense_bytes": quantized_dense_bytes,
        "unquantized_dense_bytes": unquantized_dense_bytes,
        "packed_weight_bytes": packed_weight_bytes,
        "scale_zero_bytes": scale_zero_bytes,
        "centroid_bytes": centroid_bytes,
        "g_idx_bytes": g_idx_bytes,
        "quantized_tensor_count": len(layer_logs) - len(unmatched_quantized_names),
        "unquantized_tensor_count": unquantized_tensor_count,
        "unmatched_quantized_tensor_names": sorted(unmatched_quantized_names),
    }


def _serializable_quantization_logs(
    layer_logs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {key: value for key, value in log.items() if not key.startswith("_")}
        for log in layer_logs
    ]


def _resolve_module(root: torch.nn.Module, name: str) -> torch.nn.Module:
    module = root
    for part in name.split("."):
        module = getattr(module, part)
    return module


def save_s3d8_quantized_checkpoint(
    model: torch.nn.Module,
    output_path: Path,
    layer_logs: list[dict[str, Any]],
    dtype: torch.dtype,
    verbose: bool = True,
) -> None:
    fmt_spec: defaultdict[str, Q.TensorFormat] = defaultdict(lambda: Q.BFLOAT16)
    target_logs = []
    for log in layer_logs:
        if log.get("quantization_format") != "s3d8":
            continue
        fmt = log.get("_fitted_format")
        if fmt is None:
            raise ValueError(f"Missing fitted S3D8 format for {log['full_name']}")
        fmt_spec[f"{log['full_name']}.weight"] = fmt
        target_logs.append(log)

    if verbose:
        print(f"Saving S3D8 quantized checkpoint to {output_path}", flush=True)
    QT.convert(
        model,
        fmt_spec=fmt_spec,
        scaling_mode="parameter",
        clip_gradient=False,
        error_weight=None,
        activation_fmt=None,
        mode="qat",
        progress=verbose,
    )
    for log in target_logs:
        weight_wrapper = _resolve_module(model, log["full_name"]).weight
        if not isinstance(weight_wrapper, QT.Sign3D8Weight):
            raise TypeError(
                f"Expected {log['full_name']}.weight to be Sign3D8Weight, "
                f"got {type(weight_wrapper)}"
            )
        scale = log["_scale"].to(
            device=weight_wrapper.scale.device,
            dtype=weight_wrapper.scale.dtype,
        )
        weight_wrapper.scale.data.copy_(scale)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    safetensors.torch.save_file(checkpoint_state(model, dtype), output_path)


def quantize(
    model_name: str,
    output_dir: Path,
    bits: int,
    quantization_format: str = "int",
    codepoints: int | None = None,
    group_size: int = DEFAULT_GROUP_SIZE,
    batch_size: int = 1,
    calibration_samples: int = DEFAULT_CALIBRATION_SAMPLES,
    calibration_max_tokens: int = DEFAULT_CALIBRATION_MAX_TOKENS,
    calibration_sort: str = DEFAULT_CALIBRATION_SORT,
    calibration_data_min_length: int = DEFAULT_CALIBRATION_DATA_MIN_LENGTH,
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
    storage_scale_zero_dtype: str = DEFAULT_STORAGE_SCALE_ZERO_DTYPE,
    verbose: bool = True,
) -> dict[str, Any]:
    if quantization_format not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported format={quantization_format!r}, expected one of "
            f"{SUPPORTED_FORMATS}"
        )
    if quantization_format == "int" and bits not in SUPPORTED_BITS:
        raise ValueError(f"Unsupported bits={bits}, expected one of {SUPPORTED_BITS}")
    if quantization_format == "int-codebook":
        if codepoints is None:
            raise ValueError("format='int-codebook' requires codepoints")
        if codepoints < 2:
            raise ValueError(f"codepoints must be >= 2, got {codepoints}")
        element_range = Q.IntFormat(
            math.log2(codepoints),
            mode="asymmetric",
        ).range
        if element_range[1] <= 0:
            raise ValueError(
                "format='int-codebook' requires at least one positive codepoint "
                f"for absmax scaling, got codepoints={codepoints}"
            )
    elif codepoints is not None:
        raise ValueError("--codepoints is only supported with format='int-codebook'")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if calibration_sort not in SUPPORTED_CALIBRATION_SORTS:
        raise ValueError(
            f"Unsupported calibration_sort={calibration_sort!r}, expected one of "
            f"{SUPPORTED_CALIBRATION_SORTS}"
        )

    dtype = DTYPES[torch_dtype]
    quantize_start = time.perf_counter()
    if verbose:
        print(
            f"Loading model {model_name!r} with dtype={torch_dtype} on {device}",
            flush=True,
        )
    model = load_model(model_name, dtype=dtype, device=device)
    if verbose:
        print("Loading processor/tokenizer", flush=True)
    processor = transformers.AutoProcessor.from_pretrained(model_name)
    tokenizer = processor.tokenizer

    if verbose:
        print(
            "Loading C4 calibration "
            f"samples={calibration_samples} split={c4_split} "
            f"data_files={c4_data_files}",
            flush=True,
        )
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
    prepared_calibration_dataset = prepare_tokenized_calibration(
        calibration_dataset,
        calibration_sort=calibration_sort,
        calibration_data_min_length=calibration_data_min_length,
    )
    if not prepared_calibration_dataset:
        raise ValueError(
            "No calibration examples remain after min-length filtering. "
            "Lower --calibration-data-min-length or increase calibration data."
        )
    calibration_batches = batch_tokenized_calibration(
        prepared_calibration_dataset,
        tokenizer=tokenizer,
        batch_size=batch_size,
    )

    config = GPTQConfig(
        quantization_format=quantization_format,
        bits=bits,
        codepoints=codepoints,
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

    text_layer_stack = mllama_text_layers(model)
    text_config = getattr(text_layer_stack.language_model, "config", None)
    config_had_use_cache = hasattr(text_config, "use_cache")
    forward_pass_use_cache = (
        text_config.use_cache if config_had_use_cache else None
    )
    if config_had_use_cache:
        text_config.use_cache = False
    try:
        if verbose:
            print(
                f"Capturing first-layer inputs from {len(calibration_batches)} "
                f"calibration batches ({len(prepared_calibration_dataset)} examples)",
                flush=True,
            )
        layer_inputs = collect_first_layer_inputs(
            text_layer_stack,
            calibration_batches,
            device=device,
            calibration_gpu_cache=calibration_gpu_cache,
        )
        if verbose:
            print(
                f"Captured {len(layer_inputs)} first-layer input batches",
                flush=True,
            )

        layer_logs = []
        total_layers = len(text_layer_stack.layers)
        quantized_layer_count = 0
        skipped_cross_attention_layers = []
        if verbose:
            print(
                f"Quantizing {total_layers} text decoder layers "
                f"from {text_layer_stack.prefix}",
                flush=True,
            )
        for layer_index, layer in enumerate(text_layer_stack.layers):
            layer_start = time.perf_counter()
            progress_name = (
                f"[layer {layer_index + 1}/{total_layers} "
                f"{layer.__class__.__name__}]"
            )
            if verbose:
                print(f"{progress_name}: start", flush=True)
            is_cross_attention_layer = is_mllama_cross_attention_layer(
                layer,
                layer_index=layer_index,
                cross_attention_layers=text_layer_stack.cross_attention_layers,
            )
            layer_inputs, logs = quantize_layer(
                layer,
                layer_inputs,
                config,
                device,
                layer_index=layer_index,
                cross_attention_layers=text_layer_stack.cross_attention_layers,
                progress_name=progress_name,
                verbose=verbose,
            )
            if is_cross_attention_layer:
                skipped_cross_attention_layers.append(
                    {
                        "layer": layer_index,
                        "full_name": f"{text_layer_stack.prefix}.{layer_index}",
                        "class": layer.__class__.__name__,
                        "reason": "mllama_text_only_cross_attention",
                    }
                )
            for log in logs:
                log["layer"] = layer_index
                log["full_name"] = (
                    f"{text_layer_stack.prefix}.{layer_index}.{log['module']}"
                )
            layer_logs.extend(logs)
            if logs:
                quantized_layer_count += 1
            if verbose:
                completed = layer_index + 1
                elapsed = time.perf_counter() - quantize_start
                eta = elapsed / completed * (total_layers - completed)
                print(
                    f"{progress_name}: done modules={len(logs)} "
                    f"layer_time={format_duration(time.perf_counter() - layer_start)} "
                    f"elapsed={format_duration(elapsed)} "
                    f"eta={format_duration(eta)}",
                    flush=True,
                )
    finally:
        if config_had_use_cache:
            text_config.use_cache = forward_pass_use_cache
    if verbose:
        print(
            f"Quantized {len(layer_logs)} modules across "
            f"{quantized_layer_count} non-cross-attention layers",
            flush=True,
        )
        print(f"Saving dense dequantized model to {output_dir}", flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    processor.save_pretrained(output_dir)
    if hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(output_dir)

    if quantization_format == "int":
        effective_weight_bits = float(bits)
    elif quantization_format == "int-codebook":
        assert codepoints is not None
        effective_weight_bits = math.log2(codepoints)
    else:
        effective_weight_bits = 8 / 3
    packed_storage = estimate_packed_storage(
        model,
        layer_logs=layer_logs,
        bits=effective_weight_bits,
        scale_zero_dtype=storage_scale_zero_dtype,
        quantization_format=quantization_format,
    )
    if verbose:
        print(
            "Estimated packed storage: "
            f"{packed_storage['estimated_packed_with_g_idx_bytes'] / 1024**3:.3f} GiB",
            flush=True,
        )
    quantized_checkpoint_path = None
    if quantization_format == "s3d8":
        quantized_checkpoint_path = output_dir / S3D8_CHECKPOINT_FILENAME
        save_s3d8_quantized_checkpoint(
            model,
            quantized_checkpoint_path,
            layer_logs,
            dtype=dtype,
            verbose=verbose,
        )

    metadata = {
        "model_name": model_name,
        "output_dir": str(output_dir),
        "artifact_type": {
            "int": "dense_dequantized_local_gptq",
            "int-codebook": "dense_dequantized_local_gptq_int_codebook",
            "s3d8": "dense_dequantized_local_gptq_s3d8",
        }[quantization_format],
        "format": quantization_format,
        "bits": bits if quantization_format == "int" else None,
        "codepoints": codepoints if quantization_format == "int-codebook" else None,
        "effective_weight_bits": effective_weight_bits,
        "quantizer_mode": (
            "scale_only_absmax_int_codebook"
            if quantization_format == "int-codebook"
            else None
        ),
        "scale_dtype": storage_scale_zero_dtype,
        "quantized_checkpoint_path": (
            str(quantized_checkpoint_path) if quantized_checkpoint_path else None
        ),
        "group_size": group_size,
        "quantization_batch_size": batch_size,
        "calibration_max_tokens": calibration_max_tokens,
        "calibration_sort": calibration_sort,
        "calibration_data_min_length": calibration_data_min_length,
        "calibration_prepared_examples": len(prepared_calibration_dataset),
        "calibration_batches": len(calibration_batches),
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
        "target_scope": "mllama_text_self_attention_decoder_layers",
        "skipped_cross_attention_layers": skipped_cross_attention_layers,
        "target_module_groups": TARGET_MODULE_GROUPS,
        "n_quantized_modules": len(layer_logs),
        "quantization_log": _serializable_quantization_logs(layer_logs),
        "estimated_packed_storage": packed_storage,
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
    load_vqa_from_s3: bool = False,
    vqa_s3_path: str | None = None,
    vqa_s3_local_path: Path | None = None,
    device: str = DEFAULT_EVAL_DEVICE,
    torch_dtype: str = DEFAULT_TORCH_DTYPE,
    resume: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    device = resolve_eval_device(device)
    tasks = list(tasks)
    task_output_state = {
        task_name: prepare_jsonl_output(
            output_dir / f"{task_name}.jsonl",
            resume=resume,
            overwrite=overwrite,
        )
        for task_name in tasks
    }
    dtype = DTYPES[torch_dtype]
    model = transformers.MllamaForConditionalGeneration.from_pretrained(
        model_dir,
        torch_dtype=dtype,
    )
    model.to(device)
    model.eval()

    processor_path = processor_name or str(model_dir)
    processor = transformers.AutoProcessor.from_pretrained(processor_path)
    artifact_size_bytes = directory_size(model_dir)

    task_results = {}
    for task_name in tasks:
        output_path = output_dir / f"{task_name}.jsonl"
        existing_results, output_mode = task_output_state[task_name]
        data = load_eval_data(
            vqa.TASKS,
            task_name,
            n_examples=n_examples,
            load_vqa_from_s3=load_vqa_from_s3,
            vqa_s3_path=vqa_s3_path,
            vqa_s3_local_path=vqa_s3_local_path,
        )
        data_to_evaluate = select_eval_data_to_run(data, existing_results)
        results = write_jsonl_stream(
            output_path,
            vqa.evaluate(
                model=model,
                processor=processor,
                task_name=task_name,
                data=data_to_evaluate,
                batch_size=batch_size,
                include_relaxed_metrics=include_relaxed_metrics,
            ),
            existing_records=existing_results,
            mode=output_mode,
        )
        task_results[task_name] = eval_records_for_data(data, results)
        partial_summary = summarise_results(
            task_results,
            task_definitions=vqa.TASKS,
            include_relaxed_metrics=include_relaxed_metrics,
        )
        partial_summary.update(
            {
                "model_dir": str(model_dir),
                "processor_name": processor_path,
                "n_examples_per_task": n_examples,
                "evaluation_batch_size": batch_size,
                "include_relaxed_metrics": include_relaxed_metrics,
                "load_vqa_from_s3": load_vqa_from_s3,
                "vqa_s3_path": vqa_s3_path,
                "vqa_s3_local_path": str(vqa_s3_local_path)
                if vqa_s3_local_path
                else None,
                "device": device,
                "torch_dtype": torch_dtype,
                "artifact_size_bytes": artifact_size_bytes,
                "partial": True,
            }
        )
        write_json(output_dir / "summary.partial.json", partial_summary)

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
            "load_vqa_from_s3": load_vqa_from_s3,
            "vqa_s3_path": vqa_s3_path,
            "vqa_s3_local_path": str(vqa_s3_local_path)
            if vqa_s3_local_path
            else None,
            "torch_dtype": torch_dtype,
            "device": device,
            "artifact_size_bytes": artifact_size_bytes,
        }
    )
    write_json(output_dir / "summary.json", summary)
    return summary


def _add_gptq_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--format",
        choices=SUPPORTED_FORMATS,
        default="int",
        help="Weight format used inside local GPTQ",
    )
    parser.add_argument("--bits", type=int, choices=SUPPORTED_BITS, default=None)
    parser.add_argument(
        "--codepoints",
        type=int,
        default=None,
        help="Number of INT codepoints for --format int-codebook",
    )
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
    parser.add_argument("--batch-size", type=int, default=1, help="Calibration batch size")
    parser.add_argument(
        "--calibration-samples", type=int, default=DEFAULT_CALIBRATION_SAMPLES
    )
    parser.add_argument(
        "--calibration-max-tokens", type=int, default=DEFAULT_CALIBRATION_MAX_TOKENS
    )
    parser.add_argument(
        "--calibration-sort",
        choices=SUPPORTED_CALIBRATION_SORTS,
        default=DEFAULT_CALIBRATION_SORT,
        help="Calibration sample ordering before batching",
    )
    parser.add_argument(
        "--calibration-data-min-length",
        type=int,
        default=DEFAULT_CALIBRATION_DATA_MIN_LENGTH,
        help="Drop calibration samples shorter than this many tokens",
    )
    parser.add_argument("--calibration-gpu-cache", action="store_true")
    parser.add_argument("--c4-data-files", default=DEFAULT_C4_DATA_FILES)
    parser.add_argument("--c4-split", default=DEFAULT_C4_SPLIT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--torch-dtype", choices=tuple(DTYPES), default=DEFAULT_TORCH_DTYPE
    )
    parser.add_argument(
        "--storage-scale-zero-dtype",
        choices=tuple(DTYPE_STORAGE_BYTES),
        default=DEFAULT_STORAGE_SCALE_ZERO_DTYPE,
        help="Assumed dtype for packed GPTQ scale/zero storage estimates",
    )
    parser.add_argument("--quiet", action="store_true")


def _add_evaluate_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--processor-name", default=None)
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--n-examples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--include-relaxed-metrics", action="store_true")
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
    parser.add_argument(
        "--device",
        default=DEFAULT_EVAL_DEVICE,
        help="Evaluation device. Use --device cpu only for intentional CPU runs.",
    )
    parser.add_argument(
        "--torch-dtype",
        choices=tuple(DTYPES),
        default=DEFAULT_TORCH_DTYPE,
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run local dense GPTQ INT4/INT3/S3D8 C4 baselines"
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
        if args.format == "int-codebook":
            if args.bits is not None:
                parser.error("--bits cannot be used with --format int-codebook")
            if args.codepoints is None:
                parser.error("--codepoints is required with --format int-codebook")
            bits = 4
        else:
            if args.codepoints is not None:
                parser.error(
                    "--codepoints can only be used with --format int-codebook"
                )
            bits = args.bits if args.bits is not None else 4
        output_dir = args.output_dir or default_output_dir(
            bits,
            args.format,
            codepoints=args.codepoints,
            group_size=args.group_size,
        )
        metadata = quantize(
            model_name=args.model_name,
            output_dir=output_dir,
            bits=bits,
            quantization_format=args.format,
            codepoints=args.codepoints,
            group_size=args.group_size,
            batch_size=args.batch_size,
            calibration_samples=args.calibration_samples,
            calibration_max_tokens=args.calibration_max_tokens,
            calibration_sort=args.calibration_sort,
            calibration_data_min_length=args.calibration_data_min_length,
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
            storage_scale_zero_dtype=args.storage_scale_zero_dtype,
            verbose=not args.quiet,
        )
        storage = metadata["estimated_packed_storage"]
        gib = 1024**3
        print(
            "Estimated packed tensor storage: "
            f"{storage['estimated_packed_with_g_idx_bytes'] / gib:.3f} GiB, "
            f"{storage['overall_bits_per_value_with_g_idx']:.3f} bits/value overall"
        )
        print(
            "Dense local artifact size: "
            f"{metadata['artifact_size_bytes'] / gib:.3f} GiB"
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
            torch_dtype=args.torch_dtype,
            resume=args.resume,
            overwrite=args.overwrite,
        )
    else:
        raise ValueError(f"Unsupported command {args.command!r}")


if __name__ == "__main__":
    main()
