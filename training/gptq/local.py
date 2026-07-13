# Copyright (c) 2026 Graphcore Ltd. All rights reserved.

"""Local dense GPTQ baseline."""

import argparse
import dataclasses
import hashlib
import itertools
import json
import math
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import urlparse

import datasets
import safetensors.torch
import torch
import torch.nn.functional as F
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
DEFAULT_VQAV2_CALIBRATION_SPLIT = "train"
VQAV2_TRAIN_CALIBRATION_DATASET = "Multimodal-Fatima/VQAv2_sample_train"
DEFAULT_C4_DATA_FILES = "en/c4-train.00001-of-01024.json.gz"
DEFAULT_C4_SPLIT = "train"
DEFAULT_GROUP_SIZE = 128
DEFAULT_TORCH_DTYPE = "bfloat16"
DEFAULT_TASKS = ("vqa", "chartqa", "docvqa", "ai2d")
SUPPORTED_BITS = (3, 4)
SUPPORTED_FORMATS = ("int", "int-codebook", "s3d8")
SUPPORTED_TARGET_SCOPES = ("text-self", "full-multimodal")
SUPPORTED_CALIBRATION_SOURCES = ("c4", "vqav2", "synthetic", "eval-task")
METADATA_FILENAME = "metadata.json"
DEFAULT_STORAGE_SCALE_ZERO_DTYPE = "bfloat16"
S3D8_CHECKPOINT_FILENAME = "gptq-s3d8.safetensors"
VISION_ATTENTION_MASK_CACHE_SIZE = 8
DEFAULT_SYNTHETIC_IMAGE_SHARDS = 8
PUBLIC_SYNTHETIC_MANIFEST = "gptq-public-synthetic.json"
VQAV2_PROTOTYPE_WARNING = (
    "VQAv2 calibration overlaps an evaluation task and is intended only for "
    "full-multimodal GPTQ prototyping; use synthetic calibration for reportable "
    "results."
)

TEXT_SELF_TARGET_MODULE_GROUPS = (
    ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
    ("self_attn.o_proj",),
    ("mlp.gate_proj", "mlp.up_proj"),
    ("mlp.down_proj",),
)
TARGET_MODULE_GROUPS = TEXT_SELF_TARGET_MODULE_GROUPS
TEXT_CROSS_TARGET_MODULE_GROUPS = (
    ("cross_attn.q_proj", "cross_attn.k_proj", "cross_attn.v_proj"),
    ("cross_attn.o_proj",),
    ("mlp.gate_proj", "mlp.up_proj"),
    ("mlp.down_proj",),
)
TEXT_SELF_IGNORED_KWARGS = frozenset(
    {
        "cross_attention_states",
        "cross_attention_mask",
        "full_text_row_masked_out_mask",
    }
)
VISION_TARGET_MODULE_GROUPS = (
    ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
    ("self_attn.o_proj",),
    ("mlp.fc1",),
    ("mlp.fc2",),
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


def _iter_layer_inputs(
    inputs: LayerInputs | Iterable[tuple[list[Any], dict[str, Any]]],
) -> Iterable[tuple[list[Any], dict[str, Any]]]:
    if isinstance(inputs, LayerInputs):
        return zip(inputs.args, inputs.kwargs)
    return iter(inputs)


def _clear_layer_inputs(inputs: LayerInputs) -> None:
    inputs.args.clear()
    inputs.kwargs.clear()


@dataclass(frozen=True)
class TextLayerStack:
    prefix: str
    language_model: torch.nn.Module
    layers: torch.nn.ModuleList
    cross_attention_layers: frozenset[int]


@dataclass(frozen=True)
class VisionLayerStack:
    prefix: str
    vision_model: torch.nn.Module
    local_layers: torch.nn.ModuleList
    global_layers: torch.nn.ModuleList


@dataclass
class VisionBatchContext:
    batch_size: int
    num_concurrent_media: int
    num_tiles: int
    num_patches: int
    num_padding_patches: int
    dim: int
    aspect_ratio_ids: torch.Tensor
    attention_mask: "DeferredVisionAttentionMask"


@dataclass
class DeferredVisionAttentionMask:
    """Compact inputs for an Mllama vision mask that is expensive to retain."""

    aspect_ratio_mask: torch.Tensor
    num_patches: int
    target_length: int
    dtype: torch.dtype

    def to(self, device: torch.device | str) -> torch.Tensor:
        resolved_device = torch.device(device)
        if resolved_device.type == "cuda" and resolved_device.index is None:
            resolved_device = torch.device("cuda", torch.cuda.current_device())
        mask_values = tuple(
            self.aspect_ratio_mask.detach().reshape(-1).to("cpu").tolist()
        )
        key = (
            tuple(self.aspect_ratio_mask.shape),
            mask_values,
            self.num_patches,
            self.target_length,
            self.dtype,
            resolved_device,
        )
        cached = _VISION_ATTENTION_MASK_CACHE.get(key)
        if cached is not None:
            _VISION_ATTENTION_MASK_CACHE.move_to_end(key)
            return cached

        attention_mask = _prepare_aspect_ratio_attention_mask(
            aspect_ratio_mask=self.aspect_ratio_mask.to(resolved_device),
            num_patches=self.num_patches,
            target_length=self.target_length,
            dtype=self.dtype,
        )
        _VISION_ATTENTION_MASK_CACHE[key] = attention_mask
        while len(_VISION_ATTENTION_MASK_CACHE) > VISION_ATTENTION_MASK_CACHE_SIZE:
            _VISION_ATTENTION_MASK_CACHE.popitem(last=False)
        return attention_mask


_VISION_ATTENTION_MASK_CACHE: OrderedDict[tuple[Any, ...], torch.Tensor] = (
    OrderedDict()
)


def clear_vision_attention_mask_cache() -> None:
    _VISION_ATTENTION_MASK_CACHE.clear()


def default_output_dir(
    bits: int,
    quantization_format: str = "int",
    codepoints: int | None = None,
    group_size: int = DEFAULT_GROUP_SIZE,
    target_scope: str = "text-self",
    calibration_source: str | None = None,
) -> Path:
    source = calibration_source or (
        "vqav2" if target_scope == "full-multimodal" else "c4"
    )
    suffix = f"-{source}-full" if target_scope == "full-multimodal" else "-c4"
    if quantization_format == "s3d8":
        return Path(f"out/gptq/llama-3.2-vision-local-gptq-s3d8{suffix}")
    if quantization_format == "int-codebook":
        if codepoints is None:
            raise ValueError("int-codebook output directories require codepoints")
        return Path(
            "out/gptq/"
            f"llama-3.2-vision-local-gptq-int-k{codepoints}-g{group_size}{suffix}"
        )
    return Path(f"out/gptq/llama-3.2-vision-local-gptq-int{bits}{suffix}")


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {seconds:.0f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m"


def _move_to_device(x: Any, device: torch.device | str) -> Any:
    if isinstance(x, DeferredVisionAttentionMask):
        return x.to(device)
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
    if isinstance(x, DeferredVisionAttentionMask):
        return DeferredVisionAttentionMask(
            aspect_ratio_mask=x.aspect_ratio_mask.detach().to(device),
            num_patches=x.num_patches,
            target_length=x.target_length,
            dtype=x.dtype,
        )
    if torch.is_tensor(x):
        return x.detach().to(device)
    if isinstance(x, dict):
        return {k: _detach_to_device(v, device) for k, v in x.items()}
    if isinstance(x, list):
        return [_detach_to_device(v, device) for v in x]
    if isinstance(x, tuple):
        return tuple(_detach_to_device(v, device) for v in x)
    return x


def _module_dtype(module: torch.nn.Module, fallback: torch.dtype = torch.float32) -> torch.dtype:
    for tensor in module.parameters(recurse=True):
        return tensor.dtype
    for tensor in module.buffers(recurse=True):
        return tensor.dtype
    return fallback


def _module_device(module: torch.nn.Module, fallback: str = "cpu") -> torch.device | str:
    for tensor in module.parameters(recurse=True):
        return tensor.device
    for tensor in module.buffers(recurse=True):
        return tensor.device
    return fallback


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


def mllama_vision_layers(model: torch.nn.Module) -> VisionLayerStack:
    for prefix in ("model.vision_model", "vision_model"):
        module = model
        try:
            for attr in prefix.split("."):
                module = getattr(module, attr)
            local_layers = module.transformer.layers
            global_layers = module.global_transformer.layers
        except AttributeError:
            continue

        if isinstance(local_layers, torch.nn.ModuleList) and isinstance(
            global_layers,
            torch.nn.ModuleList,
        ):
            return VisionLayerStack(
                prefix=prefix,
                vision_model=module,
                local_layers=local_layers,
                global_layers=global_layers,
            )

    raise AttributeError(
        "Unable to resolve Mllama vision layers. Tried model.vision_model "
        "and vision_model."
    )


def mllama_multimodal_projector(model: torch.nn.Module) -> tuple[str, torch.nn.Linear]:
    for prefix in ("model.multi_modal_projector", "multi_modal_projector"):
        try:
            module = _resolve_module(model, prefix)
        except AttributeError:
            continue
        if isinstance(module, torch.nn.Linear):
            return prefix, module
    raise AttributeError("Unable to resolve Mllama multi_modal_projector Linear")


def mllama_lm_head(model: torch.nn.Module) -> tuple[str, torch.nn.Linear]:
    for prefix in ("lm_head", "language_model.lm_head"):
        try:
            module = _resolve_module(model, prefix)
        except AttributeError:
            continue
        if isinstance(module, torch.nn.Linear):
            return prefix, module
    raise AttributeError("Unable to resolve Mllama lm_head Linear")


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


def _dataset_batches(data: Any, batch_size: int) -> Iterable[dict[str, Any]]:
    data_iter = getattr(data, "iter", None)
    if callable(data_iter):
        yield from data_iter(batch_size)
        return

    rows = iter(data)
    while chunk := list(itertools.islice(rows, batch_size)):
        if isinstance(chunk[0], dict):
            yield {key: [row[key] for row in chunk] for key in chunk[0]}
        else:
            raise TypeError(
                "Unsupported calibration dataset rows; expected dict-like rows"
            )


def _processor_batch_to_plain_dict(batch: Any) -> dict[str, Any]:
    if isinstance(batch, dict):
        return dict(batch)
    data = getattr(batch, "data", None)
    if isinstance(data, dict):
        return dict(data)
    return dict(batch)


def _count_calibration_examples(batches: Iterable[dict[str, Any]]) -> int:
    total = 0
    for batch in batches:
        input_ids = batch.get("input_ids")
        if torch.is_tensor(input_ids) and input_ids.ndim > 0:
            total += int(input_ids.shape[0])
        elif input_ids is not None:
            total += len(input_ids)
        else:
            raise ValueError("Calibration batch is missing input_ids")
    return total


def load_vqav2_calibration_batches(
    processor: Any,
    *,
    n_samples: int,
    batch_size: int,
    split: str,
    max_tokens: int,
    load_from_s3: bool = False,
) -> list[dict[str, Any]]:
    if split == "train":
        data = datasets.load_dataset(
            VQAV2_TRAIN_CALIBRATION_DATASET,
            split="train",
        )
        data = data.select_columns(["question_id", "question", "image", "answers"])
        data = data.shuffle(625464).select(range(min(n_samples, len(data))))
    else:
        data = vqa.VQA.data(
            split=split,
            limit=n_samples,
            load_from_s3=load_from_s3,
        )
    system_template = vqa.LLAMA_PROMPT_TEMPLATES["instruct"]
    batches = []
    for batch in _dataset_batches(data, batch_size):
        images = batch["image"]
        prompts = [
            system_template.format(
                prompt=vqa.VQA.QUESTION_TEMPLATE.format(question=question)
            )
            for question in batch["question"]
        ]
        encoded = processor(
            [[image] for image in images],
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_tokens,
        )
        batches.append(_processor_batch_to_plain_dict(encoded))
    return batches


def load_eval_task_calibration_batches(
    processor: Any,
    *,
    task_name: str,
    n_samples: int,
    batch_size: int,
    split: str,
    max_tokens: int,
    load_vqa_from_s3: bool = False,
) -> list[dict[str, Any]]:
    task = vqa.TASKS[task_name]
    kwargs: dict[str, Any] = {
        "split": split,
        "limit": n_samples,
    }
    if task_name == "vqa":
        kwargs["load_from_s3"] = load_vqa_from_s3
    data = task.data(**kwargs)
    system_template = vqa.LLAMA_PROMPT_TEMPLATES["instruct"]
    batches = []
    for batch in _dataset_batches(data, batch_size):
        prepared = task.prepare_batch(batch, system_template=system_template)
        encoded = processor(
            [[image] for image in prepared.images],
            prepared.prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_tokens,
        )
        batches.append(_processor_batch_to_plain_dict(encoded))
    return batches


def load_synthetic_calibration_batches(
    processor: Any,
    *,
    calibration_data_paths: Iterable[str | Path],
    n_samples: int,
    batch_size: int,
    max_tokens: int,
    synthetic_image_shards: int = DEFAULT_SYNTHETIC_IMAGE_SHARDS,
) -> list[dict[str, Any]]:
    paths = [str(path) for path in calibration_data_paths]
    if not paths:
        raise ValueError(
            "--calibration-data-path is required with --calibration-source synthetic"
        )

    paths = [
        _stage_public_synthetic_data(
            path,
            synthetic_image_shards=synthetic_image_shards,
        )
        for path in paths
    ]

    from utility import LOCAL_DATA_PATH

    local_paths = [Path(LOCAL_DATA_PATH) / path for path in paths]
    public_manifests = [path / PUBLIC_SYNTHETIC_MANIFEST for path in local_paths]
    if any(path.is_file() for path in public_manifests):
        if not all(path.is_file() for path in public_manifests):
            raise ValueError(
                "Cannot mix public subset caches and full synthetic datasets in "
                "one calibration run"
            )
        return _load_public_synthetic_calibration_batches(
            processor,
            local_paths=local_paths,
            n_samples=n_samples,
            batch_size=batch_size,
            max_tokens=max_tokens,
        )

    from train_data import Dataset

    dataset = Dataset(paths, n_examples=[None] * len(paths))
    batches = []
    datums = itertools.islice(dataset.get_datums(), n_samples)
    while batch := list(itertools.islice(datums, batch_size)):
        encoded = processor(
            [[datum.image] for datum in batch],
            [datum.out.replace("<|begin_of_text|>", "") for datum in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_tokens,
        )
        batches.append(_processor_batch_to_plain_dict(encoded))
    return batches


def _load_public_synthetic_calibration_batches(
    processor: Any,
    *,
    local_paths: list[Path],
    n_samples: int,
    batch_size: int,
    max_tokens: int,
) -> list[dict[str, Any]]:
    calibration_datasets = []
    for source_id, local_path in enumerate(local_paths):
        manifest = json.loads((local_path / PUBLIC_SYNTHETIC_MANIFEST).read_text())
        image_datasets = [
            datasets.Dataset.from_file(filename)
            for filename in manifest["image_files"]
        ]
        image_data = datasets.concatenate_datasets(image_datasets)
        image_data = image_data.add_column(
            "_synthetic_source_id",
            [source_id] * len(image_data),
        )
        calibration_datasets.append(image_data)

    data = datasets.concatenate_datasets(calibration_datasets).shuffle(563673)
    if len(data) < n_samples:
        raise ValueError(
            "The staged synthetic image subset is smaller than the requested "
            f"calibration set: available={len(data)}, requested={n_samples}. "
            "Increase --synthetic-image-shards."
        )
    data = data.select(range(n_samples))

    required_indices: list[set[int]] = [set() for _ in local_paths]
    for source_id, index in zip(data["_synthetic_source_id"], data["index"]):
        required_indices[int(source_id)].add(int(index))

    outputs: list[dict[int, str]] = []
    for local_path, required in zip(local_paths, required_indices):
        selected_outputs: dict[int, str] = {}
        for filename in sorted((local_path / "out").glob("out*.jsonl")):
            with filename.open() as file:
                for line in file:
                    index, text = json.loads(line)
                    if index in required:
                        selected_outputs[index] = text
                        if len(selected_outputs) == len(required):
                            break
            if len(selected_outputs) == len(required):
                break
        missing = required - selected_outputs.keys()
        if missing:
            raise ValueError(
                f"Synthetic rollout outputs are missing {len(missing)} selected "
                f"image indices under {local_path}"
            )
        outputs.append(selected_outputs)

    batches = []
    for start in range(0, len(data), batch_size):
        rows = data[start : start + batch_size]
        texts = [
            outputs[int(source_id)][int(index)].replace("<|begin_of_text|>", "")
            for source_id, index in zip(
                rows["_synthetic_source_id"],
                rows["index"],
            )
        ]
        encoded = processor(
            [[image] for image in rows["image"]],
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_tokens,
        )
        batches.append(_processor_batch_to_plain_dict(encoded))
    return batches


def _parse_public_synthetic_s3_path(path: str) -> tuple[str, str, str]:
    parsed = urlparse(path)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"Expected an s3:// URI, got {path!r}")

    key_parts = parsed.path.strip("/").split("/")
    data_indices = [index for index, part in enumerate(key_parts) if part == "data"]
    if not data_indices:
        raise ValueError(
            "Synthetic S3 paths must be below a data/ prefix, for example "
            "s3://bucket/project/data/generation/model/dataset/run"
        )
    data_index = data_indices[-1]
    data_prefix = "/".join(key_parts[: data_index + 1])
    relative_path = "/".join(key_parts[data_index + 1 :])
    if not relative_path.startswith("generation/"):
        raise ValueError(
            "Synthetic calibration requires a generated rollout path below "
            "data/generation/, not a raw data/datasets/ path"
        )
    return parsed.netloc, data_prefix, relative_path


def _download_public_s3_prefix(
    bucket: str,
    prefix: str,
    destination: Path,
    *,
    include: Callable[[str], bool] | None = None,
) -> None:
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    client = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    prefix = prefix.rstrip("/") + "/"
    paginator = client.get_paginator("list_objects_v2")
    objects = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents", ()):
            relative = item["Key"][len(prefix) :]
            if relative and (include is None or include(relative)):
                objects.append((item["Key"], relative, int(item["Size"])))
    if not objects:
        raise FileNotFoundError(f"No public S3 objects found at s3://{bucket}/{prefix}")

    destination.mkdir(parents=True, exist_ok=True)
    for key, relative, size in objects:
        output_path = destination / relative
        if output_path.is_file() and output_path.stat().st_size == size:
            continue
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(
            f"Downloading s3://{bucket}/{key} "
            f"({size / 1024**3:.2f} GiB) to {output_path}",
            flush=True,
        )
        client.download_file(bucket, key, str(output_path))


def _stage_public_synthetic_data(
    path: str,
    *,
    synthetic_image_shards: int = DEFAULT_SYNTHETIC_IMAGE_SHARDS,
) -> str:
    if not path.startswith("s3://"):
        return path
    if synthetic_image_shards < 1:
        raise ValueError(
            f"synthetic_image_shards must be >= 1, got {synthetic_image_shards}"
        )

    from train_data import load_config
    from utility import LOCAL_DATA_PATH

    bucket, data_prefix, relative_path = _parse_public_synthetic_s3_path(path)
    local_path = Path(LOCAL_DATA_PATH) / relative_path
    _download_public_s3_prefix(
        bucket,
        f"{data_prefix}/{relative_path}",
        local_path,
        include=lambda name: name == "config.json" or name.startswith("out/"),
    )

    config = load_config(local_path)
    image_relative_path = f"datasets/{config.dataset_name}-{config.split}"
    cache_key = hashlib.sha256(
        f"{bucket}/{data_prefix}/{image_relative_path}".encode()
    ).hexdigest()[:12]
    image_local_path = (
        Path(LOCAL_DATA_PATH)
        / "gptq-public-cache"
        / cache_key
        / f"{config.dataset_name}-{config.split}"
    )
    image_prefix = f"{data_prefix}/{image_relative_path}"
    _download_public_s3_prefix(
        bucket,
        image_prefix,
        image_local_path,
        include=lambda name: name in {"dataset_info.json", "state.json"},
    )
    state = json.loads((image_local_path / "state.json").read_text())
    image_filenames = [item["filename"] for item in state["_data_files"]]
    shard_count = min(synthetic_image_shards, len(image_filenames))
    shard_indices = [
        min(
            len(image_filenames) - 1,
            ((2 * index + 1) * len(image_filenames)) // (2 * shard_count),
        )
        for index in range(shard_count)
    ]
    selected_filenames = {image_filenames[index] for index in shard_indices}
    _download_public_s3_prefix(
        bucket,
        image_prefix,
        image_local_path,
        include=selected_filenames.__contains__,
    )
    manifest = {
        "source": path,
        "image_shards": shard_count,
        "image_files": [
            str(image_local_path / image_filenames[index])
            for index in shard_indices
        ],
    }
    local_path.mkdir(parents=True, exist_ok=True)
    (local_path / PUBLIC_SYNTHETIC_MANIFEST).write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return relative_path


def _run_layer(
    layer: torch.nn.Module,
    args: list[Any],
    kwargs: dict[str, Any],
    device: str,
    ignored_kwargs: frozenset[str] = frozenset(),
) -> Any:
    args = [_move_to_device(arg, device) for arg in args]
    kwargs = {
        k: _move_to_device(v, device)
        for k, v in kwargs.items()
        if k not in ignored_kwargs
    }
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
    ignored_kwargs: frozenset[str] = frozenset(),
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
                _run_layer(
                    layer,
                    args,
                    kwargs,
                    device,
                    ignored_kwargs=ignored_kwargs,
                )
    finally:
        for handle in handles:
            handle.remove()

    logs = []
    for name, module in subset.items():
        result = quantizers.pop(name).quantize()
        module.weight.data.copy_(result.weight.to(module.weight.device))
        logs.append(_quantized_linear_log(name, module, result, config))

    return logs


def _quantized_linear_log(
    name: str,
    module: torch.nn.Linear,
    result: Any,
    config: GPTQConfig,
) -> dict[str, Any]:
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
    return log


def quantize_standalone_linear(
    full_name: str,
    module: torch.nn.Linear,
    inputs: LayerInputs | Iterable[tuple[list[Any], dict[str, Any]]],
    config: GPTQConfig,
    device: str,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    if verbose:
        print(f"[{full_name}]: quantizing standalone Linear", flush=True)
    quantizer = GPTQLinearQuantizer(module=module, config=config)
    with torch.inference_mode():
        for args, _ in _iter_layer_inputs(inputs):
            args = [_move_to_device(arg, device) for arg in args]
            quantizer.add_batch(args[0].data)
    result = quantizer.quantize()
    module.weight.data.copy_(result.weight.to(module.weight.device))
    log = _quantized_linear_log(full_name.rsplit(".", 1)[-1], module, result, config)
    log["full_name"] = full_name
    if verbose:
        print(
            f"[{full_name}]: loss={log['avg_loss']:.6g} "
            f"damp={log['damp_percent']:.5g} "
            f"time={format_duration(log['duration'])}",
            flush=True,
        )
    return [log]


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
    return quantize_layer_groups(
        layer,
        inputs,
        config,
        device,
        TARGET_MODULE_GROUPS,
        progress_name=progress_name,
        verbose=verbose,
    )


def validate_module_groups(
    layer: torch.nn.Module,
    module_groups: tuple[tuple[str, ...], ...],
    label: str,
) -> None:
    layer_modules = dict(layer.named_modules())
    missing = [
        name
        for group in module_groups
        for name in group
        if name not in layer_modules
        or not isinstance(layer_modules[name], torch.nn.Linear)
    ]
    if missing:
        raise ValueError(
            f"Unexpected GPTQ target scope for {label}: missing linear modules "
            f"{missing}."
        )


def quantize_layer_groups(
    layer: torch.nn.Module,
    inputs: LayerInputs,
    config: GPTQConfig,
    device: str,
    module_groups: tuple[tuple[str, ...], ...],
    progress_name: str | None = None,
    verbose: bool = True,
    ignored_kwargs: frozenset[str] = frozenset(),
) -> tuple[LayerInputs, list[dict[str, Any]]]:
    validate_module_groups(
        layer,
        module_groups,
        progress_name or layer.__class__.__name__,
    )
    layer_modules = dict(layer.named_modules())
    logs = []

    for group in module_groups:
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
            group_logs = _quantize_subset(
                layer,
                subset,
                inputs,
                config,
                device,
                ignored_kwargs=ignored_kwargs,
            )
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
            output = _run_layer(
                layer,
                args,
                kwargs,
                device,
                ignored_kwargs=ignored_kwargs,
            )
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


def _prepare_aspect_ratio_attention_mask(
    aspect_ratio_mask: torch.Tensor,
    num_patches: int,
    target_length: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    batch_size, max_num_tiles = aspect_ratio_mask.shape
    attention_mask = aspect_ratio_mask.view(batch_size, max_num_tiles, 1, 1).to(dtype)
    attention_mask = attention_mask.repeat(1, 1, target_length, 1)
    pad_patches = target_length - num_patches
    if pad_patches:
        attention_mask[:, :, -pad_patches:] = 0
    attention_mask = 1 - attention_mask
    attention_mask = attention_mask.reshape(batch_size, max_num_tiles * target_length, 1)
    attention_mask = (
        attention_mask
        @ attention_mask.transpose(-1, -2)
        * torch.finfo(dtype).min
    )
    return attention_mask.unsqueeze(1)


def _prepare_cross_attention_mask(
    cross_attention_mask: torch.Tensor,
    num_vision_tokens: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, text_total_length, *_ = cross_attention_mask.shape
    cross_attention_mask = cross_attention_mask.repeat_interleave(
        num_vision_tokens,
        dim=3,
    )
    cross_attention_mask = cross_attention_mask.view(batch_size, text_total_length, -1)
    cross_attention_mask = cross_attention_mask.unsqueeze(1)
    inverted = (1.0 - cross_attention_mask).to(dtype)
    negative_inf = torch.finfo(dtype).min
    prepared = inverted.masked_fill(inverted.to(torch.bool), negative_inf)
    full_text_row_masked_out_mask = (
        (prepared != negative_inf).any(dim=-1).type_as(prepared)[..., None]
    )
    prepared *= full_text_row_masked_out_mask
    return prepared, full_text_row_masked_out_mask


def _prepare_causal_mask(
    language_model: torch.nn.Module,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor | None,
    past_key_values: Any,
    position_ids: torch.Tensor,
) -> Any:
    try:
        from transformers.masking_utils import create_causal_mask
    except ImportError:
        return _prepare_first_layer_attention_mask(attention_mask)

    config = getattr(language_model, "config", None)
    if config is None:
        return _prepare_first_layer_attention_mask(attention_mask)
    try:
        return create_causal_mask(
            config=config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )
    except Exception:
        return _prepare_first_layer_attention_mask(attention_mask)


def collect_vision_transformer_inputs(
    vision_stack: VisionLayerStack,
    calibration_batches: Iterable[dict[str, Any]],
    device: str,
    calibration_gpu_cache: bool = False,
) -> tuple[LayerInputs, list[VisionBatchContext]]:
    clear_vision_attention_mask_cache()
    storage_device = device if calibration_gpu_cache else "cpu"
    cached_args = []
    cached_kwargs = []
    contexts = []
    vision_model = vision_stack.vision_model
    vision_model.eval()
    with torch.inference_mode():
        for batch in calibration_batches:
            pixel_values = _move_to_device(batch["pixel_values"], device)
            aspect_ratio_ids = _move_to_device(batch["aspect_ratio_ids"], device)
            aspect_ratio_mask = _move_to_device(batch["aspect_ratio_mask"], device)
            (
                batch_size,
                num_concurrent_media,
                num_tiles,
                num_channels,
                height,
                width,
            ) = pixel_values.shape
            pixel_values = pixel_values.reshape(
                batch_size * num_concurrent_media * num_tiles,
                num_channels,
                height,
                width,
            )
            aspect_ratio_ids = aspect_ratio_ids.reshape(
                batch_size * num_concurrent_media,
                -1,
            )

            patch_weight = vision_model.patch_embedding.weight
            patch_embeds = vision_model.patch_embedding(
                pixel_values.to(patch_weight.device, patch_weight.dtype)
            )
            hidden_state = patch_embeds.flatten(2).transpose(1, 2)
            _, num_patches, dim = hidden_state.shape
            hidden_state = hidden_state.reshape(
                batch_size * num_concurrent_media,
                num_tiles,
                -1,
                dim,
            )
            hidden_state = vision_model.pre_tile_positional_embedding(
                hidden_state,
                aspect_ratio_ids,
            )
            hidden_state = hidden_state.reshape(
                batch_size * num_concurrent_media * num_tiles,
                num_patches,
                dim,
            )
            hidden_state = vision_model.apply_class_embedding(hidden_state)
            num_patches += 1
            hidden_state = hidden_state.reshape(
                batch_size * num_concurrent_media,
                num_tiles,
                num_patches,
                dim,
            )
            hidden_state = vision_model.gated_positional_embedding(
                hidden_state,
                aspect_ratio_ids,
            )
            hidden_state = vision_model.layernorm_pre(hidden_state)

            num_padding_patches = (8 - (hidden_state.shape[-2] % 8)) % 8
            hidden_state = F.pad(
                hidden_state,
                (0, 0, 0, num_padding_patches),
                mode="constant",
                value=0,
            )
            compact_attention_mask = aspect_ratio_mask.reshape(
                batch_size * num_concurrent_media,
                -1,
            )
            attention_mask = DeferredVisionAttentionMask(
                aspect_ratio_mask=_detach_to_device(
                    compact_attention_mask,
                    storage_device,
                ),
                num_patches=getattr(vision_model, "num_patches", num_patches),
                target_length=hidden_state.shape[2],
                dtype=_module_dtype(vision_model, hidden_state.dtype),
            )
            hidden_state = hidden_state.view(
                batch_size * num_concurrent_media,
                -1,
                dim,
            )

            cached_args.append([_detach_to_device(hidden_state, storage_device)])
            cached_kwargs.append({"attention_mask": attention_mask})
            contexts.append(
                VisionBatchContext(
                    batch_size=batch_size,
                    num_concurrent_media=num_concurrent_media,
                    num_tiles=num_tiles,
                    num_patches=num_patches,
                    num_padding_patches=num_padding_patches,
                    dim=dim,
                    aspect_ratio_ids=_detach_to_device(
                        aspect_ratio_ids,
                        storage_device,
                    ),
                    attention_mask=attention_mask,
                )
            )
    return LayerInputs(cached_args, cached_kwargs), contexts


def quantize_layer_sequence(
    layers: torch.nn.ModuleList,
    prefix: str,
    inputs: LayerInputs,
    config: GPTQConfig,
    device: str,
    module_groups: tuple[tuple[str, ...], ...],
    verbose: bool = True,
    capture_after_layers: frozenset[int] = frozenset(),
) -> tuple[LayerInputs, list[dict[str, Any]], dict[int, LayerInputs]]:
    logs = []
    captured: dict[int, LayerInputs] = {}
    total_layers = len(layers)
    for layer_index, layer in enumerate(layers):
        layer_start = time.perf_counter()
        progress_name = (
            f"[{prefix}.{layer_index} {layer_index + 1}/{total_layers} "
            f"{layer.__class__.__name__}]"
        )
        if verbose:
            print(f"{progress_name}: start", flush=True)
        previous_inputs = inputs
        inputs, layer_logs = quantize_layer_groups(
            layer,
            inputs,
            config,
            device,
            module_groups,
            progress_name=progress_name,
            verbose=verbose,
        )
        if layer_index == 0 or layer_index - 1 not in capture_after_layers:
            _clear_layer_inputs(previous_inputs)
        for log in layer_logs:
            log["layer"] = layer_index
            log["full_name"] = f"{prefix}.{layer_index}.{log['module']}"
        logs.extend(layer_logs)
        if layer_index in capture_after_layers:
            captured[layer_index] = inputs
        if verbose:
            print(
                f"{progress_name}: done modules={len(layer_logs)} "
                f"layer_time={format_duration(time.perf_counter() - layer_start)}",
                flush=True,
            )
    return inputs, logs, captured


def build_global_vision_inputs(
    vision_model: torch.nn.Module,
    local_outputs: LayerInputs,
    contexts: list[VisionBatchContext],
    device: str,
    calibration_gpu_cache: bool = False,
) -> LayerInputs:
    storage_device = device if calibration_gpu_cache else "cpu"
    next_args = []
    next_kwargs = []
    with torch.inference_mode():
        for args, context in zip(local_outputs.args, contexts):
            hidden_state = _move_to_device(args[0], device)
            aspect_ratio_ids = _move_to_device(context.aspect_ratio_ids, device)
            hidden_state = vision_model.layernorm_post(hidden_state)
            hidden_state = hidden_state.reshape(
                context.batch_size * context.num_concurrent_media,
                context.num_tiles,
                context.num_patches + context.num_padding_patches,
                context.dim,
            )
            hidden_state = vision_model.post_tile_positional_embedding(
                hidden_state,
                aspect_ratio_ids,
            )
            hidden_state = hidden_state.reshape(
                context.batch_size * context.num_concurrent_media,
                context.num_tiles * (context.num_patches + context.num_padding_patches),
                context.dim,
            )
            next_args.append([_detach_to_device(hidden_state, storage_device)])
            next_kwargs.append(
                {
                    "attention_mask": _detach_to_device(
                        context.attention_mask,
                        storage_device,
                    )
                }
            )
    return LayerInputs(next_args, next_kwargs)


def _remove_vision_padding(
    hidden_state: torch.Tensor,
    context: VisionBatchContext,
) -> torch.Tensor:
    hidden_state = hidden_state.reshape(
        context.batch_size * context.num_concurrent_media,
        context.num_tiles,
        context.num_patches + context.num_padding_patches,
        -1,
    )
    if context.num_padding_patches:
        hidden_state = hidden_state[:, :, : -context.num_padding_patches]
    return hidden_state.reshape(
        context.batch_size,
        context.num_concurrent_media,
        context.num_tiles,
        context.num_patches,
        -1,
    )


def build_projector_inputs(
    global_outputs: LayerInputs,
    local_captures: dict[int, LayerInputs],
    contexts: list[VisionBatchContext],
    intermediate_layers_indices: Iterable[int],
    device: str,
    calibration_gpu_cache: bool = False,
) -> LayerInputs:
    storage_device = device if calibration_gpu_cache else "cpu"
    next_args = []
    next_kwargs = []
    with torch.inference_mode():
        for batch_index, context in enumerate(contexts):
            hidden_state = _remove_vision_padding(
                _move_to_device(global_outputs.args[batch_index][0], device),
                context,
            )
            intermediate_states = []
            for layer_index in intermediate_layers_indices:
                captured_inputs = local_captures[layer_index]
                intermediate_states.append(
                    _move_to_device(captured_inputs.args[batch_index][0], device)
                )
            if intermediate_states:
                intermediate_hidden_states = torch.stack(
                    intermediate_states,
                    dim=-1,
                )
                intermediate_hidden_states = _remove_vision_padding(
                    intermediate_hidden_states,
                    context,
                )
                hidden_state = torch.cat(
                    [hidden_state, intermediate_hidden_states],
                    dim=-1,
                )
            next_args.append([_detach_to_device(hidden_state, storage_device)])
            next_kwargs.append({})
    return LayerInputs(next_args, next_kwargs)


def iter_projector_inputs(
    global_outputs: LayerInputs,
    local_captures: dict[int, LayerInputs],
    contexts: list[VisionBatchContext],
    intermediate_layers_indices: Iterable[int],
    device: str,
    consume: bool = False,
) -> Iterable[tuple[list[Any], dict[str, Any]]]:
    """Build one projector input at a time instead of retaining the full cache."""

    intermediate_layers_indices = tuple(intermediate_layers_indices)
    with torch.inference_mode():
        for batch_index, context in enumerate(contexts):
            hidden_state = _remove_vision_padding(
                _move_to_device(global_outputs.args[batch_index][0], device),
                context,
            )
            intermediate_states = [
                _move_to_device(
                    local_captures[layer_index].args[batch_index][0],
                    device,
                )
                for layer_index in intermediate_layers_indices
            ]
            if intermediate_states:
                intermediate_hidden_states = _remove_vision_padding(
                    torch.stack(intermediate_states, dim=-1),
                    context,
                )
                hidden_state = torch.cat(
                    [hidden_state, intermediate_hidden_states],
                    dim=-1,
                )
            yield [hidden_state], {}
            if consume:
                global_outputs.args[batch_index].clear()
                global_outputs.kwargs[batch_index].clear()
                for layer_index in intermediate_layers_indices:
                    local_captures[layer_index].args[batch_index].clear()
                    local_captures[layer_index].kwargs[batch_index].clear()


def run_projector(
    projector: torch.nn.Linear,
    projector_inputs: LayerInputs | Iterable[tuple[list[Any], dict[str, Any]]],
    hidden_size: int,
    device: str,
    calibration_gpu_cache: bool = False,
) -> list[torch.Tensor]:
    storage_device = device if calibration_gpu_cache else "cpu"
    states = []
    with torch.inference_mode():
        for args, kwargs in _iter_layer_inputs(projector_inputs):
            args = [_move_to_device(arg, device) for arg in args]
            kwargs = {k: _move_to_device(v, device) for k, v in kwargs.items()}
            projected = projector(*args, **kwargs)
            projected = projected.reshape(-1, projected.shape[-2], hidden_size)
            states.append(_detach_to_device(projected, storage_device))
    return states


def collect_multimodal_text_first_layer_inputs(
    model: torch.nn.Module,
    text_layer_stack: TextLayerStack,
    calibration_batches: Iterable[dict[str, Any]],
    cross_attention_states: list[torch.Tensor],
    device: str,
    calibration_gpu_cache: bool = False,
    use_cache: bool = False,
) -> LayerInputs:
    storage_device = device if calibration_gpu_cache else "cpu"
    cached_args = []
    cached_kwargs = []
    language_model = text_layer_stack.language_model
    vision_model = mllama_vision_layers(model).vision_model
    model_dtype = _module_dtype(model)
    with torch.inference_mode():
        for batch, cross_states in zip(calibration_batches, cross_attention_states):
            batch = _move_to_device(batch, device)
            input_ids = batch["input_ids"]
            attention_mask = batch.get("attention_mask")
            position_ids = batch.get("position_ids")
            past_key_values = batch.get("past_key_values")
            embedding_weight = getattr(language_model.embed_tokens, "weight", None)
            if torch.is_tensor(embedding_weight):
                input_ids = input_ids.to(embedding_weight.device)
            inputs_embeds = language_model.embed_tokens(input_ids)
            if position_ids is None:
                past_seen_tokens = (
                    past_key_values.get_seq_length()
                    if past_key_values is not None
                    else 0
                )
                position_ids = torch.arange(
                    inputs_embeds.shape[1],
                    device=inputs_embeds.device,
                )
                position_ids = position_ids + past_seen_tokens
                position_ids = position_ids.unsqueeze(0)
            else:
                position_ids = position_ids.to(inputs_embeds.device)
            causal_mask = _prepare_causal_mask(
                language_model,
                inputs_embeds,
                attention_mask,
                past_key_values,
                position_ids,
            )
            position_embeddings = language_model.rotary_emb(
                inputs_embeds,
                position_ids=position_ids,
            )

            cross_attention_mask = batch.get("cross_attention_mask")
            if cross_attention_mask is not None:
                cross_attention_mask, full_text_row_masked_out_mask = (
                    _prepare_cross_attention_mask(
                        cross_attention_mask,
                        num_vision_tokens=vision_model.num_patches,
                        dtype=model_dtype,
                    )
                )
                past_seen_tokens = (
                    past_key_values.get_seq_length()
                    if past_key_values is not None
                    else 0
                )
                current_pos = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    + past_seen_tokens
                )
                cross_attention_mask = cross_attention_mask[:, :, current_pos]
                full_text_row_masked_out_mask = full_text_row_masked_out_mask[
                    :,
                    :,
                    current_pos,
                ]
            else:
                full_text_row_masked_out_mask = None

            cached_args.append([_detach_to_device(inputs_embeds, storage_device)])
            cached_kwargs.append(
                {
                    "cross_attention_states": _detach_to_device(
                        cross_states,
                        storage_device,
                    ),
                    "cross_attention_mask": _detach_to_device(
                        cross_attention_mask,
                        storage_device,
                    ),
                    "attention_mask": _detach_to_device(causal_mask, storage_device),
                    "full_text_row_masked_out_mask": _detach_to_device(
                        full_text_row_masked_out_mask,
                        storage_device,
                    ),
                    "position_ids": _detach_to_device(position_ids, storage_device),
                    "past_key_values": past_key_values,
                    "use_cache": use_cache,
                    "position_embeddings": _detach_to_device(
                        position_embeddings,
                        storage_device,
                    ),
                }
            )
    return LayerInputs(cached_args, cached_kwargs)


def build_lm_head_inputs(
    language_model: torch.nn.Module,
    final_text_inputs: LayerInputs,
    device: str,
    calibration_gpu_cache: bool = False,
) -> LayerInputs:
    storage_device = device if calibration_gpu_cache else "cpu"
    next_args = []
    next_kwargs = []
    with torch.inference_mode():
        for args in final_text_inputs.args:
            hidden_states = _move_to_device(args[0], device)
            if hasattr(language_model, "norm"):
                hidden_states = language_model.norm(hidden_states)
            next_args.append([_detach_to_device(hidden_states, storage_device)])
            next_kwargs.append({})
    return LayerInputs(next_args, next_kwargs)


def offload_completed_modules_for_lm_head(
    vision_stack: VisionLayerStack,
    text_layer_stack: TextLayerStack,
    projector: torch.nn.Linear,
    lm_head: torch.nn.Linear,
    device: str,
) -> None:
    """Free GPU memory from modules that are no longer used before lm_head GPTQ."""

    text_layer_stack.language_model.to("cpu")
    vision_stack.vision_model.to("cpu")
    projector.to("cpu")
    lm_head.to(device)
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()


def load_model(
    model_name_or_path: str, dtype: torch.dtype, device: str
) -> torch.nn.Module:
    kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
    }
    if device != "cpu":
        kwargs["device_map"] = {"": device}
    return transformers.MllamaForConditionalGeneration.from_pretrained(
        model_name_or_path,
        **kwargs,
    )


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


def quantize_full_multimodal_scope(
    model: torch.nn.Module,
    text_layer_stack: TextLayerStack,
    calibration_batches: list[dict[str, Any]],
    config: GPTQConfig,
    device: str,
    calibration_gpu_cache: bool,
    verbose: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    layer_logs: list[dict[str, Any]] = []
    vision_stack = mllama_vision_layers(model)
    if verbose:
        print(
            f"Capturing vision inputs from {len(calibration_batches)} batches",
            flush=True,
        )
    vision_inputs, vision_contexts = collect_vision_transformer_inputs(
        vision_stack,
        calibration_batches,
        device=device,
        calibration_gpu_cache=calibration_gpu_cache,
    )
    # The remaining stages only need the encoded text fields. Keeping normalized
    # pixels for every sample would otherwise add several GiB to every later peak.
    for batch in calibration_batches:
        for key in ("pixel_values", "aspect_ratio_ids", "aspect_ratio_mask"):
            batch.pop(key, None)

    intermediate_layers = frozenset(
        getattr(vision_stack.vision_model, "intermediate_layers_indices", ())
    )
    if verbose:
        print(
            f"Quantizing {len(vision_stack.local_layers)} vision encoder layers",
            flush=True,
        )
    local_outputs, local_logs, local_captures = quantize_layer_sequence(
        vision_stack.local_layers,
        f"{vision_stack.prefix}.transformer.layers",
        vision_inputs,
        config,
        device,
        VISION_TARGET_MODULE_GROUPS,
        verbose=verbose,
        capture_after_layers=intermediate_layers,
    )
    del vision_inputs
    missing_captures = intermediate_layers - frozenset(local_captures)
    if missing_captures:
        raise ValueError(
            "Vision intermediate layer capture failed for indices "
            f"{sorted(missing_captures)}"
        )
    layer_logs.extend(local_logs)

    global_inputs = build_global_vision_inputs(
        vision_stack.vision_model,
        local_outputs,
        vision_contexts,
        device=device,
        calibration_gpu_cache=calibration_gpu_cache,
    )
    del local_outputs
    if verbose:
        print(
            f"Quantizing {len(vision_stack.global_layers)} global vision layers",
            flush=True,
        )
    global_outputs, global_logs, _ = quantize_layer_sequence(
        vision_stack.global_layers,
        f"{vision_stack.prefix}.global_transformer.layers",
        global_inputs,
        config,
        device,
        VISION_TARGET_MODULE_GROUPS,
        verbose=verbose,
    )
    clear_vision_attention_mask_cache()
    del global_inputs
    layer_logs.extend(global_logs)

    projector_prefix, projector = mllama_multimodal_projector(model)
    projector_logs = quantize_standalone_linear(
        projector_prefix,
        projector,
        iter_projector_inputs(
            global_outputs,
            local_captures,
            vision_contexts,
            getattr(vision_stack.vision_model, "intermediate_layers_indices", ()),
            device=device,
        ),
        config,
        device,
        verbose=verbose,
    )
    layer_logs.extend(projector_logs)
    cross_attention_states = run_projector(
        projector,
        iter_projector_inputs(
            global_outputs,
            local_captures,
            vision_contexts,
            getattr(vision_stack.vision_model, "intermediate_layers_indices", ()),
            device=device,
            consume=True,
        ),
        hidden_size=projector.out_features,
        device=device,
        calibration_gpu_cache=calibration_gpu_cache,
    )
    del global_outputs, local_captures, vision_contexts

    if verbose:
        print(
            f"Capturing multimodal text first-layer inputs from "
            f"{len(calibration_batches)} batches",
            flush=True,
        )
    text_inputs = collect_multimodal_text_first_layer_inputs(
        model,
        text_layer_stack,
        calibration_batches,
        cross_attention_states,
        device=device,
        calibration_gpu_cache=calibration_gpu_cache,
    )
    del cross_attention_states

    total_layers = len(text_layer_stack.layers)
    if verbose:
        print(
            f"Quantizing {total_layers} full multimodal text decoder layers "
            f"from {text_layer_stack.prefix}",
            flush=True,
        )
    for layer_index, layer in enumerate(text_layer_stack.layers):
        layer_kind = mllama_layer_kind(
            layer,
            layer_index=layer_index,
            cross_attention_layers=text_layer_stack.cross_attention_layers,
        )
        module_groups = (
            TEXT_CROSS_TARGET_MODULE_GROUPS
            if layer_kind == "cross_attention"
            else TEXT_SELF_TARGET_MODULE_GROUPS
        )
        progress_name = (
            f"[layer {layer_index + 1}/{total_layers} "
            f"{layer.__class__.__name__}]"
        )
        text_inputs, logs = quantize_layer_groups(
            layer,
            text_inputs,
            config,
            device,
            module_groups,
            progress_name=progress_name,
            verbose=verbose,
            ignored_kwargs=(
                TEXT_SELF_IGNORED_KWARGS
                if layer_kind == "self_attention"
                else frozenset()
            ),
        )
        for log in logs:
            log["layer"] = layer_index
            log["full_name"] = (
                f"{text_layer_stack.prefix}.{layer_index}.{log['module']}"
            )
        layer_logs.extend(logs)

    lm_head_prefix, lm_head = mllama_lm_head(model)
    lm_head_inputs = build_lm_head_inputs(
        text_layer_stack.language_model,
        text_inputs,
        device=device,
        calibration_gpu_cache=calibration_gpu_cache,
    )
    del text_inputs
    if config.quantization_format == "s3d8" and str(device).startswith("cuda"):
        if verbose:
            print(
                "[lm_head]: offloading completed model body to CPU for S3D8 fitting",
                flush=True,
            )
        offload_completed_modules_for_lm_head(
            vision_stack,
            text_layer_stack,
            projector,
            lm_head,
            device,
        )
    layer_logs.extend(
        quantize_standalone_linear(
            lm_head_prefix,
            lm_head,
            lm_head_inputs,
            config,
            device,
            verbose=verbose,
        )
    )
    return layer_logs, []


def quantize(
    model_name: str,
    output_dir: Path,
    bits: int,
    quantization_format: str = "int",
    target_scope: str = "text-self",
    calibration_source: str | None = None,
    codepoints: int | None = None,
    group_size: int = DEFAULT_GROUP_SIZE,
    batch_size: int = 1,
    calibration_samples: int = DEFAULT_CALIBRATION_SAMPLES,
    calibration_max_tokens: int = DEFAULT_CALIBRATION_MAX_TOKENS,
    calibration_sort: str = DEFAULT_CALIBRATION_SORT,
    calibration_data_min_length: int = DEFAULT_CALIBRATION_DATA_MIN_LENGTH,
    calibration_gpu_cache: bool = False,
    calibration_split: str | None = None,
    calibration_data_paths: Iterable[str | Path] = (),
    synthetic_image_shards: int = DEFAULT_SYNTHETIC_IMAGE_SHARDS,
    calibration_task: str = "vqa",
    load_vqa_from_s3: bool = False,
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
    if target_scope not in SUPPORTED_TARGET_SCOPES:
        raise ValueError(
            f"Unsupported target_scope={target_scope!r}, expected one of "
            f"{SUPPORTED_TARGET_SCOPES}"
        )
    resolved_calibration_source = calibration_source
    if resolved_calibration_source is None:
        resolved_calibration_source = (
            "vqav2" if target_scope == "full-multimodal" else "c4"
        )
    if resolved_calibration_source not in SUPPORTED_CALIBRATION_SOURCES:
        raise ValueError(
            f"Unsupported calibration_source={resolved_calibration_source!r}, "
            f"expected one of {SUPPORTED_CALIBRATION_SOURCES}"
        )
    if calibration_split is None:
        calibration_split = (
            DEFAULT_VQAV2_CALIBRATION_SPLIT
            if resolved_calibration_source == "vqav2"
            else "validation"
        )
    if target_scope == "text-self" and resolved_calibration_source != "c4":
        raise ValueError("target_scope='text-self' currently requires C4 calibration")
    if target_scope == "full-multimodal" and resolved_calibration_source == "c4":
        raise ValueError(
            "target_scope='full-multimodal' requires multimodal calibration "
            "(vqav2, synthetic, or eval-task)"
        )
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
    if calibration_samples < 1:
        raise ValueError(
            f"calibration_samples must be >= 1, got {calibration_samples}"
        )
    if calibration_max_tokens < 1:
        raise ValueError(
            f"calibration_max_tokens must be >= 1, got {calibration_max_tokens}"
        )
    if calibration_sort not in SUPPORTED_CALIBRATION_SORTS:
        raise ValueError(
            f"Unsupported calibration_sort={calibration_sort!r}, expected one of "
            f"{SUPPORTED_CALIBRATION_SORTS}"
        )
    if synthetic_image_shards < 1:
        raise ValueError(
            f"synthetic_image_shards must be >= 1, got {synthetic_image_shards}"
        )

    calibration_data_paths = tuple(str(path) for path in calibration_data_paths)
    staged_calibration_data_paths = calibration_data_paths
    if resolved_calibration_source == "synthetic":
        if verbose and any(
            path.startswith("s3://") for path in calibration_data_paths
        ):
            print("Staging public synthetic calibration data", flush=True)
        staged_calibration_data_paths = tuple(
            _stage_public_synthetic_data(
                path,
                synthetic_image_shards=synthetic_image_shards,
            )
            for path in calibration_data_paths
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

    if resolved_calibration_source == "c4":
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
        calibration_prepared_examples = len(prepared_calibration_dataset)
        calibration_metadata = {
            "dataset": "allenai/c4",
            "data_files": c4_data_files,
            "split": c4_split,
            "n_samples": calibration_samples,
            "field": "text",
            "source": resolved_calibration_source,
        }
    elif resolved_calibration_source == "vqav2":
        if verbose:
            print(
                "Loading VQAv2 multimodal calibration from Hugging Face "
                f"samples={calibration_samples} split={calibration_split}",
                flush=True,
            )
            if calibration_split != "train":
                print(f"Warning: {VQAV2_PROTOTYPE_WARNING}", flush=True)
        calibration_batches = load_vqav2_calibration_batches(
            processor,
            n_samples=calibration_samples,
            batch_size=batch_size,
            split=calibration_split,
            max_tokens=calibration_max_tokens,
            load_from_s3=load_vqa_from_s3,
        )
        calibration_prepared_examples = _count_calibration_examples(
            calibration_batches
        )
        calibration_metadata = {
            "dataset": (
                VQAV2_TRAIN_CALIBRATION_DATASET
                if calibration_split == "train"
                else "lmms-lab/VQAv2"
            ),
            "split": calibration_split,
            "n_samples": calibration_samples,
            "source": resolved_calibration_source,
            "load_from_s3": (
                load_vqa_from_s3 if calibration_split != "train" else False
            ),
            **(
                {"prototype_warning": VQAV2_PROTOTYPE_WARNING}
                if calibration_split != "train"
                else {}
            ),
        }
    elif resolved_calibration_source == "eval-task":
        if calibration_task not in vqa.TASKS:
            raise ValueError(
                f"Unsupported calibration_task={calibration_task!r}, expected one of "
                f"{tuple(vqa.TASKS)}"
            )
        if verbose:
            print(
                "Loading eval-task multimodal calibration "
                f"task={calibration_task} samples={calibration_samples} "
                f"split={calibration_split}",
                flush=True,
            )
        calibration_batches = load_eval_task_calibration_batches(
            processor,
            task_name=calibration_task,
            n_samples=calibration_samples,
            batch_size=batch_size,
            split=calibration_split,
            max_tokens=calibration_max_tokens,
            load_vqa_from_s3=load_vqa_from_s3,
        )
        calibration_prepared_examples = _count_calibration_examples(
            calibration_batches
        )
        calibration_metadata = {
            "dataset": calibration_task,
            "split": calibration_split,
            "n_samples": calibration_samples,
            "source": resolved_calibration_source,
            "load_vqa_from_s3": load_vqa_from_s3,
        }
    elif resolved_calibration_source == "synthetic":
        if verbose:
            print(
                "Loading synthetic multimodal calibration "
                f"samples={calibration_samples}",
                flush=True,
            )
        calibration_batches = load_synthetic_calibration_batches(
            processor,
            calibration_data_paths=staged_calibration_data_paths,
            n_samples=calibration_samples,
            batch_size=batch_size,
            max_tokens=calibration_max_tokens,
            synthetic_image_shards=synthetic_image_shards,
        )
        calibration_prepared_examples = _count_calibration_examples(
            calibration_batches
        )
        calibration_metadata = {
            "dataset": "synthetic",
            "paths": [str(path) for path in calibration_data_paths],
            "n_samples": calibration_samples,
            "source": resolved_calibration_source,
            "image_shards": synthetic_image_shards,
        }
    else:
        raise ValueError(
            f"Unsupported calibration source {resolved_calibration_source!r}"
        )
    if not calibration_batches:
        raise ValueError("No calibration batches were loaded")
    if (
        resolved_calibration_source != "c4"
        and calibration_prepared_examples < calibration_samples
    ):
        raise ValueError(
            "Calibration source returned fewer examples than requested: "
            f"requested={calibration_samples}, loaded={calibration_prepared_examples}"
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
        if target_scope == "full-multimodal":
            layer_logs, skipped_cross_attention_layers = quantize_full_multimodal_scope(
                model,
                text_layer_stack,
                calibration_batches,
                config,
                device,
                calibration_gpu_cache,
                verbose,
            )
            quantized_layer_count = len(
                {log["full_name"].rsplit(".", 1)[0] for log in layer_logs}
            )
        else:
            if verbose:
                print(
                    f"Capturing first-layer inputs from {len(calibration_batches)} "
                    f"calibration batches ({calibration_prepared_examples} examples)",
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
                        f"layer_time="
                        f"{format_duration(time.perf_counter() - layer_start)} "
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
            f"{quantized_layer_count} targeted layers/modules",
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
        "calibration_prepared_examples": calibration_prepared_examples,
        "calibration_batches": len(calibration_batches),
        "calibration_gpu_cache": calibration_gpu_cache,
        "torch_dtype": torch_dtype,
        "gptq": dataclasses.asdict(config),
        "calibration": calibration_metadata,
        "device": device,
        "target_scope": (
            "mllama_full_multimodal_heavy_linears"
            if target_scope == "full-multimodal"
            else "mllama_text_self_attention_decoder_layers"
        ),
        "target_scope_option": target_scope,
        "calibration_source": resolved_calibration_source,
        "skipped_cross_attention_layers": skipped_cross_attention_layers,
        "target_module_groups": (
            {
                "vision": VISION_TARGET_MODULE_GROUPS,
                "text_self": TEXT_SELF_TARGET_MODULE_GROUPS,
                "text_cross": TEXT_CROSS_TARGET_MODULE_GROUPS,
                "standalone": ("model.multi_modal_projector", "lm_head"),
            }
            if target_scope == "full-multimodal"
            else TARGET_MODULE_GROUPS
        ),
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
    parser.add_argument(
        "--target-scope",
        choices=SUPPORTED_TARGET_SCOPES,
        default="text-self",
        help="Quantization target scope. Default keeps the legacy decoder-only path.",
    )
    parser.add_argument(
        "--calibration-source",
        choices=SUPPORTED_CALIBRATION_SOURCES,
        default=None,
        help=(
            "Calibration source. Defaults to c4 for --target-scope text-self and "
            "vqav2 for --target-scope full-multimodal."
        ),
    )
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
        help="C4 calibration sample ordering before batching",
    )
    parser.add_argument(
        "--calibration-data-min-length",
        type=int,
        default=DEFAULT_CALIBRATION_DATA_MIN_LENGTH,
        help="Drop C4 calibration samples shorter than this many tokens",
    )
    parser.add_argument(
        "--calibration-gpu-cache",
        action="store_true",
        help=(
            "Keep activation caches on the quantization GPU; leave disabled "
            "for large runs"
        ),
    )
    parser.add_argument(
        "--calibration-split",
        default=None,
        help=(
            "Split for multimodal calibration; defaults to train for VQAv2 "
            "and validation for eval-task"
        ),
    )
    parser.add_argument(
        "--calibration-data-path",
        dest="calibration_data_paths",
        action="append",
        default=[],
        help=(
            "Synthetic calibration shard path or public s3:// rollout URI; "
            "repeat for multiple shards"
        ),
    )
    parser.add_argument(
        "--synthetic-image-shards",
        type=int,
        default=DEFAULT_SYNTHETIC_IMAGE_SHARDS,
        help=(
            "Number of evenly distributed public ImageNet Arrow shards to "
            "download for synthetic calibration"
        ),
    )
    parser.add_argument(
        "--calibration-task",
        default="vqa",
        choices=tuple(vqa.TASKS),
        help="Task used with --calibration-source eval-task",
    )
    parser.add_argument(
        "--load-vqa-from-s3",
        action="store_true",
        help="Load VQAv2 calibration from the legacy S3 cache instead of Hugging Face",
    )
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
            target_scope=args.target_scope,
            calibration_source=args.calibration_source,
        )
        metadata = quantize(
            model_name=args.model_name,
            output_dir=output_dir,
            bits=bits,
            quantization_format=args.format,
            target_scope=args.target_scope,
            calibration_source=args.calibration_source,
            codepoints=args.codepoints,
            group_size=args.group_size,
            batch_size=args.batch_size,
            calibration_samples=args.calibration_samples,
            calibration_max_tokens=args.calibration_max_tokens,
            calibration_sort=args.calibration_sort,
            calibration_data_min_length=args.calibration_data_min_length,
            calibration_gpu_cache=args.calibration_gpu_cache,
            calibration_split=args.calibration_split,
            calibration_data_paths=args.calibration_data_paths,
            synthetic_image_shards=args.synthetic_image_shards,
            calibration_task=args.calibration_task,
            load_vqa_from_s3=args.load_vqa_from_s3,
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
