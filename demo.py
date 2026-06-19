#!/usr/bin/env python3
"""Standalone HuggingFace demo for the Llama-Mobile QAT vision model."""

from __future__ import annotations

import argparse
import io
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Iterator

import PIL.Image
import requests
from safetensors import safe_open
import torch
import torch.nn.functional as F
import transformers

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

PACKED_S3D8_FORMAT = "llama-mobile-s3d8-packed-v1"
FORMAT_METADATA_KEY = "llama_mobile_format"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run an interactive prompt/image chat with Meta Llama 3.2 Vision Instruct, "
            "optionally loading a released Llama-Mobile QAT checkpoint."
        ),
        epilog=(
            "Example checkpoint download: wget "
            "https://graphcore-research-public.s3.eu-west-1.amazonaws.com/"
            "2026-llama-mobile/hf_models/vision-11B-s3d8-packed.safetensors"
        ),
    )
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.2-11B-Vision-Instruct",
        help="HuggingFace model id or local path.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Optional BF16 or packed S3D8 safetensors checkpoint to load after the base model.",
    )
    parser.add_argument(
        "--image",
        help="Image URL or local path.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=100,
        help="Maximum number of generated tokens.",
    )
    parser.add_argument(
        "--int8-activations",
        action="store_true",
        help="Simulate per-channel INT8 activation quantization on linear inputs.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help='Device to run on. "auto" uses CUDA when available.',
    )
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
        help="Torch dtype for model loading.",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Enable sampling. By default generation is deterministic.",
    )
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def load_image(url_or_path: str) -> PIL.Image.Image:
    if url_or_path.startswith(("http://", "https://")):
        response = requests.get(url_or_path, timeout=30)
        response.raise_for_status()
        image = PIL.Image.open(io.BytesIO(response.content)).convert("RGB")
    else:
        image = PIL.Image.open(url_or_path).convert("RGB")

    scale = min(560 / image.width, 560 / image.height)
    size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
    return image.resize(size)


def quantise_channel_int8(x: torch.Tensor) -> torch.Tensor:
    scale = x.abs().amax(dim=tuple(range(x.ndim - 1)), keepdim=True).clamp_min(1e-8) / 127
    q = torch.clamp(torch.round(x / scale), -128, 127).to(torch.int8)
    return q.to(x.dtype) * scale


@torch.compile(fullgraph=True)
def dequantise_packed_s3d8(
    packed: torch.Tensor,
    scale: torch.Tensor,
    table: torch.Tensor,
    rows: int,
    cols: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    packed_rows = (rows + 2) // 3
    packed_cols = packed.numel() // packed_rows

    idx = ((packed >> 1) & 0x1F).long()
    values = table.to(dtype=dtype)[idx]
    sign0 = packed & 0x01
    sign1 = (packed >> 6) & 0x01
    sign2 = sign0 ^ ((packed >> 7) & 0x01)
    signs = torch.stack((sign0, sign1, sign2), dim=-1).bool()
    values = torch.where(signs, -values, values)

    weight = (
        values.view(packed_rows, packed_cols, 3)
        .permute(0, 2, 1)
        .reshape(packed_rows * 3, packed_cols)[:rows, :cols]
    )
    return weight * scale.to(dtype=dtype)


if triton is not None:

    @triton.jit
    def _triton_dequant_s3d8_kernel(
        packed,
        scale,
        table,
        out,
        rows: tl.constexpr,
        cols: tl.constexpr,
        packed_cols: tl.constexpr,
        n_col_blocks: tl.constexpr,
        block_m: tl.constexpr,
        block_n: tl.constexpr,
    ):
        pid = tl.program_id(0)
        pid_m = pid // n_col_blocks
        pid_n = pid - pid_m * n_col_blocks
        offs_m = pid_m * block_m + tl.arange(0, block_m)
        offs_n = pid_n * block_n + tl.arange(0, block_n)
        mask = (offs_m[:, None] < rows) & (offs_n[None, :] < cols)

        lane = offs_m % 3
        packed_group = offs_m // 3
        byte = tl.load(
            packed + packed_group[:, None] * packed_cols + offs_n[None, :],
            mask=mask,
            other=0,
        ).to(tl.int32)

        idx = (byte >> 1) & 31
        value = tl.load(
            table + idx * 3 + lane[:, None],
            mask=mask,
            other=0,
        ).to(tl.float32)

        sign0 = byte & 1
        sign1 = (byte >> 6) & 1
        sign2 = sign0 ^ ((byte >> 7) & 1)
        sign = tl.where(lane[:, None] == 0, sign0, tl.where(lane[:, None] == 1, sign1, sign2))
        value = tl.where(sign != 0, -value, value)
        value *= tl.load(scale + offs_m[:, None], mask=offs_m[:, None] < rows, other=0).to(tl.float32)

        tl.store(out + offs_m[:, None] * cols + offs_n[None, :], value, mask=mask)


def dequantise_packed_s3d8_triton(
    packed: torch.Tensor,
    scale: torch.Tensor,
    table: torch.Tensor,
    rows: int,
    cols: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("triton is not installed")
    if packed.device.type != "cuda":
        raise RuntimeError("dequantise_packed_s3d8_triton requires CUDA tensors")

    packed_rows = (rows + 2) // 3
    packed_cols = packed.numel() // packed_rows
    out = torch.empty((rows, cols), device=packed.device, dtype=dtype)
    block_m = 16
    block_n = 64
    n_row_blocks = triton.cdiv(rows, block_m)
    n_col_blocks = triton.cdiv(cols, block_n)
    grid = (n_row_blocks * n_col_blocks,)
    _triton_dequant_s3d8_kernel[grid](
        packed,
        scale,
        table,
        out,
        rows,
        cols,
        packed_cols,
        n_col_blocks,
        block_m,
        block_n,
        num_warps=4,
    )
    return out


class PackedS3D8Weight(torch.nn.Module):
    def __init__(self, shape: torch.Size | tuple[int, int], device: torch.device | str | None = None) -> None:
        super().__init__()
        if len(shape) != 2:
            raise ValueError(f"Packed S3D8 weights must be rank-2, got {tuple(shape)}")
        self.shape = tuple(shape)
        rows, cols = self.shape
        packed_rows = (rows + 2) // 3
        packed_cols = cols + (-cols % 16)
        self.register_buffer("packed", torch.empty(packed_rows * packed_cols, dtype=torch.uint8, device=device))
        self.register_buffer("scale", torch.empty(rows, 1, dtype=torch.bfloat16, device=device))
        self.register_buffer("table", torch.empty(32, 3, dtype=torch.int8, device=device))

    def forward(self, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        rows, cols = self.shape
        if self.packed.device.type == "cuda":
            return dequantise_packed_s3d8_triton(self.packed, self.scale, self.table, rows, cols, dtype)
        return dequantise_packed_s3d8(self.packed, self.scale, self.table, rows, cols, dtype)


class PackedS3D8Linear(torch.nn.Module):
    def __init__(self, linear: torch.nn.Linear) -> None:
        super().__init__()
        self.weight = PackedS3D8Weight(linear.weight.shape, device=linear.weight.device)
        if linear.bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = torch.nn.Parameter(linear.bias.detach().clone())

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight(dtype=input.dtype), self.bias)


class PackedS3D8Embedding(torch.nn.Module):
    def __init__(self, embedding: torch.nn.Embedding) -> None:
        super().__init__()
        self.weight = PackedS3D8Weight(embedding.weight.shape, device=embedding.weight.device)
        self.dtype = embedding.weight.dtype
        self.padding_idx = embedding.padding_idx
        self.max_norm = embedding.max_norm
        self.norm_type = embedding.norm_type
        self.scale_grad_by_freq = embedding.scale_grad_by_freq
        self.sparse = embedding.sparse

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.embedding(
            input,
            self.weight(dtype=self.dtype),
            padding_idx=self.padding_idx,
            max_norm=self.max_norm,
            norm_type=self.norm_type,
            scale_grad_by_freq=self.scale_grad_by_freq,
            sparse=self.sparse,
        )


def load_safetensors_checkpoint(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    with safe_open(path, framework="pt", device="cpu") as checkpoint:
        metadata = checkpoint.metadata() or {}
        return {key: checkpoint.get_tensor(key) for key in checkpoint.keys()}, metadata


def is_packed_s3d8_checkpoint(path: Path | None) -> bool:
    if path is None:
        return False
    with safe_open(path, framework="pt", device="cpu") as checkpoint:
        metadata = checkpoint.metadata() or {}
    return metadata.get(FORMAT_METADATA_KEY) == PACKED_S3D8_FORMAT


def replace_packed_s3d8_modules(
    module: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
    prefix: str = "",
) -> None:
    for name, child in list(module.named_children()):
        child_prefix = f"{prefix}.{name}" if prefix else name
        packed_key = f"{child_prefix}.weight.packed"
        if packed_key in state_dict:
            if isinstance(child, torch.nn.Linear):
                replacement = PackedS3D8Linear(child)
            elif isinstance(child, torch.nn.Embedding):
                replacement = PackedS3D8Embedding(child)
            else:
                raise TypeError(
                    f"Packed S3D8 checkpoint contains {packed_key}, "
                    f"but {child_prefix} is {type(child).__name__}"
                )
            module.add_module(name, replacement)
            continue

        replace_packed_s3d8_modules(child, state_dict, child_prefix)


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path) -> None:
    state_dict, metadata = load_safetensors_checkpoint(checkpoint_path)
    if metadata.get(FORMAT_METADATA_KEY) == PACKED_S3D8_FORMAT:
        replace_packed_s3d8_modules(model, state_dict)
        model.load_state_dict(state_dict, assign=True)
    else:
        model.load_state_dict(state_dict)


def materialize_mllama_rotary_buffers(
    model: transformers.MllamaForConditionalGeneration,
    dtype: torch.dtype,
) -> None:
    rotary_emb = model.language_model.model.rotary_emb
    if rotary_emb.inv_freq.device.type != "meta":
        return
    inv_freq, rotary_emb.attention_scaling = rotary_emb.rope_init_fn(rotary_emb.config, device="cpu")
    rotary_emb.register_buffer("inv_freq", inv_freq.to(dtype=dtype), persistent=False)
    rotary_emb.original_inv_freq = rotary_emb.inv_freq


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_model(args: argparse.Namespace, dtype: torch.dtype) -> transformers.MllamaForConditionalGeneration:
    if is_packed_s3d8_checkpoint(args.checkpoint):
        config = transformers.AutoConfig.from_pretrained(args.model)
        with torch.device("meta"):
            model = transformers.MllamaForConditionalGeneration(config)
        model.to(dtype=dtype)
        load_checkpoint(model, args.checkpoint)
        materialize_mllama_rotary_buffers(model, dtype)
        return model.to(device=resolve_device(args.device), dtype=dtype)

    model = transformers.MllamaForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map=str(resolve_device(args.device)),
    )
    if args.checkpoint:
        load_checkpoint(model, args.checkpoint)
    return model


@contextmanager
def int8_activations(model: torch.nn.Module) -> Iterator[None]:
    handles = [
        module.register_forward_pre_hook(lambda _module, args: (quantise_channel_int8(args[0]), *args[1:]))
        for module in model.modules()
        if isinstance(module, (torch.nn.Linear, PackedS3D8Linear))
    ]
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def user_message(prompt: str, include_image: bool) -> dict[str, object]:
    content = [{"type": "text", "text": prompt}]
    if include_image:
        content.append({"type": "image"})
    return {"role": "user", "content": content}


def assistant_message(response: str) -> dict[str, object]:
    return {"role": "assistant", "content": [{"type": "text", "text": response}]}


def generate_response(
    model: transformers.MllamaForConditionalGeneration,
    processor: transformers.AutoProcessor,
    messages: list[dict[str, object]],
    image: PIL.Image.Image | None,
    max_new_tokens: int,
    do_sample: bool,
) -> str:
    processor_args = {
        "text": processor.apply_chat_template(messages, add_generation_prompt=True),
        "add_special_tokens": False,
        "return_tensors": "pt",
    }
    if image is not None:
        processor_args["images"] = image

    inputs = processor(**processor_args)
    model_inputs = inputs.to(model.device)
    sample_args = {} if do_sample else {"do_sample": False, "temperature": None, "top_p": None}
    outputs = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        **sample_args,
    )
    input_token_count = model_inputs["input_ids"].shape[-1]
    return processor.tokenizer.decode(outputs[0][input_token_count:], skip_special_tokens=True).strip()


def chat(
    model: transformers.MllamaForConditionalGeneration,
    processor: transformers.AutoProcessor,
    image: PIL.Image.Image | None,
    max_new_tokens: int,
    do_sample: bool,
) -> None:
    messages: list[dict[str, object]] = []
    include_image = image is not None

    while True:
        try:
            prompt = input("> " if sys.stdin.isatty() else "")
        except EOFError:
            break
        prompt = prompt.strip()
        if not prompt:
            continue

        messages.append(user_message(prompt, include_image))
        include_image = False
        response = generate_response(
            model=model,
            processor=processor,
            messages=messages,
            image=image,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
        )
        print(response, flush=True)
        messages.append(assistant_message(response))


def main() -> None:
    args = parse_args()
    dtype = dtype_from_name(args.dtype)

    model = load_model(args, dtype)
    processor = transformers.AutoProcessor.from_pretrained(args.model)
    model.requires_grad_(False)

    with (int8_activations(model) if args.int8_activations else nullcontext()):
        chat(
            model=model,
            processor=processor,
            image=(load_image(args.image) if args.image else None),
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
        )


if __name__ == "__main__":
    main()
