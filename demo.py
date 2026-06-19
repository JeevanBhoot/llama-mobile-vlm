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
import safetensors.torch
import torch
import transformers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run an interactive prompt/image chat with Meta Llama 3.2 Vision Instruct, "
            "optionally loading the released Llama-Mobile QAT BF16 checkpoint."
        ),
        epilog=(
            "Example checkpoint download: wget "
            "https://graphcore-research-public.s3.eu-west-1.amazonaws.com/"
            "2026-llama-mobile/proud-sponge-1878-bf16.safetensors"
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
        help="Optional safetensors checkpoint to load after the base model.",
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
        "--device-map",
        default="auto",
        help='Transformers device_map argument. Use "auto" by default.',
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


@contextmanager
def int8_activations(model: torch.nn.Module) -> Iterator[None]:
    handles = [
        module.register_forward_pre_hook(lambda _module, args: (quantise_channel_int8(args[0]), *args[1:]))
        for module in model.modules()
        if isinstance(module, torch.nn.Linear)
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

    model = transformers.MllamaForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map=args.device_map,
    )
    processor = transformers.AutoProcessor.from_pretrained(model.config._name_or_path)
    model.requires_grad_(False)

    if args.checkpoint:
        state_dict = safetensors.torch.load_file(args.checkpoint, device="cpu")
        model.load_state_dict(state_dict)

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
