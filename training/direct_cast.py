# Copyright (c) 2026 Graphcore Ltd. All rights reserved.

"""Direct-cast Hugging Face Llama weights to a quantised checkpoint.

The output is a safetensors checkpoint containing ``weight_formats`` quantisation
metadata. It can be consumed by ``squashedtensors.py --checkpoint``.
"""

import argparse
from pathlib import Path
from typing import Literal

import safetensors.torch
import torch
import transformers
import weight_formats.fit as F
import weight_formats.quantisation as Q
import weight_formats.quantisation_training as QT
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.models.mllama.modeling_mllama import MllamaForConditionalGeneration

FMT_CHANNEL_INT8 = Q.LinearScalingFormat(
    Q.IntFormat(8),
    scale_format=Q.BFLOAT16,
    block_shape=(1, None),
    scaling="absmax",
)

FMT_CHANNEL_S3D8 = F.Scaled(
    8 / 3,
    "s3d8",
    scale_format=Q.BFLOAT16,
    block_shape=(1, None),
    scaling="absmax",
    args=dict(threshold=1e-3),
)

DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def format_for(name: Literal["int8", "s3d8"]) -> Q.TensorFormat | F.Scaled:
    if name == "int8":
        return FMT_CHANNEL_INT8
    if name == "s3d8":
        return FMT_CHANNEL_S3D8
    raise ValueError(f"Unsupported format {name!r}")


def model_class(model_name_or_path: str) -> type[LlamaForCausalLM] | type[MllamaForConditionalGeneration]:
    config = transformers.AutoConfig.from_pretrained(model_name_or_path)
    if config.model_type == "llama":
        return LlamaForCausalLM
    if config.model_type == "mllama":
        return MllamaForConditionalGeneration
    raise ValueError(
        f"Unsupported model type {config.model_type!r} for {model_name_or_path}, "
        "expected 'llama' or 'mllama'"
    )


def load_model(model_name_or_path: str, dtype: torch.dtype, device: str) -> torch.nn.Module:
    cls = model_class(model_name_or_path)
    model = cls.from_pretrained(model_name_or_path, torch_dtype=dtype)
    if device != "cpu":
        model.to(device)
    return model


def checkpoint_state(model: torch.nn.Module, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    state = QT.save(model)
    out = {}
    for key, tensor in state.items():
        if tensor.dtype.is_floating_point:
            tensor = tensor.to(dtype)
        # safetensors.save_file rejects shared storage. QT.convert preserves
        # tied weights such as Llama embed_tokens/lm_head, so clone each tensor
        # after moving it to CPU to make the checkpoint self-contained.
        out[key] = tensor.cpu().contiguous().clone()
    return out


def direct_cast(
    model_name_or_path: str,
    output_path: Path,
    fmt_name: Literal["int8", "s3d8"],
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cpu",
) -> None:
    model = load_model(model_name_or_path, dtype=dtype, device=device)
    QT.convert(
        model,
        fmt_spec=format_for(fmt_name),
        scaling_mode="dynamic",
        clip_gradient=False,
        error_weight=None,
        activation_fmt=None,
        mode="qat",
        progress=True,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    safetensors.torch.save_file(checkpoint_state(model, dtype), output_path)


def _run() -> None:
    parser = argparse.ArgumentParser(
        description="Direct-cast a Hugging Face Llama model to an INT8 or S3D8 checkpoint"
    )
    parser.add_argument(
        "model_name_or_path",
        type=str,
        help="Hugging Face model name or path",
    )
    parser.add_argument(
        "output_path",
        type=Path,
        help="Output safetensors checkpoint path",
    )
    parser.add_argument(
        "--format",
        choices=("int8", "s3d8"),
        required=True,
        help="Direct-cast quantisation format",
    )
    parser.add_argument(
        "--dtype",
        choices=tuple(DTYPES),
        default="bfloat16",
        help="Floating-point dtype for non-quantised tensors and checkpoint metadata",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device used while converting, e.g. 'cpu' or 'cuda'",
    )
    args = parser.parse_args()
    direct_cast(
        args.model_name_or_path,
        args.output_path,
        args.format,
        dtype=DTYPES[args.dtype],
        device=args.device,
    )


if __name__ == "__main__":
    _run()
