# Copyright (c) 2026 Graphcore Ltd. All rights reserved.

"""Shared quantisation formats and checkpoint helpers."""

from typing import Literal

import torch
import weight_formats.fit as F
import weight_formats.quantisation as Q
import weight_formats.quantisation_training as QT

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


def format_for(name: Literal["int8", "s3d8"]) -> Q.TensorFormat | F.Scaled:
    if name == "int8":
        return FMT_CHANNEL_INT8
    if name == "s3d8":
        return FMT_CHANNEL_S3D8
    raise ValueError(f"Unsupported format {name!r}")


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
