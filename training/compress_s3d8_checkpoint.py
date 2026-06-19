# Copyright (c) 2026 Graphcore Ltd. All rights reserved.

"""Convert a QAT S3D8 training checkpoint into a packed inference checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from safetensors import safe_open
import safetensors.torch
import torch
import weight_formats.quantisation as Q
from weight_formats.nearest_neighbour import nearest_neighbour_torch

PACKED_S3D8_FORMAT = "llama-mobile-s3d8-packed-v1"
FORMAT_METADATA_KEY = "llama_mobile_format"


def safe_div(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a / torch.where(b == 0, 1, b)


def squeeze_odd_dims(tensor: torch.Tensor) -> torch.Tensor:
    assert all(dim == 1 for dim in tensor.shape[1::2])
    return tensor.reshape(tensor.shape[::2])


def pack_s3d8_weight(
    master: torch.Tensor,
    centroids: torch.Tensor,
    fmt: Q.LinearScalingFormat,
    max_nearest_neighbour_bytes: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if master.ndim != 2:
        raise ValueError(f"S3D8 packing only supports rank-2 weights, got {tuple(master.shape)}")
    if fmt.block_shape != (1, None):
        raise ValueError(f"S3D8 packing only supports block_shape=(1, None), got {fmt.block_shape}")
    if fmt.scaling != "absmax":
        raise ValueError(f"S3D8 packing only supports absmax scaling, got {fmt.scaling!r}")

    scale = Q.blocked_scale(
        master.reshape(Q.blocked_shape(master, fmt.block_shape)),
        fmt.scaling,
        fmt.element_format.range,
    )
    scale = fmt.scale_format.quantise(scale)
    tensor = safe_div(master.reshape(Q.blocked_shape(master, fmt.block_shape)), scale).reshape(master.shape)

    d_out, d_in = tensor.shape
    tensor = (
        torch.nn.functional.pad(tensor, (0, -(d_in % -16), 0, -(d_out % -3)))
        .unflatten(0, (-1, 3))
        .permute(0, 2, 1)
        .flatten(end_dim=1)
    )
    table = centroids.to(torch.int8)
    idx = nearest_neighbour_torch(
        tensor.abs().to(torch.float32),
        centroids.to(torch.float32),
        max_bytes=max_nearest_neighbour_bytes,
    )
    sign = tensor.lt(0)
    packed = (
        (idx << 1)
        | sign[:, 0].to(idx.dtype)
        | (sign[:, 1].to(idx.dtype) << 6)
        | ((sign[:, 0] ^ sign[:, 2]).to(idx.dtype) << 7)
    ).to(torch.uint8)
    return (
        packed.cpu().contiguous(),
        squeeze_odd_dims(scale).to(torch.bfloat16).cpu().contiguous(),
        table.cpu().contiguous(),
    )


def _quantisation_meta(meta_tensor: torch.Tensor) -> dict[str, object]:
    import json

    return json.loads(meta_tensor.numpy().tobytes())


def convert_checkpoint(
    input_path: Path,
    output_path: Path,
    device: str = "cpu",
    max_nearest_neighbour_bytes: int = 512 * 1024 * 1024,
) -> None:
    output: dict[str, torch.Tensor] = {}
    with safe_open(input_path, framework="pt", device="cpu") as checkpoint:
        keys = list(checkpoint.keys())
        meta = _quantisation_meta(checkpoint.get_tensor("_quantisation_meta"))
        if meta.get("scaling_mode") != "dynamic":
            raise ValueError(f"Only dynamic S3D8 checkpoints are supported, got {meta.get('scaling_mode')!r}")

        fmt_by_weight = {
            name: Q.TensorFormat.load(fmt_spec)
            for name, fmt_spec in meta["fmt"].items()
        }
        for i, key in enumerate(keys, 1):
            if key == "_quantisation_meta" or key.endswith(".centroids"):
                continue
            if key.endswith(".master"):
                weight_name = key.removesuffix(".master")
                fmt = fmt_by_weight.get(weight_name)
                if not (
                    isinstance(fmt, Q.LinearScalingFormat)
                    and isinstance(fmt.element_format, Q.Sign3D8Format)
                ):
                    raise ValueError(f"Unsupported quantised format for {weight_name}: {fmt}")
                print(f"[{i}/{len(keys)}] packing {weight_name}", file=sys.stderr)
                master = checkpoint.get_tensor(key).to(device)
                centroids = checkpoint.get_tensor(f"{weight_name}.centroids").to(device)
                packed, scale, table = pack_s3d8_weight(
                    master,
                    centroids,
                    fmt,
                    max_nearest_neighbour_bytes=max_nearest_neighbour_bytes,
                )
                output[f"{weight_name}.packed"] = packed
                output[f"{weight_name}.scale"] = scale
                output[f"{weight_name}.table"] = table
                continue

            output[key] = checkpoint.get_tensor(key).cpu().contiguous()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    safetensors.torch.save_file(
        output,
        output_path,
        metadata={FORMAT_METADATA_KEY: PACKED_S3D8_FORMAT},
    )


def _run() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a QAT S3D8 training safetensors checkpoint to packed S3D8 safetensors"
    )
    parser.add_argument("input_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--device", default="cpu", help="Device used while packing, e.g. 'cpu' or 'cuda'")
    parser.add_argument(
        "--max-nearest-neighbour-bytes",
        type=int,
        default=512 * 1024 * 1024,
        help="Approximate maximum temporary bytes for nearest-neighbour chunks",
    )
    args = parser.parse_args()
    convert_checkpoint(
        args.input_path,
        args.output_path,
        device=args.device,
        max_nearest_neighbour_bytes=args.max_nearest_neighbour_bytes,
    )


if __name__ == "__main__":
    _run()
