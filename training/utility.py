import copy
import dataclasses
import subprocess
import tempfile
import typing
import unittest.mock as um
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional, TypeVar

import safetensors.torch
import torch
import weight_formats.quantisation_training as QT
from torch import Tensor, nn
from transformers import MllamaVisionModel, PreTrainedTokenizerBase

T = TypeVar("T")

LOCAL_DATA_PATH = f"{Path(__file__).parent}/data"
S3_DATA_PATH = "s3://graphcore-research/2024-10-squashedllama/data"

LLAMA_PROMPT_TEMPLATES = dict(
    instruct="<|start_header_id|>user<|end_header_id|>"
    "\n\n<|image|>{prompt}<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n",
    simple="<|image|>{prompt}",
)


def batches(
    iterable: Iterable[T], batch_size: int, drop_last: bool = False
) -> Iterable[list[T]]:
    """Chunks `iterable` into batches of consecutive values."""
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch and not drop_last:
        yield batch


def distributed_batches(
    iterable: Iterable[T], batch_size: int, rank: int, world_size: int
) -> Iterable[list[T]]:
    assert batch_size % world_size == 0, "Batch size should be divisible by world size"
    device_batch_size = batch_size // world_size
    for i, device_batch in enumerate(
        batches(iterable, batch_size=device_batch_size, drop_last=True)
    ):
        if i % world_size == rank:
            yield device_batch


def convert_module(
    model: nn.Module, replace: Callable[[nn.Module], Optional[nn.Module]]
) -> nn.Module:
    """Generic recursive module conversion."""

    def _convert(original: nn.Module) -> nn.Module:
        replacement = replace(original)
        if replacement is not None:
            replacement.to(next(original.parameters()).dtype)
            replacement.to(next(original.parameters()).device)
            replacement.load_state_dict(original.state_dict(), strict=False)
            return replacement

        # Recursive (lazy) copy
        result = original
        for name, child in original.named_children():
            replacement = _convert(child)
            if replacement is not child:
                if result is original:
                    result = copy.copy(original)
                    # Copy _modules, otherwise add_module() modifies `original`
                    result._modules = original._modules.copy()
                result.add_module(name, replacement)
        return result

    return _convert(model)


@contextmanager
def set_padding_side_left(tokenizer: PreTrainedTokenizerBase) -> Iterator[None]:
    with um.patch.object(tokenizer, "padding_side", "left"):
        yield


@contextmanager
def record_memory(
    path: str = "out/memory.pickle", max_entries: int = 100_000
) -> Iterable[None]:
    torch.cuda.memory._record_memory_history(max_entries=max_entries)
    try:
        yield
    finally:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.cuda.memory._dump_snapshot(path)
        torch.cuda.memory._record_memory_history(enabled=None)


def merge_params_mllama(m: MllamaVisionModel) -> None:
    """Merge in-place gate params in vision global transformer layers"""

    def merge(layer: torch.nn.Module) -> None:
        if not hasattr(layer, "is_gated"):
            return
        layer.is_gated = False
        with torch.no_grad():
            layer.self_attn.o_proj.weight.mul_(layer.gate_attn.tanh())
            layer.mlp.fc2.weight.mul_(layer.gate_ffn.tanh())
            layer.mlp.fc2.bias.mul_(layer.gate_ffn.tanh())
            del layer.gate_attn, layer.gate_ffn

    for layer in m.global_transformer.layers:
        merge(layer)


def check_s3_access() -> None:
    """Check that we have credentials for AWS S3 access."""
    subprocess.check_call(
        ["aws", "s3", "ls", "s3://graphcore-research"], stdout=subprocess.DEVNULL
    )


def get_unsharded_quantised_params(
    model: nn.Module, dtype: torch.dtype
) -> dict[str, Tensor]:
    """Save model parameters and quantisation metadata on CPU.

    Note that this requires enough free host memory to hold the whole model.
    """
    with torch.no_grad():
        unsharded_tensors = {}
        state_dict = QT.save(model)
        for key in state_dict:
            tensor = state_dict[key]
            if tensor.dtype.is_floating_point:
                tensor = tensor.to(dtype)
            if isinstance(tensor, torch.distributed.tensor.DTensor):
                tensor = tensor.full_tensor()
            unsharded_tensors[key] = tensor.cpu()
    return unsharded_tensors


def save_params_to_s3(params: dict[str, Tensor], s3_path: str | None) -> None:
    """Save model parameters to a .safetensors file and sync to S3.

    s3_path -- the path to save the object to in S3; should be s3://bucket/key...
               (this can be `None` for `rank != 0` when using distributed training)
    """

    if not torch.distributed.is_initialized() or (torch.distributed.get_rank() == 0):
        assert s3_path is not None
        with tempfile.NamedTemporaryFile() as f:
            safetensors.torch.save_file(params, f.name)
            subprocess.check_call(["aws", "s3", "cp", f.name, s3_path])


def from_dict(cls, data):
    """Recursively reconstruct a dataclass from a dict."""
    args = typing.get_args(cls)
    if args:
        dc_args = [a for a in args if dataclasses.is_dataclass(a)]
        cls = dc_args[0] if len(dc_args) == 1 else cls
        if len(dc_args) > 1:
            print(f"Warning: Multiple dataclass types: {dc_args}")

    # Handle the case where tuple was converted to a list
    if isinstance(data, list) and (
        cls is tuple  # tuple
        or typing.get_origin(cls) is tuple  # tuple[int, int]
        or any(
            a is tuple or typing.get_origin(a) is tuple for a in args
        )  # tuple[int, int] | None
    ):
        return tuple(data)

    if not dataclasses.is_dataclass(cls):
        return data
    return cls(
        **{f.name: (from_dict(f.type, data[f.name])) for f in dataclasses.fields(cls)}
    )
