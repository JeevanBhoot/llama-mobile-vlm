import copy
import unittest.mock as um
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Optional, TypeVar

import safetensors.torch
import torch
from torch import Tensor, nn
from transformers import MllamaVisionModel, PreTrainedTokenizerBase

T = TypeVar("T")


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


# NOTE: Copied from weight_formats.experiments.qat to avoid extra imports


class AttrDict(dict):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.__dict__ = self


class XEnt_Destructive(torch.autograd.Function):
    """A somewhat dangerous in-place Cross-Entropy, optimised for memory-efficiency.

    Both forward and backward passes mutate `input_logits` in-place. It is unsafe for
    `input_logits` to be consumed by any other operation.
    """

    @staticmethod
    def forward(ctx, input_logits, target_p, mask):
        input_logp = input_logits.sub_(torch.logsumexp(input_logits, -1, keepdim=True))
        del input_logits
        ctx.save_for_backward(input_logp, target_p, mask)
        return torch.dot(target_p.flatten(), input_logp.flatten()).neg()

    @staticmethod
    def backward(ctx, grad_output):
        input_logp, target_p, mask = ctx.saved_tensors
        grad_input_logits = (
            input_logp.exp_().sub_(target_p).mul_(mask).mul_(grad_output)
        )
        del input_logp
        return grad_input_logits, None, None


def compute_kl_loss(
    model: nn.Module, reference_model: nn.Module, batch: dict[str, Tensor]
) -> Tensor:
    """Computes the KL divergence between the output of two models, with care for memory."""
    with torch.no_grad():
        reference_logp = torch.log_softmax(
            reference_model(**batch, use_cache=False).logits, -1
        )
        reference_p = reference_logp.exp().mul_(batch["attention_mask"].unsqueeze(-1))
        # Calculate reference entropy here, so we can free up `reference_logp`
        reference_ent = torch.dot(reference_p.flatten(), reference_logp.flatten()).neg()
        del reference_logp

    xent = XEnt_Destructive.apply(
        model(**batch, use_cache=False).logits,
        reference_p,
        batch["attention_mask"].unsqueeze(-1),
    )
    return xent - reference_ent


def save_model(model: nn.Module, path: Path | None, dtype: torch.dtype) -> None:
    """Save a model to a local .safetensors file.

    Note that this requires enough free memory to hold the whole model on one shard.
    """
    with torch.no_grad():
        unsharded_tensors = {}
        state_dict = model.state_dict()
        for key in state_dict:
            tensor = state_dict[key].to(dtype)
            if isinstance(tensor, torch.distributed.tensor.DTensor):
                tensor = tensor.full_tensor()
            unsharded_tensors[key.replace("._orig_mod", "")] = tensor
        if not torch.distributed.is_initialized() or (torch.distributed.get_rank() == 0):
            path.parent.mkdir(parents=True, exist_ok=True)
            safetensors.torch.save_file(unsharded_tensors, path)
