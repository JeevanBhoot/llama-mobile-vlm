from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.autograd.function import FunctionCtx

from utility import convert_module

from .quantisation import TensorFormat


class STEQuantise(torch.autograd.Function):
    @staticmethod
    def forward(ctx: FunctionCtx, inp: Tensor, format: TensorFormat) -> Tensor:
        return format.quantise(inp)

    @staticmethod
    def backward(ctx: FunctionCtx, grad_out: Tensor) -> Tuple[Optional[Tensor]]:
        return grad_out, None


class STELinear(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: FunctionCtx,
        inp: Tensor,
        weight: Tensor,
        bias: Optional[Tensor],
        format: TensorFormat,
    ) -> Tensor:
        ctx.save_for_backward(inp, weight)
        ctx.format = format
        out = inp @ format.quantise(weight).T
        if bias is not None:
            out += bias
        return out

    @staticmethod
    def backward(ctx: FunctionCtx, grad_out: Tensor) -> Tuple[Optional[Tensor]]:
        inp, weight = ctx.saved_tensors
        grad_inp = grad_weight = grad_bias = None

        shape = inp.shape
        inp, grad_out = inp.flatten(end_dim=-2), grad_out.flatten(end_dim=-2)
        if ctx.needs_input_grad[0]:
            grad_inp = (grad_out @ ctx.format.quantise(weight)).reshape(shape)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_out.T @ inp
        if ctx.needs_input_grad[2]:
            grad_bias = grad_out.sum(dim=0)
        return grad_inp, grad_weight, grad_bias, None


ste_quantise = STEQuantise.apply


class QuantLinear(torch.nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight_fmt: TensorFormat,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self.weight_fmt = weight_fmt

    def forward(self, inp: Tensor) -> Tensor:
        return STELinear.apply(inp, self.weight, self.bias, self.weight_fmt)
        # return F.linear(inp, ste_quantise(self.weight, self.weight_fmt), self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, weight_format={self.weight_fmt}"
        )


def quantise_linear_layers(
    model: torch.nn.Module, weight_fmt: TensorFormat
) -> torch.nn.Module:
    def _replace(m: torch.nn.Module) -> Optional[torch.nn.Module]:
        if isinstance(m, torch.nn.Linear):
            return QuantLinear(
                in_features=m.in_features,
                out_features=m.out_features,
                weight_fmt=weight_fmt,
                bias=m.bias is not None,
            )
        else:
            return None

    model = convert_module(model, _replace)
    return model
