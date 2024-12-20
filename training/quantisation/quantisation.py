# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

"""Utilities for "fake quantisation"."""

import math
import re
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence, Tuple, Union, cast, Literal

import torch
from torch import Tensor, nn

Shape = Tuple[int, ...]


class TensorFormat:
    """Quantisation formats for tensors."""

    def quantise(self, tensor: Tensor) -> Tensor:
        raise NotImplementedError

    def count_bits(self, shape: Shape) -> int:
        raise NotImplementedError


# Scalar formats


@dataclass
class ScalarFormat(TensorFormat):
    """Elementwise scalar formats (abstract base class).

    Subclasses define: `_type`, `__str__`, `bits`, `max_absolute_value`, `quantise`
    """

    def __str__(self) -> str:
        raise NotImplementedError

    @property
    def bits(self) -> int:
        raise NotImplementedError

    @property
    def max_absolute_value(self) -> float:
        raise NotImplementedError

    def count_bits(self, shape: Shape) -> int:
        return self.bits * math.prod(shape)


@dataclass
class FPFormat(ScalarFormat):
    """Note that this format does not reserve an exponent code for specials.

    For exponent e : [0, 2^E - 1], mantissa m : [0, 2^M - 1], the represented value is:

        2^(e - 2^(E-1))) * (1 + m / 2^M)   if e != 0  (normal)
        2^(1 - 2^(E-1))) * (m / 2^M)       if e == 0  (subnormal)
    """

    exponent_bits: int
    mantissa_bits: int
    rounding: Literal["nearest", "to_inf", "to_zero"]
    _type: str = "fp"

    def __post_init__(self) -> None:
        assert self.exponent_bits >= 2, "FPFormat requires at least 2 exponent bits"

    def __str__(self) -> str:
        return f"E{self.exponent_bits}M{self.mantissa_bits}"

    @property
    def bits(self) -> int:
        return 1 + self.exponent_bits + self.mantissa_bits

    @property
    def max_absolute_value(self) -> float:
        max_exponent = 2 ** (self.exponent_bits - 1) - 1
        return cast(float, 2**max_exponent * (2 - 2**-self.mantissa_bits))

    @property
    def min_absolute_normal(self) -> float:
        min_exponent = 1 - 2 ** (self.exponent_bits - 1)
        return cast(float, 2**min_exponent)

    @property
    def min_absolute_subnormal(self) -> float:
        return self.min_absolute_normal * 2.0**-self.mantissa_bits

    def quantise(self, x: Tensor) -> Tensor:
        assert x.dtype in [
            torch.float32,
            torch.bfloat16,
        ], "Quantising is only supported from bfloat16 and float32"

        absmax = self.max_absolute_value
        downscale = 2.0 ** (127 - 2 ** (self.exponent_bits - 1))
        m_bits_before = {torch.float32: 23, torch.bfloat16: 7}[x.dtype]
        int_dtype = {torch.float32: torch.int32, torch.bfloat16: torch.int16}[x.dtype]
        mask = (
            2 ** (m_bits_before - self.mantissa_bits) - 1
            if m_bits_before > self.mantissa_bits
            else 0
        )
        if self.rounding == "nearest":
            offset = mask >> 1
        elif self.rounding == "to_inf":
            offset = mask
        elif self.rounding == "to_zero":
            offset = 0

        q = torch.clip(x, -float(absmax), float(absmax))
        q /= downscale
        q = ((q.view(int_dtype) + offset) & ~mask).view(x.dtype)
        q *= downscale

        return q.to(x.dtype)


@dataclass
class FPFormat_NoSign(FPFormat):
    _type: str = "fp_nosign"

    def __str__(self) -> str:
        return f"{super().__str__()}+"

    @property
    def bits(self) -> int:
        return self.exponent_bits + self.mantissa_bits

    def quantise(self, x: Tensor) -> Tensor:
        return super().quantise(x.clamp_min(0))


@dataclass
class TorchFormat(ScalarFormat):
    dtype: str
    _type: str = "torch"

    def __init__(self, dtype: Union[torch.dtype, str], _type: str = "torch"):
        assert _type == "torch"
        self.dtype = str(dtype).replace("torch.", "")

    @property
    def torch_dtype(self) -> torch.dtype:
        return cast(torch.dtype, getattr(torch, self.dtype))

    def quantise(self, x: Tensor) -> Tensor:
        amax = self.max_absolute_value
        return torch.clip(x, -amax, amax).to(self.torch_dtype).to(x.dtype)

    def __str__(self) -> str:
        return str(self.torch_dtype).replace("torch.", "").upper()

    @property
    def bits(self) -> int:
        torch_dtype = self.torch_dtype
        return (
            torch.finfo(torch_dtype).bits
            if torch_dtype.is_floating_point
            else torch.iinfo(torch_dtype).bits
        )

    @property
    def max_absolute_value(self) -> float:
        torch_dtype = self.torch_dtype
        return (
            torch.finfo(torch_dtype).max
            if torch_dtype.is_floating_point
            else torch.iinfo(torch_dtype).max
        )

    def count_bits(self, shape: Shape) -> int:
        return self.bits * math.prod(shape)


@dataclass
class IntFormat(ScalarFormat):
    bits_: int
    _type: str = "int"

    def __str__(self) -> str:
        return f"E0M{self.bits_ - 1}"

    @property
    def bits(self) -> int:
        return self.bits_

    @property
    def max_absolute_value(self) -> float:
        return 2.0 ** (self.bits_ - 1) - 1

    def quantise(self, x: Tensor) -> Tensor:
        return torch.clip(
            torch.round(x), -self.max_absolute_value, self.max_absolute_value
        )


@dataclass
class ExpCeilFormat(ScalarFormat):
    """An exponent-only format for positive numbers, with no zero."""

    bits_: int
    _type: str = "exp"

    def __str__(self) -> str:
        return f"EXP{self.bits_}"

    @property
    def bits(self) -> int:
        return self.bits_

    @property
    def max_absolute_value(self) -> float:
        return cast(float, 2 ** (2**self.bits_ - 1 - self.exponent_bias))

    @property
    def exponent_bias(self) -> float:
        return 2.0 ** (self.bits_ - 1) - 1

    def quantise(self, x: Tensor) -> Tensor:
        y: Tensor = 2 ** torch.clip(
            torch.ceil(torch.log2(x)),
            -self.exponent_bias,
            2**self.bits_ - 1 - self.exponent_bias,
        )
        return y


@dataclass
class LUTFormat(ScalarFormat):
    values: Tuple[float, ...]
    name: str
    _type: str = "lut"

    @classmethod
    def create(cls, values: Tensor, name: str) -> "LUTFormat":
        return cls(values=tuple(values.tolist()), name=name)

    def __post_init__(self) -> None:
        self.values = tuple(self.values)
        n = len(self.values)
        assert 2 ** int(math.log2(n)) == n, "table size must be a power of 2"

    def __str__(self) -> str:
        return f"LUT{self.bits}[{self.name}]"

    @property
    def bits(self) -> int:
        return int(math.ceil(math.log2(len(self.values))))

    @property
    def max_absolute_value(self) -> float:
        return max(abs(v) for v in self.values)

    def quantise(self, x: Tensor) -> Tensor:
        values = torch.tensor(self.values, dtype=x.dtype)
        return values[(x[..., None] - values).abs().argmin(-1)]


def parse(value: str) -> ScalarFormat:
    if value == "FP32":
        return FP32
    if value == "FP16":
        return FP16
    if value == "BFLOAT16":
        return BFLOAT16
    m = re.match(r"^E(\d+)M(\d+)(-(RN|RZ|RI))?$", value)
    if m:
        exponent_bits = int(m.group(1))
        mantissa_bits = int(m.group(2))
        if exponent_bits == 0:
            assert not m.group(3)
            return IntFormat(1 + mantissa_bits)
        if exponent_bits >= 2:
            rounding = {
                None: "nearest",
                "-RN": "nearest",
                "-RZ": "to_zero",
                "-RI": "to_inf",
            }[m.group(3)]
            return FPFormat(exponent_bits, mantissa_bits, rounding)
        raise ValueError(f"No format {value!r} available (note: E1M6 == E0M7)")
    m = re.match(r"EXP(\d+)", value)
    if m:
        return ExpCeilFormat(int(m.group(1)))
    raise ValueError(f"Couldn't parse {value!r}")


def lut_function(fn: Callable[[Tensor], Tensor], bits: int, name: str) -> LUTFormat:
    """A lookup table quantiser based on mapping [-1, 1] via a function"""
    return LUTFormat.create(fn(torch.linspace(-1, 1, steps=2**bits)), name)


def nf_approx(bits: int) -> LUTFormat:
    return lut_function(
        lambda n: cast(Tensor, (n + n**3) / 2), bits=bits, name="NF-approx"
    )


FP32 = TorchFormat(torch.float32)
FP16 = TorchFormat(torch.float16)
BFLOAT16 = TorchFormat(torch.bfloat16)
# See: QLoRA [https://arxiv.org/abs/2305.14314]
NF4 = LUTFormat(
    (
        -1.0,
        -0.6961928009986877,
        -0.5250730514526367,
        -0.39491748809814453,
        -0.28444138169288635,
        -0.18477343022823334,
        -0.09105003625154495,
        0.0,
        0.07958029955625534,
        0.16093020141124725,
        0.24611230194568634,
        0.33791524171829224,
        0.44070982933044434,
        0.5626170039176941,
        0.7229568362236023,
        1.0,
    ),
    "NF",
)


# Tensor formats


@dataclass
class LinearScalingFormat(TensorFormat):
    """Specifies a scheme for quantising tensors.

    group_shapes -- list of groupings (size of groups in each dimension)
                    e.g. [(1, 8)]                input-groups of size 8
                         [(2, 2)]                square groups of 2x2 (4 elements)
                         [(1, None)]             per-output-channel scaling
                         [(None, 1), (None, 1)]  sqrt(input*output)-channel scaling
    """

    GroupShape = Tuple[Optional[int], ...]

    element_format: ScalarFormat
    group_shapes: Sequence[GroupShape]
    scale_format: TensorFormat
    scale_combiner: Optional[str]

    _type: str = "linear"

    def __str__(self) -> str:
        group = ":".join(
            ".".join("*" if g is None else str(g) for g in group_shape)
            for group_shape in self.group_shapes
        )
        return f"{self.element_format}{{{group}:{self.scale_format}}}"

    @staticmethod
    def _group_shape_for(tensor_shape: Shape, group_shape: GroupShape) -> Shape:
        assert len(tensor_shape) == len(group_shape), f"{tensor_shape} vs {group_shape}"
        return tuple((t if g is None else g) for t, g in zip(tensor_shape, group_shape))

    def count_bits(self, shape: Shape) -> int:
        """Count the number of bits in the quantised representation."""
        count = self.element_format.count_bits(shape)
        for group_shape in self.group_shapes:
            count += self.scale_format.count_bits(
                tuple(
                    t // g
                    for t, g in zip(shape, self._group_shape_for(shape, group_shape))
                )
            )
        return count

    @staticmethod
    def _group_scale_for(absratio: Tensor, group_shape: Shape) -> Tensor:
        full_grouped_shape = tuple(
            s
            for size, group_size in zip(absratio.shape, group_shape)
            for s in [size // group_size, group_size]
        )
        grouped_absratio = absratio.reshape(full_grouped_shape)
        for dim in range(1, len(full_grouped_shape), 2):
            grouped_absratio = grouped_absratio.max(dim=dim, keepdim=True).values
        return grouped_absratio.broadcast_to(full_grouped_shape).reshape(absratio.shape)

    def scale_for(self, tensor: Tensor) -> Tensor:
        """Get the quantised scaling tensor to apply to quantise a given tensor."""
        absratio = tensor.abs() / self.element_format.max_absolute_value
        scales = []
        for group_shape in self.group_shapes:
            scales.append(
                self.scale_format.quantise(
                    self._group_scale_for(
                        absratio, self._group_shape_for(absratio.shape, group_shape)
                    )
                )
            )
        if len(scales) == 1:
            return scales[0]
        if self.scale_combiner == "prod":
            return torch.pow(
                cast(Tensor, math.prod(scales, start=torch.ones_like(absratio))),
                1 / len(scales),
            )
        if self.scale_combiner == "min":
            return torch.minimum(*scales)
        assert False, f"unknown scale_combiner {self.scale_combiner}"

    def quantise(self, tensor: Tensor) -> Tensor:
        """Quantise a tensor under the scheme."""
        scale = self.scale_for(tensor)
        return self.element_format.quantise(tensor / scale) * scale


def tensor_scaling_format(
    element_format: ScalarFormat, scale_format: ScalarFormat = BFLOAT16
) -> LinearScalingFormat:
    """A per-(2D)-tensor scaling format."""
    return LinearScalingFormat(
        element_format, [(None, None)], scale_format, scale_combiner=None
    )


def channel_scaling_format(
    element_format: ScalarFormat, per: str, scale_format: TensorFormat = BFLOAT16
) -> LinearScalingFormat:
    """A per-channel scaling format.

    per -- "input|output|inout-prod|inout-min"
    """
    groups = cast(
        Sequence[LinearScalingFormat.GroupShape],
        {
            "input": [(None, 1)],
            "output": [(1, None)],
            "inout-prod": [(None, 1), (1, None)],
            "inout-min": [(None, 1), (1, None)],
        }[per],
    )
    scale_combiner = {"inout-prod": "prod", "inout-min": "min"}.get(per)
    return LinearScalingFormat(element_format, groups, scale_format, scale_combiner)


def group_scaling_format(
    element_format: ScalarFormat,
    grouping: str,
    group_size: int,
    scale_format: TensorFormat = BFLOAT16,
) -> LinearScalingFormat:
    """A simplified grouped scaling format: only 1D groups & no "inout" product.

    groups -- "input|output"
    """
    return LinearScalingFormat(
        element_format,
        dict(input=[(1, group_size)], output=[(group_size, 1)])[grouping],
        scale_format,
        scale_combiner=None,
    )


# Model parameters


def _match_shape(shape: Shape, pattern: Tuple[Optional[int], ...]) -> bool:
    return (len(shape) == len(pattern)) and all(
        p is None or p == s for s, p in zip(shape, pattern)
    )


@dataclass
class ParameterRule:
    pattern: Optional[str]
    shape: Optional[Tuple[Optional[int], ...]]
    format: TensorFormat

    def match(self, name: str, parameter: Tensor) -> bool:
        if self.pattern is not None and not re.search(self.pattern, name):
            return False
        if self.shape is not None and not _match_shape(parameter.shape, self.shape):
            return False
        return True


def quantise_model(model: nn.Module, rules: Iterable[ParameterRule] = []) -> float:
    """In-place quantise a model, returning the size of the quantised model (bytes).

    rules -- an ordered list or rules. For the first rule which matches `pattern` and
             `shape` (if specified), use the given quantisation format.
             If no rule matches, use a `TorchFormat` with the existing parameter type
             (no quantisation).
    """
    bitcount = 0
    for name, p in model.named_parameters():
        for rule in rules:
            if rule.match(name, p):
                format_ = rule.format
                break
        else:
            format_ = TorchFormat(p.dtype)
        assert not hasattr(p, "quantisation_format"), "double-quantisation not allowed"
        p.data = format_.quantise(p.data)
        p.quantisation_format = format_  # type:ignore[attr-defined]
        bitcount += format_.count_bits(p.data.shape)
    return bitcount / 8


# Exports

__all__ = [
    "TensorFormat",
    "ScalarFormat",
    "FPFormat",
    "TorchFormat",
    "IntFormat",
    "ExpCeilFormat",
    "FPFormat_NoSign",
    "LUTFormat",
    "parse",
    "lut_function",
    "nf_approx",
    "FP32",
    "FP16",
    "BFLOAT16",
    "NF4",
    "LinearScalingFormat",
    "tensor_scaling_format",
    "channel_scaling_format",
    "group_scaling_format",
    "ParameterRule",
    "quantise_model",
]
