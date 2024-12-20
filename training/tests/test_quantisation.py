# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import dataclasses
import json
from collections import OrderedDict
from typing import cast, Any

import pytest
import torch
import torch.nn.functional as F
from torch import nn, tensor

from .. import quantisation as Q


def _json_roundtrip(s: Any) -> Any:
    return type(s)(**json.loads(json.dumps(dataclasses.asdict(s))))


def test_int_format() -> None:
    fmt = Q.parse("E0M7")
    assert fmt.bits == 8
    assert fmt.max_absolute_value == 127
    assert set(fmt.quantise(torch.linspace(-150, 150, steps=1000)).tolist()) == set(
        range(-127, 128)
    )
    assert set(
        Q.parse("E0M3").quantise(torch.linspace(-10, 10, steps=100)).tolist()
    ) == set(range(-7, 8))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fp_format(dtype: torch.dtype) -> None:
    e2m1 = cast(Q.FPFormat, Q.parse("E2M1"))
    assert e2m1.bits == 4
    assert e2m1.max_absolute_value == 3
    assert e2m1.min_absolute_normal == 0.5
    assert e2m1.min_absolute_subnormal == 0.25
    for fmt, limit, steps, expected in [
        (Q.parse("E2M1"), 4, 100, {0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3}),
        (Q.parse("E3M0"), 10, 1000, {0, 0.125, 0.25, 0.5, 1, 2, 4, 8}),
        (Q.parse("E2M0"), 10, 1000, {0, 0.5, 1, 2}),
    ]:
        x = torch.linspace(-limit, limit, steps=steps, dtype=dtype)
        y = fmt.quantise(x)
        assert y.dtype == dtype
        assert set(y.tolist()) == {n for absn in expected for n in [-absn, absn]}


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fp_format_rounding(dtype: torch.dtype) -> None:
    eps = torch.finfo(dtype).eps
    x = tensor([1.5 - eps, 1.5, 1.5 + eps])
    assert torch.equal(Q.parse("E2M1-RN").quantise(x), torch.tensor([1.5, 1.5, 1.5]))
    assert torch.equal(Q.parse("E2M1-RI").quantise(x), torch.tensor([1.5, 1.5, 2.0]))
    assert torch.equal(Q.parse("E2M1-RZ").quantise(x), torch.tensor([1.0, 1.5, 1.5]))


def test_torch_format() -> None:
    assert Q.FP16.max_absolute_value == 65504
    assert torch.equal(
        Q.FP16.quantise(tensor([2**-25 * 0.99, 2**-25 * 1.01, -1, 1e5])),
        tensor([0, 2**-24, -1, 65504.0]),
    )


def test_exp_ceil_format() -> None:
    fmt = Q.parse("EXP8")
    amax = 2 ** (2**7)
    assert fmt.max_absolute_value == amax
    # Always round up
    assert torch.equal(
        fmt.quantise(tensor([1.01 / amax, 2.000001, amax * 1.01], dtype=torch.float64)),
        tensor([2 / amax, 4, fmt.max_absolute_value], dtype=torch.float64),
    )


def test_lut_format() -> None:
    fmt = Q.LUTFormat((-1, -0.125, 0.125, 1), "fours")
    assert str(fmt) == "LUT2[fours]"
    assert fmt.bits == 2
    assert fmt.max_absolute_value == 1
    assert torch.equal(
        fmt.quantise(tensor([0.8, 0.6, -0.001, -1.2])),
        tensor([1, 1, -0.125, -1]),
    )


def test_scalar_formats() -> None:
    for fmt in [
        Q.FP16,
        Q.FP32,
        Q.NF4,
        Q.nf_approx(5),
        Q.parse("E0M3"),
        Q.parse("E2M2"),
        Q.ExpCeilFormat(3),
    ]:
        assert 0 < fmt.max_absolute_value
        assert 1 <= fmt.bits <= 32
        assert 600 <= fmt.count_bits((20, 30))

        x = torch.linspace(-20, 20, steps=100).view(2, 1, 50)
        if isinstance(fmt, Q.ExpCeilFormat):
            x.abs_()
        qx = fmt.quantise(x)
        assert qx.shape == x.shape
        assert torch.all(qx <= fmt.max_absolute_value)

        assert _json_roundtrip(fmt) == fmt


# Tensor scaling formats


def test_linear_scaling_format() -> None:
    torch.manual_seed(23875)
    tensor = torch.randn((10, 20))
    e3m4 = Q.parse("E3M4")

    # 1. Per-tensor scaling
    per_tensor = Q.tensor_scaling_format(e3m4)
    assert per_tensor.count_bits(tensor.shape) == 8 * 200 + 16
    per_tensor_mse = F.mse_loss(per_tensor.quantise(tensor), tensor)

    # 2. Per-output channel scaling
    per_output_channel = Q.channel_scaling_format(e3m4, per="output")
    assert per_output_channel.count_bits(tensor.shape) == 8 * 200 + 16 * 10
    per_output_channel_mse = F.mse_loss(per_output_channel.quantise(tensor), tensor)
    assert per_output_channel_mse < per_tensor_mse  # could be ==, but unlikely

    # 3. Input-group scaling
    input_group = Q.group_scaling_format(e3m4, grouping="input", group_size=5)
    assert input_group.count_bits(tensor.shape) == 8 * 200 + 16 * 10 * (20 // 5)
    input_group_mse = F.mse_loss(input_group.quantise(tensor), tensor)
    assert input_group_mse < per_output_channel_mse  # could be ==, but unlikely

    # 4. Others
    for format_ in [
        Q.channel_scaling_format(e3m4, per="output"),
        Q.channel_scaling_format(e3m4, per="inout-prod"),
        Q.channel_scaling_format(e3m4, per="inout-min"),
        Q.channel_scaling_format(
            e3m4, per="input", scale_format=Q.tensor_scaling_format(e3m4)
        ),
    ]:
        quantised = format_.quantise(tensor)
        assert quantised.shape == tensor.shape
        assert torch.all(~torch.isnan(quantised))
        assert format_.count_bits(tensor.shape) > 8 * tensor.nelement()


# Model parameters


def test_quantise_model() -> None:
    m = nn.Sequential(
        OrderedDict([("up", nn.Linear(10, 20)), ("down", nn.Linear(20, 10))])
    )
    assert len(set(m.up.weight.flatten().tolist())) > 16
    rules = [
        Q.ParameterRule("up", (None,), Q.BFLOAT16),
        Q.ParameterRule("weight", None, Q.tensor_scaling_format(Q.parse("E2M1"))),
    ]
    nbytes = Q.quantise_model(m, rules)
    assert nbytes == pytest.approx(
        10 * 20 * (0.5 + 0.5)  # weights
        + (2 + 2)  # weight scales
        + (20 * 2 + 10 * 4)  # biases
    )
    assert len(set(m.up.weight.flatten().tolist())) < 16
    assert str(m.up.weight.quantisation_format) == "E2M1{*.*:BFLOAT16}"
    assert str(m.down.weight.quantisation_format) == "E2M1{*.*:BFLOAT16}"
    assert str(m.up.bias.quantisation_format) == "BFLOAT16"
    assert str(m.down.bias.quantisation_format) == "FLOAT32"
