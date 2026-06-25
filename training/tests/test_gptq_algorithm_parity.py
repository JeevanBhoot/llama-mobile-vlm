# Copyright (c) 2026 Graphcore Ltd. All rights reserved.

import pytest
import torch

pytestmark = pytest.mark.gptqmodel
pytest.importorskip(
    "gptqmodel",
    reason=(
        "Install GPTQModel with training/gptq/requirements.txt to run GPTQ "
        "parity tests."
    ),
)

from gptqmodel.looper.named_module import NamedModule
from gptqmodel.quantization.config import QuantizeConfig
from gptqmodel.quantization.gptq import GPTQ
from gptqmodel.quantization.quantizer import Quantizer

from gptq.algorithm import GPTQConfig, GPTQLinearQuantizer, UniformAffineQuantizer


def reference_quantizer(bits: int, sym: bool, x: torch.Tensor) -> Quantizer:
    qcfg = QuantizeConfig(bits=bits, sym=sym)
    quantizer = Quantizer(qcfg=qcfg)
    quantizer.configure(perchannel=True)
    quantizer.find_params(x, weight=True)
    return quantizer


@pytest.mark.parametrize("sym", [True, False])
def test_uniform_affine_quantizer_matches_gptqmodel(sym) -> None:
    x = torch.tensor(
        [
            [-1.0, -0.25, 0.5, 1.0],
            [-2.0, 0.0, 1.0, 3.0],
        ],
        dtype=torch.float32,
    )

    expected = reference_quantizer(bits=4, sym=sym, x=x)
    actual = UniformAffineQuantizer(bits=4, sym=sym)
    actual.find_params(x)

    torch.testing.assert_close(actual.scale, expected.scale)
    torch.testing.assert_close(actual.zero, expected.zero)
    torch.testing.assert_close(actual.quantize(x), expected.quantize(x))


def run_reference_gptq(
    linear: torch.nn.Linear,
    calibration: torch.Tensor,
    bits: int,
    group_size: int,
    desc_act: bool,
    act_group_aware: bool,
    blocksize: int = 8,
):
    qcfg = QuantizeConfig(
        bits=bits,
        group_size=group_size,
        desc_act=desc_act,
        act_group_aware=act_group_aware,
        sym=True,
        static_groups=False,
        damp_percent=0.05,
        damp_auto_increment=0.01,
    )
    named_module = NamedModule(
        linear,
        name="linear",
        full_name="layers.0.linear",
        layer_index=0,
    )
    gptq = GPTQ(named_module, qcfg=qcfg)
    gptq.quantizer.configure(perchannel=True)
    gptq.add_batch(calibration, None)
    return gptq.quantize(blocksize=blocksize)


def run_local_gptq(
    linear: torch.nn.Linear,
    calibration: torch.Tensor,
    bits: int,
    group_size: int,
    desc_act: bool,
    act_group_aware: bool,
    blocksize: int = 8,
):
    quantizer = GPTQLinearQuantizer(
        linear,
        GPTQConfig(
            bits=bits,
            group_size=group_size,
            desc_act=desc_act,
            act_group_aware=act_group_aware,
            sym=True,
            static_groups=False,
            damp_percent=0.05,
            damp_auto_increment=0.01,
            blocksize=blocksize,
        ),
    )
    quantizer.add_batch(calibration)
    return quantizer.quantize()


@pytest.mark.parametrize("bits", [3, 4])
@pytest.mark.parametrize(
    "group_size,desc_act,act_group_aware",
    [
        (128, False, True),
        (16, False, True),
        (128, False, False),
        (128, True, False),
        (-1, False, False),
        (-1, True, False),
    ],
)
def test_linear_gptq_matches_gptqmodel(
    bits,
    group_size,
    desc_act,
    act_group_aware,
) -> None:
    torch.manual_seed(625464)
    linear = torch.nn.Linear(32, 8, bias=False)
    calibration = torch.randn(3, 7, 32)

    expected_weight, expected_scale, expected_zero, expected_g_idx, *_ = run_reference_gptq(
        linear,
        calibration,
        bits=bits,
        group_size=group_size,
        desc_act=desc_act,
        act_group_aware=act_group_aware,
    )
    actual = run_local_gptq(
        linear,
        calibration,
        bits=bits,
        group_size=group_size,
        desc_act=desc_act,
        act_group_aware=act_group_aware,
    )

    torch.testing.assert_close(actual.weight, expected_weight)
    torch.testing.assert_close(actual.scale, expected_scale)
    torch.testing.assert_close(actual.zero, expected_zero)
    torch.testing.assert_close(actual.g_idx, expected_g_idx)


def test_dead_hessian_columns_match_gptqmodel() -> None:
    torch.manual_seed(625464)
    linear = torch.nn.Linear(8, 4, bias=False)
    calibration = torch.randn(2, 5, 8)
    calibration[..., 3] = 0

    expected_weight, expected_scale, expected_zero, expected_g_idx, *_ = run_reference_gptq(
        linear,
        calibration,
        bits=4,
        group_size=128,
        desc_act=False,
        act_group_aware=True,
    )
    actual = run_local_gptq(
        linear,
        calibration,
        bits=4,
        group_size=128,
        desc_act=False,
        act_group_aware=True,
    )

    torch.testing.assert_close(actual.weight, expected_weight)
    torch.testing.assert_close(actual.scale, expected_scale)
    torch.testing.assert_close(actual.zero, expected_zero)
    torch.testing.assert_close(actual.g_idx, expected_g_idx)
