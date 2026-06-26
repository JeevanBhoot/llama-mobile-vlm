# Copyright (c) 2026 Graphcore Ltd. All rights reserved.

import pytest
import torch
import weight_formats.quantisation as Q

from gptq.algorithm import GPTQConfig, GPTQLinearQuantizer, S3D8Quantizer


def test_cholesky_failure_raises_when_damp_cannot_increment() -> None:
    linear = torch.nn.Linear(2, 2, bias=False)
    quantizer = GPTQLinearQuantizer(
        linear,
        GPTQConfig(
            bits=4,
            group_size=-1,
            act_group_aware=False,
            damp_percent=0.01,
            damp_auto_increment=0.0,
        ),
    )
    quantizer.nsamples = 1
    quantizer.H = torch.tensor([[1.0, 2.0], [2.0, 1.0]])

    with pytest.raises(torch._C._LinAlgError):
        quantizer.quantize()


def test_s3d8_quantizer_handles_rows_not_divisible_by_three() -> None:
    torch.manual_seed(625464)
    weight = torch.randn(5, 7)
    quantizer = S3D8Quantizer()

    quantizer.find_params(weight)
    quantized = quantizer.quantize(weight)

    assert quantized.shape == weight.shape
    assert quantizer.scale.shape == (5, 1)
    assert quantizer.centroid_shape == (32, 3)
    assert quantizer.fmt is not None
    assert isinstance(quantizer.fmt.element_format, Q.Sign3D8Format)


def test_s3d8_quantizer_uses_stored_row_scale_for_columns() -> None:
    torch.manual_seed(625464)
    weight = torch.randn(5, 7)
    quantizer = S3D8Quantizer()
    quantizer.find_params(weight)

    column = weight[:, 3]
    actual = quantizer.quantize(column)
    assert quantizer.fmt is not None
    scale = quantizer.scale
    expected = (
        quantizer.fmt.element_format.quantise(Q.safe_div(column.unsqueeze(1), scale))
        * scale
    ).flatten()

    torch.testing.assert_close(actual, expected)


def test_s3d8_gptq_has_no_nans_for_dead_hessian_columns() -> None:
    torch.manual_seed(625464)
    linear = torch.nn.Linear(8, 5, bias=False)
    calibration = torch.randn(2, 5, 8)
    calibration[..., 3] = 0
    quantizer = GPTQLinearQuantizer(
        linear,
        GPTQConfig(
            quantization_format="s3d8",
            group_size=4,
            blocksize=4,
        ),
    )
    quantizer.add_batch(calibration)

    result = quantizer.quantize()

    assert result.quantization_format == "s3d8"
    assert result.weight.shape == linear.weight.shape
    assert not torch.isnan(result.weight).any()
    assert result.zero.numel() == 0
    assert result.g_idx.numel() == 0
    assert result.centroid_shape == (32, 3)
