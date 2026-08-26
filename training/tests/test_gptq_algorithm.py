# Copyright (c) 2026 Graphcore Ltd. All rights reserved.

import math

import pytest
import torch
import weight_formats.quantisation as Q

from gptq.algorithm import (
    AbsmaxCodebookIntQuantizer,
    AffineCodebookIntQuantizer,
    GPTQConfig,
    GPTQLinearQuantizer,
    S3D8Quantizer,
    UniformAffineQuantizer,
)


def test_gptq_reduces_error() -> None:
    torch.manual_seed(6)
    linear = torch.nn.Linear(8, 5, bias=False)
    calibration = torch.randn(4, 6, 8)

    gptq = GPTQLinearQuantizer(
        linear,
        GPTQConfig(
            bits=2,
            group_size=-1,
            blocksize=8,
            act_group_aware=False,
        ),
    )
    gptq.add_batch(calibration)
    gptq_weight = gptq.quantize().weight

    naive = UniformAffineQuantizer(bits=2, sym=True)
    naive.find_params(linear.weight)
    naive_weight = naive.quantize(linear.weight)

    reference = linear(calibration)
    gptq_error = torch.mean(
        (torch.nn.functional.linear(calibration, gptq_weight) - reference) ** 2
    )
    naive_error = torch.mean(
        (torch.nn.functional.linear(calibration, naive_weight) - reference) ** 2
    )

    assert gptq_error < naive_error


def test_k6_absmax() -> None:
    quantizer = AbsmaxCodebookIntQuantizer(codepoints=6)
    weight = torch.tensor([[-6.0, 4.0, 0.0]])

    quantizer.find_params(weight)

    expected_scale = Q.BFLOAT16.quantise(torch.tensor([3.0])).reshape(1, 1)
    torch.testing.assert_close(quantizer.scale, expected_scale)
    torch.testing.assert_close(
        quantizer.quantize(weight),
        torch.tensor([[-6.0, 3.0, 0.0]]),
    )
    assert quantizer.effective_bits == pytest.approx(math.log2(6))


def test_k6_affine_params() -> None:
    quantizer = AffineCodebookIntQuantizer(
        codepoints=6,
        sym=True,
        scale_zero_dtype="bfloat16",
    )

    quantizer.find_params(torch.tensor([[-2.0, -0.5, 0.0, 1.0, 2.0]]))

    expected_scale = torch.tensor([[0.8]]).to(torch.bfloat16).to(torch.float32)
    torch.testing.assert_close(quantizer.scale, expected_scale, rtol=0, atol=0)
    torch.testing.assert_close(quantizer.zero, torch.tensor([[3.0]]))
    assert quantizer.scale.item() != 0.8


def test_power_of_two_codebook_matches_int() -> None:
    torch.manual_seed(625464)
    linear = torch.nn.Linear(8, 5, bias=False)
    calibration = torch.randn(2, 5, 8)
    ordinary = GPTQLinearQuantizer(
        linear,
        GPTQConfig(bits=3, group_size=4, blocksize=4),
    )
    codebook = GPTQLinearQuantizer(
        linear,
        GPTQConfig(
            quantization_format="int-codebook-affine",
            bits=None,
            codepoints=8,
            group_size=4,
            blocksize=4,
        ),
    )
    ordinary.add_batch(calibration)
    codebook.add_batch(calibration)

    expected = ordinary.quantize()
    actual = codebook.quantize()

    torch.testing.assert_close(actual.weight, expected.weight)
    torch.testing.assert_close(actual.scale, expected.scale)
    torch.testing.assert_close(actual.zero, expected.zero)
    torch.testing.assert_close(actual.g_idx, expected.g_idx)


def test_dead_columns_are_finite() -> None:
    torch.manual_seed(625464)
    linear = torch.nn.Linear(8, 5, bias=False)
    calibration = torch.randn(2, 5, 8)
    calibration[..., 3] = 0
    quantizer = GPTQLinearQuantizer(
        linear,
        GPTQConfig(
            quantization_format="int-codebook",
            codepoints=6,
            group_size=4,
            blocksize=4,
        ),
    )
    quantizer.add_batch(calibration)

    result = quantizer.quantize()

    assert torch.isfinite(result.weight).all()


def test_s3d8_reuses_fitted_params() -> None:
    torch.manual_seed(625464)
    weight = torch.randn(5, 20)
    quantizer = S3D8Quantizer()
    quantizer.find_params(weight)
    assert quantizer.fmt is not None
    scale = quantizer.scale.clone()
    centroids = torch.tensor(quantizer.fmt.element_format.lut.values)

    quantized = quantizer.quantize(weight)
    quantized_column = quantizer.quantize(weight[:, 3])

    assert quantized.shape == weight.shape
    assert quantizer.centroid_shape == (32, 3)
    torch.testing.assert_close(quantized_column, quantized[:, 3])
    torch.testing.assert_close(quantizer.scale, scale)
    torch.testing.assert_close(
        torch.tensor(quantizer.fmt.element_format.lut.values),
        centroids,
    )
