# Copyright (c) 2026 Graphcore Ltd. All rights reserved.

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


@pytest.mark.parametrize(
    ("codepoints", "expected_levels"),
    [
        (4, [-2.0, -1.0, 0.0, 1.0]),
        (6, [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0]),
        (7, [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]),
        (16, list(range(-8, 8))),
    ],
)
def test_absmax_codebook_int_quantizer_levels_include_zero(
    codepoints, expected_levels
) -> None:
    quantizer = AbsmaxCodebookIntQuantizer(codepoints=codepoints)

    assert quantizer.levels.tolist() == expected_levels
    assert 0.0 in quantizer.levels.tolist()
    assert quantizer.effective_bits == pytest.approx(
        torch.log2(torch.tensor(codepoints)).item()
    )


def test_int_codebook_rejects_no_positive_codepoint_for_absmax() -> None:
    with pytest.raises(ValueError, match="positive codepoint"):
        GPTQConfig(quantization_format="int-codebook", codepoints=2)


def test_absmax_codebook_int4_uses_positive_scale_denominator() -> None:
    quantizer = AbsmaxCodebookIntQuantizer(codepoints=16)
    weight = torch.tensor([[-8.0, 7.0]], dtype=torch.float32)

    quantizer.find_params(weight)

    expected_scale = Q.BFLOAT16.quantise(torch.tensor([8.0 / 7.0])).reshape(1, 1)
    torch.testing.assert_close(quantizer.scale, expected_scale)
    assert quantizer.scale_denominator == 7.0


@pytest.mark.parametrize(
    ("codepoints", "expected_scale", "expected_zero"),
    [(5, 1.0, 2.0), (6, 0.8, 3.0)],
)
def test_affine_codebook_symmetric_params_use_all_resolution(
    codepoints, expected_scale, expected_zero
) -> None:
    weight = torch.tensor([[-2.0, -0.5, 0.0, 1.0, 2.0]])
    quantizer = AffineCodebookIntQuantizer(
        codepoints=codepoints,
        sym=True,
        scale_zero_dtype="float32",
    )

    quantizer.find_params(weight)

    torch.testing.assert_close(quantizer.scale, torch.tensor([[expected_scale]]))
    torch.testing.assert_close(quantizer.zero, torch.tensor([[expected_zero]]))
    assert int(quantizer.maxq) == codepoints - 1


def test_affine_parameter_dtype_rounds_before_quantization() -> None:
    weight = torch.tensor([[-1.0, 0.2, 1.0]])
    fp32 = UniformAffineQuantizer(
        bits=3, sym=True, scale_zero_dtype="float32"
    )
    bf16 = UniformAffineQuantizer(
        bits=3, sym=True, scale_zero_dtype="bfloat16"
    )

    fp32.find_params(weight)
    bf16.find_params(weight)

    expected = fp32.scale.to(torch.bfloat16).to(torch.float32)
    torch.testing.assert_close(bf16.scale, expected)
    assert not torch.equal(bf16.scale, fp32.scale)


@pytest.mark.parametrize("codepoints,bits", [(4, 2), (8, 3), (16, 4)])
@pytest.mark.parametrize("scale_zero_dtype", ["bfloat16", "float16", "float32"])
@pytest.mark.parametrize("sym,mse", [(True, 0.0), (False, 0.0), (True, 2.0)])
def test_affine_codebook_quantizer_matches_power_of_two_int(
    codepoints, bits, scale_zero_dtype, sym, mse
) -> None:
    weight = torch.tensor(
        [
            [-2.0, -0.3, 0.0, 0.7, 1.5],
            [-0.2, 0.0, 0.1, 1.1, 3.0],
        ]
    )
    expected = UniformAffineQuantizer(
        bits=bits,
        sym=sym,
        mse=mse,
        scale_zero_dtype=scale_zero_dtype,
    )
    actual = AffineCodebookIntQuantizer(
        codepoints=codepoints,
        sym=sym,
        mse=mse,
        scale_zero_dtype=scale_zero_dtype,
    )

    expected.find_params(weight)
    actual.find_params(weight)

    torch.testing.assert_close(actual.scale, expected.scale)
    torch.testing.assert_close(actual.zero, expected.zero)
    torch.testing.assert_close(actual.quantize(weight), expected.quantize(weight))


@pytest.mark.parametrize("codepoints,bits", [(4, 2), (8, 3), (16, 4)])
@pytest.mark.parametrize("scale_zero_dtype", ["bfloat16", "float16", "float32"])
def test_affine_codebook_gptq_matches_power_of_two_int(
    codepoints, bits, scale_zero_dtype
) -> None:
    torch.manual_seed(625464)
    linear = torch.nn.Linear(8, 5, bias=False)
    calibration = torch.randn(2, 5, 8)

    expected = GPTQLinearQuantizer(
        linear,
        GPTQConfig(
            bits=bits,
            group_size=4,
            blocksize=4,
            scale_zero_dtype=scale_zero_dtype,
        ),
    )
    actual = GPTQLinearQuantizer(
        linear,
        GPTQConfig(
            quantization_format="int-codebook-affine",
            bits=None,
            codepoints=codepoints,
            group_size=4,
            blocksize=4,
            scale_zero_dtype=scale_zero_dtype,
        ),
    )
    expected.add_batch(calibration)
    actual.add_batch(calibration)

    expected_result = expected.quantize()
    actual_result = actual.quantize()

    torch.testing.assert_close(actual_result.weight, expected_result.weight)
    torch.testing.assert_close(actual_result.scale, expected_result.scale)
    torch.testing.assert_close(actual_result.zero, expected_result.zero)
    torch.testing.assert_close(actual_result.g_idx, expected_result.g_idx)


@pytest.mark.parametrize(
    "desc_act,act_group_aware,static_groups,group_size",
    [
        (False, False, False, -1),
        (True, False, False, 4),
        (False, False, True, 4),
    ],
)
def test_affine_codebook_gptq_anchor_parity_across_group_modes(
    desc_act, act_group_aware, static_groups, group_size
) -> None:
    torch.manual_seed(625464)
    linear = torch.nn.Linear(8, 5, bias=False)
    calibration = torch.randn(2, 5, 8)
    common = dict(
        group_size=group_size,
        blocksize=4,
        desc_act=desc_act,
        act_group_aware=act_group_aware,
        static_groups=static_groups,
        sym=False,
        mse=2.0,
        scale_zero_dtype="bfloat16",
    )
    expected = GPTQLinearQuantizer(linear, GPTQConfig(bits=3, **common))
    actual = GPTQLinearQuantizer(
        linear,
        GPTQConfig(
            quantization_format="int-codebook-affine",
            bits=None,
            codepoints=8,
            **common,
        ),
    )
    expected.add_batch(calibration)
    actual.add_batch(calibration)

    expected_result = expected.quantize()
    actual_result = actual.quantize()

    torch.testing.assert_close(actual_result.weight, expected_result.weight)
    torch.testing.assert_close(actual_result.scale, expected_result.scale)
    torch.testing.assert_close(actual_result.zero, expected_result.zero)
    torch.testing.assert_close(actual_result.g_idx, expected_result.g_idx)


def test_affine_codebook_k4_through_k16_are_finite_and_reduce_error() -> None:
    weight = torch.linspace(-1.0, 1.0, 257).reshape(1, -1)
    errors = []

    for codepoints in range(4, 17):
        quantizer = AffineCodebookIntQuantizer(
            codepoints=codepoints,
            sym=True,
            scale_zero_dtype="float32",
        )
        quantizer.find_params(weight)
        quantized = quantizer.quantize(weight)
        assert torch.isfinite(quantized).all()
        assert quantizer.scale.shape == (1, 1)
        assert quantizer.zero.shape == (1, 1)
        errors.append(torch.mean((quantized - weight) ** 2).item())

    assert all(
        next_error <= error + 1e-12
        for error, next_error in zip(errors, errors[1:])
    )


def test_int_codebook_gptq_has_no_nans_for_dead_hessian_columns() -> None:
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

    assert result.quantization_format == "int-codebook"
    assert result.weight.shape == linear.weight.shape
    assert result.effective_bits == pytest.approx(
        torch.log2(torch.tensor(6)).item()
    )
    assert not torch.isnan(result.weight).any()
    assert result.zero.numel() == 0
    assert result.g_idx.shape == (8,)


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
