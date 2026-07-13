# Copyright (c) 2026 Graphcore Ltd. All rights reserved.

"""Local GPTQ implementation for readable PTQ experiments."""

import math
import time
from dataclasses import dataclass
from typing import Literal

import torch
import weight_formats.fit as F
import weight_formats.quantisation as Q
from torch import Tensor

from quant_formats import FMT_CHANNEL_S3D8


@dataclass
class GPTQConfig:
    quantization_format: Literal[
        "int", "int-codebook", "int-codebook-affine", "s3d8"
    ] = "int"
    bits: int | None = 4
    codepoints: int | None = None
    group_size: int = 128
    blocksize: int = 128
    damp_percent: float = 0.05
    damp_auto_increment: float = 0.01
    desc_act: bool = False
    act_group_aware: bool = True
    static_groups: bool = False
    sym: bool = True
    mse: float = 0.0
    scale_zero_dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"

    def __post_init__(self) -> None:
        if self.quantization_format not in (
            "int",
            "int-codebook",
            "int-codebook-affine",
            "s3d8",
        ):
            raise ValueError(
                f"Unsupported GPTQ quantization format: {self.quantization_format}"
            )
        if self.quantization_format == "int" and self.bits not in (2, 3, 4, 8):
            raise ValueError(f"Unsupported GPTQ bit width: {self.bits}")
        if self.quantization_format in ("int-codebook", "int-codebook-affine"):
            if self.codepoints is None:
                raise ValueError(
                    f"{self.quantization_format} quantization requires codepoints"
                )
            if self.codepoints < 2:
                raise ValueError("codepoints must be at least 2")
            if self.quantization_format == "int-codebook":
                element_range = Q.IntFormat(
                    math.log2(self.codepoints),
                    mode="asymmetric",
                ).range
                if element_range[1] <= 0:
                    raise ValueError(
                        "int-codebook absmax scaling requires at least one positive "
                        f"codepoint, got codepoints={self.codepoints}"
                    )
        if self.scale_zero_dtype not in ("bfloat16", "float16", "float32"):
            raise ValueError(
                f"Unsupported scale/zero dtype: {self.scale_zero_dtype}"
            )
        if self.group_size != -1 and self.group_size <= 0:
            raise ValueError("group_size must be -1 or a positive integer")
        if self.static_groups and self.group_size == -1:
            raise ValueError("static_groups requires a positive group_size")
        if self.act_group_aware and self.group_size == -1:
            raise ValueError("act_group_aware requires a positive group_size")
        if self.act_group_aware and self.desc_act:
            raise ValueError("act_group_aware requires desc_act=False")
        if self.blocksize <= 0:
            raise ValueError("blocksize must be a positive integer")
        if not 0 < self.damp_percent < 1:
            raise ValueError("damp_percent must be between 0 and 1")
        if self.damp_auto_increment < 0:
            raise ValueError("damp_auto_increment must be non-negative")
        if self.mse < 0:
            raise ValueError("mse must be non-negative")


@dataclass
class QuantizedLinear:
    weight: Tensor
    scale: Tensor
    zero: Tensor
    g_idx: Tensor
    duration: float
    avg_loss: float
    damp_percent: float
    quantization_format: str = "int"
    effective_bits: float | None = None
    fitted_format: Q.TensorFormat | None = None
    centroid_shape: tuple[int, ...] | None = None


class UniformAffineQuantizer:
    """Per-row affine quantizer matching GPTQModel's Quantizer."""

    def __init__(
        self,
        bits: int | float,
        sym: bool,
        mse: float = 0.0,
        grid: int = 100,
        maxshrink: float = 0.8,
        scale_zero_dtype: str = "bfloat16",
        codepoints: int | None = None,
    ):
        self.bits = bits
        self.sym = sym
        self.mse = mse
        self.grid = grid
        self.maxshrink = maxshrink
        self.scale_zero_dtype = scale_zero_dtype
        self.codepoints = codepoints or 2**bits
        self.maxq = torch.tensor(self.codepoints - 1)
        self.symmetric_zero = self.codepoints // 2
        self.scale = torch.empty(0)
        self.zero = torch.empty(0)

    def copy(self) -> "UniformAffineQuantizer":
        quantizer = UniformAffineQuantizer(
            bits=self.bits,
            sym=self.sym,
            mse=self.mse,
            grid=self.grid,
            maxshrink=self.maxshrink,
            scale_zero_dtype=self.scale_zero_dtype,
            codepoints=self.codepoints,
        )
        quantizer.scale = self.scale.clone()
        quantizer.zero = self.zero.clone()
        quantizer.maxq = self.maxq.clone()
        return quantizer

    def find_params(self, weight: Tensor) -> None:
        x = weight.flatten(1)
        device = x.device
        self.maxq = self.maxq.to(device)

        zeros = torch.zeros(x.shape[0], device=device)
        xmin = torch.minimum(x.min(1)[0], zeros)
        xmax = torch.maximum(x.max(1)[0], zeros)

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            has_negative = xmin < 0
            if torch.any(has_negative):
                xmin[has_negative] = -xmax[has_negative]

        dead = (xmin == 0) & (xmax == 0)
        xmin[dead] = -1
        xmax[dead] = 1

        self.scale = (xmax - xmin) / self.maxq
        if self.sym:
            self.zero = torch.full_like(self.scale, self.symmetric_zero)
        else:
            self.zero = torch.round(-xmin / self.scale)

        if self.mse > 0.0:
            self._shrink_to_mse(x, xmin, xmax)

        parameter_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[self.scale_zero_dtype]
        self.scale = self.scale.to(parameter_dtype).to(dtype=x.dtype)
        self.zero = self.zero.to(parameter_dtype).to(dtype=x.dtype)

        shape = [-1] + [1] * (len(weight.shape) - 1)
        self.scale = self.scale.reshape(shape)
        self.zero = self.zero.reshape(shape)

    def quantize(self, x: Tensor) -> Tensor:
        q = torch.clamp(torch.round(x / self.scale) + self.zero, 0, self.maxq)
        return self.scale * (q - self.zero)

    def _shrink_to_mse(self, x: Tensor, xmin: Tensor, xmax: Tensor) -> None:
        best = torch.full([x.shape[0]], float("inf"), device=x.device)
        scale = self.scale.flatten()
        zero = self.zero.flatten()
        maxq = self.maxq

        for i in range(int(self.maxshrink * self.grid)):
            shrink = 1 - i / self.grid
            xmin1 = shrink * xmin
            xmax1 = shrink * xmax
            scale1 = (xmax1 - xmin1) / maxq
            zero1 = zero if self.sym else torch.round(-xmin1 / scale1)
            q = torch.clamp(
                torch.round(x / scale1.unsqueeze(1)) + zero1.unsqueeze(1),
                0,
                maxq,
            )
            q = scale1.unsqueeze(1) * (q - zero1.unsqueeze(1))
            err = (q - x).abs().pow_(self.mse).sum(1)
            improved = err < best
            if torch.any(improved):
                best[improved] = err[improved]
                scale[improved] = scale1[improved]
                zero[improved] = zero1[improved]


class AffineCodebookIntQuantizer(UniformAffineQuantizer):
    """Uniform affine INT quantizer with an arbitrary number of codepoints."""

    def __init__(
        self,
        codepoints: int,
        sym: bool,
        mse: float = 0.0,
        grid: int = 100,
        maxshrink: float = 0.8,
        scale_zero_dtype: str = "bfloat16",
    ):
        if codepoints < 2:
            raise ValueError("codepoints must be at least 2")
        self.effective_bits = math.log2(codepoints)
        super().__init__(
            bits=self.effective_bits,
            sym=sym,
            mse=mse,
            grid=grid,
            maxshrink=maxshrink,
            scale_zero_dtype=scale_zero_dtype,
            codepoints=codepoints,
        )

    def copy(self) -> "AffineCodebookIntQuantizer":
        quantizer = AffineCodebookIntQuantizer(
            codepoints=self.codepoints,
            sym=self.sym,
            mse=self.mse,
            grid=self.grid,
            maxshrink=self.maxshrink,
            scale_zero_dtype=self.scale_zero_dtype,
        )
        quantizer.scale = self.scale.clone()
        quantizer.zero = self.zero.clone()
        quantizer.maxq = self.maxq.clone()
        return quantizer


class AbsmaxCodebookIntQuantizer:
    """Scale-only INT codebook quantizer for fractional effective widths."""

    def __init__(
        self,
        codepoints: int,
        scale_format: Q.TensorFormat = Q.BFLOAT16,
    ):
        if codepoints < 2:
            raise ValueError("codepoints must be at least 2")
        self.codepoints = codepoints
        self.effective_bits = math.log2(codepoints)
        self.element_format = Q.IntFormat(self.effective_bits, mode="asymmetric")
        self.scale_format = scale_format
        self.levels = torch.tensor(self.element_format.centroids)
        _, max_level = self.element_format.range
        self.scale_denominator = float(max_level)
        if self.scale_denominator <= 0:
            raise ValueError(
                "int-codebook absmax scaling requires at least one positive "
                f"codepoint, got range={self.element_format.range}"
            )
        self.scale = torch.empty(0)
        self.zero = torch.empty(0)

    def copy(self) -> "AbsmaxCodebookIntQuantizer":
        quantizer = AbsmaxCodebookIntQuantizer(
            codepoints=self.codepoints,
            scale_format=self.scale_format,
        )
        quantizer.scale = self.scale.clone()
        quantizer.zero = self.zero.clone()
        return quantizer

    def find_params(self, weight: Tensor) -> None:
        x = weight.flatten(1)
        absmax = x.abs().amax(1)
        eps = torch.finfo(x.dtype).smallest_normal
        scale = absmax.div(self.scale_denominator).clamp_min_(eps)
        scale = self.scale_format.quantise(scale).to(device=x.device, dtype=x.dtype)

        shape = [-1] + [1] * (len(weight.shape) - 1)
        self.scale = scale.reshape(shape)
        self.zero = torch.empty(0, device=x.device)

    def quantize(self, x: Tensor) -> Tensor:
        q = self.element_format.quantise(Q.safe_div(x, self.scale))
        return (q * self.scale).to(dtype=x.dtype)


def _fallback_s3d8_format() -> Q.LinearScalingFormat:
    return Q.LinearScalingFormat(
        Q.Sign3D8Format(
            Q.VectorLUTFormat.create(
                torch.cartesian_prod(
                    torch.linspace(0, 127, 4),
                    torch.linspace(0, 127, 4),
                    torch.linspace(0, 127, 2),
                ),
                Q.TorchFormat(torch.int8),
                "S3D8",
                range=(0, 127),
            )
        ),
        scale_format=Q.BFLOAT16,
        block_shape=(1, None),
        scaling="absmax",
    )


class S3D8Quantizer:
    """S3D8 vector quantizer using deployable per-output-row scaling."""

    effective_bits = 8 / 3

    def __init__(
        self,
        fmt_spec: F.Scaled | Q.LinearScalingFormat = FMT_CHANNEL_S3D8,
    ):
        self.fmt_spec = fmt_spec
        self.fmt: Q.LinearScalingFormat | None = None
        self.scale = torch.empty(0)

    def find_params(self, weight: Tensor) -> None:
        if weight.ndim != 2:
            raise ValueError(
                f"S3D8 quantization expects a rank-2 weight, got {weight.ndim}"
            )
        if weight.numel() == 0:
            raise ValueError("S3D8 quantization requires a non-empty weight")

        n_vectors = ((weight.shape[0] + 2) // 3) * weight.shape[1]
        if torch.count_nonzero(weight).item() == 0 or n_vectors < 32:
            fmt = _fallback_s3d8_format()
        else:
            fmt = (
                self.fmt_spec.fit(weight)
                if isinstance(self.fmt_spec, F.Scaled)
                else self.fmt_spec
            )
        if not isinstance(fmt, Q.LinearScalingFormat) or not isinstance(
            fmt.element_format, Q.Sign3D8Format
        ):
            raise TypeError(f"Expected a Sign3D8 linear scaling format, got {fmt}")

        _, scale = fmt.normalise(weight)
        self.fmt = fmt
        self.scale = scale[:, :1].contiguous()

    @property
    def centroid_shape(self) -> tuple[int, ...] | None:
        if self.fmt is None:
            return None
        return (
            len(self.fmt.element_format.lut.values),
            self.fmt.element_format.lut.dim,
        )

    def quantize(self, x: Tensor) -> Tensor:
        if self.fmt is None:
            raise ValueError("S3D8 quantizer parameters have not been fitted")
        squeeze = False
        if x.ndim == 1:
            x = x.unsqueeze(1)
            squeeze = True
        if x.ndim != 2 or x.shape[0] != self.scale.shape[0]:
            raise ValueError(
                "S3D8 quantization expects shape "
                f"({self.scale.shape[0]}, columns), got {tuple(x.shape)}"
            )

        scale = self.scale.to(device=x.device)
        q = self.fmt.element_format.quantise(Q.safe_div(x, scale)) * scale
        q = q.to(dtype=x.dtype)
        return q.flatten() if squeeze else q


def _compute_local_perms(
    diag_h: Tensor,
    group_size: int,
) -> tuple[Tensor, Tensor]:
    n = diag_h.numel()
    n_groups = n // group_size
    if n_groups == 0:
        return (
            torch.empty(0, group_size, dtype=torch.long, device=diag_h.device),
            torch.empty(0, group_size, dtype=diag_h.dtype, device=diag_h.device),
        )
    grouped = diag_h[: n_groups * group_size].view(n_groups, group_size)
    values, indices = torch.sort(grouped, dim=1, descending=True)
    return indices.to(dtype=torch.long), values


def _compute_global_perm(
    diag_h: Tensor,
    group_size: int,
    precomputed_values: Tensor | None = None,
) -> Tensor:
    n_groups = diag_h.numel() // group_size
    if n_groups == 0:
        return torch.empty(0, dtype=torch.long, device=diag_h.device)
    if precomputed_values is not None:
        scores = precomputed_values[:, 0]
    else:
        scores = (
            diag_h[: n_groups * group_size]
            .view(n_groups, group_size)
            .max(dim=1)
            .values
        )
    try:
        return torch.argsort(scores, descending=True, stable=True)
    except TypeError:
        return torch.argsort(scores, descending=True)


def _compose_final_perm(
    local_perms: Tensor,
    global_perm: Tensor,
    group_size: int,
) -> Tensor:
    n_groups, local_group_size = local_perms.shape
    if local_group_size != group_size:
        raise ValueError(
            f"group_size mismatch: local={local_group_size}, expected={group_size}"
    )
    base = torch.arange(n_groups, device=local_perms.device) * group_size
    perm_2d = (local_perms + base.view(n_groups, 1))[
        global_perm.to(local_perms.device)
    ]
    return perm_2d.reshape(-1)


def _extend_perm_with_tail(perm: Tensor, columns: int) -> Tensor:
    covered = int(perm.numel())
    if covered >= columns:
        return perm
    tail = torch.arange(covered, columns, dtype=perm.dtype, device=perm.device)
    return torch.cat((perm, tail), dim=0)


def _invert_perm(perm: Tensor) -> Tensor:
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.numel(), device=perm.device)
    return inv


class GPTQLinearQuantizer:
    """Collect activation Hessians and GPTQ-quantize one Linear weight."""

    def __init__(self, module: torch.nn.Linear, config: GPTQConfig):
        self.module = module
        self.config = config
        self.device = module.weight.device
        self.rows, self.columns = module.weight.shape
        self.nsamples = 0
        self.H: Tensor | None = None

    def add_batch(self, inp: Tensor) -> None:
        if len(inp.shape) > 2:
            inp = inp.reshape((-1, inp.shape[-1]))
        batch_size = inp.shape[0]
        inp = inp.t().to(self.device)

        if self.H is None:
            self.H = torch.zeros((self.columns, self.columns), device=self.device)

        self.nsamples += batch_size
        inp = inp.float()
        self.H += inp.matmul(inp.t())

    @torch.inference_mode()
    def quantize(self) -> QuantizedLinear:
        if self.H is None or self.nsamples == 0:
            raise ValueError("No calibration batches were added")

        start = time.time()
        W = self.module.weight.detach().clone().float()
        quantizer = self._new_quantizer()
        if self.config.quantization_format in (
            "int",
            "int-codebook",
            "int-codebook-affine",
        ):
            quantizer.find_params(W)

        H = self.H
        self.H = None
        H.mul_(2.0 / self.nsamples)

        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        scale = []
        zero = []
        now_idx = 1

        groups = []
        if (
            self.config.quantization_format
            in ("int", "int-codebook", "int-codebook-affine")
            and self.config.static_groups
        ):
            for i in range(0, self.columns, self.config.group_size):
                group_quantizer = self._new_quantizer()
                group_quantizer.find_params(W[:, i : i + self.config.group_size])
                scale.append(group_quantizer.scale)
                if group_quantizer.zero.numel():
                    zero.append(group_quantizer.zero)
                groups.append(group_quantizer)

        if self.config.desc_act:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)
            act_group_perm = None
            global_group_perm = None
        elif self.config.act_group_aware:
            diag_h = torch.diag(H)
            local_perms, local_values = _compute_local_perms(
                diag_h,
                self.config.group_size,
            )
            global_group_perm = _compute_global_perm(
                diag_h,
                self.config.group_size,
                precomputed_values=local_values,
            )
            act_group_perm = _extend_perm_with_tail(
                _compose_final_perm(
                    local_perms,
                    global_group_perm,
                    self.config.group_size,
                ),
                self.columns,
            )
            W = W[:, act_group_perm]
            H = H[act_group_perm][:, act_group_perm]
            perm = None
            invperm = None
        else:
            perm = None
            invperm = None
            act_group_perm = None
            global_group_perm = None

        if self.config.quantization_format == "s3d8":
            quantizer.find_params(W)

        total_loss = torch.zeros((), dtype=W.dtype, device=W.device)
        Q = torch.empty(
            W.shape,
            dtype=self.module.weight.dtype,
            device=W.device,
        )
        Hinv, damp_percent = self._inverse_hessian(H)

        for i1 in range(0, self.columns, self.config.blocksize):
            i2 = min(i1 + self.config.blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if (
                    self.config.quantization_format
                    in ("int", "int-codebook", "int-codebook-affine")
                    and self.config.group_size != -1
                ):
                    if not self.config.static_groups:
                        if (i1 + i) % self.config.group_size == 0:
                            quantizer.find_params(
                                W[:, i1 + i : i1 + i + self.config.group_size]
                            )

                        if ((i1 + i) // self.config.group_size) - now_idx == -1:
                            scale.append(quantizer.scale)
                            if quantizer.zero.numel():
                                zero.append(quantizer.zero)
                            now_idx += 1
                    else:
                        idx = i1 + i
                        if self.config.desc_act:
                            assert perm is not None
                            idx = int(perm[idx])
                        quantizer = groups[idx // self.config.group_size]

                q = quantizer.quantize(w.unsqueeze(1)).flatten()
                Q1[:, i] = q
                total_loss += torch.sum((w - q) ** 2 / d**2) / 2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        avg_loss = total_loss.item() / self.nsamples
        if math.isnan(avg_loss):
            raise ValueError("Quantization failed due to NaN loss")

        if self.config.quantization_format == "s3d8":
            assert isinstance(quantizer, S3D8Quantizer)
            if self.config.desc_act:
                assert invperm is not None
                Q = Q[:, invperm]
            elif self.config.act_group_aware:
                assert act_group_perm is not None
                Q = Q[:, _invert_perm(act_group_perm).to(device=Q.device)]
            return QuantizedLinear(
                weight=Q.reshape_as(self.module.weight).type_as(self.module.weight),
                scale=quantizer.scale,
                zero=torch.empty(0, device=Q.device),
                g_idx=torch.empty(0, dtype=torch.int32, device=Q.device),
                duration=time.time() - start,
                avg_loss=avg_loss,
                damp_percent=damp_percent,
                quantization_format="s3d8",
                effective_bits=quantizer.effective_bits,
                fitted_format=quantizer.fmt,
                centroid_shape=quantizer.centroid_shape,
            )

        group_size = (
            self.config.group_size
            if self.config.group_size != -1
            else self.columns
        )
        if self.config.static_groups and self.config.desc_act:
            assert perm is not None
            g_idx = [int(perm[i]) // group_size for i in range(self.columns)]
        else:
            g_idx = [i // group_size for i in range(self.columns)]
        g_idx_tensor = torch.tensor(g_idx, dtype=torch.int32, device=Q.device)

        if self.config.desc_act:
            assert invperm is not None
            Q = Q[:, invperm]
            g_idx_tensor = g_idx_tensor[invperm]
        elif self.config.act_group_aware:
            assert act_group_perm is not None
            assert global_group_perm is not None
            Q = Q[:, _invert_perm(act_group_perm).to(device=Q.device)]
            inv_global_group_perm = _invert_perm(global_group_perm).tolist()
            reordered_group_count = len(inv_global_group_perm)
            scale = [scale[i] for i in inv_global_group_perm] + scale[
                reordered_group_count:
            ]
            if zero:
                zero = [zero[i] for i in inv_global_group_perm] + zero[
                    reordered_group_count:
                ]

        if not scale:
            scale.append(quantizer.scale)
            if quantizer.zero.numel():
                zero.append(quantizer.zero)

        if self.config.quantization_format == "int-codebook":
            assert isinstance(quantizer, AbsmaxCodebookIntQuantizer)
            return QuantizedLinear(
                weight=Q.reshape_as(self.module.weight).type_as(self.module.weight),
                scale=torch.cat(scale, dim=1),
                zero=torch.empty(0, device=Q.device),
                g_idx=g_idx_tensor,
                duration=time.time() - start,
                avg_loss=avg_loss,
                damp_percent=damp_percent,
                quantization_format="int-codebook",
                effective_bits=quantizer.effective_bits,
            )

        if self.config.quantization_format == "int-codebook-affine":
            assert isinstance(quantizer, AffineCodebookIntQuantizer)
            return QuantizedLinear(
                weight=Q.reshape_as(self.module.weight).type_as(self.module.weight),
                scale=torch.cat(scale, dim=1),
                zero=torch.cat(zero, dim=1),
                g_idx=g_idx_tensor,
                duration=time.time() - start,
                avg_loss=avg_loss,
                damp_percent=damp_percent,
                quantization_format="int-codebook-affine",
                effective_bits=quantizer.effective_bits,
            )

        return QuantizedLinear(
            weight=Q.reshape_as(self.module.weight).type_as(self.module.weight),
            scale=torch.cat(scale, dim=1),
            zero=torch.cat(zero, dim=1),
            g_idx=g_idx_tensor,
            duration=time.time() - start,
            avg_loss=avg_loss,
            damp_percent=damp_percent,
        )

    def _new_quantizer(
        self,
    ) -> (
        UniformAffineQuantizer
        | AbsmaxCodebookIntQuantizer
        | AffineCodebookIntQuantizer
        | S3D8Quantizer
    ):
        if self.config.quantization_format == "s3d8":
            return S3D8Quantizer()
        if self.config.quantization_format == "int-codebook":
            assert self.config.codepoints is not None
            scale_format = (
                Q.BFLOAT16
                if self.config.scale_zero_dtype == "bfloat16"
                else Q.TorchFormat(
                    {
                        "float16": torch.float16,
                        "float32": torch.float32,
                    }[self.config.scale_zero_dtype]
                )
            )
            return AbsmaxCodebookIntQuantizer(
                codepoints=self.config.codepoints,
                scale_format=scale_format,
            )
        if self.config.quantization_format == "int-codebook-affine":
            assert self.config.codepoints is not None
            return AffineCodebookIntQuantizer(
                codepoints=self.config.codepoints,
                sym=self.config.sym,
                mse=self.config.mse,
                scale_zero_dtype=self.config.scale_zero_dtype,
            )
        assert self.config.bits is not None
        return UniformAffineQuantizer(
            bits=self.config.bits,
            sym=self.config.sym,
            mse=self.config.mse,
            scale_zero_dtype=self.config.scale_zero_dtype,
        )

    def _inverse_hessian(self, H: Tensor) -> tuple[Tensor, float]:
        damp_percent = self.config.damp_percent
        diag = torch.arange(self.columns, device=self.device)
        original_diag = H.diagonal().clone()
        diag_mean = torch.mean(original_diag)
        while 1 > damp_percent > 0:
            H[diag, diag] = original_diag
            H[diag, diag] += damp_percent * diag_mean
            try:
                chol = torch.linalg.cholesky(H)
                inv = torch.cholesky_inverse(chol)
                H[diag, diag] = original_diag
                return torch.linalg.cholesky(inv, upper=True), damp_percent
            except torch._C._LinAlgError:
                H[diag, diag] = original_diag
                if self.config.damp_auto_increment == 0:
                    raise
                damp_percent += self.config.damp_auto_increment

        raise ValueError(
            f"damp_percent must stay between 0 and 1, current is {damp_percent}"
        )
