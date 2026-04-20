"""Generate test data for gentest_tensor."""

import inspect
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn, tensor


@dataclass
class ChannelInt8Data:
    data: Tensor
    scale: Tensor

    def __post_init__(self) -> None:
        assert self.data.dtype == torch.int8
        assert self.scale.dtype == torch.bfloat16
        assert self.data.ndim == 2
        assert self.scale.shape == (self.data.shape[0],)


class TestFile:
    def __init__(self) -> None:
        self.offset = 0
        self.buffer_list: list[bytes] = []
        self.tests = []

    @staticmethod
    def _align(offset: int, alignment: int = 4) -> int:
        return offset + (-offset % alignment)

    def _store_tensor(self, t: Tensor) -> dict[str, Any]:
        pad = -self.offset % 4
        self.buffer_list.append(b"\0" * pad)
        self.offset += pad
        start_offset = self.offset
        _, dtype = str(t.dtype).split(".")
        buffer = t.contiguous().view(-1).view(torch.uint8).numpy().tobytes()
        self.buffer_list.append(buffer)
        self.offset += len(buffer)
        return dict(
            type="tensor", dtype=dtype, shape=list(t.shape), offset=start_offset
        )

    def store(self, t: int | float | list | Tensor | ChannelInt8Data) -> dict[str, Any]:
        if isinstance(t, (int, float)):
            return dict(type="scalar", data=t)
        if isinstance(t, list):
            return dict(type="list", data=t)
        if isinstance(t, ChannelInt8Data):
            out = self._store_tensor(t.data)
            out["scale"] = self._store_tensor(t.scale)
            return out
        if isinstance(t, Tensor):
            return self._store_tensor(t)
        raise ValueError(f"Unsupported type: {type(t)}")

    def add(self, op: str, name: str, **data: int | float | list | Tensor) -> None:
        self.tests.append(
            dict(
                op=op,
                name=name,
                data={k: self.store(v) for k, v in data.items()},
            )
        )

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            meta_bytes = json.dumps(dict(tests=self.tests)).encode("utf-8")
            f.write("TESTDATA".encode("utf-8"))
            f.write(len(meta_bytes).to_bytes(8, sys.byteorder))
            f.write(meta_bytes)
            payload_bytes = b"".join(self.buffer_list)
            f.write(len(payload_bytes).to_bytes(8, sys.byteorder))
            f.write(payload_bytes)


def _rotate(z: Tensor, angle: Tensor) -> Tensor:
    zx, zy = z.unflatten(-1, (2, -1)).movedim(-2, 0)
    while angle.ndim < z.ndim:
        angle.unsqueeze_(1)
    return torch.cat(
        [zx * angle.cos() - zy * angle.sin(), zy * angle.cos() + zx * angle.sin()], -1
    )


class Tests:
    @classmethod
    def all(cls, tests: TestFile) -> None:
        for name, method in inspect.getmembers(cls, predicate=inspect.isroutine):
            if not (name.startswith("_") or name == "all"):
                method(tests)

    # Data movement and type conversion

    @staticmethod
    def casts(tests: TestFile) -> None:
        torch.manual_seed(0xD4E6F5A9B3C2D1E0)
        x = torch.randn(7, 13)
        tests.add("cast", "unit", float=x, bf16=x.to(torch.bfloat16))
        finfo = torch.finfo(torch.bfloat16)
        for name, value in dict(
            min=finfo.min,
            max=finfo.max,
            small=finfo.smallest_normal,
            small_neg=-finfo.smallest_normal,
        ).items():
            x = tensor(value).mul(1 + 2**-8)  # adjust so that float value != bf16 value
            tests.add("cast", name, float=x, bf16=x.to(torch.bfloat16))

    @staticmethod
    def concat(tests: TestFile) -> None:
        torch.manual_seed(0xF1F505CF541D67E8)
        t0 = torch.randn(7, 13)
        t1 = torch.randn(5, 13)
        tests.add("concat2", "dim0", t0=t0, t1=t1, dim=0, output=torch.cat([t0, t1], 0))
        t0 = torch.randn(7, 13)
        t1 = torch.randn(7, 17)
        tests.add("concat2", "dim1", t0=t0, t1=t1, dim=1, output=torch.cat([t0, t1], 1))

    @staticmethod
    def tile(tests: TestFile) -> None:
        torch.manual_seed(0xFF6C46CE4EB71B3)
        tensor_ = torch.randn(3, 5)
        reps = [7]
        output = tensor_.repeat(*reps, 1, 1)
        tests.add("tile", "1D", tensor=tensor_, reps=reps, output=output)
        reps = [2, 7]
        output = tensor_.repeat(*reps, 1, 1)
        tests.add("tile", "2D", tensor=tensor_, reps=reps, output=output)

    # Maths/NN ops

    @staticmethod
    def add(tests: TestFile) -> None:
        torch.manual_seed(0x1B45E6A9EFDAABBA)
        x = torch.randn(5, 23).mul(3)
        y = torch.randn(5, 23).mul(0.7)
        tests.add("add", "basic", x=x, y=y, output=x + y)
        tests.add("broadcastAdd", "basic", x=x, y=y[0], output=x + y[0])

    @staticmethod
    def nonlinearities(tests: TestFile) -> None:
        torch.manual_seed(0x4882A4CC4699C100)
        x = torch.randn(5, 100).mul(3)
        gate = torch.randn(5, 100).mul(3)
        tests.add("gelu", "basic", x=x, output=nn.functional.gelu(x))
        tests.add(
            "swiGlu",
            "basic",
            x=x,
            gate=gate,
            output=x.mul(nn.functional.silu(gate)),
        )

    @staticmethod
    def norms(tests: TestFile) -> None:
        torch.manual_seed(0xFE218FECC5B4C9C3)
        scale = tensor([1, 0.1, 10, 3e-4, 2e5])
        x = torch.randn(5, 480).add(-0.2) * scale[:, None]
        weight, bias, epsilon = torch.randn(480).exp(), torch.randn(480), 1e-5

        output = nn.functional.layer_norm(x, (480,), weight, bias, eps=epsilon)
        tests.add(
            "layerNorm",
            "basic",
            x=x,
            weight=weight,
            bias=bias,
            epsilon=epsilon,
            output=output,
        )
        output = nn.functional.rms_norm(x, (480,), weight, eps=epsilon)
        tests.add(
            "rmsNorm",
            "basic",
            x=x,
            weight=weight,
            epsilon=epsilon,
            output=output,
        )

    @staticmethod
    def embeddingLookup(tests: TestFile) -> None:
        torch.manual_seed(0xA8B6B8AEF045B5CE)
        weight = torch.randn(91, 32).bfloat16()
        tokens = [10, 64, 0, 1, 90]
        output = weight[tensor(tokens)]
        tests.add(
            "embeddingLookup",
            "basic",
            weight=weight,
            tokens=tokens,
            output=output,
        )

        torch.manual_seed(0xA20EB8EB7546F6F0)
        weight_i8 = torch.randint(-8, 8, (91, 32), dtype=torch.int8)
        scale = (0.05 + 0.2 * torch.rand(91)).to(torch.bfloat16)
        output = (weight_i8.bfloat16() * scale[:, None])[tensor(tokens)]
        tests.add(
            "embeddingLookup",
            "channel_int8",
            weight=ChannelInt8Data(weight_i8, scale),
            tokens=tokens,
            output=output,
        )

    @staticmethod
    def projection(tests: TestFile) -> None:
        torch.manual_seed(0x19DFB9E938DCE261)
        weight = torch.randn(128, 256).bfloat16() / (256**0.5)
        x = torch.randn(64, 256).bfloat16()
        output = x @ weight.T
        tests.add("projection", "regular", weight=weight, x=x, output=output)

        torch.manual_seed(0x73DAB3442659F0EE)
        weight = torch.randn(64, 143).bfloat16() / (143**0.5)
        x = torch.randn(32, 143).bfloat16()
        output = x @ weight.T
        tests.add("projection", "odd-k", weight=weight, x=x, output=output)

        torch.manual_seed(0x73DAB3442659F0EE)
        weight = torch.randn(79, 64).bfloat16() / (64**0.5)
        x = torch.randn(47, 64).bfloat16()
        output = x @ weight.T
        tests.add("projection", "odd-mn", weight=weight, x=x, output=output)

        torch.manual_seed(0x5BA87BB13DF4F97)
        weight = torch.randn(3, 5).bfloat16() / (5**0.5)
        x = torch.randn(7, 5).bfloat16()
        output = x @ weight.T
        tests.add("projection", "small", weight=weight, x=x, output=output)

        torch.manual_seed(0x796C93DFEDE7751A)
        weight = torch.randn(137, 79).bfloat16() / (79**0.5)
        x = torch.randn(37, 79).bfloat16()
        output = x @ weight.T
        tests.add("projection", "prime", weight=weight, x=x, output=output)

        torch.manual_seed(0x54CD9CD9B9E8CD2E)
        weight_i8 = torch.randint(-9, 9, (53, 41), dtype=torch.int8)
        scale = (0.02 + 0.3 * torch.rand(53)).bfloat16()
        x = torch.randn(17, 41).bfloat16()
        output = x @ (weight_i8.bfloat16() * scale[:, None]).T
        tests.add(
            "projection",
            "channel_int8",
            weight=ChannelInt8Data(weight_i8, scale),
            x=x,
            output=output,
        )

    @staticmethod
    def rotate(tests: TestFile) -> None:
        torch.manual_seed(0x5E62430FA002CCF6)
        x = torch.randn(100, 2, 7, 32)
        offset = 11
        freq = 1000 ** torch.arange(0, 32, 2).div(32).neg()
        angle = torch.arange(offset, x.shape[0] + offset)[:, None] * freq
        output = _rotate(x, angle)
        tests.add(
            "rotate",
            "basic",
            x=x,
            freq=freq.tolist(),
            offset=offset,
            output=output,
        )

    @staticmethod
    def attention(tests: TestFile) -> None:
        torch.manual_seed(0x918ECDBAE4B5A6B)
        query = torch.randn(13, 3, 5, 32)
        key = torch.randn(17, 3, 32)
        value = torch.randn(17, 3, 32)
        for causal in [False, True]:
            out = (
                nn.functional.scaled_dot_product_attention(
                    query.movedim(0, -2).flatten(end_dim=1),
                    key.movedim(0, -2),
                    value.movedim(0, -2),
                    # Note: is_causal=true doesn't match expected behaviour when
                    # the query is shorter than the key, so use `attn_mask`.
                    attn_mask=(
                        torch.tril(
                            torch.ones(13, 17, dtype=torch.bool), diagonal=17 - 13
                        )
                        if causal
                        else None
                    ),
                    enable_gqa=True,
                )
                .movedim(-2, 0)
                .unflatten(1, query.shape[1:3])
            )
            tests.add(
                "attention",
                "causal" if causal else "noncausal",
                query=query,
                key=key,
                value=value,
                output=out,
                causal=causal,
            )


if __name__ == "__main__":
    tests = TestFile()
    Tests.all(tests)
    tests.save(Path(sys.argv[1]))
