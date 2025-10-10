"""Generate test data for gentest_tensor."""

import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn, tensor
import inspect


class TestFile:
    def __init__(self) -> None:
        self.offset = 0
        self.buffer_list = []
        self.tests = []

    def store(self, t: int | float | list | Tensor) -> dict[str, Any]:
        if isinstance(t, (int, float)):
            return dict(type="scalar", data=t)
        if isinstance(t, list):
            return dict(type="list", data=t)
        if isinstance(t, Tensor):
            if t.dtype != torch.float32:
                raise ValueError(f"Unsupported dtype: {t.dtype}, must be torch.float32")
            try:
                return dict(type="tensor", shape=t.shape, offset=self.offset)
            finally:
                self.offset += t.numel() * 4
                self.buffer_list.append(t.flatten())
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
            payload_bytes = torch.cat(self.buffer_list).numpy().tobytes()
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

    # Math/NN ops

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
        weight = torch.randn(91, 32)
        tokens = [10, 64, 0, 1, 90]
        output = weight[tensor(tokens)]
        tests.add(
            "embeddingLookup",
            "basic",
            weight=weight,
            tokens=tokens,
            output=output,
        )

    @staticmethod
    def projection(tests: TestFile) -> None:
        torch.manual_seed(0x5BA87BB13DF4F97)
        weight = torch.randn(32, 48) / (48**0.5)
        x = torch.randn(7, 48)
        output = x @ weight.T
        tests.add("projection", "small", weight=weight, x=x, output=output)

        torch.manual_seed(0x796C93DFEDE7751A)
        weight = torch.randn(137, 79) / (79**0.5)
        x = torch.randn(37, 79)
        output = x @ weight.T
        tests.add("projection", "prime", weight=weight, x=x, output=output)

    @staticmethod
    def rotate(tests: TestFile) -> None:
        torch.manual_seed(0x5E62430FA002CCF6)
        input = torch.randn(100, 2, 7, 32)
        offset = 11
        freq = 1000 ** torch.arange(0, 32, 2).div(32).neg()
        angle = torch.arange(offset, input.shape[0] + offset)[:, None] * freq
        output = _rotate(input, angle)
        tests.add(
            "rotate",
            "basic",
            input=input,
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
