"""Generate test data for gentest_tensor."""

import json
import sys
from pathlib import Path
from typing import Any

from matplotlib.pylab import indices
import torch
from torch import Tensor


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
        cls.projection(tests)
        cls.embeddingLookup(tests)
        cls.rotate(tests)

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
    def embeddingLookup(tests: TestFile) -> None:
        torch.manual_seed(0xA8B6B8AEF045B5CE)
        weight = torch.randn(91, 32)
        tokens = [10, 64, 0, 1, 90]
        output = weight[torch.tensor(tokens)]
        tests.add(
            "embeddingLookup",
            "basic",
            weight=weight,
            tokens=tokens,
            output=output,
        )

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


if __name__ == "__main__":
    tests = TestFile()
    Tests.all(tests)
    tests.save(Path(sys.argv[1]))
