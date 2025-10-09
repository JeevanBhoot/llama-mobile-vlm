"""Generate test data for gentest_tensor."""

import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch import Tensor


class TestFile:
    def __init__(self) -> None:
        self.offset = 0
        self.buffer_list = []
        self.tests = []

    def store(self, t: Tensor) -> dict[str, Any]:
        if t.dtype != torch.float32:
            raise ValueError(f"Unsupported dtype: {t.dtype}, must be torch.float32")
        try:
            return dict(type="tensor", shape=t.shape, offset=self.offset)
        finally:
            self.offset += t.numel() * 4
            self.buffer_list.append(t.flatten())

    def add(self, op: str, name: str, **tensors: Tensor) -> None:
        self.tests.append(
            dict(
                op=op,
                name=name,
                data={k: self.store(v) for k, v in tensors.items()},
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


def generate_tests(tests: TestFile) -> None:
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


if __name__ == "__main__":
    tests = TestFile()
    generate_tests(tests)
    tests.save(Path(sys.argv[1]))
