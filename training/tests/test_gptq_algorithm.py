# Copyright (c) 2026 Graphcore Ltd. All rights reserved.

import pytest
import torch

from gptq.algorithm import GPTQConfig, GPTQLinearQuantizer


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
