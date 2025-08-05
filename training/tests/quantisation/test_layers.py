import torch
from torch import nn

import quantisation as Q
import quantisation.layers as QL


def test_quant_linear_forward() -> None:
    in_features, out_features = 3, 1
    w = torch.tensor([[0.25, 0.75, 1.0]], dtype=torch.float32)
    linear = torch.nn.Linear(in_features, out_features, bias=False, dtype=torch.float32)
    linear.weight.data = w
    quant_linear = QL.QuantLinear(
        in_features,
        out_features,
        weight_fmt=Q.tensor_scaling_format(Q.parse("E0M1"), scale_format=Q.FP32),
        bias=False,
        dtype=torch.float32,
    )
    quant_linear.weight.data = w

    x = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    torch.testing.assert_close(linear(x).item(), 4.75)
    torch.testing.assert_close(quant_linear(x).item(), 5.0)


def test_quant_linear_backward() -> None:
    in_features, out_features = 128, 1
    quant_linear = QL.QuantLinear(
        in_features,
        out_features,
        weight_fmt=Q.tensor_scaling_format(Q.parse("E0M1"), scale_format=Q.FP32),
        bias=False,
        dtype=torch.float32,
    )

    x = torch.randn((1, in_features), dtype=torch.float32, requires_grad=True)
    y = quant_linear(x)
    y.backward()
    torch.testing.assert_close(quant_linear.weight.grad, x)
    torch.testing.assert_close(
        x.grad,
        quant_linear.weight_fmt.quantise(quant_linear.weight.data),
    )


def test_quantise_linear_layers() -> None:
    class DummyModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.c = nn.ModuleList([nn.Linear(32, 64), nn.Linear(64, 32)])

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(32, 32)
            self.b = DummyModule()

    model = DummyModel()
    q_model = QL.quantise_linear_layers(model, weight_fmt=Q.IntFormat(4))

    for m in model.modules():
        if isinstance(m, nn.Linear):
            assert not isinstance(m, QL.QuantLinear)
    for m in q_model.modules():
        if isinstance(m, nn.Linear):
            assert isinstance(m, QL.QuantLinear)
