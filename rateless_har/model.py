"""Depthwise-separable CNN with per-channel weight / per-tensor activation QAT."""

import torch
from torch import nn
from torch.nn import functional as F
from torch.ao.quantization import (
    FakeQuantize,
    MovingAverageMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver,
)


def activation_quantizer():
    return FakeQuantize(
        observer=MovingAverageMinMaxObserver,
        quant_min=-128,
        quant_max=127,
        dtype=torch.qint8,
        qscheme=torch.per_tensor_symmetric,
    )


def weight_quantizer():
    return FakeQuantize(
        observer=MovingAveragePerChannelMinMaxObserver,
        quant_min=-127,
        quant_max=127,
        dtype=torch.qint8,
        qscheme=torch.per_channel_symmetric,
        ch_axis=0,
    )


class QConv(nn.Module):
    def __init__(self, cin, cout, kernel, stride=(1, 1), groups=1):
        super().__init__()
        self.conv = nn.Conv2d(
            cin, cout, kernel, stride, (kernel[0] // 2, kernel[1] // 2), groups=groups, bias=False
        )
        self.weight_quant = weight_quantizer()
        self.bn = nn.BatchNorm2d(cout)
        self.activation_quant = activation_quantizer()

    def forward(self, x):
        c = self.conv
        y = F.conv2d(
            x, self.weight_quant(c.weight), None, c.stride, c.padding, c.dilation, c.groups
        )
        return self.activation_quant(F.relu6(self.bn(y)))


class QLinear(nn.Module):
    def __init__(self, din, dout):
        super().__init__()
        self.linear = nn.Linear(din, dout)
        self.weight_quant = weight_quantizer()

    def forward(self, x):
        return F.linear(x, self.weight_quant(self.linear.weight), self.linear.bias)


class TinyHAR(nn.Module):
    def __init__(self, classes, feature_dim=128):
        super().__init__()
        if classes < 2:
            raise ValueError("at least two classes required")
        self.input_quant = activation_quantizer()
        self.blocks = nn.ModuleList([QConv(1, 16, (3, 5))])
        for cin, cout, stride in [(16, 32, (1, 1)), (32, 64, (1, 2)), (64, 96, (1, 2))]:
            self.blocks.append(QConv(cin, cin, (3, 3), stride, groups=cin))
            self.blocks.append(QConv(cin, cout, (1, 1)))
        self.projection = QLinear(96, feature_dim)
        self.feature_quant = activation_quantizer()
        self.head, self.margin_head = QLinear(feature_dim, classes), QLinear(feature_dim, 1)
        self.dropout_head = nn.Sequential(nn.Dropout(0.2), nn.Linear(feature_dim, classes))

    def features(self, x):
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError("expected [N,1,sensors,time]")
        x = self.input_quant(x)
        for block in self.blocks:
            x = block(x)
        return self.feature_quant(self.projection(x.mean((2, 3))))

    def forward(self, x, masked=False):
        z = self.features(x)
        partial = z
        if masked:
            ratios = torch.tensor([0.25, 0.5, 0.75, 1.0], device=z.device)
            visible = ratios[torch.randint(0, 4, (len(z), 1), device=z.device)]
            partial = z * (torch.rand_like(z) < visible).to(z.dtype)
        return self.head(partial), z, self.margin_head(partial).squeeze(-1)


def model_profile(model, channels=6, samples=100):
    macs, handles = [0], []

    def hook(module, inputs, output):
        if isinstance(module, QConv):
            c = module.conv
            macs[0] += (
                output.numel() * (c.in_channels // c.groups) * c.kernel_size[0] * c.kernel_size[1]
            )
        else:
            macs[0] += output.numel() * module.linear.in_features

    for module in model.modules():
        if isinstance(module, (QConv, QLinear)):
            handles.append(module.register_forward_hook(hook))
    with torch.no_grad():
        model(torch.zeros(1, 1, channels, samples))
    for handle in handles:
        handle.remove()
    return dict(
        parameters=sum(p.numel() for p in model.parameters()),
        macs=macs[0],
        input_channels=channels,
        input_samples=samples,
        note="Auxiliary margin MACs included; dropout training head parameters included.",
    )
