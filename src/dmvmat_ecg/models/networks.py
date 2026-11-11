"""Signal-appropriate five-view encoders, capacity control and upstream XResNet."""

import random

import numpy as np
import torch
from torch import nn

from .heads import weight_init
from .vendor.xresnet1d import xresnet1d101


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class CNNEncoder(nn.Module):
    def __init__(self, channels, widths=(32, 64, 64), embedding=64):
        super().__init__()
        layers = []
        for ni, no, k in zip((channels, *widths[:-1]), widths, (7, 5, 3), strict=True):
            layers.extend(
                [
                    nn.Conv1d(ni, no, k, stride=2, padding=k // 2, bias=False),
                    nn.BatchNorm1d(no),
                    nn.ReLU(),
                ]
            )
        self.features = nn.Sequential(*layers)
        self.project = nn.Sequential(
            nn.Linear(2 * widths[-1], embedding), nn.ReLU(), nn.LayerNorm(embedding)
        )

    def forward(self, x):
        z = self.features(x)
        return self.project(torch.cat([z.mean(-1), z.amax(-1)], dim=1))


def fusion(ni, no):
    return nn.Sequential(nn.Linear(ni, no), nn.ReLU(), nn.LayerNorm(no), nn.Dropout(0.2))


class FiveViewECG(nn.Module):
    def __init__(self):
        super().__init__()
        self.view_1 = CNNEncoder(6)
        self.view_2 = CNNEncoder(3)
        self.view_3 = CNNEncoder(3)
        self.view_4 = CNNEncoder(3)
        self.view_5 = CNNEncoder(12)
        self.fusion_1 = fusion(5 * 64, 128)
        self.fusion_2 = fusion(128, 64)
        self.prediction_head = nn.Linear(64, 5)

    def representations(self, wave, rhythm, spectrum):
        views = dict(
            view_1=self.view_1(wave[:, :6]),
            view_2=self.view_2(wave[:, 6:9]),
            view_3=self.view_3(wave[:, 9:12]),
            view_4=self.view_4(rhythm),
            view_5=self.view_5(spectrum),
        )
        views["fusion_1"] = self.fusion_1(torch.cat(list(views.values()), dim=1))
        views["fusion_2"] = self.fusion_2(views["fusion_1"])
        return views

    def forward(self, wave, rhythm=None, spectrum=None):
        return self.prediction_head(self.representations(wave, rhythm, spectrum)["fusion_2"])

    def selectable_modules(self):
        return {
            n: getattr(self, n)
            for n in ("view_1", "view_2", "view_3", "view_4", "view_5", "fusion_1", "fusion_2")
        }


class FlatCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = CNNEncoder(12, widths=(64, 128, 208), embedding=128)
        self.fusion_1 = fusion(128, 128)
        self.fusion_2 = fusion(128, 64)
        self.prediction_head = nn.Linear(64, 5)

    def forward(self, wave, rhythm=None, spectrum=None):
        return self.prediction_head(self.fusion_2(self.fusion_1(self.encoder(wave))))


class ReferenceXResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = xresnet1d101(
            input_channels=12, num_classes=5, kernel_size=5, ps_head=0.5, lin_ftrs_head=[128]
        )
        self.backbone.apply(weight_init)

    def forward(self, wave, rhythm=None, spectrum=None):
        return self.backbone(wave)


def build(name, seed=101):
    seed_all(seed)
    if name == "xresnet101":
        return ReferenceXResNet()
    if name == "flat_cnn":
        return FlatCNN()
    if name in ("multiview", "multiview_aug"):
        return FiveViewECG()
    raise ValueError("Unregistered model")


def parameters(model):
    return sum(p.numel() for p in model.parameters())


def macs(model, inputs):
    """Conv1d and Linear multiply-accumulates only; excludes FFT/pooling/normalization."""
    total = [0]
    hooks = []

    def count(module, args, output):
        if isinstance(module, nn.Conv1d):
            total[0] += output.numel() * module.in_channels // module.groups * module.kernel_size[0]
        elif isinstance(module, nn.Linear):
            total[0] += output.numel() * module.in_features

    for m in model.modules():
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            hooks.append(m.register_forward_hook(count))
    model.eval()
    with torch.no_grad():
        model(*inputs)
    for hook in hooks:
        hook.remove()
    return total[0]
