"""GPL-3.0 FastAI-v1 head equivalents for the pinned upstream architecture.

Reference create_head1d uses bn_drop_lin: BN -> Dropout -> Linear -> activation.
This explicit compatibility layer avoids importing an obsolete training framework.
"""

import torch
from torch import nn


class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.shape[0], -1)


class AdaptiveConcatPool1d(nn.Module):
    def forward(self, x):
        return torch.cat([x.amax(-1, keepdim=True), x.mean(-1, keepdim=True)], dim=1)


def create_head1d(
    nf, nc, lin_ftrs=None, ps=0.5, bn_final=False, bn=True, act="relu", concat_pooling=True
):
    if not concat_pooling or act != "relu":
        raise ValueError("Only the registered upstream head configuration is implemented")
    widths = [2 * nf, *(lin_ftrs or []), nc]
    probs = [ps / 2] * (len(widths) - 2) + [ps]
    layers = [AdaptiveConcatPool1d(), Flatten()]
    for k, (ni, no, p) in enumerate(zip(widths[:-1], widths[1:], probs, strict=True)):
        if bn:
            layers.append(nn.BatchNorm1d(ni))
        if p:
            layers.append(nn.Dropout(p))
        layers.append(nn.Linear(ni, no))
        if k < len(widths) - 2:
            layers.append(nn.ReLU(inplace=True))
    if bn_final:
        layers.append(nn.BatchNorm1d(nc, momentum=0.01))
    return nn.Sequential(*layers)


def weight_init(m):
    """Matches upstream fastai_model.fit's post-construction apply(weight_init)."""
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    if isinstance(m, nn.BatchNorm1d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
