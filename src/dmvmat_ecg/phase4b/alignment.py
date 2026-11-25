"""Differentiable, explicitly declared four-kernel biased MMD."""

import torch


def mk_mmd(source, target):
    if source.ndim != 2 or target.ndim != 2 or source.shape[1] != target.shape[1]:
        raise ValueError("Aligned representation matrices required")
    if (
        min(len(source), len(target)) < 2
        or not torch.isfinite(source).all()
        or not torch.isfinite(target).all()
    ):
        raise ValueError("Finite nontrivial source/target batches required")
    values = torch.cat([source, target], dim=0)
    distances = torch.cdist(values, values, p=2).square()
    positive = distances.detach()[distances.detach() > 0]
    scale = positive.median() if positive.numel() else distances.new_tensor(1e-8)
    scale = scale.clamp_min(1e-8)
    kernels = (
        sum(
            torch.exp(-distances / (2 * scale * multiplier**2))
            for multiplier in (0.5, 1.0, 2.0, 4.0)
        )
        / 4
    )
    n = len(source)
    value = kernels[:n, :n].mean() + kernels[n:, n:].mean() - 2 * kernels[:n, n:].mean()
    return value.clamp_min(0)


def alignment_loss(source_reps, target_reps):
    if set(source_reps) != set(target_reps) or len(source_reps) != 7:
        raise ValueError("All seven module representations required")
    return sum(mk_mmd(source_reps[n], target_reps[n]) for n in source_reps)
