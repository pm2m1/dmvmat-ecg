"""Synthetic CPU/MPS parity and optimizer probe; never reads ECG records."""

import copy
import importlib.metadata
import json
import os
import platform
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import torch

from dmvmat_ecg.models.networks import build, macs, parameters
from dmvmat_ecg.preprocessing.signals import views
from dmvmat_ecg.utils.io import source_hash, write_json


def main():
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        raise RuntimeError("Silent operator fallback is forbidden")
    available = torch.backends.mps.is_available()
    results = []
    for name in ("xresnet101", "flat_cnn", "multiview"):
        torch.manual_seed(1701)
        length = 250 if name == "xresnet101" else 1000
        x = torch.randn(4, 12, length)
        inputs = {"wave": x}
        if name == "multiview":
            _, r, f = views(x.numpy())
            inputs.update(rhythm=torch.tensor(r), spectrum=torch.tensor(f))
        cpu = build(name).eval()
        target = torch.tensor(np.random.default_rng(8).integers(0, 2, (4, 5)), dtype=torch.float32)
        result = dict(
            model=name,
            parameters=parameters(cpu),
            per_window_macs=macs(cpu, tuple(v[:1] for v in inputs.values())),
            length=length,
            MAC_scope="Conv1d+Linear only, excludes preprocessing/pooling/normalization",
        )
        loss = torch.nn.functional.binary_cross_entropy_with_logits(cpu(**inputs), target)
        loss.backward()
        assert torch.isfinite(loss) and all(
            p.grad is not None and torch.isfinite(p.grad).all() for p in cpu.parameters()
        )
        result["cpu_forward_backward"] = "passed"
        if available:
            try:
                gpu = copy.deepcopy(cpu).to("mps").eval()
                gpu.zero_grad(set_to_none=True)
                gpu_inputs = {k: v.to("mps") for k, v in inputs.items()}
                with warnings.catch_warnings(record=True) as caught:
                    out_cpu = cpu(**inputs)
                    out_gpu = gpu(**gpu_inputs)
                    torch.testing.assert_close(out_gpu.cpu(), out_cpu, atol=1e-3, rtol=1e-3)
                    gpu_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        out_gpu, target.to("mps")
                    )
                    torch.testing.assert_close(gpu_loss.cpu(), loss, atol=1e-4, rtol=1e-3)
                    gpu_loss.backward()
                    for a, b in zip(cpu.parameters(), gpu.parameters(), strict=True):
                        torch.testing.assert_close(b.grad.cpu(), a.grad, atol=3e-3, rtol=1e-2)
                    # Includes train-mode BN, dropout, registered optimizer and norm checks.
                    gpu.train()
                    optimizer = torch.optim.AdamW(gpu.parameters(), lr=0.01, weight_decay=0.01)
                    optimizer.zero_grad(set_to_none=True)
                    loss2 = torch.nn.functional.binary_cross_entropy_with_logits(
                        gpu(**gpu_inputs), target.to("mps")
                    )
                    loss2.backward()
                    torch.nn.utils.clip_grad_norm_(
                        gpu.parameters(), float("inf"), error_if_nonfinite=True
                    )
                    optimizer.step()
                    assert all(torch.isfinite(p).all() for p in gpu.parameters())
                    result.update(
                        mps_parity="passed",
                        mps_train_step="passed",
                        warnings=[str(w.message) for w in caught],
                    )
                del gpu
                torch.mps.empty_cache()
            except Exception as error:
                result.update(
                    mps_parity="failed", reason=type(error).__name__ + ": " + str(error)[:2000]
                )
        else:
            result["mps_parity"] = "unavailable"
        results.append(result)
        print(json.dumps(result), flush=True)
    selected = (
        "mps" if available and all(r.get("mps_parity") == "passed" for r in results) else "cpu"
    )
    receipt = dict(
        device=selected,
        common_device=True,
        synthetic_only=True,
        results=results,
        source_hash=source_hash(ROOT),
        python=platform.python_version(),
        versions={
            n: importlib.metadata.version(n)
            for n in ("torch", "wfdb", "numpy", "pandas", "scipy", "scikit-learn")
        },
        mps_available=available,
        mps_built=torch.backends.mps.is_built(),
        silent_fallback=False,
        cpu_mps_fitted_weight_identity_claimed=False,
    )
    write_json(ROOT / "results/device_probe.json", receipt)
    print("SELECTED_COMMON_DEVICE", selected, flush=True)


if __name__ == "__main__":
    main()
