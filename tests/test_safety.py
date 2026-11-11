"""Budget, role and synthetic frozen-state safety checks."""

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
import torch

from dmvmat_ecg.evaluation.state import Study
from dmvmat_ecg.models.networks import build
from dmvmat_ecg.preprocessing.signals import views
from dmvmat_ecg.transfer.interface import MODULES, apply_mask, frozen_snapshot
from dmvmat_ecg.utils.io import write_json


def test_hard_global_attempt_ceiling_and_restart(tmp_path):
    config = json.loads((Path(__file__).parents[1] / "configs/study.json").read_text())
    write_json(tmp_path / "configs/study.json", config)
    study = Study(tmp_path)
    study.set("phase", "development")
    # Eleven registered slots, then two permitted failed-slot repairs.
    for name in ("xresnet101", "flat_cnn", "multiview"):
        study.finish(study.reserve(name, 17, "smoke", config), "failed", {})
    for name in config["models"]:
        for seed in config["seeds"]:
            study.finish(study.reserve(name, seed, "full", config), "failed", {})
    for name in ("xresnet101", "flat_cnn"):
        study.finish(study.reserve(name, 17, "smoke", config), "failed", {})
    restarted = Study(tmp_path)
    assert len(restarted.runs()) == 13
    with pytest.raises(RuntimeError, match="exhausted"):
        restarted.reserve("multiview", 17, "smoke", config)


def test_concurrent_duplicate_reservations_are_unique(tmp_path):
    config = json.loads((Path(__file__).parents[1] / "configs/study.json").read_text())
    write_json(tmp_path / "configs/study.json", config)
    Study(tmp_path).set("phase", "development")

    def reserve(_):
        s = Study(tmp_path)
        try:
            return s.reserve("flat_cnn", 101, "full", config)
        except (PermissionError, sqlite3.OperationalError):
            return None
        finally:
            s.db.close()

    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(reserve, range(3)))
    assert sum(r is not None for r in results) == 1
    assert len(Study(tmp_path).runs()) == 1


def test_synthetic_head_update_preserves_frozen_weights_and_buffers():
    m = build("multiview", 27)
    mask = dict.fromkeys(MODULES, "SHARE")
    apply_mask(m, mask)
    before = frozen_snapshot(m, mask)
    head = m.prediction_head.weight.detach().clone()
    x = np.random.default_rng(8).normal(size=(4, 12, 1000)).astype("float32")
    wave, rhythm, spectrum = [torch.tensor(v) for v in views(x)]
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=0.01)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        m(wave, rhythm, spectrum), torch.zeros(4, 5)
    )
    loss.backward()
    opt.step()
    assert not torch.equal(head, m.prediction_head.weight)
    for n, module in m.selectable_modules().items():
        for key, value in module.state_dict().items():
            assert torch.equal(value, before[n][key])
