"""Fictional numeric records only. Never discovers project data/checkpoints."""

import numpy as np
import torch
from torch import nn

from dmvmat_ecg.phase2.adaptation import state_hash
from dmvmat_ecg.phase2.contracts import MODULES
from dmvmat_ecg.phase3b.engine import Source
from dmvmat_ecg.phase3b.evidence import Decision
from dmvmat_ecg.phase3b.protocol import canonical
from dmvmat_ecg.phase3b.roles import Normalization, RoleData, UnitRole, array_hash


class Tiny(nn.Module):
    input_schema = "fictional-vector"

    def __init__(self):
        super().__init__()
        for n in MODULES[:5]:
            setattr(
                self,
                n,
                nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4), nn.ReLU(), nn.Dropout(0.2)),
            )
        self.fusion_1 = nn.Sequential(
            nn.Linear(20, 4), nn.BatchNorm1d(4), nn.ReLU(), nn.Dropout(0.2)
        )
        self.fusion_2 = nn.Sequential(
            nn.Linear(4, 4), nn.BatchNorm1d(4), nn.ReLU(), nn.Dropout(0.2)
        )
        self.prediction_head = nn.Linear(4, 2)

    def selectable_modules(self):
        return {n: getattr(self, n) for n in MODULES}

    def representations(self, wave):
        result = {n: getattr(self, n)(wave) for n in MODULES[:5]}
        result["fusion_1"] = self.fusion_1(torch.cat(list(result.values()), 1))
        result["fusion_2"] = self.fusion_2(result["fusion_1"])
        return result

    def forward(self, wave):
        return self.prediction_head(self.representations(wave)["fusion_2"])


def factory(seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return Tiny()


def fixture_data(*, budget=64, unknown=False):
    rng = np.random.default_rng(17)
    roles, counters, arrays = [], {}, {}
    for k, (name, size) in enumerate(
        (
            ("source_train", 32),
            ("source_stop", 24),
            ("F", budget // 2),
            ("G", budget // 2),
            ("O", 16),
        )
    ):
        x = rng.normal(size=(size, 4)).astype("float32")
        y = np.column_stack([np.arange(size) % 2, (np.arange(size) // 2) % 2]).astype("float32")
        obs = np.ones_like(y, dtype=bool)
        if unknown and name == "G":
            obs[:3, 1] = False
            y[:3, 1] = np.nan
        counters[name] = dict(features=0, outcomes=0)
        arrays[name] = [x, y, obs]

        def features(n=name):
            counters[n]["features"] += 1
            return arrays[n][0].copy()

        def outcomes(n=name):
            counters[n]["outcomes"] += 1
            return arrays[n][1].copy(), arrays[n][2].copy()

        source = name.startswith("source")
        records = tuple(
            str((k + 1) * 1000 + i) if source else f"fictional:{name}:{i}" for i in range(size)
        )
        roles.append(
            UnitRole(
                name,
                records,
                tuple(f"{name}:group:{i}" for i in range(size)),
                "patient" if source else "record_cluster",
                "known" if source else "unknown",
                features,
                outcomes,
                array_hash([x]),
                array_hash([y, obs]),
            )
        )
    return RoleData("synthetic:phase3b:test", roles), counters, arrays


def source_fixture(data):
    model = factory(15101).eval()
    return Source(
        model,
        Normalization.fit(data),
        state_hash(model),
        data.role("source_train").identity,
        data.role("source_stop").identity,
    )


def decision_fixture(source, data, mask):
    return Decision(
        canonical(
            dict(
                policy="A-record-v1",
                mask=mask,
                source_checkpoint=source.checkpoint_hash,
                normalization=source.normalization.identity,
                source_identity=data.role("source_train").identity,
                G_identity=data.role("G").identity,
                sampling_unit="record_cluster",
                patient_linkage="unknown",
            )
        )
    )
