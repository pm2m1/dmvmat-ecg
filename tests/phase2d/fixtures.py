import numpy as np
import torch
from torch import nn

from dmvmat_ecg.phase2.contracts import MODULES


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        for n in MODULES[:5]:
            setattr(
                self,
                n,
                nn.Sequential(nn.Linear(12, 4), nn.BatchNorm1d(4), nn.ReLU(), nn.Dropout(0.2)),
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

    def representations(self, wave, rhythm=None, spectrum=None):
        x = wave.mean(-1)
        r = {n: getattr(self, n)(x) for n in MODULES[:5]}
        r["fusion_1"] = self.fusion_1(torch.cat(list(r.values()), 1))
        r["fusion_2"] = self.fusion_2(r["fusion_1"])
        return r

    def forward(self, wave, rhythm=None, spectrum=None):
        return self.prediction_head(self.representations(wave, rhythm, spectrum)["fusion_2"])


def reader(row):
    rng = np.random.default_rng(row["ecg_id"] + 555)
    x = rng.normal(0, 0.1, (12, 1000)).astype("float32")
    # Fictional signal generation needs no label access (O outcomes are redacted).
    x[:3] += (row["ecg_id"] % 2) * 0.1
    x[3:6] += ((row["ecg_id"] // 2) % 2) * 0.1
    return x
