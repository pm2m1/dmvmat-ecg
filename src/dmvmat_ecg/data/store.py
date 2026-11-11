"""Role-gated caches; the decoder/auditor is separate from predictor access."""

import numpy as np
from torch.utils.data import Dataset

from dmvmat_ecg.preprocessing.signals import STARTS, augment, normalize, views
from dmvmat_ecg.utils.io import sha


class Store:
    def __init__(self, study):
        self.study = study

    def load(self, role):
        self.study.require(role)
        manifest = self.study.get("dataset")
        root = self.study.root / "private/cache"
        for name in (f"{role}_x.npy", f"{role}_meta.npz"):
            if sha(root / name) != manifest["cache_hashes"][name]:
                raise PermissionError("Cache changed")
        x = np.load(root / f"{role}_x.npy", mmap_mode="r", allow_pickle=False)
        meta = np.load(root / f"{role}_meta.npz", allow_pickle=False)
        expected = self.study.get("memberships")[role]
        if meta["ecg_id"].tolist() != expected:
            raise PermissionError("Role membership mismatch")
        allowed = range(1, 9) if role == "train" else (9,) if role == "validation" else (10,)
        if (
            not np.isin(meta["fold"], list(allowed)).all()
            or meta["y"].shape != (len(x), 5)
            or not np.isin(meta["y"], [0, 1]).all()
        ):
            raise PermissionError("Fold/target schema mismatch")
        return x, {k: meta[k] for k in meta.files}


class CropDataset(Dataset):
    def __init__(
        self,
        x,
        y,
        ids,
        stats,
        *,
        training,
        multiview=False,
        augmentation=False,
        seed=101,
        window=1000,
    ):
        if augmentation and not training:
            raise PermissionError("No held-out augmentation")
        self.x, self.y, self.ids, self.stats = x, y, np.asarray(ids), stats
        self.training, self.multiview, self.augmentation = training, multiview, augmentation
        self.seed, self.epoch = seed, 0
        if window not in (250, 1000):
            raise ValueError("Unregistered window length")
        self.window = window
        self.starts = STARTS if window == 250 else (0,)

    def __len__(self):
        return len(self.x) if self.training else len(self.x) * len(self.starts)

    def __getitem__(self, idx):
        record = idx if self.training else idx // len(self.starts)
        rng = np.random.default_rng(self.seed + 100003 * self.epoch + int(self.ids[record]))
        # Upstream excludes the last possible random start (749 is its maximum).
        start = (
            (int(rng.integers(0, 1000 - self.window)) if self.window < 1000 else 0)
            if self.training
            else self.starts[idx % len(self.starts)]
        )
        x = np.array(self.x[record, :, start : start + self.window], copy=True)
        if self.augmentation:
            x = augment(
                x,
                np.random.default_rng(
                    self.seed + 700001 + 100003 * self.epoch + int(self.ids[record])
                ),
                training=True,
                state=self.stats,
            )
        wave = normalize(x, self.stats)
        inputs = {"wave": wave}
        if self.multiview:
            _, rhythm, spectrum = views(wave)
            inputs.update(rhythm=rhythm, spectrum=spectrum)
        return inputs, np.array(self.y[record], dtype=np.float32), record
