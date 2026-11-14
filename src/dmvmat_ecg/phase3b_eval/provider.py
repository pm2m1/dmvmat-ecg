"""O-only evaluation adapter; physical fields first, diagnoses exclusively post-seal."""

import io
from pathlib import Path

import numpy as np

from dmvmat_ecg.phase3.signals import physical_wave, to_100hz_first10
from dmvmat_ecg.phase3b.jobs import atomic_new, file_hash, save_json
from dmvmat_ecg.phase3b.protocol import digest

from .access import ROOT, bytes_hash, read_json
from .signals import decode_labels, physical_metadata


class OProvider:
    def __init__(self, root, session):
        self.root = Path(root).resolve()
        self.study = self.root / "private/phase3b_cpsc_v1"
        self.folder = self.study / "evaluation_v1"
        if (
            self.root != ROOT
            or session.folder.resolve() != self.folder.resolve()
            or session.anchor.resolve() != self.study.resolve()
        ):
            raise PermissionError("Only the canonical, anchored O evaluation is permitted")
        self.session = session
        self.contract = read_json(self.study / "MEMBERSHIPS.json")
        self.ids = tuple(self.contract["membership"]["O"])
        self.raw = self.root / "data/raw/cinc2021_v1_0_3"
        if file_hash(self.raw / "SHA256SUMS.txt") != self.contract["target_publisher_index_hash"]:
            raise PermissionError("Frozen target publisher checksum index changed")
        self.publisher = {}
        for line in (self.raw / "SHA256SUMS.txt").read_text().splitlines():
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                self.publisher[parts[1].lstrip(" *").removeprefix("./")] = parts[0]
        if len(self.ids) != 3860 or len(set(self.ids)) != 3860:
            raise PermissionError("Frozen O membership differs")
        if set(self.ids).intersection(self.contract["membership"]["FEAS"]):
            raise PermissionError("FEAS cannot enter O")

    def _permit(self, record):
        self.session.check()
        if record not in set(self.ids):
            raise PermissionError("Only frozen O IDs are available to the evaluation adapter")

    def signals(self):
        self.session.check()
        metas = {}
        excluded = []
        eligible = []
        for record in self.ids:
            self._permit(record)
            meta = physical_metadata(
                self.raw / (record + ".hea"),
                self.folder / "physical_headers" / (digest(record) + ".hea"),
            )
            metas[record] = meta
            (excluded if meta["duration"] < 10 else eligible).append(record)
        # All exclusions fixed from physical header data before any scoring/outcome access.
        self.eligible = tuple(eligible)
        self.metas = metas
        save_json(
            self.folder / "HEADER_ONLY_MEMBERSHIPS.json",
            dict(
                eligible=eligible,
                excluded_short=excluded,
                metadata=metas,
                header_policy="Exactly first13 unbuffered WFDB lines; diagnosis comments not read",
            ),
        )
        waves = []
        for record in eligible:
            self._permit(record)
            meta = metas[record]
            mat = self.raw / Path(record).parent / meta["signal_file"]
            key = str(mat.relative_to(self.raw))
            if file_hash(mat) != self.publisher.get(key):
                raise PermissionError("O signal checksum differs: " + key)
            waves.append(to_100hz_first10(physical_wave(self.raw / record, meta), meta["fs"]))
        x = np.stack(waves)
        payload = io.BytesIO()
        np.save(payload, x, allow_pickle=False)
        atomic_new(self.folder / "O_signals.npy", payload.getvalue())
        summary = dict(
            total_O=len(self.ids),
            evaluated=len(eligible),
            excluded_short=len(excluded),
            truncated_first10=sum(metas[i]["duration"] > 10 for i in eligible),
            sampling_rates=sorted({metas[i]["fs"] for i in eligible}),
            physical_prefix_membership_hash=digest(
                {i: metas[i]["physical_prefix_hash"] for i in self.ids}
            ),
            eligible_membership_hash=digest(eligible),
            outcomes_read=False,
        )
        save_json(self.folder / "HEADER_ONLY_SUMMARY.json", summary)
        return x, summary

    def outcomes_once(self):
        def load():
            frozen = read_json(self.root / "results/phase3a/LABEL_MAPPING.json")
            y = []
            observed = []
            for record in self.eligible:
                # The authoritative outcome marker has already been consumed. Read once.
                payload = (self.raw / (record + ".hea")).read_bytes()
                if bytes_hash(payload) != self.publisher.get(record + ".hea"):
                    raise PermissionError("O header release checksum differs")
                a, b = decode_labels(payload, frozen)
                y.append(a)
                observed.append(b)
            y, observed = np.asarray(y, dtype=float), np.asarray(observed, dtype=bool)
            stream = io.BytesIO()
            np.savez_compressed(stream, y=y, observed=observed, ids=np.asarray(self.eligible))
            atomic_new(self.folder / "O_outcomes_once.npz", stream.getvalue())
            return y, observed

        return self.session.consume(load)
