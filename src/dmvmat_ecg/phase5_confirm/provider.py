"""Georgia test signals only until the complete190-predictor seal."""

import hashlib
import json
from pathlib import Path

import numpy as np

from dmvmat_ecg.phase3.signals import physical_wave, to_100hz_first10
from dmvmat_ecg.phase3b_eval.signals import physical_metadata
from dmvmat_ecg.phase3b_eval_v2.decoder import decode_labels, validity_gate
from dmvmat_ecg.phase3b_real.boundary import MAPPING_HASH
from dmvmat_ecg.phase4.safety import digest, exclusion_hash, new_json, project_json, sha
from dmvmat_ecg.phase4c_confirm.contract import save_array, utc
from dmvmat_ecg.phase4c_confirm.provider import signal_only_mat
from dmvmat_ecg.phase4c_confirm.scope import capability
from dmvmat_ecg.phase5.common import GEORGIA

from .contract import ASSIGNMENT, AUDIT, MAPPING, RAW, RUN


class Provider:
    def __init__(self, scope):
        if scope.stage != "predict":
            raise PermissionError("No test signals before all180 replay matches")
        self.scope = scope
        m = project_json(
            ASSIGNMENT, [("test",), ("test_sha256",), ("pool_sha256",), ("label_blind",)]
        )
        self.ids = tuple(m["test",])
        self.groups = self.ids
        if (
            len(self.ids) != 4081
            or len(set(self.ids)) != 4081
            or not m["label_blind",]
            or digest(self.ids) != m["test_sha256",]
        ):
            raise PermissionError("Frozen label-blind test identity differs")
        if any(
            not r.startswith("training/georgia/") or ".." in Path(r).parts or Path(r).is_absolute()
            for r in self.ids
        ):
            raise PermissionError("Non-Georgia/unsafe test identity")
        excluded = set(json.loads((GEORGIA / "exclusion_hashes.json").read_text()))
        if {exclusion_hash(r) for r in self.ids} != excluded:
            raise PermissionError("Test exclusions differ")
        pool = json.loads((GEORGIA / "records.json").read_text())
        if set(pool) & set(self.ids) or digest(pool) != m["pool_sha256",]:
            raise PermissionError("Pool/test binding differs")
        expected = project_json(AUDIT, [("publisher_index_sha256",)])["publisher_index_sha256",]
        if sha(RAW / "SHA256SUMS.txt") != expected or sha(MAPPING) != MAPPING_HASH:
            raise PermissionError("Original publisher index/mapping differs")
        self.publisher = {}
        for line in (RAW / "SHA256SUMS.txt").read_text().splitlines():
            h, name = line.split(maxsplit=1)
            self.publisher[name.lstrip(" *").removeprefix("./")] = h
        scope.raw_paths = {RAW / (r + suffix) for r in self.ids for suffix in (".hea", ".mat")}
        self.binding = dict(
            records=4081,
            record_order_sha256=digest(self.ids),
            groups_sha256=digest(self.groups),
            assignment_sha256=sha(ASSIGNMENT),
            mapping_sha256=MAPPING_HASH,
            publisher_index_sha256=expected,
            grouping="historical canonical native-duplicate representative IDs",
        )

    def signals(self):
        x = np.empty((4081, 12, 1000), np.float32)
        prefixes = {}
        for i, r in enumerate(self.ids):
            with capability("prefix"):
                meta = physical_metadata(
                    RAW / (r + ".hea"), RUN / "physical_headers" / (digest(r) + ".hea")
                )
            mat = RAW / Path(r).parent / meta["signal_file"]
            if meta["duration"] < 10 or mat != RAW / (r + ".mat"):
                raise PermissionError("Frozen physical eligibility differs; no silent exclusions")
            with capability("signal"):
                signal_only_mat(mat)
                if sha(mat) != self.publisher.get(r + ".mat"):
                    raise PermissionError("Signal publisher hash differs")
                x[i] = to_100hz_first10(physical_wave(RAW / r, meta), meta["fs"])
            prefixes[r] = meta["physical_prefix_hash"]
            if (i + 1) % 250 == 0:
                print(f"GEORGIA_SIGNAL_ONLY {i + 1}/4081", flush=True)
        save_array(RUN / "signals.npy", x)
        new_json(
            RUN / "SIGNALS_COMPLETE.json",
            dict(
                timestamp=utc(),
                test_binding=self.binding,
                signals_sha256=sha(RUN / "signals.npy"),
                physical_prefixes_sha256=digest(prefixes),
                labels_read=False,
            ),
        )
        return x

    def outcomes(self, manifest):
        if manifest["test_binding"] != self.binding:
            raise PermissionError("Outcome row binding differs")
        self.scope.outcomes()
        frozen = json.loads(MAPPING.read_text())
        labels = []
        observed = []
        for r in self.ids:
            with capability("annotation"):
                payload = (RAW / (r + ".hea")).read_bytes()
            if hashlib.sha256(payload).hexdigest() != self.publisher.get(r + ".hea"):
                raise PermissionError("Post-seal header publisher hash differs")
            a, b = decode_labels(payload, frozen)
            labels.append(a)
            observed.append(b)
        y, obs = np.asarray(labels, float), np.asarray(observed, bool)
        gate = validity_gate(y, obs)
        new_json(RUN / "OUTCOME_VALIDITY.json", gate)
        if not gate["passed"]:
            raise PermissionError("V2 validity gate failed before metrics")
        save_array(RUN / "outcomes.npy", y)
        save_array(RUN / "observed.npy", obs)
        return y, obs
