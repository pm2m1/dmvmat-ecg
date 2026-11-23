"""New signal-only capability; frozen V2 annotations only after prediction seal."""

import json
from pathlib import Path

import numpy as np
from scipy.io import whosmat

from dmvmat_ecg.phase3.signals import physical_wave, to_100hz_first10
from dmvmat_ecg.phase3b_eval.signals import physical_metadata
from dmvmat_ecg.phase3b_eval_v2.decoder import decode_labels, validity_gate
from dmvmat_ecg.phase3b_real.boundary import MAPPING_HASH
from dmvmat_ecg.phase4.safety import digest, exclusion_hash, new_json, project_json, sha

from .contract import RUN, save_array, utc
from .scope import BUNDLE, MAPPING, MEMBERSHIPS, RAW, capability


def signal_only_mat(path):
    variables = whosmat(path)
    if (
        len(variables) != 1
        or variables[0][0] != "val"
        or len(variables[0][1]) != 2
        or variables[0][1][0] != 12
    ):
        raise PermissionError("MAT is not an exclusively declared signal matrix")


class Provider:
    def __init__(self, scope, run=RUN):
        if scope.stage != "predict":
            raise PermissionError("No test membership or signals before108 replay matches")
        self.scope, self.folder = scope, Path(run)
        fields = project_json(
            MEMBERSHIPS,
            [
                ("membership", "O"),
                ("target_publisher_index_hash",),
                ("mapping_hash",),
                ("split_seed",),
            ],
        )
        self.ids = tuple(fields["membership", "O"])
        if (
            len(self.ids) != 3860
            or len(set(self.ids)) != 3860
            or fields["split_seed",] != 77101
            or fields["mapping_hash",] != MAPPING_HASH
        ):
            raise PermissionError("Frozen test role identity differs")
        excluded = set(json.loads((BUNDLE / "exclusion_hashes.json").read_text()))
        if {exclusion_hash(r) for r in self.ids} != excluded:
            raise PermissionError("Test roles differ from original exclusion binding")
        pool = set(json.loads((BUNDLE / "records.json").read_text()))
        if pool & set(self.ids):
            raise PermissionError("Adaptation/test overlap")
        if any(Path(r).is_absolute() or ".." in Path(r).parts for r in self.ids):
            raise PermissionError("Unsafe record path")
        self.scope.raw_paths = {RAW / (r + suffix) for r in self.ids for suffix in (".hea", ".mat")}
        if sha(RAW / "SHA256SUMS.txt") != fields["target_publisher_index_hash",]:
            raise PermissionError("Publisher index differs")
        # Publisher checksum metadata contains no outcomes. Do not open headers.
        self.publisher = {}
        for line in (RAW / "SHA256SUMS.txt").read_text().splitlines():
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                self.publisher[parts[1].lstrip(" *").removeprefix("./")] = parts[0]
        if sha(MAPPING) != MAPPING_HASH:
            raise PermissionError("Frozen label schema differs")
        # Exactly the original evaluator's canonical singleton grouping:
        # corrected_run.analyze(..., groups=eligible). No new duplicate search.
        self.groups = self.ids
        self.binding = dict(
            records=len(self.ids),
            record_order_sha256=digest(self.ids),
            groups_sha256=digest(self.groups),
            grouping="original canonical record IDs as in Experiment4 evaluator",
            projected_membership_sha256=digest({"/".join(k): v for k, v in fields.items()}),
            mapping_sha256=MAPPING_HASH,
        )

    def signals(self):
        x = np.empty((3860, 12, 1000), dtype=np.float32)
        prefix = {}
        for i, r in enumerate(self.ids):
            with capability("prefix"):
                meta = physical_metadata(
                    RAW / (r + ".hea"), self.folder / "physical_headers" / (digest(r) + ".hea")
                )
            if meta["duration"] < 10:
                raise PermissionError("Frozen physical rule yields a different3860-record cohort")
            mat = RAW / Path(r).parent / meta["signal_file"]
            if mat != RAW / (r + ".mat"):
                raise PermissionError("Header-declared signal identity differs")
            with capability("signal"):
                signal_only_mat(mat)  # BEFORE hashing/reading any MAT payload.
                if sha(mat) != self.publisher.get(str(mat.relative_to(RAW))):
                    raise PermissionError("Signal publisher checksum differs")
                x[i] = to_100hz_first10(physical_wave(RAW / r, meta), meta["fs"])
            prefix[r] = meta["physical_prefix_hash"]
            if (i + 1) % 250 == 0:
                print(f"SIGNAL_ONLY_PROGRESS {i + 1}/3860", flush=True)
        save_array(self.folder / "signals.npy", x)
        new_json(
            self.folder / "SIGNALS_COMPLETE.json",
            dict(
                timestamp=utc(),
                **self.binding,
                signals_sha256=sha(self.folder / "signals.npy"),
                physical_prefixes_sha256=digest(prefix),
                outcomes_read=False,
            ),
        )
        return x

    def outcomes(self, manifest):
        if manifest["test_binding"] != self.binding:
            raise PermissionError("Prediction/outcome row binding differs")
        self.scope.outcomes()
        mapping = json.loads(MAPPING.read_text())
        ys, obs = [], []
        for r in self.ids:
            with capability("annotation"):
                payload = (RAW / (r + ".hea")).read_bytes()
            import hashlib

            if hashlib.sha256(payload).hexdigest() != self.publisher.get(r + ".hea"):
                raise PermissionError("Post-seal header publisher checksum differs")
            a, b = decode_labels(payload, mapping)
            ys.append(a)
            obs.append(b)
        y, observed = np.asarray(ys, float), np.asarray(obs, bool)
        gate = validity_gate(y, observed)
        new_json(self.folder / "OUTCOME_VALIDITY.json", gate)
        if not gate["passed"]:
            raise PermissionError("V2 validity gate failed before metrics")
        save_array(self.folder / "outcomes.npy", y)
        save_array(self.folder / "observed.npy", observed)
        return y, observed, gate
