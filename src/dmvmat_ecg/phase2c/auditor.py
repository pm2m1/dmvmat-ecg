"""Restricted cohort-auditor interface: aggregates, not predictive access.

Like prior guards, this controls supported Python paths, not an adversarial user
with arbitrary filesystem privileges. Execution endpoints deliberately do not exist.
"""

import copy
import json
import sqlite3
from pathlib import Path

from dmvmat_ecg.phase2b.contract import digest, file_hash, jobs
from dmvmat_ecg.phase2b.ledger import atomic_bytes
from dmvmat_ecg.phase2b.preflight import load_metadata

from .splits import SPEC_HASH, VERSION, build_candidate, support, validate

OLD_HASH = "b5b3dcd127fa6d15317e82f3d704f25b193f269651db094a85e9fc6429ce644f"


def immutable_json(path, obj):
    data = json.dumps(obj, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
    path = Path(path)
    if path.exists():
        if path.read_bytes() != data:
            raise PermissionError("Existing immutable candidate artifact differs")
    else:
        atomic_bytes(path, data)
    return file_hash(path)


def ledger_snapshot(path):
    """Read only the Phase2B identity/state/counters; never old fold10 memberships."""
    db = sqlite3.connect(f"file:{Path(path).resolve()}?mode=ro", uri=True)
    try:
        meta = dict(
            db.execute(
                "SELECT k,v FROM meta WHERE k IN ('identity','phase','development_authorization','evaluation_authorization')"
            )
        )
        parsed = {k: json.loads(v) for k, v in meta.items()}
        return dict(
            **parsed,
            attempts=db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0],
            evaluation_accesses=db.execute("SELECT COUNT(*) FROM access").fetchone()[0],
        )
    finally:
        db.close()


def fit_matrix(manifest):
    validate(manifest)
    output = []
    for name, spec in jobs().items():
        roles = set(spec["fit_roles"] + [spec["stop_role"]])
        if spec.get("evidence_role"):
            roles.add(spec["evidence_role"])
        target = {
            r["patient_id"]
            for k in roles
            if not k.startswith("source_")
            for r in manifest["roles"][k]
        }
        with_c = target | {r["patient_id"] for r in manifest["roles"]["C"]}
        output.append(
            dict(
                job=name,
                spec=spec,
                role_hashes={k: manifest["role_hashes"][k] for k in sorted(roles)},
                target_labels_before_C=len(target),
                target_labels_with_C=len(with_c),
            )
        )
    return dict(planned=33, ceiling=36, reserve=3, matrix=output, attempts_reserved=0)


def context(manifest, code_hash, dependency_hash):
    validate(manifest)
    return dict(
        namespace="draft:phase2c-joint-pattern-v1",
        status="draft",
        split_version=VERSION,
        spec_hash=SPEC_HASH,
        manifest_hash=digest(manifest),
        dataset=manifest["provenance"],
        domain_definition=dict(source_age=[18, 59], target_age=[60, 89]),
        labels=["CD", "STTC"],
        role_hashes=manifest["role_hashes"],
        patient_manifest_hash=manifest["population"]["canonical_membership_hash"],
        method_policy="unchanged Phase2 EvidenceConfig A/B",
        preprocessing_policy="fresh source_train scalar normalization only; no historical artifacts",
        fit_matrix_hash=digest(fit_matrix(manifest)),
        code_hash=code_hash,
        dependency_hash=dependency_hash,
        predictor_attempts=0,
        evaluation_accesses=0,
        training_authorization=False,
        evaluation_authorization=False,
    )


class DraftBindings:
    """Verify lineage and deny all predictive operations for this unapproved draft."""

    def __init__(self, manifest, binding, *, files=None):
        validate(manifest)
        expected = context(manifest, binding["code_hash"], binding["dependency_hash"])
        if binding != expected:
            raise PermissionError("Mixed split/version/role/configuration context")
        self._manifest = copy.deepcopy(manifest)
        self._binding = copy.deepcopy(binding)
        self._files = dict(files or {})

    def verify(self):
        if digest(self._manifest) != self._binding["manifest_hash"]:
            raise PermissionError("Changed patient/group manifest")
        if any(file_hash(p) != h for p, h in self._files.items()):
            raise PermissionError("Pinned source/configuration integrity changed")
        validate(self._manifest)

    def request(self, role, operation):
        self.verify()
        if operation != "audit_counts" or role not in self._manifest["roles"]:
            raise PermissionError(
                "DRAFT: real fitting, signal, label-worker and evaluation access DENIED"
            )
        return copy.deepcopy(self._manifest["summary"][role])

    def development_view(self):
        """No record/patient identities or outcomes, including O, reach workers."""
        self.verify()
        return dict(
            split_version=VERSION,
            manifest_hash=self._binding["manifest_hash"],
            status="DRAFT",
            counts=copy.deepcopy(self._manifest["summary"]),
            role_hashes=copy.deepcopy(self._manifest["role_hashes"]),
            training_authorization=False,
            evaluation_authorization=False,
        )

    def verify_fresh_source_lineage(self, lineage):
        self.verify()
        expected = dict(
            namespace=self._binding["namespace"],
            split_version=VERSION,
            manifest_hash=self._binding["manifest_hash"],
            labels=["CD", "STTC"],
            origin="random-scratch",
            fresh=True,
            source_training_hash=self._manifest["role_hashes"]["source_train"],
            source_selection_hash=self._manifest["role_hashes"]["source_stop"],
            preprocessing_fit_role="source_train",
            preprocessing_membership_hash=self._manifest["role_hashes"]["source_train"],
            code_hash=self._binding["code_hash"],
            dependency_hash=self._binding["dependency_hash"],
        )
        if any(lineage.get(k) != v for k, v in expected.items()):
            raise PermissionError("Historical/foreign checkpoint or normalization lineage rejected")
        # Identity verification alone can never turn this draft into a fitted artifact.
        return True


def generate_metadata_candidate(root):
    """The only real entry point: release-verified folds1–9 metadata, no signals."""
    root = Path(root).resolve()
    rows, provenance = load_metadata(root)
    old_path = root / "private/phase2b_internal/preflight" / f"{OLD_HASH}.json"
    old = json.loads(old_path.read_text())
    if digest(old) != OLD_HASH or old["provenance"] != provenance:
        raise PermissionError("Original candidate/dataset differs")
    new = build_candidate(rows, provenance)
    # Repeating precisely the same contract is a determinism check, not a seed trial.
    if new != build_candidate(rows, provenance):
        raise AssertionError("Nondeterministic candidate")
    original_canonical = {
        r["ecg_id"]: r
        for name in ["source_train", "source_stop", "O", "E_reserve", "C_reserve", "A"]
        for r in old["roles"][name]
    }
    current_canonical = {
        r["ecg_id"]: r
        for name in ["source_train", "source_stop", "O", "E_reserve", "C_reserve", "A"]
        for r in new["roles"][name]
    }
    if original_canonical != current_canonical:
        raise PermissionError("Eligibility/canonical population changed")
    for domain, names in [
        ("source", ["source_train", "source_stop"]),
        ("target", ["O", "E_reserve", "C_reserve", "A"]),
    ]:
        if {r["patient_id"] for k in names for r in old["roles"][k]} != {
            r["patient_id"] for k in names for r in new["roles"][k]
        }:
            raise PermissionError("Source/target domain boundary changed")
    old_counts = {k: support(v) for k, v in old["roles"].items()}
    overlap = {
        name: len(
            {r["patient_id"] for r in old["roles"][name]}
            & {r["patient_id"] for r in new["roles"][name]}
        )
        for name in old["roles"]
    }
    return new, dict(
        original_manifest_hash=OLD_HASH,
        original_role_counts=old_counts,
        old_new_same_role_patients=overlap,
        canonical_population_unchanged=True,
        domains_unchanged=True,
    )
