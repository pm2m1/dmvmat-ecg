"""Authenticate all180 Georgia states against the frozen validation release."""

import io
import json

import torch

from dmvmat_ecg.phase4.safety import digest, project_json, sha
from dmvmat_ecg.phase5 import sources, targets
from dmvmat_ecg.phase5.common import ARMS, BUDGETS, DRAWS, GEORGIA, SEEDS
from dmvmat_ecg.phase5.georgia import Target
from dmvmat_ecg.phase5.replay import replay

from .contract import ASSIGNMENT, DEV, ROOT, Scope


def release():
    p = project_json(
        DEV / "VALIDATION_RELEASE_BINDING.json",
        [
            ("sources",),
            ("reference_fits", "Georgia"),
            ("Georgia_bundle_sha256",),
            ("protocol_sha256",),
            ("report_sha256",),
        ],
    )
    if p["Georgia_bundle_sha256",] != sha(GEORGIA / "MANIFEST.json"):
        raise PermissionError("Frozen Georgia bundle differs")
    if p["protocol_sha256",] != sha(ROOT / "docs/phase5/PHASE5_GEORGIA_PROTOCOL.md") or p[
        "report_sha256",
    ] != sha(ROOT / "docs/phase5/PHASE5_VALIDATION_RESULTS.md"):
        raise PermissionError("Frozen public protocol/evidence differs")
    return p


def preflight():
    frozen = release()
    manifest = json.loads((GEORGIA / "MANIFEST.json").read_text())
    if sha(ASSIGNMENT) != manifest["assignment_sha256"]:
        raise PermissionError("ID-only assignment hash differs")
    if json.loads((DEV / "SOURCE_CODE_BINDING.json").read_text()) != sources.binding():
        raise PermissionError("Frozen source numerical code differs")
    if json.loads((DEV / "Georgia/TARGET_CODE_BINDING.json").read_text()) != targets.binding():
        raise PermissionError("Frozen target numerical code differs")
    rows = [sources.descriptor(a, s)[3] for s in SEEDS for a in ARMS]
    if rows != frozen["sources",]:
        raise PermissionError("Frozen source roster differs")
    refs = frozen["reference_fits", "Georgia"]
    if len(refs) != 180:
        raise PermissionError("Incomplete frozen validation references")
    t = Target()
    found = set()
    for ref in refs:
        path = ROOT / "private/phase5_replication_run" / ref["report"]
        if not path.resolve().is_relative_to(DEV / "Georgia/fits") or sha(path) != ref["sha256"]:
            raise PermissionError("Validation report hash/path differs")
        r = json.loads(path.read_text())
        arm, s, d, b = (r[k] for k in ("arm", "source_seed", "draw_seed", "budget"))
        _, check = targets.read(t, "Georgia", arm, s, d, b)
        expected = digest(
            dict(
                hospital="Georgia",
                bundle=t.identity,
                split=t.split,
                arm=arm,
                s=s,
                d=d,
                b=b,
                source=r["source_sha256"],
                code=targets.binding(),
            )
        )
        if (
            check["identity"] != expected
            or r["state_hash"] != ref["state_hash"]
            or r["predictions_sha256"] != ref["predictions_sha256"]
            or r["reused_historical_validation"]
        ):
            raise PermissionError("Reference identity/state differs")
        found.add((arm, s, d, b))
    expected = {(a, s, d, b) for a in ARMS for s in SEEDS for d in DRAWS for b in BUDGETS}
    if found != expected:
        raise PermissionError("Missing/duplicate validation jobs")
    return dict(
        sources=rows,
        references=180,
        bundle_sha256=t.identity,
        split_sha256=digest(t.split),
        release_sha256=sha(DEV / "VALIDATION_RELEASE_BINDING.json"),
        assignment_sha256=sha(ASSIGNMENT),
    )


def job(args):
    Scope().install()
    arm, s, d, b = args
    model, norm, r = replay("Georgia", arm, s, d, b)
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    return stream.getvalue(), dict(
        method="b" if arm == "plain" else "c",
        arm=arm,
        source_seed=s,
        draw_seed=d,
        budget=b,
        source_sha256=r["source_sha256"],
        state_hash=r["state_hash"],
        normalization=norm.__dict__,
        history_sha256=digest(r["history"]),
        A_bitwise_match=True,
        B_bitwise_match=True,
        state_history_recipe_match=True,
    )


def source_states():
    rows = {}
    for s in SEEDS:
        model, norm, _, r = sources.descriptor("plain", s)
        stream = io.BytesIO()
        torch.save(model.state_dict(), stream)
        rows[f"a/s{s}"] = (
            stream.getvalue(),
            dict(
                method="a",
                source_seed=s,
                source_sha256=r["sha256"],
                state_hash=r["state_hash"],
                normalization=norm.__dict__,
            ),
        )
    return rows
