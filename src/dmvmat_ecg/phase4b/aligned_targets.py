"""Lambda selection on fold9/A only; production selector from aligned sources."""

import concurrent.futures
import json
import multiprocessing
from types import SimpleNamespace

import numpy as np
import torch

from dmvmat_ecg.phase2.adaptation import state_hash
from dmvmat_ecg.phase2.contracts import MODULES
from dmvmat_ecg.phase3b.evidence import Decision, RepresentationBank, analyze
from dmvmat_ecg.phase3b.models import inputs
from dmvmat_ecg.phase3b.roles import UnitRole, array_hash
from dmvmat_ecg.phase4.runtime import mask_index
from dmvmat_ecg.phase4.safety import new_json

from .common import RUN, load_source, setup
from .features import Features
from .source_data import SourceData
from .target_fit import fit, key, save_fit
from .train_alignment import tag


def choose_lambda(candidates, original_mean):
    passing = [
        row
        for row in candidates
        if row["fold9_macro_AUROC"] >= original_mean - 0.05
        and all(v > 0.5 for pair in row["per_seed_AUROC"] for v in pair)
    ]
    if not passing:
        return None
    return min(
        passing, key=lambda r: (-r["A_macro_AP"], -r["fold9_macro_AUROC"], r["lambda_value"])
    )["lambda_value"]


def full_candidate(s, d, lam):
    t = setup()
    return save_fit(t, s, d, 128, "full", aligned=tag(lam))


def replay(target, s, d, b, method, *, aligned, mask=127):
    original = save_fit(target, s, d, b, method, aligned=aligned, mask=mask)
    probabilities, r = fit(target, s, d, b, method, aligned=aligned, mask=mask)
    with np.load(RUN / "predictions" / (original["file"] + ".npz"), allow_pickle=False) as prior:
        if any(not np.array_equal(probabilities[n], prior[n]) for n in ("A", "B")):
            raise RuntimeError("STOP_ALIGNED_TARGET_REPRODUCTION:" + method)
    if r["state_hash"] != original["state_hash"] or r["history"] != original["history"]:
        raise RuntimeError("STOP_ALIGNED_TARGET_STATE_REPRODUCTION:" + method)
    return dict(
        method=method,
        status="PASS",
        repeat_seconds=r["runtime_seconds"],
        A_B_predictions_bitwise_identical=True,
        state_history_identical=True,
    )


def candidate_selection():
    t = setup()
    p = RUN / "LAMBDA_SELECTION.json"
    if p.exists():
        return json.loads(p.read_text())
    gate = RUN / "ALIGNED_FULL_REPRODUCIBILITY.json"
    if not gate.exists():
        new_json(gate, replay(t, 15101, 16101, 128, "full", aligned=tag(0.1)))
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=3, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        futures = [
            pool.submit(full_candidate, s, d, lam)
            for lam in (0.1, 1.0)
            for s in (15101, 15102, 15103)
            for d in (16101, 16102, 16103)
        ]
        for f in concurrent.futures.as_completed(futures):
            f.result()
    original = json.loads((RUN / "ORIGINAL_FOLD9_BASELINES.json").read_text())
    baseline = float(np.mean([r["macro_AUROC"] for r in original.values()]))
    rows = []
    for lam in (0.1, 1.0):
        entries = [
            json.loads((RUN / "aligned_sources" / f"{tag(lam)}_{s}.json").read_text())
            for s in (15101, 15102, 15103)
        ]
        fits = [
            json.loads(
                (RUN / "fits" / (key(s, d, 128, "full", aligned=tag(lam)) + ".json")).read_text()
            )
            for s in (15101, 15102, 15103)
            for d in (16101, 16102, 16103)
        ]
        rows.append(
            dict(
                lambda_value=lam,
                fold9_macro_AUROC=float(np.mean([r["fold9_macro_AUROC"] for r in entries])),
                per_seed_AUROC=[r["fold9_AUROC"] for r in entries],
                A_macro_AP=float(np.mean([r["A_macro_AP"] for r in fits])),
            )
        )
    selected = choose_lambda(rows, baseline)
    qualified = selected is not None
    if selected is None:
        # Complete the requested benchmark even when neither source preserves
        # source utility. This is explicitly a source-collapse descriptive arm.
        selected = min(
            rows, key=lambda r: (-r["fold9_macro_AUROC"], -r["A_macro_AP"], r["lambda_value"])
        )["lambda_value"]
    value = dict(
        status="SELECTED" if qualified else "SOURCE_COLLAPSE_DESCRIPTIVE_FALLBACK",
        source_collapse_screen_passed=qualified,
        selected_lambda=selected,
        candidates=rows,
        original_mean_fold9_AUROC=baseline,
        B_used_for_selection=False,
    )
    new_json(p, value)
    print(json.dumps(value, indent=2), flush=True)
    return value


def source_bank(model, norm, source, features):
    records = source.records["source_train"]
    canonical = {}
    for i, (record, group) in enumerate(zip(records["ids"], records["groups"], strict=True)):
        if group not in canonical or int(record) < int(records["ids"][canonical[group]]):
            canonical[group] = i
    ix = np.array([canonical[g] for g in sorted(canonical)], dtype=int)
    values = {n: [] for n in MODULES}
    model.eval()
    with torch.no_grad():
        for start in range(0, len(ix), 128):
            reps = model.representations(**features.batch("source_train", ix[start : start + 128]))
            for n in MODULES:
                values[n].append(reps[n].numpy())
    y = source.y["source_train"][ix]
    return RepresentationBank(
        {n: np.concatenate(chunks) for n, chunks in values.items()},
        y,
        np.ones(y.shape, dtype=bool),
        tuple(records["groups"][i] for i in ix),
        "source_train",
        "patient",
        "known",
        source.manifest["roles"]["source_train"]["identity"],
        state_hash(model),
        norm.identity,
    )


def decision(target, source, model, norm, bank, s, d, b, aligned):
    p = RUN / "aligned_decisions" / f"{aligned}_{s}_{d}_B{b}.json"
    ids = target.ids("G", d, b)
    x = target.signals("G", d, b)
    y = target.labels("G", d, b)
    observed = np.ones(y.shape, dtype=bool)
    role = UnitRole(
        "G",
        tuple(ids),
        tuple(ids),
        "record_cluster",
        "unknown",
        lambda: x,
        lambda: (y, observed),
        array_hash([x]),
        array_hash([y, observed]),
    )
    if p.exists():
        result = Decision(p.read_bytes())
    else:
        model.eval()
        with torch.no_grad():
            reps = model.representations(**inputs(model, norm.transform(x)))
        gb = RepresentationBank(
            {n: reps[n].numpy() for n in MODULES},
            y,
            observed,
            tuple(ids),
            "G",
            "record_cluster",
            "unknown",
            role.identity,
            state_hash(model),
            norm.identity,
        )
        result = analyze(bank, gb)
        new_json(p, result.report)
    src = SimpleNamespace(checkpoint_hash=state_hash(model), normalization=norm)
    source_identity = SimpleNamespace(identity=source.manifest["roles"]["source_train"]["identity"])
    data = SimpleNamespace(
        role=lambda name: (
            source_identity if name == "source_train" else role if name == "G" else None
        )
    )
    mask = result.validate(src, data)
    return mask_index(mask)


def selected_source_jobs(s, lam):
    torch.set_num_threads(4)
    t = setup()
    source = SourceData()
    features = Features()
    aligned = tag(lam)
    model, norm, _, _ = load_source(t, s, aligned)
    bank = source_bank(model, norm, source, features)
    save_fit(t, s, 16101, 0, "source", aligned=aligned)
    if s == 15101:
        gate = RUN / "ALIGNED_SOURCE_ONLY_REPRODUCIBILITY.json"
        if not gate.exists():
            new_json(gate, replay(t, s, 16101, 0, "source", aligned=aligned))
        m = decision(t, source, model, norm, bank, s, 16101, 128, aligned)
        gate = RUN / "ALIGNED_SELECTOR_REPRODUCIBILITY.json"
        if not gate.exists():
            new_json(gate, replay(t, s, 16101, 128, "aligned_selector", aligned=aligned, mask=m))
    for d in (16101, 16102, 16103):
        for b in (64, 128, 256):
            save_fit(t, s, d, b, "full", aligned=aligned)
            m = decision(t, source, model, norm, bank, s, d, b, aligned)
            save_fit(t, s, d, b, "aligned_selector", aligned=aligned, mask=m)
            print(f"ALIGNED_TARGET_COMPLETE seed={s} draw={d} budget={b} mask={m:07b}", flush=True)
    return dict(source_seed=s, status="COMPLETE")


def run():
    choice = candidate_selection()
    if choice["selected_lambda"] is not None:
        # Gate selector replay for one seed precedes the other source groups.
        selected_source_jobs(15101, choice["selected_lambda"])
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=2, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            futures = [
                pool.submit(selected_source_jobs, s, choice["selected_lambda"])
                for s in (15102, 15103)
            ]
            for f in futures:
                f.result()
    p = RUN / "ALIGNED_TARGETS_COMPLETE.json"
    if not p.exists():
        new_json(
            p,
            dict(
                status="COMPLETE",
                lambda_selection_status=choice["status"],
                selected_lambda=choice["selected_lambda"],
            ),
        )


if __name__ == "__main__":
    run()
