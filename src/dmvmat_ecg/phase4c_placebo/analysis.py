"""Six fixed validation contrasts; no test provider or outcome operation."""

import json

import numpy as np

from dmvmat_ecg.phase4.safety import digest, new_json, sha
from dmvmat_ecg.phase4b.common import Target
from dmvmat_ecg.phase4b.statistics import boots, intervals, measures
from dmvmat_ecg.phase4b.target_fit import key

from .data import OLD, ROOT, RUN, setup
from .target import name


def read_fit(root, filename, target, s, d, b, expected_method):
    report = json.loads((root / "fits" / (filename + ".json")).read_text())
    path = root / "predictions" / (filename + ".npz")
    if (
        report["status"] != "COMPLETE"
        or report["source_seed"] != s
        or report["draw_seed"] != d
        or report["budget"] != b
        or report["updates"] != 200
        or report["mask_index"] != 127
        or sha(path) != report["predictions_sha256"]
    ):
        raise PermissionError("Reference fit/prediction identity differs")
    if expected_method == "full":
        source_entry = target.sources[str(s)]
        compatible = report["method"] in {"full", "oracle"} and report["aligned"] is None
    elif expected_method == "aligned_full":
        source_entry = json.loads((OLD / "aligned_sources" / f"lambda0.1_{s}.json").read_text())
        compatible = report["method"] == "full" and report["aligned"] == "lambda0.1"
    elif expected_method == "placebo_full":
        source_entry = json.loads((RUN / "source_models" / f"placebo_s{s}.json").read_text())
        compatible = report["method"] == "placebo_full"
    else:
        raise ValueError("Unknown comparison method")
    if not compatible or report["source_sha256"] != source_entry["sha256"]:
        raise PermissionError("Model family/source checkpoint differs")
    with np.load(path, allow_pickle=False) as a:
        if set(a.files) != {"A", "B"}:
            raise PermissionError("Unexpected prediction roles")
        p = {n: a[n] for n in ("A", "B")}
    for role in p:
        if (
            p[role].shape != (500, 2)
            or p[role].dtype != np.float32
            or not np.isfinite(p[role]).all()
            or ((p[role] < 0) | (p[role] > 1)).any()
        ):
            raise PermissionError("Invalid validation probabilities")
    if not np.array_equal(measures(target.labels("A"), p["A"])[0], report["A_AP"]):
        raise PermissionError("Stored A score differs")
    return p, report


def run():
    setup("target")
    t = Target()
    if json.loads((RUN / "TARGETS_COMPLETE.json").read_text())["status"] != "COMPLETE":
        raise PermissionError("All validation fits must finish")
    destination = ROOT / "docs/phase4/PHASE4B_PLACEBO_AGGREGATES.json"
    if destination.exists():
        raise FileExistsError("Completed placebo analysis cannot be overwritten")
    binding = json.loads((OLD / "SPLIT_BINDING.json").read_text())
    if binding["bundle_sha256"] != t.identity or binding["split_sha256"] != digest(t.split):
        raise PermissionError("Old and new validation rows differ")
    y = t.labels("B")
    weights = (
        np.random.default_rng(20261127)
        .multinomial(500, np.full(500, 1 / 500), size=10000)
        .astype(np.int16)
    )
    metrics = []
    contrasts = []
    frozen = []
    for b in (64, 128, 256):
        points = {n: [] for n in ("full", "aligned_full", "placebo_full")}
        points_A = {n: [] for n in points}
        samples = {n: [] for n in points}
        for s in (15101, 15102, 15103):
            for d in (16101, 16102, 16103):
                files = {
                    "full": (
                        OLD,
                        key(s, d, 64, "oracle", mask=127) if b == 64 else key(s, d, b, "full"),
                    ),
                    "aligned_full": (OLD, key(s, d, b, "full", aligned="lambda0.1")),
                    "placebo_full": (RUN, name(s, d, b)),
                }
                for method, (root, filename) in files.items():
                    p, r = read_fit(root, filename, t, s, d, b, method)
                    points[method].append(measures(y, p["B"]))
                    points_A[method].append(measures(t.labels("A"), p["A"]))
                    samples[method].append(boots(y, p["B"], weights))
                    if method == "placebo_full":
                        frozen.append(
                            dict(
                                source_seed=s,
                                draw_seed=d,
                                budget=b,
                                report=filename + ".json",
                                report_sha256=sha(RUN / "fits" / (filename + ".json")),
                                predictions_sha256=r["predictions_sha256"],
                                source_sha256=r["source_sha256"],
                                state_hash=r["state_hash"],
                                normalization=r["normalization"],
                                identity=r["identity"],
                                history_sha256=digest(r["history"]),
                            )
                        )
        mean = {n: np.mean(v, axis=0) for n, v in points.items()}
        mean_A = {n: np.mean(v, axis=0) for n, v in points_A.items()}
        sampled = {n: np.mean(v, axis=0) for n, v in samples.items()}
        for method in mean:
            p = mean[method]
            metrics.append(
                dict(
                    budget=b,
                    method=method,
                    macro_AP=float(p[0].mean()),
                    macro_AUROC=float(p[1].mean()),
                    A_macro_AP=float(mean_A[method][0].mean()),
                    A_macro_AUROC=float(mean_A[method][1].mean()),
                    A_CD_AP=float(mean_A[method][0, 0]),
                    A_STTC_AP=float(mean_A[method][0, 1]),
                    CD_AP=float(p[0, 0]),
                    STTC_AP=float(p[0, 1]),
                    CD_AUROC=float(p[1, 0]),
                    STTC_AUROC=float(p[1, 1]),
                )
            )
        for a, z in (("aligned_full", "placebo_full"), ("placebo_full", "full")):
            p = mean[a] - mean[z]
            v = sampled[a] - sampled[z]
            contrasts.append(
                dict(
                    budget=b,
                    contrast=a + "-minus-" + z,
                    AP_difference=float(p[0].mean()),
                    AP_ci=intervals(v[:, 0].mean(-1)),
                    AUROC_difference=float(p[1].mean()),
                    AUROC_ci=intervals(v[:, 1].mean(-1)),
                    per_label={
                        label: dict(
                            AP_difference=float(p[0, j]),
                            AP_ci=intervals(v[:, 0, j]),
                            AUROC_difference=float(p[1, j]),
                            AUROC_ci=intervals(v[:, 1, j]),
                        )
                        for j, label in enumerate(("CD", "STTC"))
                    },
                )
            )
        print(f"PLACEBO_VALIDATION_BOOTSTRAP_COMPLETE budget={b}", flush=True)
    original = json.loads((OLD / "ORIGINAL_FOLD9_BASELINES.json").read_text())
    aligned = json.loads((OLD / "ALIGNMENT_GRID_COMPLETE.json").read_text())
    sources = []
    for s in (15101, 15102, 15103):
        p = RUN / "source_models" / f"placebo_s{s}.json"
        r = json.loads(p.read_text())
        cp = RUN / r["checkpoint"]
        if sha(cp) != r["sha256"] or r["CPSC_inputs_read"]:
            raise PermissionError("Source-only checkpoint proof differs")
        a = next(v for v in aligned["candidates"] if v["seed"] == s and v["lambda_value"] == 0.1)
        sources.append(
            dict(
                source_seed=s,
                original_macro_AUROC=original[str(s)]["macro_AUROC"],
                aligned_macro_AUROC=a["fold9_macro_AUROC"],
                placebo_macro_AUROC=r["fold9_macro_AUROC"],
                placebo_per_label_AUROC=r["fold9_AUROC"],
                selected_epoch=r["selected_epoch"],
                source_sha256=r["sha256"],
                source_state_hash=r["state_hash"],
                source_runtime_seconds=r["runtime_seconds"],
            )
        )
    result = dict(
        schema="phase4c-placebo-validation-v1",
        lambda_value=0.1,
        sources=sources,
        metrics=metrics,
        contrasts=contrasts,
        bootstrap=dict(
            draws=10000,
            seed=20261127,
            unit="same split-B record across all models/labels/episodes",
            pointwise=True,
        ),
        CPSC_pretraining_inputs_used=False,
        test_access=False,
        selection_on_B=False,
    )
    new_json(destination, result)
    new_json(
        RUN / "FROZEN_VALIDATION_REFERENCES.json",
        dict(
            schema="placebo-full-replay-references-v1",
            bundle_sha256=t.identity,
            split_sha256=digest(t.split),
            references=frozen,
            source_input_manifest_sha256=sha(RUN / "SOURCE_INPUTS.json"),
            test_access=False,
        ),
    )
    append_report(result)
    print(
        json.dumps(
            dict(source_AUROC=[r["placebo_macro_AUROC"] for r in sources], contrasts=contrasts),
            indent=2,
        ),
        flush=True,
    )


def append_report(r):
    path = ROOT / "docs/phase4/PHASE4B_RESULTS.md"
    b = json.loads((RUN / "PREVIOUS_RESULTS_BINDING.json").read_text())
    if sha(path) != b["sha256"] or path.stat().st_size != b["bytes"]:
        raise PermissionError("Previous report changed; do not overwrite or blindly append")
    lines = [
        "",
        "## Placebo-alignment control",
        "",
        "This section was added by the Phase4C amendment after the original Phase4B study. [Prospective validation protocol](PLACEBO_ALIGNMENT_PROTOCOL.md) · [Full new aggregates](PHASE4B_PLACEBO_AGGREGATES.json). These are reused validation B results, not CPSC test findings. No test membership, signal, label, prediction, metric or seal was accessed.",
        "",
        "Three fresh placebo sources used only PTB-XL folds1–8 training and fold9 selection. Everything matched lambda=0.1 alignment except that the second independent 32-record stream was PTB training data rather than unlabelled CPSC. The BCE/source batch stayed128; all seven module-wise MMD terms, four bandwidth multipliers, seeds, 50 epochs and source-only normalization were fixed. All sources are reported without selecting a favorable seed or lambda.",
        "",
        "| Seed | Original fold9 macro AUROC | Aligned lambda=0.1 | Placebo | Placebo selected epoch |",
        "|---:|---:|---:|---:|---:|",
    ]
    for s in r["sources"]:
        lines.append(
            f"| {s['source_seed']} | {s['original_macro_AUROC']:.6f} | {s['aligned_macro_AUROC']:.6f} | {s['placebo_macro_AUROC']:.6f} | {s['selected_epoch']} |"
        )
    lines += [
        "",
        f"Mean placebo fold9 macro AUROC: **{np.mean([s['placebo_macro_AUROC'] for s in r['sources']]):.6f}**. Checkpoints, SHA-256 values and full fold9 summaries are retained privately; no source-training CPSC input was permitted by the OS guard.",
        "",
        "Full FT reused the same nine episodes and F+G draws, 64/128/256 optimization labels, batch32 replacement sampling, original seeds and 200 fixed AdamW updates. A/B are the original500/500 splits. No A/B result selected a source, hyperparameter or epoch; source checkpoints use fold9 AP with BCE ties as before.",
        "",
        "| Budget | Method | A macro AP | A macro AUROC | B macro AP | B macro AUROC |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for v in r["metrics"]:
        lines.append(
            f"| {v['budget']} | {v['method']} | {v['A_macro_AP']:.6f} | {v['A_macro_AUROC']:.6f} | {v['macro_AP']:.6f} | {v['macro_AUROC']:.6f} |"
        )
    lines += [
        "",
        "| Budget | Contrast | Macro AP difference [pointwise 95% CI] | Macro AUROC difference [pointwise 95% CI] |",
        "|---:|---|---|---|",
    ]
    for v in r["contrasts"]:
        lines.append(
            f"| {v['budget']} | {v['contrast']} | {v['AP_difference']:+.6f} [{v['AP_ci']['low']:+.6f}, {v['AP_ci']['high']:+.6f}] | {v['AUROC_difference']:+.6f} [{v['AUROC_ci']['low']:+.6f}, {v['AUROC_ci']['high']:+.6f}] |"
        )
    lines += [
        "",
        "Intervals use10,000 common paired B-record resamples (seed20261127), fixed fitted models and equal means of nine episode-wise metrics, not prediction ensembles or independent cohorts. Per-label differences/support are in the new JSON. These exploratory pointwise validation contrasts are not the future test Holm family.",
        "",
        "The aligned-minus-placebo contrast probes target-domain exposure beyond matched within-source MMD regularization/extra forward passes. It still includes domain-dependent BN effects; it is not proof of an MMD-gradient-only mechanism. Unfavorable or null placebo findings do not change the fixed confirmatory roster.",
        "",
        "AdaBN+full is removed from the planned Phase4C roster because its reported ranking results coincide with full FT after BN-statistic updating during200 steps. This does not assert universal state/score equality: only10/27 previous A/B vector pairs were bitwise-identical. Its historical results above remain unchanged.",
        "",
        "The amended confirmatory protocol keeps the primary M3-minus-M2 at128 labels and substitutes placebo+full as M4 for the existing key secondary and unchanged95-secondary Holm family. Future target-state replay is authentication only; all81 fits must reproduce stored A/B predictions bitwise and state/recipe bindings before test access. No confirmation replay or test evaluation was run here.",
    ]
    with path.open("a") as out:
        out.write("\n".join(lines) + "\n")
    if sha_bytes(path.read_bytes()[: b["bytes"]]) != b["sha256"]:
        raise PermissionError("Original report prefix no longer intact")


def sha_bytes(data):
    import hashlib

    return hashlib.sha256(data).hexdigest()


if __name__ == "__main__":
    run()
