"""Nine fixed validation contrasts; no test provider or outcome operation."""

import json

import numpy as np

from dmvmat_ecg.phase4.safety import digest, new_json, sha
from dmvmat_ecg.phase4b.common import Target
from dmvmat_ecg.phase4b.statistics import boots, intervals, measures
from dmvmat_ecg.phase4b.target_fit import key
from dmvmat_ecg.phase4c_placebo.target import name as placebo_name

from .data import OLD, PLACEBO, ROOT, RUN, setup
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
        source_entry = json.loads((PLACEBO / "source_models" / f"placebo_s{s}.json").read_text())
        compatible = report["method"] == "placebo_full"
    elif expected_method == "retrain_full":
        source_entry = json.loads((RUN / "source_models" / f"retrain_s{s}.json").read_text())
        compatible = report["method"] == "retrain_full"
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
    destination = ROOT / "docs/phase4/PHASE4B_RETRAIN_AGGREGATES.json"
    if destination.exists():
        raise FileExistsError("Completed retrain analysis cannot be overwritten")
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
        points = {n: [] for n in ("full", "aligned_full", "placebo_full", "retrain_full")}
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
                    "placebo_full": (PLACEBO, placebo_name(s, d, b)),
                    "retrain_full": (RUN, name(s, d, b)),
                }
                for method, (root, filename) in files.items():
                    p, r = read_fit(root, filename, t, s, d, b, method)
                    points[method].append(measures(y, p["B"]))
                    points_A[method].append(measures(t.labels("A"), p["A"]))
                    samples[method].append(boots(y, p["B"], weights))
                    if method == "retrain_full":
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
        for a, z in (
            ("retrain_full", "full"),
            ("placebo_full", "retrain_full"),
            ("aligned_full", "retrain_full"),
        ):
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
        print(f"RETRAIN_VALIDATION_BOOTSTRAP_COMPLETE budget={b}", flush=True)
    original = json.loads((OLD / "ORIGINAL_FOLD9_BASELINES.json").read_text())
    aligned = json.loads((OLD / "ALIGNMENT_GRID_COMPLETE.json").read_text())
    sources = []
    for s in (15101, 15102, 15103):
        p = RUN / "source_models" / f"retrain_s{s}.json"
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
                retrain_macro_AUROC=r["fold9_macro_AUROC"],
                retrain_per_label_AUROC=r["fold9_AUROC"],
                selected_epoch=r["selected_epoch"],
                source_sha256=r["sha256"],
                source_state_hash=r["state_hash"],
                source_runtime_seconds=r["runtime_seconds"],
            )
        )
    result = dict(
        schema="phase4c-retrain-validation-v1",
        lambda_value=0.0,
        sources=sources,
        source_comparison=json.loads((RUN / "SOURCE_COMPARISON.json").read_text()),
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
            schema="retrain-full-replay-references-v1",
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
            dict(source_AUROC=[r["retrain_macro_AUROC"] for r in sources], contrasts=contrasts),
            indent=2,
        ),
        flush=True,
    )


def append_report(r):
    path = ROOT / "docs/phase4/PHASE4B_RESULTS.md"
    binding = json.loads((RUN / "PRESERVATION_BEFORE.json").read_text())["report"]
    if sha(path) != binding["sha256"] or path.stat().st_size != binding["bytes"]:
        raise PermissionError("Earlier report changed; refuse overwrite/duplicate append")
    lines = [
        "",
        "## Plain-retrain control",
        "",
        "Amendment2, validation only. [Prospective protocol](PLAIN_RETRAIN_PROTOCOL.md) · [Structured aggregates](PHASE4B_RETRAIN_AGGREGATES.json). Original and placebo results above remain unchanged. No CPSC test data/outputs or PTB-XL fold10 were accessed.",
        "",
        "The new parameterized source loop reproduces the frozen placebo loop at lambda0.1 on invented fixtures. At lambda0 it optimizes BCE only: MMD is diagnostic under no-grad. It retains the independently sampled PTB batch32, second train-mode forward, BN updates and dropout/RNG consumption. This is a matched lambda0 control, not a claim that the original single-pass source recipe was repeated unchanged. Source folds1–8, fold9 AP selection/BCE ties, source batch128, all50 epochs, seeds, optimizer/schedule, normalization and first-stream ordering remain fixed. No CPSC data entered source training.",
        "",
        "| Seed | Original fold9 macro AUROC | Aligned lambda0.1 | Retrain lambda0 | Retrain selected epoch |",
        "|---:|---:|---:|---:|---:|",
    ]
    for s in r["sources"]:
        lines.append(
            f"| {s['source_seed']} | {s['original_macro_AUROC']:.6f} | {s['aligned_macro_AUROC']:.6f} | {s['retrain_macro_AUROC']:.6f} | {s['selected_epoch']} |"
        )
    lines += [
        "",
        f"Mean lambda0 fold9 macro AUROC: **{np.mean([s['retrain_macro_AUROC'] for s in r['sources']]):.6f}**. Three new source checkpoints and their SHA-256 values are private/gitignored; no adapted checkpoint was saved.",
        "",
        "| Seed | Checkpoint file bytes identical? | Full state bytes identical? | Changed parameter tensors | Changed buffers | Original selected epoch | New epoch |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for s in r["source_comparison"]["sources"]:
        lines.append(
            f"| {s['source_seed']} | {s['checkpoint_file_bytes_identical']} | {s['full_state_bitwise_identical']} | {s['changed_parameter_tensors']} | {s['changed_buffer_tensors']} | {s['original_selected_epoch']} | {s['retrain_selected_epoch']} |"
        )
    lines += [
        "",
        "| Seed | Original base-code hash matches? | Original real-executor hash matches? | Optimizer state stored? | RNG state stored? | Library-version record stored? |",
        "|---:|---|---|---|---|---|",
    ]
    for s in r["source_comparison"]["sources"]:
        lines.append(
            f"| {s['source_seed']} | {s['original_source_code_matches_current_public']} | {s['original_real_executor_matches_current_public']} | {s['original_has_optimizer_state']} | {s['original_has_rng_state']} | {s['original_has_library_version_record']} |"
        )
    lines += [
        "",
        "**Pipeline comparison.** Original Experiment4 sources use `phase3b_real/engine.py:train_source`, one supervised `model(...)` forward per source minibatch. The aligned/placebo/retrain path uses `representations(...)` plus the head, followed by a second independent32-record representation forward. The second pass updates BN statistics and consumes dropout RNG; it does not advance the dedicated first-stream shuffle generator. Later supervised dropout masks can therefore differ even at lambda0. Diagnostic no-grad MMD consumes no randomness and contributes no gradient. Source settings, row identities and normalization are checked directly against each original checkpoint. Source-view cache payloads match the aligned-source caches; input-view operations are per record and synthetic tests check chunk/full invariance. Original and new checkpoint provenance/history schemas and code paths differ independently of numerical state. Exact current-versus-recorded executor-hash matches and changed tensor details are retained in the comparison record; no claim is based on file hashes alone.",
        "",
        "Concretely, the backbone has15 BN layers in the five CNN encoders and Dropout(0.2) in each of the two fusion modules. The additional train-mode representation pass updates those BN layers and invokes both dropout modules again. The first source permutation uses the same separately seeded generator and source-row order; the independent second stream does not advance it. A historical per-batch order/RNG/optimizer trace is not reconstructed from final checkpoints. Missing stored runtime-version/state information, as indicated above, limits claims about an exact first historical divergence; code-defined differences and direct final-state comparisons are distinguished.",
        "",
        "Synthetic tests show that an extra forward changes BN counters/dropout RNG while leaving the first supervised BCE gradient unchanged. This identifies concrete pipeline differences, not a quantitative causal decomposition of the final selected models. The retrain-minus-original comparison below measures their combined effect under the fixed recipe.",
        "",
        "All target fits reuse the nine source/draw episodes, original F+G memberships and A/B500/500 validation rows, batch32 replacement sampling, original seeds and exactly200 AdamW updates. No A/B outcome selected a source, epoch, lambda or setting.",
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
        "| Budget | Contrast | Macro AP difference [pointwise95% CI] | Macro AUROC difference [pointwise95% CI] |",
        "|---:|---|---|---|",
    ]
    for v in r["contrasts"]:
        lines.append(
            f"| {v['budget']} | {v['contrast']} | {v['AP_difference']:+.6f} [{v['AP_ci']['low']:+.6f}, {v['AP_ci']['high']:+.6f}] | {v['AUROC_difference']:+.6f} [{v['AUROC_ci']['low']:+.6f}, {v['AUROC_ci']['high']:+.6f}] |"
        )
    lines += [
        "",
        "Intervals use10,000 identical paired split-B record resamples (seed20261127), conditional on the fixed models, with equal episode-wise CD/STTC means. They do not pool repeated draws as independent cohorts. Per-label results and supported draws are in the JSON. These pointwise reused-validation comparisons are not confirmatory test results.",
        "",
        "Retrain−original measures the new training-pipeline effect; placebo−retrain isolates adding the weighted within-source MMD gradient in the matched two-forward recipe; aligned−retrain combines that addition with target-domain substitution, including domain-dependent BN effects. None alone establishes a clinically validated or MMD-gradient-only target-alignment mechanism.",
        "",
        "The new confirmatory document is **v2 DRAFT, UNADOPTED**. The author must choose P-reg or P-align and lock the final protocol before any test access. No candidate primary was selected by this computation; no confirmatory replay or test operation was run.",
    ]
    with path.open("a") as out:
        out.write("\n".join(lines) + "\n")
        out.flush()
    import hashlib

    if hashlib.sha256(path.read_bytes()[: binding["bytes"]]).hexdigest() != binding["sha256"]:
        raise PermissionError("Earlier report prefix changed")


if __name__ == "__main__":
    run()
