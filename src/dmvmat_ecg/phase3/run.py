"""Offline FEAS-only frozen checkpoint execution. No fitting or test-fold access."""

import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from .download import sha
from .inference import fold9, infer, reference_models
from .labels import targets
from .prepare import load_feas, source_proxy_labels, write_new
from .splits import DOMAINS, digest
from .statistics import bootstrap, mmd, percentile, prior_adjusted_AP, ranking


def freeze(root):
    root = Path(root)
    out = root / "results/phase3a"
    refs, stats = reference_models(root)
    splits = json.loads((out / "SPLIT_SUMMARY.json").read_text())
    mapping_path = out / "LABEL_MAPPING.json"
    if sha(mapping_path) != splits["mapping_sha256"]:
        raise PermissionError("Frozen mapping changed")
    code = {
        str(p.relative_to(root)): sha(p)
        for folder in [root / "src/dmvmat_ecg/phase3", root / "scripts/phase3"]
        for p in sorted(folder.glob("*.py"))
    }
    protocol = {
        "task": "Exploratory zero-shot recorded-annotation proxy feasibility; no clinical validation",
        "models": refs,
        "stats": stats,
        "source_reference": "fold9 only; full superclass AND restricted matched SCP union, latter primary for DROP",
        "source_cache_sha256": {
            name: sha(root / "private/cache" / name)
            for name in ["validation_x.npy", "validation_meta.npz", "standardization.json"]
        },
        "mapping_sha256": sha(mapping_path),
        "split_summary_sha256": sha(out / "SPLIT_SUMMARY.json"),
        "code_sha256": code,
        "device": "CPU float32, four torch threads",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "estimand": "mean of seed101/202 metric values; NOT probability-averaged ensemble",
        "bootstrap": {
            "draws": 2000,
            "source_seed": 74311,
            "target_seed": 74312,
            "unit": "record",
            "interval": "95percent percentile; undefined draws reported, never imputed",
            "drop": "independent source/target record resampling; paired models/labels/seeds within population",
        },
        "mmd": {
            "sample_cap_per_population": 256,
            "sampling": "deterministic SHA256(ID,74321), label-free",
            "kernel": "single RBF pooled positive-distance median",
            "permutations": 999,
            "seed": 74321,
            "p": "(1+exceedances)/(999+1)",
            "adjustment": "Holm across seven modules separately per domain/model seed",
        },
        "verdict_model": "historical five-view noaug, mean seed-wise metrics",
        "verdict": {
            "drop": "macro(CD-proxy,STTC-proxy) AUROC(target)-AUROC(proxy-matched fold9) <= -0.05",
            "alternative": "BOTH target proxy AP95percent upper bounds are below source proxy AP reweighted to target FEAS prevalence; descriptive pure-prior-shift benchmark only, no confirmation of signal shift",
            "support": "FEAS-prevalence-estimated RESERVED positives >=150 for EACH proxy label",
            "checks": "checksum/lead/unit/finite/length/mapping checks pass; clinical endpoint equivalence explicitly NOT claimed",
            "SUITABLE": "drop or alternative AND support AND checks, annotation-proxy adaptation feasibility only",
            "BORDERLINE": "some but not all conditions or undefined metric",
            "NOT_SUITABLE": "no meaningful degradation and no alternative evidence, or insufficient support",
            "selection": "largest macro AUROC drop among suitable domains; remaining support as tie-break; descriptive target choice only",
        },
        "limits": [
            "No true-negative clinical ascertainment",
            "No PTB-XL cross-waveform duplicate audit, human waiver",
            "Native-target duplicate audit does not prove patient independence",
            "Full10s CNN pipeline replaces original2.5s request",
            "First10sec truncation; short records excluded BEFORE split",
            "RESERVED never loaded for inference or later label analysis",
            "Published PTBXL test numbers not tuning criteria",
        ],
    }
    write_new(out / "RUN_PROTOCOL.json", protocol)
    return protocol


def predict(root):
    root = Path(root)
    out = root / "results/phase3a"
    protocol = json.loads((out / "RUN_PROTOCOL.json").read_text())
    for path, expected in protocol["code_sha256"].items():
        if sha(root / path) != expected:
            raise PermissionError("Code changed after prescoring freeze")
    if (
        sha(out / "LABEL_MAPPING.json") != protocol["mapping_sha256"]
        or sha(out / "SPLIT_SUMMARY.json") != protocol["split_summary_sha256"]
    ):
        raise PermissionError("Mapping/split changed before scoring")
    folder = root / "data/manifests/phase3a_v1/predictions"
    folder.mkdir(exist_ok=False)
    frozen = json.loads((out / "LABEL_MAPPING.json").read_text())
    x, meta = fold9(root)
    proxy_y, code_y = source_proxy_labels(
        root / "data/raw/ptbxl_v1_0_3/ptbxl_database.csv", meta["ecg_id"], frozen
    )
    np.savez(folder / "fold9_labels.npz", proxy=proxy_y, full=meta["y"][:, [3, 2]], code=code_y)
    runtimes = []
    for ref in protocol["models"]:
        key = ref["model"] + "_" + str(ref["seed"])
        p, reps, elapsed = infer(
            ref, x, protocol["stats"], representations=ref["model"] == "multiview"
        )
        np.savez(folder / ("fold9_" + key + ".npz"), p=p, **reps)
        runtimes.append(
            {"population": "fold9", "key": key, "inference_seconds": elapsed, "records": len(x)}
        )
    for domain in DOMAINS:
        x, rows, manifest = load_feas(root, domain)
        ids = [row["id"] for row in rows]
        y = np.array([targets(row["header"]["dx"], frozen) for row in rows], dtype=np.int8)
        code_y = np.array(
            [
                [int(code in row["header"]["dx"]) for code in frozen["per_code_descriptive_tasks"]]
                for row in rows
            ],
            dtype=np.int8,
        )
        np.savez(folder / (domain + "_labels.npz"), proxy=y, code=code_y)
        for ref in protocol["models"]:
            key = ref["model"] + "_" + str(ref["seed"])
            p, reps, elapsed = infer(
                ref,
                x,
                protocol["stats"],
                ids=ids,
                manifest=manifest,
                representations=ref["model"] == "multiview",
            )
            np.savez(folder / (domain + "_" + key + ".npz"), p=p, **reps)
            runtimes.append(
                {"population": domain, "key": key, "inference_seconds": elapsed, "records": len(x)}
            )
        print(json.dumps({"finished_FEAS_prediction": domain, "records": len(x)}), flush=True)
    write_new(
        out / "PREDICTION_SEAL.json",
        {
            "protocol_sha256": sha(out / "RUN_PROTOCOL.json"),
            "artifacts": {p.name: sha(p) for p in sorted(folder.glob("*.npz"))},
            "inference_runtime": runtimes,
            "no_training": True,
            "reserved_access": False,
        },
    )


def mean_metrics(y, scores):
    rows = [ranking(y, p) for p in scores]
    result = []
    for j in range(2):
        entry = {}
        for k in rows[0][j]:
            values = [r[j][k] for r in rows]
            entry[k] = None if any(v is None for v in values) else float(np.mean(values))
        result.append(entry)
    return result


def analyze(root):
    root = Path(root)
    out = root / "results/phase3a"
    folder = root / "data/manifests/phase3a_v1/predictions"
    seal = json.loads((out / "PREDICTION_SEAL.json").read_text())
    if any(sha(folder / name) != value for name, value in seal["artifacts"].items()):
        raise PermissionError("Sealed predictions changed")
    if sha(out / "RUN_PROTOCOL.json") != seal["protocol_sha256"]:
        raise PermissionError("Prescoring protocol changed")
    frozen = json.loads((out / "LABEL_MAPPING.json").read_text())
    populations = ["fold9", *DOMAINS]
    models = ["multiview", "flat_cnn"]
    labels, code_labels, predictions, metrics, boot, per_code = {}, {}, {}, {}, {}, {}
    seedwise = {}
    start = time.perf_counter()
    for population in populations:
        with np.load(folder / (population + "_labels.npz")) as f:
            labels[population], code_labels[population] = f["proxy"].copy(), f["code"].copy()
            full_y = f["full"].copy() if population == "fold9" else None
        predictions[population] = {
            model: [
                np.load(folder / f"{population}_{model}_{seed}.npz")["p"] for seed in [101, 202]
            ]
            for model in models
        }
        metrics[population] = {
            model: mean_metrics(labels[population], scores)
            for model, scores in predictions[population].items()
        }
        seedwise[population] = {
            model: {
                str(seed): ranking(labels[population], scores[index])
                for index, seed in enumerate([101, 202])
            }
            for model, scores in predictions[population].items()
        }
        boot[population] = bootstrap(
            labels[population],
            predictions[population],
            seed=74311 if population == "fold9" else 74312,
        )
        for model in models:
            for j in range(2):
                metrics[population][model][j]["CI95"] = {
                    key: percentile(boot[population][model][key][:, j]) for key in ["ap", "auroc"]
                }
        per_code[population] = {}
        for k, (code, task) in enumerate(frozen["per_code_descriptive_tasks"].items()):
            j = ["CD", "STTC"].index(task["label"])
            cy = code_labels[population][:, k]
            score = {
                m: [np.stack([p[:, j], p[:, j]], 1) for p in ps]
                for m, ps in predictions[population].items()
            }
            per_code[population][code] = {
                "name": task["name"],
                "score_output": task["label"],
                "annotation_prevalence": float(cy.mean()),
                "positive": int(cy.sum()),
                "models": {
                    m: mean_metrics(np.stack([cy, cy], 1), ps)[0] for m, ps in score.items()
                },
            }
        if population == "fold9":
            metrics["fold9_full_superclass"] = {
                m: mean_metrics(full_y, ps) for m, ps in predictions[population].items()
            }
            full_boot = bootstrap(full_y, predictions[population], seed=74311)
            for model in models:
                for j in range(2):
                    metrics["fold9_full_superclass"][model][j]["CI95"] = {
                        key: percentile(full_boot[model][key][:, j]) for key in ["ap", "auroc"]
                    }
    drops, verdicts = {}, {}
    split_summary = json.loads((out / "SPLIT_SUMMARY.json").read_text())["domains"]
    for domain in DOMAINS:
        drops[domain] = {}
        for model in models:
            prior_ap = prior_adjusted_AP(
                labels["fold9"], predictions["fold9"][model], labels[domain].mean(0)
            )
            drops[domain][model] = []
            for j in range(2):
                entry = {"source_AP_prior_reweighted_to_target": prior_ap[j]}
                for key in ["auroc", "ap"]:
                    a, b = metrics[domain][model][j][key], metrics["fold9"][model][j][key]
                    entry[key + "_DROP"] = a - b if a is not None and b is not None else None
                    entry[key + "_DROP_CI95"] = percentile(
                        boot[domain][model][key][:, j] - boot["fold9"][model][key][:, j]
                    )
                drops[domain][model].append(entry)
        d = [row["auroc_DROP"] for row in drops[domain]["multiview"]]
        macro = float(np.mean(d)) if all(v is not None for v in d) else None
        ap_condition = all(
            metrics[domain]["multiview"][j]["CI95"]["ap"]["upper"] is not None
            and drops[domain]["multiview"][j]["source_AP_prior_reweighted_to_target"] is not None
            and metrics[domain]["multiview"][j]["CI95"]["ap"]["upper"]
            < drops[domain]["multiview"][j]["source_AP_prior_reweighted_to_target"]
            for j in range(2)
        )
        support = min(split_summary[domain]["EXPECTED_RESERVED_positive_CD_STTC"]) >= 150
        degraded = macro is not None and macro <= -0.05
        suitable = support and (degraded or ap_condition)
        verdicts[domain] = {
            "status": "SUITABLE"
            if suitable
            else "BORDERLINE"
            if support and (macro is None or macro < 0)
            else "NOT SUITABLE",
            "macro_proxy_matched_AUROC_DROP": macro,
            "AP_prior_adjusted_condition": ap_condition,
            "expected_reserved_support_at_least150_each": support,
            "scope": "annotation proxy feasibility only; external clinical validation out of scope",
        }
    write_new(
        out / "METRICS.json",
        {
            "metrics": metrics,
            "seedwise_metrics": seedwise,
            "drops": drops,
            "verdicts": verdicts,
            "per_code_descriptive": per_code,
            "analysis_seconds": time.perf_counter() - start,
            "bootstrap_draws": 2000,
            "undefined_handling": "undefined resample metrics excluded from percentile calculation, counts reported; not replaced by zero",
        },
    )
    return verdicts


def shifts(root):
    root = Path(root)
    folder = root / "data/manifests/phase3a_v1"
    pred = folder / "predictions"
    _, meta = fold9(root)
    source_ids = [str(v) for v in meta["ecg_id"]]
    si = sorted(range(len(source_ids)), key=lambda i: digest([74321, source_ids[i]]))[:256]
    results = {}
    for domain in DOMAINS:
        manifest = json.loads((folder / (domain + "_split.json")).read_text())
        ids = manifest["FEAS"]
        ti = sorted(range(len(ids)), key=lambda i: digest([74321, ids[i]]))[:256]
        results[domain] = {}
        for seed in [101, 202]:
            source = np.load(pred / f"fold9_multiview_{seed}.npz")
            target = np.load(pred / f"{domain}_multiview_{seed}.npz")
            rows = {
                name: mmd(source[name][si], target[name][ti])
                for name in [
                    "view_1",
                    "view_2",
                    "view_3",
                    "view_4",
                    "view_5",
                    "fusion_1",
                    "fusion_2",
                ]
            }
            order = sorted(rows, key=lambda name: rows[name]["p"])
            previous = 0
            for rank, name in enumerate(order):
                previous = max(previous, min(1.0, (7 - rank) * rows[name]["p"]))
                rows[name]["p_Holm7"] = previous
            results[domain][str(seed)] = rows
    write_new(root / "results/phase3a/MODULE_MMD.json", results)
    return results
