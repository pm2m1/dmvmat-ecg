"""Aggregate-only reporting, no outcome provider or predictor modification."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dmvmat_ecg.phase3b.jobs import save_json
from dmvmat_ecg.phase3b.protocol import BUDGETS, METHODS, SOURCE_SEEDS


def fmt(x):
    return "NA" if x is None or not np.isfinite(x) else f"{x:.5f}"


def interval(row, key="CI95"):
    values = row[key]["interval"] if row.get(key) else None
    return "not estimable" if values is None else "[" + ", ".join(fmt(v) for v in values) + "]"


def verdict(ci):
    if ci is None:
        return "Primary uncertainty is not estimable under the valid-replicate rule."
    if ci[0] > 0:
        return "Conditional exploratory evidence that A beats Full-cost here."
    if ci[1] < 0:
        return "Full-cost is better than A here under the conditional comparison."
    return "Inconclusive; the interval includes zero. This is not evidence of no effect."


def report(root, result, y, observed, headers, seal_hash, timing):
    root = Path(root)
    out = root / "results/phase3b_eval_v2"
    primary = result["comparisons"]["B128/A-minus-full_cost"]
    head = result["comparisons"]["B128/A-minus-head_cost"]
    random = result["comparisons"]["B128/A-minus-random_a"]
    supported = all(
        r["Bonferroni_CI9875"]["interval"] is not None and r["Bonferroni_CI9875"]["interval"][0] > 0
        for r in (head, random)
    )
    source_data = json.loads((root / "results/phase3b/SOURCE_VALIDATION_RESULTS.json").read_text())
    membership = json.loads((root / "private/phase3b_cpsc_v1/MEMBERSHIPS.json").read_text())
    source_y = np.array([r["y"] for r in membership["source"]["source_stop"]])
    source_prevalence = source_y.mean(0)
    reference = []
    for s in SOURCE_SEEDS:
        point = source_data[str(s)]["report"]["source_stop_per_label"]
        target = result["episode"][f"source/{s}"]
        for j, label in enumerate(("CD-proxy", "STTC-proxy")):
            reference.append(
                dict(
                    source_seed=s,
                    label=label,
                    source_fold9_AP=point[j]["AP"],
                    source_fold9_AUROC=point[j]["AUROC"],
                    source_fold9_prevalence=float(source_prevalence[j]),
                    O_AP=target["per_label"][j]["AP"],
                    O_AUROC=target["per_label"][j]["AUROC"],
                    O_prevalence=target["per_label"][j]["prevalence_AP_reference"],
                    O_minus_fold9_AP=target["per_label"][j]["AP"] - point[j]["AP"]
                    if target["per_label"][j]["AP"] is not None
                    else None,
                    O_minus_fold9_AUROC=target["per_label"][j]["AUROC"] - point[j]["AUROC"]
                    if target["per_label"][j]["AUROC"] is not None
                    else None,
                )
            )
    save_json(
        out / "SOURCE_FOLD9_TRANSFER_REFERENCE.json",
        dict(
            rows=reference,
            limitation="Source fold9 selected checkpoints; AP depends on prevalence and populations differ.",
        ),
    )
    table = []
    label_table = []
    for b in BUDGETS:
        for method in ("source_only", *METHODS):
            r = result["aggregate"][f"B{b}/{method}"]
            m = r["macro"]
            table.append(
                "|"
                + str(b)
                + "|"
                + method
                + "|"
                + "|".join(fmt(m[k]) for k in ("AP", "AUROC", "Brier", "log_loss", "ECE15"))
                + "|"
            )
            for row in r["per_label"]:
                label_table.append(
                    "|"
                    + str(b)
                    + "|"
                    + method
                    + "|"
                    + row["label"]
                    + "|"
                    + "|".join(
                        fmt(row[k])
                        for k in (
                            "AP",
                            "AUROC",
                            "prevalence_AP_reference",
                            "Brier",
                            "log_loss",
                            "ECE15",
                            "precision",
                            "recall",
                            "specificity",
                            "F1",
                            "balanced_accuracy",
                        )
                    )
                    + "|"
                )
    comparisons = []
    for key, r in result["comparisons"].items():
        comparisons.append(
            "|"
            + key
            + "|"
            + fmt(r["delta_AP"])
            + "|"
            + fmt(r["delta_AUROC"])
            + "|"
            + interval(r)
            + "|"
            + interval(r, "Bonferroni_CI9875")
            + "|"
        )
    source_table = "\n".join(
        "|"
        + str(r["source_seed"])
        + "|"
        + r["label"]
        + "|"
        + "|".join(
            fmt(r[k])
            for k in (
                "source_fold9_AP",
                "source_fold9_AUROC",
                "source_fold9_prevalence",
                "O_AP",
                "O_AUROC",
                "O_prevalence",
                "O_minus_fold9_AUROC",
            )
        )
        + "|"
        for r in reference
    )
    primary_episodes = "\n".join(
        f"|{r['source_seed']}|{r['draw_seed']}|{fmt(r['delta_AP'])}|{fmt(r['delta_AUROC'])}|"
        for r in primary["episodes"]
    )
    recovery = "\n".join(
        "|" + key + "|" + fmt(r["delta_AP"]) + "|" + fmt(r["delta_AUROC"]) + "|"
        for key, r in result["recovery"].items()
    )
    negative = "\n".join(
        "|" + key + "|" + str(r["AP_worse_count"]) + "/9|" + fmt(r["mean_delta_AP"]) + "|"
        for key, r in result["negative_transfer"].items()
    )
    prevalence = [
        float(y[observed[:, j], j].mean()) if observed[:, j].any() else None for j in range(2)
    ]
    text = f"""# Phase3B corrected locked CPSC O evaluation V2

**Disclosure:** The first O read failed because the decoder did not recognise
`# Dx:` and returned zero observed labels, conveying no usable outcome information.
All predictions were sealed beforehand and remain unchanged. This is the corrected
second outcome read under explicit human authorization, not a new inference run.
The V1 report and blocker are unchanged and superseded by this completed V2 report.

**Pre-access revision:** Before the corrected read, the protocol owner replaced
the impossible all-R frozen-reference requirement with exact agreement on all711
frozen-reference R records plus live unmodified-adapter agreement on all3,859 R
records. Both had zero disagreements. The revision is transcribed privately in
STEP3_REVISION_AUTHORIZATION_V2.json and bound to DECODER_R_VALIDATION.json.
No split, endpoint, prediction, model, roster or statistical analysis changed.

The corrected validity gate requires at least95% observed labels and both classes
for each label. It passed before any metric calculation (see the aggregate gate receipt).

## Executive finding

Primary A−Full-cost at B128: Δ mean raw AP **{fmt(primary["delta_AP"])}**;
95% paired record-group interval **{interval(primary)}**.
Mean paired ΔAUROC: **{fmt(primary["delta_AUROC"])}** (descriptive).
{verdict(primary["CI95"]["interval"])}
Evidence-guided selection beyond ordinary freezing requires A also to beat
Head-cost and Random-A. Both Bonferroni98.75% lower bounds exceed zero:
**{supported}**. Beating Full-cost alone is a policy-level result.
No model, mask, checkpoint, threshold or seed was changed after O access.

## Population and pre-outcome signal audit

Frozen O membership3,860 records; evaluated **{headers["evaluated"]}**;
short records excluded **{headers["excluded_short"]}**; first10s truncation
**{headers["truncated_first10"]}**. All exclusions were determined from the
physical WFDB header prefix before any score/outcome access. No padding or
windows. Inputs12 canonical leads,mV,100Hz; source or target-only frozen
normalization as registered. Diagnosis comments were not read during signal
metadata preparation; only the first13 unbuffered header lines were parsed.
Every MAT signal passed its publisher checksum. Full header checksums and
diagnoses were read only inside the authorized post-seal outcome accesses.
There are exactly two recorded outcome accesses: the failed V1 decode and this
single authorized corrected V2 decode. The first marker was never reset.

CD-proxy observed prevalence/random AP baseline **{fmt(prevalence[0])}**;
STTC-proxy **{fmt(prevalence[1])}**. An unlisted code means not annotated,
not clinically absent. Unknown labels remain masked; observed counts and
unknowns appear in every per-label metric record.

## Complete mean episode-wise performance

Source-only means three source models, reused across budgets/draws. Other
rows average nine source×draw episodes. These are means of metrics, not
probability ensembles, pooled AP or an outer-selected winning model.

|B|Procedure|Mean AP|Mean AUROC|Mean Brier|Mean log loss|Mean ECE15|
|---:|---|---:|---:|---:|---:|---:|
{chr(10).join(table)}

|B|Procedure|Label|AP|AUROC|Prevalence/AP baseline|Brier|Log loss|ECE15|Precision|Recall|Specificity|F1|Balanced accuracy|
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(label_table)}

All138 logical procedures, per-label values, unknown counts and confusion
counts are in PER_EPISODE_METRICS.json. No exclusions of abstaining episodes
or masks equal to controls. Precision etc use0.5; undefined denominators are
NA. Probabilities are raw, uncalibrated frozen sigmoid outputs. Only log loss
clips to[1e-7,1−1e-7]. AP/AUROC/Brier/ECE inputs are not clipped. ECE uses15
fixed equal-width bins; these are diagnostic statistics, not fitted calibrators.

## Primary paired episodes

|Source|Draw|ΔAP A−Full-cost|ΔAUROC|
|---:|---:|---:|---:|
{primary_episodes}

Primary is the equal mean of all nine differences at B128. Budget64 and256
do not replace it. Practical AP margin+0.01 is an engineering criterion, not
a clinical threshold; distinguish a point estimate≥0.01 from a CI lower bound>0.01.

## Registered comparison family

|Contrast|Δ mean AP|Δ mean AUROC|95% AP interval|Bonferroni98.75% AP interval|
|---|---:|---:|---|---|
{chr(10).join(comparisons)}

Only B128 A−Full-cost is primary. The four B128 secondary AP comparisons
form the registered Bonferroni family. Other budgets are exploratory; intervals
there do not establish simultaneous control across the entire result package.

Bootstrap10,000 draws,seed78701; shared record/known-duplicate-group weights
across all labels,methods,seeds,draws and budgets. Whole CD/STTC vectors are
preserved. Primary valid draws **{primary["CI95"]["valid"]}**, undefined
**{primary["CI95"]["undefined"]}**; no redraws. No primary interval when fewer
than9,500 replicates are valid. Training seeds/draws are not independent patients.
Intervals are conditional on the fitted models and realized design; they do not
capture model/selector refitting uncertainty, target selection from FEAS,
unknown within-patient correlation, source checkpoint selection or label error.

## Source transfer gap (fold9 reference versus O Source-only)

|Seed|Label|Fold9 AP|Fold9 AUROC|Fold9 AP baseline|O AP|O AUROC|O AP baseline|O−fold9 AUROC|
|---:|---|---:|---:|---:|---:|---:|---:|---:|
{source_table}

Fold9 selected source checkpoints and is an optimistic development reference.
Source and target populations/prevalences and annotation practices differ.
AP changes cannot be interpreted as pure hospital signal-shift effects.

## Descriptive recovery contrasts fixed before O access

|Contrast|Mean ΔAP|Mean ΔAUROC|
|---|---:|---:|
{recovery}

These show how much adaptation changes ranking relative to Source-only;
they are not additional primary hypotheses or clinical recovery estimates.

## Negative transfer

|Procedure|AP worse than paired Source-only|Mean ΔAP|
|---|---:|---:|
{negative}

Counts are paired fitted episodes on the same O population, not independent
clinical events. Equality from aliases stays included. Increased ADAPT counts,
valid geometry, MMD significance and mask stability do not by themselves show
predictive benefit. Module masks/drift and label costs remain the frozen
development findings; all target arms received the same F∪G labels and200 updates.

## Execution integrity and limits

Human authorization was transcribed under private/phase3b_cpsc_v1/. Locked
study hash30a0c75ab9a159773bc87e3941ddfdee91c1860482d08e20a7b87ae601d25073.
Complete138-procedure prediction seal **{seal_hash}** preceded all O outcome
access. Aliases resolve to shared artifacts without disappearing from the roster.
Original V1 inference runs **{len(timing)}**; no new inference in V2.
Original summed forward time
**{sum(v["seconds"] for v in timing.values()):.2f}s**, excluding preparation/load.
Predictor-fit ledger remains121/144; no training,fine-tuning,recalibration,
threshold selection,mask/seed/checkpoint changes or reserve use. Post-evaluation
tests and the final integrity receipt document re-entry refusal and unchanged
frozen artifacts. No Phase2 artifact,PTBXL fold10 or CPSC FEAS data were accessed.

This is **CPSC-only exploratory annotation-proxy evidence at record level**.
Target patient identifiers are unknown; exact native deduplication is not proof
of patient independence. PTBXL-versus-target waveform overlap was not audited
by explicit prior waiver. Phase3A FEAS outcomes chose the target and informed
the protocol; RESERVED annotations were previously used for stratification.
This history precludes a pristine independent confirmation claim. Labels are
recorded-code unions, not clinically adjudicated positives/negatives; label
definition/reporting mismatch can contribute to the apparent hospital gap.
First10s truncation can omit findings described by full-record annotations.
No external clinical validation, clinical utility, causal transfer mechanism
or future cardiovascular-event prediction is established. A CI crossing zero
is inconclusive, not proof of no effect. No further experiment is executed here.
"""
    with (root / "docs/phase3/PHASE3B_LOCKED_EVALUATION_RESULTS_V2.md").open("w") as stream:
        stream.write(text)
    figures = out / "figures"
    figures.mkdir()
    methods = ("source_only", *METHODS)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for method in methods:
        axes[0].plot(
            BUDGETS,
            [result["aggregate"][f"B{b}/{method}"]["macro"]["AP"] for b in BUDGETS],
            marker="o",
            label=method,
        )
        axes[1].plot(
            BUDGETS,
            [result["aggregate"][f"B{b}/{method}"]["macro"]["AUROC"] for b in BUDGETS],
            marker="o",
            label=method,
        )
    for ax, label in zip(axes, ("Mean episode AP", "Mean episode AUROC"), strict=True):
        ax.set(xlabel="F∪G record-label budget", ylabel=label)
    axes[1].legend(fontsize=7)
    fig.suptitle("Locked CPSC O — annotation proxies, record-level")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(figures / ("budget_ranking." + ext), dpi=200)
    plt.close(fig)
    keys = [
        f"B128/A-minus-{m}"
        for m in ("full_cost", "head_cost", "random_a", "source_only", "target_only_cost")
    ]
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, key in enumerate(keys):
        row = result["comparisons"][key]
        bounds = row["CI95"]["interval"]
        point = row["delta_AP"]
        if bounds:
            ax.hlines(i, bounds[0], bounds[1], color="navy")
            ax.plot(point, i, "o", color="navy")
    ax.set_yticks(range(len(keys)), [k.split("/")[-1] for k in keys])
    ax.axvline(0, color="black", ls="--")
    ax.set(
        xlabel="Δ mean raw AP; ordinary95% paired conditional interval",
        title="Primary and registered secondary B128 contrasts",
    )
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(figures / ("paired_AP_differences." + ext), dpi=200)
    plt.close(fig)
