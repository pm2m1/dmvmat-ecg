"""Post-evaluation reporting-only refinement from immutable aggregate metrics.

Fixes clipped confusion titles and adds explicit evidence/seed interpretation.
Does not modify locked scripts, metrics, preprocessing, thresholds or models.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dmvmat_ecg.labels.superclasses import CLASSES
from dmvmat_ecg.utils.io import sha, write_json

root = Path(__file__).resolve().parents[1]
assert (root / "results/evaluation_completion.json").exists()
metrics_path = root / "results/locked_metrics.json"
before_metrics = sha(metrics_path)
data = json.loads(metrics_path.read_text())
figures = json.loads((root / "results/figure_inventory.json").read_text())
old = figures["locked_confusion_matrices"]["files"].copy()
plt.rcParams.update({"font.size": 10, "figure.constrained_layout.use": True, "savefig.dpi": 220})
fig, axes = plt.subplots(4, 5, figsize=(13, 10))
names = ["xresnet101", "flat_cnn", "multiview", "multiview_aug"]
pretty = ["XResNet val-best", "Flat CNN", "Five-view", "Five-view + aug"]
for row, name in enumerate(names):
    for j, label in enumerate(CLASSES):
        info = data["runs"][f"{name}:101"]["validation_thresholds"]["labels"][label]
        matrix = np.array(info["confusion_matrix"])
        ax = axes[row, j]
        ax.imshow(matrix, cmap="Blues")
        for a in range(2):
            for b in range(2):
                ax.text(
                    b,
                    a,
                    str(matrix[a, b]),
                    ha="center",
                    va="center",
                    color="white" if matrix[a, b] > matrix.max() / 2 else "black",
                )
        ax.set(
            xticks=[0, 1],
            yticks=[0, 1],
            xlabel="Predicted",
            ylabel="Observed",
            title=f"{pretty[row]}\n{label}, t={info['threshold']:.2f}",
        )
new = {}
for suffix in ("png", "pdf"):
    path = root / "results/figures" / f"locked_confusion_matrices.{suffix}"
    fig.savefig(path)
    new[str(path.relative_to(root))] = sha(path)
plt.close(fig)
figures["locked_confusion_matrices"]["files"] = new
figures["locked_confusion_matrices"]["caption"] += (
    " Reporting-only revision: titles wrapped after visual review; all numerical values unchanged."
)
write_json(root / "results/figure_inventory.json", figures)
summary_path = root / "docs/FINAL_RESEARCH_SUMMARY.md"
summary = summary_path.read_text()
if "## Direct answers and seed stability" not in summary:
    summary += """
## Direct answers and seed stability

The reference reproduction succeeded as a competitive **modern adaptation**:
the prespecified final-epoch procedure attained macro AUROC **0.92800**, close
to the published approximately 0.928; its macro AP was **0.82241**. The
validation-best procedure attained AUROC **0.92985**, AP **0.82759**. Different
dataset/software versions and checkpoint conventions prevent an exact historical
reproduction or a paired superiority claim against the published number.

Our no-augmentation five-view network attained macro AUROC **0.92139** and
macro AP **0.81405**. Relative to the matched flat CNN, both seeds improved
AUROC (seed 101: +0.00423; seed 202: +0.00388). The mean difference +0.00406
has a positive 97.5% conditional bootstrap interval, **[+0.000085, +0.007856]**,
whose lower endpoint is extremely close to zero. With only 2,000 bootstrap draws,
finite Monte Carlo resolution and omitted retraining/model-selection uncertainty
matter. This is limited conditional evidence, not established general superiority.
The AP improvement +0.00748 has an interval spanning zero.

Relative to XResNet validation-best, five-view underperformed in both seeds and
on both macro ranking metrics: AUROC difference -0.00847 (adjusted interval
[-0.01312, -0.00400]); AP difference -0.01354 (exploratory interval
[-0.02253, -0.00408]). The proposed lightweight design did not match the stronger
reference's ranking under this protocol. No equivalence margin was registered.

Exploratory label findings versus flat CNN: HYP AUROC improved by about 0.01112
and HYP AP by 0.03183; CD AP improved by 0.01220. These per-label intervals are
unadjusted and do not establish medically meaningful improvements. There is no
uniform gain over XResNet. Bounded augmentation changed macro AUROC by -0.00071
and AP by +0.00293, both intervals spanning zero: no reliable ranking benefit.

The small CNNs show substantial late overfitting in development curves; their
validation-selected checkpoints were retained. Global pooling after short local
receptive fields limits what the rhythm branch can represent. These are observed
behavior/static architectural limits, not a causal explanation of the result.
View-specific utility remains untested. The matched comparison changes clinical
grouping and deterministic representations together, so it does not isolate one
fusion mechanism or one derived view.

Achieving 0.92–0.93 macro AUROC is realistic **on this benchmark**: our actual
five-view and reference results lie in that range. Macro AP is a distinct
prevalence-dependent quantity and is lower. No external validity, future-event
prediction or selective-transfer advantage follows. A defensible next contribution
is a separately registered view/receptive-field ablation and an overlap-audited,
clinically harmonized external ECG study; do not tune again against this opened
fold 10.
"""
    summary_path.write_text(summary)
results_path = root / "docs/EXPERIMENT_RESULTS.md"
text = results_path.read_text()
if "## Explicit seed-wise locked ranking" not in text:
    text += "\n## Explicit seed-wise locked ranking\n\n| Procedure | Seed | Macro AUROC | Macro AP |\n|---|---:|---:|---:|\n"
    for name in data["summary"]:
        for seed in (101, 202):
            row = data["runs"][f"{name}:{seed}"]["validation_thresholds"]
            text += f"| {name} | {seed} | {row['macro_auroc']:.6f} | {row['macro_ap']:.6f} |\n"
    text += "\nSee the final summary's direct answers for interpretation. These seeds share one test population and do not constitute independent validation cohorts.\n"
    results_path.write_text(text)
assert sha(metrics_path) == before_metrics
write_json(
    root / "results/reporting_revision.json",
    dict(
        reason="Visual QA found clipped long confusion titles; wrapped labels, unchanged counts. Added explicit conclusion/seed tables from same sealed metrics.",
        original_figure_hashes=old,
        revised_figure_hashes=new,
        metrics_sha256_preserved=before_metrics,
        additional_prediction=False,
        additional_training=False,
        threshold_changes=False,
        locked_source_or_scripts_modified=False,
    ),
)
inventory = json.loads((root / "results/public_release_inventory.json").read_text())
for base in ("docs", "results", "scripts"):
    for file in (root / base).rglob("*"):
        if (
            file.is_file()
            and file.suffix in (".md", ".json", ".py", ".png", ".pdf")
            and file.name != "public_release_inventory.json"
            and "__pycache__" not in file.parts
        ):
            inventory["eligible_files"][str(file.relative_to(root))] = sha(file)
write_json(root / "results/public_release_inventory.json", inventory)
print("Reporting-only layout/interpretation refinement complete; sealed metrics unchanged")
