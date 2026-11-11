"""Scientific figures from recorded histories and sealed predictions only."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

from dmvmat_ecg.evaluation.state import Study
from dmvmat_ecg.labels.superclasses import CLASSES
from dmvmat_ecg.utils.io import sha, write_json

root = Path(__file__).resolve().parents[1]
study = Study(root)
if study.get("phase") != "evaluated" or not study.get("prediction_seal"):
    raise PermissionError("Figures need the completed locked evaluation")
result = json.loads((root / "results/locked_metrics.json").read_text())
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 220,
        "figure.constrained_layout.use": True,
    }
)
out = root / "results/figures"
out.mkdir(exist_ok=True)
inventory = {}


def save(fig, name, caption):
    files = {}
    for extension in ("png", "pdf"):
        path = out / f"{name}.{extension}"
        fig.savefig(path)
        files[str(path.relative_to(root))] = sha(path)
    inventory[name] = dict(caption=caption, files=files)
    plt.close(fig)


names = ["xresnet101", "xresnet101_final", "flat_cnn", "multiview", "multiview_aug"]
colors = dict(zip(names, ["#325c96", "#93a9c9", "#de8a24", "#268f72", "#a75aa4"], strict=True))
pretty = dict(
    xresnet101="XResNet val-best",
    xresnet101_final="XResNet final",
    flat_cnn="Flat CNN",
    multiview="Five-view",
    multiview_aug="Five-view + augmentation",
)
predictions = {}
for name in names:
    file = root / "private/test_predictions" / f"{name}-101.npz"
    if sha(file) != study.get("prediction_seal")[f"{name}:101"]:
        raise PermissionError("Sealed predictions changed")
    item = np.load(file, allow_pickle=False)
    predictions[name] = item["p"]
y = item["y"]
runs = [r for r in study.runs() if r["status"] == "successful" and r["metadata"]["stage"] == "full"]
fig, axes = plt.subplots(4, 3, figsize=(13, 11))
for row, name in enumerate([n for n in names if n != "xresnet101_final"]):
    for r in [r for r in runs if r["metadata"]["model"] == name]:
        h = r["metadata"]["history"]
        epochs = [v["epoch"] for v in h]
        seed = r["metadata"]["seed"]
        for col, metric in enumerate(["loss", "macro_auroc", "macro_ap"]):
            train = [v["train_loss"] if col == 0 else v["train_metrics"][metric] for v in h]
            valid = [
                v["validation_loss"] if col == 0 else v["validation_metrics"][metric] for v in h
            ]
            axes[row, col].plot(
                epochs,
                train,
                color=colors[name],
                alpha=0.45,
                linestyle="--",
                label=f"online train {seed}",
            )
            axes[row, col].plot(
                epochs,
                valid,
                color=colors[name],
                alpha=1 if seed == 101 else 0.65,
                label=f"validation {seed}",
            )
            axes[row, col].set(xlabel="Completed epoch", ylabel=metric, title=pretty[name])
            axes[row, col].legend(fontsize=7)
save(
    fig,
    "development_learning_curves",
    "Development only. Online training metrics reflect changing weights and train-mode augmentation/dropout; validation metrics use the fixed epoch checkpoint. XResNet validation uses max over seven crops.",
)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, metric in zip(axes, ("macro_auroc", "macro_ap"), strict=True):
    values = [result["summary"][n][metric] for n in names]
    ci = [
        result["summary"][n]["auroc_ci95" if metric == "macro_auroc" else "ap_ci95"] for n in names
    ]
    ax.bar(range(len(names)), values, color=[colors[n] for n in names])
    # Percentile intervals can be asymmetric around observed estimates.
    for i, limits in enumerate(ci):
        ax.plot([i, i], limits, color="black")
    ax.set_xticks(range(len(names)), [pretty[n] for n in names], rotation=30, ha="right")
    ax.set(ylabel=metric, ylim=(0, 1), title="Locked fold10; mean of paired seeds")
save(
    fig,
    "locked_macro_comparison",
    "Patient-cluster95% conditional intervals. Same test participants; mean seed-wise metrics, no ensemble. Published0.928 is not plotted as our result.",
)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for col, metric in enumerate(("per_label_auroc", "per_label_ap")):
    for i, name in enumerate(names):
        axes[col].bar(
            np.arange(5) + (i - 2) * 0.15,
            result["summary"][name][metric],
            width=0.15,
            color=colors[name],
            label=pretty[name],
        )
    if col == 1:
        axes[col].scatter(
            np.arange(5),
            y.mean(0),
            color="black",
            marker="_",
            s=150,
            label="Prevalence / AP reference",
        )
    axes[col].set(xticks=np.arange(5), xticklabels=CLASSES, ylabel=metric, ylim=(0, 1))
    axes[col].legend(fontsize=7)
save(
    fig,
    "locked_per_label_comparison",
    "Locked test per-label ranking; mean metrics across seeds. AP prevalence baselines shown explicitly.",
)

for kind in ("roc", "pr", "calibration"):
    fig, axes = plt.subplots(1, 5, figsize=(18, 3.5))
    for j, label in enumerate(CLASSES):
        ax = axes[j]
        for name in names:
            p = predictions[name][:, j]
            if kind == "roc":
                xx, yy, _ = roc_curve(y[:, j], p)
                ax.plot(xx, yy, color=colors[name], label=pretty[name])
            elif kind == "pr":
                precision, recall, _ = precision_recall_curve(y[:, j], p)
                ax.plot(recall, precision, color=colors[name], label=pretty[name])
            else:
                bins = result["runs"][f"{name}:101"]["validation_thresholds"]["labels"][label][
                    "bins"
                ]
                bins = [b for b in bins if b["count"]]
                ax.plot(
                    [b["prediction"] for b in bins],
                    [b["observed"] for b in bins],
                    ".-",
                    color=colors[name],
                    label=pretty[name],
                )
        if kind == "pr":
            ax.axhline(y[:, j].mean(), linestyle=":", color="gray")
        else:
            ax.plot([0, 1], [0, 1], ":", color="gray")
        ax.set(
            xlim=(0, 1),
            ylim=(0, 1),
            title=label,
            xlabel={
                "roc": "False-positive rate",
                "pr": "Recall",
                "calibration": "Mean predicted probability",
            }[kind],
            ylabel={"roc": "Sensitivity", "pr": "Precision", "calibration": "Observed fraction"}[
                kind
            ],
        )
    axes[-1].legend(fontsize=6)
    save(
        fig,
        "locked_" + kind,
        "Locked test, prespecified representative seed101 for legibility; no probability averaging or recalibration. Calibration uses ten fixed equal-width bins.",
    )

fig, axes = plt.subplots(4, 5, figsize=(13, 10))
for i, name in enumerate([n for n in names if n != "xresnet101_final"]):
    for j, label in enumerate(CLASSES):
        info = result["runs"][f"{name}:101"]["validation_thresholds"]["labels"][label]
        cm = np.array(info["confusion_matrix"])
        ax = axes[i, j]
        ax.imshow(cm, cmap="Blues")
        for a in range(2):
            for b in range(2):
                ax.text(
                    b,
                    a,
                    str(cm[a, b]),
                    ha="center",
                    va="center",
                    color="white" if cm[a, b] > cm.max() / 2 else "black",
                )
        ax.set(
            xticks=[0, 1],
            yticks=[0, 1],
            xlabel="Predicted",
            ylabel="Observed",
            title=f"{pretty[name]} / {label}\nt={info['threshold']:.2f}",
        )
save(
    fig,
    "locked_confusion_matrices",
    "Seed101 only; each threshold selected using fold9F1 before lock. Counts are recordings, not independent patients.",
)

probe = json.loads((root / "results/device_probe.json").read_text())
param = {r["model"]: r["parameters"] for r in probe["results"]}
mac = {r["model"]: r["per_window_macs"] for r in probe["results"]}
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for name in ("xresnet101", "flat_cnn", "multiview"):
    m = [r["metadata"] for r in runs if r["metadata"]["model"] == name]
    xs = [
        param[name] / 1e6,
        mac[name] * (7 if name == "xresnet101" else 1) / 1e6,
        np.mean([v["training_seconds"] for v in m]) / 60,
    ]
    for ax, x in zip(axes, xs, strict=True):
        ax.scatter(
            x, result["summary"][name]["macro_auroc"], color=colors[name], s=60, label=pretty[name]
        )
for ax, label in zip(
    axes,
    [
        "Trainable parameters (millions)",
        "Conv/Linear MACs per recording (millions)",
        "Mean training minutes, excluding validation",
    ],
    strict=True,
):
    ax.set(xlabel=label, ylabel="Locked macro AUROC")
    ax.legend(fontsize=7)
save(
    fig,
    "locked_parameter_compute_comparison",
    "MACs exclude FFT, normalization and pooling. XResNet includes seven inference crops. Training time measured on common CPU and concurrent paired-seed processes.",
)

fig, ax = plt.subplots(figsize=(12, 5))
ax.axis("off")
for i, label in enumerate(
    [
        "Limb leads (6)",
        "V1–V3 (3)",
        "V4–V6 (3)",
        "Learned temporal/rhythm (3)",
        "Hann FFT0.5–40Hz (12)",
    ]
):
    yy = 0.9 - i * 0.19
    ax.text(0.02, yy, label, bbox=dict(boxstyle="round", fc="#e4eef7"), fontsize=10)
    ax.annotate("", xy=(0.55, 0.52), xytext=(0.38, yy), arrowprops=dict(arrowstyle="->"))
ax.text(0.48, 0.52, "5 encoders ×64\nConcatenate320", bbox=dict(boxstyle="round", fc="#cbe6dc"))
ax.annotate("", xy=(0.77, 0.52), xytext=(0.68, 0.52), arrowprops=dict(arrowstyle="->"))
ax.text(0.76, 0.52, "Fusion128→64\n5 sigmoid outputs", bbox=dict(boxstyle="round", fc="#f8e5c6"))
ax.set_title("Conceptual diagram — five-view ECG architecture; seven future selectable modules")
save(
    fig,
    "conceptual_architecture",
    "Conceptual architecture, not an empirical result. Deterministic derived views contain no independent measurements; future selective transfer remains untested.",
)
write_json(root / "results/figure_inventory.json", inventory)
print(json.dumps(dict(figures=len(inventory), formats=["png", "pdf"])))
