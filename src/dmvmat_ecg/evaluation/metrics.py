import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from dmvmat_ecg.labels.superclasses import CLASSES


def ranking(y, p):
    y, p = np.asarray(y), np.asarray(p)
    if y.shape != p.shape or y.ndim != 2 or y.shape[1] != 5 or not np.isfinite(p).all():
        raise ValueError("Aligned five-label scores required")
    auc = []
    ap = []
    for j in range(5):
        valid = len(np.unique(y[:, j])) == 2
        auc.append(float(roc_auc_score(y[:, j], p[:, j])) if valid else None)
        ap.append(float(average_precision_score(y[:, j], p[:, j])) if y[:, j].sum() > 0 else None)
    return dict(
        macro_auroc=float(np.mean(auc)) if all(a is not None for a in auc) else None,
        macro_ap=float(np.mean(ap)) if all(a is not None for a in ap) else None,
        micro_ap=float(average_precision_score(y.ravel(), p.ravel())),
        per_label_auroc=auc,
        per_label_ap=ap,
    )


def complete(y, p, thresholds):
    result = ranking(y, p)
    labels = {}
    for j, name in enumerate(CLASSES):
        scores = p[:, j]
        pred = scores >= thresholds[j]
        truth = y[:, j]
        tn, fp, fn, tp = confusion_matrix(truth, pred, labels=[0, 1]).ravel()
        bins = []
        for b in range(10):
            ix = np.minimum((scores * 10).astype(int), 9) == b
            bins.append(
                dict(
                    lower=b / 10,
                    upper=(b + 1) / 10,
                    count=int(ix.sum()),
                    prediction=float(scores[ix].mean()) if ix.any() else None,
                    observed=float(truth[ix].mean()) if ix.any() else None,
                )
            )
        labels[name] = dict(
            auroc=result["per_label_auroc"][j],
            ap=result["per_label_ap"][j],
            prevalence=float(truth.mean()),
            precision=float(precision_score(truth, pred, zero_division=0)),
            recall=float(recall_score(truth, pred, zero_division=0)),
            specificity=float(tn / (tn + fp)),
            f1=float(f1_score(truth, pred, zero_division=0)),
            brier=float(brier_score_loss(truth, scores)),
            ece=sum(b["count"] * abs(b["prediction"] - b["observed"]) for b in bins if b["count"])
            / len(truth),
            bins=bins,
            confusion_matrix=[[int(tn), int(fp)], [int(fn), int(tp)]],
            threshold=float(thresholds[j]),
        )
    result.update(
        labels=labels,
        macro_f1=float(np.mean([v["f1"] for v in labels.values()])),
        mean_brier=float(np.mean([v["brier"] for v in labels.values()])),
        mean_ece=float(np.mean([v["ece"] for v in labels.values()])),
    )
    return result


def thresholds_from_validation(y, p):
    thresholds = []
    for j in range(5):
        candidates = [
            (float(f1_score(y[:, j], p[:, j] >= t, zero_division=0)), float(t))
            for t in np.arange(5, 96) / 100
        ]
        best = max(s for s, _ in candidates)
        thresholds.append(
            min((t for s, t in candidates if s == best), key=lambda t: (abs(t - 0.5), t))
        )
    return thresholds
