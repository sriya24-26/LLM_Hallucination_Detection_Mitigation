"""Accuracy, precision, recall, F1, AUROC and the bootstrap confidence interval.  Owner: Kriti.
Positive class = hallucinated (label 1). The bootstrap resamples whole examples (example_id), never single claims."""
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import CFG, SEED  # noqa: E402


def confusion_metrics(y, pred):
    y, pred = np.asarray(y).astype(bool), np.asarray(pred).astype(bool)
    tp, fp = int((y & pred).sum()), int((~y & pred).sum())
    tn, fn = int((~y & ~pred).sum()), int((y & ~pred).sum())
    n = tp + fp + tn + fn
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(accuracy=(tp + tn) / n if n else float("nan"), precision=prec, recall=rec, f1=f1,
                tp=tp, fp=fp, tn=tn, fn=fn)


def auroc(y, score):
    y = np.asarray(y).astype(int)
    if len(y) == 0 or y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, score))


def bootstrap_ci(groups, fn, n=None, seed=SEED, alpha=0.05):
    """95% interval of fn(row_indices) when whole groups (examples) are resampled with replacement."""
    n = n or CFG["evaluation"]["bootstrap_resamples"]
    uniq, inv = np.unique(np.asarray(groups), return_inverse=True)
    members = [np.where(inv == i)[0] for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        pick = rng.integers(0, len(uniq), len(uniq))
        vals.append(fn(np.concatenate([members[i] for i in pick])))
    lo, hi = np.nanpercentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)
