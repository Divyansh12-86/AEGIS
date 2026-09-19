"""Evaluation metrics (PRD §20, M11 — subset for the first experiment)."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy.stats import rankdata


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC AUC via the rank statistic (average-rank ties, no sklearn)."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    pos = labels == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUROC needs both classes present")
    ranks = rankdata(scores)
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def auprc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Average precision (step-integral PR area, sklearn-compatible).

    AP = sum over distinct thresholds t (descending) of (R_prev - R_t) * P_t,
    where selection at t is score >= t (ties grouped into one threshold) and
    a (R=0, P=1) sentinel closes the curve. No sklearn dependency; verified
    equal to sklearn's average_precision_score including tie handling.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    n_pos = int((labels == 1).sum())
    if n_pos == 0 or (labels == 0).sum() == 0:
        raise ValueError("AUPRC needs both classes present")
    prec, rec = [], []
    for t in np.unique(scores):  # ascending: lowest threshold first
        sel = scores >= t
        tp = int(((labels == 1) & sel).sum())
        fp = int(((labels == 0) & sel).sum())
        prec.append(tp / (tp + fp))
        rec.append(tp / n_pos)
    prec.append(1.0)
    rec.append(0.0)  # sentinel: empty selection
    prec, rec = np.asarray(prec), np.asarray(rec)
    return float(max(0.0, -np.sum(np.diff(rec) * prec[:-1])))


def f1_at_threshold(
    scores: np.ndarray, labels: np.ndarray, threshold: float
) -> Tuple[float, float, float]:
    """Binary precision/recall/F1 at a given threshold."""
    pred = np.asarray(scores) > threshold
    labels = np.asarray(labels, dtype=int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def rul_metrics(pred: np.ndarray, true: np.ndarray) -> Dict[str, float]:
    """RMSE, MAE, and NASA/PHM asymmetric score (PRD §20 RUL row)."""
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    err = pred - true
    rmse = float(np.sqrt((err ** 2).mean()))
    mae = float(np.abs(err).mean())
    # NASA asymmetric score: late RUL predictions penalized exponentially
    s = np.where(
        err < 0,
        np.exp(-err / 13) - 1,   # over-estimated RUL (pred > true): heavy
        1 - np.exp(err / 10),    # under-estimated: lighter
    )
    return {"RMSE": rmse, "MAE": mae, "PHM_score": float(s.mean())}


def early_detection_rate(
    scores: np.ndarray, labels: np.ndarray, threshold: float,
    horizon: int = 10,
) -> float:
    """Fraction of anomalous units detected within `horizon` windows of onset
    (PRD §20 Anomaly row, early detection rate). Operates on a binary label
    sequence: first flagged window within [onset, onset+horizon)."""
    labels = np.asarray(labels, dtype=int)
    pred = np.asarray(scores) > threshold
    onsets = np.nonzero((labels == 1) & (np.concatenate([[0], labels[:-1]]) == 0))[0]
    if len(onsets) == 0:
        raise ValueError("no anomaly onsets in labels")
    hits = 0
    for o in onsets:
        hi = min(o + horizon, len(pred))
        if pred[o:hi].any():
            hits += 1
    return hits / len(onsets)
