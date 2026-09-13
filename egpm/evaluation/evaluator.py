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
