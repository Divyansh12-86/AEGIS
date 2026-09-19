"""PRD §23 statistical reporting: unit-level aggregation, bootstrap CIs,
paired Wilcoxon. Windows within a unit are NOT independent — every reported
interval/test aggregates to the unit first, then bootstraps units."""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy import stats as sps

from .evaluator import auroc


def auroc_bootstrap_ci(
    scores: Sequence[float],
    labels: Sequence[int],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """Clip-level bootstrap CI for AUROC (units = clips, PRD §23 item 4/9).

    Resamples clips with replacement, recomputes AUROC per replicate,
    returns (point_auroc, lo, hi). Degenerate replicates (one class absent)
    are skipped.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    point = auroc(scores, labels)
    rng = np.random.default_rng(seed)
    n = len(scores)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        s, l = scores[idx], labels[idx]
        if (l == 1).any() and (l == 0).any():
            vals.append(auroc(s, l))
    if not vals:
        return point, float("nan"), float("nan")
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)


def per_unit_metrics(
    preds: Sequence[np.ndarray],
    trues: Sequence[np.ndarray],
) -> np.ndarray:
    """RMSE per unit — the independent data points (PRD §23 item 4).

    preds/trues: one array of per-window predictions/labels per unit.
    """
    if len(preds) != len(trues) or len(preds) == 0:
        raise ValueError("need equal, non-empty per-unit lists")
    out = [
        float(np.sqrt(((np.asarray(p) - np.asarray(t)) ** 2).mean()))
        for p, t in zip(preds, trues)
    ]
    return np.asarray(out, dtype=np.float64)


def bootstrap_ci(
    values: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float]:
    """Percentile bootstrap CI over units (the independent samples)."""
    v = np.asarray(values, dtype=np.float64)
    if len(v) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    means = v[idx].mean(axis=1)
    return tuple(np.quantile(means, [alpha / 2, 1 - alpha / 2]).tolist())


def paired_wilcoxon(
    a: np.ndarray, b: np.ndarray, alternative: str = "two-sided"
) -> Dict[str, float]:
    """Paired Wilcoxon over units: EGPM arm vs baseline arm (PRD §23 item 8).

    NaN-safe: units dropped pairwise (a unit with NaN in either arm is
    excluded from both). n<6 → p=NaN with a note (too few units to test).
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        raise ValueError("paired arrays must share shape (units)")
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    n = len(a)
    if n < 2 or np.allclose(a, b):
        return {"statistic": float("nan"), "p": float("nan"), "n": int(n),
                "note": "no paired differences to test"}
    if n < 6:
        return {"statistic": float("nan"), "p": float("nan"), "n": int(n),
                "note": f"only {n} units; Wilcoxon underpowered (needs >=6)"}
    res = sps.wilcoxon(a, b, alternative=alternative)
    return {"statistic": float(res.statistic), "p": float(res.pvalue),
            "n": int(n)}


def summarize_units(
    per_arm_unit: Dict[str, np.ndarray],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
    reference_arm: str = "",
) -> Dict:
    """Full §23 block: per-arm unit RMSE mean + bootstrap CI, plus paired
    Wilcoxon of every arm against `reference_arm`."""
    out: Dict[str, Dict] = {}
    for arm, v in per_arm_unit.items():
        v = np.asarray(v, float)
        lo, hi = bootstrap_ci(v, n_boot, alpha, seed)
        out[arm] = {
            "mean": float(np.mean(v)) if len(v) else float("nan"),
            "ci95": [lo, hi],
            "per_unit": v.tolist(),
            "n_units": int(len(v)),
        }
    if reference_arm and reference_arm in out:
        for arm in out:
            if arm == reference_arm:
                continue
            out[arm]["wilcoxon_vs_ref"] = paired_wilcoxon(
                per_arm_unit[arm], per_arm_unit[reference_arm]
            )
    return out
