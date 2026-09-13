"""Interpretability validation metrics (PRD §24, M-experimental).

Six operational metrics, per the PRD's definitions (and its renamings):

  * Event coherence — do windows sharing an event token exhibit similar raw
    sensor behavior? Silhouette score over window features grouped by token
    is ONE signal among several (PRD §24 explicitly says not the sole one),
    so we also report the pooled within-vs-between distance ratio.
  * Event stability — does the same vocabulary recur across seeds after
    Hungarian/optimal matching? Reported as (a) assignment agreement under
    optimal matching and (b) matched usage-distribution overlap.
  * Cross-unit consistency — trained on a subset of units, does the
    event/state distribution transfer to held-out units? Jensen-Shannon
    divergence between train and held-out usage distributions.
  * Temporal validity — do inferred state transitions align with known
    fault-onset timestamps where ground truth exists? Reported as boundary
    alignment: mean |inferred_boundary - true_boundary| and detection lag.
  * Explanation faithfulness — mask the flagged event/state subsequence,
    recompute the anomaly score, measure the reduction. THE primary
    faithfulness test (replaces the mislabeled "rule fidelity", Decision 9).
  * Anomaly-score rank correlation — Spearman correlation between the
    anomaly score and the binary anomaly label (the renamed metric);
    discriminative power only, never reported as faithfulness.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr


# ---------------------------------------------------------------------------
# Event coherence (PRD §24 row 1)
# ---------------------------------------------------------------------------
def event_coherence(
    window_features: np.ndarray, event_ids: np.ndarray
) -> Dict[str, float]:
    """Coherence of windows grouped by event token.

    window_features: [N, d] raw sensor features per window (preprocessed X
    flattened, per PRD §24 'similar raw sensor behavior').
    event_ids: [N] token assignment.

    Returns:
      silhouette: mean silhouette coefficient over windows with the
        token-grouping (in [-1, 1]; higher = more coherent). Degenerate
        single-token groupings return 0 with a note.
      within_between_ratio: mean within-token pairwise distance divided by
        mean between-token pairwise distance (< 1 = coherent).
    """
    X = np.asarray(window_features, dtype=np.float64)
    ids = np.asarray(event_ids)
    N = len(ids)
    if X.shape[0] != N:
        raise ValueError("features/events length mismatch")
    tokens, counts = np.unique(ids, return_counts=True)
    out: Dict[str, float] = {"silhouette": 0.0, "within_between_ratio": float("nan"),
                             "n_tokens": len(tokens)}
    if len(tokens) < 2 or N < 3:
        return out
    # pairwise distance matrix (N can be large; cap for safety)
    if N > 4000:
        idx = np.random.default_rng(0).choice(N, 4000, replace=False)
        X, ids = X[idx], ids[idx]
        N = len(ids)
    d2 = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    same = ids[:, None] == ids[None, :]
    # silhouette per point: (b - a) / max(a, b)
    sil = np.zeros(N)
    for i in range(N):
        own = ids == ids[i]
        own[i] = False
        if own.sum() == 0 or (own == False).sum() == 0:
            sil[i] = 0.0
            continue
        a = d2[i, own].mean()
        # nearest other token's mean distance
        b = np.inf
        for t in np.unique(ids):
            if t == ids[i]:
                continue
            b = min(b, d2[i, ids == t].mean())
        sil[i] = 0.0 if max(a, b) == 0 else (b - a) / max(a, b)
    out["silhouette"] = float(np.mean(sil))
    # pooled distance ratio
    within = d2[same & ~np.eye(N, dtype=bool)]
    between = d2[~same]
    if len(within) and len(between):
        out["within_between_ratio"] = float(within.mean() / between.mean())
    return out


# ---------------------------------------------------------------------------
# Event stability across seeds (PRD §24 row 2)
# ---------------------------------------------------------------------------
def _hungarian_usage_match(usage_a: np.ndarray, usage_b: np.ndarray) -> float:
    """Optimal (Hungarian) overlap of two usage histograms, invariant to
    token relabeling. Returns matched-overlap in [0, 1]."""
    K = max(len(usage_a), len(usage_b))
    a = np.zeros(K); a[: len(usage_a)] = usage_a
    b = np.zeros(K); b[: len(usage_b)] = usage_b
    a /= a.sum(); b /= b.sum()
    # cost of matching token i (a) to token j ( b ): -min(a_i, b_j) overlap
    cost = -np.minimum(a[:, None], b[None, :])
    row, col = linear_sum_assignment(cost)
    return float(np.minimum(a[row], b[col]).sum())


def event_stability(
    event_runs_seed_a: List[np.ndarray],
    event_runs_seed_b: List[np.ndarray],
    n_codes: Optional[int] = None,
) -> Dict[str, float]:
    """Vocabulary recurrence across two seeds after optimal matching.

    Compares per-seed usage histograms over the SAME units (callers must
    tokenize identical runs with both seeds' encoders/codebooks).
    Returns matched usage overlap in [0, 1] (1 = identical distribution
    modulo relabeling).
    """
    def usage(runs):
        nonlocalK = n_codes
        K = nonlocalK if nonlocalK is not None else (
            int(max(r.max() for r in runs)) + 1 if runs else 1
        )
        h = np.zeros(K)
        for r in runs:
            h += np.bincount(np.asarray(r, dtype=np.int64), minlength=K)
        return h
    ua, ub = usage(event_runs_seed_a), usage(event_runs_seed_b)
    overlap = _hungarian_usage_match(ua, ub)
    return {"matched_usage_overlap": overlap, "n_codes": len(ua)}


def hungarian_state_match(
    posterior_runs_seed_a: List[np.ndarray],
    posterior_runs_seed_b: List[np.ndarray],
) -> float:
    """Optimal state-index alignment between two seeds' posteriors.

    Aligns per-state mean posterior mass (rows are states) via the
    Hungarian algorithm and returns the matched mass overlap in [0, 1]
    (PRD §25 'state permutation across seeds' mitigation — always match
    before comparing across seeds).
    """
    a = np.stack([
        np.concatenate(runs, axis=0).mean(axis=0)
        for runs in [posterior_runs_seed_a]
    ])[0]
    b = np.concatenate([np.concatenate(runs, axis=0).mean(axis=0)
                        for runs in [posterior_runs_seed_b]])
    M = len(a)
    if len(b) != M:
        raise ValueError("seeds must share state count")
    a /= a.sum(); b /= b.sum()
    cost = -np.minimum(a[:, None], b[None, :])
    row, col = linear_sum_assignment(cost)
    return float(np.minimum(a[row], b[col]).sum())


# ---------------------------------------------------------------------------
# Cross-unit consistency (PRD §24 row 3)
# ---------------------------------------------------------------------------
def cross_unit_consistency(
    train_event_runs: List[np.ndarray],
    heldout_event_runs: List[np.ndarray],
    n_codes: Optional[int] = None,
) -> Dict[str, float]:
    """Jensen-Shannon divergence of event usage: train vs held-out units."""
    def usage(runs):
        K = n_codes if n_codes is not None else (
            int(max(r.max() for r in runs)) + 1 if runs else 1
        )
        h = np.zeros(K)
        for r in runs:
            h += np.bincount(np.asarray(r, dtype=np.int64), minlength=K)
        return h / max(h.sum(), 1e-12)
    from scipy.spatial.distance import jensenshannon
    jsd = float(jensenshannon(usage(train_event_runs), usage(heldout_event_runs)))
    return {"jsd": jsd, "jsd_sqrt_normalized": jsd}


# ---------------------------------------------------------------------------
# Temporal validity (PRD §24 row 4)
# ---------------------------------------------------------------------------
def temporal_validity(
    inferred_boundaries: np.ndarray, true_onsets: np.ndarray, tolerance: int = 2
) -> Dict[str, float]:
    """Alignment of inferred state-segment boundaries with known fault onsets.

    inferred_boundaries: indices where the Viterbi path changes state.
    true_onsets: ground-truth change indices (e.g. N-CMAPSS health-state
    transitions).
    Returns mean absolute boundary lag, matched-detection rate within
    `tolerance` windows, and detection lag (inferred minus true, matched).
    """
    ib = np.asarray(sorted(set(int(x) for x in inferred_boundaries)))
    to = np.asarray(sorted(set(int(x) for x in true_onsets)))
    if len(to) == 0:
        raise ValueError("no true onsets provided")
    matched, lags = [], []
    for t in to:
        if len(ib) == 0:
            break
        j = int(np.argmin(np.abs(ib - t)))
        lag = int(ib[j] - t)
        if abs(lag) <= tolerance:
            matched.append(t)
            lags.append(lag)
    return {
        "detection_rate": len(matched) / len(to),
        "mean_abs_lag": float(np.mean(np.abs(lags))) if lags else float("nan"),
        "mean_lag": float(np.mean(lags)) if lags else float("nan"),
        "n_true_onsets": len(to),
        "n_inferred_boundaries": len(ib),
    }


# ---------------------------------------------------------------------------
# Explanation faithfulness (PRD §24 row 5; replaces "rule fidelity")
# ---------------------------------------------------------------------------
def explanation_faithfulness(
    v_seq: np.ndarray,
    flagged_windows: List[int],
    hsmm,
    scorer,
) -> Dict[str, float]:
    """Mask the flagged event/state subsequence -> recompute the anomaly
    score -> measure the reduction (PRD §24's primary faithfulness test).

    State-aware masking: each flagged window's token is replaced by its
    Viterbi state's MOST-LIKELY emission — neutralizing emission surprisal
    without fabricating implausible transitions (a modal-token mask can
    RAISE scores by creating context mismatch, which measures the wrong
    thing). The score must DROP if the explanation reflects what actually
    drove the prediction; reduction > 0 is the pass signal.
    """
    v = np.asarray(v_seq, dtype=np.int64).copy()
    if len(flagged_windows) == 0:
        return {"score_reduction": 0.0, "before": float("nan"),
                "after": float("nan"), "n_flagged": 0}
    from ..grammar.viterbi import SegmentalViterbi
    vit = SegmentalViterbi(hsmm).decode(v)
    before = float(np.mean(scorer.score_sequence(hsmm, v, viterbi=vit).score))
    for t in flagged_windows:
        s = int(vit.path[t])
        v[t] = int(np.argmax(hsmm.B[s]))
    after = float(np.mean(scorer.score_sequence(hsmm, v).score))
    return {
        "score_reduction": before - after,
        "before": before,
        "after": after,
        "n_flagged": len(flagged_windows),
    }


# ---------------------------------------------------------------------------
# Anomaly-score rank correlation (renamed per PRD Decision 9)
# ---------------------------------------------------------------------------
def anomaly_score_rank_correlation(
    scores: np.ndarray, labels: np.ndarray
) -> Dict[str, float]:
    """Spearman correlation between anomaly scores and binary labels.

    Measures DISCRIMINATIVE POWER only (PRD Decision 9: never call this
    'rule fidelity' — faithfulness is the masking experiment above).
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    if len(scores) != len(labels):
        raise ValueError("scores/labels length mismatch")
    if len(np.unique(labels)) < 2:
        return {"spearman": float("nan"), "n": len(labels)}
    rho, p = spearmanr(scores, labels)
    return {"spearman": float(rho), "p_value": float(p), "n": len(labels)}
