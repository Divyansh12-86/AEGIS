"""Ablation A2: no HSMM — pooled event model (PRD §22, M10).

The event sequence is pooled (unigram occurrence histogram) directly to the
prediction heads, with no state/duration model at all. Tests whether the
HSMM is earning its cost (RQ2's other half).

Also provides the pooling feature used by A2's heads: unigram frequencies
plus run-level co-occurrence (the bag-of-events representation).
"""
from __future__ import annotations

import numpy as np


class PooledEventModel:
    """Bag-of-events representation (no temporal model).

    fit() memorizes the training unigram distribution (normal-only for the
    anomaly task, per PRD §10.4). score_run() flags rare/unseen events —
    the duration-free, order-free anomaly floor.
    """

    def __init__(self, n_events: int, smoothing: float = 0.5):
        if n_events < 1:
            raise ValueError("n_events must be >= 1")
        self.K = n_events
        self.smoothing = smoothing
        self.log_unigram: np.ndarray | None = None

    def fit(self, seqs) -> "PooledEventModel":
        counts = np.zeros(self.K)
        for v in seqs:
            v = np.asarray(v, dtype=np.int64)
            counts += np.bincount(v, minlength=self.K)
        counts += self.smoothing
        self.log_unigram = np.log(counts / counts.sum())
        return self

    # ------------------------------------------------------------------
    def features(self, v: np.ndarray) -> np.ndarray:
        """Unigram histogram [K] (row-normalized) — A2's head input."""
        if self.log_unigram is None:
            raise RuntimeError("fit first")
        v = np.asarray(v, dtype=np.int64)
        h = np.bincount(v, minlength=self.K).astype(np.float64)
        return h / max(h.sum(), 1.0)

    def score_run(self, v: np.ndarray) -> np.ndarray:
        """Per-window surprisal under the unigram model (order-free)."""
        if self.log_unigram is None:
            raise RuntimeError("fit first")
        v = np.asarray(v, dtype=np.int64)
        return -self.log_unigram[v]

    # ------------------------------------------------------------------
    @staticmethod
    def cooccurrence_features(v: np.ndarray, n_events: int) -> np.ndarray:
        """Bigram co-occurrence histogram (flattened [K*K]) — still pooled
        (no segmentation), capturing simple pair statistics."""
        v = np.asarray(v, dtype=np.int64)
        counts = np.zeros((n_events, n_events))
        np.add.at(counts, (v[:-1], v[1:]), 1.0)
        flat = counts.ravel()
        return flat / max(flat.sum(), 1.0)
