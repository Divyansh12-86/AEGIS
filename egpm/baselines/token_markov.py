"""VQ-events + plain first-order Markov scorer (PRD Deliverable E arm ii).

The mid-point between the continuous-AE baseline and the full HSMM: learned
VQ event tokens scored by a plain first-order transition table — events
without explicit duration modeling. This isolates exactly the PRD's
Deliverable-F question: does the HSMM's duration modeling earn its cost
over a plain Markov chain on the same discrete events?
"""
from __future__ import annotations

from typing import Dict

import numpy as np


class TokenMarkovModel:
    """First-order Markov chain over event tokens (normal-trained)."""

    def __init__(self, n_events: int, smoothing: float = 1.0):
        if n_events < 1:
            raise ValueError("n_events must be >= 1")
        self.K = n_events
        self.smoothing = smoothing
        self.log_trans: np.ndarray | None = None

    def fit(self, seqs) -> "TokenMarkovModel":
        """Fit the transition table on normal sequences (add-smoothing)."""
        counts = np.zeros((self.K, self.K))
        for v in seqs:
            v = np.asarray(v, dtype=np.int64)
            np.add.at(counts, (v[:-1], v[1:]), 1.0)
        counts += self.smoothing
        self.log_trans = np.log(counts / counts.sum(axis=1, keepdims=True))
        return self

    def score_run(self, v: np.ndarray) -> np.ndarray:
        """Per-window surprisal -log P(v_t | v_{t-1}); first window uses the
        smoothed marginal."""
        if self.log_trans is None:
            raise RuntimeError("fit first")
        v = np.asarray(v, dtype=np.int64)
        T = len(v)
        out = np.zeros(T)
        # marginal from the transition table's stationary-ish row mean
        row = np.exp(self.log_trans).mean(axis=0)
        row = row / row.sum()
        out[0] = -np.log(max(row[v[0]], 1e-300))
        for t in range(1, T):
            out[t] = -self.log_trans[v[t - 1], v[t]]
        return out
