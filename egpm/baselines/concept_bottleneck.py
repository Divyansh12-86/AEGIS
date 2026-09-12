"""Concept Bottleneck baseline without temporal structure (PRD §21, M9).

Does the TEMPORAL structure (the HSMM) add value beyond a flat concept
bottleneck? This baseline discretizes each window into one of K learned
concepts (k-means on raw windows — the non-temporal analog of the VQ
event), then classifies/regresses from concept-occurrence histograms
alone. No sequence model, no states, no durations.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class WindowKMeans:
    """Tiny k-means over flattened windows (concept induction, deterministic
    seeding via k-means++ with fixed rng)."""

    def __init__(self, n_clusters: int, n_iter: int = 15, seed: int = 0):
        if n_clusters < 1:
            raise ValueError("n_clusters must be >= 1")
        self.K = n_clusters
        self.n_iter = n_iter
        self.seed = seed
        self.centroids: np.ndarray | None = None

    def fit(self, flat_windows: np.ndarray) -> "WindowKMeans":
        X = np.asarray(flat_windows, dtype=np.float64)
        rng = np.random.default_rng(self.seed)
        # k-means++ init
        n = X.shape[0]
        centers = [X[rng.integers(n)]]
        for _ in range(1, self.K):
            d2 = np.min(
                ((X[:, None, :] - np.array(centers)[None, :, :]) ** 2).sum(-1),
                axis=1,
            )
            probs = d2 / max(d2.sum(), 1e-12)
            centers.append(X[rng.choice(n, p=probs)])
        C = np.array(centers)
        for _ in range(self.n_iter):
            d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
            lab = d2.argmin(axis=1)
            for k in range(self.K):
                pts = X[lab == k]
                if len(pts):
                    C[k] = pts.mean(axis=0)
                else:
                    C[k] = X[rng.integers(n)]
        self.centroids = C
        return self

    def assign(self, flat_windows: np.ndarray) -> np.ndarray:
        if self.centroids is None:
            raise RuntimeError("fit first")
        X = np.asarray(flat_windows, dtype=np.float64)
        d2 = ((X[:, None, :] - self.centroids[None, :, :]) ** 2).sum(-1)
        return d2.argmin(axis=1)


class ConceptBottleneck(nn.Module):
    """Flat concept histogram -> task heads (PRD §21 CBM baseline).

    The concept layer is k-means over raw windows (no learned encoder, no
    temporal modeling — isolating the HSMM's contribution per RQ2). The
    head consumes only the concept OCCURRENCE HISTOGRAM of the run.
    """

    def __init__(self, n_concepts: int, out_dim: int = 1, hidden: int = 32):
        super().__init__()
        self.n_concepts = n_concepts
        self.head = nn.Sequential(
            nn.Linear(n_concepts, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, concept_histograms: torch.Tensor) -> torch.Tensor:
        """histograms: [B, K] (row-normalized counts) -> [B, out_dim]."""
        if concept_histograms.shape[-1] != self.n_concepts:
            raise ValueError(
                f"histograms must be [B, {self.n_concepts}]; "
                f"got {concept_histograms.shape}"
            )
        return self.head(concept_histograms)

    # ------------------------------------------------------------------
    @staticmethod
    def histogram(concept_ids: np.ndarray, n_concepts: int) -> np.ndarray:
        """Concept-occurrence histogram of one run (row-normalized)."""
        ids = np.asarray(concept_ids, dtype=np.int64)
        h = np.bincount(ids, minlength=n_concepts).astype(np.float64)
        return h / max(h.sum(), 1)
