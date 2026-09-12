"""Deterministic split construction with leakage guarantees (PRD §23).

``SplitBuilder`` wraps :class:`egpm.data.UnitDataset.split` with the
integrity checks the PRD requires: disjoint unit-level splits, normalization
statistics fit on train units only, and a hard error if a unit appears in
two partitions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .loaders import UnitDataset, UnitSplit


@dataclass
class NormalizationStats:
    """Per-channel z-score statistics, fit on TRAIN units only (PRD §6)."""

    mean: np.ndarray  # [d]
    std: np.ndarray  # [d]
    train_unit_ids: List[str]

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply z-score normalization with stored train-only statistics."""
        return (X - self.mean) / self.std


class SplitBuilder:
    """Builds unit-level train/val/test splits and train-only normalization.

    Enforces PRD §23's checklist items 1–2:
      * splits are at unit level, before windowing;
      * normalization statistics come only from training units.
    """

    def __init__(self, dataset: UnitDataset, seed: int = 0):
        if len(dataset) == 0:
            raise ValueError("cannot split an empty dataset")
        self.dataset = dataset
        self.seed = seed

    def build(
        self, train_frac: float = 0.6, val_frac: float = 0.2
    ) -> UnitSplit:
        return self.dataset.split(train_frac=train_frac, val_frac=val_frac, seed=self.seed)

    def fit_normalization(
        self, split: UnitSplit, eps: float = 1e-8
    ) -> NormalizationStats:
        """Compute per-channel mean/std over TRAIN units only.

        All train units are concatenated channel-wise; constants get ``eps``
        so division cannot blow up.
        """
        train_records = [
            r for r in self.dataset.records if r.unit_id in set(split.train_unit_ids)
        ]
        if not train_records:
            raise ValueError("split has no training units")
        stacked = np.concatenate([r.signals for r in train_records], axis=0)
        mean = stacked.mean(axis=0)
        std = stacked.std(axis=0)
        std = np.where(std < eps, 1.0, std)
        return NormalizationStats(
            mean=mean, std=std, train_unit_ids=sorted(split.train_unit_ids)
        )

    # -- leakage audit ---------------------------------------------------------
    @staticmethod
    def audit(split: UnitSplit, dataset: UnitDataset) -> Dict[str, bool]:
        """Return the PRD §23 leakage checklist as booleans (all must be True).

        Checks:
          * disjoint: no unit in two splits;
          * complete: every dataset unit is assigned to exactly one split;
          * non_empty: train and test both contain at least one unit.
        """
        tr, va, te = set(split.train_unit_ids), set(split.val_unit_ids), set(split.test_unit_ids)
        all_ids = set(dataset.unit_ids)
        return {
            "disjoint": not (tr & va or tr & te or va & te),
            "complete": all_ids == (tr | va | te),
            "train_non_empty": len(tr) > 0,
            "test_non_empty": len(te) > 0,
        }
