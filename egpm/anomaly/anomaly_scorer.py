"""Grammar-native anomaly scoring (PRD §11, M6).

At window i, given the Viterbi (or MAP marginal) state s_hat_i and elapsed
dwell tau_i, the score combines three surprise signals:

    a_i = -log B(v_i | s_hat_i)
          - lambda_trans * 1[transition at i] * log A(s_hat_{i-1}, s_hat_i)
          - lambda_dur  * log Sbar(tau_i | s_hat_i)

where Sbar is the duration SURVIVAL function (probability of dwelling at
least tau_i steps), so implausibly long or short stays are penalized without
requiring the segment to have ended (PRD §11 wording).

Thresholding: percentile of a_i over the VALIDATION split's normal-only
windows. Test labels are never touched for threshold selection (PRD §11,
§29 — verified by a dedicated test).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from ..grammar.hsmm import HSMM
from ..grammar.viterbi import SegmentalViterbi, ViterbiResult


@dataclass
class AnomalyComponents:
    """Per-window decomposition of the anomaly score (PRD §18 anomaly_causes)."""

    emission_surprisal: np.ndarray  # [T]
    transition_surprisal: np.ndarray  # [T] (0 where no transition)
    duration_surprisal: np.ndarray  # [T]
    score: np.ndarray  # [T] weighted sum
    flagged_component: List[str]  # [T] dominant signal at each window


@dataclass
class AnomalyScorer:
    """Grammar-native scorer with validation-only thresholding."""

    lambda_trans: float = 1.0
    lambda_dur: float = 1.0
    threshold_percentile: float = 99.0
    threshold: Optional[float] = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.threshold_percentile <= 0 or self.threshold_percentile > 100:
            raise ValueError("threshold_percentile must be in (0, 100]")

    # ------------------------------------------------------------------
    def score_sequence(
        self, hsmm: HSMM, v_seq: np.ndarray, viterbi: Optional[ViterbiResult] = None
    ) -> AnomalyComponents:
        """Score all windows of one event sequence."""
        v_seq = np.asarray(v_seq, dtype=np.int64)
        T = len(v_seq)
        if viterbi is None:
            viterbi = SegmentalViterbi(hsmm).decode(v_seq)
        path = viterbi.path
        tau = viterbi.duration_estimates

        with np.errstate(divide="ignore"):
            logB = np.log(np.clip(hsmm.B, 1e-300, None))
            logA = np.log(np.clip(hsmm.A, 1e-300, None))
        S = hsmm.D.survival()  # [M, Dmax+1]

        emission = -logB[path, v_seq]  # [T]
        transition = np.zeros(T)
        for t in range(1, T):
            if path[t] != path[t - 1]:
                transition[t] = -logA[path[t - 1], path[t]]
        # duration surprisal: -log S(tau) — mass of dwelling >= tau
        dur = np.empty(T)
        for t in range(T):
            tau_t = int(np.clip(tau[t], 1, hsmm.d_max))
            s_t = path[t]
            dur[t] = -np.log(max(S[s_t, tau_t], 1e-300))

        score = emission \
            + self.lambda_trans * transition \
            + self.lambda_dur * dur

        # dominant component per window (for the explanation object)
        flagged: List[str] = []
        for t in range(T):
            parts = {
                "emission": emission[t],
                "transition": self.lambda_trans * transition[t],
                "duration": self.lambda_dur * dur[t],
            }
            flagged.append(max(parts, key=parts.get))
        return AnomalyComponents(
            emission_surprisal=emission,
            transition_surprisal=transition,
            duration_surprisal=dur,
            score=score,
            flagged_component=flagged,
        )

    # ------------------------------------------------------------------
    def fit_threshold(self, val_scores: np.ndarray) -> float:
        """Set the threshold from VALIDATION-split normal-only scores ONLY.

        PRD §11: e.g. 99th percentile over validation normal windows. Test
        data must never reach this function (enforced by test).
        """
        if len(val_scores) == 0:
            raise ValueError("validation scores empty — cannot set threshold")
        val_scores = np.asarray(val_scores, dtype=np.float64)
        self.threshold = float(np.percentile(val_scores, self.threshold_percentile))
        return self.threshold

    # ------------------------------------------------------------------
    def classify(self, scores: np.ndarray) -> np.ndarray:
        """Binary anomaly decisions at the fitted threshold."""
        if self.threshold is None:
            raise RuntimeError(
                "threshold not fitted — call fit_threshold(validation scores) first"
            )
        return (np.asarray(scores) > self.threshold).astype(np.int64)
