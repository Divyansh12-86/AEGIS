"""Explicit-duration HSMM core (PRD §10, M4).

G_EGPM = (pi, A, B, D):
  * pi  in Delta^(M-1)      — initial state distribution
  * A   [M, M], ZERO diagonal — embedded transition matrix; dwell is carried
    by D, NOT by self-transitions (avoids double-counting dwell in RUL, §12)
  * B(v | m)                — categorical emission over K event tokens
  * D(tau | m)              — duration distribution, support {1..D_max}
    (parametric: discretized Gamma / shifted negative binomial, or
    non-parametric histogram — PRD §10.1 Open Experimental Decision)

Inference (standard method, Yu & Kobayashi 2003; Yu 2010 survey — cited, not
re-derived): explicit-duration forward algorithm over segment endpoints,
symmetric backward, segmental Viterbi for decoding. Complexity
O(T * M^2 * D_max).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


def _normalize_rows(P: np.ndarray) -> np.ndarray:
    """Map arbitrary rows to a stochastic matrix via softmax (constrained
    parameterization, PRD §15 'Gradient flow' note)."""
    P = np.asarray(P, dtype=np.float64)
    P = P - P.max(axis=-1, keepdims=True)
    e = np.exp(P)
    return e / e.sum(axis=-1, keepdims=True)


@dataclass
class DurationHistogram:
    """Non-parametric duration distribution per state (PRD §10.1 option b).

    pmf over support {1..D_max}; falls back to a discretized geometric tail
    for tau > D_max via `tail_prob` (mass beyond support, spread geometrically
    with the last-bin ratio) — required so the forward algorithm stays
    everywhere-positive for numerical stability.
    """

    pmf: np.ndarray  # [M, D_max]

    def __post_init__(self) -> None:
        if self.pmf.ndim != 2:
            raise ValueError("pmf must be [M, D_max]")
        if (self.pmf < 0).any():
            raise ValueError("pmf must be nonnegative")

    @classmethod
    def from_means(cls, mean_dwell: np.ndarray, d_max: int) -> "DurationHistogram":
        """Discretized geometric-ish prior: pmf(tau) ∝ p^(tau-1) tuned so that
        E[tau] = mean_dwell, truncated at d_max and renormalized."""
        M = len(mean_dwell)
        pmf = np.zeros((M, d_max))
        taus = np.arange(1, d_max + 1, dtype=np.float64)
        for m, mu in enumerate(mean_dwell):
            # geometric with success prob 1/mu (clipped for stability)
            p = float(np.clip(1.0 / max(mu, 1.0), 1e-3, 1.0))
            q = (1 - p) ** (taus - 1) * p
            pmf[m] = q / q.sum()
        return cls(pmf=pmf)

    def normalize(self) -> "DurationHistogram":
        s = self.pmf.sum(axis=1, keepdims=True)
        s = np.where(s <= 0, 1.0, s)
        return DurationHistogram(pmf=self.pmf / s)

    def survival(self) -> np.ndarray:
        """S_m(tau) = P(D >= tau) = sum_{t>=tau} pmf(t | m). [M, D_max+1] with
        S[:, 0] = 1."""
        M, D = self.pmf.shape
        S = np.ones((M, D + 1), dtype=np.float64)
        tail = np.cumsum(self.pmf[:, ::-1], axis=1)[:, ::-1]  # S(tau=1..)
        S[:, 1:] = np.maximum(tail, 0.0)
        return S

    def mean(self) -> np.ndarray:
        D = self.pmf.shape[1]
        taus = np.arange(1, D + 1, dtype=np.float64)
        return (self.pmf * taus).sum(axis=1)

    def mean_residual(self, tau_elapsed: np.ndarray) -> np.ndarray:
        """r_m(tau) = E[D - tau | D >= tau] elementwise; tau clipped to [0, D_max].

        Returns [M, D_max+1]: row m, column tau = residual life given tau
        elapsed steps (tau in 0..D_max). tau=0 -> mean dwell.
        """
        M, D = self.pmf.shape
        S = self.survival()  # [M, D+1], S[:, tau] = P(D >= tau)
        # E[(D - tau) 1{D >= tau}] = sum_{t >= tau} (t - tau) pmf(t)
        taus = np.arange(1, D + 1, dtype=np.float64)
        R = np.zeros((M, D + 1), dtype=np.float64)
        for m in range(M):
            for tau in range(0, D + 1):
                mask = taus >= tau
                num = float(((taus[mask] - tau) * self.pmf[m][mask]).sum())
                R[m, tau] = num / max(S[m, tau], 1e-300)
        return R


class HSMM:
    """Hidden Semi-Markov Model with explicit durations (PRD §10).

    State M-1 (index M-1, 0-based) is the ABSORBING failure state: its row of
    A is e_{M-1} (self-identity), its emission is uniform over events, and
    its duration is degenerate at D_max is NOT assumed — a geometric tail is
    used instead, since after failure the process simply stays put.
    """

    def __init__(
        self,
        n_states: int,
        n_events: int,
        pi: Optional[np.ndarray] = None,
        A: Optional[np.ndarray] = None,
        B: Optional[np.ndarray] = None,
        D: Optional[DurationHistogram] = None,
        d_max: int = 20,
        mean_dwell: Optional[np.ndarray] = None,
        absorbing_state: Optional[int] = None,
        keep_diagonal: bool = False,
        seed: int = 0,
    ):
        if n_states < 2:
            raise ValueError("n_states must be >= 2")
        self.M = int(n_states)
        self.K = int(n_events)
        self.d_max = int(d_max)
        rng = np.random.default_rng(seed)

        # pi: initial distribution, zero mass on absorbing state
        self.absorbing = self.M - 1 if absorbing_state is None else int(absorbing_state)
        if pi is None:
            pi = rng.dirichlet(np.ones(self.M))
        pi = np.asarray(pi, dtype=np.float64)
        pi = pi / pi.sum()
        if not keep_diagonal:  # degenerate A2b models keep absorbing start mass
            pi[self.absorbing] = 0.0
        pi = pi / max(pi.sum(), 1e-300)
        self.pi = pi

        # A: embedded zero-diagonal transition matrix over transient states.
        # The ABSORBING failure state's row is the identity (PRD §29 invariant:
        # "the failure state's row of A is the identity"). Self-transitions on
        # the diagonal are handled by this special case only; transient states
        # keep a zero diagonal (dwell is modeled by D, not by self-loops).
        # EXCEPTION (keep_diagonal=True, d_max=1): ablation A2b encodes a
        # first-order HMM as a duration-degenerate HSMM — its dwell lives ON
        # the diagonal, which must be preserved verbatim.
        if A is None:
            A = rng.dirichlet(np.ones(self.M), size=self.M)
        A = np.asarray(A, dtype=np.float64).copy()
        if keep_diagonal:
            if self.d_max != 1:
                raise ValueError(
                    "keep_diagonal is only valid for d_max=1 (geometric-dwell "
                    "A2b encoding); explicit-duration models must zero it"
                )
            A = A / A.sum(axis=1, keepdims=True).clip(1e-300)
        else:
            np.fill_diagonal(A, 0.0)
            for i in range(self.M):
                if i == self.absorbing:
                    A[i] = 0.0
                    A[i, i] = 1.0  # identity row: absorbing state stays
                    continue
                r = A[i]
                s = r.sum()
                if s > 0:
                    A[i] = r / s
                else:
                    others = [j for j in range(self.M) if j != i]
                    for j in others:
                        A[i, j] = 1.0 / len(others)
        self.A = A

        # B: emission distributions [M, K]
        if B is None:
            B = rng.dirichlet(np.ones(self.K) * 2.0, size=self.M)
        B = np.asarray(B, dtype=np.float64)
        self.B = B / B.sum(axis=1, keepdims=True)

        # D: duration histograms
        if D is None:
            if mean_dwell is None:
                mean_dwell = np.maximum(3.0, rng.uniform(2, 10, size=self.M))
            D = DurationHistogram.from_means(mean_dwell, self.d_max)
        D = D.normalize()
        if D.pmf.shape != (self.M, self.d_max):
            raise ValueError(
                f"duration pmf must be [{self.M}, {self.d_max}]; got {D.pmf.shape}"
            )
        self.D = D

    # ------------------------------------------------------------------
    def validate(self) -> None:
        """Check PRD §29 mathematical invariants (raises on violation).

        For the duration-degenerate A2b encoding (d_max=1, keep_diagonal),
        a non-zero diagonal is the geometric dwell itself and is legal.
        """
        M, K = self.M, self.K
        if self.pi.shape != (M,) or abs(self.pi.sum() - 1) > 1e-9:
            raise ValueError("pi must sum to 1")
        if self.A.shape != (M, M):
            raise ValueError("A must be [M, M]")
        degenerate = self.d_max == 1
        if not degenerate and np.abs(np.diag(self.A)).max() > 1e-12:
            # only the absorbing state may carry a (unit) diagonal entry
            diag = np.diag(self.A).copy()
            diag[self.absorbing] = 0.0
            if np.abs(diag).max() > 1e-12:
                raise ValueError("A diagonal must be zero (embedded chain)")
        if np.abs(self.A.sum(axis=1) - 1).max() > 1e-9:
            raise ValueError("A rows must sum to 1")
        if self.B.shape != (M, K) or np.abs(self.B.sum(axis=1) - 1).max() > 1e-9:
            raise ValueError("B rows must sum to 1 with shape [M, K]")
        if np.abs(self.D.pmf.sum(axis=1) - 1).max() > 1e-9:
            raise ValueError("D rows must sum to 1")

    # ------------------------------------------------------------------
    def event_log_emission(self, v_seq: np.ndarray) -> np.ndarray:
        """log B(v_t | m) for all t, m -> [T, M]."""
        v_seq = np.asarray(v_seq)
        with np.errstate(divide="ignore"):
            logB = np.log(self.B)  # [M, K]
        return logB[:, v_seq].T  # [T, M]

    def sample(self, T: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Sample an event sequence from the generative process (for MC RUL).

        Semantics match the forward algorithm exactly: the initial state is
        drawn from pi and EMITS for its full dwell; each subsequent state is
        drawn from the embedded row A[s] of the state whose segment just
        ended. The absorbing state re-enters itself (identity row) with a
        fresh dwell draw each time.
        """
        rng = rng or np.random.default_rng()
        events: List[int] = []
        s = int(rng.choice(self.M, p=self.pi))
        remaining = 0
        for _ in range(T):
            if remaining == 0:
                # a new segment begins: state s persists (either freshly
                # drawn at t=0 or transitioned at the previous boundary)
                d = int(rng.choice(np.arange(1, self.d_max + 1), p=self.D.pmf[s]))
                remaining = d
            events.append(int(rng.choice(self.K, p=self.B[s])))
            remaining -= 1
            if remaining == 0 and s != self.absorbing:
                # segment ends: transition now; emissions resume next loop
                s = int(rng.choice(self.M, p=self.A[s]))
        return np.array(events, dtype=np.int64)
