"""Duration-corrected phase-type RUL estimation (PRD §12, M5).

The naive (I-Q)^-1 1 formula counts expected TRANSITIONS, not expected TIME,
and ignores that inference happens MID-DWELL. The PRD's corrected formula:

    E[RUL | s_i, tau] = [ r_i(tau) - mu_i ] + (Phi mu)_i

where:
  * Q       — sub-stochastic embedded (zero-diagonal) transition matrix over
              the M-1 non-absorbing states;
  * Phi     — (I - Q)^-1 fundamental matrix: Phi_ij = expected # visits to j
              starting from a FRESH ENTRY into i, before absorption;
  * mu      — vector of mean state durations over the non-absorbing states;
  * (Phi mu)_i — fresh-entry expected total remaining time from state i
              (includes one full mu_i for the current visit);
  * r_i(tau) — mean residual dwell: E[D_i - tau | D_i >= tau], replacing the
              already-counted full mu_i with what's actually left.

Median / prediction intervals: Monte Carlo simulation of the generative
process from the belief p_i to absorption (PRD §12; Phase-2 intervals are
implemented here but the MVP reports the expected value).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..grammar.hsmm import HSMM


@dataclass
class RULResult:
    """PRD §28 prediction output contract for the RUL head."""

    expected: float
    median: Optional[float] = None
    interval: Optional[tuple] = None
    per_state_expected: Optional[np.ndarray] = None  # [M] incl. absorbing=0


class PhaseTypeRUL:
    """RUL from HSMM parameters via the corrected phase-type formula."""

    def __init__(self, hsmm: HSMM):
        self.hsmm = hsmm
        hsmm.validate()
        self._precompute()

    # ------------------------------------------------------------------
    def _precompute(self) -> None:
        M = self.hsmm.M
        absb = self.hsmm.absorbing
        self.transient = [i for i in range(M) if i != absb]
        n = len(self.transient)
        # Q: sub-stochastic block over transient states
        Q = self.hsmm.A[np.ix_(self.transient, self.transient)].copy()
        # absorbing probability from each transient state
        self.absorb_p = self.hsmm.A[np.ix_(self.transient, [absb])].ravel()
        # fundamental matrix Phi = (I - Q)^-1
        # A transient class that can never reach the absorbing state gives
        # a singular (I - Q): RUL is genuinely infinite there. pinv would
        # silently return ~0 — raise instead (degenerate fit, PRD §12).
        absorb_p = self.hsmm.A[np.ix_(self.transient, [absb])].ravel()
        # multi-step reachability: absorb probability through any chain
        P = self.hsmm.A.copy()
        reach = absorb_p.copy()
        for _ in range(len(self.transient)):
            reach = np.maximum(reach, (P @ self.hsmm.A)[
                np.ix_(self.transient, [absb])
            ].ravel())
            P = P @ self.hsmm.A
        if (reach <= 1e-12).any():
            raise ValueError(
                "degenerate HSMM: some transient state cannot reach the "
                "absorbing state; expected RUL is infinite (PRD §12)"
            )
        self.Phi = np.linalg.inv(np.eye(n) - Q)
        # mean dwell mu over transient states
        self.mu = self.hsmm.D.mean()[self.transient]
        # fresh-entry expected total time: (Phi mu)
        self.fresh_total = self.Phi @ self.mu
        # residual-life table r_i(tau) for tau = 0..D_max
        self.R = self.hsmm.D.mean_residual(np.zeros(len(self.transient), dtype=int))

    # ------------------------------------------------------------------
    def expected_rul(
        self, state: int, tau_elapsed: int = 0, belief: Optional[np.ndarray] = None
    ) -> float:
        """E[RUL | s_i = state, tau] (PRD §12 boxed formula).

        With a belief vector p over transient states (posterior), computes
        the belief-weighted expectation over MAP state or full mixture.
        """
        M = self.hsmm.M
        absb = self.hsmm.absorbing
        if state == absb:
            return 0.0
        if state not in self.transient:
            raise ValueError(f"state {state} not a valid non-absorbing state")
        idx = self.transient.index(state)
        tau = int(np.clip(tau_elapsed, 0, self.hsmm.d_max))
        # r_i(tau): residual life of the CURRENT visit given tau elapsed
        r_tau = float(self.R[idx, tau])
        # (Phi mu)_i: fresh-entry total remaining time (includes full mu_i)
        fresh = float(self.fresh_total[idx])
        return (r_tau - self.mu[idx]) + fresh

    # ------------------------------------------------------------------
    def expected_rul_from_belief(
        self, belief: np.ndarray, tau_elapsed: Optional[np.ndarray] = None
    ) -> float:
        """Belief-weighted RUL: sum_i p_i * E[RUL | i, tau_i].

        tau_elapsed: optional [M] per-state elapsed dwell estimates (e.g.
        from FB expected dwell); when None, tau = 0 per state (fresh entry).
        Absorbing state contributes 0.
        """
        belief = np.asarray(belief, dtype=np.float64)
        M = self.hsmm.M
        if belief.shape != (M,):
            raise ValueError(f"belief must be [M]={M}; got {belief.shape}")
        belief = belief / max(belief.sum(), 1e-300)
        if tau_elapsed is None:
            tau_elapsed = np.zeros(M, dtype=int)
        tau_elapsed = np.asarray(tau_elapsed, dtype=int)
        total = 0.0
        for i in self.transient:
            total += belief[i] * self.expected_rul(i, int(tau_elapsed[i]))
        return float(total)

    # ------------------------------------------------------------------
    def simulate_to_absorption(
        self,
        belief: np.ndarray,
        n_trajectories: int = 200,
        max_steps: int = 10_000,
        rng: Optional[np.random.Generator] = None,
        start_elapsed: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Monte Carlo RUL samples from belief p_i to absorption (PRD §12).

        Used for median / prediction intervals (Phase 2 fields). Each
        trajectory: sample initial state ~ belief (restricted to transient
        unless belief puts mass on absorbing, which ends immediately),
        sample remaining dwell given elapsed (conditional), then follow
        embedded transitions until absorption.
        """
        rng = rng or np.random.default_rng()
        belief = np.asarray(belief, dtype=np.float64)
        M = self.hsmm.M
        absb = self.hsmm.absorbing
        belief = belief / max(belief.sum(), 1e-300)
        if start_elapsed is None:
            start_elapsed = np.zeros(M, dtype=int)
        # conditional dwell sampler: given elapsed tau in state i, sample the
        # remaining dwell from the conditional distribution
        pmf = self.hsmm.D.pmf  # [M, Dmax]
        surv = self.hsmm.D.survival()  # [M, Dmax+1]

        samples = np.empty(n_trajectories, dtype=np.float64)
        for n in range(n_trajectories):
            s = int(rng.choice(M, p=belief))
            if s == absb:
                samples[n] = 0.0
                continue
            steps = 0.0
            while s != absb and steps < max_steps:
                tau0 = int(np.clip(start_elapsed[s], 0, self.hsmm.d_max))
                # conditional remaining dwell P(rem = m | D >= tau0):
                # D = tau0 + m total; P(D = t | D >= tau0) = pmf(t)/S(tau0)
                taus = np.arange(tau0 + 1, self.hsmm.d_max + 1)
                w = pmf[s, taus - 1] / max(surv[s, tau0], 1e-300)
                if w.sum() <= 0:
                    dwell_total = max(self.hsmm.d_max, 1)
                else:
                    dwell_total = int(rng.choice(taus, p=w / w.sum()))
                rem = dwell_total - tau0
                steps += rem
                # next state from embedded row (absorbing possible)
                s = int(rng.choice(M, p=self.hsmm.A[s]))
            samples[n] = steps
        return samples

    # ------------------------------------------------------------------
    def rul_from_belief(
        self,
        belief: np.ndarray,
        tau_elapsed: Optional[np.ndarray] = None,
        n_mc: int = 0,
        interval_q: float = 0.9,
        rng: Optional[np.random.Generator] = None,
    ) -> RULResult:
        """Full RULResult: expected always; median/interval if n_mc > 0."""
        expected = self.expected_rul_from_belief(belief, tau_elapsed)
        per_state = np.zeros(self.hsmm.M)
        for i in self.transient:
            per_state[i] = self.expected_rul(i, 0)
        median = None
        interval = None
        if n_mc > 0:
            sims = self.simulate_to_absorption(
                belief, n_trajectories=n_mc, rng=rng,
                start_elapsed=tau_elapsed,
            )
            median = float(np.median(sims))
            lo = (1 - interval_q) / 2
            interval = (
                float(np.quantile(sims, lo)),
                float(np.quantile(sims, 1 - lo)),
            )
        return RULResult(
            expected=expected,
            median=median,
            interval=interval,
            per_state_expected=per_state,
        )
