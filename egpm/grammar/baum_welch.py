"""Baum-Welch (EM) fitting for the explicit-duration HSMM (PRD §16 Stage 2, M4).

E-step: occupancy forward-backward gives
  * state/elapsed posterior xi_t(i, e) = P(v, s_t=i, elapsed e);
  * segment-boundary statistics: eta_t(i, j) = P(segment in i ends at t,
    next segment starts at t+1 in j);
  * per-state duration counts aggregated over segments.

M-step (constrained re-estimation):
  pi_i      = expected # segments starting at position 0 in state i
              (normalized over non-absorbing states);
  A(i, j)   = eta(i, j) / sum_k eta(i, k)   (zero diagonal preserved;
              absorbing row kept identity);
  B(i, v)   = sum of xi_t(i, e) over t with v_t = v, all e (Laplace floor);
  D(i, d)   = expected # segments of state i with length d (histogram MLE
              with smoothing; PRD §10.1 non-parametric option).

Convergence: monitored on per-iteration total log-likelihood (EM is
guaranteed monotone for exact E/M steps). Restarts with different inits are
supported per PRD §25 (unstable-EM mitigation).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from .hsmm import HSMM, DurationHistogram
from .forward_backward import ForwardBackward, ForwardBackwardResult

_EPS = 1e-300


@dataclass
class EMTrace:
    log_likelihoods: List[float]

    @property
    def converged(self) -> bool:
        return (
            len(self.log_likelihoods) >= 2
            and self.log_likelihoods[-1] >= self.log_likelihoods[-2]
            and (self.log_likelihoods[-1] - self.log_likelihoods[-2]) < 1e-6
        )


class BaumWelch:
    """EM fitting for (pi, A, B, D) with explicit durations."""

    def __init__(
        self,
        n_states: int,
        n_events: int,
        d_max: int = 20,
        tol: float = 1e-6,
        max_iter: int = 50,
        min_iter: int = 3,
        seed: int = 0,
        restarts: int = 0,
    ):
        self.n_states = n_states
        self.n_events = n_events
        self.d_max = d_max
        self.tol = tol
        self.max_iter = max_iter
        self.min_iter = min_iter
        self.seed = seed
        self.restarts = restarts

    # ------------------------------------------------------------------
    def _e_step(self, hsmm: HSMM, seqs: Sequence[np.ndarray]):
        """Collect expected counts from one forward-backward per sequence."""
        fb = ForwardBackward(hsmm)
        M, K, Dmax = hsmm.M, hsmm.K, hsmm.d_max
        absb = hsmm.absorbing
        results = []
        total_ll = 0.0
        # accumulators
        pi_counts = np.zeros(M)
        eta = np.zeros((M, M))  # segment boundary state-pair counts
        B_counts = np.zeros((M, K))
        D_counts = np.zeros((M, Dmax))
        ll_prev: Optional[np.ndarray] = None
        for v in seqs:
            v = np.asarray(v, dtype=np.int64)
            res = fb.run(v)
            total_ll += res.log_likelihood
            results.append(res)
            # xi: posterior over (state, elapsed) [T, M, Dmax]
            with np.errstate(divide="ignore"):
                joint = res.alpha + res.beta
            xi = np.exp(joint - np.log(np.exp(res.log_likelihood)))
            xi = np.nan_to_num(xi, nan=0.0, posinf=0.0, neginf=0.0)
            T = len(v)
            # initial-state counts: posterior of s_0 (any elapsed; fresh seg)
            pi_counts += xi[0].sum(axis=1)
            # emission counts
            for t in range(T):
                B_counts[:, v[t]] += xi[t].sum(axis=1)
            # duration + transition counts need per-segment statistics:
            # P(segment (i, d) ends at t) = xi-style mass via alpha/beta:
            # ends(i, d, t) = [alpha_t(i, d) * h_i(d) * M_i(t+1)] / P(v)
            #               + [alpha_t(i,d)*h_i(d)*beta-exit...] — simplest:
            # use the same decomposition as beta: end mass at t in state i
            # with elapsed d = alpha_t(i,d)*h_i(d)*M_i(t+1) / P(v) where
            # M is the after-end value; and the last-segment end at T-1:
            # alpha_{T-1}(i,d)*h_i(d) / P(v).
            log_hazard = np.log(np.clip(hsmm.D.pmf, _EPS, None)) - np.log(
                np.clip(hsmm.D.survival()[:, 1:], _EPS, None)
            )  # [M, Dmax]
            logP = res.log_likelihood
            # ends at T-1
            end_last = np.exp(res.alpha[T - 1] + log_hazard - logP)
            D_counts += end_last  # [M, Dmax]
            # ends at t < T-1: need M_i(t+1) — recompute via beta structure:
            # reuse beta: beta_t(i,d) = c*B*beta_{t+1} + h_i(d)*M_i(t+1)
            # => h_i(d)*M_i(t+1) = beta_t(i,d) - c*B*beta_{t+1}(i,d+1).
            # Simpler and exact: segment end mass at t (state i, elapsed d):
            #   alpha_t(i,d) * [beta_t(i,d) - cont_part] / P(v)
            # cont_part = c_i(d) * B(v_{t+1}|i) * beta_{t+1}(i, d+1)
            # but beta includes BOTH end and cont; end mass needs M only:
            # end_mass_t(i,d) = alpha_t(i,d) * h_i(d) * M_i(t+1) / P(v)
            # We recompute M_i(t+1) from stored pieces of the FB pass —
            # cleanest: recompute N/M tables as in backward(). To avoid
            # duplicating logic, approximate is NOT acceptable (PRD §16) —
            # we recompute here exactly:
            M_after, N_vals = self._after_end_values(hsmm, v)  # log [M, T+1] each
            for t in range(T - 1):
                log_end = (
                    res.alpha[t] + log_hazard + M_after[:, t + 1][:, None] - logP
                )  # [M, Dmax]
                end_mass = np.exp(log_end)
                D_counts += end_mass
                # eta(i, j): expected count of boundary (seg i ends at t,
                # seg j starts at t+1). The end mass for state i is summed
                # over d; the next-segment state distribution given the end
                # is A(i,j) * N_j(t+1) / M_i(t+1) (Bayes over the boundary).
                log_N = N_vals  # [M, T+1]
                denom = np.exp(M_after[:, t + 1])[:, None]  # [M, 1]
                nj = np.exp(log_N)  # [M, T+1]
                end_i = end_mass.sum(axis=1)  # [M]
                frac = np.where(
                    denom > 0, nj[:, t + 1][None, :] / np.maximum(denom, _EPS), 0.0
                )  # [i, j] = N_j(t+1)/M_i(t+1)
                eta += end_i[:, None] * hsmm.A * frac
        return {
            "ll": total_ll,
            "pi": pi_counts,
            "eta": eta,
            "B": B_counts,
            "D": D_counts,
        }

    # ------------------------------------------------------------------
    def _after_end_values(self, hsmm: HSMM, v: np.ndarray) -> np.ndarray:
        """log M_j(s): value after a segment of state j ended at s-1 (s in 0..T)."""
        from scipy.special import logsumexp
        T = len(v)
        M, Dmax = hsmm.M, hsmm.d_max
        logA = np.log(np.clip(hsmm.A, _EPS, None))
        logD = np.log(np.clip(hsmm.D.pmf, _EPS, None))
        logB = np.log(np.clip(hsmm.B, _EPS, None))
        c = np.zeros((M, T + 1))
        c[:, 1:] = np.cumsum(logB[:, v], axis=1)
        log_M = np.full((M, T + 1), -np.inf)
        log_M[:, T] = 0.0
        log_N = np.full((M, T + 1), -np.inf)
        for s in range(T, 0, -1):
            for j in range(M):
                d_hi = min(Dmax, T - s)
                if d_hi < 1:
                    log_N[j, s] = -np.inf
                    continue
                lens = np.arange(1, d_hi + 1)
                emis = c[j, s + lens] - c[j, s]
                ends_at_T = (s + lens - 1) == (T - 1)
                after = np.where(ends_at_T, 0.0, log_M[j, s + lens])
                log_N[j, s] = logsumexp(logD[j, lens - 1] + emis + after)
            for j in range(M):
                log_M[j, s] = logsumexp(logA[j] + log_N[:, s])
        return log_M, log_N

    # ------------------------------------------------------------------
    def _m_step(self, hsmm: HSMM, counts: dict) -> HSMM:
        M, K = hsmm.M, hsmm.K
        absb = hsmm.absorbing
        # pi: no mass on absorbing
        pi = counts["pi"].copy()
        pi[absb] = 0.0
        if pi.sum() <= 0:
            pi = np.ones(M)
            pi[absb] = 0.0
        pi = pi / pi.sum()
        # A from eta; keep absorbing row = identity
        eta = counts["eta"]
        A = eta.copy()
        np.fill_diagonal(A, 0.0)
        A[absb] = 0.0
        A[absb, absb] = 1.0
        for i in range(M):
            if i == absb:
                continue
            s = A[i].sum()
            if s > 0:
                A[i] = A[i] / s
            else:
                others = [j for j in range(M) if j != i]
                for j in others:
                    A[i, j] = 1.0 / len(others)
        # B with Laplace floor
        B = counts["B"] + 1e-3
        B = B / B.sum(axis=1, keepdims=True)
        # D histogram with smoothing
        Dc = counts["D"] + 1e-6
        D = DurationHistogram(pmf=Dc / Dc.sum(axis=1, keepdims=True))
        return HSMM(
            n_states=M,
            n_events=K,
            pi=pi,
            A=A,
            B=B,
            D=D,
            d_max=hsmm.d_max,
            absorbing_state=absb,
        )

    # ------------------------------------------------------------------
    def fit(
        self, seqs: Sequence[np.ndarray], init: Optional[HSMM] = None, verbose: bool = False
    ) -> tuple:
        """Fit parameters; returns (hsmm, trace). Restart-aware."""
        if len(seqs) == 0:
            raise ValueError("no sequences to fit")
        best = None
        for r in range(self.restarts + 1):
            hsmm = init if (init is not None and r == 0) else HSMM(
                n_states=self.n_states,
                n_events=self.n_events,
                d_max=self.d_max,
                seed=self.seed + 1000 * r,
            )
            trace = EMTrace(log_likelihoods=[])
            for it in range(self.max_iter):
                counts = self._e_step(hsmm, seqs)
                trace.log_likelihoods.append(counts["ll"])
                if it >= self.min_iter:
                    if (
                        len(trace.log_likelihoods) >= 2
                        and abs(trace.log_likelihoods[-1] - trace.log_likelihoods[-2])
                        < self.tol
                    ):
                        break
                hsmm = self._m_step(hsmm, counts)
            if verbose:
                print(f"restart {r}: ll {trace.log_likelihoods[0]:.3f} -> {trace.log_likelihoods[-1]:.3f}")
            if best is None or trace.log_likelihoods[-1] > best[1].log_likelihoods[-1]:
                best = (hsmm, trace)
        return best

    # ------------------------------------------------------------------
    def fit_holdout(
        self,
        train_seqs: Sequence[np.ndarray],
        val_seqs: Sequence[np.ndarray],
        max_iter: int = 100,
        patience: int = 5,
        verbose: bool = False,
    ) -> tuple:
        """Fit with validation-based early stopping (PRD §25 overfitting row)."""
        hsmm = HSMM(
            n_states=self.n_states,
            n_events=self.n_events,
            d_max=self.d_max,
            seed=self.seed,
        )
        trace = EMTrace(log_likelihoods=[])
        val_trace: List[float] = []
        best_val = -np.inf
        best_hsmm = None
        since_best = 0
        for it in range(max_iter):
            counts = self._e_step(hsmm, train_seqs)
            trace.log_likelihoods.append(counts["ll"])
            hsmm = self._m_step(hsmm, counts)
            val_ll = sum(ForwardBackward(hsmm).forward(np.asarray(v))[1] for v in val_seqs)
            val_trace.append(val_ll)
            if verbose:
                print(f"iter {it}: train {counts['ll']:.3f} val {val_ll:.3f}")
            if val_ll > best_val + 1e-8:
                best_val = val_ll
                best_hsmm = HSMM(
                    n_states=hsmm.M, n_events=hsmm.K, pi=hsmm.pi.copy(),
                    A=hsmm.A.copy(), B=hsmm.B.copy(),
                    D=DurationHistogram(pmf=hsmm.D.pmf.copy()),
                    d_max=hsmm.d_max, absorbing_state=hsmm.absorbing,
                )
                since_best = 0
            else:
                since_best += 1
                if since_best >= patience:
                    break
        return best_hsmm, (trace, val_trace)
