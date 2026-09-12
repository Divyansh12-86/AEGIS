"""Ablation A2b: HSMM vs first-order HMM (PRD §22, M10).

Replaces the explicit duration D with the implicit geometric dwell of a
standard HMM: self-transitions on the diagonal. Directly tests whether
explicit duration modeling — the HSMM's most expensive component relative
to a plain HMM — earns its keep (PRD §22, Deliverable F risk).

Implementation reuses the validated HSMM machinery by encoding a
first-order HMM as an HSMM with Dmax = 1 (every segment has length exactly
1, dwell carried entirely by the diagonal self-transition probability). The
likelihood of that constrained HSMM is exactly the standard HMM forward
likelihood:

    alpha_t(j) = B(v_t|j) * [sum_i alpha_{t-1}(i) A(i,j)]   (diagonal allowed)

so both inference paths (through the general HSMM code and through a
dedicated first-order recursion) must agree — the cross-validation test.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.special import logsumexp

from .hsmm import HSMM, DurationHistogram
from .forward_backward import ForwardBackward


class FirstOrderHMM:
    """Standard HMM with diagonal self-transitions (A2b ablation model).

    Built as a duration-degenerate HSMM (Dmax=1, geometric dwell via the
    diagonal), sharing the same data contracts and inference API so the
    Deliverable-E comparison is apples-to-apples.
    """

    def __init__(self, n_states: int, n_events: int, seed: int = 0,
                 pi: Optional[np.ndarray] = None,
                 A: Optional[np.ndarray] = None,
                 B: Optional[np.ndarray] = None):
        if n_states < 2:
            raise ValueError("n_states must be >= 2")
        self.M = int(n_states)
        self.K = int(n_events)
        rng = np.random.default_rng(seed)
        if A is None:
            A = rng.dirichlet(np.ones(self.M) * 2.0, size=self.M)
        A = np.asarray(A, dtype=np.float64).copy()
        A = A / A.sum(axis=1, keepdims=True)
        self.A = A  # diagonal ALLOWED (dwell = geometric via self-loop)
        if pi is None:
            pi = rng.dirichlet(np.ones(self.M))
        pi = np.asarray(pi, dtype=np.float64)
        self.pi = pi / pi.sum()
        if B is None:
            B = rng.dirichlet(np.ones(self.K) * 2.0, size=self.M)
        B = np.asarray(B, dtype=np.float64)
        self.B = B / B.sum(axis=1, keepdims=True)

    # ------------------------------------------------------------------
    def to_hsmm(self) -> HSMM:
        """Express this HMM as a duration-degenerate HSMM.

        Dmax=1: every 'segment' has length exactly 1; the state's dwell
        distribution is the implicit geometric carried by the diagonal,
        which `keep_diagonal=True` preserves verbatim (the A2b encoding).
        """
        pmf = np.ones((self.M, 1))  # P(d=1)=1 for every state
        return HSMM(
            n_states=self.M,
            n_events=self.K,
            pi=self.pi.copy(),
            A=self.A.copy(),  # diagonal kept — to_hsmm bypasses zero-diag ctor
            B=self.B.copy(),
            D=DurationHistogram(pmf=pmf),
            d_max=1,
            keep_diagonal=True,
            seed=0,
        )

    # ------------------------------------------------------------------
    def forward_loglik(self, v_seq: np.ndarray) -> float:
        """Dedicated first-order forward (log-space Rabiner).

        Recurrence: alpha_t(i) = logB(i, v_t) + logsum_j [ logA(j, i) + alpha_{t-1}(j) ]
        (the sum runs over PREDECESSORS j — axis-0 logsumexp of logA).
        """
        v = np.asarray(v_seq, dtype=np.int64)
        T = len(v)
        logB = np.log(np.clip(self.B, 1e-300, None))
        logA = np.log(np.clip(self.A, 1e-300, None))
        logPi = np.log(np.clip(self.pi, 1e-300, None))
        alpha = logPi + logB[:, v[0]]  # [M]
        for t in range(1, T):
            alpha = logB[:, v[t]] + logsumexp(logA + alpha[:, None], axis=0)
        return float(logsumexp(alpha))

    # ------------------------------------------------------------------
    def hsmm_loglik(self, v_seq: np.ndarray) -> float:
        """Same likelihood through the general HSMM forward machinery."""
        hsmm = self.to_hsmm()
        # to_hsmm's A bypasses the constructor's zero-diagonal/renormalization
        # only when given explicitly; enforce invariants manually for the
        # degenerate case: rows must still sum to 1 (they do — copied), and
        # Dmax=1 durations are proper.
        fb = ForwardBackward(hsmm)
        return fb.forward(np.asarray(v_seq, dtype=np.int64))[1]

    # ------------------------------------------------------------------
    def posterior(self, v_seq: np.ndarray) -> np.ndarray:
        """Filtered state posterior p_t(m) = P(s_t | v_{1:t}) (causal)."""
        v = np.asarray(v_seq, dtype=np.int64)
        T = len(v)
        logB = np.log(np.clip(self.B, 1e-300, None))
        logA = np.log(np.clip(self.A, 1e-300, None))
        logPi = np.log(np.clip(self.pi, 1e-300, None))
        alpha = logPi + logB[:, v[0]]
        out = np.zeros((T, self.M))
        out[0] = np.exp(alpha - logsumexp(alpha))
        for t in range(1, T):
            alpha = logB[:, v[t]] + logsumexp(logA + alpha[:, None], axis=0)
            out[t] = np.exp(alpha - logsumexp(alpha))
        return out

    # ------------------------------------------------------------------
    def sample(self, T: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        s = rng.choice(self.M, p=self.pi)
        events = []
        for _ in range(T):
            events.append(int(rng.choice(self.K, p=self.B[s])))
            s = int(rng.choice(self.M, p=self.A[s]))
        return np.array(events, dtype=np.int64)


@dataclass
class HMMFitResult:
    log_likelihoods: list
    n_iterations: int


class FirstOrderHMMFitter:
    """Standard Baum-Welch for the first-order HMM (A2b training arm).

    E-step: alpha/beta recursions with full diagonal; M-step: closed-form
    pi/A/B updates. Monotone in likelihood (EM guarantee), same early-stop
    conventions as the HSMM fitter.
    """

    def __init__(self, max_iter: int = 50, tol: float = 1e-6, seed: int = 0):
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed

    def fit(self, seqs, n_states: int, n_events: int) -> tuple:
        if len(seqs) == 0:
            raise ValueError("no sequences")
        hmm = FirstOrderHMM(n_states, n_events, seed=self.seed)
        trace: list = []
        for it in range(self.max_iter):
            # ----- E-step over all sequences -----
            counts_pi = np.zeros(n_states)
            counts_A = np.zeros((n_states, n_states))
            counts_B = np.zeros((n_states, n_events))
            total_ll = 0.0
            for v in seqs:
                v = np.asarray(v, dtype=np.int64)
                T = len(v)
                logB = np.log(np.clip(hmm.B, 1e-300, None))
                logA = np.log(np.clip(hmm.A, 1e-300, None))
                logPi = np.log(np.clip(hmm.pi, 1e-300, None))
                # forward
                af = np.zeros((T, n_states))
                af[0] = logPi + logB[:, v[0]]
                for t in range(1, T):
                    af[t] = logB[:, v[t]] + logsumexp(
                        logA + af[t - 1][:, None], axis=0
                    )
                # backward
                ab = np.zeros((T, n_states))
                ab[T - 1] = 0.0
                for t in range(T - 2, -1, -1):
                    ab[t] = logsumexp(
                        logA + (logB[:, v[t + 1]] + ab[t + 1])[None, :], axis=1
                    )
                ll = logsumexp(af[T - 1])
                total_ll += ll
                # gamma_t(i) = af*ab/ll
                with np.errstate(divide="ignore"):
                    lg = af + ab - ll
                gamma = np.exp(lg)
                # xi_t(i,j) ∝ gamma_t(i) * A(i,j) * B(v_{t+1}|j) * beta_{t+1}(j) / beta_t(i)
                counts_pi += gamma[0]
                for t in range(T):
                    counts_B[:, v[t]] += gamma[t]
                for t in range(T - 1):
                    # log xi = lg[t, i] + logA(i,j) + logB(j, v_{t+1}) + ab[t+1, j] - ab[t, i]
                    log_xi = (
                        lg[t][:, None]
                        + logA
                        + (logB[:, v[t + 1]] + ab[t + 1])[None, :]
                        - ab[t][:, None]
                    )
                    counts_A += np.exp(log_xi - logsumexp(log_xi))
            trace.append(total_ll)
            # ----- M-step -----
            pi = counts_pi / counts_pi.sum()
            A = counts_A / counts_A.sum(axis=1, keepdims=True).clip(1e-300)
            B = counts_B + 1e-3  # Laplace floor
            B = B / B.sum(axis=1, keepdims=True)
            hmm = FirstOrderHMM(n_states, n_events, seed=self.seed,
                                pi=pi, A=A, B=B)
            if it >= 2 and abs(trace[-1] - trace[-2]) < self.tol:
                break
        return hmm, HMMFitResult(trace, len(trace))
