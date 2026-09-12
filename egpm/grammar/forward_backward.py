"""Explicit-duration forward-backward for the HSMM (PRD §10.2, M4).

Implementation uses the standard occupancy (elapsed-dwell) parametrization of
the explicit-duration forward-backward (Yu & Kobayashi 2003; Yu 2010 survey —
cited per PRD §10.2, not re-derived):

    alpha_t(i, e) = P(v_0..v_t, s_t = i, current segment has elapsed dwell e)

    alpha_t(i, 1)      = B(v_t|i) * sum_j A(j, i) * [sum_e' alpha_{t-1}(j, e') * h_j(e')]
    alpha_t(i, e>1)    = alpha_{t-1}(i, e-1) * B(v_t|i) * S_i(e)/S_i(e-1)

where h_j(e) = pmf(e)/S(e) is the duration HAZARD (segment ends exactly at
elapsed e) and S_i(e)/S_i(e-1) the conditional continuation probability.
Zero-diagonal A forbids consecutive same-state segments for transient
states; the absorbing state's identity row (PRD §29) permits failure
segments of total dwell > Dmax to chain. Sequence likelihood uses the
PRD §10.2 convention — the final segment ends at the last observation:

    P(v) = sum_i sum_e alpha_{T-1}(i, e) * h_i(e)

Complexity O(T * M * D_max) with the hazard regrouping (equivalent to the
boundary formulation in the PRD, which is O(T * M^2 * D_max); both match
brute-force enumeration — see tests/test_hsmm_likelihood.py).

The state posterior is the true smoothing posterior:
    p_t(i)  ∝ sum_e alpha_t(i, e) * beta_t(i, e)
with the exact identity sum_{i,e} alpha_t(i,e) beta_t(i,e) = P(v) for every t
(checked in CI per PRD §29).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp  # type: ignore

from .hsmm import HSMM

_NEG_INF = -np.inf


@dataclass
class ForwardBackwardResult:
    log_likelihood: float
    state_posterior: np.ndarray  # [T, M], rows sum to 1
    expected_dwell: np.ndarray  # [T, M] E[elapsed dwell | s_t = i]
    alpha: np.ndarray  # [T, M, Dmax] log alpha_t(i, e)
    beta: np.ndarray  # [T, M, Dmax] log beta_t(i, e)
    alpha_state_marginal: np.ndarray  # [T, M] logsumexp_e alpha
    beta_state_marginal: np.ndarray  # [T, M] logsumexp_e beta


class ForwardBackward:
    """Explicit-duration forward-backward (PRD §10.2, §29)."""

    def __init__(self, hsmm: HSMM):
        self.hsmm = hsmm
        self._validate_model()

    def _validate_model(self) -> None:
        self.hsmm.validate()

    # ------------------------------------------------------------------
    @staticmethod
    def _obs_to_logE(hsmm: HSMM, obs: np.ndarray) -> np.ndarray:
        """Observation -> per-window log-emission matrix, orientation [M, T].

        ``obs`` may be:
          * 1-D int array [T] — event-token sequence; emissions from the
            HSMM's categorical B (discrete path, PRD §10);
          * 2-D float array [T, M] — external log-emissions (one row per
            window), the injection seam used by ablation A1's continuous
            Gaussian emissions (PRD §22) and any emission model that can
            produce per-(window, state) log-densities.
        """
        obs = np.asarray(obs)
        if obs.ndim == 1:
            v = obs.astype(np.int64)
            logB = np.log(np.clip(hsmm.B, 1e-300, None))
            return logB[:, v]  # [M, T]
        if obs.ndim == 2 and obs.shape[1] == hsmm.M:
            return np.asarray(obs, dtype=np.float64).T  # [M, T]
        raise ValueError(
            f"observations must be tokens [T] or log-emissions [T, {hsmm.M}]; "
            f"got shape {obs.shape}"
        )

    # ------------------------------------------------------------------
    def _precompute(self, obs: np.ndarray):
        """Shared log-space tables: emissions, hazards, continuation probs."""
        M, K = self.hsmm.M, self.hsmm.K
        Dmax = self.hsmm.d_max
        logA = np.log(np.clip(self.hsmm.A, 1e-300, None))  # [M, M]
        logPi = np.log(np.clip(self.hsmm.pi, 1e-300, None))  # [M]
        logD = np.log(np.clip(self.hsmm.D.pmf, 1e-300, None))  # [M, Dmax]

        # survival S(e) = P(D >= e), e in 0..Dmax+1;  index e
        S = np.zeros((M, Dmax + 2), dtype=np.float64)
        S[:, 0] = 1.0
        pmf = self.hsmm.D.pmf
        S[:, 1 : Dmax + 1] = np.maximum(
            np.cumsum(pmf[:, ::-1], axis=1)[:, ::-1], 0.0
        )
        S[:, Dmax + 1] = 0.0
        logS = np.log(np.clip(S, 1e-300, None))

        # hazard h_i(e) = pmf(e)/S(e), e in 1..Dmax -> index e-1
        log_hazard = logD - logS[:, 1 : Dmax + 1]  # [M, Dmax]
        # continuation c_i(e) = S(e+1)/S(e), from elapsed e to e+1
        # index e-1 (elapsed e); e = Dmax gives -inf (support exhausted)
        log_cont = logS[:, 2:] - logS[:, 1 : Dmax + 1]  # [M, Dmax]
        # clip: S(Dmax+1)=0 -> log -inf is correct; guard nan from 0/0
        log_cont = np.where(np.isfinite(log_cont), log_cont, _NEG_INF)

        # per-window log emissions, [M, T] (tokens OR injected logE)
        logE_T = self._obs_to_logE(self.hsmm, obs)
        T = logE_T.shape[1]
        # prefix sums c[j, s] = sum_{tau < s} logE(j, tau)
        c = np.zeros((M, T + 1), dtype=np.float64)
        c[:, 1:] = np.cumsum(logE_T, axis=1)

        return (logA, logPi, logD, logS, log_hazard, log_cont,
                c, logE_T)

    # ------------------------------------------------------------------
    def forward(self, obs: np.ndarray):
        """Occupancy forward pass.

        Returns (alpha [T, M, Dmax] log, log_likelihood). ``obs``: token
        sequence [T] or external log-emissions [T, M] (ablation A1 seam).
        """
        T = self._obs_len(obs)
        M = self.hsmm.M
        Dmax = self.hsmm.d_max
        logA, logPi, _, _, log_hazard, log_cont, _, logE_T = self._precompute(obs)

        alpha = np.full((T, M, Dmax), _NEG_INF, dtype=np.float64)
        # t = 0: fresh segment, elapsed 1
        alpha[0, :, 0] = logPi + logE_T[:, 0]
        # zero-diagonal masking (PRD §10.1): same-state segments are illegal
        # for transient states; the absorbing identity row is legal. For
        # DEGENERATE d_max=1 models (A2b geometric-dwell encoding) the
        # diagonal is the dwell and must be kept for all states.
        degenerate = Dmax == 1
        if degenerate:
            incoming_mask = np.zeros((M, M), dtype=bool)
        else:
            incoming_mask = np.eye(M, dtype=bool)
            incoming_mask[self.hsmm.absorbing, self.hsmm.absorbing] = False
        for t in range(1, T):
            # termination mass per previous state j: sum_e' alpha*hazard
            term = logsumexp(alpha[t - 1] + log_hazard, axis=1)  # [M]
            # new segment starts at t: state i receives
            #   log sum_j A(j, i) exp(term_j)  -- logsumexp, NOT matrix @
            incoming = np.where(
                incoming_mask, _NEG_INF, logA + term[:, None]
            )  # [j, i]
            incoming = logsumexp(incoming, axis=0)  # [M] over j
            alpha[t, :, 0] = logE_T[:, t] + incoming
            # continuation: elapsed e-1 -> e
            alpha[t, :, 1:] = (
                alpha[t - 1, :, : Dmax - 1]
                + logE_T[:, t, None]
                + log_cont[:, : Dmax - 1]
            )
        ll = float(logsumexp(alpha[T - 1] + log_hazard))
        return alpha, ll

    # ------------------------------------------------------------------
    @staticmethod
    def _obs_len(obs: np.ndarray) -> int:
        obs = np.asarray(obs)
        return obs.shape[0] if obs.ndim == 1 else obs.shape[0]

    # ------------------------------------------------------------------
    def backward(self, obs: np.ndarray) -> np.ndarray:
        """Occupancy backward pass (corrected segment-accounting form).

        beta[t, i, e-1] = log P(v_{t+1..T-1} observed, parse completes |
        s_t = i, current segment elapsed e and still running at t).

        Two intermediates per start position s (s in 1..T):
          N_j(s) — a segment of state j STARTING at s: sum over lengths d of
                   D(d|j) * emissions * [factor 1 if it ends at T-1 (its
                   length probability is already D(d); the parse ends there),
                   else M_j(s+d)]
          M_j(s) — value AFTER a segment of state j just ended at s-1:
                   sum_k A(j, k) * N_k(s)
        Base: M_j(T) = 1 (nothing left). beta base at T-1 applies the
        CURRENT segment's hazard (its end is the parse's last event).

        Recurrence (validated against exhaustive enumeration in tests):
          beta_t(i,e) = c_i(e) * B(v_{t+1}|i) * beta_{t+1}(i, e+1)   [continue]
                      + h_i(e) * M_i(t+1)                            [end at t]
        """
        T = self._obs_len(obs)
        M = self.hsmm.M
        Dmax = self.hsmm.d_max
        logA, _, logD, _, log_hazard, log_cont, c, logE_T = self._precompute(obs)

        # --- N_j(s) for s = T-1 .. 1 (segment starts at s, in log space) ---
        # log_N[j, s]
        log_N = np.full((M, T + 1), _NEG_INF, dtype=np.float64)
        # M_j(s) for s = 1..T (s = position AFTER previous segment's end)
        log_M = np.full((M, T + 1), _NEG_INF, dtype=np.float64)
        # base M_j(T) = 1
        log_M[:, T] = 0.0
        for s in range(T, 0, -1):  # s = start position of next segment
            for j in range(M):
                d_hi = min(Dmax, T - s)  # segment must fit in [s, T-1]
                if d_hi < 1:
                    log_N[j, s] = _NEG_INF
                    continue
                lens = np.arange(1, d_hi + 1)
                emis = c[j, s + lens] - c[j, s]  # [d_hi]
                # end factor: 1 if s + d - 1 == T - 1 else M_j(s + d)
                ends_at_T = (s + lens - 1) == (T - 1)  # [d_hi]
                after = np.where(ends_at_T, 0.0, log_M[j, s + lens])  # [d_hi]
                log_N[j, s] = logsumexp(logD[j, lens - 1] + emis + after)
            # M_j(s) = log sum_k A(j, k) * N_k(s)
            for j in range(M):
                log_M[j, s] = logsumexp(logA[j] + log_N[:, s])
        # special: M at s=0 unused (no segment starts at 0 in backward)

        beta = np.full((T, M, Dmax), _NEG_INF, dtype=np.float64)
        # base: current segment ends AT T-1
        beta[T - 1] = log_hazard
        for t in range(T - 2, -1, -1):
            # end-at-t term: h_i(e) * M_i(t+1), broadcast over elapsed axis
            end_term = log_hazard + log_M[:, t + 1][:, None]  # [M, Dmax]
            # cont term: c_i(e) * B(v_{t+1}|i) * beta_{t+1}(i, e+1)
            cont = np.full((M, Dmax), _NEG_INF, dtype=np.float64)
            cont[:, : Dmax - 1] = (
                beta[t + 1, :, 1:]
                + logE_T[:, t + 1, None]
                + log_cont[:, : Dmax - 1]
            )
            beta[t] = np.logaddexp(end_term, cont)
        return beta

    # ------------------------------------------------------------------
    def run(self, obs: np.ndarray) -> ForwardBackwardResult:
        """Full forward-backward: smoothing posterior p_t(i) (PRD §5.1).

        ``obs``: event tokens [T] (discrete) or log-emissions [T, M]
        (ablation A1's continuous emissions).
        """
        alpha, ll = self.forward(obs)
        beta = self.backward(obs)
        joint = alpha + beta  # [T, M, Dmax]
        norm = logsumexp(joint, axis=(1, 2), keepdims=True)  # [T, 1, 1]
        gamma = np.exp(joint - norm)  # posterior over (state, elapsed)
        state_posterior = gamma.sum(axis=2)  # [T, M]
        state_posterior /= state_posterior.sum(axis=1, keepdims=True).clip(1e-300)
        e_idx = np.arange(1, self.hsmm.d_max + 1, dtype=np.float64)
        expected_dwell = (gamma * e_idx[None, None, :]).sum(axis=2)  # [T, M]
        denom = gamma.sum(axis=2).clip(1e-300)
        expected_dwell = np.where(denom > 0, expected_dwell / denom, 0.0)
        return ForwardBackwardResult(
            log_likelihood=ll,
            state_posterior=state_posterior,
            expected_dwell=expected_dwell,
            alpha=alpha,
            beta=beta,
            alpha_state_marginal=logsumexp(alpha, axis=2),
            beta_state_marginal=logsumexp(beta, axis=2),
        )
