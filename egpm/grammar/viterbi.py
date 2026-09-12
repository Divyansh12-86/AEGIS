"""Segmental Viterbi decoding for the HSMM (PRD §10.3, M4).

Jointly maximizes over segment boundaries, state assignments, and durations:

    delta_t(j, d) = max over segmentations of v_1..v_t where the last segment
    has state j and length d (covers [t-d+1..t]).

Backpointers store (prev_state, segment_length) so the decoded path contains
per-segment states + elapsed durations for the explanation object (PRD §18).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .hsmm import HSMM
from .forward_backward import ForwardBackward

_LOG_EPS = -700.0


@dataclass
class ViterbiResult:
    path: np.ndarray  # [T] state index at each t
    segments: List[Tuple[int, int, int]]  # (state, start, end) inclusive
    log_likelihood: float  # log P(v, best path) — must be <= total log-lik
    duration_estimates: np.ndarray  # [T] elapsed dwell within segment at t


class SegmentalViterbi:
    """Most-probable state-and-duration path (PRD §10.3)."""

    def __init__(self, hsmm: HSMM):
        self.hsmm = hsmm
        hsmm.validate()

    # ------------------------------------------------------------------
    @staticmethod
    def _obs_len(obs: np.ndarray) -> int:
        return int(np.asarray(obs).shape[0])

    # ------------------------------------------------------------------
    def decode(self, obs: np.ndarray) -> ViterbiResult:
        """Most-probable state-and-duration path.

        ``obs``: event tokens [T] (discrete) or external log-emissions
        [T, M] (ablation A1 seam), mirroring ForwardBackward.
        """
        T = self._obs_len(obs)
        M = self.hsmm.M
        Dmax = self.hsmm.d_max
        logA = np.log(np.clip(self.hsmm.A, 1e-300, None))
        logD = np.log(np.clip(self.hsmm.D.pmf, 1e-300, None))
        # per-window log emissions [M, T] (tokens OR injected logE)
        logE_T = ForwardBackward._obs_to_logE(self.hsmm, obs)

        # prefix emissions for O(1) window products
        c = np.zeros((M, T + 1), dtype=np.float64)
        c[:, 1:] = np.cumsum(logE_T, axis=1)

        # zero-diagonal masking consistent with ForwardBackward:
        # transient same-state segments illegal; absorbing self-loop legal;
        # degenerate d_max=1 models (A2b) keep the diagonal (it IS the dwell)
        degenerate = Dmax == 1

        # delta[t, j] = best log-lik of v_1..v_t with a segment ending in j at t
        delta = np.full((T, M), _LOG_EPS, dtype=np.float64)
        # back state[t, j] = previous state argmax; back len[t, j] = its length
        back_state = np.full((T, M), -1, dtype=np.int64)
        back_len = np.zeros((T, M), dtype=np.int64)

        for t in range(T):
            for j in range(M):
                best_score = _LOG_EPS
                best_d, best_i = 1, -1
                for d in range(1, min(Dmax, t + 1) + 1):
                    emis = c[j, t + 1] - c[j, t + 1 - d]
                    if t - d < 0:
                        cand = np.log(np.clip(self.hsmm.pi[j], 1e-300, None)) \
                            + logD[j, d - 1] + emis
                        i_star = -1  # sequence start
                    else:
                        ai = delta[t - d] + logA[:, j]
                        # mask same-state segments per the zero-diagonal rule
                        if not degenerate and j != self.hsmm.absorbing:
                            ai[j] = _LOG_EPS
                        i_star = int(np.argmax(ai))
                        cand = ai[i_star] + logD[j, d - 1] + emis
                    if cand > best_score:
                        best_score, best_d, best_i = cand, d, i_star
                delta[t, j] = best_score
                back_len[t, j] = best_d
                back_state[t, j] = best_i

        # final state: argmax over delta[T-1]
        j_final = int(np.argmax(delta[T - 1]))
        best_ll = float(delta[T - 1, j_final])

        # backtrack segments
        segments: List[Tuple[int, int, int]] = []
        t = T - 1
        j = j_final
        while t >= 0:
            d = int(back_len[t, j])
            start = t - d + 1
            segments.append((j, start, t))
            prev = int(back_state[t, j])
            if prev < 0:
                break
            t = start - 1
            j = prev
        segments.reverse()

        # expand to per-step path + elapsed dwell
        path = np.zeros(T, dtype=np.int64)
        durations = np.zeros(T, dtype=np.int64)
        for (s, a, b) in segments:
            for pos in range(a, b + 1):
                path[pos] = s
                durations[pos] = pos - a + 1

        return ViterbiResult(
            path=path,
            segments=segments,
            log_likelihood=best_ll,
            duration_estimates=durations,
        )
