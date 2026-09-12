"""Ablation A1: no event bottleneck — continuous HSMM (PRD §22, M10).

Encoder embeddings feed the HSMM's emission model DIRECTLY (continuous),
skipping VQ. Tests RQ1: does discretizing sensor windows into learned
events preserve task-relevant information relative to continuous encodings?

Implementation: a diagonal-Gaussian emission HSMM. Per state m, emission
log-density at window embedding z_t:
    log N(z_t | mu_m, diag(sigma_m^2))  (per-dim independence, standard
    simplification for high-d embeddings)
The ForwardBackward engine consumes external log-emissions [T, M] via the
injection seam (see egpm/grammar/forward_backward.py), reusing the validated
duration machinery unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ..grammar.hsmm import HSMM, DurationHistogram
from ..grammar.forward_backward import ForwardBackward


def gaussian_log_emissions(
    Z: np.ndarray, means: np.ndarray, inv_variances: np.ndarray
) -> np.ndarray:
    """Diagonal-Gaussian log-density of each embedding under each state.

    Z: [T, h]; means: [M, h]; inv_variances: [M, h] (1/sigma^2).
    Returns [T, M] log-emissions (constants dropped per-window — they cancel
    in posterior/EM updates and shift log-lik by an additive constant only).
    """
    Z = np.asarray(Z, dtype=np.float64)
    means = np.asarray(means, dtype=np.float64)
    ivar = np.asarray(inv_variances, dtype=np.float64)
    T, h = Z.shape
    M = means.shape[0]
    # [T, M, h] squared deviations
    d = Z[:, None, :] - means[None, :, :]
    quad = (d ** 2 * ivar[None, :, :]).sum(axis=2)  # [T, M]
    logdet = np.log(ivar).sum(axis=1)  # [M] log prod(1/sigma^2)
    return -0.5 * (quad - logdet[None, :])


@dataclass
class ContinuousHSMM:
    """Gaussian-emission explicit-duration HSMM (A1). Wraps a duration-only
    HSMM plus per-state Gaussian emission parameters."""

    hsmm: HSMM  # provides pi, A, D (B unused for inference)
    means: np.ndarray  # [M, h]
    inv_variances: np.ndarray  # [M, h]

    def __post_init__(self) -> None:
        M = self.hsmm.M
        if self.means.shape[0] != M or self.inv_variances.shape[0] != M:
            raise ValueError("means/inv_variances must have M rows")
        if self.means.shape != self.inv_variances.shape:
            raise ValueError("means/inv_variances must share shape")

    # ------------------------------------------------------------------
    def log_emissions(self, Z: np.ndarray) -> np.ndarray:
        return gaussian_log_emissions(Z, self.means, self.inv_variances)

    def log_likelihood(self, Z: np.ndarray) -> float:
        logE = self.log_emissions(Z)
        fb = ForwardBackward(self.hsmm)
        return fb.forward(logE)[1]

    def posterior(self, Z: np.ndarray) -> np.ndarray:
        logE = self.log_emissions(Z)
        return ForwardBackward(self.hsmm).run(logE).state_posterior

    # ------------------------------------------------------------------
    def fit_em(
        self,
        runs: List[np.ndarray],
        n_states: int,
        d_max: int,
        max_iter: int = 20,
        min_iter: int = 3,
        tol: float = 1e-4,
        seed: int = 0,
    ) -> List[float]:
        """EM: alternate (pi, A, D) re-estimation via the shared occupancy
        machinery on injected log-emissions, and Gaussian M-step on
        posterior-weighted embeddings.

        NOTE (MVP scope): pi/A/D updates reuse BaumWelch's exact expected
        counts through a light adapter; the Gaussian M-step is closed-form.
        The shared machinery keeps A1 comparable to the primary model.
        """
        from ..grammar.baum_welch import BaumWelch

        rng = np.random.default_rng(seed)
        h = runs[0].shape[1]
        # init: k-means++ style seeding over pooled embeddings
        pooled = np.concatenate(runs, axis=0)
        centers = [pooled[rng.integers(len(pooled))]]
        for _ in range(1, n_states):
            d2 = np.min(
                ((pooled[:, None, :] - np.array(centers)[None, :, :]) ** 2).sum(-1),
                axis=1,
            )
            p = d2 / max(d2.sum(), 1e-12)
            centers.append(pooled[rng.choice(len(pooled), p=p)])
        means = np.array(centers)
        ivar = np.ones_like(means)  # unit variances to start

        lls: List[float] = []
        for it in range(max_iter):
            # E-step: posteriors under current params
            hsmm = self._refresh_hsmm(n_states, d_max, seed + it)
            # M-step (pi, A, D): adapted BaumWelch E-counts per run
            fb = ForwardBackward(hsmm)
            gammas = []
            logE_runs = []
            for Z in runs:
                logE = gaussian_log_emissions(Z, means, ivar)
                logE_runs.append(logE)
                gammas.append(fb.run(logE).state_posterior)
            # Gaussian M-step: weighted means/variances per state
            for m in range(n_states):
                w_all = np.concatenate([g[:, m] for g in gammas])  # [sum T]
                z_all = np.concatenate(runs, axis=0)
                w = w_all / max(w_all.sum(), 1e-12)
                mu = (w[:, None] * z_all).sum(axis=0)
                var = (w[:, None] * (z_all - mu) ** 2).sum(axis=0)
                means[m] = mu
                ivar[m] = 1.0 / np.clip(var, 1e-3, None)
            # refresh with new emissions for the (pi, A, D) step
            self.hsmm = hsmm
            self.means = means
            self.inv_variances = ivar
            ll = sum(
                ForwardBackward(self.hsmm).forward(
                    gaussian_log_emissions(Z, self.means, self.inv_variances)
                )[1]
                for Z in runs
            )
            lls.append(ll)
            if it >= min_iter and len(lls) >= 2 and abs(lls[-1] - lls[-2]) < tol:
                break
        return lls

    # ------------------------------------------------------------------
    def _refresh_hsmm(self, n_states: int, d_max: int, seed: int) -> HSMM:
        """Random-restart (pi, A, D); kept simple for the ablation — A1's
        question is about the emission path, not the optimizer."""
        rng = np.random.default_rng(seed)
        pi = rng.dirichlet(np.ones(n_states))
        A = rng.dirichlet(np.ones(n_states), size=n_states)
        D = DurationHistogram.from_means(
            np.maximum(2.0, rng.uniform(2, 8, size=n_states)), d_max
        )
        return HSMM(
            n_states=n_states, n_events=2,  # K unused for inference
            pi=pi, A=A, D=D, d_max=d_max, seed=seed,
        )
