"""Test causal forward filtering vs offline smoothing posterior (PRD §5 / Audit).

Sanity checks:
1. P(S_t | v_0:t) from causal_filter is strictly identical whether evaluated on
   the full sequence v_0:T-1, a truncated prefix v_0:t, or a sequence with
   perturbed future tokens v_t+1:T-1.
2. In contrast, the smoothing posterior P(S_t | v_0:T-1) changes when future
   tokens are perturbed or removed.
"""
import numpy as np
import pytest

from egpm.grammar import HSMM, ForwardBackward
from egpm.grammar.hsmm import DurationHistogram


def _make_toy_hsmm(M: int = 3, K: int = 6, d_max: int = 5) -> HSMM:
    return HSMM(
        n_states=M,
        n_events=K,
        d_max=d_max,
        seed=42,
    )


def test_causal_filter_strictly_prefix_invariant():
    hsmm = _make_toy_hsmm(M=4, K=8, d_max=6)
    fb = ForwardBackward(hsmm)

    rng = np.random.default_rng(123)
    T = 25
    tokens = rng.integers(0, hsmm.K, size=T)

    t = 7  # early timestep

    # 1. Full sequence causal filter
    res_full = fb.causal_filter(tokens)

    # 2. Truncated prefix sequence up to t
    res_prefix = fb.causal_filter(tokens[: t + 1])

    # 3. Perturbed future tokens (prefix identical, future completely altered)
    tokens_perturbed = tokens.copy()
    tokens_perturbed[t + 1 :] = (tokens_perturbed[t + 1 :] + rng.integers(1, hsmm.K, size=T - t - 1)) % hsmm.K
    res_perturbed = fb.causal_filter(tokens_perturbed)

    # Assert causal posterior at t is identical to machine precision
    np.testing.assert_allclose(
        res_full.state_posterior[t],
        res_prefix.state_posterior[t],
        atol=1e-12,
        err_msg="Causal posterior at t changed when sequence was truncated!",
    )
    np.testing.assert_allclose(
        res_full.expected_dwell[t],
        res_prefix.expected_dwell[t],
        atol=1e-12,
        err_msg="Causal expected dwell at t changed when sequence was truncated!",
    )
    np.testing.assert_allclose(
        res_full.state_posterior[t],
        res_perturbed.state_posterior[t],
        atol=1e-12,
        err_msg="Causal posterior at t changed when future tokens were perturbed!",
    )
    np.testing.assert_allclose(
        res_full.expected_dwell[t],
        res_perturbed.expected_dwell[t],
        atol=1e-12,
        err_msg="Causal expected dwell at t changed when future tokens were perturbed!",
    )


def test_smoothing_posterior_is_future_dependent():
    hsmm = _make_toy_hsmm(M=4, K=8, d_max=6)
    fb = ForwardBackward(hsmm)

    rng = np.random.default_rng(123)
    T = 25
    tokens = rng.integers(0, hsmm.K, size=T)

    t = 7  # early timestep

    res_smooth_full = fb.run(tokens, mode="smoothing")

    tokens_perturbed = tokens.copy()
    tokens_perturbed[t + 1 :] = (tokens_perturbed[t + 1 :] + rng.integers(1, hsmm.K, size=T - t - 1)) % hsmm.K
    res_smooth_perturbed = fb.run(tokens_perturbed, mode="smoothing")

    # The smoothing posterior at t MUST differ when future tokens change
    diff = np.max(np.abs(res_smooth_full.state_posterior[t] - res_smooth_perturbed.state_posterior[t]))
    assert diff > 1e-4, f"Expected smoothing posterior to change with perturbed future, but diff was {diff}"
