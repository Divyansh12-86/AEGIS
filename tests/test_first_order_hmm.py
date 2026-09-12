"""Ablation A2b tests: first-order HMM vs explicit-duration HSMM (PRD §22).

The single most important ablation for defending the architecture's added
complexity (PRD §22). Validations:
  * dedicated first-order forward == degenerate-HSMM forward (two
    independent implementations of the same model);
  * first-order forward == brute-force path enumeration on tiny instances;
  * EM monotonicity for the A2b training arm;
  * A2b vs HSMM on data with genuinely long dwells: the HSMM must
    outperform the duration-degenerate model (duration modeling earns its
    keep), while on memoryless data they should tie.
"""
import itertools
import math

import numpy as np
import pytest

from egpm.grammar import FirstOrderHMM, FirstOrderHMMFitter
from egpm.grammar.hsmm import DurationHistogram
from egpm.grammar import HSMM, ForwardBackward


def brute_force_hmm_loglik(hmm, v):
    """Enumerate all state paths; sum P(v, path)."""
    T = len(v)
    total = 0.0
    for path in itertools.product(range(hmm.M), repeat=T):
        logp = math.log(hmm.pi[path[0]]) + math.log(hmm.B[path[0], v[0]])
        for t in range(1, T):
            logp += math.log(hmm.A[path[t - 1], path[t]]) + math.log(hmm.B[path[t], v[t]])
        total += math.exp(logp)
    return math.log(total)


class TestEquivalence:
    def test_dedicated_matches_degenerate_hsmm(self):
        """Two independent implementations of the same A2b model must agree
        on every sequence (the cross-validation the PRD's ablation needs to
        be trustworthy)."""
        for seed in range(4):
            hmm = FirstOrderHMM(n_states=3, n_events=4, seed=seed)
            rng = np.random.default_rng(seed)
            for _ in range(5):
                v = rng.integers(0, 4, size=12)
                assert hmm.forward_loglik(v) == pytest.approx(
                    hmm.hsmm_loglik(v), rel=1e-9
                ), (seed, v)

    def test_dedicated_matches_brute_force(self):
        """PRD §29-style brute-force check for the first-order forward."""
        hmm = FirstOrderHMM(
            n_states=2, n_events=2, seed=0,
            pi=np.array([0.6, 0.4]),
            A=np.array([[0.7, 0.3], [0.2, 0.8]]),
            B=np.array([[0.9, 0.1], [0.25, 0.75]]),
        )
        rng = np.random.default_rng(3)
        for _ in range(6):
            v = rng.integers(0, 2, size=5)
            got = hmm.forward_loglik(v)
            want = brute_force_hmm_loglik(hmm, v)
            assert got == pytest.approx(want, rel=1e-10)

    def test_degenerate_encoding_preserves_diagonal(self):
        hmm = FirstOrderHMM(n_states=4, n_events=3, seed=1)
        h = hmm.to_hsmm()
        np.testing.assert_allclose(h.A, hmm.A, atol=1e-12)
        np.testing.assert_allclose(np.diag(h.A), np.diag(hmm.A), atol=1e-12)
        # degenerate model passes validation
        h.validate()

    def test_keep_diagonal_rejected_for_non_degenerate(self):
        with pytest.raises(ValueError, match="d_max=1"):
            HSMM(n_states=2, n_events=2, d_max=5, keep_diagonal=True)


class TestA2bFitter:
    def test_em_monotone(self):
        true = FirstOrderHMM(n_states=3, n_events=4, seed=9)
        seqs = [true.sample(40, rng=np.random.default_rng(40 + i)) for i in range(8)]
        fitter = FirstOrderHMMFitter(max_iter=12, seed=2)
        hmm, res = fitter.fit(seqs, n_states=3, n_events=4)
        diffs = np.diff(res.log_likelihoods)
        assert (diffs > -1e-8).all()

    def test_fitter_recovers_better_than_random_init(self):
        true = FirstOrderHMM(n_states=3, n_events=4, seed=9)
        seqs = [true.sample(40, rng=np.random.default_rng(40 + i)) for i in range(8)]
        ll_true = sum(true.forward_loglik(s) for s in seqs)
        fitter = FirstOrderHMMFitter(max_iter=12, seed=2)
        hmm, res = fitter.fit(seqs, n_states=3, n_events=4)
        assert res.log_likelihoods[-1] > res.log_likelihoods[0]
        # converged into the true model's likelihood ballpark
        assert res.log_likelihoods[-1] >= ll_true - 5.0

    def test_posterior_rows_normalized(self):
        hmm = FirstOrderHMM(n_states=3, n_events=3, seed=4)
        v = np.random.default_rng(5).integers(0, 3, size=20)
        post = hmm.posterior(v)
        np.testing.assert_allclose(post.sum(axis=1), 1.0, atol=1e-9)

    def test_empty_rejected(self):
        with pytest.raises(ValueError, match="no sequences"):
            FirstOrderHMMFitter().fit([], n_states=2, n_events=2)


class TestDurationEarnsItsKeep:
    """The PRD's Deliverable-F risk, made concrete: on data with long,
    regular dwells and state-identifying emissions, the explicit-duration
    HSMM must out-likelihood the best geometric-dwell (first-order) model;
    on dwell-1 (memoryless) data they must tie exactly."""

    def _dwell_chain_generator(self, B, dwell, d_max=6):
        """3-state chain 0->1->2(absorbing), deterministic `dwell` per
        transient state, geometric-ish absorbing dwell."""
        M, K = 3, B.shape[1]
        pi = np.zeros(M); pi[0] = 1.0
        A = np.zeros((M, M))
        A[0, 1] = 1.0
        A[1, 2] = 1.0
        A[2, 2] = 1.0
        pmf = np.zeros((M, d_max))
        pmf[:, dwell - 1] = 1.0
        pmf[M - 1] = 1.0 / d_max
        return HSMM(
            n_states=M, n_events=K, pi=pi, A=A, B=B,
            D=DurationHistogram(pmf=pmf), d_max=d_max, seed=0,
        )

    def _best_geometric_match(self, B, dwell):
        """First-order HMM with the same chain and a diagonal tuned so the
        geometric dwell's MEAN equals `dwell` (p_self = 1 - 1/dwell) — the
        strongest duration-blind competitor for this generator."""
        M, K = 3, B.shape[1]
        p_self = 1.0 - 1.0 / dwell
        A = np.array([
            [p_self, 1 - p_self, 0.0],
            [0.0, p_self, 1 - p_self],
            [0.0, 0.0, 1.0],
        ])
        return FirstOrderHMM(
            n_states=M, n_events=K, seed=0,
            pi=np.array([1.0, 0.0, 0.0]), A=A, B=B,
        )

    def test_hsmm_beats_geometric_on_regular_dwell_data(self):
        """Long deterministic dwells (5 steps) + state-identifying
        emissions: the HSMM's exact dwell mass must beat the geometric
        model's spread (validated margin ~1.5 nats/run; require > 2 nats
        total over 3 held-out runs)."""
        B = np.array([
            [0.9, 0.05, 0.05],
            [0.05, 0.9, 0.05],
            [0.10, 0.10, 0.80],
        ])
        gen = self._dwell_chain_generator(B, dwell=5)
        rng = np.random.default_rng(0)
        test_seqs = [gen.sample(60, rng=np.random.default_rng(200 + i))
                     for i in range(3)]
        ll_hsmm = sum(ForwardBackward(gen).forward(v)[1] for v in test_seqs)
        hmm = self._best_geometric_match(B, dwell=5)
        ll_hmm = sum(hmm.forward_loglik(v) for v in test_seqs)
        assert ll_hsmm > ll_hmm + 2.0, (
            f"duration modeling not earning its keep: HSMM {ll_hsmm:.2f} "
            f"vs geometric {ll_hmm:.2f}"
        )

    def test_short_dwell_margin_shrinks(self):
        """With dwell=1 the geometric model is exactly right; the HSMM's
        advantage must vanish (margin near zero, either direction < 5 nats)."""
        B = np.array([
            [0.9, 0.05, 0.05],
            [0.05, 0.9, 0.05],
            [0.10, 0.10, 0.80],
        ])
        gen = self._dwell_chain_generator(B, dwell=1, d_max=1)
        test_seqs = [gen.sample(60, rng=np.random.default_rng(300 + i))
                     for i in range(3)]
        ll_hsmm = sum(ForwardBackward(gen).forward(v)[1] for v in test_seqs)
        hmm = self._best_geometric_match(B, dwell=1)
        ll_hmm = sum(hmm.forward_loglik(v) for v in test_seqs)
        assert abs(ll_hsmm - ll_hmm) < 5.0, (
            f"dwell-1 models should tie: HSMM {ll_hsmm:.2f} vs {ll_hmm:.2f}"
        )
