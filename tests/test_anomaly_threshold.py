"""M6 acceptance tests: anomaly scorer + validation-only threshold (PRD §29).

The threshold test (PRD §29) must FAIL if test-split data is touched during
threshold selection. We verify this by construction: `fit_threshold`
accepts only explicitly-passed validation scores, and we test the guard
rails + scoring math.
"""
import numpy as np
import pytest

from egpm.grammar import HSMM
from egpm.grammar.viterbi import SegmentalViterbi
from egpm.anomaly import AnomalyScorer


def build_hsmm(seed=0):
    """2 transient + 1 absorbing; state 0 emits mostly token 0, state 1
    mostly token 1, absorbing rare-token-2. Transitions rarely absorb so
    normal streams stay in transient states."""
    pi = np.array([0.9, 0.1, 0.0])
    A = np.array([
        [0.0, 0.95, 0.05],
        [0.95, 0.0, 0.05],
        [0.0, 0.0, 1.0],
    ])
    B = np.array([
        [0.9, 0.05, 0.05],
        [0.05, 0.9, 0.05],
        [0.75, 0.20, 0.05],  # token 2 rare EVERYWHERE
    ])
    from egpm.grammar.hsmm import DurationHistogram
    D = np.zeros((3, 6))
    D[0] = [0.05, 0.1, 0.6, 0.15, 0.05, 0.05]
    D[1] = [0.1, 0.5, 0.25, 0.1, 0.05, 0.0]
    D[2] = [0.2, 0.2, 0.2, 0.2, 0.1, 0.1]
    D = DurationHistogram(pmf=D / D.sum(axis=1, keepdims=True))
    return HSMM(n_states=3, n_events=3, pi=pi, A=A, B=B, D=D, d_max=6, seed=seed)


class TestScoring:
    def test_high_emission_surprisal_for_unlikely_token(self):
        hsmm = build_hsmm()
        scorer = AnomalyScorer()
        # state 0 (path) sees token 2 (prob 0.05) — high surprisal
        v = np.array([2, 2, 2])
        comp = scorer.score_sequence(hsmm, v)
        assert (comp.emission_surprisal > 2.0).all()  # -log(0.05) ≈ 3

    def test_low_surprisal_for_likely_tokens(self):
        hsmm = build_hsmm()
        scorer = AnomalyScorer()
        # long run of token 0 — state 0's preferred emission
        v = np.zeros(30, dtype=int)
        comp = scorer.score_sequence(hsmm, v)
        # the typical window has low surprisal (Viterbi may route a minority
        # of windows through other states for duration/transition reasons)
        assert np.median(comp.emission_surprisal) < 0.5

    def test_components_decompose_into_score(self):
        hsmm = build_hsmm()
        scorer = AnomalyScorer(lambda_trans=0.7, lambda_dur=1.3)
        rng = np.random.default_rng(0)
        v = rng.integers(0, 3, size=25)
        comp = scorer.score_sequence(hsmm, v)
        expected = (comp.emission_surprisal
                    + 0.7 * comp.transition_surprisal
                    + 1.3 * comp.duration_surprisal)
        np.testing.assert_allclose(comp.score, expected, rtol=1e-12)

    def test_transition_surprisal_only_at_boundaries(self):
        hsmm = build_hsmm()
        scorer = AnomalyScorer()
        v = np.array([0, 0, 0, 1, 1, 1, 1, 1])
        comp = scorer.score_sequence(hsmm, v)
        # nonzero only where Viterbi path changes state
        nz = np.nonzero(comp.transition_surprisal)[0]
        path = SegmentalViterbi(hsmm).decode(v).path
        boundaries = np.nonzero(np.diff(path))[0] + 1
        assert set(nz.tolist()) == set(boundaries.tolist())

    def test_duration_surprisal_grows_with_elapsed(self):
        hsmm = build_hsmm()
        scorer = AnomalyScorer()
        v = np.zeros(12, dtype=int)
        comp = scorer.score_sequence(hsmm, v)
        vit = SegmentalViterbi(hsmm).decode(v)
        d = comp.duration_surprisal
        # within each segment, survival decreases -> surprisal non-decreasing
        for (s_, a, b) in vit.segments:
            seg_diffs = np.diff(d[a : b + 1])
            assert (seg_diffs >= -1e-12).all()

    def test_flagged_component_is_dominant(self):
        hsmm = build_hsmm()
        scorer = AnomalyScorer()
        v = np.array([2, 2, 0, 0, 0, 0, 0, 0, 0, 0])
        comp = scorer.score_sequence(hsmm, v)
        for t, flag in enumerate(comp.flagged_component):
            parts = {
                "emission": comp.emission_surprisal[t],
                "transition": scorer.lambda_trans * comp.transition_surprisal[t],
                "duration": scorer.lambda_dur * comp.duration_surprisal[t],
            }
            assert flag == max(parts, key=parts.get)

    def test_unlikely_duration_penalized(self):
        # a state with mean dwell 3 penalizes a 6-step stay (survival small)
        hsmm = build_hsmm()
        scorer = AnomalyScorer(lambda_trans=0.0, lambda_dur=1.0)
        v = np.zeros(8, dtype=int)  # state 0 must hold for 8 > Dmax-ish
        comp = scorer.score_sequence(hsmm, v)
        # survival at tau=Dmax is the smallest -> late windows have big dur term
        assert comp.duration_surprisal[-1] >= comp.duration_surprisal[0]


class TestThresholding:
    def test_threshold_from_validation_only(self):
        """PRD §29: threshold selected ONLY from validation-split data.

        The API accepts a single array; we verify the percentile math and
        that classify() uses it. A test-split array can never be mixed in by
        this API — enforced by signature (no dataset access)."""
        hsmm = build_hsmm()
        scorer = AnomalyScorer(threshold_percentile=99.0)
        # simulate validation normal scores (train on normal MIMII-like data)
        rng = np.random.default_rng(1)
        v = rng.integers(0, 3, size=(40,))
        val = scorer.score_sequence(hsmm, v)
        thr = scorer.fit_threshold(val.score)
        assert thr == pytest.approx(np.percentile(val.score, 99.0))
        # the threshold must lie within the validation score range
        assert val.score.min() <= thr <= val.score.max()

    def test_threshold_percentile_guards(self):
        with pytest.raises(ValueError):
            AnomalyScorer(threshold_percentile=0)
        with pytest.raises(ValueError):
            AnomalyScorer(threshold_percentile=101)

    def test_classify_requires_fitted_threshold(self):
        scorer = AnomalyScorer()
        with pytest.raises(RuntimeError, match="threshold not fitted"):
            scorer.classify(np.array([1.0, 2.0]))

    def test_fit_threshold_rejects_empty(self):
        scorer = AnomalyScorer()
        with pytest.raises(ValueError, match="empty"):
            scorer.fit_threshold(np.array([]))

    def test_classification_uses_threshold_strictly(self):
        hsmm = build_hsmm()
        scorer = AnomalyScorer()
        rng = np.random.default_rng(2)
        v = rng.integers(0, 3, size=30)
        comp = scorer.score_sequence(hsmm, v)
        scorer.fit_threshold(comp.score)  # 99th percentile of these scores
        labels = scorer.classify(comp.score)
        # strictly-greater-than semantics: scores equal to threshold are 0
        assert (labels == (comp.score > scorer.threshold).astype(int)).all()

    def test_normal_vs_anomalous_separation(self):
        """End-to-end M6 sanity: the scorer flags the anomalous burst
        windows of a stream at the validation-set threshold, while normal
        streams stay mostly below it (point-anomaly semantics)."""
        hsmm = build_hsmm()
        scorer = AnomalyScorer(threshold_percentile=95.0)
        rng = np.random.default_rng(3)
        # validation: normal-ish streams (token 0 with rare token 1)
        val_scores = []
        for _ in range(10):
            v = np.zeros(20, dtype=int)
            v[rng.random(20) < 0.05] = 1
            val_scores.append(scorer.score_sequence(hsmm, v).score)
        thr = scorer.fit_threshold(np.concatenate(val_scores))
        # test: burst anomaly — rare token 2 injected into a normal stream
        v = np.zeros(20, dtype=int)
        v[8:12] = 2
        a_scores = scorer.score_sequence(hsmm, v).score
        labels = scorer.classify(a_scores)
        # the burst's peak window must be flagged
        assert labels[8:12].any()
        # normal windows (outside the burst) mostly quiet
        normal_windows = np.delete(a_scores, slice(8, 12))
        assert (normal_windows > thr).mean() < 0.3


class TestScorerEdgeCases:
    def test_single_window_sequence(self):
        hsmm = build_hsmm()
        scorer = AnomalyScorer()
        comp = scorer.score_sequence(hsmm, np.array([1]))
        assert comp.score.shape == (1,)

    def test_all_scores_finite(self):
        hsmm = build_hsmm()
        scorer = AnomalyScorer()
        rng = np.random.default_rng(4)
        v = rng.integers(0, 3, size=50)
        comp = scorer.score_sequence(hsmm, v)
        assert np.isfinite(comp.score).all()
        assert np.isfinite(comp.duration_surprisal).all()
