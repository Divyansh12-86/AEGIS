"""§24 interpretability validation tests (PRD, experimental-completion bar).

Metric semantics are validated on CONSTRUCTED ground truth (known-coherent /
known-random clusters, known relabelings, known onsets), and the runner's
report schema on a small synthetic pipeline.
"""
import json

import numpy as np
import pytest

from egpm.data import UnitDataset, UnitRecord
from egpm.grammar import HSMM
from egpm.anomaly import AnomalyScorer
from egpm.interpretability import (
    event_coherence,
    event_stability,
    hungarian_state_match,
    cross_unit_consistency,
    temporal_validity,
    explanation_faithfulness,
    anomaly_score_rank_correlation,
    run_interpretability_suite,
)


# ---------------------------------------------------------------------------
# Coherence
# ---------------------------------------------------------------------------
class TestEventCoherence:
    def test_separated_clusters_coherent(self):
        rng = np.random.default_rng(0)
        # three well-separated token groups
        centers = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
        X = np.concatenate([c + 0.1 * rng.standard_normal((20, 2)) for c in centers])
        ids = np.repeat([0, 1, 2], 20)
        r = event_coherence(X, ids)
        assert r["silhouette"] > 0.8
        assert r["within_between_ratio"] < 0.2

    def test_random_assignment_incoherent(self):
        rng = np.random.default_rng(1)
        X = rng.standard_normal((60, 2))
        ids = rng.integers(0, 3, 60)
        r = event_coherence(X, ids)
        assert r["silhouette"] < 0.15
        assert r["within_between_ratio"] > 0.7

    def test_single_token_degenerate(self):
        X = np.random.default_rng(2).standard_normal((10, 3))
        r = event_coherence(X, np.zeros(10, dtype=int))
        assert r["silhouette"] == 0.0

    def test_length_mismatch_rejected(self):
        with pytest.raises(ValueError):
            event_coherence(np.zeros((5, 2)), np.zeros(4))


# ---------------------------------------------------------------------------
# Stability
# ---------------------------------------------------------------------------
class TestEventStability:
    def test_identical_usage_overlap_one(self):
        runs = [np.array([0, 0, 1, 1, 2, 2])]
        r = event_stability(runs, runs, n_codes=4)
        assert r["matched_usage_overlap"] == pytest.approx(1.0)

    def test_relabeling_invariant(self):
        # seed B permutes token ids but keeps the distribution
        a = [np.array([0, 0, 1, 1, 1, 2, 2])]
        b = [np.array([3, 3, 2, 2, 2, 1, 1])]  # 0<->3, 1<->2
        r = event_stability(a, b, n_codes=4)
        assert r["matched_usage_overlap"] == pytest.approx(1.0)

    def test_different_shapes_low_overlap(self):
        # matched_usage_overlap compares distribution SHAPES modulo
        # relabeling: concentrated vs spread usage must differ
        a = [np.array([0] * 10)]  # collapsed: one token
        b = [np.array([0, 1, 2, 3] * 3)]  # uniform across 4 tokens
        r = event_stability(a, b, n_codes=4)
        assert r["matched_usage_overlap"] < 0.5

    def test_same_shape_different_labels_full_overlap(self):
        # identical usage SHAPES under relabeling are 'stable' by this
        # metric's semantics (token identity alone carries no information)
        a = [np.array([0] * 6 + [1] * 4)]
        b = [np.array([2] * 6 + [3] * 4)]
        r = event_stability(a, b, n_codes=4)
        assert r["matched_usage_overlap"] == pytest.approx(1.0)

    def test_state_match_relabeling_invariant(self):
        post_a = [np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1]])]
        post_b = [np.array([[0.1, 0.8, 0.1], [0.1, 0.2, 0.7]])]  # shifted by 1
        assert hungarian_state_match(post_a, post_b) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Cross-unit consistency
# ---------------------------------------------------------------------------
class TestCrossUnitConsistency:
    def test_identical_distributions_zero_jsd(self):
        runs = [np.array([0, 1, 2, 0, 1, 2])]
        r = cross_unit_consistency(runs, runs, n_codes=4)
        assert r["jsd"] == pytest.approx(0.0, abs=1e-9)

    def test_shifted_distributions_positive_jsd(self):
        train = [np.array([0, 0, 0, 0, 1, 1])]
        heldout = [np.array([2, 2, 2, 2, 3, 3])]
        r = cross_unit_consistency(train, heldout, n_codes=4)
        assert r["jsd"] > 0.5


# ---------------------------------------------------------------------------
# Temporal validity
# ---------------------------------------------------------------------------
class TestTemporalValidity:
    def test_exact_boundaries_detected(self):
        r = temporal_validity([10, 20], [10, 20], tolerance=2)
        assert r["detection_rate"] == 1.0
        assert r["mean_abs_lag"] == 0.0

    def test_lagged_boundaries_measured(self):
        r = temporal_validity([12, 24], [10, 20], tolerance=4)
        assert r["detection_rate"] == 1.0
        assert r["mean_lag"] == pytest.approx(3.0)

    def test_undetected_onsets_counted(self):
        r = temporal_validity([50], [10, 20], tolerance=2)
        assert r["detection_rate"] == pytest.approx(0.0)

    def test_no_truth_rejected(self):
        with pytest.raises(ValueError):
            temporal_validity([10], [])


# ---------------------------------------------------------------------------
# Faithfulness (the primary §24 experiment)
# ---------------------------------------------------------------------------
class TestFaithfulness:
    def _scored_hsmm(self):
        # state 0 loves token 0; token 4 never emitted -> high surprisal
        from egpm.grammar.hsmm import DurationHistogram
        B = np.array([
            [0.9, 0.02, 0.02, 0.02, 0.04],
            [0.02, 0.9, 0.02, 0.02, 0.04],
            [0.02, 0.02, 0.9, 0.02, 0.04],
        ])
        D = np.zeros((3, 4))
        D[:, 2] = 1.0
        return HSMM(
            n_states=3, n_events=5,
            pi=np.array([1.0, 0.0, 0.0]),
            A=np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
            B=B, D=DurationHistogram(pmf=D), d_max=4,
        )

    def test_masking_flagged_windows_reduces_score(self):
        hsmm = self._scored_hsmm()
        scorer = AnomalyScorer()
        v = np.array([0, 0, 0, 4, 4, 0, 0, 0])  # rare-token burst
        comp = scorer.score_sequence(hsmm, v)
        flagged = list(np.nonzero(v == 4)[0])
        f = explanation_faithfulness(v, flagged, hsmm, scorer)
        assert f["score_reduction"] > 0.0
        assert f["n_flagged"] == 2

    def test_masking_neutral_windows_no_inversion(self):
        # masking NON-anomalous windows must not produce a large positive
        # reduction (the metric's specificity check)
        hsmm = self._scored_hsmm()
        scorer = AnomalyScorer()
        v = np.array([0, 0, 0, 4, 4, 0, 0, 0])
        neutral = [0, 1]  # well-explained windows
        f = explanation_faithfulness(v, neutral, hsmm, scorer)
        assert f["score_reduction"] < 0.5

    def test_empty_flag_list(self):
        hsmm = self._scored_hsmm()
        f = explanation_faithfulness(np.array([0, 0]), [], hsmm, AnomalyScorer())
        assert f["score_reduction"] == 0.0 and f["n_flagged"] == 0


# ---------------------------------------------------------------------------
# Rank correlation (renamed; never faithfulness — Decision 9)
# ---------------------------------------------------------------------------
class TestRankCorrelation:
    def test_perfect_ranking_positive(self):
        # scipy's Spearman averages ranks for the binary label's ties, so
        # the CEILING for a 50/50 binary label at n=200 is sqrt(3)/2
        # (~0.866) — 0.866 IS perfect separation. The metric must hit that
        # ceiling on perfectly separated scores and stay strongly positive
        # on the small case.
        r = anomaly_score_rank_correlation(
            np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1])
        )
        assert r["spearman"] > 0.8
        rng = np.random.default_rng(0)
        n = 200
        scores = np.concatenate([rng.uniform(0, 0.4, n // 2),
                                 rng.uniform(0.6, 1.0, n // 2)])
        labels = np.concatenate([np.zeros(n // 2, int), np.ones(n // 2, int)])
        r_big = anomaly_score_rank_correlation(scores, labels)
        ceiling = np.sqrt(3) / 2  # exact max for 50/50 binary labels
        assert r_big["spearman"] > 0.95 * ceiling

    def test_inverse_ranking(self):
        r = anomaly_score_rank_correlation(
            np.array([0.9, 0.8, 0.2, 0.1]), np.array([0, 0, 1, 1])
        )
        assert r["spearman"] < -0.8

    def test_single_class_returns_nan(self):
        r = anomaly_score_rank_correlation(np.array([0.1, 0.2]), np.array([1, 1]))
        assert np.isnan(r["spearman"])

    def test_length_mismatch_rejected(self):
        with pytest.raises(ValueError):
            anomaly_score_rank_correlation(np.array([0.1]), np.array([0, 1]))


# ---------------------------------------------------------------------------
# Full-suite runner
# ---------------------------------------------------------------------------
@pytest.fixture(scope="class")
def suite_dataset():
    """Clean §24 fixture: shared nominal behavior, distinct anomaly only."""
    rng = np.random.default_rng(0)
    ds = UnitDataset()
    for u in range(10):
        T = 96
        t = np.arange(T)
        phase = t // 32
        freqs = [0.7, 1.9, 3.1]
        base = np.stack(
            [np.sin(freqs[phase[i]] * t[i] + c) for i in range(T) for c in range(2)],
            axis=0,
        ).reshape(T, 2)
        drift = (t / T) ** 2 * 1.2
        signals = base + drift[:, None] + 0.05 * rng.standard_normal((T, 2))
        health = phase + 1
        anom = np.zeros(T, dtype=int)
        if u >= 7:
            anom[:] = 1
            signals[56:88] += rng.standard_normal((32, 2)) * 3.0
        ds.add(UnitRecord(
            unit_id=f"synth_u{u}", signals=signals, timestamps=t.astype(float),
            health_labels=health, anomaly_labels=anom,
        ))
    return ds


class TestInterpretabilityRunner:
    def test_report_schema_all_six_metrics(self, suite_dataset):
        report = run_interpretability_suite(
            suite_dataset, window_length=8, stage1_steps=40,
            n_states=3, d_max=6, n_codes=8,
        )
        parsed = json.loads(report.to_json())
        assert parsed["experiment"] == "interpretability_suite_sec24"
        for key in ("event_coherence", "event_stability",
                    "cross_unit_consistency", "temporal_validity",
                    "explanation_faithfulness", "anomaly_score_rank_correlation"):
            assert key in parsed["results"], key

    def test_pass_signals_on_clean_fixture(self, suite_dataset):
        """The constructed ground truth must yield the PRD's pass signals:
        coherent tokens, stable vocabulary, transferable usage, detected
        onsets, and a positive faithfulness reduction."""
        report = run_interpretability_suite(
            suite_dataset, window_length=8, stage1_steps=40,
            n_states=3, d_max=6, n_codes=8,
        )
        r = report.results
        assert r["event_coherence"]["within_between_ratio"] < 0.9
        assert r["event_stability"]["matched_usage_overlap"] > 0.5
        assert r["cross_unit_consistency"]["jsd"] < 0.3
        assert r["temporal_validity"]["mean_detection_rate"] > 0.5
        assert r["explanation_faithfulness"]["mean_score_reduction"] > 0.0

    def test_hsmm_fit_excludes_anomalous_train_units(self, suite_dataset):
        """PRD §10.4 audit: anomalous runs must never feed Stage-2 EM
        (verified indirectly — the anomalous runs must score HIGHER than
        normal runs under the normal-fit model)."""
        report = run_interpretability_suite(
            suite_dataset, window_length=8, stage1_steps=40,
            n_states=3, d_max=6, n_codes=8,
        )
        rho = report.results["anomaly_score_rank_correlation"]["spearman"]
        assert rho > 0, f"normal-fit HSMM failed to rank anomalies: {rho}"
