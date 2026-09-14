"""§23 unit-level statistics: per-unit aggregation, bootstrap CI, Wilcoxon."""
import numpy as np

from egpm.evaluation.stats import (
    per_unit_metrics, bootstrap_ci, paired_wilcoxon, summarize_units,
)


def test_per_unit_rmse_matches_manual():
    preds = [np.array([1.0, 2.0]), np.array([0.0, 0.0])]
    trues = [np.array([0.0, 0.0]), np.array([0.0, 0.0])]
    np.testing.assert_allclose(
        per_unit_metrics(preds, trues), [np.sqrt(2.5), 0.0]
    )


def test_bootstrap_ci_brackets_mean():
    rng = np.random.default_rng(0)
    v = rng.normal(10.0, 1.0, size=40)
    lo, hi = bootstrap_ci(v, seed=0)
    assert lo < v.mean() < hi


def test_wilcoxon_detects_paired_difference():
    a = np.array([5.0, 6.0, 4.0, 7.0, 5.0, 6.0, 4.0, 7.0])
    w = paired_wilcoxon(a, a + 2.0)
    assert w["n"] == 8 and w["p"] < 0.05


def test_wilcoxon_underpowered_below_six_units():
    w = paired_wilcoxon(np.array([5.0, 6.0]), np.array([1.0, 2.0]))
    assert np.isnan(w["p"]) and "underpowered" in w["note"]


def test_summarize_units_full_block():
    per_arm = {
        "A": np.array([10.0, 12.0, 11.0, 9.0, 12.0, 10.0]),
        "F": np.array([8.0, 8.0, 9.0, 8.0, 9.0, 8.0]),
    }
    s = summarize_units(per_arm, reference_arm="F")
    assert s["F"]["mean"] < s["A"]["mean"]
    assert len(s["F"]["ci95"]) == 2
    assert "wilcoxon_vs_ref" in s["A"]
    assert "wilcoxon_vs_ref" not in s["F"]
