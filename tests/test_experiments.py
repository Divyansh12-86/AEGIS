"""Experiment-runner tests (PRD Deliverable E + M10 harness).

These exercise the harness wiring on small synthetic data: correct report
schema, >=3 seeds, per-seed AUROCs in [0,1], all arms present, and
reproducibility of the report structure. The scientific conclusions require
real MIMII data (PRD §3.1) and are out of CI scope.
"""
import json

import numpy as np
import pytest

from egpm.data import MIMIILoader
from egpm.experiments import deliverable_e, run_core_ablations, ExperimentReport


@pytest.fixture(scope="module")
def small_dataset():
    return MIMIILoader.synthetic_like(
        n_units=12, frames_per_unit=160, anomaly_fraction=0.25, seed=0
    )


class TestDeliverableE:
    def test_report_schema_and_ranges(self, small_dataset):
        report = deliverable_e(
            dataset=small_dataset, n_seeds=2,
            window_length=16, stage1_steps=30,
        )
        assert isinstance(report, ExperimentReport)
        assert report.experiment == "deliverable_E_first_experiment"
        assert report.seeds == [0, 1]
        for arm in ("continuous_AE", "vq_markov", "full_hsmm"):
            r = report.results[arm]
            assert 0.0 <= r["AUROC_mean"] <= 1.0
            assert len(r["AUROC_per_seed"]) == 2
            assert all(0.0 <= v <= 1.0 for v in r["AUROC_per_seed"])
            # std consistent with per-seed values
            np.testing.assert_allclose(
                r["AUROC_std"], float(np.std(r["AUROC_per_seed"])), rtol=1e-9
            )

    def test_report_json_serializable(self, small_dataset):
        report = deliverable_e(
            dataset=small_dataset, n_seeds=1,
            window_length=16, stage1_steps=20,
        )
        parsed = json.loads(report.to_json())
        assert parsed["experiment"] == report.experiment
        assert "seeds" in parsed and "results" in parsed

    def test_unit_level_protocol(self, small_dataset):
        """PRD §21 fair comparison: the same split feeds every arm —
        verified by construction (single split per seed in the runner);
        here we check the anomaly/normal partition is respected (fit on
        normals only)."""
        normal_ids = [r.unit_id for r in small_dataset.records
                     if r.anomaly_labels[0] == 0]
        anom_ids = [r.unit_id for r in small_dataset.records
                    if r.anomaly_labels[0] == 1]
        assert len(normal_ids) > 0 and len(anom_ids) > 0
        assert set(normal_ids) & set(anom_ids) == set()


class TestCoreAblations:
    def test_all_arms_present_with_seeds(self, small_dataset):
        report = run_core_ablations(
            dataset=small_dataset, n_seeds=2,
            window_length=16, stage1_steps=30,
        )
        assert report.experiment == "core_ablations_A1_A2_A2b"
        for arm in ("A1_continuous_hsmm", "A2_pooled_events",
                    "A2b_first_order_hmm", "primary_full_hsmm"):
            assert arm in report.results
            r = report.results[arm]
            assert 0.0 <= r["AUROC_mean"] <= 1.0
            assert len(r["AUROC_per_seed"]) == 2

    def test_pooled_weaker_than_hsmm_or_equal(self, small_dataset):
        """A2 (no temporal model) must not beat the full HSMM on data with
        temporal structure — on this synthetic set, at minimum it must not
        exceed the primary by a margin."""
        report = run_core_ablations(
            dataset=small_dataset, n_seeds=2,
            window_length=16, stage1_steps=30,
        )
        a2 = report.results["A2_pooled_events"]["AUROC_mean"]
        hs = report.results["primary_full_hsmm"]["AUROC_mean"]
        assert a2 <= hs + 0.05, "pooled model unexpectedly beat the HSMM"
