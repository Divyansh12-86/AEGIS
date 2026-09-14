"""Heads-run harness contract: runs on synthetic hs-labeled data, produces
schema-valid report with unit CIs."""
import numpy as np
import pytest

from egpm.data import UnitDataset, UnitRecord
from egpm.experiments.heads_run import run_heads_experiment


def _hs_dataset(seed=0):
    """Small synthetic N-CMAPSS-shaped dataset: hs flips 1->2 mid-run."""
    rng = np.random.default_rng(seed)
    ds = UnitDataset()
    for i in range(10):  # 10 units -> 6/2/2 split -> >=2 test units for CIs
        T = 120
        flip = T // 2
        x = np.linspace(0, 1, T)
        signals = np.stack([x, 1 - x], axis=1) + 0.01 * rng.standard_normal((T, 2))
        health = np.where(np.arange(T) < flip, 1, 2)
        ds.add(UnitRecord(
            unit_id=f"u{i}", signals=signals,
            timestamps=np.arange(T, dtype=float),
            health_labels=health.astype(np.int64),
        ))
    return ds


@pytest.fixture(scope="module")
def report():
    return run_heads_experiment(
        _hs_dataset(), n_seeds=1, window_length=4,
        stage1_steps=30, em_max_iter=4, em_restarts=1,
        verbose=False,
    )


class TestHeadsRun:
    def test_schema_and_ranges(self, report):
        assert report["experiment"] == "fault_health_heads_m7"
        v = report["results"]["fault_accuracy"]
        assert 0.0 <= v["mean"] <= 1.0
        assert v["unit_ci95"][0] <= v["unit_ci95"][1]
        v = report["results"]["health_spearman_vs_hs"]
        assert -1.0 <= v["mean"] <= 1.0  # Spearman lives in [-1, 1]

    def test_perfect_labels_perfect_fault(self, report):
        # synthetic hs is perfectly separable from the ramp -> accuracy 1.0
        assert report["results"]["fault_accuracy"]["mean"] == 1.0
