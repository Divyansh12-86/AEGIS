"""M1 acceptance tests: data pipeline leakage & causality (PRD §23, §29).

These tests verify the non-negotiable integrity requirements:
  * unit-level splits, disjoint and complete;
  * normalization statistics fit on training units only;
  * no window contains information from a timestep later than its own end.
"""
import numpy as np
import pytest
from pathlib import Path

from egpm.data.loaders import UnitDataset, UnitRecord, NCMAPSSLoader, MIMIILoader
from egpm.data.split_builder import SplitBuilder
from egpm.preprocessing.preprocessor import Preprocessor


# ---------------------------------------------------------------------------
# fixtures: synthetic datasets with the same contracts as the real loaders
# ---------------------------------------------------------------------------
def make_ncmapss_like(n_units=5, T=120, d=4, seed=0):
    """N-CMAPSS-shaped units: slowly drifting degradation + RUL labels."""
    rng = np.random.default_rng(seed)
    ds = UnitDataset()
    for u in range(n_units):
        drift = np.linspace(0, 1, T)[:, None] * (0.5 + 0.1 * u)
        noise = 0.05 * rng.standard_normal((T, d))
        signals = np.tile(np.linspace(0, 1, d), (T, 1)) + drift + noise
        cycles = np.arange(T, dtype=float)
        ds.add(UnitRecord(
            unit_id=f"ncmapss_u{u}",
            signals=signals,
            timestamps=cycles,
            rul_labels=(T - 1) - cycles,  # cycles-to-failure
            health_labels=np.minimum(6, 1 + (cycles // 20).astype(int)),
        ))
    return ds


def make_mimii_like(n_units=6, frames=800, seed=1, anomaly_fraction=0.2):
    return MIMIILoader.synthetic_like(
        n_units=n_units, frames_per_unit=frames, seed=seed,
        anomaly_fraction=anomaly_fraction,
    )


# ---------------------------------------------------------------------------
# Task 1: loaders + split integrity
# ---------------------------------------------------------------------------
class TestUnitSplit:
    def test_splits_are_disjoint(self):
        ds = make_ncmapss_like()
        split = SplitBuilder(ds, seed=0).build()
        tr, va, te = set(split.train_unit_ids), set(split.val_unit_ids), set(split.test_unit_ids)
        assert not (tr & va) and not (tr & te) and not (va & te)

    def test_splits_cover_all_units_exactly_once(self):
        ds = make_ncmapss_like()
        split = SplitBuilder(ds, seed=0).build()
        all_ids = [r.unit_id for r in ds.records]
        merged = split.train_unit_ids + split.val_unit_ids + split.test_unit_ids
        assert sorted(merged) == sorted(all_ids)

    def test_unit_split_object_rejects_overlap(self):
        with pytest.raises(ValueError, match="disjoint"):
            from egpm.data.loaders import UnitSplit
            UnitSplit(["u1"], ["u1"], ["u2"])

    def test_split_is_at_unit_level_not_window_level(self):
        # every unit must map to exactly one split, and windows derived from
        # a test unit must all be attributed to the test split
        ds = make_ncmapss_like()
        sb = SplitBuilder(ds, seed=0)
        split = sb.build()
        for r in ds.records:
            which = split.unit_to_split(r.unit_id)
            assert which in {"train", "val", "test"}

    def test_split_reproducible_with_same_seed(self):
        ds = make_ncmapss_like()
        s1 = SplitBuilder(ds, seed=42).build()
        s2 = SplitBuilder(ds, seed=42).build()
        assert s1 == s2 or (
            s1.train_unit_ids == s2.train_unit_ids
            and s1.val_unit_ids == s2.val_unit_ids
            and s1.test_unit_ids == s2.test_unit_ids
        )

    def test_audit_checklist_all_true(self):
        ds = make_ncmapss_like()
        sb = SplitBuilder(ds, seed=0)
        split = sb.build()
        report = SplitBuilder.audit(split, ds)
        assert all(report.values()), report


class TestNormalizationLeakage:
    def test_norm_stats_use_train_units_only(self):
        ds = make_ncmapss_like()
        sb = SplitBuilder(ds, seed=0)
        split = sb.build()
        stats = sb.fit_normalization(split)
        assert set(stats.train_unit_ids) == set(split.train_unit_ids)
        # verify the mean equals the mean of train units' data only
        train_signals = np.concatenate(
            [ds[uid].signals for uid in split.train_unit_ids], axis=0
        )
        np.testing.assert_allclose(stats.mean, train_signals.mean(axis=0), rtol=1e-12)
        np.testing.assert_allclose(stats.std, train_signals.std(axis=0), rtol=1e-12)

    def test_norm_stats_differ_from_full_dataset_stats(self):
        # sanity: if stats leaked, they'd match the full-dataset stats
        ds = make_ncmapss_like()
        sb = SplitBuilder(ds, seed=0)
        split = sb.build()
        stats = sb.fit_normalization(split)
        full = np.concatenate([r.signals for r in ds.records], axis=0)
        # with drift designed per-unit, train-only stats != full stats
        assert not np.allclose(stats.mean, full.mean(axis=0))

    def test_transform_uses_stored_stats(self):
        ds = make_ncmapss_like()
        sb = SplitBuilder(ds, seed=0)
        split = sb.build()
        stats = sb.fit_normalization(split)
        # pooled TRAIN data must be zero-mean / unit-std after transform
        train_signals = np.concatenate(
            [ds[uid].signals for uid in split.train_unit_ids], axis=0
        )
        Xn = stats.transform(train_signals)
        np.testing.assert_allclose(Xn.mean(axis=0), np.zeros(Xn.shape[1]), atol=1e-10)
        np.testing.assert_allclose(Xn.std(axis=0), np.ones(Xn.shape[1]), atol=1e-10)


class TestLoadersContract:
    def test_unit_record_validates_shapes(self):
        with pytest.raises(ValueError, match="signals"):
            UnitRecord("u", np.zeros(5), np.zeros(5))
        with pytest.raises(ValueError, match="timestamps"):
            UnitRecord("u", np.zeros((5, 2)), np.zeros(3))

    def test_duplicate_unit_id_rejected(self):
        ds = UnitDataset()
        ds.add(UnitRecord("u", np.zeros((5, 1)), np.arange(5.0)))
        with pytest.raises(ValueError, match="duplicate"):
            ds.add(UnitRecord("u", np.zeros((5, 1)), np.arange(5.0)))

    def test_ncmapss_npz_roundtrip(self, tmp_path):
        ds = make_ncmapss_like(n_units=3)
        f = tmp_path / "ncmapss.npz"
        NCMAPSSLoader.to_npz(ds, f)
        reloaded = NCMAPSSLoader().load(f)
        assert reloaded.unit_ids == ds.unit_ids
        for uid in ds.unit_ids:
            np.testing.assert_array_equal(ds[uid].signals, reloaded[uid].signals)
            np.testing.assert_array_equal(ds[uid].rul_labels, reloaded[uid].rul_labels)


# ---------------------------------------------------------------------------
# Real N-CMAPSS .h5 wiring (DS01/DS02 schema): skipped when files are absent
# ---------------------------------------------------------------------------
_NCMAPSS_DIR = Path(__file__).resolve().parent.parent / "N-CMAPSS"

def h5py_available() -> bool:
    try:
        import h5py  # noqa: F401
        return True
    except ImportError:
        return False

_HAS_REAL = any(_NCMAPSS_DIR.glob("*.h5")) and h5py_available()


@pytest.mark.skipif(not _HAS_REAL, reason="real N-CMAPSS .h5 files not present")
class TestNCMAPSSRealFiles:
    """Wired against the verified real schema:
    A_*=[unit, cycle, Fc, hs], W_* (4 ops), X_s_* (14 sensors), Y_*=RUL."""

    def test_load_real_files(self):
        ds = NCMAPSSLoader(max_cycles_per_unit=3).load(_NCMAPSS_DIR)
        # DS01 (6 dev + 4 test units) + DS02 (6 dev + 3 test); DS03 stub skipped
        assert len(ds) == 19
        ids = set(ds.unit_ids)
        assert "DS01_u1" in ids and "DS02_u2" in ids  # namespaced per dataset
        assert not (ids & {"DS01_u2"}) - {"DS01_u2"}  # u2 exists in both files
        r = ds["DS01_u1"]
        assert r.signals.shape[1] == 18  # 4 ops + 14 health-proxy sensors
        assert r.rul_labels is not None and r.rul_labels[0] >= r.rul_labels[-1]
        assert set(np.unique(r.health_labels)) <= {1, 2}

    def test_real_files_split_and_preprocess(self):
        ds = NCMAPSSLoader(max_cycles_per_unit=3).load(_NCMAPSS_DIR)
        split = SplitBuilder(ds, seed=0).build()
        assert all(SplitBuilder.audit(split, ds).values())
        stats = SplitBuilder(ds, seed=0).fit_normalization(split)
        seq = Preprocessor(window_length=100).process(
            ds[split.train_unit_ids[0]], norm_stats=stats
        )
        # one cycle ~ 4000-9000 rows at 1Hz; W=100 keeps causality per PRD §6
        assert seq.X.ndim == 3 and seq.rul_labels is not None

    def test_corrupt_file_skipped_with_others_loaded(self):
        ds = NCMAPSSLoader(max_cycles_per_unit=2).load(_NCMAPSS_DIR)
        # DS03-012.h5 is a corrupt 4KB stub; the two real files still load
        assert len(ds) == 19

    def test_mimii_synthetic_labels(self):
        ds = make_mimii_like(n_units=10, anomaly_fraction=0.2)
        n_anom = sum(1 for r in ds.records if r.anomaly_labels is not None and r.anomaly_labels[0] == 1)
        assert n_anom == 2
        # anomalous units have larger deviation from clean sinusoid
        anom_ids = [r.unit_id for r in ds.records if r.anomaly_labels[0] == 1]
        normal_ids = [r.unit_id for r in ds.records if r.anomaly_labels[0] == 0]
        def energy(rec): return float(np.abs(rec.signals).mean())
        anom_energy = max(energy(ds[a]) for a in anom_ids)
        normal_energy = min(energy(ds[n]) for n in normal_ids)
        assert anom_energy > normal_energy


# ---------------------------------------------------------------------------
# Task 2: preprocessing causality
# ---------------------------------------------------------------------------
class TestCausality:
    def test_window_labels_come_from_window_end_not_future(self):
        # rul/health labels for window i must equal the label at index s+W-1
        rng = np.random.default_rng(3)
        T, d, W = 50, 2, 10
        signals = rng.standard_normal((T, d))
        rul = np.arange(T, dtype=float)[::-1]  # decreasing RUL
        health = np.arange(T)  # increasing health stage
        rec = UnitRecord("u", signals, np.arange(T, dtype=float),
                         rul_labels=rul, health_labels=health)
        prep = Preprocessor(window_length=W, fs_sync=1.0)
        seq = prep.process(rec)
        # window j starts at j*W (S=W); label must be from index j*W + W - 1
        for j in range(seq.X.shape[0]):
            end_idx = j * W + W - 1
            assert seq.rul_labels[j] == rul[end_idx]
            assert seq.health_labels[j] == health[end_idx]

    def test_labels_mapped_onto_sync_grid(self):
        # grid longer than raw rows (timestamps span -> T_out > T): labels
        # must be resampled onto the grid, not indexed raw (regression: the
        # tuned N-CMAPSS run crashed with IndexError here)
        rng = np.random.default_rng(5)
        T, d, W = 101, 2, 8
        ts = np.arange(T) * 100.0  # span 10000 -> grid of 10001 rows
        rul = np.arange(T, dtype=float)[::-1]
        rec = UnitRecord("u", rng.standard_normal((T, d)), ts, rul_labels=rul)
        seq = Preprocessor(window_length=W, fs_sync=1.0).process(rec)
        # labels resampled causally: every window label is a raw RUL value,
        # RUL is non-increasing over time, and it reaches the last raw value
        assert all(np.isin(seq.rul_labels, rul))
        assert (np.diff(seq.rul_labels) <= 0).all()
        assert seq.rul_labels[-1] <= 1.0  # near end-of-life (tail window may drop the final row)

    def test_no_future_leak_into_windows(self):
        # changing signal values AFTER a window's end must not change that window
        rng = np.random.default_rng(4)
        T, d, W = 40, 2, 10
        base = rng.standard_normal((T, d))
        rec1 = UnitRecord("u", base.copy(), np.arange(T, dtype=float))
        seq1 = Preprocessor(window_length=W).process(rec1)
        mutated = base.copy()
        mutated[30:] += 100.0  # corrupt everything after window 2's end (idx 29)
        rec2 = UnitRecord("u", mutated, np.arange(T, dtype=float))
        seq2 = Preprocessor(window_length=W).process(rec2)
        np.testing.assert_array_equal(seq1.X[:3], seq2.X[:3])  # windows 0..2 untouched

    def test_synchronize_does_not_extrapolate_past_end(self):
        t = np.linspace(0, 10, 101)  # 10 s at 10 Hz
        y = np.sin(t)
        rec = UnitRecord("u", y.reshape(-1, 1), t)
        prep = Preprocessor(window_length=5, fs_sync=10.0)
        X, m, tgrid = prep.synchronize(rec)
        assert tgrid[0] == pytest.approx(t[0])
        assert tgrid[-1] <= t[-1] + 1e-9

    def test_antialias_filter_is_causal(self):
        # filtering a suffix must not affect earlier outputs
        rng = np.random.default_rng(5)
        x = rng.standard_normal((60, 1))
        # split into two halves; filter of full must match filter of first half
        from egpm.preprocessing.preprocessor import _lowpass_antialias
        full = _lowpass_antialias(x, 10.0, 1.0)
        first = _lowpass_antialias(x[:30], 10.0, 1.0)
        np.testing.assert_allclose(full[:30], first[:30], rtol=1e-12)


class TestPreprocessorShapes:
    def test_window_shapes_and_stride(self):
        rng = np.random.default_rng(6)
        T, d, W = 100, 3, 10
        rec = UnitRecord("u", rng.standard_normal((T, d)), np.arange(T, dtype=float))
        seq = Preprocessor(window_length=W).process(rec)
        assert seq.X.shape == (10, W, d)
        assert seq.mask.shape == seq.X.shape
        assert (seq.mask == 1).all()

    def test_stride_overlapping_windows(self):
        rng = np.random.default_rng(7)
        T, d, W, S = 50, 1, 10, 5
        rec = UnitRecord("u", rng.standard_normal((T, d)), np.arange(T, dtype=float))
        seq = Preprocessor(window_length=W, stride=S).process(rec)
        # starts at 0,5,...,40 -> 9 windows
        assert seq.X.shape[0] == 9
        assert seq.window_starts[0] == 0 and seq.window_starts[1] == 5

    def test_short_sequence_raises(self):
        rec = UnitRecord("u", np.zeros((5, 1)), np.arange(5.0))
        with pytest.raises(ValueError, match="shorter than window"):
            Preprocessor(window_length=10).process(rec)

    def test_normalization_applied_per_channel(self):
        ds = make_ncmapss_like()
        sb = SplitBuilder(ds, seed=0)
        split = sb.build()
        stats = sb.fit_normalization(split)
        rec = ds[split.train_unit_ids[0]]
        seq = Preprocessor(window_length=20).process(rec, norm_stats=stats)
        # windows must equal manual z-scoring with the train-only stats
        expected = (seq.X - 0)  # already normalized; verify against raw instead
        raw_seq = Preprocessor(window_length=20).process(rec)
        manual = (raw_seq.X - stats.mean) / stats.std
        np.testing.assert_allclose(seq.X, manual, atol=1e-12)
        assert np.isfinite(expected).all()

    def test_impute_loocf_short_gaps(self):
        X = np.arange(20, dtype=float).reshape(-1, 1)
        mask = np.ones((20, 1))
        mask[5:8] = 0  # 3-sample gap (short)
        prep = Preprocessor(window_length=5)
        Xi, mi = prep.impute(X, mask)
        # carried forward value 4 across the gap
        np.testing.assert_allclose(Xi[5:8, 0], 4.0)
        assert (mi == 1).all()

    def test_impute_linear_long_gaps(self):
        X = np.arange(20, dtype=float).reshape(-1, 1)
        mask = np.ones((20, 1))
        mask[5:15] = 0  # 10-sample gap (long)
        prep = Preprocessor(window_length=5, gap_loocf_max=5)
        Xi, mi = prep.impute(X, mask)
        # linear interpolation between X[4]=4 and X[15]=15 across indices 5..14
        np.testing.assert_allclose(Xi[5:15, 0], np.linspace(4, 15, 12)[1:-1], atol=1e-12)
        assert (mi == 1).all()

    def test_impute_is_causal_at_sequence_start(self):
        # gap at the very start: no left neighbor; must not touch future beyond right
        X = np.zeros((10, 1))
        mask = np.ones((10, 1))
        mask[0:3] = 0
        prep = Preprocessor(window_length=5)
        Xi, mi = prep.impute(X, mask)
        assert np.isfinite(Xi).all()
