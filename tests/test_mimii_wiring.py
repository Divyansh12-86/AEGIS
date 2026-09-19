"""Tests for the MIMII real-data wiring: spectral front-end, fixed loader
layout (id_XX/{normal,abnormal}), auprc/auroc_bootstrap_ci, and (when the
13 GB fan/ tree is present) the real files themselves.
"""
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from egpm.data.loaders import MIMIILoader
from egpm.data.split_builder import SplitBuilder
from egpm.preprocessing.spectral import logmel_features, _mel_filterbank
from egpm.preprocessing.preprocessor import Preprocessor
from egpm.evaluation import auroc, auprc
from egpm.evaluation.stats import auroc_bootstrap_ci


# ---------------------------------------------------------------------------
# spectral front-end
# ---------------------------------------------------------------------------
def test_logmel_shapes_and_causality():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((16000, 2)) * 0.1
    F, ts, fps = logmel_features(x, 16000.0)
    assert F.shape == (59, 128)  # (16000-1024)//256 + 1 frames, 64 mel * 2 ch
    assert fps == 62.5
    assert (np.diff(ts) > 0).all()
    # causality: frame k unchanged when everything after its end is zeroed
    k = 10
    end = int(round(ts[k] * 16000.0))
    x3 = x.copy()
    x3[end:, :] = 0.0
    F3, _, _ = logmel_features(x3, 16000.0)
    np.testing.assert_allclose(F[k], F3[k], rtol=1e-12)
    # and not vacuous: changing frame k+1's samples changes frame k+1
    x4 = x.copy()
    x4[end:end + 256, :] = 0.0
    F4, _, _ = logmel_features(x4, 16000.0)
    assert not np.allclose(F[k + 1], F4[k + 1])


def test_mel_filterbank_no_dead_rows():
    H = _mel_filterbank(16000.0, 1024, 64)
    assert H.shape == (64, 513)
    assert (H >= 0).all()
    assert (H.sum(axis=1) > 0).all(), "every mel band must cover >= 1 FFT bin"


# ---------------------------------------------------------------------------
# loader on synthetic wavs in the download layout (fan/id_XX/{normal,abnormal})
# ---------------------------------------------------------------------------
def _write_fan_tree(tmp_path, n_norm=3, n_anom=2, fs=16000, frames=1600, ch=8, seed=0):
    rng = np.random.default_rng(seed)
    root = tmp_path / "fan"
    for id_ in ("id_00", "id_02"):
        for split, n in (("normal", n_norm), ("abnormal", n_anom)):
            d = root / id_ / split
            d.mkdir(parents=True)
            for i in range(n):
                x = rng.standard_normal((frames, ch)) * 0.05
                if split == "abnormal":
                    x[:, 0] += 0.5  # loud channel = injected fault stand-in
                sf.write(d / f"{i:08d}.wav", x, fs)
    return root


def test_mimii_loader_download_layout(tmp_path):
    _write_fan_tree(tmp_path)
    ds = MIMIILoader(machines=("fan",), feature="logmel",
                     max_clips_per_label=10).load(tmp_path / "fan")
    # 2 ids x (3 normal + 2 abnormal)
    assert len(ds) == 10
    ids = set(ds.unit_ids)
    # exactly one of the label tokens _normal_ / _abnormal_ per unit id
    # (note: 'normal' is a substring of 'abnormal' — compare delimited)
    assert all((("_normal_" in u) ^ ("_abnormal_" in u)) for u in ids)
    # no collisions: same stems exist in normal/ and abnormal/ dirs
    assert len(ids) == len(ds.records)
    r = next(r for r in ds.records if "_abnormal_" in r.unit_id)
    assert r.signals.ndim == 2
    assert set(np.unique(r.anomaly_labels)) == {1}
    r_n = next(r for r in ds.records if "_normal_" in r.unit_id)
    assert set(np.unique(r_n.anomaly_labels)) == {0}


def test_mimii_loader_label_from_directory_not_stem(tmp_path):
    """Regression: label must come from the abnormal/ DIRECTORY, never the stem."""
    _write_fan_tree(tmp_path, n_norm=2, n_anom=2)
    ds = MIMIILoader(machines=("fan",), feature="raw").load(tmp_path / "fan")
    for r in ds.records:
        expected = 1 if "_abnormal_" in r.unit_id else 0
        assert int(r.anomaly_labels[0]) == expected


def test_mimii_loader_multichannel_raw_not_mangled(tmp_path):
    """Regression: 8-channel wavs must load as [T, 8], not [T*8, 1]."""
    _write_fan_tree(tmp_path, n_norm=1, n_anom=1, frames=1600, ch=8)
    ds = MIMIILoader(machines=("fan",), feature="raw").load(tmp_path / "fan")
    for r in ds.records:
        assert r.signals.shape == (1600, 8)


def test_mimii_loader_max_clips_cap(tmp_path):
    _write_fan_tree(tmp_path, n_norm=5, n_anom=5)
    ds = MIMIILoader(machines=("fan",), feature="logmel",
                     max_clips_per_label=2).load(tmp_path / "fan")
    assert len(ds) == 8  # 2 ids x 2 labels x 2 clips


def test_mimii_loader_append_deltas(tmp_path):
    _write_fan_tree(tmp_path, n_norm=1, n_anom=1)
    base = MIMIILoader(machines=("fan",), feature="logmel").load(tmp_path / "fan")
    delt = MIMIILoader(machines=("fan",), feature="logmel",
                       append_deltas=True).load(tmp_path / "fan")
    r0, r1 = base.records[0], delt.records[0]
    assert r1.signals.shape[1] == 2 * r0.signals.shape[1]
    # first row's delta = 0 by construction (prepend first frame)
    assert np.allclose(r1.signals[0, r0.signals.shape[1]:], 0.0)
    np.testing.assert_allclose(r1.signals[:, : r0.signals.shape[1]], r0.signals)


def test_mimii_loader_db_layout_still_works(tmp_path):
    """Legacy archive layout (dB token in path) must keep loading."""
    rng = np.random.default_rng(3)
    root = tmp_path / "archive"
    for name, lab in (("normal_0dB_normal_000", 0), ("anomaly_0dB_id_00_000", 1)):
        d = root / "fan" / "0dB" / "normal"
        d.mkdir(parents=True, exist_ok=True)
        x = rng.standard_normal((1600, 1)) * 0.05
        sf.write(d / f"{name}.wav", x, 16000)
    ds = MIMIILoader(machines=("fan",), dbs=("0dB",), feature="raw").load(root)
    assert len(ds) == 2
    labs = sorted(int(r.anomaly_labels[0]) for r in ds.records)
    assert labs == [0, 1]


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def test_auprc_matches_known_values():
    # perfect separation
    assert auprc(np.array([.9, .8, .1, .2]), np.array([1, 1, 0, 0])) == 1.0
    # anti-separated: positives at the bottom -> step AP
    ap = auprc(np.array([.1, .2, .8, .9]), np.array([1, 1, 0, 0]))
    assert ap == pytest.approx((1 / 3 + 2 / 4) / 2)
    # both classes required
    with pytest.raises(ValueError):
        auprc(np.array([1.0, 2.0]), np.array([1, 1]))


def test_auprc_sklearn_equivalent_ties():
    """Verified against sklearn's average_precision_score incl. tie grouping."""
    sklearn = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(1)
    for _ in range(50):
        s = rng.integers(0, 4, 15).astype(float)  # heavy ties
        l = rng.integers(0, 2, 15)
        if (l == 1).sum() in (0, 15):
            continue
        assert abs(auprc(s, l) - sklearn.average_precision_score(l, s)) < 1e-12


def test_auroc_bootstrap_ci_reasonable():
    rng = np.random.default_rng(2)
    s = rng.random(200)
    l = (rng.random(200) < (s > 0.5)).astype(int)
    l[0], l[-1] = 1, 0  # guarantee both classes
    pt, lo, hi = auroc_bootstrap_ci(s, l, n_boot=200, seed=0)
    assert 0.0 <= lo <= pt <= hi <= 1.0
    noisy = s + rng.normal(0, 1.0, 200)
    pt2, lo2, hi2 = auroc_bootstrap_ci(noisy, l, n_boot=200, seed=0)
    assert lo2 < hi2  # nondegenerate


# ---------------------------------------------------------------------------
# real files (skipped when the 13 GB tree is absent)
# ---------------------------------------------------------------------------
_REPO = Path(__file__).resolve().parent.parent
_HAS_FAN = (_REPO / "fan").is_dir() and any((_REPO / "fan").rglob("*.wav"))


@pytest.fixture(scope="class")
def _real_fan_ds():
    return MIMIILoader(machines=("fan",), feature="logmel",
                       max_clips_per_label=4, append_deltas=True).load(_REPO / "fan")


@pytest.mark.skipif(not _HAS_FAN, reason="real MIMII fan/ wavs not present")
class TestMIMIIRealFiles:
    """Wired against the verified real download: 16 kHz, 8-ch, 10 s clips,
    fan/id_XX/{normal,abnormal} layout, colliding stems across labels."""

    def test_shapes_and_labels(self, _real_fan_ds):
        real_ds = _real_fan_ds
        assert len(real_ds) == 32  # 4 ids x 2 labels x 4 clips
        for r in real_ds.records:
            # mean-channel logmel + deltas: (64 mel mean + 64 std) * 2 = 256
            assert r.signals.shape == (622, 256)
            expected = 1 if "_abnormal_" in r.unit_id else 0
            assert int(r.anomaly_labels[0]) == expected

    def test_split_and_preprocess(self, _real_fan_ds):
        real_ds = _real_fan_ds
        sub = type(real_ds)()
        for r in real_ds.records:
            if "id_00" in r.unit_id:
                sub.add(r)
        split = SplitBuilder(sub, seed=0).build()
        assert all(SplitBuilder.audit(split, sub).values())
        stats = SplitBuilder(sub, seed=0).fit_normalization(split)
        seq = Preprocessor(window_length=16, fs_sync=1.0).process(
            sub[split.train_unit_ids[0]], norm_stats=stats
        )
        assert seq.X.shape[1:] == (16, 256)
        assert seq.X.shape[0] == 622 // 16
