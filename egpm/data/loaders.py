"""Unit-level dataset contracts and loaders (PRD §27 `data/`, M1).

Every sequence is owned by a *unit* (one asset run). Splits are made at the
unit level BEFORE any windowing; normalization statistics are computed only
from training units (PRD §6).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np


@dataclass
class UnitRecord:
    """One asset run: a multivariate sensor sequence plus optional labels.

    Attributes:
        unit_id: stable identifier of the unit (asset run).
        signals: array [T, d] — raw (unsynchronized, unnormalized) channels,
            rows ordered by native time.
        timestamps: array [T] — per-row native timestamps (seconds).
        health_labels: optional [T] integer auxiliary health-state labels
            (e.g. N-CMAPSS health states 1..6). May be None.
        rul_labels: optional [T] float remaining-useful-life labels
            (cycles-to-failure). May be None.
        fault_label: optional integer fault-class label for the run.
        anomaly_labels: optional [T] binary {0,1} anomaly labels
            (e.g. MIMII sound files marked anomalous).
    """

    unit_id: str
    signals: np.ndarray
    timestamps: np.ndarray
    health_labels: Optional[np.ndarray] = None
    rul_labels: Optional[np.ndarray] = None
    fault_label: Optional[int] = None
    anomaly_labels: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if self.signals.ndim != 2:
            raise ValueError(f"signals must be 2-D [T, d]; got {self.signals.shape}")
        T = self.signals.shape[0]
        if self.timestamps.shape != (T,):
            raise ValueError(
                f"timestamps must have shape [{T},]; got {self.timestamps.shape}"
            )
        for name, arr in (
            ("health_labels", self.health_labels),
            ("rul_labels", self.rul_labels),
            ("anomaly_labels", self.anomaly_labels),
        ):
            if arr is not None and arr.shape != (T,):
                raise ValueError(f"{name} must have shape [{T},]; got {arr.shape}")


@dataclass
class UnitDataset:
    """Collection of unit records with unit-level split support (PRD §28)."""

    records: List[UnitRecord] = field(default_factory=list)

    # -- basic accessors -----------------------------------------------------
    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, unit_id: str) -> UnitRecord:
        by_id = {r.unit_id: r for r in self.records}
        if unit_id not in by_id:
            raise KeyError(f"unknown unit_id: {unit_id}")
        return by_id[unit_id]

    @property
    def unit_ids(self) -> List[str]:
        return [r.unit_id for r in self.records]

    def add(self, record: UnitRecord) -> None:
        if any(r.unit_id == record.unit_id for r in self.records):
            raise ValueError(f"duplicate unit_id: {record.unit_id}")
        self.records.append(record)

    # -- unit-level splitting (PRD §6: split at unit level, pre-windowing) ---
    def split(
        self,
        train_frac: float = 0.6,
        val_frac: float = 0.2,
        seed: int = 0,
    ) -> "UnitSplit":
        """Randomly partition *units* (not windows) into train/val/test.

        The remaining fraction (1 - train - val) becomes test. Test units are
        never touched except once, for final reporting (PRD §23).
        """
        if not (0.0 < train_frac < 1.0) or not (0.0 <= val_frac < 1.0):
            raise ValueError("train/val fractions must be in (0, 1) and [0, 1)")
        if train_frac + val_frac >= 1.0:
            raise ValueError("train_frac + val_frac must be < 1.0")
        rng = np.random.default_rng(seed)
        ids = np.array(self.unit_ids, dtype=object)
        rng.shuffle(ids)
        n = len(ids)
        n_train = max(1, int(round(train_frac * n)))
        n_val = max(0, int(round(val_frac * n))) if val_frac > 0 else 0
        n_train = min(n_train, n - n_val if n_val else n)
        train_ids = sorted(ids[:n_train].tolist())
        val_ids = sorted(ids[n_train : n_train + n_val].tolist())
        test_ids = sorted(ids[n_train + n_val :].tolist())
        return UnitSplit(
            train_unit_ids=train_ids,
            val_unit_ids=val_ids,
            test_unit_ids=test_ids,
        )


@dataclass
class UnitSplit:
    """Unit-level partition of a dataset (guaranteed disjoint, PRD §23)."""

    train_unit_ids: List[str]
    val_unit_ids: List[str]
    test_unit_ids: List[str]

    def __post_init__(self) -> None:
        sets = [set(self.train_unit_ids), set(self.val_unit_ids), set(self.test_unit_ids)]
        if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
            raise ValueError("unit splits must be disjoint (no unit in two splits)")

    def unit_to_split(self, unit_id: str) -> str:
        if unit_id in self.train_unit_ids:
            return "train"
        if unit_id in self.val_unit_ids:
            return "val"
        if unit_id in self.test_unit_ids:
            return "test"
        raise KeyError(f"unit {unit_id} not assigned to any split")


class NCMAPSSLoader:
    """Loader for N-CMAPSS (NASA Modular Aero-Propulsion System Simulation).

    The real distribution ships ``.h5`` files (also readable when renamed
    ``.mat`` — MATLAB v7.3/HDF5) with the schema verified against
    ``N-CMAPSS DS01 005.h5`` / ``DS02 006.h5``:

    * ``A_dev``/``A_test`` — [T, 4]: unit, cycle, Fc, hs (health state)
    * ``W_dev``/``W_test`` — [T, 4]: operational settings (alt, Mach, TRA, T2)
    * ``X_s_dev``/``X_s_test`` — [T, 14]: health-proxy sensors
    * ``Y_dev``/``Y_test`` — [T, 1]: ground-truth RUL in cycles (0..N-1,
      constant within a cycle, verified equal to max_cycle - cycle)

    Multiple units live in one file (contiguous row blocks, unit ids disjoint
    across dev/test); the same unit number appears in different dataset files,
    so unit_ids are namespaced by dataset (e.g. ``DS01_u2`` vs ``DS02_u2``).
    ``hs`` is stored 1-based as health_labels. A pre-extracted ``.npz``
    archive (see :meth:`NCMAPSSLoader.to_npz`) is also accepted.
    """

    def __init__(self, max_cycles_per_unit: Optional[int] = None):
        self.max_cycles_per_unit = max_cycles_per_unit

    # -- public API ----------------------------------------------------------
    def load(self, path: str | Path) -> UnitDataset:
        """Load all units under ``path`` (directory of .h5/.mat/.npz, or single file).

        Unreadable files in a directory (corrupt/partial downloads) are
        skipped with a warning; at least one unit must load overall.
        """
        path = Path(path)
        if path.is_dir():
            files = sorted(
                p for p in path.iterdir() if p.suffix in {".h5", ".mat", ".npz"}
            )
            if not files:
                raise FileNotFoundError(f"no .h5/.mat/.npz files under {path}")
            dataset = UnitDataset()
            for f in files:
                try:
                    self._load_one_into(f, dataset)
                except OSError as e:  # corrupt file: skip, keep loading the rest
                    print(f"[NCMAPSSLoader] skipping unreadable {f.name}: {e}")
            if len(dataset) == 0:
                raise ValueError(f"no readable dataset files under {path}")
            return dataset
        if path.is_file():
            dataset = UnitDataset()
            self._load_one_into(path, dataset)
            return dataset
        raise FileNotFoundError(str(path))

    # -- internals -----------------------------------------------------------
    def _load_one_into(self, file: Path, dataset: UnitDataset) -> None:
        if file.suffix == ".npz":
            self._load_npz(file, dataset)
        else:
            self._load_h5(file, dataset)

    @staticmethod
    def _dataset_prefix(file: Path) -> str:
        """Namespace units by dataset file (DS01_u2 vs DS02_u2)."""
        m = re.search(r"(DS\d+[a-z]?)", file.stem, re.IGNORECASE)
        if m:
            return m.group(1).upper()
        return re.sub(r"[^A-Za-z0-9]+", "_", file.stem).strip("_") or "ds"

    def _load_h5(self, file: Path, dataset: UnitDataset) -> None:
        try:
            import h5py  # N-CMAPSS files are MATLAB v7.3 (HDF5)
        except ImportError as e:  # pragma: no cover - optional dep
            raise ImportError(
                "h5py is required to read N-CMAPSS .h5/.mat files: pip install h5py"
            ) from e
        prefix = self._dataset_prefix(file)
        with h5py.File(file, "r") as f:
            for part in ("dev", "test"):
                A = np.array(f[f"A_{part}"])  # [T, 4]: unit, cycle, Fc, hs
                Y = np.array(f[f"Y_{part}"]).ravel()  # [T]: ground-truth RUL
                W = np.array(f[f"W_{part}"])  # [T, 4]
                if f"X_s_{part}" in f:
                    Xs = np.array(f[f"X_s_{part}"])  # [T, n_sensors]
                    signals_all = np.concatenate([W, Xs], axis=1)
                else:
                    signals_all = W
                for unit in np.unique(A[:, 0]):
                    rows = A[:, 0] == unit  # units are contiguous row blocks
                    signals = signals_all[rows].astype(np.float64)
                    rul = Y[rows].astype(np.float64)
                    hs = A[rows, 3].astype(np.int64) + 1  # 1-based health state
                    if self.max_cycles_per_unit is not None:
                        cycles = A[rows, 1]
                        unique_cycles = np.unique(cycles)
                        cutoff = unique_cycles[
                            min(self.max_cycles_per_unit, len(unique_cycles)) - 1
                        ]
                        keep = cycles <= cutoff
                        signals, rul, hs = signals[keep], rul[keep], hs[keep]
                    dataset.add(
                        UnitRecord(
                            unit_id=f"{prefix}_u{int(unit)}",
                            signals=signals,
                            timestamps=np.arange(len(signals), dtype=np.float64),
                            health_labels=hs,
                            rul_labels=rul,
                        )
                    )

    def _load_npz(self, file: Path, dataset: UnitDataset) -> None:
        with np.load(file, allow_pickle=False) as z:
            unit_ids = sorted(
                k.split("/", 1)[1]
                for k in z.files
                if k.startswith("signals/")
            )
            if not unit_ids:
                raise ValueError(f"{file.name}: npz missing 'signals/<unit_id>' arrays")
            for uid in unit_ids:
                health = z[f"health/{uid}"] if f"health/{uid}" in z.files else None
                rul = z[f"rul/{uid}"] if f"rul/{uid}" in z.files else None
                fault = z[f"fault/{uid}"] if f"fault/{uid}" in z.files else None
                anomaly = z[f"anomaly/{uid}"] if f"anomaly/{uid}" in z.files else None
                dataset.add(
                    UnitRecord(
                        unit_id=uid,
                        signals=z[f"signals/{uid}"],
                        timestamps=z[f"timestamps/{uid}"],
                        health_labels=health,
                        rul_labels=rul,
                        fault_label=int(fault) if fault is not None else None,
                        anomaly_labels=anomaly,
                    )
                )

    # -- export for CI / tests -------------------------------------------------
    @staticmethod
    def to_npz(dataset: UnitDataset, out_file: str | Path) -> None:
        """Serialize a UnitDataset to .npz so downstream stages don't need .mat."""
        out: Dict[str, np.ndarray] = {}
        for r in dataset.records:
            out[f"signals/{r.unit_id}"] = r.signals
            out[f"timestamps/{r.unit_id}"] = r.timestamps
            if r.health_labels is not None:
                out[f"health/{r.unit_id}"] = r.health_labels
            if r.rul_labels is not None:
                out[f"rul/{r.unit_id}"] = r.rul_labels
            if r.fault_label is not None:
                out[f"fault/{r.unit_id}"] = np.array(r.fault_label, dtype=np.int64)
            if r.anomaly_labels is not None:
                out[f"anomaly/{r.unit_id}"] = r.anomaly_labels
        np.savez_compressed(out_file, **out)


class MIMIILoader:
    """Loader for MIMII (machine sound anomaly detection).

    Supports two on-disk layouts:

    * **Download layout** (verified on disk): ``<machine>/id_XX/{normal,
      abnormal}/NNNNNNNN.wav`` — 16 kHz, 8-channel, 10 s clips. The
      normal/anomalous label comes from the *directory* name; unit ids are
      namespaced by machine/id/split/stem (normal and abnormal share stems).
    * **Archive layout** (original MIMII distribution): machine and dB level
      appear in the path (``normal_6dB/...`` or a ``0dB``/``6dB``/``min6dB``
      token); label from the ``anomaly`` token. Kept for backwards
      compatibility with the synthetic tests.

    With ``feature="logmel"`` (default), clips are converted to causal
    log-mel frames (``egpm.preprocessing.spectral``): ``signals`` = [n_frames,
    n_mels * n_channels], ``timestamps`` = frame-end times. With
    ``feature="raw"``, signals are the raw samples [n_samples, n_channels].
    """

    def __init__(
        self,
        machines: Sequence[str] = ("pump", "fan", "valve", "slider"),
        dbs: Sequence[str] = ("0dB", "6dB", "min6dB"),
        fs_sync: float = 16000.0,
        clip_seconds: float = 10.0,
        feature: str = "raw",
        n_fft: int = 1024,
        hop: int = 256,
        n_mels: int = 64,
        max_clips_per_label: Optional[int] = None,
        append_deltas: bool = False,
        machine_ids: Optional[Sequence[str]] = None,
        channel_mode: str = "mean",
    ):
        if feature not in ("raw", "logmel"):
            raise ValueError(f"feature must be 'raw' or 'logmel'; got {feature!r}")
        self.machines = list(machines)
        self.dbs = list(dbs)
        self.fs_sync = fs_sync
        self.clip_seconds = clip_seconds
        self.feature = feature
        self.n_fft = n_fft
        self.hop = hop
        self.n_mels = n_mels
        self.max_clips_per_label = max_clips_per_label
        self.append_deltas = append_deltas
        self.machine_ids = list(machine_ids) if machine_ids is not None else None
        self.channel_mode = channel_mode

    # -- public API ----------------------------------------------------------
    def load(self, path: str | Path) -> UnitDataset:
        """Load all wav files under ``path`` matching configured machine/ids/dB.

        ``max_clips_per_label`` caps clips per (unit-group, label) — e.g. a
        smoke run pays for 40 clips instead of 5500.
        """
        try:
            import soundfile as sf  # lazy: optional dependency
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "soundfile is required to read MIMII wav files: pip install soundfile"
            ) from e
        path = Path(path)
        if not path.is_dir():
            raise FileNotFoundError(str(path))
        dataset = UnitDataset()
        counts: Dict[str, int] = {}
        root_name = path.name
        for wav in sorted(path.rglob("*.wav")):
            rel = wav.relative_to(path)
            parts = rel.parts
            # machine dir: first path part if it names a configured machine,
            # else the root directory itself (download layout: the root IS
            # the machine, e.g. .../fan/id_00/normal/x.wav, possibly called
            # with path=.../fan or path=.../fan/id_00).
            machine = parts[0] if parts and parts[0] in self.machines else None
            if machine is None and re.match(r"(fan|pump|valve|slider)$", root_name):
                machine = root_name
            if machine is None:
                # root is .../<machine>/id_XX — machine is its parent dir
                machine = path.parent.name if re.match(r"id_\d+", root_name) else root_name
            if machine not in self.machines:
                continue
            name = wav.stem
            rel_str = str(rel)
            # dB filter: only applies to archive-layout paths that carry a
            # dB token; download-layout paths (no dB anywhere) always match.
            if any(db in rel_str for db in ("0dB", "6dB", "min6dB")):
                if not any(db in rel_str for db in self.dbs):
                    continue
            # label: from directory ('abnormal') or archive token ('anomaly')
            is_anom = "abnormal" in parts or "anomaly" in name
            # machine-instance id: an id_XX directory relative to root, or the
            # root itself when called per-id (root = .../<machine>/id_XX).
            group = machine
            m = re.search(r"(id_\d+)", rel_str)
            if m is None:
                m = re.match(r"(id_\d+)$", root_name)
            if m:
                group = f"{machine}_{m.group(1)}"
            if self.machine_ids is not None and not any(
                mid in group for mid in self.machine_ids
            ):
                continue
            key = f"{group}/{'abnormal' if is_anom else 'normal'}"
            if self.max_clips_per_label is not None:
                if counts.get(key, 0) >= self.max_clips_per_label:
                    continue
                counts[key] = counts.get(key, 0) + 1
            audio, fs = sf.read(wav, dtype="float64", always_2d=True)
            if fs != self.fs_sync:
                raise ValueError(
                    f"{wav}: expected fs={self.fs_sync}, got {fs} (resample upstream)"
                )
            n = int(self.clip_seconds * self.fs_sync)
            audio = audio[:n]
            if audio.shape[1] == 1:
                audio = audio[:, 0]
            T = audio.shape[0]
            labels = np.zeros(T, dtype=np.int64)
            if is_anom:
                labels[:] = 1
            if self.feature == "logmel":
                from ..preprocessing.spectral import logmel_features

                feats, _, _ = logmel_features(
                    audio, fs, n_fft=self.n_fft, hop=self.hop,
                    n_mels=self.n_mels, channel_mode=self.channel_mode,
                )
                if self.append_deltas:
                    # first-order temporal deltas (dynamics): faults add
                    # impulsive/sweep components static mel means average out
                    d = np.diff(feats, axis=0, prepend=feats[:1])
                    feats = np.concatenate([feats, d], axis=1)
                # timestamps = frame indices (same convention as
                # synthetic_like): runners pass Preprocessor(fs_sync=1.0),
                # so the sync grid is the identity on frames.
                dataset.add(
                    UnitRecord(
                        unit_id=f"mimii_{group}_{'abnormal' if is_anom else 'normal'}_{name}",
                        signals=feats,
                        timestamps=np.arange(len(feats), dtype=np.float64),
                        anomaly_labels=np.full(len(feats), int(is_anom), dtype=np.int64),
                    )
                )
            else:
                dataset.add(
                    UnitRecord(
                        unit_id=f"mimii_{group}_{'abnormal' if is_anom else 'normal'}_{name}",
                        signals=audio if audio.ndim == 2 else audio.reshape(-1, 1),
                        timestamps=np.arange(T, dtype=np.float64) / self.fs_sync,
                        anomaly_labels=labels,
                    )
                )
        if len(dataset) == 0:
            raise FileNotFoundError(
                f"no MIMII wav files matched under {path} for machines={self.machines}"
            )
        return dataset

    @staticmethod
    def synthetic_like(
        n_units: int,
        frames_per_unit: int = 16000,
        fs_sync: float = 16000.0,
        anomaly_fraction: float = 0.2,
        seed: int = 0,
        n_channels: int = 1,
    ) -> UnitDataset:
        """Synthesize a MIMII-shaped dataset (for tests / smoke runs).

        Normal units: stationary oscillation + noise. Anomalous units: the
        same plus injected impulsive bursts (stand-in for real faults).
        """
        rng = np.random.default_rng(seed)
        dataset = UnitDataset()
        n_anom = int(round(anomaly_fraction * n_units))
        for i in range(n_units):
            is_anom = i < n_anom
            t = np.arange(frames_per_unit) / fs_sync
            base = np.stack(
                [np.sin(2 * np.pi * (50 + 30 * c) * t) + 0.1 * rng.standard_normal(frames_per_unit)
                 for c in range(n_channels)],
                axis=1,
            )
            labels = np.zeros(frames_per_unit, dtype=np.int64)
            if is_anom:
                burst_len = max(1, frames_per_unit // 10)
                n_bursts = max(1, frames_per_unit // 50)
                hi = max(1, frames_per_unit - burst_len)
                burst_starts = rng.integers(0, hi, size=n_bursts)
                for s in burst_starts:
                    for c in range(n_channels):
                        base[s : s + burst_len, c] += rng.standard_normal(burst_len) * 4.0
                labels[:] = 1
            dataset.add(
                UnitRecord(
                    unit_id=f"mimii_synth_{i:03d}{'_anom' if is_anom else ''}",
                    signals=base,
                    timestamps=np.arange(frames_per_unit, dtype=np.float64),
                    anomaly_labels=labels,
                )
            )
        return dataset
