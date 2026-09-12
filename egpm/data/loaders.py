"""Unit-level dataset contracts and loaders (PRD §27 `data/`, M1).

Every sequence is owned by a *unit* (one asset run). Splits are made at the
unit level BEFORE any windowing; normalization statistics are computed only
from training units (PRD §6).
"""
from __future__ import annotations

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

    The public dataset (DS01: 6 train units, DS02–DS08a additional units) is
    distributed as .mat (MATLAB v7.3 / HDF5) files. Because the dataset is not
    bundled with the repo, this loader accepts either:

    1. A directory of ``*.mat`` files (one per unit) each containing the
       standard N-CMAPSS variable groups: ``A``, ``W``, ``X_s``, ``X_v``
       (sensor/health/aux matrices, rows = cycles) and ``Y`` (unit + cycle).
    2. A pre-extracted ``.npz`` archive with per-unit arrays (produced by
       :meth:`NCMAPSSLoader.to_npz`), so CI can run without MATLAB files.

    Only ``X_s`` (health proxies) and ``W`` (operational settings) are used as
    signals by default; ``aux_health`` rows are decoded into health-state
    labels 1..6 when present.
    """

    #: channels of ``Y``: (unit number, time cycle)
    _Y_COLS = ("unit", "cycle")
    #: N-CMAPSS auxiliary health code: 3 bits
    _HEALTH_BITS = 3

    def __init__(self, use_health_proxies: bool = True):
        self.use_health_proxies = use_health_proxies

    # -- public API ----------------------------------------------------------
    def load(self, path: str | Path) -> UnitDataset:
        """Load all units under ``path`` (directory of .mat/.npz, or single file)."""
        path = Path(path)
        if path.is_dir():
            files = sorted(
                p for p in path.iterdir() if p.suffix in {".mat", ".npz"}
            )
            if not files:
                raise FileNotFoundError(f"no .mat/.npz files under {path}")
            dataset = UnitDataset()
            for f in files:
                self._load_one_into(f, dataset)
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
            self._load_mat(file, dataset)

    def _load_mat(self, file: Path, dataset: UnitDataset) -> None:
        try:
            import h5py  # N-CMAPSS .mat files are MATLAB v7.3 (HDF5)
        except ImportError as e:  # pragma: no cover - optional dep
            raise ImportError(
                "h5py is required to read N-CMAPSS .mat files: pip install h5py"
            ) from e
        with h5py.File(file, "r") as f:
            Y = np.array(f["Y"]).T  # [T, 2]: unit, cycle
            W = np.array(f["W"]).T  # [T, n_oper]
            if self.use_health_proxies and "X_s" in f:
                Xs = np.array(f["X_s"]).T  # [T, n_health]
            else:
                Xs = np.empty((Y.shape[0], 0))
            if "aux_health" in f:
                bits = np.array(f["aux_health"]).T  # [T, 3]
                health = bits @ (2 ** np.arange(self._HEALTH_BITS))
                health = np.clip(health, 1, 6)
            else:
                health = None
            signals = np.concatenate(
                [c for c in (W, Xs) if c.size], axis=1
            )
            unit_ids = np.unique(Y[:, 0]).astype(int).tolist()
            if len(unit_ids) != 1:
                raise ValueError(
                    f"{file.name}: expected exactly one unit per .mat file, "
                    f"found {unit_ids}"
                )
            cycles = Y[:, 1]
            dataset.add(
                UnitRecord(
                    unit_id=f"ncmapss_u{unit_ids[0]}",
                    signals=signals.astype(np.float64),
                    timestamps=cycles.astype(np.float64),
                    health_labels=health,
                    rul_labels=self._compute_rul(cycles),
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

    def _compute_rul(self, cycles: np.ndarray) -> np.ndarray:
        """Piecewise-linear RUL label: RUL = (max_cycle - cycle) per unit."""
        last = cycles.max()
        return (last - cycles).astype(np.float64)

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

    MIMII ships as 16 kHz mono wav files organized as
    ``<machine>/<split>/<wav_id>.wav`` where wav_id encodes the machine ID and
    normal/anomalous attribute. Each wav is a ~10 s clip = one "unit"; anomaly
    labels are derived from the filename. When the raw dataset is unavailable,
    tests use :meth:`synthetic_like` which produces the same contract.
    """

    _ANOMALY_TAG = ("normal", "anomaly")

    def __init__(
        self,
        machines: Sequence[str] = ("pump", "fan", "valve", "slider"),
        dbs: Sequence[str] = ("0dB", "6dB", "min6dB"),
        fs_sync: float = 16000.0,
        clip_seconds: float = 10.0,
    ):
        self.machines = list(machines)
        self.dbs = list(dbs)
        self.fs_sync = fs_sync
        self.clip_seconds = clip_seconds

    def load(self, path: str | Path) -> UnitDataset:
        """Load all wav files under ``path`` matching configured machine/dB."""
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
        for wav in sorted(path.rglob("*.wav")):
            rel = wav.relative_to(path)
            parts = rel.parts
            machine = parts[0] if parts else ""
            if machine not in self.machines:
                continue
            name = wav.stem
            if not any(db in name or db in str(rel) for db in self.dbs):
                continue
            is_anom = "anomaly" in name
            audio, fs = sf.read(wav, dtype="float64")
            if fs != self.fs_sync:
                raise ValueError(
                    f"{wav}: expected fs={self.fs_sync}, got {fs} (resample upstream)"
                )
            n = int(self.clip_seconds * self.fs_sync)
            audio = audio[:n]
            labels = np.zeros(len(audio), dtype=np.int64)
            if is_anom:
                labels[:] = 1
            dataset.add(
                UnitRecord(
                    unit_id=f"mimii_{wav.stem}",
                    signals=audio.reshape(-1, 1),
                    timestamps=np.arange(len(audio), dtype=np.float64),
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
