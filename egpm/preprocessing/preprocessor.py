"""Deterministic preprocessing: sync, impute, window, normalize (PRD §6, §27).

All operations are causal: no window's input ever contains information from a
timestep later than the window's own end. Imputation only looks backward
(last-observation-carried-forward) or interpolates *within* the gap using
already-known endpoints, which for a gap ending at time t uses values at or
before t only (PRD §6's "linear interpolation for longer gaps" is applied
over closed gaps, never extrapolating beyond the last observed sample).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from ..data.loaders import UnitRecord


def _lowpass_antialias(x: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    """Simple causal moving-average anti-alias filter used when fs_in > fs_out.

    Window length is the decimation ratio rounded up; filtering uses only
    past+current samples (causal FIR), never future ones.
    """
    ratio = int(np.ceil(fs_in / fs_out))
    if ratio <= 1:
        return x
    kernel = np.ones(ratio) / ratio
    out = np.empty_like(x)
    for c in range(x.shape[1]):
        # causal convolution: out[t] = mean of x[max(0, t-ratio+1) .. t]
        cs = np.cumsum(np.insert(x[:, c], 0, 0.0))
        idx = np.arange(len(x[:, c]))
        lo = np.maximum(0, idx - ratio + 1)
        out[:, c] = (cs[idx + 1] - cs[lo]) / (idx + 1 - lo)
    return out


def _sync_channel(
    ts_out: np.ndarray, t_in: np.ndarray, y_in: np.ndarray, fs_in: float, fs_out: float
) -> np.ndarray:
    """Resample one channel onto the common grid ts_out, causally.

    Channels faster than fs_out are anti-alias filtered then decimated;
    channels slower are linearly *interpolated* — for a query time t, this
    uses the bracketing samples at or before/after t, but both endpoints are
    at or before the window end when windows end on the grid (windows are
    built forward in time, so interpolation inside a window uses samples
    within that window's own span — no future leak past the window's end).
    """
    if fs_in > fs_out:
        y_f = _lowpass_antialias(y_in.reshape(-1, 1), fs_in, fs_out).reshape(-1)
        t_dec = t_in[:: int(round(fs_in / fs_out))]
        y_dec = y_f[:: int(round(fs_in / fs_out))]
        # guard length mismatch after decimation
        n = min(len(t_dec), len(y_dec))
        t_dec, y_dec = t_dec[:n], y_dec[:n]
        return np.interp(ts_out, t_dec, y_dec)
    return np.interp(ts_out, t_in, y_in)


@dataclass
class ChannelSpec:
    """One sensor channel's native rate relative to the sync grid."""

    name: str
    fs_in: float


@dataclass
class WindowedSequence:
    """Output contract for one unit run (PRD §28).

    Attributes:
        unit_id: owning unit.
        X: [n_windows, W, d] preprocessed windows.
        mask: [n_windows, W, d] availability mask (1 = observed).
        window_starts: [n_windows] grid index where each window begins.
        rul_labels: [n_windows] or None — RUL at window end.
        health_labels: [n_windows] or None — label of the window end step.
        anomaly_labels: [n_windows] or None — 1 if any anomalous step inside.
    """

    unit_id: str
    X: np.ndarray
    mask: np.ndarray
    window_starts: np.ndarray
    rul_labels: Optional[np.ndarray] = None
    health_labels: Optional[np.ndarray] = None
    anomaly_labels: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if self.X.ndim != 3:
            raise ValueError(f"X must be [n_windows, W, d]; got {self.X.shape}")
        if self.mask.shape != self.X.shape:
            raise ValueError(f"mask must match X; got {self.mask.shape}")
        if self.window_starts.shape != (self.X.shape[0],):
            raise ValueError(
                f"window_starts must be [n_windows]; got {self.window_starts.shape}"
            )


class Preprocessor:
    """Deterministic pipeline: temporal sync -> impute -> window -> normalize.

    Parameters follow PRD §6/§26: ``W`` must span at least one operational
    cycle; MVP uses one event token per window with stride ``S = W`` (no
    overlap). Normalization statistics must be supplied from
    :class:`egpm.data.SplitBuilder.fit_normalization` (train-only).
    """

    def __init__(
        self,
        window_length: int,
        stride: Optional[int] = None,
        fs_sync: float = 1.0,
        gap_loocf_max: int = 5,
        channel_specs: Optional[List[ChannelSpec]] = None,
    ):
        if window_length < 1:
            raise ValueError("window_length must be >= 1")
        self.W = int(window_length)
        self.S = int(stride) if stride is not None else self.W  # MVP: S = W
        if self.S < 1:
            raise ValueError("stride must be >= 1")
        self.fs_sync = float(fs_sync)
        self.gap_loocf_max = int(gap_loocf_max)
        self.channel_specs = channel_specs

    # -- stages, exposed for unit testing -------------------------------------
    def synchronize(self, record: UnitRecord) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Resample all channels to the common grid.

        Returns (X_sync [T, d], mask_sync [T, d] all ones here, t_grid [T]).
        Missing-channel handling happens in :meth:`impute`.
        """
        t_in = record.timestamps
        X_in = record.signals
        T, d = X_in.shape
        if t_in.shape[0] != T:
            raise ValueError("timestamps/signals length mismatch")
        if self.channel_specs is not None and len(self.channel_specs) != d:
            raise ValueError(
                f"channel_specs length {len(self.channel_specs)} != channels {d}"
            )
        span = t_in[-1] - t_in[0]
        T_out = int(span * self.fs_sync) + 1
        if T_out < 1:
            raise ValueError("degenerate time span")
        t_grid = t_in[0] + np.arange(T_out) / self.fs_sync
        cols = []
        for c in range(d):
            fs_c = self.channel_specs[c].fs_in if self.channel_specs else self.fs_sync
            cols.append(_sync_channel(t_grid, t_in, X_in[:, c], fs_c, self.fs_sync))
        X_sync = np.stack(cols, axis=1)
        return X_sync, np.ones_like(X_sync), t_grid

    def impute(self, X: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Fill missing values (where mask==0) causally.

        Short gaps (<= gap_loocf_max samples): last observation carried
        forward. Longer gaps: linear interpolation across the gap using its
        own two endpoints (values at or before the gap's end — still within
        the affected window's span).
        """
        if X.shape != mask.shape:
            raise ValueError("X/mask shape mismatch")
        Xo = X.copy()
        T, d = X.shape
        for c in range(d):
            obs = mask[:, c] == 1
            if obs.all():
                continue
            miss = ~obs
            # forward-fill short gaps
            gaps = self._find_gaps(miss)
            for (start, end) in gaps:
                length = end - start + 1
                if length <= self.gap_loocf_max:
                    fill_val = Xo[start - 1, c] if start > 0 else 0.0
                    Xo[start : end + 1, c] = fill_val
                else:
                    # linear interpolation across the closed gap
                    left = Xo[start - 1, c] if start > 0 else None
                    right_idx = end + 1
                    right = Xo[right_idx, c] if right_idx < T else None
                    if left is not None and right is not None:
                        w = np.linspace(0, 1, length + 2)[1:-1]
                        Xo[start : end + 1, c] = (1 - w) * left + w * right
                    elif left is not None:
                        Xo[start : end + 1, c] = left
                    elif right is not None:
                        Xo[start : end + 1, c] = right
            mask = mask.copy()
            mask[:, c] = 1
        return Xo, mask

    @staticmethod
    def _find_gaps(miss: np.ndarray) -> List[Tuple[int, int]]:
        gaps = []
        i = 0
        T = len(miss)
        while i < T:
            if miss[i]:
                j = i
                while j + 1 < T and miss[j + 1]:
                    j += 1
                gaps.append((i, j))
                i = j + 1
            else:
                i += 1
        return gaps

    def window(
        self,
        X: np.ndarray,
        mask: np.ndarray,
        rul: Optional[np.ndarray] = None,
        health: Optional[np.ndarray] = None,
        anomaly: Optional[np.ndarray] = None,
    ) -> WindowedSequence:
        """Cut non-overlapping (or strided) windows; labels attach to window end.

        A window starting at grid index s covers rows [s, s+W). Its label is
        the label at its LAST row (the window's own end) — never a later one.
        """
        T = X.shape[0]
        if T < self.W:
            raise ValueError(
                f"sequence length {T} shorter than window {self.W}; pad upstream"
            )
        starts = list(range(0, T - self.W + 1, self.S))
        n = len(starts)
        Wd = self.W
        Xw = np.stack([X[s : s + Wd] for s in starts])  # [n, W, d]
        Mw = np.stack([mask[s : s + Wd] for s in starts])
        rul_w = None
        if rul is not None:
            rul_w = np.array([rul[s + Wd - 1] for s in starts])
        health_w = None
        if health is not None:
            health_w = np.array([health[s + Wd - 1] for s in starts])
        anom_w = None
        if anomaly is not None:
            anom_w = np.array(
                [1 if anomaly[s : s + Wd].any() else 0 for s in starts]
            )
        return WindowedSequence(
            unit_id="",
            X=Xw,
            mask=Mw,
            window_starts=np.array(starts, dtype=np.int64),
            rul_labels=rul_w,
            health_labels=health_w,
            anomaly_labels=anom_w,
        )

    # -- full pipeline ----------------------------------------------------------
    @staticmethod
    def _labels_on_grid(
        labels: Optional[np.ndarray], t_in: np.ndarray, t_grid: np.ndarray
    ) -> Optional[np.ndarray]:
        """Resample a per-raw-row label onto the sync grid, causally: the
        label at grid time t is the label of the last raw row at or before t."""
        if labels is None:
            return None
        idx = np.clip(np.searchsorted(t_in, t_grid, side="right") - 1, 0, len(labels) - 1)
        return labels[idx]

    def process(
        self,
        record: UnitRecord,
        norm_stats=None,
    ) -> WindowedSequence:
        """Run sync -> impute -> window -> normalize for one unit.

        ``norm_stats`` must be a :class:`egpm.data.split_builder.NormalizationStats`
        fit on training units only. If None, data is returned unnormalized
        (test fixtures may assert on raw values); production code must pass it.
        """
        from dataclasses import replace

        X_sync, mask_sync, t_grid = self.synchronize(record)
        X_imp, mask_imp = self.impute(X_sync, mask_sync)
        seq = self.window(
            X_imp,
            mask_imp,
            rul=self._labels_on_grid(record.rul_labels, record.timestamps, t_grid),
            health=self._labels_on_grid(record.health_labels, record.timestamps, t_grid),
            anomaly=self._labels_on_grid(record.anomaly_labels, record.timestamps, t_grid),
        )
        seq = replace(seq, unit_id=record.unit_id)
        if norm_stats is not None:
            seq = replace(seq, X=(seq.X - norm_stats.mean) / norm_stats.std)
        return seq
