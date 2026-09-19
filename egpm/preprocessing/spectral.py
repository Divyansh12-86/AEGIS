"""Causal log-mel spectral front-end for machine-sound clips (MIMII, PRD §6).

Framing is causal: frame k is computed from samples [k*hop, k*hop+n_fft) and
its timestamp is its END sample index (k*hop + n_fft), so a frame never
contains samples after its own end (PRD §6 window-causality rule).

Uses only scipy + numpy — no librosa/torchaudio dependency.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import stft

# Slaney-style mel scale constants (HTS, fs=16000 territory)
_MEL_BREAK_HZ = 1000.0 / (200.0 / 3.0)  # linear->log break frequency in Hz


def _hz_to_mel(f_hz):
    f = np.asarray(f_hz, dtype=np.float64)
    b = _MEL_BREAK_HZ
    return np.where(f < b, f, b * np.log(1.0 + f / b) / np.log(2.0))


def _mel_to_hz(m_mel):
    m = np.asarray(m_mel, dtype=np.float64)
    b = _MEL_BREAK_HZ
    return np.where(m < b, m, b * (2.0 ** (m / b) - 1.0))


def _mel_filterbank(fs: float, n_fft: int, n_mels: int,
                    f_min: float = 40.0, f_max: float = None) -> np.ndarray:
    """Triangular mel filterbank: [n_mels, n_fft//2 + 1] (Slaney scale).

    Triangle edges are quantized to the nearest FFT bins and forced strictly
    increasing, so every mel band spans at least two bins — no dead rows at
    coarse frequency resolution. ``f_min=40`` Hz: machine-sound energy of
    interest lives above this; sub-40Hz bins are DC/rumble.
    """
    if f_max is None:
        f_max = fs / 2.0
    n_freqs = n_fft // 2 + 1
    mel_pts = _hz_to_mel(np.array([f_min, f_max]))
    edges_hz = _mel_to_hz(np.linspace(mel_pts[0], mel_pts[1], n_mels + 2))
    bin_hz = fs / n_fft
    bins = np.clip(np.round(edges_hz / bin_hz).astype(int), 0, n_freqs - 1)
    for i in range(1, len(bins)):  # strictly increasing edge bins
        bins[i] = max(bins[i], bins[i - 1] + 1)
    bins = np.minimum(bins, n_freqs - 1)
    H = np.zeros((n_mels, n_freqs))
    for m in range(n_mels):
        lo, mid, hi = bins[m], bins[m + 1], bins[m + 2]
        if hi <= mid:  # top clipped: band collapsed, skip
            continue
        if mid > lo:
            H[m, lo:mid] = (np.arange(lo, mid) - lo) / (mid - lo)
        H[m, mid:hi + 1] = 1.0 - (np.arange(mid, hi + 1) - mid) / (hi - mid + 1.0)
    return H


def logmel_features(
    audio: np.ndarray,
    fs: float,
    n_fft: int = 1024,
    hop: int = 256,
    n_mels: int = 64,
    eps: float = 1e-10,
    channel_mode: str = "mean",
) -> tuple:
    """Causal log-mel features for a multichannel clip.

    Args:
        audio: [n_samples] or [n_samples, C] time-domain clip.
        fs: sample rate (Hz).
        n_fft: FFT size (frame length in samples).
        hop: frame stride in samples.
        n_mels: mel bands per channel.
        channel_mode: 'mean' -> [n_frames, 2*n_mels] (per-band mean & std
            across the 8 mics; ceiling: collapses spatial channel differences);
            'full' -> [n_frames, n_mels*C] (raw per-channel).

    Returns:
        (features [n_frames, n_feats], timestamps [n_frames] in seconds,
         frames_per_second)
    """
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError(f"audio must be [n_samples] or [n_samples, C]; got {x.shape}")

    n_samples, C = x.shape
    # causal framing: frame k = samples [k*hop, k*hop+n_fft), no zero-padding
    # of a "future" window; scipy.stft with boundary=None + padded n_fft hits
    # exactly this: k-th frame covers [k*hop, k*hop+n_fft).
    frames = []
    for c in range(C):
        f, t, Zxx = stft(x[:, c], fs=fs, nperseg=n_fft, noverlap=n_fft - hop,
                         boundary=None, padded=False)
        frames.append(np.abs(Zxx))  # [n_freqs, n_frames]
    mag = np.stack(frames, axis=-1)                # [n_freqs, n_frames, C]
    n_freqs, n_frames, C = mag.shape
    H = _mel_filterbank(fs, n_fft, n_mels)
    mel = H @ mag.reshape(n_freqs, -1)             # [n_mels, n_frames*C]
    mel = mel.reshape(n_mels, n_frames, C)
    # channels concatenated: [n_frames, n_mels, C] -> [n_frames, n_mels*C]
    if channel_mode not in ("full", "mean"):
        raise ValueError(f"channel_mode must be 'full' or 'mean'")
    if channel_mode == "full":
        # keep raw per-channel features: [n_frames, n_mels*C]
        feats = np.transpose(mel, (1, 0, 2)).reshape(n_frames, n_mels * C)
    else:
        # fold channels: per-(mel_band) mean & std over the 8 mics
        # [n_mels, n_frames, C] -> mean/std on C -> [n_mels, n_frames, 2]
        band_mean = mel.mean(axis=2, keepdims=False)      # [n_mels, n_frames]
        band_std = mel.std(axis=2, keepdims=False)       # [n_mels, n_frames]
        feats = np.concatenate([
            np.transpose(band_mean, (1, 0)),
            np.transpose(band_std, (1, 0)),
        ], axis=1)                                        # [n_frames, 2*n_mels]
    feats = np.log(feats.astype(np.float64) + eps)
    # timestamps = frame END times (causal: frame k's info ends at its end)
    ts = (np.arange(1, n_frames + 1) * hop + n_fft - hop) / fs
    return feats.astype(np.float32), ts, fs / hop
