"""Core baselines (PRD §21, M9 — subset: CNN-RUL + AutoEncoder anomaly).

Both consume the SAME data contracts as EGPM (windows [N, W, d] + masks):
the fair-comparison protocol (PRD §21) requires identical unit-level splits
and equivalent preprocessing. OmniAnomaly/GrammarViz/CBM are additional
core baselines to be added before final reporting; this module ships the two
needed for the first experiment (PRD Deliverable E).
"""
from __future__ import annotations

import torch
from torch import nn


class CNNRUL(nn.Module):
    """CNN over windows for direct RUL regression (Babu et al., 2016 style).

    Baseline family: continuous-representation, no event/grammar structure.
    Answers RQ1: does event-grammar structure beat direct conv feature
    learning for RUL?
    """

    def __init__(self, n_channels: int, window_length: int, hidden: int = 32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_channels, hidden, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """X/mask: [B, W, d] -> RUL [B]."""
        m = (mask[:, :, 0] > 0).float().unsqueeze(1)  # [B, 1, W]
        h = X.permute(0, 2, 1) * m
        feat = self.conv(h).squeeze(-1)  # [B, hidden]
        return self.head(feat).squeeze(-1)


class WindowAutoEncoder(nn.Module):
    """Continuous-embedding reconstruction anomaly baseline (PRD Deliverable E).

    A simple conv autoencoder: anomaly score = window reconstruction MSE.
    Baseline family: continuous latent, no discrete events, no temporal
    model. Answers RQ1/RQ2: is grammar likelihood competitive with
    latent-space reconstruction likelihood (the OmniAnomaly comparison)?
    """

    def __init__(self, n_channels: int, window_length: int, hidden: int = 32,
                 latent: int = 8):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(n_channels, hidden, 5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden, latent, 3, padding=1),
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose1d(latent, hidden, 3, padding=1),
            nn.GELU(),
            nn.ConvTranspose1d(hidden, n_channels, 5, padding=2),
        )

    def forward(self, X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """X/mask [B, W, d] -> reconstruction [B, W, d]."""
        h = self.enc(X.permute(0, 2, 1))
        rec = self.dec(h).permute(0, 2, 1)
        return rec

    @torch.no_grad()
    def anomaly_scores(self, X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Per-window MSE (masked) — the anomaly score [B]."""
        rec = self.forward(X, mask)
        se = ((rec - X) ** 2 * mask).sum(dim=(1, 2)) / mask.sum(dim=(1, 2)).clamp_min(1.0)
        return se
