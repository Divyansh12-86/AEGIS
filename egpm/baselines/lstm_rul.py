"""LSTM-RUL baseline (PRD §21, M9).

Does explicit event/state structure beat implicit sequential memory? The
LSTM consumes the same preprocessed windows as EGPM and regresses RUL
directly — the standard deep-learning RUL baseline (Li et al. 2018 style,
answering the same research question as the CNN variant with a recurrent
inductive bias).
"""
from __future__ import annotations

import torch
from torch import nn


class LSTMRUL(nn.Module):
    """Window-sequence LSTM for direct RUL regression.

    Input: the run's windows [B, T, W, d] + mask [B, T, W, d] — the same
    unit-level runs EGPM sees. A per-window MLP embeds each window, the
    LSTM runs over the run's window sequence, and the last hidden state
    regresses RUL (PRD §21: identical data contracts, no event structure).
    """

    def __init__(self, n_channels: int, window_length: int,
                 hidden: int = 32, rnn_hidden: int = 32):
        super().__init__()
        self.window_embed = nn.Sequential(
            nn.Linear(window_length * n_channels, hidden),
            nn.GELU(),
        )
        self.lstm = nn.LSTM(hidden, rnn_hidden, batch_first=True)
        self.head = nn.Linear(rnn_hidden, 1)

    def forward(self, X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """X/mask: [B, T, W, d] -> RUL at the final window [B]."""
        if X.ndim != 4:
            raise ValueError(f"X must be [B, T, W, d]; got {X.shape}")
        B, T, W, d = X.shape
        m = (mask[:, :, 0, 0] > 0).float()  # [B, T]
        flat = X.reshape(B, T, W * d)
        h = self.window_embed(flat)  # [B, T, hidden]
        out, _ = self.lstm(h)
        last = out[:, -1, :]  # [B, rnn_hidden]
        return self.head(last).squeeze(-1)
