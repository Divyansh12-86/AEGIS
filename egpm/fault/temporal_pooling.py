"""Lightweight temporal pooling over the state-belief trajectory (PRD §13)."""
from __future__ import annotations

import torch
from torch import nn


class BeliefGRU(nn.Module):
    """Small GRU pooling p_{1:T} -> run-level summary (PRD §13 'small GRU').

    Input: [B, T, M] belief trajectories; output: [B, hidden].
    """

    def __init__(self, n_states: int, hidden: int = 32, n_layers: int = 1):
        super().__init__()
        self.gru = nn.GRU(n_states, hidden, num_layers=n_layers, batch_first=True)
        self.hidden = hidden

    def forward(self, beliefs: torch.Tensor) -> torch.Tensor:
        if beliefs.ndim != 3:
            raise ValueError(f"beliefs must be [B, T, M]; got {beliefs.shape}")
        out, _ = self.gru(beliefs)
        return out[:, -1, :]  # last hidden state
