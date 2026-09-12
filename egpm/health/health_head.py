"""Health index head (PRD §14, M7).

    h_i = sum_m p_i(m) * w_m,    w_m in [0, 1]

MVP default: fixed ORDINAL severity prior over the designed state ordering
(startup < stable < transition < abnormal < degrade < failure), keeping the
index interpretable (PRD §14 option a). Option (b) — learned weights with a
monotonicity regularizer — is an Open Experimental Decision; this class
supports both via `learn_weights`.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


def default_severity_weights(n_states: int) -> np.ndarray:
    """Linear ordinal prior: state 0 healthiest (w=1), failure w=0 (PRD
    health index convention: higher = healthier)."""
    if n_states < 2:
        raise ValueError("n_states must be >= 2")
    w = np.linspace(1.0, 0.0, n_states)
    return w


class HealthHead(nn.Module):
    """Belief-weighted health index in [0, 1].

    h_i = sum_m p_i(m) w_m. Weights are either a fixed ordinal prior
    (MVP default) or learned with a monotonicity penalty (Phase-2 option).
    NOTE: severity weights are NOT assumed physically valid (PRD §14) —
    validity is checked against N-CMAPSS auxiliary health-state labels.
    """

    def __init__(self, n_states: int, weights=None, learn_weights: bool = False):
        super().__init__()
        if weights is None:
            weights = default_severity_weights(n_states)
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != (n_states,):
            raise ValueError(f"weights must be [{n_states}]")
        if (w < 0).any() or (w > 1).any():
            raise ValueError("weights must lie in [0, 1]")
        if learn_weights:
            self.weights = nn.Parameter(torch.as_tensor(w, dtype=torch.float32))
        else:
            self.register_buffer(
                "weights", torch.as_tensor(w, dtype=torch.float32)
            )
        self.learn_weights = learn_weights

    # ------------------------------------------------------------------
    def forward(self, beliefs: torch.Tensor) -> torch.Tensor:
        """beliefs: [B, M] or [B, T, M] -> health in [0, 1] (same leading dims)."""
        if beliefs.shape[-1] != len(self.weights):
            raise ValueError(
                f"belief last dim must be M={len(self.weights)}; got {beliefs.shape}"
            )
        h = torch.einsum("...m,m->...", beliefs, self.weights.to(beliefs.dtype))
        return h

    # ------------------------------------------------------------------
    def monotonicity_penalty(self) -> torch.Tensor:
        """Penalty for learned weights violating the ordinal ordering
        (Phase-2 option b): sum of max(0, w_{m+1} - w_m + margin)."""
        if not self.learn_weights:
            return torch.zeros(())
        w = self.weights
        diffs = w[1:] - w[:-1]  # must be <= 0 (health decreasing toward failure)
        return torch.relu(diffs + 0.01).sum()
