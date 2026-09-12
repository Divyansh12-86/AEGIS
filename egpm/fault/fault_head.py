"""Fault classification head (PRD §13, M7).

Input: pooled state-belief trajectory + aggregate duration statistics
(total dwell per state, transition counts) — NEVER raw z/z_hat (Decision 5:
no continuous feature bypasses the bottleneck; A4 tests the skip as an
ablation). Output: softmax over C labeled fault classes.
"""
from __future__ import annotations

import torch
from torch import nn

from .temporal_pooling import BeliefGRU


class FaultHead(nn.Module):
    """Fault classifier from belief trajectory + duration statistics.

    Feature vector per run:
      * GRU-pooled belief summary [hidden]
      * total dwell per state [M]
      * transition counts per (i, j) [M * M] (row-normalized)
    """

    def __init__(self, n_states: int, n_classes: int, hidden: int = 32,
                 gru_hidden: int = 32, use_raw_embedding: bool = False,
                 embed_dim: int = 0):
        super().__init__()
        if use_raw_embedding and embed_dim <= 0:
            raise ValueError("use_raw_embedding requires embed_dim > 0")
        self.use_raw_embedding = use_raw_embedding  # ONLY for ablation A4
        self.n_states = n_states
        self.pool = BeliefGRU(n_states, hidden=gru_hidden)
        # features: gru_hidden + M (dwell) + M*M (transition counts)
        in_dim = gru_hidden + n_states + n_states * n_states
        if use_raw_embedding:
            in_dim += embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_classes),
        )

    # ------------------------------------------------------------------
    @staticmethod
    def duration_statistics(beliefs: torch.Tensor) -> torch.Tensor:
        """Aggregate stats: total expected dwell per state + expected
        transition counts. beliefs: [B, T, M]."""
        B, T, M = beliefs.shape
        dwell = beliefs.sum(dim=1)  # [B, M] total expected occupancy
        # expected transition counts: sum_t p_t(i) * A(i, j)? A lives in the
        # HSMM, not here — use successive-belief co-occurrence proxy:
        # trans[i, j] = sum_t p_t(i) p_{t+1}(j) (first-order co-flow)
        p_t = beliefs[:, :-1, :].sum(dim=1)  # [B, M]
        p_t1 = beliefs[:, 1:, :].sum(dim=1)  # [B, M]
        outer = torch.einsum("bi,bj->bij", p_t, p_t1)  # [B, M, M]
        trans = outer.reshape(B, M * M)
        # normalize counts to proportions for scale invariance
        trans = trans / trans.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return torch.cat([dwell, trans], dim=1)  # [B, M + M*M]

    # ------------------------------------------------------------------
    def forward(
        self,
        beliefs: torch.Tensor,
        raw_embedding: torch.Tensor = None,
    ) -> torch.Tensor:
        """beliefs: [B, T, M] -> logits [B, C].

        Passing ``raw_embedding`` without ``use_raw_embedding=True`` is a
        contract violation (bottleneck bypass, PRD Decision 5) and raises.
        """
        if beliefs.ndim != 3:
            raise ValueError(f"beliefs must be [B, T, M]; got {beliefs.shape}")
        if raw_embedding is not None and not self.use_raw_embedding:
            raise ValueError(
                "raw_embedding passed but use_raw_embedding=False — continuous "
                "features must not bypass the event/state bottleneck "
                "(PRD Decision 5; enable only for ablation A4)"
            )
        pooled = self.pool(beliefs)
        stats = self.duration_statistics(beliefs)
        feats = torch.cat([pooled, stats], dim=1)
        if self.use_raw_embedding:
            # ablation A4 ONLY: skip connection with raw embedding
            if raw_embedding is None:
                raise ValueError("use_raw_embedding=True requires raw_embedding")
            feats = torch.cat([feats, raw_embedding], dim=1)
        return self.mlp(feats)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_proba(self, beliefs: torch.Tensor, raw_embedding=None) -> torch.Tensor:
        return torch.softmax(self.forward(beliefs, raw_embedding), dim=-1)
