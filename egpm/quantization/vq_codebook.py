"""VQ-VAE event discovery with the corrected straight-through estimator (PRD §8, M3).

Event primitive = a recurring learned behavioral pattern in the encoder's
window-level representation. Mechanism (Decision 2, resolved-conflicts table):

    v_i   = argmin_k || z_i - c_k ||^2
    z_hat = z + sg(c_v - z)          # forwards c_v; gradient flows to z  (*)

(*) This is the corrected form. The original methodology's
    `h + sg(h - e)` would forward `h` itself, silently undoing quantization.

Codebook updates use EMA toward the running mean of assigned embeddings
(standard VQ-VAE practice). Collapse monitoring (perplexity + effective
codebook size) is REQUIRED and exposed on every forward.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


@dataclass
class VQOutput:
    """Quantizer output contract (PRD §28).

    Attributes:
        event_ids: [B] long, in 0..K-1 (shifted to 1..K only at reporting).
        quantized: [B, h] — the codebook vectors c_v (what downstream sees).
        straight_through: [B, h] — z + sg(c_v - z): value == c_v forward,
            gradient == identity to z backward.
        codebook_stats: dict with perplexity, usage_histogram,
            effective_size, mean_usage_entropy.
    """

    event_ids: torch.Tensor
    quantized: torch.Tensor
    straight_through: torch.Tensor
    codebook_stats: dict


class VQCodebook(nn.Module):
    """Learnable event vocabulary {c_1..c_K} with EMA updates (PRD §8).

    Args:
        embed_dim: h — must match encoder output dim.
        n_codes: K — vocabulary size (tunable, PRD §26).
        commitment_beta: beta in the commitment loss ||z - sg(c_v)||^2.
        ema_decay: EMA decay for codebook updates (0.99 typical).
    """

    def __init__(self, embed_dim: int, n_codes: int = 32,
                 commitment_beta: float = 0.25, ema_decay: float = 0.99):
        super().__init__()
        if embed_dim < 1 or n_codes < 1:
            raise ValueError("embed_dim and n_codes must be >= 1")
        self.K = n_codes
        self.h = embed_dim
        self.beta = commitment_beta
        self.ema_decay = ema_decay

        # EMA state (buffers, not parameters — updated by rule, not autograd)
        codebook = torch.randn(n_codes, embed_dim) * 0.5
        self.register_buffer("codebook", codebook)
        self.register_buffer("ema_cluster_size", torch.zeros(n_codes))
        self.register_buffer("ema_embed_sum", codebook.clone())

    # ------------------------------------------------------------------
    def forward(self, z: torch.Tensor) -> VQOutput:
        """Quantize batch of embeddings z [B, h] -> event ids + ST estimates."""
        if z.ndim != 2 or z.shape[1] != self.h:
            raise ValueError(f"z must be [B, {self.h}]; got {tuple(z.shape)}")
        # distances: [B, K]
        d2 = torch.cdist(z.unsqueeze(0), self.codebook.unsqueeze(0)).squeeze(0) ** 2
        ids = d2.argmin(dim=1)  # [B]
        c_v = self.codebook[ids]  # [B, h]
        # corrected straight-through: forward value is c_v; grad to z is d(c_v)/dz = I
        st = z + (c_v - z).detach()
        stats = self._collapse_stats(ids)
        if self.training:
            self._ema_update(z.detach(), ids)
        return VQOutput(
            event_ids=ids,
            quantized=c_v,
            straight_through=st,
            codebook_stats=stats,
        )

    # ------------------------------------------------------------------
    def loss_terms(self, z: torch.Tensor, out: Optional[VQOutput] = None) -> dict:
        """Reconstruction + codebook losses (PRD §8, Eq. after the ST def).

        L_recon (recon only, vs targets) is computed in the trainer; here we
        provide the VQ-internal terms:
          codebook:  || sg(z) - c_v ||^2      (EMA reduces this in practice;
                     kept for logging/diagnostics)
          commit:    beta * || z - sg(c_v) ||^2  (gradient to encoder only)
        """
        if out is None:
            out = self(z)
        codebook_loss = (out.quantized - z.detach()).pow(2).sum(dim=1).mean()
        commit_loss = self.beta * (z - out.quantized.detach()).pow(2).sum(dim=1).mean()
        return {
            "codebook": codebook_loss,
            "commitment": commit_loss,
        }

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _ema_update(self, z: torch.Tensor, ids: torch.Tensor) -> None:
        """EMA codebook update toward mean of assigned embeddings (PRD §8)."""
        B = z.shape[0]
        one_hot = torch.zeros(B, self.K, device=z.device, dtype=z.dtype)
        one_hot.scatter_(1, ids.unsqueeze(1), 1.0)
        counts = one_hot.sum(dim=0)  # [K]
        sum_z = one_hot.t() @ z  # [K, h]
        self.ema_cluster_size.mul_(self.ema_decay).add_(counts, alpha=1 - self.ema_decay)
        self.ema_embed_sum.mul_(self.ema_decay).add_(sum_z, alpha=1 - self.ema_decay)
        # Laplace smoothing to avoid zero counts
        n = self.ema_cluster_size + 1e-5
        self.codebook.copy_(self.ema_embed_sum / n.unsqueeze(1))

    # ------------------------------------------------------------------
    def _collapse_stats(self, ids: torch.Tensor) -> dict:
        """Perplexity + effective size monitoring (PRD §8, required)."""
        with torch.no_grad():
            usage = torch.bincount(ids, minlength=self.K).to(torch.float64)
            usage = usage / max(int(ids.numel()), 1)
            usage_hist = usage / usage.sum().clamp_min(1e-12)
            entropy = -(usage_hist * (usage_hist + 1e-12).log()).sum()
            perplexity = torch.exp(entropy)
            effective = int((usage > 1e-3).sum())
        return {
            "perplexity": float(perplexity),
            "usage_histogram": usage_hist.cpu().numpy(),
            "effective_size": effective,
            "mean_usage_entropy": float(entropy),
        }

    # ------------------------------------------------------------------
    @torch.no_grad()
    def event_id(self, z: torch.Tensor) -> torch.Tensor:
        """Deterministic argmin assignment for inference (PRD §4)."""
        d2 = torch.cdist(z.unsqueeze(0), self.codebook.unsqueeze(0)).squeeze(0) ** 2
        return d2.argmin(dim=1)
