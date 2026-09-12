"""Neural sensor encoder (PRD §7, M2).

Architecture, deliberately minimal per the PRD:
  1. per-channel causal 1-D convolution (preserves channel identity);
  2. cross-channel fusion via a dilated causal convolution stack
     (geometrically increasing dilation);
  3. optional self-attention over window positions (config flag
     ``use_attention``, tested by ablation A5);
  4. window pooling (mean or attention) to one embedding z [h].

The decoder mirrors the encoder for the VQ reconstruction objective
(PRD §8): g_phi(z_hat) -> X_hat. Both operate on masked windows
(batch, W, d) + mask (batch, W, d).
"""
from __future__ import annotations

import math

import torch
from torch import nn


class CausalConv1d(nn.Module):
    """1-D causal convolution: pads only on the left (no future leak)."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int = 1):
        super().__init__()
        if kernel < 1:
            raise ValueError("kernel must be >= 1")
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(
            in_ch, out_ch, kernel, dilation=dilation, padding=0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        if self.pad > 0:
            x = torch.nn.functional.pad(x, (self.pad, 0))
        return self.conv(x)


class PerChannelCausalBlock(nn.Module):
    """Depthwise-style causal conv applied per sensor channel (PRD §7.1).

    Implemented as grouped convolution with groups == channels so each sensor
    keeps its own filter, then pointwise mixed position-wise.
    """

    def __init__(self, channels: int, hidden: int, kernel: int = 5):
        super().__init__()
        self.depthwise = nn.Conv1d(
            channels, channels, kernel, groups=channels, padding=0
        )
        self.pad = kernel - 1
        self.pointwise = nn.Conv1d(channels, hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, C, T]
        x = torch.nn.functional.pad(x, (self.pad, 0))
        return self.pointwise(self.depthwise(x))


class DilatedFusionStack(nn.Module):
    """Cross-channel fusion: stacked causal dilated convs (PRD §7.2).

    Dilation grows geometrically (1, 2, 4, ...) for a large receptive field
    without parameter blow-up.
    """

    def __init__(self, in_ch: int, hidden: int, n_blocks: int = 4, kernel: int = 3,
                 base_dilation: int = 2):
        super().__init__()
        if n_blocks < 1:
            raise ValueError("n_blocks must be >= 1")
        blocks = []
        for b in range(n_blocks):
            dilation = base_dilation ** b
            blocks.append(
                nn.Sequential(
                    CausalConv1d(in_ch if b == 0 else hidden, hidden,
                                 kernel, dilation=dilation),
                    nn.GELU(),
                )
            )
        self.blocks = nn.Sequential(*blocks)
        self.receptive_field = 1 + sum((kernel - 1) * (base_dilation ** b)
                                       for b in range(n_blocks))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)  # [B, hidden, T]


class WindowSelfAttention(nn.Module):
    """Single-head self-attention over window positions (PRD §7.3)."""

    def __init__(self, dim: int):
        super().__init__()
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, T, dim]
        B, T, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        att = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(D), dim=-1)
        return self.proj(att @ v)


class WindowPooling(nn.Module):
    """Pool window positions to one embedding (PRD §7.4): mean or attention."""

    def __init__(self, dim: int, mode: str = "mean"):
        super().__init__()
        if mode not in ("mean", "attention"):
            raise ValueError(f"pooling mode must be mean|attention, got {mode}")
        self.mode = mode
        if mode == "attention":
            self.score = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: [B, T, h]; mask: [B, T] (1 = valid)
        if self.mode == "mean":
            m = mask.unsqueeze(-1).to(x.dtype)
            return (x * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
        scores = self.score(x).squeeze(-1)
        scores = scores.masked_fill(mask == 0, -torch.inf)
        w = torch.softmax(scores, dim=-1).unsqueeze(-1)
        return (x * w).sum(dim=1)


class SensorEncoder(nn.Module):
    """Window [B, W, d] (+mask) -> embedding z [B, h] (PRD §28 contract)."""

    def __init__(
        self,
        n_channels: int,
        hidden: int = 64,
        embed_dim: int = 32,
        n_conv_blocks: int = 4,
        kernel: int = 3,
        use_attention: bool = False,
        pooling: str = "mean",
    ):
        super().__init__()
        self.per_channel = PerChannelCausalBlock(n_channels, hidden)
        self.fusion = DilatedFusionStack(hidden, hidden, n_blocks=n_conv_blocks,
                                         kernel=kernel)
        self.use_attention = use_attention
        if use_attention:
            self.attention = WindowSelfAttention(hidden)
        self.pool = WindowPooling(hidden, mode=pooling)
        self.proj = nn.Linear(hidden, embed_dim)

    def forward(self, X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # X: [B, W, d], mask: [B, W, d] -> z: [B, h]
        if X.ndim != 3 or mask.shape != X.shape:
            raise ValueError(f"X/mask must be [B, W, d]; got {X.shape}/{mask.shape}")
        m = (mask[:, :, 0] > 0).float()  # [B, W] — per-window availability
        h = X.permute(0, 2, 1)  # [B, d, W]
        h = self.per_channel(h)  # [B, hidden, W]
        h = self.fusion(h)  # [B, hidden, W]
        h = h.permute(0, 2, 1)  # [B, W, hidden]
        if self.use_attention:
            h = h + self.attention(h)  # residual attention
        z = self.pool(h, m)  # [B, hidden]
        return self.proj(z)  # [B, embed_dim]


class SensorDecoder(nn.Module):
    """Mirror of the encoder for the VQ reconstruction objective g_phi (PRD §8).

    Maps a (quantized) embedding [B, h] back to a window reconstruction
    [B, W, d] plus an availability mask of ones (reconstruction targets use
    the original mask externally).
    """

    def __init__(self, n_channels: int, window_length: int,
                 embed_dim: int = 32, hidden: int = 64):
        super().__init__()
        self.W = int(window_length)
        self.in_proj = nn.Linear(embed_dim, hidden)
        # positional code for the W positions we must generate
        self.pos = nn.Parameter(torch.zeros(1, self.W, hidden))
        nn.init.normal_(self.pos, std=0.02)
        self.block = CausalConv1d(hidden, hidden, kernel=5)
        self.out = nn.Linear(hidden, n_channels)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B, h] -> X_hat: [B, W, d]
        B = z.shape[0]
        h = self.in_proj(z).unsqueeze(1).expand(B, self.W, -1) + self.pos
        h = h + self.block(h.permute(0, 2, 1)).permute(0, 2, 1)  # causal residual
        return self.out(torch.nn.functional.gelu(h))
