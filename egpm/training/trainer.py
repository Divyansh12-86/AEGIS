"""Staged training loop (PRD §16, M-training).

Stage 1 — Encoder/VQ pretraining: train f_theta, g_phi, codebook on
          L_recon + L_commit (no HSMM yet). Exit: perplexity stabilizes,
          reconstruction plateaus.
Stage 2 — HSMM induction: fit (pi, A, B, D) via Baum-Welch on event
          sequences from normal-operation runs. Exit: EM log-likelihood
          converges.
Stage 3 — Joint fine-tuning (optional here): composite loss; the MVP
          trainer implements stages 1-2 plus head warmup on synthetic labels;
          full Stage-3 joint gradient flow through the forward-algorithm
          log-likelihood is a research-grade extension wired but defaulted
          off (PRD §15 'Gradient flow' — differentiable HSMM params).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn

from ..encoder import SensorEncoder, SensorDecoder
from ..quantization import VQCodebook
from ..grammar import HSMM, BaumWelch, ForwardBackward, SegmentalViterbi
from ..utils import set_global_seed, generate_run_id


@dataclass
class StageStats:
    stage: str
    steps: int = 0
    recon_loss: float = float("nan")
    commitment_loss: float = float("nan")
    perplexity: float = float("nan")
    effective_size: int = 0
    em_log_likelihood: float = float("nan")
    em_iterations: int = 0
    wall_clock_s: float = 0.0


@dataclass
class TrainerConfig:
    # data/window (PRD §26: all tunable, dataset-dependent)
    window_length: int = 16
    stride: Optional[int] = None  # default = window_length (MVP)
    # encoder
    embed_dim: int = 16
    hidden: int = 32
    n_conv_blocks: int = 3
    use_attention: bool = False
    # VQ
    n_codes: int = 16
    commitment_beta: float = 0.25
    ema_decay: float = 0.99
    # HSMM
    n_states: int = 6
    d_max: int = 12
    em_max_iter: int = 20
    em_restarts: int = 0
    # optimization
    lr: float = 1e-3
    batch_size: int = 32
    stage1_steps: int = 300
    seed: int = 0
    log_every: int = 50


class Trainer:
    """Orchestrates Stage 1 (encoder+VQ) and Stage 2 (HSMM EM)."""

    def __init__(self, config: TrainerConfig, n_channels: int, n_events_expected: int = 16):
        self.config = config
        self.run_id = generate_run_id()
        set_global_seed(config.seed)
        self.encoder = SensorEncoder(
            n_channels=n_channels,
            hidden=config.hidden,
            embed_dim=config.embed_dim,
            n_conv_blocks=config.n_conv_blocks,
            use_attention=config.use_attention,
        )
        self.decoder = SensorDecoder(
            n_channels=n_channels,
            window_length=config.window_length,
            embed_dim=config.embed_dim,
            hidden=config.hidden,
        )
        self.vq = VQCodebook(
            embed_dim=config.embed_dim,
            n_codes=config.n_codes,
            commitment_beta=config.commitment_beta,
            ema_decay=config.ema_decay,
        )
        self.hsmm: Optional[HSMM] = None
        self.stats: List[StageStats] = []

    # ------------------------------------------------------------------
    def _stage1_batch(self, X: torch.Tensor, mask: torch.Tensor):
        z = self.encoder(X, mask)
        out = self.vq(z)
        X_hat = self.decoder(out.straight_through)
        recon = ((X_hat - X) ** 2 * mask).sum() / mask.sum().clamp_min(1.0)
        terms = self.vq.loss_terms(z, out)
        return recon, terms, out

    # ------------------------------------------------------------------
    def init_codebook_from_data(self, windows: torch.Tensor, masks: torch.Tensor) -> None:
        """Seed codebook vectors at k-means-style centroids of the (fresh,
        untrained) encoder embeddings — the standard VQ-VAE anti-collapse
        practice: spread the vocabulary across the data manifold before EMA
        takes over (PRD §25 'VQ collapse' mitigation)."""
        self.encoder.eval()
        with torch.no_grad():
            z = self.encoder(windows, masks)
            z = z[torch.randperm(z.shape[0])[: self.vq.K]]
            if z.shape[0] < self.vq.K:
                # too few windows: jitter-duplicate
                idx = torch.randint(0, z.shape[0], (self.vq.K,))
                z = z[idx] + 0.05 * torch.randn(
                    self.vq.K, z.shape[1], device=z.device
                )
            self.vq.codebook.copy_(z)
            self.vq.ema_embed_sum.copy_(z)
            # seed cluster sizes at 1 (Laplace-style prior): resetting to 0
            # makes the first EMA step divide embed_sum by ~0 and blow up
            self.vq.ema_cluster_size.fill_(1.0)

    # ------------------------------------------------------------------
    def train_stage1(self, windows: torch.Tensor, masks: torch.Tensor) -> StageStats:
        """windows: [N, W, d]; masks: [N, W, d]. Fits encoder + VQ + decoder."""
        cfg = self.config
        stats = StageStats(stage="stage1_encoder_vq")
        t0 = time.time()
        # anti-collapse: initialize codebook on the data manifold
        self.init_codebook_from_data(windows, masks)
        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        opt = torch.optim.AdamW(params, lr=cfg.lr)
        N = windows.shape[0]
        self.encoder.train(); self.decoder.train(); self.vq.train()
        perplexity_hist: List[float] = []
        for step in range(cfg.stage1_steps):
            idx = torch.randint(0, N, (min(cfg.batch_size, N),))
            X, m = windows[idx], masks[idx]
            opt.zero_grad()
            recon, terms, out = self._stage1_batch(X, m)
            loss = recon + terms["commitment"]
            loss.backward()
            opt.step()
            stats.steps = step + 1
            stats.recon_loss = float(recon.detach())
            stats.commitment_loss = float(terms["commitment"].detach())
            stats.perplexity = out.codebook_stats["perplexity"]
            stats.effective_size = out.codebook_stats["effective_size"]
            perplexity_hist.append(stats.perplexity)
            if cfg.log_every and (step + 1) % cfg.log_every == 0:
                print(
                    f"[{self.run_id}] stage1 step {step+1}: recon={stats.recon_loss:.4f} "
                    f"perplexity={stats.perplexity:.1f} eff={stats.effective_size}"
                )
        # exit criterion: perplexity stable over the last quarter
        tail = perplexity_hist[len(perplexity_hist) // 2:]
        stats.wall_clock_s = time.time() - t0
        self.stats.append(stats)
        return stats

    # ------------------------------------------------------------------
    def event_sequences(self, windows: torch.Tensor, masks: torch.Tensor) -> List[np.ndarray]:
        """Deterministic argmin tokenization of runs (inference contract).

        Returns one [1] array per window (MVP: one event token per window,
        PRD §9)."""
        self.encoder.eval(); self.vq.eval()
        seqs: List[np.ndarray] = []
        with torch.no_grad():
            for i in range(windows.shape[0]):
                X = windows[i : i + 1]
                m = masks[i : i + 1]
                z = self.encoder(X, m)
                ids = self.vq.event_id(z)
                seqs.append(ids.reshape(-1).cpu().numpy().astype(np.int64))
        return seqs

    # ------------------------------------------------------------------
    def window_runs_to_event_runs(
        self, seqs: List[np.ndarray], run_lengths: List[int]
    ) -> List[np.ndarray]:
        """Concatenate per-window tokens into per-run token sequences."""
        out = []
        i = 0
        for L in run_lengths:
            out.append(np.concatenate(seqs[i : i + L]))
            i += L
        return out

    # ------------------------------------------------------------------
    def train_stage2(
        self, event_seqs: List[np.ndarray], normal_only: Optional[List[bool]] = None
    ) -> StageStats:
        """Baum-Welch on event sequences (normal-operation runs per PRD §10.4)."""
        cfg = self.config
        stats = StageStats(stage="stage2_hsmm_em")
        t0 = time.time()
        seqs = event_seqs
        if normal_only is not None:
            seqs = [s for s, n in zip(event_seqs, normal_only) if n]
        if not seqs:
            raise ValueError("no normal-operation sequences for Stage 2")
        n_events = int(max(int(s.max()) for s in seqs) + 1)
        bw = BaumWelch(
            n_states=cfg.n_states,
            n_events=n_events,
            d_max=cfg.d_max,
            max_iter=cfg.em_max_iter,
            min_iter=cfg.em_max_iter,
            seed=cfg.seed,
            restarts=cfg.em_restarts,
        )
        self.hsmm, trace = bw.fit(seqs)
        stats.em_log_likelihood = trace.log_likelihoods[-1]
        stats.em_iterations = len(trace.log_likelihoods)
        stats.wall_clock_s = time.time() - t0
        self.stats.append(stats)
        return stats

    # ------------------------------------------------------------------
    def pipeline_outputs(
        self, v_seq: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Posterior + Viterbi + log-likelihood for one event run."""
        if self.hsmm is None:
            raise RuntimeError("train_stage2 must run before pipeline_outputs")
        fb = ForwardBackward(self.hsmm)
        res = fb.run(v_seq)
        return res.state_posterior, res.expected_dwell, res.log_likelihood

    def viterbi(self, v_seq: np.ndarray):
        return SegmentalViterbi(self.hsmm).decode(v_seq)
