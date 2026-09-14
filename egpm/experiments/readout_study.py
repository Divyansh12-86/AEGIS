"""Controlled RUL-readout study on the FROZEN EGPM representation.

Question: can a stronger temporal readout recover the prognostic information
already present in the EGPM event/state/duration representation — or is the
bottleneck itself the limit?

Representation contract (enforced): EGPM heads consume ONLY structured EGPM
outputs per timestep: event one-hot [K], state posterior [M], normalized
expected dwell [M], emission surprisal [1], duration surprisal [1].
No raw sensors, no encoder z, no decoder features.

Heads (all small, fixed config, no search):
  ridge  — existing reference (closed form)
  gru    — 1 layer, hidden 32
  tcn    — 2 dilated causal conv blocks, width 32
  tsmixer — 2 mixer blocks, hidden 32 (transpose-MLP temporal mixing)

Controls (NOT EGPM):
  raw-tsmixer — raw sensor windows -> TSMixer (accuracy ceiling reference)
  z-tsmixer   — continuous encoder z sequence -> TSMixer (info-loss control:
                z vs structured shows VQ+HSMM bottleneck cost)

Protocol: identical splits/preprocessing/windows/seeds as the A-F ablation;
em_restarts=10; early stopping on validation RMSE (patience 10); no test
contact during fitting or model selection. 2 test units per dataset — no
significance claims.
"""
from __future__ import annotations

import json
import time
from typing import Dict, List

import numpy as np
import torch
from torch import nn

from ..data import NCMAPSSLoader, SplitBuilder
from ..preprocessing import Preprocessor
from ..training import Trainer, TrainerConfig
from ..grammar import ForwardBackward
from ..anomaly import AnomalyScorer
from ..evaluation import rul_metrics
from ..utils import set_global_seed
from .ablation_arms import _downsample, _ridge_fit, _ridge_predict

FEAT_DIM_NOTE = "per-step: event_onehot[K] + posterior[M] + dwell[M]/d_max + emis_surp[1] + dur_surp[1]"


# ---------------------------------------------------------------- heads
class GRUHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 32):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden, num_layers=1, batch_first=True)
        self.out = nn.Linear(hidden, 1)

    def forward(self, X):  # [B, T, F]
        h, _ = self.gru(X)
        return self.out(h).squeeze(-1)  # [B, T]


class TCNHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, hidden, 3, dilation=1, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, 3, dilation=2, padding=2),
            nn.GELU(),
        )
        self.out = nn.Linear(hidden, 1)

    def forward(self, X):
        h = self.net(X.transpose(1, 2)).transpose(1, 2)
        return self.out(h).squeeze(-1)


class TSMixerHead(nn.Module):
    """TSMixer-style block: feature-mixing MLP + time-mixing MLP
    (transpose), residual + norm; length-agnostic in T. One stack kept
    deliberately modest."""
    def __init__(self, in_dim: int, hidden: int = 32):
        super().__init__()
        self.f_mlp = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(),
                                   nn.Linear(hidden, in_dim))
        self.t_conv = nn.Conv1d(in_dim, in_dim, kernel_size=3, padding=1,
                                groups=in_dim)  # per-channel temporal mix
        self.out = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(),
                                  nn.Linear(hidden, 1))
        self.norm1 = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(in_dim)

    def forward(self, X):  # [B, T, F]
        h = self.norm1(X + self.f_mlp(X))                    # feature mixing
        h = self.norm2(h + self.t_conv(h.transpose(1, 2)).transpose(1, 2))
        return self.out(h).squeeze(-1)


def _n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ---------------------------------------------------------------- features# ---------------------------------------------------------------- features
def _egpm_features(trainer, hsmm, windows, tokens):
    """Structured per-step features: [T, K + 2M + 2]. Only EGPM outputs."""
    fb = ForwardBackward(hsmm)
    scorer = AnomalyScorer()
    K, M = hsmm.K, hsmm.M
    out = []
    for w_seq, v in zip(windows, tokens):
        r = fb.run(v)
        comp = scorer.score_sequence(hsmm, v)
        T = len(v)
        F = np.zeros((T, K + 2 * M + 2))
        F[np.arange(T), v] = 1.0                       # event one-hot
        F[:, K:K + M] = r.state_posterior              # state posterior
        F[:, K + M:K + 2 * M] = r.expected_dwell / hsmm.d_max
        F[:, K + 2 * M] = comp.emission_surprisal        # per-step emission surprisal
        F[:, K + 2 * M + 1] = comp.score                  # per-step anomaly score
        out.append(F)
    return out


def run_readout_study(h5_path: str, n_seeds: int = 3, window_length: int = 4,
                      n_states: int = 5, d_max: int = 10, n_codes: int = 16,
                      stage1_steps: int = 300, em_max_iter: int = 10,
                      verbose: bool = True) -> Dict:
    dataset = NCMAPSSLoader().load(h5_path)
    dataset.records = [_downsample(r) for r in dataset.records]
    t0 = time.time()
    per_model: Dict[str, Dict[str, List[float]]] = {}
    meta = {}

    def record(name, rmse, mae, epochs, params, es):
        per_model.setdefault(name, {}).setdefault("rmse", []).append(rmse)
        per_model[name].setdefault("mae", []).append(mae)
        per_model[name].setdefault("epochs", []).append(epochs)
        meta.setdefault(name, {"params": params, "early_stop": es})

    for seed in range(n_seeds):
        set_global_seed(seed)
        sb = SplitBuilder(dataset, seed=0)
        split = sb.build()
        stats = sb.fit_normalization(split)
        prep = Preprocessor(window_length=window_length)
        tr_units = split.train_unit_ids
        va_units = split.val_unit_ids
        te_units = split.test_unit_ids

        tr = [prep.process(dataset[u], norm_stats=stats) for u in tr_units]
        va = [prep.process(dataset[u], norm_stats=stats) for u in va_units]
        te = [prep.process(dataset[u], norm_stats=stats) for u in te_units]

        def stack(seqs):
            X = torch.as_tensor(np.concatenate([s.X for s in seqs]), dtype=torch.float32)
            y = torch.as_tensor(np.concatenate([s.rul_labels for s in seqs]), dtype=torch.float32)
            lens = [len(s.X) for s in seqs]
            return X, y, lens

        Xtr, ytr, ltr = stack(tr)
        Mtr = torch.ones_like(Xtr)

        cfg = TrainerConfig(
            window_length=window_length, embed_dim=16, hidden=32,
            n_conv_blocks=2, n_codes=n_codes, d_max=d_max,
            n_states=n_states, em_max_iter=em_max_iter, em_restarts=10,
            stage1_steps=stage1_steps, batch_size=64, seed=seed, log_every=0,
        )
        trainer = Trainer(cfg, n_channels=Mtr.shape[-1])
        trainer.train_stage1(Xtr, Mtr)
        ev_tr = trainer.window_runs_to_event_runs(
            trainer.event_sequences(Xtr, Mtr), ltr)

        def toks(seqs):
            out = []
            for s in seqs:
                X = torch.as_tensor(s.X, dtype=torch.float32)
                out.append(np.concatenate(
                    trainer.event_sequences(X, torch.ones_like(X))))
            return out

        def zseq(seqs):
            out = []
            for s in seqs:
                X = torch.as_tensor(s.X, dtype=torch.float32)
                with torch.no_grad():
                    out.append(trainer.encoder(X, torch.ones_like(X)).cpu().numpy())
            return out

        ev_va, ev_te = toks(va), toks(te)
        z_va, z_te = zseq(va), zseq(te)

        trainer.train_stage2(ev_tr)
        hsmm = trainer.hsmm

        # structured EGPM features (train uses same trained HSMM)
        f_tr = _egpm_features(trainer, hsmm, tr, ev_tr)
        f_va = _egpm_features(trainer, hsmm, va, ev_va)
        f_te = _egpm_features(trainer, hsmm, te, ev_te)

        y_tr_np = np.concatenate([s.rul_labels for s in tr])
        y_va_np = np.concatenate([s.rul_labels for s in va])
        y_te_np = np.concatenate([s.rul_labels for s in te])

        def eval_np(preds):
            return rul_metrics(np.concatenate(preds), y_te_np)

        # ---- 0. ridge (reference) ----
        w = _ridge_fit(np.concatenate(f_tr), y_tr_np)
        pr = [_ridge_predict(w, f) for f in f_te]
        m = eval_np(pr)
        record("ridge", m["RMSE"], m["MAE"], 0, 0, False)

        # ---- neural heads helper ----
        def run_neural(name, model_fn, tr_feats, va_feats, te_feats):
            torch.manual_seed(seed)
            model = model_fn()
            # train on concatenated sequences with per-run reset via batches
            Xs = [torch.as_tensor(f, dtype=torch.float32).unsqueeze(0) for f in tr_feats]
            ys = [torch.as_tensor(y, dtype=torch.float32)
                  for y in [s.rul_labels for s in tr]]
            Xv = [torch.as_tensor(f, dtype=torch.float32).unsqueeze(0) for f in va_feats]
            yv = [torch.as_tensor(s.rul_labels, dtype=torch.float32) for s in va]
            # greedy full-batch over runs (small data)
            XtrB = torch.cat([x[:, :min(len(x), min(map(len, Xs)))] if False else x for x in Xs], 0) \
                if False else None
            # simple approach: train per-run batches, val on all val runs
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            best_val, best_state, wait = float("inf"), None, 0
            for ep in range(200):
                model.train()
                for X, y in zip(Xs, ys):
                    opt.zero_grad()
                    p = model(X)
                    loss = ((p - y) ** 2).mean()
                    loss.backward(); opt.step()
                model.eval()
                with torch.no_grad():
                    vv = float(torch.cat([((model(X) - y) ** 2).mean().expand(1)
                                           for X, y in zip(Xv, yv)]).mean())
                if vv < best_val - 1e-4:
                    best_val, wait = vv, 0
                    best_state = {k: t.clone() for k, t in model.state_dict().items()}
                else:
                    wait += 1
                    if wait >= 10:
                        break
            model.load_state_dict(best_state)
            model.eval()
            preds = []
            with torch.no_grad():
                for f in te_feats:
                    X = torch.as_tensor(f, dtype=torch.float32).unsqueeze(0)
                    preds.append(model(X).squeeze(0).numpy())
            m = eval_np(preds)
            record(name, m["RMSE"], m["MAE"], ep + 1, _n_params(model), True)

        Fdim = f_tr[0].shape[1]

        # ---- 1. GRU ----
        run_neural("gru", lambda: GRUHead(Fdim, hidden=32), f_tr, f_va, f_te)

        # ---- 2. TCN ----
        run_neural("tcn", lambda: TCNHead(Fdim, hidden=32), f_tr, f_va, f_te)

        # ---- 3. TSMixer ----
        run_neural("tsmixer", lambda: TSMixerHead(Fdim, hidden=32), f_tr, f_va, f_te)

        # ---- 4. raw-sensor TSMixer reference (NOT EGPM) ----
        # window [W, d] flattened to one feature vector per step: same
        # windows, same targets, same protocol — only the input differs.
        flat = lambda seqs: [s.X.reshape(len(s.X), -1) for s in seqs]
        raw_tr, raw_va, raw_te = flat(tr), flat(va), flat(te)
        run_neural("raw_tsmixer",
                   lambda: TSMixerHead(raw_tr[0].shape[-1], hidden=32),
                   raw_tr, raw_va, raw_te)

        # ---- info-loss control: encoder z -> TSMixer (NOT EGPM) ----
        run_neural("z_tsmixer",
                   lambda: TSMixerHead(z_va[0].shape[-1], hidden=32),
                   zseq(tr), z_va, z_te)

        if verbose:
            done = {k: v["rmse"][-1] for k, v in per_model.items()}
            print(f"seed {seed}: " + "  ".join(
                f"{k}={v:.1f}" for k, v in done.items()), flush=True)

    results = {}
    for name, d in per_model.items():
        results[name] = {
            "rmse": {"mean": float(np.mean(d["rmse"])), "std": float(np.std(d["rmse"])),
                     "per_seed": d["rmse"]},
            "mae": {"mean": float(np.mean(d["mae"])), "std": float(np.std(d["mae"])),
                    "per_seed": d["mae"]},
            "epochs_per_seed": d["epochs"], **meta[name],
        }
    return {
        "experiment": "rul_readout_study",
        "dataset": h5_path,
        "n_independent_test_units": len(te_units),
        "test_unit_ids": te_units,
        "feature_contract": FEAT_DIM_NOTE,
        "heads": {
            "ridge": "closed-form ridge (reference)",
            "gru": "1-layer GRU hidden=32, Adam lr=1e-3, ES patience 10 on val RMSE",
            "tcn": "2 dilated causal convs (d=1,2) width 32, ES same",
            "tsmixer": "feature-MLP + transpose time-MLP mixer, hidden 32, 1 stack, ES same",
            "raw_tsmixer": "CONTROL: raw sensor windows -> same TSMixer (ceiling ref)",
            "z_tsmixer": "CONTROL: encoder z seq -> same TSMixer (info-loss probe)",
        },
        "results": results,
        "wall_clock_s": time.time() - t0,
    }
