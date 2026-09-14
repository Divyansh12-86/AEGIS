"""M7 real-data run: fault + health heads on N-CMAPSS (tuned protocol).

Same protocol as the A–F ablation: unit-level splits, train-only
normalization, 3 seeds, per-unit stats + bootstrap CIs (egpm.evaluation.stats).

Fault head: run-level hs class (0/1) from belief trajectory + duration
stats (egpm.fault.FaultHead, NO raw-embedding bypass).
Health head: fixed ordinal severity weights over state posterior
(egpm.health.HealthHead); validity checked via per-unit Spearman
correlation with hs labels (PRD §14: weights NOT assumed valid).
"""
from __future__ import annotations

import json
import sys
import time
from typing import Dict, List

import numpy as np
import torch
from scipy.stats import spearmanr

from ..data import NCMAPSSLoader, SplitBuilder, UnitDataset
from ..preprocessing import Preprocessor
from ..training import Trainer, TrainerConfig
from ..grammar import ForwardBackward
from ..fault import FaultHead
from ..health import HealthHead
from ..evaluation.stats import bootstrap_ci, paired_wilcoxon
from ..utils import set_global_seed
from .ablation_arms import _downsample


def run_heads_experiment(
    h5_path: str, n_seeds: int = 3, window_length: int = 4,
    n_states: int = 5, d_max: int = 10, n_codes: int = 16,
    stage1_steps: int = 300, em_max_iter: int = 10, em_restarts: int = 3,
    verbose: bool = True,
) -> Dict:
    dataset = h5_path if isinstance(h5_path, UnitDataset) else NCMAPSSLoader().load(h5_path)
    dataset.records = [_downsample(r) for r in dataset.records]
    t0 = time.time()

    fault_acc_per_seed: List[float] = []
    fault_acc_per_unit: List[List[float]] = []   # [seed][unit]
    health_rho_per_seed: List[float] = []
    health_rho_per_unit: List[List[float]] = []  # [seed][unit]
    classes = sorted({int(c) for r in dataset.records
                      if r.health_labels is not None
                      for c in np.unique(r.health_labels)})
    cfg = {
        "dataset": h5_path, "n_seeds": n_seeds, "window_length": window_length,
        "n_states": n_states, "d_max": d_max, "n_codes": n_codes,
        "stage1_steps": stage1_steps, "em_max_iter": em_max_iter,
        "em_restarts": em_restarts, "hs_classes": classes,
        "fault_labels": "run-level hs class (N-CMAPSS auxiliary health state)",
        "health_validation": "per-unit Spearman(health_index, hs_label), PRD §14",
    }

    for seed in range(n_seeds):
        set_global_seed(seed)
        sb = SplitBuilder(dataset, seed=0)   # fixed split: same units as A–F
        split = sb.build()
        stats = sb.fit_normalization(split)
        prep = Preprocessor(window_length=window_length)

        tr = [prep.process(dataset[u], norm_stats=stats) for u in split.train_unit_ids]
        va = [prep.process(dataset[u], norm_stats=stats) for u in split.val_unit_ids]
        te = [prep.process(dataset[u], norm_stats=stats) for u in split.test_unit_ids]

        Xtr = torch.as_tensor(np.concatenate([s.X for s in tr]), dtype=torch.float32)
        Mtr = torch.ones_like(Xtr)
        run_lens = [len(s.X) for s in tr]

        tcfg = TrainerConfig(
            window_length=window_length, embed_dim=16, hidden=32,
            n_conv_blocks=2, n_codes=n_codes, d_max=d_max,
            n_states=n_states, em_max_iter=em_max_iter,
            em_restarts=em_restarts, stage1_steps=stage1_steps,
            batch_size=64, seed=seed, log_every=0,
        )
        trainer = Trainer(tcfg, n_channels=Xtr.shape[-1])
        trainer.train_stage1(Xtr, Mtr)
        ev_tr = trainer.window_runs_to_event_runs(
            trainer.event_sequences(Xtr, Mtr), run_lens)

        def belief_runs(seqs):
            fb = ForwardBackward(trainer.hsmm)
            out = []
            for s in seqs:
                X = torch.as_tensor(s.X, dtype=torch.float32)
                v = np.concatenate(trainer.event_sequences(X, torch.ones_like(X)))
                out.append(fb.run(v).state_posterior)   # [T, M]
            return out

        trainer.train_stage2(ev_tr)
        b_tr, b_va, b_te = belief_runs(tr), belief_runs(va), belief_runs(te)

        # run-level fault labels: hs class (constant within early life;
        # majority vote over the run), mapped to contiguous class indices
        cls_index = {c: i for i, c in enumerate(classes)}

        def label_of(seqs):
            ys = []
            for s in seqs:
                hl = s.health_labels
                ys.append(cls_index[int(np.bincount(hl.astype(int)).argmax())])
            return ys

        y_tr, y_va, y_te = label_of(tr), label_of(va), label_of(te)

        # ---------------- fault head (trained, val early-stop) -------------
        torch.manual_seed(seed)
        fault = FaultHead(n_states=n_states, n_classes=len(classes))
        opt = torch.optim.Adam(fault.parameters(), lr=1e-3)
        yt = torch.as_tensor(y_tr, dtype=torch.long)

        def fault_loss(feats, ys):
            logits = torch.cat([
                fault(torch.as_tensor(b[None], dtype=torch.float32))
                for b in feats
            ])
            return torch.nn.functional.cross_entropy(
                logits, torch.as_tensor(ys, dtype=torch.long))

        best, best_state, wait = float("inf"), None, 0
        for _ in range(100):
            fault.train()
            opt.zero_grad()
            loss = fault_loss(b_tr, y_tr)
            loss.backward(); opt.step()
            fault.eval()
            with torch.no_grad():
                vv = float(fault_loss(b_va, y_va))
            if vv < best - 1e-4:
                best, wait = vv, 0
                best_state = {k: t.clone() for k, t in fault.state_dict().items()}
            else:
                wait += 1
                if wait >= 10:
                    break
        fault.load_state_dict(best_state)
        fault.eval()
        pred = []
        with torch.no_grad():
            for b in b_te:
                logits = fault(torch.as_tensor(b[None], dtype=torch.float32))
                pred.append(int(logits.argmax(dim=-1).item()))
        # per-unit accuracy: each test unit is one sample (run-level label)
        acc_units = [float(p == y) for p, y in zip(pred, y_te)]
        fault_acc_per_seed.append(float(np.mean(acc_units)))
        fault_acc_per_unit.append(acc_units)

        # ---------------- health head (fixed weights, no training) ---------
        health = HealthHead(n_states=n_states)
        rho_units = []
        with torch.no_grad():
            for s, b in zip(te, b_te):
                h = health(torch.as_tensor(b, dtype=torch.float32)).numpy()
                hl = s.health_labels.astype(int)
                if len(np.unique(hl)) < 2:
                    continue
                rho_units.append(float(spearmanr(h, hl).statistic))
        health_rho_per_seed.append(float(np.mean(rho_units)))
        health_rho_per_unit.append(rho_units)

        if verbose:
            print(f"seed {seed}: fault_acc={fault_acc_per_seed[-1]:.3f} "
                  f"health_spearman={health_rho_per_seed[-1]:.3f}", flush=True)

    def agg(per_seed, per_unit):
        u = np.mean(np.array(per_unit), axis=0)  # units shared across seeds
        lo, hi = bootstrap_ci(u, seed=0)
        return {
            "mean": float(np.mean(per_seed)),
            "std_over_seeds": float(np.std(per_seed)),
            "per_seed": per_seed,
            "unit_mean": float(np.mean(u)),
            "unit_ci95": [lo, hi],
            "per_unit": u.tolist(),
            "n_units": int(len(u)),
        }

    report = {
        "experiment": "fault_health_heads_m7",
        "config": cfg,
        "results": {
            "fault_accuracy": agg(fault_acc_per_seed, fault_acc_per_unit),
            "health_spearman_vs_hs": agg(health_rho_per_seed, health_rho_per_unit),
        },
        "notes": "Fault: run-level hs class from belief trajectory + duration "
                 "stats (no raw-embedding bypass, PRD Decision 5). Health: "
                 "fixed ordinal severity weights validated against hs labels "
                 "(PRD §14 — validity is checked, not assumed). Unit-level "
                 "bootstrap CIs per §23; only "
                 f"{len(split.test_unit_ids)} test units → CIs wide.",
        "wall_clock_s": time.time() - t0,
    }
    return report


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "N-CMAPSS/N-CMAPSS DS01 005.h5"
    rep = run_heads_experiment(path)
    out = f"heads_report_{path.split()[-2]}.json"
    with open(out, "w") as f:
        json.dump(rep, f, indent=2)
    print(f"-> {out}")
    for k, v in rep["results"].items():
        print(f"{k:24s} {v['mean']:.3f}  unit_ci95 {v['unit_ci95']}")
