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

import copy
import json
import sys
import time
from typing import Dict, List

import numpy as np
import torch
from scipy.stats import spearmanr
from torch import nn

from ..data import NCMAPSSLoader, SplitBuilder, UnitDataset
from ..preprocessing import Preprocessor
from ..training import Trainer, TrainerConfig
from ..grammar import ForwardBackward
from ..fault import FaultHead
from ..health import HealthHead, fit_learned_weights
from ..evaluation.stats import bootstrap_ci, paired_wilcoxon
from ..utils import set_global_seed
from .ablation_arms import _downsample


def run_heads_experiment(
    h5_path: str, n_seeds: int = 3, window_length: int = 4,
    n_states: int = 5, d_max: int = 10, n_codes: int = 16,
    stage1_steps: int = 300, em_max_iter: int = 10, em_restarts: int = 3,
    fault_target: str = "hs",
    verbose: bool = True,
) -> Dict:
    if fault_target not in ("hs", "fc"):
        raise ValueError(f"fault_target must be 'hs' or 'fc'; got {fault_target!r}")
    dataset = h5_path if isinstance(h5_path, UnitDataset) else NCMAPSSLoader().load(h5_path)
    dataset.records = [_downsample(r) for r in dataset.records]
    t0 = time.time()

    fault_acc_per_seed: List[float] = []
    fault_acc_per_unit: List[List[float]] = []   # [seed][unit]
    health_rho_per_seed: List[float] = []
    health_rho_per_unit: List[List[float]] = []  # [seed][unit]
    health_ab_per_seed: List[Dict] = []  # per-seed a/b details
    a3_per_seed: List[Dict] = []  # per-seed A3 joint-vs-single details
    fault_modes: List[str] = []  # per-seed: trained | degenerate_majority
    if fault_target == "fc":
        if any(r.fault_label is None for r in dataset.records):
            raise ValueError("fault_target='fc' requires fault_label on every unit")
        classes = sorted({int(r.fault_label) for r in dataset.records})
    else:
        classes = sorted({int(c) for r in dataset.records
                          if r.health_labels is not None
                          for c in np.unique(r.health_labels)})
    cfg = {
        "dataset": h5_path, "n_seeds": n_seeds, "window_length": window_length,
        "n_states": n_states, "d_max": d_max, "n_codes": n_codes,
        "stage1_steps": stage1_steps, "em_max_iter": em_max_iter,
        "em_restarts": em_restarts, "hs_classes": classes,
            "fault_labels": ("run-level Fc flight-envelope class (N-CMAPSS A-matrix col 3)"
                             if fault_target == "fc"
                             else "run-level hs class (N-CMAPSS auxiliary health state)"),
            "fault_mode_per_seed": fault_modes,
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

        # run-level fault labels: target chosen by `fault_target`
        #   "hs": majority health-state class over the run (MVP default)
        #   "fc": the unit's flight-envelope class (constant per unit)
        cls_index = {c: i for i, c in enumerate(classes)}

        def label_of(seqs, ids):
            if fault_target == "fc":
                return [cls_index[int(dataset[u].fault_label)] for u in ids]
            ys = []
            for s in seqs:
                hl = s.health_labels
                ys.append(cls_index[int(np.bincount(hl.astype(int)).argmax())])
            return ys

        y_tr, y_va, y_te = (label_of(tr, split.train_unit_ids),
                            label_of(va, split.val_unit_ids),
                            label_of(te, split.test_unit_ids))

        # ---------------- fault head (trained, val early-stop) -------------
        torch.manual_seed(seed)
        if len(set(y_tr)) < 2:
            # degenerate: train split has one class (e.g. DS01/DS02 run-level
            # majority hs is always 1 — early life dominates every run, and
            # DS02 dev units are all Fc=3) — constant majority predictor is
            # the honest maximum
            fault_mode = "degenerate_majority"
            pred = [y_tr[0]] * len(y_te)
        else:
            fault_mode = "trained"
            fault = FaultHead(n_states=n_states, n_classes=len(classes))
            opt = torch.optim.Adam(fault.parameters(), lr=1e-3)

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

        # ---------------- health heads: §14 a/b -----------------------
        # a) fixed ordinal prior (MVP default, no training)
        health_fixed = HealthHead(n_states=n_states)
        # b) learned weights fitted on TRAIN splits only (§14 option b)
        y_tr_windows = [s.health_labels for s in tr]
        health_learned = HealthHead(n_states=n_states, learn_weights=True)
        fit_learned_weights(health_learned, b_tr, y_tr_windows, seed=seed)

        def rho_units(head, bl, seql):
            out = []
            with torch.no_grad():
                for s, b in zip(seql, bl):
                    h = head(torch.as_tensor(b, dtype=torch.float32)).numpy()
                    hl = s.health_labels.astype(int)
                    if len(np.unique(hl)) < 2:
                        continue
                    out.append(float(spearmanr(h, hl).statistic))
            return out

        # decision: on validation only; test untouched for final report
        rho_fixed_va = rho_units(health_fixed, b_va, va)
        rho_learned_va = rho_units(health_learned, b_va, va)
        chosen_on_val = "learned" if float(np.mean(rho_learned_va)) > float(np.mean(rho_fixed_va)) else "fixed"
        rho_fixed_te = rho_units(health_fixed, b_te, te)
        rho_learned_te = rho_units(health_learned, b_te, te)

        health_ab_per_seed.append({
            "ab": "b" if chosen_on_val == "learned" else "a",
            "val_spearman_fixed": float(np.mean(rho_fixed_va)),
            "val_spearman_learned": float(np.mean(rho_learned_va)),
            "test_spearman_fixed": float(np.mean(rho_fixed_te)),
            "test_spearman_learned": float(np.mean(rho_learned_te)),
            "learned_weights": [float(w) for w in health_learned.weights.tolist()],
        })
        health_rho_per_seed.append(
            float(np.mean(rho_learned_te if chosen_on_val == "learned" else rho_fixed_te)))
        health_rho_per_unit.append(rho_units(
            health_learned if chosen_on_val == "learned" else health_fixed, b_te, te))

        # ---------------- A3: joint multi-task vs single-task copies --------
        # identical fresh heads per arm; joint trains all in one optimizer,
        # single-task trains each alone on the same losses/steps/lr (PRD §22).
        torch.manual_seed(seed + 1000)
        h_lo = min(float(np.asarray(s.health_labels).min()) for s in tr)
        h_hi = max(float(np.asarray(s.health_labels).max()) for s in tr)
        has_rul = all(s.rul_labels is not None for s in tr + te)

        def a3_fresh():
            return {
                "fault": FaultHead(n_states=n_states, n_classes=len(classes)),
                "health": HealthHead(n_states=n_states, learn_weights=True),
                "rul": nn.Linear(n_states, 1),
            }

        def a3_losses(heads, beliefs, seqs, ys_fault):
            out = {}
            if len(set(ys_fault)) >= 2:
                out["fault"] = lambda h: torch.nn.functional.cross_entropy(
                    torch.cat([h(torch.as_tensor(b[None], dtype=torch.float32))
                               for b in beliefs]),
                    torch.as_tensor(ys_fault, dtype=torch.long))
            if h_hi > h_lo:
                def hl(h):
                    preds = torch.cat([h(torch.as_tensor(b, dtype=torch.float32))
                                       for b in beliefs])
                    tgts = torch.cat([torch.as_tensor(
                        (h_hi - s.health_labels.astype(np.float64)) / (h_hi - h_lo),
                        dtype=torch.float32) for s in seqs])
                    return torch.nn.functional.mse_loss(preds, tgts)
                out["health"] = hl
            if has_rul:
                def rl(h):
                    preds = torch.cat([h(torch.as_tensor(b, dtype=torch.float32)).squeeze(-1)
                                       for b in beliefs])
                    tgts = torch.cat([torch.as_tensor(s.rul_labels.astype(np.float64),
                                      dtype=torch.float32) for s in seqs])
                    return torch.nn.functional.huber_loss(preds, tgts)
                out["rul"] = rl
            return out

        def a3_eval(heads, beliefs, seqs, ys):
            out = {}
            if len(set(y_tr)) >= 2:
                pred = [int(heads["fault"](torch.as_tensor(b[None], dtype=torch.float32))
                            .argmax(dim=-1).item()) for b in beliefs]
                out["fault_acc"] = float(np.mean([p == y for p, y in zip(pred, ys)]))
            if h_hi > h_lo:
                rhos = []
                with torch.no_grad():
                    for b, s in zip(beliefs, seqs):
                        h = heads["health"](torch.as_tensor(b, dtype=torch.float32)).numpy()
                        hl = s.health_labels.astype(int)
                        if len(np.unique(hl)) >= 2:
                            rhos.append(float(spearmanr(h, hl).statistic))
                out["health_spearman"] = float(np.mean(rhos)) if rhos else float("nan")
            if has_rul:
                with torch.no_grad():
                    pr = torch.cat([heads["rul"](torch.as_tensor(b, dtype=torch.float32))
                                    .squeeze(-1) for b in beliefs]).numpy()
                tg = np.concatenate([s.rul_labels.astype(np.float64) for s in seqs])
                out["rul_rmse"] = float(np.sqrt(np.mean((pr - tg) ** 2)))
            return out

        base_heads = a3_fresh()
        joint_heads = copy.deepcopy(base_heads)
        single_heads = copy.deepcopy(base_heads)
        joint_losses = a3_losses(joint_heads, b_tr, tr, y_tr)
        Trainer.stage3_joint(joint_heads, joint_losses, steps=150, lr=5e-3)
        for name, fn in a3_losses(single_heads, b_tr, tr, y_tr).items():
            Trainer.stage3_joint({name: single_heads[name]}, {name: fn},
                                 steps=150, lr=5e-3)
        a3_per_seed.append({
            "tasks": sorted(joint_losses),
            "joint": a3_eval(joint_heads, b_te, te, y_te),
            "single_task": a3_eval(single_heads, b_te, te, y_te),
        })
        fault_modes.append(fault_mode)

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

    def a3_agg():
        tasks = sorted({m for s in a3_per_seed for m in s["joint"]})
        joint = {t: float(np.mean([s["joint"][t] for s in a3_per_seed]))
                 for t in tasks}
        single = {t: float(np.mean([s["single_task"][t] for s in a3_per_seed]))
                  for t in tasks}
        return {"joint_mean": joint, "single_task_mean": single,
                "delta_joint_minus_single": {t: joint[t] - single[t] for t in tasks}}

    report = {
        "experiment": "fault_health_heads_m7",
        "config": cfg,
        "results": {
            "fault_accuracy": agg(fault_acc_per_seed, fault_acc_per_unit),
            "health_spearman_vs_hs": agg(health_rho_per_seed, health_rho_per_unit),
        },
        "multitask_a3": {
            "note": "A3 (PRD §22): Stage-3 joint multi-task head training vs "
                    "independent single-task copies, identical heads/steps/lr, "
                    "frozen beliefs. delta > 0 means joint wins (for accuracy/"
                    "spearman; for rul_rmse delta < 0 means joint wins). "
                    "Anomaly task absent: no anomaly labels on N-CMAPSS. "
                    "NOTE: heads share NO trainable parameters and the "
                    "encoder/VQ/HSMM stay frozen, so joint ≡ single-task "
                    "exactly (per-parameter adaptive updates decouple) — the "
                    "§15 multi-task coupling only bites in the full-gradient "
                    "Stage-3 variant (encoder flow), which is the research-grade "
                    "extension this result motivates.",
            "per_seed": a3_per_seed,
            **a3_agg(),
        },
        "health_ab_decision": {
            "note": "§14: a=fixed ordinal prior evaluated; b=learned "
                    "monotone-regularized weights trained on train units "
                    "only; variant chosen by validation Spearman before "
                    "final test eval.",
            "per_seed": health_ab_per_seed,
            "majority": (
                "b" if sum(1 for s in health_ab_per_seed if s["ab"] == "b")
                       >= (len(health_ab_per_seed) + 1) // 2 else "a"
            ),
        },
        "notes": f"Fault: run-level {fault_target} class from belief trajectory "
                 "+ duration stats (no raw-embedding bypass, PRD Decision 5). "
                 "Health: §14 a/b resolved on VALIDATION split (Spearman), "
                 "winner reported on test. Unit-level bootstrap CIs per §23; "
                 f"only {len(split.test_unit_ids)} test units → CIs wide.",
        "wall_clock_s": time.time() - t0,
    }
    return report


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "N-CMAPSS/N-CMAPSS DS01 005.h5"
    fault_target = sys.argv[2] if len(sys.argv) > 2 else "hs"
    rep = run_heads_experiment(path, fault_target=fault_target)
    if fault_target == "hs":
        out = f"heads_report_{path.split()[-2]}.json"
    else:
        out = f"heads_report_{path.split()[-2]}_fc.json"
    with open(out, "w") as f:
        json.dump(rep, f, indent=2)
    print(f"-> {out}")
    for k, v in rep["results"].items():
        print(f"{k:24s} {v['mean']:.3f}  unit_ci95 {v['unit_ci95']}")
