"""Causal re-run of dwell-only vs full features for N-CMAPSS DS02 (and DS01 for completeness)."""
from __future__ import annotations

import time
import numpy as np
import torch
from typing import List, Dict

from egpm.data.loaders import NCMAPSSLoader
from egpm.data.split_builder import SplitBuilder
from egpm.preprocessing.preprocessor import Preprocessor
from egpm.training.trainer import Trainer, TrainerConfig
from egpm.grammar.forward_backward import ForwardBackward
from egpm.evaluation.stats import per_unit_metrics
from egpm.evaluation.evaluator import rul_metrics
from egpm.utils import set_global_seed
from egpm.experiments.ablation_arms import (
    _downsample, _ridge_fit, _ridge_predict, rul_metrics
)


def run_dwell_vs_full_causal(
    h5_path: str,
    n_seeds: int = 3,
    window_length: int = 4,
    n_states: int = 5,
    d_max: int = 10,
    n_codes: int = 16,
    stage1_steps: int = 300,
    em_max_iter: int = 10,
    em_restarts: int = 3,
) -> Dict:
    """Run dwell-only vs full features under causal filtering."""
    dataset = NCMAPSSLoader().load(h5_path)
    dataset.records = [_downsample(r) for r in dataset.records]

    arms = ["dwell_only", "full_posterior_dwell"]
    per_arm_seed: Dict[str, Dict[str, List[float]]] = {a: {} for a in arms}

    for seed in range(n_seeds):
        set_global_seed(seed)
        sb = SplitBuilder(dataset, seed=0)
        split = sb.build()
        stats = sb.fit_normalization(split)
        prep = Preprocessor(window_length=window_length)

        train = [prep.process(dataset[u], norm_stats=stats) for u in split.train_unit_ids]
        test = [prep.process(dataset[u], norm_stats=stats) for u in split.test_unit_ids]
        Xtr = torch.as_tensor(np.concatenate([s.X for s in train]), dtype=torch.float32)
        Mtr = torch.ones_like(Xtr)
        run_lens = [len(s.X) for s in train]
        y_tr = np.concatenate([s.rul_labels for s in train])

        tcfg = TrainerConfig(
            window_length=window_length, embed_dim=16, hidden=32,
            n_conv_blocks=2, n_codes=n_codes, d_max=d_max,
            n_states=n_states, em_max_iter=em_max_iter,
            em_restarts=em_restarts, stage1_steps=stage1_steps,
            batch_size=64, seed=seed, log_every=0,
        )
        trainer = Trainer(tcfg, n_channels=Xtr.shape[-1])
        t0_all = time.time()
        trainer.train_stage1(Xtr, Mtr)
        
        ev_train = trainer.window_runs_to_event_runs(
            trainer.event_sequences(Xtr, Mtr), run_lens
        )

        def toks(seq_list):
            out = []
            for s in seq_list:
                X = torch.as_tensor(s.X, dtype=torch.float32)
                out.append(np.concatenate(trainer.event_sequences(X, torch.ones_like(X))))
            return out

        ev_test = toks(test)
        trainer.train_stage2(ev_train)
        hsmm = trainer.hsmm

        # CAUSAL features
        def fb_feats_causal(h, runs):
            out = []
            for v in runs:
                r = ForwardBackward(h).causal_filter(v)  # CAUSAL
                out.append(np.concatenate([r.state_posterior,
                                           r.expected_dwell / h.d_max], axis=1))
            return out

        def fb_dwell_only_causal(h, runs):
            out = []
            for v in runs:
                r = ForwardBackward(h).causal_filter(v)  # CAUSAL
                out.append(r.expected_dwell / h.d_max)
            return out

        full_feat_train = fb_feats_causal(hsmm, ev_train)
        full_feat_test = fb_feats_causal(hsmm, ev_test)
        dwell_feat_train = fb_dwell_only_causal(hsmm, ev_train)
        dwell_feat_test = fb_dwell_only_causal(hsmm, ev_test)

        def fit_and_eval(feat_train, feat_test, arm):
            w = _ridge_fit(np.concatenate(feat_train), y_tr)
            preds, trues = [], []
            for i, s in enumerate(test):
                p = _ridge_predict(w, feat_test[i])
                preds.append(p); trues.append(s.rul_labels)
            m = rul_metrics(np.concatenate(preds), np.concatenate(trues))
            per_arm_seed[arm].setdefault("rmse", []).append(m["RMSE"])
            per_arm_seed[arm].setdefault("mae", []).append(m["MAE"])
            per_arm_seed[arm].setdefault("rmse_per_unit", []).append(
                per_unit_metrics(preds, trues).tolist()
            )

        t_arm = time.time()
        fit_and_eval(full_feat_train, full_feat_test, "full_posterior_dwell")

        t_arm = time.time()
        fit_and_eval(dwell_feat_train, dwell_feat_test, "dwell_only")

    results = {}
    for arm in arms:
        auc = np.array(per_arm_seed[arm].get("rmse", []))
        apc = np.array(per_arm_seed[arm].get("mae", []))
        results[arm] = {
            "rmse": {"mean": float(auc.mean()), "std": float(auc.std()), "per_seed": per_arm_seed[arm].get("rmse", [])},
            "mae": {"mean": float(apc.mean()), "std": float(apc.std()), "per_seed": per_arm_seed[arm].get("mae", [])},
        }
        if "rmse_per_unit" in per_arm_seed[arm]:
            results[arm]["rmse_per_unit"] = {
                "mean": float(np.mean(per_arm_seed[arm]["rmse_per_unit"])),
                "per_seed": per_arm_seed[arm]["rmse_per_unit"],
            }
    return results


if __name__ == "__main__":
    import json
    
    print("=== Running causal dwell vs full for DS01 ===")
    ds01_result = run_dwell_vs_full_causal("N-CMAPSS/N-CMAPSS DS01 005.h5")
    for arm, data in ds01_result.items():
        print(f"  {arm}: RMSE={data['rmse']['mean']:.4f} (±{data['rmse']['std']:.4f})")
    
    print("\n=== Running causal dwell vs full for DS02 ===")
    ds02_result = run_dwell_vs_full_causal("N-CMAPSS/N-CMAPSS DS02 006.h5")
    for arm, data in ds02_result.items():
        print(f"  {arm}: RMSE={data['rmse']['mean']:.4f} (±{data['rmse']['std']:.4f})")
    
    with open("causal_dwell_vs_full_results.json", "w") as f:
        json.dump({"DS01": ds01_result, "DS02": ds02_result}, f, indent=2)
    print("\nSaved to causal_dwell_vs_full_results.json")