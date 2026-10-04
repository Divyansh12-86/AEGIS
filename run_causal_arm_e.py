"""Causal re-run of Arm E (VQ+HSMM+ridge) for N-CMAPSS DS01 and DS02.
Uses ForwardBackward.causal_filter() instead of ForwardBackward.run() (smoothing)."""
from __future__ import annotations

import time
import numpy as np
import torch

from egpm.data.loaders import NCMAPSSLoader
from egpm.data.split_builder import SplitBuilder
from egpm.preprocessing.preprocessor import Preprocessor
from egpm.training.trainer import Trainer, TrainerConfig
from egpm.grammar.forward_backward import ForwardBackward
from egpm.evaluation.stats import per_unit_metrics
from egpm.utils import set_global_seed
from egpm.evaluation.evaluator import rul_metrics
from egpm.experiments.ablation_arms import _downsample, _ridge_fit, _ridge_predict, _onsets, temporal_validity


def run_arm_e_causal(
    h5_path: str,
    n_seeds: int = 3,
    window_length: int = 4,
    n_states: int = 5,
    d_max: int = 10,
    n_codes: int = 16,
    stage1_steps: int = 300,
    em_max_iter: int = 10,
    em_restarts: int = 3,
) -> dict:
    """Run Arm E (VQ+HSMM+ridge) with causal filtering only."""
    dataset = NCMAPSSLoader().load(h5_path)
    dataset.records = [_downsample(r) for r in dataset.records]

    per_seed_rmse = []
    per_seed_mae = []
    per_seed_rmse_per_unit = []
    per_seed_onset_detection = []
    per_seed_onset_lag = []
    per_seed_runtime = []

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
        t0 = time.time()
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

        # Train HSMM (stage 2)
        trainer.train_stage2(ev_train)
        hsmm_E = trainer.hsmm

        # CAUSAL features: posterior + expected-dwell using causal_filter (forward-only)
        def fb_feats_causal(h, runs):
            out = []
            for v in runs:
                r = ForwardBackward(h).causal_filter(v)  # <-- CAUSAL
                out.append(np.concatenate([r.state_posterior,
                                           r.expected_dwell / h.d_max], axis=1))
            return out

        e_feat_train = fb_feats_causal(hsmm_E, ev_train)
        e_feat_test = fb_feats_causal(hsmm_E, ev_test)

        def onset_extra_E(h):
            def f(i):
                onsets = _onsets(test[i].health_labels)
                if len(onsets) == 0:
                    return np.nan, np.nan
                from egpm.grammar import SegmentalViterbi
                vit = SegmentalViterbi(h).decode(ev_test[i])
                bnd = [a for (_, a, _) in vit.segments if 0 < a < len(vit.path) - 1]
                tv = temporal_validity(bnd, onsets, tolerance=max(2, h.d_max // 2))
                return tv["detection_rate"], tv["mean_abs_lag"]
            return f

        w = _ridge_fit(np.concatenate(e_feat_train), y_tr)
        preds, trues, det, lag = [], [], [], []
        for i, s in enumerate(test):
            p = _ridge_predict(w, e_feat_test[i])
            preds.append(p); trues.append(s.rul_labels)
            det_i, lag_i = onset_extra_E(hsmm_E)(i)
            det.append(det_i); lag.append(lag_i)
        
        m = rul_metrics(np.concatenate(preds), np.concatenate(trues))
        per_seed_rmse.append(m["RMSE"])
        per_seed_mae.append(m["MAE"])
        per_seed_rmse_per_unit.append(per_unit_metrics(preds, trues).tolist())
        per_seed_onset_detection.append(float(np.mean(det)))
        per_seed_onset_lag.append(float(np.nanmean(lag)))
        per_seed_runtime.append(time.time() - t0)

    return {
        "rmse": {"mean": float(np.mean(per_seed_rmse)), "std": float(np.std(per_seed_rmse)), "per_seed": per_seed_rmse},
        "mae": {"mean": float(np.mean(per_seed_mae)), "std": float(np.std(per_seed_mae)), "per_seed": per_seed_mae},
        "rmse_per_unit": {"mean": float(np.mean(per_seed_rmse_per_unit)), "per_seed": per_seed_rmse_per_unit},
        "onset_detection": {"mean": float(np.mean(per_seed_onset_detection)), "per_seed": per_seed_onset_detection},
        "onset_lag": {"mean": float(np.mean(per_seed_onset_lag)), "per_seed": per_seed_onset_lag},
        "runtime_s": {"mean": float(np.mean(per_seed_runtime)), "per_seed": per_seed_runtime},
    }


if __name__ == "__main__":
    import json
    
    print("=== Running Arm E CAUSAL re-run for DS01 ===")
    ds01_result = run_arm_e_causal("N-CMAPSS/N-CMAPSS DS01 005.h5")
    print(f"DS01 E_vq_hsmm RMSE (causal): {ds01_result['rmse']['mean']:.4f}")
    print(f"Per seed: {[f'{x:.4f}' for x in ds01_result['rmse']['per_seed']]}")
    
    print("\n=== Running Arm E CAUSAL re-run for DS02 ===")
    ds02_result = run_arm_e_causal("N-CMAPSS/N-CMAPSS DS02 006.h5")
    print(f"DS02 E_vq_hsmm RMSE (causal): {ds02_result['rmse']['mean']:.4f}")
    print(f"Per seed: {[f'{x:.4f}' for x in ds02_result['rmse']['per_seed']]}")
    
    # Save results
    with open("causal_arm_e_results.json", "w") as f:
        json.dump({"DS01": ds01_result, "DS02": ds02_result}, f, indent=2)
    print("\nSaved to causal_arm_e_results.json")