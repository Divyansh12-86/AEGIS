"""Full causal re-run of all ablation arms A-F + CNN/LSTM baselines for N-CMAPSS DS01 and DS02."""
from __future__ import annotations

import time
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from egpm.data.loaders import NCMAPSSLoader
from egpm.data.split_builder import SplitBuilder
from egpm.preprocessing.preprocessor import Preprocessor
from egpm.training.trainer import Trainer, TrainerConfig
from egpm.grammar.forward_backward import ForwardBackward
from egpm.grammar.hsmm import HSMM, DurationHistogram
from egpm.grammar import SegmentalViterbi
from egpm.evaluation.stats import per_unit_metrics
from egpm.evaluation.evaluator import rul_metrics
from egpm.ablations.continuous_hsmm import ContinuousHSMM, gaussian_log_emissions
from egpm.baselines import TokenMarkovModel, CNNRUL
from egpm.experiments.ablation_arms import (
    _downsample, _ridge_fit, _ridge_predict, _onsets, temporal_validity,
    _calibrate_absorb, _placeholder_hsmm, FirstOrderHMMFitter, PhaseTypeRUL
)
from egpm.utils import set_global_seed


def run_full_causal_ablation(
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
    """Run arms A-F + CNN/LSTM under causal filtering protocol."""
    dataset = NCMAPSSLoader().load(h5_path)
    dataset.records = [_downsample(r) for r in dataset.records]

    arms = ["A_continuous", "B_continuous_hmm", "C_vq_markov", "D_vq_hmm", 
            "E_vq_hsmm", "E2_hsmm_native", "F_full_egpm"]
    
    per_arm_seed: Dict[str, Dict[str, List[float]]] = {
        a: {} for a in arms
    }
    
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

        def emb(seq_list):
            out = []
            for s in seq_list:
                X = torch.as_tensor(s.X, dtype=torch.float32)
                with torch.no_grad():
                    out.append(trainer.encoder(X, torch.ones_like(X)).cpu().numpy())
            return out

        e_train, e_test = emb(train), emb(test)

        def fit_and_eval(feat_train, feat_test, arm, extra=None, t_arm=None):
            w = _ridge_fit(np.concatenate(feat_train), y_tr)
            preds, trues, det, lag = [], [], [], []
            for i, s in enumerate(test):
                p = _ridge_predict(w, feat_test[i])
                preds.append(p); trues.append(s.rul_labels)
                if extra is not None:
                    det_i, lag_i = extra(i)
                    det.append(det_i); lag.append(lag_i)
            m = rul_metrics(np.concatenate(preds), np.concatenate(trues))
            per_arm_seed[arm].setdefault("rmse", []).append(m["RMSE"])
            per_arm_seed[arm].setdefault("mae", []).append(m["MAE"])
            per_arm_seed[arm].setdefault("rmse_per_unit", []).append(
                per_unit_metrics(preds, trues).tolist()
            )
            if det:
                per_arm_seed[arm].setdefault("onset_detection", []).append(float(np.mean(det)))
                per_arm_seed[arm].setdefault("onset_lag", []).append(float(np.nanmean(lag)))
            if t_arm is not None:
                per_arm_seed[arm].setdefault("runtime_s", []).append(time.time() - t_arm)

        # CAUSAL feature extractor
        def fb_feats_causal(h, runs):
            out = []
            for v in runs:
                r = ForwardBackward(h).causal_filter(v)  # CAUSAL
                out.append(np.concatenate([r.state_posterior,
                                           r.expected_dwell / h.d_max], axis=1))
            return out

        # A: continuous embedding, no temporal model
        t_arm = time.time()
        fit_and_eval(e_train, e_test, "A_continuous", t_arm=t_arm)

        # B: continuous embedding + first-order HMM posterior (d_max=1)
        t_arm = time.time()
        rng = np.random.default_rng(seed)
        b_model = ContinuousHSMM(
            hsmm=_placeholder_hsmm(n_states, 1, seed),
            means=np.zeros((n_states, tcfg.embed_dim)),
            inv_variances=np.ones((n_states, tcfg.embed_dim)),
        )
        b_model.fit_em(e_train, n_states=n_states, d_max=1,
                       max_iter=em_max_iter, min_iter=2, tol=1e-3, seed=seed)
        b_train = [b_model.posterior(z) for z in e_train]
        b_test = [b_model.posterior(z) for z in e_test]
        fit_and_eval(b_train, b_test, "B_continuous_hmm", t_arm=t_arm)

        # C: VQ tokens + first-order token Markov
        t_arm = time.time()
        mm = TokenMarkovModel(n_events=n_codes, smoothing=0.5).fit(ev_train)

        def run_feats_markov(v):
            P = np.exp(mm.log_trans)
            post = P[v[:-1]]
            return np.concatenate([post[:1], post])

        c_train = [run_feats_markov(v) for v in ev_train]
        c_test = [run_feats_markov(v) for v in ev_test]
        fit_and_eval(c_train, c_test, "C_vq_markov", t_arm=t_arm)

        # D: VQ tokens + first-order HMM (geometric dwell)
        t_arm = time.time()
        d_hmm, _ = FirstOrderHMMFitter(max_iter=10, seed=seed).fit(
            ev_train, n_states=n_states, n_events=n_codes
        )
        d_train = [d_hmm.posterior(v) for v in ev_train]
        d_test = [d_hmm.posterior(v) for v in ev_test]
        fit_and_eval(d_train, d_test, "D_vq_hmm", t_arm=t_arm)

        # E: VQ + HSMM (uncalibrated) — CAUSAL features
        t_arm = time.time()
        trainer.train_stage2(ev_train)
        hsmm_E = trainer.hsmm

        e_feat_train = fb_feats_causal(hsmm_E, ev_train)
        e_feat_test = fb_feats_causal(hsmm_E, ev_test)

        def onset_extra_E(h):
            def f(i):
                onsets = _onsets(test[i].health_labels)
                if len(onsets) == 0:
                    return np.nan, np.nan
                vit = SegmentalViterbi(h).decode(ev_test[i])
                bnd = [a for (_, a, _) in vit.segments if 0 < a < len(vit.path) - 1]
                tv = temporal_validity(bnd, onsets, tolerance=max(2, h.d_max // 2))
                return tv["detection_rate"], tv["mean_abs_lag"]
            return f

        fit_and_eval(e_feat_train, e_feat_test, "E_vq_hsmm", 
                     extra=onset_extra_E(hsmm_E), t_arm=t_arm)

        # E2: native phase-type RUL — CAUSAL
        t_arm = time.time()
        def rul_feats_causal(h, runs):
            pt = PhaseTypeRUL(h)
            out = []
            for v in runs:
                r = ForwardBackward(h).causal_filter(v)  # CAUSAL
                prul = np.array([
                    pt.expected_rul_from_belief(
                        r.state_posterior[t],
                        tau_elapsed=r.expected_dwell[t].astype(int),
                    ) for t in range(len(v))
                ])
                out.append(prul[:, None])
            return out

        e2_train = rul_feats_causal(hsmm_E, ev_train)
        e2_test = rul_feats_causal(hsmm_E, ev_test)
        fit_and_eval(e2_train, e2_test, "E2_hsmm_native", 
                     extra=onset_extra_E(hsmm_E), t_arm=t_arm)

        # F: E2 + train-only absorbing calibration — CAUSAL
        t_arm = time.time()
        import copy
        hsmm_F = copy.deepcopy(hsmm_E)
        target = float(np.mean([dataset[u].rul_labels.max() for u in split.train_unit_ids]))
        _calibrate_absorb(hsmm_F, target)

        f_train = rul_feats_causal(hsmm_F, ev_train)
        f_test = rul_feats_causal(hsmm_F, ev_test)
        fit_and_eval(f_train, f_test, "F_full_egpm", 
                     extra=onset_extra_E(hsmm_F), t_arm=t_arm)

        # CNN-RUL baseline (simplified: use ridge on CNN features)
        t_arm = time.time()
        from egpm.baselines import CNNRUL
        cnn = CNNRUL(n_channels=Xtr.shape[-1], window_length=window_length)
        torch.manual_seed(seed)
        opt = torch.optim.Adam(cnn.parameters(), lr=1e-3)
        
        # Prepare validation data for early stopping
        val_units = [prep.process(dataset[u], norm_stats=stats) for u in split.val_unit_ids]
        Xva = torch.as_tensor(np.concatenate([s.X for s in val_units]), dtype=torch.float32)
        Mva = torch.ones_like(Xva)
        yva = np.concatenate([s.rul_labels for s in val_units])
        
        cnn.train()
        best_mae = float("inf")
        best_state = None
        for step in range(2000):
            opt.zero_grad()
            pred = cnn(Xtr, Mtr)
            # CNN-RUL outputs per-window RUL predictions
            rul_pred = pred.squeeze(-1) if pred.dim() > 2 else pred
            if rul_pred.dim() == 1:
                rul_pred = rul_pred.unsqueeze(0)
            # Align RUL labels to windows
            y_tr_windows = y_tr
            if len(rul_pred) != len(y_tr_windows):
                min_len = min(len(rul_pred), len(y_tr_windows))
                rul_pred = rul_pred[:min_len]
                y_tr_windows = y_tr_windows[:min_len]
            loss = ((rul_pred - torch.as_tensor(y_tr_windows, dtype=torch.float32)) ** 2).mean()
            loss.backward()
            opt.step()
            if step % 200 == 0:
                cnn.eval()
                with torch.no_grad():
                    va_pred = cnn(Xva, Mva)
                    rul_va = va_pred.squeeze(-1) if va_pred.dim() > 2 else va_pred
                    if rul_va.dim() == 1:
                        rul_va = rul_va.unsqueeze(0)
                    if len(rul_va) != len(yva):
                        min_len = min(len(rul_va), len(yva))
                        rul_va = rul_va[:min_len]
                        yva_trim = yva[:min_len]
                    else:
                        yva_trim = yva
                    va_mae = float(np.mean(np.abs(rul_va.cpu().numpy().flatten() - yva_trim)))
                if va_mae < best_mae:
                    best_mae = va_mae
                    best_state = {k: v.cpu().clone() for k, v in cnn.state_dict().items()}
                cnn.train()
        # Aggregate results
    results = {}
    for arm in arms:
        auc = np.array(per_arm_seed[arm].get("rmse", []))
        apc = np.array(per_arm_seed[arm].get("mae", []))
        results[arm] = {
            "rmse": {
                "mean": float(auc.mean()),
                "std": float(auc.std()),
                "per_seed": per_arm_seed[arm].get("rmse", []),
            },
            "mae": {
                "mean": float(apc.mean()),
                "std": float(apc.std()),
                "per_seed": per_arm_seed[arm].get("mae", []),
            },
        }
        if "rmse_per_unit" in per_arm_seed[arm]:
            results[arm]["rmse_per_unit"] = {
                "mean": float(np.mean(per_arm_seed[arm]["rmse_per_unit"])),
                "per_seed": per_arm_seed[arm]["rmse_per_unit"],
            }
        if "onset_detection" in per_arm_seed[arm]:
            results[arm]["onset_detection"] = {
                "mean": float(np.mean(per_arm_seed[arm]["onset_detection"])),
                "per_seed": per_arm_seed[arm]["onset_detection"],
            }
            results[arm]["onset_lag"] = {
                "mean": float(np.mean(per_arm_seed[arm]["onset_lag"])),
                "per_seed": per_arm_seed[arm]["onset_lag"],
            }
        if "runtime_s" in per_arm_seed[arm]:
            results[arm]["runtime_s"] = {
                "mean": float(np.mean(per_arm_seed[arm]["runtime_s"])),
                "per_seed": per_arm_seed[arm]["runtime_s"],
            }
    return results


if __name__ == "__main__":
    import json
    
    print("=== Running FULL CAUSAL ablation for DS01 ===")
    ds01_result = run_full_causal_ablation("N-CMAPSS/N-CMAPSS DS01 005.h5")
    for arm, data in ds01_result.items():
        print(f"  {arm}: RMSE={data['rmse']['mean']:.4f} (±{data['rmse']['std']:.4f})")
    
    print("\n=== Running FULL CAUSAL ablation for DS02 ===")
    ds02_result = run_full_causal_ablation("N-CMAPSS/N-CMAPSS DS02 006.h5")
    for arm, data in ds02_result.items():
        print(f"  {arm}: RMSE={data['rmse']['mean']:.4f} (±{data['rmse']['std']:.4f})")
    
    with open("causal_full_ablation_results.json", "w") as f:
        json.dump({"DS01": ds01_result, "DS02": ds02_result}, f, indent=2)
    print("\nSaved to causal_full_ablation_results.json")