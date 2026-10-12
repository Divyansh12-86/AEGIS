import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
"""W=16 LOEO Sweep: Compare W=4 vs W=16 for Arm E (VQ+HSMM+Ridge) on DS01 and DS02.
Uses causal filtering only, 6-fold LOEO on training engines, stride=4."""
from __future__ import annotations

import json
import time
import numpy as np
import torch

from egpm.data.loaders import NCMAPSSLoader
from egpm.data.split_builder import SplitBuilder, NormalizationStats
from egpm.experiments.ablation_arms import _downsample, _ridge_fit, _ridge_predict
from egpm.preprocessing.preprocessor import Preprocessor
from egpm.training.trainer import Trainer, TrainerConfig
from egpm.grammar.forward_backward import ForwardBackward
from egpm.evaluation.evaluator import rul_metrics
from egpm.utils import set_global_seed


def run_loeo_w16(
    dataset_path: str,
    window_length: int = 16,
    stride: int = 4,
    n_states: int = 5,
    d_max: int = 10,
    n_codes: int = 16,
    stage1_steps: int = 300,
    em_max_iter: int = 10,
    em_restarts: int = 10,
    seed: int = 0,
) -> dict:
    """Run 6-fold LOEO for Arm E (VQ+HSMM+Ridge) at specified window length."""
    raw_dataset = NCMAPSSLoader().load(dataset_path)
    raw_dataset.records = [_downsample(r) for r in raw_dataset.records]

    # Standard split to get train/val/test units
    sb = SplitBuilder(raw_dataset, seed=0)
    split = sb.build()
    train_unit_ids = split.train_unit_ids
    val_unit_ids = split.val_unit_ids
    test_unit_ids = split.test_unit_ids

    print(f"Dataset: {dataset_path}")
    print(f"Training engines for LOEO: {train_unit_ids}")
    print(f"Test engines (locked): {test_unit_ids}")

    loeo_results = {"folds": {}, "summary": {}}
    models_list = ["Arm_E_F_struct"]  # Arm E uses F_struct (gamma + dwell)

    for k_fold in range(len(train_unit_ids)):
        val_unit = train_unit_ids[k_fold]
        tr_fold = [u for j, u in enumerate(train_unit_ids) if j != k_fold]
        print(f"\nFold {k_fold + 1}/6: Held-out = {val_unit}, Train = {tr_fold}")

        t0 = time.time()

        # 1. Fit normalization strictly on tr_fold (5 training units)
        train_records_fold = [r for r in raw_dataset.records if r.unit_id in set(tr_fold)]
        stacked_fold = np.concatenate([r.signals for r in train_records_fold], axis=0)
        mean_fold = stacked_fold.mean(axis=0)
        std_fold = stacked_fold.std(axis=0)
        std_fold = np.where(std_fold < 1e-8, 1.0, std_fold)
        norm_stats_fold = NormalizationStats(mean=mean_fold, std=std_fold, train_unit_ids=sorted(tr_fold))

        # 2. Process windows with specified window_length and stride
        prep = Preprocessor(window_length=window_length, stride=stride)
        tr_seqs_fold = [prep.process(raw_dataset[u], norm_stats=norm_stats_fold) for u in tr_fold]
        val_seq_fold = prep.process(raw_dataset[val_unit], norm_stats=norm_stats_fold)

        # 3. Train full pipeline (encoder -> VQ -> HSMM) on tr_fold
        set_global_seed(0)
        tcfg = TrainerConfig(
            window_length=window_length,
            embed_dim=16,
            hidden=32,
            n_conv_blocks=2,  # MVP uses 2 conv blocks
            n_codes=n_codes,
            d_max=d_max,
            n_states=n_states,
            em_max_iter=em_max_iter,
            em_restarts=em_restarts,
            aux_rul_lambda=0.0,  # No aux RUL for Arm E
            state_dim=None,  # No subspace decoupling
            dead_code_revival=False,
            stage1_steps=stage1_steps,
            batch_size=64,
            seed=seed,
            log_every=0,
        )
        trainer_fold = Trainer(tcfg, n_channels=18)
        X_tr_tensor = torch.as_tensor(np.concatenate([s.X for s in tr_seqs_fold]), dtype=torch.float32)
        M_tr_tensor = torch.ones_like(X_tr_tensor)
        y_tr_tensor = torch.as_tensor(np.concatenate([s.rul_labels for s in tr_seqs_fold]), dtype=torch.float32)
        
        trainer_fold.train_stage1(X_tr_tensor, M_tr_tensor)
        
        ltr_fold = [len(s.X) for s in tr_seqs_fold]
        ev_tr_fold = trainer_fold.window_runs_to_event_runs(
            trainer_fold.event_sequences(X_tr_tensor, M_tr_tensor), ltr_fold
        )
        trainer_fold.train_stage2(ev_tr_fold)
        fb_fold = ForwardBackward(trainer_fold.hsmm)

        # 4. Extract causal F_struct (gamma + dwell) for train and val
        r_tr_fold = [fb_fold.causal_filter(ev) for ev in ev_tr_fold]
        gamma_tr_fold = np.concatenate([r.state_posterior for r in r_tr_fold], axis=0)
        dwell_tr_fold = np.concatenate([r.expected_dwell / trainer_fold.hsmm.d_max for r in r_tr_fold], axis=0)
        f_struct_tr_fold = np.concatenate([gamma_tr_fold, dwell_tr_fold], axis=1)

        X_val_tensor = torch.as_tensor(val_seq_fold.X, dtype=torch.float32)
        ev_val_fold = np.concatenate(trainer_fold.event_sequences(X_val_tensor, torch.ones_like(X_val_tensor)))
        r_val_fold = fb_fold.causal_filter(ev_val_fold)
        gamma_val_fold = r_val_fold.state_posterior
        dwell_val_fold = r_val_fold.expected_dwell / trainer_fold.hsmm.d_max
        f_struct_val_fold = np.concatenate([gamma_val_fold, dwell_val_fold], axis=1)

        # 5. Fit Ridge on F_struct (Arm E features) and predict
        y_tr_fold = np.concatenate([s.rul_labels for s in tr_seqs_fold])
        y_val_fold = val_seq_fold.rul_labels

        w = _ridge_fit(f_struct_tr_fold, y_tr_fold)
        pred = _ridge_predict(w, f_struct_val_fold)
        m = rul_metrics(pred, y_val_fold)

        fold_metrics = {
            "rmse": float(m["RMSE"]),
            "mae": float(m["MAE"]),
            "phm": float(m["PHM_score"]),
        }

        loeo_results["folds"][val_unit] = {
            "metrics": fold_metrics,
            "n_windows": len(val_seq_fold.X),
            "wall_clock_s": time.time() - t0,
        }

        print(f"  Fold {val_unit}: RMSE={m['RMSE']:.2f}, MAE={m['MAE']:.2f}, PHM={m['PHM_score']:.2f}")

    # Summary
    print(f"\n--- LOEO SUMMARY ({dataset_path}, W={window_length}, S={stride}) ---")
    header = f"{'Held-out':<10} {'N':<4} {'Arm_E_F_struct':<15}"
    print(header)
    print("-" * len(header))

    rmse_list = []
    for u in train_unit_ids:
        f_res = loeo_results["folds"][u]
        m_dict = f_res["metrics"]
        n_w = f_res["n_windows"]
        rmse_list.append(m_dict["rmse"])
        print(f"{u:<10} {n_w:<4} {m_dict['rmse']:<15.2f}")

    loeo_results["summary"] = {
        "window_length": window_length,
        "stride": stride,
        "engine_count": len(train_unit_ids),
        "mean_rmse": float(np.mean(rmse_list)),
        "std_rmse": float(np.std(rmse_list)),
        "median_rmse": float(np.median(rmse_list)),
        "min_rmse": float(np.min(rmse_list)),
        "max_rmse": float(np.max(rmse_list)),
    }

    print(f"\nMean RMSE: {loeo_results['summary']['mean_rmse']:.2f} ± {loeo_results['summary']['std_rmse']:.2f}")
    print(f"Median RMSE: {loeo_results['summary']['median_rmse']:.2f}")
    print(f"Min/Max: {loeo_results['summary']['min_rmse']:.2f} / {loeo_results['summary']['max_rmse']:.2f}")

    return loeo_results


def run_loeo_comparison():
    """Run W=4 and W=16 LOEO on both DS01 and DS02, compare results."""
    
    # W=4 baseline numbers from existing LOEO (reproduce for reference)
    print("=" * 80)
    print("PART 1: W=4 BASELINE (Reference)")
    print("=" * 80)
    
    # These are the W=4 numbers from existing LOEO run (Arm E equivalent = Raw+F_struct)
    w4_baseline = {
        "DS01": {
            "DS01_u3": 25.50, "DS01_u4": 17.65, "DS01_u5": 12.59,
            "DS01_u6": 13.56, "DS01_u7": 17.65, "DS01_u8": 16.87
        },
        # DS02 W=4 baseline - need to run
    }
    
    print("\nDS01 W=4 Baseline (Arm E / Raw+F_struct):")
    for u, v in w4_baseline["DS01"].items():
        print(f"  {u}: {v:.2f}")
    print(f"Mean: {np.mean(list(w4_baseline['DS01'].values())):.2f}")
    
    # Run W=16 for DS01
    print("\n" + "=" * 80)
    print("PART 2: W=16 LOEO ON DS01")
    print("=" * 80)
    ds01_w16 = run_loeo_w16(
        "N-CMAPSS/N-CMAPSS DS01 005.h5",
        window_length=16,
        stride=4,
        n_states=5,
        d_max=10,
        n_codes=16,
        stage1_steps=300,
        em_max_iter=10,
        em_restarts=3,  # Use 3 restarts per protocol
    )
    
    # Run W=16 for DS02
    print("\n" + "=" * 80)
    print("PART 3: W=16 LOEO ON DS02")
    print("=" * 80)
    ds02_w16 = run_loeo_w16(
        "N-CMAPSS/N-CMAPSS DS02 006.h5",
        window_length=16,
        stride=4,
        n_states=5,
        d_max=10,
        n_codes=16,
        stage1_steps=300,
        em_max_iter=10,
        em_restarts=3,
    )
    
    # Also run W=4 on DS02 for baseline
    print("\n" + "=" * 80)
    print("PART 4: W=4 LOEO ON DS02 (Baseline)")
    print("=" * 80)
    ds02_w4 = run_loeo_w16(
        "N-CMAPSS/N-CMAPSS DS02 006.h5",
        window_length=4,
        stride=4,
        n_states=5,
        d_max=10,
        n_codes=16,
        stage1_steps=300,
        em_max_iter=10,
        em_restarts=3,
    )
    
    # Summary comparison
    print("\n" + "=" * 80)
    print("SUMMARY COMPARISON")
    print("=" * 80)
    
    print("\nDS01 Comparison (Arm E / F_struct):")
    print(f"  W=4 Mean RMSE: {np.mean(list(w4_baseline['DS01'].values())):.2f}")
    print(f"  W=16 Mean RMSE: {ds01_w16['summary']['mean_rmse']:.2f}")
    print(f"  W=4 Median RMSE: {np.median(list(w4_baseline['DS01'].values())):.2f}")
    print(f"  W=16 Median RMSE: {ds01_w16['summary']['median_rmse']:.2f}")
    
    # Check success criterion: W=16 beats W=4 on majority of folds (>=4/6)
    w4_wins = 0
    w16_wins = 0
    for u in ["DS01_u3", "DS01_u4", "DS01_u5", "DS01_u6", "DS01_u7", "DS01_u8"]:
        w4_rmse = w4_baseline["DS01"][u]
        w16_rmse = ds01_w16["folds"][u]["metrics"]["rmse"]
        if w16_rmse < w4_rmse:
            w16_wins += 1
        else:
            w4_wins += 1
    print(f"\nDS01 Fold wins: W=16 beats W=4 on {w16_wins}/6 folds")
    print(f"Success criterion (>=4/6): {'PASS' if w16_wins >= 4 else 'FAIL'}")
    
    # DS02 comparison
    print("\nDS02 Comparison (Arm E / F_struct):")
    print(f"  W=4 Mean RMSE: {ds02_w4['summary']['mean_rmse']:.2f}")
    print(f"  W=16 Mean RMSE: {ds02_w16['summary']['mean_rmse']:.2f}")
    print(f"  W=4 Median RMSE: {ds02_w4['summary']['median_rmse']:.2f}")
    print(f"  W=16 Median RMSE: {ds02_w16['summary']['median_rmse']:.2f}")
    
    # Save results
    results = {
        "w4_baseline_ds01": w4_baseline["DS01"],
        "w16_ds01": {k: v["metrics"]["rmse"] for k, v in ds01_w16["folds"].items()},
        "w4_ds02": {k: v["metrics"]["rmse"] for k, v in ds02_w4["folds"].items()},
        "w16_ds02": {k: v["metrics"]["rmse"] for k, v in ds02_w16["folds"].items()},
        "summary_ds01": ds01_w16["summary"],
        "summary_ds02_w4": ds02_w4["summary"],
        "summary_ds02_w16": ds02_w16["summary"],
    }
    
    out_path = Path("results/conformal/w16_loeo_sweep_results.json") if Path("results/conformal").exists() else Path("w16_loeo_sweep_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to {out_path}")
    
    return results


if __name__ == "__main__":
    run_loeo_comparison()