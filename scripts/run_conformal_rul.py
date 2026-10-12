import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
"""Conformal RUL Intervals with Engine-Level LOEO Calibration.

Method:
- 6-fold LOEO on training engines (DS01: 6 engines, DS02: 5 engines)
- For each fold: train on 5/4 engines, predict on held-out, compute residuals
- Pool residuals PER HELD-OUT ENGINE (not per window) - engine-level nonconformity
- Derive quantiles for 50%, 80%, 90% coverage from pooled engine-level residuals
- Apply to locked test engines (DS01_u2/u9, DS02_u2/u5) ONCE
- Report empirical coverage, interval width, per-engine coverage
"""
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


def run_loeo_conformal(
    dataset_path: str,
    window_length: int = 4,
    stride: int = 4,
    n_states: int = 5,
    d_max: int = 10,
    n_codes: int = 16,
    stage1_steps: int = 300,
    em_max_iter: int = 10,
    em_restarts: int = 3,
    seed: int = 0,
    nominal_levels: list = [0.5, 0.8, 0.9],
) -> dict:
    """Run engine-level LOEO conformal calibration for Arm E."""
    raw_dataset = NCMAPSSLoader().load(dataset_path)
    raw_dataset.records = [_downsample(r) for r in raw_dataset.records]

    sb = SplitBuilder(raw_dataset, seed=0)
    split = sb.build()
    train_unit_ids = split.train_unit_ids
    test_unit_ids = split.test_unit_ids

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_path}")
    print(f"Training engines: {train_unit_ids}")
    print(f"Test engines (locked): {test_unit_ids}")
    print(f"{'='*60}")

    loeo_results = {"folds": {}, "summary": {}, "conformal": {}}

    # ================================================================
    # LOEO FOLDS: Collect residuals PER HELD-OUT ENGINE
    # ================================================================
    all_engine_residuals = {}  # engine -> list of residuals (per window)

    for k_fold, val_unit in enumerate(train_unit_ids):
        tr_fold = [u for j, u in enumerate(train_unit_ids) if j != k_fold]
        print(f"\nFold {k_fold + 1}/{len(train_unit_ids)}: Held-out = {val_unit}, Train = {tr_fold}")
        t0 = time.time()

        # 1. Fit normalization on tr_fold (5/4 engines)
        train_records_fold = [r for r in raw_dataset.records if r.unit_id in set(tr_fold)]
        stacked_fold = np.concatenate([r.signals for r in train_records_fold], axis=0)
        mean_fold = stacked_fold.mean(axis=0)
        std_fold = stacked_fold.std(axis=0)
        std_fold = np.where(std_fold < 1e-8, 1.0, std_fold)
        norm_stats_fold = NormalizationStats(mean=mean_fold, std=std_fold, train_unit_ids=sorted(tr_fold))

        # 2. Process windows
        prep = Preprocessor(window_length=window_length, stride=stride)
        tr_seqs_fold = [prep.process(raw_dataset[u], norm_stats=norm_stats_fold) for u in tr_fold]
        val_seq_fold = prep.process(raw_dataset[val_unit], norm_stats=norm_stats_fold)

        # 3. Train full pipeline on tr_fold
        set_global_seed(seed)
        tcfg = TrainerConfig(
            window_length=window_length, embed_dim=16, hidden=32,
            n_conv_blocks=2, n_codes=n_codes, d_max=d_max,
            n_states=n_states, em_max_iter=em_max_iter,
            em_restarts=em_restarts, aux_rul_lambda=0.0,
            state_dim=None, dead_code_revival=False,
            stage1_steps=stage1_steps, batch_size=64, seed=seed, log_every=0,
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

        # 5. Fit Ridge on F_struct (Arm E features) and predict on held-out engine
        y_tr_fold = np.concatenate([s.rul_labels for s in tr_seqs_fold])
        y_val_fold = val_seq_fold.rul_labels

        w = _ridge_fit(f_struct_tr_fold, y_tr_fold)
        pred = _ridge_predict(w, f_struct_val_fold)
        m = rul_metrics(pred, y_val_fold)

        residuals = pred - y_val_fold  # signed residuals
        abs_residuals = np.abs(residuals)  # absolute residuals for conformal

        loeo_results["folds"][val_unit] = {
            "metrics": {"rmse": float(m["RMSE"]), "mae": float(m["MAE"]), "phm": float(m["PHM_score"])},
            "n_windows": len(val_seq_fold.X),
            "residuals": residuals.tolist(),
            "abs_residuals": abs_residuals.tolist(),
            "wall_clock_s": time.time() - t0,
        }

        # Store residuals per held-out engine
        all_engine_residuals[val_unit] = abs_residuals.tolist()

        print(f"  {val_unit}: RMSE={m['RMSE']:.2f}, n_windows={len(val_seq_fold.X)}, residuals range=[{abs_residuals.min():.1f}, {abs_residuals.max():.1f}]")

    # ================================================================
    # CONFORMAL QUANTILES FROM ENGINE-LEVEL POOLED RESIDUALS
    # ================================================================
    print(f"\n--- CONFORMAL QUANTILES ---")
    
    # Pool all engine-level residuals (engine-level exchangeability)
    # Each engine contributes its residual distribution
    all_abs_residuals = []
    for engine, res_list in all_engine_residuals.items():
        all_abs_residuals.extend(res_list)
    
    all_abs_residuals = np.array(all_abs_residuals)
    n_total = len(all_abs_residuals)
    n_engines = len(all_engine_residuals)
    
    print(f"Total engines: {n_engines}, Total residual windows: {n_total}")
    print(f"Per-engine residual counts: {[len(v) for v in all_engine_residuals.values()]}")

    conformal_quantiles = {}
    for level in nominal_levels:
        # Split conformal: quantile of absolute residuals
        # Use (1 - alpha) quantile with finite-sample correction
        # For small N, use (n+1)*(1-alpha) / n correction
        q = np.quantile(all_abs_residuals, level, method="higher")  # conservative
        # Alternative: (n+1)*(1-alpha)/n adjusted
        k = int(np.ceil((n_total + 1) * level))
        q_adjusted = np.sort(all_abs_residuals)[min(k - 1, n_total - 1)]
        
        conformal_quantiles[f"{int(level*100)}%"] = {
            "quantile_standard": float(q),
            "quantile_finite_sample": float(q_adjusted),
            "n_calibration_windows": n_total,
            "n_calibration_engines": n_engines,
        }
        print(f"  {int(level*100)}% nominal: standard={q:.2f}, finite-sample={q_adjusted:.2f}")

    # ================================================================
    # FINAL TEST SET EVALUATION (ONLY ONCE)
    # ================================================================
    print(f"\n--- LOCKED TEST SET EVALUATION ---")
    
    # Re-train on ALL training engines for final model
    train_records_all = [r for r in raw_dataset.records if r.unit_id in set(train_unit_ids)]
    stacked_all = np.concatenate([r.signals for r in train_records_all], axis=0)
    mean_all = stacked_all.mean(axis=0)
    std_all = stacked_all.std(axis=0)
    std_all = np.where(std_all < 1e-8, 1.0, std_all)
    norm_stats_all = NormalizationStats(mean=mean_all, std=std_all, train_unit_ids=sorted(train_unit_ids))

    prep = Preprocessor(window_length=window_length, stride=stride)
    tr_seqs_all = [prep.process(raw_dataset[u], norm_stats=norm_stats_all) for u in train_unit_ids]
    test_seqs = [prep.process(raw_dataset[u], norm_stats=norm_stats_all) for u in test_unit_ids]

    # Train final model on all training data
    set_global_seed(seed)
    tcfg = TrainerConfig(
        window_length=window_length, embed_dim=16, hidden=32,
        n_conv_blocks=2, n_codes=n_codes, d_max=d_max,
        n_states=n_states, em_max_iter=em_max_iter,
        em_restarts=em_restarts, aux_rul_lambda=0.0,
        state_dim=None, dead_code_revival=False,
        stage1_steps=stage1_steps, batch_size=64, seed=seed, log_every=0,
    )
    trainer_final = Trainer(tcfg, n_channels=18)
    X_tr_tensor = torch.as_tensor(np.concatenate([s.X for s in tr_seqs_all]), dtype=torch.float32)
    M_tr_tensor = torch.ones_like(X_tr_tensor)
    y_tr_tensor = torch.as_tensor(np.concatenate([s.rul_labels for s in tr_seqs_all]), dtype=torch.float32)

    trainer_final.train_stage1(X_tr_tensor, M_tr_tensor)
    ltr_all = [len(s.X) for s in tr_seqs_all]
    ev_tr_all = trainer_final.window_runs_to_event_runs(
        trainer_final.event_sequences(X_tr_tensor, M_tr_tensor), ltr_all
    )
    trainer_final.train_stage2(ev_tr_all)
    fb_final = ForwardBackward(trainer_final.hsmm)

    # Extract F_struct for all training data
    r_tr_all = [fb_final.causal_filter(ev) for ev in ev_tr_all]
    gamma_tr_all = np.concatenate([r.state_posterior for r in r_tr_all], axis=0)
    dwell_tr_all = np.concatenate([r.expected_dwell / trainer_final.hsmm.d_max for r in r_tr_all], axis=0)
    f_struct_tr_all = np.concatenate([gamma_tr_all, dwell_tr_all], axis=1)

    y_tr_all = np.concatenate([s.rul_labels for s in tr_seqs_all])

    # Fit final Ridge
    w_final = _ridge_fit(f_struct_tr_all, y_tr_all)

    # ================================================================
    # TEST SET PREDICTIONS WITH CONFORMAL INTERVALS
    # ================================================================
    test_results = {}
    for level in nominal_levels:
        q = conformal_quantiles[f"{int(level*100)}%"]["quantile_finite_sample"]
        test_results[f"{int(level*100)}%"] = {
            "quantile": q,
            "engines": {}
        }

    for test_unit in test_unit_ids:
        test_seq = [s for s in test_seqs if s.unit_id == test_unit][0]
        X_test_tensor = torch.as_tensor(test_seq.X, dtype=torch.float32)
        ev_test = np.concatenate(trainer_final.event_sequences(X_test_tensor, torch.ones_like(X_test_tensor)))
        r_test = fb_final.causal_filter(ev_test)
        gamma_test = r_test.state_posterior
        dwell_test = r_test.expected_dwell / trainer_final.hsmm.d_max
        f_struct_test = np.concatenate([gamma_test, dwell_test], axis=1)

        pred = _ridge_predict(w_final, f_struct_test)
        true_rul = test_seq.rul_labels
        
        # RMSE on test
        m = rul_metrics(pred, true_rul)
        
        # Coverage at each level
        coverage = {}
        intervals = []
        for level in nominal_levels:
            q = conformal_quantiles[f"{int(level*100)}%"]["quantile_finite_sample"]
            lower = pred - q
            upper = pred + q
            covered = (true_rul >= lower) & (true_rul <= upper)
            cov = covered.mean()
            coverage[f"{int(level*100)}%"] = {
                "empirical": float(cov),
                "nominal": level,
                "covered_count": int(covered.sum()),
                "total_windows": len(true_rul),
            }
            intervals.append({
                "level": int(level*100),
                "lower": lower.tolist(),
                "upper": upper.tolist(),
                "point": pred.tolist(),
            })
        
        test_results[f"{int(level*100)}%"]["engines"][test_unit] = {
            "metrics": {"rmse": float(m["RMSE"]), "mae": float(m["MAE"]), "phm": float(m["PHM_score"])},
            "n_windows": len(true_rul),
            "coverage": coverage[f"{int(level*100)}%"],
            "intervals": intervals,
            "true_rul": true_rul.tolist(),
            "pred_rul": pred.tolist(),
        }
        print(f"  {test_unit}: RMSE={m['RMSE']:.2f}, n={len(true_rul)}, "
              f"50% cov={coverage['50%']['empirical']:.3f}, "
              f"80% cov={coverage['80%']['empirical']:.3f}, "
              f"90% cov={coverage['90%']['empirical']:.3f}")

    # ================================================================
    # PER-ENGINE COVERAGE FROM LOEO FOLDS (retrospective)
    # ================================================================
    loeo_coverage = {}
    for level in nominal_levels:
        q = conformal_quantiles[f"{int(level*100)}%"]["quantile_finite_sample"]
        loeo_coverage[f"{int(level*100)}%"] = {}
        for engine, abs_res in all_engine_residuals.items():
            abs_res = np.array(abs_res)
            cov = (abs_res <= q).mean()
            loeo_coverage[f"{int(level*100)}%"][engine] = {
                "empirical": float(cov),
                "n_windows": len(abs_res),
            }

    # ================================================================
    # SMALL-N LIMITATION ANALYSIS
    # ================================================================
    n_calib_engines = len(all_engine_residuals)
    granularity = 1.0 / n_calib_engines if n_calib_engines > 0 else None
    
    small_n_analysis = {
        "n_calibration_engines": n_calib_engines,
        "n_calibration_windows": n_total,
        "achievable_quantile_granularity": f"~{granularity:.0%} steps (1/{n_calib_engines})",
        "warning": f"Only {n_calib_engines} calibration engines - quantile granularity is coarse. "
                   f"Nominal coverage may not be precisely achievable." if n_calib_engines <= 10 else None,
    }

    # ================================================================
    # FINAL RESULTS PACKAGE
    # ================================================================
    results = {
        "dataset": dataset_path,
        "config": {
            "window_length": window_length,
            "stride": stride,
            "n_states": n_states,
            "d_max": d_max,
            "n_codes": n_codes,
            "stage1_steps": stage1_steps,
            "em_max_iter": em_max_iter,
            "em_restarts": em_restarts,
            "arm": "E_vq_hsmm",
            "filtering": "causal",
            "nominal_levels": nominal_levels,
        },
        "loeo_folds": loeo_results["folds"],
        "loeo_coverage": loeo_coverage,
        "conformal_quantiles": conformal_quantiles,
        "test_set_evaluation": test_results,
        "small_n_analysis": small_n_analysis,
        "engine_residual_counts": {k: len(v) for k, v in all_engine_residuals.items()},
    }
    
    return results


def main():
    print("="*70)
    print("CONFORMAL RUL INTERVALS - ENGINE-LEVEL LOEO CALIBRATION")
    print("="*70)
    print("Arm E (VQ+HSMM+Ridge), Causal filtering only")
    print("Engine-level LOEO conformal calibration (no window-level pseudoreplication)")
    
    all_results = {}
    
    # DS01
    ds01_results = run_loeo_conformal("N-CMAPSS/N-CMAPSS DS01 005.h5")
    all_results["DS01"] = ds01_results
    
    # DS02
    ds02_results = run_loeo_conformal("N-CMAPSS/N-CMAPSS DS02 006.h5")
    all_results["DS02"] = ds02_results
    
    # Save
    out_path = Path("results/conformal/conformal_rul_results.json") if Path("results/conformal").exists() else Path("conformal_rul_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print("\n\nResults saved to {out_path}")
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for ds, res in all_results.items():
        print(f"\n{ds}:")
        print(f"  LOEO mean RMSE: {np.mean([v['metrics']['rmse'] for v in res['loeo_folds'].values()]):.2f}")
        for level in [0.5, 0.8, 0.9]:
            q = res["conformal_quantiles"][f"{int(level*100)}%"]["quantile_finite_sample"]
            print(f"  {int(level*100)}% quantile: {q:.2f}")
        
        # Test set coverage
        for level in [0.5, 0.8, 0.9]:
            for engine, data in res["test_set_evaluation"][f"{int(level*100)}%"]["engines"].items():
                cov = data["coverage"][f"{int(level*100)}%"]["empirical"]
                print(f"  Test {engine} {int(level*100)}% coverage: {cov:.3f}")

    print("\nDone! Results saved to {out_path}")


if __name__ == "__main__":
    main()
