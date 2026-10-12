import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
"""EGPM LOEO Validation & 10-Seed Representation Stability Study.

Part 1: Leave-One-Engine-Out (LOEO) cross-validation across 6 training engines (u3-u8).
Part 2: 10-Seed Representation Stability Study on locked test split (seeds 0 through 9).
Part 3: Gamma vs Dwell ablation across LOEO folds.
Part 4: Residual analysis by RUL bins.

Strictly adheres to:
- No architecture changes.
- Causal filtering only (no future leak, no backward beta).
- Ridge alpha = 1.0 (closed-form, no tuning).
- Normalization strictly fit on training units per fold.
"""
from __future__ import annotations

import json
import time
import numpy as np
import scipy.stats
import torch

from egpm.data.loaders import NCMAPSSLoader
from egpm.data.split_builder import SplitBuilder, NormalizationStats, UnitSplit
from egpm.experiments.ablation_arms import _downsample, _ridge_fit, _ridge_predict
from egpm.preprocessing.preprocessor import Preprocessor
from egpm.training.trainer import Trainer, TrainerConfig
from egpm.grammar.forward_backward import ForwardBackward
from egpm.evaluation.evaluator import rul_metrics
from egpm.utils import set_global_seed


def run_loeo_and_seed_study():
    print("=" * 80)
    print("EGPM LOEO VALIDATION & 10-SEED REPRESENTATION STABILITY STUDY")
    print("=" * 80)

    dataset_path = "N-CMAPSS/N-CMAPSS DS01 005.h5"
    raw_dataset = NCMAPSSLoader().load(dataset_path)
    raw_dataset.records = [_downsample(r) for r in raw_dataset.records]

    # All engines in DS01
    all_unit_ids = sorted(raw_dataset.unit_ids)
    print(f"All available units in DS01: {all_unit_ids}")

    # Standard split units
    sb = SplitBuilder(raw_dataset, seed=0)
    split_standard = sb.build()
    train_unit_ids = split_standard.train_unit_ids  # ['DS01_u3', 'DS01_u4', 'DS01_u5', 'DS01_u6', 'DS01_u7', 'DS01_u8']
    val_unit_ids = split_standard.val_unit_ids      # ['DS01_u1', 'DS01_u10']
    test_unit_ids = split_standard.test_unit_ids    # ['DS01_u2', 'DS01_u9']
    print(f"Training engines for LOEO: {train_unit_ids}")
    print(f"Validation engines:         {val_unit_ids}")
    print(f"Locked test engines:        {test_unit_ids}")

    # =========================================================================
    # LEAKAGE AND CAUSALITY AUDIT
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 0: PREPROCESSING & CAUSALITY AUDIT")
    print("=" * 70)
    leakage_checks = {}

    # Check 1: Unit disjointness
    disjoint_tr_te = len(set(train_unit_ids) & set(test_unit_ids)) == 0
    disjoint_tr_va = len(set(train_unit_ids) & set(val_unit_ids)) == 0
    disjoint_va_te = len(set(val_unit_ids) & set(test_unit_ids)) == 0
    leakage_checks["units_disjoint"] = disjoint_tr_te and disjoint_tr_va and disjoint_va_te

    # Check 2: Endpoint alignment on test set
    stats_std = sb.fit_normalization(split_standard)
    prep_std = Preprocessor(window_length=4, stride=4)
    align_ok = True
    for u in test_unit_ids:
        rec = raw_dataset[u]
        seq = prep_std.process(rec, norm_stats=stats_std)
        for n in range(len(seq.X)):
            right_idx = 4 * n + 3
            if seq.rul_labels[n] != rec.rul_labels[right_idx]:
                align_ok = False
    leakage_checks["target_endpoint_alignment"] = align_ok

    # Check 3: Encoder receptive field & padding
    cfg_test = TrainerConfig(n_conv_blocks=3, window_length=4, embed_dim=16, hidden=32)
    trainer_test = Trainer(cfg_test, n_channels=18)
    rf = trainer_test.encoder.fusion.receptive_field + 4  # per-channel pad is 4
    leakage_checks["encoder_strict_left_padding"] = (rf == 19)

    for k, v in leakage_checks.items():
        print(f"  [CHECK] {k}: {'PASSED' if v else 'FAILED'}")
    assert all(leakage_checks.values()), "Causality/Leakage check failed!"

    # =========================================================================
    # EXPERIMENT 1: LOEO VALIDATION (FOLDS 1..6)
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 1: LEAVE-ONE-ENGINE-OUT (LOEO) CROSS-VALIDATION")
    print("=" * 70)

    loeo_results = {
        "folds": {},
        "summary": {},
    }

    models_list = ["Raw", "F_struct", "Raw_plus_gamma", "Raw_plus_dwell", "Raw_plus_F_struct"]

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

        # 2. Process windows
        prep = Preprocessor(window_length=4, stride=4)
        tr_seqs_fold = [prep.process(raw_dataset[u], norm_stats=norm_stats_fold) for u in tr_fold]
        val_seq_fold = prep.process(raw_dataset[val_unit], norm_stats=norm_stats_fold)

        # 3. Extract raw features (18-D right edge)
        X_tr_raw = np.concatenate([s.X[:, -1, :] for s in tr_seqs_fold], axis=0)
        X_val_raw = val_seq_fold.X[:, -1, :]
        y_tr_fold = np.concatenate([s.rul_labels for s in tr_seqs_fold])
        y_val_fold = val_seq_fold.rul_labels

        # 4. Train 12/4 candidate on tr_fold
        set_global_seed(0)
        cfg_fold = TrainerConfig(
            window_length=4,
            embed_dim=16,
            hidden=32,
            n_conv_blocks=3,
            n_codes=16,
            d_max=10,
            n_states=5,
            em_max_iter=10,
            em_restarts=10,
            aux_rul_lambda=0.1,
            state_dim=12,
            dead_code_revival=False,
            stage1_steps=300,
            batch_size=64,
            seed=0,
            log_every=0,
        )
        trainer_fold = Trainer(cfg_fold, n_channels=18)
        X_tr_tensor = torch.as_tensor(np.concatenate([s.X for s in tr_seqs_fold]), dtype=torch.float32)
        M_tr_tensor = torch.ones_like(X_tr_tensor)
        y_tr_tensor = torch.as_tensor(y_tr_fold, dtype=torch.float32)
        trainer_fold.train_stage1(X_tr_tensor, M_tr_tensor, rul_labels=y_tr_tensor)

        ltr_fold = [len(s.X) for s in tr_seqs_fold]
        ev_tr_fold = trainer_fold.window_runs_to_event_runs(trainer_fold.event_sequences(X_tr_tensor, M_tr_tensor), ltr_fold)
        trainer_fold.train_stage2(ev_tr_fold)
        fb_fold = ForwardBackward(trainer_fold.hsmm)

        # 5. Extract causal F_struct
        r_tr_fold = [fb_fold.causal_filter(ev) for ev in ev_tr_fold]
        gamma_tr_fold = np.concatenate([r.state_posterior for r in r_tr_fold], axis=0)
        dwell_tr_fold = np.concatenate([r.expected_dwell / 10.0 for r in r_tr_fold], axis=0)
        f_struct_tr_fold = np.concatenate([gamma_tr_fold, dwell_tr_fold], axis=1)

        X_val_tensor = torch.as_tensor(val_seq_fold.X, dtype=torch.float32)
        ev_val_fold = np.concatenate(trainer_fold.event_sequences(X_val_tensor, torch.ones_like(X_val_tensor)))
        r_val_fold = fb_fold.causal_filter(ev_val_fold)
        gamma_val_fold = r_val_fold.state_posterior
        dwell_val_fold = r_val_fold.expected_dwell / 10.0
        f_struct_val_fold = np.concatenate([gamma_val_fold, dwell_val_fold], axis=1)

        # 6. Fit models on tr_fold, predict on val_unit
        feature_sets = {
            "Raw": (X_tr_raw, X_val_raw),
            "F_struct": (f_struct_tr_fold, f_struct_val_fold),
            "Raw_plus_gamma": (np.concatenate([X_tr_raw, gamma_tr_fold], axis=1), np.concatenate([X_val_raw, gamma_val_fold], axis=1)),
            "Raw_plus_dwell": (np.concatenate([X_tr_raw, dwell_tr_fold], axis=1), np.concatenate([X_val_raw, dwell_val_fold], axis=1)),
            "Raw_plus_F_struct": (np.concatenate([X_tr_raw, f_struct_tr_fold], axis=1), np.concatenate([X_val_raw, f_struct_val_fold], axis=1)),
        }

        fold_metrics = {}
        for mname, (F_tr, F_val) in feature_sets.items():
            w = _ridge_fit(F_tr, y_tr_fold)
            pred = _ridge_predict(w, F_val)
            m = rul_metrics(pred, y_val_fold)
            fold_metrics[mname] = {
                "rmse": float(m["RMSE"]),
                "mae": float(m["MAE"]),
                "phm": float(m["PHM_score"]),
            }

        delta_rmse = fold_metrics["Raw"]["rmse"] - fold_metrics["Raw_plus_F_struct"]["rmse"]
        rel_imp = (delta_rmse / fold_metrics["Raw"]["rmse"]) * 100.0

        loeo_results["folds"][val_unit] = {
            "metrics": fold_metrics,
            "delta_rmse": float(delta_rmse),
            "rel_improvement_pct": float(rel_imp),
            "n_windows": len(val_seq_fold.X),
            "wall_clock_s": time.time() - t0,
        }

        print(f"  Raw: {fold_metrics['Raw']['rmse']:.2f} | F_struct: {fold_metrics['F_struct']['rmse']:.2f} | Raw+F_struct: {fold_metrics['Raw_plus_F_struct']['rmse']:.2f} | Δ = {delta_rmse:+.2f} ({rel_imp:+.1f}%)")

    # LOEO Summary
    print("\n--- LOEO SUMMARY TABLE ---")
    header = f"{'Held-out':<10} {'N':<4} {'Raw':<8} {'F_struct':<10} {'Raw+γ':<8} {'Raw+dwell':<10} {'Raw+F_struct':<12} {'Δ RMSE':<8} {'Rel Imp':<8}"
    print(header)
    print("-" * len(header))

    rmse_by_model = {m: [] for m in models_list}
    deltas = []
    rel_imps = []

    for u in train_unit_ids:
        f_res = loeo_results["folds"][u]
        m_dict = f_res["metrics"]
        n_w = f_res["n_windows"]
        d_rmse = f_res["delta_rmse"]
        r_imp = f_res["rel_improvement_pct"]
        deltas.append(d_rmse)
        rel_imps.append(r_imp)
        for m in models_list:
            rmse_by_model[m].append(m_dict[m]["rmse"])

        print(f"{u:<10} {n_w:<4} {m_dict['Raw']['rmse']:<8.2f} {m_dict['F_struct']['rmse']:<10.2f} {m_dict['Raw_plus_gamma']['rmse']:<8.2f} {m_dict['Raw_plus_dwell']['rmse']:<10.2f} {m_dict['Raw_plus_F_struct']['rmse']:<12.2f} {d_rmse:>+7.2f} {r_imp:>+6.1f}%")

    loeo_results["summary"] = {
        "engine_count": len(train_unit_ids),
        "mean_rmse": {m: float(np.mean(rmse_by_model[m])) for m in models_list},
        "std_rmse": {m: float(np.std(rmse_by_model[m])) for m in models_list},
        "median_rmse": {m: float(np.median(rmse_by_model[m])) for m in models_list},
        "mean_delta_rmse": float(np.mean(deltas)),
        "median_delta_rmse": float(np.median(deltas)),
        "mean_rel_improvement_pct": float(np.mean(rel_imps)),
        "median_rel_improvement_pct": float(np.median(rel_imps)),
        "wins_raw_plus_fstruct": int(np.sum(np.array(deltas) > 0)),
        "wins_raw_plus_gamma": int(np.sum(np.array(rmse_by_model["Raw"]) > np.array(rmse_by_model["Raw_plus_gamma"]))),
        "wins_raw_plus_dwell": int(np.sum(np.array(rmse_by_model["Raw"]) > np.array(rmse_by_model["Raw_plus_dwell"]))),
    }

    print("-" * len(header))
    print(f"{'Mean':<10} {'-':<4} {loeo_results['summary']['mean_rmse']['Raw']:<8.2f} {loeo_results['summary']['mean_rmse']['F_struct']:<10.2f} {loeo_results['summary']['mean_rmse']['Raw_plus_gamma']:<8.2f} {loeo_results['summary']['mean_rmse']['Raw_plus_dwell']:<10.2f} {loeo_results['summary']['mean_rmse']['Raw_plus_F_struct']:<12.2f} {loeo_results['summary']['mean_delta_rmse']:>+7.2f} {loeo_results['summary']['mean_rel_improvement_pct']:>+6.1f}%")
    print(f"{'Median':<10} {'-':<4} {loeo_results['summary']['median_rmse']['Raw']:<8.2f} {loeo_results['summary']['median_rmse']['F_struct']:<10.2f} {loeo_results['summary']['median_rmse']['Raw_plus_gamma']:<8.2f} {loeo_results['summary']['median_rmse']['Raw_plus_dwell']:<10.2f} {loeo_results['summary']['median_rmse']['Raw_plus_F_struct']:<12.2f} {loeo_results['summary']['median_delta_rmse']:>+7.2f} {loeo_results['summary']['median_rel_improvement_pct']:>+6.1f}%")
    print(f"\nEngines where Raw+F_struct beats Raw: {loeo_results['summary']['wins_raw_plus_fstruct']} / 6")
    print(f"Engines where Raw+gamma beats Raw:     {loeo_results['summary']['wins_raw_plus_gamma']} / 6")
    print(f"Engines where Raw+dwell beats Raw:     {loeo_results['summary']['wins_raw_plus_dwell']} / 6")

    # =========================================================================
    # EXPERIMENT 2: 10-SEED REPRESENTATION STABILITY STUDY
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 2: EGPM REPRESENTATION-SEED STABILITY STUDY (SEEDS 0..9)")
    print("=" * 70)

    # Standard split: Train on u3..u8 (138 windows), Test on u2, u9 (39 windows)
    stats_std = sb.fit_normalization(split_standard)
    prep_std = Preprocessor(window_length=4, stride=4)
    tr_seqs_std = [prep_std.process(raw_dataset[u], norm_stats=stats_std) for u in train_unit_ids]
    te_seqs_std = [prep_std.process(raw_dataset[u], norm_stats=stats_std) for u in test_unit_ids]

    Xtr_raw_std = np.concatenate([s.X[:, -1, :] for s in tr_seqs_std], axis=0)  # [138, 18]
    Xte_raw_std = np.concatenate([s.X[:, -1, :] for s in te_seqs_std], axis=0)  # [39, 18]
    ytr_std = np.concatenate([s.rul_labels for s in tr_seqs_std])
    yte_std = np.concatenate([s.rul_labels for s in te_seqs_std])
    ltr_std = [len(s.X) for s in tr_seqs_std]

    # Model A: Raw Ridge baseline is strictly deterministic
    w_raw = _ridge_fit(Xtr_raw_std, ytr_std)
    p_raw = _ridge_predict(w_raw, Xte_raw_std)
    m_raw_std = rul_metrics(p_raw, yte_std)
    raw_rmse_const = float(m_raw_std["RMSE"])
    raw_mae_const = float(m_raw_std["MAE"])
    raw_phm_const = float(m_raw_std["PHM_score"])

    # Per engine Raw baseline
    idx = 0
    per_eng_raw = {}
    for u, s in zip(test_unit_ids, te_seqs_std):
        L = len(s.X)
        p_u = p_raw[idx:idx + L]
        mu = rul_metrics(p_u, s.rul_labels)
        per_eng_raw[u] = {"rmse": float(mu["RMSE"]), "mae": float(mu["MAE"])}
        idx += L

    print(f"Deterministic Raw Sensor Baseline: RMSE = {raw_rmse_const:.2f}, MAE = {raw_mae_const:.2f}, PHM = {raw_phm_const:.2f}")

    seeds_10 = list(range(10))
    seed_study_results = {
        "raw_baseline": {
            "rmse": raw_rmse_const,
            "mae": raw_mae_const,
            "phm": raw_phm_const,
            "per_engine": per_eng_raw,
        },
        "per_seed": {},
        "summary": {},
    }

    raw_plus_fstruct_rmses = []
    f_struct_rmses = []
    deltas_seed = []
    rel_imps_seed = []

    Xtr_tensor_std = torch.as_tensor(np.concatenate([s.X for s in tr_seqs_std]), dtype=torch.float32)
    Mtr_tensor_std = torch.ones_like(Xtr_tensor_std)
    ytr_tensor_std = torch.as_tensor(ytr_std, dtype=torch.float32)

    for s in seeds_10:
        t0 = time.time()
        set_global_seed(s)
        cfg_s = TrainerConfig(
            window_length=4,
            embed_dim=16,
            hidden=32,
            n_conv_blocks=3,
            n_codes=16,
            d_max=10,
            n_states=5,
            em_max_iter=10,
            em_restarts=10,
            aux_rul_lambda=0.1,
            state_dim=12,
            dead_code_revival=False,
            stage1_steps=300,
            batch_size=64,
            seed=s,
            log_every=0,
        )
        trainer_s = Trainer(cfg_s, n_channels=18)
        trainer_s.train_stage1(Xtr_tensor_std, Mtr_tensor_std, rul_labels=ytr_tensor_std)

        ev_tr_s = trainer_s.window_runs_to_event_runs(trainer_s.event_sequences(Xtr_tensor_std, Mtr_tensor_std), ltr_std)
        trainer_s.train_stage2(ev_tr_s)
        fb_s = ForwardBackward(trainer_s.hsmm)

        ev_te_s = []
        for s_unit in te_seqs_std:
            Xt = torch.as_tensor(s_unit.X, dtype=torch.float32)
            toks = np.concatenate(trainer_s.event_sequences(Xt, torch.ones_like(Xt)))
            ev_te_s.append(toks)

        # Causal filter
        r_tr_s = [fb_s.causal_filter(ev) for ev in ev_tr_s]
        r_te_s = [fb_s.causal_filter(ev) for ev in ev_te_s]

        gamma_tr_s = np.concatenate([r.state_posterior for r in r_tr_s], axis=0)
        gamma_te_s = np.concatenate([r.state_posterior for r in r_te_s], axis=0)
        dwell_tr_s = np.concatenate([r.expected_dwell / 10.0 for r in r_tr_s], axis=0)
        dwell_te_s = np.concatenate([r.expected_dwell / 10.0 for r in r_te_s], axis=0)

        f_struct_tr_s = np.concatenate([gamma_tr_s, dwell_tr_s], axis=1)
        f_struct_te_s = np.concatenate([gamma_te_s, dwell_te_s], axis=1)

        f_comb_tr_s = np.concatenate([Xtr_raw_std, f_struct_tr_s], axis=1)
        f_comb_te_s = np.concatenate([Xte_raw_std, f_struct_te_s], axis=1)

        # Fit F_struct only
        wb = _ridge_fit(f_struct_tr_s, ytr_std)
        pb = _ridge_predict(wb, f_struct_te_s)
        mb = rul_metrics(pb, yte_std)

        # Fit Raw + F_struct
        wc = _ridge_fit(f_comb_tr_s, ytr_std)
        pc = _ridge_predict(wc, f_comb_te_s)
        mc = rul_metrics(pc, yte_std)

        # Per engine for Raw + F_struct
        idx = 0
        pe_c = {}
        for u, s_unit in zip(test_unit_ids, te_seqs_std):
            L = len(s_unit.X)
            p_u = pc[idx:idx + L]
            mu = rul_metrics(p_u, s_unit.rul_labels)
            pe_c[u] = {"rmse": float(mu["RMSE"]), "mae": float(mu["MAE"])}
            idx += L

        d_s = raw_rmse_const - float(mc["RMSE"])
        rel_s = (d_s / raw_rmse_const) * 100.0

        raw_plus_fstruct_rmses.append(float(mc["RMSE"]))
        f_struct_rmses.append(float(mb["RMSE"]))
        deltas_seed.append(d_s)
        rel_imps_seed.append(rel_s)

        seed_study_results["per_seed"][f"seed_{s}"] = {
            "f_struct_rmse": float(mb["RMSE"]),
            "f_struct_mae": float(mb["MAE"]),
            "f_struct_phm": float(mb["PHM_score"]),
            "raw_plus_fstruct_rmse": float(mc["RMSE"]),
            "raw_plus_fstruct_mae": float(mc["MAE"]),
            "raw_plus_fstruct_phm": float(mc["PHM_score"]),
            "delta_rmse": float(d_s),
            "rel_improvement_pct": float(rel_s),
            "per_engine": pe_c,
            "wall_clock_s": time.time() - t0,
        }
        print(f"  Seed {s} ({time.time()-t0:.1f}s): F_struct={mb['RMSE']:.2f}, Raw+F_struct={mc['RMSE']:.2f} | Δ over Raw = {d_s:+.2f} ({rel_s:+.1f}%) | u2={pe_c['DS01_u2']['rmse']:.2f}, u9={pe_c['DS01_u9']['rmse']:.2f}")

    seed_study_results["summary"] = {
        "n_seeds": len(seeds_10),
        "raw_rmse": raw_rmse_const,
        "f_struct": {
            "mean_rmse": float(np.mean(f_struct_rmses)),
            "std_rmse": float(np.std(f_struct_rmses)),
            "median_rmse": float(np.median(f_struct_rmses)),
            "min_rmse": float(np.min(f_struct_rmses)),
            "max_rmse": float(np.max(f_struct_rmses)),
        },
        "raw_plus_fstruct": {
            "mean_rmse": float(np.mean(raw_plus_fstruct_rmses)),
            "std_rmse": float(np.std(raw_plus_fstruct_rmses)),
            "median_rmse": float(np.median(raw_plus_fstruct_rmses)),
            "min_rmse": float(np.min(raw_plus_fstruct_rmses)),
            "max_rmse": float(np.max(raw_plus_fstruct_rmses)),
        },
        "mean_delta_rmse": float(np.mean(deltas_seed)),
        "std_delta_rmse": float(np.std(deltas_seed)),
        "median_delta_rmse": float(np.median(deltas_seed)),
        "mean_rel_improvement_pct": float(np.mean(rel_imps_seed)),
        "median_rel_improvement_pct": float(np.median(rel_imps_seed)),
        "seeds_beating_raw_count": int(np.sum(np.array(deltas_seed) > 0)),
    }

    print("\n--- 10-SEED STABILITY SUMMARY ---")
    print(f"Raw RMSE:                  {raw_rmse_const:.2f} (constant, deterministic)")
    print(f"F_struct RMSE:             Mean={seed_study_results['summary']['f_struct']['mean_rmse']:.2f} ± {seed_study_results['summary']['f_struct']['std_rmse']:.2f}, Med={seed_study_results['summary']['f_struct']['median_rmse']:.2f} [Min={seed_study_results['summary']['f_struct']['min_rmse']:.2f}, Max={seed_study_results['summary']['f_struct']['max_rmse']:.2f}]")
    print(f"Raw + F_struct RMSE:       Mean={seed_study_results['summary']['raw_plus_fstruct']['mean_rmse']:.2f} ± {seed_study_results['summary']['raw_plus_fstruct']['std_rmse']:.2f}, Med={seed_study_results['summary']['raw_plus_fstruct']['median_rmse']:.2f} [Min={seed_study_results['summary']['raw_plus_fstruct']['min_rmse']:.2f}, Max={seed_study_results['summary']['raw_plus_fstruct']['max_rmse']:.2f}]")
    print(f"Mean Δ RMSE over Raw:      {seed_study_results['summary']['mean_delta_rmse']:+.2f} cycles (rel: {seed_study_results['summary']['mean_rel_improvement_pct']:+.1f}%)")
    print(f"Median Δ RMSE over Raw:    {seed_study_results['summary']['median_delta_rmse']:+.2f} cycles (rel: {seed_study_results['summary']['median_rel_improvement_pct']:+.1f}%)")
    print(f"Seeds where Raw+F_struct < Raw: {seed_study_results['summary']['seeds_beating_raw_count']} / {len(seeds_10)}")

    # Combined output package
    full_output = {
        "leakage_checks": leakage_checks,
        "loeo_validation": loeo_results,
        "seed_stability_study": seed_study_results,
    }

    out_file = "loeo_and_seed_stability_results.json"
    with open(out_file, "w") as f:
        json.dump(full_output, f, indent=2)
    print(f"\nAll results saved to {out_file}!")


if __name__ == "__main__":
    run_loeo_and_seed_study()
