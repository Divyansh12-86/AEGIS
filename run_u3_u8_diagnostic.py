"""EGPM u3/u8 Transfer-Failure Diagnostic Audit.

Diagnoses whether the transfer failures of DS01_u3 and DS01_u8 in LOEO
are caused by:
(a) Data coverage / flight envelope shift (operational conditions W outside pool distribution)
(b) Representation failure (VQ / HSMM mismodeling u3/u8)
(c) Inconclusive

Checks performed:
1. Codebook utilization: per-engine VQ usage histogram and perplexity (via vq_codebook._collapse_stats).
2. Hungarian-matched event stability (via metrics._hungarian_usage_match).
3. Operational-condition coverage (channels 0..3: alt, Mach, TRA, T2) vs comparison pool {u4, u5, u6, u7}.
4. Normalization drift: train stats (mean, std) including vs excluding u3 and u8.
5. Gamma and dwell distributions: state posteriors and dwell for held-out u3/u8 vs held-out u4-u7.
6. Error decomposition by RUL bins (>50, 20-50, <20) for u3 and u8.
"""
from __future__ import annotations

import json
import time
import numpy as np
import scipy.stats
import torch

from egpm.data.loaders import NCMAPSSLoader
from egpm.data.split_builder import SplitBuilder, NormalizationStats
from egpm.experiments.ablation_arms import _downsample, _ridge_fit, _ridge_predict
from egpm.preprocessing.preprocessor import Preprocessor
from egpm.training.trainer import Trainer, TrainerConfig
from egpm.grammar.forward_backward import ForwardBackward
from egpm.evaluation.evaluator import rul_metrics
from egpm.interpretability.metrics import _hungarian_usage_match
from egpm.utils import set_global_seed


def run_diagnostic():
    print("=" * 80)
    print("EGPM — u3/u8 TRANSFER-FAILURE DIAGNOSTIC AUDIT")
    print("=" * 80)

    dataset_path = "N-CMAPSS/N-CMAPSS DS01 005.h5"
    raw_dataset = NCMAPSSLoader().load(dataset_path)
    raw_dataset.records = [_downsample(r) for r in raw_dataset.records]

    train_unit_ids = ['DS01_u3', 'DS01_u4', 'DS01_u5', 'DS01_u6', 'DS01_u7', 'DS01_u8']
    comparison_pool = ['DS01_u4', 'DS01_u5', 'DS01_u6', 'DS01_u7']
    target_units = ['DS01_u3', 'DS01_u8']

    results = {}

    # =========================================================================
    # PART 1.3: OPERATIONAL-CONDITION COVERAGE (DATA LEVEL)
    # =========================================================================
    print("\n" + "=" * 70)
    print("CHECK 3: OPERATIONAL-CONDITION COVERAGE (W ARRAY: alt, Mach, TRA, T2)")
    print("=" * 70)
    
    channel_names = ["alt (ft)", "Mach (-)", "TRA (deg)", "T2 (deg R)"]
    
    # Extract raw W arrays
    W_pool = np.concatenate([raw_dataset[u].signals[:, :4] for u in comparison_pool], axis=0)
    W_u3 = raw_dataset['DS01_u3'].signals[:, :4]
    W_u8 = raw_dataset['DS01_u8'].signals[:, :4]

    pool_mean = W_pool.mean(axis=0)
    pool_std = W_pool.std(axis=0)
    pool_p5 = np.percentile(W_pool, 5, axis=0)
    pool_p95 = np.percentile(W_pool, 95, axis=0)
    pool_min = W_pool.min(axis=0)
    pool_max = W_pool.max(axis=0)

    op_results = {
        "pool": {
            "n_samples": int(len(W_pool)),
            "mean": pool_mean.tolist(),
            "std": pool_std.tolist(),
            "p5": pool_p5.tolist(),
            "p95": pool_p95.tolist(),
            "min": pool_min.tolist(),
            "max": pool_max.tolist(),
        },
        "units": {}
    }

    print(f"Comparison pool {comparison_pool} (N={len(W_pool)} samples):")
    for c, cname in enumerate(channel_names):
        print(f"  {cname:<15}: mean={pool_mean[c]:10.2f}, std={pool_std[c]:8.2f}, 5th%={pool_p5[c]:10.2f}, 95th%={pool_p95[c]:10.2f}, range=[{pool_min[c]:.2f}, {pool_max[c]:.2f}]")

    for u in train_unit_ids:
        Wu = raw_dataset[u].signals[:, :4]
        u_mean = Wu.mean(axis=0)
        u_std = Wu.std(axis=0)
        # Outliers relative to pool [p5, p95]
        outliers_per_ch = [float(np.mean((Wu[:, c] < pool_p5[c]) | (Wu[:, c] > pool_p95[c]))) for c in range(4)]
        # Any channel outside pool [p5, p95]
        any_outlier = float(np.mean(np.any((Wu < pool_p5) | (Wu > pool_p95), axis=1)))
        # Outside pool full [min, max]
        outside_full_range = [float(np.mean((Wu[:, c] < pool_min[c]) | (Wu[:, c] > pool_max[c]))) for c in range(4)]
        # KS-test against pool
        ks_stats = [scipy.stats.ks_2samp(Wu[:, c], W_pool[:, c]) for c in range(4)]

        op_results["units"][u] = {
            "n_samples": int(len(Wu)),
            "mean": u_mean.tolist(),
            "std": u_std.tolist(),
            "outliers_p5_p95_per_ch": outliers_per_ch,
            "any_ch_outlier_p5_p95": any_outlier,
            "outside_full_range_per_ch": outside_full_range,
            "ks_statistic": [float(ks.statistic) for ks in ks_stats],
            "ks_pvalue": [float(ks.pvalue) for ks in ks_stats],
        }

        print(f"\nUnit {u} (N={len(Wu)} samples):")
        for c, cname in enumerate(channel_names):
            print(f"  {cname:<15}: mean={u_mean[c]:10.2f}, std={u_std[c]:8.2f} | out[p5,p95]={outliers_per_ch[c]*100:5.1f}% | out[min,max]={outside_full_range[c]*100:4.1f}% | KS={ks_stats[c].statistic:.3f} (p={ks_stats[c].pvalue:.2e})")
        print(f"  -> Any channel outside pool [5th, 95th]: {any_outlier*100:.1f}%")

    results["operational_coverage"] = op_results

    # =========================================================================
    # PART 1.4: NORMALIZATION DRIFT
    # =========================================================================
    print("\n" + "=" * 70)
    print("CHECK 4: NORMALIZATION DRIFT")
    print("=" * 70)

    # 1. Full training pool (all 6 units)
    rec_all = [raw_dataset[u] for u in train_unit_ids]
    stk_all = np.concatenate([r.signals for r in rec_all], axis=0)
    mu_all = stk_all.mean(axis=0)
    sig_all = stk_all.std(axis=0)

    # 2. Excluding u3
    rec_ex_u3 = [raw_dataset[u] for u in train_unit_ids if u != 'DS01_u3']
    stk_ex_u3 = np.concatenate([r.signals for r in rec_ex_u3], axis=0)
    mu_ex_u3 = stk_ex_u3.mean(axis=0)
    sig_ex_u3 = stk_ex_u3.std(axis=0)

    # 3. Excluding u8
    rec_ex_u8 = [raw_dataset[u] for u in train_unit_ids if u != 'DS01_u8']
    stk_ex_u8 = np.concatenate([r.signals for r in rec_ex_u8], axis=0)
    mu_ex_u8 = stk_ex_u8.mean(axis=0)
    sig_ex_u8 = stk_ex_u8.std(axis=0)

    # 4. Excluding both u3 and u8 (pool of 4)
    rec_pool4 = [raw_dataset[u] for u in comparison_pool]
    stk_pool4 = np.concatenate([r.signals for r in rec_pool4], axis=0)
    mu_pool4 = stk_pool4.mean(axis=0)
    sig_pool4 = stk_pool4.std(axis=0)

    norm_drift_results = {
        "mu_all": mu_all.tolist(),
        "sig_all": sig_all.tolist(),
        "shift_ex_u3": {
            "delta_mu": (mu_all - mu_ex_u3).tolist(),
            "rel_delta_mu_pct": ((mu_all - mu_ex_u3) / np.where(sig_all < 1e-8, 1.0, sig_all) * 100.0).tolist(),
            "sig_ratio": (sig_all / np.where(sig_ex_u3 < 1e-8, 1.0, sig_ex_u3)).tolist(),
        },
        "shift_ex_u8": {
            "delta_mu": (mu_all - mu_ex_u8).tolist(),
            "rel_delta_mu_pct": ((mu_all - mu_ex_u8) / np.where(sig_all < 1e-8, 1.0, sig_all) * 100.0).tolist(),
            "sig_ratio": (sig_all / np.where(sig_ex_u8 < 1e-8, 1.0, sig_ex_u8)).tolist(),
        },
        "shift_pool4": {
            "delta_mu": (mu_all - mu_pool4).tolist(),
            "rel_delta_mu_pct": ((mu_all - mu_pool4) / np.where(sig_all < 1e-8, 1.0, sig_all) * 100.0).tolist(),
            "sig_ratio": (sig_all / np.where(sig_pool4 < 1e-8, 1.0, sig_pool4)).tolist(),
        }
    }

    print(f"{'Ch':<4} {'Name':<12} {'μ_all':<10} {'μ_ex_u3':<10} {'Δμ_u3 (in σ %)':<16} {'σ_all/σ_ex_u3':<14} {'Δμ_u8 (in σ %)':<16} {'σ_all/σ_ex_u8':<14}")
    print("-" * 92)
    names_18 = ["alt", "Mach", "TRA", "T2"] + [f"s_{i}" for i in range(1, 15)]
    for c in range(18):
        shift_u3_sig = (mu_all[c] - mu_ex_u3[c]) / max(sig_all[c], 1e-8) * 100.0
        sig_ratio_u3 = sig_all[c] / max(sig_ex_u3[c], 1e-8)
        shift_u8_sig = (mu_all[c] - mu_ex_u8[c]) / max(sig_all[c], 1e-8) * 100.0
        sig_ratio_u8 = sig_all[c] / max(sig_ex_u8[c], 1e-8)
        print(f"{c:<4} {names_18[c]:<12} {mu_all[c]:<10.2f} {mu_ex_u3[c]:<10.2f} {shift_u3_sig:+15.2f}% {sig_ratio_u3:14.3f} {shift_u8_sig:+15.2f}% {sig_ratio_u8:14.3f}")

    results["normalization_drift"] = norm_drift_results

    # =========================================================================
    # PART 1.1 & 1.2: VQ UTILIZATION & HUNGARIAN STABILITY (STANDARD MODEL)
    # =========================================================================
    print("\n" + "=" * 70)
    print("CHECK 1 & 2: VQ CODEBOOK UTILIZATION & HUNGARIAN MATCH (STANDARD 12/4 MODEL)")
    print("=" * 70)

    # Train standard 12/4 candidate on all 6 units
    sb = SplitBuilder(raw_dataset, seed=0)
    split_standard = sb.build()
    stats_std = sb.fit_normalization(split_standard)
    prep_std = Preprocessor(window_length=4, stride=4)

    tr_seqs_std = [prep_std.process(raw_dataset[u], norm_stats=stats_std) for u in train_unit_ids]

    set_global_seed(0)
    cfg_std = TrainerConfig(
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
    trainer_std = Trainer(cfg_std, n_channels=18)
    X_all_tensor = torch.as_tensor(np.concatenate([s.X for s in tr_seqs_std]), dtype=torch.float32)
    M_all_tensor = torch.ones_like(X_all_tensor)
    y_all_tensor = torch.as_tensor(np.concatenate([s.rul_labels for s in tr_seqs_std]), dtype=torch.float32)
    trainer_std.train_stage1(X_all_tensor, M_all_tensor, rul_labels=y_all_tensor)

    # Extract tokens for each engine under standard model
    tokens_per_unit_std = {}
    vq_stats_per_unit_std = {}
    for u, s in zip(train_unit_ids, tr_seqs_std):
        X_u = torch.as_tensor(s.X, dtype=torch.float32)
        M_u = torch.ones_like(X_u)
        ev_u = np.concatenate(trainer_std.event_sequences(X_u, M_u))
        tokens_per_unit_std[u] = ev_u
        ev_u_tensor = torch.as_tensor(ev_u, dtype=torch.long)
        cstats = trainer_std.vq._collapse_stats(ev_u_tensor)
        vq_stats_per_unit_std[u] = {
            "perplexity": cstats["perplexity"],
            "effective_size": cstats["effective_size"],
            "mean_usage_entropy": cstats["mean_usage_entropy"],
            "usage_histogram": cstats["usage_histogram"].tolist(),
            "n_tokens": len(ev_u),
        }

    # Pooled comparison pool tokens
    ev_pool_std = np.concatenate([tokens_per_unit_std[u] for u in comparison_pool])
    cstats_pool = trainer_std.vq._collapse_stats(torch.as_tensor(ev_pool_std, dtype=torch.long))
    vq_stats_per_unit_std["pool_u4_u7"] = {
        "perplexity": cstats_pool["perplexity"],
        "effective_size": cstats_pool["effective_size"],
        "mean_usage_entropy": cstats_pool["mean_usage_entropy"],
        "usage_histogram": cstats_pool["usage_histogram"].tolist(),
        "n_tokens": len(ev_pool_std),
    }

    print("\n--- VQ CODEBOOK UTILIZATION (Standard Model) ---")
    print(f"{'Unit':<12} {'N tokens':<10} {'Perplexity':<12} {'Active Codes (Eff Size)':<24} {'Entropy':<10}")
    print("-" * 68)
    for u in train_unit_ids + ["pool_u4_u7"]:
        st = vq_stats_per_unit_std[u]
        print(f"{u:<12} {st['n_tokens']:<10} {st['perplexity']:<12.2f} {st['effective_size']:<24} {st['mean_usage_entropy']:<10.2f}")

    print("\nDetailed Token Histogram (K=16) across units:")
    print("Token ID:     " + " ".join(f"{k:4d}" for k in range(16)))
    for u in train_unit_ids + ["pool_u4_u7"]:
        hist = np.array(vq_stats_per_unit_std[u]["usage_histogram"])
        print(f"{u:<12}: " + " ".join(f"{hist[k]*100:4.1f}" for k in range(16)))

    # Hungarian usage match
    print("\n--- HUNGARIAN USAGE MATCH (Standard Model) ---")
    h_matches = {}
    pool_hist = np.array(vq_stats_per_unit_std["pool_u4_u7"]["usage_histogram"])
    
    # Target units vs pool
    h_matches["u3_vs_pool"] = _hungarian_usage_match(np.array(vq_stats_per_unit_std["DS01_u3"]["usage_histogram"]), pool_hist)
    h_matches["u8_vs_pool"] = _hungarian_usage_match(np.array(vq_stats_per_unit_std["DS01_u8"]["usage_histogram"]), pool_hist)
    print(f"Hungarian match u3 vs Pool {comparison_pool}: {h_matches['u3_vs_pool']:.4f}")
    print(f"Hungarian match u8 vs Pool {comparison_pool}: {h_matches['u8_vs_pool']:.4f}")

    # Target units vs each comparison engine
    for tgt in target_units:
        tgt_hist = np.array(vq_stats_per_unit_std[tgt]["usage_histogram"])
        for comp in comparison_pool:
            comp_hist = np.array(vq_stats_per_unit_std[comp]["usage_histogram"])
            match = _hungarian_usage_match(tgt_hist, comp_hist)
            h_matches[f"{tgt}_vs_{comp}"] = match
            print(f"  {tgt} vs {comp}: {match:.4f}")

    # Within comparison pool baseline
    within_pool_matches = []
    for i in range(len(comparison_pool)):
        for j in range(i + 1, len(comparison_pool)):
            h1 = np.array(vq_stats_per_unit_std[comparison_pool[i]]["usage_histogram"])
            h2 = np.array(vq_stats_per_unit_std[comparison_pool[j]]["usage_histogram"])
            m = _hungarian_usage_match(h1, h2)
            within_pool_matches.append(m)
            h_matches[f"{comparison_pool[i]}_vs_{comparison_pool[j]}"] = m
            print(f"  Baseline: {comparison_pool[i]} vs {comparison_pool[j]}: {m:.4f}")

    h_matches["within_pool_mean"] = float(np.mean(within_pool_matches))
    h_matches["within_pool_std"] = float(np.std(within_pool_matches))
    h_matches["within_pool_min"] = float(np.min(within_pool_matches))
    h_matches["within_pool_max"] = float(np.max(within_pool_matches))
    print(f"Within-pool baseline match: {h_matches['within_pool_mean']:.4f} ± {h_matches['within_pool_std']:.4f} [range: {h_matches['within_pool_min']:.4f} - {h_matches['within_pool_max']:.4f}]")

    results["vq_stats_standard"] = vq_stats_per_unit_std
    results["hungarian_matches"] = h_matches

    # =========================================================================
    # PART 1.1, 1.5, 1.6: LOEO FOLDS EVALUATION (γ, DWELL, RESIDUALS BY RUL BIN)
    # =========================================================================
    print("\n" + "=" * 70)
    print("CHECK 5 & 6: LOEO HELD-OUT γ/DWELL DISTRIBUTIONS & RUL BIN RESIDUALS")
    print("=" * 70)

    loeo_fold_diagnostics = {}

    for k_fold, val_unit in enumerate(train_unit_ids):
        tr_fold = [u for j, u in enumerate(train_unit_ids) if j != k_fold]
        print(f"\nEvaluating LOEO Fold {k_fold + 1}/6: Held-out = {val_unit}")

        # Normalization fit strictly on tr_fold
        train_records_fold = [r for r in raw_dataset.records if r.unit_id in set(tr_fold)]
        stacked_fold = np.concatenate([r.signals for r in train_records_fold], axis=0)
        mean_fold = stacked_fold.mean(axis=0)
        std_fold = np.where(stacked_fold.std(axis=0) < 1e-8, 1.0, stacked_fold.std(axis=0))
        norm_stats_fold = NormalizationStats(mean=mean_fold, std=std_fold, train_unit_ids=sorted(tr_fold))

        prep = Preprocessor(window_length=4, stride=4)
        tr_seqs = [prep.process(raw_dataset[u], norm_stats=norm_stats_fold) for u in tr_fold]
        val_seq = prep.process(raw_dataset[val_unit], norm_stats=norm_stats_fold)

        # Raw features
        X_tr_raw = [s.X[:, -1, :] for s in tr_seqs]
        X_va_raw = val_seq.X[:, -1, :]
        y_tr = [s.rul_labels for s in tr_seqs]
        y_va = val_seq.rul_labels
        y_tr_flat = np.concatenate(y_tr)
        X_tr_raw_flat = np.concatenate(X_tr_raw, axis=0)

        # Train 12/4 candidate on tr_fold
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
        X_tr_tensor = torch.as_tensor(np.concatenate([s.X for s in tr_seqs]), dtype=torch.float32)
        M_tr_tensor = torch.ones_like(X_tr_tensor)
        y_tr_tensor = torch.as_tensor(y_tr_flat, dtype=torch.float32)
        trainer_fold.train_stage1(X_tr_tensor, M_tr_tensor, rul_labels=y_tr_tensor)

        ltr_fold = [len(s.X) for s in tr_seqs]
        ev_tr_fold = trainer_fold.window_runs_to_event_runs(trainer_fold.event_sequences(X_tr_tensor, M_tr_tensor), ltr_fold)
        trainer_fold.train_stage2(ev_tr_fold)
        fb_fold = ForwardBackward(trainer_fold.hsmm)

        # Causal filtering on train and val
        r_tr_fold = [fb_fold.causal_filter(ev) for ev in ev_tr_fold]
        gamma_tr = [r.state_posterior for r in r_tr_fold]
        dwell_tr = [r.expected_dwell / 10.0 for r in r_tr_fold]
        f_struct_tr = [np.concatenate([g, d], axis=1) for g, d in zip(gamma_tr, dwell_tr)]
        f_struct_tr_flat = np.concatenate(f_struct_tr, axis=0)

        X_val_tensor = torch.as_tensor(val_seq.X, dtype=torch.float32)
        ev_val = np.concatenate(trainer_fold.event_sequences(X_val_tensor, torch.ones_like(X_val_tensor)))
        r_val = fb_fold.causal_filter(ev_val)
        gamma_va = r_val.state_posterior      # [T, 5]
        dwell_va = r_val.expected_dwell / 10.0 # [T, 5]
        f_struct_va = np.concatenate([gamma_va, dwell_va], axis=1)

        # Token stats for held-out fold
        cstats_fold = trainer_fold.vq._collapse_stats(torch.as_tensor(ev_val, dtype=torch.long))

        # Ridge models
        w_raw = _ridge_fit(X_tr_raw_flat, y_tr_flat)
        pred_raw = _ridge_predict(w_raw, X_va_raw)

        raw_plus_f_tr = np.concatenate([X_tr_raw_flat, f_struct_tr_flat], axis=1)
        raw_plus_f_va = np.concatenate([X_va_raw, f_struct_va], axis=1)
        w_comb = _ridge_fit(raw_plus_f_tr, y_tr_flat)
        pred_comb = _ridge_predict(w_comb, raw_plus_f_va)

        raw_plus_dwell_tr = np.concatenate([X_tr_raw_flat, np.concatenate(dwell_tr, axis=0)], axis=1)
        raw_plus_dwell_va = np.concatenate([X_va_raw, dwell_va], axis=1)
        w_dwell = _ridge_fit(raw_plus_dwell_tr, y_tr_flat)
        pred_dwell = _ridge_predict(w_dwell, raw_plus_dwell_va)

        m_raw = rul_metrics(pred_raw, y_va)
        m_comb = rul_metrics(pred_comb, y_va)
        m_dwell = rul_metrics(pred_dwell, y_va)

        # Distributional gamma & dwell stats
        mean_gamma_val = gamma_va.mean(axis=0)  # [5]
        mean_dwell_val = dwell_va.mean(axis=0)  # [5]
        mean_gamma_tr = np.concatenate(gamma_tr, axis=0).mean(axis=0)
        mean_dwell_tr = np.concatenate(dwell_tr, axis=0).mean(axis=0)

        # Check 6: Residual decomposition by RUL bins
        bins = {
            "early (>50)": y_va > 50,
            "mid (20-50)": (y_va >= 20) & (y_va <= 50),
            "late (<20)": y_va < 20,
        }
        bin_metrics = {}
        for bname, mask in bins.items():
            if np.sum(mask) > 0:
                p_r = pred_raw[mask]
                p_c = pred_comb[mask]
                y_b = y_va[mask]
                rmse_r = float(np.sqrt(np.mean((p_r - y_b) ** 2)))
                mae_r = float(np.mean(np.abs(p_r - y_b)))
                rmse_c = float(np.sqrt(np.mean((p_c - y_b) ** 2)))
                mae_c = float(np.mean(np.abs(p_c - y_b)))
                delta_b = rmse_r - rmse_c
                bin_metrics[bname] = {
                    "n_samples": int(np.sum(mask)),
                    "raw_rmse": rmse_r,
                    "raw_mae": mae_r,
                    "comb_rmse": rmse_c,
                    "comb_mae": mae_c,
                    "delta_rmse": delta_b,
                    "mean_actual_rul": float(np.mean(y_b)),
                    "mean_pred_raw": float(np.mean(p_r)),
                    "mean_pred_comb": float(np.mean(p_c)),
                }

        loeo_fold_diagnostics[val_unit] = {
            "n_windows": len(val_seq.X),
            "vq_stats": {
                "perplexity": cstats_fold["perplexity"],
                "effective_size": cstats_fold["effective_size"],
                "usage_histogram": cstats_fold["usage_histogram"].tolist(),
            },
            "gamma_dwell": {
                "val_mean_gamma": mean_gamma_val.tolist(),
                "val_mean_dwell": mean_dwell_val.tolist(),
                "tr_mean_gamma": mean_gamma_tr.tolist(),
                "tr_mean_dwell": mean_dwell_tr.tolist(),
            },
            "overall_metrics": {
                "raw_rmse": float(m_raw["RMSE"]),
                "comb_rmse": float(m_comb["RMSE"]),
                "dwell_rmse": float(m_dwell["RMSE"]),
                "delta_rmse": float(m_raw["RMSE"] - m_comb["RMSE"]),
            },
            "bin_breakdown": bin_metrics,
        }

        print(f"  Overall: Raw RMSE={m_raw['RMSE']:.2f}, Raw+F_struct RMSE={m_comb['RMSE']:.2f}, Δ={m_raw['RMSE']-m_comb['RMSE']:+.2f}")
        print(f"  Mean Gamma held-out: {[round(x, 3) for x in mean_gamma_val]} (Train: {[round(x, 3) for x in mean_gamma_tr]})")
        print(f"  Mean Dwell held-out: {[round(x, 3) for x in mean_dwell_val]} (Train: {[round(x, 3) for x in mean_dwell_tr]})")
        print("  RUL Bin Breakdown:")
        for bname, bm in bin_metrics.items():
            print(f"    {bname:<12} (N={bm['n_samples']:2d}): Raw RMSE={bm['raw_rmse']:5.2f}, Comb RMSE={bm['comb_rmse']:5.2f}, Δ={bm['delta_rmse']:+5.2f}")

    results["loeo_diagnostics"] = loeo_fold_diagnostics

    # Save results to json
    with open("u3_u8_diagnostic_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved diagnostic results to u3_u8_diagnostic_results.json")


if __name__ == "__main__":
    run_diagnostic()
