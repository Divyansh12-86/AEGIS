"""EGPM Event-State Information Audit.

Evaluates whether the recovered discrete event vocabulary (12/4 subspace decoupling)
represents meaningful machine operating/degradation structure or superficial token diversity.

Strictly adheres to:
- No architecture changes.
- No retraining with different hyperparameters.
- Same dataset (DS01), W=4, S=4, RF=19, causal filtering, Ridge lambda=1.0, seeds [0, 1, 2].
- Evaluates Experiments 1-8 and outputs data for Section 1-9.
"""
from __future__ import annotations

import json
import time
import numpy as np
import scipy.stats
import torch

from egpm.data.loaders import NCMAPSSLoader
from egpm.data.split_builder import SplitBuilder
from egpm.experiments.ablation_arms import _downsample, _ridge_fit, _ridge_predict
from egpm.preprocessing.preprocessor import Preprocessor
from egpm.training.trainer import Trainer, TrainerConfig
from egpm.grammar.forward_backward import ForwardBackward
from egpm.evaluation import rul_metrics
from egpm.interpretability.metrics import event_stability
from egpm.utils import set_global_seed


CHANNEL_NAMES = [
    "alt", "Mach", "TRA", "T2",        # Flight conditions (0..3)
    "T24", "T30", "T48", "T50",        # Temperatures (4..7)
    "P15", "P2", "P21", "P24", "Ps30", "P40", "P50",  # Pressures (8..14)
    "Nf", "Nc", "Wf"                   # Speeds and fuel flow (15..17)
]


def compute_transition_stats(event_seqs, n_codes=16):
    counts = np.zeros((n_codes, n_codes), dtype=np.float64)
    total_transitions = 0
    self_transitions = 0
    for seq in event_seqs:
        for t in range(1, len(seq)):
            i, j = int(seq[t - 1]), int(seq[t])
            counts[i, j] += 1.0
            total_transitions += 1
            if i == j:
                self_transitions += 1

    if total_transitions == 0:
        return {
            "matrix": counts.tolist(),
            "total_transitions": 0,
            "self_transition_rate": 0.0,
            "n_observed_transitions": 0,
            "entropy_bits": 0.0,
            "entropy_nats": 0.0,
            "top_transitions": [],
        }

    self_trans_rate = float(self_transitions / total_transitions)
    n_observed = int(np.sum(counts > 0))

    # Conditional entropy H(E_t | E_{t-1})
    row_sums = counts.sum(axis=1)
    ent_bits = 0.0
    ent_nats = 0.0
    top_trans = []

    for i in range(n_codes):
        if row_sums[i] > 0:
            p_i = row_sums[i] / total_transitions
            row_p = counts[i] / row_sums[i]
            for j in range(n_codes):
                if counts[i, j] > 0:
                    top_trans.append({"from": i, "to": j, "count": int(counts[i, j]), "prob": float(row_p[j])})
                    ent_bits -= p_i * row_p[j] * np.log2(row_p[j])
                    ent_nats -= p_i * row_p[j] * np.log(row_p[j])

    top_trans.sort(key=lambda x: x["count"], reverse=True)

    # Row-normalized transition matrix
    with np.errstate(divide='ignore', invalid='ignore'):
        trans_mat = np.where(row_sums[:, None] > 0, counts / row_sums[:, None], 0.0)

    return {
        "matrix": trans_mat.tolist(),
        "counts": counts.tolist(),
        "total_transitions": total_transitions,
        "self_transition_rate": self_trans_rate,
        "n_observed_transitions": n_observed,
        "entropy_bits": float(ent_bits),
        "entropy_nats": float(ent_nats),
        "top_transitions": top_trans[:10],
    }


def compute_run_lengths(event_seqs):
    run_lens = []
    for seq in event_seqs:
        if len(seq) == 0:
            continue
        cur_len = 1
        for t in range(1, len(seq)):
            if seq[t] == seq[t - 1]:
                cur_len += 1
            else:
                run_lens.append(cur_len)
                cur_len = 1
        run_lens.append(cur_len)
    if not run_lens:
        return {"mean": 0.0, "median": 0.0, "min": 0, "max": 0, "distribution": {}}
    counts = np.bincount(run_lens)
    dist = {int(k): int(counts[k]) for k in range(len(counts)) if counts[k] > 0}
    return {
        "mean": float(np.mean(run_lens)),
        "median": float(np.median(run_lens)),
        "std": float(np.std(run_lens)),
        "min": int(np.min(run_lens)),
        "max": int(np.max(run_lens)),
        "distribution": dist,
        "all_lengths": run_lens,
    }


def compute_running_frequency(seqs, n_codes=16):
    out = []
    for seq in seqs:
        L = len(seq)
        freq = np.zeros((L, n_codes), dtype=np.float64)
        c = np.zeros(n_codes, dtype=np.float64)
        for t in range(L):
            c[int(seq[t])] += 1.0
            freq[t] = c / (t + 1.0)
        out.append(freq)
    return out


def run_audit():
    print("=" * 75)
    print("EGPM EVENT-STATE INFORMATION AUDIT (12/4 SUBSPACE)")
    print("=" * 75)

    dataset_path = "N-CMAPSS/N-CMAPSS DS01 005.h5"
    raw_dataset = NCMAPSSLoader().load(dataset_path)
    raw_dataset.records = [_downsample(r) for r in raw_dataset.records]

    sb = SplitBuilder(raw_dataset, seed=0)
    split = sb.build()
    stats = sb.fit_normalization(split)

    tr_units = split.train_unit_ids
    val_units = split.val_unit_ids
    te_units = split.test_unit_ids
    print(f"Train units: {tr_units}")
    print(f"Test units:  {te_units}")

    prep = Preprocessor(window_length=4, stride=4)
    tr_seqs = [prep.process(raw_dataset[u], norm_stats=stats) for u in tr_units]
    te_seqs = [prep.process(raw_dataset[u], norm_stats=stats) for u in te_units]

    Xtr = torch.as_tensor(np.concatenate([s.X for s in tr_seqs]), dtype=torch.float32)
    Mtr = torch.ones_like(Xtr)
    ytr = np.concatenate([s.rul_labels for s in tr_seqs])
    yte = np.concatenate([s.rul_labels for s in te_seqs])
    ltr = [len(s.X) for s in tr_seqs]
    lte = [len(s.X) for s in te_seqs]

    # Pre-extract mean sensor values per window across all 18 channels
    # Standardized: mean across window timesteps
    Xtr_std_mean = np.concatenate([s.X.mean(axis=1) for s in tr_seqs], axis=0)  # [138, 18]
    Xte_std_mean = np.concatenate([s.X.mean(axis=1) for s in te_seqs], axis=0)  # [39, 18]

    # Raw physical values per window (mean over window of raw records)
    raw_tr_list = []
    for u in tr_units:
        rec = raw_dataset[u]
        # windowing right-aligned or start:s, s+W
        # Preprocessor uses s:s+W
        w_means = [rec.signals[s:s+4].mean(axis=0) for s in range(0, len(rec.signals) - 3, 4)]
        raw_tr_list.append(np.array(w_means))
    raw_tr_mean = np.concatenate(raw_tr_list, axis=0)

    raw_te_list = []
    for u in te_units:
        rec = raw_dataset[u]
        w_means = [rec.signals[s:s+4].mean(axis=0) for s in range(0, len(rec.signals) - 3, 4)]
        raw_te_list.append(np.array(w_means))
    raw_te_mean = np.concatenate(raw_te_list, axis=0)

    seeds = [0, 1, 2]
    audit_data = {
        "seeds": seeds,
        "train_units": tr_units,
        "test_units": te_units,
        "exp1_rul_readouts": {},
        "exp2_operating_regimes": {},
        "exp3_degradation": {},
        "exp4_persistence": {},
        "exp5_transitions": {},
        "exp6_shuffled_baseline": {},
        "exp7_stability": {},
        "exp8_sensor_associations": {},
    }

    per_seed_data = {}

    for s in seeds:
        print(f"\n--- Loading / Reproducing Seed {s} ---")
        t0 = time.time()
        set_global_seed(s)
        cfg = TrainerConfig(
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
        trainer = Trainer(cfg, n_channels=Xtr.shape[-1])
        ytr_tensor = torch.as_tensor(ytr, dtype=torch.float32)
        s1_stats = trainer.train_stage1(Xtr, Mtr, rul_labels=ytr_tensor)

        # Discrete events:
        ev_tr = trainer.window_runs_to_event_runs(trainer.event_sequences(Xtr, Mtr), ltr)
        trainer.train_stage2(ev_tr)
        fb = ForwardBackward(trainer.hsmm)

        ev_te = []
        for s_unit in te_seqs:
            Xt = torch.as_tensor(s_unit.X, dtype=torch.float32)
            toks = np.concatenate(trainer.event_sequences(Xt, torch.ones_like(Xt)))
            ev_te.append(toks)

        # Causal filter
        r_tr = [fb.causal_filter(ev) for ev in ev_tr]
        r_te = [fb.causal_filter(ev) for ev in ev_te]

        gamma_tr = np.concatenate([r.state_posterior for r in r_tr], axis=0)
        gamma_te = np.concatenate([r.state_posterior for r in r_te], axis=0)

        dwell_tr = np.concatenate([r.expected_dwell / trainer.hsmm.d_max for r in r_tr], axis=0)
        dwell_te = np.concatenate([r.expected_dwell / trainer.hsmm.d_max for r in r_te], axis=0)

        f_struct_tr = np.concatenate([gamma_tr, dwell_tr], axis=1)
        f_struct_te = np.concatenate([gamma_te, dwell_te], axis=1)

        ev_tr_flat = np.concatenate(ev_tr)
        ev_te_flat = np.concatenate(ev_te)

        per_seed_data[s] = {
            "trainer": trainer,
            "fb": fb,
            "ev_tr": ev_tr,
            "ev_te": ev_te,
            "ev_tr_flat": ev_tr_flat,
            "ev_te_flat": ev_te_flat,
            "gamma_tr": gamma_tr,
            "gamma_te": gamma_te,
            "dwell_tr": dwell_tr,
            "dwell_te": dwell_te,
            "f_struct_tr": f_struct_tr,
            "f_struct_te": f_struct_te,
            "perplexity": float(s1_stats.perplexity),
        }
        print(f"Seed {s} ready in {time.time()-t0:.1f}s. Perplexity={s1_stats.perplexity:.2f}, active={len(np.unique(ev_te_flat))}/16")

    # =========================================================================
    # EXPERIMENT 1: EVENT ID vs RUL
    # =========================================================================
    print("\n" + "=" * 70)
    print("RUNNING EXPERIMENT 1: EVENT ID vs RUL READOUTS")
    print("=" * 70)

    # Candidate feature representations:
    # A. event ID one-hot (16-D)
    # B. event ID one-hot + running frequency (32-D)
    # C. HSMM gamma (5-D)
    # D. HSMM gamma + dwell (10-D)
    # E. full existing structural F_struct (10-D)

    readout_names = [
        "A_event_onehot",
        "B_event_onehot_plus_freq",
        "C_hsmm_gamma",
        "D_hsmm_gamma_dwell",
        "E_full_f_struct",
    ]

    exp1_results = {r: {"rmse_per_seed": [], "mae_per_seed": [], "per_engine": {u: {"rmse": [], "mae": []} for u in te_units}} for r in readout_names}

    for s in seeds:
        d = per_seed_data[s]
        # A: one-hot
        eye16 = np.eye(16)
        x_tr_a = eye16[d["ev_tr_flat"]]
        x_te_a = eye16[d["ev_te_flat"]]

        # B: one-hot + running frequency
        freq_tr = np.concatenate(compute_running_frequency(d["ev_tr"], 16), axis=0)
        freq_te = np.concatenate(compute_running_frequency(d["ev_te"], 16), axis=0)
        x_tr_b = np.concatenate([x_tr_a, freq_tr], axis=1)
        x_te_b = np.concatenate([x_te_a, freq_te], axis=1)

        # C: gamma
        x_tr_c = d["gamma_tr"]
        x_te_c = d["gamma_te"]

        # D: gamma + dwell
        x_tr_d = d["f_struct_tr"]
        x_te_d = d["f_struct_te"]

        # E: full f_struct
        x_tr_e = d["f_struct_tr"]
        x_te_e = d["f_struct_te"]

        feature_dict = {
            "A_event_onehot": (x_tr_a, x_te_a),
            "B_event_onehot_plus_freq": (x_tr_b, x_te_b),
            "C_hsmm_gamma": (x_tr_c, x_te_c),
            "D_hsmm_gamma_dwell": (x_tr_d, x_te_d),
            "E_full_f_struct": (x_tr_e, x_te_e),
        }

        for rname, (X_tr_r, X_te_r) in feature_dict.items():
            w = _ridge_fit(X_tr_r, ytr)
            preds = _ridge_predict(w, X_te_r)
            m = rul_metrics(preds, yte)
            exp1_results[rname]["rmse_per_seed"].append(m["RMSE"])
            exp1_results[rname]["mae_per_seed"].append(m["MAE"])

            # Per engine:
            idx_start = 0
            for u, s_unit in zip(te_units, te_seqs):
                L = len(s_unit.X)
                p_u = preds[idx_start:idx_start + L]
                mu = rul_metrics(p_u, s_unit.rul_labels)
                exp1_results[rname]["per_engine"][u]["rmse"].append(mu["RMSE"])
                exp1_results[rname]["per_engine"][u]["mae"].append(mu["MAE"])
                idx_start += L

    for rname in readout_names:
        rmses = exp1_results[rname]["rmse_per_seed"]
        maes = exp1_results[rname]["mae_per_seed"]
        exp1_results[rname]["rmse_mean"] = float(np.mean(rmses))
        exp1_results[rname]["rmse_std"] = float(np.std(rmses))
        exp1_results[rname]["mae_mean"] = float(np.mean(maes))
        exp1_results[rname]["mae_std"] = float(np.std(maes))
        print(f"  {rname:25s}: RMSE = {np.mean(rmses):.2f} ± {np.std(rmses):.2f} | MAE = {np.mean(maes):.2f} ± {np.std(maes):.2f}")
        for u in te_units:
            print(f"     Engine {u:8s}: RMSE = {np.mean(exp1_results[rname]['per_engine'][u]['rmse']):.2f} | MAE = {np.mean(exp1_results[rname]['per_engine'][u]['mae']):.2f}")

    audit_data["exp1_rul_readouts"] = exp1_results

    # =========================================================================
    # EXPERIMENT 2 & 3: OPERATING REGIME & DEGRADATION ASSOCIATIONS
    # =========================================================================
    print("\n" + "=" * 70)
    print("RUNNING EXPERIMENT 2 & 3: EVENT ID vs OPERATING REGIMES & DEGRADATION")
    print("=" * 70)

    # Analyze for each seed and pooled
    # Focus on representative Seed 0 (as primary reference) and pooled across seeds
    # Let's inspect Seed 0 in detail:
    s0_events = per_seed_data[0]["ev_te_flat"]  # test set events
    s0_tr_events = per_seed_data[0]["ev_tr_flat"]  # train set events

    exp2_results = {}
    exp3_results = {}

    for s in seeds:
        ev_te = per_seed_data[s]["ev_te_flat"]
        ev_tr = per_seed_data[s]["ev_tr_flat"]
        
        # Test unit evaluations
        regimes_seed = {}
        degrad_seed = {}
        for k in range(16):
            idx_te = np.where(ev_te == k)[0]
            idx_tr = np.where(ev_tr == k)[0]
            cnt_te = len(idx_te)
            cnt_tr = len(idx_tr)
            
            if cnt_te > 0:
                # Flight conditions: standardized [alt, Mach, TRA, T2]
                fc_std_te = Xte_std_mean[idx_te, :4]
                fc_raw_te = raw_te_mean[idx_te, :4]
                rul_te = yte[idx_te]

                regimes_seed[k] = {
                    "count_test": cnt_te,
                    "count_train": cnt_tr,
                    "std_mean_alt": float(np.mean(fc_std_te[:, 0])),
                    "std_mean_Mach": float(np.mean(fc_std_te[:, 1])),
                    "std_mean_TRA": float(np.mean(fc_std_te[:, 2])),
                    "std_mean_T2": float(np.mean(fc_std_te[:, 3])),
                    "std_std_alt": float(np.std(fc_std_te[:, 0])),
                    "std_std_Mach": float(np.std(fc_std_te[:, 1])),
                    "std_std_TRA": float(np.std(fc_std_te[:, 2])),
                    "std_std_T2": float(np.std(fc_std_te[:, 3])),
                    "raw_mean_alt_ft": float(np.mean(fc_raw_te[:, 0])),
                    "raw_mean_Mach": float(np.mean(fc_raw_te[:, 1])),
                    "raw_mean_TRA_pct": float(np.mean(fc_raw_te[:, 2])),
                    "raw_mean_T2_R": float(np.mean(fc_raw_te[:, 3])),
                }

                degrad_seed[k] = {
                    "count": cnt_te,
                    "mean_rul": float(np.mean(rul_te)),
                    "median_rul": float(np.median(rul_te)),
                    "std_rul": float(np.std(rul_te)),
                    "min_rul": float(np.min(rul_te)),
                    "max_rul": float(np.max(rul_te)),
                    "p25_rul": float(np.percentile(rul_te, 25)),
                    "p75_rul": float(np.percentile(rul_te, 75)),
                }

        exp2_results[f"seed_{s}"] = regimes_seed
        exp3_results[f"seed_{s}"] = degrad_seed

    # Print summary for Seed 0
    print("\n--- Seed 0 Event Associations (Test Set, N=39) ---")
    print(f"{'Token':<6} {'N':<4} {'Alt(std)':<10} {'Mach(std)':<10} {'TRA(std)':<10} {'T2(std)':<10} {'Mean RUL':<10} {'Med RUL':<10} {'Std RUL':<10} {'RUL Range':<15}")
    for k, v in sorted(exp2_results["seed_0"].items()):
        d_k = exp3_results["seed_0"][k]
        print(f"E{k:<5} {v['count_test']:<4} {v['std_mean_alt']:>9.2f} {v['std_mean_Mach']:>9.2f} {v['std_mean_TRA']:>9.2f} {v['std_mean_T2']:>9.2f} {d_k['mean_rul']:>9.1f} {d_k['median_rul']:>9.1f} {d_k['std_rul']:>9.1f} [{d_k['min_rul']:.0f} - {d_k['max_rul']:.0f}]")

    audit_data["exp2_operating_regimes"] = exp2_results
    audit_data["exp3_degradation"] = exp3_results

    # =========================================================================
    # EXPERIMENT 4: TEMPORAL PERSISTENCE
    # =========================================================================
    print("\n" + "=" * 70)
    print("RUNNING EXPERIMENT 4: TEMPORAL PERSISTENCE")
    print("=" * 70)

    # Compare 12/4 model across seeds with 16/0 control
    # 12/4 run lengths:
    run_stats_12_4 = []
    self_trans_12_4 = []
    trans_ent_12_4 = []

    for s in seeds:
        ev_te = per_seed_data[s]["ev_te"]
        rl = compute_run_lengths(ev_te)
        t_stats = compute_transition_stats(ev_te, 16)
        run_stats_12_4.append(rl)
        self_trans_12_4.append(t_stats["self_transition_rate"])
        trans_ent_12_4.append(t_stats["entropy_bits"])
        print(f"  Seed {s}: Mean Run={rl['mean']:.2f}, Med={rl['median']:.1f}, Self-Trans Rate={t_stats['self_transition_rate']:.3f}, Trans Entropy={t_stats['entropy_bits']:.3f} bits")

    # 16/0 Control metrics (from subspace_experiment_results.json):
    with open("subspace_experiment_results.json") as f:
        sub_results = json.load(f)
    a1 = sub_results["Arm 1 (Control 16/0)"]
    a1_seqs = a1["event_vocabulary"]["representative_sequences"]
    # Reconstruct 16/0 test sequences for each seed:
    # Seed 0: DS01_u2_seed0 (19), DS01_u9_seed0 (20)
    # Seed 1: DS01_u2_seed1 (19)
    # Seed 2: DS01_u2_seed2 (19)
    # Mean run length for 16/0 was 13.8125
    mean_rl_16_0 = a1["event_vocabulary"]["mean_run_length"]
    active_16_0 = a1["vq_diagnostics"]["active_codes_mean"]
    perp_16_0 = a1["vq_diagnostics"]["perplexity_mean"]

    exp4_results = {
        "candidate_12_4": {
            "mean_run_length": float(np.mean([r["mean"] for r in run_stats_12_4])),
            "mean_run_std": float(np.std([r["mean"] for r in run_stats_12_4])),
            "median_run_length": float(np.median([r["median"] for r in run_stats_12_4])),
            "self_transition_rate_mean": float(np.mean(self_trans_12_4)),
            "self_transition_rate_std": float(np.std(self_trans_12_4)),
            "transition_entropy_bits_mean": float(np.mean(trans_ent_12_4)),
            "transition_entropy_bits_std": float(np.std(trans_ent_12_4)),
            "per_seed_runs": [r["distribution"] for r in run_stats_12_4],
        },
        "control_16_0": {
            "mean_run_length": mean_rl_16_0,
            "active_codes_mean": active_16_0,
            "perplexity_mean": perp_16_0,
            "collapse_behavior": "Static collapse to single code (39 steps) in seeds 1 & 2; slow toggle in seed 0",
        },
    }
    print(f"\nPersistence Comparison:")
    print(f"  12/4 Subspace: Mean run length = {exp4_results['candidate_12_4']['mean_run_length']:.2f} ± {exp4_results['candidate_12_4']['mean_run_std']:.2f}")
    print(f"  16/0 Control:  Mean run length = {mean_rl_16_0:.2f} (collapsed single-code dominance)")
    audit_data["exp4_persistence"] = exp4_results

    # =========================================================================
    # EXPERIMENT 5: TRANSITION STRUCTURE
    # =========================================================================
    print("\n" + "=" * 70)
    print("RUNNING EXPERIMENT 5: TRANSITION STRUCTURE (16x16 MATRIX)")
    print("=" * 70)

    exp5_results = {}
    for s in seeds:
        ev_te = per_seed_data[s]["ev_te"]
        t_stats = compute_transition_stats(ev_te, 16)
        exp5_results[f"seed_{s}"] = t_stats
        print(f"  Seed {s}: Total trans={t_stats['total_transitions']}, Observed={t_stats['n_observed_transitions']}, Self-trans={t_stats['self_transition_rate']:.3f}, H(E_t|E_t-1)={t_stats['entropy_bits']:.3f} bits")
        print(f"    Top 3 transitions: {t_stats['top_transitions'][:3]}")

    audit_data["exp5_transitions"] = exp5_results

    # =========================================================================
    # EXPERIMENT 6: SHUFFLED EVENT BASELINE
    # =========================================================================
    print("\n" + "=" * 70)
    print("RUNNING EXPERIMENT 6: SHUFFLED EVENT BASELINE CONTROL")
    print("=" * 70)

    # Shuffled control: permute event tokens within each test sequence
    # Evaluate over 50 permutations per seed:
    # 1. Transition entropy & self-transition rate of shuffled sequences
    # 2. Event one-hot Ridge RUL test RMSE
    # 3. HSMM causal filtering -> Ridge RUL test RMSE (using frozen HSMM & Ridge weights)

    n_shuffles = 50
    exp6_results = {}

    for s in seeds:
        d = per_seed_data[s]
        fb = d["fb"]
        trainer = d["trainer"]
        ev_te_orig = d["ev_te"]

        # Train one-hot Ridge model:
        x_tr_onehot = np.eye(16)[d["ev_tr_flat"]]
        w_onehot = _ridge_fit(x_tr_onehot, ytr)
        m_orig_onehot = rul_metrics(_ridge_predict(w_onehot, np.eye(16)[d["ev_te_flat"]]), yte)

        # Train EGPM F_struct Ridge model:
        w_egpm = _ridge_fit(d["f_struct_tr"], ytr)
        m_orig_egpm = rul_metrics(_ridge_predict(w_egpm, d["f_struct_te"]), yte)

        shuf_trans_entropies = []
        shuf_self_trans = []
        shuf_onehot_rmses = []
        shuf_onehot_maes = []
        shuf_egpm_rmses = []
        shuf_egpm_maes = []

        rng = np.random.default_rng(1000 + s)

        for rep in range(n_shuffles):
            # Permute each test sequence independently
            shuf_ev_te = [rng.permutation(seq) for seq in ev_te_orig]
            shuf_flat = np.concatenate(shuf_ev_te)

            # 1. Transition stats
            t_shuf = compute_transition_stats(shuf_ev_te, 16)
            shuf_trans_entropies.append(t_shuf["entropy_bits"])
            shuf_self_trans.append(t_shuf["self_transition_rate"])

            # 2. One-hot RUL
            p_shuf_onehot = _ridge_predict(w_onehot, np.eye(16)[shuf_flat])
            m_s_onehot = rul_metrics(p_shuf_onehot, yte)
            shuf_onehot_rmses.append(m_s_onehot["RMSE"])
            shuf_onehot_maes.append(m_s_onehot["MAE"])

            # 3. HSMM causal filtering on shuffled sequence
            shuf_r_te = [fb.causal_filter(seq) for seq in shuf_ev_te]
            shuf_gamma = np.concatenate([r.state_posterior for r in shuf_r_te], axis=0)
            shuf_dwell = np.concatenate([r.expected_dwell / trainer.hsmm.d_max for r in shuf_r_te], axis=0)
            shuf_f_struct = np.concatenate([shuf_gamma, shuf_dwell], axis=1)

            p_shuf_egpm = _ridge_predict(w_egpm, shuf_f_struct)
            m_s_egpm = rul_metrics(p_shuf_egpm, yte)
            shuf_egpm_rmses.append(m_s_egpm["RMSE"])
            shuf_egpm_maes.append(m_s_egpm["MAE"])

        exp6_results[f"seed_{s}"] = {
            "orig_onehot_rmse": float(m_orig_onehot["RMSE"]),
            "orig_onehot_mae": float(m_orig_onehot["MAE"]),
            "orig_egpm_rmse": float(m_orig_egpm["RMSE"]),
            "orig_egpm_mae": float(m_orig_egpm["MAE"]),
            "shuf_onehot_rmse_mean": float(np.mean(shuf_onehot_rmses)),
            "shuf_onehot_rmse_std": float(np.std(shuf_onehot_rmses)),
            "shuf_onehot_mae_mean": float(np.mean(shuf_onehot_maes)),
            "shuf_egpm_rmse_mean": float(np.mean(shuf_egpm_rmses)),
            "shuf_egpm_rmse_std": float(np.std(shuf_egpm_rmses)),
            "shuf_egpm_mae_mean": float(np.mean(shuf_egpm_maes)),
            "orig_trans_entropy_bits": float(compute_transition_stats(ev_te_orig, 16)["entropy_bits"]),
            "shuf_trans_entropy_bits_mean": float(np.mean(shuf_trans_entropies)),
            "orig_self_trans_rate": float(compute_transition_stats(ev_te_orig, 16)["self_transition_rate"]),
            "shuf_self_trans_rate_mean": float(np.mean(shuf_self_trans)),
        }
        print(f"  Seed {s}:")
        print(f"    One-hot RMSE: True = {m_orig_onehot['RMSE']:.2f} -> Shuffled = {np.mean(shuf_onehot_rmses):.2f} ± {np.std(shuf_onehot_rmses):.2f} (Δ = {np.mean(shuf_onehot_rmses) - m_orig_onehot['RMSE']:+.2f})")
        print(f"    EGPM RMSE:    True = {m_orig_egpm['RMSE']:.2f} -> Shuffled = {np.mean(shuf_egpm_rmses):.2f} ± {np.std(shuf_egpm_rmses):.2f} (Δ = {np.mean(shuf_egpm_rmses) - m_orig_egpm['RMSE']:+.2f})")
        print(f"    Trans Entropy: True = {exp6_results[f'seed_{s}']['orig_trans_entropy_bits']:.3f} -> Shuffled = {np.mean(shuf_trans_entropies):.3f} bits")
        print(f"    Self-Trans:    True = {exp6_results[f'seed_{s}']['orig_self_trans_rate']:.3f} -> Shuffled = {np.mean(shuf_self_trans):.3f}")

    audit_data["exp6_shuffled_baseline"] = exp6_results

    # =========================================================================
    # EXPERIMENT 7: CODEBOOK STABILITY
    # =========================================================================
    print("\n" + "=" * 70)
    print("RUNNING EXPERIMENT 7: CODEBOOK STABILITY")
    print("=" * 70)

    # Hungarian matched usage overlap and Jaccard token stability
    ev_runs_per_seed = [per_seed_data[s]["ev_te"] for s in seeds]
    active_sets = [set(per_seed_data[s]["ev_te_flat"]) for s in seeds]

    # Hungarian usage overlap
    stab_01 = event_stability(ev_runs_per_seed[0], ev_runs_per_seed[1], n_codes=16)["matched_usage_overlap"]
    stab_12 = event_stability(ev_runs_per_seed[1], ev_runs_per_seed[2], n_codes=16)["matched_usage_overlap"]
    stab_02 = event_stability(ev_runs_per_seed[0], ev_runs_per_seed[2], n_codes=16)["matched_usage_overlap"]
    mean_hungarian = float(np.mean([stab_01, stab_12, stab_02]))

    # Jaccard overlap
    jaccard_01 = len(active_sets[0] & active_sets[1]) / len(active_sets[0] | active_sets[1])
    jaccard_12 = len(active_sets[1] & active_sets[2]) / len(active_sets[1] | active_sets[2])
    jaccard_02 = len(active_sets[0] & active_sets[2]) / len(active_sets[0] | active_sets[2])
    mean_jaccard = float(np.mean([jaccard_01, jaccard_12, jaccard_02]))

    exp7_results = {
        "hungarian_usage_overlap_mean": mean_hungarian,
        "hungarian_usage_overlap_pairs": {"seed0_seed1": stab_01, "seed1_seed2": stab_12, "seed0_seed2": stab_02},
        "jaccard_token_stability_mean": mean_jaccard,
        "jaccard_token_stability_pairs": {"seed0_seed1": jaccard_01, "seed1_seed2": jaccard_12, "seed0_seed2": jaccard_02},
        "active_codes_per_seed": [len(s_set) for s_set in active_sets],
        "active_codes_mean": float(np.mean([len(s_set) for s_set in active_sets])),
        "active_codes_std": float(np.std([len(s_set) for s_set in active_sets])),
        "code_usage_distributions": [
            np.bincount(per_seed_data[s]["ev_te_flat"], minlength=16).tolist() for s in seeds
        ],
    }
    print(f"Hungarian Matched Usage Overlap: {mean_hungarian:.3f} (pairs: {stab_01:.3f}, {stab_12:.3f}, {stab_02:.3f})")
    print(f"Jaccard Token Stability:         {mean_jaccard:.3f} (pairs: {jaccard_01:.3f}, {jaccard_12:.3f}, {jaccard_02:.3f})")
    print(f"Active codes per seed:           {exp7_results['active_codes_per_seed']}")

    audit_data["exp7_stability"] = exp7_results

    # =========================================================================
    # EXPERIMENT 8: SENSOR ASSOCIATIONS
    # =========================================================================
    print("\n" + "=" * 70)
    print("RUNNING EXPERIMENT 8: EVENT-CONDITIONED SENSOR ASSOCIATIONS")
    print("=" * 70)

    # For each active event ID, compute mean standardized sensor vector across all 18 channels
    # Report major deviations from global mean (std deviations > 0.5)
    exp8_results = {}

    for s in seeds:
        ev_te = per_seed_data[s]["ev_te_flat"]
        s_assoc = {}
        for k in range(16):
            idx = np.where(ev_te == k)[0]
            if len(idx) == 0:
                continue
            sensor_vec = Xte_std_mean[idx]  # [N_k, 18]
            mean_vec = np.mean(sensor_vec, axis=0)
            std_vec = np.std(sensor_vec, axis=0)

            # Major deviations from global mean (threshold |mean| > 0.5)
            deviations = []
            for ch_idx, ch_name in enumerate(CHANNEL_NAMES):
                dev = float(mean_vec[ch_idx])
                if abs(dev) > 0.4:
                    deviations.append({
                        "channel": ch_name,
                        "channel_idx": ch_idx,
                        "deviation_std": dev,
                        "is_operating_cond": ch_idx < 4,
                    })
            deviations.sort(key=lambda x: abs(x["deviation_std"]), reverse=True)

            s_assoc[int(k)] = {
                "count": len(idx),
                "mean_standardized_vector": [float(x) for x in mean_vec],
                "std_vector": [float(x) for x in std_vec],
                "major_deviations": deviations,
            }
        exp8_results[f"seed_{s}"] = s_assoc

    # Print summary for Seed 0
    print("\n--- Seed 0 Event-Conditioned Sensor Associations ---")
    for k, v in sorted(exp8_results["seed_0"].items()):
        dev_strs = [f"{d['channel']}: {d['deviation_std']:+.2f}σ" for d in v["major_deviations"][:4]]
        print(f"Event {k:2d} (N={v['count']:2d}): {', '.join(dev_strs) if dev_strs else 'No deviations > 0.4σ'}")

    audit_data["exp8_sensor_associations"] = exp8_results

    # Save to JSON
    out_file = "event_information_audit_results.json"
    with open(out_file, "w") as f:
        json.dump(audit_data, f, indent=2)
    print(f"\nAll audit results successfully saved to {out_file}!")


if __name__ == "__main__":
    run_audit()
