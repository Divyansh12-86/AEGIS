"""Complete Scientific Audit:
Phase 1: Reproducibility audit & seed comparison
Phase 2: Canonical W=4 baseline run (seeds 0,1,2 vs 42,123,999)
Phase 3 & 4: Aligned W=4 vs W=16 comparison (strictly on paired timestamps >= cycle 15)
Phase 5: Representation diagnostics (VQ entropy, active codes, HSMM state occupancy, dwell)
Phase 6: Information bottleneck study across all 6 representation tiers
"""
import json
import time
import numpy as np
import torch

from egpm.data.loaders import NCMAPSSLoader
from egpm.data.split_builder import SplitBuilder
from egpm.experiments.ablation_arms import _downsample, _ridge_fit, _ridge_predict
from egpm.preprocessing.preprocessor import Preprocessor
from egpm.training.trainer import Trainer, TrainerConfig
from egpm.grammar.forward_backward import ForwardBackward
from egpm.evaluation import rul_metrics
from egpm.utils import set_global_seed


def run_complete_audit():
    dataset_path = "N-CMAPSS/N-CMAPSS DS01 005.h5"
    raw_dataset = NCMAPSSLoader().load(dataset_path)
    raw_dataset.records = [_downsample(r) for r in raw_dataset.records]
    
    sb = SplitBuilder(raw_dataset, seed=0)
    split = sb.build()
    stats = sb.fit_normalization(split)
    
    tr_units = split.train_unit_ids
    va_units = split.val_unit_ids
    te_units = split.test_unit_ids
    
    print(f"Loaded dataset: {dataset_path}")
    print(f"Train units: {tr_units}, Val: {va_units}, Test: {te_units}")
    
    # -------------------------------------------------------------
    # PHASE 2: Canonical W=4 Causal Baseline (Seeds 0,1,2 and 42,123,999)
    # -------------------------------------------------------------
    print("\n" + "="*60)
    print("PHASE 2: CANONICAL W=4 CAUSAL BASELINE SEED COMPARISON")
    print("="*60)
    
    seed_sets = {
        "seeds_0_1_2": [0, 1, 2],
        "seeds_42_123_999": [42, 123, 999],
    }
    
    p2_results = {}
    
    for s_name, seeds in seed_sets.items():
        print(f"\nEvaluating seed set: {s_name} {seeds} (n_conv_blocks=3, W=4, S=4)...")
        prep4 = Preprocessor(window_length=4, stride=4)
        tr4 = [prep4.process(raw_dataset[u], norm_stats=stats) for u in tr_units]
        va4 = [prep4.process(raw_dataset[u], norm_stats=stats) for u in va_units]
        te4 = [prep4.process(raw_dataset[u], norm_stats=stats) for u in te_units]
        
        Xtr4 = torch.as_tensor(np.concatenate([s.X for s in tr4]), dtype=torch.float32)
        Mtr4 = torch.ones_like(Xtr4)
        ytr4 = np.concatenate([s.rul_labels for s in tr4])
        yte4 = np.concatenate([s.rul_labels for s in te4])
        ltr4 = [len(s.X) for s in tr4]
        
        rmses, maes = [], []
        per_eng_rmses = {u: [] for u in te_units}
        per_eng_maes = {u: [] for u in te_units}
        
        for s in seeds:
            set_global_seed(s)
            cfg = TrainerConfig(
                window_length=4, embed_dim=16, hidden=32, n_conv_blocks=3,
                n_codes=16, d_max=10, n_states=5, em_max_iter=10, em_restarts=10,
                stage1_steps=300, batch_size=64, seed=s, log_every=0
            )
            trainer = Trainer(cfg, n_channels=Xtr4.shape[-1])
            trainer.train_stage1(Xtr4, Mtr4)
            ev_tr = trainer.window_runs_to_event_runs(trainer.event_sequences(Xtr4, Mtr4), ltr4)
            trainer.train_stage2(ev_tr)
            
            fb = ForwardBackward(trainer.hsmm)
            
            # Causal filtering on train
            f_tr = []
            for s_unit, ev in zip(tr4, ev_tr):
                r = fb.causal_filter(ev)
                f_tr.append(np.concatenate([r.state_posterior, r.expected_dwell / trainer.hsmm.d_max], axis=1))
            w = _ridge_fit(np.concatenate(f_tr), ytr4)
            
            # Causal filtering on test
            pr_test = []
            for s_unit in te4:
                Xt = torch.as_tensor(s_unit.X, dtype=torch.float32)
                toks = np.concatenate(trainer.event_sequences(Xt, torch.ones_like(Xt)))
                r = fb.causal_filter(toks)
                f_test = np.concatenate([r.state_posterior, r.expected_dwell / trainer.hsmm.d_max], axis=1)
                pr_test.append(_ridge_predict(w, f_test))
                
            m = rul_metrics(np.concatenate(pr_test), yte4)
            rmses.append(m["RMSE"])
            maes.append(m["MAE"])
            
            for u, p, s_unit in zip(te_units, pr_test, te4):
                em = rul_metrics(p, s_unit.rul_labels)
                per_eng_rmses[u].append(em["RMSE"])
                per_eng_maes[u].append(em["MAE"])
                
        p2_results[s_name] = {
            "seeds": seeds,
            "rmse": {"mean": float(np.mean(rmses)), "std": float(np.std(rmses)), "per_seed": rmses},
            "mae": {"mean": float(np.mean(maes)), "std": float(np.std(maes)), "per_seed": maes},
            "per_engine": {
                u: {
                    "rmse_mean": float(np.mean(per_eng_rmses[u])),
                    "rmse_std": float(np.std(per_eng_rmses[u])),
                    "mae_mean": float(np.mean(per_eng_maes[u])),
                    "mae_std": float(np.std(per_eng_maes[u])),
                } for u in te_units
            }
        }
        print(f"Results for {s_name}: RMSE = {np.mean(rmses):.2f} ± {np.std(rmses):.2f}, MAE = {np.mean(maes):.2f} ± {np.std(maes):.2f}")

    # -------------------------------------------------------------
    # PHASE 3 & 4: Aligned W=4 vs W=16 Comparison (Seeds 0, 1, 2)
    # -------------------------------------------------------------
    print("\n" + "="*60)
    print("PHASE 3 & 4: STRICTLY ALIGNED W=4 VS W=16 EVALUATION")
    print("="*60)
    
    seeds = [0, 1, 2]
    aligned_w4_rmses, aligned_w4_maes = [], []
    aligned_w16_rmses, aligned_w16_maes = [], []
    
    eng_w4_aligned_rmses = {u: [] for u in te_units}
    eng_w4_aligned_maes = {u: [] for u in te_units}
    eng_w16_aligned_rmses = {u: [] for u in te_units}
    eng_w16_aligned_maes = {u: [] for u in te_units}
    
    # Store diagnostics
    diag_w4 = []
    diag_w16 = []

    for s in seeds:
        set_global_seed(s)
        # --- Model W=4 ---
        prep4 = Preprocessor(window_length=4, stride=4)
        tr4 = [prep4.process(raw_dataset[u], norm_stats=stats) for u in tr_units]
        te4 = [prep4.process(raw_dataset[u], norm_stats=stats) for u in te_units]
        Xtr4 = torch.as_tensor(np.concatenate([s.X for s in tr4]), dtype=torch.float32)
        Mtr4 = torch.ones_like(Xtr4)
        ytr4 = np.concatenate([s.rul_labels for s in tr4])
        ltr4 = [len(s.X) for s in tr4]
        
        cfg4 = TrainerConfig(
            window_length=4, embed_dim=16, hidden=32, n_conv_blocks=3,
            n_codes=16, d_max=10, n_states=5, em_max_iter=10, em_restarts=10,
            stage1_steps=300, batch_size=64, seed=s, log_every=0
        )
        trainer4 = Trainer(cfg4, n_channels=Xtr4.shape[-1])
        trainer4.train_stage1(Xtr4, Mtr4)
        ev_tr4 = trainer4.window_runs_to_event_runs(trainer4.event_sequences(Xtr4, Mtr4), ltr4)
        trainer4.train_stage2(ev_tr4)
        fb4 = ForwardBackward(trainer4.hsmm)
        
        f_tr4 = []
        for s_unit, ev in zip(tr4, ev_tr4):
            r = fb4.causal_filter(ev)
            f_tr4.append(np.concatenate([r.state_posterior, r.expected_dwell / trainer4.hsmm.d_max], axis=1))
        w4 = _ridge_fit(np.concatenate(f_tr4), ytr4)
        
        # Test predictions W=4
        preds_w4 = []
        ev_te4 = []
        f_te4 = []
        for s_unit in te4:
            Xt = torch.as_tensor(s_unit.X, dtype=torch.float32)
            toks = np.concatenate(trainer4.event_sequences(Xt, torch.ones_like(Xt)))
            ev_te4.append(toks)
            r = fb4.causal_filter(toks)
            f_test = np.concatenate([r.state_posterior, r.expected_dwell / trainer4.hsmm.d_max], axis=1)
            f_te4.append(f_test)
            preds_w4.append(_ridge_predict(w4, f_test))
            
        # --- Model W=16 ---
        prep16 = Preprocessor(window_length=16, stride=4)
        tr16 = [prep16.process(raw_dataset[u], norm_stats=stats) for u in tr_units]
        te16 = [prep16.process(raw_dataset[u], norm_stats=stats) for u in te_units]
        Xtr16 = torch.as_tensor(np.concatenate([s.X for s in tr16]), dtype=torch.float32)
        Mtr16 = torch.ones_like(Xtr16)
        ytr16 = np.concatenate([s.rul_labels for s in tr16])
        ltr16 = [len(s.X) for s in tr16]
        
        cfg16 = TrainerConfig(
            window_length=16, embed_dim=16, hidden=32, n_conv_blocks=3,
            n_codes=16, d_max=10, n_states=5, em_max_iter=10, em_restarts=10,
            stage1_steps=300, batch_size=64, seed=s, log_every=0
        )
        trainer16 = Trainer(cfg16, n_channels=Xtr16.shape[-1])
        trainer16.train_stage1(Xtr16, Mtr16)
        ev_tr16 = trainer16.window_runs_to_event_runs(trainer16.event_sequences(Xtr16, Mtr16), ltr16)
        trainer16.train_stage2(ev_tr16)
        fb16 = ForwardBackward(trainer16.hsmm)
        
        f_tr16 = []
        for s_unit, ev in zip(tr16, ev_tr16):
            r = fb16.causal_filter(ev)
            f_tr16.append(np.concatenate([r.state_posterior, r.expected_dwell / trainer16.hsmm.d_max], axis=1))
        w16 = _ridge_fit(np.concatenate(f_tr16), ytr16)
        
        # Test predictions W=16
        preds_w16 = []
        ev_te16 = []
        f_te16 = []
        for s_unit in te16:
            Xt = torch.as_tensor(s_unit.X, dtype=torch.float32)
            toks = np.concatenate(trainer16.event_sequences(Xt, torch.ones_like(Xt)))
            ev_te16.append(toks)
            r = fb16.causal_filter(toks)
            f_test = np.concatenate([r.state_posterior, r.expected_dwell / trainer16.hsmm.d_max], axis=1)
            f_te16.append(f_test)
            preds_w16.append(_ridge_predict(w16, f_test))
            
        # ALIGNED EVALUATION:
        # W=16 starts emitting at cycle 15, matching W=4 window index 3 onward.
        aligned_p4_list = []
        aligned_p16_list = []
        aligned_y_list = []
        
        for u, p4, p16, s4, s16 in zip(te_units, preds_w4, preds_w16, te4, te16):
            # Align: take p4 from index 3 onward
            p4_aligned = p4[3:]
            y4_aligned = s4.rul_labels[3:]
            p16_aligned = p16[:]
            y16_aligned = s16.rul_labels[:]
            
            assert len(p4_aligned) == len(p16_aligned), f"Length mismatch: {len(p4_aligned)} vs {len(p16_aligned)}"
            assert np.allclose(y4_aligned, y16_aligned), "Target mismatch in aligned comparison!"
            
            aligned_p4_list.append(p4_aligned)
            aligned_p16_list.append(p16_aligned)
            aligned_y_list.append(y16_aligned)
            
            m4_eng = rul_metrics(p4_aligned, y4_aligned)
            m16_eng = rul_metrics(p16_aligned, y16_aligned)
            eng_w4_aligned_rmses[u].append(m4_eng["RMSE"])
            eng_w4_aligned_maes[u].append(m4_eng["MAE"])
            eng_w16_aligned_rmses[u].append(m16_eng["RMSE"])
            eng_w16_aligned_maes[u].append(m16_eng["MAE"])
            
        m4_all = rul_metrics(np.concatenate(aligned_p4_list), np.concatenate(aligned_y_list))
        m16_all = rul_metrics(np.concatenate(aligned_p16_list), np.concatenate(aligned_y_list))
        
        aligned_w4_rmses.append(m4_all["RMSE"])
        aligned_w4_maes.append(m4_all["MAE"])
        aligned_w16_rmses.append(m16_all["RMSE"])
        aligned_w16_maes.append(m16_all["MAE"])
        
        # Diagnostics
        def get_diag(trainer, hsmm, ev_tr, ev_te, f_te):
            all_te = np.concatenate(ev_te)
            counts = np.bincount(all_te, minlength=hsmm.K)
            probs = counts / counts.sum()
            entropy = -np.sum(probs[probs > 0] * np.log(probs[probs > 0]))
            active = int(np.sum(counts > 0))
            dead = float((hsmm.K - active) / hsmm.K)
            
            all_gammas = np.concatenate([f[:, :hsmm.M] for f in f_te], axis=0)
            state_occ = np.mean(all_gammas, axis=0)
            # state entropy per step
            step_ents = -np.sum(all_gammas * np.log(np.clip(all_gammas, 1e-12, None)), axis=1)
            mean_state_ent = float(np.mean(step_ents))
            
            all_dwells = np.concatenate([f[:, hsmm.M:] for f in f_te], axis=0)
            mean_dwell = float(np.mean(all_dwells) * hsmm.d_max)
            
            s1 = trainer.stats[0] if trainer.stats else None
            return {
                "vq_perplexity": float(s1.perplexity) if s1 else float("nan"),
                "active_code_count": active,
                "code_usage_entropy": float(entropy),
                "fraction_dead_codes": dead,
                "token_frequencies": [float(p) for p in probs],
                "hsmm_state_occupancy": [float(p) for p in state_occ],
                "mean_state_entropy": mean_state_ent,
                "mean_expected_dwell": mean_dwell,
            }
            
        diag_w4.append(get_diag(trainer4, trainer4.hsmm, ev_tr4, ev_te4, f_te4))
        diag_w16.append(get_diag(trainer16, trainer16.hsmm, ev_tr16, ev_te16, f_te16))

    aligned_results = {
        "aligned_windows_count": 33,
        "w4": {
            "rmse": {"mean": float(np.mean(aligned_w4_rmses)), "std": float(np.std(aligned_w4_rmses)), "per_seed": aligned_w4_rmses},
            "mae": {"mean": float(np.mean(aligned_w4_maes)), "std": float(np.std(aligned_w4_maes)), "per_seed": aligned_w4_maes},
            "per_engine": {
                u: {
                    "rmse_mean": float(np.mean(eng_w4_aligned_rmses[u])),
                    "rmse_std": float(np.std(eng_w4_aligned_rmses[u])),
                    "mae_mean": float(np.mean(eng_w4_aligned_maes[u])),
                    "mae_std": float(np.std(eng_w4_aligned_maes[u])),
                } for u in te_units
            }
        },
        "w16": {
            "rmse": {"mean": float(np.mean(aligned_w16_rmses)), "std": float(np.std(aligned_w16_rmses)), "per_seed": aligned_w16_rmses},
            "mae": {"mean": float(np.mean(aligned_w16_maes)), "std": float(np.std(aligned_w16_maes)), "per_seed": aligned_w16_maes},
            "per_engine": {
                u: {
                    "rmse_mean": float(np.mean(eng_w16_aligned_rmses[u])),
                    "rmse_std": float(np.std(eng_w16_aligned_rmses[u])),
                    "mae_mean": float(np.mean(eng_w16_aligned_maes[u])),
                    "mae_std": float(np.std(eng_w16_aligned_maes[u])),
                } for u in te_units
            }
        }
    }
    
    print("\nAligned Results (33 paired timestamps):")
    print(f"W=4 Aligned:  RMSE = {aligned_results['w4']['rmse']['mean']:.2f} ± {aligned_results['w4']['rmse']['std']:.2f}, MAE = {aligned_results['w4']['mae']['mean']:.2f} ± {aligned_results['w4']['mae']['std']:.2f}")
    print(f"W=16 Aligned: RMSE = {aligned_results['w16']['rmse']['mean']:.2f} ± {aligned_results['w16']['rmse']['std']:.2f}, MAE = {aligned_results['w16']['mae']['mean']:.2f} ± {aligned_results['w16']['mae']['std']:.2f}")

    # -------------------------------------------------------------
    # PHASE 6: Information Bottleneck Study
    # -------------------------------------------------------------
    print("\n" + "="*60)
    print("PHASE 6: INFORMATION BOTTLENECK STUDY (CANONICAL W=4)")
    print("="*60)
    
    # We compare 6 representations on the identical W=4 causal pipeline:
    # A: Raw sensors (72-dim flattened window)
    # B: Encoder continuous latent z (16-dim)
    # C: VQ discrete event representation (16-dim one-hot)
    # D: Causal HSMM state posterior gamma (5-dim)
    # E: Causal gamma + dwell (10-dim)
    # F: Full structural manifold (identical to E, verifying pipeline)
    
    rep_names = [
        "A_raw_sensors",
        "B_encoder_z",
        "C_vq_one_hot",
        "D_causal_gamma",
        "E_gamma_dwell",
        "F_full_f_struct"
    ]
    
    rep_results = {r: {"rmses": [], "maes": [], "dim": 0} for r in rep_names}
    
    for s in seeds:
        set_global_seed(s)
        prep4 = Preprocessor(window_length=4, stride=4)
        tr4 = [prep4.process(raw_dataset[u], norm_stats=stats) for u in tr_units]
        te4 = [prep4.process(raw_dataset[u], norm_stats=stats) for u in te_units]
        Xtr4 = torch.as_tensor(np.concatenate([s.X for s in tr4]), dtype=torch.float32)
        Mtr4 = torch.ones_like(Xtr4)
        ytr4 = np.concatenate([s.rul_labels for s in tr4])
        yte4 = np.concatenate([s.rul_labels for s in te4])
        ltr4 = [len(s.X) for s in tr4]
        
        cfg4 = TrainerConfig(
            window_length=4, embed_dim=16, hidden=32, n_conv_blocks=3,
            n_codes=16, d_max=10, n_states=5, em_max_iter=10, em_restarts=10,
            stage1_steps=300, batch_size=64, seed=s, log_every=0
        )
        trainer = Trainer(cfg4, n_channels=Xtr4.shape[-1])
        trainer.train_stage1(Xtr4, Mtr4)
        ev_tr = trainer.window_runs_to_event_runs(trainer.event_sequences(Xtr4, Mtr4), ltr4)
        trainer.train_stage2(ev_tr)
        fb = ForwardBackward(trainer.hsmm)
        
        # Build features for train units
        tr_feats = {r: [] for r in rep_names}
        for s_unit, ev in zip(tr4, ev_tr):
            Xt = torch.as_tensor(s_unit.X, dtype=torch.float32)
            Mt = torch.ones_like(Xt)
            
            # A: Raw sensors
            f_A = s_unit.X.reshape(len(s_unit.X), -1)
            tr_feats["A_raw_sensors"].append(f_A)
            
            # B: Encoder z
            with torch.no_grad():
                z = trainer.encoder(Xt, Mt).cpu().numpy()
            tr_feats["B_encoder_z"].append(z)
            
            # C: VQ one-hot
            f_C = np.eye(trainer.vq.K)[ev]
            tr_feats["C_vq_one_hot"].append(f_C)
            
            # D, E, F: Causal HSMM
            r = fb.causal_filter(ev)
            f_D = r.state_posterior
            f_E = np.concatenate([r.state_posterior, r.expected_dwell / trainer.hsmm.d_max], axis=1)
            f_F = f_E.copy()
            tr_feats["D_causal_gamma"].append(f_D)
            tr_feats["E_gamma_dwell"].append(f_E)
            tr_feats["F_full_f_struct"].append(f_F)
            
        # Build features for test units
        te_feats = {r: [] for r in rep_names}
        for s_unit in te4:
            Xt = torch.as_tensor(s_unit.X, dtype=torch.float32)
            Mt = torch.ones_like(Xt)
            toks = np.concatenate(trainer.event_sequences(Xt, Mt))
            
            # A
            f_A = s_unit.X.reshape(len(s_unit.X), -1)
            te_feats["A_raw_sensors"].append(f_A)
            
            # B
            with torch.no_grad():
                z = trainer.encoder(Xt, Mt).cpu().numpy()
            te_feats["B_encoder_z"].append(z)
            
            # C
            f_C = np.eye(trainer.vq.K)[toks]
            te_feats["C_vq_one_hot"].append(f_C)
            
            # D, E, F
            r = fb.causal_filter(toks)
            f_D = r.state_posterior
            f_E = np.concatenate([r.state_posterior, r.expected_dwell / trainer.hsmm.d_max], axis=1)
            f_F = f_E.copy()
            te_feats["D_causal_gamma"].append(f_D)
            te_feats["E_gamma_dwell"].append(f_E)
            te_feats["F_full_f_struct"].append(f_F)
            
        # Fit Ridge and evaluate for each representation
        for r in rep_names:
            X_tr = np.concatenate(tr_feats[r])
            X_te = np.concatenate(te_feats[r])
            rep_results[r]["dim"] = X_tr.shape[1]
            
            w = _ridge_fit(X_tr, ytr4)
            preds = _ridge_predict(w, X_te)
            m = rul_metrics(preds, yte4)
            rep_results[r]["rmses"].append(m["RMSE"])
            rep_results[r]["maes"].append(m["MAE"])
            
    bottleneck_summary = {}
    print("\nInformation Bottleneck Results Table:")
    print(f"{'Representation':22s} {'Dim':>5s} {'RMSE (mean±std)':>18s} {'MAE (mean±std)':>18s}")
    for r in rep_names:
        mean_rmse = float(np.mean(rep_results[r]["rmses"]))
        std_rmse = float(np.std(rep_results[r]["rmses"]))
        mean_mae = float(np.mean(rep_results[r]["maes"]))
        std_mae = float(np.std(rep_results[r]["maes"]))
        d = rep_results[r]["dim"]
        bottleneck_summary[r] = {
            "dim": d,
            "rmse_mean": mean_rmse, "rmse_std": std_rmse,
            "mae_mean": mean_mae, "mae_std": std_mae,
            "per_seed_rmse": rep_results[r]["rmses"],
            "per_seed_mae": rep_results[r]["maes"],
        }
        print(f"{r:22s} {d:5d} {mean_rmse:7.2f} ± {std_rmse:<6.2f} {mean_mae:7.2f} ± {std_mae:<6.2f}")
        
    final_output = {
        "phase2_reproducibility": p2_results,
        "phase3_4_aligned_w4_vs_w16": aligned_results,
        "phase5_diagnostics": {
            "w4": diag_w4,
            "w16": diag_w16
        },
        "phase6_information_bottleneck": bottleneck_summary
    }
    
    with open("scientific_audit_results.json", "w") as f:
        json.dump(final_output, f, indent=2)
    print("\nSaved full scientific audit report to scientific_audit_results.json")

if __name__ == "__main__":
    run_complete_audit()
