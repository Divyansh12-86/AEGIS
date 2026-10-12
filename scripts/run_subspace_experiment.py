import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
"""Subspace Decoupling Controlled Experiment:
Arm 1: Control (16/0) - full 16-D latent to both VQ and RUL
Arm 2: 12/4 Subspace - z_state in R^12 to VQ, z_prog in R^4 to RUL
Arm 3: 8/8 Subspace - z_state in R^8 to VQ, z_prog in R^8 to RUL

Fixed configuration:
- Dataset: N-CMAPSS DS01
- Train: u3, u4, u5, u6, u7, u8
- Val: u1, u10
- Test: u2, u9
- W=4, S=4, n_conv_blocks=3 (RF=19)
- VQ: K=16, beta=0.25, ema_decay=0.99
- HSMM: M=5, D_max=10
- Causal filtering (no beta, no smoothing)
- Ridge lambda=1.0
- lambda_RUL = 0.1
- Seeds: [0, 1, 2]
- Dead-code revival disabled
"""
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
from egpm.utils import set_global_seed


def compute_pca_spectrum(X):
    # X: [N, d]
    X_c = X - np.mean(X, axis=0)
    _, S, _ = np.linalg.svd(X_c, full_matrices=False)
    var = (S ** 2) / np.sum(S ** 2)
    pc1 = float(var[0])
    pc12 = float(var[0] + var[1]) if len(var) > 1 else pc1
    pc123 = float(var[0] + var[1] + var[2]) if len(var) > 2 else pc12
    return pc1, pc12, pc123


def compute_correlations(X, y):
    # X: [N, d], y: [N]
    p_corrs, s_corrs = [], []
    for d in range(X.shape[1]):
        col = X[:, d]
        r_p = float(np.corrcoef(col, y)[0, 1]) if np.std(col) > 1e-6 else 0.0
        r_s = float(scipy.stats.spearmanr(col, y).statistic) if np.std(col) > 1e-6 else 0.0
        p_corrs.append(r_p)
        s_corrs.append(r_s)
    max_p = float(np.max(np.abs(p_corrs)))
    max_s = float(np.max(np.abs(s_corrs)))
    mean_s = float(np.mean(np.abs(s_corrs)))
    mono_dims = int(np.sum(np.abs(s_corrs) > 0.5))
    return max_p, max_s, mean_s, mono_dims


def run_subspace_experiment():
    print("=" * 75)
    print("EGPM SUBSPACE DECOUPLING CONTROLLED EXPERIMENT (λ_RUL = 0.1)")
    print("=" * 75)

    dataset_path = "N-CMAPSS/N-CMAPSS DS01 005.h5"
    raw_dataset = NCMAPSSLoader().load(dataset_path)
    raw_dataset.records = [_downsample(r) for r in raw_dataset.records]

    # Preprocessing strictly on train split
    sb = SplitBuilder(raw_dataset, seed=0)
    split = sb.build()
    stats = sb.fit_normalization(split)

    tr_units = split.train_unit_ids
    val_units = split.val_unit_ids
    te_units = split.test_unit_ids
    print(f"Train engines:      {tr_units}")
    print(f"Validation engines: {val_units}")
    print(f"Test engines:       {te_units}")

    # Build windows W=4, S=4
    prep = Preprocessor(window_length=4, stride=4)
    tr_seqs = [prep.process(raw_dataset[u], norm_stats=stats) for u in tr_units]
    val_seqs = [prep.process(raw_dataset[u], norm_stats=stats) for u in val_units]
    te_seqs = [prep.process(raw_dataset[u], norm_stats=stats) for u in te_units]

    Xtr = torch.as_tensor(np.concatenate([s.X for s in tr_seqs]), dtype=torch.float32)
    Mtr = torch.ones_like(Xtr)
    ytr = np.concatenate([s.rul_labels for s in tr_seqs])
    yte = np.concatenate([s.rul_labels for s in te_seqs])
    ltr = [len(s.X) for s in tr_seqs]

    # Causality verification: right edge s + W - 1
    for u, s in zip(tr_units, tr_seqs):
        rec = raw_dataset[u]
        assert s.rul_labels[0] == rec.rul_labels[3], f"Train target alignment mismatch on {u}"
    for u, s in zip(te_units, te_seqs):
        rec = raw_dataset[u]
        assert s.rul_labels[0] == rec.rul_labels[3], f"Test target alignment mismatch on {u}"

    arms = [
        {"name": "Arm 1 (Control 16/0)", "state_dim": None, "prog_dim": 16},
        {"name": "Arm 2 (12/4 Subspace)", "state_dim": 12, "prog_dim": 4},
        {"name": "Arm 3 (8/8 Subspace)",  "state_dim": 8,  "prog_dim": 8},
    ]

    seeds = [0, 1, 2]
    all_results = {}

    for arm in arms:
        arm_name = arm["name"]
        s_dim = arm["state_dim"]
        p_dim = arm["prog_dim"]
        print(f"\n{'='*70}")
        print(f"RUNNING {arm_name.upper()} across seeds {seeds}")
        print(f"{'='*70}")

        # Metrics lists across seeds
        prog_rmses, prog_maes = [], []
        egpm_rmses, egpm_maes = [], []
        per_eng_prog = {u: {"rmse": [], "mae": []} for u in te_units}
        per_eng_egpm = {u: {"rmse": [], "mae": []} for u in te_units}

        # VQ Diagnostics
        vq_perplexities = []
        active_codes_list = []
        dead_fractions = []
        code_entropies = []
        code_freqs_list = []
        code_usage_per_seed = []

        # Latent Geometry Diagnostics
        pca_z16_list = []
        pca_state_list = []
        pca_prog_list = []
        prog_corr_list = []

        # HSMM Diagnostics
        hsmm_occupancies = []
        hsmm_entropies = []
        hsmm_dwells = []
        hsmm_trans_entropies = []

        # Event Vocabulary & Sequences
        event_sequences_per_seed = {}
        mean_run_lengths = []

        for s in seeds:
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
                state_dim=s_dim,
                dead_code_revival=False,
                stage1_steps=300,
                batch_size=64,
                seed=s,
                log_every=0,
            )
            trainer = Trainer(cfg, n_channels=Xtr.shape[-1])
            ytr_tensor = torch.as_tensor(ytr, dtype=torch.float32)
            s1_stats = trainer.train_stage1(Xtr, Mtr, rul_labels=ytr_tensor)

            # Extract continuous latent z [N, 16]
            with torch.no_grad():
                z_tr = trainer.encoder(Xtr, Mtr).cpu().numpy()
                z_te_list = []
                for s_unit in te_seqs:
                    Xt = torch.as_tensor(s_unit.X, dtype=torch.float32)
                    Mt = torch.ones_like(Xt)
                    z_te_list.append(trainer.encoder(Xt, Mt).cpu().numpy())
                z_te = np.concatenate(z_te_list)

            # Subspace slices:
            if s_dim is not None:
                z_state_tr = z_tr[:, :s_dim]
                z_prog_tr = z_tr[:, s_dim:]
                z_state_te = z_te[:, :s_dim]
                z_prog_te = z_te[:, s_dim:]
                z_prog_te_list = [u_z[:, s_dim:] for u_z in z_te_list]
            else:
                z_state_tr = z_tr
                z_prog_tr = z_tr
                z_state_te = z_te
                z_prog_te = z_te
                z_prog_te_list = z_te_list

            # -------------------------------------------------------------
            # 1. Prognostic RUL Readout: z_prog -> Ridge (closed-form lambda=1.0)
            # -------------------------------------------------------------
            w_prog = _ridge_fit(z_prog_tr, ytr)
            preds_prog_list = [_ridge_predict(w_prog, u_prog) for u_prog in z_prog_te_list]
            preds_prog = np.concatenate(preds_prog_list)
            m_prog = rul_metrics(preds_prog, yte)
            prog_rmses.append(m_prog["RMSE"])
            prog_maes.append(m_prog["MAE"])

            for u, p_u, s_u in zip(te_units, preds_prog_list, te_seqs):
                mu = rul_metrics(p_u, s_u.rul_labels)
                per_eng_prog[u]["rmse"].append(mu["RMSE"])
                per_eng_prog[u]["mae"].append(mu["MAE"])

            # -------------------------------------------------------------
            # 2. Structural EGPM Readout: z_state -> VQ -> events -> HSMM -> gamma+dwell -> Ridge
            # -------------------------------------------------------------
            ev_tr = trainer.window_runs_to_event_runs(trainer.event_sequences(Xtr, Mtr), ltr)
            trainer.train_stage2(ev_tr)
            fb = ForwardBackward(trainer.hsmm)

            # Train structural features
            f_tr = []
            for s_unit, ev in zip(tr_seqs, ev_tr):
                r = fb.causal_filter(ev)
                f_tr.append(np.concatenate([r.state_posterior, r.expected_dwell / trainer.hsmm.d_max], axis=1))
            w_egpm = _ridge_fit(np.concatenate(f_tr), ytr)

            # Test structural features & event sequences
            preds_egpm_list = []
            ev_te_list = []
            f_te_list = []
            for s_unit in te_seqs:
                Xt = torch.as_tensor(s_unit.X, dtype=torch.float32)
                toks = np.concatenate(trainer.event_sequences(Xt, torch.ones_like(Xt)))
                ev_te_list.append(toks)
                r = fb.causal_filter(toks)
                f_test = np.concatenate([r.state_posterior, r.expected_dwell / trainer.hsmm.d_max], axis=1)
                f_te_list.append(f_test)
                preds_egpm_list.append(_ridge_predict(w_egpm, f_test))

            preds_egpm = np.concatenate(preds_egpm_list)
            m_egpm = rul_metrics(preds_egpm, yte)
            egpm_rmses.append(m_egpm["RMSE"])
            egpm_maes.append(m_egpm["MAE"])

            for u, p_u, s_u in zip(te_units, preds_egpm_list, te_seqs):
                mu = rul_metrics(p_u, s_u.rul_labels)
                per_eng_egpm[u]["rmse"].append(mu["RMSE"])
                per_eng_egpm[u]["mae"].append(mu["MAE"])

            # -------------------------------------------------------------
            # 3. VQ Health Diagnostics
            # -------------------------------------------------------------
            all_te_events = np.concatenate(ev_te_list)
            counts = np.bincount(all_te_events, minlength=trainer.vq.K)
            probs = counts / counts.sum()
            ent = -np.sum(probs[probs > 0] * np.log(probs[probs > 0]))
            active = int(np.sum(counts > 0))
            dead = float((trainer.vq.K - active) / trainer.vq.K)

            vq_perplexities.append(float(s1_stats.perplexity))
            active_codes_list.append(active)
            dead_fractions.append(dead)
            code_entropies.append(float(ent))
            code_freqs_list.append(probs.tolist())
            code_usage_per_seed.append(counts.tolist())

            # -------------------------------------------------------------
            # 4. Latent Geometry Diagnostics
            # -------------------------------------------------------------
            pc1_z16, pc12_z16, pc123_z16 = compute_pca_spectrum(z_te)
            pca_z16_list.append([pc1_z16, pc12_z16, pc123_z16])

            pc1_state, pc12_state, pc123_state = compute_pca_spectrum(z_state_te)
            pca_state_list.append([pc1_state, pc12_state, pc123_state])

            pc1_prog, pc12_prog, pc123_prog = compute_pca_spectrum(z_prog_te)
            pca_prog_list.append([pc1_prog, pc12_prog, pc123_prog])

            max_p, max_s, mean_s, mono_d = compute_correlations(z_prog_te, yte)
            prog_corr_list.append([max_p, max_s, mean_s, mono_d])

            # -------------------------------------------------------------
            # 5. HSMM Diagnostics
            # -------------------------------------------------------------
            all_gammas = np.concatenate([f[:, :trainer.hsmm.M] for f in f_te_list], axis=0)
            state_occ = np.mean(all_gammas, axis=0)
            hsmm_occupancies.append(state_occ.tolist())
            step_ents = -np.sum(all_gammas * np.log(np.clip(all_gammas, 1e-12, None)), axis=1)
            hsmm_entropies.append(float(np.mean(step_ents)))

            all_dwells = np.concatenate([f[:, trainer.hsmm.M:] for f in f_te_list], axis=0)
            hsmm_dwells.append(float(np.mean(all_dwells) * trainer.hsmm.d_max))

            A = trainer.hsmm.A
            row_ents = []
            for row in A:
                r_pos = row[row > 0]
                row_ents.append(-float(np.sum(r_pos * np.log(r_pos))) if len(r_pos) > 0 else 0.0)
            hsmm_trans_entropies.append(float(np.mean(row_ents)))

            # -------------------------------------------------------------
            # 6. Event Vocabulary & Sequences
            # -------------------------------------------------------------
            # Run lengths for event tokens:
            run_lens = []
            for ev_seq in ev_te_list:
                # count run lengths
                if len(ev_seq) == 0:
                    continue
                cur_run = 1
                for i in range(1, len(ev_seq)):
                    if ev_seq[i] == ev_seq[i - 1]:
                        cur_run += 1
                    else:
                        run_lens.append(cur_run)
                        cur_run = 1
                run_lens.append(cur_run)
            mean_run_lengths.append(float(np.mean(run_lens)) if run_lens else 0.0)

            # Store representative sequence (DS01_u2, index 0 in te_units)
            event_sequences_per_seed[f"seed_{s}"] = {
                u: ev_seq.tolist() for u, ev_seq in zip(te_units, ev_te_list)
            }

            dt = time.time() - t0
            print(f"  Seed {s} ({dt:.1f}s): prog RMSE={m_prog['RMSE']:.2f}, EGPM RMSE={m_egpm['RMSE']:.2f}, active={active}/16, perp={s1_stats.perplexity:.2f}, z_state PC1={pc1_state*100:.1f}%, z_prog PC1={pc1_prog*100:.1f}%, max |r_s|={max_s:.3f}")

        # Compute Jaccard active token stability across seeds
        active_sets = [set(np.where(np.array(cnts) > 0)[0]) for cnts in code_usage_per_seed]
        jaccard_01 = len(active_sets[0] & active_sets[1]) / len(active_sets[0] | active_sets[1]) if len(active_sets[0] | active_sets[1]) > 0 else 1.0
        jaccard_12 = len(active_sets[1] & active_sets[2]) / len(active_sets[1] | active_sets[2]) if len(active_sets[1] | active_sets[2]) > 0 else 1.0
        jaccard_02 = len(active_sets[0] & active_sets[2]) / len(active_sets[0] | active_sets[2]) if len(active_sets[0] | active_sets[2]) > 0 else 1.0
        mean_jaccard = float(np.mean([jaccard_01, jaccard_12, jaccard_02]))

        arm_summary = {
            "name": arm_name,
            "state_dim": s_dim,
            "prog_dim": p_dim,
            "prognostic_readout": {
                "rmse_mean": float(np.mean(prog_rmses)),
                "rmse_std": float(np.std(prog_rmses)),
                "mae_mean": float(np.mean(prog_maes)),
                "mae_std": float(np.std(prog_maes)),
                "per_seed_rmse": prog_rmses,
                "per_seed_mae": prog_maes,
                "per_engine": {
                    u: {
                        "rmse_mean": float(np.mean(per_eng_prog[u]["rmse"])),
                        "rmse_std": float(np.std(per_eng_prog[u]["rmse"])),
                        "mae_mean": float(np.mean(per_eng_prog[u]["mae"])),
                        "mae_std": float(np.std(per_eng_prog[u]["mae"])),
                    } for u in te_units
                },
            },
            "egpm_readout": {
                "rmse_mean": float(np.mean(egpm_rmses)),
                "rmse_std": float(np.std(egpm_rmses)),
                "mae_mean": float(np.mean(egpm_maes)),
                "mae_std": float(np.std(egpm_maes)),
                "per_seed_rmse": egpm_rmses,
                "per_seed_mae": egpm_maes,
                "per_engine": {
                    u: {
                        "rmse_mean": float(np.mean(per_eng_egpm[u]["rmse"])),
                        "rmse_std": float(np.std(per_eng_egpm[u]["rmse"])),
                        "mae_mean": float(np.mean(per_eng_egpm[u]["mae"])),
                        "mae_std": float(np.std(per_eng_egpm[u]["mae"])),
                    } for u in te_units
                },
            },
            "vq_diagnostics": {
                "perplexity_mean": float(np.mean(vq_perplexities)),
                "perplexity_std": float(np.std(vq_perplexities)),
                "active_codes_mean": float(np.mean(active_codes_list)),
                "active_codes_per_seed": active_codes_list,
                "active_fraction_mean": float(np.mean(active_codes_list) / 16.0),
                "dead_fraction_mean": float(np.mean(dead_fractions)),
                "code_entropy_mean": float(np.mean(code_entropies)),
                "per_seed_code_usage": code_usage_per_seed,
                "full_16_code_frequencies": np.mean(code_freqs_list, axis=0).tolist(),
                "mean_jaccard_token_stability": mean_jaccard,
            },
            "latent_geometry": {
                "z16_pca": {
                    "pc1_mean": float(np.mean([p[0] for p in pca_z16_list])),
                    "pc12_mean": float(np.mean([p[1] for p in pca_z16_list])),
                    "pc123_mean": float(np.mean([p[2] for p in pca_z16_list])),
                },
                "z_state_pca": {
                    "pc1_mean": float(np.mean([p[0] for p in pca_state_list])),
                    "pc12_mean": float(np.mean([p[1] for p in pca_state_list])),
                    "pc123_mean": float(np.mean([p[2] for p in pca_state_list])),
                },
                "z_prog_pca": {
                    "pc1_mean": float(np.mean([p[0] for p in pca_prog_list])),
                    "pc12_mean": float(np.mean([p[1] for p in pca_prog_list])),
                    "pc123_mean": float(np.mean([p[2] for p in pca_prog_list])),
                },
                "z_prog_correlations": {
                    "max_pearson_mean": float(np.mean([c[0] for c in prog_corr_list])),
                    "max_spearman_mean": float(np.mean([c[1] for c in prog_corr_list])),
                    "mean_spearman_mean": float(np.mean([c[2] for c in prog_corr_list])),
                    "strong_monotonic_dims_mean": float(np.mean([c[3] for c in prog_corr_list])),
                },
            },
            "hsmm_diagnostics": {
                "state_occupancy_mean": np.mean(hsmm_occupancies, axis=0).tolist(),
                "mean_state_entropy": float(np.mean(hsmm_entropies)),
                "mean_expected_dwell": float(np.mean(hsmm_dwells)),
                "mean_transition_entropy": float(np.mean(hsmm_trans_entropies)),
            },
            "event_vocabulary": {
                "mean_run_length": float(np.mean(mean_run_lengths)),
                "representative_sequences": {
                    "DS01_u2_seed0": event_sequences_per_seed["seed_0"]["DS01_u2"],
                    "DS01_u2_seed1": event_sequences_per_seed["seed_1"]["DS01_u2"],
                    "DS01_u2_seed2": event_sequences_per_seed["seed_2"]["DS01_u2"],
                    "DS01_u9_seed0": event_sequences_per_seed["seed_0"]["DS01_u9"],
                },
            },
        }

        all_results[arm_name] = arm_summary
        print(f"\nArm Summary for {arm_name}:")
        print(f"  z_prog Ridge:  RMSE = {arm_summary['prognostic_readout']['rmse_mean']:.2f} ± {arm_summary['prognostic_readout']['rmse_std']:.2f}")
        print(f"  EGPM Ridge:    RMSE = {arm_summary['egpm_readout']['rmse_mean']:.2f} ± {arm_summary['egpm_readout']['rmse_std']:.2f}")
        print(f"  VQ Health:     K_active = {arm_summary['vq_diagnostics']['active_codes_mean']:.1f}/16, perp = {arm_summary['vq_diagnostics']['perplexity_mean']:.2f}, dead = {arm_summary['vq_diagnostics']['dead_fraction_mean']*100:.1f}%")
        print(f"  z_state PC1:   {arm_summary['latent_geometry']['z_state_pca']['pc1_mean']*100:.1f}% (PC1+PC2: {arm_summary['latent_geometry']['z_state_pca']['pc12_mean']*100:.1f}%)")
        print(f"  z_prog PC1:    {arm_summary['latent_geometry']['z_prog_pca']['pc1_mean']*100:.1f}%")
        print(f"  z_prog |r_s|:  max = {arm_summary['latent_geometry']['z_prog_correlations']['max_spearman_mean']:.3f}, strong dims = {arm_summary['latent_geometry']['z_prog_correlations']['strong_monotonic_dims_mean']:.1f}")

    out_path = Path("results/ablations/subspace_experiment_results.json") if Path("results/ablations").exists() else Path("subspace_experiment_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved full subspace experiment results to subspace_experiment_results.json")


if __name__ == "__main__":
    run_subspace_experiment()
