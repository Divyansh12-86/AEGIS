"""Controlled Lambda_RUL Sweep Experiment for EGPM.
Evaluates the trade-off between prognostic information in z and discrete event diversity in VQ.

Arms:
λ_RUL ∈ [0.0, 0.005, 0.01, 0.03, 0.05, 0.1, 1.0]
Seeds: [0, 1, 2]

Fixed configuration:
- Dataset: N-CMAPSS DS01
- Train: u3, u4, u5, u6, u7, u8
- Val: u1, u10
- Test: u2, u9
- W=4, S=4, n_conv_blocks=3 (RF=19)
- VQ: K=16, embed_dim=16
- HSMM: M=5, D_max=10
- Causal filtering (no beta, no smoothing)
- Ridge readout (lambda=1.0)
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


def run_lambda_sweep():
    print("=" * 75)
    print("EGPM λ_RUL CONTROLLED SWEEP: PROGNOSTIC SUPERVISION VS VQ HEALTH")
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

    lambdas = [0.0, 0.005, 0.01, 0.03, 0.05, 0.1, 1.0]
    seeds = [0, 1, 2]
    sweep_results = {}

    for l_rul in lambdas:
        l_str = f"lambda_{l_rul}"
        print(f"\n{'='*70}")
        print(f"EVALUATING λ_RUL = {l_rul} across seeds {seeds}")
        print(f"{'='*70}")

        z_rmses, z_maes = [], []
        egpm_rmses, egpm_maes = [], []
        per_eng_z = {u: {"rmse": [], "mae": []} for u in te_units}
        per_eng_egpm = {u: {"rmse": [], "mae": []} for u in te_units}

        # VQ Diagnostics
        vq_perplexities = []
        active_codes_list = []
        code_entropies = []
        dead_fractions = []
        code_freqs_list = []

        # HSMM Diagnostics
        hsmm_occupancies = []
        hsmm_entropies = []
        hsmm_dwells = []
        hsmm_trans_entropies = []

        # Latent Correlations & PCA
        pearson_corrs = []
        spearman_corrs = []
        max_pearson_list = []
        max_spearman_list = []
        dim_monotonicity_list = []
        pc1_var_list = []
        pca_top3_list = []

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
                aux_rul_lambda=l_rul,
                stage1_steps=300,
                batch_size=64,
                seed=s,
                log_every=0,
            )
            trainer = Trainer(cfg, n_channels=Xtr.shape[-1])
            ytr_tensor = torch.as_tensor(ytr, dtype=torch.float32)
            s1_stats = trainer.train_stage1(Xtr, Mtr, rul_labels=ytr_tensor)

            # Continuous latent z
            with torch.no_grad():
                z_tr = trainer.encoder(Xtr, Mtr).cpu().numpy()
                z_te_list = []
                for s_unit in te_seqs:
                    Xt = torch.as_tensor(s_unit.X, dtype=torch.float32)
                    Mt = torch.ones_like(Xt)
                    z_te_list.append(trainer.encoder(Xt, Mt).cpu().numpy())
                z_te = np.concatenate(z_te_list)

            # 1. z-only Ridge readout (lambda=1.0)
            w_z = _ridge_fit(z_tr, ytr)
            preds_z_list = [_ridge_predict(w_z, z_u) for z_u in z_te_list]
            preds_z = np.concatenate(preds_z_list)
            mz = rul_metrics(preds_z, yte)
            z_rmses.append(mz["RMSE"])
            z_maes.append(mz["MAE"])

            for u, p_u, s_u in zip(te_units, preds_z_list, te_seqs):
                mu = rul_metrics(p_u, s_u.rul_labels)
                per_eng_z[u]["rmse"].append(mu["RMSE"])
                per_eng_z[u]["mae"].append(mu["MAE"])

            # 2. PCA Concentration on z_te
            z_centered = z_te - np.mean(z_te, axis=0)
            _, S, _ = np.linalg.svd(z_centered, full_matrices=False)
            var_explained = (S ** 2) / np.sum(S ** 2)
            pc1_var_list.append(float(var_explained[0]))
            pca_top3_list.append([float(v) for v in var_explained[:3]])

            # 3. Latent z Correlations with RUL
            p_corrs = []
            s_corrs = []
            for d in range(z_te.shape[1]):
                col = z_te[:, d]
                r_p = float(np.corrcoef(col, yte)[0, 1]) if np.std(col) > 1e-6 else 0.0
                r_s = float(scipy.stats.spearmanr(col, yte).statistic) if np.std(col) > 1e-6 else 0.0
                p_corrs.append(r_p)
                s_corrs.append(r_s)

            pearson_corrs.append(p_corrs)
            spearman_corrs.append(s_corrs)
            max_pearson_list.append(float(np.max(np.abs(p_corrs))))
            max_spearman_list.append(float(np.max(np.abs(s_corrs))))
            mono_dims = int(np.sum(np.abs(s_corrs) > 0.5))
            dim_monotonicity_list.append(mono_dims)

            # 4. Discrete Events & Causal HSMM Pipeline
            ev_tr = trainer.window_runs_to_event_runs(trainer.event_sequences(Xtr, Mtr), ltr)
            trainer.train_stage2(ev_tr)
            fb = ForwardBackward(trainer.hsmm)

            # Causal filtering on train
            f_tr = []
            for s_unit, ev in zip(tr_seqs, ev_tr):
                r = fb.causal_filter(ev)
                f_tr.append(np.concatenate([r.state_posterior, r.expected_dwell / trainer.hsmm.d_max], axis=1))
            w_egpm = _ridge_fit(np.concatenate(f_tr), ytr)

            # Causal filtering on test
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

            # 5. VQ Diagnostics
            all_te_events = np.concatenate(ev_te_list)
            counts = np.bincount(all_te_events, minlength=trainer.vq.K)
            probs = counts / counts.sum()
            ent = -np.sum(probs[probs > 0] * np.log(probs[probs > 0]))
            active = int(np.sum(counts > 0))
            dead = float((trainer.vq.K - active) / trainer.vq.K)

            vq_perplexities.append(float(s1_stats.perplexity))
            active_codes_list.append(active)
            code_entropies.append(float(ent))
            dead_fractions.append(dead)
            code_freqs_list.append(probs.tolist())

            # 6. HSMM Diagnostics
            all_gammas = np.concatenate([f[:, :trainer.hsmm.M] for f in f_te_list], axis=0)
            state_occ = np.mean(all_gammas, axis=0)
            hsmm_occupancies.append(state_occ.tolist())
            step_ents = -np.sum(all_gammas * np.log(np.clip(all_gammas, 1e-12, None)), axis=1)
            hsmm_entropies.append(float(np.mean(step_ents)))

            all_dwells = np.concatenate([f[:, trainer.hsmm.M:] for f in f_te_list], axis=0)
            hsmm_dwells.append(float(np.mean(all_dwells) * trainer.hsmm.d_max))

            # HSMM Transition usage: row entropy of A
            A = trainer.hsmm.A
            row_ents = []
            for row in A:
                r_pos = row[row > 0]
                if len(r_pos) > 0:
                    row_ents.append(-float(np.sum(r_pos * np.log(r_pos))))
                else:
                    row_ents.append(0.0)
            hsmm_trans_entropies.append(float(np.mean(row_ents)))

            dt = time.time() - t0
            print(f"  Seed {s} ({dt:.1f}s): z RMSE={mz['RMSE']:.2f}, EGPM RMSE={m_egpm['RMSE']:.2f}, active={active}/16, perp={s1_stats.perplexity:.2f}, max|r_s|={np.max(np.abs(s_corrs)):.3f}, PC1 var={var_explained[0]*100:.1f}%")

        arm_summary = {
            "lambda_rul": l_rul,
            "z_readout": {
                "rmse_mean": float(np.mean(z_rmses)),
                "rmse_std": float(np.std(z_rmses)),
                "mae_mean": float(np.mean(z_maes)),
                "mae_std": float(np.std(z_maes)),
                "per_seed_rmse": z_rmses,
                "per_seed_mae": z_maes,
                "per_engine": {
                    u: {
                        "rmse_mean": float(np.mean(per_eng_z[u]["rmse"])),
                        "rmse_std": float(np.std(per_eng_z[u]["rmse"])),
                        "mae_mean": float(np.mean(per_eng_z[u]["mae"])),
                        "mae_std": float(np.std(per_eng_z[u]["mae"])),
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
                "code_entropy_mean": float(np.mean(code_entropies)),
                "dead_fraction_mean": float(np.mean(dead_fractions)),
                "per_code_frequencies_mean": np.mean(code_freqs_list, axis=0).tolist(),
            },
            "hsmm_diagnostics": {
                "state_occupancy_mean": np.mean(hsmm_occupancies, axis=0).tolist(),
                "mean_state_entropy": float(np.mean(hsmm_entropies)),
                "mean_expected_dwell": float(np.mean(hsmm_dwells)),
                "mean_transition_entropy": float(np.mean(hsmm_trans_entropies)),
            },
            "correlations": {
                "max_pearson_mean": float(np.mean(max_pearson_list)),
                "max_spearman_mean": float(np.mean(max_spearman_list)),
                "mean_abs_spearman_over_dims": float(np.mean(np.abs(spearman_corrs))),
                "strong_monotonic_dims_mean": float(np.mean(dim_monotonicity_list)),
                "per_seed_max_spearman": max_spearman_list,
            },
            "pca_diagnostics": {
                "pc1_variance_mean": float(np.mean(pc1_var_list)),
                "pc1_variance_std": float(np.std(pc1_var_list)),
                "top3_variance_mean": np.mean(pca_top3_list, axis=0).tolist(),
            },
        }

        sweep_results[l_str] = arm_summary
        print(f"\nArm Summary for λ={l_rul}:")
        print(f"  z Ridge:      RMSE = {arm_summary['z_readout']['rmse_mean']:.2f} ± {arm_summary['z_readout']['rmse_std']:.2f}")
        print(f"  EGPM Ridge:   RMSE = {arm_summary['egpm_readout']['rmse_mean']:.2f} ± {arm_summary['egpm_readout']['rmse_std']:.2f}")
        print(f"  Spearman:     max |r_s| = {arm_summary['correlations']['max_spearman_mean']:.3f}, strong dims = {arm_summary['correlations']['strong_monotonic_dims_mean']:.1f}")
        print(f"  VQ Health:    K_active = {arm_summary['vq_diagnostics']['active_codes_mean']:.1f}/16, perp = {arm_summary['vq_diagnostics']['perplexity_mean']:.2f}, dead = {arm_summary['vq_diagnostics']['dead_fraction_mean']*100:.1f}%")
        print(f"  PCA PC1 Var:  {arm_summary['pca_diagnostics']['pc1_variance_mean']*100:.1f}%")

    with open("lambda_sweep_results.json", "w") as f:
        json.dump(sweep_results, f, indent=2)
    print("\nSaved full sweep results to lambda_sweep_results.json")


if __name__ == "__main__":
    run_lambda_sweep()
