"""Controlled Prognostic Auxiliary Loss Experiment:
Evaluates whether adding auxiliary prognostic supervision (Huber loss)
to the Stage-1 SensorEncoder recovers prognostic information in z and
improves causal EGPM downstream.

Conditions:
- A: λ_RUL = 0.0 (Reconstruction only)
- B: λ_RUL = 0.1 (Weak prognostic)
- C: λ_RUL = 1.0 (Stronger prognostic)

Fixed configuration:
- Dataset: N-CMAPSS DS01
- W=4, S=4, n_conv_blocks=3 (RF=19)
- VQ: K=16, embed_dim=16
- HSMM: M=5, D_max=10
- Causal filtering (no beta, no smoothing)
- Ridge readout (lambda=1.0)
- Seeds: [0, 1, 2]
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


def run_experiment():
    print("=" * 70)
    print("EGPM AUXILIARY PROGNOSTIC SUPERVISION CONTROLLED EXPERIMENT")
    print("=" * 70)

    dataset_path = "N-CMAPSS/N-CMAPSS DS01 005.h5"
    raw_dataset = NCMAPSSLoader().load(dataset_path)
    raw_dataset.records = [_downsample(r) for r in raw_dataset.records]

    # Preprocessing strictly on train split
    sb = SplitBuilder(raw_dataset, seed=0)
    split = sb.build()
    stats = sb.fit_normalization(split)

    tr_units = split.train_unit_ids
    te_units = split.test_unit_ids
    print(f"Train engines: {tr_units}")
    print(f"Test engines:  {te_units}")

    # Build windows W=4, S=4
    prep = Preprocessor(window_length=4, stride=4)
    tr_seqs = [prep.process(raw_dataset[u], norm_stats=stats) for u in tr_units]
    te_seqs = [prep.process(raw_dataset[u], norm_stats=stats) for u in te_units]

    Xtr = torch.as_tensor(np.concatenate([s.X for s in tr_seqs]), dtype=torch.float32)
    Mtr = torch.ones_like(Xtr)
    ytr = np.concatenate([s.rul_labels for s in tr_seqs])
    yte = np.concatenate([s.rul_labels for s in te_seqs])
    ltr = [len(s.X) for s in tr_seqs]
    lte = [len(s.X) for s in te_seqs]

    # Target alignment check: RUL is taken at window right edge s + W - 1
    # Verify labels match
    for u, s in zip(tr_units, tr_seqs):
        rec = raw_dataset[u]
        # check that s.rul_labels[0] == rec.rul_labels[3]
        assert s.rul_labels[0] == rec.rul_labels[3], "Target alignment mismatch on train!"
    for u, s in zip(te_units, te_seqs):
        rec = raw_dataset[u]
        assert s.rul_labels[0] == rec.rul_labels[3], "Target alignment mismatch on test!"

    conditions = [
        {"name": "Condition A (Recon only)", "lambda_rul": 0.0},
        {"name": "Condition B (Weak aux)", "lambda_rul": 0.1},
        {"name": "Condition C (Stronger aux)", "lambda_rul": 1.0},
    ]

    seeds = [0, 1, 2]
    all_results = {}

    for cond in conditions:
        c_name = cond["name"]
        l_rul = cond["lambda_rul"]
        print(f"\n" + "-" * 60)
        print(f"Running {c_name} with λ_RUL = {l_rul} across seeds {seeds}...")
        print("-" * 60)

        z_rmses, z_maes = [], []
        egpm_rmses, egpm_maes = [], []
        per_eng_z = {u: {"rmse": [], "mae": []} for u in te_units}
        per_eng_egpm = {u: {"rmse": [], "mae": []} for u in te_units}

        vq_perplexities = []
        active_codes_list = []
        code_entropies = []
        dead_fractions = []
        hsmm_occupancies = []
        hsmm_entropies = []
        hsmm_dwells = []

        pearson_corrs = []
        spearman_corrs = []
        max_pearson_list = []
        max_spearman_list = []
        dim_monotonicity_list = []

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
            # Train Stage 1
            ytr_tensor = torch.as_tensor(ytr, dtype=torch.float32)
            s1_stats = trainer.train_stage1(Xtr, Mtr, rul_labels=ytr_tensor)

            # Extract continuous latent z
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

            # 2. Complete EGPM Pipeline: VQ -> Causal HSMM -> F_struct -> Ridge
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

            # 3. VQ & HSMM Quality Diagnostics
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

            all_gammas = np.concatenate([f[:, :trainer.hsmm.M] for f in f_te_list], axis=0)
            state_occ = np.mean(all_gammas, axis=0)
            hsmm_occupancies.append(state_occ.tolist())
            step_ents = -np.sum(all_gammas * np.log(np.clip(all_gammas, 1e-12, None)), axis=1)
            hsmm_entropies.append(float(np.mean(step_ents)))

            all_dwells = np.concatenate([f[:, trainer.hsmm.M:] for f in f_te_list], axis=0)
            hsmm_dwells.append(float(np.mean(all_dwells) * trainer.hsmm.d_max))

            # 4. Latent z Correlation Analysis with RUL
            # Compute Pearson & Spearman for each dimension d in 0..15 on test set
            p_corrs = []
            s_corrs = []
            for d in range(z_te.shape[1]):
                col = z_te[:, d]
                # Pearson
                r_p = float(np.corrcoef(col, yte)[0, 1]) if np.std(col) > 1e-6 else 0.0
                # Spearman
                r_s = float(scipy.stats.spearmanr(col, yte).statistic) if np.std(col) > 1e-6 else 0.0
                p_corrs.append(r_p)
                s_corrs.append(r_s)

            pearson_corrs.append(p_corrs)
            spearman_corrs.append(s_corrs)
            max_pearson_list.append(float(np.max(np.abs(p_corrs))))
            max_spearman_list.append(float(np.max(np.abs(s_corrs))))

            # Count dimensions with strong monotonicity (|r_s| > 0.5)
            mono_dims = int(np.sum(np.abs(s_corrs) > 0.5))
            dim_monotonicity_list.append(mono_dims)

            dt = time.time() - t0
            print(f"  Seed {s} ({dt:.1f}s): z RMSE={mz['RMSE']:.2f}, EGPM RMSE={m_egpm['RMSE']:.2f}, active codes={active}, max |r_s|={np.max(np.abs(s_corrs)):.3f}")

        cond_result = {
            "name": c_name,
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
            "diagnostics": {
                "vq_perplexity_mean": float(np.mean(vq_perplexities)),
                "vq_perplexity_std": float(np.std(vq_perplexities)),
                "active_codes_mean": float(np.mean(active_codes_list)),
                "active_codes_per_seed": active_codes_list,
                "code_entropy_mean": float(np.mean(code_entropies)),
                "dead_fraction_mean": float(np.mean(dead_fractions)),
                "hsmm_occupancy_mean": np.mean(hsmm_occupancies, axis=0).tolist(),
                "mean_state_entropy": float(np.mean(hsmm_entropies)),
                "mean_expected_dwell": float(np.mean(hsmm_dwells)),
            },
            "correlations": {
                "max_pearson_mean": float(np.mean(max_pearson_list)),
                "max_spearman_mean": float(np.mean(max_spearman_list)),
                "mean_abs_pearson_over_dims": float(np.mean(np.abs(pearson_corrs))),
                "mean_abs_spearman_over_dims": float(np.mean(np.abs(spearman_corrs))),
                "strong_monotonic_dims_mean": float(np.mean(dim_monotonicity_list)),
                "per_seed_max_spearman": max_spearman_list,
                "per_seed_max_pearson": max_pearson_list,
            },
        }

        all_results[c_name] = cond_result
        print(f"Summary for {c_name}:")
        print(f"  z-only Ridge:  RMSE = {cond_result['z_readout']['rmse_mean']:.2f} ± {cond_result['z_readout']['rmse_std']:.2f}, MAE = {cond_result['z_readout']['mae_mean']:.2f} ± {cond_result['z_readout']['mae_std']:.2f}")
        print(f"  Full EGPM:     RMSE = {cond_result['egpm_readout']['rmse_mean']:.2f} ± {cond_result['egpm_readout']['rmse_std']:.2f}, MAE = {cond_result['egpm_readout']['mae_mean']:.2f} ± {cond_result['egpm_readout']['mae_std']:.2f}")
        print(f"  VQ Perplexity: {cond_result['diagnostics']['vq_perplexity_mean']:.2f}, Active Codes: {cond_result['diagnostics']['active_codes_mean']:.1f}/16")
        print(f"  Max |Spearman r|: {cond_result['correlations']['max_spearman_mean']:.3f}, Strong Monotonic Dims (|r|>0.5): {cond_result['correlations']['strong_monotonic_dims_mean']:.1f}/16")

    with open("aux_rul_experiment_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved all experimental results to aux_rul_experiment_results.json")


if __name__ == "__main__":
    run_experiment()
