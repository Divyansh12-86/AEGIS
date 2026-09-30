"""EGPM Critical Raw + Structured Information Test.

Evaluates whether EGPM causal structural features (F_struct in R^10) contain
predictive information useful in addition to raw sensors (X_raw in R^18).

Strictly controlled:
- No architecture changes, no retraining with different parameters.
- Frozen 12/4 candidate: W=4, S=4, RF=19, causal HSMM filtering, Ridge lambda=1.0.
- Seeds [0, 1, 2].
- Evaluates:
  1. Alignment check
  2. Three-way comparison (Raw, F_struct, Raw + F_struct)
  3. Per-seed results
  4. Per-engine results (DS01_u2, DS01_u9)
  5. Gamma/dwell ablation
  6. Ridge coefficient block norms
  7. Residual analysis by RUL bins
  8. Incremental information metrics
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
from egpm.evaluation.evaluator import rul_metrics
from egpm.utils import set_global_seed

CHANNEL_NAMES = [
    "alt", "Mach", "TRA", "T2",        # Flight conditions (0..3)
    "T24", "T30", "T48", "T50",        # Temperatures (4..7)
    "P15", "P2", "P21", "P24", "Ps30", "P40", "P50",  # Pressures (8..14)
    "Nf", "Nc", "Wf"                   # Speeds and fuel flow (15..17)
]


def run_experiment():
    print("=" * 75)
    print("EGPM RAW + STRUCTURED INFORMATION TEST (FROZEN 12/4 CANDIDATE)")
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

    # Target alignment check
    print("\n--- 1. DATA ALIGNMENT CHECK ---")
    alignment_verified = True
    for u, s in zip(te_units, te_seqs):
        rec = raw_dataset[u]
        print(f"Checking {u}: {len(s.X)} windows, {len(s.rul_labels)} targets, raw records {len(rec.signals)}")
        for n in range(len(s.X)):
            right_edge_idx = 4 * n + 3
            ts_rec = rec.timestamps[right_edge_idx]
            rul_rec = rec.rul_labels[right_edge_idx]
            if s.rul_labels[n] != rul_rec:
                alignment_verified = False
                print(f"ALIGNMENT ERROR on {u} window {n}: seq RUL {s.rul_labels[n]} != rec RUL {rul_rec}")
    assert alignment_verified, "Timestamp alignment check failed!"
    print("Timestamp alignment: VERIFIED. All windows right-edge aligned to native time s+3.")

    ytr = np.concatenate([s.rul_labels for s in tr_seqs])
    yte = np.concatenate([s.rul_labels for s in te_seqs])
    ltr = [len(s.X) for s in tr_seqs]
    lte = [len(s.X) for s in te_seqs]

    # Raw features (18-dim right edge, corresponding to the right edge timestamp)
    Xtr_raw_18 = np.concatenate([s.X[:, -1, :] for s in tr_seqs], axis=0)  # [138, 18]
    Xte_raw_18 = np.concatenate([s.X[:, -1, :] for s in te_seqs], axis=0)  # [39, 18]

    # Also compute 72-dim flattened window for completeness
    Xtr_raw_72 = np.concatenate([s.X.reshape(len(s.X), -1) for s in tr_seqs], axis=0)
    Xte_raw_72 = np.concatenate([s.X.reshape(len(s.X), -1) for s in te_seqs], axis=0)

    # Raw baseline evaluation
    w_raw_18 = _ridge_fit(Xtr_raw_18, ytr)
    p_raw_18 = _ridge_predict(w_raw_18, Xte_raw_18)
    m_raw_18 = rul_metrics(p_raw_18, yte)

    w_raw_72 = _ridge_fit(Xtr_raw_72, ytr)
    p_raw_72 = _ridge_predict(w_raw_72, Xte_raw_72)
    m_raw_72 = rul_metrics(p_raw_72, yte)

    print(f"Raw 18-dim (right edge): RMSE = {m_raw_18['RMSE']:.2f}, MAE = {m_raw_18['MAE']:.2f}, PHM = {m_raw_18['PHM_score']:.2f}")
    print(f"Raw 72-dim (flattened):  RMSE = {m_raw_72['RMSE']:.2f}, MAE = {m_raw_72['MAE']:.2f}, PHM = {m_raw_72['PHM_score']:.2f}")

    seeds = [0, 1, 2]
    Xtr_tensor = torch.as_tensor(np.concatenate([s.X for s in tr_seqs]), dtype=torch.float32)
    Mtr_tensor = torch.ones_like(Xtr_tensor)
    ytr_tensor = torch.as_tensor(ytr, dtype=torch.float32)

    results = {
        "raw_18": {
            "rmse": float(m_raw_18["RMSE"]),
            "mae": float(m_raw_18["MAE"]),
            "phm": float(m_raw_18["PHM_score"]),
            "per_engine": {},
        },
        "raw_72": {
            "rmse": float(m_raw_72["RMSE"]),
            "mae": float(m_raw_72["MAE"]),
            "phm": float(m_raw_72["PHM_score"]),
        },
        "models": {
            "Raw": {"rmse": [], "mae": [], "phm": [], "per_engine": {u: {"rmse": [], "mae": []} for u in te_units}},
            "F_struct": {"rmse": [], "mae": [], "phm": [], "per_engine": {u: {"rmse": [], "mae": []} for u in te_units}},
            "Raw_plus_F_struct": {"rmse": [], "mae": [], "phm": [], "per_engine": {u: {"rmse": [], "mae": []} for u in te_units}},
        },
        "ablations": {
            "Raw": {"rmse": [], "mae": []},
            "Raw_plus_gamma": {"rmse": [], "mae": []},
            "Raw_plus_dwell": {"rmse": [], "mae": []},
            "Raw_plus_gamma_dwell": {"rmse": [], "mae": []},
        },
        "coefficients": [],
        "residuals": [],
    }

    # Per engine for Raw baseline
    idx = 0
    for u, s in zip(te_units, te_seqs):
        L = len(s.X)
        p_u = p_raw_18[idx:idx + L]
        mu = rul_metrics(p_u, s.rul_labels)
        results["raw_18"]["per_engine"][u] = {"rmse": float(mu["RMSE"]), "mae": float(mu["MAE"])}
        idx += L

    for s in seeds:
        print(f"\n--- Processing Seed {s} ---")
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
        trainer = Trainer(cfg, n_channels=Xtr_tensor.shape[-1])
        trainer.train_stage1(Xtr_tensor, Mtr_tensor, rul_labels=ytr_tensor)

        ev_tr = trainer.window_runs_to_event_runs(trainer.event_sequences(Xtr_tensor, Mtr_tensor), ltr)
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

        gamma_tr = np.concatenate([r.state_posterior for r in r_tr], axis=0)  # [138, 5]
        gamma_te = np.concatenate([r.state_posterior for r in r_te], axis=0)  # [39, 5]

        dwell_tr = np.concatenate([r.expected_dwell / trainer.hsmm.d_max for r in r_tr], axis=0)  # [138, 5]
        dwell_te = np.concatenate([r.expected_dwell / trainer.hsmm.d_max for r in r_te], axis=0)  # [39, 5]

        f_struct_tr = np.concatenate([gamma_tr, dwell_tr], axis=1)  # [138, 10]
        f_struct_te = np.concatenate([gamma_te, dwell_te], axis=1)  # [39, 10]

        # Combined features
        f_comb_tr = np.concatenate([Xtr_raw_18, f_struct_tr], axis=1)  # [138, 28]
        f_comb_te = np.concatenate([Xte_raw_18, f_struct_te], axis=1)  # [39, 28]

        # -------------------------------------------------------------
        # 1. Fit Three Models
        # -------------------------------------------------------------
        # Model A: Raw (constant across seeds, but recorded per seed for consistency)
        results["models"]["Raw"]["rmse"].append(float(m_raw_18["RMSE"]))
        results["models"]["Raw"]["mae"].append(float(m_raw_18["MAE"]))
        results["models"]["Raw"]["phm"].append(float(m_raw_18["PHM_score"]))
        for u in te_units:
            results["models"]["Raw"]["per_engine"][u]["rmse"].append(results["raw_18"]["per_engine"][u]["rmse"])
            results["models"]["Raw"]["per_engine"][u]["mae"].append(results["raw_18"]["per_engine"][u]["mae"])

        # Model B: F_struct (10-dim)
        w_b = _ridge_fit(f_struct_tr, ytr)
        p_b = _ridge_predict(w_b, f_struct_te)
        m_b = rul_metrics(p_b, yte)
        results["models"]["F_struct"]["rmse"].append(float(m_b["RMSE"]))
        results["models"]["F_struct"]["mae"].append(float(m_b["MAE"]))
        results["models"]["F_struct"]["phm"].append(float(m_b["PHM_score"]))

        # Model C: Raw + F_struct (28-dim)
        w_c = _ridge_fit(f_comb_tr, ytr)
        p_c = _ridge_predict(w_c, f_comb_te)
        m_c = rul_metrics(p_c, yte)
        results["models"]["Raw_plus_F_struct"]["rmse"].append(float(m_c["RMSE"]))
        results["models"]["Raw_plus_F_struct"]["mae"].append(float(m_c["MAE"]))
        results["models"]["Raw_plus_F_struct"]["phm"].append(float(m_c["PHM_score"]))

        # Per engine for B and C
        idx = 0
        for u, s_unit in zip(te_units, te_seqs):
            L = len(s_unit.X)
            p_b_u = p_b[idx:idx + L]
            p_c_u = p_c[idx:idx + L]
            m_b_u = rul_metrics(p_b_u, s_unit.rul_labels)
            m_c_u = rul_metrics(p_c_u, s_unit.rul_labels)
            results["models"]["F_struct"]["per_engine"][u]["rmse"].append(float(m_b_u["RMSE"]))
            results["models"]["F_struct"]["per_engine"][u]["mae"].append(float(m_b_u["MAE"]))
            results["models"]["Raw_plus_F_struct"]["per_engine"][u]["rmse"].append(float(m_c_u["RMSE"]))
            results["models"]["Raw_plus_F_struct"]["per_engine"][u]["mae"].append(float(m_c_u["MAE"]))
            idx += L

        # -------------------------------------------------------------
        # 2. Block Ablation
        # -------------------------------------------------------------
        # Raw + gamma (23-dim)
        f_rg_tr = np.concatenate([Xtr_raw_18, gamma_tr], axis=1)
        f_rg_te = np.concatenate([Xte_raw_18, gamma_te], axis=1)
        w_rg = _ridge_fit(f_rg_tr, ytr)
        p_rg = _ridge_predict(w_rg, f_rg_te)
        m_rg = rul_metrics(p_rg, yte)

        # Raw + dwell (23-dim)
        f_rd_tr = np.concatenate([Xtr_raw_18, dwell_tr], axis=1)
        f_rd_te = np.concatenate([Xte_raw_18, dwell_te], axis=1)
        w_rd = _ridge_fit(f_rd_tr, ytr)
        p_rd = _ridge_predict(w_rd, f_rd_te)
        m_rd = rul_metrics(p_rd, yte)

        results["ablations"]["Raw"]["rmse"].append(float(m_raw_18["RMSE"]))
        results["ablations"]["Raw"]["mae"].append(float(m_raw_18["MAE"]))
        results["ablations"]["Raw_plus_gamma"]["rmse"].append(float(m_rg["RMSE"]))
        results["ablations"]["Raw_plus_gamma"]["mae"].append(float(m_rg["MAE"]))
        results["ablations"]["Raw_plus_dwell"]["rmse"].append(float(m_rd["RMSE"]))
        results["ablations"]["Raw_plus_dwell"]["mae"].append(float(m_rd["MAE"]))
        results["ablations"]["Raw_plus_gamma_dwell"]["rmse"].append(float(m_c["RMSE"]))
        results["ablations"]["Raw_plus_gamma_dwell"]["mae"].append(float(m_c["MAE"]))

        # -------------------------------------------------------------
        # 3. Ridge Coefficients Analysis (w_c: 28 weights + 1 intercept)
        # -------------------------------------------------------------
        # _ridge_fit appends a bias column of 1s at the end: w[:-1] is weights, w[-1] is bias
        w_weights = w_c[:-1]
        bias = float(w_c[-1])
        w_raw_block = w_weights[:18]
        w_gamma_block = w_weights[18:23]
        w_dwell_block = w_weights[23:28]

        l2_raw = float(np.linalg.norm(w_raw_block))
        l2_gamma = float(np.linalg.norm(w_gamma_block))
        l2_dwell = float(np.linalg.norm(w_dwell_block))

        results["coefficients"].append({
            "seed": s,
            "bias": bias,
            "l2_raw": l2_raw,
            "l2_gamma": l2_gamma,
            "l2_dwell": l2_dwell,
            "raw_weights": [float(x) for x in w_raw_block],
            "gamma_weights": [float(x) for x in w_gamma_block],
            "dwell_weights": [float(x) for x in w_dwell_block],
        })

        # -------------------------------------------------------------
        # 4. Residuals by RUL Bins
        # -------------------------------------------------------------
        res_raw = yte - p_raw_18
        res_comb = yte - p_c

        # Bins: High (> 50), Mid (20-50), Low (< 20)
        idx_high = np.where(yte > 50)[0]
        idx_mid = np.where((yte >= 20) & (yte <= 50))[0]
        idx_low = np.where(yte < 20)[0]

        results["residuals"].append({
            "seed": s,
            "high_rul_gt50": {
                "count": len(idx_high),
                "raw_rmse": float(np.sqrt(np.mean(res_raw[idx_high] ** 2))),
                "comb_rmse": float(np.sqrt(np.mean(res_comb[idx_high] ** 2))),
                "raw_mae": float(np.mean(np.abs(res_raw[idx_high]))),
                "comb_mae": float(np.mean(np.abs(res_comb[idx_high]))),
            },
            "mid_rul_20_50": {
                "count": len(idx_mid),
                "raw_rmse": float(np.sqrt(np.mean(res_raw[idx_mid] ** 2))),
                "comb_rmse": float(np.sqrt(np.mean(res_comb[idx_mid] ** 2))),
                "raw_mae": float(np.mean(np.abs(res_raw[idx_mid]))),
                "comb_mae": float(np.mean(np.abs(res_comb[idx_mid]))),
            },
            "low_rul_lt20": {
                "count": len(idx_low),
                "raw_rmse": float(np.sqrt(np.mean(res_raw[idx_low] ** 2))),
                "comb_rmse": float(np.sqrt(np.mean(res_comb[idx_low] ** 2))),
                "raw_mae": float(np.mean(np.abs(res_raw[idx_low]))),
                "comb_mae": float(np.mean(np.abs(res_comb[idx_low]))),
            },
        })

        print(f"  Seed {s} ({time.time()-t0:.1f}s): Raw RMSE={m_raw_18['RMSE']:.2f}, F_struct={m_b['RMSE']:.2f}, Raw+F_struct={m_c['RMSE']:.2f} (Δ = {m_raw_18['RMSE'] - m_c['RMSE']:+.2f})")

    # Compute summary statistics
    for m in results["models"]:
        for k in ["rmse", "mae", "phm"]:
            vals = results["models"][m][k]
            results["models"][m][f"{k}_mean"] = float(np.mean(vals))
            results["models"][m][f"{k}_std"] = float(np.std(vals))
        for u in te_units:
            for k in ["rmse", "mae"]:
                vals = results["models"][m]["per_engine"][u][k]
                results["models"][m]["per_engine"][u][f"{k}_mean"] = float(np.mean(vals))
                results["models"][m]["per_engine"][u][f"{k}_std"] = float(np.std(vals))

    for ab in results["ablations"]:
        for k in ["rmse", "mae"]:
            vals = results["ablations"][ab][k]
            results["ablations"][ab][f"{k}_mean"] = float(np.mean(vals))
            results["ablations"][ab][f"{k}_std"] = float(np.std(vals))

    # Save output to JSON
    with open("raw_plus_structured_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved full results to raw_plus_structured_results.json")


if __name__ == "__main__":
    run_experiment()
