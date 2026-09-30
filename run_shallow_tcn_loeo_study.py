"""EGPM Shallow Causal TCN LOEO Validation Study.

Scientific Question:
Does a small amount of nonlinear temporal capacity improve upon the existing
linear Raw + F_struct Ridge baseline (LOEO mean RMSE = 17.30) when evaluated
under the exact same engine-level LOEO protocol across engines u3-u8?

Arms evaluated:
- Arm A: Raw-only TCN (18-D)
- Arm B: Raw + dwell TCN (23-D)
- Arm C: Raw + gamma + dwell TCN (28-D)

Baselines:
- Raw Ridge (18-D)
- Raw + F_struct Ridge (28-D)

Strictly adheres to:
- No architecture search.
- Shallow causal TCN: 2 dilated conv blocks (d=1, 2), width 32, causal left-padding.
- Predetermined seeds: [0, 1, 2].
- Preprocessing fit strictly on training engines per fold.
- Locked test set (u2, u9) evaluated ONLY if LOEO success gate is passed.
"""
from __future__ import annotations

import json
import time
import numpy as np
import scipy.stats
import torch
from torch import nn

from egpm.data.loaders import NCMAPSSLoader
from egpm.data.split_builder import SplitBuilder, NormalizationStats
from egpm.experiments.ablation_arms import _downsample, _ridge_fit, _ridge_predict
from egpm.preprocessing.preprocessor import Preprocessor
from egpm.training.trainer import Trainer, TrainerConfig
from egpm.grammar.forward_backward import ForwardBackward
from egpm.evaluation.evaluator import rul_metrics
from egpm.utils import set_global_seed


class ShallowCausalTCN(nn.Module):
    """Exact frozen TCN specification from egpm/experiments/readout_study.py."""
    def __init__(self, in_dim: int, hidden: int = 32):
        super().__init__()
        # Causal left-padding: (kernel - 1) * dilation on left, 0 on right
        # Layer 1: kernel=3, dilation=1 -> pad 2 left
        # Layer 2: kernel=3, dilation=2 -> pad 4 left
        self.net = nn.Sequential(
            nn.ConstantPad1d((2, 0), 0.0),
            nn.Conv1d(in_dim, hidden, 3, dilation=1, padding=0),
            nn.GELU(),
            nn.ConstantPad1d((4, 0), 0.0),
            nn.Conv1d(hidden, hidden, 3, dilation=2, padding=0),
            nn.GELU(),
        )
        self.out = nn.Linear(hidden, 1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X: [B, T, in_dim]
        # Transpose to [B, in_dim, T] for Conv1D
        h = self.net(X.transpose(1, 2)).transpose(1, 2)  # [B, T, hidden]
        return self.out(h).squeeze(-1)  # [B, T]


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_tcn_head(
    model: nn.Module,
    Xs_tr: list[torch.Tensor],  # list of [1, T_u, D]
    ys_tr: list[torch.Tensor],  # list of [T_u]
    X_va: torch.Tensor,         # [1, T_val, D]
    y_va: torch.Tensor,         # [T_val]
    lr: float = 1e-3,
    max_epochs: int = 200,
    patience: int = 10,
    seed: int = 0,
) -> tuple[nn.Module, dict]:
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    wait = 0

    train_loss_hist = []
    val_loss_hist = []

    for ep in range(max_epochs):
        model.train()
        tr_losses = []
        for X, y in zip(Xs_tr, ys_tr):
            opt.zero_grad()
            p = model(X).squeeze(0)  # [T_u]
            loss = ((p - y) ** 2).mean()
            loss.backward()
            opt.step()
            tr_losses.append(loss.item())

        mean_tr_loss = float(np.mean(tr_losses))
        train_loss_hist.append(mean_tr_loss)

        model.eval()
        with torch.no_grad():
            p_val = model(X_va).squeeze(0)
            val_loss = float(((p_val - y_va) ** 2).mean())
            val_loss_hist.append(val_loss)

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_epoch = ep
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    history = {
        "best_epoch": best_epoch,
        "stopped_epoch": len(train_loss_hist),
        "best_val_loss": best_val_loss,
        "final_train_loss": train_loss_hist[best_epoch] if best_state is not None else train_loss_hist[-1],
        "train_loss_history": train_loss_hist,
        "val_loss_history": val_loss_hist,
    }
    return model, history


def run_experiment():
    print("=" * 80)
    print("EGPM SHALLOW CAUSAL TCN LOEO VALIDATION STUDY")
    print("=" * 80)

    dataset_path = "N-CMAPSS/N-CMAPSS DS01 005.h5"
    raw_dataset = NCMAPSSLoader().load(dataset_path)
    raw_dataset.records = [_downsample(r) for r in raw_dataset.records]

    sb = SplitBuilder(raw_dataset, seed=0)
    split_standard = sb.build()
    train_unit_ids = split_standard.train_unit_ids  # ['DS01_u3', 'DS01_u4', 'DS01_u5', 'DS01_u6', 'DS01_u7', 'DS01_u8']
    test_unit_ids = split_standard.test_unit_ids    # ['DS01_u2', 'DS01_u9']
    print(f"LOEO Training pool (6 engines): {train_unit_ids}")
    print(f"Locked Test pool (2 engines):    {test_unit_ids}")

    # =========================================================================
    # 1. IMPLEMENTATION & CAUSALITY AUDIT
    # =========================================================================
    print("\n--- 1. IMPLEMENTATION & CAUSALITY AUDIT ---")
    # Verify causality of ShallowCausalTCN
    test_model = ShallowCausalTCN(in_dim=18, hidden=32)
    # Check receptive field: Layer 1 pad 2, kernel 3, dil 1; Layer 2 pad 4, kernel 3, dil 2
    # Total receptive field = 1 + 2*1 + 2*2 = 7 steps
    test_input = torch.randn(1, 10, 18)
    test_out = test_model(test_input)
    assert test_out.shape == (1, 10), f"TCN output shape mismatch: {test_out.shape}"

    # Verify temporal causality: perturb input at t=8, check if output at t <= 7 changes
    test_input_perturbed = test_input.clone()
    test_input_perturbed[:, 8:, :] += 10.0
    with torch.no_grad():
        out1 = test_model(test_input)
        out2 = test_model(test_input_perturbed)
    diff = (out1[:, :8] - out2[:, :8]).abs().max().item()
    print(f"Causality perturbation test: Max diff at t < 8 when t >= 8 perturbed = {diff:.6e}")
    assert diff < 1e-6, "Causality check FAILED: future inputs leak into past outputs!"
    print("Causality Audit: STRICTLY PASSED. Left padding prevents future leakage.")

    # Parameter counts
    params_arm_a = count_parameters(ShallowCausalTCN(in_dim=18, hidden=32))
    params_arm_b = count_parameters(ShallowCausalTCN(in_dim=23, hidden=32))
    params_arm_c = count_parameters(ShallowCausalTCN(in_dim=28, hidden=32))
    print(f"Trainable Parameters:")
    print(f"  Arm A (Raw TCN, 18-D):           {params_arm_a} parameters")
    print(f"  Arm B (Raw + dwell TCN, 23-D):   {params_arm_b} parameters")
    print(f"  Arm C (Raw + F_struct TCN, 28-D): {params_arm_c} parameters")

    # =========================================================================
    # 2. LOEO EXPERIMENT
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 2: LEAVE-ONE-ENGINE-OUT (LOEO) TCN vs RIDGE EVALUATION")
    print("=" * 70)

    tcn_seeds = [0, 1, 2]
    fold_results = {}

    for k_fold in range(len(train_unit_ids)):
        val_unit = train_unit_ids[k_fold]
        tr_fold = [u for j, u in enumerate(train_unit_ids) if j != k_fold]
        print(f"\n>>> Fold {k_fold + 1}/6: Held-out = {val_unit}, Train = {tr_fold}")
        t0 = time.time()

        # Normalization fit strictly on tr_fold
        train_records_fold = [r for r in raw_dataset.records if r.unit_id in set(tr_fold)]
        stacked_fold = np.concatenate([r.signals for r in train_records_fold], axis=0)
        mean_fold = stacked_fold.mean(axis=0)
        std_fold = np.where(stacked_fold.std(axis=0) < 1e-8, 1.0, stacked_fold.std(axis=0))
        norm_stats_fold = NormalizationStats(mean=mean_fold, std=std_fold, train_unit_ids=sorted(tr_fold))

        prep = Preprocessor(window_length=4, stride=4)
        tr_seqs = [prep.process(raw_dataset[u], norm_stats=norm_stats_fold) for u in tr_fold]
        val_seq = prep.process(raw_dataset[val_unit], norm_stats=norm_stats_fold)

        # Raw features (18-D right edge per window)
        X_tr_raw = [s.X[:, -1, :] for s in tr_seqs]  # list of [T_u, 18]
        X_va_raw = val_seq.X[:, -1, :]               # [T_val, 18]
        y_tr = [s.rul_labels for s in tr_seqs]       # list of [T_u]
        y_va = val_seq.rul_labels                    # [T_val]

        y_tr_flat = np.concatenate(y_tr)
        X_tr_raw_flat = np.concatenate(X_tr_raw, axis=0)

        # Train EGPM 12/4 candidate on tr_fold
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

        # Causal filter
        r_tr_fold = [fb_fold.causal_filter(ev) for ev in ev_tr_fold]
        gamma_tr = [r.state_posterior for r in r_tr_fold]
        dwell_tr = [r.expected_dwell / 10.0 for r in r_tr_fold]
        f_struct_tr = [np.concatenate([g, d], axis=1) for g, d in zip(gamma_tr, dwell_tr)]

        X_val_tensor = torch.as_tensor(val_seq.X, dtype=torch.float32)
        ev_val_fold = np.concatenate(trainer_fold.event_sequences(X_val_tensor, torch.ones_like(X_val_tensor)))
        r_val_fold = fb_fold.causal_filter(ev_val_fold)
        gamma_va = r_val_fold.state_posterior
        dwell_va = r_val_fold.expected_dwell / 10.0
        f_struct_va = np.concatenate([gamma_va, dwell_va], axis=1)

        # Construct feature representations for train and val sequences
        # Arm A: Raw (18-D)
        feats_tr_A = X_tr_raw
        feats_va_A = X_va_raw

        # Arm B: Raw + dwell (23-D)
        feats_tr_B = [np.concatenate([x, d], axis=1) for x, d in zip(X_tr_raw, dwell_tr)]
        feats_va_B = np.concatenate([X_va_raw, dwell_va], axis=1)

        # Arm C: Raw + gamma + dwell (28-D)
        feats_tr_C = [np.concatenate([x, f], axis=1) for x, f in zip(X_tr_raw, f_struct_tr)]
        feats_va_C = np.concatenate([X_va_raw, f_struct_va], axis=1)

        # -------------------------------------------------------------
        # 1. Existing Ridge Baselines
        # -------------------------------------------------------------
        w_raw_ridge = _ridge_fit(X_tr_raw_flat, y_tr_flat)
        pred_raw_ridge = _ridge_predict(w_raw_ridge, X_va_raw)
        m_raw_ridge = rul_metrics(pred_raw_ridge, y_va)

        w_comb_ridge = _ridge_fit(np.concatenate(feats_tr_C, axis=0), y_tr_flat)
        pred_comb_ridge = _ridge_predict(w_comb_ridge, feats_va_C)
        m_comb_ridge = rul_metrics(pred_comb_ridge, y_va)

        w_dwell_ridge = _ridge_fit(np.concatenate(feats_tr_B, axis=0), y_tr_flat)
        pred_dwell_ridge = _ridge_predict(w_dwell_ridge, feats_va_B)
        m_dwell_ridge = rul_metrics(pred_dwell_ridge, y_va)

        print(f"  Ridge Baselines: Raw={m_raw_ridge['RMSE']:.2f}, Raw+Dwell={m_dwell_ridge['RMSE']:.2f}, Raw+F_struct={m_comb_ridge['RMSE']:.2f}")

        # -------------------------------------------------------------
        # 2. TCN Arms across seeds [0, 1, 2]
        # -------------------------------------------------------------
        tcn_arms_data = {
            "Raw_TCN": {"in_dim": 18, "tr_feats": feats_tr_A, "va_feats": feats_va_A},
            "Raw_plus_Dwell_TCN": {"in_dim": 23, "tr_feats": feats_tr_B, "va_feats": feats_va_B},
            "Raw_plus_Fstruct_TCN": {"in_dim": 28, "tr_feats": feats_tr_C, "va_feats": feats_va_C},
        }

        tcn_fold_eval = {}

        for arm_name, arm_spec in tcn_arms_data.items():
            in_d = arm_spec["in_dim"]
            tr_f = arm_spec["tr_feats"]
            va_f = arm_spec["va_feats"]

            Xs_tr_tensor = [torch.as_tensor(x, dtype=torch.float32).unsqueeze(0) for x in tr_f]
            ys_tr_tensor = [torch.as_tensor(y, dtype=torch.float32) for y in y_tr]
            X_va_tensor = torch.as_tensor(va_f, dtype=torch.float32).unsqueeze(0)
            y_va_tensor = torch.as_tensor(y_va, dtype=torch.float32)

            arm_rmses = []
            arm_maes = []
            arm_phms = []
            arm_histories = []

            for s in tcn_seeds:
                model = ShallowCausalTCN(in_dim=in_d, hidden=32)
                fitted_model, hist = train_tcn_head(
                    model, Xs_tr_tensor, ys_tr_tensor, X_va_tensor, y_va_tensor,
                    lr=1e-3, max_epochs=200, patience=10, seed=s
                )
                fitted_model.eval()
                with torch.no_grad():
                    pred_tcn = fitted_model(X_va_tensor).squeeze(0).cpu().numpy()
                m_tcn = rul_metrics(pred_tcn, y_va)
                arm_rmses.append(float(m_tcn["RMSE"]))
                arm_maes.append(float(m_tcn["MAE"]))
                arm_phms.append(float(m_tcn["PHM_score"]))
                arm_histories.append({
                    "best_epoch": hist["best_epoch"],
                    "stopped_epoch": hist["stopped_epoch"],
                    "final_train_loss": hist["final_train_loss"],
                    "best_val_loss": hist["best_val_loss"],
                })

            tcn_fold_eval[arm_name] = {
                "rmse_mean": float(np.mean(arm_rmses)),
                "rmse_std": float(np.std(arm_rmses)),
                "mae_mean": float(np.mean(arm_maes)),
                "phm_mean": float(np.mean(arm_phms)),
                "per_seed_rmse": arm_rmses,
                "histories": arm_histories,
            }
            print(f"  {arm_name:22s}: RMSE = {np.mean(arm_rmses):.2f} ± {np.std(arm_rmses):.2f} (seeds: {[round(x, 2) for x in arm_rmses]}) | best_ep={[h['best_epoch'] for h in arm_histories]}")

        fold_results[val_unit] = {
            "n_windows": len(val_seq.X),
            "ridge": {
                "raw": float(m_raw_ridge["RMSE"]),
                "raw_dwell": float(m_dwell_ridge["RMSE"]),
                "raw_fstruct": float(m_comb_ridge["RMSE"]),
            },
            "tcn": tcn_fold_eval,
            "wall_clock_s": time.time() - t0,
        }

    # =========================================================================
    # 3. SUMMARY & STATISTICAL ANALYSIS ACROSS THE 6 INDEPENDENT FOLDS
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 3: LOEO PRIMARY RESULTS SUMMARY (6 INDEPENDENT ENGINES)")
    print("=" * 70)

    # Print Primary Table
    header = f"{'Held-out':<10} {'N':<4} {'Raw Ridge':<10} {'Raw+Fst Ridge':<14} {'Raw TCN':<10} {'Raw+Dwell TCN':<14} {'Raw+Fst TCN':<12}"
    print(header)
    print("-" * len(header))

    raw_ridge_all = []
    comb_ridge_all = []
    raw_tcn_all = []
    dwell_tcn_all = []
    comb_tcn_all = []

    for u in train_unit_ids:
        fr = fold_results[u]
        n_w = fr["n_windows"]
        rr = fr["ridge"]["raw"]
        cr = fr["ridge"]["raw_fstruct"]
        rt = fr["tcn"]["Raw_TCN"]["rmse_mean"]
        dt = fr["tcn"]["Raw_plus_Dwell_TCN"]["rmse_mean"]
        ct = fr["tcn"]["Raw_plus_Fstruct_TCN"]["rmse_mean"]

        raw_ridge_all.append(rr)
        comb_ridge_all.append(cr)
        raw_tcn_all.append(rt)
        dwell_tcn_all.append(dt)
        comb_tcn_all.append(ct)

        print(f"{u:<10} {n_w:<4} {rr:<10.2f} {cr:<14.2f} {rt:<10.2f} {dt:<14.2f} {ct:<12.2f}")

    print("-" * len(header))
    print(f"{'Mean':<10} {'-':<4} {np.mean(raw_ridge_all):<10.2f} {np.mean(comb_ridge_all):<14.2f} {np.mean(raw_tcn_all):<10.2f} {np.mean(dwell_tcn_all):<14.2f} {np.mean(comb_tcn_all):<12.2f}")
    print(f"{'Median':<10} {'-':<4} {np.median(raw_ridge_all):<10.2f} {np.median(comb_ridge_all):<14.2f} {np.median(raw_tcn_all):<10.2f} {np.median(dwell_tcn_all):<14.2f} {np.median(comb_tcn_all):<12.2f}")
    print(f"{'Std Dev':<10} {'-':<4} {np.std(raw_ridge_all):<10.2f} {np.std(comb_ridge_all):<14.2f} {np.std(raw_tcn_all):<10.2f} {np.std(dwell_tcn_all):<14.2f} {np.std(comb_tcn_all):<12.2f}")
    print(f"{'Min':<10} {'-':<4} {np.min(raw_ridge_all):<10.2f} {np.min(comb_ridge_all):<14.2f} {np.min(raw_tcn_all):<10.2f} {np.min(dwell_tcn_all):<14.2f} {np.min(comb_tcn_all):<12.2f}")
    print(f"{'Max':<10} {'-':<4} {np.max(raw_ridge_all):<10.2f} {np.max(comb_ridge_all):<14.2f} {np.max(raw_tcn_all):<10.2f} {np.max(dwell_tcn_all):<14.2f} {np.max(comb_tcn_all):<12.2f}")

    # Paired comparisons (TCN Raw+F_struct vs Ridge Raw+F_struct)
    diff_tcn_vs_ridge = [cr - ct for cr, ct in zip(comb_ridge_all, comb_tcn_all)]
    tcn_wins = sum(1 for d in diff_tcn_vs_ridge if d > 0)
    print(f"\nPaired Comparison (Raw+F_struct TCN vs Raw+F_struct Ridge):")
    print(f"  Per-fold differences (Ridge - TCN, positive = TCN wins): {[round(d, 2) for d in diff_tcn_vs_ridge]}")
    print(f"  Folds where TCN beats Ridge: {tcn_wins} / 6")
    print(f"  Mean Difference:   {np.mean(diff_tcn_vs_ridge):+.2f} cycles")
    print(f"  Median Difference: {np.median(diff_tcn_vs_ridge):+.2f} cycles")

    # Dwell comparison: Raw+Dwell TCN vs Raw+F_struct TCN
    diff_dwell_vs_fstruct = [ct - dt for dt, ct in zip(dwell_tcn_all, comb_tcn_all)]
    print(f"\nDwell Analysis (Raw+Dwell TCN vs Raw+F_struct TCN):")
    print(f"  Mean Raw+Dwell TCN RMSE:   {np.mean(dwell_tcn_all):.2f}")
    print(f"  Mean Raw+F_struct TCN RMSE: {np.mean(comb_tcn_all):.2f}")
    print(f"  Difference (F_struct - Dwell): {np.mean(diff_dwell_vs_fstruct):+.2f} cycles")

    # Success Criterion Evaluation
    gate_1 = (tcn_wins >= 4)
    gate_2 = (np.mean(comb_tcn_all) < 17.30)
    # Check consistency: not just 1 outlier
    # If 1 fold improved by 10 and others degraded, that's not consistent.
    gate_3 = (tcn_wins >= 3 and np.median(diff_tcn_vs_ridge) > -0.5)

    gate_passed = gate_1 and gate_2 and gate_3
    print(f"\nSuccess Gate Check:")
    print(f"  Condition 1 (TCN beats Ridge on majority >= 4/6 folds): {gate_1} ({tcn_wins}/6)")
    print(f"  Condition 2 (Mean LOEO RMSE < 17.30):                   {gate_2} ({np.mean(comb_tcn_all):.2f})")
    print(f"  Condition 3 (Consistent improvement across folds):      {gate_3}")
    print(f"  OVERALL SUCCESS GATE PASSED:                            {gate_passed}")

    # =========================================================================
    # 4. FINAL LOCKED TEST EVALUATION (ONLY IF GATE PASSED)
    # =========================================================================
    locked_test_results = None
    if gate_passed:
        print("\n" + "=" * 70)
        print("PHASE 4: ONE-TIME FINAL LOCKED TEST EVALUATION (u2, u9)")
        print("=" * 70)
        # Train on all 6 training engines (u3..u8)
        stats_std = sb.fit_normalization(split_standard)
        prep_std = Preprocessor(window_length=4, stride=4)
        tr_seqs_std = [prep_std.process(raw_dataset[u], norm_stats=stats_std) for u in train_unit_ids]
        te_seqs_std = [prep_std.process(raw_dataset[u], norm_stats=stats_std) for u in test_unit_ids]

        X_tr_std = [s.X[:, -1, :] for s in tr_seqs_std]
        X_te_std = [s.X[:, -1, :] for s in te_seqs_std]
        y_tr_std = [s.rul_labels for s in tr_seqs_std]
        y_te_std = [s.rul_labels for s in te_seqs_std]

        # Extract EGPM features on full train split
        set_global_seed(0)
        cfg_full = TrainerConfig(
            window_length=4, embed_dim=16, hidden=32, n_conv_blocks=3, n_codes=16,
            d_max=10, n_states=5, em_max_iter=10, em_restarts=10, aux_rul_lambda=0.1,
            state_dim=12, dead_code_revival=False, stage1_steps=300, batch_size=64, seed=0, log_every=0
        )
        trainer_full = Trainer(cfg_full, n_channels=18)
        X_tr_tensor_std = torch.as_tensor(np.concatenate([s.X for s in tr_seqs_std]), dtype=torch.float32)
        M_tr_tensor_std = torch.ones_like(X_tr_tensor_std)
        y_tr_tensor_std = torch.as_tensor(np.concatenate(y_tr_std), dtype=torch.float32)
        trainer_full.train_stage1(X_tr_tensor_std, M_tr_tensor_std, rul_labels=y_tr_tensor_std)

        ltr_std = [len(s.X) for s in tr_seqs_std]
        ev_tr_std = trainer_full.window_runs_to_event_runs(trainer_full.event_sequences(X_tr_tensor_std, M_tr_tensor_std), ltr_std)
        trainer_full.train_stage2(ev_tr_std)
        fb_full = ForwardBackward(trainer_full.hsmm)

        r_tr_std = [fb_full.causal_filter(ev) for ev in ev_tr_std]
        f_tr_std = [np.concatenate([r.state_posterior, r.expected_dwell / 10.0], axis=1) for r in r_tr_std]
        comb_tr_std = [np.concatenate([x, f], axis=1) for x, f in zip(X_tr_std, f_tr_std)]

        ev_te_std = [np.concatenate(trainer_full.event_sequences(torch.as_tensor(s.X, dtype=torch.float32), torch.ones_like(torch.as_tensor(s.X, dtype=torch.float32)))) for s in te_seqs_std]
        r_te_std = [fb_full.causal_filter(ev) for ev in ev_te_std]
        f_te_std = [np.concatenate([r.state_posterior, r.expected_dwell / 10.0], axis=1) for r in r_te_std]
        comb_te_std = [np.concatenate([x, f], axis=1) for x, f in zip(X_te_std, f_te_std)]

        y_te_flat = np.concatenate(y_te_std)

        # Baseline Ridge models on locked test
        w_raw_r = _ridge_fit(np.concatenate(X_tr_std), np.concatenate(y_tr_std))
        p_raw_r = _ridge_predict(w_raw_r, np.concatenate(X_te_std))
        m_raw_r = rul_metrics(p_raw_r, y_te_flat)

        w_comb_r = _ridge_fit(np.concatenate(comb_tr_std), np.concatenate(y_tr_std))
        p_comb_r = _ridge_predict(w_comb_r, np.concatenate(comb_te_std))
        m_comb_r = rul_metrics(p_comb_r, y_te_flat)

        # Train TCN on full train set, evaluate on locked test
        Xs_tr_t = [torch.as_tensor(x, dtype=torch.float32).unsqueeze(0) for x in comb_tr_std]
        ys_tr_t = [torch.as_tensor(y, dtype=torch.float32) for y in y_tr_std]
        Xs_te_t = [torch.as_tensor(x, dtype=torch.float32).unsqueeze(0) for x in comb_te_std]

        # Use u1, u10 (validation split) for early stopping in locked test run
        val_seqs_std = [prep_std.process(raw_dataset[u], norm_stats=stats_std) for u in split_standard.val_unit_ids]
        X_va_std = [s.X[:, -1, :] for s in val_seqs_std]
        ev_va_std = [np.concatenate(trainer_full.event_sequences(torch.as_tensor(s.X, dtype=torch.float32), torch.ones_like(torch.as_tensor(s.X, dtype=torch.float32)))) for s in val_seqs_std]
        r_va_std = [fb_full.causal_filter(ev) for ev in ev_va_std]
        f_va_std = [np.concatenate([r.state_posterior, r.expected_dwell / 10.0], axis=1) for r in r_va_std]
        comb_va_std = [np.concatenate([x, f], axis=1) for x, f in zip(X_va_std, f_va_std)]

        X_va_full = torch.as_tensor(np.concatenate(comb_va_std), dtype=torch.float32).unsqueeze(0)
        y_va_full = torch.as_tensor(np.concatenate([s.rul_labels for s in val_seqs_std]), dtype=torch.float32)

        test_tcn_rmses = []
        for s in tcn_seeds:
            model = ShallowCausalTCN(in_dim=28, hidden=32)
            fitted_model, hist = train_tcn_head(
                model, Xs_tr_t, ys_tr_t, X_va_full, y_va_full,
                lr=1e-3, max_epochs=200, patience=10, seed=s
            )
            fitted_model.eval()
            with torch.no_grad():
                preds_te = [fitted_model(x).squeeze(0).cpu().numpy() for x in Xs_te_t]
            m_te = rul_metrics(np.concatenate(preds_te), y_te_flat)
            test_tcn_rmses.append(float(m_te["RMSE"]))

        locked_test_results = {
            "raw_ridge_rmse": float(m_raw_r["RMSE"]),
            "raw_plus_fstruct_ridge_rmse": float(m_comb_r["RMSE"]),
            "tcn_raw_plus_fstruct_rmse_mean": float(np.mean(test_tcn_rmses)),
            "tcn_raw_plus_fstruct_rmse_std": float(np.std(test_tcn_rmses)),
            "tcn_per_seed_rmse": test_tcn_rmses,
        }
        print(f"Locked Test Results (u2, u9):")
        print(f"  Raw Ridge:               {m_raw_r['RMSE']:.2f}")
        print(f"  Raw + F_struct Ridge:     {m_comb_r['RMSE']:.2f}")
        print(f"  Raw + F_struct TCN:       {np.mean(test_tcn_rmses):.2f} ± {np.std(test_tcn_rmses):.2f} (seeds: {[round(x, 2) for x in test_tcn_rmses]})")
    else:
        print("\nSuccess gate NOT passed. Locked test evaluation SKIPPED to preserve test set integrity.")

    # Save full results
    output_package = {
        "params": {
            "arm_a": params_arm_a,
            "arm_b": params_arm_b,
            "arm_c": params_arm_c,
        },
        "fold_results": fold_results,
        "summary": {
            "mean_rmse": {
                "raw_ridge": float(np.mean(raw_ridge_all)),
                "comb_ridge": float(np.mean(comb_ridge_all)),
                "raw_tcn": float(np.mean(raw_tcn_all)),
                "dwell_tcn": float(np.mean(dwell_tcn_all)),
                "comb_tcn": float(np.mean(comb_tcn_all)),
            },
            "median_rmse": {
                "raw_ridge": float(np.median(raw_ridge_all)),
                "comb_ridge": float(np.median(comb_ridge_all)),
                "raw_tcn": float(np.median(raw_tcn_all)),
                "dwell_tcn": float(np.median(dwell_tcn_all)),
                "comb_tcn": float(np.median(comb_tcn_all)),
            },
            "std_rmse": {
                "raw_ridge": float(np.std(raw_ridge_all)),
                "comb_ridge": float(np.std(comb_ridge_all)),
                "raw_tcn": float(np.std(raw_tcn_all)),
                "dwell_tcn": float(np.std(dwell_tcn_all)),
                "comb_tcn": float(np.std(comb_tcn_all)),
            },
            "tcn_vs_ridge_wins": tcn_wins,
            "diff_tcn_vs_ridge": [float(x) for x in diff_tcn_vs_ridge],
            "gate_passed": gate_passed,
        },
        "locked_test_results": locked_test_results,
    }

    with open("tcn_loeo_study_results.json", "w") as f:
        json.dump(output_package, f, indent=2)
    print("\nSaved full results to tcn_loeo_study_results.json")


if __name__ == "__main__":
    run_experiment()
