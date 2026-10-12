# Data and Protocol Documentation

## Datasets

### N-CMAPSS
- **Source**: NASA Modular Aero-Propulsion System Simulation (N-CMAPSS)
- **Files**: `N-CMAPSS DS01 005.h5`, `N-CMAPSS DS02 006.h5` (MATLAB v7.3 / HDF5)
- **Schema verified**:
  - `A_dev` / `A_test`: [T, 4] = unit, cycle, Fc, hs
  - `W_dev` / `W_test`: [T, 4] = operational settings (alt, Mach, TRA, T2)
  - `X_s_dev` / `X_s_test`: [T, 14] = health-proxy sensors
  - `Y_dev` / `Y_test`: [T, 1] = ground-truth RUL (cycles to failure)
- **Units**: DS01: 10 (6 dev + 4 test), DS02: 9 (6 dev + 3 test)
- **Labels**:
  - `health_labels` = hs (1..6, 1-based)
  - `rul_labels` = cycles to failure (0 at failure)
  - `fault_label` = Fc (flight condition, constant per unit)

### MIMII Fan
- **Source**: MIMII Dataset (Hitachi)
- **Layout**: Download layout `fan/id_XX/{normal,abnormal}/*.wav`
- **Format**: 16 kHz, 8-channel, 10-second WAV clips
- **Machine IDs**: `id_00`, `id_02`, `id_04`, `id_06`
- **Clips per ID**: ~1400 normal + ~400 abnormal (varies)
- **Features**: Causal log-mel (64 mels × 8 mics → 512 dims; `channel_mode="mean"`) + first-order deltas = 1024 dims
- **Labels**: `anomaly_labels` = 1 for abnormal clips (all frames), 0 for normal

## Protocol (Locked — PROTOCOL.md v1.0)

### N-CMAPSS
| Parameter | Value |
|-----------|-------|
| Split method | Unit-level random (train 0.6, val 0.2, test 0.2) |
| Split seed | 0 (fixed across all experiments) |
| Downsample | 1 row per cycle (stride = T / max_cycle) |
| Window length | 4 |
| Stride | 4 (non-overlapping) |
| Normalization | Per-channel z-score, train units only |
| Test units (DS01) | 2 (DS01_u2, DS01_u9) |
| Test units (DS02) | 2 (DS02_u2, DS02_u5) |
| Seeds | 3 (split seed fixed at 0) |

### MIMII Fan
| Parameter | Value |
|-----------|-------|
| Split method | Clip-level random, normal clips only (train 0.7, val 0.15, test 0.15) |
| Split seeds | 0, 1, 2 (3 seeds per experiment) |
| Abnormal clips | ALL held out for evaluation (never in train/val) |
| Features | Causal log-mel (64 mels × 8 ch) + deltas = 1024 dims |
| Window length | 16 frames (~0.26 s per token) |
| Stride | 16 (non-overlapping) |
| Normalization | Per-channel z-score, train normal clips only |
| Seeds | 0, 1, 2 (3 seeds per experiment) |

## Model Architecture (Locked)

| Component | Parameter | Value |
|-----------|-----------|-------|
| Encoder | Hidden dim | 32 |
| Encoder | Embed dim | 16 |
| Encoder | Conv blocks | 3 |
| Encoder | Kernel size | 3 |
| Encoder | Dilation base | 2 |
| Encoder | Attention | OFF (ablation A5) |
| Encoder | Pooling | Mean |
| VQ | Vocabulary size (K) | 16 (N-CMAPSS), 8 (MIMII) |
| VQ | Commitment β | 0.25 |
| VQ | EMA decay | 0.99 |
| HSMM | States (M) | 5 (N-CMAPSS), 3 (MIMII) |
| HSMM | Max duration (D_max) | 10 (N-CMAPSS), 12 (MIMII) |
| HSMM | Duration model | Non-parametric histogram |
| HSMM | EM max iter | 10 |
| HSMM | EM restarts | 3 (N-CMAPSS), 0 (MIMII) |
| RUL decoder | Primary | Ridge on [posterior, dwell] |
| Anomaly scorer | λ_trans | 1.0 |
| Anomaly scorer | λ_dur | 1.0 |
| Anomaly scorer | Threshold pct | 99.0 (validation normal only) |
| Fault head | Target | `fault_target="fc"` (3-class) or `"hs"` (2-class) |
| Health head | Default | Fixed ordinal prior (learned weights don't beat noise) |

## Training Protocol (Locked)

| Stage | Description | Key Parameters |
|-------|-------------|----------------|
| Stage 1 | Encoder + VQ + Decoder (recon + commitment) | 300 steps, batch 32/128, AdamW 1e-3, k-means++ init |
| Stage 2 | HSMM Baum-Welch (normal ops only) | 10 iter, 3 restarts (NCM), k-means++ codebook init |
| Stage 3 | Joint head fine-tuning (frozen rep) | 200 steps, masked composite loss, AdamW 1e-3 (research-grade only) |

## Evaluation Metrics (Locked)

| Task | Primary Metric | Secondary | Statistical Unit |
|------|----------------|-----------|------------------|
| RUL (N-CMAPSS) | RMSE | MAE, PHM score | Unit (engine) |
| Anomaly (MIMII) | AUROC | AUPRC, F1@thresh | Clip (grouped by ID) |
| Fault | Accuracy (run-level) | — | Unit |
| Health | Spearman vs hs | — | Unit |
| Interpretability | 6 §24 metrics | — | Unit/Clip |

## Statistical Rigor (Locked)

- **Independent unit**: Engine (NCM, n=2) / Clip grouped by ID (MIMII, n=4 IDs)
- **Seeds**: 3 (MIMII), 1 split seed (NCM, fixed)
- **Reporting**: Mean ± std across seeds, 95% bootstrap CI (unit-level), paired Wilcoxon
- **Test lock**: Test sets touched ONCE for final reporting
- **Limitations explicit**: 2 test units → wide CIs; MIMII clips within ID correlated; health weak; Fc degenerate on DS02

## Test Set Lock

- **N-CMAPSS DS01**: Test units = {DS01_u2, DS01_u9}
- **N-CMAPSS DS02**: Test units = {DS02_u2, DS02_u5}
- **MIMII**: All abnormal clips + held-out normal per ID (never touched until final eval)
- **Violation = invalid experiment**