# EGPM Locked Evaluation Protocol
**Version:** 1.0 (frozen)  
**Date:** 2025-09-28  
**Status:** FROZEN — no parameter changes after this point without explicit version bump  
**Reference:** AUDIT.md (repository integrity verified)

---

## 1. Datasets and Splits

### N-CMAPSS (RUL / Fault / Health)
| Property | Value |
|----------|-------|
| Files | `N-CMAPSS/N-CMAPSS DS01 005.h5`, `N-CMAPSS/N-CMAPSS DS02 006.h5` |
| Units per file | DS01: 10 (6 dev + 4 test), DS02: 9 (6 dev + 3 test) |
| Signals per unit | 18 channels (4 ops + 14 sensors) |
| Labels per unit | `health_labels` (hs 1..6), `rul_labels` (cycles to failure), `fault_label` (Fc 1..3) |
| **Split method** | Unit-level random partition (train/val/test) |
| **Split fractions** | Train 0.6, Val 0.2, Test 0.2 (fixed seed=0) |
| **Split seed** | 0 (frozen across all experiments) |
| **Downsample** | 1 row per cycle (stride = T / max_cycle) |
| **Normalization** | Per-channel z-score fit on TRAIN units only |
| **Test units (DS01)** | 2 (DS01_u2, DS01_u9) |
| **Test units (DS02)** | 2 (DS02_u11, DS02_u14) |

### MIMII Fan (Anomaly Detection)
| Property | Value |
|----------|-------|
| Root directory | `fan/` |
| Machine IDs | `id_00`, `id_02`, `id_04`, `id_06` |
| Clips per ID | ~1400 normal + ~400 abnormal (varies by ID) |
| Audio format | 16 kHz, 8-channel, 10 s clips (WAV) |
| **Split method** | Clip-level random partition of NORMAL clips only |
| **Split fractions** | Train 0.7, Val 0.15, Test 0.15 |
| **Split seed** | 0, 1, 2 (3 seeds per experiment) |
| **Abnormal clips** | ALL held out for evaluation (never in train/val) |
| **Features** | Causal log-mel (64 mels × 8 mics → 512 dims; `channel_mode="mean"`) |
| **Deltas** | First-order temporal deltas appended (1024 dims total) |
| **Window length** | 16 frames (~0.26 s per token) |
| **Stride** | 16 (non-overlapping, one token per window) |
| **Normalization** | Per-channel z-score fit on TRAIN normal clips only |

---

## 2. Windowing and Preprocessing (Frozen)

| Parameter | N-CMAPSS | MIMII |
|-----------|----------|-------|
| Window length (W) | 4 | 16 |
| Stride (S) | 4 (non-overlapping) | 16 (non-overlapping) |
| Sync fs | 1.0 (native grid) | 1.0 (frame grid) |
| Max gap LOOCF | 5 | 5 |
| Label assignment | Window END index (s + W - 1) | Window END index |
| Anomaly label | Any anomalous frame in window → 1 | N/A (N-CMAPSS) / per-window mean (MIMII) |

**Causality guarantees (enforced by code):**
- Synchronization: no future samples in interpolation
- Imputation: LOOCF for short gaps; linear interpolation across gap uses endpoints at or before gap end
- Window label: always at window's last row (s + W - 1)
- Normalization: train-unit statistics only

---

## 3. Model Architecture (Frozen)

### Encoder
| Parameter | Value |
|-----------|-------|
| Hidden dim | 32 |
| Embed dim | 16 |
| Conv blocks | 3 |
| Kernel size | 3 |
| Dilation base | 2 (geometric: 1, 2, 4) |
| Attention | OFF (ablation A5 only) |
| Pooling | Mean (attention for A5 only) |

### VQ Codebook
| Parameter | Value |
|-----------|-------|
| Vocabulary size (K) | 16 (N-CMAPSS), 8 (MIMII) |
| Commitment β | 0.25 |
| EMA decay | 0.99 |
| Init | k-means++ on fresh encoder embeddings |
| Collapse monitoring | Perplexity + effective size (exposed every forward) |

### HSMM
| Parameter | Value |
|-----------|-------|
| States (M) | 5 (N-CMAPSS), 3 (MIMII) |
| Max duration (D_max) | 10 (N-CMAPSS), 12 (MIMII) |
| Duration model | Non-parametric histogram (DurationHistogram) |
| Initial π | Dirichlet(1), zero mass on absorbing state |
| Transitions A | Zero-diagonal for transient; absorbing = identity |
| Emissions B | Categorical over K tokens (Dirichlet prior) |
| Duration prior | Discretized geometric from mean_dwell init |
| EM max iter | 10 (N-CMAPSS), 10 (MIMII) |
| EM restarts | 3 (N-CMAPSS), 0 (MIMII) |

### RUL Head
| Parameter | Value |
|-----------|-------|
| Primary | Ridge regression (closed-form) on HSMM posterior features |
| Phase-type | `PhaseTypeRUL` class (duration-corrected formula) |
| Features | `state_posterior` [T,M] + `expected_dwell` [T,M] |
| Calibration | Train-only absorbing column scaling (Arm F only) |

### Anomaly Scorer
| Parameter | Value |
|-----------|-------|
| λ_trans | 1.0 |
| λ_dur | 1.0 |
| Threshold percentile | 99.0 |
| Threshold source | VALIDATION normal windows ONLY |
| Score components | Emission + λ_trans·Transition + λ_dur·Duration |

### Fault Head
| Parameter | Value |
|-----------|-------|
| Input | Beliefs [B,T,M] + dwell + transition counts |
| Pooling | GRU (hidden=32) over beliefs |
| Classes | `fault_target="hs"` (health state majority) or `"fc"` (flight condition) |
| Bottleneck | Enforced: `use_raw_embedding` ONLY for ablation A4 |

### Health Head
| Parameter | Value |
|-----------|-------|
| Default | Fixed ordinal prior: `linspace(1.0, 0.0, M)` |
| Option (b) | Learned weights + monotonicity penalty (margin 0.01) |
| Fitting | TRAIN labels only; clamps to [0,1] |

---

## 4. Training Protocol (Frozen)

### Stage 1: Encoder + VQ
| Parameter | Value |
|-----------|-------|
| Steps | 300 |
| Batch size | 32 (N-CMAPSS), 128 (MIMII) |
| LR | 1e-3 (AdamW) |
| Codebook init | k-means++ on fresh encoder embeddings |
| Anti-collapse | Laplace-smoothed EMA cluster sizes (init=1) |
| Exit criterion | Perplexity stable over last 50% of steps |

### Stage 2: HSMM Induction
| Parameter | Value |
|-----------|-------|
| Data | Normal-operation runs only (PRD §10.4) |
| Algorithm | Baum-Welch (EM) |
| Max iter | 10 |
| Restarts | 3 (N-CMAPSS), 0 (MIMII) |
| Seed | Fixed per experiment |
| Early stopping | Holdout log-likelihood (if implemented) |

### Stage 3: Joint Head Fine-tuning
| Parameter | Value |
|-----------|-------|
| Status | Research-grade extension (not default) |
| Method | `Trainer.stage3_joint` — masked composite loss |
| Heads frozen | Encoder, VQ, HSMM |
| Tasks | Fault (CE) + Health (MSE) + RUL (Huber) |
| Masking | `None` loss = task absent (PRD §15) |
| LR | 1e-3 (AdamW) |
| Steps | 200 |

---

## 5. Evaluation Metrics (Frozen)

### RUL (N-CMAPSS)
| Metric | Formula / Note |
|--------|----------------|
| RMSE | `sqrt(mean((y_pred - y_true)^2))` per window, aggregated per unit |
| MAE | `mean(|y_pred - y_true|)` per unit |
| PHM Score | Asymmetric (early penalty) — if computed |
| Per-unit aggregation | Mean over unit's windows → bootstrap CI over units |
| Statistical unit | **Unit (engine)** — not window |

### Anomaly Detection (MIMII)
| Metric | Formula / Note |
|--------|----------------|
| AUROC | `sklearn.metrics.roc_auc_score` |
| AUPRC | `sklearn.metrics.average_precision_score` |
| F1 at threshold | Validation-selected threshold (99th pct normal) |
| Per-clip correctness | `(score > threshold) == label` per clip |
| Wilcoxon | Paired per-clip vs primary arm (worst-case p across seeds) |
| Statistical unit | **Clip** (500+ clips per ID → powered) |

### Fault / Health
| Metric | Formula / Note |
|--------|----------------|
| Fault accuracy | Run-level majority class (hs) or Fc label |
| Health Spearman | `spearmanr(health_index, hs_label)` per unit |
| Health a/b decision | Validation Spearman → winner evaluated on test |

### Interpretability (§24)
| Metric | Target |
|--------|--------|
| Event coherence | Silhouette > 0 (weak is acceptable) |
| Event stability | Matched usage overlap > 0.6 |
| Cross-unit JSD | < 0.5 |
| Temporal validity | Detection rate > 0.5 (N-CMAPSS only) |
| Explanation faithfulness | Score reduction > 0 (masking test) |
| Anomaly rank correlation | Spearman (discriminative only, not faithfulness) |

---

## 6. Statistical Rigor (Frozen)

### Independent Units
| Dataset | Independent Unit | N_test |
|---------|------------------|--------|
| N-CMAPSS DS01 | Engine/unit | 2 |
| N-CMAPSS DS02 | Engine/unit | 2 |
| MIMII fan | Clip (but grouped by machine ID) | 4 IDs × 300+ clips |

### Reporting Requirements
- **Mean ± std** across seeds (3 seeds minimum)
- **95% bootstrap CI** (unit-level for N-CMAPSS; clip-level for MIMII)
- **Paired tests** where same units/clips evaluated (Wilcoxon)
- **Per-seed numbers** always reported (no hiding behind averages)
- **Effect size** for ablations: `relative_improvement = (baseline - EGPM) / baseline`

### Known Statistical Limitations (Explicitly Stated)
1. N-CMAPSS: Only 2 independent test units → CIs wide, Wilcoxon underpowered by construction
2. MIMII: Clips within same machine ID may be correlated; 4 IDs is the true replication
3. Health index: 2-class hs at run level → fault accuracy 1.0 is constant-predictor artifact
4. Fc classification: DS02 train units all Fc=3 → degenerate single-class split

---

## 7. Baselines (Locked)

| Baseline | Dataset | Config |
|----------|---------|--------|
| CNN-RUL | N-CMAPSS | Same windows/splits; per-window MSE + val early-stop |
| LSTM-RUL | N-CMAPSS | Same windows/splits; per-timestep head; masked MSE + val early-stop |
| GrammarViz | MIMII | SAX (word_len=4, alphabet=4) + Sequitur n-gram novelty |
| Concept Bottleneck | MIMII | k-means (K=8) on raw windows; nearest-normal histogram distance |
| Continuous AE | MIMII | WindowAutoEncoder recon MSE (baseline i) |
| VQ Markov | MIMII | First-order token Markov (baseline ii) |
| VQ HSMM | MIMII | Full HSMM surprisal (primary) |
| Continuous HSMM | MIMII | Gaussian-emission HSMM on embeddings (A1) |

**Fair-comparison protocol:** Identical splits, preprocessing, windows, seeds, tuning budget.

---

## 8. Ablation Arms (Locked)

| Arm | Representation | Temporal | Notes |
|-----|---------------|----------|-------|
| A | Continuous encoder | None | Window AE embedding |
| B | Continuous encoder | HMM (d_max=1) | Continuous HMM |
| C | VQ tokens | Markov | First-order token Markov |
| D | VQ tokens | HMM | Geometric dwell (d_max=1) |
| E | VQ tokens | HSMM | Explicit duration, no calibration |
| F | VQ tokens | HSMM + calibration | Train-only absorbing column scaling |
| CNN | Raw windows | CNN | CNN-RUL baseline |
| LSTM | Raw windows | LSTM | LSTM-RUL baseline |

**RUL decoder:** Identical ridge head for all arms A–F (shared prediction head protocol).

---

## 9. Random Seeds (Frozen)

| Purpose | Seeds |
|---------|-------|
| Data split (N-CMAPSS) | 0 (fixed) |
| Data split (MIMII) | 0, 1, 2 (3 seeds) |
| Model init / training | Same as split seed per run |
| EM restarts | Separate RNG, same base seed |
| Bootstrap CI | Seed 0 (fixed) |
| Hungarian matching | Deterministic (scipy) |

---

## 10. Compute Budget (Estimated)

| Experiment | Est. Time (CPU) | Est. Time (GPU) | Parallelism |
|------------|-----------------|-----------------|-------------|
| N-CMAPSS ablation (8 arms × 3 seeds) | ~45 min | ~15 min | Seeds parallel |
| MIMII fan (7 arms × 4 IDs × 3 seeds) | ~14 h | ~3 h | IDs parallel |
| Heads run (3 seeds) | ~3 min | ~1 min | Seeds parallel |
| RUL readout study | ~5 min | ~2 min | — |
| Interpretability suite | ~9 min/ID | ~3 min/ID | IDs parallel |

---

## 11. Artifacts and Naming Convention

| Artifact | Pattern | Contents |
|----------|---------|----------|
| N-CMAPSS ablation | `ablation_report_DS{01,02}.json` | Per-arm RMSE/MAE/onset/stability + unit CI + E-decomp |
| MIMII fan | `mimii_fan_id_{00,02,04,06}_3seeds.json` | Per-arm AUROC/AUPRC + Wilcoxon |
| MIMII interpretability | `mimii_fan_id_{00,02,04,06}_interpretability.json` | 6 §24 metrics + temporal proxy |
| Heads run | `heads_report_DS{01,02}.json` + `_fc.json` | Fault acc + health Spearman + A3 |
| RUL readout | `readout_study_DS{01,02}.json` | Frozen EGPM features + various heads |
| Cross-dataset | `exp3_cross_dataset.json` | Train DS01→test DS02 + reverse |
| EM restart sweep | `exp2_em_restart_sweep.json` | RMSE vs restarts |

---

## 12. Protocol Change Procedure

**This protocol is FROZEN.** Changes require:

1. Version bump (e.g., 1.0 → 1.1)
2. Explicit rationale in changelog
3. Re-run of ALL affected experiments from scratch
3. No cherry-picking — if protocol changes, ALL numbers regenerated

**Exception:** Bug fixes that correct implementation to MATCH this protocol (not change the protocol itself) may be applied without version bump, but must be documented in AUDIT.md.

---

## 13. Test Set Lock

**The test sets for N-CMAPSS DS01, DS02 and MIMII fan id_00/02/04/06 are LOCKED.**

- No model selection on test data
- No threshold tuning on test data  
- No hyperparameter search on test data
- No architecture decisions based on test performance
- Test touched ONCE for final reporting in each experiment

**Violation = invalid experiment.**

---

## 14. Sign-Off

| Role | Name | Date | Signature |
|--------|------|------|-----------|
| Lead ML Researcher | [Automated] | 2025-09-28 | Protocol v1.0 FROZEN |

---

*This protocol is derived from AUDIT.md and reflects the actual implemented codebase as of 2025-09-28. All experiments from this point forward MUST adhere to these parameters.*