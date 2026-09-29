# EGPM Repository and Data Pipeline Audit
**Generated:** 2025-09-28  
**Auditor:** Lead ML Researcher (Schneider Electric PredictX 2026)  
**Scope:** Complete codebase audit for experimental integrity per PRD §23 checklist

---

## 1. Repository Structure Overview

```
egpm/
├── data/                    # Unit-level dataset contracts and loaders
│   ├── loaders.py           # NCMAPSSLoader, MIMIILoader, UnitRecord, UnitDataset, UnitSplit
│   └── split_builder.py     # SplitBuilder, NormalizationStats (leakage guarantees)
├── preprocessing/           # Deterministic preprocessing pipeline
│   ├── preprocessor.py      # sync, impute, window, normalize (causal)
│   └── spectral.py          # Causal log-mel front-end for MIMII
├── encoder/                 # Neural sensor encoder (causal conv stack)
│   └── sensor_encoder.py    # Per-channel causal + dilated fusion + optional attention
├── quantization/            # VQ-VAE event discovery
│   └── vq_codebook.py       # Corrected straight-through estimator, EMA updates
├── grammar/                 # Explicit-duration HSMM
│   ├── hsmm.py              # HSMM core with DurationHistogram
│   ├── forward_backward.py  # Explicit-duration forward-backward + smoothing
│   ├── viterbi.py           # Segmental Viterbi decoding
│   └── baum_welch.py        # EM fitting for HSMM
├── rul/                     # Phase-type RUL estimation
│   └── phase_type_rul.py    # Duration-corrected formula + Monte Carlo intervals
├── anomaly/                 # Grammar-native anomaly scoring
│   └── anomaly_scorer.py    # Viterbi-based scoring + validation-only thresholding
├── fault/                   # Fault classification head
│   └── fault_head.py        # Belief GRU + duration statistics (no raw embedding bypass)
├── health/                  # Health index head
│   └── health_head.py       # Belief-weighted index (fixed ordinal or learned monotone)
├── explanation/             # Structured explanation object
│   └── explanation_builder.py
├── training/                # Staged training loop
│   └── trainer.py           # Stage 1 (encoder+VQ), Stage 2 (HSMM EM), Stage 3 (joint heads)
├── evaluation/              # Metrics and statistical tests
│   ├── rul_metrics.py
│   └── stats.py             # bootstrap_ci, paired_wilcoxon, etc.
├── experiments/             # Controlled ablations and real-data runs
│   ├── ablation_arms.py     # N-CMAPSS arms A-F + baselines
│   ├── heads_run.py         # Fault/health heads on real N-CMAPSS
│   ├── mimii_run.py         # MIMII fan real-data experiments
│   └── readout_study.py     # RUL readout ablation
├── interpretability/        # §24 validation metrics
│   └── metrics.py           # Coherence, stability, temporal, faithfulness, rank corr
├── baselines/               # Core baselines (CNN-RUL, LSTM-RUL, GrammarViz, CBM)
├── utils/                   # Seeding, run IDs
└── tests/                   # 249 tests (mathematical invariants, contracts, leakage audits)
```

---

## 2. Data Loaders and Unit-Level Contracts

### NCMAPSSLoader (loaders.py:141-249)
- **Schema verified**: A_dev/A_test [T,4] = unit, cycle, Fc, hs; W [T,4] ops; X_s [T,14] sensors; Y [T] RUL
- **Unit-level grouping**: Contiguous row blocks by unit ID; namespaced by dataset (DS01_u2 vs DS02_u2)
- **Labels**: 
  - `health_labels` = hs (1-based, 1..6)
  - `rul_labels` = Y (cycles to failure)
  - `fault_label` = Fc (flight condition, constant per unit)
  - No anomaly labels (N-CMAPSS has no anomalous runs in dev/test)
- **Optional truncation**: `max_cycles_per_unit` caps training to early cycles (prevents label leakage from late degradation)

### MIMIILoader (loaders.py:296-500)
- **Layouts supported**: 
  - Download layout: `<machine>/id_XX/{normal,abnormal}/*.wav` (16kHz, 8-ch, 10s clips)
  - Archive layout: dB tokens in path (backwards compatibility)
- **Labels**: `anomaly_labels` = 1 for abnormal clips (all frames), 0 for normal
- **Features**: Causal log-mel (spectral.py) with per-band mean/std over 8 mics + optional deltas
- **Per-clip units**: Each 10s clip is a `UnitRecord` with clip-level label

### UnitDataset / UnitRecord (loaders.py:17-116)
- **Contract enforced**: 
  - Signals [T, d], timestamps [T], labels all [T] matching T
  - `split()` method: unit-level random partition (train/val/test) with fixed seed
  - `UnitSplit` guarantees disjoint sets (raises on overlap)

---

## 3. Split and Normalization Audit (PRD §23 Checklist)

### SplitBuilder (split_builder.py)
- **Unit-level splits**: `build()` calls `UnitDataset.split()` — splits occur BEFORE windowing ✅
- **Train-only normalization**: `fit_normalization()` concatenates ONLY train unit signals for mean/std ✅
- **Leakage audit** (`audit()` method): 
  - Disjoint: no unit in two splits ✅
  - Complete: all units assigned ✅
  - Train/test non-empty ✅
- **Fixed seed**: Split seed fixed across experiments (seed=0 default) for reproducibility

### Preprocessor (preprocessor.py)
- **Causal sync**: `_sync_channel` uses only past/current samples; no future extrapolation ✅
- **Causal impute**: LOOCF for short gaps; linear interpolation across gap uses endpoints at or before gap end ✅
- **Window labels**: Window covering [s, s+W) gets label at s+W-1 (window END) — never later ✅
- **Normalization**: Applied via `SplitBuilder.fit_normalization` output; test receives train-only stats ✅

### Windowing (preprocessor.py:201-243)
- Stride defaults to window_length (MVP: non-overlapping, one token per window)
- Labels attached to window END index (s+W-1) — consistent with causal constraint ✅

---

## 4. Encoder → VQ → HSMM Pipeline

### Encoder (encoder/sensor_encoder.py)
- **Causal conv**: Left padding only, no future leak ✅
- **Per-channel depthwise**: Preserves channel identity ✅
- **Dilated fusion**: Geometric dilation (1,2,4...) for large receptive field ✅
- **Optional attention**: Flagged, default OFF (ablation A5) ✅
- **Pooling**: Mean or attention over window positions with mask support ✅

### VQ Codebook (quantization/vq_codebook.py)
- **Corrected straight-through**: `st = z + (c_v - z).detach()` — forward passes `c_v`, gradient flows to `z` ✅
- **EMA updates**: Standard VQ-VAE with Laplace smoothing ✅
- **Collapse monitoring**: Perplexity + effective size exposed on every forward (REQUIRED) ✅
- **Deterministic argmin**: `event_id()` for inference (no stochasticity) ✅

### HSMM (grammar/hsmm.py + forward_backward.py + baum_welch.py)
- **Explicit duration**: DurationHistogram (non-parametric pmf over {1..D_max}) ✅
- **Embedded transitions**: Zero-diagonal A for transient states; absorbing state identity row ✅
- **Forward-backward**: Occupancy parametrization O(T·M·D_max); validated against brute-force enumeration ✅
- **Smoothing posterior**: `p_t(i) ∝ Σ_e α_t(i,e)β_t(i,e)`; exact identity `Σ_{i,e} αβ = P(v)` at every t (CI checked) ✅
- **Segmental Viterbi**: Decodes state segments with duration-aware path cost ✅
- **Baum-Welch**: EM with restarts, early-stopping on holdout log-likelihood ✅

---

## 5. RUL, Anomaly, Fault, Health Heads

### PhaseTypeRUL (rul/phase_type_rul.py)
- **Duration-corrected formula**: `E[RUL|s,τ] = (r(τ) - μ) + Φμ` — not naive `(I-Q)⁻¹1` ✅
- **Mid-dwell correction**: Uses residual dwell `r(τ)` instead of full `μ` ✅
- **Monte Carlo intervals**: `simulate_to_absorption()` for median/prediction intervals ✅
- **Degeneracy check**: Raises if transient state cannot reach absorbing state ✅

### AnomalyScorer (anomaly/anomaly_scorer.py)
- **Three-component score**: Emission + transition + duration surprisal ✅
- **Validation-only threshold**: 99th percentile of VALIDATION normal windows only ✅
- **Test isolation**: Test labels never touch threshold selection (enforced by test) ✅

### FaultHead (fault/fault_head.py)
- **Input contract**: Beliefs [B,T,M] + duration stats — NEVER raw z/z_hat ✅
- **Bottleneck enforcement**: `use_raw_embedding` flag ONLY for ablation A4; raises if violated ✅
- **Features**: GRU-pooled belief + dwell + transition counts (row-normalized) ✅

### HealthHead (health/health_head.py)
- **Fixed ordinal prior** (MVP): Linear 1.0 → 0.0 over states ✅
- **Learned option (b)**: Monotonicity penalty `Σ relu(w_{m+1} - w_m + 0.01)` ✅
- **Train-only fitting**: `fit_learned_weights` uses TRAIN labels only; clamps to [0,1] ✅

---

## 6. Training Loop and Stage 3

### Trainer (training/trainer.py)
- **Stage 1**: Encoder + VQ + Decoder (recon + commitment loss) ✅
- **Stage 2**: HSMM Baum-Welch on normal-operation event sequences (PRD §10.4) ✅
- **Stage 3 (static)**: `stage3_joint` — masked composite loss over frozen representation heads ✅
  - Masking: `None` loss = task absent (PRD §15 missing-label mask) ✅
  - Encoder/VQ/HSMM frozen; full gradient through forward-algorithm is research-grade extension ✅

---

## 7. Experiments and Baselines

### Ablation Arms (experiments/ablation_arms.py)
| Arm | Representation | Temporal Model | RUL Head |
|-----|---------------|----------------|----------|
| A | Continuous encoder | None | Ridge |
| B | Continuous encoder | HMM (d_max=1) | Ridge |
| C | VQ tokens | Markov (1st order) | Ridge |
| D | VQ tokens | HMM (geometric dwell) | Ridge |
| E | VQ tokens | HSMM (explicit duration) | Ridge |
| F | VQ tokens | HSMM + absorb calibration | Ridge |
| CNN | Raw windows | CNN | CNN-RUL |
| LSTM | Raw windows | LSTM | LSTM-RUL |

- **Shared protocol**: Identical splits, preprocessing, windows, seeds, ridge head ✅
- **Calibration knob**: F vs E isolates absorbing-rate calibration (train-only) ✅
- **CNN/LSTM baselines**: Same windows/splits, per-window and per-run respectively ✅

### MIMII Run (experiments/mimii_run.py)
- **7 arms**: continuous_AE, pooled_events, vq_markov, vq_hsmm, continuous_hsmm, grammarviz, concept_bottleneck
- **Split**: 70/15/15 clip-level (normal only train; eval = held-out normal + all abnormal) ✅
- **Thresholds**: Validation-only per arm (99th pct of arm's validation normal scores) ✅
- **Wilcoxon**: Paired per-clip correctness vs primary arm, worst-case p across seeds ✅

---

## 8. Interpretability Metrics (interpretability/metrics.py)

1. **Event coherence**: Silhouette + within/between distance ratio on raw window features grouped by token
2. **Event stability**: Hungarian-matched usage overlap across seeds
3. **Cross-unit consistency**: JSD of event usage (train vs held-out)
4. **Temporal validity**: Viterbi boundary alignment with true health onsets (N-CMAPSS)
5. **Temporal onset proxy**: First-detection-lag for MIMII (no true onsets)
6. **Explanation faithfulness**: Mask flagged subsequence → recompute anomaly score → measure reduction (state-aware masking) ✅
7. **Anomaly rank correlation**: Spearman(score, label) — renamed from "rule fidelity" per PRD Decision 9 ✅

---

## 9. Test Suite Coverage (249 tests passing)

### Mathematical Invariants
- Forward vs brute-force likelihood enumeration (test_hsmm_likelihood.py)
- α·β = P(v) at every t (forward-backward consistency)
- Viterbi likelihood bound
- EM monotonicity
- RUL formula vs closed-form synthetic cases
- Monte Carlo RUL agreement

### Contract Tests
- Data-contract shapes per module
- Unit-split leakage audits
- Preprocessing causality
- VQ straight-through correctness
- Validation-only thresholding

### Statistical Tests
- Bootstrap CI (unit-level)
- Paired Wilcoxon (unit-level aggregation)
- Per-unit RMSE with CI

### Leakage Audits
- `SplitBuilder.audit()` verified in CI
- Normalization stats from train units only
- No unit in two splits
- Window labels use end index only

---

## 10. Issues Found and Fixes Applied

### Issue 1: MIMIILoader dB Filter Logic (FIXED)
- **Problem**: Archive layout dB filter was excluding download-layout paths
- **Fix**: Download layout paths (no dB token) now always match; filter only applies when dB token present

### Issue 2: Unit ID Collision in MIMII (FIXED)
- **Problem**: Normal and abnormal clips with same stem shared unit_id
- **Fix**: Unit IDs namespaced by `group/label/stem` (e.g., `mimii_fan_id_00_normal_00000000`)

### Issue 3: HealthHead Learned Weights (IMPLEMENTED)
- **Added**: `learn_weights` option with monotonicity penalty + `fit_learned_weights()` 
- **Decision**: Fixed ordinal prior retained as default (learned did not beat noise)

### Issue 4: MIMII Temporal Validity (IMPLEMENTED)
- **Added**: `temporal_onset_proxy` — first-detection-lag as pseudo-onset for MIMII
- **Caveat**: Explicitly noted as proxy, not true onset alignment

### Issue 5: Fault Head Fc Target (IMPLEMENTED)
- **Added**: `fault_target="fc"` option in heads_run; degenerate-split guard for DS02
- **Finding**: Old 1.0 accuracy was constant-predictor artifact (all units majority hs=1)

### Issue 5: Stage 3 Joint Training (IMPLEMENTED)
- **Added**: `Trainer.stage3_joint` with masked composite loss
- **A3 finding**: Delta ≡ 0 — frozen representation ⇒ no shared params ⇒ joint ≡ single-task

### Issue 6: LSTM Baseline (IMPLEMENTED)
- **Added**: Per-timestep head for per-window evaluation; causal test passes
- **Results**: DS01 38.8, DS02 41.2 — worse than CNN (34.0/28.9) and E (19.8/17.3)

### Issue 7: GrammarViz + CBM Arms (IMPLEMENTED)
- **Added**: GrammarViz (SAX+Sequitur) and Concept Bottleneck on MIMII
- **Results**: GrammarViz 0.152 (anti-correlated), CBM 0.414

---

## 11. Remaining Gaps and Phase-2 Items

| Item | Status | Notes |
|------|--------|-------|
| MC RUL interval wiring | Exists in code (`simulate_to_absorption`), not in explanation output | Phase-2 |
| Streaming fixed-lag decode | Not implemented | Phase-2 |
| ToyADMOS generalization | Not tested | Phase-2 |
| Config versioning (`configs/`) | Not needed (single config) | Phase-2 |
| Parallel per-(id,seed) driver | Sequential only | Engineering |
| OmniAnomaly proper | Flagged/gated; not implemented | Phase-2 |
| Air-compressor dataset | Not available | Phase-2 |

---

## 12. Conclusion: Audit Passes with Documented Caveats

**All PRD §23 leakage checks PASS:**
- ✅ Splits at unit level, before windowing
- ✅ Normalization statistics from training units only
- ✅ No window contains future information
- ✅ Overlapping windows not treated as independent (per-unit/bootstrap aggregation)
- ✅ Hyperparameter tuning on validation split only
- ✅ Test set touched once for final reporting

**Known Limitations (honestly reported):**
1. **N-CMAPSS test units**: Only 2 independent test units per dataset — CIs wide, Wilcoxon underpowered by construction
2. **MIMII unit definition**: Clips = units; 407 abnormal clips but only 4 machine IDs (id_00..id_06)
3. **Health index**: Weak signal (Spearman 0.21/0.10) — representation issue, not weighting
4. **Native phase-type RUL**: Loses to ridge readout; absorbing calibration hurts DS02
5. **Discretization on MIMII**: VQ bottleneck discards acoustic detail; continuous representations win
6. **A3 joint training**: No benefit with frozen representation — needs full-gradient Stage 3

**No blocking bugs. Experimental evidence package ready for Phase 1 protocol freeze.**

---

## 13. Files Created/Modified During Audit

### Created
- `experiments/AUDIT.md` — this document

### Modified (pre-existing fixes, not part of this audit)
- `egpm/data/loaders.py` — MIMII dB filter, unit ID namespacing
- `egpm/health/health_head.py` — learned weights option
- `egpm/experiments/heads_run.py` — `fault_target="fc"`, A3 multitask, fault_mode
- `egpm/training/trainer.py` — `stage3_joint` static method
- `egpm/experiments/mimii_run.py` — GrammarViz + CBM arms
- `egpm/interpretability/metrics.py` — `temporal_onset_proxy`
- `egpm/baselines/lstm_rul.py` — per-timestep head + causal test

All modifications are covered by the existing 249-test suite.