# EGPM Experimental Execution Plan
**Protocol:** v1.0 (frozen, PROTOCOL.md)  
**Status:** Phase 0-1 complete, Phase 2 complete  
**Next:** Phase 3-9 (core scientific evidence)

---

## Phase Status

| Phase | Description | Status | Priority |
|-------|-------------|--------|----------|
| 0 | Repository & Data Audit | ✅ DONE (AUDIT.md) | — |
| 1 | Locked Protocol | ✅ DONE (PROTOCOL.md) | — |
| 2 | N-CMAPSS RUL Ablations | ✅ DONE (ablation_report_DS01/02.json) | — |
| 3 | Degradation Progression | 🔄 NEXT | P0 |
| 4 | Event Vocabulary Analysis | 🔄 NEXT | P0 |
| 5 | Information-Loss Study | 🔄 NEXT | P0 |
| 6 | MIMII Anomaly Deep Dive | 🔄 NEXT | P0 |
| 7 | Cross-Domain Generalization | ⏳ | P1 |
| 8 | Interpretability/Faithfulness | ⏳ | P1 |
| 9 | RUL Readout Study | ✅ PARTIAL (readout_study_DS01/02.json) | P0 |
| 10 | Robustness | ⏳ | P1 |
| 11 | Data Efficiency | ⏳ | P2 |
| 12 | Early Warning | ⏳ | P2 |
| 13 | Demo Metrics | ⏳ | P2 |
| 14 | Uncertainty | ⏳ | P2 |
| 15 | Baseline Research | ⏳ | P2 |
| 16 | Statistical Validation | ⏳ | P1 |
| 17 | Failure Analysis | ⏳ | P1 |
| 18 | Final Ablation Table | ⏳ | P0 |
| 19 | Best Defensible Config | ⏳ | P0 |
| 20 | Schneider Package | ⏳ | P0 |

---

## Execution Order (P0 First)

### 1. Phase 3: Degradation Progression (P0)
**Goal:** Demonstrate EGPM captures degradation, not just useful features.

**Experiments:**
- `run_degradation_trajectories.py` — Plot 10+ test trajectories: sensors, VQ events, HSMM states, state age, predicted RUL, true RUL, transitions
- `run_onset_detection.py` — Onset F1, precision, recall, median/mean detection lag per test unit
- `run_state_persistence.py` — Avg state duration, duration prediction error, state occupancy
- `run_monotonicity.py` — Test if degradation-related stats change monotonically with RUL

**Outputs:** Trajectory plots (PNG), onset detection table, monotonicity report

### 2. Phase 4: Event Vocabulary Analysis (P0)
**Goal:** Determine if VQ events have behavioral semantics.

**Experiments:**
- `run_event_dictionary.py` — For each code: freq, duration, sensor profile, RUL dist, state dist, transition neighbors, fault association, top-5 representative windows
- `run_event_visualizations.py` — Event×sensor heatmap, event×state heatmap, event transition graph, duration distributions

**Outputs:** Event dictionary (JSON/CSV), visualization PNGs

### 3. Phase 5: Information-Loss Study (P0)
**Goal:** Does VQ discard anomaly-relevant information?

**Experiments:**
- `run_information_loss.py` — Linear probes on continuous vs quantized representations for:
  - N-CMAPSS: RUL probe, fault/regime probe, degradation-stage probe
  - MIMII: Supervised probe, anomaly separability, reconstruction error
- `run_continuous_vs_vq_hsmm.py` — Continuous HSMM vs VQ HSMM (isolates VQ vs HSMM vs scoring)

**Outputs:** Probe comparison tables, information loss quantification

### 4. Phase 6: MIMII Anomaly Deep Dive (P0)
**Goal:** Understand why continuous beats discrete on MIMII.

**Experiments:**
- `run_anomaly_components.py` — Per-ID per-arm: emission/transition/duration surprisal distributions
- `run_anomaly_threshold_sweep.py` — AUROC/AUPRC vs threshold percentile
- `run_anomaly_per_id.py` — Deep dive on id_00 (worst) vs id_06 (best)

**Outputs:** Component analysis, threshold sensitivity, per-ID diagnostic reports

### 5. Phase 8: Interpretability/Faithfulness (P1)
**Goal:** Experimental demonstration of interpretability.

**Experiments:**
- `run_faithfulness.py` — Event masking, state masking, transition masking, duration masking, counterfactual substitution
- `run_stability.py` — Explanation consistency across seeds, event consistency across similar trajectories

**Outputs:** Faithfulness table, stability metrics

### 6. Phase 9: RUL Readout Study (P0 - extend existing)
**Goal:** Why does native phase-type lose to ridge?

**Experiments:**
- `run_rul_feature_ablation.py` — Ridge vs phase-type with feature groups: posterior only, +dwell, +transitions, +entropy, +event counts
- `run_rul_decoder_capacity.py` — Ridge vs ElasticNet vs MLP vs TSMixer on frozen EGPM features

**Outputs:** Feature ablation table, decoder comparison

### 7. Phase 16: Statistical Validation (P1)
**Goal:** Rigorous stats for all claims.

**Experiments:**
- `run_statistical_validation.py` — Paired tests, bootstrap CIs, effect sizes, limitation statements for all main results

**Outputs:** Statistical validation appendix

### 8. Phase 17: Failure Analysis (P1)
**Goal:** Understand worst 10% predictions.

**Experiments:**
- `run_failure_analysis.py` — Worst 10% RUL predictions: taxonomy (early/late degradation, unusual ops, noisy, unseen, codebook failure, state confusion, duration misest, decoder failure)

**Outputs:** Failure taxonomy, worst-case trajectory plots

### 7. Phase 18-19: Final Table & Best Config (P0)
**Goal:** Consolidated evidence and final recommendation.

**Experiments:**
- `generate_final_ablation_table.py` — One table: Representation | Temporal | Duration | Decoder | RUL RMSE | Anomaly AUROC | Onset F1 | Stability | Faithfulness
- `select_best_config.py` — Based on validation: predictive perf, interpretability, robustness, consistency, cost

**Outputs:** FINAL_ABLATION_TABLE.csv, BEST_CONFIG.md

### 8. Phase 20: Schneider Package (P0)
**Goal:** Competition-ready deliverables.

**Files:**
1. FINAL_RESULTS.md
2. EXECUTIVE_SUMMARY.md
3. ABLATION_TABLE.csv
4. BASELINE_COMPARISON.csv
5. ROBUSTNESS_RESULTS.csv
6. INTERPRETABILITY_RESULTS.csv
7. FAILURE_ANALYSIS.md
8. DATA_AND_PROTOCOL.md
9. LIMITATIONS.md
10. CLAIMS_AUDIT.md
11. FIGURE_INDEX.md
12. DEMO_METRICS.md

---

## Parallelism Strategy

| Experiment | Est. Time | Parallelism |
|------------|-----------|-------------|
| Degradation trajectories | 5 min | Per trajectory |
| Event dictionary | 2 min | Per event code |
| Information loss probes | 10 min | Per dataset/probe |
| MIMII anomaly components | 5 min | Per ID |
| Faithfulness masking | 5 min | Per test trajectory |
| RUL feature ablation | 3 min | Per feature group |
| Failure analysis | 5 min | Per failure case |

**Total estimated CPU time:** ~2-3 hours (parallelizable across 4-8 cores)

---

## Immediate Next Step

Start with **Phase 3: Degradation Progression** — trajectory plotting for 10 representative test units across DS01 and DS02. This is the foundational evidence for "EGPM models degradation."

Let me create the first script and run it.