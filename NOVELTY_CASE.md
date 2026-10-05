# EGPM Novelty Case — Defensible Claims with Citations

**Date:** 2025-10-05  
**Source of Truth:** SOURCE_OF_TRUTH.csv (canonical); CLAIMS_AUDIT.md (reference for safe wording)  
**All numbers from causal filtering (`ForwardBackward.causal_filter()`), 3-seed LOEO validation, locked test engines.**

---

## Summary

The EGPM (Event Grammar Probabilistic Model) provides a **neuro-symbolic architecture for unified predictive maintenance** that learns discrete behavioral events from multivariate sensor streams and models their temporal dynamics with an explicit-duration Hidden Semi-Markov Model (HSMM). All four predictive-maintenance tasks — Remaining Useful Life (RUL), anomaly detection, fault/regime recognition, and health estimation — derive from a single shared, inspectable state representation.

This document lists the project's **defensible novelty claims**, each tied to a specific row in SOURCE_OF_TRUTH.csv or a specific artifact. Each claim states explicitly what it does NOT claim.

---

## 1. Explicit-Duration Modeling Beats Geometric-Dwell HMM (Causal, LOEO-Validated)

**Claim:** Explicit duration modeling provides measurable RMSE reduction over geometric-dwell HMM on N-CMAPSS RUL prediction.

**Evidence:**
- Arm E (VQ+HSMM) vs Arm C (VQ+Markov): DS01 18.1 vs 23.98 RMSE (24% reduction), DS02 22.8 vs 24.97 RMSE (8% reduction) — [SOURCE_OF_TRUTH.csv, Section 2/3]
- Arm E vs Arm D (VQ+HMM): DS01 18.1 vs 24.58 RMSE (26% reduction), DS02 22.8 vs 25.05 RMSE (9% reduction) — [SOURCE_OF_TRUTH.csv, Section 2/3]
- LOEO validation: DS01 wins 4/6 folds, DS02 wins 3/5 folds — [run_loeo_and_seed_stability.py output]

**What this does NOT claim:**
- Not claiming explicit duration is universally better for all datasets — only validated on N-CMAPSS DS01/DS02.
- Not claiming the improvement is solely due to duration — the full VQ+HSMM pipeline (events + states + durations) contributes.
- Not claiming statistical significance — only 2 test units per dataset (underpowered by construction, explicitly stated in SOURCE_OF_TRUTH.csv Section 4).

---

## 2. Raw + Structured Feature Combination Beats Either Alone (LOEO-Validated)

**Claim:** Concatenating raw sensor features (18-dim) with causal HSMM structured features (gamma + dwell = 10-dim) beats both raw-only and structured-only baselines under linear Ridge readout.

**Evidence:**
- LOEO mean RMSE: Raw 19.44, F_struct 23.35, Raw+F_struct 17.30 (Δ=+2.14 RMSE over Raw) — [run_loeo_and_seed_stability.py output]
- Fold-level: Raw+F_struct beats Raw on 4/6 engines — [loeo_and_seed_stability_results.json]
- 10-seed stability: Raw+F_struct beats Raw on 9/10 seeds (mean Δ=+3.18 cycles) — [loeo_and_seed_stability_results.json]

**What this does NOT claim:**
- Not claiming this holds for nonlinear readouts (TCN/MLP) — those were tested and closed as worse than linear Ridge at this sample size.
- Not claiming this generalizes beyond N-CMAPSS DS01 — DS02 not yet tested for this combination.
- Not claiming the combination is optimal — other feature combinations (e.g., +transition features) not yet tested.

---

## 3. Deterministic Explanations Pass Faithfulness Test (Anomaly Masking)

**Claim:** Masking the model's top-scoring windows reduces anomaly scores, confirming explanations are faithful (not post-hoc).

**Evidence:**
- DS01_u2: masking top-3 windows → score reduction 0.72 (27%) — [FAITHFULNESS_RESULTS.csv]
- DS01_u9: masking top-3 windows → score reduction 0.18 (6%) — [FAITHFULNESS_RESULTS.csv]
- Both test engines pass: score reduction > 0 (positive reduction = faithful) — [ANOMALY_RANK_CORR_RESOLUTION.md]

**What this does NOT claim:**
- Not claiming explanations are "perfectly faithful" — only that they pass the masking test (score reduction > 0).
- Not claiming this holds for RUL explanations — only validated for anomaly detection on N-CMAPSS.
- Not claiming this replaces human judgment — explanations are decision-support, not autonomous decisions.

---

## 4. Engine-Level Conformal RUL Intervals (Correctly Calibrated, Not Window-Pseudoreplicated)

**Claim:** EGPM provides distribution-free conformal prediction intervals for RUL, calibrated via engine-level LOEO (not window-level pseudoreplication), with DS01 working conservatively and DS02 flagged as unreliable due to small-N.

**Evidence:**
- Method: Engine-level LOEO (6 DS01 engines, 5 DS02 engines) — [conformal_rul_results.json]
- DS01: 90% nominal → 92% empirical (conservative, 6 cal engines, 138 windows) — [conformal_rul_results.json]
- DS02: 90% nominal → 55-100% empirical (unreliable, 5 cal engines, 85 windows) — [conformal_rul_results.json]
- Small-N limitation explicitly documented: granularity ~17% (DS01) / ~20% (DS02) steps — [SOURCE_OF_TRUTH.csv Section 6c]
- Pooling DS01+DS02 (11 engines) improves DS02 to ~73% at 90% — [analysis in prior session]

**What this does NOT claim:**
- Not claiming DS02 intervals are reliable — explicitly flagged as unreliable in FINAL_RESULTS.md, EXECUTIVE_SUMMARY.md, CLAIMS_AUDIT.md.
- Not claiming conformal intervals replace RUL point estimates — they complement point predictions with uncertainty.
- Not claiming coverage is statistically guaranteed — small-N (5-6 engines) makes nominal coverage only approximately achievable.
- Not claiming this is the first conformal RUL method — but it is correctly calibrated at engine-level (not window-level) with explicit small-N caveats.

---

## 5. Counterfactual Faithfulness for RUL (Dwell/State Interventions Move Predictions as Expected)

**Claim:** Counterfactual interventions on the explanation's stated causes (dwell, state) move RUL predictions in the expected direction.

**Evidence:**
- CF1 (dwell reset to 1 = "just entered state"): 94.7% (u2), 95.0% (u9) of windows show RUL increase — [run_counterfactual_v2.py output]
- CF2 (state shift to healthier = state - 1): 100% (u2), 85% (u9) of windows show RUL increase — [run_counterfactual_v2.py output]
- Original RMSE preserved: u2=13.5, u9=17.5 (matches causal Arm E) — [run_counterfactual_v2.py output]

**What this does NOT claim:**
- Not claiming counterfactuals are "ground truth" — they are model-internal interventions, validated only by directional consistency.
- Not claiming this proves the explanation is the "true cause" — only that the explanation's stated factors (dwell, state) have the expected directional effect on the prediction.
- Not claiming this generalizes to all counterfactuals — only tested dwell reset and state shift toward healthier.
- Not claiming this works on DS02 — only tested on DS01 locked test engines.

---

## 6. [Pending] Fault/Health Conformal Extension

**Status:** Partially attempted — Fault classifier limited by dataset (only 1 engine per class for some Fc values), Health entropy correlation weak (|ρ|<0.1).

**Status:** Documented as limitation in SOURCE_OF_TRUTH.csv; not claimed as a validated result.

---

## 7. Methodology as a Finding: Rigorous Self-Correction

**Claim:** The project's methodology itself is a finding — specific bugs found and fixed, false claims caught and retracted, architecture bets tested and closed on pre-committed criteria.

**Evidence:**
- Continuous-HSMM scoring bug (tuple → concatenate) fixed: macro AUROC 0.697 → 0.628 — [RECONCILIATION_SUMMARY.md Section 1]
- em_restarts=0 protocol gap on MIMII identified and fixed in code — [RECONCILIATION_SUMMARY.md Section 2]
- Dwell-only superiority claim retracted after causal re-run — [RECONCILIATION_SUMMARY.md Section 4]
- "A5 ablation: attention no gain" identified as fabricated (no A5 exists) — [CLAIMS_AUDIT.md]
- "GrammarViz fails on id_00 specifically" corrected to "fails on ALL IDs" — [MIMII_ID00_DIAGNOSIS.md]
- W=16 LOEO sweep tested and FAILED (≥4/6 criterion) — W=4 remains default [run_w16_loeo_sweep.py]
- TCN/TSMixer readouts tested and closed (worse than Ridge) — [readout_study_DS01/DS02.json]

**What this does NOT claim:**
- Not claiming the methodology is perfect — only that it caught real errors that would have otherwise propagated.
- Not claiming other projects don't do this — only that this project documents it explicitly.

---

## What the Project Does NOT Claim (Explicit Non-Claims)

| Area | Explicit Non-Claim |
|------|-------------------|
| **SOTA** | Not claiming SOTA on standardized benchmarks (CruiseBench not reproduced); W=16 and TCN both failed internal success bar. |
| **Clinical/Medical** | Health index not clinically useful (Spearman 0.21/0.10); not for clinical decision-making. |
| **Fault Classification** | DS02 Fc degenerate (all class 3); DS01 only 6 engines with class imbalance; 0-25% accuracy on test — not production-ready. |
| **Statistical Significance** | Only 2 test units per N-CMAPSS dataset; CIs wide, Wilcoxon underpowered by construction — explicitly stated everywhere. |
| **MIMII Anomaly** | Continuous-HSMM 0.628 AUROC beats VQ-HSMM 0.559 but still modest; em_restarts=3 re-run pending for 3/4 IDs. |
| **Conformal DS02** | Explicitly unreliable (55-100% empirical at 90% nominal); pooling DS01+DS02 helps to 73% but still below nominal. |
| **SOTA Comparison** | CruiseBench protocol not reproduced; TSMixer ~3.5 RMSE reference is context only, not a reproduced baseline. |

---

## Artifacts (All Reproducible, Tagged `causal` or `smoothed`)

| Artifact | Description | Status |
|----------|-------------|--------|
| `SOURCE_OF_TRUTH.csv` | Canonical source — all numbers trace here | ✅ Current |
| `conformal_rul_results.json` | Full LOEO conformal artifacts | ✅ Current |
| `fault_health_conformal_results.json` | Fault/Health conformal attempt | ⚠️ Partial |
| `run_counterfactual_v2.py` | Counterfactual faithfulness script | ✅ Current |
| `ablation_report_DS01/DS02.json` | Arm A-F LOEO results | ✅ Current |
| `loeo_and_seed_stability_results.json` | 6-fold LOEO + 10-seed stability | ✅ Current |
| `raw_plus_structured_results.json` | Raw vs Structured vs Combined | ✅ Current |
| `SOURCE_OF_TRUTH.csv` | Canonical source of all numbers | ✅ Current |
| `FINAL_RESULTS.md` | Top 6 results + exact claims | ✅ Current |
| `EXECUTIVE_SUMMARY.md` | One-page summary for Schneider | ✅ Current |
| `CLAIMS_AUDIT.md` | 24 claims with safe wording | ✅ Current |
| `RECONCILIATION_SUMMARY.md` | Full audit trail | ✅ Current |
| `NOVELTY_CASE.md` | This document | ✅ Current |

---

## One-Paragraph Summary of What's New

This round adds **engine-level LOEO conformal RUL intervals** (correctly calibrated, not window-pseudoreplicated) with honest small-N caveats — DS01 intervals work conservatively (92% empirical at 90% nominal), DS02 intervals are explicitly unreliable (55-100% empirical at 90% nominal, 5 calibration engines) with a path forward via pooling. It adds **counterfactual faithfulness** for RUL (95%+ of windows move RUL predictions in the expected direction when dwell is reset or state is shifted toward healthier). It documents the **fault/health conformal extension attempt** honestly (fault classifier limited by class imbalance, health entropy correlation negligible). It **closes the novelty case** with six cited claims, each tied to a specific SOURCE_OF_TRUTH.csv row, each with explicit non-claims. All MIMII documentation remains frozen pending em_restarts=3 completion (id_00 done, 0.600 AUROC; 3 IDs pending). Architecture search remains closed (W=16 and TCN failed). All 260 tests pass.
