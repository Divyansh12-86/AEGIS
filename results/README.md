# AEGIS Experimental Results Registry

This directory contains the frozen experimental results, cross-validation metrics, and claims evidence artifacts for the **Event Grammar Probabilistic Model (EGPM)**.

## Directory Structure

```
results/
├── ncmapss/                 # NASA N-CMAPSS Turbofan Benchmarks (DS01 & DS02)
│   ├── ablation_report_DS01.json       # Arms A–F LOEO cross-validation on DS01
│   ├── ablation_report_DS02.json       # Arms A–F LOEO cross-validation on DS02
│   ├── heads_report_DS01.json          # Multi-task heads benchmark (RUL, Fault, Health)
│   ├── heads_report_DS01_fc.json       # Flight condition classification head on DS01
│   ├── heads_report_DS02.json          # Multi-task heads benchmark on DS02
│   ├── heads_report_DS02_fc.json       # Flight condition classification head on DS02
│   ├── readout_study_DS01.json         # RUL readout architectures comparison (Ridge/TCN/TSMixer)
│   ├── readout_study_DS02.json         # RUL readout architectures comparison on DS02
│   ├── tuned_run_ds01.json             # 3-seed tuned verification on DS01
│   ├── tuned_run_ds02.json             # 3-seed tuned verification on DS02
│   ├── interpretability_ds01.json      # Codebook stability and interpretability suite
│   └── interpretability_ds02.json      # Codebook stability and interpretability suite
│
├── mimii/                   # MIMII Industrial Acoustic Anomaly Detection Benchmarks
│   ├── mimii_fan_id_00_3seeds.json     # 3-seed evaluation across 7 arms on machine ID 00
│   ├── mimii_fan_id_02_3seeds.json     # 3-seed evaluation across 7 arms on machine ID 02
│   ├── mimii_fan_id_04_3seeds.json     # 3-seed evaluation across 7 arms on machine ID 04
│   ├── mimii_fan_id_06_3seeds.json     # 3-seed evaluation across 7 arms on machine ID 06
│   ├── mimii_fan_id_00_interpretability.json
│   ├── mimii_fan_id_02_interpretability.json
│   ├── mimii_fan_id_04_interpretability.json
│   └── mimii_fan_id_06_interpretability.json
│
├── conformal/               # Engine-Level Conformal Uncertainty Quantification
│   ├── conformal_rul_results.json      # Engine-level LOEO conformal RUL intervals
│   ├── loeo_and_seed_stability_results.json # 6-fold LOEO & 10-seed representation stability
│   ├── w16_loeo_sweep_results.json     # W=16 window duration sweep
│   ├── robustness_val_results.json     # Robustness validation under noise & degradation
│   └── robustness_val_K.json           # Codebook size K sensitivity sweep
│
└── ablations/               # Controlled Causal Ablations & Scientific Audits
    ├── causal_full_ablation_results.json    # Complete causal ablation table
    ├── causal_arm_e_results.json            # Causal filtering Arm E re-run
    ├── causal_dwell_vs_full_results.json    # Dwell-only vs full feature comparison
    ├── raw_plus_structured_results.json     # Raw sensor + structured feature combinations
    ├── aux_rul_experiment_results.json      # Auxiliary RUL loss pathway ablation
    ├── exp2_em_restart_sweep.json           # EM random restart stability study
    ├── exp3_cross_dataset.json              # Cross-dataset transfer evaluation
    ├── exp_w4_causal.json                   # Causal W=4 baseline run
    ├── exp_w16_causal.json                  # Causal W=16 baseline run
    ├── lambda_sweep_results.json            # Multi-task loss weight sweep
    ├── subspace_experiment_results.json     # Latent subspace decoupling study
    ├── revival_experiment_results.json      # Codebook dead-code revival study
    ├── tcn_loeo_study_results.json          # Shallow TCN architecture search
    ├── regularized_tcn_loeo_study_results.json # Regularized TCN architecture search
    ├── scientific_audit_results.json        # End-to-end scientific audit
    ├── event_information_audit_results.json # Discrete event information retention audit
    └── u3_u8_diagnostic_results.json       # Transfer failure diagnostic audit
```

## Python Loading Utility

```python
from results.loader import load_result

# Load any result file by filename (searches all subdirectories automatically):
data = load_result("ablation_report_DS01.json")
mimii = load_result("mimii_fan_id_00_3seeds.json")
conformal = load_result("conformal_rul_results.json")
```
