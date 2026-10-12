# Limitations — EGPM MVP

## Fundamental Limitations

### 1. Phase-Type RUL Catastrophic Failure on DS02
- **Observation**: Native phase-type RUL RMSE = 310.4 (DS02) vs ridge 23.3
- **Root Cause**: Absorbing state unreachable from some transient states in fitted HSMM; fundamental matrix (I-Q)⁻¹ ill-conditioned
- **Impact**: Cannot use native formula for production; must use ridge decoder
- **Mitigation**: Add absorbing reachability constraint during HSMM fitting; investigate geometric-dwell encoding

### 2. Discretization Loses Acoustic Information on MIMII
- **Observation**: Continuous-HSMM (A1) AUROC=0.628 vs VQ+HSMM=0.559 (macro)
- **Root Cause**: VQ bottleneck (K=8) discards fine-grained spectral detail needed for acoustic anomaly discrimination
- **Impact**: EGPM primary architecture unsuitable for acoustic anomaly tasks without late-fusion
- **Mitigation**: Late-fusion architecture (continuous anomaly head + discrete grammar explanation)

### 3. Severely Underpowered N-CMAPSS Statistics
- **Observation**: Only 2 independent test units per dataset
- **Consequence**: Bootstrap CIs wide (e.g., DS01 E: [19.4, 20.3] vs F: [21.0, 24.7] overlap); Wilcoxon underpowered (p=0.5 with 2 units)
- **Impact**: Cannot claim statistical significance; effect sizes reported instead
- **Mitigation**: Cross-dataset transfer (DS01→DS02) provides complementary evidence; explicit statement in all reports

### 4. Health Index is Not Clinically Useful
- **Observation**: Spearman vs hs = 0.21 (DS01), 0.10 (DS02)
- **Root Cause**: Latent states don't align with auxiliary health states; learned weights don't beat fixed prior (val Δ ≤ 0.002 vs seed std 0.3-0.4)
- **Impact**: Health head cannot be used for clinical decision-making
- **Mitigation**: Treat as weak degradation proxy only; investigate better latent supervision

### 5. Fault Head on hs is Constant Predictor
- **Observation**: Run-level hs majority class = 1 for ALL units → fault accuracy 1.0 is constant-predictor artifact
- **Root Cause**: Early life dominates every run (>50% cycles in hs=1)
- **Impact**: Reported 1.0 accuracy is NOT discrimination evidence
- **Mitigation**: Use Fc (flight condition) as richer 3-class target; Fc results: DS01 0.50 acc (vs 0.33 chance), DS02 1.0 but Fc-imbalanced

### 6. Fc Classification Imbalanced on DS02
- **Observation**: DS02 train units all Fc=3 → degenerate single-class split
- **Consequence**: Fc accuracy 1.0 is constant predictor; DS01 0.50 is honest (3-class, 2 test units)
- **Impact**: Cannot evaluate fault head on DS02
- **Mitigation**: Use DS01 for fault evaluation; collect balanced data

### 7. Native Phase-Type RUL Loses to Ridge on Both Datasets
- **Observation**: DS01: 23.4 vs 18.1; DS02: 25.6 vs 22.8 (ridge better on both, causal)
- **Root Cause**: Fundamental matrix (I-Q)⁻¹ assumes reachability; mid-dwell correction insufficient
- **Impact**: Phase-type formula not production-ready; use ridge decoder

### 8. Absorbing Calibration Hurts RUL
- **Observation**: Arm F (calibrated) worse than E on both datasets (DS01: 19.3 vs 18.1; DS02: 28.2 vs 22.8, causal)
- **Root Cause**: Single-knob calibration overfits to train-unit lifespan distribution
- **Impact**: Calibration knob should be disabled in production

### 9. MIMII Anomaly: Continuous Beats Discrete
- **Observation**: Continuous-AE (0.621) and Continuous-HSMM (0.628) both beat VQ+HSMM (0.559)
- **Implication**: VQ bottleneck discards fine-grained spectral info critical for acoustic anomaly
- **Implication**: For acoustic anomaly, continuous representations preferred

### 10. GrammarViz Severely Anti-Correlated
- **Observation**: GrammarViz AUROC = 0.152 (macro) — consistently below 0.5
- **Root Cause**: SAX + Sequitur novelty scoring inverted on this data
- **Impact**: GrammarViz not a valid baseline for MIMII fan

### 11. Concept Bottleneck Weak on MIMII
- **Observation**: CBM AUROC = 0.414 (macro) — below chance on some IDs
- **Root Cause**: k-means concepts (K=8) on raw windows lose temporal structure entirely
- **Impact**: CBM not competitive without temporal model

### 12. Event Coherence Weak (Silhouette ~0)
- **Observation**: Silhouette scores ~0 on both datasets (DS01: -0.003, DS02: -0.003)
- **Implication**: VQ events don't form well-separated clusters in sensor space
- **Implication**: Events are functional (useful for HSMM) not semantic

### 13. A3 Multi-Task Delta ≡ 0
- **Observation**: Joint head training vs single-task copies → delta ≡ 0
- **Root Cause**: Frozen representation → no shared trainable params → AdamW per-param updates decouple
- **Implication**: Multi-task coupling requires full-gradient Stage 3 (encoder flow)

### 14. Only 4 MIMII Machine IDs
- **Observation**: 4 IDs (id_00, 02, 04, 06) — limited generalization evidence
- **Consequence**: ID-specific effects dominate (id_00 hard, id_06 easy)
- **Mitigation**: Cross-ID transfer experiments needed

### 15. VQ Codebook Underutilization
- **Observation**: DS01 uses 11/16 codes; DS02 uses 8/16 codes; MIMII K=8 likely too small
- **Consequence**: Unused capacity; potential collapse for rare operating regimes
- **Mitigation**: Increase K, add diversity regularization

### 17. State-Health Label Misalignment
- **Observation**: HSMM infers states 0-4 but true health labels are 1-2
- **Consequence**: Latent states don't map to auxiliary labels
- **Implication**: Health index via fixed ordinal prior is inherently limited

### 18. Training Stability Variance
- **Observation**: EM restarts needed for stability (3 restarts → seed std 2.6→0.5)
- **Implication**: Single EM run unreliable; must report multiple restarts

### 19. No Cross-Fleet Transfer Validation
- **Observation**: Only tested DS01↔DS02 and MIMII fan
- **Gap**: No air compressor, ToyADMOS, or cross-fleet validation
- **Implication**: Generalization claims limited to these datasets

### 18. No Edge/Deployment Optimization
- **Gap**: No quantization, distillation, beam-pruned inference
- **Implication**: Not production-ready for edge deployment

### 20. No Natural Language Explanations
- **Gap**: Structured explanation object exists but no NL rendering layer
- **Implication**: Engineers need NL interface for adoption

---

## Summary: What Works vs What Doesn't

| Works | Doesn't Work |
|-------|--------------|
| VQ+HSMM+ridge RUL (18.1/22.8 RMSE causal) | Native phase-type RUL |
| Explicit duration (15% gain) | Absorbing calibration |
| Dwell features for RUL | Joint multi-task (frozen rep) |
| Faithfulness masking | Learned health weights |
| Event stability (0.66-0.93) | Event coherence (silhouette ~0) |
| Event vocabulary stability | MIMII discrete anomaly |
| Cross-dataset transfer | Health index (weak) |
| | Fc on DS02 (degenerate) |
| | GrammarViz on MIMII |
| | Phase-type RUL DS02 |