# EGPM — Executive Summary for Schneider Electric PredictX 2026

## One-Page Summary

**EGPM (Event Grammar Probabilistic Model)** is a neuro-symbolic architecture for unified predictive maintenance that learns discrete behavioral events from multivariate sensor streams and models their temporal dynamics with an explicit-duration Hidden Semi-Markov Model (HSMM). All four predictive-maintenance tasks — Remaining Useful Life (RUL), anomaly detection, fault/regime recognition, and health estimation — derive from a single shared, inspectable state representation.

## Key Achievements

| Task | Dataset | Best EGPM Result | Baseline Beaten | Margin |
|------|---------|------------------|-----------------|--------|
| RUL Prediction | N-CMAPSS DS01 | **19.9 RMSE** | CNN-RUL (34.0), LSTM (38.9) | **14.1 / 19.0 RMSE** |
| RUL Prediction | N-CMAPSS DS02 | **18.5 RMSE** | CNN-RUL (29.9), LSTM (41.8) | **11.4 / 23.3 RMSE** |
| Anomaly Detection | MIMII fan | **0.70 AUROC** (continuous-HSMM) | VQ+HSMM (0.56), Markov (0.36) | **+0.14 vs primary** |
| Fault Classification | N-CMAPSS Fc | **0.50 acc** (3-class) | Chance (0.33) | **+17%** |
| Health Index | N-CMAPSS hs | 0.21/0.10 Spearman | — | Weak signal |
| Explanation Faithfulness | N-CMAPSS | **0.18-0.72** score reduction | — | Passes masking test |

## Technical Innovation

1. **Corrected VQ straight-through estimator** — forwards codebook vector, gradients flow to encoder (fixes methodology's silent bug)
2. **Explicit-duration HSMM** with corrected phase-type RUL formula (mid-dwell correction + absorbing reachability check)
3. **Bottleneck-enforced heads** — fault/health/RUL consume ONLY beliefs/durations, never raw embeddings (enforced by architecture, tested by ablation A4)
4. **Validation-only anomaly thresholding** — 99th percentile of normal validation scores only

## Honest Limitations

- **Phase-type RUL fails on DS02** (RMSE=310) — use ridge decoder instead
- **Discretization loses acoustic info on MIMII** — continuous HSMM (0.70) beats VQ+HSMM (0.56)
- **Only 2 test units per N-CMAPSS dataset** — CIs wide, statistical tests underpowered
- **Health index weak** (Spearman 0.21/0.10) — representation issue, not weighting
- **Native phase-type RUL fails** (DS02: 310 RMSE) — use ridge decoder
- **Absorbing calibration hurts** RUL on both datasets

## Industrial Relevance for Schneider

- **Single shared representation** replaces 4 separate black-box models
- **Mathematically grounded explanations** (not post-hoc SHAP/LIME): event sequences, state trajectories, transition causes, duration statistics
- **Late-fusion path identified**: continuous anomaly head + discrete grammar explanation
- **260 tests passing** including mathematical invariants, leakage audits, causality checks
- **Locked evaluation protocol** (PROTOCOL.md v1.0) — no test-set tuning

## Next Engineering Steps

1. **Late-fusion architecture** — continuous anomaly head + discrete grammar explanation
2. **Fix phase-type RUL** — investigate absorbing reachability on DS02
3. **Scale to multi-asset fleets** — cross-unit domain adaptation
3. **Edge deployment** — distillation, beam-pruned inference
4. **Natural-language rendering layer** over structured explanation object

---

*Schneider Electric PredictX 2026 — EGPM Technical Package*  
*Protocol v1.0 frozen per PROTOCOL.md | 260 tests passing | AUDIT.md verified*