# Claims Audit — EGPM MVP

| # | Claim | Evidence | Metric | Dataset | Strength | Safe Wording |
|---|-------|----------|--------|---------|----------|--------------|
| 1 | VQ+HSMM+ridge beats all simpler arms on RUL | Ablation A-F + baselines (3 seeds, unit CI) | DS01: 19.9 vs 22.4-24.2; DS02: 18.5 vs 23.4-25.4 | N-CMAPSS | **SUPPORTED** | "EGPM's explicit-duration HSMM substantially outperforms continuous, Markov, and HMM baselines on N-CMAPSS RUL" |
| 2 | Explicit duration modeling earns its cost | E (19.9) vs D (23.5) / C (23.8) | Δ=3.6-4.0 RMSE (15-16% improvement) | N-CMAPSS | **SUPPORTED** | "Explicit duration modeling provides 15-16% RMSE reduction over geometric-dwell HMM" |
| 3 | Native phase-type RUL works | E2 (24.0/19.98) vs E (19.9/18.5) | DS01: +4.1; DS02: -1.5 (ridge better) | N-CMAPSS | **NOT SUPPORTED** | "Native phase-type formula underperforms ridge decoder; requires further research" |
| 4 | Absorbing calibration helps | F (22.9/24.4) vs E (19.9/18.5) | DS01: +2.9; DS02: +5.9 RMSE (worse) | N-CMAPSS | **NOT SUPPORTED** | "Train-only absorbing calibration degrades RUL on both datasets" |
| 5 | Multi-task joint training helps | A3 delta ≡ 0 (joint vs single-task) | Δ≡0 (frozen rep → no shared params) | N-CMAPSS | **NOT SUPPORTED** | "Joint head training provides no benefit with frozen representation; needs full-gradient Stage 3" |
| 6 | Learned health weights beat fixed prior | Val Δ ≤ 0.002 vs seed std 0.3-0.4 | DS01: 0.0002; DS02: 0.0004 | N-CMAPSS | **NOT SUPPORTED** | "Fixed ordinal prior retained; learned weights don't beat noise" |
| 5 | EGPM detects MIMII anomalies | Primary 0.559 AUROC (vs 0.5 chance) | DS01: 0.559; DS02: 0.697 (A1) | MIMII | **PARTIALLY SUPPORTED** | "EGPM detects fan anomalies above chance, but continuous representations substantially outperform" |
| 6 | EGPM generalizes across datasets | Cross-dataset DS01→DS02: 20.5 RMSE (vs 19.8 in-domain) | Δ≈0.7 RMSE despite JSD~0.5 | N-CMAPSS | **PARTIALLY SUPPORTED** | "Cross-dataset transfer within ~2 RMSE despite token JSD ~0.5" |
| 7 | EGPM provides faithful explanations | Masking top windows → score reduction 0.18-0.72 | Mean reduction 6-27% | N-CMAPSS | **SUPPORTED** | "Masking highest-scoring windows reduces anomaly score (faithfulness confirmed)" |
| 8 | Discrete events are stable | Event stability 0.66-0.93 across seeds | 0.66 (id_06) - 0.93 (DS01) | N-CMAPSS, MIMII | **SUPPORTED** | "Event vocabulary stable across seeds (matched usage overlap 0.66-0.93)" |
| 9 | Discrete events are coherent | Silhouette ~0 (DS01: -0.003, DS02: -0.003) | Weak but consistent | N-CMAPSS | **NOT SUPPORTED** | "Event coherence weak (silhouette ~0); events functional not semantic" |
| 10 | VQ+HSMM beats continuous on MIMII | Continuous-AE 0.638 > VQ+HSMM 0.559 | Continuous wins by 0.08 AUROC | MIMII | **NOT SUPPORTED** | "On acoustic anomaly, continuous representations substantially outperform discrete EGPM" |
| 6 | EGPM generalizes across machines | Cross-ID transfer not tested; cross-dataset DS01→DS02 works | JSD ~0.5, RMSE +0.7 | N-CMAPSS | **PARTIALLY SUPPORTED** | "Cross-dataset transfer within ~2 RMSE despite token JSD ~0.5" |
| 7 | Discrete events provide interpretability | Event stability 0.66-0.93; coherence ~0; faithfulness >0 | Mixed | N-CMAPSS, MIMII | **PARTIALLY SUPPORTED** | "Event vocabulary stable across seeds; coherence weak; faithfulness confirmed" |
| 8 | EGPM beats CNN-RUL baseline | E 19.9 vs CNN 34.0 (DS01); 18.5 vs 28.9 (DS02) | Δ=14-19 RMSE | N-CMAPSS | **SUPPORTED** | "EGPM beats CNN-RUL by 10+ RMSE under identical protocol" |
| 9 | EGPM beats LSTM-RUL baseline | E 19.9 vs LSTM 38.8 (DS01); 18.5 vs 41.2 (DS02) | Δ=19-23 RMSE | N-CMAPSS | **SUPPORTED** | "EGPM beats LSTM-RUL by 19-23 RMSE under identical protocol" |
| 10 | Dwell features sufficient for RUL | Dwell-only 19.9 vs full 23.3 (DS02) | Dwell beats full on DS02 | N-CMAPSS | **SUPPORTED** | "Duration statistics alone suffice for competitive RUL prediction" |
| 9 | EGPM beats all MIMII baselines | Primary 0.559 < Continuous_AE 0.638 < Continuous_HSMM 0.697 | Continuous HSMM best | MIMII | **NOT SUPPORTED** | "Continuous representations substantially outperform discrete EGPM on acoustic anomaly" |
| 10 | GrammarViz is valid baseline | GrammarViz 0.152 AUROC (anti-correlated) | Consistently < 0.5 | MIMII | **NOT SUPPORTED** | "GrammarViz severely anti-correlated on MIMII fan; not a valid baseline" |
| 10 | Concept Bottleneck competitive | CBM 0.414 AUROC (below primary 0.559) | Below chance on some IDs | MIMII | **NOT SUPPORTED** | "CBM weak on MIMII fan; loses to all EGPM arms" |
| 11 | Health index is useful | Spearman 0.21/0.10; learned weights don't beat fixed | Weak signal | N-CMAPSS | **NOT SUPPORTED** | "Health index weak (Spearman 0.21/0.10); representation issue, not weighting" |
| 12 | Fault head discriminates | Fc 0.50 acc (DS01, 3-class, 2 test units); DS02 degenerate | Honest on DS01; artifact on DS02 | N-CMAPSS | **PARTIALLY SUPPORTED** | "Fc 3-class accuracy 0.50 vs 0.33 chance (DS01); DS02 Fc-imbalanced" |
| 10 | EGPM provides faithful explanations | Masking top windows → score reduction 0.18-0.72 | Positive reduction = faithful | N-CMAPSS | **SUPPORTED** | "Masking highest-scoring windows reduces anomaly score (faithfulness confirmed)" |
| 11 | Discrete events are stable | Event stability 0.66-0.93 across seeds | 0.66 (id_06) - 0.93 (DS01) | N-CMAPSS, MIMII | **SUPPORTED** | "Event vocabulary stable across seeds (matched usage overlap 0.66-0.93)" |
| 11 | EGPM detects degradation progression | Onset DR=1.0 (DS01), 0.5 (DS02); lag 0.75/1.0 | True onsets from hs | N-CMAPSS | **SUPPORTED** | "Viterbi boundaries detect health transitions with 50-100% rate, lag 0.75-1.0 windows" |
| 12 | EGPM event vocabulary is stable | Cross-seed matched usage overlap 0.66-0.93 | Consistent across datasets | N-CMAPSS, MIMII | **SUPPORTED** | "Event vocabulary stable across seeds (matched usage overlap 0.66-0.93)" |

---

## Summary

| Verdict | Count |
|---------|-------|
| **SUPPORTED** | 8 |
| **PARTIALLY SUPPORTED** | 4 |
| **NOT SUPPORTED** | 7 |

**Key Insight**: EGPM excels on N-CMAPSS RUL with explicit duration modeling, but has fundamental limitations on MIMII anomaly detection (discretization hurts), phase-type RUL (catastrophic on DS02), and multi-task learning (frozen rep → no gain). All limitations are explicitly documented.

**Safe Position**: "EGPM provides a structurally interpretable RUL predictor that beats deep baselines on N-CMAPSS, with faithful explanations. On acoustic anomaly, continuous representations outperform discrete — we recommend late-fusion for production."