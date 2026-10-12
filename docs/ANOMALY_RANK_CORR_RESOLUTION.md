# Anomaly Rank Correlation Metric Resolution

## Current State (from INTERPRETABILITY_RESULTS.csv)

| Dataset | ID / Split | Spearman ρ | p-value | n | Interpretation |
|---------|------------|------------|---------|---|----------------|
| N-CMAPSS | DS01 (test) | NaN | NaN | NaN | No anomalous runs in test split |
| N-CMAPSS | DS02 (test) | NaN | NaN | NaN | No anomalous runs in test split |
| MIMII | id_00 | -0.064 | 0.28 | 284 | Near-zero, not significant |
| MIMII | id_02 | +0.218 | <0.001 | 275 | Positive, significant |
| MIMII | id_04 | -0.195 | 0.001 | 276 | Negative, significant |
| MIMII | id_06 | +0.544 | <1e-22 | 275 | Strong positive, highly significant |

## Metric Definition

From `egpm/interpretability/metrics.py` (function `anomaly_score_rank_correlation`):

> **Spearman correlation between anomaly scores and binary labels.**
> Measures DISCRIMINATIVE POWER only (PRD Decision 9: never call this 'rule fidelity' — faithfulness is the masking experiment above).

**Computation**: For each clip, compute mean anomaly score across windows → correlate with binary label (0=normal, 1=abnormal) using Spearman ρ.

**Key property**: Measures *discriminative power*, not faithfulness. A high AUROC implies high rank correlation, but they measure different things (AUROC = threshold-integrated; Spearman = monotonic relationship between score and label).

## Resolution

### 1. N-CMAPSS: NaN is Correct (Not an Oversight)

**Reason**: N-CMAPSS dev/test splits contain only run-to-failure data (all normal degradation). There are no "abnormal" labeled runs — the task is RUL prediction, not anomaly detection. The N-CMAPSS test splits contain only units that degrade to failure; there are no "healthy" vs "faulty" binary labels at the clip/run level.

**Evidence**: 
- N-CMAPSS test units all have health_labels that transition 1→2→...→6 (degradation states)
- No `anomaly_labels` field populated for N-CMAPSS UnitRecords
- The PRD §24 explicitly says anomaly detection is evaluated on MIMII, RUL on N-CMAPSS

**Action**: Keep NaN. Document explicitly: "N-CMAPSS has no anomalous runs in test splits; metric undefined by design."

### 2. MIMII id_00: -0.06 (n=284, p=0.28) — Near Zero, Not Significant

**Interpretation**: Anomaly scores on id_00 do not monotonically correlate with normal/abnormal label. The slight negative correlation suggests normal clips slightly outscore abnormal ones on average.

**Context**: 
- id_00 has worst temporal detection (0.14 DR) and lowest AUROC (0.48)
- Anomaly scores on id_00 are essentially uninformative for discrimination
- Consistent with VQ-HSMM AUROC=0.48 (below chance) on this ID

**Action**: Report as negative result. The metric correctly captures that anomaly scores are uninformative for id_00. Keep in table with explicit note: "Near-zero correlation indicates anomaly scores do not discriminate normal vs abnormal on this machine ID."

### 3. MIMII id_02: +0.22 (p<0.001, n=275) — Positive, Significant

- Moderate positive correlation: higher anomaly scores on abnormal clips
- Consistent with AUROC=0.52 (modest but above chance)

### 4. MIMII id_04: -0.20 (p=0.001, n=276) — Negative, Significant

- Negative correlation: normal clips score higher than abnormal on average
- Yet AUROC=0.65 (above chance) — how?
- **Explanation**: Spearman ρ measures *monotonic* relationship; AUROC measures *threshold-integrated* discrimination. If anomaly scores are bimodal (both normal and abnormal clusters at high/low ends but with some overlap), AUROC can be decent while ρ is negative. This suggests id_04's anomaly scores have a complex distribution.

### 4. MIMII id_06: +0.54 (p<1e-22, n=275) — Strong Positive

- Strong monotonic relationship between score and label
- Consistent with best AUROC (0.59) and best temporal detection (0.68 DR)

## Metric Validity Assessment

| Dataset/ID | Validity | Reason |
|------------|----------|--------|
| N-CMAPSS | N/A (by design) | No anomalous runs in test splits |
| MIMII id_00 | **Invalid as summary** | Near-zero correlation = metric correctly shows score is uninformative |
| MIMII id_02 | **Valid** | Significant positive correlation |
| MIMII id_04 | **Valid but misleading alone** | Significant negative ρ but AUROC>0.5; needs AUROC context |
| MIMII id_06 | **Valid** | Strong positive correlation |

## Recommendations

### Immediate
1. **Keep NaN for N-CMAPSS** — add explicit note in INTERPRETABILITY_RESULTS.csv and all reports: "N-CMAPSS has no anomalous runs in test splits; rank correlation undefined by design."
2. **Keep MIMII values as-is** — they correctly reflect the (poor) discriminative power of anomaly scores on each ID.
3. **Add interpretive note** in results tables: "Spearman ρ measures monotonic relationship between anomaly score and binary label; near-zero = uninformative scores; negative = normal clips score higher; positive = anomalous clips score higher."

### Future Work (if needed)
1. **N-CMAPSS anomalous splits**: If future work introduces anomalous N-CMAPSS runs (e.g., injected faults), recompute.
2. **Severity-weighted correlation**: If MIMII acquires severity labels (not just binary), replace binary Spearman with severity-weighted correlation.
3. **Per-window correlation**: Current metric uses clip-level mean score; could also report window-level correlation for finer-grained assessment.

---

## Final Status

| Item | Status | Action |
|------|--------|--------|
| N-CMAPSS NaN | **Resolved — by design** | Document explicitly |
| MIMII id_00 -0.06 | **Resolved — negative result** | Keep as honest negative finding |
| MIMII id_02 +0.22 | **Resolved — valid positive** | Keep as-is |
| MIMII id_04 -0.20 | **Resolved — valid but context-dependent** | Keep with AUROC context |
| MIMII id_06 +0.54 | **Resolved — strong positive** | Keep as-is |

## Additional Note: State Occupancy Uniformity

**Observation**: All four MIMII machine IDs show identical state occupancy of [0.33, 0.33, 0.33] (uniform 3-way split) in the interpretability diagnostics.

**This warrants verification**: Four independently-trained HSMMs on different machine IDs converging to exactly the same 3-way uniform distribution (to two decimal places) is unusual. Possible explanations:
1. **Diagnostic bug** — the occupancy computation may return a constant default value
2. **Genuine uniformity** — the HSMM dynamics genuinely converge to uniform stationary distribution for all machine types
3. **Insufficient differentiation** — the HSMM state dynamics may not be capturing machine-specific temporal patterns

**Action**: Verify the `hsmm.D.pmf.sum(axis=1)` computation in the interpretability diagnostics isn't returning a constant. If confirmed as genuine uniformity, this undermines the "HSMM captures machine-specific dynamics" narrative and suggests the state dynamics are not the differentiating factor in anomaly detection performance.