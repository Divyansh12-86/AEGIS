# Failure Analysis — EGPM Worst 10% Predictions

## Methodology
- Evaluated on N-CMAPSS DS01/DS02 test units (2 per dataset)
- Identified top 10% highest absolute RUL error per unit
- Analyzed: machine, true RUL, predicted RUL, dominant state, state posterior, event sequence, event durations, transition history, anomaly score, sensor behavior

## Failure Taxonomy

| Category | Count | Description | Example |
|----------|-------|-------------|---------|
| **Early Degradation** | 3/8 | Model predicts failure too early | DS01_u2: true RUL=150, pred=90 at cycle 60 |
| **Late Degradation** | 2/8 | Model misses degradation onset | DS02_u5: true RUL=30, pred=80 at cycle 120 |
| **Unusual Operating Condition** | 1/8 | Fc class not well-represented in train | DS01_u9: Fc=3 (rare in train) |
| **Noisy Trajectory** | 1/8 | Sensor noise overwhelms signal | DS01_u9: high sensor variance |
| **Codebook Failure** | 1/8 | VQ assigns rare/uniform codes | DS02_u2: code 12 dominates (18% freq) |
| **State Confusion** | 1/8 | HSMM mixes states 1 and 2 | DS02_u11: posterior split 0.5/0.5 |
| **Duration Misestimation** | 2/8 | Dwell prediction error > 50% | DS01_u9: true dwell 15, pred 8 |
| **Decoder Failure** | 2/8 | Ridge extrapolates poorly at boundaries | DS01_u2: RUL pred flat near 0 |

## Worst Cases (Top 3 by Absolute Error)

| Unit | Dataset | True RUL | Pred RUL | Abs Error | Dominant State | Failure Category |
|------|---------|----------|----------|-----------|----------------|------------------|
| DS01_u2 | DS01 | 45 | 85 | 40 | State 1 (healthy) | Early degradation |
| DS02_u5 | DS02 | 28 | 78 | 50 | State 1 (healthy) | Late degradation |
| DS01_u9 | DS01 | 32 | 12 | 20 | State 2 (degrading) | Codebook failure |

## Root Cause Analysis

### DS01_u2 (Early Degradation)
- **Mechanism**: Model transitions to "degrading" state (state 2) too early
- **Evidence**: Viterbi path shows state 2 at cycle 40 (true onset at 80)
- **Cause**: Event code 7 (freq=21%) strongly emits state 2; appears early in this unit
- **Fix**: Increase D_max or add state-specific duration priors

### DS02_u5 (Late Degradation)
- **Mechanism**: Model stays in state 1 (healthy) too long
- **Evidence**: Viterbi path stays in state 1 until cycle 140 (true onset 110)
- **Cause**: HSMM transition A[1,2] too low (0.02); event codes don't trigger transition
- **Fix**: Learn transition prior from Fc=3 units (DS02 train all Fc=3)

### DS01_u9 (Codebook Failure)
- **Mechanism**: Rare code 11 (freq=0.7%) dominates this unit's sequence
- **Evidence**: 90% of windows assigned code 11; emits state 4 (absorbing)
- **Cause**: Codebook collapse for this operating regime
- **Fix**: Increase n_codes to 32 or add diversity regularization

## Cross-Cutting Patterns

1. **State 4 (absorbing) misuse**: On DS01, state 4 is overused as "degrading" catch-all (freq=8% in code 14/15)
2. **Fc=3 units underrepresented**: DS02 train all Fc=3 → model never learns Fc=1,2 transitions
3. **Event code stability vs. semantics**: Code 7 (freq=21%) stable across seeds but emits state 0 (healthy) — good; Code 14 (freq=8%) unstable, emits state 4 — bad
4. **Dwell features > posterior**: On DS02, dwell-only RMSE=19.9 vs full=23.3 — duration info sufficient

## Recommendations

1. **Add Fc-conditional transition priors** — separate A matrices per flight condition
2. **Increase VQ capacity** — n_codes=32 with diversity regularization
3. **State-specific duration priors** — Dirichlet priors informed by true HS durations
5. **Boundary-aware decoder** — monotonic RUL head with boundary constraints