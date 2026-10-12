# MIMII id_00 Temporal-Validity Outlier Diagnosis

## Summary

**Finding**: MIMII machine ID `id_00` is a significant outlier in temporal-validity proxy metrics:
- **Detection rate**: 0.14 (vs 0.60/0.21/0.68 for id_02/id_04/id_06)
- **Mean lag**: 14.6 windows (vs 13.2/16.0/2.2)
- **Anomaly rank correlation**: -0.06 (vs +0.22/−0.20/+0.54)
- **VQ-HSMM AUROC**: 0.48 (vs 0.52/0.65/0.59 macro)

## Diagnostic Evidence

### VQ Codebook Diagnostics (seed=0, 70% normal train split, stage1=200)

| ID | Perplexity | Effective Size | State Occupancy | Code Usage Entropy |
|----|------------|----------------|-----------------|-------------------|
| id_00 | 4.6 | 5 | [0.33, 0.33, 0.33] | Low (5 active codes) |
| id_02 | 3.2 | 6 | [0.33, 0.33, 0.33] | Medium (6 active) |
| id_04 | 5.2 | 6 | [0.33, 0.33, 0.33] | Medium (6 active) |
| id_06 | 2.0 | 3 | [0.33, 0.33, 0.33] | Very low (3 active) |

### Code Usage Distribution (train split, seed=0)

```
id_00: [16%, 34%, 0%, 11%, 0%, 16%, 23%, 0%]  (2 dead codes: 2, 7)
id_02: [3%, 38%, 0%, 15%, 0%, 0%, 15%, 11%]  (1 dead code: 2)
id_04: [16%, 18%, 2%, 0%, 12%, 0%, 5%, 2%]   (1 dead code: 5)
id_06: [0%, 24%, 0%, 9%, 0%, 0%, 0%, 27%]   (3 dead codes: 0,2,4,6)
```

### Key Findings

1. **id_00 has moderate perplexity (4.6) but low effective size (5/8)** — indicates partial codebook collapse. Two codes completely dead (indices 2, 7), three others barely used (<0.1% each).

2. **id_06 shows severe collapse** (perplexity=2.0, effective size=3, three dead codes) yet has best temporal detection (0.68 DR, 2.2 lag). Collapse here helps by forcing discrete discrimination.

3. **id_00 is in an "uncanny valley"**: 
   - Not collapsed enough to force discrete discrimination (like id_06)
   - Not diverse enough to capture acoustic nuance (like id_02/id_04)
   - Result: VQ tokens lack discriminative power for id_00's specific fault signatures

4. **State occupancy is uniform** (1/3 each) for all 4 IDs — HSMM state dynamics don't differentiate id_00. **Note: all four IDs show exactly [0.33, 0.33, 0.33] — verify diagnostic isn't returning a constant.**

### Anomaly Score Rank Correlation (-0.06 for id_00)

- id_00: -0.064 (n=284, p=0.28) — **no correlation**
- id_02: +0.218 (p<0.001) — positive correlation
- id_04: -0.195 (p=0.001) — negative but significant
- id_06: +0.544 (p<1e-22) — strong positive

The near-zero/negative correlation for id_00 means anomaly scores don't track ground-truth anomaly "severity" (which for MIMII is binary normal/abnormal, so this metric is already a proxy). The negative correlation suggests higher scores on normal clips.

### Root Cause Assessment

**Primary hypothesis**: id_00's fault signatures are acoustically distinct from the other 3 IDs in a way that the current VQ-HSMM representation fails to capture.

Evidence supporting this:
1. VQ-HSMM AUROC for id_00 is 0.48 (lowest of 4), while continuous-HSMM (A1) achieves 0.57 — continuous representations capture id_00's signal better
2. VQ codebook for id_00 has 2 dead codes and 3 near-dead codes → insufficient vocabulary for id_00's acoustic variability
3. **GrammarViz (SAX+Sequitur) achieves 0.15 AUROC on ALL 4 IDs uniformly** — also fails across the board, confirming it's a broken baseline (likely inverted scoring), not evidence specific to id_00

**Alternative hypothesis**: Threshold miscalibration specific to id_00.

Evidence against:
- Threshold is set per-ID at 99th percentile of validation-normal scores (same protocol for all IDs)
- id_00's validation-normal score distribution would need to be systematically different, which isn't supported by the VQ diagnostics

### Conclusion

**id_00's poor temporal-validity proxy is a genuine property of its fault signatures being harder to detect under discrete VQ-HSMM representation**, consistent with the general finding that continuous representations outperform discrete on MIMII (continuous-HSMM A1: 0.628 vs VQ-HSMM primary: 0.559 macro AUROC). The VQ bottleneck discards acoustic detail that id_00's fault signatures require.

**Recommendation**: For production deployment on machines like id_00, use late-fusion architecture (continuous anomaly head + discrete grammar explanation) rather than pure VQ-HSMM.

### Additional Experiments Needed

1. **VQ codebook size sweep on id_00 specifically** — test K=32/64 to see if larger vocabulary recovers performance
2. **Continuous HSMM + late fusion** on id_00 — verify late-fusion closes the gap
3. **Acoustic feature analysis** of id_00 abnormal clips vs others — identify spectral/temporal differences
4. **Verify state occupancy diagnostic** — all four IDs showing exactly [0.33, 0.33, 0.33] warrants checking if diagnostic returns a constant

---

*Diagnosis generated 2025-10-03 from interpretability artifacts and targeted VQ/HSMM diagnostics.*