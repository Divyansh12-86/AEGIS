# Exploratory W=32 Downsampled Run (Not CruiseBench-Comparable)

> **CruiseBench reproduction not completed (memory); EXPLORATORY_W32.md is not benchmark-comparable.**
>
> - Window length: W=32 (protocol requires W=256)
> - Downsampled data (protocol requires native 1Hz)
> - No CPM masking implemented
> - Full CruiseBench (W=256, S=10, native 1Hz): hit OOM (~4.7 GB feature memory exceeded environment)
> - Results are exploratory only — not benchmark-comparable and not partial progress toward CruiseBench.

---

## 1. Protocol Constraints & Environment Limits

Full CruiseBench protocol execution failed due to memory constraints:
- **Full CruiseBench specification** (W=256, S=10, native 1Hz): ~4.7 GB feature memory — exceeds execution environment memory limits (OOM).
- **Intermediate attempt** (W=128, S=10 with downsampling): sequence too short after downsampling.
- **Exploratory scratchpad run** (W=32, S=4 with downsampling): yielded 12–13 windows per test engine.

Because window length was truncated by 8×, data was downsampled, and CPM masking was omitted, this run does **not** reproduce CruiseBench and must not be compared to published CruiseBench benchmarks.

---

## 2. Exploratory Scratchpad Numbers (DS01, Downsampled, W=32, S=4)

### RUL Prediction (Causal)
| Test Engine | RMSE | Windows |
|-------------|------|---------|
| DS01_u2 | 17.4 | 12 |
| DS01_u9 | 16.0 | 13 |

### Counterfactual Consistency Check
| Test Unit | CF1 (dwell=1) | CF2 (state-1) | n_windows |
|-----------|---------------|---------------|-----------|
| DS01_u2 | 91.7% | 91.7% | 12 |
| DS01_u9 | 92.3% | 92.3% | 13 |

*Note on Counterfactual numbers*:
- In this exploratory W=32 scratchpad run, n=12 (u2) and n=13 (u9) windows were evaluated.
- The validated Track B benchmark result is **CF1: 94.7–95.0%** (u2: 94.7%, u9: 95.0%, n=19/20) and **CF2: 85.0–100%** (u2: 100%, u9: 85.0%, n=19/20) evaluated at W=4, S=4 with causal filtering across 39 windows on 2 engines.
- No W=64 counterfactual run was executed.

---

## 3. Comparison with Validated Track B (W=4, S=4, Causal)

| Metric | Exploratory W=32 (Downsampled) | Track B Validated (W=4, S=4, Causal) |
|--------|--------------------------------|--------------------------------------|
| Protocol match | Non-comparable scratchpad | Project Standard (causal LOEO) |
| DS01_u2 RMSE | 17.4 | 13.5 |
| DS01_u9 RMSE | 16.0 | 17.6 |
| CF1 % increase | 91.7–92.3% (n=12/13) | 94.7–95.0% (n=19/20) |
| CF2 % increase | 91.7–92.3% (n=12/13) | 85.0–100% (n=19/20) |

---

## 4. Conclusion

CruiseBench reproduction was not completed due to environment memory limits (native 1Hz W=256 OOM). The exploratory W=32 run documented here was an internal check under downsampled conditions, not CruiseBench-compatible, and is not claimed as benchmark reproduction.

**Reference Scripts** (located in `scripts/exploratory/`):
- `scripts/exploratory/run_track_a_small.py` — exploratory run (W=32, S=4)
- `scripts/exploratory/run_track_a_proper.py` — full protocol attempt (OOM)
- `scripts/exploratory/run_track_a_chunked.py` — chunked attempt (OOM)
- `scripts/exploratory/run_track_a_compromise.py` — W=128 attempt (sequence too short)
