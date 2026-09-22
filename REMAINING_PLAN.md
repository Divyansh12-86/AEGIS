# Implementation Plan — finishing all remaining EGPM tasks

Status: plan, 2026-09-18. Baseline state: MVP (M1–M11) complete, 237 tests green,
real results on N-CMAPSS + MIMII fan. This plan covers everything the PRD leaves
open, in dependency order.

## Remaining task inventory (from PRD + current README findings)

1. **Health-index decision (PRD §14)** — the current fixed-ordinal weighting is a
   *documented weak signal* (Spearman 0.21 DS01 / 0.10 DS02). The PRD says: resolve
   (a) fixed ordinal prior vs (b) learned weights with monotonicity regularizer,
   on validation. Must run the experiment, not just pick.
2. **Multi-task training (MVP §15 scope, ablation A3)** — heads currently fitted
   per task; joint training + A3 comparison is unimplemented.
3. **Cross-dataset transfer interpretation** (exp3 harness exists; the "use it
   to probe event-code universality" follow-up isn't run).
4. **Health/temporal validity with real signals** — suite currently skips
   `temporal_validity` on MIMII (no onset labels); synthetic onset-proxy for
   a real run.
5. **Full baselines on MIMII** — OmniAnomaly flag is Phase-2; CBM (concept
   bottleneck) exists but only synthetic-tested; MIMII run would close §21.
6. **Fault-class generalization beyond N-CMAPSS hs** — currently hs is 2-class
   at run level (validates wiring only). N-CMAPSS flight-envelope variants give
   a richer target; heads_run can consume those.
7. **MC-based RUL interval** (PRD §32's Phase-2 carve-out): simulation-based
   RUL uncertainty by Monte-Carlo through the phase-type distribution.
8. **Streaming/partial-history anomaly scoring** — currently full-clip. PRD
   §25's open question: fixed-lag Viterbi re-decode vs online smoother on MIMII
   (needs `t0..t` inference API + latency measurement, do it honestly).
9. **OmniAnomaly proper** — ablation family behind a flag; the numbers from
   `cnn_rul.WindowAutoEncoder` currently stand in.
10. **Config management** — PRD §30 wants versioned per-run configs; currently
    stitched into experiment JSONs. Replace with a tiny `configs/` dir if >1
    (not before). Never add heavy config machinery for a value that lives in
   exactly one place.
11. **ToyADMOS generalization check** — secondary dataset, PRD§3.2.
12. **Air-compressor seed dataset** — compressor fault labels would exercise
    the fault head on multi-class data (PRD §13 mentions it).

## Phases

### Phase A — close the documented gaps (hours, no new architecture)
- **A1. Health-index a/b experiment** (§14): implement
  `health_head(learnable=True)` — logistic-regression weights `w_m` with a
  monotonicity penalty hinge(min(0, w_m - w_{m+1})); train on train-fra
  (learnable=True) — logistic-regression weights `w_m` with a monotonicity
  regularizer hinge(min(0, w_m - w_{m+1})); train on the train fraction only,
  compare Spearman vs the fixed baseline on VALIDATION (§23).
  File: `egpm/health/health_head.py`, new test asserting monotonicity
  penalty zero for monotone weights and > 0 for non-monotone.
  If (b) wins by > 3 Spearman std-devs, switch the default in README.
- **A2. Multi-task trainer + A3 ablation**: `Trainer.stage3_joint(heads, Xtr,
  task_logits)` — losses: anomaly BCE + fault CE + health MSE + RUL huber;
  A3 = single-task copies on the same splits, per head. Compare per-task
  metric; keep RQ5's answer in `ablation_report_DS01.json`.
- **A3'. OmniAnomaly flag**: `baselines/omnianomaly_v2.py` (recurrent,
  stochastic, planar-normalizing-flow likelihood) as an optional arm — gated
  behind `run_omnianomaly=True` in the driver reports. ~200 lines, test it
  contract-compatible with `WindowAutoEncoder.anomaly_scores`.
- **A4'. CBM on MIMII**: run `concept_bottleneck.py` against the fan set; metric: AUROC and concept-activation JSD vs token usage.

### Phase B — strengthen the real-MIMII evidence (hours)
- **B1. Health state tracking across runs** on `interpretability_ds*.json`:
  add mean dwell per HSMM state as a health proxy and check Spearman
  correlation (cheap, no new training).
- **B2. MIMII temporal validity via proxy onsets**: the suite skips temporal
  checks because MIMII lacks in-clip onset labels; use the anomaly *first
  detection lag* as a proxy (anomalous clips are anomalous from t=0, so the
  first window whose cumulative score exceeds the per-id threshold is a
  pseudo-onset) — add as `temporal_proxies` in `mimii_fan_*_interpretability.json`
  with an explicit caveat note.
- **B3. Per-id health trajectories**: run the health head per-id on MIMII
  and plot/record the curve shape vs the anomaly-flagged runs.

### Phase C — RUL uncertainty + streaming (medium)
- **C1. Monte-Carlo RUL intervals** (PRD §12 note): simulate N paths through
  the fitted HSMM until absorption; report mean/median/5-95 pct interval under
  each explainable caller's current state belief. Files:
  `egpm/rul/phase_type_rul.py` (add `simulate_rul(hsmm, state_belief, n=1000)`).
  Test: synthetic closed-form 2-state chain — interval coverage vs analytic.
  Runs on N-CMAPSS: add `interval_coverage_95` to `ablation_report_DS*.json`.
- **C2. Fixed-lag streaming decoder**: Viterbi re-decode on the last L windows
  per new window (parameter `lag_window=64`); measure AUROC-vs-latency on
  MIMII and Viterbi-score divergence. Add a `StreamingAnomalyScorer` wrapper
  around `AnomalyScorer.score_sequence`; test that early anomaly detection
  rate on clip-level data rises.

### Phase D — more baselines on real MIMII (one at a time)
- **D1. GrammarViz** (`egpm/baselines/grammarviz.py`) on fan logs.
- **D2. SAX on raw windows** + first-order Markov, per PRD §21's extended list.
- **D3. Anomaly Transformer** is Phase-2; only run if A/B/C look strong.
- **D4. AutoRUL** only if the team decides it's needed; not before.

### Phase E — generalization targets
- **E1. ToyADMOS** (PRD§3.2) generalization check under the Phase-2 scope —
  evaluate the frozen fan grammar on ToyADMOS without retraining; record
  cross-domain AUROC only.
- **E2. Air-compressor dataset** — if available, fault-head multi-class test.
- **E3. Cross-machine transfer within MIMII** (fan→pump): reuse pump data if
  present.

### Phase F — engineering/robustness at scale
- **F1. Config version bookkeeping**: turns `exp*.json`, `mimii_fan_*.json`
  into config-keyed runs with a `config_hash`; implement the tiniest version
  (no Hydra until YAML actually has two configs).
- **F2. Memory profile runs** with `resource` in `scripts/mimii_driver.py`;
  write peak RSS into the JSON artifacts.
- **F3. Parallel per-(id, seed) harness** — embarrassingly parallel; the
  current driver is sequential. ~4x wall-clock win on a single box.

### Phase G — documentation (after the numbers land)
- **G1.** README "Progress" section synthesized from the actual JSONs (not
  hand-edited), matching the current table format.
- **G2.** `docs/mimii_decisions.md` short memo justifying the delta-features /
  mean-channel fold choices (they revised the numbers substantially).

## Order of execution (laziness-ranked)

1. **First**: A1, B1-B3, C1, C2, F1. All tiny code, high signal, closing
answers to questions the PRD explicitly opens.
2. **Then**: A2/A3' paired with D1/D2 as data supports.
3. **E** only after the current pipeline is demonstrably stable on the
   primary datasets.
4. **Never build unless the current evidence demands it**: OmniAnomaly full
   port (D3), hierarchical PCFG, LLM rendering layer, edge deployment
   optimizations, streaming for production.

## Risks + how each phase mitigates them

- **"Continuous HSMM beats everything"**: the A1 result is real and the PRD
  already acknowledges this happens (continuous encoder features win on RUL on
  N-CMAPSS too). The MIMII task needs a real answer (not just synthetic) of
  whether late-fusion continuous heads + grammar explanation is sufficient.
  We run paired-Wilcoxon per-id (already doing) and report the raw numbers.
- **Sequence length semantics**: with 19-token clips the HSMM barely learns
  duration structure. The `window_length=16` default matters; add explicit
  sensitivity to window length in the driver's logs (it's a knob anyway).
- **Poor CBM/AE convergence**: ae_steps=200 is visibly low; raising to
  ~800 adds ~30-60s/arm — schedule also fine but mention it and revisit only
  if the AE arm beats the primary arm again.

## What stays out of scope (per PRD + current README)

- PLC/production deployment, multi-asset fleet ops, autonomous control.
- LLM used as the primary interpretability mechanism (rendering only).
- OmniAnomaly/Attention-LSTM/TranAD unless A/B/C fail.
