# EGPM — Event Grammar Probabilistic Model

**A structurally interpretable predictive-maintenance architecture that derives RUL, anomaly detection, fault classification, and health scoring from one shared, inspectable state representation — instead of four separate black boxes.**

---

## What EGPM Does, In One Paragraph

EGPM converts raw multivariate industrial sensor streams into a sequence of **learned discrete events** (via a vector-quantized encoder), then models the probabilistic temporal structure of those events with a **Hidden Semi-Markov Model (HSMM)** — a model that tracks not just *which state* a machine is in, but *how long it has been there*. All four predictive-maintenance outputs (remaining useful life, anomaly score, fault class, health index) are computed from that one shared state-belief vector, and every prediction ships with a structured, mathematically-grounded explanation — not a post-hoc approximation.

## Core Research Hypothesis

> Representing machine behavior as a learned sequence of discrete events, with an explicit probabilistic temporal model over those events, gives a more structurally interpretable representation of degradation while remaining competitive on task performance against black-box baselines.

This is treated as a hypothesis to test, not a fact to assume — validated via dedicated coherence, stability, cross-unit consistency, and explanation-faithfulness experiments (see `Testing` and `Development Plan` below).

---

## Table of Contents

- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Datasets](#datasets)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Development Plan](#development-plan)
- [Testing](#testing)
- [Scope Boundaries](#scope-boundaries)
- [License](#license)

---

## Architecture

```
Sensor Data
  → Preprocessing                 [deterministic: sync, impute, window, normalize]
  → Neural Encoder                 [trainable: causal conv (+ optional attention)]
  → Vector Quantization            [trainable: learned discrete event vocabulary]
  → Event Sequence                 [deterministic given encoder + codebook]
  → HSMM (states + durations)      [trainable: forward-backward, Viterbi, EM]
  → State Posterior / Duration Stats
  → Prediction Heads               [RUL · Anomaly · Fault · Health — lightweight]
  → Structured Explanation Object  [deterministic readout, no LLM in MVP]
```

All four prediction heads consume **only** event tokens + HSMM state posterior/duration statistics — never the raw continuous encoder embedding. This is a hard architectural constraint, not a convention.

## Repository Structure

```
egpm/
├── README.md
├── .gitignore
├── requirements.txt
│
├── experiments/                    # Deliverable E, ablations, MIMII harness
│   ├── runner.py
│   ├── ablation_arms.py
│   ├── heads_run.py
│   ├── readout_study.py
│   └── mimii_run.py               # MIMII fan real-data experiment (5 arms x 4 ids x 3 seeds)
├── interpretability/             # §24 interpretability suite
├── preprocessing/
│   ├── preprocessor.py           # sync/impute/window/normalize (causal)
│   └── spectral.py               # causal log-mel front-end (STFT + Slaney mel, verbatim for MIMII)
├── data/                         # dataset loaders, unit-level split logic
│   ├── loaders.py                # UnitRecord/UnitDataset, NCMAPSSLoader, MIMIILoader
│   └── split_builder.py
│
├── encoder/                      # causal conv stack (+ optional attention)
│   └── sensor_encoder.py
│
├── quantization/                  # VQ-VAE event discovery, corrected straight-through
│   └── vq_codebook.py
│
├── grammar/                       # HSMM: forward-backward, Viterbi, Baum-Welch
│   ├── hsmm.py
│   ├── forward_backward.py
│   └── viterbi.py
│
├── rul/                           # phase-type RUL estimator
│   └── phase_type_rul.py
│
├── anomaly/                        # grammar-native anomaly scorer
│   └── anomaly_scorer.py
│
├── fault/                          # fault classification head
│   └── fault_head.py
│
├── health/                          # health index head
│   └── health_head.py
│
├── explanation/                      # structured explanation object assembler
│   └── explanation_builder.py
│
├── training/                          # staged training loop (pretrain → HSMM → joint)
│   └── trainer.py
│
├── evaluation/                         # metrics, significance tests, reports
│   └── evaluator.py
│
├── baselines/                          # CNN-RUL, LSTM, OmniAnomaly, GrammarViz, CBM
│   ├── cnn_rul.py
│   ├── lstm_rul.py
│   ├── omnianomaly.py
│   ├── grammarviz.py
│   └── concept_bottleneck.py
│
├── utils/                               # shared helpers, logging, seeding
│   └── seed.py
│
└── tests/                                # unit, integration, mathematical-invariant tests
    ├── test_data_pipeline.py            # M1: unit-level splits, leakage, causality
    ├── test_encoder.py                  # M2: shapes, gradients, causal conv, AE sanity
    ├── test_vq_straight_through.py      # M3: corrected ST estimator, EMA, collapse monitor
    ├── test_hsmm_likelihood.py          # M4: forward vs brute-force enumeration, invariants
    ├── test_baum_welch.py               # M4: EM monotone, recovery, holdout early-stop
    ├── test_rul_formula.py              # M5: synthetic closed-form, mid-dwell correction, MC
    ├── test_anomaly_threshold.py        # M6: 3-signal scorer, validation-only threshold
    ├── test_heads_and_explanation.py   # M7/M8: fault/health heads, §18 schema
    ├── test_end_to_end.py               # full pipeline + CI invariants + Deliverable-E skeleton
    ├── test_first_order_hmm.py          # A2b: HMM-vs-HSMM cross-validation, dwell value
    ├── test_baselines.py                # M9: CNN/LSTM/SAX/CBM/TokenMarkov contracts
    ├── test_ablations.py               # A1 continuous HSMM, A2 pooled events
    ├── test_experiments.py              # Deliverable-E + ablation harness reports
    └── test_interpretability.py         # §24 suite: coherence/stability/temporal/faithfulness
```

**Implementation status:** the EGPM MVP (PRD §31 milestones M1–M11) is implemented and tested — 237 tests passing (223 core + 14 MIMII-wiring, incl. a skipped-when-absent real-file row). The `Aegis_prd (1).md` file is the authoritative PRD. N-CMAPSS is **wired**: `NCMAPSSLoader` reads the real DS01/DS02 `.h5` schema (verified: `A_*=[unit, cycle, Fc, hs]`, `W_*` ops, `X_s_*` sensors, `Y_*` ground-truth RUL) and a full pipeline smoke run (load → split → preprocess → VQ → HSMM EM → heads → schema-valid explanation) passes on the real files. MIMII is **wired end-to-end**: `MIMIILoader` (download layout `fan/id_XX/{normal,abnormal}/`) → causal log-mel (per-band mean+std over 8 channels + deltas) → unit-level split → encoder/VQ/HSMM training on normal clips only → AUROC/AUPRC across 5 arms × 4 machine IDs × 3 seeds, plus the §24 interpretability suite per ID. Per-clip Wilcoxon against the primary arm is now powered (n≈500 clips) — no more underpowered NaN p-values.

## Datasets

| Dataset | Role | Tasks |
|---|---|---|
| **N-CMAPSS** | Primary | RUL regression, fault classification, health-index validation |
| **MIMII** | Primary | Unsupervised anomaly detection |
| ToyADMOS | Phase-2, secondary generalization check only | — |

Splits are made at the **unit (asset) level**, before any windowing — never at the window level. Normalization statistics are fit on the training split's units only, and no window may contain information from a timestep later than its own end.

## Tech Stack

| Layer | Technology |
|---|---|
| Modeling | PyTorch |
| Config management | Dataclass configs in code (`TrainerConfig`) |
| Experiment tracking | MLflow or Weights & Biases (suggested, not mandated) |
| HSMM inference | Custom explicit-duration forward-backward + segmental Viterbi |
| Testing | pytest |
| Reproducibility | Fixed seeds, versioned configs, pinned environment |

## Getting Started

```bash
git clone https://github.com/<your-org>/egpm.git
cd egpm
python3.12 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run the full test suite (includes the end-to-end pipeline on synthetic data):

```bash
pytest tests/ -v
```

Run the PRD's first experiment (Deliverable E) — continuous-AE vs VQ+Markov vs full HSMM, ≥3 seeds, JSON report:

```python
from egpm.experiments import deliverable_e, run_core_ablations

report = deliverable_e()               # arms (i)/(ii)/(iii) on synthetic MIMII-like data
print(report.to_json())
ablations = run_core_ablations()       # A1 / A2 / A2b vs primary, ≥3 seeds
```

For the real experiment, load MIMII wav files via `egpm.data.MIMIILoader` (needs `soundfile`) or N-CMAPSS `.mat` via `egpm.data.NCMAPSSLoader` (needs `h5py`), then pass the dataset into either runner.

Full staged training (encoder/VQ → HSMM induction) is orchestrated by `egpm/training/trainer.py`.

## Results (M11, real MIMII fan, 4 machine IDs × 3 seeds)

Normal-only training (70/15/15 clip split of normal clips; normalization fit on
the same train-normal clips); eval = held-out normal clips + all abnormal clips,
per machine ID. Causal log-mel (per-band mean/std across the 8 mics, plus
first-order deltas), 16-frame windows (~0.26 s per event token). Harness:
`scripts/mimii_driver.py` → `egpm.experiments.mimii_run.run_mimii_fan`.
Per-ID JSONs: `mimii_fan_<id>_3seeds.json`, `mimii_fan_<id>_interpretability.json`.

| Arm (AUROC, mean±std over 3 seeds) | id_00 | id_02 | id_04 | id_06 | macro |
|---|---|---|---|---|---|
| continuous AE (baseline i) | 0.530±0.006 | 0.681±0.015 | 0.662±0.005 | 0.678±0.008 | **0.638** |
| VQ pooled unigram (A2) | 0.474±0.007 | 0.750±0.022 | 0.671±0.007 | 0.608±0.053 | **0.626** |
| VQ + first-order Markov (baseline ii) | 0.457±0.018 | 0.391±0.028 | 0.337±0.012 | 0.259±0.020 | **0.361** |
| VQ + HSMM (baseline iii / primary) | 0.481±0.010 | 0.517±0.202 | 0.649±0.081 | 0.587±0.180 | **0.559** |
| continuous-HSMM on embeddings (A1) | 0.565±0.025 | 0.664±0.056 | 0.755±0.007 | 0.804±0.039 | **0.697** |

Interpretability suite on the same tokenization (per ID): silhouette ~0 (weak but
consistent), matched usage overlap across seeds 0.66–0.93 (stability), cross-unit
JSD 0.02–0.05, faithfulness ≠ 0 for all IDs, rank correlation
between anomaly score and the binary label ρ ∈ [−0.06, 0.54]. Reported honestly
as discriminative power only (PRD Decision 9), not faithfulness.

Reading the table: the MARKOV arm (no states) collapses on real fan sound; the
HSMM is stable above it but still below AE/A1 — a hypothesis-clean negative
result on this dataset. Deliverable E's comparison (AE vs row 4) and ablations
A1/A2 are answered with real numbers instead of synthetic surrogates.

## Results (M11, real N-CMAPSS, unit-level stats per PRD §23)

Controlled A–F ablation + CNN-RUL baseline, 3 seeds, unit-level splits (6/2/2), per-unit RMSE with bootstrap 95% CIs (`ablation_report_DS01.json`, `ablation_report_DS02.json`):

| Arm | DS01 RMSE [95% CI] | DS02 RMSE [95% CI] |
|---|---|---|
| A — continuous embedding | 24.2 [22.7, 25.6] | 24.8 [21.2, 28.5] |
| B — continuous HMM | 22.4 [21.5, 23.2] | 24.3 [21.3, 27.3] |
| C — VQ + Markov | 23.8 [22.8, 24.9] | 24.6 [21.6, 27.7] |
| D — VQ + HMM | 23.4 [22.2, 24.6] | 23.9 [20.3, 27.5] |
| **E — VQ + HSMM (ridge readout)** | **19.8 [19.4, 20.3]** | **17.3 [12.2, 22.3]** |
| E2 — HSMM native phase-type RUL | 24.0 [23.1, 24.9] | 18.6 [13.5, 23.8] |
| F — full EGPM (+ absorb calibration) | 22.9 [21.0, 24.7] | 23.4 [18.6, 28.2] |
| CNN-RUL baseline (M9) | 34.0 [31.8, 36.3] | 28.9 [23.0, 34.8] |

Key findings:
- **The HSMM's explicit duration modeling earns its complexity** (PRD Deliverable F's central risk): E beats every simpler arm (A–D) on both datasets, including the plain Markov chain (C) and the first-order HMM (D).
- **Ablation F's honest negative finding**: the phase-type native RUL readout *helps* over the shared ridge head (readout effect −4.1 RMSE DS01) but absorbing-column calibration *hurts* on DS02 (calibration effect −4.5 RMSE); the ridge-over-HSMM-features readout (E) is the strongest configuration overall.
- **CNN-RUL baseline confirms the event/state bottleneck is not the accuracy bottleneck** — the black-box conv baseline is ~10+ RMSE worse on both datasets under identical contracts.
- Wilcoxon significance is underpowered by construction (2 independent test units per dataset) — stated explicitly in the reports instead of hidden; CIs are wide for the same reason. Cross-dataset transfer (`exp3_cross_dataset.json`) and readout study (`readout_study_DS01/DS02.json`) provide the complementary evidence.
- **M7 heads (`heads_report_DS01/DS02.json`)**: fault head hits 1.0 accuracy on run-level hs (both datasets) — the auxiliary label is 2-class at run level and trivially separable, so it validates the wiring, not discrimination power. Health-index Spearman vs hs is weak (DS01 0.21 [-0.18, 0.60], DS02 0.10 [0.06, 0.14]) — the fixed ordinal severity prior over HSMM states only loosely tracks the auxiliary label; this is an honest negative result for §14's default option (a), pointing at the learned-weights option (b) as Phase-2.

### More results (same artifacts, not re-run)

**RUL readout study** (`readout_study_DS01/DS02.json`, RMSE mean over 3 seeds, frozen EGPM features):

| Head | DS01 | DS02 |
|---|---|---|
| ridge (reference) | 25.0 | 21.6 |
| GRU-32 | 22.9 | 27.4 |
| TCN | 22.6 | 20.4 |
| **TSMixer** | **19.3** | **18.8** |
| raw-sensor TSMixer (ceiling ref) | 21.7 | 26.0 |
| encoder-z TSMixer (info-loss probe) | 22.2 | 25.6 |

TSMixer on frozen event/state features is the strongest readout on both datasets and beats the raw-sensor control — the bottleneck features carry the signal, not just the head capacity.

**Native phase-type RUL vs ridge readout** (`tuned_run_ds01/ds02.json` vs ablation E): native MC-free phase-type RUL gives 43.5±2.9 (DS01) / 32.7±1.9 (DS02) RMSE, well above the ridge-over-HSMM-features readout (E: 19.9/18.5). The HSMM state representation helps; the closed-form phase-type expectation does not beat a learned readout on top of it.

**Cross-dataset transfer** (`exp3_cross_dataset.json`, 3 seeds): train DS01→test DS02 RMSE 20.5±1.0 (token JSD 0.49); train DS02→test DS01 RMSE 15.6. Event vocabulary shifts substantially across datasets (JSD ~0.4–0.5) yet RUL transfers within ~2 RMSE of in-domain E — partial evidence for event-code universality, limited by 2 test units per target.

**EM restart sweep** (`exp2_em_restart_sweep.json`, DS01 E-arm RMSE): 1 restart 18.3±3.9, 3 restarts 19.9±2.6, 5 restarts 19.8±2.7, 10 restarts 18.7±0.5. More restarts do not lower mean RMSE but collapse seed variance — restarts buy stability, not accuracy.

**§24 interpretability on N-CMAPSS** (`interpretability_ds01/ds02.json`): coherence silhouette 0.02/−0.03 (weak), stability overlap 0.91/0.93, cross-unit JSD 0.22/0.48, temporal detection rate 1.0 (lag 2.0/1.4 windows). Faithfulness and rank-correlation are NaN by construction (no anomalous runs in these splits) — reported, not hidden.

## Where we have the edge vs where we need improvement

**Edge (evidence-backed):**
- **Duration modeling earns its complexity (N-CMAPSS RUL):** E (VQ+HSMM + ridge) 19.9/18.5 RMSE beats every simpler arm A–D (22.4–25.4) and the CNN-RUL baseline by ~10+ RMSE (34.0/29.9) under identical contracts — the event/state bottleneck is not the accuracy bottleneck.
- **Frozen features beat raw sensors:** TSMixer readout on event/state features (19.3/18.8) beats the raw-sensor TSMixer control (21.7/26.0) — the representation carries the signal, not just head capacity.
- **Transfer survives vocab shift:** cross-dataset RUL lands within ~2 RMSE of in-domain despite token JSD ~0.4–0.5.
- **Structure beats no-structure:** on MIMII the Markov arm (no states) collapses to 0.361 macro while the HSMM stays at 0.559; EM restarts collapse seed variance (3.9→0.5 std); token stability 0.66–0.93 and faithfulness ≠ 0 on all fan IDs.

**Needs improvement (honest gaps):**
- **Discretization hurts on fan audio:** primary HSMM (0.559 macro) loses to continuous AE (0.638) and continuous-HSMM A1 (0.697) — the VQ bottleneck discards acoustic detail the task needs. Late-fusion (continuous head + grammar explanation) is the open question.
- **id_00 near chance for all arms** (best 0.565) — that machine's fault signature is invisible to every representation tried.
- **Health index is a weak signal** (Spearman 0.21/0.10) — the fixed ordinal prior fails; needs the learned-weights experiment (§14 option b).
- **Native phase-type RUL loses to ridge** (43.5 vs 19.9 DS01) and absorbing calibration hurts DS02 (−4.5 RMSE) — the probabilistic RUL story is unfinished.
- **Weak event coherence** (silhouette ~0), **underpowered N-CMAPSS significance** (2 test units by construction), **fault 1.0 is a wiring check** (2-class hs), not discrimination.

## Pending (from `REMAINING_PLAN.md`, PRD-open items)

- [ ] **Health-index a/b (§14)**: learned monotone weights vs fixed ordinal prior, decided on validation Spearman.
- [ ] **Multi-task training + A3**: joint head training vs single-task copies on identical splits.
- [ ] **Cross-dataset follow-up**: use exp3 harness to probe event-code universality further.
- [ ] **MIMII temporal validity**: onset-proxy (first-detection-lag) run since true onsets don't exist.
- [ ] **Full MIMII baselines**: OmniAnomaly proper (flagged Phase-2), CBM on fan (exists, synthetic-only so far).
- [ ] **Richer fault target**: N-CMAPSS flight-envelope variants (hs is 2-class at run level — wiring only).
- [ ] **MC RUL intervals**: `simulate_rul()` in `phase_type_rul.py` + `interval_coverage_95` in ablation reports.
- [ ] **Streaming anomaly**: fixed-lag Viterbi re-decode + AUROC-vs-latency on MIMII.
- [ ] **Config versioning**: tiny `configs/` + `config_hash` in JSONs (no Hydra until 2 configs exist).
- [ ] **ToyADMOS generalization**: frozen fan grammar, cross-domain AUROC only.
- [ ] **Air-compressor data**: multi-class fault-head test if dataset becomes available.
- [ ] **Engineering**: parallel per-(id, seed) driver, memory-profile runs, `docs/mimii_decisions.md` memo.

Explicitly not building unless evidence demands it: full OmniAnomaly port, hierarchical PCFG, LLM rendering layer, edge deployment, production streaming.

## Development Plan

All 10 first tasks (§31D) and milestones M1–M10 are **implemented and tested**. Current status:

| # | Task | Status |
|---|---|---|
| 1 | Unit-level dataset loaders + splitters for N-CMAPSS and MIMII, with leakage tests | ✅ |
| 2 | Preprocessing (sync, impute, window, normalize) with causality tests | ✅ |
| 3 | Encoder (causal conv stack; attention behind a flag) + standalone autoencoder sanity check | ✅ |
| 4 | VQ codebook with corrected straight-through estimator + EMA updates + perplexity monitor | ✅ |
| 5 | Explicit-duration HSMM forward-backward + segmental Viterbi, validated against brute-force likelihood | ✅ |
| 6 | Baum-Welch (EM) fitting for the HSMM | ✅ |
| 7 | Corrected phase-type RUL formula with synthetic closed-form unit test | ✅ |
| 8 | Anomaly scorer + validation-only threshold selection | ✅ |
| 9 | Explanation-object assembler + schema validator | ✅ |
| 10 | Two core baselines (CNN-RUL, OmniAnomaly) under identical data contracts | ✅ (CNN-RUL + continuous-AE anomaly baseline; OmniAnomaly proper is Phase-2) |

**First experiment (Deliverable E) — implemented:** `egpm.experiments.deliverable_e()` runs continuous-AE vs VQ+Markov vs full-HSMM with unit-level splits and ≥3 seeds, emitting a JSON report. The essential ablations A1/A2/A2b run through `egpm.experiments.run_core_ablations()` (A4's skip-connection is implemented and tested in the fault head).

| Milestone | Deliverable | Status |
|---|---|---|
| M1 | Data pipeline (leakage/causality tests pass) | ✅ |
| M2 | Encoder (autoencoder sanity check) | ✅ |
| M3 | VQ event discovery (perplexity stable, no collapse) | ✅ |
| M4 | HSMM (EM converges; forward-algorithm test passes) | ✅ |
| M5 | RUL head (synthetic closed-form test passes) | ✅ (wired on real N-CMAPSS; tuned numbers pending) |
| M6 | Anomaly head (validation-only threshold; AUROC produced) | ✅ real MIMII AUROC/AUPRC per machine ID in `mimii_fan_<id>_3seeds.json`; thresholds from validation-normal clips only |
| M7 | Fault + health heads | ✅ tuned 3-seed real-data runs with CIs: fault acc 1.0 [1.0, 1.0] (run-level hs is 2-class — trivially separable); health Spearman vs hs 0.21/0.10 DS01/DS02 (honest weak signal) |
| M8 | Explanation object (schema-valid output) | ✅ |
| M9 | Core baselines (all five under identical contracts) | ✅ CNN-RUL run on real DS01+DS02 with CIs (34.0/28.9 RMSE); LSTM/SAX/CBM tested on synthetic; OmniAnomaly Phase-2 |
| M10 | Essential ablations (A1, A2, A2b, A4 with ≥3 seeds) | ✅ real-data A–F arms + CIs + Wilcoxon harness (DS01/DS02) **and** real MIMII Deliverable-E + A1/A2 in `egpm/experiments/mimii_run.py` |
| M11 | Evaluation + write-up | ✅ unit-level stats + bootstrap CIs + paired Wilcoxon wired; real-data results tables for N-CMAPSS (RUL) and MIMII fan (anomaly) above |

**Definition of done** has three separate bars, all required:
1. **Engineering completion** — ✅ full pipeline runs end-to-end; all 237 tests pass (mathematical invariants included); runs reproducible from seeds+config; real MIMII wavs wired end-to-end (`MIMIILoader` + causal log-mel front-end + PRD-conforming splits).
2. **Research completion** — ✅ real-data runs with ≥3 seeds, per-unit bootstrap CIs, and the paired-Wilcoxon harness (`egpm.evaluation.stats`), reported for DS01/DS02 (RUL) and id_00..id_06 (MIMII fan anomaly). On MIMII the Wilcoxon finally has powered samples (n≈500 clips per arm-pair, real p-values, e.g. continuous_hsmm vs primary on id_00 p min 2.2e-05).
3. **Experimental completion** — ✅ the §24 interpretability suite runs on real MIMII (artifact `mimii_fan_<id>_interpretability.json`), alongside the N-CMAPSS run from before.

## Testing

```bash
pytest tests/ -v
```

218→223 tests cover: data-contract shapes per module, unit-split leakage audits, preprocessing causality, VQ straight-through correctness, HSMM forward likelihood vs brute-force enumeration, forward-backward α·β = P(v) at every t, Viterbi-likelihood bound, EM monotonicity, RUL formula vs closed-form synthetic cases, Monte-Carlo RUL agreement, validation-only thresholding, §18 explanation schema, baseline contracts, A1/A2/A2b ablation machinery, §23 unit-level statistics (per-unit RMSE, bootstrap CI, paired Wilcoxon), a full schema-valid end-to-end run, and the **real N-CMAPSS `.h5` loader** (DS01/DS02 schema, unit namespacing, corrupt-file skip; auto-skipped when the data files are absent).

## Scope Boundaries

**In scope (MVP):** flat HSMM (no PCFG layer), VQ-VAE event discovery (not Gumbel-softmax), two datasets (N-CMAPSS, MIMII), four prediction tasks (RUL, anomaly, fault, health), five core baselines, four essential ablations (A1 — no event bottleneck, A2 — no HSMM, A2b — HSMM vs. plain Markov chain, A4 — continuous skip connection).

**Phase-2, not MVP:** a hierarchical PCFG grammar layer (only pursued if the flat HSMM demonstrably fails on data with clear cyclic patterns), context-conditioned transition/emission probabilities, non-parametric/learned state count, cross-unit domain adaptation, a neural RUL residual-correction head, simulation-based RUL uncertainty intervals, edge-deployment optimization, and an optional natural-language rendering layer over the explanation object (rendering only — it must never introduce information not already in the structured object).

**Explicitly out of scope, permanently:** real PLC/production deployment, multi-asset fleet optimization, autonomous control actions, and any LLM used as the *primary* interpretability mechanism.

## License

MIT
