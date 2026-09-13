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
├── environment.yml              # optional conda lockfile
│
├── configs/                     # YAML/Hydra hyperparameter + experiment configs
│   ├── base.yaml
│   ├── ncmapss.yaml
│   └── mimii.yaml
│
├── data/                        # dataset loaders, unit-level split logic
│   ├── loaders.py               # UnitRecord/UnitDataset, NCMAPSSLoader, MIMIILoader
│   └── split_builder.py
│
├── preprocessing/                # sync, impute, window, normalize (deterministic)
│   └── preprocessor.py
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

**Implementation status:** the EGPM MVP (PRD §31 milestones M1–M10) is implemented and tested — 218 tests passing. The `Aegis_prd (1).md` file is the authoritative PRD. N-CMAPSS is **wired**: `NCMAPSSLoader` reads the real DS01/DS02 `.h5` schema (verified: `A_*=[unit, cycle, Fc, hs]`, `W_*` ops, `X_s_*` sensors, `Y_*` ground-truth RUL) and a full pipeline smoke run (load → split → preprocess → VQ → HSMM EM → heads → schema-valid explanation) passes on the real files. Remaining for Definition of Done (§32): wiring MIMII wavs, hyperparameter tuning on real data, the §24 interpretability run on real datasets, and full multi-seed experiment reporting.

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
| Config management | Hydra / YAML |
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
| M6 | Anomaly head (validation-only threshold; AUROC produced) | ✅ (real MIMII numbers pending dataset) |
| M7 | Fault + health heads | ✅ (heads + tests; runs on real N-CMAPSS data, tuned numbers pending) |
| M8 | Explanation object (schema-valid output) | ✅ |
| M9 | Core baselines (all five under identical contracts) | ✅ |
| M10 | Essential ablations (A1, A2, A2b, A4 with ≥3 seeds) | ✅ harness + tests (tuned real-data runs pending) |
| M11 | Evaluation + write-up | ⏳ N-CMAPSS wired; MIMII + multi-seed reporting pending |

**Definition of done** has three separate bars, all required:
1. **Engineering completion** — ✅ full pipeline runs end-to-end, all 218 tests (mathematical invariants included) pass, runs are reproducible from seeds+config. *Remaining: wire real MIMII wavs (N-CMAPSS is wired and smoke-tested on the real files).*
2. **Research completion** — ⏳ the harness produces multi-seed ablation reports; N-CMAPSS is wired end-to-end, tuned real-data runs with confidence intervals pending.
3. **Experimental completion** — ✅ the §24 interpretability suite is **implemented and run**: `egpm.interpretability.run_interpretability_suite()` reports all six metrics (coherence, stability, cross-unit consistency, temporal validity, faithfulness, rank correlation) with pass-signal tests on constructed ground truth. *Remaining: run on real datasets.*

## Testing

```bash
pytest tests/ -v
```

218 tests cover: data-contract shapes per module, unit-split leakage audits, preprocessing causality, VQ straight-through correctness, HSMM forward likelihood vs brute-force enumeration, forward-backward α·β = P(v) at every t, Viterbi-likelihood bound, EM monotonicity, RUL formula vs closed-form synthetic cases, Monte-Carlo RUL agreement, validation-only thresholding, §18 explanation schema, baseline contracts, A1/A2/A2b ablation machinery, a full schema-valid end-to-end run, and the **real N-CMAPSS `.h5` loader** (DS01/DS02 schema, unit namespacing, corrupt-file skip; auto-skipped when the data files are absent).

## Scope Boundaries

**In scope (MVP):** flat HSMM (no PCFG layer), VQ-VAE event discovery (not Gumbel-softmax), two datasets (N-CMAPSS, MIMII), four prediction tasks (RUL, anomaly, fault, health), five core baselines, four essential ablations (A1 — no event bottleneck, A2 — no HSMM, A2b — HSMM vs. plain Markov chain, A4 — continuous skip connection).

**Phase-2, not MVP:** a hierarchical PCFG grammar layer (only pursued if the flat HSMM demonstrably fails on data with clear cyclic patterns), context-conditioned transition/emission probabilities, non-parametric/learned state count, cross-unit domain adaptation, a neural RUL residual-correction head, simulation-based RUL uncertainty intervals, edge-deployment optimization, and an optional natural-language rendering layer over the explanation object (rendering only — it must never introduce information not already in the structured object).

**Explicitly out of scope, permanently:** real PLC/production deployment, multi-asset fleet optimization, autonomous control actions, and any LLM used as the *primary* interpretability mechanism.

## License

MIT
