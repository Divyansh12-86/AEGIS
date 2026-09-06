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
│   ├── ncmapss_loader.py
│   ├── mimii_loader.py
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
    ├── test_preprocessing.py
    ├── test_vq_straight_through.py
    ├── test_hsmm_likelihood.py
    ├── test_rul_formula.py
    ├── test_anomaly_threshold.py
    └── test_end_to_end.py
```

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
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run the end-to-end pipeline on a synthetic sanity dataset first (before touching real data):

```bash
pytest tests/test_end_to_end.py -v
```

Train Stage 1 (encoder + VQ pretraining) on a configured dataset:

```bash
python -m training.trainer --config configs/ncmapss.yaml --stage 1
```

Full staged training (encoder/VQ → HSMM induction → joint fine-tuning) is orchestrated by `training/trainer.py`.

## Development Plan

Implementation proceeds in the exact order below. Do not start a task before its dependency's acceptance criteria pass.

| # | Task |
|---|---|
| 1 | Unit-level dataset loaders + splitters for N-CMAPSS and MIMII, with leakage tests |
| 2 | Preprocessing (sync, impute, window, normalize) with causality tests |
| 3 | Encoder (causal conv stack; attention behind a flag) + standalone autoencoder sanity check |
| 4 | VQ codebook with corrected straight-through estimator + EMA updates + perplexity monitor |
| 5 | Explicit-duration HSMM forward-backward + segmental Viterbi, validated against brute-force likelihood |
| 6 | Baum-Welch (EM) fitting for the HSMM |
| 7 | Corrected phase-type RUL formula with synthetic closed-form unit test |
| 8 | Anomaly scorer + validation-only threshold selection |
| 9 | Explanation-object assembler + schema validator |
| 10 | Two core baselines (CNN-RUL, OmniAnomaly) under identical data contracts |

**First experiment to run before scaling anything else:** on MIMII (anomaly-only), compare a continuous-embedding baseline vs. VQ-events-with-plain-Markov vs. full HSMM-with-duration. This is the cheapest possible test of whether the architecture's central, most expensive bet — explicit duration modeling — is worth its cost.

| Milestone | Deliverable | Depends on | Acceptance criteria |
|---|---|---|---|
| M1 | Data pipeline | — | Passes leakage/causality tests on N-CMAPSS and MIMII |
| M2 | Encoder | M1 | Reconstructs a held-out window with reasonable error in a standalone autoencoder sanity check |
| M3 | VQ event discovery | M2 | Codebook perplexity stable, no collapse, over a full training run |
| M4 | HSMM | M3 | EM converges; forward-algorithm likelihood test passes |
| M5 | RUL head | M4 | Passes synthetic closed-form RUL unit test; runs on N-CMAPSS |
| M6 | Anomaly head | M4 | Threshold selection test passes; produces AUROC/AUPRC on MIMII |
| M7 | Fault + health heads | M4, M5 | Both run on N-CMAPSS with labeled data |
| M8 | Explanation object | M4–M7 | Schema-valid output for a sample run |
| M9 | Core baselines | M1 | All five core baselines run under identical data contracts |
| M10 | Essential ablations | M2–M8 | A1, A2, A2b, A4 all run with ≥3 seeds |
| M11 | Evaluation + write-up | M9, M10 | Full metrics report with confidence intervals |

**Definition of done** has three separate bars, all required:
1. **Engineering completion** — full pipeline runs end-to-end on both datasets, all mathematical-invariant tests pass, and a run is fully reproducible from its config file alone.
2. **Research completion** — core baselines and essential ablations complete with ≥3 seeds and reported confidence intervals.
3. **Experimental completion** — the interpretability validation experiments (coherence, stability, cross-unit consistency, temporal validity, faithfulness) have all been run and reported, not merely defined.

## Testing

```bash
pytest tests/ -v
```

Covers, at minimum: data-contract shape tests per module, VQ straight-through correctness, HSMM forward-algorithm likelihood vs. brute-force enumeration, forward-backward normalization invariants, Viterbi-likelihood bound, RUL formula vs. closed-form synthetic case, and a full schema-valid end-to-end run.

## Scope Boundaries

**In scope (MVP):** flat HSMM (no PCFG layer), VQ-VAE event discovery (not Gumbel-softmax), two datasets, four prediction tasks, five core baselines, four essential ablations (A1, A2, A2b, A4).

**Explicitly out of scope:** real PLC/production deployment, multi-asset fleet optimization, autonomous control actions, any LLM as the primary interpretability mechanism, cross-fleet transfer, edge-deployment optimization.

**In scope (MVP):** flat HSMM (no PCFG layer), VQ-VAE event discovery (not Gumbel-softmax), two datasets (N-CMAPSS, MIMII), four prediction tasks (RUL, anomaly, fault, health), five core baselines, four essential ablations (A1 — no event bottleneck, A2 — no HSMM, A2b — HSMM vs. plain Markov chain, A4 — continuous skip connection).

**Phase-2, not MVP:** a hierarchical PCFG grammar layer (only pursued if the flat HSMM demonstrably fails on data with clear cyclic patterns), context-conditioned transition/emission probabilities, non-parametric/learned state count, cross-unit domain adaptation, a neural RUL residual-correction head, simulation-based RUL uncertainty intervals, edge-deployment optimization, and an optional natural-language rendering layer over the explanation object (rendering only — it must never introduce information not already in the structured object).

**Explicitly out of scope, permanently:** real PLC/production deployment, multi-asset fleet optimization, autonomous control actions, and any LLM used as the *primary* interpretability mechanism.

## License

MIT
