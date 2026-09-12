# EGPM — Event Grammar Probabilistic Model
### Final Technical PRD for Implementation

---

## Architecture Decisions and Resolved Conflicts

| # | Issue | Current documents | Final decision | Reason |
|---|-------|-------------------|-----------------|--------|
| 1 | Core temporal model | PRD proposes an HSMM; Methodology proposes a Neural PCFG parsed with the inside algorithm | **HSMM is the MVP core model.** A hierarchical PCFG layer is an optional Phase-2 extension | HSMM's explicit states and duration distributions map directly onto degradation stages and RUL via phase-type theory. Neither document argues machine behavior has genuine recursive/self-embedding structure, which is what a CFG's expressiveness (and its $O(L^3M^3)$ parsing cost) is actually for. Duration/RUL semantics are native to HSMM and absent from a standard PCFG |
| 2 | Discretization mechanism | Methodology mixes VQ-VAE and Gumbel-softmax terminology and states the straight-through estimator backwards: $\hat h_\ell = e_{k^*} + \text{sg}(h_\ell - e_{k^*})$ evaluates to $h_\ell$ in the forward pass, not $e_{k^*}$ | **Vector quantization (VQ-VAE) with a corrected straight-through estimator is primary.** Gumbel-softmax is an ablation | $\hat h = h + \text{sg}(e_{k^*} - h)$ is the form that actually forwards $e_{k^*}$ while passing gradient straight through to $h$. The original formula was mathematically wrong, not just non-standard |
| 3 | RUL mechanism | Methodology uses an MLP over an undefined grammar vector; PRD proposes $E[T]=(I-Q)^{-1}\mathbf{1}$ with no duration or elapsed-time correction | **Grammar/state-derived RUL via a duration-corrected phase-type formula (Section 12) is primary.** A neural residual-correction head is an optional Phase-2 addition | $(I-Q)^{-1}\mathbf{1}$ counts expected *transitions*, not expected *time*, and ignores that inference happens mid-dwell in the current state. Both defects are fixed below |
| 4 | Shared representation | Methodology aggregates PCFG non-terminal marginals into $\mathbf{g}\in\mathbb{R}^{d_g}$ with $d_g$ never related to $M$; PRD uses the HSMM belief $\alpha_t$ | **Use the time-indexed HSMM state-belief vector** $p_t\in\Delta^{M-1}$, dimension fixed at the number of states $M$ | Well-defined by construction, unlike an aggregation whose dimensionality was never pinned down |
| 5 | Symbolic bottleneck | Methodology claims a strict bottleneck but also folds sequence log-likelihood and (optionally) SHAP-on-grammar-state into the explanation path | **All primary heads consume only event tokens + HSMM state posterior/duration statistics.** No continuous encoder feature bypasses the bottleneck in the primary model; a skip connection is tested only as ablation A4 | Keeps the interpretability claim structurally true instead of quietly true-with-exceptions |
| 6 | Event semantics | Methodology asserts specific physical meanings ("bearing resonance," "valve discharge") as established fact | **Events are defined only as "a recurring learned behavioral pattern."** Physical labels are assigned post hoc, only after the event-coherence validation experiment (Section 24) passes | These meanings were never experimentally demonstrated in either source document |
| 7 | Complexity/latency numbers | Methodology states specific unmeasured numbers ("<10M MACs," "15–25ms," "$2.1\times10^6$ operations," "3–5× speedup") | Asymptotic complexity is given (Sections 13, 20); concrete numbers are marked **TBD**, to be measured and reported with exact hardware | These were invented estimates, not measurements, and are removed as fact claims |
| 8 | Codebook/rule entropy regularization | Methodology forces uniform codebook and rule usage as an unconditional objective | **Monitor utilization/perplexity; diversity regularization is optional and weak, its effect on downstream performance is itself ablation A6** | Uniform usage is not obviously correct — real machine behavior may legitimately concentrate on a handful of dominant events |
| 9 | "Rule fidelity" metric | Methodology calls Spearman correlation between grammar score and the binary anomaly label "rule fidelity" | **Renamed "anomaly-score rank correlation."** A separate faithfulness experiment (mask the flagged event subsequence → remeasure anomaly score) is added | Correlation with the label measures discriminative power, not whether the explanation reflects what the model actually used |
| 10 | Novelty language | Prior documents (PRD-level, general project framing) use "first"/"no existing work" style claims | Contribution is stated as a specific, falsifiable claim (Section 33), no superlatives | Unverified superlative claims aren't defensible without a systematic prior-art search, which has not been run |

---

## 1. Executive Summary

EGPM converts multivariate industrial sensor streams into a sequence of learned discrete events, models the probabilistic temporal structure of those events with a Hidden Semi-Markov Model (HSMM), and derives all four predictive-maintenance outputs — remaining useful life (RUL), anomaly detection, fault classification, and a continuous health index — from one shared, inspectable state-belief representation. Explanations are a direct readout of that representation, not a post-hoc approximation fitted after the fact.

The central, falsifiable research hypothesis: *representing machine behavior as a learned sequence of discrete events, with an explicit probabilistic temporal model over those events, gives a more structurally interpretable representation of degradation while remaining competitive on task performance against black-box baselines.* This document specifies the MVP scope, exact mathematics, software architecture, and implementation order needed to test that hypothesis, not to assume it.

---

## 2. Problem Definition

### 2.1 Industrial problem
Maintenance engineers need to know not just *that* an asset is degrading but *which operational stage it is in*, *how long it has been there*, and *why a given reading was flagged* — information current black-box predictive-maintenance (PdM) systems do not expose.

### 2.2 Machine-learning problem
Learn a mapping from a raw multivariate sensor sequence $\mathbf{X}_{1:T}$ to an anomaly indicator $a_t$, an RUL estimate $y_t$, a fault class $c_t$ (when labeled), a health index $h_t$, and a structured explanation $\mathcal{E}_t$ — such that every output is derivable from one shared, inspectable intermediate representation rather than from an opaque per-task hidden state.

### 2.3 Research gap
No reviewed PdM approach jointly (a) learns a discrete event vocabulary end-to-end, (b) models the probabilistic order and *duration* of those events, and (c) drives anomaly detection, RUL, fault classification, and health estimation from that one shared structure with intrinsic (not post-hoc) explanations. Existing work sacrifices interpretability for accuracy (CNN/LSTM/OmniAnomaly-style latent models) or sacrifices learned structure for interpretability (SAX/GrammarViz's fixed symbolic breakpoints, Concept Bottleneck Models' hand-labeled concepts with no temporal structure).

### 2.4 Core research hypothesis
Stated in Section 1. Restated formally: predictions computed exclusively from a discrete event sequence and its HSMM state posterior achieve task performance competitive with continuous-representation baselines, while providing explanations that are (i) coherent, (ii) stable across seeds, and (iii) faithful to what drove the prediction — properties defined operationally in Section 24.

### 2.5 Research questions
- **RQ1** — Does discretizing sensor windows into learned events preserve task-relevant information relative to continuous encodings? (Ablation A1)
- **RQ2** — Does explicit state+duration structure (HSMM) improve RUL/anomaly performance over a simpler pooled or first-order-Markov representation? (Ablations A2, A2b)
- **RQ3** — Are learned events and states coherent and stable enough to support human-meaningful explanation? (Section 24)
- **RQ4** — Is the HSMM-derived explanation faithful to the prediction it accompanies? (Section 24, faithfulness experiment)
- **RQ5** — Does multi-task training improve the shared representation over single-task training? (Ablation A3)

---

## 3. Scope

### 3.1 MVP
Preprocessing → encoder (conv required, attention optional/flagged) → VQ event discovery → HSMM (states + durations) → four prediction heads → structured explanation object. Evaluated on **N-CMAPSS** (RUL, fault classification) and **MIMII** (anomaly detection) as the two primary datasets. Core baseline subset (Section 21) and essential ablations A1, A2, A2b, A4 (Section 22).

### 3.2 Phase-2 extensions
- Hierarchical PCFG layer on top of the HSMM (pursue only if the flat HSMM demonstrably fails to capture cyclic motifs — Open Experimental Decision, Section 10)
- Context-conditioned transition/emission probabilities
- Non-parametric/learned state count
- Cross-unit domain adaptation
- Neural RUL residual-correction head
- Simulation-based RUL uncertainty intervals
- Edge-deployment optimization (distillation, beam-pruned inference)
- Natural-language rendering layer over the structured explanation object (rendering only — must not introduce or alter information)
- ToyADMOS as a secondary generalization check

### 3.3 Explicitly out of scope
Real PLC/production deployment; multi-asset fleet optimization; autonomous control actions; any LLM as the *primary* interpretability mechanism.

---

## 4. System Architecture

```
Sensor Data
  → Preprocessing                [deterministic]
  → Neural Encoder                [trainable]
  → Vector Quantization           [trainable; soft-assigned in training, deterministic argmin at inference]
  → Event Sequence                [deterministic, given encoder + codebook]
  → HSMM (states, durations)      [trainable parameters; probabilistic inference: forward-backward, Viterbi]
  → State Posterior / Duration Stats
  → Prediction Heads (RUL, Anomaly, Fault, Health)   [trainable, lightweight]
  → Explanation Object            [deterministic function of the above]
```

- **Trainable components:** encoder $f_\theta$, codebook $\{c_k\}$, HSMM parameters $(\pi, A, B, D)$, prediction heads.
- **Probabilistic components:** VQ soft assignment (training only), the HSMM itself (state and duration are latent).
- **Deterministic components:** preprocessing, event-sequence construction, explanation assembly.
- **Inference-only components:** Viterbi decoding for the explanation's state trajectory (training uses forward-backward marginals, not the hard Viterbi path, as the training signal).

---

## 5. Formal Problem Formulation

### 5.1 Notation

| Symbol | Meaning |
|---|---|
| $d$ | number of sensor channels |
| $\mathbf{x}_t\in\mathbb{R}^d$ | raw synchronized sensor observation at native time $t$ |
| $W, S$ | window length, stride |
| $\mathbf{X}^{(i)}\in\mathbb{R}^{W\times d}$ | preprocessed window $i$ |
| $\mathbf{m}^{(i)}\in\{0,1\}^{W\times d}$ | availability mask for window $i$ |
| $T_{\text{seq}}$ | number of windows in one asset run |
| $z_i = f_\theta(\mathbf{X}^{(i)}) \in\mathbb{R}^h$ | continuous encoder embedding for window $i$ (window-pooled) |
| $\{c_1,\dots,c_K\}\subset\mathbb{R}^h$ | learnable event codebook, vocabulary size $K$ |
| $v_i = \arg\min_k \lVert z_i - c_k\rVert^2 \in \{1,\dots,K\}$ | discrete **event token** for window $i$ |
| $\hat z_i = z_i + \text{sg}(c_{v_i} - z_i)$ | straight-through quantized embedding (forwards $c_{v_i}$) |
| $s_i \in \{1,\dots,M\}$ | latent **state** at window $i$; state $M$ is the absorbing failure state |
| $\pi \in \Delta^{M-1}$ | initial state distribution |
| $A \in \mathbb{R}^{M\times M}$ | embedded (zero-diagonal) state transition matrix |
| $B(v\mid m)$ | emission distribution over event tokens given state $m$ |
| $D(\tau\mid m)$ | duration distribution of state $m$ (support $1,\dots,D_{\max}$) |
| $p_i \in \Delta^{M-1}$ | state-belief vector, $p_i(m)=P(s_i=m\mid v_{1:i})$ |
| $Q$ | sub-stochastic transition matrix over the $M{-}1$ non-absorbing states |
| $\Phi = (I-Q)^{-1}$ | fundamental matrix |
| $\mu_m = \mathbb{E}[D(\cdot\mid m)]$ | mean dwell time of state $m$ |
| $y_i$ | RUL estimate at window $i$ |
| $a_i$ | anomaly score at window $i$ |
| $\hat p(c_i=j)$ | fault-class probability |
| $h_i\in[0,1]$ | health index |
| $\mathcal{E}_i$ | structured explanation object |

*Event* = what happened (short-lived, discrete token). *State* = where the machine is (longer-duration latent regime). *The HSMM* = how behavior evolves between and within states, including how long it stays.

### 5.2 Key relationships
$$
\mathbf{X}^{(i)} \xrightarrow{f_\theta} z_i \xrightarrow{\text{VQ}} v_i,\ \hat z_i \xrightarrow{\text{HSMM}} p_i,\ \text{Viterbi path } \hat s_{1:T_{\text{seq}}} \xrightarrow{\text{heads}} (y_i, a_i, \hat p(c_i), h_i)
$$

All four heads receive only $\{v_{1:i}, p_i, \text{duration statistics}\}$ — never $z_i$ or $\hat z_i$ directly (Decision 5, Section "Architecture Decisions").

---

## 6. Data Pipeline

**Required steps** (reused from the methodology's uncontested preprocessing, which is standard and correctly specified):
- **Temporal synchronization** — resample all channels to a common rate $f_{\text{sync}}$, anti-alias filter channels above it, linearly interpolate channels below it.
- **Missing-value handling** — binary mask $\mathbf{m}_t$; last-observation-carried-forward for short gaps, linear interpolation for longer gaps; mask passed to the encoder.
- **Window segmentation** — window length $W$ should span at least one operational cycle; MVP uses one event token per window ($S=W$, no overlap) for interpretability. Overlapping windows ($S<W$) are a Phase-2 resolution increase.
- **Normalization** — per-channel z-score, statistics computed **only on the training split, only from training units**.

**Optional steps:**
- STFT/MFCC frequency-domain features for vibration/acoustic channels, concatenated along the channel dimension.
- Learnable first-layer noise-rejection convolution (in place of a fixed filter).

**Data-leakage prevention (required, non-negotiable):**
1. Train/validation/test splits are made **at the unit (asset) level**, before any windowing.
2. Normalization statistics are fit on the training partition only.
3. No window's input may contain information from a timestep later than the window's own end (causality).
4. Overlapping windows are never treated as statistically independent samples in evaluation — metrics and significance tests operate at the unit level (Section 23).
5. Hyperparameters are tuned on the validation split only; the test split is touched exactly once, for final reporting.

---

## 7. Neural Sensor Encoder

MVP encoder, deliberately not over-engineered:

1. **Per-channel causal 1D convolution** (required) — captures local, short-duration patterns (spikes, brief oscillations) per sensor before cross-channel mixing, preserving physical channel identity.
2. **Cross-channel fusion via dilated causal convolution stack** (required) — geometrically increasing dilation for a large receptive field without parameter blow-up; the workhorse component.
3. **Self-attention** (**optional**, config flag `use_attention`) — captures long-range dependencies dilated convolution may miss; ablation A5 tests whether it earns its cost given the HSMM already models sequential structure downstream.
4. **Window pooling** — mean- or attention-pool over the window's temporal positions to a single embedding $z_i\in\mathbb{R}^h$, since the MVP assigns one event per window.

All dimensions (filter count, kernel size, number of conv blocks, attention heads/layers, $h$) are configurable hyperparameters (Section 26), not fixed by this specification.

---

## 8. Event Primitive Discovery

**Definition.** An event is a recurring learned behavioral pattern in the encoder's window-level representation — nothing more is asserted until Section 24's validation passes.

**Mechanism (VQ-VAE with corrected straight-through estimator, primary):**
$$
z_i = f_\theta(\mathbf{X}^{(i)}),\qquad v_i = \arg\min_k \lVert z_i - c_k \rVert^2
$$
$$
\hat z_i = z_i + \text{sg}(c_{v_i} - z_i) \qquad \text{(forwards } c_{v_i}\text{; gradient flows straight to } z_i)
$$
$$
\hat{\mathbf X}^{(i)} = g_\phi(\hat z_i)
$$
$$
\mathcal{L}_{\text{recon}} = \lVert \mathbf{X}^{(i)} - \hat{\mathbf X}^{(i)}\rVert^2 + \lVert \text{sg}[z_i] - c_{v_i}\rVert^2 + \beta \lVert z_i - \text{sg}[c_{v_i}]\rVert^2
$$

**Codebook updates:** EMA updates toward the running mean of assigned embeddings (standard VQ-VAE practice), not gradient descent on the codebook loss term directly, to reduce instability.

**Collapse monitoring (required, not optional):** track codebook **perplexity** $= \exp(-\sum_k \bar q_k \log \bar q_k)$ where $\bar q_k$ is the empirical usage frequency, and **effective codebook size** (number of codes with usage above a small threshold) every epoch. A weak diversity regularizer (entropy bonus on $\bar q_k$) is available but its effect on downstream task performance is an open question resolved by ablation A6 — it is not applied by default.

**Ablation, not primary:** Gumbel-softmax categorical assignment (soft at train time, hard argmax at inference) is implemented as an alternative event-discovery mechanism for a head-to-head comparison against VQ-VAE, not as the MVP default.

---

## 9. Event Sequence Representation

Each unit run of $T_{\text{seq}}$ windows produces $v_1,\dots,v_{T_{\text{seq}}}$. Repeated tokens (persistence) are expected and meaningful — a stable regime should emit the same event repeatedly. Runs of different lengths across units are handled by padding + masking in batched training (padded positions excluded from all losses and from HSMM likelihood). Sub-window temporal resolution (multiple tokens per window, as in the earlier PCFG-oriented methodology) is a Phase-2 option, not MVP.

---

## 10. Probabilistic Temporal Event Grammar (HSMM)

This is the central mechanism of EGPM.

### 10.1 Definition
$$
G_{\text{EGPM}} = (\pi, A, B, D)
$$
- $\pi \in \Delta^{M-1}$: initial state distribution over $M$ latent states (MVP default $M=6$: `startup, stable, transition, abnormal, degrade, failure` — `failure` is the absorbing state, index $M$; **required/tunable**, see Section 26).
- $A\in\mathbb{R}^{M\times M}$: **embedded, zero-diagonal** transition matrix — self-dwell is captured by $D$, not by repeated self-transitions in $A$. This avoids double-counting dwell time in the RUL calculation (Section 12).
- $B(v\mid m)$: categorical emission distribution over the $K$ event tokens, per state.
- $D(\tau\mid m)$: duration distribution per state, support $\{1,\dots,D_{\max}\}$ (parametric — e.g. discretized Gamma or shifted negative binomial — or non-parametric histogram; **Open Experimental Decision**, resolved by validation-set log-likelihood comparison).

### 10.2 Likelihood — explicit-duration forward algorithm
This is a standard method (Yu & Kobayashi, 2003; Yu, 2010 survey on hidden semi-Markov models), not a novel EGPM contribution — cited here rather than re-derived from scratch, and applied to industrial event sequences rather than its original domains. The forward variable is defined over segment endpoints:
$$
\alpha_t(j) = \sum_{d=1}^{\min(t,D_{\max})} \Big[\sum_{i\ne j} \alpha_{t-d}(i)\, A(i,j)\Big]\, D(d\mid j)\, \prod_{\tau=t-d+1}^{t} B(v_\tau \mid j)
$$
with $\alpha_0(i)=\pi(i)$. Total sequence likelihood is obtained by summing the forward variable over all valid final segment endpoints. Backward variables are defined symmetrically; forward-backward gives the state-belief $p_i(m)$ used downstream. Complexity: $O(T_{\text{seq}}\cdot M^2\cdot D_{\max})$ — linear in sequence length, unlike the $O(L^3M^3)$ CFG parsing it replaces.

### 10.3 Decoding
Segmental Viterbi (the HSMM analogue of standard Viterbi, maximizing over segment boundaries and durations jointly) gives the most probable state-and-duration path $\hat s_{1:T_{\text{seq}}}$, used for the explanation object's state trajectory.

### 10.4 Learning normal behavior
The HSMM is fit primarily on sequences from **normal operating regions** of each run (Section 15/16), so that low likelihood under the learned model is itself the anomaly signal, rather than requiring separately labeled anomalies.

### 10.5 Degradation in the state trajectory
Whether posterior mass shifts toward "late-stage" states as a unit degrades is a **hypothesis**, validated empirically against N-CMAPSS's auxiliary health-state labels (Section 24, temporal validity), not assumed true by construction.

**Open Experimental Decision — PCFG extension:** if held-out log-likelihood or fault/RUL performance shows the flat HSMM systematically fails on runs with clearly cyclic sub-patterns (e.g. repeated load/unload cycles the state set can't represent), layer a PCFG over event *motifs* in Phase 2. Do not add this speculatively.

---

## 11. Anomaly Detection

Grammar-native score combining three surprise signals at window $i$, given the Viterbi (or MAP marginal) state $\hat s_i$:
$$
a_i = -\log B(v_i\mid \hat s_i) \;-\; \lambda_{\text{trans}}\cdot \mathbb{1}[\text{transition at } i]\log A(\hat s_{i-1}, \hat s_i) \;-\; \lambda_{\text{dur}}\cdot \log \bar D(\tau_i \mid \hat s_i)
$$
where $\tau_i$ is elapsed dwell time in the current state and $\bar D$ is the survival function (probability of dwelling at least $\tau_i$ steps), so an implausibly long or short stay is penalized without requiring the segment to have ended yet. $\lambda_{\text{trans}}, \lambda_{\text{dur}}$ are tunable weights.

**Thresholding:** set from the empirical percentile (e.g. 99th) of $a_i$ over the **validation split's normal-only** windows. Test labels are never used for threshold selection.

---

## 12. RUL Estimation

This must be mathematically rigorous and duration-aware — the naive $(I-Q)^{-1}\mathbf{1}$ formula in the source PRD counts expected *transitions*, not expected *time*, and ignores partial elapsed dwell. The corrected derivation:

Let $Q$ be the embedded (zero-diagonal) sub-stochastic transition matrix over the $M{-}1$ non-absorbing states, $\Phi=(I-Q)^{-1}$ the fundamental matrix ($\Phi_{ij}$ = expected number of future visits to state $j$, starting a **fresh entry** into state $i$, before absorption), and $\mu\in\mathbb{R}^{M-1}$ the vector of mean state durations. Then:

$$
\text{(fresh-entry expected total remaining time from state } i) = (\Phi\mu)_i
$$

This is not yet the deployed quantity — a fielded system is almost always observed **mid-dwell**, having already spent $\tau$ steps in its current state $i$. Let $r_i(\tau) = \mathbb{E}[D_i - \tau \mid D_i \ge \tau]$ be the mean residual life of state $i$'s duration distribution given $\tau$ elapsed steps. Since $(\Phi\mu)_i$ already includes one full $\mu_i$ term for the current (in-progress) visit, replace it with the residual correction:

$$
\boxed{\ \mathbb{E}[\text{RUL} \mid s_i, \tau] \;=\; \big[r_i(\tau) - \mu_i\big] \;+\; (\Phi\mu)_i\ }
$$

**Median RUL and prediction intervals:** no closed form is claimed. Compute via **Monte Carlo simulation** of the HSMM generative process forward from the current belief $p_i$ to absorption, taking empirical quantiles across simulated trajectories (**Open Experimental Decision** — number of simulated trajectories, and whether to condition on the full belief $p_i$ vs. only the MAP state, are resolved empirically).

A neural residual-correction head ($y_i = \mathbb{E}[\text{RUL}] + f_{\text{res}}(p_i)$) is a Phase-2 addition, evaluated only to check whether the grammar-derived estimate leaves systematic error — it is not the primary mechanism.

---

## 13. Fault Classification

Input: the pooled state-belief trajectory $p_{1:T_{\text{seq}}}$ plus aggregate duration statistics (total dwell per state, transition counts) — pooled via a lightweight temporal model (small GRU or attention pooling), **not** raw $z_i/\hat z_i$. Output: softmax over $C$ labeled fault classes. Trained only where fault labels exist (e.g. N-CMAPSS failure modes, or the air-compressor seed dataset's component labels if incorporated in Phase 2).

---

## 14. Health Index

$$
h_i = \sum_{m=1}^{M} p_i(m)\, w_m,\qquad w_m \in [0,1]
$$

**Open Experimental Decision:** whether $w_m$ is (a) a fixed ordinal prior over the designed state ordering (MVP default — keeps the index interpretable and avoids overclaiming physical calibration) or (b) learned with a monotonicity regularizer against the assumed degradation ordering. State severity weights are **not** assumed physically valid; resolve (a) vs (b) by comparing correlation with N-CMAPSS's auxiliary health-state labels on the validation split.

---

## 15. Multi-task Learning

**Composite loss:**
$$
\mathcal{L} = \lambda_1\mathcal{L}_{\text{recon}} + \lambda_2\mathcal{L}_{\text{commit}} + \lambda_3\big(-\log P_{\text{HSMM}}(v_{1:T_{\text{seq}}})\big) + \lambda_4\mathcal{L}_{\text{RUL}} + \lambda_5\mathcal{L}_{\text{anom}} + \lambda_6\mathcal{L}_{\text{fault}} + \lambda_7\mathcal{L}_{\text{health}}
$$
- $\mathcal{L}_{\text{RUL}}$: Huber loss against ground-truth cycles-to-failure.
- $\mathcal{L}_{\text{anom}}$: BCE where labels exist; otherwise the HSMM log-likelihood term itself is the (unsupervised) anomaly training signal.
- $\mathcal{L}_{\text{fault}}$: cross-entropy where labels exist.
- $\mathcal{L}_{\text{health}}$: supervised regression where labels exist; otherwise a weak self-supervised ordering loss over the assumed state severity ordering.
- **Missing labels:** the corresponding loss term is masked out per-sample, not zero-filled.
- **Task weights $\lambda_{1..7}$:** start equal, tune on the validation split (**tunable, dataset-dependent** — no value is asserted here).

**Gradient flow:** the VQ straight-through estimator carries gradient from all downstream losses back into the encoder. HSMM parameters $(A,B,D)$ are constrained-parameterized (softmax/positive-definite reparameterizations) so the forward-algorithm log-likelihood is differentiable and can also receive gradient during Stage 3 fine-tuning, on top of the EM fit from Stage 2 (Section 16).

---

## 16. Training Procedure

| Stage | What's trained | What's frozen | Objective | Exit criterion |
|---|---|---|---|---|
| 1 — Encoder/VQ pretraining | $f_\theta, g_\phi$, codebook | HSMM (not yet introduced) | $\mathcal{L}_{\text{recon}}+\mathcal{L}_{\text{commit}}$ | Codebook perplexity stabilizes; reconstruction loss plateaus |
| 2 — HSMM induction | $(\pi,A,B,D)$ via EM (Baum-Welch, generalized for explicit duration) | Encoder lightly fine-tuned or frozen | Event-sequence log-likelihood on normal-operation runs | EM log-likelihood converges; sanity-check state trajectories against N-CMAPSS health-state labels |
| 3 — Joint fine-tuning | Everything | Nothing | Full composite loss $\mathcal{L}$, low VQ temperature | Validation composite loss / primary task metric plateaus |

Optimizer: AdamW (standard default). Learning rate, batch size, and stage lengths are **tunable/dataset-dependent** (Section 26) — not fixed here. Early stopping on validation loss. Checkpoint every epoch. Random seeds fixed and logged per run (Section 30).

---

## 17. Inference Pipeline

**Offline (batch):** preprocess full run → encode all windows → quantize (deterministic argmin, no Gumbel noise) → run full forward-backward for state posteriors + segmental Viterbi for the explanation path → run all four heads → assemble explanation object.

**Streaming:** on each new window, update the forward variable incrementally (amortized $O(M^2 D_{\max})$ per step); heads recompute from the updated belief. Full segmental Viterbi over an unbounded history is not causal for streaming use; **Open Experimental Decision:** re-decode Viterbi over a fixed recent lag window, or use an online fixed-lag smoother approximation — resolved by comparing explanation stability vs. latency in Phase 2.

---

## 18. Explanation System

Structured object, returned for every prediction — this is the primary interpretability mechanism; no LLM natural-language layer is part of the MVP.

```json
{
  "events": [{"window_index": int, "event_id": int}],
  "state_trajectory": [{"window_index": int, "state": int, "elapsed_duration": int}],
  "transition_probabilities": {"from_state": int, "to_state": int, "probability": float},
  "duration_estimates": {"state": int, "mean": float, "residual_given_elapsed": float},
  "anomaly_causes": [{"window_index": int, "component": "emission|transition|duration", "surprisal": float}],
  "RUL_estimate": {"expected": float, "median": "float | null (Phase 2)", "interval": "[low, high] | null (Phase 2)"},
  "fault_prediction": {"class": int, "probability": float},
  "health_index": float
}
```

A natural-language rendering layer over this object is an optional Phase-2 interface addition and must only phrase these fields, never introduce information not present in them.

---

## 19. Experimental Evaluation

Run each of the following, per Section 20's metrics, on the datasets in Section 3.1:
- RUL on N-CMAPSS, cross-operating-envelope.
- Anomaly detection on MIMII, unsupervised (trained normal-only).
- Fault classification on N-CMAPSS's labeled failure modes.
- Health index correlation against N-CMAPSS auxiliary health-state labels.
- Event/grammar quality experiments (Section 24).
- Explanation faithfulness experiment (Section 24).

---

## 20. Metrics

| Task | Metrics |
|---|---|
| RUL | RMSE, MAE, NASA/PHM asymmetric score; calibration/interval coverage **if** Phase-2 uncertainty estimation is implemented |
| Anomaly | AUROC, AUPRC, F1 at validation-selected threshold, early detection rate |
| Fault | Macro F1, per-class recall |
| Health | Correlation with auxiliary health-state labels; smoothness (successive-difference variance) |
| Interpretability | Event coherence, event stability, cross-unit consistency, temporal validity, **anomaly-score rank correlation** (renamed from "rule fidelity"), explanation faithfulness, expert agreement (only if domain experts are actually recruited — do not assume this happens) |

---

## 21. Baseline Suite

**Core (MVP) baselines** — the minimum needed to answer the research questions in Section 2.5:

| Family | Baseline | Research question it answers |
|---|---|---|
| RUL | CNN (Babu et al., 2016) | Does event-grammar structure beat direct convolutional feature learning for RUL? |
| RUL | LSTM | Does explicit event/state structure beat implicit sequential memory? |
| Anomaly | OmniAnomaly | Is grammar likelihood competitive with latent-space reconstruction likelihood? |
| Symbolic | GrammarViz (Sequitur on SAX) | Do the neural encoder and differentiable HSMM add value over classical, non-learned grammar induction? |
| Neuro-symbolic | Concept Bottleneck Model (no temporal structure) | Does the *temporal* structure (HSMM) add value beyond a flat concept bottleneck? |

**Extended (Phase-2) baselines:** Attention-LSTM, USAD, MSCRED, TranAD, Anomaly Transformer, SAX (raw), DeepProbLog, LSTM+SHAP, AutoRUL (a **strong automated baseline**, not an "upper bound" — no systematic search has established that).

**Fair-comparison protocol (all baselines):** identical unit-level splits, equivalent preprocessing where architecturally appropriate, comparable input information, a defined tuning protocol on validation only, multiple random seeds (≥3), and reported confidence intervals or standard deviations.

---

## 22. Ablation Study

**Essential (MVP):**
- **A1 — No event bottleneck.** Encoder embeddings feed the HSMM's emission model directly (continuous), skipping VQ. Tests RQ1.
- **A2 — No HSMM.** Event sequence pooled (mean) directly to heads, no state/duration model. Tests whether the HSMM is earning its cost at all.
- **A2b — HSMM vs. first-order Markov (no duration).** Replace $D$ with an implicit geometric duration (standard HMM). Directly tests whether explicit duration modeling — the most expensive part of the HSMM relative to a plain HMM — earns its keep. This is the single most important ablation for defending the architecture's added complexity.
- **A4 — Continuous skip connection.** Heads receive both $p_i$ and raw $\hat z_i$. Tests whether the bottleneck constraint costs predictive accuracy.

**Optional (Phase 2, run if resources allow):**
- A3 — No multi-task learning (independent single-task copies).
- A5 — Encoder without self-attention.
- A6 — No codebook/rule diversity regularization.
- Context-conditioned vs. static transition/emission probabilities.
- Learned VQ events vs. fixed SAX events.
- Learned events vs. random discretization (sanity floor).
- Event order shuffled (sanity/control check, not a true ablation of a design choice).

---

## 23. Experimental Integrity

Checklist, required for every reported result:
- [ ] Splits are unit-level, made before windowing
- [ ] Normalization statistics fit on training units only
- [ ] No window contains information from a later timestep than its own end
- [ ] Overlapping windows are not treated as independent in significance testing — use unit-level aggregation/bootstrap
- [ ] Hyperparameter tuning uses the validation split only
- [ ] Test split is evaluated exactly once, for final numbers
- [ ] ≥3 random seeds per reported configuration
- [ ] Statistical testing (e.g. paired Wilcoxon across units) between EGPM and each baseline
- [ ] Confidence intervals or standard deviations reported, not point estimates alone
- [ ] Full reproducibility artifacts logged (Section 30)

---

## 24. Interpretability Validation

- **Event coherence** — do windows sharing an event token exhibit similar raw sensor behavior (silhouette score as one signal among several, not the sole metric)?
- **Event stability** — does the same event vocabulary recur across random seeds after Hungarian/optimal-transport matching?
- **Cross-unit consistency** — trained on a subset of units, does the event/state distribution transfer to held-out units?
- **Temporal validity** — do inferred state transitions align with known operational or fault-onset timestamps where such ground truth exists (e.g. N-CMAPSS flight-phase changes)?
- **Explanation faithfulness** — identify the event/state subsequence flagged as the anomaly cause → mask or corrupt it → recompute the anomaly score → measure the score's reduction. This is the primary faithfulness test and replaces the mislabeled "rule fidelity" correlation metric.
- **Expert agreement** — only if domain experts are actually recruited; not assumed as part of the MVP timeline.

---

## 25. Failure Modes and Risks

| Risk | Symptom | Diagnostic | Mitigation |
|---|---|---|---|
| VQ collapse | Perplexity crashes toward 1 | Monitor per-epoch codebook usage histogram | EMA updates, commitment loss, optionally the diversity regularizer (A6) |
| Vocabulary too large | Sparse, unstable emission distributions per state | Per-state emission entropy very high | Sweep $K$ down; check downstream task metric doesn't improve further |
| Vocabulary too small | States become emission-indistinguishable | Emission distributions across states converge | Sweep $K$ up |
| Unstable/degenerate HSMM | States merge or become unreachable during EM | Track $A$'s row entropy, state occupancy over training | Regularize transition entropy weakly; restart EM with different init |
| State permutation across seeds | Same physical state gets a different index each run | Compare state orderings post-hoc via Hungarian matching before reporting stability metrics | Always match before comparing across seeds |
| Duration misspecification | Poor fit for genuinely irregular dwell times | Compare parametric vs. non-parametric $D$ on validation log-likelihood | Prefer non-parametric histogram $D$ if parametric fit is poor |
| Cross-unit domain shift | Model trained on some units fails on others | Held-out-unit validation performance drop | Per-unit normalization; Phase-2 domain adaptation |
| Cold start on new machine type | No usable vocabulary/grammar at deployment | N/A pre-deployment | Phase-2: pretrain on one fleet, fine-tune on the new one |
| Label scarcity (fault/health) | Only RUL trains well | Per-task loss curves diverge in usefulness | Stages 1–2 need no labels; add labeled heads last (Section 16) |
| Class imbalance (fault types) | Macro F1 dominated by majority class | Per-class recall inspection | Class-weighted loss, stratified sampling |
| Computational cost | Forward-backward slower than expected at scale | Profile per-run wall-clock early, not late | $D_{\max}$ pruning; batched forward-backward |
| Weak event semantics | Coherence/stability metrics fail | Section 24 results | Do not assign physical labels until metrics pass; reconsider encoder capacity or $K$ |
| HSMM overfitting | Train log-likelihood high, validation low | Standard train/val likelihood gap | Reduce $M$/$D_{\max}$, add transition-entropy regularization |

---

## 26. Hyperparameter Configuration

| Hyperparameter | Status | Notes |
|---|---|---|
| Window length $W$ | Required, dataset-dependent | Must span ≥1 operational cycle |
| Stride $S$ | Required | MVP default $S=W$ |
| Embedding dim $h$ | Tunable | — |
| Vocabulary size $K$ | Tunable, dataset-dependent | Sweep against Section 25's collapse/sparsity diagnostics |
| Number of HSMM states $M$ | Tunable | MVP default 6 (incl. absorbing failure state) |
| Max duration $D_{\max}$ | Tunable, dataset-dependent | Set from the longest plausible normal dwell in the data |
| Encoder depth / dilation levels | Tunable | — |
| Attention heads/layers | Tunable | Only relevant if `use_attention=true` |
| Learning rate(s) per stage | Tunable | No fixed value asserted |
| Batch size | Tunable, hardware-dependent | — |
| Loss weights $\lambda_{1..7}$, $\lambda_{\text{trans}}, \lambda_{\text{dur}}$ | Tunable | Start equal, tune on validation |

No numeric defaults beyond "start equal" are asserted as fact; all are search ranges to sweep, per doc conventions above.

---

## 27. Software Architecture

```
egpm/
    data/            preprocessing/            encoder/
    quantization/     grammar/                  rul/
    anomaly/          fault/                    health/
    explanation/       training/                 evaluation/
    baselines/         configs/                  utils/
    tests/
```

| Module | Responsibility | Input | Output | Key classes | Tests |
|---|---|---|---|---|---|
| `data/` | Load raw datasets, unit-level split | Raw dataset files | Unit-indexed raw sequences | `UnitDataset`, `SplitBuilder` | Split leakage checks |
| `preprocessing/` | Sync, impute, window, normalize | Raw sequences | `X, mask` tensors | `Preprocessor` | Shape, causality, no-future-leakage tests |
| `encoder/` | Window → embedding | `X, mask` | `z` | `SensorEncoder` | Shape, gradient-flow tests |
| `quantization/` | VQ event discovery | `z` | `v, ẑ, codebook stats` | `VQCodebook` | Straight-through correctness, EMA update tests |
| `grammar/` | HSMM fit/inference | `v_{1:T}` | `p_i, Viterbi path, seq log-lik` | `HSMM`, `ForwardBackward`, `Viterbi` | Likelihood correctness, row-sum invariants |
| `rul/` | RUL from HSMM | `p_i, (Φ,μ,Q)` | `y_i` (+ interval, Phase 2) | `PhaseTypeRUL` | Formula unit tests (Section 29) |
| `anomaly/` | Grammar-native score | `v_i, \hat s_i` | `a_i` | `AnomalyScorer` | Threshold-from-val-only test |
| `fault/` | Fault classification | `p_{1:T}` | `\hat p(c)` | `FaultHead` | Shape, no-raw-feature-bypass test |
| `health/` | Health index | `p_i, w` | `h_i` | `HealthHead` | Range-in-[0,1] test |
| `explanation/` | Assemble structured object | all of the above | JSON explanation | `ExplanationBuilder` | Schema validation |
| `training/` | Staged training loop | configs, data | checkpoints | `Trainer` | Stage-transition tests |
| `evaluation/` | Metrics, significance tests | predictions, labels | reports | `Evaluator` | Metric correctness |
| `baselines/` | Baseline model implementations | same data contracts | predictions | one class per baseline | Contract-compliance tests |
| `configs/` | Hyperparameter/config files | — | — | YAML/Hydra configs | Schema validation |
| `utils/` | Shared utilities | — | — | — | — |
| `tests/` | All unit/integration tests | — | — | — | — |

---

## 28. Data Contracts

```
Preprocessor output:
  X:    [batch, W, d]
  mask: [batch, W, d]

Encoder output:
  z: [batch, h]

Quantizer output:
  event_ids:           [batch]              (int, 1..K)
  quantized_embedding:  [batch, h]
  codebook_stats:       {perplexity, usage_histogram}

HSMM output:
  state_posterior:      [batch, T_seq, M]
  viterbi_path:         [batch, T_seq]        (int, 1..M)
  duration_estimates:   [batch, T_seq]
  sequence_log_likelihood: [batch]
  transition_statistics, duration_statistics: aggregated per-run

Prediction outputs:
  RUL:    {expected: float, median: float|null, interval: [float,float]|null}
  anomaly_probability: float
  fault_probabilities: [C]
  health_score: float

Explanation output:
  structured JSON object as defined in Section 18
```

---

## 29. Testing Strategy

- **Unit/shape tests** for every module's data contract (Section 28).
- **Numerical stability tests** — no NaN/Inf in log-likelihood computations at extreme (near-zero) probabilities.
- **VQ tests** — straight-through estimator forwards $c_{v_i}$ exactly; gradient reaches $z_i$ unchanged when the loss is applied directly to $\hat z_i$.
- **HSMM likelihood tests** — forward-algorithm total likelihood matches brute-force enumeration on a small synthetic $(M, D_{\max}, T)$ instance.
- **Forward-backward consistency** — $\sum_m p_i(m) = 1$ for every $i$.
- **Viterbi tests** — Viterbi path likelihood $\le$ total sequence likelihood, always.
- **RUL calculation tests** — on a synthetic 2-state absorbing chain with known closed-form expected time, the implemented formula (Section 12) matches analytically.
- **Anomaly scoring tests** — threshold is selected only from validation-split data in code, verified by a test that fails if test-split data is touched.
- **End-to-end integration test** — full pipeline runs on a small synthetic dataset without error and produces a schema-valid explanation object.
- **Mathematical invariants, checked automatically in CI:**
  - All probability distributions sum to 1 (within floating-point tolerance)
  - Transition matrix rows (over non-diagonal + absorption) sum to 1
  - State-belief vectors sum to 1
  - Viterbi likelihood ≤ total sequence likelihood
  - The failure state's row of $A$ is the identity (absorbing)
  - No future timestep enters a streaming inference call (causality test)

---

## 30. Reproducibility

Fixed random seeds per run, logged. All hyperparameters captured in versioned config files (YAML, e.g. via Hydra). Pinned environment (`requirements.txt`/lockfile). Dataset version/hash recorded. Model checkpoints saved per stage. Experiment tracking (e.g. MLflow or Weights & Biases — suggested, not mandated by this spec) logging all metrics and hardware information (needed later for honest latency benchmarking per Decision 7 in the resolved-conflicts table).

---

## 31. MVP Development Plan

| Milestone | Deliverable | Depends on | Acceptance criteria |
|---|---|---|---|
| M1 | Data pipeline | — | Passes leakage/causality tests on N-CMAPSS and MIMII |
| M2 | Encoder | M1 | Reconstructs a held-out window with reasonable error in a standalone autoencoder sanity check |
| M3 | VQ event discovery | M2 | Codebook perplexity stable, no collapse, over a full training run |
| M4 | HSMM | M3 | EM converges; forward-algorithm likelihood test (Section 29) passes |
| M5 | RUL head | M4 | Passes synthetic closed-form RUL unit test; runs on N-CMAPSS |
| M6 | Anomaly head | M4 | Threshold selection test passes; produces AUROC/AUPRC on MIMII |
| M7 | Fault + health heads | M4, M5 | Both run on N-CMAPSS with labeled data |
| M8 | Explanation object | M4–M7 | Schema-valid output for a sample run |
| M9 | Core baselines | M1 | All five core baselines (Section 21) run under identical data contracts |
| M10 | Essential ablations | M2–M8 | A1, A2, A2b, A4 all run with ≥3 seeds |
| M11 | Evaluation + write-up | M9, M10 | Full metrics report with CIs per Section 23's checklist |

---

## 32. Definition of Done

- **Engineering completion:** full pipeline (M1–M8) runs end-to-end on both primary datasets, all Section 29 tests pass, a run is fully reproducible from its config file alone.
- **Research completion:** core baselines (M9) and essential ablations (M10) complete with ≥3 seeds and reported confidence intervals.
- **Experimental completion:** Section 24's interpretability validation experiments (coherence, stability, cross-unit consistency, temporal validity, faithfulness) have all been run and reported — not merely defined.

---

## 33. Final Research Contribution

EGPM contributes a predictive-maintenance architecture in which a learned discrete event vocabulary and an explicit, duration-aware probabilistic state model (HSMM) jointly produce RUL, anomaly, fault, and health predictions from one shared, inspectable representation, with explanations read directly off that representation rather than approximated post hoc. The claim under test — not asserted as established — is that this constrained representation can match competitive black-box baselines on task metrics while providing measurably coherent, stable, and faithful explanations (Section 24). No claim of being "first" or that "no existing work" does this is made; the baseline suite (Section 21) exists specifically to situate this contribution against the nearest prior art rather than assert novelty by omission.

---

## Final Deliverables

**A. Final architecture summary.** Sensor data → deterministic preprocessing → trainable conv(+optional attention) encoder → trainable VQ event discovery → deterministic event sequence → trainable HSMM (probabilistic states + durations) → state posterior/duration statistics → four lightweight prediction heads → deterministic structured explanation. No continuous feature bypasses the event/state bottleneck in the primary model.

**B. Final mathematical dependency graph.**
```
X → z (encoder) → v, ẑ (VQ) → HSMM(π,A,B,D) → p_i, Viterbi path
p_i, Φ, μ → RUL (Section 12)
v_i, ŝ_i → anomaly score (Section 11)
p_{1:T} (pooled) → fault class (Section 13)
p_i, w → health index (Section 14)
all of the above → explanation object (Section 18)
```

**C. Final module/codebase dependency graph.**
```
data → preprocessing → encoder → quantization → grammar → {rul, anomaly, fault, health} → explanation
training depends on: data, preprocessing, encoder, quantization, grammar, all heads
evaluation depends on: training outputs + baselines
baselines depend on: data, preprocessing only (independent model implementations)
```

**D. First 10 implementation tasks.**
1. Implement unit-level dataset loaders + splitters for N-CMAPSS and MIMII, with leakage tests (M1).
2. Implement preprocessing (`sync`, `impute`, `window`, `normalize`) with causality tests.
3. Implement the encoder (conv stack; attention behind a flag) and a standalone autoencoder sanity check.
4. Implement the VQ codebook with the **corrected** straight-through estimator and EMA updates; add the perplexity/collapse monitor.
5. Implement the explicit-duration HSMM forward-backward and segmental Viterbi; validate against brute-force likelihood on a tiny synthetic instance.
6. Implement Baum-Welch (EM) fitting for the HSMM (Stage 2 training).
7. Implement the corrected phase-type RUL formula (Section 12) with its synthetic closed-form unit test.
8. Implement the anomaly scorer and validation-only threshold selection.
9. Implement the explanation-object assembler and schema validator.
10. Implement two core baselines (CNN-RUL, OmniAnomaly) under the same data contracts, to get comparison numbers as early as possible.

**E. First experiment to run.** Before scaling to the full pipeline: on **MIMII, anomaly detection only**, compare (i) a continuous-embedding baseline (e.g. a simple autoencoder reconstruction score), (ii) VQ events scored with a plain first-order Markov transition table (no duration model), and (iii) the full HSMM with duration modeling. This directly tests RQ1 and RQ2 cheaply, before any multi-task or multi-dataset investment, and specifically answers whether the HSMM's duration modeling — its most expensive component relative to a plain Markov chain — is earning its complexity.

**F. Biggest technical risk.** That the HSMM's duration-aware state structure does not outperform a much simpler discrete-event Markov model (ablation A2b) on real data — i.e., that the central architectural bet (explicit duration modeling) does not earn its computational and implementation cost. This is exactly what Deliverable E is designed to surface early, before it is buried under the rest of the pipeline.

**G. Exact MVP boundary.** Two datasets (N-CMAPSS, MIMII), four tasks, VQ-VAE (not Gumbel-softmax) event discovery, flat HSMM (no PCFG layer), five core baselines, four essential ablations (A1, A2, A2b, A4), no cross-fleet transfer, no edge-deployment optimization, no LLM-based explanation layer, no uncertainty intervals beyond the Monte Carlo RUL simulation marked Phase 2.
