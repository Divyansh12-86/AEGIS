# AEGIS (EGPM) — Architectural Component Domain Division

**Schneider Electric PredictX 2026**  
**Purpose:** Technical ownership breakdown between **Computer Science (CS/AI)** and **Electrical / CORE / EEE** engineering domains for architectural defense and jury Q&A.

---

## 1. End-to-End Architectural Stack Ownership Map

```text
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                AEGIS ARCHITECTURAL PIPELINE                                      │
├───────────────────────────────┬──────────────────────────────────┬───────────────────────────────┤
│ ARCHITECTURAL LAYER           │ CS / AI SPECIALIST RESPONSIBILITY│ EEE / CORE SPECIALIST RESPONSIBILITY
├───────────────────────────────┼──────────────────────────────────┼───────────────────────────────┤
│ 1. Telemetry & Transducers    │ Data shape contracts [B, W, C],  │ Physical sensors (PT100, Hall,│
│    (Sensors & Data Ingestion) │ streaming pipeline, PyTorch tensors│ Rogowski), thermal time constants
├───────────────────────────────┼──────────────────────────────────┼───────────────────────────────┤
│ 2. Causal Normalization       │ Zero future-leakage algorithm,   │ Physical window scale relative│
│    & Sliding Window (W=16)    │ streaming causal buffer          │ to 50/60 Hz & thermal drift   │
├───────────────────────────────┼──────────────────────────────────┼───────────────────────────────┤
│ 3. 1D Causal Convolutional    │ Dilated causal CNN mechanics,    │ Transducer cross-correlation, │
│    Sensor Encoder             │ receptive field, tensor gradient │ harmonic noise vs signal      │
├───────────────────────────────┼──────────────────────────────────┼───────────────────────────────┤
│ 4. Vector Quantization        │ Straight-Through Estimator (STE),│ Physical event semantics      │
│    (K=16 Discrete Codebook)   │ codebook loss, perplexity        │ (micro-arcing, blade flutter) │
├───────────────────────────────┼──────────────────────────────────┼───────────────────────────────┤
│ 5. Temporal Grammar Core      │ Baum-Welch EM, Segmental Viterbi,│ Non-exponential wear physics, │
│    (Explicit-Duration HSMM)   │ non-parametric duration matrix   │ 5 degradation physical regimes│
├───────────────────────────────┼──────────────────────────────────┼───────────────────────────────┤
│ 6. Symbolic Bottleneck Gate   │ Strict bottleneck enforcement,   │ Why raw feature bypass        │
│    (No-Raw-Bypass Contract)   │ Ablation A4 test verification    │ fails safety certifications   │
├───────────────────────────────┼──────────────────────────────────┼───────────────────────────────┤
│ 7. Four Multi-Task Heads:     │                                  │                               │
│    • Conformal RUL (Ridge)    │ LOEO calibration, non-conformity │ Arc-flash safety margins, RUL │
│    • Acoustic Anomaly Scorer  │ Spectrograms, 99th% threshold    │ Fan bearing cage spalling, CRAH
│    • Fault / Regime Head      │ Bottleneck GRU temporal pooling  │ Load envelopes, harmonics     │
│    • Health Index Head        │ Monotonic prior optimization     │ Asset degradation lifecycle   │
├───────────────────────────────┼──────────────────────────────────┼───────────────────────────────┤
│ 8. Explainability & CF Engine │ JSON ExplanationObject schema,   │ Technician root-cause action, │
│    (Counterfactual What-If)   │ perturbation faithfulness test   │ CF1 contact dwell reset impact│
├───────────────────────────────┼──────────────────────────────────┼───────────────────────────────┤
│ 9. Edge Runtime & Deployment  │ <15ms latency, <1.4 MB footprint,│ Schneider Harmony edge iPC,   │
│    (EcoStruxure Integration)  │ 260/260 automated test suite     │ EcoStruxure Asset Advisor     │
└───────────────────────────────┴──────────────────────────────────┴───────────────────────────────┘
```

---

## 2. Detailed Component-by-Component Responsibilities

### Component 1: Sensor Transducers & Causal Windowing
* **CS Domain Owns:**
  * **Tensor Dimensionality:** Enforces $[B, W, C]$ shape invariants ($W=16$ windows, $C=14$ channels).
  * **Causal Preprocessing:** Online streaming normalization using *only* historical running mean/variance; zero lookahead into future windows ($t$ never accesses $t+1$).
  * **Code Reference:** `egpm/preprocessing/preprocessor.py`, `egpm/data/split_builder.py`.
* **EEE Domain Owns:**
  * **Sensor Physics:** Transducer types (PT100 RTDs for contact/busbar temperature, Rogowski coils for current transients, high-frequency piezo accelerometers, partial discharge RF sensors).
  * **Window Physics ($W=16$):** Explaining why $W=16$ (downsampled steps) aligns with mechanical and thermodynamic time constants of electrical switchgear heating and cooling cycles.
  * **Noise Rejection:** Differentiating electrical switching noise from true degradation drift.

---

### Component 2: 1D Causal Convolutional Encoder
* **CS Domain Owns:**
  * **Causal CNN Architecture:** Dilated 1D convolutions with causal left-padding, ensuring receptive field does not leak future time steps.
  * **Gradient Mechanics:** Normalization layers (LayerNorm/BatchNorm) and residual connections configured for stable backpropagation.
  * **Code Reference:** `egpm/encoder/sensor_encoder.py`.
* **EEE Domain Owns:**
  * **Signal Feature Extraction:** Explaining how causal convolutions capture multi-channel phase shifts between current spikes and temperature rises (Joule heating delay $\Delta t = \tau_{\text{thermal}}$).

---

### Component 3: Vector Quantizer ($K=16$) with Corrected STE
* **CS Domain Owns:**
  * **Corrected Straight-Through Estimator (STE):**
    $$\hat{h} = h + \text{sg}(e_{k^*} - h)$$
    Forwards quantized codebook vector $e_{k^*}$ while allowing gradients $\frac{\partial \mathcal{L}}{\partial \hat{h}}$ to flow directly back to continuous encoder $h$.
  * **Codebook Metrics:** Codebook utilization, perplexity monitoring, and codebook dead-code revival.
  * **Code Reference:** `egpm/quantization/vq_codebook.py`.
* **EEE Domain Owns:**
  * **Physical Event Semantics:** Translating abstract integer tokens $v_t \in \{0, \dots, 15\}$ into concrete physical operating modes:
    * *Token #7:* Contact resistance spike + transient thermal rise (switchgear breaker fretting).
    * *Token #12:* High-frequency acoustic resonance chirp (cooling fan bearing cage instability).
    * *Token #3:* Nominal steady-state balanced operation.

---

### Component 4: Explicit-Duration Hidden Semi-Markov Model (HSMM)
* **CS Domain Owns:**
  * **Mathematical Algorithms:**
    * Baum-Welch Expectation-Maximization (EM) parameter estimation.
    * Segmental Viterbi dynamic programming decoder ($O(T \cdot M^2 \cdot D_{\max})$).
    * Causal Forward-Backward filtering: $p_t(m) = P(S_t = m \mid v_{1:t})$.
    * Non-parametric duration distribution matrix $p_m(d)$ for $d \in \{1, \dots, D_{\max}\}$.
  * **Code Reference:** `egpm/grammar/hsmm.py`, `egpm/grammar/forward_backward.py`, `egpm/grammar/viterbi.py`.
* **EEE Domain Owns:**
  * **Wear Physics vs Memoryless Fallacy:**
    * Standard HMMs/Markov chains assume geometric decay ($P(d) = (1-a_{ii})a_{ii}^{d-1}$), implying the asset is "as good as new" at every step (constant hazard rate).
    * Real electrical contacts and mechanical bearings suffer non-linear cumulative wear (wear accumulation hazard increases with dwell time).
  * **The 5 Physical States:**
    * State 1: Fresh / Commissioned Asset (Nominal contact resistance).
    * State 2: Bedding-In / Thermal Stabilization.
    * State 3: Incipient Micro-Fretting / Surface Oxidation.
    * State 4: Accelerated Thermal Degradation & High Resistance.
    * State 5: Absorbing Pre-Failure / Critical Arc-Fault Risk.

---

### Component 5: The Symbolic Bottleneck Gate
* **CS Domain Owns:**
  * **Bottleneck Invariant:** Primary heads are strictly prohibited from receiving continuous latent embeddings $z$. They take *only* $[p_t, \tau_t]$ (beliefs and elapsed dwell).
  * **Contract Verification:** Automated test `test_no_raw_feature_bypass_by_default()` enforces this by raising `TypeError` if continuous tensors are passed.
  * **Code Reference:** `egpm/fault/fault_head.py`, `tests/test_heads_and_explanation.py`.
* **EEE Domain Owns:**
  * **Engineering Certification & Trust:**
    * Black-box latent bypasses cannot be safety-certified by electrical utilities or field inspectors.
    * Forcing predictions through discrete states and dwell times guarantees that decisions are auditable under industrial safety standards (IEC 62271 for switchgear).

---

### Component 6: Multi-Task Heads & Uncertainty Quantification
* **Head [A]: Remaining Useful Life (RUL) & Conformal Calibration**
  * **CS Owns:** Causal Linear Ridge readout over $[p_t, \tau_t]$; Leave-One-Engine-Out (LOEO) Conformal calibration producing distribution-free intervals; why closed-form phase-type formula failed on DS02.
  * **EEE Owns:** Maintenance scheduling economics; using the 90% lower bound (e.g. "at least 31.8 cycles left") to coordinate planned breaker maintenance shutdowns rather than unplanned emergency outages.
* **Head [B]: Acoustic Anomaly Scorer**
  * **CS Owns:** Log-mel spectrogram extraction; Viterbi emission surprisal; validation-only 99th-percentile thresholding.
  * **EEE Owns:** MIMII fan acoustics mapping to Data Center CRAH/CRAC cooling units; preventing motor burnouts and thermal runaway in high-density server halls.
* **Head [C]: Operational Regime / Fault Indicator**
  * **CS Owns:** Bottleneck GRU temporal pooling; cross-entropy classification loss.
  * **EEE Owns:** Operational load envelope identification ($F_c$ electrical load states, harmonic saturation).
* **Head [D]: Health Degradation Index**
  * **CS Owns:** Monotonic degradation ranking loss; Spearman rank correlation verification.
  * **EEE Owns:** Overall asset health metric for fleet-wide ranking in EcoStruxure dashboards.

---

### Component 7: Explainability & Counterfactual Simulation
* **CS Domain Owns:**
  * **Deterministic Schema:** Assembles `ExplanationObject` JSON (events, Viterbi path, dwell estimates, anomaly surprisal components) with zero LLM dependency.
  * **Perturbation Faithfulness:** Validates explanation validity via masking tests (score drops 0.18–0.72).
  * **Code Reference:** `egpm/explanation/explanation_builder.py`, `tests/test_interpretability.py`.
* **EEE Domain Owns:**
  * **Technician Root-Cause Action:** Interpreting *why* the breaker flagged an alarm (e.g., stationary in State 4 for 8 cycles due to elevated contact temperature $T_{24}$).
  * **Counterfactual "What-If" Planning (CF1):** Simulating dwell reset—showing how a minor maintenance action (re-torquing contacts, lubricating mechanical linkages) resets state dwell and extends operational RUL by +32 cycles.

---

### Component 8: Edge Runtime & Commercial EcoStruxure Deployment
* **CS Domain Owns:**
  * **Computational Efficiency:** Sub-15ms inference latency, <1.4 MB memory footprint, zero PyTorch dependency required for inference (runs on lightweight C++ / ONNX runtime).
  * **Verification Suite:** 260 automated pytest tests passing in CI.
  * **Code Reference:** `scripts/demo_predictx.py`, `pyproject.toml`.
* **EEE Domain Owns:**
  * **Hardware Integration:** Hosting the engine on Schneider Harmony edge industrial PCs or EcoStruxure Micro Data Centers.
  * **SCADA / Telemetry Protocols:** Ingesting sensor streams over Modbus TCP, IEC 61850 (substation standard), and OPC-UA into the EcoStruxure Asset Advisor cloud.
  * **Quantified ROI:** $1.08M downtime savings in hyperscale facilities; $180,000/yr CAPEX savings in switchgear breaker fleets.

---

## 3. Quick Defense Cheat Sheet: Who Answers Which Judge Question

| Question Prompt From Jury | Domain Lead | 30-Second Answer Anchor |
| :--- | :---: | :--- |
| *"Why didn't you use an Attention / Transformer model?"* | **CS** | *"Transformers lack explicit dwell duration distributions, have $O(T^2)$ edge complexity, and attention weights are not causal explanations. Our HSMM models non-exponential wear directly in $O(T \cdot M^2 \cdot D_{\max})$."* |
| *"How does your N-CMAPSS turbofan data relate to a Masterpact breaker?"* | **EEE** | *"N-CMAPSS is our peer-reviewed surrogate for continuous thermodynamic wear. In a breaker, contact resistance heating ($I^2R$) and partial discharge follow identical non-linear degradation trajectories."* |
| *"How do you guarantee your model never leaks future information?"* | **CS** | *"Every layer is strictly causal: 1D convolutions use left-padding only, preprocessing normalizes with historical running statistics, and HSMM inference uses causal filtering $\alpha_t$, never smoothed non-causal $\gamma_t$."* |
| *"How does a facility manager use your conformal prediction interval?"* | **EEE** | *"Point estimates give false confidence. Our 90% conformal interval provides a guaranteed lower bound. If the lower bound is 32 cycles, the plant manager can safely schedule maintenance during the next weekend outage."* |
| *"Why does Vector Quantization cause an acoustic penalty on MIMII?"* | **CS** | *"Quantizing continuous spectrograms into 16 discrete tokens discards subtle high-frequency harmonic phase information. That's why we propose our late-fusion architecture: continuous features for acoustic triggers, discrete grammar for RUL and explanations."* |
| *"How do you deploy this on factory floor hardware?"* | **EEE** | *"AEGIS takes under 1.4 MB of memory and runs in 14.2 milliseconds. It runs directly on Schneider Harmony edge iPCs connected via Modbus TCP or IEC 61850 without requiring a cloud connection or edge GPU."* |
