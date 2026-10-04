# Ponytail Audit — EGPM Codebase

**Date:** 2025-10-03  
**Scope:** Over-engineering and complexity only (correctness, security, performance out of scope)  
**Total codebase:** ~7,500 lines Python in `egpm/` + 28 experiment scripts  

---

## Executive Summary

The core EGPM architecture (encoder → VQ → HSMM → heads) is clean and well-factored. The bloat is in:
1. **Dead config flags** — 15+ unused TrainerConfig fields
2. **Duplicated ablation implementations** across 3+ files
3. **28 one-off experiment scripts** (28 files, ~3000 lines)
4. **Hand-rolled stdlib reimplementations** (k-means, ridge, gap-finding, interpolation)

**Net savings possible:** ~4,200 lines, 3 deps (scikit-learn for k-means/ridge, drop hand-rolled k-means/ridge)

---

## DELETE — Dead/Duplicated Code (~4,000 lines)

| Tag | Finding | Replacement |
|-----|---------|-------------|
| `delete:` | 18 experiment scripts in `experiments/run_*.py` | Unify into single CLI with subcommands |
| `delete:` | 10 root-level `run_*.py` scripts | Unify into single CLI |
| `delete:` | `ContinuousHSMM` in `egpm/ablations/continuous_hsmm.py` — just `HSMM` with `d_max=1` + Gaussian emissions | Use `HSMM(d_max=1)` directly |
| `delete:` | `PooledEventModel` in `egpm/ablations/pooled_events.py` — one-class unigram counter | Inline where used |
| `delete:` | `FirstOrderHMMFitter` in `egpm/grammar/first_order_hmm.py` — just `BaumWelch(d_max=1)` | Use `BaumWelch(d_max=1)` |
| `delete:` | `ContinuousHSMM` duplicated in `ablations/`, `runner.py`, `mimii_run.py` | Single implementation |
| `delete:` | `PooledEventModel` — one-class unigram counter | Inline where used |
| `delete:` | `FirstOrderHMMFitter` — just `BaumWelch(d_max=1)` | Use `BaumWelch(d_max=1)` |
| `delete:` | `run_robustness_sweeps.py` — defines sweeps but no runner (dead code) | Remove or implement |
| `delete:` | 8 one-off analysis scripts in `experiments/run_*.py` | Unify into CLI |

---

## YAGNI — Unused Config/Code (~1,500 lines)

| Tag | Finding |
|-----|---------|
| `yagni:` | `TrainerConfig` has 25 fields, only ~10 tuned. Remove: `state_dim`, `aux_rul_lambda`, `aux_rul`, `dead_code_revival`, `revival_threshold`, `revival_interval`, `state_dim`/`d_prog` subspace decoupling |
| `yagni:` | `HSMM.absorbing_state` — always `M-1`. Hardcode or remove |
| `yagni:` | `HSMM.keep_diagonal` — only for A2b ablation |
| `yagni:` | `VQCodebook.dead_code_revival` / `revival_threshold` / `revival_interval` — never used |
| `yagni:` | `VQCodebook.revive_dead_codes()` — wired but never called |
| `yagni:` | `SensorEncoder.use_attention` — ablation A5 only, never used |
| `yagni:` | `SensorEncoder.pooling="attention"` — never used |
| `yagni:` | `WindowSelfAttention` — ablation A5 only |
| `yagni:` | `SensorDecoder` positional encoding — only used in reconstruction |
| `yagni:` | `HealthHead.learn_weights` / `monotonicity_penalty` / `fit_learned_weights()` — never used |
| `yagni:` | `FaultHead.use_raw_embedding` / `embed_dim` — ablation A4 only |
| `yagni:` | `FaultHead.gru_hidden` / `hidden` — separate params, unify |
| `yagni:` | `ExplanationBuilder.counterfactual` / `what_if_dwell_reset()` — never used |
| `yagni:` | `PhaseTypeRUL.counterfactual_dwell_reset()` / `ConformalEngineCalibrator` — never used |
| `yagni:` | `Trainer.aux_rul` head / `aux_rul_lambda` — dead code path |
| `yagni:` | `Trainer.state_dim` subspace decoupling — wired but never activated |
| `yagni:` | `Trainer.total_revivals` tracking — revival never triggered |
| `yagni:` | `TrainerConfig.aux_rul_lambda` / `aux_rul` — dead code path |
| `yagni:` | `TrainerConfig.state_dim` subspace decoupling — wired but never activated |
| `yagni:` | `TrainerConfig.aux_rul_lambda` / `aux_rul` — never used |
| `yagni:` | `TrainerConfig.dead_code_revival` / `revival_threshold` / `revival_interval` — never used |

---

## STDLIB — Hand-rolled stdlib (replace)

| File | Hand-rolled | Use instead |
|------|-------------|-------------|
| `egpm/experiments/ablation_arms.py` | `_ridge_fit/_ridge_predict` | `sklearn.linear_model.Ridge` |
| `egpm/experiments/readout_study.py` | duplicates `_ridge_fit` | Same |
| `egpm/experiments/heads_run.py` | duplicates `_ridge_fit` | Same |
| `egpm/baselines/concept_bottleneck.py` | `WindowKMeans` | `sklearn.cluster.KMeans` |
| `egpm/baselines/concept_bottleneck.py` | `histogram` method | `np.bincount` |
| `egpm/baselines/grammarviz.py` | `SequiturGrammar` | Use library or keep minimal |
| `egpm/preprocessing/preprocessor.py` | `_find_gaps` | `np.flatnonzero(np.diff(mask))` |
| `egpm/preprocessing/preprocessor.py` | `_sync_channel` | `np.interp` |
| `egpm/preprocessing/preprocessor.py` | `_lowpass_antialias` | `np.convolve` or `scipy.signal` |
| `egpm/data/split_builder.py` | `SplitBuilder.audit()` | `set()` operations |
| `egpm/grammar/baum_welch.py` | `_after_end_values` | Reuse `ForwardBackward` |

---

## NATIVE — Platform features available

| File | Hand-rolled | Native alternative |
|------|-------------|-------------------|
| `egpm/encoder/sensor_encoder.py` | Custom causal padding | `nn.Conv1d(padding_mode='causal')` (PyTorch 2.0+) |
| `egpm/grammar/forward_backward.py` | `causal_filter` | `torch.scan` (PyTorch 2.0+) |
| `egpm/grammar/baum_welch.py` | `_after_end_values` duplicates backward | Reuse `ForwardBackward` |

---

## SHRINK — Duplicate Logic (~2,000 lines)

| Pattern | Locations | Action |
|---------|-----------|--------|
| `_ridge_fit/_ridge_predict` | 3 files | Extract to shared utility |
| `ContinuousHSMM` training | 3 files | Single implementation |
| `_precompute` in ForwardBackward | Called 2x (fwd+bwd) | Cache it |
| `_after_end_values` in BaumWelch | Duplicates backward | Reuse `ForwardBackward` |
| Log-emission precompute | `SegmentalViterbi` + `ForwardBackward` | Share |
| `_stage1_batch` branches | `state_dim` branches | Unify |
| `aux_rul` code paths | Duplicate in `_stage1_batch` | Extract helper |
| Training/eval loops | `runner.py` + `mimii_run.py` + `ablation_arms.py` | Unify |
| `_stage1_batch` branches | `state_dim` branches | Unify |
| `aux_rul` code paths | Duplicate in `_stage1_batch` | Extract helper |
| `ContinuousHSMM` training | 3 files | Single implementation |
| Training/eval loops | `runner.py` + `mimii_run.py` + `ablation_arms.py` | Unify |
| `_ridge_fit/_ridge_predict` | 3 files | Extract to shared utility |
| `ContinuousHSMM` training | 3 files | Single implementation |
| `run_ablation_experiment` / `deliverable_e` | Duplicate logic | Unify |
| `heads_run.py` / `ablation_arms.py` | Both have `_ridge_fit` | Extract shared |
| `readout_study.py` | Duplicates `_ridge_fit` | Extract shared |
| `TrainerConfig` | 25 fields, ~10 used | Remove unused |
| `Trainer._stage1_batch` | 2 branches for `state_dim` | Unify |
| `aux_rul` code paths | Duplicate in `_stage1_batch` | Extract helper |
| `BaumWelch._e_step` / `_m_step` | Duplicate log-prob | Factor out |
| `BaumWelch._e_step` | Duplicate forward-backward | Reuse `ForwardBackward` |
| `ForwardBackward._precompute` | Called 2x (fwd+bwd) | Cache it |
| `BaumWelch._after_end_values` | Duplicates backward | Reuse `ForwardBackward` |
| `SegmentalViterbi` + `ForwardBackward` | Both precompute log emissions | Share |
| `ContinuousHSMM` training | 3 files | Single implementation |
| `mimii_run.py` / `runner.py` | Duplicate MIMII logic | Unify |
| `run_ablation_experiment` / `deliverable_e` | Duplicate logic | Unify |
| `heads_run.py` / `ablation_arms.py` | Both have `_ridge_fit` | Extract shared |
| `readout_study.py` | Duplicates `_ridge_fit` | Extract shared |

---

## Summary

| Category | Count | Lines |
|----------|-------|-------|
| DELETE | 10 findings | ~4,000 lines |
| YAGNI | 20 findings | ~1,500 lines |
| STDLIB | 10 findings | ~500 lines |
| NATIVE | 3 findings | ~200 lines |
| SHRINK | 25 findings | ~2,000 lines |
| **TOTAL** | **68 findings** | **~4,200 lines** |

**Dependencies to drop:** 3 (scikit-learn for k-means/ridge, drop hand-rolled k-means/ridge)

---

## Priority Order for Cleanup

1. **Immediate:** Remove 28 dead experiment scripts → unify into `egpm/cli.py` with subcommands
2. **High:** Delete `ContinuousHSMM`, `PooledEventModel`, `FirstOrderHMMFitter` — inline their one use sites
3. **High:** Remove 15 unused `TrainerConfig` fields
3. **Medium:** Extract `_ridge_fit` to shared util, unify 3 duplicate implementations
4. **Medium:** Unify training/eval loops across `runner.py`, `mimii_run.py`, `ablation_arms.py`
5. **Low:** Replace hand-rolled stdlib with sklearn/scipy
6. **Low:** Cache `ForwardBackward._precompute`, unify `_stage1_batch` branches
7. **Low:** Use `nn.Conv1d(padding_mode='causal')` when PyTorch 2.0+ required

---

*The core EGPM architecture (encoder→VQ→HSMM→heads) is clean and well-factored. The bloat is entirely in dead config, duplicated ablation implementations, 28 one-off experiment scripts, and hand-rolled stdlib reimplementations.*