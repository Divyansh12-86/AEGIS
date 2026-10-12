# MIMII Implementation Plan — wiring the real `fan/` dataset

Status: plan. Data on disk verified 2026-09-15.

## 0. Verified facts about the on-disk data

- Layout: `fan/id_XX/{normal,abnormal}/NNNNNNNN.wav` (4 machine IDs: id_00, id_02, id_04, id_06).
- Every wav: 16 kHz, **8 channels**, 16-bit PCM, 160 000 frames = 10.0 s.
- Counts: normal 1011/1016/1033/1015, abnormal 407/359/348/361 (4075 + 1475 clips).
- Filenames **collide across normal/abnormal** (`00000000.wav` exists in both) — unit IDs must be namespaced.
- **No dB tokens in any path** (`0dB/6dB/min6dB` appear nowhere).
- 13 GB, currently untracked in git — must be gitignored, never committed.

## 1. Gaps found in existing code (read, not guessed)

`egpm/data/loaders.py::MIMIILoader.load` (lines 316–360) is the only real-code
change needed, but it currently **rejects or mangles this dataset**:

1. **dB filter kills everything**: `if not any(db in name or db in str(rel) for db in self.dbs)`
   — no dB string exists in these paths → 0 matches → `FileNotFoundError`.
2. **Mono assumption is wrong**: `sf.read` returns `[160000, 8]`; the code does
   `audio.reshape(-1, 1)` → mangles 8 channels into a fake 1.28 M-sample mono signal.
3. **Anomaly label from the wrong place**: `is_anom = "anomaly" in wav.stem` — the
   stem never contains "anomaly"; the label lives in the *directory* name (`abnormal`).
   Result: every clip labeled normal.
4. **unit_id collision**: `mimii_{wav.stem}` collides for normal/abnormal pairs →
   `UnitDataset.add` raises on duplicates.

Everything else is reusable as-is (rung 2): `UnitDataset/SplitBuilder/audit`,
`Preprocessor` (window/normalize, causal), `Trainer` (stage1 VQ + stage2 HSMM EM),
`AnomalyScorer` (validation-only threshold), `auroc/f1_at_threshold`,
`per-unit stats/bootstrap/Wilcoxon`, `deliverable_e`, `run_core_ablations`,
`run_interpretability_suite` — all already accept a `UnitDataset`.

## 2. Design decisions (ladder applied)

- **Feature front-end: log-mel, not raw waveforms.** MIMII anomaly cues are
  spectral (bearing tones, imbalance harmonics); every MIMII baseline uses
  spectral features; a 32-sample raw window at 16 kHz is 2 ms of audio and
  carries no machine-sound structure. Front-end goes **inside the MIMII load
  path** (`UnitRecord.signals` = frame features, timestamps = frame-end times),
  so `Preprocessor/Trainer/runners` run untouched.
- **Causality preserved**: STFT frames defined to *end* at their last sample
  (no centered/boundary padding); a frame's data never includes samples after
  its end — satisfies PRD §6's window-causality rule.
- **Unit = one 10 s clip.** Clip-level splits = PRD unit-level splits. ~1400
  clips per ID → unit-level stats finally well powered (fixes the "Wilcoxon
  NaN with 2 test units" limitation from N-CMAPSS).
- **Per-machine-ID models** (id_00/02/04/06 trained and evaluated separately),
  matching the MIMII protocol — each machine sounds different; pooling would
  blur anomaly signal. Implemented as a loop over id-subsets; no new code paths.
- **scipy + numpy only** for the front-end (`scipy.signal.stft` + ~25-line mel
  filterbank). No librosa, no new dependency.
- **Normal-only training** per PRD §10.4 — `_normal_anomaly_split` in
  `runner.py` already does exactly this (70% normals train, rest eval).

## 3. Phases

### Phase 1 — git hygiene (1 line)
`.gitignore`: add `fan/` (next to the existing `N-CMAPSS/*.h5` rules).

### Phase 2 — fix `MIMIILoader` (`egpm/data/loaders.py`, ~40 changed lines)
- Detect the `id_XX/{normal,abnormal}` layout from the relative path parts;
  keep the legacy `machine/dB/normal/...` layout working (it's covered by tests).
- `is_anom = "abnormal" in rel.parts` (directory label).
- Multichannel: `signals = audio` as `[T, C]` (no reshape when `sf.read`
  returns 2-D).
- dB filter only applies when dB tokens actually appear in paths (or an
  explicit `dbs` config is given); default = no filtering.
- unit_id: `mimii_fan_id00_normal_00000000` — machine, id, split, stem.
- Add `max_clips_per_label: Optional[int]` (default None) so tests and smoke
  runs don't pay for 5500 clips; keeps 41 GB float64 RAM blow-up off the table.
- Keep the 16 kHz fs check and the 10 s trim.

### Phase 3 — log-mel front-end (`egpm/preprocessing/spectral.py`, new, ~45 lines)
- `logmel_features(wav: [T, C], fs, n_fft, hop, n_mels) -> [F, C*n_mels]`:
  causal framing + STFT (scipy), magnitude, mel filterbank (numpy, Slaney
  scale), `log(x + eps)`, channels concatenated on axis 1.
- Defaults (knobs, not constants — PRD §26 dataset-dependent):
  `n_fft=1024` (64 ms), `hop=256` (16 ms) → 62.5 fps → ~624 frames/clip,
  `n_mels=64` → d = 8×64 = 512.
- Wired via `MIMIILoader(feature="logmel", ...)`; loader emits frame-rate
  timestamps `arange(F)/fps`. Downstream call sites pass
  `Preprocessor(window_length=W_frames, fs_sync=62.5)` → identity resample
  (verified against `synchronize`'s grid math: T_out = F).
- `feature="raw"` keeps today's behavior (synthetic tests stay valid).

### Phase 4 — evaluation gaps (`egpm/evaluation/`, ~20 lines total)
- `evaluator.py`: `auprc(scores, labels)` — average precision via `rankdata`
  (~8 lines, no sklearn). PRD §20 requires AUROC **and** AUPRC.
- `stats.py`: `auroc_bootstrap_ci(scores, labels, n_boot=2000)` — resample
  clips, recompute AUROC (~12 lines, reuses the rng pattern).
- Early-detection rate: **degenerate for MIMII** (onset = clip start); report
  recall-at-threshold instead and note why. No new code.

### Phase 5 — experiment suite (`egpm/experiments/mimii_run.py`, new, ~90 lines)
One entry point following the repo's function+JSON convention:

```
run_mimii_fan(root="fan", ids=("id_00","id_02","id_04","id_06"), n_seeds=3)
```

Per id (loop over a 3-line `UnitDataset` prefix-subset):
1. `MIMIILoader(machines=("fan",), feature="logmel", ...).load(root)`, subset by id.
2. `deliverable_e(dataset, ...)` → arms (i) AE, (ii) VQ+Markov, (iii) HSMM.
3. `run_core_ablations(dataset, ...)` → A1/A2/A2b vs primary.
4. `AnomalyScorer.fit_threshold` on validation normal-only clips → F1@threshold,
   recall@threshold on eval (M6 wiring on real data).
5. `run_interpretability_suite(dataset, ...)` → §24 six metrics.
6. Write `mimii_fan_<id>_<experiment>.json` at repo root (matches
   `ablation_report_DS01.json` convention), including per-seed AUROCs,
   clip-level bootstrap CIs, config, and the §23 checklist booleans.

Config defaults for real audio (knobs left): `window_length=16` frames
(~256 ms), `n_codes=32`, `n_states=4`, `d_max=12`, `stage1_steps=400`,
`embed_dim=16` (bigger than the synthetic 8 — 512-d features need it).
Per-clip sequences are ~38 tokens (W=S=16) → HSMM EM cheap.

§23 statistics: AUROC + AUPRC with clip-level bootstrap CIs per arm; paired
Wilcoxon across arms on **per-clip correctness at the shared threshold**
(raw scores live on incomparable scales across arms — correctness is the
honest paired unit; ~675 eval clips per id is well powered).

### Phase 6 — tests (`tests/test_data_pipeline.py`, ~110 lines)
Follow the repo's two existing patterns:
1. **Synthetic-wav loader test (runs everywhere, no 13 GB needed)**: write
   tiny 8-ch wavs into `tmp_path/id_00/{normal,abnormal}/`, assert —
   signals `[F, 512]`, anomaly labels from directories, unit-id uniqueness
   across normal/abnormal stems, dB-less paths load, `max_clips_per_label`
   cap works.
2. **`TestMIMIIRealFiles`** (skipif `fan/` absent, mirrors
   `TestNCMAPSSRealFiles`): capped load of one id; shapes/labels/split +
   `SplitBuilder.audit` all-true; `Preprocessor` causality on a real clip.
3. `auprc`/`auroc_bootstrap_ci` checks on constructed cases (one known-answer
   case each).
4. One log-mel self-check: frame k of the features must not change when
   samples after `(k+1)*hop` are zeroed (causality invariant).
5. `pytest tests/ -v` — all 223 existing tests must still pass.

### Phase 7 — run + report
- `python -c "from egpm.experiments.mimii_run import run_mimii_fan; run_mimii_fan()"`
- Update README: M6/M11 "pending dataset wiring" → real numbers; Definition of
  Done items 2/3 close.

## 4. Compute budget (rough)

Per id per seed: ~700 train clips × ~39 windows = ~27 k windows × [16×512]
float32 ≈ 850 MB working set; stage1 (400 steps × batch 64) minutes on
CPU/MPS. Suite = 4 ids × 3 seeds × (E + ablations + interpretability) ≈
30–90 min total. Fits a laptop run; if not, `max_clips_per_label` is the knob.

## 5. Risks

- **VQ collapse on real audio** — mitigated already: `init_codebook_from_data`
  seeds on the data manifold + EMA + perplexity monitor (M3).
- **HSMM on short sequences** (~38 tokens/clip): EM across ~700 runs per id
  gives it plenty of data; `d_max=12 < 38` so segmental Viterbi is well-posed.
- **z-score on log-mel** — standard; `fit_normalization` already guards
  zero-std columns.
- **Class imbalance in eval** (~300 normal vs ~375 anomalous per id) — mild;
  AUPRC reported alongside AUROC for exactly this reason.

## 6. Explicitly skipped (YAGNI / out of scope)

- OmniAnomaly proper (Phase-2 per README), ToyADMOS, per-dB variants (this
  download has no dB split), resampling, streaming/incremental loading,
  librosa/torchaudio dependency, early-detection-rate on clip-level labels.

## 7. Total footprint

~200 lines of new code + ~40 changed in `loaders.py` + ~110 test lines.
Only one real-code file is touched (`loaders.py`); everything else is additive.
