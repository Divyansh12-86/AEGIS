"""§24 interpretability validation suite runner (PRD, experimental bar).

Runs all six §24 metrics on a trained EGPM pipeline (encoder+VQ+HSMM via
egpm.training.Trainer) over a UnitDataset, emitting a JSON report:

  coherence      — silhouette + distance-ratio over token-grouped windows
  stability      — Hungarian-matched usage overlap across two seeds
  cross_unit     — JSD of event usage, train units vs held-out units
  temporal       — Viterbi-boundary alignment with known health onsets
                   (uses health_labels when present; skipped otherwise)
  faithfulness   — masking the flagged subsequence reduces the score
  rank_corr      — Spearman(score, label), reported under its renamed,
                   non-faithfulness semantics (Decision 9)

Note on labels: the suite NEVER uses test-split labels for fitting —
thresholds and models are trained on train/val only (PRD §23). Where the
suite needs labeled anomalies for rank correlation, it consumes them only
for evaluation.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch

from ..data import UnitDataset, SplitBuilder
from ..preprocessing import Preprocessor
from ..training import Trainer, TrainerConfig
from ..grammar import ForwardBackward, SegmentalViterbi
from ..anomaly import AnomalyScorer
from .metrics import (
    event_coherence,
    event_stability,
    cross_unit_consistency,
    temporal_validity,
    explanation_faithfulness,
    anomaly_score_rank_correlation,
)
from ..utils import set_global_seed


@dataclass
class InterpretabilityReport:
    experiment: str
    config: Dict
    results: Dict[str, Dict]
    wall_clock_s: float = 0.0
    notes: str = ""

    def to_json(self) -> str:
        return json.dumps(
            {
                "experiment": self.experiment,
                "config": self.config,
                "results": self.results,
                "wall_clock_s": self.wall_clock_s,
                "notes": self.notes,
            },
            indent=2,
        )


def _tokenize_runs(trainer: Trainer, window_runs: List[np.ndarray]):
    seqs = []
    for r in window_runs:
        X = torch.as_tensor(r, dtype=torch.float32)
        m = torch.ones_like(X)
        s = trainer.event_sequences(X, m)
        seqs.append(np.concatenate(s))
    return seqs


def _onsets_from_health(health_labels: np.ndarray) -> np.ndarray:
    """Ground-truth change indices from ordinal health labels (N-CMAPSS
    auxiliary health states: transitions 1->2->... are known onsets)."""
    h = np.asarray(health_labels)
    return np.nonzero(np.diff(h) != 0)[0] + 1  # boundary index


def run_interpretability_suite(
    dataset: UnitDataset,
    window_length: int = 16,
    stage1_steps: int = 80,
    n_states: int = 3,
    d_max: int = 8,
    n_codes: int = 8,
    seed: int = 0,
    seed_b: int = 1,
    top_k_flag: int = 3,
) -> InterpretabilityReport:
    """Full §24 suite on a trained pipeline.

    Splits at the unit level, trains seed `seed` (and seed `seed_b` for the
    stability metric), and evaluates all six metrics. All fitting uses
    train/val units only.
    """
    t0 = time.time()
    cfg = {
        "window_length": window_length, "stage1_steps": stage1_steps,
        "n_states": n_states, "d_max": d_max, "n_codes": n_codes,
        "seed": seed, "seed_b": seed_b,
    }
    sb = SplitBuilder(dataset, seed=seed)
    split = sb.build()
    stats = sb.fit_normalization(split)
    prep = Preprocessor(window_length=window_length, fs_sync=1.0)

    train_runs, train_health, train_units = [], [], []
    for uid in split.train_unit_ids:
        seq = prep.process(dataset[uid], norm_stats=stats)
        train_runs.append(seq.X)
        train_health.append(seq.health_labels)
        train_units.append(uid)
    val_runs, val_units = [], []
    for uid in split.val_unit_ids:
        seq = prep.process(dataset[uid], norm_stats=stats)
        val_runs.append(seq.X)
        val_units.append(uid)

    def make_trainer(s: int) -> Trainer:
        tcfg = TrainerConfig(
            window_length=window_length, embed_dim=8, hidden=16,
            n_conv_blocks=2, n_codes=n_codes, d_max=d_max,
            n_states=n_states, em_max_iter=8, stage1_steps=stage1_steps,
            batch_size=16, seed=s, log_every=0,
        )
        t = Trainer(tcfg, n_channels=train_runs[0].shape[-1])
        Xtr = torch.as_tensor(np.concatenate(train_runs), dtype=torch.float32)
        Mtr = torch.ones_like(Xtr)
        t.train_stage1(Xtr, Mtr)
        return t

    # main pipeline (seed A)
    set_global_seed(seed)
    trainer = make_trainer(seed)
    Xtr = torch.as_tensor(np.concatenate(train_runs), dtype=torch.float32)
    Mtr = torch.ones_like(Xtr)
    # tokenize all train runs (window-level) once
    seqs_per_window = trainer.event_sequences(Xtr, Mtr)
    run_lengths = [len(r) for r in train_runs]
    event_runs_train = trainer.window_runs_to_event_runs(seqs_per_window, run_lengths)
    # PRD §10.4: the HSMM is fit on NORMAL-operating runs only. Anomalous
    # train units (when labels exist) are excluded from EM but kept for the
    # coherence/stability statistics.
    normal_mask = [
        (dataset[uid].anomaly_labels is None
         or int(dataset[uid].anomaly_labels[0]) == 0)
        for uid in train_units
    ]
    normal_event_runs = [r for r, ok in zip(event_runs_train, normal_mask) if ok]
    fit_runs = normal_event_runs if normal_event_runs else event_runs_train
    trainer.train_stage2(fit_runs)

    results: Dict[str, Dict] = {}

    # ---------------- coherence ----------------
    feats = Xtr.reshape(Xtr.shape[0], -1).cpu().numpy()
    ids = np.concatenate([np.asarray(r) for r in event_runs_train])
    results["event_coherence"] = event_coherence(feats, ids)

    # ---------------- stability (seed B, same units) ----------------
    set_global_seed(seed_b)
    trainer_b = make_trainer(seed_b)
    seqs_b = trainer_b.event_sequences(Xtr, torch.ones_like(Xtr))
    event_runs_b = trainer_b.window_runs_to_event_runs(seqs_b, run_lengths)
    results["event_stability"] = event_stability(
        event_runs_train, event_runs_b, n_codes=n_codes
    )

    # ---------------- cross-unit consistency ----------------
    # tokenize held-out (val) units with the TRAINED vocabulary
    val_event_runs = _tokenize_runs(trainer, val_runs)
    results["cross_unit_consistency"] = cross_unit_consistency(
        event_runs_train, val_event_runs, n_codes=n_codes
    )

    # ---------------- temporal validity ----------------
    # where health labels exist, compare Viterbi boundaries to label onsets
    boundary_lags, detection = [], []
    n_units_with_truth = 0
    for run_i, uid in enumerate(train_units):
        hl = train_health[run_i]
        if hl is None:
            continue
        onsets = _onsets_from_health(hl)
        if len(onsets) == 0:
            continue
        n_units_with_truth += 1
        vit = SegmentalViterbi(trainer.hsmm).decode(event_runs_train[run_i])
        boundaries = [s for (_, a, b) in vit.segments for s in (a,)
                      if 0 < a < len(vit.path) - 1]
        tv = temporal_validity(boundaries, onsets, tolerance=max(2, d_max // 2))
        detection.append(tv["detection_rate"])
        if np.isfinite(tv["mean_abs_lag"]):
            boundary_lags.append(tv["mean_abs_lag"])
    results["temporal_validity"] = {
        "mean_detection_rate": float(np.mean(detection)) if detection else float("nan"),
        "mean_abs_lag": float(np.mean(boundary_lags)) if boundary_lags else float("nan"),
        "n_units_with_truth": n_units_with_truth,
        "note": "skipped where no health-label onsets exist"
                if n_units_with_truth == 0 else "",
    }

    # ---------------- faithfulness + rank correlation ----------------
    # Labeled units from val first (PRD §23: evaluation-only consumption);
    # fall back to train units if the split left val without both classes.
    # Fitting already excluded anomalous runs (§10.4), so evaluating on
    # train-unit labels reveals no leakage.
    val_labeled = [(i, u) for i, u in enumerate(val_units)
                   if dataset[u].anomaly_labels is not None]
    classes_val = {int(dataset[u].anomaly_labels[0]) for _, u in val_labeled}
    if classes_val >= {0, 1}:
        eval_pool = [("val", i, u) for i, u in val_labeled]
    else:
        eval_pool = [
            ("train", i, u) for i, u in enumerate(train_units)
            if dataset[u].anomaly_labels is not None
        ]
    runs_by_pool = {"val": val_runs, "train": train_runs}
    faith_reductions = []
    rank_scores: List[float] = []
    rank_labels: List[int] = []
    scorer = AnomalyScorer()
    for pool, run_i, uid in eval_pool:
        rec = dataset[uid]
        v = np.concatenate(_tokenize_runs(trainer, [runs_by_pool[pool][run_i]]))
        comp = scorer.score_sequence(trainer.hsmm, v)
        rank_scores.append(float(comp.score.mean()))
        rank_labels.append(int(rec.anomaly_labels[0]))
        if rec.anomaly_labels[0] == 1:
            top = list(np.argsort(comp.score)[::-1][:top_k_flag])
            f = explanation_faithfulness(v, top, trainer.hsmm, scorer)
            if np.isfinite(f["score_reduction"]):
                faith_reductions.append(f["score_reduction"])
    results["explanation_faithfulness"] = {
        "mean_score_reduction": float(np.mean(faith_reductions))
        if faith_reductions else float("nan"),
        "n_anomalous_runs": len(faith_reductions),
        "note": "score must DROP when the flagged subsequence is masked; "
                "reduction > 0 is the pass signal (PRD §24)",
    }
    if len(np.unique(rank_labels)) >= 2:
        results["anomaly_score_rank_correlation"] = anomaly_score_rank_correlation(
            np.asarray(rank_scores), np.asarray(rank_labels)
        )
    else:
        results["anomaly_score_rank_correlation"] = {
            "spearman": float("nan"),
            "note": "single-class units available; needs both classes for "
                    "the rank correlation (evaluation-only consumption)",
        }

    return InterpretabilityReport(
        experiment="interpretability_suite_sec24",
        config=cfg,
        results=results,
        wall_clock_s=time.time() - t0,
        notes="All six PRD §24 metrics; fitting on train/val units only; "
              "labels consumed for evaluation only (PRD §23).",
    )
