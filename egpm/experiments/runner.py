"""Experiment runner (PRD Deliverable E, §19–§22, M11).

The PRD's FIRST experiment, before any multi-task or multi-dataset
investment: on MIMII-style anomaly data, compare
  (i)   a continuous-embedding baseline (window AE reconstruction score);
  (ii)  VQ events scored with a plain first-order Markov transition table
        (no duration model);
  (iii) the full HSMM with duration modeling.

This directly tests RQ1 and RQ2 cheaply and answers whether duration
modeling — the HSMM's most expensive component — earns its complexity
(Deliverable F's risk).

Also provides run_core_ablations(): the A1/A2/A2b/A4 arms with >=3 seeds
each, emitting a PRD §23-style JSON report (unit-level evaluation, seeds
and config logged).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch

from ..data import UnitDataset, SplitBuilder, MIMIILoader
from ..preprocessing import Preprocessor
from ..training import Trainer, TrainerConfig
from ..grammar import HSMM, ForwardBackward, FirstOrderHMMFitter
from ..grammar.hsmm import DurationHistogram
from ..anomaly import AnomalyScorer
from ..baselines import WindowAutoEncoder, TokenMarkovModel
from ..ablations import ContinuousHSMM, PooledEventModel
from ..evaluation import auroc
from ..utils import set_global_seed


@dataclass
class ExperimentReport:
    """JSON-serializable experiment report (PRD §23/§30 artifacts)."""

    experiment: str
    config: Dict
    seeds: List[int]
    results: Dict[str, Dict[str, float]]  # model -> {AUROC_mean, AUROC_std, ...}
    wall_clock_s: float = 0.0
    notes: str = ""

    def to_json(self) -> str:
        return json.dumps(
            {
                "experiment": self.experiment,
                "config": self.config,
                "seeds": self.seeds,
                "results": self.results,
                "wall_clock_s": self.wall_clock_s,
                "notes": self.notes,
            },
            indent=2,
        )


def _windows_of(dataset: UnitDataset, unit_ids, prep: Preprocessor, stats) -> List[np.ndarray]:
    return [prep.process(dataset[uid], norm_stats=stats).X for uid in unit_ids]


def _normal_anomaly_split(dataset: UnitDataset, seed: int):
    """Per-seed normal-only train split + eval sets (PRD §10.4, §21)."""
    normal_ids = [r.unit_id for r in dataset.records
                  if r.anomaly_labels is None or r.anomaly_labels[0] == 0]
    anom_ids = [r.unit_id for r in dataset.records
                if r.anomaly_labels is not None and r.anomaly_labels[0] == 1]
    rng = np.random.default_rng(seed)
    rng.shuffle(normal_ids)
    n_train = max(1, int(0.7 * len(normal_ids)))
    train_normals, eval_normals = normal_ids[:n_train], normal_ids[n_train:]
    if not eval_normals or not anom_ids:
        raise ValueError("need both normal-eval and anomalous units")
    return train_normals, eval_normals, anom_ids


def deliverable_e(
    dataset: Optional[UnitDataset] = None,
    n_seeds: int = 3,
    window_length: int = 32,
    stage1_steps: int = 120,
    n_states: int = 3,
    d_max: int = 8,
    n_codes: int = 8,
    verbose: bool = False,
) -> ExperimentReport:
    """Run the PRD's first experiment on a MIMII-shaped dataset.

    All three arms share: identical unit-level splits, identical
    preprocessing, identical tokenization where applicable — the PRD §21
    fair-comparison protocol.
    """
    if dataset is None:
        dataset = MIMIILoader.synthetic_like(
            n_units=16, frames_per_unit=320, anomaly_fraction=0.25, seed=0
        )
    seeds = list(range(n_seeds))
    per_seed: Dict[str, List[float]] = {
        "continuous_AE": [], "vq_markov": [], "full_hsmm": []
    }
    cfg = {
        "window_length": window_length, "stage1_steps": stage1_steps,
        "n_states": n_states, "d_max": d_max, "n_codes": n_codes,
        "n_units": len(dataset),
    }
    t0 = time.time()
    for seed in seeds:
        set_global_seed(seed)
        sb = SplitBuilder(dataset, seed=seed)
        split = sb.build()
        stats = sb.fit_normalization(split)
        train_normals, eval_normals, anom_ids = _normal_anomaly_split(dataset, seed)

        prep = Preprocessor(window_length=window_length, fs_sync=1.0)
        train_runs = _windows_of(dataset, train_normals, prep, stats)
        eval_norm = _windows_of(dataset, eval_normals, prep, stats)
        eval_anom = _windows_of(dataset, anom_ids, prep, stats)

        Xtr = torch.as_tensor(np.concatenate(train_runs), dtype=torch.float32)
        Mtr = torch.ones_like(Xtr)

        # ---------------- arm (i): continuous AE ----------------
        ae = WindowAutoEncoder(n_channels=Xtr.shape[-1],
                               window_length=Xtr.shape[1])
        opt = torch.optim.Adam(ae.parameters(), lr=1e-3)
        ae.train()
        for _ in range(stage1_steps):
            opt.zero_grad()
            rec = ae(Xtr, Mtr)
            loss = (((rec - Xtr) ** 2) * Mtr).mean()
            loss.backward()
            opt.step()
        ae.eval()

        def ae_scores(runs):
            out = []
            for r in runs:
                X = torch.as_tensor(r, dtype=torch.float32)
                m = torch.ones_like(X)
                out.append(ae.anomaly_scores(X, m).mean().item())
            return np.array(out)

        auc_ae = auroc(
            np.concatenate([ae_scores(eval_anom), ae_scores(eval_norm)]),
            np.concatenate([np.ones(len(eval_anom)), np.zeros(len(eval_norm))]),
        )
        per_seed["continuous_AE"].append(float(auc_ae))

        # ---------------- shared VQ tokenization (arms ii & iii) --------
        tcfg = TrainerConfig(
            window_length=window_length, embed_dim=8, hidden=16,
            n_conv_blocks=2, n_codes=n_codes, d_max=d_max,
            n_states=n_states, em_max_iter=8, stage1_steps=stage1_steps,
            batch_size=16, seed=seed, log_every=0,
        )
        trainer = Trainer(tcfg, n_channels=Xtr.shape[-1])
        trainer.train_stage1(Xtr, Mtr)
        run_lengths = [len(r) for r in train_runs]
        seqs = trainer.event_sequences(Xtr, Mtr)
        train_event_runs = trainer.window_runs_to_event_runs(seqs, run_lengths)

        def event_runs_of(window_runs):
            out = []
            for r in window_runs:
                X = torch.as_tensor(r, dtype=torch.float32)
                m = torch.ones_like(X)
                s = trainer.event_sequences(X, m)
                out.append(np.concatenate(s))
            return out

        ev_norm = event_runs_of(eval_norm)
        ev_anom = event_runs_of(eval_anom)

        # ---------------- arm (ii): VQ + first-order Markov -----------
        mm = TokenMarkovModel(n_events=n_codes, smoothing=0.5).fit(train_event_runs)

        def mm_scores(runs):
            return np.array([mm.score_run(v).mean() for v in runs])

        auc_mm = auroc(
            np.concatenate([mm_scores(ev_anom), mm_scores(ev_norm)]),
            np.concatenate([np.ones(len(ev_anom)), np.zeros(len(ev_norm))]),
        )
        per_seed["vq_markov"].append(float(auc_mm))

        # ---------------- arm (iii): full HSMM with duration ---------
        trainer.train_stage2(train_event_runs)

        def hsmm_scores(runs):
            out = []
            for v in runs:
                scorer = AnomalyScorer()
                out.append(scorer.score_sequence(trainer.hsmm, v).score.mean())
            return np.array(out)

        auc_hs = auroc(
            np.concatenate([hsmm_scores(ev_anom), hsmm_scores(ev_norm)]),
            np.concatenate([np.ones(len(ev_anom)), np.zeros(len(ev_norm))]),
        )
        per_seed["full_hsmm"].append(float(auc_hs))
        if verbose:
            print(f"seed {seed}: AE {auc_ae:.3f}  Markov {auc_mm:.3f}  HSMM {auc_hs:.3f}")

    results: Dict[str, Dict[str, float]] = {}
    for arm, vals in per_seed.items():
        arr = np.array(vals)
        results[arm] = {
            "AUROC_mean": float(arr.mean()),
            "AUROC_std": float(arr.std()),
            "AUROC_per_seed": vals,
        }
    return ExperimentReport(
        experiment="deliverable_E_first_experiment",
        config=cfg,
        seeds=seeds,
        results=results,
        wall_clock_s=time.time() - t0,
        notes="PRD Deliverable E: continuous-AE vs VQ+Markov vs full HSMM; "
              "unit-level splits, normal-only fitting, >=3 seeds (PRD §21/§23).",
    )


def run_core_ablations(
    dataset: Optional[UnitDataset] = None,
    n_seeds: int = 3,
    window_length: int = 32,
    stage1_steps: int = 100,
    n_states: int = 3,
    d_max: int = 8,
    n_codes: int = 8,
    verbose: bool = False,
) -> ExperimentReport:
    """Run the M10 essential ablations A1, A2, A2b on MIMII-style data
    (A4's skip-connection arm lives in the fault-head tests — the MVP has
    no multi-task training yet). Reports anomaly AUROC per arm per seed.

    A1: continuous Gaussian-emission HSMM on encoder embeddings (no VQ).
    A2: pooled unigram event model (no HSMM).
    A2b: VQ events + first-order HMM (geometric dwell, no explicit D).
    Primary: the full HSMM reference numbers (same as deliverable_e arm iii).
    """
    if dataset is None:
        dataset = MIMIILoader.synthetic_like(
            n_units=16, frames_per_unit=320, anomaly_fraction=0.25, seed=0
        )
    seeds = list(range(n_seeds))
    arms = ["A1_continuous_hsmm", "A2_pooled_events",
            "A2b_first_order_hmm", "primary_full_hsmm"]
    per_seed: Dict[str, List[float]] = {a: [] for a in arms}
    cfg = {
        "window_length": window_length, "stage1_steps": stage1_steps,
        "n_states": n_states, "d_max": d_max, "n_codes": n_codes,
        "n_units": len(dataset), "n_seeds": n_seeds,
    }
    t0 = time.time()
    for seed in seeds:
        set_global_seed(seed)
        sb = SplitBuilder(dataset, seed=seed)
        split = sb.build()
        stats = sb.fit_normalization(split)
        train_normals, eval_normals, anom_ids = _normal_anomaly_split(dataset, seed)
        prep = Preprocessor(window_length=window_length, fs_sync=1.0)
        train_runs = _windows_of(dataset, train_normals, prep, stats)
        eval_norm = _windows_of(dataset, eval_normals, prep, stats)
        eval_anom = _windows_of(dataset, anom_ids, prep, stats)

        Xtr = torch.as_tensor(np.concatenate(train_runs), dtype=torch.float32)
        Mtr = torch.ones_like(Xtr)

        # shared encoder for embeddings (A1 needs z, others need events)
        tcfg = TrainerConfig(
            window_length=window_length, embed_dim=8, hidden=16,
            n_conv_blocks=2, n_codes=n_codes, d_max=d_max,
            n_states=n_states, em_max_iter=8, stage1_steps=stage1_steps,
            batch_size=16, seed=seed, log_every=0,
        )
        trainer = Trainer(tcfg, n_channels=Xtr.shape[-1])
        trainer.train_stage1(Xtr, Mtr)
        run_lengths = [len(r) for r in train_runs]
        seqs = trainer.event_sequences(Xtr, Mtr)
        train_event_runs = trainer.window_runs_to_event_runs(seqs, run_lengths)

        # embeddings per run (for A1)
        def embed_runs(window_runs):
            out = []
            for r in window_runs:
                X = torch.as_tensor(r, dtype=torch.float32)
                m = torch.ones_like(X)
                with torch.no_grad():
                    z = trainer.encoder(X, m)
                out.append(z.cpu().numpy())
            return out

        def event_runs_of(window_runs):
            out = []
            for r in window_runs:
                X = torch.as_tensor(r, dtype=torch.float32)
                m = torch.ones_like(X)
                s = trainer.event_sequences(X, m)
                out.append(np.concatenate(s))
            return out

        emb_train = embed_runs(train_runs)
        emb_norm = embed_runs(eval_norm)
        emb_anom = embed_runs(eval_anom)
        ev_norm = event_runs_of(eval_norm)
        ev_anom = event_runs_of(eval_anom)

        def auc_of(score_norm, score_anom):
            return auroc(
                np.concatenate([score_anom, score_norm]),
                np.concatenate([np.ones(len(score_anom)), np.zeros(len(score_norm))]),
            )

        # ---------------- A1: continuous Gaussian HSMM ----------------
        # hsmm is a placeholder; fit_em replaces it (means/ivar start flat)
        rng = np.random.default_rng(seed)
        a1 = ContinuousHSMM(
            hsmm=HSMM(
                n_states=n_states, n_events=2,
                pi=rng.dirichlet(np.ones(n_states)),
                A=rng.dirichlet(np.ones(n_states), size=n_states),
                D=DurationHistogram.from_means(np.full(n_states, 2.0), d_max),
                d_max=d_max, seed=seed,
            ),
            means=np.zeros((n_states, tcfg.embed_dim)),
            inv_variances=np.ones((n_states, tcfg.embed_dim)),
        )
        a1.fit_em(emb_train, n_states=n_states, d_max=d_max,
                  max_iter=6, min_iter=2, tol=1e-3, seed=seed)
        a1_norm = [a1.log_likelihood(z) / len(z) for z in emb_norm]
        a1_anom = [a1.log_likelihood(z) / len(z) for z in emb_anom]
        # anomaly score = NEGATIVE mean log-lik (lower likelihood = anomaly)
        per_seed["A1_continuous_hsmm"].append(
            auc_of([-x for x in a1_norm], [-x for x in a1_anom])
        )

        # ---------------- A2: pooled unigram events -------------------
        pooled = PooledEventModel(n_events=n_codes).fit(train_event_runs)
        a2_norm = [pooled.score_run(v).mean() for v in ev_norm]
        a2_anom = [pooled.score_run(v).mean() for v in ev_anom]
        per_seed["A2_pooled_events"].append(auc_of(a2_norm, a2_anom))

        # ---------------- A2b: VQ events + first-order HMM ------------
        fitter = FirstOrderHMMFitter(max_iter=10, seed=seed)
        hmm_fit, _ = fitter.fit(train_event_runs, n_states=n_states, n_events=n_codes)

        def hmm_ll_runs(runs):
            return [hmm_fit.forward_loglik(v) / len(v) for v in runs]

        a2b_norm = hmm_ll_runs(ev_norm)
        a2b_anom = hmm_ll_runs(ev_anom)
        per_seed["A2b_first_order_hmm"].append(
            auc_of([-x for x in a2b_norm], [-x for x in a2b_anom])
        )

        # ---------------- primary: full HSMM --------------------------
        trainer.train_stage2(train_event_runs)

        def hsmm_ll_runs(runs):
            out = []
            for v in runs:
                fb = ForwardBackward(trainer.hsmm)
                out.append(fb.forward(v)[1] / len(v))
            return out

        hs_norm = hsmm_ll_runs(ev_norm)
        hs_anom = hsmm_ll_runs(ev_anom)
        per_seed["primary_full_hsmm"].append(
            auc_of([-x for x in hs_norm], [-x for x in hs_anom])
        )
        if verbose:
            print(f"seed {seed}: " + ", ".join(
                f"{a}={per_seed[a][-1]:.3f}" for a in arms))

    results = {
        a: {
            "AUROC_mean": float(np.mean(v)),
            "AUROC_std": float(np.std(v)),
            "AUROC_per_seed": v,
        }
        for a, v in per_seed.items()
    }
    return ExperimentReport(
        experiment="core_ablations_A1_A2_A2b",
        config=cfg,
        seeds=seeds,
        results=results,
        wall_clock_s=time.time() - t0,
        notes="M10 essential ablations (A4 lives in fault-head tests until "
              "multi-task training is wired). Unit-level splits, >=3 seeds.",
    )


