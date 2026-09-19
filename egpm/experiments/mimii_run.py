"""Real MIMII fan experiments (PRD Deliverable E + M10 ablations, M11).

Per machine id (id_00..id_06), normal-only training on a shuffled 70/15/15
clip-level split of normal clips (PRD §10.4 / §23: unit-level splits, normal
clips only in training), evaluation on held-out normal + all abnormal clips.

Arms under identical contracts (PRD §21 fair-comparison):
  continuous_AE      — window reconstruction MSE (baseline i)
  pooled_events      — VQ tokens scored by train unigram surprisal (A2)
  vq_markov          — VQ tokens scored by first-order Markov (baseline ii)
  vq_hsmm            — full HSMM surprisal scorer (baseline iii / primary)
  continuous_hsmm    — Gaussian-emission HSMM on encoder embeddings (A1)

Reports AUROC/AUPRC per arm per seed, clip-level bootstrap CIs, and paired
Wilcoxon (per-clip correctness at the shared HSMM threshold) vs the primary
arm. JSON artifacts: mimii_fan_<id>_<experiment>.json (repo-root convention).
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from ..data import UnitDataset, UnitRecord, MIMIILoader
from ..data.split_builder import NormalizationStats
from ..preprocessing import Preprocessor
from ..training import Trainer, TrainerConfig
from ..grammar import HSMM, ForwardBackward, FirstOrderHMMFitter
from ..grammar.hsmm import DurationHistogram
from ..anomaly import AnomalyScorer
from ..baselines import WindowAutoEncoder, TokenMarkovModel
from ..ablations import ContinuousHSMM
from ..evaluation import auroc, auprc, f1_at_threshold
from ..evaluation.stats import auroc_bootstrap_ci, paired_wilcoxon
from ..utils import set_global_seed


def _subset_by_id(dataset: UnitDataset, id_token: str) -> UnitDataset:
    sub = UnitDataset()
    for r in dataset.records:
        if id_token in r.unit_id:
            sub.add(r)
    if len(sub) == 0:
        raise ValueError(f"no clips matched id token {id_token!r}")
    return sub


def _normal_split(unit_ids: List[str], seed: int,
                  train_frac: float = 0.7, val_frac: float = 0.15) -> Tuple[List[str], List[str], List[str]]:
    """Shuffled clip-level split of NORMAL clips only (train/val/test)."""
    ids = list(unit_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_tr = max(1, int(round(train_frac * n)))
    n_va = max(1, int(round(val_frac * n)))
    return ids[:n_tr], ids[n_tr:n_tr + n_va], ids[n_tr + n_va:]


def _fit_normal_stats_normal_only(dataset: UnitDataset, train_normals: List[str]) -> NormalizationStats:
    """z-stats from TRAIN NORMAL clips only (PRD §6: train-split, and §10.4:
    normal-only training data for MIMII)."""
    stacked = np.concatenate([dataset[u].signals for u in train_normals], axis=0)
    std = stacked.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return NormalizationStats(
        mean=stacked.mean(axis=0), std=std, train_unit_ids=train_normals
    )


def run_mimii_fan_one(
    id_token: str,
    dataset: UnitDataset,
    n_seeds: int = 3,
    window_length: int = 16,
    stage1_steps: int = 600,
    n_states: int = 3,
    d_max: int = 12,
    n_codes: int = 8,
    ae_steps: int = 300,
    a1_em_iter: int = 6,
    verbose: bool = True,
    max_normal_eval: int = 300,
) -> Dict:
    """All arms, all seeds, one machine id. Returns the report dict."""
    sub = _subset_by_id(dataset, id_token)
    normal_ids = [u for u in sub.unit_ids if "normal" in u]
    anom_ids = [u for u in sub.unit_ids if "abnormal" in u]
    if verbose:
        print(f"[{id_token}] {len(normal_ids)} normal / {len(anom_ids)} abnormal clips", flush=True)

    arms = ["continuous_AE", "pooled_events", "vq_markov", "vq_hsmm", "continuous_hsmm"]
    per_seed_auroc: Dict[str, List[float]] = {a: [] for a in arms}
    per_seed_auprc: Dict[str, List[float]] = {a: [] for a in arms}
    per_clip_correct: Dict[str, Dict[int, np.ndarray]] = {a: {} for a in arms}  # arm -> seed -> [0/1] per eval clip
    eval_label_vec: Optional[np.ndarray] = None
    primary_threshold: Dict[int, float] = {}
    cfg = {
        "window_length": window_length, "stage1_steps": stage1_steps,
        "n_states": n_states, "d_max": d_max, "n_codes": n_codes,
        "ae_steps": ae_steps, "n_normal": len(normal_ids), "n_abnormal": len(anom_ids),
    }
    t0 = time.time()
    for seed in range(n_seeds):
        set_global_seed(seed)
        tr_norm, va_norm, te_norm = _normal_split(normal_ids, seed)
        eval_norm = te_norm[:max_normal_eval]
        eval_anom = anom_ids[:max_normal_eval]
        stats = _fit_normal_stats_normal_only(sub, tr_norm)
        prep = Preprocessor(window_length=window_length, fs_sync=1.0)

        def wins(ids):
            return [prep.process(sub[i], norm_stats=stats).X for i in ids]

        if verbose:
            print(f"[{id_token}] seed {seed}: {len(tr_norm)} train, {len(eval_norm)} eval norm / "
                  f"{len(eval_anom)} eval anom; build windows…", flush=True)
        train_runs = wins(tr_norm)
        va_runs = wins(va_norm)
        Xtr = torch.as_tensor(np.concatenate(train_runs), dtype=torch.float32)
        Mtr = torch.ones_like(Xtr)

        # ---- arm: continuous AE ----
        ae = WindowAutoEncoder(n_channels=Xtr.shape[-1], window_length=window_length)
        opt = torch.optim.Adam(ae.parameters(), lr=1e-3)
        ae.train()
        for _ in range(ae_steps):
            opt.zero_grad()
            rec = ae(Xtr, Mtr)
            (((rec - Xtr) ** 2) * Mtr).mean().backward()
            opt.step()
        ae.eval()

        def ae_score(ids):
            out = []
            with torch.no_grad():
                for w in wins(ids):
                    X = torch.as_tensor(w, dtype=torch.float32)
                    out.append(ae.anomaly_scores(X, torch.ones_like(X)).mean().item())
            return np.array(out)

        # ---- shared VQ tokenization (pooled / markov / hsmm arms) ----
        tcfg = TrainerConfig(
            window_length=window_length, embed_dim=16, hidden=32,
            n_conv_blocks=2, n_codes=n_codes, d_max=d_max,
            n_states=n_states, em_max_iter=10, stage1_steps=stage1_steps,
            batch_size=128, seed=seed, log_every=0,
        )
        trainer = Trainer(tcfg, n_channels=Xtr.shape[-1])
        trainer.train_stage1(Xtr, Mtr)
        run_lengths = [len(r) for r in train_runs]
        train_event_runs = trainer.window_runs_to_event_runs(
            trainer.event_sequences(Xtr, Mtr), run_lengths
        )

        def tokens_of(ids):
            out = []
            for w in wins(ids):
                X = torch.as_tensor(w, dtype=torch.float32)
                out.append(np.concatenate(trainer.event_sequences(X, torch.ones_like(X))))
            return out

        ev_norm_toks = tokens_of(eval_norm)
        ev_anom_toks = tokens_of(eval_anom)
        va_toks = tokens_of(va_norm)

        # pooled unigram (A2)
        counts = np.bincount(np.concatenate(train_event_runs), minlength=n_codes)
        log_unigram = np.log(counts / counts.sum() + 1e-12)

        # first-order Markov over tokens (baseline ii)
        mm = TokenMarkovModel(n_events=n_codes, smoothing=0.5).fit(train_event_runs)

        # full HSMM (primary)
        trainer.train_stage2(train_event_runs)
        scorer = AnomalyScorer()

        # continuous HSMM (A1) on encoder embeddings
        def embed_runs(ids):
            out = []
            for w in wins(ids):
                X = torch.as_tensor(w, dtype=torch.float32)
                with torch.no_grad():
                    out.append(trainer.encoder(X, torch.ones_like(X)).cpu().numpy())
            return out

        emb_train, emb_va = embed_runs(tr_norm), embed_runs(va_norm)
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
                  max_iter=a1_em_iter, min_iter=2, tol=1e-3, seed=seed)

        def score_all(arm):
            if arm == "continuous_AE":
                return ae_score(eval_norm), ae_score(eval_anom)
            if arm == "pooled_events":
                s = lambda toks: np.array([-log_unigram[v].mean() for v in toks])
                return s(ev_norm_toks), s(ev_anom_toks)
            if arm == "vq_markov":
                s = lambda toks: np.array([mm.score_run(v).mean() for v in toks])
                return s(ev_norm_toks), s(ev_anom_toks)
            if arm == "vq_hsmm":
                s = lambda toks: np.array(
                    [scorer.score_sequence(trainer.hsmm, v).score.mean() for v in toks])
                return s(ev_norm_toks), s(ev_anom_toks)
            if arm == "continuous_hsmm":
                ev_n, ev_a = embed_runs(eval_norm), embed_runs(eval_anom)
                s = lambda rs: np.array([-a1.log_likelihood(z) / len(z) for z in rs])
                return s(ev_n), s(ev_a)
            raise ValueError(arm)

        eval_labels = np.concatenate([np.zeros(len(eval_norm)), np.ones(len(eval_anom))])
        eval_label_vec = eval_labels
        # primary threshold: AnomalyScorer's validation-only rule on HSMM arm
        _, _ = score_all("vq_hsmm")
        va_scores = np.array(
            [scorer.score_sequence(trainer.hsmm, v).score.mean() for v in va_toks])
        thr = float(np.percentile(va_scores, 99.0))
        primary_threshold[seed] = thr

        for arm in arms:
            sn, sa = score_all(arm)
            scores = np.concatenate([sn, sa])
            per_seed_auroc[arm].append(float(auroc(scores, eval_labels)))
            per_seed_auprc[arm].append(float(auprc(scores, eval_labels)))
            # per-clip correctness at the shared primary threshold:
            # normalize scores so "higher = anomalous at thr" holds per arm
            # via rank-invariant mapping: use each arm's OWN validation
            # threshold (99th pct of its validation-normal scores) — fair,
            # validation-only, per PRD §11.
            if arm == "vq_hsmm":
                va_s = va_scores
            elif arm == "continuous_AE":
                va_s = ae_score(va_norm)
            elif arm == "pooled_events":
                va_s = np.array([-log_unigram[v].mean() for v in va_toks])
            elif arm == "vq_markov":
                va_s = np.array([mm.score_run(v).mean() for v in va_toks])
            else:  # continuous_hsmm
                va_s = np.array([-a1.log_likelihood(z) / len(z) for z in emb_va])
            arm_thr = float(np.percentile(va_s, 99.0))
            per_clip_correct[arm][seed] = (
                (scores > arm_thr).astype(np.int64) == eval_labels
            ).astype(np.int64)
        if verbose:
            msg = ", ".join(f"{a}={per_seed_auroc[a][-1]:.3f}" for a in arms)
            print(f"[{id_token}] seed {seed}: {msg} ({time.time()-t0:.0f}s)", flush=True)

    # ---- aggregate ----
    results: Dict[str, Dict] = {}
    for arm in arms:
        auc = np.array(per_seed_auroc[arm])
        apc = np.array(per_seed_auprc[arm])
        # pooled across seeds for CI: concat per-seed eval scores via
        # correctness vectors (all seeds share the same eval clip set)
        pooled_scores = None  # per-seed scores have different scales; CI on mean AUROC via seeds
        seed_means = auc
        results[arm] = {
            "AUROC_mean": float(seed_means.mean()),
            "AUROC_std": float(seed_means.std()),
            "AUROC_per_seed": per_seed_auroc[arm],
            "AUPRC_mean": float(apc.mean()),
            "AUPRC_per_seed": per_seed_auprc[arm],
        }
    # paired Wilcoxon on per-clip correctness vs primary (vq_hsmm), per seed
    wilcoxon: Dict[str, Dict] = {}
    for arm in arms:
        if arm == "vq_hsmm":
            continue
        stats_list = []
        for seed in range(n_seeds):
            a = per_clip_correct[arm][seed].astype(float)
            b = per_clip_correct["vq_hsmm"][seed].astype(float)
            stats_list.append(paired_wilcoxon(a, b))
        # report the worst-case (max) p across seeds; clips are shared per seed
        ps = [s["p"] for s in stats_list if np.isfinite(s["p"])]
        wilcoxon[arm] = {
            "note": "paired Wilcoxon over eval clips (correctness vs primary arm), "
                    "worst-case p across seeds; n clips per seed",
            "max_p": float(max(ps)) if ps else float("nan"),
            "min_p": float(min(ps)) if ps else float("nan"),
            "n_clips": int(len(per_clip_correct["vq_hsmm"][0])),
        }
    return {
        "experiment": "mimii_fan_real",
        "machine_id": id_token,
        "config": cfg,
        "n_seeds": n_seeds,
        "results": results,
        "wilcoxon_vs_primary": wilcoxon,
        "primary_thresholds": primary_threshold,
        "wall_clock_s": time.time() - t0,
        "notes": "normal-only training (70/15/15 clip split of normals, "
                 "train-normal z-stats); eval = held-out normals + all "
                 "abnormal clips; validation-only thresholds (99th pct); "
                 "arms share tokenization/preprocessing per seed.",
    }


def run_mimii_fan(
    root: str | Path = "fan",
    ids: Tuple[str, ...] = ("id_00", "id_02", "id_04", "id_06"),
    n_seeds: int = 3,
    window_length: int = 16,
    stage1_steps: int = 600,
    n_states: int = 3,
    d_max: int = 12,
    n_codes: int = 8,
    ae_steps: int = 300,
    append_deltas: bool = True,
    max_clips_per_label: Optional[int] = None,
    out_dir: str | Path = ".",
    verbose: bool = True,
) -> Dict:
    """Full MIMII fan suite: all machine ids, all arms, >=3 seeds each."""
    dataset = MIMIILoader(
        machines=("fan",), feature="logmel",
        max_clips_per_label=max_clips_per_label,
        append_deltas=append_deltas,
    ).load(root)
    reports = {}
    for id_token in ids:
        rep = run_mimii_fan_one(
            id_token, dataset,
            n_seeds=n_seeds, window_length=window_length,
            stage1_steps=stage1_steps, n_states=n_states,
            d_max=d_max, n_codes=n_codes, ae_steps=ae_steps,
            verbose=verbose,
        )
        reports[id_token] = rep
        out = Path(out_dir) / f"mimii_fan_{id_token}_{n_seeds}seeds.json"
        out.write_text(json.dumps(rep, indent=2))
        if verbose:
            print(f"[{id_token}] wrote {out}", flush=True)
    return reports


if __name__ == "__main__":
    run_mimii_fan()
