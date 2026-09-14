"""Controlled ablation experiment A-F on N-CMAPSS (RQ: do learned discrete
events + explicit duration modeling beat continuous and simpler temporal
baselines on predictive maintenance?).

Arms (differ ONLY in representation + temporal model; everything else —
splits, preprocessing, windowing, seeds, tuning budget, metrics — identical):

  A. continuous encoder            — window AE embedding, no temporal model
  B. continuous + first-order Markov — AE embedding + HMM over embeddings
  C. VQ events + Markov            — VQ tokens + first-order token Markov
  D. VQ events + HMM               — VQ tokens + first-order HMM (geometric dwell)
  E. VQ events + HSMM              — VQ tokens + explicit-duration HSMM,
                                     no absorbing-rate calibration
  F. full EGPM                     — VQ + HSMM + absorbing-rate calibration
                                     (the one tuned knob; E vs F isolates it)

RUL protocol per arm: linear readout from the arm's state/token posterior
to RUL, fit on train units only (ridge regression, closed form). This is the
identical-prediction-head protocol: no arm gets a bespoke head.

Metrics: RUL RMSE/MAE (primary), onset detection rate + lag, event
stability (seeds), anomaly/faithfulness/coherence where applicable.

Leakage controls (inherited, unchanged): unit-level splits before windowing,
train-only normalization, validation-only thresholds, test touched once.

Honesty rule: if a simpler arm matches or beats F, the report says so.
No arm is tuned beyond the shared budget; E vs F differ only in the
calibration knob (train-only).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch

from ..data import NCMAPSSLoader, SplitBuilder, UnitDataset, UnitRecord
from ..preprocessing import Preprocessor
from ..training import Trainer, TrainerConfig
from ..grammar import ForwardBackward, FirstOrderHMMFitter
from ..rul import PhaseTypeRUL
from ..evaluation import rul_metrics, per_unit_metrics, summarize_units
from ..interpretability.metrics import event_stability, temporal_validity
from ..utils import set_global_seed


ARMS = ["A_continuous", "B_continuous_hmm", "C_vq_markov",
        "D_vq_hmm", "E_vq_hsmm", "E2_hsmm_native", "F_full_egpm",
        "CNN_RUL_baseline"]


@dataclass
class ArmResult:
    arm: str
    rmse: float
    mae: float
    onset_detection: float
    onset_lag: float
    stability: float
    n_test_windows: int
    detail: Dict


def _downsample(rec: UnitRecord) -> UnitRecord:
    """1 row/cycle (rows are per-timestep; cycles are the RUL unit).
    ponytail: stride subsample; anti-alias averaging if quality suffers."""
    T = len(rec.signals)
    cycles = int(rec.rul_labels.max()) + 1 if rec.rul_labels is not None else T
    step = max(1, T // cycles)
    lbl = lambda a: a[::step] if a is not None else None
    return UnitRecord(rec.unit_id, rec.signals[::step],
                      np.arange(len(rec.signals[::step]), dtype=float),
                      health_labels=lbl(rec.health_labels),
                      rul_labels=lbl(rec.rul_labels))


def _ridge_fit(X: np.ndarray, y: np.ndarray, lam: float = 1.0) -> np.ndarray:
    """Closed-form ridge: the identical prediction head every arm shares.
    ponytail: no per-arm MLPs; a shared linear head isolates the
    representation+temporal-model effect (the actual research question)."""
    Xb = np.concatenate([X, np.ones((len(X), 1))], axis=1)
    A = Xb.T @ Xb + lam * np.eye(Xb.shape[1])
    A[-1, -1] -= lam  # don't penalize the intercept
    return np.linalg.solve(A, Xb.T @ y)


def _ridge_predict(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    Xb = np.concatenate([X, np.ones((len(X), 1))], axis=1)
    return Xb @ w


def _onsets(health: np.ndarray) -> np.ndarray:
    h = np.asarray(health)
    return np.nonzero(np.diff(h) != 0)[0] + 1


def _calibrate_absorb(hsmm, target_life: float) -> None:
    """Train-only single-knob calibration of the absorbing column so
    fresh-entry mean lifetime matches train-unit lifespan (F vs E knob)."""
    from scipy.optimize import brentq
    absb = hsmm.absorbing
    tr_ = [i for i in range(hsmm.M) if i != absb]
    mu = hsmm.D.mean()[tr_]
    A0 = hsmm.A.copy()

    def life(p: float) -> float:
        A2 = A0.copy()
        for i in tr_:
            o = A2[i, tr_].sum()
            A2[i, tr_] = A2[i, tr_] * (1 - p) / max(o, 1e-12)
            A2[i, absb] = p
        Phi = np.linalg.inv(np.eye(len(tr_)) - A2[np.ix_(tr_, tr_)])
        return float((Phi @ mu).mean())

    p = brentq(lambda p: life(p) - target_life, 1e-4, 0.9)
    for i in tr_:
        o = hsmm.A[i, tr_].sum()
        hsmm.A[i, tr_] = hsmm.A[i, tr_] * (1 - p) / max(o, 1e-12)
        hsmm.A[i, absb] = p


def run_ablation_experiment(
    h5_path: str,
    n_seeds: int = 3,
    window_length: int = 4,
    n_states: int = 5,
    d_max: int = 10,
    n_codes: int = 16,
    stage1_steps: int = 300,
    em_max_iter: int = 10,
    em_restarts: int = 3,
) -> Dict:
    """Run arms A-F under identical protocol. Returns the JSON-serializable
    report; per-seed numbers kept so nothing is hidden by an average."""
    dataset = NCMAPSSLoader().load(h5_path)
    dataset.records = [_downsample(r) for r in dataset.records]

    cfg = {
        "dataset": h5_path, "n_seeds": n_seeds, "window_length": window_length,
        "n_states": n_states, "d_max": d_max, "n_codes": n_codes,
        "stage1_steps": stage1_steps, "em_max_iter": em_max_iter,
        "em_restarts": em_restarts, "protocol": "shared ridge head, "
        "unit-level splits (seed-fixed), train-only normalization, "
        "identical windows; F = E + train-only absorbing calibration",
    }
    per_arm_seed: Dict[str, Dict[str, List[float]]] = {
        a: {} for a in ARMS
    }
    ev_by_seed: Dict[int, List[np.ndarray]] = {}
    t0 = time.time()
    for seed in range(n_seeds):
        set_global_seed(seed)
        sb = SplitBuilder(dataset, seed=0)  # fixed split seed: same units
        split = sb.build()
        stats = sb.fit_normalization(split)
        prep = Preprocessor(window_length=window_length)

        train = [prep.process(dataset[u], norm_stats=stats) for u in split.train_unit_ids]
        test = [prep.process(dataset[u], norm_stats=stats) for u in split.test_unit_ids]
        Xtr = torch.as_tensor(np.concatenate([s.X for s in train]), dtype=torch.float32)
        Mtr = torch.ones_like(Xtr)
        run_lens = [len(s.X) for s in train]
        y_tr = np.concatenate([s.rul_labels for s in train])

        # ---- shared encoder/VQ stage (A uses it as embedder, C-F tokenize) --
        tcfg = TrainerConfig(
            window_length=window_length, embed_dim=16, hidden=32,
            n_conv_blocks=2, n_codes=n_codes, d_max=d_max,
            n_states=n_states, em_max_iter=em_max_iter,
            em_restarts=em_restarts, stage1_steps=stage1_steps,
            batch_size=64, seed=seed, log_every=0,
        )
        trainer = Trainer(tcfg, n_channels=Xtr.shape[-1])
        trainer.train_stage1(Xtr, Mtr)
        ev_train = trainer.window_runs_to_event_runs(
            trainer.event_sequences(Xtr, Mtr), run_lens
        )

        def toks(seq_list):
            out = []
            for s in seq_list:
                X = torch.as_tensor(s.X, dtype=torch.float32)
                out.append(np.concatenate(trainer.event_sequences(X, torch.ones_like(X))))
            return out

        ev_test = toks(test)

        def emb(seq_list):
            out = []
            for s in seq_list:
                X = torch.as_tensor(s.X, dtype=torch.float32)
                with torch.no_grad():
                    out.append(trainer.encoder(X, torch.ones_like(X)).cpu().numpy())
            return out

        e_train, e_test = emb(train), emb(test)

        # ---- per-arm posteriors: train features -> fit ridge -> test preds --
        def fit_and_eval(feat_train, feat_test, arm, extra=None, t_arm=None):
            w = _ridge_fit(np.concatenate(feat_train), y_tr)
            preds, trues, det, lag = [], [], [], []
            for i, s in enumerate(test):
                p = _ridge_predict(w, feat_test[i])
                preds.append(p); trues.append(s.rul_labels)
                if extra is not None:
                    det_i, lag_i = extra(i)
                    det.append(det_i); lag.append(lag_i)
            m = rul_metrics(np.concatenate(preds), np.concatenate(trues))
            per_arm_seed[arm].setdefault("rmse", []).append(m["RMSE"])
            per_arm_seed[arm].setdefault("mae", []).append(m["MAE"])
            per_arm_seed[arm].setdefault("rmse_per_unit", []).append(
                per_unit_metrics(preds, trues).tolist()
            )
            if det:
                per_arm_seed[arm].setdefault("onset_detection", []).append(float(np.mean(det)))
                per_arm_seed[arm].setdefault("onset_lag", []).append(float(np.nanmean(lag)))
            if t_arm is not None:
                per_arm_seed[arm].setdefault("runtime_s", []).append(time.time() - t_arm)

        # A: continuous embedding, no temporal model
        t_arm = time.time()
        fit_and_eval(e_train, e_test, "A_continuous", t_arm=t_arm)

        # B: continuous embedding + first-order HMM posterior
        t_arm = time.time()
        # (Gaussian-emission HMM = ContinuousHSMM at d_max=1: geometric dwell)
        from ..ablations import ContinuousHSMM
        rng = np.random.default_rng(seed)
        b_model = ContinuousHSMM(
            hsmm=_placeholder_hsmm(n_states, 1, seed),  # d_max=1: geometric dwell
            means=np.zeros((n_states, tcfg.embed_dim)),
            inv_variances=np.ones((n_states, tcfg.embed_dim)),
        )
        b_model.fit_em(e_train, n_states=n_states, d_max=1,
                       max_iter=em_max_iter, min_iter=2, tol=1e-3, seed=seed)
        b_train = [b_model.posterior(z) for z in e_train]
        b_test = [b_model.posterior(z) for z in e_test]
        fit_and_eval(b_train, b_test, "B_continuous_hmm", t_arm=t_arm)

        # C: VQ tokens + first-order token Markov posterior features
        t_arm = time.time()
        from ..baselines import TokenMarkovModel
        mm = TokenMarkovModel(n_events=n_codes, smoothing=0.5).fit(ev_train)

        def run_feats_markov(v):
            P = np.exp(mm.log_trans)
            post = P[v[:-1]]  # P(next | prev) as the per-step feature
            return np.concatenate([post[:1], post])  # first step repeats

        c_train = [run_feats_markov(v) for v in ev_train]
        c_test = [run_feats_markov(v) for v in ev_test]
        fit_and_eval(c_train, c_test, "C_vq_markov", t_arm=t_arm)

        # D: VQ tokens + first-order HMM (geometric dwell via diagonal)
        t_arm = time.time()
        d_hmm, _ = FirstOrderHMMFitter(max_iter=10, seed=seed).fit(
            ev_train, n_states=n_states, n_events=n_codes
        )
        d_train = [d_hmm.posterior(v) for v in ev_train]
        d_test = [d_hmm.posterior(v) for v in ev_test]
        fit_and_eval(d_train, d_test, "D_vq_hmm", t_arm=t_arm)

        # E: VQ + HSMM (uncalibrated) — posterior + expected-dwell features
        t_arm = time.time()
        trainer.train_stage2(ev_train)
        hsmm_E = trainer.hsmm

        def fb_feats(h, runs):
            out = []
            for v in runs:
                r = ForwardBackward(h).run(v)
                out.append(np.concatenate([r.state_posterior,
                                           r.expected_dwell / h.d_max], axis=1))
            return out

        e_feat_train = fb_feats(hsmm_E, ev_train)
        e_feat_test = fb_feats(hsmm_E, ev_test)

        def onset_extra_E(h):
            def f(i):
                onsets = _onsets(test[i].health_labels)
                if len(onsets) == 0:
                    return np.nan, np.nan
                from ..grammar import SegmentalViterbi
                vit = SegmentalViterbi(h).decode(ev_test[i])
                bnd = [a for (_, a, _) in vit.segments if 0 < a < len(vit.path) - 1]
                tv = temporal_validity(bnd, onsets, tolerance=max(2, h.d_max // 2))
                return tv["detection_rate"], tv["mean_abs_lag"]
            return f

        fit_and_eval(e_feat_train, e_feat_test, "E_vq_hsmm", extra=onset_extra_E(hsmm_E), t_arm=t_arm)

        def rul_feats(h, runs):
            # native phase-type readout: posterior-weighted expected RUL
            # with mid-dwell tau (PRD §12 formula, belief-weighted)
            pt = PhaseTypeRUL(h)
            out = []
            for v in runs:
                r = ForwardBackward(h).run(v)
                prul = np.array([
                    pt.expected_rul_from_belief(
                        r.state_posterior[t],
                        tau_elapsed=r.expected_dwell[t].astype(int),
                    ) for t in range(len(v))
                ])
                out.append(prul[:, None])
            return out

        # E2: VQ + HSMM + native phase-type RUL, NO absorbing calibration
        t_arm = time.time()
        # (isolates readout: E1 vs E2; calibration: E2 vs F=E3)
        e2_train = rul_feats(hsmm_E, ev_train)
        e2_test = rul_feats(hsmm_E, ev_test)
        fit_and_eval(e2_train, e2_test, "E2_hsmm_native", extra=onset_extra_E(hsmm_E), t_arm=t_arm)

        # F: E2 + train-only absorbing calibration (isolates the knob)
        t_arm = time.time()

        # F: E2 + train-only absorbing calibration (isolates the knob)
        import copy
        hsmm_F = copy.deepcopy(hsmm_E)
        target = float(np.mean([dataset[u].rul_labels.max() for u in split.train_unit_ids]))
        _calibrate_absorb(hsmm_F, target)

        f_train = rul_feats(hsmm_F, ev_train)
        f_test = rul_feats(hsmm_F, ev_test)
        fit_and_eval(f_train, f_test, "F_full_egpm", extra=onset_extra_E(hsmm_F), t_arm=t_arm)

        # CNN-RUL real-data baseline (M9): same windows/splits, trained on
        # window MSE with val early-stop; per-window predict like every arm.
        t_arm = time.time()
        from ..baselines import CNNRUL
        cnn = CNNRUL(n_channels=Xtr.shape[-1], window_length=window_length)
        torch.manual_seed(seed)
        opt = torch.optim.Adam(cnn.parameters(), lr=1e-3)
        Xva = torch.as_tensor(
            np.concatenate([prep.process(dataset[u], norm_stats=stats).X
                             for u in split.val_unit_ids]),
            dtype=torch.float32,
        )
        yva = np.concatenate([
            prep.process(dataset[u], norm_stats=stats).rul_labels
            for u in split.val_unit_ids
        ])
        Mva = torch.ones_like(Xva)
        ytr_t = torch.as_tensor(y_tr, dtype=torch.float32)
        best, best_state, wait = float("inf"), None, 0
        for _ in range(100):
            cnn.train()
            opt.zero_grad()
            loss = ((cnn(Xtr, Mtr) - ytr_t) ** 2).mean()
            loss.backward(); opt.step()
            cnn.eval()
            with torch.no_grad():
                vv = float(((cnn(Xva, Mva) - torch.as_tensor(yva, dtype=torch.float32)) ** 2).mean())
            if vv < best - 1e-4:
                best, wait = vv, 0
                best_state = {k: t.clone() for k, t in cnn.state_dict().items()}
            else:
                wait += 1
                if wait >= 10:
                    break
        cnn.load_state_dict(best_state)
        cnn.eval()
        cnn_preds = []
        with torch.no_grad():
            for s in test:
                X = torch.as_tensor(s.X, dtype=torch.float32)
                cnn_preds.append(cnn(X, torch.ones_like(X)).numpy())
        per_arm_seed["CNN_RUL_baseline"].setdefault("rmse", []).append(
            rul_metrics(np.concatenate(cnn_preds), np.concatenate(
                [s.rul_labels for s in test]))["RMSE"])
        per_arm_seed["CNN_RUL_baseline"].setdefault("mae", []).append(
            rul_metrics(np.concatenate(cnn_preds), np.concatenate(
                [s.rul_labels for s in test]))["MAE"])
        per_arm_seed["CNN_RUL_baseline"].setdefault("rmse_per_unit", []).append(
            per_unit_metrics(cnn_preds, [s.rul_labels for s in test]).tolist())
        per_arm_seed["CNN_RUL_baseline"].setdefault("runtime_s", []).append(
            time.time() - t_arm)

        # event stability: this seed's train tokenization vs next seed's
        # (same units, different encoder init) — matched usage overlap
        ev_by_seed[seed] = ev_train

    # stability between consecutive seeds (metric consumes run lists)
    for s in range(n_seeds - 1):
        st = event_stability(ev_by_seed[s], ev_by_seed[s + 1], n_codes=n_codes)
        for arm in ("C_vq_markov", "D_vq_hmm", "E_vq_hsmm", "E2_hsmm_native", "F_full_egpm"):
            per_arm_seed[arm].setdefault("stability", []).append(
                st["matched_usage_overlap"]
            )
    per_arm_seed.setdefault("A_continuous", {})
    per_arm_seed.setdefault("B_continuous_hmm", {})
    results = {}
    for arm in ARMS:
        d = per_arm_seed[arm]
        results[arm] = {
            k: {"mean": float(np.mean(v)), "std": float(np.std(v)), "per_seed": v}
            for k, v in d.items() if k != "rmse_per_unit"
        }

    # §23 unit-level stats: windows within a unit are not independent, so the
    # independent sample is per-unit RMSE. Units are shared across seeds →
    # average each unit over seeds first, then bootstrap over units.
    per_unit_rmse = {
        arm: np.mean(np.array(per_arm_seed[arm]["rmse_per_unit"]), axis=0)
        for arm in ARMS if per_arm_seed[arm].get("rmse_per_unit")
    }
    unit_stats = summarize_units(per_unit_rmse, reference_arm="F_full_egpm")
    # E1/E2/E3 decomposition: readout effect (E1 vs E2), calibration effect
    # (E2 vs E3=F). Both effects computed per seed, paired.
    decomp = {}
    for metric in ("rmse", "mae"):
        e1 = results["E_vq_hsmm"].get(metric, {}).get("per_seed", [])
        e2 = results["E2_hsmm_native"].get(metric, {}).get("per_seed", [])
        e3 = results["F_full_egpm"].get(metric, {}).get("per_seed", [])
        if e1 and e2 and e3 and len(e1) == len(e2) == len(e3):
            decomp[f"readout_effect_{metric}"] = {  # E1 - E2 (>0: native readout worse)
                "per_seed": [a - b for a, b in zip(e1, e2)],
                "mean": float(np.mean([a - b for a, b in zip(e1, e2)])),
            }
            decomp[f"calibration_effect_{metric}"] = {  # E2 - E3 (>0: calibration hurts)
                "per_seed": [b - c for b, c in zip(e2, e3)],
                "mean": float(np.mean([b - c for b, c in zip(e2, e3)])),
            }
    report = {
        "experiment": "controlled_ablation_A_F",
        "config": cfg,
        "n_independent_test_units": len(split.test_unit_ids),
        "n_train_units": len(split.train_unit_ids),
        "n_val_units": len(split.val_unit_ids),
        "statistical_limitations": [
            f"only {len(split.test_unit_ids)} independent test units — CIs are "
            "wide and Wilcoxon underpowered by construction; per-seed numbers "
            "reflect encoder/EM variance (seeds share the same test units)",
        ],
        "results": results,
        "unit_level_stats": unit_stats,
        "E_decomposition": decomp,
        "notes": "If a simpler arm matches/beats F, that is the finding. "
                 "E1 vs E2 = readout effect; E2 vs F = absorbing-calibration "
                 "effect (F's readout is identical to E2's). "
                 "A-E use the shared ridge head over each arm's representation; "
                 "E2/F use the model-native phase-type RUL. "
                 "unit_level_stats follows PRD §23: per-unit aggregation, "
                 "bootstrap CIs, paired Wilcoxon vs F.",
    }
    return report


def _placeholder_hsmm(n_states: int, d_max: int, seed: int):
    """d_max=1 degenerate HSMM for the continuous-HMM arm (B)."""
    from ..grammar import HSMM
    from ..grammar.hsmm import DurationHistogram
    rng = np.random.default_rng(seed)
    return HSMM(
        n_states=n_states, n_events=2,
        pi=rng.dirichlet(np.ones(n_states)),
        A=rng.dirichlet(np.ones(n_states), size=n_states),
        D=DurationHistogram(pmf=np.ones((n_states, 1))),
        d_max=1, seed=seed,
    )


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "N-CMAPSS/N-CMAPSS DS01 005.h5"
    rep = run_ablation_experiment(path)
    out = f"ablation_report_{path.split()[-1].split('.')[0]}.json"
    with open(out, "w") as f:
        json.dump(rep, f, indent=2)
    print(f"-> {out}  |  test units: {rep['n_independent_test_units']}")
    print(f"{'arm':20s} {'RMSE':>8s} {'MAE':>7s} {'onset%':>7s} {'lag':>5s} {'stab':>5s} {'sec':>5s}")
    for arm, d in rep["results"].items():
        g = lambda k: d.get(k, {}).get("mean", float("nan"))
        print(f"{arm:20s} {g('rmse'):8.1f} {g('mae'):7.1f} "
              f"{100 * g('onset_detection') if g('onset_detection') == g('onset_detection') else float('nan'):7.1f} "
              f"{g('onset_lag'):5.1f} {g('stability'):5.2f} {g('runtime_s'):5.1f}")
    print("\nper-seed RMSE:")
    for arm, d in rep["results"].items():
        ps = d.get("rmse", {}).get("per_seed", [])
        print(f"  {arm:20s} {[round(x, 1) for x in ps]}")
    print("\nE-decomposition (positive = second stage worse):")
    for k, v in rep["E_decomposition"].items():
        print(f"  {k:28s} mean {v['mean']:+.1f}  per_seed {[round(x, 1) for x in v['per_seed']]}")
