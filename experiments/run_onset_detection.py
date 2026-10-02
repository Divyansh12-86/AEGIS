"""Phase 3: Onset Detection Metrics — Viterbi boundary alignment with health state transitions.

Computes per-unit and aggregate:
- Detection rate (within tolerance)
- Mean absolute lag
- Mean lag (signed)
- Number of true onsets
"""
from __future__ import annotations

import numpy as np
from pathlib import Path

from egpm.data import NCMAPSSLoader, SplitBuilder
from egpm.preprocessing import Preprocessor
from egpm.training import Trainer, TrainerConfig
from egpm.grammar import ForwardBackward, SegmentalViterbi
from egpm.experiments.ablation_arms import _downsample
from egpm.utils import set_global_seed


def _onsets_from_health(health_labels: np.ndarray) -> np.ndarray:
    """Ground-truth change indices from ordinal health labels."""
    h = np.asarray(health_labels)
    return np.nonzero(np.diff(h) != 0)[0] + 1


def temporal_validity(
    inferred_boundaries: np.ndarray,
    true_onsets: np.ndarray,
    tolerance: int = 2,
) -> dict:
    """Alignment of inferred state-segment boundaries with known fault onsets."""
    ib = np.asarray(sorted(set(int(x) for x in inferred_boundaries)))
    to = np.asarray(sorted(set(int(x) for x in true_onsets)))
    if len(to) == 0:
        raise ValueError("no true onsets provided")
    matched, lags = [], []
    for t in to:
        if len(ib) == 0:
            break
        j = int(np.argmin(np.abs(ib - t)))
        lag = int(ib[j] - t)
        if abs(lag) <= tolerance:
            matched.append(t)
            lags.append(lag)
    return {
        "detection_rate": len(matched) / len(to),
        "mean_abs_lag": float(np.mean(np.abs(lags))) if lags else float("nan"),
        "mean_lag": float(np.mean(lags)) if lags else float("nan"),
        "n_true_onsets": len(to),
        "n_inferred_boundaries": len(ib),
    }


def load_dataset(h5_path: str):
    from egpm.data import NCMAPSSLoader
    dataset = NCMAPSSLoader().load(h5_path)
    dataset.records = [_downsample(r) for r in dataset.records]
    return dataset


def build_trainer_and_tokenize(dataset, window_length=4, n_states=5, d_max=10, n_codes=16, stage1_steps=300, seed=0):
    import torch
    set_global_seed(seed)
    sb = SplitBuilder(dataset, seed=0)
    split = sb.build()
    stats = sb.fit_normalization(split)
    prep = Preprocessor(window_length=window_length)

    train = [prep.process(dataset[u], norm_stats=stats) for u in split.train_unit_ids]
    test = [prep.process(dataset[u], norm_stats=stats) for u in split.test_unit_ids]

    Xtr = torch.as_tensor(np.concatenate([s.X for s in train]), dtype=torch.float32)
    Mtr = torch.ones_like(Xtr)
    run_lens = [len(s.X) for s in train]

    from egpm.training import Trainer, TrainerConfig
    tcfg = TrainerConfig(
        window_length=window_length, embed_dim=16, hidden=32,
        n_conv_blocks=3, n_codes=n_codes, d_max=d_max,
        n_states=n_states, em_max_iter=10, stage1_steps=stage1_steps,
        batch_size=64, seed=seed, log_every=0,
    )
    trainer = Trainer(tcfg, n_channels=Xtr.shape[-1])
    trainer.train_stage1(Xtr, Mtr)
    ev_tr = trainer.window_runs_to_event_runs(
        trainer.event_sequences(Xtr, Mtr), run_lens
    )
    trainer.train_stage2(ev_tr)
    return trainer, split, stats, prep, test


def process_unit(trainer, prep, stats, unit_id, dataset):
    import torch
    seq = prep.process(dataset[unit_id], norm_stats=stats)
    X = torch.as_tensor(seq.X, dtype=torch.float32)
    M = torch.ones_like(X)
    v = np.concatenate(trainer.event_sequences(X, M))
    from egpm.grammar import ForwardBackward, SegmentalViterbi
    fb = ForwardBackward(trainer.hsmm)
    fb_res = fb.run(v)
    vit = SegmentalViterbi(trainer.hsmm).decode(v)
    return {
        "seq": seq,
        "v": v,
        "vit_path": vit.path,
        "vit_segments": vit.segments,
    }


def main():
    results = {}
    for ds_name, h5_path in [("DS01", "N-CMAPSS/N-CMAPSS DS01 005.h5"),
                              ("DS02", "N-CMAPSS/N-CMAPSS DS02 006.h5")]:
        print(f"\n=== {ds_name} Onset Detection ===")
        dataset = load_dataset(h5_path)
        trainer, split, stats, prep, test = build_trainer_and_tokenize(dataset)

        print(f"Test units: {split.test_unit_ids}")
        ds_results = {}
        for unit_id in split.test_unit_ids + split.train_unit_ids[:2]:
            unit_data = process_unit(trainer, prep, stats, unit_id, dataset)
            seq = unit_data["seq"]
            vit_segments = unit_data["vit_segments"]
            vit_path = unit_data["vit_path"]

            # Ground truth onsets from health labels
            hl = unit_data["seq"].health_labels
            if hl is None:
                continue
            onsets = _onsets_from_health(hl)
            if len(onsets) == 0:
                print(f"  {unit_id}: no health transitions")
                continue

            # Viterbi boundaries (state changes)
            boundaries = [s for (_, s, _) in unit_data["vit_segments"]
                         if 0 < s < len(unit_data["vit_path"]) - 1]

            tv = temporal_validity(np.array(boundaries), onsets, tolerance=max(2, 10 // 2))
            tv["unit_id"] = unit_data["seq"].unit_id if hasattr(unit_data["seq"], "unit_id") else "unknown"
            ds_results[unit_id] = tv
            print(f"  {unit_id}: DR={tv['detection_rate']:.2f}, lag={tv['mean_lag']:.1f}±{tv['mean_abs_lag']:.1f}, n_onsets={tv['n_true_onsets']}")

        results[ds_name] = ds_results

    return results


if __name__ == "__main__":
    results = main()
    print("\n=== Summary ===")
    for ds, ds_results in results.items():
        drs = [r["detection_rate"] for r in ds_results.values() if not np.isnan(r["detection_rate"])]
        lags = [r["mean_abs_lag"] for r in ds_results.values() if not np.isnan(r["mean_abs_lag"])]
        print(f"{ds}: mean DR={np.mean(drs):.3f}, mean |lag|={np.mean(lags):.2f}")