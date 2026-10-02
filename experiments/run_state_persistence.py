"""Phase 3: State Persistence Metrics — Duration prediction and state occupancy."""
from __future__ import annotations

import numpy as np

from egpm.data import NCMAPSSLoader, SplitBuilder
from egpm.preprocessing import Preprocessor
from egpm.training import Trainer, TrainerConfig
from egpm.grammar import SegmentalViterbi
from egpm.experiments.ablation_arms import _downsample
from egpm.utils import set_global_seed


def load_dataset(h5_path: str):
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
    return trainer, split, stats, prep


def process_unit(trainer, prep, stats, unit_id, dataset):
    import torch
    seq = prep.process(dataset[unit_id], norm_stats=stats)
    X = torch.as_tensor(seq.X, dtype=torch.float32)
    M = torch.ones_like(X)
    v = np.concatenate(trainer.event_sequences(X, M))
    from egpm.grammar import SegmentalViterbi
    vit = SegmentalViterbi(trainer.hsmm).decode(v)
    return {"seq": seq, "vit_path": vit.path, "vit_segments": vit.segments}


def compute_persistence(seq, vit_segments):
    true_hs = seq.health_labels

    # Inferred state durations
    inf_durations = {}
    for state, start, end in vit_segments:
        dur = end - start + 1
        inf_durations.setdefault(state, []).append(dur)

    # True health state durations
    true_durations = {}
    if true_hs is not None:
        h = true_hs
        changes = np.nonzero(np.diff(h) != 0)[0] + 1
        boundaries = np.concatenate([[0], changes, [len(h)]])
        for i in range(len(boundaries) - 1):
            s = int(h[boundaries[i]])
            dur = boundaries[i + 1] - boundaries[i]
            true_durations.setdefault(s, []).append(dur)

    inf = {k: {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": len(v)}
           for k, v in inf_durations.items()}
    true = {k: {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": len(v)}
            for k, v in true_durations.items()}
    return {"inferred": inf, "true": true}


def main():
    for ds_name, h5_path in [("DS01", "N-CMAPSS/N-CMAPSS DS01 005.h5"),
                              ("DS02", "N-CMAPSS/N-CMAPSS DS02 006.h5")]:
        print(f"\n=== {ds_name} State Persistence ===")
        dataset = load_dataset(h5_path)
        trainer, split, stats, prep = build_trainer_and_tokenize(dataset)

        for unit_id in split.test_unit_ids + split.train_unit_ids[:2]:
            unit_data = process_unit(trainer, None, None, unit_id, dataset)
            # Fix: pass correct arguments
            import torch
            seq = prep.process(dataset[unit_id], norm_stats=stats)
            X = torch.as_tensor(seq.X, dtype=torch.float32)
            M = torch.ones_like(X)
            v = np.concatenate(trainer.event_sequences(X, M))
            from egpm.grammar import SegmentalViterbi
            vit = SegmentalViterbi(trainer.hsmm).decode(v)
            pers = compute_persistence(seq, vit.segments)
            print(f"  {unit_id}: inf {pers['inferred']} | true {pers['true']}")


if __name__ == "__main__":
    from egpm.data import NCMAPSSLoader
    from egpm.preprocessing import Preprocessor
    from egpm.training import Trainer, TrainerConfig
    from egpm.utils import set_global_seed
    import torch

    for ds_name, h5_path in [("DS01", "N-CMAPSS/N-CMAPSS DS01 005.h5"),
                              ("DS02", "N-CMAPSS/N-CMAPSS DS02 006.h5")]:
        print(f"\n=== {ds_name} State Persistence ===")
        dataset = load_dataset(h5_path)
        trainer, split, stats, prep = build_trainer_and_tokenize(dataset)

        for unit_id in split.test_unit_ids + split.train_unit_ids[:2]:
            seq = prep.process(dataset[unit_id], norm_stats=stats)
            X = torch.as_tensor(seq.X, dtype=torch.float32)
            M = torch.ones_like(X)
            v = np.concatenate(trainer.event_sequences(X, M))
            from egpm.grammar import SegmentalViterbi
            vit = SegmentalViterbi(trainer.hsmm).decode(v)
            pers = compute_persistence(seq, vit.segments)
            print(f"  {unit_id}:")
            print(f"    inferred: {pers['inferred']}")
            print(f"    true:     {pers['true']}")