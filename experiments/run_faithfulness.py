"""Phase 8: Interpretability/Faithfulness — Event/state masking, counterfactuals."""
from __future__ import annotations

import numpy as np

from egpm.data import NCMAPSSLoader, SplitBuilder
from egpm.preprocessing import Preprocessor
from egpm.training import Trainer, TrainerConfig
from egpm.grammar import ForwardBackward, SegmentalViterbi
from egpm.anomaly import AnomalyScorer
from egpm.interpretability.metrics import explanation_faithfulness
from egpm.experiments.ablation_arms import _downsample
from egpm.utils import set_global_seed


def load_dataset(h5_path: str):
    from egpm.data import NCMAPSSLoader
    from egpm.experiments.ablation_arms import _downsample
    dataset = NCMAPSSLoader().load(h5_path)
    dataset.records = [_downsample(r) for r in dataset.records]
    return dataset


def build_trainer(dataset, window_length=4, n_states=5, d_max=10, n_codes=16, stage1_steps=300, seed=0):
    import torch
    set_global_seed(seed)
    sb = SplitBuilder(dataset, seed=0)
    split = sb.build()
    stats = sb.fit_normalization(split)
    prep = Preprocessor(window_length=window_length)

    train = [prep.process(dataset[u], norm_stats=stats) for u in split.train_unit_ids]

    import torch
    Xtr = torch.as_tensor(np.concatenate([s.X for s in [prep.process(dataset[u], norm_stats=stats) for u in split.train_unit_ids]]), dtype=torch.float32)
    Mtr = torch.ones_like(Xtr)
    run_lens = [len(s.X) for s in [prep.process(dataset[u], norm_stats=stats) for u in split.train_unit_ids]]

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


def run_faithfulness(trainer, seq, top_k=3):
    """Compute explanation faithfulness by masking top-k scoring windows."""
    import torch
    X = torch.as_tensor(seq.X, dtype=torch.float32)
    M = torch.ones_like(X)
    v = np.concatenate(trainer.event_sequences(X, M))

    scorer = AnomalyScorer()
    comp = scorer.score_sequence(trainer.hsmm, v)
    scores = comp.score

    # Top-k scoring windows
    top_windows = np.argsort(scores)[::-1][:top_k]

    # Faithfulness: mask top windows and measure score reduction
    from egpm.interpretability.metrics import explanation_faithfulness
    return explanation_faithfulness(v, top_windows.tolist(), trainer.hsmm, scorer)


def run_stability_across_seeds(dataset, n_seeds=3):
    """Event stability: Hungarian-matched usage overlap across seeds."""
    from egpm.interpretability.metrics import event_stability

    overlaps = []
    for seed_a in range(n_seeds):
        trainer_a, _, _, _ = build_trainer(dataset, seed=seed_a)
        # This is simplified - real stability needs same units tokenized with both seeds
        pass
    return overlaps


def main():
    from egpm.data import NCMAPSSLoader
    from egpm.preprocessing import Preprocessor
    from egpm.training import Trainer, TrainerConfig
    from egpm.data import SplitBuilder
    from egpm.experiments.ablation_arms import _downsample
    from egpm.utils import set_global_seed
    import torch

    for ds_name, h5_path in [("DS01", "N-CMAPSS/N-CMAPSS DS01 005.h5"),
                              ("DS02", "N-CMAPSS/N-CMAPSS DS02 006.h5")]:
        print(f"\n=== {ds_name} Faithfulness ===")
        dataset = load_dataset(h5_path)

        # Use seed=0 trainer for faithfulness
        trainer, split, stats, prep = build_trainer(dataset, seed=0)

        # Test on test units
        for unit_id in split.test_unit_ids:
            seq = prep.process(dataset[unit_id], norm_stats=stats)
            X = torch.as_tensor(seq.X, dtype=torch.float32)
            M = torch.ones_like(X)
            v = np.concatenate(trainer.event_sequences(X, M))

            scorer = AnomalyScorer()
            comp = scorer.score_sequence(trainer.hsmm, v)
            top_windows = np.argsort(comp.score)[::-1][:3]

            faith = explanation_faithfulness(v, top_windows.tolist(), trainer.hsmm, scorer)
            print(f"  DS01_u2: score_reduction={faith['score_reduction']:.4f} (before={faith['before']:.4f}, after={faith['after']:.4f})")

    print("\n[Note] Stability and counterfactual experiments need full seed runs")


def load_dataset(h5_path: str):
    from egpm.data import NCMAPSSLoader
    from egpm.experiments.ablation_arms import _downsample
    dataset = NCMAPSSLoader().load(h5_path)
    dataset.records = [_downsample(r) for r in dataset.records]
    return dataset


def build_trainer(dataset, window_length=4, n_states=5, d_max=10, n_codes=16, stage1_steps=300, seed=0):
    import torch
    set_global_seed(seed)
    sb = SplitBuilder(dataset, seed=0)
    split = sb.build()
    stats = sb.fit_normalization(split)
    prep = Preprocessor(window_length=window_length)

    train = [prep.process(dataset[u], norm_stats=stats) for u in split.train_unit_ids]

    Xtr = torch.as_tensor(np.concatenate([s.X for s in [prep.process(dataset[u], norm_stats=stats) for u in split.train_unit_ids]]), dtype=torch.float32)
    Mtr = torch.ones_like(Xtr)
    run_lens = [len(s.X) for s in [prep.process(dataset[u], norm_stats=stats) for u in split.train_unit_ids]]

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


if __name__ == "__main__":
    from egpm.data import NCMAPSSLoader, SplitBuilder
    from egpm.preprocessing import Preprocessor
    from egpm.training import Trainer, TrainerConfig
    from egpm.experiments.ablation_arms import _downsample
    from egpm.utils import set_global_seed
    import torch

    main()