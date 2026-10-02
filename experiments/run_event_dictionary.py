"""Phase 4: Event Vocabulary Analysis — Behavioral semantics of learned VQ codes."""
from __future__ import annotations

import json
import numpy as np

from egpm.data import NCMAPSSLoader, SplitBuilder
from egpm.preprocessing import Preprocessor
from egpm.training import Trainer, TrainerConfig
from egpm.experiments.ablation_arms import _downsample
from egpm.utils import set_global_seed


def load_dataset(h5_path: str):
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


def collect_training_events(trainer, dataset, split, stats, prep):
    """Collect all VQ events + labels from training units."""
    import torch
    all_events = []
    all_rul = []
    all_hs = []
    all_sensors_list = []

    for u in split.train_unit_ids:
        seq = prep.process(dataset[u], norm_stats=stats)
        X = torch.as_tensor(seq.X, dtype=torch.float32)
        M = torch.ones_like(X)
        v = np.concatenate(trainer.event_sequences(X, M))
        all_events.extend(v.tolist())
        all_rul.extend(seq.rul_labels.tolist())
        all_hs.extend(seq.health_labels.tolist())
        all_sensors_list.append(seq.X)

    all_events = np.array(all_events)
    all_rul = np.array(all_rul)
    all_hs = np.array(all_hs)
    all_sensors = np.concatenate(all_sensors_list, axis=0)
    return all_events, all_rul, all_hs, all_sensors


def build_event_dictionary(trainer, all_events, all_rul, all_hs, all_sensors):
    """Analyze each VQ code."""
    K = trainer.vq.K
    dictionary = {}

    for k in range(K):
        mask = all_events == k
        if not mask.any():
            dictionary[k] = {"note": "unused code"}
            continue

        code_rul = all_rul[mask]
        code_hs = all_hs[mask]
        code_sensors = all_sensors[mask]

        freq = float(mask.mean())
        rul_mean = float(code_rul.mean())
        rul_std = float(code_rul.std())

        hs_dist = {int(h): float((code_hs == h).mean()) for h in np.unique(code_hs)}

        sensor_mean = code_sensors.mean(axis=(0, 1))
        sensor_std = code_sensors.std(axis=(0, 1))

        b_k = trainer.hsmm.B[:, k]
        state_dist = {int(s): float(p) for s, p in enumerate(b_k)}

        dictionary[k] = {
            "frequency": freq,
            "mean_rul": float(code_rul.mean()),
            "std_rul": rul_std,
            "health_state_dist": hs_dist,
            "state_emission_probs": state_dist,
            "sensor_mean": sensor_mean.tolist(),
            "sensor_std": sensor_std.tolist(),
        }

    return dictionary


def main():
    for ds_name, h5_path in [("DS01", "N-CMAPSS/N-CMAPSS DS01 005.h5"),
                              ("DS02", "N-CMAPSS/N-CMAPSS DS02 006.h5")]:
        print(f"\n=== {ds_name} Event Dictionary ===")
        dataset = load_dataset(h5_path)
        trainer, split, stats, prep = build_trainer(dataset)

        all_events, all_rul, all_hs, all_sensors = collect_training_events(
            trainer, dataset, split, stats, prep
        )

        dictionary = build_event_dictionary(trainer, all_events, all_rul, all_hs, all_sensors)

        out_path = f"experiments/figures/event_dictionary_{ds_name}.json"
        with open(out_path, "w") as f:
            json.dump(dictionary, f, indent=2)
        print(f"Saved {out_path}")

        for k, v in dictionary.items():
            if "note" not in v:
                print(f"  Code {k}: freq={v['frequency']:.3f}, RUL={v['mean_rul']:.1f}±{v['std_rul']:.1f}, "
                      f"HS={v['health_state_dist']}, state_emit={v['state_emission_probs']}")


if __name__ == "__main__":
    from egpm.data import NCMAPSSLoader, SplitBuilder
    from egpm.preprocessing import Preprocessor
    from egpm.training import Trainer, TrainerConfig
    from egpm.experiments.ablation_arms import _downsample
    from egpm.utils import set_global_seed
    import torch
    import json

    main()