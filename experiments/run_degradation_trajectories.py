"""Phase 3: Degradation Progression — Trajectory visualization for test units.

Generates per-unit plots showing:
1. Sensor trajectories (ops + sensors)
2. VQ event IDs over time
3. HSMM state posterior heatmap
4. Viterbi path (inferred state sequence)
5. State age / elapsed dwell
6. Predicted RUL vs true RUL
7. Health state transitions
8. Event transitions

Saves PNGs to experiments/figures/degradation_trajectories/
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from egpm.data import NCMAPSSLoader, SplitBuilder
from egpm.preprocessing import Preprocessor
from egpm.training import Trainer, TrainerConfig
from egpm.grammar import ForwardBackward, SegmentalViterbi
from egpm.rul import PhaseTypeRUL
from egpm.experiments.ablation_arms import _downsample
from egpm.utils import set_global_seed

OUT_DIR = Path("experiments/figures/degradation_trajectories")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset(h5_path: str):
    dataset = NCMAPSSLoader().load(h5_path)
    dataset.records = [_downsample(r) for r in dataset.records]
    return dataset


def build_trainer_and_tokenize(dataset, window_length=4, n_states=5, d_max=10, n_codes=16, stage1_steps=300, seed=0):
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
    return trainer, split, stats, prep, test, dataset


def process_unit(trainer, prep, stats, unit_id, dataset):
    seq = prep.process(dataset[unit_id], norm_stats=stats)
    X = torch.as_tensor(seq.X, dtype=torch.float32)
    M = torch.ones_like(X)
    v = np.concatenate(trainer.event_sequences(X, M))
    fb = ForwardBackward(trainer.hsmm)
    fb_res = fb.run(v)
    vit = SegmentalViterbi(trainer.hsmm).decode(v)
    rul_est = PhaseTypeRUL(trainer.hsmm)
    return {
        "seq": seq,
        "v": v,
        "posterior": fb_res.state_posterior,      # [T, M]
        "expected_dwell": fb_res.expected_dwell,  # [T, M]
        "vit_path": vit.path,                     # [T]
        "vit_segments": vit.segments,             # [(state, start, end)]
        "log_likelihood": fb_res.log_likelihood,
    }


def plot_trajectory(unit_id, unit_data, trainer, dataset, save_path):
    seq = unit_data["seq"]
    v = unit_data["v"]
    posterior = unit_data["posterior"]
    expected_dwell = unit_data["expected_dwell"]
    vit_path = unit_data["vit_path"]
    vit_segments = unit_data["vit_segments"]
    T = len(v)

    # True labels
    true_rul = seq.rul_labels
    true_hs = seq.health_labels
    true_onsets = np.nonzero(np.diff(true_hs) != 0)[0] + 1 if true_hs is not None else []

    # Predicted RUL from posterior
    rul_est = PhaseTypeRUL(trainer.hsmm)
    pred_rul = np.array([
        rul_est.expected_rul_from_belief(posterior[t], expected_dwell[t])
        for t in range(T)
    ])

    # State age (elapsed dwell) per window
    state_age = np.zeros(T)
    for s, (state, start, end) in enumerate(vit_segments):
        for t in range(start, min(end + 1, T)):
            state_age[t] = t - start

    fig, axes = plt.subplots(8, 1, figsize=(14, 20), sharex=True)

    # 1. Sensor trajectories (all channels, up to 6)
    n_ch = min(6, seq.X.shape[1])
    sensors = seq.X[:, :n_ch]
    for c in range(n_ch):
        axes[0].plot(sensors[:, c], alpha=0.7, label=f"ch{c}")
    axes[0].set_ylabel("Sensor (norm)")
    axes[0].set_title(f"{unit_id} — First {n_ch} Sensor Channels")
    if n_ch > 0:
        axes[0].legend(ncol=3, fontsize=7)

    # 2. VQ Event IDs
    axes[1].plot(v, 'k.', markersize=2)
    axes[1].set_ylabel("Event ID")
    axes[1].set_title("VQ Event Sequence (one token per window)")
    axes[1].set_ylim(-0.5, trainer.vq.K - 0.5)

    # 3. HSMM State Posterior Heatmap
    im = axes[2].imshow(posterior.T, aspect='auto', origin='lower', cmap='viridis')
    axes[2].set_ylabel("State")
    axes[2].set_title("HSMM State Posterior p_t(m)")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    # 4. Viterbi Path (inferred state sequence)
    axes[3].plot(vit_path, 'r-', linewidth=1.5, label="Viterbi")
    if true_hs is not None:
        # Map health states (1-6) to state indices if possible
        axes[3].plot(true_hs - 1, 'b--', linewidth=1, alpha=0.5, label="True HS (shifted)")
    axes[3].set_ylabel("State")
    axes[3].set_title("Viterbi Path vs True Health State")
    axes[3].legend()

    # 5. State Age (elapsed dwell in current state)
    axes[4].plot(state_age, 'g-')
    axes[4].set_ylabel("Age (windows)")
    axes[4].set_title("Elapsed Dwell in Current Viterbi State")

    # 6. Predicted RUL vs True RUL
    axes[5].plot(true_rul, 'b-', label="True RUL", linewidth=1.5)
    axes[5].plot(pred_rul, 'r--', label="Predicted RUL (belief-weighted)", linewidth=1.5)
    axes[5].set_ylabel("RUL (cycles)")
    axes[5].set_title("RUL: True vs Predicted (PhaseTypeRUL)")
    axes[5].legend()
    axes[5].set_ylim(bottom=0)

    # 7. Health state transitions
    if len(true_onsets) > 0:
        axes[6].vlines(true_onsets, 0, 1, colors='b', linestyles='--', linewidth=2, label="True HS Onset")
    vit_boundaries = [s for (_, s, _) in vit_segments if 0 < s < T]
    if len(vit_boundaries) > 0:
        axes[6].vlines(vit_boundaries, 0, 1, colors='r', linestyles=':', linewidth=2, label="Viterbi Boundary")
    axes[6].set_ylim(0, 1)
    axes[6].set_ylabel("Transition")
    axes[6].set_title("Health State Transitions (True vs Inferred)")
    axes[6].legend()

    # 8. Event transitions
    event_changes = np.nonzero(np.diff(v) != 0)[0] + 1
    if len(event_changes) > 0:
        axes[7].vlines(event_changes, 0, 1, colors='k', linestyles='-', linewidth=0.5)
    axes[7].set_ylim(0, 1)
    axes[7].set_ylabel("Event Change")
    axes[7].set_title("VQ Event Transitions")
    axes[7].set_xlabel("Window Index")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    import torch
    for ds_name, h5_path in [("DS01", "N-CMAPSS/N-CMAPSS DS01 005.h5"),
                              ("DS02", "N-CMAPSS/N-CMAPSS DS02 006.h5")]:
        print(f"\n=== Processing {ds_name} ===")
        dataset = load_dataset(h5_path)
        trainer, split, stats, prep, test, dataset = build_trainer_and_tokenize(dataset)

        print(f"Test units: {split.test_unit_ids}")
        for unit_id in split.test_unit_ids:
            print(f"  Processing {unit_id}...")
            unit_data = process_unit(trainer, prep, stats, unit_id, dataset)
            save_path = OUT_DIR / f"{ds_name}_{unit_id}_trajectory.png"
            plot_trajectory(unit_id, unit_data, trainer, dataset, save_path)
            print(f"    Saved: {save_path}")

        # Also process a few train units for "successful" examples
        for unit_id in split.train_unit_ids[:2]:
            print(f"  Processing train unit {unit_id}...")
            unit_data = process_unit(trainer, prep, stats, unit_id, dataset)
            save_path = OUT_DIR / f"{ds_name}_{unit_id}_trajectory_train.png"
            plot_trajectory(unit_id, unit_data, trainer, dataset, save_path)
            print(f"    Saved: {save_path}")

    print(f"\nAll trajectory plots saved to {OUT_DIR}")


if __name__ == "__main__":
    main()