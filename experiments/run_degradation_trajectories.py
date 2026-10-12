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
import sys
from pathlib import Path

# Ensure aegis-1 root is on python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
    from mpl_toolkits.axes_grid1 import make_axes_locatable

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

    # Professional publication styling
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(
        8, 1, figsize=(13, 17), sharex=True,
        gridspec_kw={'height_ratios': [2.2, 1.4, 2.0, 1.6, 1.3, 2.2, 0.9, 0.9]}
    )
    fig.patch.set_facecolor('white')

    # 1. Sensor trajectories (average across window length W, up to 6 channels)
    n_ch = min(6, seq.X.shape[2])
    sensors = seq.X.mean(axis=1)[:, :n_ch]
    palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    for c in range(n_ch):
        axes[0].plot(sensors[:, c], color=palette[c % len(palette)], linewidth=1.8, alpha=0.85, label=f"Sensor {c}")
    axes[0].set_ylabel("Sensor (norm)", fontsize=9, fontweight='bold')
    axes[0].set_title(f"{unit_id} — Multichannel Sensor Telemetry (First {n_ch} Channels)", fontsize=11, fontweight='bold', pad=8)
    axes[0].legend(loc='upper right', ncol=min(n_ch, 6), fontsize=8, framealpha=0.9)
    axes[0].grid(True, linestyle='--', alpha=0.3)

    # 2. VQ Event IDs (discrete event stream)
    axes[1].step(range(T), v, where='mid', color='#64748b', linewidth=1, alpha=0.6)
    axes[1].scatter(range(T), v, c=v, cmap='tab20', s=35, edgecolors='black', linewidth=0.6, zorder=3)
    axes[1].set_ylabel("Event ID", fontsize=9, fontweight='bold')
    axes[1].set_title(r"Vector-Quantized Operational Events ($v_t \in \mathcal{V}$)", fontsize=11, fontweight='bold', pad=8)
    axes[1].set_ylim(-0.5, trainer.vq.K - 0.5)
    axes[1].grid(True, linestyle='--', alpha=0.3)

    # 3. HSMM State Posterior Heatmap
    im = axes[2].imshow(posterior.T, aspect='auto', origin='lower', cmap='magma', vmin=0, vmax=1)
    axes[2].set_ylabel("HSMM State", fontsize=9, fontweight='bold')
    axes[2].set_title(r"HSMM Latent State Posterior Distribution $p_t(m)$", fontsize=11, fontweight='bold', pad=8)
    axes[2].set_yticks(range(posterior.shape[1]))
    axes[2].set_yticklabels([f"State {m}" for m in range(posterior.shape[1])], fontsize=8)
    divider2 = make_axes_locatable(axes[2])
    cax2 = divider2.append_axes('right', size='2%', pad=0.08)
    cb2 = plt.colorbar(im, cax=cax2)
    cb2.set_label("Belief", fontsize=8)
    cb2.ax.tick_params(labelsize=8)

    # 4. Viterbi Path vs True Health State
    axes[3].step(range(T), vit_path, 'r-', where='post', linewidth=2.0, label="Inferred State (Viterbi)")
    if true_hs is not None:
        axes[3].step(range(T), true_hs - 1, 'b--', where='post', linewidth=1.5, alpha=0.7, label="True Health State (shifted)")
    axes[3].set_ylabel("State Index", fontsize=9, fontweight='bold')
    axes[3].set_title("Inferred Regime Sequence vs Ground Truth Health", fontsize=11, fontweight='bold', pad=8)
    axes[3].set_yticks(range(trainer.hsmm.M))
    axes[3].legend(loc='upper left', fontsize=8, framealpha=0.9)
    axes[3].grid(True, linestyle='--', alpha=0.3)

    # 5. State Age (elapsed dwell in current state)
    axes[4].plot(state_age, color='#059669', linewidth=1.8)
    axes[4].fill_between(range(T), state_age, color='#059669', alpha=0.15)
    axes[4].set_ylabel("Dwell (windows)", fontsize=9, fontweight='bold')
    axes[4].set_title(r"Elapsed Dwell in Active Viterbi State (State Age $\tau_t$)", fontsize=11, fontweight='bold', pad=8)
    axes[4].grid(True, linestyle='--', alpha=0.3)

    # 6. Predicted RUL vs True RUL
    axes[5].plot(true_rul, color='#2563eb', linewidth=2.0, label="True RUL")
    axes[5].plot(pred_rul, color='#dc2626', linestyle='--', linewidth=2.0, label="Predicted RUL (Belief-Weighted)")
    axes[5].fill_between(range(T), true_rul, pred_rul, color='#ef4444', alpha=0.12, label="RUL Error")
    axes[5].set_ylabel("RUL (cycles)", fontsize=9, fontweight='bold')
    axes[5].set_title("Remaining Useful Life (RUL): True vs Predicted", fontsize=11, fontweight='bold', pad=8)
    axes[5].set_ylim(bottom=0)
    rmse = np.sqrt(np.mean((true_rul - pred_rul)**2))
    axes[5].text(
        0.02, 0.15, f"Trajectory RMSE: {rmse:.2f} cycles", transform=axes[5].transAxes,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#cbd5e1', alpha=0.9),
        fontsize=8, fontweight='bold'
    )
    axes[5].legend(loc='upper right', fontsize=8, framealpha=0.9)
    axes[5].grid(True, linestyle='--', alpha=0.3)

    # 7. Health state transitions
    if len(true_onsets) > 0:
        axes[6].vlines(true_onsets, 0, 1, colors='#2563eb', linestyles='--', linewidth=2, label="True HS Onset")
    vit_boundaries = [s for (_, s, _) in vit_segments if 0 < s < T]
    if len(vit_boundaries) > 0:
        axes[6].vlines(vit_boundaries, 0, 1, colors='#dc2626', linestyles=':', linewidth=2, label="Inferred Boundary")
    axes[6].set_ylim(0, 1)
    axes[6].set_yticks([])
    axes[6].set_ylabel("Transitions", fontsize=9, fontweight='bold')
    axes[6].set_title("Health State Regime Transitions (True vs Inferred)", fontsize=10, fontweight='bold', pad=6)
    axes[6].legend(loc='upper right', fontsize=8, framealpha=0.9)
    axes[6].grid(True, linestyle='--', alpha=0.3)

    # 8. Event transitions
    event_changes = np.nonzero(np.diff(v) != 0)[0] + 1
    if len(event_changes) > 0:
        axes[7].vlines(event_changes, 0, 1, colors='#475569', linestyles='-', linewidth=1.2)
    axes[7].set_ylim(0, 1)
    axes[7].set_yticks([])
    axes[7].set_ylabel("Event Shift", fontsize=9, fontweight='bold')
    axes[7].set_title("VQ Operational Event Change Points", fontsize=10, fontweight='bold', pad=6)
    axes[7].set_xlabel("Window Index (Operational Time)", fontsize=10, fontweight='bold')
    axes[7].grid(True, linestyle='--', alpha=0.3)

    # Harmonize horizontal widths for all axes to match colorbar on axis 2
    for ax in axes:
        if ax != axes[2]:
            div = make_axes_locatable(ax)
            dummy_cax = div.append_axes('right', size='2%', pad=0.08)
            dummy_cax.axis('off')

    plt.suptitle(f"EGPM Diagnostic Degradation Trajectory Dashboard — {unit_id}", fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def main():
    import shutil

    for ds_name, h5_path in [("DS01", "N-CMAPSS/N-CMAPSS DS01 005.h5"),
                              ("DS02", "N-CMAPSS/N-CMAPSS DS02 006.h5")]:
        print(f"\n=== Processing {ds_name} ===")
        dataset = load_dataset(h5_path)
        trainer, split, stats, prep, test, dataset = build_trainer_and_tokenize(dataset)

        print(f"Test units: {split.test_unit_ids}")
        for unit_id in split.test_unit_ids:
            clean_name = unit_id if unit_id.startswith(ds_name) else f"{ds_name}_{unit_id}"
            print(f"  Processing test unit {clean_name}...")
            unit_data = process_unit(trainer, prep, stats, unit_id, dataset)
            save_path = OUT_DIR / f"{clean_name}_trajectory.png"
            plot_trajectory(unit_id, unit_data, trainer, dataset, save_path)
            print(f"    Saved: {save_path}")

            # Also maintain backward-compatible duplicate name if needed
            legacy_path = OUT_DIR / f"{ds_name}_{clean_name}_trajectory.png"
            if legacy_path != save_path:
                shutil.copyfile(save_path, legacy_path)

        # Also process train units for successful examples
        for unit_id in split.train_unit_ids[:2]:
            clean_name = unit_id if unit_id.startswith(ds_name) else f"{ds_name}_{unit_id}"
            print(f"  Processing train unit {clean_name}...")
            unit_data = process_unit(trainer, prep, stats, unit_id, dataset)
            save_path = OUT_DIR / f"{clean_name}_trajectory_train.png"
            plot_trajectory(unit_id, unit_data, trainer, dataset, save_path)
            print(f"    Saved: {save_path}")

            legacy_path = OUT_DIR / f"{ds_name}_{clean_name}_trajectory_train.png"
            if legacy_path != save_path:
                shutil.copyfile(save_path, legacy_path)

    print(f"\nAll trajectory plots saved to {OUT_DIR}")


if __name__ == "__main__":
    main()