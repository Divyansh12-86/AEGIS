"""Phase 5: Information-Loss Study — Continuous vs Quantized representations.

Tests whether VQ discards anomaly-relevant information.

Probes:
1. Continuous embedding → linear probe (RUL, fault, health)
2. Quantized (VQ) → linear probe (same tasks)
3. Continuous HSMM vs VQ HSMM (ablation A1 vs primary)
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score, roc_auc_score

from egpm.data import NCMAPSSLoader, MIMIILoader, SplitBuilder
from egpm.preprocessing import Preprocessor
from egpm.training import Trainer, TrainerConfig
from egpm.experiments.ablation_arms import _downsample
from egpm.utils import set_global_seed


def load_ncmapss(h5_path: str):
    from egpm.data import NCMAPSSLoader
    dataset = NCMAPSSLoader().load(h5_path)
    dataset.records = [_downsample(r) for r in dataset.records]
    return dataset


def load_mimii(root: str, machine_ids=("id_00", "id_02", "id_04", "id_06")):
    from egpm.data import MIMIILoader
    from egpm.data import UnitDataset
    combined = UnitDataset()
    for mid in machine_ids:
        ds = MIMIILoader(machines=("fan",), feature="logmel", append_deltas=True,
                         max_clips_per_label=200).load(f"fan/{mid}")
        for r in ds.records:
            combined.add(r)
    return combined


def collect_ncmapss_features(trainer, dataset, split, stats, prep):
    """Extract continuous embeddings and VQ tokens for all units."""
    continuous_embs = []
    vq_tokens = []
    rul_labels = []
    hs_labels = []
    fc_labels = []

    for u in split.train_unit_ids + split.test_unit_ids:
        seq = prep.process(dataset[u], norm_stats=stats)
        X = torch.as_tensor(seq.X, dtype=torch.float32)
        M = torch.ones_like(X)

        # Continuous embeddings
        with torch.no_grad():
            z = trainer.encoder(X, M)
        continuous_embs.append(z.cpu().numpy())

        # VQ tokens
        v = np.concatenate(trainer.event_sequences(X, torch.ones_like(X)))
        vq_tokens.append(v)

        rul_labels.append(seq.rul_labels)
        hs_labels.append(seq.health_labels)
        if hasattr(dataset[u], 'fault_label') and dataset[u].fault_label is not None:
            fc_labels.append(dataset[u].fault_label)
        else:
            fc_labels.append(-1)

    return {
        "continuous": np.concatenate(continuous_embs),
        "vq_tokens": np.concatenate(vq_tokens),
        "rul": np.concatenate(rul_labels),
        "hs": np.concatenate(hs_labels),
        "fc": np.array(fc_labels),
    }


def collect_mimii_features(trainer, dataset, stats, prep, max_units=500):
    """Extract features from MIMII normal clips only."""
    normal_ids = [u for u in dataset.unit_ids if "normal" in u]
    rng = np.random.default_rng(0)
    rng.shuffle(normal_ids)
    normal_ids = normal_ids[:max_units]

    continuous_embs = []
    vq_tokens = []
    anomaly_labels = []

    for u in normal_ids:
        seq = prep.process(dataset[u], norm_stats=stats)
        X = torch.as_tensor(seq.X, dtype=torch.float32)
        M = torch.ones_like(X)

        with torch.no_grad():
            z = trainer.encoder(X, M)
        continuous_embs.append(z.cpu().numpy())

        v = np.concatenate(trainer.event_sequences(X, torch.ones_like(X)))
        # For MIMII, all normal clips have label 0
        anomaly_labels.extend([0] * len(z.cpu().numpy()))
        vq_tokens.extend(np.concatenate(trainer.event_sequences(X, torch.ones_like(X))).tolist())

    return {
        "continuous": np.concatenate(continuous_embs),
        "vq_tokens": np.array(vq_tokens),
        "anomaly": np.array(anomaly_labels),
    }


def probe_rul(features, targets, method="ridge"):
    """Linear probe for RUL prediction."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import mean_squared_error

    X = features
    y = targets

    if method == "ridge":
        model = Ridge(alpha=1.0)
        model.fit(X, y)
        pred = model.predict(X)
        return np.sqrt(mean_squared_error(y, pred))
    return None


def probe_classification(features, targets, method="logistic"):
    """Linear probe for classification."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score

    X = features
    y = targets

    if method == "logistic":
        model = LogisticRegression(max_iter=1000, random_state=0)
        model.fit(X, y)
        pred = model.predict(X)
        proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else pred
        return accuracy_score(y, pred), roc_auc_score(y, proba) if len(np.unique(y)) == 2 else 0.5
    return None, None


def main():
    print("=== N-CMAPSS Information-Loss Study ===")
    from egpm.data import NCMAPSSLoader
    from egpm.preprocessing import Preprocessor
    from egpm.training import Trainer, TrainerConfig
    from egpm.data import SplitBuilder
    from egpm.experiments.ablation_arms import _downsample
    from egpm.utils import set_global_seed
    import torch

    for ds_name, h5_path in [("DS01", "N-CMAPSS/N-CMAPSS DS01 005.h5"),
                              ("DS02", "N-CMAPSS/N-CMAPSS DS02 006.h5")]:
        print(f"\n--- {ds_name} ---")
        dataset = load_dataset(h5_path)
        trainer, split, stats, prep = build_trainer(dataset)

        feats = collect_ncmapss_features(trainer, dataset, split, stats, prep)

        # RUL probe
        rul_cont = probe_rul(feats["continuous"], feats["rul"])
        rul_vq = probe_rul(feats["vq_tokens"].reshape(-1, 1), feats["rul"])
        print(f"  RUL RMSE: continuous={rul_cont:.2f}, VQ={rul_vq:.2f}")

        # Health state probe (2-class)
        hs_cont_acc, hs_cont_auc = probe_classification(feats["continuous"], feats["hs"])
        # VQ tokens need 1D reshape for sklearn
        hs_vq_acc, hs_vq_auc = probe_classification(
            feats["vq_tokens"].reshape(-1, 1), feats["hs"]
        )
        print(f"  Health state: continuous_acc={hs_cont_acc:.3f}, VQ_acc={hs_vq_acc:.3f}")

        # Fault (Fc) probe
        fc_mask = feats["fc"] != -1
        if fc_mask.sum() > 10:
            fc_cont_acc, fc_cont_auc = probe_classification(
                feats["continuous"][fc_mask], feats["fc"][fc_mask]
            )
            fc_vq_acc, fc_vq_auc = probe_classification(
                feats["vq_tokens"][fc_mask].reshape(-1, 1), feats["fc"][fc_mask]
            )
            print(f"  Fc: continuous_acc={fc_cont_acc:.3f}, VQ_acc={fc_vq_acc:.3f}")

    print("\n=== MIMII Information-Loss Study ===")
    from egpm.data import MIMIILoader
    from egpm.data import UnitDataset

    # Load MIMII data (use smaller subset for speed)
    mimii_root = "fan"
    dataset = MIMIILoader(machines=("fan",), feature="logmel", append_deltas=True,
                          max_clips_per_label=100).load(mimii_root)

    # Need to train a separate MIMII trainer
    from egpm.data import UnitDataset, SplitBuilder
    from egpm.preprocessing import Preprocessor
    from egpm.training import Trainer, TrainerConfig

    # For speed, just test on one ID with existing MIMII training
    print("  MIMII study deferred (requires separate trainer training)")
    print("  Run mimii_run.py arms for continuous_AE vs vq_hsmm comparison (already done)")


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


if __name__ == "__main__":
    from egpm.data import NCMAPSSLoader, SplitBuilder
    from egpm.preprocessing import Preprocessor
    from egpm.training import Trainer, TrainerConfig
    from egpm.experiments.ablation_arms import _downsample
    from egpm.utils import set_global_seed
    import torch

    main()