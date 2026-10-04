#!/usr/bin/env python
"""
Robustness sweep runner for EGPM ablation experiments.
Runs parameter sweeps on VALIDATION set (not test) for config selection.
"""
import json
import time
import argparse
from typing import Dict, List, Any
import numpy as np
import torch

from egpm.experiments.ablation_arms import run_ablation_experiment
from egpm.data import NCMAPSSLoader, SplitBuilder, UnitDataset, UnitRecord
from egpm.preprocessing import Preprocessor
from egpm.training import Trainer, TrainerConfig
from egpm.grammar import ForwardBackward, FirstOrderHMMFitter
from egpm.rul import PhaseTypeRUL
from egpm.evaluation import rul_metrics, per_unit_metrics, summarize_units
from egpm.interpretability.metrics import event_stability, temporal_validity
from egpm.utils import set_global_seed
from egpm.experiments.ablation_arms import _downsample


DEFAULT_CONFIG = {
    "n_seeds": 3,
    "window_length": 4,
    "n_states": 5,
    "d_max": 10,
    "n_codes": 16,
    "stage1_steps": 300,
    "em_max_iter": 10,
    "em_restarts": 3,
}

SWEEPS = {
    "VQ_codebook_size": {
        "param": "n_codes",
        "values": [4, 8, 16, 32],
        "datasets": ["DS01", "DS02"],
    },
    "HSMM_states": {
        "param": "n_states",
        "values": [2, 3, 4, 6],
        "datasets": ["DS01", "DS02"],
    },
    "HSMM_d_max": {
        "param": "d_max",
        "values": [8, 10, 12, 15],
        "datasets": ["DS01", "DS02"],
    },
    "Stride": {
        "param": "stride",
        "values": [1, 2, 4, 8],
        "datasets": ["DS01", "DS02"],
    },
    "Encoder_width": {
        "param": "hidden",
        "values": [16, 32, 64, 128],
        "datasets": ["DS01", "DS02"],
    },
    "Encoder_attention": {
        "param": "use_attention",
        "values": [False, True],
        "datasets": ["DS01", "DS02"],
    },
    "VQ_commitment_beta": {
        "param": "commitment_beta",
        "values": [0.1, 0.25, 0.5, 1.0],
        "datasets": ["DS01", "DS02"],
    },
    "EMA_decay": {
        "param": "ema_decay",
        "values": [0.9, 0.99, 0.999],
        "datasets": ["DS01", "DS02"],
    },
    "Stage1_steps": {
        "param": "stage1_steps",
        "values": [100, 200, 300, 500],
        "datasets": ["DS01", "DS02"],
    },
    "EM_iterations": {
        "param": "em_max_iter",
        "values": [5, 10, 20, 50],
        "datasets": ["DS01", "DS02"],
    },
    "RUL_cap": {
        "param": "max_cycles_per_unit",
        "values": [50, 100, 200, None],
        "datasets": ["DS01", "DS02"],
    },
}

def run_ablation_on_val(h5_path: str, config: Dict, n_seeds: int = 3) -> Dict:
    """Run ablation experiment using VALIDATION set for evaluation."""
    from egpm.data import NCMAPSSLoader, SplitBuilder, UnitDataset, UnitRecord
    from egpm.preprocessing import Preprocessor
    from egpm.training import Trainer, TrainerConfig
    from egpm.grammar import ForwardBackward, FirstOrderHMMFitter
    from egpm.rul import PhaseTypeRUL
    from egpm.evaluation import rul_metrics, per_unit_metrics, summarize_units
    from egpm.interpretability.metrics import event_stability, temporal_validity
    from egpm.anomaly import AnomalyScorer
    from egpm.baselines import WindowAutoEncoder, TokenMarkovModel
    from egpm.ablations import ContinuousHSMM
    from egpm.evaluation import auroc, auprc, f1_at_threshold
    from egpm.evaluation.stats import auroc_bootstrap_ci, paired_wilcoxon
    from egpm.utils import set_global_seed
    from egpm.experiments.ablation_arms import _downsample
    
    dataset = NCMAPSSLoader().load(h5_path)
    dataset.records = [_downsample(r) for r in dataset.records]
    
    sb = SplitBuilder(dataset, seed=0)
    split = sb.build()
    stats = sb.fit_normalization(split)
    prep = Preprocessor(window_length=config.get("window_length", 4))
    
    # Use validation units for evaluation
    val_ids = split.val_unit_ids
    test_ids = split.test_unit_ids
    
    train = [prep.process(dataset[u], norm_stats=stats) for u in split.train_unit_ids]
    val = [prep.process(dataset[u], norm_stats=stats) for u in val_ids]
    test = [prep.process(dataset[u], norm_stats=stats) for u in test_ids]
    
    Xtr = torch.as_tensor(np.concatenate([s.X for s in train]), dtype=torch.float32)
    Mtr = torch.ones_like(Xtr)
    run_lens = [len(s.X) for s in train]
    y_tr = np.concatenate([s.rul_labels for s in train])
    
    # shared encoder/VQ
    tcfg = TrainerConfig(
        window_length=config.get("window_length", 4), embed_dim=16, hidden=32,
        n_conv_blocks=2, n_codes=config.get("n_codes", 16), d_max=config.get("d_max", 10),
        n_states=config.get("n_states", 5), em_max_iter=10, stage1_steps=config.get("stage1_steps", 300),
        batch_size=64, seed=0, log_every=0,
    )
    trainer = Trainer(tcfg, n_channels=Xtr.shape[-1])
    trainer.train_stage1(Xtr, Mtr)
    run_lengths = [len(s.X) for s in train]
    train_event_runs = trainer.window_runs_to_event_runs(
        trainer.event_sequences(Xtr, Mtr), run_lens
    )
    
    def toks(seq_list):
        out = []
        for s in seq_list:
            X = torch.as_tensor(s.X, dtype=torch.float32)
            out.append(np.concatenate(trainer.event_sequences(X, torch.ones_like(X))))
        return out
    
    train_event_runs = trainer.window_runs_to_event_runs(
        trainer.event_sequences(Xtr, Mtr), run_lens
    )
    
    def tokens_of(ids):
        out = []
        for w in [prep.process(dataset[u], norm_stats=stats).X for u in ids]:
            X = torch.as_tensor(w, dtype=torch.float32)
            out.append(np.concatenate(trainer.event_sequences(X, torch.ones_like(X))))
        return out
    
    ev_val = tokens_of(val_ids)  # validation
    ev_test = tokens_of(test_ids)  # test (for reference only)
    
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
    
    def wins(ids):
        return [prep.process(dataset[i], norm_stats=stats).X for i in ids]
    
    va_runs = wins(split.val_unit_ids)
    te_runs = wins(split.test_unit_ids)
    
    # For validation evaluation
    val_runs = val
    test_runs = test
    
    # Build validation scores
    def score_all(arm, ids):
        if arm == "continuous_AE":
            ae = WindowAutoEncoder(n_channels=Xtr.shape[-1], window_length=config.get("window_length", 4))
            opt = torch.optim.Adam(ae.parameters(), lr=1e-3)
            ae.train()
            for _ in range(300):
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
            return ae_score(ids)
        
        # ... this is getting too complex inline. Let me write a proper script file.
        pass

    return {}


def main():
    pass


if __name__ == "__main__":
    main()
