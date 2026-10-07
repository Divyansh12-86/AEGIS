"""
Track A: CruiseBench Protocol - Compromise Version
Uses W=128, S=10 for memory efficiency while maintaining protocol spirit.
"""

import json
import time
import numpy as np
import torch

from egpm.data.loaders import NCMAPSSLoader
from egpm.data.split_builder import SplitBuilder, NormalizationStats
from egpm.experiments.ablation_arms import _downsample, _ridge_fit, _ridge_predict
from egpm.preprocessing.preprocessor import Preprocessor
from egpm.training.trainer import Trainer, TrainerConfig
from egpm.grammar.forward_backward import ForwardBackward
from egpm.evaluation.evaluator import rul_metrics
from egpm.utils import set_global_seed
from egpm.experiments.ablation_arms import _downsample, _ridge_fit, _ridge_predict


def main():
    print("="*70)
    print("TRACK A: CRUISEBENCH PROTOCOL - W=128, S=10 (Compromise)")
    print("="*70)
    
    raw_dataset = NCMAPSSLoader().load("N-CMAPSS/N-CMAPSS DS01 005.h5")
    raw_dataset.records = [_downsample(r) for r in raw_dataset.records]  # Use downsampling for memory
    
    print(f"Loaded {len(raw_dataset.records)} units (downsampled)")
    
    sb = SplitBuilder(raw_dataset, seed=0)
    split = sb.build()
    stats = sb.fit_normalization(split)
    
    print(f"Train: {split.train_unit_ids}")
    print(f"Test: {split.test_unit_ids}")
    
    # Compromise: W=128, S=10 (reduces memory ~4x vs W=256)
    prep = Preprocessor(window_length=128, stride=10, fs_sync=1.0)
    
    print("Processing training data...")
    tr_seqs = []
    for u in split.train_unit_ids:
        seq = prep.process(raw_dataset[u], norm_stats=stats)
        print(f"  {u}: {len(seq.X)} windows")
        tr_seqs.append(seq)
    
    test_seqs = [prep.process(raw_dataset[u], norm_stats=stats) for u in ['DS01_u2', 'DS01_u9']]
    for s in test_seqs:
        print(f"  {s.unit_id}: {len(s.X)} windows")
    
    set_global_seed(0)
    tcfg = TrainerConfig(
        window_length=128, embed_dim=32, hidden=64, n_conv_blocks=2, n_codes=32, d_max=20,
        n_states=8, em_max_iter=10, em_restarts=3, aux_rul_lambda=0.0,
        state_dim=12, dead_code_revival=False, stage1_steps=500, batch_size=64, seed=0, log_every=0,
    )
    
    trainer = Trainer(tcfg, n_channels=18)
    
    for i, s in enumerate(tr_seqs):
        X = torch.as_tensor(s.X, dtype=torch.float32)
        M = torch.ones_like(X)
        trainer.train_stage1(X, M)
        print(f"  Engine {i+1}/{len(tr_seqs)} stage1 done")
    
    ev_tr = []
    for s in tr_seqs:
        X = torch.as_tensor(s.X, dtype=torch.float32)
        ev = trainer.event_sequences(X, torch.ones_like(X))
        ev_tr.extend(ev)
    
    print("Training Stage 2...")
    trainer.train_stage2(ev_tr)
    
    fb = ForwardBackward(trainer.hsmm)
    tr_feats = []
    y_all = []
    for s in tr_seqs:
        X = torch.as_tensor(s.X, dtype=torch.float32)
        v = np.concatenate(trainer.event_sequences(X, torch.ones_like(X)))
        r = fb.causal_filter(v)
        feats = np.concatenate([r.state_posterior, r.expected_dwell / trainer.hsmm.d_max], axis=1)
        tr_feats.append(feats)
        y_all.append(s.rul_labels)
    
    tr_feats = np.concatenate(tr_feats, axis=0)
    y_all = np.concatenate(y_all)
    
    w = _ridge_fit(tr_feats, y_all)
    
    for test_unit in ['DS01_u2', 'DS01_u9']:
        test_seq = [s for s in [prep.process(raw_dataset[u], norm_stats=stats) for u in ['DS01_u2', 'DS01_u9']] if s.unit_id == test_unit][0]
        X_test = torch.as_tensor(test_seq.X, dtype=torch.float32)
        v_test = np.concatenate(trainer.event_sequences(X_test, torch.ones_like(X_test)))
        r_test = ForwardBackward(trainer.hsmm).causal_filter(v_test)
        
        orig_feats = np.concatenate([r_test.state_posterior, r_test.expected_dwell / trainer.hsmm.d_max], axis=1)
        orig_pred = _ridge_predict(w, orig_feats)
        
        cf_dwell = r_test.expected_dwell.copy()
        cf_dwell[:] = 1.0
        cf1_feats = np.concatenate([r_test.state_posterior, cf_dwell / trainer.hsmm.d_max], axis=1)
        cf1_pred = _ridge_predict(w, cf1_feats)
        
        dominant = r_test.state_posterior.argmax(axis=1)
        shifted = np.clip(dominant - 1, 0, 4)
        cf2_state = np.zeros_like(r_test.state_posterior)
        for t in range(len(shifted)):
            cf2_state[t, shifted[t]] = 1.0
        cf2_feats = np.concatenate([cf2_state, r_test.expected_dwell / trainer.hsmm.d_max], axis=1)
        cf2_pred = _ridge_predict(w, cf2_feats)
        
        true_rul = test_seq.rul_labels
        orig_rmse = np.sqrt(np.mean((orig_pred - true_rul)**2))
        cf1_increased = (cf1_pred > orig_pred).mean()
        cf2_increased = (cf2_pred > orig_pred).mean()
        
        print(f"\n{test_unit}:")
        print(f"  Orig RMSE: {orig_rmse:.1f}")
        print(f"  CF1 (dwell=1): {cf1_increased*100:.1f}% increase")
        print(f"  CF2 (state-1): {cf2_increased*100:.1f}% increase")
    
    return {"status": "done"}

if __name__ == "__main__":
    from egpm.data.loaders import NCMAPSSLoader
    from egpm.data.split_builder import SplitBuilder, NormalizationStats
    from egpm.experiments.ablation_arms import _downsample, _ridge_fit, _ridge_predict
    from egpm.preprocessing.preprocessor import Preprocessor
    from egpm.training.trainer import Trainer, TrainerConfig
    from egpm.grammar.forward_backward import ForwardBackward
    from egpm.evaluation.evaluator import rul_metrics
    from egpm.utils import set_global_seed
    from egpm.experiments.ablation_arms import _downsample, _ridge_fit, _ridge_predict
    import torch
    import numpy as np
    import time
    import json
    from egpm.preprocessing.preprocessor import Preprocessor
    from egpm.training.trainer import Trainer, TrainerConfig
    from egpm.grammar.forward_backward import ForwardBackward
    from egpm.utils import set_global_seed
    from egpm.experiments.ablation_arms import _downsample, _ridge_fit, _ridge_predict

    set_global_seed(0)
    result = main()
    print("\nDone!")
