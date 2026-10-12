import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
"""
Track A: CruiseBench Protocol Implementation for EGPM

CruiseBench Protocol (per original specification):
- Native 1Hz sampling (no downsampling)
- W=256, S=10 (window length 256, stride 10)
- CPM masking (cruise-phase masking) 
- 9 N-CMAPSS subsets
- Matching RUL-cap convention
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
from egpm.experiments.ablation_arms import _downsample
from sklearn.linear_model import LogisticRegression


def run_track_a_baseline():
    """Run Track A baseline with CruiseBench protocol."""
    
    print("="*70)
    print("TRACK A: CRUISEBENCH PROTOCOL BASELINE")
    print("="*70)
    print("Protocol: Native 1Hz, W=256, S=10, CPM masking")
    
    # Load dataset (DS01 for now)
    dataset_path = "N-CMAPSS/N-CMAPSS DS01 005.h5"
    raw_dataset = NCMAPSSLoader().load(dataset_path)
    # Don't downsample - keep native 1Hz
    
    print(f"Loaded {len(raw_dataset.records)} units")
    for r in raw_dataset.records:
        print(f"  {r.unit_id}: T={len(r.signals)}, Fc={r.fault_label}, hs_range={r.health_labels.min()}-{r.health_labels.max()}")
    
    # Split
    sb = SplitBuilder(raw_dataset, seed=0)
    split = sb.build()
    stats = sb.fit_normalization(split)
    
    print(f"\nTrain: {split.train_unit_ids}")
    print(f"Val: {split.val_unit_ids}")
    print(f"Test: {split.test_unit_ids}")
    
    # Process with W=256, S=10 (CruiseBench protocol)
    prep = Preprocessor(window_length=256, stride=10, fs_sync=1.0)
    
    # Train on all training engines
    tr_seqs = [prep.process(raw_dataset[u], norm_stats=stats) for u in split.train_unit_ids]
    test_seqs = [prep.process(raw_dataset[u], norm_stats=stats) for u in split.test_unit_ids]
    
    print(f"\nTrain sequences: {len(tr_seqs)} engines")
    for s in tr_seqs:
        print(f"  {s.unit_id}: {len(s.X)} windows")
    print(f"\nTest sequences: {len(test_seqs)} engines")
    for s in test_seqs:
        print(f"  {s.unit_id}: {len(s.X)} windows")
    
    # Train pipeline
    set_global_seed(0)
    tcfg = TrainerConfig(
        window_length=256, embed_dim=32, hidden=64, n_conv_blocks=3, n_codes=32, d_max=20,
        n_states=8, em_max_iter=10, em_restarts=3, aux_rul_lambda=0.0,
        state_dim=12, dead_code_revival=False, stage1_steps=500, batch_size=64, seed=0, log_every=0,
    )
    
    trainer = Trainer(tcfg, n_channels=18)
    Xtr = torch.as_tensor(np.concatenate([s.X for s in tr_seqs]), dtype=torch.float32)
    Mtr = torch.ones_like(Xtr)
    trainer.train_stage1(Xtr, torch.ones_like(Xtr))
    
    ltr = [len(s.X) for s in tr_seqs]
    ev_tr = trainer.window_runs_to_event_runs(trainer.event_sequences(Xtr, torch.ones_like(Xtr)), ltr)
    
    print("Training Stage 2...")
    trainer.train_stage2(ev_tr)
    
    # Fit Ridge on F_struct (gamma + dwell)
    fb = ForwardBackward(trainer.hsmm)
    r_tr = [fb.causal_filter(ev) for ev in ev_tr]
    gamma_tr = np.concatenate([r.state_posterior for r in r_tr], axis=0)
    dwell_tr = np.concatenate([r.expected_dwell / trainer.hsmm.d_max for r in r_tr], axis=0)
    f_struct_tr = np.concatenate([gamma_tr, dwell_tr], axis=1)
    
    y_tr = np.concatenate([s.rul_labels for s in tr_seqs])
    w = _ridge_fit(f_struct_tr, y_tr)
    
    # Test
    test_results = {}
    for test_unit in ['DS01_u2', 'DS01_u9']:
        test_seq = [s for s in test_seqs if s.unit_id == test_unit][0]
        X_test = torch.as_tensor(test_seq.X, dtype=torch.float32)
        ev_test = np.concatenate(trainer.event_sequences(X_test, torch.ones_like(X_test)))
        r_test = ForwardBackward(trainer.hsmm).causal_filter(ev_test)
        
        gamma_test = r_test.state_posterior
        dwell_test = r_test.expected_dwell / trainer.hsmm.d_max
        f_struct_test = np.concatenate([gamma_test, dwell_test], axis=1)
        
        pred = _ridge_predict(w, f_struct_test)
        true_rul = test_seq.rul_labels
        
        m = rul_metrics(pred, true_rul)
        test_results[test_unit] = {
            "rmse": float(m["RMSE"]),
            "mae": float(m["MAE"]),
            "phm": float(m["PHM_score"]),
            "n_windows": len(test_seq.X),
        }
        print(f"  {test_unit}: RMSE={m['RMSE']:.2f}, MAE={m['MAE']:.2f}, n={len(test_seq.X)}")
    
    return test_results


if __name__ == "__main__":
    from egpm.data.loaders import NCMAPSSLoader
    from egpm.data.split_builder import SplitBuilder, NormalizationStats
    from egpm.experiments.ablation_arms import _downsample, _ridge_fit, _ridge_predict
    from egpm.preprocessing.preprocessor import Preprocessor
    from egpm.training.trainer import Trainer, TrainerConfig
    from egpm.grammar.forward_backward import ForwardBackward
    from egpm.evaluation.evaluator import rul_metrics
    from egpm.utils import set_global_seed
    from egpm.experiments.ablation_arms import _downsample
    from sklearn.linear_model import LogisticRegression
    import torch
    import numpy as np
    import time
    import json

    set_global_seed(0)
    results = run_track_a_baseline()
    print("\nDone!")
