"""
Track A: CruiseBench Protocol - Chunked Processing
Uses incremental ridge fitting to handle large feature matrices.
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


class IncrementalRidge:
    """Incremental ridge regression using Woodbury identity."""
    def __init__(self, n_features, alpha=1.0):
        self.n_features = n_features
        self.alpha = alpha
        self.A = alpha * np.eye(n_features)
        self.b = np.zeros(n_features)
        self.n_samples = 0
    
    def partial_fit(self, X, y):
        """Incremental update: (X^T X + alpha I) w = X^T y"""
        self.A += X.T @ X
        self.b += X.T @ y
        self.n_samples += len(X)
    
    def predict(self, X):
        w = np.linalg.solve(self.A, self.b)
        return X @ w
    
    def get_weights(self):
        return np.linalg.solve(self.A, self.b)


def main():
    print("="*70)
    print("TRACK A: CRUISEBENCH PROTOCOL - Native 1Hz (Chunked)")
    print("="*70)
    
    raw_dataset = NCMAPSSLoader().load("N-CMAPSS/N-CMAPSS DS01 005.h5")
    # NO DOWNSAMPLING
    
    print(f"Loaded {len(raw_dataset.records)} units (native 1Hz)")
    
    sb = SplitBuilder(raw_dataset, seed=0)
    split = sb.build()
    stats = sb.fit_normalization(split)
    
    print(f"Train: {split.train_unit_ids}")
    print(f"Test: {split.test_unit_ids}")
    
    # CruiseBench: W=256, S=10
    prep = Preprocessor(window_length=256, stride=10, fs_sync=1.0)
    
    # Process test data first (smaller)
    test_seqs = [prep.process(raw_dataset[u], norm_stats=stats) for u in ['DS01_u2', 'DS01_u9']]
    for s in test_seqs:
        print(f"  Test {s.unit_id}: {len(s.X)} windows")
    
    # Process training data one engine at a time
    set_global_seed(0)
    tcfg = TrainerConfig(
        window_length=256, embed_dim=32, hidden=64, n_conv_blocks=3, n_codes=32, d_max=20,
        n_states=8, em_max_iter=10, em_restarts=3, aux_rul_lambda=0.0,
        state_dim=12, dead_code_revival=False, stage1_steps=500, batch_size=32, seed=0, log_every=0,
    )
    
    trainer = Trainer(tcfg, n_channels=18)
    
    print("Training Stage 1 (per engine)...")
    for u in split.train_unit_ids:
        print(f"  Processing {u}...")
        seq = prep.process(raw_dataset[u], norm_stats=stats)
        print(f"  {u}: {len(seq.X)} windows")
        
        X = torch.as_tensor(seq.X, dtype=torch.float32)
        M = torch.ones_like(X)
        trainer.train_stage1(X, M)
    
    # Stage 2 on all events
    print("Training Stage 2...")
    all_ev_tr = []
    for u in split.train_unit_ids:
        seq = prep.process(raw_dataset[u], norm_stats=stats)
        X = torch.as_tensor(seq.X, dtype=torch.float32)
        ev = trainer.event_sequences(X, torch.ones_like(X))
        all_ev_tr.extend(ev)
    
    trainer.train_stage2(all_ev_tr)
    
    # Incremental Ridge fitting
    fb = ForwardBackward(trainer.hsmm)
    ridge = None
    y_all = []
    
    for u in split.train_unit_ids:
        seq = prep.process(raw_dataset[u], norm_stats=stats)
        X = torch.as_tensor(seq.X, dtype=torch.float32)
        v = np.concatenate(trainer.event_sequences(X, torch.ones_like(X)))
        r = ForwardBackward(trainer.hsmm).causal_filter(v)
        feats = np.concatenate([r.state_posterior, r.expected_dwell / trainer.hsmm.d_max], axis=1)
        
        if ridge is None:
            ridge = IncrementalRidge(feats.shape[1], alpha=1.0)
        ridge.partial_fit(feats, seq.rul_labels)
        y_all.append(seq.rul_labels)
    
    w = ridge.get_weights()
    
    # Test with counterfactuals
    for test_unit in ['DS01_u2', 'DS01_u9']:
        test_seq = prep.process(raw_dataset[test_unit], norm_stats=stats)
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
