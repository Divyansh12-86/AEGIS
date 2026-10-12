import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
"""Part 3: Counterfactual Faithfulness Test for RUL.
Tests if counterfactual interventions on HSMM state/dwell move RUL predictions in expected direction.
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


def get_beliefs(seqs, trainer, fb):
    feats = []
    for s in seqs:
        X = torch.as_tensor(s.X, dtype=torch.float32)
        v = np.concatenate(trainer.event_sequences(X, torch.ones_like(X)))
        r = fb.causal_filter(v)
        feats.append(np.concatenate([r.state_posterior, r.expected_dwell / trainer.hsmm.d_max, r.state_posterior @ trainer.hsmm.A], axis=1))
    return np.concatenate(feats, axis=0)


def run_counterfactual_faithfulness():
    """Test if counterfactual interventions on state/dwell move RUL predictions in expected direction."""
    
    raw_dataset = NCMAPSSLoader().load("N-CMAPSS/N-CMAPSS DS01 005.h5")
    raw_dataset.records = [_downsample(r) for r in raw_dataset.records]

    sb = SplitBuilder(raw_dataset, seed=0)
    split = sb.build()
    train_unit_ids = split.train_unit_ids
    test_unit_ids = ['DS01_u2', 'DS01_u9']

    print("="*60)
    print("PART 3: COUNTERFACTUAL FAITHFULNESS FOR RUL")
    print("="*60)

    # Train full pipeline on ALL training engines
    train_records = [r for r in raw_dataset.records if r.unit_id in set(train_unit_ids)]
    stacked = np.concatenate([r.signals for r in train_records], axis=0)
    mean_all = stacked.mean(axis=0)
    std_all = stacked.std(axis=0)
    std_all = np.where(std_all < 1e-8, 1.0, std_all)
    stats = NormalizationStats(mean=mean_all, std=std_all, train_unit_ids=sorted(train_unit_ids))

    prep = Preprocessor(window_length=4, stride=4)
    tr_seqs = [prep.process(raw_dataset[u], norm_stats=stats) for u in train_unit_ids]
    test_seqs = [prep.process(raw_dataset[u], norm_stats=stats) for u in ['DS01_u2', 'DS01_u9']]

    set_global_seed(0)
    tcfg = TrainerConfig(
        window_length=4, embed_dim=16, hidden=32, n_conv_blocks=2, n_codes=16, d_max=10,
        n_states=5, em_max_iter=10, em_restarts=3, aux_rul_lambda=0.0,
        state_dim=None, dead_code_revival=False, stage1_steps=300, batch_size=64, seed=0, log_every=0,
    )
    trainer = Trainer(tcfg, n_channels=18)
    Xtr = torch.as_tensor(np.concatenate([s.X for s in tr_seqs]), dtype=torch.float32)
    Mtr = torch.ones_like(Xtr)
    trainer.train_stage1(Xtr, torch.ones_like(Xtr))
    ltr = [len(s.X) for s in tr_seqs]
    ev_tr = trainer.window_runs_to_event_runs(trainer.event_sequences(Xtr, torch.ones_like(Xtr)), ltr)
    trainer.train_stage2(ev_tr)
    fb = ForwardBackward(trainer.hsmm)
    ridge_w = _ridge_fit(np.concatenate([b for b in [np.concatenate([r.state_posterior, r.expected_dwell / trainer.hsmm.d_max], axis=1) for r in [ForwardBackward(trainer.hsmm).causal_filter(np.concatenate(trainer.event_sequences(torch.as_tensor(s.X, dtype=torch.float32), torch.ones_like(s.X)))) for s in tr_seqs]]], axis=0), np.concatenate([s.rul_labels for s in tr_seqs]))

    # Get beliefs for test engines
    fb_obj = ForwardBackward(trainer.hsmm)
    test_results = {}

    for test_unit in ['DS01_u2', 'DS01_u9']:
        test_seq = [s for s in [prep.process(raw_dataset[u], norm_stats=stats) for u in ['DS01_u2', 'DS01_u9']] if s.unit_id == test_unit][0]
        
        X_test = torch.as_tensor(test_seq.X, dtype=torch.float32)
        v_test = np.concatenate(trainer.event_sequences(X_test, torch.ones_like(X_test)))
        r_test = ForwardBackward(trainer.hsmm).causal_filter(v_test)
        
        # Original predictions
        orig_feats = np.concatenate([r_test.state_posterior, r_test.expected_dwell / trainer.hsmm.d_max], axis=1)
        orig_pred = _ridge_predict(_ridge_fit(np.concatenate([np.concatenate([r.state_posterior, r.expected_dwell / trainer.hsmm.d_max], axis=1) for r in [ForwardBackward(trainer.hsmm).causal_filter(np.concatenate(trainer.event_sequences(torch.as_tensor(s.X, dtype=torch.float32), torch.ones_like(s.X)))) for s in tr_seqs]], axis=0), np.concatenate([s.rul_labels for s in tr_seqs])), orig_feats)
        
        # Counterfactual 1: Reset dwell to 1 (just entered state) -> should increase RUL
        cf_dwell_reset = r_test.expected_dwell.copy()
        cf_dwell_reset[:] = 1.0  # just entered state
        cf1_feats = np.concatenate([r_test.state_posterior, cf_dwell_reset / trainer.hsmm.d_max], axis=1)
        cf1_pred = _ridge_predict(_ridge_fit(np.concatenate([np.concatenate([r.state_posterior, r.expected_dwell / trainer.hsmm.d_max], axis=1) for r in [ForwardBackward(trainer.hsmm).causal_filter(np.concatenate(trainer.event_sequences(torch.as_tensor(s.X, dtype=torch.float32), torch.ones_like(s.X)))) for s in tr_seqs]], axis=0), np.concatenate([s.rul_labels for s in tr_seqs])), cf1_feats)
        
        # Counterfactual 2: Shift to "healthier" state (lower index = healthier)
        # Find dominant state, shift to previous (healthier) state
        dominant_state = r_test.state_posterior.argmax(axis=1)
        shifted_state = np.clip(dominant_state - 1, 0, 4)  # shift to healthier (lower index)
        cf2_state_post = np.zeros_like(r_test.state_posterior)
        for t in range(len(shifted_state)):
            cf2_state_post[t, shifted_state[t]] = 1.0
        cf2_feats = np.concatenate([cf2_state_post, r_test.expected_dwell / trainer.hsmm.d_max], axis=1)
        
        # For simplicity, let's just test the dwell reset counterfactual
        # This is the clearest counterfactual: "what if we just entered this state?"
        
        # Re-train ridge on all training data for clean counterfactual
        tr_feats_all = []
        for s in tr_seqs:
            X = torch.as_tensor(s.X, dtype=torch.float32)
            v = np.concatenate(trainer.event_sequences(X, torch.ones_like(X)))
            r = ForwardBackward(trainer.hsmm).causal_filter(v)
            feats = np.concatenate([r.state_posterior, r.expected_dwell / trainer.hsmm.d_max], axis=1)
            tr_feats.append(feats)
        tr_feats_all = np.concatenate([f for f in [np.concatenate([r.state_posterior, r.expected_dwell / trainer.hsmm.d_max], axis=1) for r in [ForwardBackward(trainer.hsmm).causal_filter(np.concatenate(trainer.event_sequences(torch.as_tensor(s.X, dtype=torch.float32), torch.ones_like(s.X))) for s in tr_seqs]]], axis=0)
        y_all = np.concatenate([s.rul_labels for s in tr_seqs])
        ridge_w = _ridge_fit(tr_feats_all, np.concatenate([s.rul_labels for s in tr_seqs]))
        
        # Test counterfactual
        cf_results = {"unit": test_unit}
        total_windows = 0
        cf1_correct = 0
        cf2_correct = 0
        total_windows = 0
        
        for test_unit in ['DS01_u2', 'DS01_u9']:
            test_seq = [s for s in [prep.process(raw_dataset[u], norm_stats=stats) for u in ['DS01_u2', 'DS01_u9']] if s.unit_id == test_unit][0]
            X_test = torch.as_tensor(test_seq.X, dtype=torch.float32)
            v_test = np.concatenate(trainer.event_sequences(X_test, torch.ones_like(X_test)))
            r_test = ForwardBackward(trainer.hsmm).causal_filter(v_test)
            
            # Original features
            orig_feats = np.concatenate([r_test.state_posterior, r_test.expected_dwell / trainer.hsmm.d_max], axis=1)
            orig_pred = _ridge_predict(_ridge_fit(
                np.concatenate([np.concatenate([r.state_posterior, r.expected_dwell / trainer.hsmm.d_max], axis=1) for r in [ForwardBackward(trainer.hsmm).causal_filter(np.concatenate(trainer.event_sequences(torch.as_tensor(s.X, dtype=torch.float32), torch.ones_like(s.X)))) for s in tr_seqs]], axis=0),
                np.concatenate([s.rul_labels for s in tr_seqs])
            ), orig_feats)
            
            # Counterfactual 1: Reset dwell to 1 (just entered state)
            cf_dwell = r_test.expected_dwell.copy()
            cf_dwell[:] = 1.0
            cf1_feats = np.concatenate([r_test.state_posterior, cf_dwell / trainer.hsmm.d_max], axis=1)
            
            # Counterfactual 2: State shift to healthier (state - 1)
            dominant = r_test.state_posterior.argmax(axis=1)
            shifted = np.clip(dominant - 1, 0, 4)
            cf2_state = np.zeros_like(r_test.state_posterior)
            for t in range(len(shifted)):
                cf2_state[t, shifted[t]] = 1.0
            cf2_feats = np.concatenate([cf2_state, r_test.expected_dwell / trainer.hsmm.d_max], axis=1)
            
            # Retrain ridge on all training for clean counterfactual
            tr_feats_list = []
            for s in tr_seqs:
                X = torch.as_tensor(s.X, dtype=torch.float32)
                v = np.concatenate(trainer.event_sequences(X, torch.ones_like(X)))
                r = ForwardBackward(trainer.hsmm).causal_filter(v)
                feats = np.concatenate([r.state_posterior, r.expected_dwell / trainer.hsmm.d_max], axis=1)
                tr_feats_list.append(feats)
            tr_feats_all = np.concatenate(tr_feats_list, axis=0)
            y_all = np.concatenate([s.rul_labels for s in tr_seqs])
            ridge_w = _ridge_fit(tr_feats_all, y_all)
            
            orig_pred = _ridge_predict(ridge_w, orig_feats)
            cf1_pred = _ridge_predict(ridge_w, cf1_feats)
            cf2_pred = _ridge_predict(ridge_w, cf2_feats)
            
            # True RUL
            true_rul = test_seq.rul_labels
            
            # Check direction: CF1 (dwell reset) should increase RUL
            cf1_increased = (cf1_pred > orig_pred).mean()
            # CF2 (healthier state) should increase RUL
            cf2_increased = (cf2_pred > orig_pred).mean()
            
            # Also check if original predictions are reasonable
            orig_rmse = np.sqrt(np.mean((orig_pred - true_rul)**2))
            
            print(f"\n{test_unit}:")
            print(f"  Orig RMSE: {orig_rmse:.1f}")
            print(f"  CF1 (dwell reset to 1): {cf1_increased*100:.1f}% windows show RUL increase (expected: increase)")
            print(f"  CF2 (state-1): {cf2_increased*100:.1f}% windows show RUL increase (expected: increase)")

    return {"counterfactual": "done"}

if __name__ == "__main__":
    from egpm.data.loaders import NCMAPSSLoader
    from egpm.data.split_builder import SplitBuilder, NormalizationStats
    from egpm.experiments.ablation_arms import _downsample, _ridge_fit, _ridge_predict
    from egpm.preprocessing.preprocessor import Preprocessor
    from egpm.training.trainer import Trainer, TrainerConfig
    from egpm.grammar.forward_backward import ForwardBackward
    from egpm.utils import set_global_seed
    from egpm.experiments.ablation_arms import _downsample
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
