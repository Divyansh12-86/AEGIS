"""Part 3: Counterfactual Faithfulness Test for RUL."""
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
        feats.append(np.concatenate([r.state_posterior, r.expected_dwell / trainer.hsmm.d_max], axis=1))
    return np.concatenate(feats, axis=0)


def main():
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
    trainer.train_stage1(Xtr, Mtr)
    ltr = [len(s.X) for s in tr_seqs]
    ev_tr = trainer.window_runs_to_event_runs(trainer.event_sequences(Xtr, Mtr), ltr)
    trainer.train_stage2(ev_tr)
    fb = ForwardBackward(trainer.hsmm)

    def get_beliefs(seqs):
        feats = []
        fb_local = ForwardBackward(trainer.hsmm)
        for s in seqs:
            X = torch.as_tensor(s.X, dtype=torch.float32)
            v = np.concatenate(trainer.event_sequences(X, torch.ones_like(X)))
            r = fb_local.causal_filter(v)
            feats.append(np.concatenate([r.state_posterior, r.expected_dwell / trainer.hsmm.d_max], axis=1))
        return np.concatenate(feats, axis=0)

    # Training features for ridge
    tr_feats = []
    for s in tr_seqs:
        X = torch.as_tensor(s.X, dtype=torch.float32)
        v = np.concatenate(trainer.event_sequences(X, torch.ones_like(X)))
        r = ForwardBackward(trainer.hsmm).causal_filter(v)
        feats.append(np.concatenate([r.state_posterior, r.expected_dwell / trainer.hsmm.d_max], axis=1))
    tr_feats = np.concatenate(feats, axis=0)
    y_all = np.concatenate([s.rul_labels for s in tr_seqs])
    ridge_w = _ridge_fit(tr_feats, y_all)

    fb_obj = ForwardBackward(trainer.hsmm)

    for test_unit in ['DS01_u2', 'DS01_u9']:
        test_seq = [s for s in [prep.process(raw_dataset[u], norm_stats=stats) for u in ['DS01_u2', 'DS01_u9']] if s.unit_id == test_unit][0]
        
        X_test = torch.as_tensor(test_seq.X, dtype=torch.float32)
        v_test = np.concatenate(trainer.event_sequences(X_test, torch.ones_like(X_test)))
        r_test = ForwardBackward(trainer.hsmm).causal_filter(v_test)
        
        # Original features
        orig_feats = np.concatenate([r_test.state_posterior, r_test.expected_dwell / trainer.hsmm.d_max], axis=1)
        orig_pred = _ridge_predict(ridge_w, orig_feats)
        
        # Counterfactual 1: Reset dwell to 1 (just entered state) -> should increase RUL
        cf_dwell = r_test.expected_dwell.copy()
        cf_dwell[:] = 1.0
        cf1_feats = np.concatenate([r_test.state_posterior, cf_dwell / trainer.hsmm.d_max], axis=1)
        cf1_pred = _ridge_predict(ridge_w, cf1_feats)
        
        # Counterfactual 2: State shift to healthier (state - 1)
        dominant = r_test.state_posterior.argmax(axis=1)
        shifted = np.clip(dominant - 1, 0, 4)
        cf2_state = np.zeros_like(r_test.state_posterior)
        for t in range(len(shifted)):
            cf2_state[t, shifted[t]] = 1.0
        cf2_feats = np.concatenate([cf2_state, r_test.expected_dwell / trainer.hsmm.d_max], axis=1)
        cf2_pred = _ridge_predict(ridge_w, cf2_feats)
        
        true_rul = test_seq.rul_labels
        orig_rmse = np.sqrt(np.mean((orig_pred - true_rul)**2))
        
        cf1_increased = (cf1_pred > orig_pred).mean()
        cf2_increased = (cf2_pred > orig_pred).mean()
        
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
