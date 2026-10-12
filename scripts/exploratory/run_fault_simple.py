import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
"""Fault Conformal using validation engines for calibration."""
import json
import time
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from egpm.data.loaders import NCMAPSSLoader
from egpm.data.split_builder import SplitBuilder, NormalizationStats
from egpm.experiments.ablation_arms import _downsample
from egpm.preprocessing.preprocessor import Preprocessor
from egpm.training.trainer import Trainer, TrainerConfig
from egpm.grammar.forward_backward import ForwardBackward
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


def main():
    raw_dataset = NCMAPSSLoader().load("N-CMAPSS/N-CMAPSS DS01 005.h5")
    raw_dataset.records = [_downsample(r) for r in raw_dataset.records]

    sb = SplitBuilder(raw_dataset, seed=0)
    split = sb.build()
    train_unit_ids = split.train_unit_ids
    val_unit_ids = ['DS01_u1', 'DS01_u10']
    test_unit_ids = ['DS01_u2', 'DS01_u9']

    print(f"Training: {train_unit_ids}")
    print(f"Val (cal): {['DS01_u1', 'DS01_u10']}")
    print(f"Test: {test_unit_ids}")

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
        fb_local = ForwardBackward(trainer.hsmm)
        feats = []
        for s in seqs:
            X = torch.as_tensor(s.X, dtype=torch.float32)
            v = np.concatenate(trainer.event_sequences(X, torch.ones_like(X)))
            r = fb_local.causal_filter(v)
            feats.append(np.concatenate([r.state_posterior, r.expected_dwell / trainer.hsmm.d_max, r.state_posterior @ trainer.hsmm.A], axis=1))
        return np.concatenate(feats, axis=0)

    # Training beliefs + labels
    tr_feats = get_beliefs(tr_seqs)
    y_fault = np.concatenate([np.array([int(raw_dataset[u].fault_label)] * len(s.X)) for s, u in zip(tr_seqs, train_unit_ids)])

    # Train classifier
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(multi_class='multinomial', solver='lbfgs', C=1.0, max_iter=1000, random_state=0)
    clf.fit(tr_feats, y_fault)

    # Calibration on validation engines (DS01_u1, DS01_u10)
    val_seqs = [prep.process(raw_dataset[u], norm_stats=stats) for u in ['DS01_u1', 'DS01_u10']]
    val_feats = get_beliefs([prep.process(raw_dataset[u], norm_stats=stats) for u in ['DS01_u1', 'DS01_u10']])
    val_labels = np.concatenate([np.array([int(raw_dataset[u].fault_label)] * len(s.X)) for u, s in zip(['DS01_u1', 'DS01_u10'], [prep.process(raw_dataset[u], norm_stats=stats) for u in ['DS01_u1', 'DS01_u10']])])

    probs_val = clf.predict_proba(val_feats)
    val_labels = np.concatenate([np.array([int(raw_dataset[u].fault_label)] * len(s.X)) for u, s in zip(['DS01_u1', 'DS01_u10'], [prep.process(raw_dataset[u], norm_stats=stats) for u in ['DS01_u1', 'DS01_u10']])])

    probs_true = probs_val[np.arange(len(val_labels)), np.array([np.where(clf.classes_ == c)[0][0] for c in val_labels])]
    cal_scores = 1 - probs_true
    cal_scores.sort()
    print(f"Calibration: n={len(cal_scores)}, range=[{cal_scores.min():.3f}, {cal_scores.max():.3f}]")
    for level in [0.5, 0.8, 0.9]:
        q = np.quantile(cal_scores, level, method="higher")
        print(f"  {int(level*100)}% quantile: {q:.3f}")

    # Test on locked test engines
    for test_unit in ['DS01_u2', 'DS01_u9']:
        test_seq = [s for s in test_seqs if s.unit_id == test_unit][0]
        test_feats = get_beliefs([test_seq])
        probs = clf.predict_proba(test_feats)
        true_fc = int(raw_dataset[test_unit].fault_label)
        test_labels = np.array([int(raw_dataset[test_unit].fault_label)] * len(test_seq.X))
        pred = clf.predict(test_feats)
        acc = (np.argmax(probs, axis=1) == np.array([int(raw_dataset[test_unit].fault_label)] * len(test_seq.X))).mean()
        
        probs_true_test = probs[:, np.where(clf.classes_ == int(raw_dataset[test_unit].fault_label))[0][0]]
        nonconf = 1 - probs_true_test
        
        print(f"\n{test_unit}: Acc={acc:.3f}")
        for level in [0.5, 0.8, 0.9]:
            q = np.quantile(np.array(cal_scores), level, method="higher")
            cov = (1 - probs[:, np.where(clf.classes_ == int(raw_dataset[test_unit].fault_label))[0][0]] <= q).mean()
            print(f"  {int(level*100)}%: q={q:.3f}, cov={cov:.3f}")

    # HEALTH: entropy vs health index correlation
    print("\n--- HEALTH TRUST SCORE (entropy vs health) ---")
    for test_unit in ['DS01_u2', 'DS01_u9']:
        test_seq = [s for s in test_seqs if s.unit_id == test_unit][0]
        r_test = ForwardBackward(trainer.hsmm).causal_filter(
            np.concatenate(trainer.event_sequences(torch.as_tensor(test_seq.X, dtype=torch.float32), torch.ones_like(torch.as_tensor(test_seq.X, dtype=torch.float32))))
        )
        entropy = -np.sum(r_test.state_posterior * np.log(np.clip(r_test.state_posterior, 1e-10, 1)), axis=1)
        health_true = test_seq.health_labels
        
        health_changes = np.abs(np.diff(health_true))
        entropy_changes = np.abs(np.diff(entropy))
        if len(health_changes) > 1:
            corr = np.corrcoef(entropy_changes, health_changes)[0,1]
            print(f"  {test_unit}: entropy-health_change corr={corr:.3f}")
        print(f"  {test_unit}: mean entropy={entropy.mean():.3f}, health range={health_true.min()}-{health_true.max()}")

    return {"fault": "done", "health": "entropy_correlation_done"}

if __name__ == "__main__":
    from egpm.data.loaders import NCMAPSSLoader
    from egpm.data.split_builder import SplitBuilder, NormalizationStats
    from egpm.experiments.ablation_arms import _downsample
    from egpm.preprocessing.preprocessor import Preprocessor
    from egpm.training.trainer import Trainer, TrainerConfig
    from egpm.grammar.forward_backward import ForwardBackward
    from egpm.utils import set_global_seed
    from egpm.experiments.ablation_arms import _downsample
    from sklearn.linear_model import LogisticRegression
    import torch
    import numpy as np
    import time
    import json

    set_global_seed(0)
    result = main()
    print("\nDone!")
