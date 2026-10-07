"""Simple Fault/Health Conformal using validation split for calibration."""
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


def run_fault_conformal():
    raw_dataset = NCMAPSSLoader().load("N-CMAPSS/N-CMAPSS DS01 005.h5")
    raw_dataset.records = [_downsample(r) for r in raw_dataset.records]

    sb = SplitBuilder(raw_dataset, seed=0)
    split = sb.build()
    train_unit_ids = split.train_unit_ids
    val_unit_ids = split.val_unit_ids  # u1, u10 for calibration
    test_unit_ids = split.test_unit_ids  # u2, u9 locked

    print(f"Training: {train_unit_ids}")
    print(f"Val (calibration): {val_unit_ids}")
    print(f"Test (locked): {test_unit_ids}")

    # 1. Train full pipeline on ALL training engines
    train_records = [r for r in raw_dataset.records if r.unit_id in set(train_unit_ids)]
    stacked = np.concatenate([r.signals for r in train_records], axis=0)
    mean_all = stacked.mean(axis=0)
    std_all = stacked.std(axis=0)
    std_all = np.where(std_all < 1e-8, 1.0, std_all)
    stats = NormalizationStats(mean=mean_all, std=std_all, train_unit_ids=sorted(train_unit_ids))

    prep = Preprocessor(window_length=4, stride=4)
    tr_seqs = [prep.process(raw_dataset[u], norm_stats=stats) for u in train_unit_ids]
    val_seqs = [prep.process(raw_dataset[u], norm_stats=stats) for u in ['DS01_u1', 'DS01_u10']]
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

    def belief_feats(seqs, trainer, fb):
        feats = []
        for s in seqs:
            X = torch.as_tensor(s.X, dtype=torch.float32)
            v = np.concatenate(trainer.event_sequences(X, torch.ones_like(X)))
            r = ForwardBackward(trainer.hsmm).causal_filter(v)
            state_post = r.state_posterior
            exp_dwell = r.expected_dwell / trainer.hsmm.d_max
            trans = state_post @ trainer.hsmm.A
            feats.append(np.concatenate([r.state_posterior, r.expected_dwell / trainer.hsmm.d_max, state_post @ trainer.hsmm.A], axis=1))
        return feats

    # Train beliefs
    tr_beliefs = np.concatenate([belief_feats([s], trainer, ForwardBackward(trainer.hsmm))[0] for s in tr_seqs], axis=0)
    y_fault = np.concatenate([np.array([int(raw_dataset[u].fault_label)] * len(s.X)) for s, u in zip(tr_seqs, train_unit_ids)])

    # Train classifier
    clf = LogisticRegression(multi_class='multinomial', solver='lbfgs', C=1.0, max_iter=1000, random_state=0)
    tr_feats = np.concatenate([np.concatenate([r.state_posterior, r.expected_dwell / trainer.hsmm.d_max, r.state_posterior @ trainer.hsmm.A], axis=1) 
                               for r in [ForwardBackward(trainer.hsmm).causal_filter(np.concatenate(trainer.event_sequences(torch.as_tensor(s.X, dtype=torch.float32), torch.ones_like(s.X))) for s in tr_seqs])], axis=0)
    # Simpler: just use the beliefs we already computed
    tr_feats = np.concatenate([b for b in tr_seqs_beliefs], axis=0) if False else None
    
    # Simpler: recompute
    tr_feats = np.concatenate([belief_feats([s], trainer, ForwardBackward(trainer.hsmm))[0] for s in tr_seqs], axis=0)
    y_fault = np.concatenate([np.array([int(raw_dataset[u].fault_label)] * len(s.X)) for s, u in zip(tr_seqs, train_unit_ids)])

    clf = LogisticRegression(multi_class='multinomial', solver='lbfgs', C=1.0, max_iter=1000, random_state=0)
    clf.fit(np.concatenate(tr_seqs_beliefs, axis=0), y_fault) if False else clf.fit(np.concatenate([belief_feats([s], trainer, ForwardBackward(trainer.hsmm))[0] for s in tr_seqs], axis=0), 
    y_fault = np.concatenate([np.array([int(raw_dataset[u].fault_label)] * len(s.X)) for s, u in zip(tr_seqs, train_unit_ids)])
    clf.fit(np.concatenate([belief_feats([s], trainer, ForwardBackward(trainer.hsmm))[0] for s in tr_seqs], axis=0), y_fault)

    # Calibration on validation engines
    val_seqs = [prep.process(raw_dataset[u], norm_stats=stats) for u in ['DS01_u1', 'DS01_u10']]
    val_feats = np.concatenate([belief_feats([s], trainer, ForwardBackward(trainer.hsmm))[0] for s in val_seqs], axis=0)
    val_labels = np.concatenate([np.array([int(raw_dataset[u].fault_label)] * len(s.X)) for u, s in zip(['DS01_u1', 'DS01_u10'], val_seqs)])

    probs_val = clf.predict_proba(np.concatenate([belief_feats([s], trainer, ForwardBackward(trainer.hsmm))[0] for s in val_seqs], axis=0))
    true_idx = np.array([np.where(clf.classes_ == c)[0][0] for c in np.concatenate([np.array([int(raw_dataset[u].fault_label)] * len(s.X)) for u, s in zip(['DS01_u1', 'DS01_u10'], val_seqs)])])
    probs_true = probs_val[np.arange(len(probs_val)), np.array([np.where(clf.classes_ == c)[0][0] for c in np.concatenate([np.array([int(raw_dataset[u].fault_label)] * len(s.X)) for u, s in zip(['DS01_u1', 'DS01_u10'], val_seqs)])])]
    cal_scores = 1 - probs_true
    cal_scores.sort()
    print(f"Calibration: n={len(cal_scores)}, range=[{cal_scores.min():.3f}, {cal_scores.max():.3f}]")
    for level in [0.5, 0.8, 0.9]:
        q = np.quantile(cal_scores, level, method="higher")
        print(f"  {int(level*100)}% quantile: {q:.3f}")

    # Test on locked test engines
    for test_unit in ['DS01_u2', 'DS01_u9']:
        test_seq = [s for s in test_seqs if s.unit_id == test_unit][0]
        test_feats = belief_feats([test_seq], trainer, ForwardBackward(trainer.hsmm))[0]
        probs = clf.predict_proba(test_feats)
        true_fc = int(raw_dataset[test_unit].fault_label)
        test_labels = np.array([int(raw_dataset[test_unit].fault_label)] * len(test_seqs[0].X))
        pred = clf.predict(test_feats)
        acc = (np.argmax(probs, axis=1) == np.array([int(raw_dataset[test_unit].fault_label)] * len(test_feats))).mean()
        
        probs_test = probs[:, np.where(clf.classes_ == int(raw_dataset[test_unit].fault_label))[0][0]]
        nonconf = 1 - probs[:, np.where(clf.classes_ == int(raw_dataset[test_unit].fault_label))[0][0]]
        
        print(f"\n{test_unit}: Acc={acc:.3f}")
        for level in [0.5, 0.8, 0.9]:
            q = np.quantile(np.array(cal_scores), level, method="higher")
            cov = (1 - probs[:, np.where(clf.classes_ == int(raw_dataset[test_unit].fault_label))[0][0]] <= q).mean()
            print(f"  {int(level*100)}%: q={q:.3f}, cov={cov:.3f}")

def belief_feats(seqs, trainer, fb):
    out = []
    for s in seqs:
        X = torch.as_tensor(s.X, dtype=torch.float32)
        v = np.concatenate(trainer.event_sequences(X, torch.ones_like(X)))
        r = ForwardBackward(trainer.hsmm).causal_filter(v)
        out.append(np.concatenate([r.state_posterior, r.expected_dwell / trainer.hsmm.d_max, r.state_posterior @ trainer.hsmm.A], axis=1))
    return out

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

    run_fault_conformal()
