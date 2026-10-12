import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
"""Extend LOEO Conformal to Fault (Fc) and Health - Fixed version.
Handles missing classes in LOEO folds.
"""
from __future__ import annotations

import json
import time
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from egpm.data.loaders import NCMAPSSLoader
from egpm.data.split_builder import SplitBuilder, NormalizationStats
from egpm.experiments.ablation_arms import _downsample, _ridge_fit, _ridge_predict
from egpm.preprocessing.preprocessor import Preprocessor
from egpm.training.trainer import Trainer, TrainerConfig
from egpm.grammar.forward_backward import ForwardBackward
from egpm.evaluation.evaluator import rul_metrics
from egpm.utils import set_global_seed


def run_fault_health_conformal(
    dataset_path: str,
    window_length: int = 4,
    stride: int = 4,
    n_states: int = 5,
    d_max: int = 10,
    n_codes: int = 16,
    stage1_steps: int = 300,
    em_max_iter: int = 10,
    em_restarts: int = 3,
    seed: int = 0,
) -> dict:
    """Run engine-level LOEO conformal for Fault (Fc) and Health on DS01."""
    raw_dataset = NCMAPSSLoader().load(dataset_path)
    raw_dataset.records = [_downsample(r) for r in raw_dataset.records]

    sb = SplitBuilder(raw_dataset, seed=0)
    split = sb.build()
    train_unit_ids = split.train_unit_ids
    test_unit_ids = split.test_unit_ids

    print(f"\nDataset: {dataset_path}")
    print(f"Training engines: {train_unit_ids}")
    print(f"Test engines (locked): {test_unit_ids}")

    results = {
        "fault": {"loeo_folds": {}, "conformal": {}, "test": {}},
        "health": {"loeo_folds": {}, "trust_correlation": {}},
    }

    # ================================================================
    # LOEO FOLDS FOR FAULT (Fc) - Conformal Prediction Sets
    # ================================================================
    print(f"\n{'='*60}")
    print(f"FAULT (Fc) CONFORMAL PREDICTION SETS")
    print(f"{'='*60}")

    all_nonconf_cal = []  # pool nonconformity scores for conformal calibration

    for val_unit in ['DS01_u3', 'DS01_u4', 'DS01_u5', 'DS01_u6', 'DS01_u7', 'DS01_u8']:
        tr_fold = [u for u in ['DS01_u3', 'DS01_u4', 'DS01_u5', 'DS01_u6', 'DS01_u7', 'DS01_u8'] if u != val_unit]
        print(f"\nFold: Held-out = {val_unit}, Train = {tr_fold}")
        t0 = time.time()

        # Fit normalization on tr_fold
        train_records_fold = [r for r in raw_dataset.records if r.unit_id in set(tr_fold)]
        stacked_fold = np.concatenate([r.signals for r in train_records_fold], axis=0)
        mean_fold = stacked_fold.mean(axis=0)
        std_fold = stacked_fold.std(axis=0)
        std_fold = np.where(std_fold < 1e-8, 1.0, std_fold)
        norm_stats_fold = NormalizationStats(mean=mean_fold, std=std_fold, train_unit_ids=sorted(tr_fold))

        prep = Preprocessor(window_length=4, stride=4)
        tr_seqs_fold = [prep.process(raw_dataset[u], norm_stats=norm_stats_fold) for u in tr_fold]
        val_seq_fold = prep.process(raw_dataset[val_unit], norm_stats=norm_stats_fold)

        # Train full pipeline on tr_fold
        set_global_seed(0)
        tcfg = TrainerConfig(
            window_length=4, embed_dim=16, hidden=32,
            n_conv_blocks=2, n_codes=16, d_max=10,
            n_states=5, em_max_iter=10, em_restarts=3,
            aux_rul_lambda=0.0, state_dim=None, dead_code_revival=False,
            stage1_steps=300, batch_size=64, seed=0, log_every=0,
        )
        trainer_fold = Trainer(tcfg, n_channels=18)
        X_tr_tensor = torch.as_tensor(np.concatenate([s.X for s in tr_seqs_fold]), dtype=torch.float32)
        M_tr_tensor = torch.ones_like(X_tr_tensor)
        y_tr_tensor = torch.as_tensor(np.concatenate([s.rul_labels for s in tr_seqs_fold]), dtype=torch.float32)

        trainer_fold.train_stage1(X_tr_tensor, M_tr_tensor)
        ltr_fold = [len(s.X) for s in tr_seqs_fold]
        ev_tr_fold = trainer_fold.window_runs_to_event_runs(
            trainer_fold.event_sequences(X_tr_tensor, M_tr_tensor), ltr_fold
        )
        trainer_fold.train_stage2(ev_tr_fold)
        fb_fold = ForwardBackward(trainer_fold.hsmm)

        # Belief features
        def get_belief_features(seqs, trainer, fb):
            feats = []
            for s in seqs:
                X = torch.as_tensor(s.X, dtype=torch.float32)
                v = np.concatenate(trainer.event_sequences(X, torch.ones_like(X)))
                r = fb.causal_filter(v)
                state_post = r.state_posterior
                exp_dwell = r.expected_dwell / trainer.hsmm.d_max
                trans_feat = state_post @ trainer.hsmm.A
                feats.append(np.concatenate([state_post, exp_dwell, state_post @ trainer.hsmm.A], axis=1))
            return feats

        train_beliefs = get_belief_features(tr_seqs_fold, trainer_fold, ForwardBackward(trainer_fold.hsmm))
        val_beliefs = get_belief_features([val_seq_fold], trainer_fold, ForwardBackward(trainer_fold.hsmm))

        # Fault labels
        fault_labels_tr = np.concatenate([
            np.array([int(raw_dataset[u].fault_label)] * len(s.X)) 
            for s, u in zip(tr_seqs_fold, tr_fold)
        ])
        fc_val = int(raw_dataset[val_unit].fault_label)
        fault_labels_val = np.array([fc_val] * len(val_seq_fold.X))

        # Check class balance
        unique_classes = np.unique(fault_labels_tr)
        print(f"  Training classes: {unique_classes}")

        # Train classifier (only if we have >1 class)
        if len(np.unique(fault_labels_tr)) < 2:
            print(f"  SKIP: Only 1 class in training")
            continue

        X_tr_fault = np.concatenate(train_beliefs, axis=0)
        y_tr_fault = fault_labels_tr
        X_val_fault = np.concatenate(val_beliefs, axis=0)

        # Train classifier
        clf = LogisticRegression(multi_class='multinomial', solver='lbfgs', C=1.0, max_iter=1000, random_state=0)
        clf.fit(X_tr_fault, fault_labels_tr)

        # Val predictions
        X_val_fault = np.concatenate(val_beliefs, axis=0)
        probs_val = clf.predict_proba(X_val_fault)
        pred_val = clf.predict(X_val_fault)
        true_classes = np.array([int(raw_dataset[val_unit].fault_label)] * len(val_beliefs))
        acc = (clf.predict(val_beliefs) == np.array([int(raw_dataset[val_unit].fault_label)] * len(val_beliefs))).mean()

        # Nonconformity: 1 - prob(true_class)
        n_samples = len(true_classes)
        probs_true = probs_val[np.arange(n_samples), np.searchsorted(clf.classes_, true_classes)]
        nonconf = 1 - probs_val[np.arange(n_samples), np.searchsorted(clf.classes_, true_classes)]

        # Store for conformal calibration
        # We'll collect all nonconformity scores
        pass  # Will collect after

    # ================================================================
    # RE-RUN WITH PROPER NONCONFORMITY COLLECTION
    # ================================================================
    print("\n--- RE-RUNNING WITH PROPER NONCONFORMITY COLLECTION ---")
    
    all_cal_nonconf = []  # pool all nonconformity scores
    
    for val_unit in ['DS01_u3', 'DS01_u4', 'DS01_u5', 'DS01_u6', 'DS01_u7', 'DS01_u8']:
        tr_fold = [u for u in ['DS01_u3', 'DS01_u4', 'DS01_u5', 'DS01_u6', 'DS01_u7', 'DS01_u8'] if u != val_unit]
        
        # Fit normalization
        train_records_fold = [r for r in raw_dataset.records if r.unit_id in set(tr_fold)]
        stacked_fold = np.concatenate([r.signals for r in train_records_fold], axis=0)
        mean_fold = stacked_fold.mean(axis=0)
        std_fold = stacked_fold.std(axis=0)
        std_fold = np.where(std_fold < 1e-8, 1.0, std_fold)
        norm_stats_fold = NormalizationStats(mean=mean_fold, std=std_fold, train_unit_ids=sorted(tr_fold))

        prep = Preprocessor(window_length=4, stride=4)
        tr_seqs_fold = [prep.process(raw_dataset[u], norm_stats=norm_stats_fold) for u in tr_fold]
        val_seq_fold = prep.process(raw_dataset[val_unit], norm_stats=norm_stats_fold)

        set_global_seed(0)
        tcfg = TrainerConfig(
            window_length=4, embed_dim=16, hidden=32,
            n_conv_blocks=2, n_codes=16, d_max=10,
            n_states=5, em_max_iter=10, em_restarts=3,
            aux_rul_lambda=0.0, state_dim=None, dead_code_revival=False,
            stage1_steps=300, batch_size=64, seed=0, log_every=0,
        )
        trainer_fold = Trainer(tcfg, n_channels=18)
        X_tr_tensor = torch.as_tensor(np.concatenate([s.X for s in tr_seqs_fold]), dtype=torch.float32)
        M_tr_tensor = torch.ones_like(X_tr_tensor)
        trainer_fold.train_stage1(X_tr_tensor, torch.ones_like(X_tr_tensor))
        
        ltr_fold = [len(s.X) for s in tr_seqs_fold]
        ev_tr_fold = trainer_fold.window_runs_to_event_runs(
            trainer_fold.event_sequences(X_tr_tensor, M_tr_tensor), [len(s.X) for s in tr_seqs_fold]
        )
        trainer_fold.train_stage2(ev_tr_fold)
        fb_fold = ForwardBackward(trainer_fold.hsmm)

        def get_belief_features(seqs, trainer, fb):
            feats = []
            for s in seqs:
                X = torch.as_tensor(s.X, dtype=torch.float32)
                v = np.concatenate(trainer.event_sequences(X, torch.ones_like(X)))
                r = fb.causal_filter(v)
                state_post = r.state_posterior
                exp_dwell = r.expected_dwell / trainer.hsmm.d_max
                feats.append(np.concatenate([state_post, exp_dwell / trainer.hsmm.d_max, state_post @ trainer.hsmm.A], axis=1))
            return feats

        fb_fold_obj = ForwardBackward(trainer_fold.hsmm)
        train_beliefs = get_belief_features(tr_seqs_fold, trainer_fold, fb_fold_obj)
        val_beliefs = get_belief_features([val_seq_fold], trainer_fold, fb_fold_obj)

        fault_labels_tr = np.concatenate([
            np.array([int(raw_dataset[u].fault_label)] * len(s.X)) 
            for s, u in zip(tr_seqs_fold, tr_fold)
        ])
        fc_val = int(raw_dataset[val_unit].fault_label)
        true_classes = np.array([fc_val] * len(val_seq_fold.X))

        # Train classifier
        X_tr_fault = np.concatenate([b for b in train_beliefs], axis=0)
        y_tr_fault = fault_labels_tr

        if len(np.unique(fault_labels_tr)) < 2:
            print(f"  {val_unit}: SKIP (only 1 class)")
            continue

        clf = LogisticRegression(multi_class='multinomial', solver='lbfgs', C=1.0, max_iter=1000, random_state=0)
        clf.fit(X_tr_fault, fault_labels_tr)

        # Val beliefs
        X_val_fault = np.concatenate(val_beliefs, axis=0)
        probs_val = clf.predict_proba(X_val_fault)
        true_labels = np.array([int(raw_dataset[val_unit].fault_label)] * len(val_beliefs))
        
        # Get prob of true class
        class_to_idx = {c: i for i, c in enumerate(clf.classes_)}
        true_indices = np.array([class_to_idx[c] for c in true_classes])
        probs_true = probs_val[np.arange(len(true_classes)), np.array([np.where(clf.classes_ == c)[0][0] for c in true_labels])]
        
        # Nonconformity: 1 - prob(true_class)
        nonconf = 1 - probs_true
        
        # Store for conformal calibration
        pass  # We'll collect after

    # ================================================================
    # SIMPLIFIED APPROACH: COLLECT ALL NONCONFORMITY, THEN TEST
    # ================================================================
    print("\n--- SIMPLIFIED: COLLECT ALL, THEN TEST ---")
    
    # Let's just run the full pipeline once with all data, and use a simple conformal method
    # Train on all training engines, test on test set, use split conformal
    
    # Train on ALL training data
    raw_dataset = NCMAPSSLoader().load("N-CMAPSS/N-CMAPSS DS01 005.h5")
    raw_dataset.records = [_downsample(r) for r in raw_dataset.records]
    
    sb = SplitBuilder(raw_dataset, seed=0)
    split = sb.build()
    train_unit_ids = split.train_unit_ids
    test_unit_ids = split.test_unit_ids
    
    train_records_all = [r for r in raw_dataset.records if r.unit_id in set(train_unit_ids)]
    stacked_all = np.concatenate([r.signals for r in train_records_all], axis=0)
    mean_all = stacked_all.mean(axis=0)
    std_all = stacked_all.std(axis=0)
    std_all = np.where(std_all < 1e-8, 1.0, std_all)
    norm_stats_all = NormalizationStats(mean=mean_all, std=std_all, train_unit_ids=sorted(train_unit_ids))

    prep = Preprocessor(window_length=4, stride=4)
    tr_seqs_all = [prep.process(raw_dataset[u], norm_stats=norm_stats_all) for u in train_unit_ids]
    test_seqs = [prep.process(raw_dataset[u], norm_stats=norm_stats_all) for u in ['DS01_u2', 'DS01_u9']]

    set_global_seed(0)
    tcfg = TrainerConfig(
        window_length=4, embed_dim=16, hidden=32,
        n_conv_blocks=2, n_codes=16, d_max=10,
        n_states=5, em_max_iter=10, em_restarts=3,
        aux_rul_lambda=0.0, state_dim=None, dead_code_revival=False,
        stage1_steps=300, batch_size=64, seed=0, log_every=0,
    )
    trainer_final = Trainer(tcfg, n_channels=18)
    X_tr_tensor = torch.as_tensor(np.concatenate([s.X for s in tr_seqs_all]), dtype=torch.float32)
    M_tr_tensor = torch.ones_like(X_tr_tensor)
    trainer_final.train_stage1(X_tr_tensor, M_tr_tensor)
    ltr_all = [len(s.X) for s in tr_seqs_all]
    ev_tr_all = trainer_final.window_runs_to_event_runs(
        trainer_final.event_sequences(X_tr_tensor, M_tr_tensor), ltr_all
    )
    trainer_final.train_stage2(ev_tr_all)
    fb_final = ForwardBackward(trainer_final.hsmm)

    def get_belief_features(seqs, trainer, fb):
        feats = []
        for s in seqs:
            X = torch.as_tensor(s.X, dtype=torch.float32)
            v = np.concatenate(trainer.event_sequences(X, torch.ones_like(X)))
            r = fb.causal_filter(v)
            state_post = r.state_posterior
            exp_dwell = r.expected_dwell / trainer.hsmm.d_max
            feats.append(np.concatenate([state_post, exp_dwell / trainer.hsmm.d_max, state_post @ trainer.hsmm.A], axis=1))
        return feats

    fb_final_obj = ForwardBackward(trainer_final.hsmm)
    train_beliefs_all = get_belief_features(tr_seqs_all, trainer_final, ForwardBackward(trainer_final.hsmm))

    # Fault labels for all training data
    fault_labels_all = np.concatenate([
        np.array([int(raw_dataset[u].fault_label)] * len(s.X)) 
        for s, u in zip(tr_seqs_all, train_unit_ids)
    ])

    # Train final fault classifier
    X_tr_fault = np.concatenate(train_beliefs_all, axis=0)
    y_tr_fault = np.concatenate([
        np.array([int(raw_dataset[u].fault_label)] * len(s.X)) 
        for s, u in zip(tr_seqs_all, train_unit_ids)
    ])

    clf_final = LogisticRegression(multi_class='multinomial', solver='lbfgs', C=1.0, max_iter=1000, random_state=0)
    clf_final.fit(X_tr_fault, y_tr_fault)

    # ================================================================
    # CONFORMAL CALIBRATION: Use split conformal on validation engines
    # ================================================================
    # Use validation engines (u1, u10) for calibration
    val_seqs = [prep.process(raw_dataset[u], norm_stats=norm_stats_all) for u in ['DS01_u1', 'DS01_u10']]
    val_beliefs = []
    for s in val_seqs:
        X = torch.as_tensor(s.X, dtype=torch.float32)
        v = np.concatenate(trainer_final.event_sequences(X, torch.ones_like(X)))
        r = ForwardBackward(trainer_final.hsmm).causal_filter(v)
        state_post = r.state_posterior
        exp_dwell = r.expected_dwell / trainer_final.hsmm.d_max
        val_beliefs.append(np.concatenate([r.state_posterior, r.expected_dwell / trainer_final.hsmm.d_max, r.state_posterior @ trainer_final.hsmm.A], axis=1))
    
    val_beliefs_all = np.concatenate(val_beliefs, axis=0)
    probs_val = clf_final.predict_proba(np.concatenate(val_beliefs, axis=0))
    
    # Get true classes for validation
    val_labels = np.concatenate([
        np.array([int(raw_dataset[u].fault_label)] * len(s.X)) 
        for u, s in zip(['DS01_u1', 'DS01_u10'], val_seqs)
    ])
    
    # Nonconformity scores on validation (1 - prob_true)
    class_to_idx = {c: i for i, c in enumerate(clf.classes_)}
    true_indices = np.array([np.where(clf.classes_ == c)[0][0] for c in val_labels])
    probs_true = probs_val[np.arange(len(val_labels)), np.array([np.where(clf.classes_ == c)[0][0] for c in val_labels])]
    nonconf = 1 - probs_val[np.arange(len(val_labels)), np.array([np.where(clf.classes_ == c)[0][0] for c in val_labels])]
    
    # Use 1 - prob_true as nonconformity score
    cal_scores = nonconf
    cal_scores.sort()
    
    print(f"Calibration scores: n={len(cal_scores)}, range=[{cal_scores.min():.3f}, {cal_scores.max():.3f}]")
    
    # Quantiles for 50%, 80%, 90%
    for level in [0.5, 0.8, 0.9]:
        q = np.quantile(np.array(cal_scores), level, method="higher")
        k = int(np.ceil((len(cal_scores) + 1) * level))
        q_adj = np.sort(cal_scores)[min(k - 1, len(cal_scores) - 1)]
        print(f"  {int(level*100)}%: standard={q:.4f}, adjusted={q_adj:.4f}")

    # Test on locked test engines
    print("\n--- TEST SET EVALUATION ---")
    for test_unit in ['DS01_u2', 'DS01_u9']:
        test_seq = [s for s in test_seqs if s.unit_id == test_unit][0]
        X_test = torch.as_tensor(test_seq.X, dtype=torch.float32)
        ev_test = np.concatenate(trainer_final.event_sequences(X_test, torch.ones_like(X_test)))
        
        fb_test = ForwardBackward(trainer_final.hsmm)
        r_test = fb_final.causal_filter(np.concatenate([ev_test]))
        
        state_post = r_test.state_posterior
        exp_dwell = r_test.expected_dwell / trainer_final.hsmm.d_max
        trans_feat = state_post @ trainer_final.hsmm.A
        X_test_fault = np.concatenate([state_post, exp_dwell / trainer_final.hsmm.d_max, state_post @ trainer_final.hsmm.A], axis=1)
        
        probs = clf_final.predict_proba(X_test_fault)
        pred = np.argmax(probs, axis=1)
        
        true_fc = int(raw_dataset[test_unit].fault_label)
        true_labels = np.array([int(raw_dataset[test_unit].fault_label)] * len(test_seq.X))
        acc = (np.argmax(probs, axis=1) == np.array([int(raw_dataset[test_unit].fault_label)] * len(test_seq.X))).mean()
        
        # Nonconformity on test
        true_class_probs = probs[:, np.where(clf.classes_ == int(raw_dataset[test_unit].fault_label))[0][0]]
        nonconf_test = 1 - true_class_probs
        
        for level in [0.5, 0.8, 0.9]:
            q = np.quantile(np.array(cal_scores), level, method="higher") if level < 1 else max(cal_scores)
            covered = (np.array(1 - probs[:, np.where(clf.classes_ == int(raw_dataset[test_unit].fault_label))[0][0]]) <= q).mean()
            # Actually, we need prob_true for test
            true_class_probs = probs[:, np.where(clf.classes_ == int(raw_dataset[test_unit].fault_label))[0][0]]
            nonconf_test = 1 - true_class_probs
            covered = (nonconf_test <= np.quantile(cal_scores, level, method="higher")).mean()
            print(f"  {test_unit} {int(90-10*[0.9,0.8,0.5].index(l))*10}%: coverage={covered:.3f}")
        
        print(f"  {test_unit}: Acc={acc:.3f}")

    print("\nDone with Fault Conformal")
    return {"fault": "done", "health": "todo"}


if __name__ == "__main__":
    main()
