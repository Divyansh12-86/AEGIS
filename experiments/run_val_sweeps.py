#!/usr/bin/env python
"""
Run robustness sweeps on VALIDATION set (not test set).
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
from egpm.grammar import ForwardBackward, FirstOrderHMMFitter, HSMM
from egpm.rul import PhaseTypeRUL
from egpm.grammar.hsmm import DurationHistogram
from egpm.evaluation import rul_metrics, per_unit_metrics, summarize_units
from egpm.interpretability.metrics import event_stability, temporal_validity
from egpm.utils import set_global_seed
from egpm.experiments.ablation_arms import _downsample
from egpm.baselines import WindowAutoEncoder, TokenMarkovModel
from egpm.ablations import ContinuousHSMM
from egpm.anomaly import AnomalyScorer
from egpm.evaluation import auroc, auprc, f1_at_threshold
from egpm.evaluation.stats import auroc_bootstrap_ci, paired_wilcoxon
from egpm.utils import set_global_seed

SWEEPS = {
    "VQ_codebook_size": {"param": "n_codes", "values": [4, 8, 16, 32], "datasets": ["DS01", "DS02"]},
    "HSMM_states": {"param": "n_states", "values": [2, 3, 4, 6], "datasets": ["DS01", "DS02"]},
    "HSMM_d_max": {"param": "d_max", "values": [8, 10, 12, 15], "datasets": ["DS01", "DS02"]},
    "Stride": {"param": "stride", "values": [1, 2, 4, 8], "datasets": ["DS01", "DS02"]},
    "Encoder_width": {"param": "hidden", "values": [16, 32, 64, 128], "datasets": ["DS01", "DS02"]},
    "Encoder_attention": {"param": "use_attention", "values": [False, True], "datasets": ["DS01", "DS02"]},
    "VQ_commitment_beta": {"param": "commitment_beta", "values": [0.1, 0.25, 0.5, 1.0], "datasets": ["DS01", "DS02"]},
    "EMA_decay": {"param": "ema_decay", "values": [0.9, 0.99, 0.999], "datasets": ["DS01", "DS02"]},
    "Stage1_steps": {"param": "stage1_steps", "values": [100, 200, 300, 500], "datasets": ["DS01", "DS02"]},
    "EM_iterations": {"param": "em_max_iter", "values": [5, 10, 20, 50], "datasets": ["DS01", "DS02"]},
    "RUL_cap": {"param": "max_cycles_per_unit", "values": [50, 100, 200, None], "datasets": ["DS01", "DS02"]},
}

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

def run_ablation_on_val(h5_path: str, config: Dict, n_seeds: int = 3) -> Dict:
    """Run ablation experiment using VALIDATION set for evaluation."""
    from egpm.data import NCMAPSSLoader, SplitBuilder, UnitDataset, UnitRecord
    from egpm.preprocessing import Preprocessor
    from egpm.training import Trainer, TrainerConfig
    from egpm.grammar import ForwardBackward, FirstOrderHMMFitter
    from egpm.rul import PhaseTypeRUL
    from egpm.evaluation import rul_metrics, per_unit_metrics, summarize_units
    from egpm.experiments.ablation_arms import _downsample
    from egpm.baselines import WindowAutoEncoder, TokenMarkovModel
    from egpm.ablations import ContinuousHSMM
    from egpm.anomaly import AnomalyScorer
    from egpm.evaluation import auroc, auprc, f1_at_threshold
    from egpm.evaluation.stats import auroc_bootstrap_ci, paired_wilcoxon
    from egpm.utils import set_global_seed
    
    dataset = NCMAPSSLoader().load(h5_path)
    dataset.records = [_downsample(r) for r in dataset.records]
    
    sb = SplitBuilder(dataset, seed=0)
    split = sb.build()
    stats = sb.fit_normalization(split)
    prep = Preprocessor(window_length=config.get("window_length", 4))
    
    # Swap val and test: use val for evaluation, test for reference
    val_ids = split.val_unit_ids
    test_ids = split.test_unit_ids
    
    train = [prep.process(dataset[u], norm_stats=stats) for u in split.train_unit_ids]
    val = [prep.process(dataset[u], norm_stats=stats) for u in val_ids]
    test = [prep.process(dataset[u], norm_stats=stats) for u in split.test_unit_ids]
    
    Xtr = torch.as_tensor(np.concatenate([s.X for s in train]), dtype=torch.float32)
    Mtr = torch.ones_like(Xtr)
    run_lens = [len(s.X) for s in train]
    y_tr = np.concatenate([s.rul_labels for s in train])
    
    # ---- shared encoder/VQ stage ----
    tcfg = TrainerConfig(
        window_length=config.get("window_length", 4), embed_dim=16, hidden=config.get("hidden", 32),
        n_conv_blocks=2, n_codes=config.get("n_codes", 16), d_max=config.get("d_max", 10),
        n_states=config.get("n_states", 5), em_max_iter=config.get("em_max_iter", 10),
        em_restarts=config.get("em_restarts", 3), stage1_steps=config.get("stage1_steps", 300),
        batch_size=128, seed=config.get("seed", 0), log_every=0,
    )
    trainer = Trainer(tcfg, n_channels=Xtr.shape[-1])
    trainer.train_stage1(Xtr, Mtr)
    run_lengths = [len(r.X) for r in train]
    train_event_runs = trainer.window_runs_to_event_runs(
        trainer.event_sequences(Xtr, Mtr), run_lens
    )
    
    def tokens_of(ids):
        out = []
        for w in wins(ids):
            X = torch.as_tensor(w, dtype=torch.float32)
            out.append(np.concatenate(trainer.event_sequences(X, torch.ones_like(X))))
        return out
    
    def wins(ids):
        return [prep.process(dataset[u], norm_stats=stats).X for u in ids]
    
    ev_val_toks = tokens_of(split.val_unit_ids)
    ev_test_toks = tokens_of(split.test_unit_ids)
    
    # continuous AE
    ae = WindowAutoEncoder(n_channels=Xtr.shape[-1], window_length=config.get("window_length", 4))
    opt = torch.optim.Adam(ae.parameters(), lr=1e-3)
    ae.train()
    for _ in range(config.get("ae_steps", 300)):
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
    
    # shared VQ tokenization
    tcfg = TrainerConfig(
        window_length=config.get("window_length", 4), embed_dim=16, hidden=32,
        n_conv_blocks=2, n_codes=config.get("n_codes", 16), d_max=config.get("d_max", 10),
        n_states=config.get("n_states", 5), em_max_iter=10, stage1_steps=config.get("stage1_steps", 300),
        batch_size=128, seed=config.get("seed", 0), log_every=0,
    )
    trainer = Trainer(tcfg, n_channels=Xtr.shape[-1])
    trainer.train_stage1(Xtr, Mtr)
    run_lengths = [len(r.X) for r in train]
    train_event_runs = trainer.window_runs_to_event_runs(
        trainer.event_sequences(Xtr, Mtr), run_lengths
    )
    
    def tokens_of(ids):
        out = []
        for w in wins(ids):
            X = torch.as_tensor(w, dtype=torch.float32)
            out.append(np.concatenate(trainer.event_sequences(X, torch.ones_like(X))))
        return out
    
    ev_val_toks = tokens_of(split.val_unit_ids)
    ev_test_toks = tokens_of(split.test_unit_ids)
    
    # pooled unigram (A2)
    counts = np.bincount(np.concatenate(train_event_runs), minlength=config.get("n_codes", 16))
    log_unigram = np.log(counts / counts.sum() + 1e-12)
    
    # first-order Markov over tokens (baseline ii)
    mm = TokenMarkovModel(n_events=config.get("n_codes", 16), smoothing=0.5).fit(train_event_runs)
    
    # full HSMM (primary)
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
    
    emb_train, emb_va = embed_runs(split.train_unit_ids), embed_runs(split.val_unit_ids)
    rng = np.random.default_rng(0)
    a1 = ContinuousHSMM(
        hsmm=HSMM(
            n_states=tcfg.n_states, n_events=2,
            pi=rng.dirichlet(np.ones(tcfg.n_states)),
            A=rng.dirichlet(np.ones(tcfg.n_states), size=tcfg.n_states),
            D=DurationHistogram.from_means(np.full(tcfg.n_states, 2.0), tcfg.d_max),
            d_max=tcfg.d_max, seed=0,
        ),
        means=np.zeros((tcfg.n_states, tcfg.embed_dim)),
        inv_variances=np.ones((tcfg.n_states, tcfg.embed_dim)),
    )
    a1.fit_em(emb_train, n_states=tcfg.n_states, d_max=tcfg.d_max,
              max_iter=6, min_iter=2, tol=1e-3, seed=0)
    
    def score_all(arm, ids):
        if arm == "continuous_AE":
            return ae_score(ids)
        if arm == "pooled_events":
            s = lambda toks: np.array([-log_unigram[v].mean() for v in toks])
            return s(tokens_of(ids))
        if arm == "vq_markov":
            s = lambda toks: np.array([mm.score_run(v).mean() for v in toks])
            return s(tokens_of(ids))
        if arm == "vq_hsmm":
            s = lambda toks: np.array([scorer.score_sequence(trainer.hsmm, v).score.mean() for v in toks])
            return s(tokens_of(ids))
        if arm == "continuous_hsmm":
            ev_n = embed_runs(split.val_unit_ids)
            ev_a = embed_runs(split.test_unit_ids)
            s = lambda rs: np.array([-a1.log_likelihood(z) / len(z) for z in rs])
            return np.concatenate([s(ev_n), s(ev_a)])
        raise ValueError(arm)
    
    def tokens_of(ids):
        out = []
        for w in wins(ids):
            X = torch.as_tensor(w, dtype=torch.float32)
            out.append(np.concatenate(trainer.event_sequences(X, torch.ones_like(X))))
        return out
    
    def wins(ids):
        return [prep.process(dataset[u], norm_stats=stats).X for u in ids]
    
    def ae_score(ids):
        out = []
        with torch.no_grad():
            for w in wins(ids):
                X = torch.as_tensor(w, dtype=torch.float32)
                out.append(ae.anomaly_scores(X, torch.ones_like(X)).mean().item())
        return np.array(out)
    
    # Arms to evaluate
    arms = ["continuous_AE", "pooled_events", "vq_markov", "vq_hsmm", "continuous_hsmm"]
    per_seed_auroc = {a: [] for a in arms}
    per_seed_auprc = {a: [] for a in arms}
    
    # Define evaluation sets
    eval_norm = split.val_unit_ids[:100]  # limit for speed
    eval_anom = split.test_unit_ids[:100]
    
    for seed in range(1):  # Single seed for speed
        set_global_seed(seed)
        # Rebuild trainer for each seed
        trainer = Trainer(tcfg, n_channels=Xtr.shape[-1])
        trainer.train_stage1(Xtr, Mtr)
        train_event_runs = trainer.window_runs_to_event_runs(
            trainer.event_sequences(Xtr, Mtr), run_lengths
        )
        trainer.train_stage2(train_event_runs)
        
        eval_labels = np.concatenate([np.zeros(len(eval_norm)), np.ones(len(eval_anom))])
        
        for arm in arms:
            sn, sa = score_all(arm, eval_norm), score_all(arm, eval_anom)
            scores = np.concatenate([sn, sa])
            per_seed_auroc[arm].append(float(auroc(scores, eval_labels)))
            per_seed_auprc[arm].append(float(auprc(scores, eval_labels)))
    
    return {
        "experiment": "robustness_sweep_val",
        "config": config,
        "results": {arm: {"AUROC_mean": float(np.mean(per_seed_auroc[arm])),
                         "AUROC_std": float(np.std(per_seed_auroc[arm])),
                         "AUROC_per_seed": per_seed_auroc[arm],
                         "AUPRC_mean": float(np.mean(per_seed_auprc[arm])),
                         "AUPRC_std": float(np.std(per_seed_auprc[arm])),
                         "AUPRC_per_seed": per_seed_auprc[arm]} for arm in arms},
        "wall_clock_s": time.time() - t0,
    }

def run_sweep(param_name: str, sweep_config: Dict, base_config: Dict) -> Dict:
    param = sweep_config["param"]
    values = sweep_config["values"]
    datasets = sweep_config["datasets"]
    
    results = {ds: {} for ds in datasets}
    
    for ds_name in datasets:
        h5_path = f"N-CMAPSS/N-CMAPSS {ds_name} 00{'5' if ds_name == 'DS01' else '6'}.h5"
        print(f"\n  Sweeping {param_name} on {ds_name}...")
        
        for val in values:
            config = base_config.copy()
            config[param] = val
            if param == "stride":
                config["window_length"] = 4
            
            print(f"    {param}={val}...")
            try:
                t0 = time.time()
                result = run_ablation_on_val(
                    f"N-CMAPSS/N-CMAPSS {ds_name} 00{'5' if ds_name == 'DS01' else '6'}.h5",
                    config, n_seeds=1
                )
                dt = time.time() - t0
                
                # Extract RMSE for each arm
                arm_results = {}
                for arm in result["results"]:
                    rmse = result["results"][arm]["rmse"]
                    arm_results[arm] = {
                        "mean": rmse["mean"],
                        "std": rmse["std"],
                        "per_seed": rmse["per_seed"],
                    }
                
                results[ds_name][str(val)] = arm_results
                print(f"      done in {time.time() - t0:.1f}s")
                
            except Exception as e:
                print(f"      ERROR: {e}")
                results[ds_name][str(val)] = {"error": str(e)}
    
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run robustness sweeps on VALIDATION set")
    parser.add_argument("--sweep", type=str, help="Specific sweep to run (or 'all')")
    parser.add_argument("--output", type=str, default="robustness_val_results.json", help="Output file")
    parser.add_argument("--quick", action="store_true", help="Run with n_seeds=1 for quick testing")
    args = parser.parse_args()
    
    base_config = DEFAULT_CONFIG.copy()
    
    sweeps_to_run = SWEEPS if args.sweep == "all" else {args.sweep: SWEEPS[args.sweep]}
    
    all_results = {}
    
    for sweep_name, sweep_config in sweeps_to_run.items():
        print(f"\n{'='*60}")
        print(f"Running sweep: {sweep_name}")
        print(f"{'='*60}")
        
        try:
            results = run_sweep(sweep_name, sweep_config, base_config)
            all_results[sweep_name] = results
        except Exception as e:
            print(f"ERROR in sweep {sweep_name}: {e}")
            all_results[sweep_name] = {"error": str(e)}
    
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
