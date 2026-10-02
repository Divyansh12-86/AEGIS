#!/usr/bin/env python
"""
Robustness sweep runner for EGPM ablation experiments.
Runs parameter sweeps for the 13 "Not yet evaluated" parameters in ROBUSTNESS_RESULTS.csv
"""
import json
import time
import argparse
from typing import Dict, List, Any
import numpy as np
import torch

from egpm.experiments.ablation_arms import run_ablation_experiment
from egpm.data import NCMAPSSLoader
from egpm.experiments.ablation_arms import _downsample
from egpm.preprocessing import Preprocessor
from egpm.training import Trainer, TrainerConfig
from egpm.grammar import ForwardBackward, SegmentalViterbi
from egpm.rul import PhaseTypeRUL
from egpm.anomaly import AnomalyScorer
from egpm.interpretability.metrics import temporal_validity, event_stability
from egpm.evaluation import auroc, auprc
from egpm.utils import set_global_seed

# Default configuration
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
        "param": "encoder_width",
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

def run_single_config(h5_path: str, config: Dict, n_seeds: int = 3) -> Dict:
    """Run a single configuration and return results."""
    from egpm.experiments.ablation_arms import run_ablation_experiment
    return run_ablation_experiment(h5_path, **config)

def run_sweep(param_name: str, sweep_config: Dict, base_config: Dict) -> Dict:
    """Run a single parameter sweep across all values and datasets."""
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
                config["window_length"] = 4  # keep window length fixed
            
            print(f"    {param}={val}...")
            try:
                t0 = time.time()
                result = run_ablation_experiment(
                    f"N-CMAPSS/N-CMAPSS {ds_name} 00{'5' if ds_name == 'DS01' else '6'}.h5",
                    n_seeds=3,
                    window_length=config.get("window_length", 4),
                    n_states=config.get("n_states", 5),
                    d_max=config.get("d_max", 10),
                    n_codes=config.get("n_codes", 16),
                    stage1_steps=config.get("stage1_steps", 300),
                    em_max_iter=config.get("em_max_iter", 10),
                    em_restarts=config.get("em_restarts", 3),
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
                print(f"      done in {dt:.1f}s")
                
            except Exception as e:
                print(f"      ERROR: {e}")
                results[ds_name][str(val)] = {"error": str(e)}
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Run robustness sweeps")
    parser.add_argument("--sweep", type=str, help="Specific sweep to run (or 'all')")
    parser.add_argument("--output", type=str, default="robustness_results.json", help="Output file")
    parser.add_argument("--quick", action="store_true", help="Run with n_seeds=1 for quick testing")
    args = parser.parse_args()
    
    base_config = DEFAULT_CONFIG.copy()
    if args.quick:
        base_config["n_seeds"] = 1
    
    all_results = {}
    
    sweeps_to_run = SWEEPS if args.sweep == "all" else {args.sweep: SWEEPS[args.sweep]}
    
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