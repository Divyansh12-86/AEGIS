import sys; sys.path.insert(0, "/Users/atharvamendhulkar/Desktop/AEGIS-1")
"""Resumable MIMII fan driver: per machine id, run
  1) mimii_fan_one (5 arms x 3 seeds) -> mimii_fan_<id>_3seeds.json
  2) interpretability suite -> mimii_fan_<id>_interpretability.json
Skips what already exists. Writes per-id artifacts immediately.
"""
import json, time, gc
from pathlib import Path
import numpy as np, torch
import os; os.chdir("/Users/atharvamendhulkar/Desktop/AEGIS-1")

torch.set_num_threads(4)

from egpm.data import MIMIILoader
from egpm.experiments.mimii_run import run_mimii_fan_one
from egpm.interpretability.runner import run_interpretability_suite
from egpm.interpretability.runner import InterpretabilityReport

print("driver start", flush=True)
for id_ in ("id_00", "id_02", "id_04", "id_06"):
    t1 = time.time()
    print(f"[driver] load {id_} …", flush=True)
    ds = MIMIILoader(machines=("fan",), feature="logmel", append_deltas=True).load(f"fan/{id_}")

    out = Path(f"mimii_fan_{id_}_3seeds.json")
    if not out.exists():
        rep = run_mimii_fan_one(id_, ds, n_seeds=3, window_length=16,
                                stage1_steps=300, n_states=3, d_max=12,
                                n_codes=8, ae_steps=200, verbose=True)
        out.write_text(json.dumps(rep, indent=2, default=str))
        print(f"done {id_} -> {out} ({(time.time()-t1)/3600:.2f} h)", flush=True)

    suite_out = Path(f"mimii_fan_{id_}_interpretability.json")
    if not suite_out.exists():
        suite = run_interpretability_suite(ds, window_length=16, stage1_steps=200,
                                           n_states=3, d_max=12, n_codes=8, seed=0, seed_b=1)
        obj = suite.__dict__ if not isinstance(suite, dict) else suite
        suite_out.write_text(json.dumps(obj, indent=2, default=str))
        print(f"done interp {id_} -> {suite_out}", flush=True)
    del ds
    gc.collect()
print("ALL DONE", flush=True)
