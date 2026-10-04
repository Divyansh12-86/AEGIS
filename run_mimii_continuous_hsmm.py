import sys
sys.path.insert(0, '/Users/atharvamendhulkar/Desktop/AEGIS-1')
from egpm.experiments.mimii_run import run_mimii_fan_one
from egpm.data import MIMIILoader

results = {}
for id_ in ['id_00', 'id_02', 'id_04', 'id_06']:
    ds = MIMIILoader(machines=('fan',), feature='logmel', append_deltas=True, max_clips_per_label=200).load(f'fan/{id_}')
    result = run_mimii_fan_one(id_, ds, n_seeds=3, window_length=16, stage1_steps=30, n_states=3, d_max=12, n_codes=8, ae_steps=30, a1_em_iter=3, verbose=True)
    results[id_] = result['results']['continuous_hsmm']['AUROC_mean']
    print(f'{id_}: continuous_hsmm AUROC = {result["results"]["continuous_hsmm"]["AUROC_mean"]:.4f} (per_seed: {result["results"]["continuous_hsmm"]["AUROC_per_seed"]})')

print("\nSummary:")
for id_, auroc in results.items():
    print(f"  {id_}: {auroc:.4f}")
macro = sum(results.values()) / len(results)
print(f"Macro AUROC: {macro:.4f}")