import sys
sys.path.insert(0, '/Users/atharvamendhulkar/Desktop/AEGIS-1')
from egpm.experiments.mimii_run import run_mimii_fan_one
from egpm.data import MIMIILoader

ds = MIMIILoader(machines=('fan',), feature='logmel', append_deltas=True, max_clips_per_label=200).load('fan/id_00')
result = run_mimii_fan_one('id_00', ds, n_seeds=1, window_length=16, stage1_steps=30, n_states=3, d_max=12, n_codes=8, ae_steps=30, a1_em_iter=3, verbose=True)
print('continuous_hsmm AUROC:', result['results']['continuous_hsmm']['AUROC_mean'])