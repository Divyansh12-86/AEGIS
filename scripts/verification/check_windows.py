#!/usr/bin/env python
import numpy as np
from egpm.data import NCMAPSSLoader, SplitBuilder
from egpm.experiments.ablation_arms import _downsample
from egpm.preprocessing import Preprocessor
from egpm.data.split_builder import NormalizationStats

ds = NCMAPSSLoader().load('N-CMAPSS/N-CMAPSS DS01 005.h5')
ds.records = [_downsample(r) for r in ds.records]

sb = SplitBuilder(ds, seed=0)
split = sb.build()

print('Val units:', split.val_unit_ids)
print('Test units:', split.test_unit_ids)

from egpm.preprocessing import Preprocessor
from egpm.data.split_builder import NormalizationStats
import numpy as np

prep = Preprocessor(window_length=4)
stacked = np.concatenate([ds[u].signals for u in ['DS01_u3','DS01_u4','DS01_u5','DS01_u6','DS01_u7','DS01_u8']], axis=0)
mean = stacked.mean(axis=0)
std = stacked.std(axis=0)
std = np.where(std < 1e-8, 1.0, std)
from egpm.data.split_builder import NormalizationStats
stats = NormalizationStats(mean=mean, std=std, train_unit_ids=['DS01_u3','DS01_u4','DS01_u5','DS01_u6','DS01_u7','DS01_u8'])

prep = Preprocessor(window_length=4)
for u in ['DS01_u1', 'DS01_u10']:
    seq = prep.process(ds[u], norm_stats=stats)
    print(f'{u}: X shape = {seq.X.shape}')

PYEOF