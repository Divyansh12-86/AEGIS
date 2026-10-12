from pathlib import Path
import json

for id_ in ['id_00', 'id_02', 'id_04', 'id_06']:
    path = Path('results/mimii') / (f'mimii_fan_{id_}_3seeds.json' if id_.startswith('id_') else f'mimii_fan_id_{id_}_3seeds.json')
    if not path.exists(): path = Path(f'mimii_fan_{id_}_3seeds.json' if id_.startswith('id_') else f'mimii_fan_id_{id_}_3seeds.json')
    with open(path) as f:
        data = json.load(f)
    auroc = data['results']['continuous_hsmm']['AUROC_mean']
    per_seed = data['results']['continuous_hsmm']['AUROC_per_seed']
    print(id_ + ': AUROC=' + str(round(auroc,4)) + ', per_seed=' + str([round(x,4) for x in per_seed]))
