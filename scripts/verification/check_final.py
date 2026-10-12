import json

for id_ in ['id_00', 'id_02', 'id_04', 'id_06']:
    with open('mimii_fan_id_' + id_ + '_3seeds.json') as f:
        data = json.load(f)
    auroc = data['results']['continuous_hsmm']['AUROC_mean']
    per_seed = data['results']['continuous_hsmm']['AUROC_per_seed']
    print(id_ + ': AUROC=' + str(round(auroc,4)) + ', per_seed=' + str([round(x,4) for x in auroc]))