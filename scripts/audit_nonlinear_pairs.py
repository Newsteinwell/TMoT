"""Audit matching of base/basis, actual update counts, IDs and labels in v11."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('out',type=Path);a=p.parse_args()
folder=a.root/'runs'/'nonlinear_matched_v11';pairs=[]
for path in sorted(folder.glob('*_gaussian_*/summary.json')):
    other=path.parent.with_name(path.parent.name.replace('_gaussian_','_er_'))
    g=json.loads(path.read_text());e=json.loads((other/'summary.json').read_text())
    assert g['input_hashes']==e['input_hashes'] and g['order']==e['order']
    for x,y in zip(g['stages'],e['stages'],strict=True):
        assert x['optimizer_steps']==y['optimizer_steps']
        assert x['historical_training_samples']==y['historical_training_samples']
    gc=torch.load(path.parent/'final.pt',map_location='cpu',weights_only=False)
    ec=torch.load(other/'final.pt',map_location='cpu',weights_only=False)
    np.testing.assert_array_equal(gc['learner'].base,ec['learner'].base)
    for key,value in vars(gc['basis']).items():
        np.testing.assert_equal(value,vars(ec['basis'])[key])
    with np.load(a.root/'features'/g['config']['encoder']/f"{g['config']['dataset']}_val.npz") as source:
        ids=source['ids'];truth=source['y'];groups=source['groups']
    lookup={str(v):i for i,v in enumerate(ids)};assert len(lookup)==len(ids)
    for run in (path.parent,other):
        for stage in g['stages']:
            with np.load(run/f"predictions_stage{stage['stage']}.npz") as predictions:
                keys=sorted(int(k.split('_')[1]) for k in predictions.files if k.startswith('ids_'))
                observed=[]
                for key in keys:
                    observed.extend(str(v) for v in predictions[f'ids_{key}'])
                    rows=[lookup[str(v)] for v in predictions[f'ids_{key}']]
                    np.testing.assert_array_equal(predictions[f'y_{key}'],truth[rows])
            seen=g['order'][:stage['stage']]
            mask=np.isin(truth,np.concatenate(seen)) if g['task']=='classification' else np.isin(groups,seen)
            expected=[str(v) for v in ids[mask]]
            assert len(observed)==len(set(observed)) and sorted(observed)==sorted(expected)
    pairs.append(path.parent.name)
assert len(pairs)==24
report=dict(complete=True,pairs=len(pairs),checked=pairs,checks=['identical basis and analytic shortcut','equal per-stage optimizer steps and replay counts','exact source validation IDs and labels','no extra or missing stage evaluation IDs'])
a.out.write_text(json.dumps(report,indent=2));print(json.dumps(report))
