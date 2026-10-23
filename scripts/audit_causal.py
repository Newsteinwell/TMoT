"""Validate raw alignment, first-stage calibration and independent ridge controls."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('out',type=Path);a=p.parse_args()
checked=[];controls=0
for path in sorted((a.root/'runs/causal_development_v19_retry').glob('*/summary.json')):
    s=json.loads(path.read_text());c=s['config'];name=c['dataset'];enc=c['encoder']
    data={split:np.load(a.root/'prepared'/f'{name}_{split}.npz') for split in ('train','val')}
    for split in data:
        assert hashlib.sha256((a.root/'prepared'/f'{name}_{split}.npz').read_bytes()).hexdigest()==s['raw_input_hashes'][split]
    model=torch.load(path.parent/'final.pt',map_location='cpu',weights_only=False)
    first=data['train']['groups']==min(data['train']['groups']);past=data['train']['x'][first,0,-1,None]
    np.testing.assert_array_equal(model['reference'].mean,past.mean(0))
    np.testing.assert_array_equal(model['target_sd'],np.maximum(data['train']['y'][first].std(0),1e-6))
    for stage in s['stages']:
        with np.load(path.parent/f"predictions_stage{stage['stage']}.npz") as pred:
            for domain in range(stage['stage']):
                mask=data['val']['groups']==domain
                np.testing.assert_array_equal(pred[f'ids_{domain}'],data['val']['ids'][mask])
                np.testing.assert_array_equal(pred[f'y_{domain}'],data['val']['y'][mask])
    if c['reference']=='none' and c['method']=='ridge':
        old=a.root/'runs/round3_validation_v9'/f"{name}_{enc}_linear_r{c['ridge']:g}_o17_s42"
        for stage in s['stages']:
            file=f"predictions_stage{stage['stage']}.npz"
            with np.load(path.parent/file) as new,np.load(old/file) as ref:
                for key in new.files:np.testing.assert_array_equal(new[key],ref[key])
        controls+=1
    checked.append(path.parent.name)
assert len(checked)==144 and controls==24
a.out.write_text(json.dumps(dict(complete=True,checked=len(checked),independent_ridge_controls=controls,runs=checked),indent=2));print('144 alignment checks, 24 exact independent controls passed')
