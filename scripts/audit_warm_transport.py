"""Check old controls, paired budgets, moments and warm transport diagnostics."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('out',type=Path);a=p.parse_args()
folder=a.root/'runs/warm_transport_v31';checked=[];errors=[]
def predictions_equal(left,right,stages):
    for i in stages:
        with np.load(left/f'predictions_stage{i}.npz') as x,np.load(right/f'predictions_stage{i}.npz') as y:
            assert set(x.files)==set(y.files)
            for key in x.files:np.testing.assert_array_equal(x[key],y[key])
for path in sorted(folder.glob('*_warm0_*/summary.json')):
    cold=json.loads(path.read_text());c=cold['config'];warm_path=folder/path.parent.name.replace('_warm0_','_warm1_')
    warm=json.loads((warm_path/'summary.json').read_text())
    old_path=a.root/'runs/task_loss_v26'/f"{c['dataset']}_{c['encoder']}_{c['residual_loss']}_{c['method']}_w0.1_o17_s42"
    old=json.loads((old_path/'summary.json').read_text())
    assert cold['input_hashes']==warm['input_hashes']==old['input_hashes']
    assert cold['order']==warm['order']==old['order']
    assert not c['warm_start'] and warm['config']['warm_start']
    predictions_equal(path.parent,old_path,range(1,len(cold['stages'])+1))
    predictions_equal(path.parent,warm_path,[1])
    for x,y,z in zip(cold['stages'],warm['stages'],old['stages']):
        for field in ('optimizer_steps','historical_training_samples','current_samples'):
            assert x[field]==y[field]==z[field]
        if y['warm_init_error'] is not None:
            assert np.isfinite(y['warm_init_error']) and y['warm_init_error']<1e-3
            errors.append(y['warm_init_error'])
    left=torch.load(path.parent/'final.pt',map_location='cpu',weights_only=False)['learner']
    right=torch.load(warm_path/'final.pt',map_location='cpu',weights_only=False)['learner']
    for key in ('base','mean','sd'):np.testing.assert_array_equal(getattr(left,key),getattr(right,key))
    for x,y in zip(left.memory.risk_moments(left.classes),right.memory.risk_moments(right.classes)):np.testing.assert_equal(x,y)
    if left.buffer is not None:
        for x,y in zip(left.buffer,right.buffer):np.testing.assert_array_equal(x,y)
        np.testing.assert_array_equal(left.keys,right.keys)
    checked.append(path.parent.name)
assert len(checked)==36
a.out.write_text(json.dumps(dict(complete=True,exact_prior_controls=36,matched_pairs=36,max_warm_residual_error=max(errors),checks=['all cold stage predictions reproduce v26','first-stage predictions identical in each pair','equal input hashes, order, updates and replay counts','equal moments, shortcut, normalization and ER buffers','warm function drift recorded on each current training stage'],checked=checked),indent=2))
print(f'36 exact controls and 36 paired audits passed; max residual drift {max(errors):.3g}')
