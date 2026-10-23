"""Verify unchanged controls and isolate small-cell merging with state audits."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('out',type=Path);a=p.parse_args()
folder=a.root/'runs/local_merge_v34';checked=[];states=[]
def equal_predictions(left,right,stages):
    for i in stages:
        with np.load(left/f'predictions_stage{i}.npz') as x,np.load(right/f'predictions_stage{i}.npz') as y:
            for key in x.files:np.testing.assert_array_equal(x[key],y[key])
for path in sorted(folder.glob('*_min1_*/summary.json')):
    high=json.loads(path.read_text());c=high['config'];lp=folder/path.parent.name.replace('_min1_','_min4_');low=json.loads((lp/'summary.json').read_text())
    old=a.root/'runs/local_support_v33'/path.parent.name.replace('_min1_','_support4_')
    previous=json.loads((old/'summary.json').read_text())
    assert high['input_hashes']==low['input_hashes']==previous['input_hashes'];assert high['order']==low['order']==previous['order']
    equal_predictions(path.parent,old,range(1,len(high['stages'])+1));equal_predictions(path.parent,lp,[1])
    for x,y in zip(high['stages'],low['stages']):
        for key in ('optimizer_steps','historical_training_samples','current_samples'):assert x[key]==y[key]
        if y['warm_init_error'] is not None:assert np.isfinite(y['warm_init_error']) and y['warm_init_error']<1e-3
    models=[]
    for directory in (path.parent,lp):
        m=torch.load(directory/'final.pt',map_location='cpu',weights_only=False)['learner'];models.append(m)
        counts=[cell.n for cells in m.local.cells.values() for cell in cells]
        if directory==lp:assert min(counts)>=4
        assert sum(counts)==m.seen and min(counts)>=1 and m.buffer is None
        states.append(dict(run=directory.name,components=len(counts),min_component_size=min(counts),singleton_components=sum(n==1 for n in counts),local_bytes=m.local.nbytes,counts=counts))
    for key in ('base','mean','sd'):np.testing.assert_array_equal(getattr(models[0],key),getattr(models[1],key))
    for x,y in zip(models[0].memory.risk_moments(models[0].classes),models[1].memory.risk_moments(models[1].classes)):np.testing.assert_equal(x,y)
    checked.append(path.parent.name)
assert len(checked)==36
a.out.write_text(json.dumps(dict(complete=True,exact_prior_controls=36,matched_pairs=36,checks=['all prior control predictions','input/order/update/replay budgets','first-stage predictions','equal aggregate moments and analytic readout','component counts, singleton cells and storage; no real-row buffer'],states=states,checked=checked),indent=2))
print('36 exact controls and 36 merge pairs verified')
