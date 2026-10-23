"""Audit local warm/cold pairs without claiming equality to full Gaussian replay."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('out',type=Path);a=p.parse_args()
folder=a.root/'runs/local_warm_v32';checked=[];errors=[];component_counts=[]
for path in sorted(folder.glob('*_warm0_*/summary.json')):
    cold=json.loads(path.read_text());wp=folder/path.parent.name.replace('_warm0_','_warm1_');warm=json.loads((wp/'summary.json').read_text())
    assert cold['input_hashes']==warm['input_hashes'] and cold['order']==warm['order']
    for x,y in zip(cold['stages'],warm['stages']):
        for key in ('optimizer_steps','historical_training_samples','current_samples'):assert x[key]==y[key]
        if y['warm_init_error'] is not None:
            assert np.isfinite(y['warm_init_error']) and y['warm_init_error']<1e-3
            errors.append(y['warm_init_error'])
    with np.load(path.parent/'predictions_stage1.npz') as x,np.load(wp/'predictions_stage1.npz') as y:
        for key in x.files:np.testing.assert_array_equal(x[key],y[key])
    left=torch.load(path.parent/'final.pt',map_location='cpu',weights_only=False)['learner'];right=torch.load(wp/'final.pt',map_location='cpu',weights_only=False)['learner']
    assert left.buffer is None and right.buffer is None
    for key in ('base','mean','sd'):np.testing.assert_array_equal(getattr(left,key),getattr(right,key))
    for x,y in zip(left.memory.risk_moments(left.classes),right.memory.risk_moments(right.classes)):np.testing.assert_equal(x,y)
    assert left.local.cells.keys()==right.local.cells.keys()
    for label,cells in left.local.cells.items():
        assert len(cells)==len(right.local.cells[label])
        for x,y in zip(cells,right.local.cells[label]):
            assert x.n==y.n
            for key in ('mean','factor','residual_sd'):np.testing.assert_array_equal(getattr(x,key),getattr(y,key))
    assert sum(cell.n for cells in left.local.cells.values() for cell in cells)==left.seen
    component_counts.append(dict(run=path.parent.name,components=sum(len(cells) for cells in left.local.cells.values()),seen=left.seen,local_bytes=left.local.nbytes))
    checked.append(path.parent.name)
assert len(checked)==36
a.out.write_text(json.dumps(dict(complete=True,matched_pairs=36,max_warm_residual_error=max(errors),checks=['first-stage predictions','input/order/update/replay budgets','moments/readout/normalization','all local cell statistics and total counts','no real-row buffer'],component_counts=component_counts,checked=checked),indent=2))
print(f'36 paired local audits passed; maximum residual drift {max(errors):.3g}')
