"""Independent empirical input-statistics, control-replay and budget checks for v44."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('out',type=Path);a=p.parse_args();torch.set_num_threads(1)
folder=a.root/'runs/arrived_calibration_v44';terminal=json.loads((folder/'complete.json').read_text());assert len(terminal)==40 and all(r['status']=='completed' for r in terminal)
checked=[]
def predictions_equal(left,right,stages):
    for stage in stages:
        with np.load(left/f'predictions_stage{stage}.npz') as l,np.load(right/f'predictions_stage{stage}.npz') as r:
            assert set(l.files)==set(r.files)
            for key in l.files:np.testing.assert_array_equal(l[key],r[key])
for path in sorted(folder.glob('*_arrived/summary.json')):
    s=json.loads(path.read_text());c=s['config'];first=path.parent.with_name(path.parent.name.removesuffix('_arrived')+'_first')
    old=a.root/'runs/history_robustness_v43'/path.parent.name.removesuffix('_arrived')
    control=json.loads((first/'summary.json').read_text());prior=json.loads((old/'summary.json').read_text())
    assert s['input_hashes']==control['input_hashes']==prior['input_hashes']
    assert s['order']==control['order']==prior['order']
    predictions_equal(first,old,range(1,len(s['stages'])+1));predictions_equal(first,path.parent,[1])
    ck=torch.load(path.parent/'final.pt',map_location='cpu',weights_only=False);ref=torch.load(first/'final.pt',map_location='cpu',weights_only=False)
    m=ck['learner'];fm=ref['learner'];assert m.calibration=='arrived' and fm.calibration=='first'
    with np.load(a.root/'features'/c['encoder']/f"{c['dataset']}_train.npz") as data:
        z=ck['basis'].transform(data['x']).astype(np.float64)
    mean=z.mean(0);sd=z.std(0);sd=np.maximum(sd,max(float(np.median(sd))*.01,1e-4))
    assert m.calibration_n==len(z)
    np.testing.assert_allclose(m.calibration_mean,mean,rtol=1e-10,atol=1e-10)
    np.testing.assert_allclose(m.calibration_m2,((z-mean)**2).sum(0),rtol=1e-9,atol=1e-8)
    np.testing.assert_allclose(m.function.mean,mean.astype(np.float32),rtol=1e-6,atol=1e-6)
    np.testing.assert_allclose(m.function.sd,sd.astype(np.float32),rtol=1e-6,atol=1e-6)
    for left,right in zip(s['stages'],control['stages'],strict=True):
        for key in ('optimizer_steps','historical_training_samples','current_samples','representative_mass'):assert left[key]==right[key]
    for left,right in zip(m.exact_memory.risk_moments(m.classes),fm.exact_memory.risk_moments(fm.classes)):np.testing.assert_array_equal(left,right)
    np.testing.assert_allclose(m.g,m.g.T,rtol=1e-10,atol=1e-10)
    checked.append(dict(run=path.parent.name,control=first.name,prior=old.name,calibration_samples=len(z),extra_stat_bytes=m.calibration_mean.nbytes+m.calibration_m2.nbytes+8))
assert len(checked)==20
report=dict(complete=True,paired_streams=20,exact_v43_controls=20,exact_first_stage_pairs=20,checked=checked)
a.out.write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='checked'}))
