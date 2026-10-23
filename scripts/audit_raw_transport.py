"""Verify fixed controls and independently reconstruct affine readouts."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from tmot.moments import ridge_readout
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('out',type=Path);a=p.parse_args();checked=[];fixed=0;references={}
for path in sorted((a.root/'runs/raw_transport_v29').glob('*/summary.json')):
    s=json.loads(path.read_text());c=s['config'];key=c['dataset'],c['encoder']
    old=a.root/'runs/raw_temporal_v28'/f"{c['dataset']}_{c['encoder']}_d512_ridge_r0.1_o17_s42"
    if key not in references:references[key]=torch.load(old/'final.pt',map_location='cpu',weights_only=False)
    reference=references[key];new=torch.load(path.parent/'final.pt',map_location='cpu',weights_only=False);m=new['learner']
    rg,rk,rv=reference['learner'].memory.risk_moments(reference['learner'].classes);g,k,v=m.memory.risk_moments(m.classes)
    np.testing.assert_array_equal(g,rg);np.testing.assert_array_equal(k,rk);np.testing.assert_equal(v,rv)
    np.testing.assert_array_equal(m.scale[:m.raw_start],np.ones(m.raw_start));np.testing.assert_array_equal(m.shift[:m.raw_start],np.zeros(m.raw_start))
    h=np.eye(m.dimension+1);h[:-1,:-1]=np.diag(m.scale);h[:-1,-1]=m.shift
    expected=ridge_readout(h@g@h.T,h@k,c['ridge']);np.testing.assert_allclose(m.readout,expected,rtol=1e-6,atol=1e-7)
    if c['method']=='fixed':
        old_summary=json.loads((old/'summary.json').read_text());assert s['input_hashes']==old_summary['input_hashes'] and s['order']==old_summary['order']
        for stage in s['stages']:
            filename=f"predictions_stage{stage['stage']}.npz"
            with np.load(path.parent/filename) as x,np.load(old/filename) as y:
                for name in x.files:np.testing.assert_array_equal(x[name],y[name])
        fixed+=1
    checked.append(path.parent.name)
assert len(checked)==147 and fixed==21
a.out.write_text(json.dumps(dict(complete=True,trajectories=len(checked),exact_fixed_controls=fixed,checks=['identical cumulative moments to independent fixed controls','unchanged Mantis coordinates','readout reconstructed using dense homogeneous transform','fixed predictions reproduce all stages'],checked=checked),indent=2));print('147 readouts and 21 exact controls verified')
