import argparse,json
from pathlib import Path
import numpy as np
import torch
from tmot.stable_residual import StableSemanticBasis
from tmot.functional_basis import FunctionalBasis
from tmot.moments import ClassificationMoments,RegressionMoments,ridge_readout
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('out',type=Path);a=p.parse_args()
torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
cache={};checked=[]
for path in sorted((a.root/'runs/functional_v36').glob('*/summary.json')):
    s=json.loads(path.read_text());c=s['config'];ck=torch.load(path.parent/'final.pt',map_location='cpu',weights_only=False)
    f=ck['basis'];m=ck['learner'];key=c['dataset'],c['encoder'],c['function_mode']
    assert f.dimension==s['effective_dimension']==640 and m.buffer is None
    assert all(not p.requires_grad for p in f.net.parameters())
    if key not in cache:
        with np.load(a.root/'features'/c['encoder']/f"{c['dataset']}_train.npz") as d:
            x=d['x'];y=d['y'].copy()
            parts=[np.flatnonzero(np.isin(y,cl)) for cl in s['order']] if s['task']=='classification' else [np.flatnonzero(d['groups']==g) for g in s['order']]
            if s['task']=='regression':y=(y-ck['target_mean'])/ck['target_sd']
            meta=json.loads((a.root/'prepared'/f"{c['dataset']}_manifest.json").read_text())
            base=StableSemanticBasis(2*meta['shapes']['train'][1],512,seed=c['basis_seed']).fit(x[parts[0]])
            fresh=FunctionalBasis(base,s['task'],mode=c['function_mode'],seed=c['seed'],epochs=c['epochs'],device='cuda').fit(x[parts[0]],y[parts[0]])
            z=np.concatenate([fresh.transform(x[ids]) for ids in parts]);target=np.concatenate([y[ids] for ids in parts])
            original=np.concatenate([base.transform(x[ids]) for ids in parts]);np.testing.assert_array_equal(z[:,:512],original)
            direct=ClassificationMoments(640) if s['task']=='classification' else RegressionMoments(640,y.shape[1])
            direct.update(z,target)
            cache[key]=(fresh,direct)
    fresh,direct=cache[key]
    assert f.optimizer_steps==fresh.optimizer_steps
    for k,v in f.net.state_dict().items():torch.testing.assert_close(v,fresh.net.state_dict()[k],rtol=0,atol=0)
    for name in ('mean','sd','hidden_mean','hidden_scale'):np.testing.assert_array_equal(getattr(f,name),getattr(fresh,name))
    g,k,v=m.memory.risk_moments(m.classes);dg,dk,dv=direct.risk_moments(m.classes)
    for left,right in ((g,dg),(k,dk),(v,dv)):np.testing.assert_allclose(left,right,rtol=1e-9,atol=1e-8)
    np.testing.assert_allclose(m.readout,ridge_readout(g,k,c['ridge']),rtol=1e-10,atol=1e-10)
    checked.append(path.parent.name)
assert len(checked)==108 and len(cache)==18
a.out.write_text(json.dumps(dict(complete=True,trajectories=108,first_arrival_reconstructions=18,checks=['first-arrival-only retraining exactly reproduces frozen functions for all ridge values','original 512 coordinates unchanged','640-dimensional cumulative moments equal direct arrived-data moments','analytic readouts reconstructed','no real-row buffer and function parameters frozen'],checked=checked),indent=2))
print('108 function/moment/readout audits passed')
