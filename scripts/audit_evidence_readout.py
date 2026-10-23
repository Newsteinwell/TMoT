"""Independently reconstruct final evidence posterior and verify fixed-basis state."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from tmot.evidence_readout import evidence_objective
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('out',type=Path);a=p.parse_args();checked=[]
for path in sorted((a.root/'runs/evidence_development_v24').glob('*/summary.json')):
    s=json.loads(path.read_text());c=s['config'];saved=torch.load(path.parent/'final.pt',map_location='cpu',weights_only=False);m=saved['learner']
    # Same moments and input basis as a known independent fixed-ridge trajectory.
    old=a.root/'runs/round3_validation_v9'/f"{c['dataset']}_{c['encoder']}_linear_r0.1_o17_s42/final.pt"
    ref=torch.load(old,map_location='cpu',weights_only=False)
    for key,value in vars(saved['basis']).items():np.testing.assert_equal(value,vars(ref['basis'])[key])
    for x,y in zip(m.memory.risk_moments(m.classes),ref['learner'].memory.risk_moments(ref['learner'].classes),strict=True):np.testing.assert_array_equal(x,y)
    g,k,v=m.memory.risk_moments(m.classes);n=m.diagnostics['observations'];mx=g[:-1,-1];my=k[-1]
    scatter=n*(g[:-1,:-1]-np.outer(mx,mx));scatter=(scatter+scatter.T)/2;cross=n*(k[:-1]-np.outer(mx,my));target=max(float(n*(v-my@my)),0.)
    t=[torch.tensor(x,dtype=torch.float64) for x in (scatter,cross,target,m.log_alpha,m.log_beta)]
    loss,mean=evidence_objective(*t[:3],n-1,*t[3:])
    np.testing.assert_allclose(loss.item(),m.last_loss,rtol=1e-7,atol=1e-6)
    np.testing.assert_allclose(mean.numpy(),m.readout,rtol=1e-6,atol=1e-7)
    for stage in s['stages']:
        e=stage['evidence'];assert e['best_evidence']<=e['initial_evidence']+1e-8
        assert np.isfinite(list(e[k] for k in ('initial_evidence','best_evidence','last_evidence','alpha_min','alpha_max','beta'))).all()
    checked.append(path.parent.name)
assert len(checked)==18
a.out.write_text(json.dumps(dict(complete=True,checked=checked,checks=['18 identical independent bases and sufficient statistics','CPU reconstruction of final objective and posterior','finite per-stage objective and improvement over initialization']),indent=2));print('18 evidence posterior and state checks passed')
