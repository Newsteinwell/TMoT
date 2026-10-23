"""Independently reconstruct final evidence posterior and verify fixed-basis state."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from tmot.evidence_readout import evidence_objective
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('out',type=Path);a=p.parse_args();checked=[];optimization=[]
for path in sorted((a.root/'runs/evidence_refined_v25').glob('*/summary.json')):
    s=json.loads(path.read_text());c=s['config'];saved=torch.load(path.parent/'final.pt',map_location='cpu',weights_only=False);m=saved['learner']
    # Same moments and input basis as a known independent fixed-ridge trajectory.
    old=a.root/'runs/round3_validation_v9'/f"{c['dataset']}_{c['encoder']}_linear_r0.1_o17_s42/final.pt"
    ref=torch.load(old,map_location='cpu',weights_only=False)
    for key,value in vars(saved['basis']).items():np.testing.assert_equal(value,vars(ref['basis'])[key])
    for x,y in zip(m.memory.risk_moments(m.classes),ref['learner'].memory.risk_moments(ref['learner'].classes),strict=True):np.testing.assert_array_equal(x,y)
    g,k,v=m.memory.risk_moments(m.classes);n=m.diagnostics['observations'];mx=g[:-1,-1];my=k[-1]
    scatter=n*(g[:-1,:-1]-np.outer(mx,mx));scatter=(scatter+scatter.T)/2;cross=n*(k[:-1]-np.outer(mx,my));target=max(float(n*(v-my@my)),0.)
    t=[torch.tensor(x,dtype=torch.float64) for x in (scatter,cross,target,m.log_alpha,m.log_beta)]
    t[3].requires_grad_(True);t[4].requires_grad_(True)
    loss,mean=evidence_objective(*t[:3],n-1,*t[3:])
    np.testing.assert_allclose(loss.item(),m.last_loss,rtol=1e-7,atol=1e-6)
    np.testing.assert_allclose(mean.detach().numpy(),m.readout,rtol=1e-6,atol=1e-7)
    loss.backward()
    optimization.append(dict(run=path.parent.name,normalized_max_gradient=max(float(t[3].grad.abs().max()),float(t[4].grad.abs()))/((n-1)*cross.shape[1]),alpha_min=m.diagnostics['alpha_min'],alpha_max=m.diagnostics['alpha_max'],beta=m.diagnostics['beta']))
    previous=json.loads((a.root/'runs/evidence_development_v24'/path.parent.name/'summary.json').read_text())
    assert previous['input_hashes']==s['input_hashes'] and previous['order']==s['order']
    for stage,old_stage in zip(s['stages'],previous['stages'],strict=True):
        refinement=stage['evidence']['refinement']
        np.testing.assert_allclose(refinement['adam_best_evidence'],old_stage['evidence']['best_evidence'],rtol=1e-8,atol=1e-6)
        assert refinement['refined_evidence']<=refinement['adam_best_evidence']+1e-6
        e=stage['evidence'];assert e['best_evidence']<=e['initial_evidence']+1e-8
        assert np.isfinite(list(e[k] for k in ('initial_evidence','best_evidence','last_evidence','alpha_min','alpha_max','beta'))).all()
    gradient=np.r_[t[3].grad.numpy(),t[4].grad.item()]/((n-1)*cross.shape[1]);theta=np.r_[m.log_alpha,m.log_beta]
    projected=gradient.copy();projected[((theta<=-12+1e-8)&(gradient>0))|((theta>=12-1e-8)&(gradient<0))]=0
    np.testing.assert_allclose(np.abs(projected).max(),m.diagnostics['refinement']['normalized_projected_gradient'],rtol=1e-4,atol=1e-7)
    optimization[-1]['refinement']=m.diagnostics['refinement']
    checked.append(path.parent.name)
assert len(checked)==18
a.out.write_text(json.dumps(dict(complete=True,checked=checked,optimization=optimization,checks=['18 identical independent bases and sufficient statistics','CPU reconstruction of final objective and posterior','finite per-stage objective and improvement over initialization']),indent=2));print('18 evidence posterior and state checks passed')
