"""v41: compare with v39 state and independently reconstruct residual-risk solve."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('out',type=Path);a=p.parse_args()
torch.set_num_threads(1);cache={};checked=[];controls=0;failures=[];diagnostics=[]
for path in sorted((a.root/'runs/regularized_innovation_v41').glob('*/summary.json')):
    s=json.loads(path.read_text());c=s['config'];ck=torch.load(path.parent/'final.pt',map_location='cpu',weights_only=False);m=ck['learner']
    key=c['dataset'],c['encoder'],c['method'];olddir=a.root/'runs/exact_base_v39'/f"{key[0]}_{key[1]}_{key[2]}_exact1_r0.1_o17_s42"
    if key not in cache:cache[key]=(torch.load(olddir/'final.pt',map_location='cpu',weights_only=False)['learner'],json.loads((olddir/'summary.json').read_text()))
    old,prior=cache[key]
    assert s['input_hashes']==prior['input_hashes'] and s['order']==prior['order']
    for field in ('g','k'):np.testing.assert_array_equal(getattr(m,field),getattr(old,field))
    for name,v in m.function.net.state_dict().items():torch.testing.assert_close(v,old.function.net.state_dict()[name],rtol=0,atol=0)
    for field in ('mean','sd','hidden_mean','hidden_scale'):np.testing.assert_array_equal(getattr(m.function,field),getattr(old.function,field))
    for row,ref in zip(s['stages'],prior['stages']):
        for field in ('optimizer_steps','historical_training_samples','representative_mass','completion'):assert row[field]==ref[field]
    if c['regularized']:
        phi=np.r_[np.arange(m.dimension),len(m.g)-1];hi=np.arange(m.dimension,len(m.g)-1);n=len(phi)
        aa=m.g[np.ix_(phi,phi)];u=m.g[np.ix_(phi,hi)];penalty=np.eye(n)*c['ridge'];penalty[-1,-1]=0
        # Cholesky solve and explicit transformed joint moments, independent of learner helper.
        chol=np.linalg.cholesky((aa+aa.T)/2+penalty)
        solved=np.linalg.solve(chol.T,np.linalg.solve(chol,np.c_[m.k[phi],u]))
        base=solved[:,:m.k.shape[1]];b=solved[:,m.k.shape[1]:]
        transform=np.zeros_like(m.g);transform[phi,:n]=np.eye(n);transform[hi,n:]=np.eye(len(hi));transform[phi,n:]=-b
        tg=transform.T@m.g@transform;tk=transform.T@m.k
        residual=(tg[n:,n:]+tg[n:,n:].T)/2
        val,vec=np.linalg.eigh(residual);cut=max(np.max(abs(val))*1e-12,1e-14)
        residual=(vec*np.where(val>cut,val,0))@vec.T
        correction=np.linalg.solve(residual+c['ridge']*np.eye(len(hi)),tk[n:]-tg[n:,:n]@base)
        expected=transform@np.r_[base,correction]
        error=float(np.max(np.abs(m.readout-expected)))
        if not np.allclose(m.readout,expected,rtol=1e-6,atol=2e-7):failures.append(dict(run=path.parent.name,max_weight_difference=error))
        diagnostics.append(dict(run=path.parent.name,max_weight_difference=error,**m.innovation_diagnostics))
    else:
        controls+=1;np.testing.assert_array_equal(m.readout,old.readout)
        for stage in range(1,len(s['stages'])+1):
            with np.load(path.parent/f'predictions_stage{stage}.npz') as x,np.load(olddir/f'predictions_stage{stage}.npz') as y:
                for field in x.files:np.testing.assert_array_equal(x[field],y[field])
    checked.append(path.parent.name)
assert len(checked)==126 and controls==18
report=dict(coverage_complete=True,complete=not failures,trajectories=126,exact_controls=18,numerical_failures=failures,numerical_tolerance=dict(rtol=1e-6,atol=2e-7),diagnostics=diagnostics,checked=checked,
            checks=['functions and complete moments unchanged from v39','paired stage budgets unchanged','independent Cholesky and transformed residual-target risk reconstruction','all-stage exact disabled controls'])
a.out.write_text(json.dumps(report,indent=2));print(f'126 trajectories; 18 exact controls; {len(failures)} numerical failures')
