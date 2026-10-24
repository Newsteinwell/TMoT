"""All-state audit for frozen history-scale multi-seed, multi-order runs."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from tmot.moments import ClassificationMoments,RegressionMoments
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('out',type=Path);a=p.parse_args()
torch.set_num_threads(1);folder=a.root/'runs/history_robustness_v43';selection=json.loads((folder/'manifest.json').read_text())['selection'];cache={};checked=[];failures=[];pairs=0;controls=0;max_mass_error=0.;max_error=0.
for path in sorted(folder.glob('*/summary.json')):
    s=json.loads(path.read_text());c=s['config'];assert c['regularized'];assert c['ridge']==selection['selected'][c['dataset']][c['method']]['ridge']
    ck=torch.load(path.parent/'final.pt',map_location='cpu',weights_only=False);m=ck['learner'];key=c['dataset'],c['order']
    if key not in cache:
        with np.load(a.root/'features/mantis'/f"{c['dataset']}_train.npz") as data:
            x=data['x'];y=data['y'].copy()
            if s['task']=='classification':parts=[np.flatnonzero(np.isin(y,cl)) for cl in s['order']];memory=ClassificationMoments(512)
            else:parts=[np.flatnonzero(data['groups']==g) for g in s['order']];y=(y-ck['target_mean'])/ck['target_sd'];memory=RegressionMoments(512,y.shape[1])
            for ids in parts:memory.update(ck['basis'].transform(x[ids]),y[ids])
            cache[key]=(memory,s['input_hashes'],s['order'],x[:20].copy(),ck['basis'].transform(x[:20]))
    memory,hashes,order,raw,basis_sample=cache[key];assert s['input_hashes']==hashes and s['order']==order
    np.testing.assert_array_equal(ck['basis'].transform(raw),basis_sample)
    for actual,expected in zip(m.exact_memory.risk_moments(m.classes),memory.risk_moments(m.classes)):np.testing.assert_allclose(actual,expected,rtol=1e-9,atol=1e-8)
    if m.buffer is not None:assert sum(v.nbytes for v in m.buffer)+m.keys.nbytes<=m.replay_bytes
    g=m.g;k=m.k;phi=np.r_[np.arange(512),len(g)-1];hi=np.arange(512,len(g)-1);n=len(phi);aa=g[np.ix_(phi,phi)];u=g[np.ix_(phi,hi)]
    penalty=np.eye(n)*c['ridge'];penalty[-1,-1]=0;chol=np.linalg.cholesky((aa+aa.T)/2+penalty)
    solved=np.linalg.solve(chol.T,np.linalg.solve(chol,np.c_[k[phi],u]));base=solved[:,:k.shape[1]];b=solved[:,k.shape[1]:]
    t=np.zeros_like(g);t[phi,:n]=np.eye(n);t[hi,n:]=np.eye(len(hi));t[phi,n:]=-b;tg=t.T@g@t;tk=t.T@k
    residual=(tg[n:,n:]+tg[n:,n:].T)/2;val,vec=np.linalg.eigh(residual);cut=max(np.max(abs(val))*1e-12,1e-14);residual=(vec*np.where(val>cut,val,0))@vec.T
    correction=np.linalg.solve(residual+c['ridge']*np.eye(len(hi)),tk[n:]-tg[n:,:n]@base);expected=t@np.r_[base,correction]
    error=float(np.max(abs(m.readout-expected)));max_error=max(max_error,error)
    if not np.allclose(m.readout,expected,rtol=1e-6,atol=2e-7):failures.append(dict(run=path.parent.name,max_weight_difference=error))
    if c['seed']==42 and c['order']==17:
        olddir=a.root/'runs/history_scale_v42'/selection['selected'][c['dataset']][c['method']]['selected_run']
        prior=json.loads((olddir/'summary.json').read_text());assert prior['input_hashes']==s['input_hashes'] and prior['order']==s['order']
        for stage in range(1,len(s['stages'])+1):
            with np.load(path.parent/f'predictions_stage{stage}.npz') as x,np.load(olddir/f'predictions_stage{stage}.npz') as y:
                for field in x.files:np.testing.assert_array_equal(x[field],y[field])
        controls+=1
    if c['method']=='local':
        ridge=selection['selected'][c['dataset']]['er']['ridge'];erdir=folder/f"{c['dataset']}_mantis_er_r{ridge:g}_o{c['order']}_s{c['seed']}";er=json.loads((erdir/'summary.json').read_text())
        assert er['input_hashes']==s['input_hashes'] and er['order']==s['order']
        for row,other in zip(s['stages'],er['stages']):
            for field in ('optimizer_steps','historical_training_samples'):assert row[field]==other[field]
            for field in ('current','history','total'):
                difference=abs(row['representative_mass'][field]-other['representative_mass'][field]);max_mass_error=max(max_mass_error,difference);assert difference<=1e-12
        pairs+=1
    checked.append(path.parent.name)
assert len(checked)==100 and controls==8 and pairs==50
report=dict(coverage_complete=True,complete=not failures,trajectories=100,matched_pairs=pairs,exact_controls=controls,numerical_failures=failures,max_readout_error=max_error,max_mass_error=max_mass_error,checked=checked,checks=['frozen per-dataset per-method settings','TRAIN original moments independently reconstructed for all ten streams','fixed basis invariant across method and seed','all final readouts independently reconstructed','50 paired update/sample budgets','8 original trajectories exactly reproduced','ER byte bound'])
a.out.write_text(json.dumps(report,indent=2));print(f'100 states, 50 pairs, 8 exact controls; {len(failures)} numerical failures')
