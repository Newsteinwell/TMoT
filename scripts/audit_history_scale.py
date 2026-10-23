"""Validate history-scale states, budgets and independently computed readouts."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from tmot.moments import ClassificationMoments,RegressionMoments,ridge_readout
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('out',type=Path);a=p.parse_args()
torch.set_num_threads(1);root=a.root;folder=root/'runs/history_scale_v42';cache={};checked=[];failures=[];storage=[];pairs=0;max_mass_error=0.
for path in sorted(folder.glob('*/summary.json')):
    s=json.loads(path.read_text());c=s['config'];m=torch.load(path.parent/'final.pt',map_location='cpu',weights_only=False);learner=m['learner']
    key=c['dataset'],c['method'];control=folder/f"{c['dataset']}_mantis_{c['method']}_reg0_r0.1_o17_s42"
    if key not in cache:
        ck=torch.load(control/'final.pt',map_location='cpu',weights_only=False);ref=ck['learner'];prior=json.loads((control/'summary.json').read_text())
        data=np.load(root/'features/mantis'/f"{c['dataset']}_train.npz");y=data['y'].copy()
        if s['task']=='classification':parts=[np.flatnonzero(np.isin(y,cl)) for cl in s['order']];memory=ClassificationMoments(512)
        else:
            parts=[np.flatnonzero(data['groups']==g) for g in s['order']];y=(y-m['target_mean'])/m['target_sd'];memory=RegressionMoments(512,y.shape[1])
        for ids in parts:memory.update(ck['basis'].transform(data['x'][ids]),y[ids])
        baseline=root/'runs/stable_development_v7'/f"{c['dataset']}_ridge_r0.1_o17_s42"
        baseline_summary=json.loads((baseline/'summary.json').read_text())
        assert baseline_summary['input_hashes']==s['input_hashes'] and baseline_summary['order']==s['order']
        baseline_state=torch.load(baseline/'final.pt',map_location='cpu',weights_only=False)
        np.testing.assert_array_equal(baseline_state['basis'].transform(data['x'][:20]),ck['basis'].transform(data['x'][:20]))
        cache[key]=(ref,prior,memory)
        storage.append(dict(dataset=c['dataset'],method=c['method'],train_samples=len(y),retained_rows=len(ref.buffer[0]) if ref.buffer is not None else 0,buffer_bytes=sum(v.nbytes for v in ref.buffer)+ref.keys.nbytes if ref.buffer is not None else 0,persistent_bytes=ref.persistent_bytes))
    ref,prior,memory=cache[key]
    assert s['input_hashes']==prior['input_hashes'] and s['order']==prior['order']
    for field in ('g','k'):np.testing.assert_array_equal(getattr(learner,field),getattr(ref,field))
    for name,v in learner.function.net.state_dict().items():torch.testing.assert_close(v,ref.function.net.state_dict()[name],rtol=0,atol=0)
    for field in ('mean','sd','hidden_mean','hidden_scale'):np.testing.assert_array_equal(getattr(learner.function,field),getattr(ref.function,field))
    for row,old in zip(s['stages'],prior['stages']):
        for field in ('optimizer_steps','historical_training_samples','representative_mass','completion'):assert row[field]==old[field]
    for actual,expected in zip(learner.exact_memory.risk_moments(learner.classes),memory.risk_moments(learner.classes)):np.testing.assert_allclose(actual,expected,rtol=1e-9,atol=1e-8)
    if learner.buffer is not None:assert sum(v.nbytes for v in learner.buffer)+learner.keys.nbytes<=learner.replay_bytes
    g=learner.g;k=learner.k
    if c['regularized']:
        phi=np.r_[np.arange(512),len(g)-1];hi=np.arange(512,len(g)-1);n=len(phi);aa=g[np.ix_(phi,phi)];u=g[np.ix_(phi,hi)]
        penalty=np.eye(n)*c['ridge'];penalty[-1,-1]=0;chol=np.linalg.cholesky((aa+aa.T)/2+penalty)
        solved=np.linalg.solve(chol.T,np.linalg.solve(chol,np.c_[k[phi],u]));base=solved[:,:k.shape[1]];b=solved[:,k.shape[1]:]
        t=np.zeros_like(g);t[phi,:n]=np.eye(n);t[hi,n:]=np.eye(len(hi));t[phi,n:]=-b;tg=t.T@g@t;tk=t.T@k
        residual=(tg[n:,n:]+tg[n:,n:].T)/2;val,vec=np.linalg.eigh(residual);cut=max(np.max(abs(val))*1e-12,1e-14);residual=(vec*np.where(val>cut,val,0))@vec.T
        correction=np.linalg.solve(residual+c['ridge']*np.eye(len(hi)),tk[n:]-tg[n:,:n]@base);expected=t@np.r_[base,correction]
    else:expected=ridge_readout(g,k,c['ridge'])
    if not np.allclose(learner.readout,expected,rtol=1e-6,atol=2e-7):failures.append(dict(run=path.parent.name,max_weight_difference=float(np.max(abs(learner.readout-expected)))))
    if c['method']=='local':
        erdir=path.parent.with_name(path.parent.name.replace('_local_','_er_'));er=json.loads((erdir/'summary.json').read_text())
        for row,other in zip(s['stages'],er['stages']):
            for field in ('optimizer_steps','historical_training_samples'):assert row[field]==other[field]
            for field in ('current','history','total'):
                error=abs(row['representative_mass'][field]-other['representative_mass'][field]);max_mass_error=max(max_mass_error,error)
                assert error<=1e-12
        with np.load(path.parent/'predictions_stage1.npz') as x,np.load(erdir/'predictions_stage1.npz') as y:
            for field in x.files:np.testing.assert_array_equal(x[field],y[field])
        pairs+=1
    checked.append(path.parent.name)
assert len(checked)==56 and pairs==28
report=dict(coverage_complete=True,complete=not failures,trajectories=56,matched_pairs=pairs,max_representative_mass_error=max_mass_error,mass_tolerance=1e-12,numerical_failures=failures,storage=storage,checked=checked,checks=['exact TRAIN moments independently rebuilt','v7 inputs/order and sampled fixed basis outputs match','functions and moments invariant across ridge/readout','28 paired budgets and exact first-stage predictions','ER buffer byte bound','independent Cholesky residual solve or joint solve'])
a.out.write_text(json.dumps(report,indent=2));print(f'56 states; 28 pairs; {len(failures)} numerical failures')
