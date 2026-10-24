"""Resolve shared states; verify readouts, moment-matched pairs and old controls."""
import argparse,hashlib,json
from collections import Counter
from pathlib import Path
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('out',type=Path);a=p.parse_args();torch.set_num_threads(1)
folder=a.root/'runs/empirical_replay_v51';terminal=json.loads((folder/'complete.json').read_text());assert len(terminal)==144 and all(r['status']=='completed' for r in terminal)
cache={};checked=[];controls=[];pairs=[];max_error=0.
def load(path):
    s=json.loads(path.read_text());leaf=torch.load(path.parent/'final.pt',map_location='cpu',weights_only=False)
    shared=(path.parent/leaf['shared_state']).resolve();assert shared.parent== (folder/'_shared').resolve()
    assert s['shared_state']==leaf['shared_state'] and s['shared_state_sha256']==leaf['shared_state_sha256']
    if shared not in cache:
        with shared.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
        cache[shared]=(torch.load(shared,map_location='cpu',weights_only=False),digest)
    ck,digest=cache[shared];assert digest==leaf['shared_state_sha256'];return s,leaf,ck

def equal_predictions(left,right,stages):
    for stage in stages:
        with np.load(left/f'predictions_stage{stage}.npz') as x,np.load(right/f'predictions_stage{stage}.npz') as y:
            assert set(x.files)==set(y.files)
            for name in x.files:np.testing.assert_array_equal(x[name],y[name])

for path in sorted(folder.glob('*/summary.json')):
    s,leaf,ck=load(path);c=s['config'];m=ck['learner'];m.readout=leaf['readout'];d=c['dataset'];e=c['encoder'];r=c['ridge']
    assert m.variant==c['variant'] and m.mode==c['method']
    if m.buffer is not None:assert sum(v.nbytes for v in m.buffer)+m.keys.nbytes<=m.replay_bytes
    phi=np.r_[np.arange(512),len(m.g)-1];hi=np.arange(512,len(m.g)-1);n=len(phi)
    aa=m.g[np.ix_(phi,phi)];u=m.g[np.ix_(phi,hi)];penalty=np.eye(n)*r;penalty[-1,-1]=0
    chol=np.linalg.cholesky((aa+aa.T)/2+penalty);solved=np.linalg.solve(chol.T,np.linalg.solve(chol,np.c_[m.k[phi],u]));base=solved[:,:m.k.shape[1]];b=solved[:,m.k.shape[1]:]
    transform=np.zeros_like(m.g);transform[phi,:n]=np.eye(n);transform[hi,n:]=np.eye(len(hi));transform[phi,n:]=-b
    tg=transform.T@m.g@transform;tk=transform.T@m.k;residual=(tg[n:,n:]+tg[n:,n:].T)/2
    vals,vec=np.linalg.eigh(residual);cut=max(np.max(abs(vals))*1e-12,1e-14);residual=(vec*np.where(vals>cut,vals,0))@vec.T
    correction=np.linalg.solve(residual+r*np.eye(len(hi)),tk[n:]-tg[n:,:n]@base);expected=transform@np.r_[base,correction]
    max_error=max(max_error,float(np.max(abs(m.readout-expected))));np.testing.assert_allclose(m.readout,expected,rtol=1e-6,atol=2e-7)
    with np.load(a.root/'features'/e/f'{d}_val.npz') as val,np.load(path.parent/f"predictions_stage{len(s['order'])}.npz") as saved:
        for k,group in enumerate(s['order']):
            inds=np.flatnonzero(np.isin(val['y'],group)) if s['task']=='classification' else np.flatnonzero(val['groups']==group)
            pred=m.predict(ck['basis'].transform(val['x'][inds]));pred=pred if s['task']=='classification' else pred*ck['target_sd']+ck['target_mean']
            np.testing.assert_array_equal(saved[f'pred_{k}'],pred);np.testing.assert_array_equal(saved[f'ids_{k}'],val['ids'][inds]);np.testing.assert_array_equal(saved[f'y_{k}'],val['y'][inds])
    if r==.1:
        with np.load(a.root/'features'/e/f'{d}_train.npz') as data:
            cl=s['task']=='classification';target=data['y'] if cl else (data['y']-ck['target_mean'])/ck['target_sd'];expected=Counter()
            parts=[np.flatnonzero(np.isin(data['y'],labels)) for labels in s['order']] if cl else [np.flatnonzero(data['groups']==group) for group in s['order']]
            for ids in parts:
                x=ck['basis'].transform(data['x'][ids]);y=target[ids]
                for j,row in enumerate(x):
                    raw=row if cl else np.r_[row,y[j]];label=int(y[j]) if cl else None
                    expected[label,hashlib.sha256(np.asarray(raw,np.float32).tobytes()).hexdigest()]+=1
        actual=Counter((None if label is None else int(label),hashlib.sha256(row.tobytes()).hexdigest()) for label,cells in m.local.cells.items() for cell in cells for row in cell.observed)
        assert expected==actual and sum(actual.values())==m.seen
    if c['variant']=='empirical':
        peer=path.parent.with_name(path.parent.name.replace('_empirical_','_gaussian_'))/'summary.json';ps,pl,pc=load(peer);ref=pc['learner']
        assert m.rng.bit_generator.state==ref.rng.bit_generator.state and m.local.rng.bit_generator.state==ref.local.rng.bit_generator.state
        assert m.local.cells.keys()==ref.local.cells.keys()
        for label,cells in m.local.cells.items():
            assert len(cells)==len(ref.local.cells[label])
            for cell,other in zip(cells,ref.local.cells[label]):
                assert cell.n==other.n
                for field in ('mean','factor','residual_sd','observed'):np.testing.assert_array_equal(getattr(cell,field),getattr(other,field))
        for field,other in zip(m.exact_memory.risk_moments(m.classes),ref.exact_memory.risk_moments(ref.classes)):np.testing.assert_array_equal(field,other)
        for left,right in zip(s['stages'],ps['stages']):
            for field in ('optimizer_steps','historical_training_samples','current_samples','representative_mass'):assert left[field]==right[field]
        equal_predictions(path.parent,peer.parent,[1]);pairs.append(path.parent.name)
    if c['variant']=='gaussian':
        old=a.root/'runs/covariance_replay_v48'/f'{d}_{e}_rank16_shrink01_r{r:g}_o17_s42'
        prior=json.loads((old/'summary.json').read_text());assert prior['input_hashes']==s['input_hashes'] and prior['order']==s['order']
        equal_predictions(path.parent,old,range(1,len(s['order'])+1));controls.append(path.parent.name)
    checked.append(path.parent.name)
    if len(checked)%30==0:print('checked',len(checked),flush=True)
assert len(checked)==144 and len(controls)==72 and len(pairs)==72 and len(cache)==24
report=dict(complete=True,checked=checked,controls=controls,paired= pairs,shared_states=len(cache),max_independent_readout_error=max_error)
a.out.write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ('checked','controls','paired')}))
