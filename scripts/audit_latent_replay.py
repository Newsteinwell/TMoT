"""Resolve shared states; verify readouts, moment-matched pairs and old controls."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('out',type=Path);a=p.parse_args();torch.set_num_threads(1)
folder=a.root/'runs/latent_replay_v47';terminal=json.loads((folder/'complete.json').read_text());assert len(terminal)==150 and all(r['status']=='completed' for r in terminal)
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
    assert m.latent_law==c['latent_law'] and m.mode==c['method']
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
    if c['variant']=='rademacher':
        peer=path.parent.with_name(path.parent.name.replace('_rademacher_','_gaussian_'))/'summary.json';ps,pl,pc=load(peer);ref=pc['learner']
        assert m.rng.bit_generator.state==ref.rng.bit_generator.state and m.local.rng.bit_generator.state==ref.local.rng.bit_generator.state
        assert m.local.cells.keys()==ref.local.cells.keys()
        for label,cells in m.local.cells.items():
            assert len(cells)==len(ref.local.cells[label])
            for cell,other in zip(cells,ref.local.cells[label]):
                assert cell.n==other.n
                for field in ('mean','factor','residual_sd'):np.testing.assert_array_equal(getattr(cell,field),getattr(other,field))
        for field,other in zip(m.exact_memory.risk_moments(m.classes),ref.exact_memory.risk_moments(ref.classes)):np.testing.assert_array_equal(field,other)
        for left,right in zip(s['stages'],ps['stages']):
            for field in ('optimizer_steps','historical_training_samples','current_samples','representative_mass'):assert left[field]==right[field]
        equal_predictions(path.parent,peer.parent,[1]);pairs.append(path.parent.name)
    if c['variant']=='gaussian':
        old=None
        if d in ('phonemespectra','lsst','handwriting'):old=a.root/'runs/harder_validation_v45'/f'{d}_{e}_local_r{r:g}_o17_s42'
        elif d in ('beijing_pm25','air_quality'):old=a.root/'runs/regularized_innovation_v41'/f'{d}_{e}_local_reg1_r{r:g}_o17_s42'
        elif d=='energy' and e=='mantis':old=a.root/'runs/history_scale_v42'/f'{d}_{e}_local_reg1_r{r:g}_o17_s42'
        if old:
            prior=json.loads((old/'summary.json').read_text());assert prior['input_hashes']==s['input_hashes'] and prior['order']==s['order']
            equal_predictions(path.parent,old,range(1,len(s['order'])+1));controls.append(path.parent.name)
    checked.append(path.parent.name)
    if len(checked)%30==0:print('checked',len(checked),flush=True)
assert len(checked)==150 and len(controls)==66 and len(pairs)==72 and len(cache)==25
report=dict(complete=True,checked=checked,controls=controls,paired= pairs,shared_states=len(cache),max_independent_readout_error=max_error)
a.out.write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ('checked','controls','paired')}))
