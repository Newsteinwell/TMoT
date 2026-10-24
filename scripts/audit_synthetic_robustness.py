"""Resolve shared states; verify readouts, moment-matched pairs and old controls."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('out',type=Path);a=p.parse_args();torch.set_num_threads(1)
folder=a.root/'runs/synthetic_robustness_v55';terminal=json.loads((folder/'complete.json').read_text());assert len(terminal)==120 and all(r['status']=='completed' for r in terminal)
memory_violations=[];cache={};checked=[];controls=[];pairs=[];max_error=0.
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

# Verify all calibration scores and selection without reselection on new streams.
calibration=[]
for encoder in ('mantis','summary'):
    scores=[]
    for cp in sorted((a.root/'runs/er64_calibration_v55').glob(f'phonemespectra_{encoder}_*/summary.json')):
        cs=json.loads(cp.read_text());assert cs['config']['seed']==42 and cs['config']['order']==17
        for stage in cs['stages']:
            with np.load(cp.parent/f"predictions_stage{stage['stage']}.npz") as pred:
                keys=sorted(int(k[2:]) for k in pred.files if k.startswith('y_'));yy=np.concatenate([pred[f'y_{k}'] for k in keys]);pp=np.concatenate([pred[f'pred_{k}'] for k in keys])
                metric=np.mean([np.mean(pp[yy==c]==yy[yy==c]) for c in np.unique(yy)]);np.testing.assert_allclose(metric,stage['macro_accuracy'],rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(cs['final_macro_accuracy'],cs['stages'][-1]['macro_accuracy'])
        scores.append(cs);calibration.append(cp.parent.name)
    assert len(scores)==6
    chosen=min(scores,key=lambda x:(-x['final_macro_accuracy'],x['config']['ridge']))['config']['ridge']
    assert chosen==json.loads((folder/'frozen.json').read_text())['selected']['phonemespectra_'+encoder+'_er64']
for path in sorted(folder.glob('*/summary.json')):
    s,leaf,ck=load(path);c=s['config'];m=ck['learner'];m.readout=leaf['readout'];d=c['dataset'];e=c['encoder'];r=c['ridge']
    assert m.mode==c['method']
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
    frozen=json.loads((folder/'frozen.json').read_text())['selected'];assert r==frozen[d+'_'+e+'_'+c['variant']]
    assert m.seed==c['seed']
    if m.mode=='er' and (d!='phonemespectra' or c['variant']=='er64'):assert len(m.buffer[0])==m.seen
    if m.mode=='local':assert m.local.teacher is None and m.points_per_cell==256
    if c['seed']==42 and c['order']==17:
        if c['variant']=='er64':old=a.root/'runs/er64_calibration_v55'/path.parent.name
        else:
            prior_rows=json.loads((a.root/'synthetic_capacity_v54_comparison.json').read_text())['selected']
            row=next(x for x in prior_rows if x['dataset']==d and x['encoder']==e and x['variant']==('task_n256' if c['variant']=='task' else 'er'))
            old=a.root/'runs'/row['source_queue']/row['source_run']
        prior=json.loads((old/'summary.json').read_text());assert prior['input_hashes']==s['input_hashes'] and prior['order']==s['order']
        equal_predictions(path.parent,old,range(1,len(s['order'])+1));controls.append(path.parent.name)
    if c['variant']!='task':
        taskridge=frozen[d+'_'+e+'_task'];peer=folder/f"{d}_{e}_task_o{c['order']}_s{c['seed']}_r{taskridge:g}"/'summary.json'
        ps,pl,pc=load(peer);ref=pc['learner']
        assert m.classes==ref.classes
        for field,other in zip(m.exact_memory.risk_moments(m.classes),ref.exact_memory.risk_moments(ref.classes)):np.testing.assert_array_equal(field,other)
        for left,right in zip(s['stages'],ps['stages']):
            for field in ('optimizer_steps','historical_training_samples','current_samples'):assert left[field]==right[field]
            for key in left['representative_mass']:np.testing.assert_allclose(left['representative_mass'][key],right['representative_mass'][key],atol=1e-12)
            if left['persistent_bytes']>right['persistent_bytes']:memory_violations.append(dict(run=path.parent.name,stage=left['stage'],er_bytes=left['persistent_bytes'],task_bytes=right['persistent_bytes']))
        pairs.append(path.parent.name)
    checked.append(path.parent.name)
    if len(checked)%30==0:print('checked',len(checked),flush=True)
assert len(checked)==120 and len(controls)==26 and len(pairs)==66 and len(cache)==120
report=dict(complete=True,checked=checked,controls=controls,paired= pairs,shared_states=len(cache),max_independent_readout_error=max_error,er_memory_exceeds_task=memory_violations,calibration_metric_checked=calibration)
a.out.write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ('checked','controls','paired')}))
