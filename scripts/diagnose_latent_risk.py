"""TRAIN-only completed-moment risk error under each learned function; no fitting."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('out',type=Path);a=p.parse_args();torch.set_num_threads(1)
folder=a.root/'runs/latent_replay_v47';cache={};rows=[]
for path in sorted(folder.glob('*/summary.json')):
    s=json.loads(path.read_text());c=s['config'];shared=(path.parent/s['shared_state']).resolve();assert shared.parent==(folder/'_shared').resolve()
    if shared not in cache:
        ck=torch.load(shared,map_location='cpu',weights_only=False);m=ck['learner'];data=np.load(a.root/'features'/c['encoder']/f"{c['dataset']}_train.npz");cl=s['task']=='classification'
        parts=[np.flatnonzero(np.isin(data['y'],labels)) for labels in s['order']] if cl else [np.flatnonzero(data['groups']==g) for g in s['order']]
        x=np.concatenate([ck['basis'].transform(data['x'][ids]) for ids in parts]);y=np.concatenate([data['y'][ids] for ids in parts]);assert len(y)==m.seen
        if cl:
            target=np.eye(len(m.classes))[[m.classes.index(int(label)) for label in y]];w=np.array([1/(len(m.classes)*np.sum(y==label)) for label in y])
        else:target=(y-ck['target_mean'])/ck['target_sd'];w=np.full(len(y),1/len(y))
        features=m.function.transform(x).astype(float);z=np.c_[features,np.ones(len(y))];g=z.T@(w[:,None]*z);k=z.T@(w[:,None]*target);v=float(np.sum(w[:,None]*target**2))
        phi=np.r_[np.arange(512),len(g)-1];hidden=np.arange(512,len(g)-1)
        np.testing.assert_allclose(m.g[np.ix_(phi,phi)],g[np.ix_(phi,phi)],rtol=1e-7,atol=1e-7);np.testing.assert_allclose(m.k[phi],k[phi],rtol=1e-7,atol=1e-7)
        error=dict(hidden_gram_relative=float(np.linalg.norm((m.g-g)[np.ix_(hidden,hidden)])/max(np.linalg.norm(g[np.ix_(hidden,hidden)]),1e-12)),hidden_target_relative=float(np.linalg.norm((m.k-k)[hidden])/max(np.linalg.norm(k[hidden]),1e-12)),cross_gram_relative=float(np.linalg.norm((m.g-g)[np.ix_(phi,hidden)])/max(np.linalg.norm(g[np.ix_(phi,hidden)]),1e-12)))
        cache[shared]=(m,z,target,w,g,k,v,error)
    m,z,target,w,g,k,v,error=cache[shared];leaf=torch.load(path.parent/'final.pt',map_location='cpu',weights_only=False);readout=leaf['readout']
    exact=float(np.sum(w[:,None]*(z@readout-target)**2));quadratic=float(np.sum(readout*(g@readout))-2*np.sum(readout*k)+v)
    np.testing.assert_allclose(exact,quadratic,rtol=1e-7,atol=1e-8)
    approximate=float(np.sum(readout*(m.g@readout))-2*np.sum(readout*m.k)+v)
    rows.append(dict(run=path.parent.name,dataset=c['dataset'],encoder=c['encoder'],variant=c['variant'],ridge=c['ridge'],train_risk_true=exact,train_risk_completed_moments=approximate,risk_bias=approximate-exact,**error))
assert len(rows)==150 and len(cache)==25
a.out.write_text(json.dumps(dict(complete=True,diagnostic='Each function has its own true TRAIN reference; this is not a shared-function causal gradient comparison',shared_states=len(cache),rows=rows),indent=2));print('150 TRAIN-risk checks; 25 shared function states')
