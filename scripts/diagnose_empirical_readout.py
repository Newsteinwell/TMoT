"""Frozen-function final-stage diagnostic; full TRAIN access, not a CL method."""
import argparse,hashlib,json,os,time
from pathlib import Path
import numpy as np
import torch
from tmot.regularized_innovation import regularized_innovation_readout
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('dataset');p.add_argument('encoder');p.add_argument('out',type=Path);a=p.parse_args();torch.set_num_threads(1)
a.out.mkdir(parents=True,exist_ok=False);rows=[]
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
for variant in ('gaussian','empirical'):
    source=a.root/'runs/empirical_replay_v51'/f'{a.dataset}_{a.encoder}_{variant}_r0.1_o17_s42'
    s=json.loads((source/'summary.json').read_text());shared=(source/s['shared_state']).resolve();assert sha(shared)==s['shared_state_sha256']
    ck=torch.load(shared,map_location='cpu',weights_only=False);m=ck['learner'];cl=s['task']=='classification';start=time.monotonic()
    with np.load(a.root/'features'/a.encoder/f'{a.dataset}_train.npz') as data:
        parts=[np.flatnonzero(np.isin(data['y'],labels)) for labels in s['order']] if cl else [np.flatnonzero(data['groups']==g) for g in s['order']]
        ids=np.concatenate(parts);assert len(ids)==len(data['y'])==m.seen and len(np.unique(ids))==len(ids)
        x=ck['basis'].transform(data['x'][ids]);y=data['y'][ids]
    if cl:
        target=np.eye(len(m.classes))[[m.classes.index(int(c)) for c in y]];w=np.array([1/(len(m.classes)*np.sum(y==c)) for c in y])
    else:
        target=(y-ck['target_mean'])/ck['target_sd'];w=np.full(len(y),1/len(y))
    z=np.c_[m.function.transform(x).astype(float),np.ones(len(y))];g=z.T@(w[:,None]*z);k=z.T@(w[:,None]*target)
    phi=np.r_[np.arange(512),len(g)-1];hi=np.arange(512,len(g)-1)
    np.testing.assert_allclose(m.g[np.ix_(phi,phi)],g[np.ix_(phi,phi)],rtol=1e-7,atol=1e-7);np.testing.assert_allclose(m.k[phi],k[phi],rtol=1e-7,atol=1e-7)
    err=dict(hidden_gram_relative=float(np.linalg.norm((m.g-g)[np.ix_(hi,hi)])/max(np.linalg.norm(g[np.ix_(hi,hi)]),1e-12)),hidden_target_relative=float(np.linalg.norm((m.k-k)[hi])/max(np.linalg.norm(k[hi]),1e-12)),cross_gram_relative=float(np.linalg.norm((m.g-g)[np.ix_(phi,hi)])/max(np.linalg.norm(g[np.ix_(phi,hi)]),1e-12)))
    with np.load(a.root/'features'/a.encoder/f'{a.dataset}_val.npz') as val:
        evals=[np.flatnonzero(np.isin(val['y'],labels)) for labels in s['order']] if cl else [np.flatnonzero(val['groups']==group) for group in s['order']]
        vx=[np.c_[m.function.transform(ck['basis'].transform(val['x'][ids])).astype(float),np.ones(len(ids))] for ids in evals]
        vy=np.concatenate([val['y'][ids] for ids in evals]);vid=np.concatenate([val['ids'][ids] for ids in evals])
    for ridge in (.001,.01,.1,1.,10.,100.):
        readout,_=regularized_innovation_readout(g,k,512,ridge)
        # Independent augmented residual-feature normal equations.
        aa=g[np.ix_(phi,phi)];u=g[np.ix_(phi,hi)];pen=np.eye(len(phi))*ridge;pen[-1,-1]=0
        solved=np.linalg.solve(aa+pen,np.c_[k[phi],u]);base=solved[:,:k.shape[1]];b=solved[:,k.shape[1]:]
        residual=z[:,hi]-z[:,phi]@b;cg=residual.T@(w[:,None]*residual);ct=residual.T@(w[:,None]*(target-z[:,phi]@base))
        corr=np.linalg.solve(cg+ridge*np.eye(len(hi)),ct);independent=np.zeros_like(k);independent[phi]=base-b@corr;independent[hi]=corr
        np.testing.assert_allclose(readout,independent,rtol=1e-6,atol=2e-7)
        original=source.with_name(source.name.replace('_r0.1_','_r'+f'{ridge:g}'+'_'));old=torch.load(original/'final.pt',map_location='cpu',weights_only=False)['readout'];oldsummary=json.loads((original/'summary.json').read_text())
        def predict(weights):
            scores=np.concatenate([part@weights for part in vx])
            return np.asarray(m.classes)[scores.argmax(1)] if cl else scores*ck['target_sd']+ck['target_mean']
        def metric(pred):return float(np.mean([np.mean(pred[vy==c]==vy[vy==c]) for c in np.unique(vy)])) if cl else float(np.sqrt(np.mean((pred-vy)**2)))
        oldpred=predict(old);pred=predict(readout)
        with np.load(original/f"predictions_stage{len(s['order'])}.npz") as saved:
            np.testing.assert_array_equal(oldpred,np.concatenate([saved[f'pred_{i}'] for i in range(len(evals))]))
            np.testing.assert_array_equal(vid,np.concatenate([saved[f'ids_{i}'] for i in range(len(evals))]))
        oldmetric=metric(oldpred);newmetric=metric(pred)
        np.testing.assert_allclose(oldmetric,oldsummary['final_macro_accuracy' if cl else 'final_rmse'],rtol=1e-10,atol=1e-12)
        oldrisk=float(np.sum(w[:,None]*(z@old-target)**2));newrisk=float(np.sum(w[:,None]*(z@readout-target)**2));approx=float(np.sum(old*(m.g@old))-2*np.sum(old*m.k)+np.sum(w[:,None]*target**2))
        name=f'{variant}_r{ridge:g}'
        np.savez_compressed(a.out/(name+'.npz'),readout=readout,pred=pred,old_pred=oldpred,y=vy,ids=vid)
        rows.append(dict(dataset=a.dataset,encoder=a.encoder,variant=variant,ridge=ridge,task=s['task'],final_original=oldmetric,final_exact=newmetric,train_risk_original=oldrisk,train_risk_exact=newrisk,train_risk_approximate=approx,independent_readout_max_error=float(np.max(abs(readout-independent))),**err))
    print(variant,'complete',flush=True)
report=dict(complete=True,source_commit=os.environ['TMOT_SOURCE_COMMIT'],diagnostic='Final-stage full TRAIN moment refit under frozen learned functions; not a bounded-memory continual learner or theoretical upper bound',input_hashes=s['input_hashes'],rows=rows)
(a.out/'report.json').write_text(json.dumps(report,indent=2))
