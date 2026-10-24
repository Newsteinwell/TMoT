"""Prefix-only, all-arrived-data refits to diagnose fixed projection loss."""
import argparse,hashlib,json,os,time
from pathlib import Path
import numpy as np
import torch
from .stable_residual import StableSemanticBasis
from .full_feature_diagnostic import unprojected_features,weighted_ridge_grid
RIDGES=(.001,.01,.1,1.,10.,100.)
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def macro(y,p):return float(np.mean([np.mean(p[y==c]==c) for c in np.unique(y)]))
def run(a):
    torch.set_num_threads(2);start=time.monotonic();root=Path(a.root);folder=Path(a.out)
    train=np.load(root/'features'/a.encoder/f'{a.dataset}_train.npz');val=np.load(root/'features'/a.encoder/f'{a.dataset}_val.npz');meta=json.loads((root/'prepared'/f'{a.dataset}_manifest.json').read_text())
    x,y=train['x'],train['y'];task=meta['task'];cl=task=='classification'
    if cl:
        labels=np.random.default_rng(17).permutation(np.unique(y));first=3 if len(labels)%2 else 2;order=[labels[:first]]+[labels[i:i+2] for i in range(first,len(labels),2)]
        parts=[np.flatnonzero(np.isin(y,c)) for c in order];evals=[np.flatnonzero(np.isin(val['y'],c)) for c in order];order=[c.tolist() for c in order];mu=0.;sd=1.
    else:
        order=sorted(np.unique(train['groups']).tolist());parts=[np.flatnonzero(train['groups']==g) for g in order];evals=[np.flatnonzero(val['groups']==g) for g in order]
        mu=y[parts[0]].mean(0);sd=np.maximum(y[parts[0]].std(0),1e-6);y=(y-mu)/sd
    basis=StableSemanticBasis(2*meta['shapes']['train'][1],512).fit(x[parts[0]])
    transform=basis.transform if a.mode=='projected' else lambda x:unprojected_features(basis,x)
    records={r:[] for r in RIDGES};matrices={r:[] for r in RIDGES};dest={};classes=[];history=[];targets=[]
    for r in RIDGES:
        dest[r]=folder/f'{a.dataset}_{a.encoder}_{a.mode}_r{r:g}_o17_s42';dest[r].mkdir(exist_ok=False)
    for stage,ids in enumerate(parts):
        history.append(transform(x[ids]));targets.append(y[ids]);hx=np.concatenate(history);hy=np.concatenate(targets)
        if cl:
            classes.extend(sorted(set(y[ids].tolist())-set(classes)));target=np.eye(len(classes))[[classes.index(int(c)) for c in hy]]
            weights=np.array([1/(len(classes)*np.sum(hy==c)) for c in hy])
        else:target=hy;weights=np.full(len(hy),1/len(hy))
        tick=time.monotonic();models,info=weighted_ridge_grid(hx,target,weights,RIDGES);fit_s=time.monotonic()-tick
        vx=[transform(val['x'][inds]) for inds in evals[:stage+1]]
        for r,(coef,bias) in models.items():
            predictions={};ys=[];ps=[];scores=[]
            for k,(z,inds) in enumerate(zip(vx,evals[:stage+1])):
                pred=z.astype(float)@coef+bias
                pred=np.asarray(classes)[pred.argmax(1)] if cl else pred*sd+mu
                truth=val['y'][inds];ys.append(truth);ps.append(pred);scores.append(macro(truth,pred) if cl else float(np.sqrt(np.mean((truth-pred)**2))))
                predictions.update({f'pred_{k}':pred,f'y_{k}':truth,f'ids_{k}':val['ids'][inds]})
            truth=np.concatenate(ys);pred=np.concatenate(ps)
            row=dict(stage=stage+1,fit_s_shared_across_six_ridges=fit_s,current_samples=len(ids),arrived_samples=len(hy),diagnostic_history_bytes=hx.nbytes+hy.nbytes,persistent_readout_bytes=coef.nbytes+bias.nbytes,scores=scores,solver=info)
            row['macro_accuracy' if cl else 'rmse']=macro(truth,pred) if cl else float(np.sqrt(np.mean((truth-pred)**2)))
            records[r].append(row);matrices[r].append(scores);np.savez_compressed(dest[r]/f'predictions_stage{stage+1}.npz',**predictions)
            with (dest[r]/'stages.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(dict(stage=stage+1,solver=info)),flush=True)
    for r,(coef,bias) in models.items():
        config=dict(root=str(root),out=str(dest[r]),dataset=a.dataset,encoder=a.encoder,mode=a.mode,method='all_arrived_refit',ridge=r,seed=42,order=17,eval_split='val')
        summary=dict(status='completed',task=task,config=config,order=order,stages=records[r],matrix=matrices[r],git_commit=os.environ['TMOT_SOURCE_COMMIT'],elapsed_s_shared_across_six_ridges=time.monotonic()-start,input_hashes={s:sha(root/'features'/a.encoder/f'{a.dataset}_{s}.npz') for s in ('train','val')},online_method=False)
        metric='macro_accuracy' if cl else 'rmse';summary['final_'+metric]=records[r][-1][metric];summary['average_'+metric]=float(np.mean([row[metric] for row in records[r]]))
        (dest[r]/'summary.json').write_text(json.dumps(summary,indent=2));torch.save(dict(coef=coef,bias=bias,basis=basis,classes=classes,target_mean=mu,target_sd=sd,mode=a.mode),dest[r]/'final.pt')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);p.add_argument('--dataset',required=True);p.add_argument('--encoder',choices=['mantis','summary'],required=True);p.add_argument('--mode',choices=['projected','full'],required=True);run(p.parse_args())
