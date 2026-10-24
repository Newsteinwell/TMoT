"""One shared learner per sampler, six regularized-innovation readouts."""
import argparse,hashlib,json,os,time
from pathlib import Path
import numpy as np
import torch
from .stable_residual import StableSemanticBasis
from .task_synthetic_replay import TaskSyntheticLearner,VARIANTS
from .regularized_innovation import regularized_innovation_readout
RIDGES=(.001,.01,.1,1.,10.,100.)
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def macro(y,p):return float(np.mean([np.mean(p[y==c]==c) for c in np.unique(y)]))
def run(a):
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
    start=time.monotonic();root=Path(a.root);folder=Path(a.out);train=np.load(root/'features'/a.encoder/f'{a.dataset}_train.npz');val=np.load(root/'features'/a.encoder/f'{a.dataset}_val.npz')
    meta=json.loads((root/'prepared'/f'{a.dataset}_manifest.json').read_text());x,y=train['x'],train['y'];task=meta['task'];cl=task=='classification'
    if cl:
        labels=np.random.default_rng(17).permutation(np.unique(y));first=3 if len(labels)%2 else 2;order=[labels[:first]]+[labels[i:i+2] for i in range(first,len(labels),2)]
        parts=[np.flatnonzero(np.isin(y,c)) for c in order];evals=[np.flatnonzero(np.isin(val['y'],c)) for c in order];order=[c.tolist() for c in order];mu=0.;sd=1.
    else:
        order=sorted(np.unique(train['groups']).tolist());parts=[np.flatnonzero(train['groups']==g) for g in order];evals=[np.flatnonzero(val['groups']==g) for g in order]
        mu=y[parts[0]].mean(0);sd=np.maximum(y[parts[0]].std(0),1e-6);y=(y-mu)/sd
    basis=StableSemanticBasis(2*meta['shapes']['train'][1],512).fit(x[parts[0]])
    method='local'
    m=TaskSyntheticLearner(512,task,42,ridge=.1,mode=method,epochs=30,device='cuda',variant=a.variant)
    group=f'{a.dataset}_{a.encoder}_{a.variant}';shared=folder/'_shared'/f'{group}.pt';shared.parent.mkdir(exist_ok=True)
    dest={};records={r:[] for r in RIDGES};matrices={r:[] for r in RIDGES}
    for r in RIDGES:dest[r]=folder/f'{group}_r{r:g}_o17_s42';dest[r].mkdir(exist_ok=False)
    for stage,ids in enumerate(parts):
        tick=time.monotonic();m.observe(basis.transform(x[ids]),y[ids]);fit_s=time.monotonic()-tick
        weights={r:regularized_innovation_readout(m.g,m.k,512,r) for r in RIDGES}
        vx=[basis.transform(val['x'][inds]) for inds in evals[:stage+1]]
        for r,(readout,diagnostics) in weights.items():
            m.readout=readout;ys=[];ps=[];scores=[];predictions={}
            for k,(z,inds) in enumerate(zip(vx,evals[:stage+1])):
                pred=m.predict(z);pred=pred if cl else pred*sd+mu;truth=val['y'][inds]
                ys.append(truth);ps.append(pred);scores.append(macro(truth,pred) if cl else float(np.sqrt(np.mean((truth-pred)**2))))
                predictions.update({f'pred_{k}':pred,f'y_{k}':truth,f'ids_{k}':val['ids'][inds]})
            truth=np.concatenate(ys);pred=np.concatenate(ps)
            row=dict(stage=stage+1,fit_s_shared_across_six_ridges=fit_s,current_samples=len(ids),persistent_bytes=m.persistent_bytes,loss=m.last_loss,scores=scores,optimizer_steps=m.optimizer_steps,historical_training_samples=m.historical_training_samples,representative_mass=m.weight_summary,completion=m.completion_diagnostics,innovation=diagnostics)
            row['macro_accuracy' if cl else 'rmse']=macro(truth,pred) if cl else float(np.sqrt(np.mean((truth-pred)**2)))
            records[r].append(row);matrices[r].append(scores);np.savez_compressed(dest[r]/f'predictions_stage{stage+1}.npz',**predictions)
            with (dest[r]/'stages.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        m.readout=weights[.1][0];print(json.dumps(dict(stage=stage+1,fit_s=fit_s)),flush=True)
    torch.save(dict(learner=m,basis=basis,target_mean=mu,target_sd=sd),shared);digest=sha(shared)
    for r,(readout,diagnostics) in weights.items():
        config=dict(root=str(root),out=str(dest[r]),dataset=a.dataset,encoder=a.encoder,variant=a.variant,method=method,replay_source=a.variant,ridge=r,seed=42,order=17,eval_split='val',dimension=512,epochs=30,regularized=True)
        ref='../_shared/'+shared.name
        summary=dict(status='completed',task=task,config=config,order=order,stages=records[r],matrix=matrices[r],git_commit=os.environ['TMOT_SOURCE_COMMIT'],elapsed_s_shared_across_six_ridges=time.monotonic()-start,input_hashes={s:sha(root/'features'/a.encoder/f'{a.dataset}_{s}.npz') for s in ('train','val')},shared_state=ref,shared_state_sha256=digest)
        metric='macro_accuracy' if cl else 'rmse';summary['final_'+metric]=records[r][-1][metric];summary['average_'+metric]=float(np.mean([row[metric] for row in records[r]]))
        torch.save(dict(readout=readout,shared_state=ref,shared_state_sha256=digest),dest[r]/'final.pt');(dest[r]/'summary.json').write_text(json.dumps(summary,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);p.add_argument('--dataset',required=True);p.add_argument('--encoder',choices=['mantis','summary'],required=True);p.add_argument('--variant',choices=VARIANTS,required=True);run(p.parse_args())
