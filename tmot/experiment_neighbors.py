"""One complete trajectory; evaluation has no model-selection side effects."""
import argparse,hashlib,json,os,platform,resource,subprocess,time
from pathlib import Path
import numpy as np
import torch
from .raw_temporal import RawAppendedBasis,RawCoordinateBasis
from .learner import MomentLearner
from .neighbor_readout import NeighborReadout
from .stable_residual import StableSemanticBasis,StableResidualLearner


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def macro_accuracy(y,p):return float(np.mean([np.mean(p[y==c]==c) for c in np.unique(y)]))

def run(a):
    torch.set_num_threads(a.threads);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    root=Path(a.root);out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    train=np.load(root/'features'/a.encoder/f'{a.dataset}_train.npz')
    test=np.load(root/'features'/a.encoder/f'{a.dataset}_{a.eval_split}.npz')
    meta=json.loads((root/'prepared'/f'{a.dataset}_manifest.json').read_text())
    task=meta['task']
    x,y=train['x'],train['y'];test_x,test_y=test['x'],test['y']
    stages=[];evals=[]
    if task=='classification':
        classes=np.random.default_rng(a.order).permutation(np.unique(y))
        first=3 if len(classes)%2 else 2
        chunks=[classes[:first]]+[classes[i:i+2] for i in range(first,len(classes),2)]
        for c in chunks:
            stages.append(np.flatnonzero(np.isin(y,c)));evals.append(np.flatnonzero(np.isin(test_y,c)))
        order=[c.tolist() for c in chunks]
        target_mu=0.;target_sd=1.
    else:
        domains=sorted(np.unique(train['groups']).tolist());order=domains
        for d in domains:
            stages.append(np.flatnonzero(train['groups']==d));evals.append(np.flatnonzero(test['groups']==d))
        target_mu=y[stages[0]].mean(0);target_sd=np.maximum(y[stages[0]].std(0),1e-6)
        y=(y-target_mu)/target_sd
    channels=meta['shapes']['train'][1]
    if a.encoder.startswith('mantis_raw_'):
        raw_dim=channels if a.encoder=='mantis_raw_last' else 9*channels
        basis=RawAppendedBasis(2*channels,raw_dim,a.dimension,seed=a.basis_seed).fit(x[stages[0]])
    elif a.encoder=='raw_windows':
        basis=RawCoordinateBasis(9*channels).fit(x[stages[0]])
    else:
        basis=StableSemanticBasis(2*channels,a.dimension,seed=a.basis_seed).fit(x[stages[0]])
    if a.method in ('neighbors_full','neighbors_buffer'):
        learner=NeighborReadout(basis.dimension,task,a.neighbors,0 if a.method=='neighbors_full' else 4*1024*1024,a.seed)
    elif a.method=='stable_residual':
        learner=StableResidualLearner(basis.dimension,task,a.seed,rank=a.bottleneck,epochs=a.epochs,
                                      ridge=a.ridge,hist_weight=a.hist_weight,residual_weight=a.residual_weight,
                                      alpha=a.alpha,device=a.device)
    else:
        learner=MomentLearner(basis.dimension,task,a.method,a.seed,bottleneck=a.bottleneck,epochs=a.epochs,
                              device=a.device,ridge=a.ridge)
    if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats()
    records=[];matrix=[]
    for stage,ids in enumerate(stages):
        # Only arrived training records enter the learner or basis transform for fit.
        t=time.perf_counter();learner.observe(basis.transform(x[ids]),y[ids]);fit_s=time.perf_counter()-t
        t=time.perf_counter();scores=[];all_y=[];all_p=[];predictions={}
        for s,inds in enumerate(evals[:stage+1]):
            pred=learner.predict(basis.transform(test_x[inds]));truth=test_y[inds]
            if task=='regression':pred=pred*target_sd+target_mu
            score=macro_accuracy(truth,pred) if task=='classification' else float(np.sqrt(np.mean((truth-pred)**2)))
            scores.append(score);all_y.append(truth);all_p.append(pred)
            predictions[f'pred_{s}']=pred;predictions[f'ids_{s}']=test['ids'][inds];predictions[f'y_{s}']=truth
        truth=np.concatenate(all_y);pred=np.concatenate(all_p)
        item=dict(stage=stage+1,fit_s=fit_s,eval_s=time.perf_counter()-t,current_samples=len(ids),
                  persistent_bytes=learner.persistent_bytes,loss=learner.last_loss,scores=scores)
        if task=='classification':
            item.update(macro_accuracy=macro_accuracy(truth,pred),accuracy=float(np.mean(truth==pred)))
        else:
            item.update(rmse=float(np.sqrt(np.mean((truth-pred)**2))),mae=float(np.mean(np.abs(truth-pred))))
        records.append(item);matrix.append(scores)
        np.savez_compressed(out/f'predictions_stage{stage+1}.npz',**predictions)
        with (out/'stages.jsonl').open('a') as f:f.write(json.dumps(item)+'\n')
        print(json.dumps(item),flush=True)
    try:commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True,stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError:commit=Path('SOURCE_COMMIT').read_text().strip() if Path('SOURCE_COMMIT').exists() else 'uncommitted'
    commit=os.environ.get('TMOT_SOURCE_COMMIT',commit)
    summary=dict(status='completed',effective_dimension=basis.dimension,task=task,config=vars(a),order=order,stages=records,matrix=matrix,
                 elapsed_s=time.perf_counter()-started,git_commit=commit,
                 max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1024 if platform.system()=='Linux' else 1),
                 gpu_peak_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
                 input_hashes={s:sha(root/'features'/a.encoder/f'{a.dataset}_{s}.npz') for s in ('train',a.eval_split)})
    if task=='classification':
        summary['final_macro_accuracy']=records[-1]['macro_accuracy'];summary['average_macro_accuracy']=float(np.mean([r['macro_accuracy'] for r in records]))
        summary['forgetting']=float(np.mean([max(matrix[t][s] for t in range(s,len(stages)-1))-matrix[-1][s] for s in range(len(stages)-1)]))
    else:summary['final_rmse']=records[-1]['rmse'];summary['average_rmse']=float(np.mean([r['rmse'] for r in records]))
    # Nearest-neighbor diagnostic explicitly retains arrived real feature rows.
    torch.save({'learner':learner,'basis':basis,'target_mean':target_mu,'target_sd':target_sd},out/'final.pt')
    tmp=out/'summary.tmp';tmp.write_text(json.dumps(summary,indent=2));tmp.rename(out/'summary.json')
    return summary


def parser():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True)
    p.add_argument('--dataset',required=True);p.add_argument('--encoder',default='mantis');p.add_argument('--method',default='stable_residual')
    p.add_argument('--neighbors',type=int,default=5)
    p.add_argument('--basis-seed',type=int,default=1701)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--order',type=int,default=17)
    p.add_argument('--dimension',type=int,default=512);p.add_argument('--bottleneck',type=int,default=64)
    p.add_argument('--epochs',type=int,default=30);p.add_argument('--hist-weight',type=float,default=1.)
    p.add_argument('--sc-weight',type=float,default=.5);p.add_argument('--lr',type=float,default=.001);p.add_argument('--ridge',type=float,default=.01)
    p.add_argument('--eval-split',choices=['val','test'],default='test')
    p.add_argument('--readout',choices=['ridge','trained'],default='ridge')
    p.add_argument('--current-loss',choices=['ce','mse'],default='ce')
    p.add_argument('--alpha',type=float,default=1.)
    p.add_argument('--residual-weight',type=float,default=.1)
    p.add_argument('--device',default='cuda');p.add_argument('--threads',type=int,default=2)
    return p

if __name__=='__main__':run(parser().parse_args())
