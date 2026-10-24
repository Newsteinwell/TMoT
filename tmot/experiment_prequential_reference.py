"""Fixed-coordinate prior experts, weights from arrived TRAIN before fitting it."""
import argparse,json,os,time
from pathlib import Path
import numpy as np
import torch
from .causal_reference import CausalReference
from .stable_residual import StableSemanticBasis
from .task_synthetic_replay import TaskSyntheticLearner
from .learner import MomentLearner
from .regularized_innovation import regularized_innovation_readout
from .experiment_synthetic_robustness import sha,RIDGES
from .prequential_weights import InverseErrorWeights
REFS=('none','append','residual')
def run(a):
 torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
 root=Path(a.root);out=Path(a.out);out.mkdir(parents=True,exist_ok=True);start=time.monotonic();group=f'{a.dataset}_{a.encoder}'
 tr=np.load(root/'features'/a.encoder/f'{a.dataset}_train.npz');val=np.load(root/'features'/a.encoder/f'{a.dataset}_val.npz');meta=json.loads((root/'prepared'/f'{a.dataset}_manifest.json').read_text());assert meta['task']=='regression'
 past={}
 for split,features in [('train',tr),('val',val)]:
  path=root/'prepared'/f'{a.dataset}_{split}.npz';assert sha(path)==meta['files'][split]
  with np.load(path) as raw:
   np.testing.assert_array_equal(raw['y'],features['y']);np.testing.assert_array_equal(raw['ids'],features['ids']);past[split]=raw['x'][:,0,-1,None].copy()
 order=sorted(np.unique(tr['groups']).tolist());parts=[np.flatnonzero(tr['groups']==g) for g in order];evals=[np.flatnonzero(val['groups']==g) for g in order];sd=np.maximum(tr['y'][parts[0]].std(0),1e-6)
 basis=StableSemanticBasis(2*meta['shapes']['train'][1],512).fit(tr['x'][parts[0]]);branches=[]
 for name in REFS:
  ref=CausalReference(name).fit(past['train'][parts[0]]);target=tr['y']-ref.offset(past['train']);mu=target[parts[0]].mean(0);target=(target-mu)/sd;dim=512+(name!='none')
  model=TaskSyntheticLearner(dim,'regression',42,ridge=.1,mode='local',epochs=30,device='cuda',variant='task',points_per_cell=256);baselines={r:MomentLearner(dim,'regression','ridge',42,ridge=r) for r in RIDGES}
  branches.append(dict(ref=ref,target=target,mu=mu,dim=dim,model=model,baselines=baselines))
 states={(method,r):InverseErrorWeights(3) for method in ('task','ridge') for r in RIDGES};records={(policy,method,r):[] for policy in ('uniform','prequential') for method,r in states};dest={k:out/f'{group}_{k[0]}_{k[1]}_r{k[2]:g}' for k in records}
 for p in dest.values():p.mkdir(exist_ok=False)
 def predict(branch,method,r,x,past_values):
  model=branch['model'] if method=='task' else branch['baselines'][r]
  if method=='task':model.readout=regularized_innovation_readout(model.g,model.k,branch['dim'],r)[0]
  z=branch['ref'].features(basis.transform(x),past_values)
  return model.predict(z)*sd+branch['mu']+branch['ref'].offset(past_values)
 for stage,ids in enumerate(parts):
  if stage:
   # All experts predict the current task BEFORE any expert observes it.
   for (method,r),state in states.items():
    pred=np.stack([predict(b,method,r,tr['x'][ids],past['train'][ids]) for b in branches]);state.update(pred,tr['y'][ids])
    np.savez_compressed(out/f'{group}_pre_stage{stage+1}_{method}_r{r:g}.npz',predictions=pred,y=tr['y'][ids],ids=tr['ids'][ids])
  for b in branches:
   x=b['ref'].features(basis.transform(tr['x'][ids]),past['train'][ids]);b['model'].observe(x,b['target'][ids])
   for model in b['baselines'].values():model.observe(x,b['target'][ids])
  for (method,r),state in states.items():
   predictions=[]
   for b,name in zip(branches,REFS):
    pp=[predict(b,method,r,val['x'][ix],past['val'][ix]) for ix in evals[:stage+1]];predictions.append(pp)
    peer=root/'runs/causal_task_v68'/f'{a.dataset}_{a.encoder}_{name}_{method}_r{r:g}'
    with np.load(peer/f'predictions_stage{stage+1}.npz') as saved:
     for j,p in enumerate(pp):np.testing.assert_array_equal(p,saved[f'pred_{j}'])
   for policy in ('uniform','prequential'):
    w=np.ones(3)/3 if policy=='uniform' else state.weights();arrays={};ys=[];ps=[];scores=[]
    for j,ix in enumerate(evals[:stage+1]):
     experts=np.stack([pp[j] for pp in predictions]);p=np.tensordot(w,experts,axes=(0,0));y=val['y'][ix];ys.append(y);ps.append(p);scores.append(float(np.sqrt(np.mean((p-y)**2))));arrays.update({f'pred_{j}':p,f'experts_{j}':experts,f'y_{j}':y,f'ids_{j}':val['ids'][ix]})
    row=dict(stage=stage+1,rmse=float(np.sqrt(np.mean((np.concatenate(ps)-np.concatenate(ys))**2))),scores=scores,weights=w.tolist(),prequential_sse=state.sse.tolist(),prequential_count=state.count)
    records[(policy,method,r)].append(row);np.savez_compressed(dest[(policy,method,r)]/f'predictions_stage{stage+1}.npz',**arrays)
  print(stage+1,'completed',flush=True)
 for key,stages in records.items():
  policy,method,r=key;s=dict(status='completed',config=dict(dataset=a.dataset,encoder=a.encoder,policy=policy,method=method,ridge=r,seed=42,eval_split='val'),order=order,stages=stages,final_rmse=stages[-1]['rmse'],average_rmse=float(np.mean([v['rmse'] for v in stages])),input_hashes={split:sha(root/'features'/a.encoder/f'{a.dataset}_{split}.npz') for split in ('train','val')},git_commit=os.environ['TMOT_SOURCE_COMMIT'],elapsed_s_shared=time.monotonic()-start,note='Three experts; 3x branch training and memory, fixed coordinates. Pre-task prediction artifacts are offline evidence, never replay memory.')
  (dest[key]/'summary.json').write_text(json.dumps(s,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);p.add_argument('--dataset',choices=['beijing_pm25','air_quality'],required=True);p.add_argument('--encoder',required=True);run(p.parse_args())
