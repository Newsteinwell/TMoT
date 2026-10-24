"""Matched inputs for task-synthetic, exact ridge and FeCAM readouts."""
import argparse,json,os,time
from pathlib import Path
import numpy as np
import torch
from .interaction_basis import InteractionBasis
from .causal_reference import CausalReference
from .task_synthetic_replay import TaskSyntheticLearner
from .fecam import FeCAMLearner
from .moments import ClassificationMoments,RegressionMoments,ridge_readout,augment
from .regularized_innovation import regularized_innovation_readout
from .experiment_synthetic_capacity import sha,macro,RIDGES
ALPHAS=(.01,.1,.5,1.,2.,10.)

def run(a):
 torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
 root=Path(a.root);out=Path(a.out);out.mkdir(exist_ok=False);start=time.monotonic()
 tr=np.load(root/'features'/a.encoder/f'{a.dataset}_train.npz');val=np.load(root/'features'/a.encoder/f'{a.dataset}_val.npz');meta=json.loads((root/'prepared'/f'{a.dataset}_manifest.json').read_text());cl=meta['task']=='classification';task=meta['task'];raw={}
 for split,f in [('train',tr),('val',val)]:
  p=root/'prepared'/f'{a.dataset}_{split}.npz';assert sha(p)==meta['files'][split]
  with np.load(p) as d:
   np.testing.assert_array_equal(d['ids'],f['ids']);np.testing.assert_array_equal(d['y'],f['y']);raw[split]=d['x'].copy()
 if cl:
  labels=np.random.default_rng(17).permutation(np.unique(tr['y']));first=3 if len(labels)%2 else 2;order=[labels[:first]]+[labels[i:i+2] for i in range(first,len(labels),2)]
  parts=[np.flatnonzero(np.isin(tr['y'],c)) for c in order];evals=[np.flatnonzero(np.isin(val['y'],c)) for c in order]
 else:
  order=sorted(np.unique(tr['groups']).tolist());parts=[np.flatnonzero(tr['groups']==g) for g in order];evals=[np.flatnonzero(val['groups']==g) for g in order]
 refmode='residual' if a.dataset=='beijing_pm25' else 'none'
 ref=CausalReference(refmode).fit(raw['train'][parts[0],0,-1,None])
 target=tr['y'] if cl else tr['y']-ref.offset(raw['train'][:,0,-1,None])
 mu=0. if cl else target[parts[0]].mean(0);sd=1. if cl else np.maximum(tr['y'][parts[0]].std(0),1e-6);target=(target-mu)/sd if not cl else target
 basis=InteractionBasis(meta['shapes']['train'][1],a.features=='interaction').fit(tr['x'][parts[0]],raw['train'][parts[0]])
 dim=basis.dimension+(refmode!='none')
 def transform(features,split,ix):return ref.features(basis.transform(features['x'][ix],raw[split][ix]),raw[split][ix,0,-1,None])
 m=TaskSyntheticLearner(dim,task,42,ridge=.1,mode='local',epochs=30,device='cuda',variant='task',points_per_cell=256)
 memory=ClassificationMoments(dim) if cl else RegressionMoments(dim,target.shape[1]);classes=[]
 fecams={alpha:FeCAMLearner(dim,alpha) for alpha in ALPHAS} if cl else {}
 records={};last_weights={}
 for stage,ids in enumerate(parts):
  x=transform(tr,'train',ids);m.observe(x,target[ids]);memory.update(x,target[ids])
  if cl:classes.extend(sorted(set(target[ids].tolist())-set(classes)))
  g,k,_=memory.risk_moments(classes);weights={('task',r):regularized_innovation_readout(m.g,m.k,dim,r)[0] for r in RIDGES}
  weights.update({('ridge',r):ridge_readout(g,k,r) for r in RIDGES})
  for f in fecams.values():f.observe(x,target[ids])
  for method,param in list(weights)+[('fecam',alpha) for alpha in fecams]:
   name=f'{method}_{param:g}';dest=out/name;dest.mkdir(exist_ok=True);ys=[];ps=[];predictions={}
   for j,ix in enumerate(evals[:stage+1]):
    z=transform(val,'val',ix)
    if method=='task':m.readout=weights[(method,param)];pred=m.predict(z)
    elif method=='fecam':pred=fecams[param].predict(z)
    else:
     scores=augment(z)@weights[(method,param)];pred=np.asarray(classes)[scores.argmax(1)] if cl else scores
    if not cl:pred=pred*sd+mu+ref.offset(raw['val'][ix,0,-1,None])
    ys.append(val['y'][ix]);ps.append(pred);predictions.update({f'pred_{j}':pred,f'y_{j}':val['y'][ix],f'ids_{j}':val['ids'][ix]})
   if a.features=='base':
    if not cl:peer=root/'runs/causal_task_v68'/f'{a.dataset}_{a.encoder}_{refmode}_{method}_r{param:g}'
    elif method=='task':peer=root/'runs/synthetic_capacity_v54'/f'{a.dataset}_{a.encoder}_task_n256_r{param:g}_o17_s42'
    elif method=='ridge':peer=root/'runs/analytic_heads_v60'/f'{a.dataset}_{a.encoder}_raw_ridge_{param:g}'
    else:peer=root/'runs/fecam_harder_v59'/f'{a.dataset}_{a.encoder}_fecam_a{param:g}_o17_s42'
    with np.load(peer/f'predictions_stage{stage+1}.npz') as saved:
     for key in predictions:np.testing.assert_array_equal(predictions[key],saved[key])
   metric=macro(np.concatenate(ys),np.concatenate(ps)) if cl else float(np.sqrt(np.mean((np.concatenate(ys)-np.concatenate(ps))**2)))
   size=m.persistent_bytes if method=='task' else fecams[param].persistent_bytes if method=='fecam' else memory.nbytes+weights[(method,param)].nbytes
   row=dict(stage=stage+1,metric=metric,persistent_bytes=size,base_control_exact=a.features=='base')
   records.setdefault(name,[]).append(row);np.savez_compressed(dest/f'predictions_stage{stage+1}.npz',**predictions)
  last_weights=weights;print(stage+1,'complete',flush=True)
 m.readout=last_weights[('task',.1)]
 state=dict(learner=m,basis=basis,reference=ref,target_mean=mu,target_sd=sd,weights=last_weights,classes=classes,ridge_memory=memory,fecam_memory=fecams[1.].memory if cl else None)
 torch.save(state,out/'shared.pt');h=sha(out/'shared.pt')
 for name,rows in records.items():
  method,param=name.split('_');s=dict(status='completed',task=task,config=dict(dataset=a.dataset,encoder=a.encoder,features=a.features,method=method,param=float(param),dimension=dim,reference=refmode,seed=42,order=17,eval_split='val'),stages=rows,final=rows[-1]['metric'],average=float(np.mean([r['metric'] for r in rows])),shared_sha256=h,git_commit=os.environ['TMOT_SOURCE_COMMIT'],input_hashes={split:sha(root/'features'/a.encoder/f'{a.dataset}_{split}.npz') for split in ('train','val')},raw_hashes=meta['files'],elapsed_s_shared=time.monotonic()-start)
  (out/name/'summary.json').write_text(json.dumps(s,indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser()
 for key in ('root','out','dataset','encoder','features'):p.add_argument('--'+key,required=True)
 run(p.parse_args())
