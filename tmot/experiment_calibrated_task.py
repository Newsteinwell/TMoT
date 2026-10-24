"""Task-synthetic learner versus identical-input ridge under causal references."""
import argparse,json,os,time
from pathlib import Path
import numpy as np
import torch
from .causal_reference import CausalReference
from .calibrated_reference import CalibratedReference
from .stable_residual import StableSemanticBasis
from .task_synthetic_replay import TaskSyntheticLearner
from .learner import MomentLearner
from .regularized_innovation import regularized_innovation_readout
from .experiment_synthetic_robustness import sha,RIDGES

def run(a):
 torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
 root=Path(a.root);out=Path(a.out);out.mkdir(parents=True,exist_ok=True);start=time.monotonic()
 tr=np.load(root/'features'/a.encoder/f'{a.dataset}_train.npz');val=np.load(root/'features'/a.encoder/f'{a.dataset}_val.npz');meta=json.loads((root/'prepared'/f'{a.dataset}_manifest.json').read_text());assert meta['task']=='regression' and a.dataset in ('beijing_pm25','air_quality')
 past={}
 for split,features in [('train',tr),('val',val)]:
  path=root/'prepared'/f'{a.dataset}_{split}.npz';assert sha(path)==meta['files'][split]
  with np.load(path) as raw:
   np.testing.assert_array_equal(raw['ids'],features['ids']);np.testing.assert_array_equal(raw['y'],features['y']);past[split]=raw['x'][:,0,-1,None].copy()
 order=sorted(np.unique(tr['groups']).tolist());parts=[np.flatnonzero(tr['groups']==g) for g in order];evals=[np.flatnonzero(val['groups']==g) for g in order]
 reference=(CalibratedReference().fit(past['train'][parts[0]],tr['y'][parts[0]]) if a.reference=='calibrated' else CausalReference(a.reference).fit(past['train'][parts[0]]));sd=np.maximum(tr['y'][parts[0]].std(0),1e-6);target=tr['y']-reference.offset(past['train']);mu=target[parts[0]].mean(0);target=(target-mu)/sd
 basis=StableSemanticBasis(2*meta['shapes']['train'][1],512).fit(tr['x'][parts[0]]);dim=512+(a.reference!='none')
 model=TaskSyntheticLearner(dim,'regression',42,ridge=.1,mode='local',epochs=30,device='cuda',variant='task',points_per_cell=256)
 baselines={r:MomentLearner(dim,'regression','ridge',42,ridge=r) for r in RIDGES};group=f'{a.dataset}_{a.encoder}_{a.reference}';records={(mode,r):[] for mode in ('task','ridge') for r in RIDGES};folders={key:out/f'{group}_{key[0]}_r{key[1]:g}' for key in records}
 for folder in folders.values():folder.mkdir(exist_ok=False)
 for stage,ids in enumerate(parts):
  x=reference.features(basis.transform(tr['x'][ids]),past['train'][ids]);model.observe(x,target[ids])
  for baseline in baselines.values():baseline.observe(x,target[ids])
  weights={r:regularized_innovation_readout(model.g,model.k,dim,r)[0] for r in RIDGES}
  vx=[reference.features(basis.transform(val['x'][ix]),past['val'][ix]) for ix in evals[:stage+1]]
  for (mode,r),folder in folders.items():
   model.readout=weights[r];learner=model if mode=='task' else baselines[r];predictions={};ys=[];ps=[];scores=[]
   for j,(z,ix) in enumerate(zip(vx,evals[:stage+1])):
    pred=learner.predict(z)*sd+mu+reference.offset(past['val'][ix]);y=val['y'][ix];ys.append(y);ps.append(pred);scores.append(float(np.sqrt(np.mean((pred-y)**2))))
    predictions.update({f'pred_{j}':pred,f'y_{j}':y,f'ids_{j}':val['ids'][ix]})
   peer=None if a.reference=='calibrated' else root/'runs/causal_task_v68'/f'{a.dataset}_{a.encoder}_{a.reference}_{mode}_r{r:g}'
   if peer is not None:
    with np.load(peer/f'predictions_stage{stage+1}.npz') as saved:
     for key in predictions:np.testing.assert_array_equal(predictions[key],saved[key])
   rmse=float(np.sqrt(np.mean((np.concatenate(ps)-np.concatenate(ys))**2)));row=dict(stage=stage+1,rmse=rmse,scores=scores,current_samples=len(ids),persistent_bytes=learner.persistent_bytes,control_verified=peer is not None)
   records[(mode,r)].append(row);np.savez_compressed(folder/f'predictions_stage{stage+1}.npz',**predictions)
  print(stage+1,'completed',flush=True)
 model.readout=weights[.1];shared=out/'_shared'/f'{group}.pt';shared.parent.mkdir(exist_ok=True);torch.save(dict(learner=model,baselines=baselines,basis=basis,reference=reference,target_mean=mu,target_sd=sd),shared);digest=sha(shared)
 for (mode,r),stages in records.items():
  summary=dict(status='completed',task='regression',config=dict(dataset=a.dataset,encoder=a.encoder,reference=a.reference,method=mode,ridge=r,dimension=dim,seed=42,order=17,eval_split='val'),order=order,stages=stages,final_rmse=stages[-1]['rmse'],average_rmse=float(np.mean([s['rmse'] for s in stages])),shared_state='../_shared/'+shared.name,shared_state_sha256=digest,input_hashes={split:sha(root/'features'/a.encoder/f'{a.dataset}_{split}.npz') for split in ('train','val')},raw_input_hashes={split:meta['files'][split] for split in ('train','val')},calibration=dict(count=getattr(reference,'calibration_count',len(parts[0])),beta=getattr(reference,'beta',np.ones(tr['y'].shape[1])).tolist(),unclipped_beta=getattr(reference,'unclipped_beta',np.ones(tr['y'].shape[1])).tolist()),git_commit=os.environ['TMOT_SOURCE_COMMIT'],elapsed_s_shared=time.monotonic()-start)
  (folders[(mode,r)]/'summary.json').write_text(json.dumps(summary,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);p.add_argument('--dataset',required=True);p.add_argument('--encoder',required=True);p.add_argument('--reference',choices=['residual','calibrated'],required=True);run(p.parse_args())
