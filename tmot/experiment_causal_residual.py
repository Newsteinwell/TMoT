"""Warm nonlinear correction of matched causal ridge, with local or ER10 replay."""
import argparse,json,os,time
from pathlib import Path
import numpy as np
import torch
from .causal_reference import CausalReference
from .stable_residual import StableSemanticBasis
from .nonlinear_moments import NonlinearMomentLearner
from .strict_residual_replay import StrictResidualER
from .moments import augment
from .experiment_synthetic_robustness import sha

def run(a):
 torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
 root=Path(a.root);out=Path(a.out);out.mkdir(exist_ok=False);start=time.monotonic()
 tr=np.load(root/'features'/a.encoder/f'{a.dataset}_train.npz');val=np.load(root/'features'/a.encoder/f'{a.dataset}_val.npz');meta=json.loads((root/'prepared'/f'{a.dataset}_manifest.json').read_text());assert meta['task']=='regression'
 past={}
 for split,features in [('train',tr),('val',val)]:
  path=root/'prepared'/f'{a.dataset}_{split}.npz';assert sha(path)==meta['files'][split]
  with np.load(path) as raw:
   np.testing.assert_array_equal(raw['ids'],features['ids']);np.testing.assert_array_equal(raw['y'],features['y']);past[split]=raw['x'][:,0,-1,None].copy()
 order=sorted(np.unique(tr['groups']).tolist());parts=[np.flatnonzero(tr['groups']==g) for g in order];evals=[np.flatnonzero(val['groups']==g) for g in order]
 reference=CausalReference(a.reference).fit(past['train'][parts[0]]);sd=np.maximum(tr['y'][parts[0]].std(0),1e-6);target=tr['y']-reference.offset(past['train']);mu=target[parts[0]].mean(0);target=(target-mu)/sd
 basis=StableSemanticBasis(2*meta['shapes']['train'][1],512).fit(tr['x'][parts[0]]);dim=512+(a.reference!='none')
 cls=StrictResidualER if a.mode=='er10' else NonlinearMomentLearner
 model=cls(dim,'regression',42,ridge=a.ridge,mode='er' if a.mode=='er10' else 'local',epochs=30,device='cuda',residual_weight=a.penalty,warm_start=True,preserve_targets=True,local_components=4,local_support=4,local_min_cell=4,replay_count=4096,loss='mse')
 peer=root/'runs/causal_task_v68'/f'{a.dataset}_{a.encoder}_{a.reference}_ridge_r{a.ridge:g}'
 stages=[]
 for stage,ids in enumerate(parts):
  x=reference.features(basis.transform(tr['x'][ids]),past['train'][ids]);model.observe(x,target[ids]);predictions={};ys=[];ps=[];bs=[]
  with np.load(peer/f'predictions_stage{stage+1}.npz') as saved:
   for j,ix in enumerate(evals[:stage+1]):
    z=reference.features(basis.transform(val['x'][ix]),past['val'][ix]);offset=mu+reference.offset(past['val'][ix])
    pred=model.predict(z)*sd+offset
    # Preserve v68 arithmetic order for the exact control comparison.
    base=(augment(z)@model.base)*sd+mu+reference.offset(past['val'][ix])
    np.testing.assert_array_equal(base,saved[f'pred_{j}']);np.testing.assert_array_equal(val['y'][ix],saved[f'y_{j}']);np.testing.assert_array_equal(val['ids'][ix],saved[f'ids_{j}'])
    predictions.update({f'pred_{j}':pred,f'base_{j}':base,f'y_{j}':val['y'][ix],f'ids_{j}':val['ids'][ix]});ys.append(val['y'][ix]);ps.append(pred);bs.append(base)
  y=np.concatenate(ys);pred=np.concatenate(ps);base=np.concatenate(bs)
  row=dict(stage=stage+1,rmse=float(np.sqrt(np.mean((pred-y)**2))),base_rmse=float(np.sqrt(np.mean((base-y)**2))),current_samples=len(ids),historical_training_samples=model.historical_training_samples,optimizer_steps=model.optimizer_steps,warm_init_error=model.warm_init_error,persistent_bytes=model.persistent_bytes,base_exact=True,buffer_count=0 if model.buffer is None else len(model.buffer[0]))
  if a.mode=='er10':row['buffer_audit']=model.buffer_audit[-1]
  stages.append(row);np.savez_compressed(out/f'predictions_stage{stage+1}.npz',**predictions);print(json.dumps(row),flush=True)
 torch.save(dict(learner=model,basis=basis,reference=reference,target_mean=mu,target_sd=sd),out/'final.pt')
 result=dict(status='completed',task='regression',config=vars(a),stages=stages,final_rmse=stages[-1]['rmse'],average_rmse=float(np.mean([s['rmse'] for s in stages])),checkpoint_sha256=sha(out/'final.pt'),input_hashes={s:sha(root/'features'/a.encoder/f'{a.dataset}_{s}.npz') for s in ('train','val')},git_commit=os.environ['TMOT_SOURCE_COMMIT'],elapsed_s=time.monotonic()-start)
 (out/'summary.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser()
 for key in ('root','out','dataset','encoder','reference','mode'):p.add_argument('--'+key,required=True)
 p.add_argument('--ridge',type=float,required=True);p.add_argument('--penalty',type=float,required=True);run(p.parse_args())
