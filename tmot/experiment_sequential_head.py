"""Matched sequential heads: FT, LwF-style CIL, and regression output distillation."""
import argparse,json,os,time
from pathlib import Path
import numpy as np
import torch
from .experiment_warm_budget import sha,macro
from .stable_residual import StableSemanticBasis
from .warm_functional_basis import WarmFunctionalBasis
from .replay_functional import IdentityBasis

def run(a):
 torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
 root=Path(a.root);out=Path(a.out);out.mkdir(exist_ok=False);start=time.monotonic()
 paths={s:root/'features'/a.encoder/f'{a.dataset}_{s}.npz' for s in ('train','val')}
 tr=np.load(paths['train']);va=np.load(paths['val']);meta=json.loads((root/'prepared'/f'{a.dataset}_manifest.json').read_text());task=meta['task'];cl=task=='classification';x=tr['x'];y=tr['y']
 if cl:
  labels=np.random.default_rng(17).permutation(np.unique(y));first=3 if len(labels)%2 else 2;order=[labels[:first]]+[labels[i:i+2] for i in range(first,len(labels),2)]
  parts=[np.flatnonzero(np.isin(y,c)) for c in order];evals=[np.flatnonzero(np.isin(va['y'],c)) for c in order];order=[c.tolist() for c in order];mu=0.;sd=1.
 else:
  order=sorted(np.unique(tr['groups']).tolist());parts=[np.flatnonzero(tr['groups']==g) for g in order];evals=[np.flatnonzero(va['groups']==g) for g in order];mu=y[parts[0]].mean(0);sd=np.maximum(y[parts[0]].std(0),1e-6);y=(y-mu)/sd
 basis=StableSemanticBasis(2*meta['shapes']['train'][1],512).fit(x[parts[0]])
 model=None;stats=None;rows=[]
 for t,ids in enumerate(parts):
  z=basis.transform(x[ids]);target=y[ids]
  if cl:
   w=np.array([1/np.sum(target==c) for c in target]);w/=w.sum()
  else:w=np.full(len(ids),1/len(ids))
  tick=time.monotonic();model=WarmFunctionalBasis(IdentityBasis(512),task,seed=42,epochs=30).fit(z,target,sample_weight=w,input_stats=stats,previous=model,distillation=a.strength)
  if stats is None:stats=(model.mean.copy(),model.sd.copy())
  fit=time.monotonic()-tick;predictions={};ys=[];ps=[];scores=[]
  for k,inds in enumerate(evals[:t+1]):
   pred=model.predict_head(basis.transform(va['x'][inds]));pred=pred if cl else pred*sd+mu;truth=va['y'][inds];ys.append(truth);ps.append(pred)
   scores.append(macro(truth,pred) if cl else float(np.sqrt(np.mean((truth-pred)**2))))
   predictions.update({f'pred_{k}':pred,f'y_{k}':truth,f'ids_{k}':va['ids'][inds]})
  truth=np.concatenate(ys);pred=np.concatenate(ps);metric='macro_accuracy' if cl else 'rmse'
  value=macro(truth,pred) if cl else float(np.sqrt(np.mean((truth-pred)**2)))
  row=dict(stage=t+1,scores=scores,current_samples=len(ids),historical_training_samples=0,stored_examples=0,fit_s=fit,optimizer_steps=model.optimizer_steps,warm_init_error=model.warm_init_error,loss=model.last_loss,distillation_loss=model.last_distillation_loss,**{metric:value})
  rows.append(row);np.savez_compressed(out/f'predictions_stage{t+1}.npz',**predictions)
  with (out/'stages.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  print(json.dumps(row),flush=True)
 torch.save(dict(model=model,basis=basis,target_mean=mu,target_sd=sd),out/'final.pt')
 summary=dict(status='completed',task=task,config=vars(a),order=order,stages=rows,git_commit=os.environ['TMOT_SOURCE_COMMIT'],input_hashes={s:sha(p) for s,p in paths.items()},elapsed_s=time.monotonic()-start,**{'final_'+metric:rows[-1][metric],'average_'+metric:float(np.mean([r[metric] for r in rows]))})
 (out/'summary.json').write_text(json.dumps(summary,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);p.add_argument('--dataset',required=True);p.add_argument('--encoder',required=True);p.add_argument('--strength',type=float,required=True);run(p.parse_args())
