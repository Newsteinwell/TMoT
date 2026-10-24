import argparse,hashlib,json,os,time
from pathlib import Path
import numpy as np
import torch
from .stable_residual import StableSemanticBasis
from .moments import ClassificationMoments
from .fecam import unit_rows
from .analytic_heads import make_heads,predict_head

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def macro(y,p):return float(np.mean([np.mean(p[y==c]==c) for c in np.unique(y)]))
def run(a):
 torch.set_num_threads(1);root=Path(a.root);out=Path(a.out);start=time.monotonic()
 tr=np.load(root/'features'/a.encoder/f'{a.dataset}_train.npz');val=np.load(root/'features'/a.encoder/f'{a.dataset}_val.npz');meta=json.loads((root/'prepared'/f'{a.dataset}_manifest.json').read_text());assert meta['task']=='classification'
 labels=np.random.default_rng(17).permutation(np.unique(tr['y']));first=3 if len(labels)%2 else 2;order=[labels[:first]]+[labels[i:i+2] for i in range(first,len(labels),2)]
 parts=[np.flatnonzero(np.isin(tr['y'],c)) for c in order];evals=[np.flatnonzero(np.isin(val['y'],c)) for c in order]
 basis=StableSemanticBasis(2*meta['shapes']['train'][1],512).fit(tr['x'][parts[0]])
 def transform(x):
  z=basis.transform(x)
  return unit_rows(z) if a.normalization=='unit' else z
 memory=ClassificationMoments(512);classes=[];records={};group=f'{a.dataset}_{a.encoder}_{a.normalization}';shared=out/'_shared'/f'{group}.pt';shared.parent.mkdir(exist_ok=True)
 for stage,ids in enumerate(parts):
  tick=time.monotonic();memory.update(transform(tr['x'][ids]),tr['y'][ids]);classes.extend(sorted(set(tr['y'][ids].tolist())-set(classes)));heads=make_heads(memory,classes);fit=time.monotonic()-tick
  for name,head in heads.items():
   dest=out/f'{group}_{name}';dest.mkdir(exist_ok=True);ys=[];ps=[];scores=[];predictions={}
   for k,inds in enumerate(evals[:stage+1]):
    y=val['y'][inds];pred=predict_head(head,transform(val['x'][inds]),classes);ys.append(y);ps.append(pred);scores.append(macro(y,pred));predictions.update({f'y_{k}':y,f'pred_{k}':pred,f'ids_{k}':val['ids'][inds]})
   # Account for this method's stats and head arrays, not all grid candidates at once.
   size=memory.nbytes+sum(v.nbytes for v in head.values() if isinstance(v,np.ndarray))
   row=dict(stage=stage+1,current_samples=len(ids),fit_s_shared=fit,persistent_bytes=size,scores=scores,macro_accuracy=macro(np.concatenate(ys),np.concatenate(ps)))
   records.setdefault(name,[]).append(row);np.savez_compressed(dest/f'predictions_stage{stage+1}.npz',**predictions)
   with (dest/'stages.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  print(stage+1,flush=True)
 torch.save(dict(memory=memory,basis=basis,classes=classes,heads=heads,normalization=a.normalization),shared);digest=sha(shared)
 for name,head in heads.items():
  s=dict(status='completed',task='classification',config=dict(dataset=a.dataset,encoder=a.encoder,normalization=a.normalization,method=head['kind'],setting=name,seed=42,order=17,eval_split='val',dimension=512),order=[c.tolist() for c in order],stages=records[name],final_macro_accuracy=records[name][-1]['macro_accuracy'],average_macro_accuracy=float(np.mean([v['macro_accuracy'] for v in records[name]])),shared_state='../_shared/'+shared.name,shared_state_sha256=digest,git_commit=os.environ['TMOT_SOURCE_COMMIT'],input_hashes={split:sha(root/'features'/a.encoder/f'{a.dataset}_{split}.npz') for split in ('train','val')},elapsed_s_shared=time.monotonic()-start)
  (out/f'{group}_{name}'/'summary.json').write_text(json.dumps(s,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);p.add_argument('--dataset',required=True);p.add_argument('--encoder',required=True);p.add_argument('--normalization',choices=['raw','unit'],required=True);run(p.parse_args())
