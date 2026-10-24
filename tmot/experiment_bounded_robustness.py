"""Frozen development choices across seeds/orders; arrival-only statistics."""
import argparse,hashlib,json,os,pickle,time
from pathlib import Path
import numpy as np
import torch
from .unit_ball_moments import bound_statistics
from .stable_residual import StableSemanticBasis
from .task_synthetic_replay import TaskSyntheticLearner
from .moments import ClassificationMoments,augment,ridge_readout
from .fecam import unit_rows
from .class_distance_completion import statistics,block_features,predict_stats
from .regularized_innovation import regularized_innovation_readout

def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def macro(y,p):return float(np.mean([np.mean(p[y==c]==c) for c in np.unique(y)]))
def frozen(root,d,e):
 row=next(r for r in json.loads((root/'bounded_completion_v63_comparison.json').read_text())['rows'] if (r['dataset'],r['encoder'])==(d,e))
 task=next(r for r in json.loads((root/'synthetic_capacity_v54_comparison.json').read_text())['selected'] if (r['dataset'],r['encoder'],r['variant'])==(d,e,'task_n256'))
 head=next(r for r in json.loads((root/'analytic_heads_v60_comparison.json').read_text())['rows'] if (r['dataset'],r['encoder'])==(d,e))
 fecam=next(r for r in json.loads((root/'fecam_harder_v59_selected.json').read_text()) if (r['config']['dataset'],r['config']['encoder'])==(d,e))
 return dict(complete=row['modes']['complete']['alpha'],bounded=row['modes']['bounded']['alpha'],original=task['ridge'],fecam=fecam['config']['alpha1'],ridge=float(head['heads']['raw_ridge']['setting'].split('_')[1]))

def run(a):
 torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
 root=Path(a.root);out=Path(a.out);params=frozen(root,a.dataset,a.encoder)
 tr=np.load(root/'features'/a.encoder/f'{a.dataset}_train.npz');val=np.load(root/'features'/a.encoder/f'{a.dataset}_val.npz');meta=json.loads((root/'prepared'/f'{a.dataset}_manifest.json').read_text());assert meta['task']=='classification'
 old=root/'runs/synthetic_robustness_v55'/f'{a.dataset}_{a.encoder}_task_o{a.order}_s{a.seed}_r{params["original"]:g}';oldsummary=json.loads((old/'summary.json').read_text())
 labels=np.random.default_rng(a.order).permutation(np.unique(tr['y']));first=3 if len(labels)%2 else 2;order=[labels[:first]]+[labels[i:i+2] for i in range(first,len(labels),2)]
 assert [g.tolist() for g in order]==oldsummary['order'];parts=[np.flatnonzero(np.isin(tr['y'],g)) for g in order];evals=[np.flatnonzero(np.isin(val['y'],g)) for g in order]
 basis=StableSemanticBasis(2*meta['shapes']['train'][1],512).fit(tr['x'][parts[0]])
 m=TaskSyntheticLearner(512,'classification',a.seed,ridge=.1,mode='local',epochs=30,device='cuda',variant='task',points_per_cell=256)
 um=ClassificationMoments(512);raw=ClassificationMoments(512);group=f'{a.dataset}_{a.encoder}_o{a.order}_s{a.seed}';records={k:[] for k in params};start=time.monotonic()
 dest={k:out/f'{group}_{k}' for k in params}
 for folder in dest.values():folder.mkdir(exist_ok=False)
 for stage,ids in enumerate(parts):
  x=basis.transform(tr['x'][ids]);m.observe(x,tr['y'][ids]);um.update(unit_rows(x),tr['y'][ids]);raw.update(x,tr['y'][ids]);stats,diag=statistics(m,um);stats['bounded'],bounds=bound_statistics(stats['complete'],um,m.classes,512)
  vx=[basis.transform(val['x'][inds]) for inds in evals[:stage+1]];joint=np.concatenate([block_features(m.function,z) for z in vx]);unit=np.concatenate([unit_rows(z) for z in vx]);rx=np.concatenate(vx);sizes=[len(v) for v in vx]
  g,k,_=raw.risk_moments(m.classes);rw=ridge_readout(g,k,params['ridge']);tw=regularized_innovation_readout(m.g,m.k,512,params['original'])[0]
  preds={mode:predict_stats(joint,stats[mode],params[mode],m.classes) for mode in ('complete','bounded')}
  preds['fecam']=predict_stats(unit,stats['base'],params['fecam'],m.classes)
  preds['ridge']=np.asarray(m.classes)[(augment(rx)@rw).argmax(1)]
  preds['original']=np.concatenate([np.asarray(m.classes)[(augment(m.function.transform(z))@tw).argmax(1)] for z in vx])
  with np.load(old/f'predictions_stage{stage+1}.npz') as saved:
   for j,pred in enumerate(np.split(preds['original'],np.cumsum(sizes)[:-1])):np.testing.assert_array_equal(pred,saved[f'pred_{j}'])
  for mode,prediction in preds.items():
   arrays={};scores=[];ys=[]
   for j,(pred,inds) in enumerate(zip(np.split(prediction,np.cumsum(sizes)[:-1]),evals[:stage+1])):
    y=val['y'][inds];ys.append(y);scores.append(macro(y,pred));arrays.update({f'pred_{j}':pred,f'y_{j}':y,f'ids_{j}':val['ids'][inds]})
   if (a.seed,a.order)==(42,17):
    if mode in ('complete','bounded'):peer=root/'runs/bounded_completion_v63_retry'/f'{a.dataset}_{a.encoder}_{mode}_a{params[mode]:g}'
    elif mode=='fecam':peer=root/'runs/fecam_harder_v59'/f'{a.dataset}_{a.encoder}_fecam_a{params[mode]:g}_o17_s42'
    else:peer=None
    if peer is not None:
     with np.load(peer/f'predictions_stage{stage+1}.npz') as saved:
      for key in arrays:np.testing.assert_array_equal(arrays[key],saved[key])
   row=dict(stage=stage+1,current_samples=len(ids),macro_accuracy=macro(np.concatenate(ys),prediction),scores=scores,unit_ball=bounds if mode=='bounded' else None)
   records[mode].append(row);np.savez_compressed(dest[mode]/f'predictions_stage{stage+1}.npz',**arrays)
   with (dest[mode]/'stages.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  print(stage+1,'completed',flush=True)
 source=(old/oldsummary['shared_state']).resolve();assert sha(source)==oldsummary['shared_state_sha256'];original=torch.load(source,map_location='cpu',weights_only=False)['learner']
 for key in ('g','k'):np.testing.assert_array_equal(getattr(m,key),getattr(original,key))
 for p,q in zip(m.function.net.parameters(),original.function.net.parameters()):torch.testing.assert_close(p,q,rtol=0,atol=0)
 for label in m.local.cells:
  for left,right in zip(m.local.cells[label],original.local.cells[label]):np.testing.assert_array_equal(left.points,right.points)
 assert pickle.dumps(m.rng.bit_generator.state)==pickle.dumps(original.rng.bit_generator.state)
 # The pre-existing v55 model is exact; retain only new unit statistics.
 shared=out/'_shared'/f'{group}.pt';shared.parent.mkdir(exist_ok=True);torch.save(dict(unit_memory=um,source_checkpoint=str(source),source_sha256=sha(source)),shared);digest=sha(shared)
 for mode,stages in records.items():
  s=dict(status='completed',config=dict(dataset=a.dataset,encoder=a.encoder,seed=a.seed,order=a.order,mode=mode,parameter=params[mode],eval_split='val'),frozen=params,order=[g.tolist() for g in order],stages=stages,final_macro_accuracy=stages[-1]['macro_accuracy'],average_macro_accuracy=float(np.mean([r['macro_accuracy'] for r in stages])),shared_state='../_shared/'+shared.name,shared_state_sha256=digest,input_hashes={split:sha(root/'features'/a.encoder/f'{a.dataset}_{split}.npz') for split in ('train','val')},git_commit=os.environ['TMOT_SOURCE_COMMIT'],elapsed_s_shared=time.monotonic()-start,final_training_state_exact=True)
  assert s['input_hashes']==oldsummary['input_hashes'];(dest[mode]/'summary.json').write_text(json.dumps(s,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);p.add_argument('--dataset',required=True);p.add_argument('--encoder',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--order',type=int,required=True);run(p.parse_args())
