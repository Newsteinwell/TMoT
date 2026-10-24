import argparse,hashlib,json,os,pickle,time
from pathlib import Path
import numpy as np
import torch
from .stable_residual import StableSemanticBasis
from .task_synthetic_replay import TaskSyntheticLearner
from .moments import ClassificationMoments,augment
from .fecam import unit_rows
from .class_distance_completion import statistics,block_features,predict_stats
from .regularized_innovation import regularized_innovation_readout
ALPHAS=(.01,.1,.5,1.,2.,10.)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def macro(y,p):return float(np.mean([np.mean(p[y==c]==c) for c in np.unique(y)]))
def run(a):
 torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
 root=Path(a.root);out=Path(a.out);tr=np.load(root/'features'/a.encoder/f'{a.dataset}_train.npz');val=np.load(root/'features'/a.encoder/f'{a.dataset}_val.npz');meta=json.loads((root/'prepared'/f'{a.dataset}_manifest.json').read_text());assert meta['task']=='classification'
 ref=next(r for r in json.loads((root/'synthetic_capacity_v54_comparison.json').read_text())['selected'] if (r['dataset'],r['encoder'],r['variant'])==(a.dataset,a.encoder,'task_n256'))
 old=root/'runs'/ref['source_queue']/ref['source_run'];oldsummary=json.loads((old/'summary.json').read_text())
 labels=np.random.default_rng(17).permutation(np.unique(tr['y']));first=3 if len(labels)%2 else 2;order=[labels[:first]]+[labels[i:i+2] for i in range(first,len(labels),2)]
 parts=[np.flatnonzero(np.isin(tr['y'],g)) for g in order];evals=[np.flatnonzero(np.isin(val['y'],g)) for g in order];assert [g.tolist() for g in order]==oldsummary['order']
 basis=StableSemanticBasis(2*meta['shapes']['train'][1],512).fit(tr['x'][parts[0]])
 m=TaskSyntheticLearner(512,'classification',42,ridge=.1,mode='local',epochs=30,device='cuda',variant='task',points_per_cell=256)
 unit_memory=ClassificationMoments(512);group=f'{a.dataset}_{a.encoder}';records={};start=time.monotonic()
 for stage,ids in enumerate(parts):
  tick=time.monotonic();x=basis.transform(tr['x'][ids]);m.observe(x,tr['y'][ids]);unit_memory.update(unit_rows(x),tr['y'][ids]);stats,diag=statistics(m,unit_memory);fit=time.monotonic()-tick
  vx=[basis.transform(val['x'][inds]) for inds in evals[:stage+1]];vj=[block_features(m.function,z) for z in vx];vu=[unit_rows(z) for z in vx]
  # Original training remains identical at every stage; only new readouts differ.
  oldweight=regularized_innovation_readout(m.g,m.k,512,ref['ridge'])[0]
  with np.load(old/f'predictions_stage{stage+1}.npz') as saved:
   for k,z in enumerate(vx):np.testing.assert_array_equal(np.asarray(m.classes)[(augment(m.function.transform(z))@oldweight).argmax(1)],saved[f'pred_{k}'])
  for mode in ('base','approx','self','complete'):
   features=vu if mode=='base' else vj;allx=np.concatenate(features);sizes=[len(v) for v in features]
   for alpha in ALPHAS:
    name=f'{group}_{mode}_a{alpha:g}';folder=out/name;folder.mkdir(exist_ok=True)
    prediction=predict_stats(allx,stats[mode],alpha,m.classes);preds=np.split(prediction,np.cumsum(sizes)[:-1]);ys=[];scores=[];arrays={}
    for k,(pred,inds) in enumerate(zip(preds,evals[:stage+1])):
     y=val['y'][inds];ys.append(y);scores.append(macro(y,pred));arrays.update({f'pred_{k}':pred,f'y_{k}':y,f'ids_{k}':val['ids'][inds]})
    if mode=='base':
     peer=root/'runs/fecam_harder_v59'/f'{a.dataset}_{a.encoder}_fecam_a{alpha:g}_o17_s42'/f'predictions_stage{stage+1}.npz'
     with np.load(peer) as control:
      for k in range(stage+1):np.testing.assert_array_equal(arrays[f'pred_{k}'],control[f'pred_{k}'])
    memory=m.persistent_bytes+unit_memory.nbytes+sum(mu.nbytes+cov.nbytes for mu,cov in stats[mode])
    row=dict(stage=stage+1,current_samples=len(ids),fit_s_shared=fit,macro_accuracy=macro(np.concatenate(ys),prediction),scores=scores,persistent_arrays_bytes=memory,completion=diag,old_training_prediction_control=True,base_fecam_prediction_control=mode=='base')
    records.setdefault(name,[]).append(row);np.savez_compressed(folder/f'predictions_stage{stage+1}.npz',**arrays)
    with (folder/'stages.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  print(stage+1,'completed',flush=True)
 source=(old/oldsummary['shared_state']).resolve();assert sha(source)==oldsummary['shared_state_sha256'];ck=torch.load(source,map_location='cpu',weights_only=False);original=ck['learner']
 for key in ('g','k'):np.testing.assert_array_equal(getattr(m,key),getattr(original,key))
 for p,q in zip(m.function.net.parameters(),original.function.net.parameters()):torch.testing.assert_close(p,q,rtol=0,atol=0)
 for label in m.local.cells:
  for left,right in zip(m.local.cells[label],original.local.cells[label]):np.testing.assert_array_equal(left.points,right.points)
 assert pickle.dumps(m.rng.bit_generator.state)==pickle.dumps(original.rng.bit_generator.state)
 shared=out/'_shared'/f'{group}.pt';shared.parent.mkdir(exist_ok=True);torch.save(dict(unit_memory=unit_memory,source_checkpoint=str(source),source_sha256=oldsummary['shared_state_sha256']),shared);digest=sha(shared)
 for name,stages in records.items():
  mode=name[len(group)+1:].split('_a')[0];alpha=float(name.rsplit('_a',1)[1])
  s=dict(status='completed',task='classification',config=dict(dataset=a.dataset,encoder=a.encoder,mode=mode,alpha=alpha,projection_regularization=.01,seed=42,order=17,eval_split='val'),order=[g.tolist() for g in order],stages=stages,final_macro_accuracy=stages[-1]['macro_accuracy'],average_macro_accuracy=float(np.mean([r['macro_accuracy'] for r in stages])),input_hashes={split:sha(root/'features'/a.encoder/f'{a.dataset}_{split}.npz') for split in ('train','val')},shared_state='../_shared/'+shared.name,shared_state_sha256=digest,git_commit=os.environ['TMOT_SOURCE_COMMIT'],elapsed_s_shared=time.monotonic()-start,final_training_state_exact=True)
  assert s['input_hashes']==oldsummary['input_hashes'];(out/name/'summary.json').write_text(json.dumps(s,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);p.add_argument('--dataset',required=True);p.add_argument('--encoder',required=True);run(p.parse_args())
