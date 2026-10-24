"""Final frozen-function diagnostic. Full-TRAIN branch is NOT a CL algorithm."""
import argparse,hashlib,json,os,pickle,time
from pathlib import Path
import numpy as np
import torch
from tmot.weighted_class_distance import class_statistics,precision,distances
from tmot.moments import augment
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('dataset');p.add_argument('encoder');p.add_argument('out',type=Path);a=p.parse_args();torch.set_num_threads(1);a.out.mkdir(parents=True,exist_ok=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
selection=json.loads((a.root/'synthetic_capacity_v54_comparison.json').read_text())['selected'];ref=next(r for r in selection if (r['dataset'],r['encoder'],r['variant'])==(a.dataset,a.encoder,'task_n256'))
source=a.root/'runs'/ref['source_queue']/ref['source_run'];s=json.loads((source/'summary.json').read_text());shared=(source/s['shared_state']).resolve();assert sha(shared)==s['shared_state_sha256'];ck=torch.load(shared,map_location='cpu',weights_only=False);m=ck['learner'];assert m.task=='classification'
def learner_digest():
 h=hashlib.sha256()
 for tensor in m.function.net.state_dict().values():h.update(tensor.cpu().numpy().tobytes())
 for arr in (m.function.mean,m.function.sd,m.function.hidden_mean):h.update(arr.tobytes())
 h.update(str(m.function.hidden_scale).encode())
 for cells in m.local.cells.values():
  for cell in cells:h.update(cell.points.tobytes())
 return h.hexdigest()
state_before=learner_digest()
rng_before=pickle.dumps((m.rng.bit_generator.state,[(c.point_rng.bit_generator.state) for cells in m.local.cells.values() for c in cells]))
tr=np.load(a.root/'features'/a.encoder/f'{a.dataset}_train.npz');val=np.load(a.root/'features'/a.encoder/f'{a.dataset}_val.npz')
for split in ('train','val'):assert sha(a.root/'features'/a.encoder/f'{a.dataset}_{split}.npz')==s['input_hashes'][split]
parts=[np.flatnonzero(np.isin(tr['y'],g)) for g in s['order']];evals=[np.flatnonzero(np.isin(val['y'],g)) for g in s['order']]
x=np.concatenate([ck['basis'].transform(tr['x'][ids]) for ids in parts]);y=np.concatenate([tr['y'][ids] for ids in parts]);assert len(y)==m.seen
vxparts=[ck['basis'].transform(val['x'][ids]) for ids in evals];vx=np.concatenate(vxparts);vy=np.concatenate([val['y'][ids] for ids in evals]);vid=np.concatenate([val['ids'][ids] for ids in evals])
leaf=torch.load(source/'final.pt',map_location='cpu',weights_only=False);oldpred=np.concatenate([np.asarray(m.classes)[(augment(m.function.transform(z))@leaf['readout']).argmax(1)] for z in vxparts])
with np.load(source/f"predictions_stage{len(s['order'])}.npz") as saved:
 np.testing.assert_array_equal(oldpred,np.concatenate([saved[f'pred_{i}'] for i in range(len(evals))]));np.testing.assert_array_equal(vid,np.concatenate([saved[f'ids_{i}'] for i in range(len(evals))]))
def transform(z,space):
 if space=='input':return z
 f=m.function.transform(z)
 return f if space=='joint' else f[:,m.dimension:]
rows=[];errors=[];start=time.monotonic()
for space in ('input','hidden','joint'):
 vv=transform(vx,space)
 for origin in ('synthetic','real_diagnostic'):
  stats=[]
  for label in m.classes:
   n=int(np.sum(y==label))
   if origin=='synthetic':
    cells=m.local.cells[label];xx=np.concatenate([c.points for c in cells]);w=np.concatenate([np.full(len(c.points),c.n/len(c.points)) for c in cells]);assert sum(c.n for c in cells)==n
   else:xx=x[y==label];w=np.ones(n)
   mu,cov,error=class_statistics(transform(xx,space),w,n);stats.append((mu,cov));errors.append(dict(space=space,origin=origin,label=int(label),mean=mu.tolist(),covariance_centering_error=error,population=n))
  for alpha in (.01,.1,.5,1.,2.,10.):
   ds=np.column_stack([distances(vv,mu,precision(cov,alpha)) for mu,cov in stats]);pred=np.asarray(m.classes)[ds.argmin(1)];metric=float(np.mean([np.mean(pred[vy==c]==c) for c in np.unique(vy)]))
   control=False
   if origin=='real_diagnostic' and space=='input':
    old=a.root/'runs/fecam_harder_v59'/f'{a.dataset}_{a.encoder}_fecam_a{alpha:g}_o17_s42'
    with np.load(old/f"predictions_stage{len(s['order'])}.npz") as saved:
     np.testing.assert_array_equal(pred,np.concatenate([saved[f'pred_{i}'] for i in range(len(evals))]));control=True
   name=f'{space}_{origin}_a{alpha:g}';np.savez_compressed(a.out/(name+'.npz'),pred=pred,y=vy,ids=vid,distance=ds)
   rows.append(dict(dataset=a.dataset,encoder=a.encoder,space=space,origin=origin,alpha=alpha,final_macro_accuracy=metric,original_tmot=ref['final'],fecam_control_exact=control,artifact=name+'.npz'))
  print(space,origin,'complete',flush=True)
assert rng_before==pickle.dumps((m.rng.bit_generator.state,[(c.point_rng.bit_generator.state) for cells in m.local.cells.values() for c in cells]));assert sha(shared)==s['shared_state_sha256'];assert learner_digest()==state_before
report=dict(complete=True,source_commit=os.environ['TMOT_SOURCE_COMMIT'],source_checkpoint=str(shared),source_checkpoint_sha256=s['shared_state_sha256'],input_hashes=s['input_hashes'],old_prediction_control=True,rng_unchanged=True,network_and_points_unchanged=True,learner_digest=state_before,elapsed_s=time.monotonic()-start,rows=rows,statistics=errors,note='Final-stage diagnostic only; real_diagnostic accesses all arrived TRAIN and is not a memory-limited CL method. Synthetic branch reads retained points with per-cell population weights; no learner updates.')
(a.out/'report.json').write_text(json.dumps(report,indent=2))
