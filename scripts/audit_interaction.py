import concurrent.futures,json,sys
from pathlib import Path
import numpy as np
import torch
from tmot.fecam import FeCAMLearner
from tmot.moments import augment
from tmot.experiment_synthetic_capacity import sha,macro

torch.set_num_threads(1)
root=Path(sys.argv[1]);queue=root/'runs/interaction_v81'
done=json.loads((queue/'complete.json').read_text());assert len(done)==20 and all(s['status']=='completed' for s in done)

def fecam_from_memory(memory,classes,alpha):
 f=FeCAMLearner(memory.dimension,alpha);f.memory=memory;f.classes=classes.copy()
 for c in classes:
  cell=memory.cells[c];mean=cell.total[:-1]/cell.n
  cov=(cell.gram[:-1,:-1]-cell.n*np.outer(mean,mean))/max(cell.n-1,1);cov=(cov+cov.T)/2
  scale=max(float(np.trace(cov)/memory.dimension),1e-12);cov+=alpha*scale*np.eye(memory.dimension)
  sd=np.sqrt(np.maximum(np.diag(cov),1e-15));f.means[c]=mean;f.precisions[c]=np.linalg.pinv(cov/np.outer(sd,sd),hermitian=True)
 return f

def audit_group(folder):
 rows=[json.loads(f.read_text()) for f in sorted(folder.glob('*/summary.json'))];assert len(rows) in (12,18)
 state=torch.load(folder/'shared.pt',map_location='cpu',weights_only=False);h=sha(folder/'shared.pt');c=rows[0]['config'];cl=rows[0]['task']=='classification'
 v=np.load(root/'features'/c['encoder']/f"{c['dataset']}_val.npz")
 with np.load(root/'prepared'/f"{c['dataset']}_val.npz") as raw:
  np.testing.assert_array_equal(raw['ids'],v['ids']);np.testing.assert_array_equal(raw['y'],v['y']);rawx=raw['x'].copy()
 for split in ('train','val'):
  assert sha(root/'features'/c['encoder']/f"{c['dataset']}_{split}.npz")==rows[0]['input_hashes'][split]
  assert sha(root/'prepared'/f"{c['dataset']}_{split}.npz")==rows[0]['raw_hashes'][split]
 index={str(v):i for i,v in enumerate(v['ids'])};assert len(index)==len(v['ids'])
 for s in rows:
  c=s['config'];assert s['shared_sha256']==h;dest=folder/f"{c['method']}_{c['param']:g}"
  for row in s['stages']:
   stage=row['stage'];ys=[];ps=[]
   with np.load(dest/f'predictions_stage{stage}.npz') as p:
    for j in range(stage):
     ix=np.array([index[str(i)] for i in p[f'ids_{j}']]);np.testing.assert_array_equal(p[f'y_{j}'],v['y'][ix]);ys.append(p[f'y_{j}']);ps.append(p[f'pred_{j}'])
   y=np.concatenate(ys);pred=np.concatenate(ps);metric=macro(y,pred) if cl else float(np.sqrt(np.mean((y-pred)**2)))
   assert row['metric']==metric
  assert s['final']==s['stages'][-1]['metric'] and s['average']==float(np.mean([r['metric'] for r in s['stages']]))
  if c['method']=='task':state['learner'].readout=state['weights'][('task',c['param'])];learner=state['learner']
  elif c['method']=='fecam':learner=fecam_from_memory(state['fecam_memory'],state['classes'],c['param'])
  else:learner=None
  with np.load(dest/f"predictions_stage{len(s['stages'])}.npz") as p:
   for j in range(len(s['stages'])):
    ix=np.array([index[str(i)] for i in p[f'ids_{j}']]);z=state['basis'].transform(v['x'][ix],rawx[ix]);z=state['reference'].features(z,rawx[ix,0,-1,None])
    if learner is not None:pred=learner.predict(z)
    else:
     score=augment(z)@state['weights'][('ridge',c['param'])];pred=np.asarray(state['classes'])[score.argmax(1)] if cl else score
    if not cl:pred=pred*state['target_sd']+state['target_mean']+state['reference'].offset(rawx[ix,0,-1,None])
    np.testing.assert_array_equal(pred,p[f'pred_{j}'])
 print(folder.name,'verified',flush=True);return rows

folders=sorted(p.parent for p in queue.glob('*/shared.pt'));assert len(folders)==20
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:groups=list(pool.map(audit_group,folders))
rows=[r for group in groups for r in group];assert len(rows)==312
report=dict(complete=True,trajectories=len(rows),stages=sum(len(r['stages']) for r in rows),exact_final_predictions=True,runs=rows)
(queue/'audit.json').write_text(json.dumps(report,indent=2));print('Audit complete',flush=True)
