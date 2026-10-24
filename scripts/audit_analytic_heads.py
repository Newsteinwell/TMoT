import json,hashlib,sys
from pathlib import Path
import numpy as np
import torch
from tmot.fecam import unit_rows
root=Path(sys.argv[1]);folder=root/'runs/analytic_heads_v60';torch.set_num_threads(1)
complete=json.loads((folder/'complete.json').read_text());assert len(complete)==12 and all(v['status']=='completed' for v in complete)
checked=[];controls=[];rows=[]
for shared in sorted((folder/'_shared').glob('*.pt')):
 ck=torch.load(shared,map_location='cpu',weights_only=False);paths=sorted(folder.glob(shared.stem+'_*/summary.json'));assert len(paths)==13
 s=json.loads(paths[0].read_text());c=s['config'];cl=ck['classes'];tr=np.load(root/'features'/c['encoder']/f"{c['dataset']}_train.npz");val=np.load(root/'features'/c['encoder']/f"{c['dataset']}_val.npz")
 def transform(x):
  z=ck['basis'].transform(x)
  return unit_rows(z) if ck['normalization']=='unit' else z
 # Recompute in original task batches, preserving transform floating-point behavior.
 x=[];y=[]
 for group in s['order']:
  ids=np.flatnonzero(np.isin(tr['y'],group));x.append(transform(tr['x'][ids]));y.append(tr['y'][ids])
 x=np.concatenate(x).astype(np.float64);y=np.concatenate(y);means=[];scatter=np.zeros((512,512))
 for label in cl:
  xx=x[y==label];z=np.c_[xx,np.ones(len(xx))];cell=ck['memory'].cells[label]
  assert cell.n==len(xx);np.testing.assert_allclose(cell.total,z.sum(0),rtol=1e-12,atol=1e-10);np.testing.assert_allclose(cell.gram,z.T@z,rtol=1e-10,atol=1e-8)
  mu=xx.mean(0);means.append(mu);scatter+=(xx-mu).T@(xx-mu)
 means=np.array(means);cov=scatter/(len(x)-len(cl));scale=max(float(np.trace(cov)/512),1e-8)
 digest=hashlib.sha256(shared.read_bytes()).hexdigest()
 for path in paths:
  s=json.loads(path.read_text());c=s['config'];h=ck['heads'][c['setting']];assert s['shared_state_sha256']==digest
  for split,expected in s['input_hashes'].items():assert hashlib.sha256((root/'features'/c['encoder']/f"{c['dataset']}_{split}.npz").read_bytes()).hexdigest()==expected
  if c['method']=='slda':
   shrink=h['shrink'];precision=np.linalg.inv((1-shrink)*cov+shrink*scale*np.eye(512));np.testing.assert_allclose(precision,h['precision'],rtol=1e-6,atol=1e-5)
  elif c['method']=='ridge':
   z=np.c_[x,np.ones(len(x))];weights=np.array([1/(len(cl)*np.sum(y==v)) for v in y]);target=np.eye(len(cl))[[cl.index(v) for v in y]];g=z.T@(weights[:,None]*z);k=z.T@(weights[:,None]*target);p=np.eye(513)*h['ridge'];p[-1,-1]=0
   np.testing.assert_allclose(h['weight'],np.linalg.solve((g+g.T)/2+p,k),rtol=1e-6,atol=1e-7)
  for stage in s['stages']:
   with np.load(path.parent/f"predictions_stage{stage['stage']}.npz") as pred:
    ys=[];ps=[]
    for i in range(stage['stage']):
     truth=pred[f'y_{i}'];prediction=pred[f'pred_{i}'];ys.append(truth);ps.append(prediction)
     score=np.mean([np.mean(prediction[truth==v]==v) for v in np.unique(truth)]);np.testing.assert_allclose(score,stage['scores'][i],atol=1e-12)
     if stage['stage']==len(s['order']):
      ids=np.flatnonzero(np.isin(val['y'],s['order'][i]));vx=transform(val['x'][ids]).astype(np.float64)
      if c['method']=='ridge':score=np.c_[vx,np.ones(len(vx))]@h['weight']
      elif c['method']=='ncm':score=-np.sum((vx[:,None,:]-means[None,:,:])**2,axis=2)
      else:score=np.column_stack([-np.einsum('ni,ni->n',(vx-mu)@precision,vx-mu) for mu in means])
      np.testing.assert_array_equal(prediction,np.asarray(cl)[score.argmax(1)]);np.testing.assert_array_equal(truth,val['y'][ids]);np.testing.assert_array_equal(pred[f'ids_{i}'],val['ids'][ids])
    yy=np.concatenate(ys);pp=np.concatenate(ps);np.testing.assert_allclose(stage['macro_accuracy'],np.mean([np.mean(pp[yy==v]==v) for v in np.unique(yy)]),atol=1e-12)
   if c['method']=='ridge' and c['normalization']=='raw':
    old=root/'runs/harder_validation_v45'/f"{c['dataset']}_{c['encoder']}_linear_r{h['ridge']:g}_o17_s42"/f"predictions_stage{stage['stage']}.npz"
    with np.load(old) as a,np.load(path.parent/old.name) as b:
     for key in a.files:np.testing.assert_array_equal(a[key],b[key])
  if c['method']=='ridge' and c['normalization']=='raw':controls.append(path.parent.name)
  np.testing.assert_allclose(s['final_macro_accuracy'],s['stages'][-1]['macro_accuracy']);np.testing.assert_allclose(s['average_macro_accuracy'],np.mean([v['macro_accuracy'] for v in s['stages']]))
  checked.append(path.parent.name);rows.append(s)
 print(shared.stem,'verified',flush=True)
assert len(checked)==156 and len(controls)==36
(root/'analytic_heads_v60_audit.json').write_text(json.dumps(dict(complete=True,checked=checked,old_controls=controls,shared_states=12,checks=['all-stage metrics','final direct TRAIN moments','independent final predictions','ridge normal equations','SLDA direct centered covariance','hashes','raw ridge v45 exact predictions']),indent=2))
(root/'analytic_heads_v60_runs.json').write_text(json.dumps(rows,indent=2))
