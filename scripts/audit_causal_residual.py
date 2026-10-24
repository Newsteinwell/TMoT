"""Recompute every metric, replay quota, control and final checkpoint prediction."""
import json,sys,os
from pathlib import Path
import numpy as np
import torch
from tmot.experiment_synthetic_robustness import sha
from tmot.moments import augment

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
root=Path(sys.argv[1]);q=root/'runs'/(sys.argv[2] if len(sys.argv)>2 else 'causal_residual_v79')
complete=json.loads((q/'complete.json').read_text());assert len(complete)==48 and all(x['status']=='completed' for x in complete)
summaries=[];initial={}
for f in sorted(q.glob('*/summary.json')):
 s=json.loads(f.read_text());c=s['config'];folder=f.parent
 assert s['status']=='completed' and len(s['stages'])==5
 tr=np.load(root/'features'/c['encoder']/f"{c['dataset']}_train.npz");v=np.load(root/'features'/c['encoder']/f"{c['dataset']}_val.npz")
 for split in ('train','val'):assert sha(root/'features'/c['encoder']/f"{c['dataset']}_{split}.npz")==s['input_hashes'][split]
 order=sorted(np.unique(tr['groups']).tolist());quota=0;parts=[np.flatnonzero(tr['groups']==g) for g in order]
 for stage,row in enumerate(s['stages'],1):
  oldquota=quota;quota+=len(parts[stage-1])//10
  expected=0 if stage==1 or (c['mode']=='er10' and not oldquota) else 4096
  assert row['historical_training_samples']==expected
  assert row['optimizer_steps']==30*((len(parts[stage-1])+expected+255)//256)
  if c.get('initialization','warm')=='cold':assert row['warm_init_error'] is None
  elif stage>1:assert row['warm_init_error']<1e-4
  if c['mode']=='er10':
   a=row['buffer_audit'];assert a['quota']==len(parts[stage-1])//10 and a['after']==quota and a['before']==oldquota
   assert len(set(a['selected_indices']))==a['quota'] and all(0<=i<a['current_samples'] for i in a['selected_indices'])
   assert row['buffer_count']==quota
  else:assert row['buffer_count']==0
  with np.load(folder/f'predictions_stage{stage}.npz') as p:
   ys=[];ps=[];bs=[]
   for j,g in enumerate(order[:stage]):
    ix=np.flatnonzero(v['groups']==g);np.testing.assert_array_equal(p[f'y_{j}'],v['y'][ix]);np.testing.assert_array_equal(p[f'ids_{j}'],v['ids'][ix])
    ys.append(p[f'y_{j}']);ps.append(p[f'pred_{j}']);bs.append(p[f'base_{j}'])
   peer=root/'runs/causal_task_v68'/f"{c['dataset']}_{c['encoder']}_{c['reference']}_ridge_r{c['ridge']:g}"
   with np.load(peer/f'predictions_stage{stage}.npz') as old:
    for j in range(stage):np.testing.assert_array_equal(p[f'base_{j}'],old[f'pred_{j}'])
   y=np.concatenate(ys);pred=np.concatenate(ps);base=np.concatenate(bs)
   assert abs(row['rmse']-float(np.sqrt(np.mean((pred-y)**2))))<1e-12
   assert abs(row['base_rmse']-float(np.sqrt(np.mean((base-y)**2))))<1e-12
   if stage==1:
    key=(c['dataset'],c['encoder'],c['reference'],c['penalty'])
    if key in initial:np.testing.assert_array_equal(pred,initial[key])
    else:initial[key]=pred.copy()
  if c.get('initialization','warm')=='cold':
   warm_folder=root/'runs/causal_residual_v79'/folder.name
   ws=json.loads((warm_folder/'summary.json').read_text());wr=ws['stages'][stage-1]
   assert all(row[k]==wr[k] for k in ('current_samples','historical_training_samples','optimizer_steps','buffer_count'))
   if c['mode']=='er10':assert row['buffer_audit']==wr['buffer_audit']
   with np.load(warm_folder/f'predictions_stage{stage}.npz') as wp,np.load(folder/f'predictions_stage{stage}.npz') as cp:
    for k in cp.files:
     if stage==1 or not k.startswith('pred_'):np.testing.assert_array_equal(cp[k],wp[k])
 assert s['final_rmse']==s['stages'][-1]['rmse']
 assert abs(s['average_rmse']-np.mean([r['rmse'] for r in s['stages']]))<1e-12
 assert sha(folder/'final.pt')==s['checkpoint_sha256']
 state=torch.load(folder/'final.pt',map_location='cuda',weights_only=False);m=state['learner'];m.device='cuda'
 with np.load(root/'prepared'/f"{c['dataset']}_val.npz") as raw:
  np.testing.assert_array_equal(raw['ids'],v['ids']);past=raw['x'][:,0,-1,None]
  with np.load(folder/'predictions_stage5.npz') as p:
   for j,g in enumerate(order):
    ix=np.flatnonzero(v['groups']==g)
    z=state['reference'].features(state['basis'].transform(v['x'][ix]),past[ix])
    pred=m.predict(z)*state['target_sd']+(state['target_mean']+state['reference'].offset(past[ix]))
    np.testing.assert_array_equal(pred,p[f'pred_{j}'])
 # Retrospective TRAIN diagnostic only; never feeds optimization or selection.
 with np.load(root/'prepared'/f"{c['dataset']}_train.npz") as raw:
  np.testing.assert_array_equal(raw['ids'],tr['ids']);past=raw['x'][:,0,-1,None]
  diagnostic={}
  for split,ix in [('old',np.flatnonzero(tr['groups']!=order[-1])),('current',parts[-1])]:
   z=state['reference'].features(state['basis'].transform(tr['x'][ix]),past[ix]);offset=state['target_mean']+state['reference'].offset(past[ix])
   pred=m.predict(z)*state['target_sd']+offset;base=(augment(z)@m.base)*state['target_sd']+offset;y=tr['y'][ix]
   diagnostic[split]=dict(base_rmse=float(np.sqrt(np.mean((base-y)**2))),residual_rmse=float(np.sqrt(np.mean((pred-y)**2))))
  s['retrospective_train_diagnostic']=diagnostic
 summaries.append(s)
 print(folder.name,'verified',flush=True)
result=dict(complete=True,trajectories=len(summaries),first_stage_pairs=len(initial),stages=240,checkpoint_predictions_exact=True,runs=summaries)
(q/'audit.json').write_text(json.dumps(result,indent=2));print('Audit complete',flush=True)
