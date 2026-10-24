"""Check retained rows against source TRAIN; rederive saved metrics and final predictions."""
import json,hashlib,sys
from pathlib import Path
import numpy as np
import torch
root=Path(sys.argv[1]);folder=root/'runs/task_budget_v57';torch.set_num_threads(1)
terminal=json.loads((folder/'complete.json').read_text());assert len(terminal)==24 and all(v['status']=='completed' for v in terminal)
checked=[];summaries=[]
for shared in sorted((folder/'_shared').glob('*.pt')):
 ck=torch.load(shared,map_location='cpu',weights_only=False);m=ck['learner'];paths=sorted(folder.glob(shared.stem+'_r*/summary.json'));assert len(paths)==6
 s=json.loads(paths[0].read_text());c=s['config'];cl=s['task']=='classification'
 train=np.load(root/'features'/c['encoder']/(c['dataset']+'_train.npz'));val=np.load(root/'features'/c['encoder']/(c['dataset']+'_val.npz'))
 xx=[];yy=[];old=0
 for group,row in zip(s['order'],m.buffer_audit):
  inds=np.flatnonzero(np.isin(train['y'],group)) if cl else np.flatnonzero(train['groups']==group)
  ids=np.asarray(row['selected_indices'],int);assert row['current_samples']==len(inds) and len(ids)==len(inds)//10
  assert len(np.unique(ids))==len(ids) and np.all((ids>=0)&(ids<len(inds)))
  assert row['before']==old;old+=len(ids);assert row['after']==old
  xx.append(ck['basis'].transform(train['x'][inds])[ids]);y=train['y'][inds]
  if not cl:y=(y-ck['target_mean'])/ck['target_sd']
  yy.append(y[ids])
 np.testing.assert_array_equal(m.buffer[0],np.concatenate(xx));np.testing.assert_array_equal(m.buffer[1],np.concatenate(yy))
 assert len(m.buffer[0])==old
 if c['variant']=='er10':assert not hasattr(m,'exact_memory')
 digest=hashlib.sha256(shared.read_bytes()).hexdigest()
 for path in paths:
  s=json.loads(path.read_text());assert s['shared_state_sha256']==digest
  leaf=torch.load(path.parent/'final.pt',map_location='cpu',weights_only=False);m.readout=leaf['readout']
  assert leaf['shared_state_sha256']==digest
  for stage in s['stages']:
   assert stage['buffer_audit']==m.buffer_audit[stage['stage']-1]
   assert stage['historical_training_samples']==(4096 if stage['buffer_audit']['before'] else 0)
   with np.load(path.parent/f"predictions_stage{stage['stage']}.npz") as z:
    ys=[];ps=[]
    for k in range(stage['stage']):
     y=z[f'y_{k}'];p=z[f'pred_{k}'];ys.append(y);ps.append(p)
     score=float(np.mean([np.mean(p[y==c]==c) for c in np.unique(y)])) if cl else float(np.sqrt(np.mean((y-p)**2)))
     np.testing.assert_allclose(score,stage['scores'][k],rtol=1e-12,atol=1e-12)
     if stage['stage']==len(s['order']):
      group=s['order'][k];inds=np.flatnonzero(np.isin(val['y'],group)) if cl else np.flatnonzero(val['groups']==group)
      pred=m.predict(ck['basis'].transform(val['x'][inds]));pred=pred if cl else pred*ck['target_sd']+ck['target_mean']
      np.testing.assert_array_equal(pred,p);np.testing.assert_array_equal(val['ids'][inds],z[f'ids_{k}']);np.testing.assert_array_equal(val['y'][inds],y)
    y=np.concatenate(ys);p=np.concatenate(ps);metric='macro_accuracy' if cl else 'rmse'
    score=float(np.mean([np.mean(p[y==c]==c) for c in np.unique(y)])) if cl else float(np.sqrt(np.mean((y-p)**2)))
    np.testing.assert_allclose(score,stage[metric],rtol=1e-12,atol=1e-12)
  np.testing.assert_allclose(s['final_'+metric],s['stages'][-1][metric]);np.testing.assert_allclose(s['average_'+metric],np.mean([r[metric] for r in s['stages']]))
  summaries.append(s);checked.append(path.parent.name)
 print(shared.stem,'verified',flush=True)
assert len(checked)==144
(root/'task_budget_v57_audit.json').write_text(json.dumps(dict(complete=True,checked=checked,shared_states=24,checks=['task quotas','selected source TRAIN rows','no duplicate selection','zero-buffer replay','all-stage metrics','final predictions','checkpoint hashes','ordinary ER has no exact_memory']),indent=2))
(root/'task_budget_v57_runs.json').write_text(json.dumps(summaries,indent=2))
