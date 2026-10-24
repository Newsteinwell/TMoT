"""Independent saved-prediction audit, controls, and final checkpoint replay."""
import json,sys,hashlib
from pathlib import Path
import numpy as np
import torch

def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def metric(y,p,cl):return float(np.mean([np.mean(p[y==c]==c) for c in np.unique(y)])) if cl else float(np.sqrt(np.mean((y-p)**2)))
root=Path(sys.argv[1]);queue=root/'runs/sequential_head_v73';torch.set_num_threads(1);reports=[]
for path in sorted(queue.glob('*/summary.json')):
 s=json.loads(path.read_text());a=s['config'];cl=s['task']=='classification';d=a['dataset'];e=a['encoder'];metric_name='macro_accuracy' if cl else 'rmse'
 for split in ('train','val'):assert sha(root/'features'/e/f'{d}_{split}.npz')==s['input_hashes'][split]
 va=np.load(root/'features'/e/f'{d}_val.npz');lookup={str(v):i for i,v in enumerate(va['ids'])}
 old=root/f'runs/warm_budget_v58_retry/{d}_{e}_warm_er10_o17_s42_r0/predictions_stage1.npz'
 control=np.load(old);first=np.load(path.parent/'predictions_stage1.npz')
 for k in control.files:np.testing.assert_array_equal(control[k],first[k])
 model=torch.load(path.parent/'final.pt',map_location='cpu',weights_only=False)
 values=[]
 for row in s['stages']:
  p=np.load(path.parent/f"predictions_stage{row['stage']}.npz");ys=[];ps=[]
  assert row['historical_training_samples']==row['stored_examples']==0 and row['warm_init_error']==0
  for k in range(row['stage']):
   inds=np.array([lookup[str(v)] for v in p[f'ids_{k}']]);truth=va['y'][inds];pred=p[f'pred_{k}'];np.testing.assert_array_equal(truth,p[f'y_{k}']);ys.append(truth);ps.append(pred)
   np.testing.assert_allclose(metric(truth,pred,cl),row['scores'][k],rtol=0,atol=1e-12)
   if row['stage']==len(s['stages']):
    actual=model['model'].predict_head(model['basis'].transform(va['x'][inds]));actual=actual if cl else actual*model['target_sd']+model['target_mean'];np.testing.assert_array_equal(actual,pred)
  value=metric(np.concatenate(ys),np.concatenate(ps),cl);np.testing.assert_allclose(value,row[metric_name],rtol=0,atol=1e-12);values.append(value)
 np.testing.assert_allclose([values[-1],np.mean(values)],[s['final_'+metric_name],s['average_'+metric_name]],rtol=0,atol=1e-12)
 reports.append(dict(dataset=d,encoder=e,strength=a['strength'],final=values[-1],average=float(np.mean(values)),stages=len(values),source=s['git_commit'],control_first_stage_exact=True,final_checkpoint_exact=True))
assert len(reports)==48
(queue/'audit.json').write_text(json.dumps(dict(status='passed',runs=reports),indent=2));print('48 trajectories passed')
