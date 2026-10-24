import json,sys
from pathlib import Path
import numpy as np
from tmot.experiment_synthetic_robustness import sha
root=Path(sys.argv[1]);folder=root/'runs/prequential_reference_v70';terminal=json.loads((folder/'complete.json').read_text());assert len(terminal)==4 and all(r['status']=='completed' for r in terminal);rows=[]
for path in sorted(folder.glob('*/summary.json')):
 s=json.loads(path.read_text());c=s['config'];d,e=c['dataset'],c['encoder'];group=f'{d}_{e}';method=c['method'];ridge=c['ridge'];tr=np.load(root/'features'/e/f'{d}_train.npz');val=np.load(root/'features'/e/f'{d}_val.npz');total=np.zeros(3);count=0
 for split in ('train','val'):assert sha(root/'features'/e/f'{d}_{split}.npz')==s['input_hashes'][split]
 for stage in s['stages']:
  t=stage['stage']
  if t>1:
   with np.load(folder/f'{group}_pre_stage{t}_{method}_r{ridge:g}.npz') as pre:
    ix=np.flatnonzero(tr['groups']==s['order'][t-1]);np.testing.assert_array_equal(pre['ids'],tr['ids'][ix]);np.testing.assert_array_equal(pre['y'],tr['y'][ix]);assert pre['predictions'].shape==(3,*pre['y'].shape)
    for expert in range(3):total[expert]+=float(np.square(pre['predictions'][expert]-pre['y']).sum())
    count+=pre['y'].size
  np.testing.assert_allclose(total,stage['prequential_sse'],rtol=1e-12,atol=1e-12);assert count==stage['prequential_count']
  if c['policy']=='uniform' or count==0:w=np.ones(3)/3
  elif np.any(total==0):w=(total==0)/np.sum(total==0)
  else:w=(1/total)/(1/total).sum()
  np.testing.assert_allclose(w,stage['weights'],rtol=1e-12,atol=1e-12)
  with np.load(path.parent/f'predictions_stage{t}.npz') as data:
   ys=[];ps=[]
   for j,g in enumerate(s['order'][:t]):
    ix=np.flatnonzero(val['groups']==g);np.testing.assert_array_equal(data[f'y_{j}'],val['y'][ix]);np.testing.assert_array_equal(data[f'ids_{j}'],val['ids'][ix])
    for expert,name in enumerate(('none','append','residual')):
     peer=root/'runs/causal_task_v68'/f'{d}_{e}_{name}_{method}_r{ridge:g}'
     with np.load(peer/f'predictions_stage{t}.npz') as old:np.testing.assert_array_equal(data[f'experts_{j}'][expert],old[f'pred_{j}'])
    p=sum(w[k]*data[f'experts_{j}'][k] for k in range(3));np.testing.assert_allclose(p,data[f'pred_{j}'],rtol=1e-12,atol=1e-12);ys.append(data[f'y_{j}']);ps.append(p);np.testing.assert_allclose(np.sqrt(np.mean((p-ys[-1])**2)),stage['scores'][j],rtol=1e-12,atol=1e-12)
   np.testing.assert_allclose(np.sqrt(np.mean((np.concatenate(ps)-np.concatenate(ys))**2)),stage['rmse'],rtol=1e-12,atol=1e-12)
 np.testing.assert_allclose(s['final_rmse'],s['stages'][-1]['rmse']);np.testing.assert_allclose(s['average_rmse'],np.mean([v['rmse'] for v in s['stages']]))
 rows.append(s)
assert len(rows)==96
selected=[]
for key in sorted({tuple(s['config'][k] for k in ('dataset','encoder','policy','method')) for s in rows}):
 choices=[s for s in rows if tuple(s['config'][k] for k in ('dataset','encoder','policy','method'))==key];best=min(choices,key=lambda s:(s['final_rmse'],s['config']['ridge']));selected.append(best);print(*key,best['final_rmse'],best['average_rmse'],best['config']['ridge'],flush=True)
(root/'prequential_reference_v70_runs.json').write_text(json.dumps(rows,indent=2))
(root/'prequential_reference_v70_comparison.json').write_text(json.dumps(dict(complete=True,trajectories=96,selected=selected,checks=['current TRAIN identities','independent cumulative SSE and weights','independent prediction mixture','exact v68 branch controls','all-stage metrics and sample identities','input hashes'],note='Single-seed development experiment; first-task uniform; 3 experts per method; no TEST.'),indent=2))
