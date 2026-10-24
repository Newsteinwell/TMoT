"""Replay an alternative fixed decision rule over unchanged expert predictions."""
import json,os,sys
from pathlib import Path
import numpy as np
from tmot.hard_prequential import hard_weights
from tmot.experiment_synthetic_robustness import sha
root=Path(sys.argv[1]);source=root/'runs/prequential_reference_v70';out=root/'runs/hard_prequential_v71';out.mkdir(exist_ok=False)
assert json.loads((root/'prequential_reference_v70_comparison.json').read_text())['complete']
rows=[];inputs={}
for sp in sorted(source.glob('*_prequential_*/summary.json')):
 s=json.loads(sp.read_text());c=s['config'];d,e=c['dataset'],c['encoder'];method=c['method'];ridge=c['ridge'];group=f'{d}_{e}';folder=out/f'{group}_hard_{method}_r{ridge:g}';folder.mkdir(exist_ok=False)
 tr=np.load(root/'features'/e/f'{d}_train.npz');val=np.load(root/'features'/e/f'{d}_val.npz');assert sha(root/'features'/e/f'{d}_train.npz')==s['input_hashes']['train'];assert sha(root/'features'/e/f'{d}_val.npz')==s['input_hashes']['val'];total=np.zeros(3);count=0;stages=[];inputs[str(sp)]=sha(sp)
 for old in s['stages']:
  t=old['stage']
  if t>1:
   path=source/f'{group}_pre_stage{t}_{method}_r{ridge:g}.npz';inputs[str(path)]=sha(path)
   with np.load(path) as pre:
    ix=np.flatnonzero(tr['groups']==s['order'][t-1]);np.testing.assert_array_equal(pre['ids'],tr['ids'][ix]);np.testing.assert_array_equal(pre['y'],tr['y'][ix]);total+=np.sum((pre['predictions']-pre['y'][None])**2,axis=(1,2));count+=pre['y'].size
  np.testing.assert_allclose(total,old['prequential_sse'],atol=1e-12,rtol=1e-12);assert count==old['prequential_count']
  weights=hard_weights(total,count)
  # Independent scalar selection, fixed REF order none/append/residual.
  chosen=None if count==0 else min(range(3),key=lambda i:(float(total[i]),i))
  arrays={};ys=[];ps=[];scores=[];path=sp.parent/f'predictions_stage{t}.npz';inputs[str(path)]=sha(path)
  with np.load(path) as saved:
   for j,g in enumerate(s['order'][:t]):
    ix=np.flatnonzero(val['groups']==g);np.testing.assert_array_equal(saved[f'y_{j}'],val['y'][ix]);np.testing.assert_array_equal(saved[f'ids_{j}'],val['ids'][ix]);experts=saved[f'experts_{j}'];p=np.tensordot(weights,experts,axes=(0,0));independent=np.mean(experts,axis=0) if chosen is None else experts[chosen]
    np.testing.assert_allclose(p,independent,atol=1e-12,rtol=1e-12);y=saved[f'y_{j}'];ys.append(y);ps.append(p);scores.append(float(np.sqrt(np.mean((p-y)**2))));arrays.update({f'pred_{j}':p,f'y_{j}':y,f'ids_{j}':saved[f'ids_{j}']})
  stages.append(dict(stage=t,rmse=float(np.sqrt(np.mean((np.concatenate(ps)-np.concatenate(ys))**2))),scores=scores,weights=weights.tolist(),chosen_reference=None if chosen is None else ('none','append','residual')[chosen],prequential_sse=total.tolist(),prequential_count=count));np.savez_compressed(folder/f'predictions_stage{t}.npz',**arrays)
 result=dict(status='completed',config={**c,'policy':'hard'},order=s['order'],stages=stages,final_rmse=stages[-1]['rmse'],average_rmse=float(np.mean([r['rmse'] for r in stages])),input_hashes=s['input_hashes'],git_commit=os.environ['TMOT_SOURCE_COMMIT'])
 (folder/'summary.json').write_text(json.dumps(result,indent=2));rows.append(result)
assert len(rows)==48
assert all(sha(Path(p))==h for p,h in inputs.items())
selected=[]
for key in sorted({tuple(r['config'][k] for k in ('dataset','encoder','method')) for r in rows}):
 best=min((r for r in rows if tuple(r['config'][k] for k in ('dataset','encoder','method'))==key),key=lambda r:(r['final_rmse'],r['config']['ridge']));selected.append(best);print(*key,best['final_rmse'],best['average_rmse'],best['config']['ridge'],[v['chosen_reference'] for v in best['stages']],flush=True)
(root/'hard_prequential_v71_runs.json').write_text(json.dumps(rows,indent=2));(root/'hard_prequential_v71_comparison.json').write_text(json.dumps(dict(complete=True,trajectories=48,selected=selected,source_hashes=inputs,checks=['TRAIN-only cumulative loss','independent hard choice','prediction mixture equivalence','VAL identities and metrics','unchanged source files'],note='Read-only rule replay on development trajectories; no retraining, no TEST; same three experts and six ridge settings as v70.'),indent=2))
(out/'complete.json').write_text(json.dumps(dict(complete=True,trajectories=48)))
