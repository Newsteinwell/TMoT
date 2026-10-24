import hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
from tmot.regularized_innovation import regularized_innovation_readout
from tmot.experiment_synthetic_robustness import sha
root=Path(sys.argv[1]);folder=root/'runs/calibrated_task_v69';terminal=json.loads((folder/'complete.json').read_text());assert len(terminal)==8 and all(r['status']=='completed' for r in terminal);torch.set_num_threads(1);rows=[]
for shared in sorted((folder/'_shared').glob('*.pt')):
 ck=torch.load(shared,map_location='cpu',weights_only=False);m=ck['learner'];ref=ck['reference'];basis=ck['basis'];mu=ck['target_mean'];sd=ck['target_sd'];digest=sha(shared);summaries=sorted(folder.glob(shared.stem+'_*/summary.json'));assert len(summaries)==12
 s=json.loads(summaries[0].read_text());c=s['config'];d,e=c['dataset'],c['encoder'];tr=np.load(root/'features'/e/f'{d}_train.npz');val=np.load(root/'features'/e/f'{d}_val.npz');rawtr=np.load(root/'prepared'/f'{d}_train.npz');rawval=np.load(root/'prepared'/f'{d}_val.npz')
 for raw,features in ((rawtr,tr),(rawval,val)):
  np.testing.assert_array_equal(raw['y'],features['y']);np.testing.assert_array_equal(raw['ids'],features['ids'])
 pasttr=rawtr['x'][:,0,-1,None];pastval=rawval['x'][:,0,-1,None];first=np.flatnonzero(tr['groups']==s['order'][0]);np.testing.assert_array_equal(ref.mean,pasttr[first].mean(0));np.testing.assert_array_equal(ref.scale,np.maximum(pasttr[first].std(0),1e-6));np.testing.assert_array_equal(sd,np.maximum(tr['y'][first].std(0),1e-6));np.testing.assert_array_equal(mu,(tr['y']-ref.offset(pasttr))[first].mean(0))
 if c['reference']=='calibrated':
  assert ref.calibration_count==len(first)
  estimated=[]
  for j in range(pasttr.shape[1]):
   p=pasttr[first,j].astype(float);target=tr['y'][first,j].astype(float)
   estimated.append(0. if np.ptp(p)==0 else np.linalg.lstsq(np.c_[p,np.ones(len(p))],target,rcond=None)[0][0])
  np.testing.assert_allclose(ref.unclipped_beta,estimated,atol=1e-10,rtol=1e-10);np.testing.assert_allclose(ref.beta,np.clip(estimated,0,1),atol=1e-10,rtol=1e-10)
 for sp in summaries:
  s=json.loads(sp.read_text());c=s['config'];assert s['shared_state_sha256']==digest
  np.testing.assert_array_equal(s['calibration']['beta'],getattr(ref,'beta',np.ones(tr['y'].shape[1])));assert s['calibration']['count']==len(first)
  for split in ('train','val'):
   assert sha(root/'features'/e/f'{d}_{split}.npz')==s['input_hashes'][split];assert sha(root/'prepared'/f'{d}_{split}.npz')==s['raw_input_hashes'][split]
  for stage in s['stages']:
   t=stage['stage'];ixs=[np.flatnonzero(val['groups']==g) for g in s['order'][:t]]
   with np.load(sp.parent/f'predictions_stage{t}.npz') as pred:
    ys=[];ps=[]
    for j,ix in enumerate(ixs):
     np.testing.assert_array_equal(pred[f'y_{j}'],val['y'][ix]);np.testing.assert_array_equal(pred[f'ids_{j}'],val['ids'][ix]);ys.append(pred[f'y_{j}']);ps.append(pred[f'pred_{j}']);np.testing.assert_allclose(np.sqrt(np.mean((ys[-1]-ps[-1])**2)),stage['scores'][j],atol=1e-12)
    np.testing.assert_allclose(np.sqrt(np.mean((np.concatenate(ys)-np.concatenate(ps))**2)),stage['rmse'],atol=1e-12)
    peer=None
    if c['reference']=='residual':peer=root/'runs/causal_task_v68'/f"{d}_{e}_residual_{c['method']}_r{c['ridge']:g}"
    if peer is not None:
     with np.load(peer/f'predictions_stage{t}.npz') as old:
      for key in old.files:np.testing.assert_array_equal(old[key],pred[key])
    if t==len(s['order']):
     m.readout=regularized_innovation_readout(m.g,m.k,c['dimension'],c['ridge'])[0];model=m if c['method']=='task' else ck['baselines'][c['ridge']]
     for j,ix in enumerate(ixs):
      x=ref.features(basis.transform(val['x'][ix]),pastval[ix]);p=model.predict(x)*sd+mu+ref.offset(pastval[ix]);np.testing.assert_array_equal(p,pred[f'pred_{j}'])
  np.testing.assert_allclose(s['final_rmse'],s['stages'][-1]['rmse']);np.testing.assert_allclose(s['average_rmse'],np.mean([v['rmse'] for v in s['stages']]))
  rows.append(s)
 print(shared.stem,'verified',flush=True)
assert len(rows)==96
(root/'calibrated_task_v69_runs.json').write_text(json.dumps(rows,indent=2))
selected=[]
for d,e,ref,method in sorted({tuple(r['config'][k] for k in ('dataset','encoder','reference','method')) for r in rows}):
 choices=[r for r in rows if tuple(r['config'][k] for k in ('dataset','encoder','reference','method'))==(d,e,ref,method)];best=min(choices,key=lambda r:(r['final_rmse'],r['config']['ridge']));selected.append(best);print(d,e,ref,method,best['final_rmse'],best['average_rmse'],best['config']['ridge'])
(root/'calibrated_task_v69_comparison.json').write_text(json.dumps(dict(complete=True,trajectories=96,selected=selected,checks=['first-task calibration','all-stage metrics and identities','independent first-task least-squares beta','v68 fixed residual controls exact','final checkpoint predictions exact','raw and feature hashes'],note='Single-seed development selection over six ridge values, no TEST and no robustness claim.'),indent=2))
