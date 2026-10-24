import json,hashlib,sys
from pathlib import Path
import numpy as np
import torch
from scipy.linalg import cho_factor,cho_solve,solve_triangular
from tmot.fecam import unit_rows
from tmot.class_distance_completion import statistics,block_features
root=Path(sys.argv[1]);folder=root/'runs/class_completion_v62';torch.set_num_threads(1)
terminal=json.loads((folder/'complete.json').read_text());assert len(terminal)==6 and all(r['status']=='completed' for r in terminal)
rows=[];max_cov_error=0.;controls=0
for path in sorted((folder/'_shared').glob('*.pt')):
 ck=torch.load(path,map_location='cpu',weights_only=False);source=Path(ck['source_checkpoint']);assert hashlib.sha256(source.read_bytes()).hexdigest()==ck['source_sha256'];original=torch.load(source,map_location='cpu',weights_only=False);m=original['learner'];basis=original['basis'];um=ck['unit_memory'];stats,_=statistics(m,um)
 summaries=sorted(folder.glob(path.stem+'_*/summary.json'));assert len(summaries)==24;s=json.loads(summaries[0].read_text());c=s['config'];d,e=c['dataset'],c['encoder'];tr=np.load(root/'features'/e/f'{d}_train.npz');val=np.load(root/'features'/e/f'{d}_val.npz')
 for group in s['order']:
  ids=np.flatnonzero(np.isin(tr['y'],group));z=unit_rows(basis.transform(tr['x'][ids]));y=tr['y'][ids]
  for label in group:
   xx=z[y==label];zz=np.c_[xx,np.ones(len(xx))];cell=um.cells[label];assert cell.n==len(xx);np.testing.assert_allclose(cell.total,zz.sum(0),rtol=1e-12,atol=1e-12);np.testing.assert_allclose(cell.gram,zz.T@zz,rtol=1e-10,atol=1e-10)
 # Independent residual observations reconstruct the corrected covariance.
 for k,label in enumerate(m.classes):
  cells=m.local.cells[label];points=np.concatenate([v.points for v in cells]);w=np.concatenate([np.full(len(v.points),v.n/len(v.points)) for v in cells]);n=sum(v.n for v in cells);f=block_features(m.function,points);mu=(w[:,None]*f).sum(0)/n;z=f-mu;den=max(n-1,1)
  aa=z[:,:512].T@(w[:,None]*z[:,:512])/den;u=z[:,:512].T@(w[:,None]*z[:,512:])/den;rho=.01*max(np.trace(aa)/512,1e-12);b=cho_solve(cho_factor(aa+rho*np.eye(512)),u)
  residual=z[:,512:]-z[:,:512]@b;rc=residual.T@(w[:,None]*residual)/den
  for mode in ('self','complete'):
   em,ec=(mu[:512],aa) if mode=='self' else stats['base'][k]
   trans=np.block([[np.eye(512),b],[np.zeros((128,512)),np.eye(128)]]);diag=np.zeros((640,640));diag[:512,:512]=ec;diag[512:,512:]=rc
   expected=trans.T@diag@trans;actual=stats[mode][k][1];err=np.linalg.norm(expected-actual)/max(np.linalg.norm(expected),1e-12);max_cov_error=max(max_cov_error,float(err));assert err<1e-7
 digest=hashlib.sha256(path.read_bytes()).hexdigest();final_ids=[np.flatnonzero(np.isin(val['y'],g)) for g in s['order']];vx=np.concatenate([basis.transform(val['x'][ids]) for ids in final_ids]);features={'base':unit_rows(vx),'approx':block_features(m.function,vx)};features['complete']=features['approx'];features['self']=features['approx']
 for sp in summaries:
  s=json.loads(sp.read_text());c=s['config'];assert s['shared_state_sha256']==digest and s['final_training_state_exact']
  for split,h in s['input_hashes'].items():assert hashlib.sha256((root/'features'/e/f'{d}_{split}.npz').read_bytes()).hexdigest()==h
  for stage in s['stages']:
   with np.load(sp.parent/f"predictions_stage{stage['stage']}.npz") as pred:
    yy=np.concatenate([pred[f'y_{i}'] for i in range(stage['stage'])]);pp=np.concatenate([pred[f'pred_{i}'] for i in range(stage['stage'])]);score=np.mean([np.mean(pp[yy==v]==v) for v in np.unique(yy)]);np.testing.assert_allclose(score,stage['macro_accuracy'],atol=1e-12)
    if c['mode']=='base':
     peer=root/'runs/fecam_harder_v59'/f"{d}_{e}_fecam_a{c['alpha']:g}_o17_s42"/f"predictions_stage{stage['stage']}.npz"
     with np.load(peer) as saved:
      for key in saved.files:np.testing.assert_array_equal(saved[key],pred[key])
    if stage['stage']==len(s['order']):
     np.testing.assert_array_equal(yy,np.concatenate([val['y'][ids] for ids in final_ids]));np.testing.assert_array_equal(np.concatenate([pred[f'ids_{i}'] for i in range(stage['stage'])]),np.concatenate([val['ids'][ids] for ids in final_ids]))
     ds=[];x=features[c['mode']]
     for mean,cov in stats[c['mode']]:
      dim=len(mean);reg=cov+c['alpha']*max(np.trace(cov)/dim,1e-12)*np.eye(dim);sd=np.sqrt(np.maximum(np.diag(reg),1e-15));l=np.linalg.cholesky(reg/np.outer(sd,sd));white=solve_triangular(l,(x-mean).T,lower=True);ds.append(np.sum(white**2,axis=0))
     np.testing.assert_array_equal(pp,np.asarray(m.classes)[np.column_stack(ds).argmin(1)])
  if c['mode']=='base':controls+=1
  np.testing.assert_allclose(s['final_macro_accuracy'],s['stages'][-1]['macro_accuracy']);np.testing.assert_allclose(s['average_macro_accuracy'],np.mean([r['macro_accuracy'] for r in s['stages']]))
  rows.append(s)
 print(path.stem,'verified',flush=True)
assert len(rows)==144 and controls==36
(root/'class_completion_v62_runs.json').write_text(json.dumps(rows,indent=2));(root/'class_completion_v62_audit.json').write_text(json.dumps(dict(complete=True,trajectories=144,base_fecam_controls=36,max_independent_covariance_relative_error=max_cov_error,checks=['direct TRAIN unit statistics','independent residual covariance','independent final triangular-solve predictions','all-stage metrics','base full trajectories exact','hashes']),indent=2))
