"""Final-state diagnostic, no new fitting and no label-dependent choice."""
import argparse,hashlib,json,os
from pathlib import Path
import numpy as np
import torch
from tmot.class_distance_completion import statistics,block_features,fast_precision
from tmot.unit_ball_moments import bound_statistics
from tmot.fecam import unit_rows
from tmot.margin_selector import select_by_margin,relative_margin
from scripts.diagnose_readout_complementarity import decomposition

def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def distances(x,stats,alpha):
 columns=[]
 for mean,cov in stats:
  delta=x-mean;v=np.einsum('ni,ni->n',delta@fast_precision(cov,alpha),delta)
  assert np.min(v)>=-1e-10;columns.append(np.maximum(v,0))
 return np.column_stack(columns)
def run(a):
 torch.set_num_threads(1);root=Path(a.root);out=Path(a.out);out.mkdir(parents=True,exist_ok=True);folder=root/'runs/bounded_robustness_v64';path=folder/'_shared'/f'{a.group}.pt';h=sha(path);ck=torch.load(path,map_location='cpu',weights_only=False);source=Path(ck['source_checkpoint']);assert sha(source)==ck['source_sha256'];model=torch.load(source,map_location='cpu',weights_only=False);m=model['learner']

 s=json.loads((folder/(a.group+'_bounded')/'summary.json').read_text());c=s['config'];valpath=root/'features'/c['encoder']/f"{c['dataset']}_val.npz";assert sha(valpath)==s['input_hashes']['val'];val=np.load(valpath)
 parts=[np.flatnonzero(np.isin(val['y'],g)) for g in s['order']];vx=[model['basis'].transform(val['x'][i]) for i in parts]
 stats,_=statistics(m,ck['unit_memory']);stats['bounded'],_=bound_statistics(stats['complete'],ck['unit_memory'],m.classes,512)
 dr=distances(np.concatenate([unit_rows(x) for x in vx]),stats['base'],s['frozen']['fecam']);dc=distances(np.concatenate([block_features(m.function,x) for x in vx]),stats['bounded'],s['frozen']['bounded'])
 rp=np.asarray(m.classes)[dr.argmin(1)];cp=np.asarray(m.classes)[dc.argmin(1)]
 for mode,pred in [('fecam',rp),('bounded',cp)]:
  with np.load(folder/(a.group+'_'+mode)/f"predictions_stage{len(parts)}.npz") as saved:
   np.testing.assert_array_equal(pred,np.concatenate([saved[f'pred_{j}'] for j in range(len(parts))]))
 pred,choice=select_by_margin(dr,dc,m.classes)
 y=np.concatenate([val['y'][i] for i in parts]);ids=np.concatenate([val['ids'][i] for i in parts]);metrics=decomposition(y,rp,pred);baseline=decomposition(y,rp,cp)
 weights=np.array([1/(len(np.unique(y))*np.sum(y==v)) for v in y]);select_rate=float(weights[choice].sum())
 np.savez_compressed(out/(a.group+'.npz'),y=y,ids=ids,classes=m.classes,reference_distances=dr,candidate_distances=dc,prediction=pred,choose_candidate=choice)
 result=dict(config=c,final_only=True,rule='candidate relative margin > reference relative margin; tie uses reference',metrics=metrics,ungated=baseline,weighted_selection_rate=select_rate,source_sha256=h,model_sha256=ck['source_sha256'],git_commit=os.environ['TMOT_SOURCE_COMMIT'])
 assert sha(path)==h and sha(source)==ck['source_sha256'];(out/(a.group+'.json')).write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);p.add_argument('--group',required=True);run(p.parse_args())
