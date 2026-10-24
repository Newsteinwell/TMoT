"""Chronological inner calibration using only first-task TRAIN records."""
import argparse,json,os
from pathlib import Path
import numpy as np
import torch
from tmot.causal_reference import CausalReference
from tmot.stable_residual import StableSemanticBasis
from tmot.task_synthetic_replay import TaskSyntheticLearner
from tmot.learner import MomentLearner
from tmot.regularized_innovation import regularized_innovation_readout
from tmot.experiment_synthetic_robustness import sha,RIDGES

def run(a):
 torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
 root=Path(a.root);out=Path(a.out);out.mkdir(parents=True,exist_ok=True);group=f'{a.dataset}_{a.encoder}';trpath=root/'features'/a.encoder/f'{a.dataset}_train.npz';tr=np.load(trpath);rawpath=root/'prepared'/f'{a.dataset}_train.npz';raw=np.load(rawpath);meta=json.loads((root/'prepared'/f'{a.dataset}_manifest.json').read_text());assert sha(rawpath)==meta['files']['train'];np.testing.assert_array_equal(tr['ids'],raw['ids']);np.testing.assert_array_equal(tr['y'],raw['y'])
 first=np.flatnonzero(tr['groups']==min(tr['groups']));cut=int(.8*len(first));assert 2<=cut<len(first);fit,held=first[:cut],first[cut:]
 intervals=[tuple(map(int,str(v).split(':'))) for v in tr['ids'][first]];assert all(intervals[j][1]<=intervals[j+1][0] for j in range(len(intervals)-1));assert not set(fit)&set(held)
 basis=StableSemanticBasis(2*meta['shapes']['train'][1],512).fit(tr['x'][fit]);past=raw['x'][:,0,-1,None];sd=np.maximum(tr['y'][fit].std(0),1e-6)
 outputs={(m,r):[] for m in ('task','ridge') for r in RIDGES};checkpoints={}
 for name in ('none','append','residual'):
  ref=CausalReference(name).fit(past[fit]);target=tr['y'][fit]-ref.offset(past[fit]);mu=target.mean(0);target=(target-mu)/sd;dim=512+(name!='none');x=ref.features(basis.transform(tr['x'][fit]),past[fit]);vx=ref.features(basis.transform(tr['x'][held]),past[held])
  m=TaskSyntheticLearner(dim,'regression',42,ridge=.1,mode='local',epochs=30,device='cuda',variant='task',points_per_cell=256);m.observe(x,target);baselines={}
  for r in RIDGES:
   b=MomentLearner(dim,'regression','ridge',42,ridge=r);b.observe(x,target);baselines[r]=b;m.readout=regularized_innovation_readout(m.g,m.k,dim,r)[0]
   for method,learner in [('task',m),('ridge',b)]:outputs[(method,r)].append(learner.predict(vx)*sd+mu+ref.offset(past[held]))
  m.readout=regularized_innovation_readout(m.g,m.k,dim,.1)[0];checkpoints[name]=dict(learner=m,baselines=baselines,reference=ref,target_mean=mu,target_sd=sd)
 for (method,r),predictions in outputs.items():np.savez_compressed(out/f'{group}_{method}_r{r:g}.npz',predictions=np.stack(predictions),y=tr['y'][held],ids=tr['ids'][held],fit_ids=tr['ids'][fit])
 shared=out/f'{group}.pt';torch.save(dict(branches=checkpoints,basis=basis),shared)
 summary=dict(dataset=a.dataset,encoder=a.encoder,fit_indices=fit.tolist(),held_indices=held.tolist(),first_task_count=len(first),fit_count=len(fit),held_count=len(held),source_hashes=dict(train=sha(trpath),raw=sha(rawpath)),checkpoint_sha256=sha(shared),git_commit=os.environ['TMOT_SOURCE_COMMIT'],note='Calibration network/basis/normalization use only first TRAIN prefix. Production experts remain v70 full-task experts; this is a refit selection proxy.')
 (out/f'{group}.json').write_text(json.dumps(summary,indent=2));print(group,'completed',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);p.add_argument('--dataset',required=True);p.add_argument('--encoder',required=True);run(p.parse_args())
