"""Reconstruct arriving TRAIN partitions and independently evaluate saved points."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from tmot.local_moments import local_partition,merge_small_cells
from tmot.covariance_replay import CovarianceCell
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('out',type=Path);a=p.parse_args();torch.set_num_threads(1);results=[]
for path in sorted((a.root/'runs/synthetic_replay_v52').glob('*_matched_r0.1_o17_s42/summary.json')):
 s=json.loads(path.read_text());c=s['config'];ck=torch.load((path.parent/s['shared_state']).resolve(),map_location='cpu',weights_only=False);m=ck['learner'];rng=np.random.default_rng(718);offset={};serial=0;items=[]
 with np.load(a.root/'features'/c['encoder']/f"{c['dataset']}_train.npz") as data:
  cl=s['task']=='classification';y=data['y'] if cl else (data['y']-ck['target_mean'])/ck['target_sd'];parts=[np.flatnonzero(np.isin(y,labels)) for labels in s['order']] if cl else [np.flatnonzero(data['groups']==g) for g in s['order']]
  for ids in parts:
   x=ck['basis'].transform(data['x'][ids]);target=y[ids]
   for label in (np.unique(target) if cl else [None]):
    mask=target==label if cl else np.ones(len(x),bool);inputs=x[mask];ys=target[mask];ys=ys[:,None] if ys.ndim==1 else ys;raw=inputs if cl else np.c_[inputs,ys];assignment=merge_small_cells(inputs,local_partition(inputs,min(4,max(1,len(inputs)//4)),rng),4)
    for component in np.unique(assignment):
     rows=raw[assignment==component].astype(np.float32);i=offset.get(label,0);cell=m.local.cells[label][i];offset[label]=i+1;serial+=1;seed=32452843+serial*7919
     np.testing.assert_array_equal(raw[assignment==component].astype(float).mean(0).astype(np.float32),cell.mean)
     initial=CovarianceCell.sample(cell,16,np.random.default_rng(seed));mean=rows.mean(0);sd=rows.std(0);sd=np.maximum(sd,max(float(np.median(sd))*.1,1e-4));rgen=np.random.default_rng(seed+1);proj=(rgen.normal(size=(512,64))/np.sqrt(512)).astype(np.float32);bias=rgen.normal(size=64).astype(np.float32)
     def feature(z):
      linear=z[:,:512]@proj;return np.c_[linear,np.maximum(linear+bias,0),z[:,512:]]
     real=(rows-mean)/sd;fr=feature(real);fm=fr.mean(0);fg=fr.T@fr/len(fr);xy=real.T@real[:,512:]/len(real)
     def loss(points):
      z=(points-mean)/sd;f=feature(z);value=np.mean((f.mean(0)-fm)**2)+np.mean((f.T@f/len(f)-fg)**2)+.1*(np.mean((z.mean(0)-real.mean(0))**2)+np.mean(((z*z).mean(0)-(real*real).mean(0))**2))
      if z.shape[1]>512:value+=np.mean((z.T@z[:,512:]/len(z)-xy)**2)
      return float(value)
     before=loss(initial);after=loss(cell.points);np.testing.assert_allclose(before,cell.fitting['initial'],rtol=1e-4,atol=1e-6);np.testing.assert_allclose(after,cell.fitting['final'],rtol=1e-4,atol=1e-6)
     assert after<=before+1e-6;items.append(dict(n=len(rows),initial=before,final=after))
 assert rng.bit_generator.state==m.local.rng.bit_generator.state
 results.append(dict(dataset=c['dataset'],encoder=c['encoder'],cells=items));print(c['dataset'],c['encoder'],len(items),flush=True)
assert len(results)==12;a.out.write_text(json.dumps(dict(complete=True,rows=results),indent=2))
