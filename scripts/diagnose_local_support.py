"""Exploratory TRAIN-only covariance outside each empirical centered subspace."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from tmot.local_moments import local_partition,merge_small_cells
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('out',type=Path);a=p.parse_args();torch.set_num_threads(1);report=[]
folder=a.root/'runs/latent_replay_v47'
for path in sorted(folder.glob('*_gaussian_r0.1_o17_s42/summary.json')):
    s=json.loads(path.read_text());c=s['config'];ck=torch.load((path.parent/s['shared_state']).resolve(),map_location='cpu',weights_only=False);m=ck['learner'];data=np.load(a.root/'features'/c['encoder']/f"{c['dataset']}_train.npz");cl=s['task']=='classification'
    parts=[np.flatnonzero(np.isin(data['y'],labels)) for labels in s['order']] if cl else [np.flatnonzero(data['groups']==g) for g in s['order']]
    y=data['y'] if cl else (data['y']-ck['target_mean'])/ck['target_sd'];rng=np.random.default_rng(42+676);offset={};cells=[]
    for ids in parts:
        x=ck['basis'].transform(data['x'][ids]);target=y[ids]
        for label in (np.unique(target) if cl else [None]):
            mask=target==label if cl else np.ones(len(x),bool);inputs=x[mask];ys=target[mask];ys=ys[:,None] if ys.ndim==1 else ys
            raw=inputs if cl else np.c_[inputs,ys];k=min(4,max(1,len(inputs)//4));assignment=merge_small_cells(inputs,local_partition(inputs,k,rng),4)
            for component in np.unique(assignment):
                rows=raw[assignment==component].astype(float);i=offset.get(label,0);cell=m.local.cells[label][i];offset[label]=i+1
                assert cell.n==len(rows);np.testing.assert_array_equal(rows.mean(0).astype(np.float32),cell.mean)
                centered=rows-rows.mean(0);_,values,vectors=np.linalg.svd(centered,full_matrices=False);keep=values>max(values.max(initial=0)*1e-10,1e-12);q=vectors[keep]
                f=cell.factor.astype(float);variance=cell.residual_sd.astype(float)**2
                total=float(np.sum(f*f)+variance.sum());factor_out=max(float(np.sum(f*f)-np.sum((q@f)**2)),0.)
                diagonal_out=float(np.sum(variance*np.maximum(1-np.sum(q*q,axis=0),0)));outside=factor_out+diagonal_out
                weight=len(rows)/(len(m.classes)*np.sum(y==label)) if cl else len(rows)/len(y)
                cells.append(dict(label=None if label is None else int(label),n=len(rows),rank=int(keep.sum()),dimension=rows.shape[1],total_variance=total,outside_variance=outside,outside_fraction=outside/total if total>1e-12 else 0.,weight=float(weight)))
    assert all(offset[label]==len(v) for label,v in m.local.cells.items());assert rng.bit_generator.state==m.local.rng.bit_generator.state
    weighted_total=sum(row['weight']*row['total_variance'] for row in cells);weighted_out=sum(row['weight']*row['outside_variance'] for row in cells)
    row=dict(dataset=c['dataset'],encoder=c['encoder'],cells=len(cells),risk_weighted_variance_fraction_outside=weighted_out/weighted_total if weighted_total>1e-12 else 0.,details=cells);report.append(row);print(json.dumps({k:v for k,v in row.items() if k!='details'}),flush=True)
assert len(report)==12
a.out.write_text(json.dumps(dict(complete=True,note='Empirical TRAIN span is not the true population manifold; outside variance may regularize or harm, not a causal performance verdict',rows=report),indent=2))
