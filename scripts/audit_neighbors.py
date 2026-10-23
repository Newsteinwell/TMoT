"""Independent retained-row and direct-distance final prediction audit."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('out',type=Path);a=p.parse_args()
cache={};checked=[]
for path in sorted((a.root/'runs/neighbors_v30').glob('*/summary.json')):
    s=json.loads(path.read_text());c=s['config'];key=c['dataset'],c['encoder']
    ck=torch.load(path.parent/'final.pt',map_location='cpu',weights_only=False);m=ck['learner']
    if key not in cache:
        with np.load(a.root/'features'/c['encoder']/f"{c['dataset']}_train.npz") as data:
            x=ck['basis'].transform(data['x']).astype(np.float32);y=data['y'].copy()
            parts=[np.flatnonzero(np.isin(y,labels)) for labels in s['order']] if s['task']=='classification' else [np.flatnonzero(data['groups']==d) for d in s['order']]
            if s['task']=='regression':y=(y-ck['target_mean'])/ck['target_sd']
            arrival=np.concatenate(parts)
        with np.load(a.root/'features'/c['encoder']/f"{c['dataset']}_val.npz") as data:
            q=ck['basis'].transform(data['x'][:6]);qid=data['ids'][:6].copy()
        cache[key]=(x[arrival],y[arrival],q,qid)
    x,y,q,qid=cache[key];indices=list(range(len(x)))
    if c['method']=='neighbors_buffer':
        cap=(4*1024*1024)//(x.shape[1]*4+y[0].nbytes);indices=list(range(min(cap,len(x))));rng=np.random.default_rng(c['seed'])
        for i in range(cap,len(x)):
            j=int(rng.integers(i+1))
            if j<cap:indices[j]=i
        assert m.persistent_bytes<=4*1024*1024
    np.testing.assert_array_equal(m.x,x[indices]);np.testing.assert_array_equal(m.y,y[indices]);assert m.seen==len(x)
    expected=[]
    for row in q:
        dist=np.sum((m.x.astype(np.float64)-row.astype(np.float64))**2,axis=1)
        target=m.y[np.argsort(dist,kind='stable')[:min(c['neighbors'],len(m.x))]]
        if s['task']=='classification':
            labels,counts=np.unique(target,return_counts=True);expected.append(labels[counts.argmax()])
        else:expected.append(target.mean(axis=0))
    expected=np.asarray(expected)
    if s['task']=='regression':expected=expected*ck['target_sd']+ck['target_mean']
    with np.load(path.parent/f"predictions_stage{len(s['stages'])}.npz") as pred:
        ids=np.concatenate([pred[f'ids_{i}'] for i in range(len(s['stages']))]);values=np.concatenate([pred[f'pred_{i}'] for i in range(len(s['stages']))])
        positions=[np.flatnonzero(ids==v).item() for v in qid]
        np.testing.assert_allclose(values[positions],expected,rtol=1e-6,atol=1e-7)
    checked.append(path.parent.name)
assert len(checked)==210
a.out.write_text(json.dumps(dict(complete=True,trajectories=210,checks=['all retained rows and targets independently reconstructed','buffer payload <=4MiB','six final VAL predictions per run recomputed with direct distances'],checked=checked),indent=2))
print('210 retained-row audits and 1260 direct-distance predictions passed')
