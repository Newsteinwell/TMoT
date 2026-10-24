"""Verify v45 class growth, fixed bases, original risk blocks and saved predictions."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from tmot.stable_residual import StableSemanticBasis
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('out',type=Path);a=p.parse_args();torch.set_num_threads(1)
folder=a.root/'runs/harder_validation_v45';cache={};checked=[];max_gram_error=0.
terminal=json.loads((folder/'complete.json').read_text());assert len(terminal)==108 and all(r['status']=='completed' for r in terminal)
for path in sorted(folder.glob('*/summary.json')):
    s=json.loads(path.read_text());c=s['config'];key=c['dataset'],c['encoder'];ck=torch.load(path.parent/'final.pt',map_location='cpu',weights_only=False);m=ck['learner'];basis=ck['basis']
    if key not in cache:
        data=np.load(a.root/'features'/c['encoder']/f"{c['dataset']}_train.npz");val=np.load(a.root/'features'/c['encoder']/f"{c['dataset']}_val.npz")
        meta=json.loads((a.root/'prepared'/f"{c['dataset']}_manifest.json").read_text());parts=[np.flatnonzero(np.isin(data['y'],cl)) for cl in s['order']]
        independent=StableSemanticBasis(2*meta['shapes']['train'][1],512).fit(data['x'][parts[0]])
        z=np.empty((len(data['x']),512),dtype=np.float32)
        for ids in parts:z[ids]=independent.transform(data['x'][ids])
        classes=[int(k) for cl in s['order'] for k in sorted(cl)];assert set(classes)==set(data['y'].tolist())
        phi=np.c_[z.astype(float),np.ones(len(z))];y=data['y'];w=np.array([1/(len(classes)*np.sum(y==label)) for label in y]);target=np.eye(len(classes))[[classes.index(int(label)) for label in y]]
        cache[key]=(independent,classes,phi.T@(w[:,None]*phi),phi.T@(w[:,None]*target),data,val)
    independent,classes,g,k,data,val=cache[key]
    for attr,value in vars(independent).items():np.testing.assert_equal(getattr(basis,attr),value)
    assert m.classes==classes and m.stage==len(s['order'])
    memory=m.memory if c['method']=='ridge' else m.exact_memory
    mg,mk,_=memory.risk_moments(classes);max_gram_error=max(max_gram_error,float(np.max(abs(mg-g))))
    np.testing.assert_allclose(mg,g,rtol=1e-8,atol=1e-8);np.testing.assert_allclose(mk,k,rtol=1e-8,atol=1e-8)
    for label in classes:assert memory.cells[label].n==int(np.sum(data['y']==label))
    if c['method']=='ridge':
        penalty=np.eye(len(g))*c['ridge'];penalty[-1,-1]=0
        np.testing.assert_allclose(m.readout,np.linalg.solve(g+penalty,k),rtol=1e-6,atol=2e-7)
    else:
        phi=np.r_[np.arange(512),len(m.g)-1]
        np.testing.assert_allclose(m.g[np.ix_(phi,phi)],g,rtol=1e-8,atol=1e-8);np.testing.assert_allclose(m.k[phi],k,rtol=1e-8,atol=1e-8)
        assert m.seen==len(data['y'])
        if m.buffer is not None:assert sum(v.nbytes for v in m.buffer)+m.keys.nbytes<=m.replay_bytes
    with np.load(path.parent/f"predictions_stage{len(s['order'])}.npz") as pred:
        for stage,labels in enumerate(s['order']):
            inds=np.flatnonzero(np.isin(val['y'],labels))
            np.testing.assert_array_equal(pred[f'ids_{stage}'],val['ids'][inds]);np.testing.assert_array_equal(pred[f'y_{stage}'],val['y'][inds])
            np.testing.assert_array_equal(m.predict(basis.transform(val['x'][inds])),pred[f'pred_{stage}'])
    checked.append(path.parent.name)
assert len(checked)==108
report=dict(complete=True,checked=checked,independent_streams=len(cache),max_original_gram_error=max_gram_error)
a.out.write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='checked'}))
