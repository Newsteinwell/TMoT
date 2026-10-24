import json,sys
from pathlib import Path
import numpy as np
import torch
from tmot.stable_residual import StableSemanticBasis
from tmot.regularized_innovation import regularized_innovation_readout
from tmot.experiment_synthetic_robustness import sha,RIDGES
from tmot.moments import augment
root=Path(sys.argv[1]);out=root/'runs/first_calibration_v72';terminal=json.loads((out/'complete.json').read_text());assert len(terminal)==4 and all(s['status']=='completed' for s in terminal);torch.set_num_threads(1);evidence=[]
for item in terminal:
 group=item['name'];s=json.loads((out/f'{group}.json').read_text());d,e=s['dataset'],s['encoder'];tr=np.load(root/'features'/e/f'{d}_train.npz');raw=np.load(root/'prepared'/f'{d}_train.npz');assert sha(root/'features'/e/f'{d}_train.npz')==s['source_hashes']['train'];assert sha(root/'prepared'/f'{d}_train.npz')==s['source_hashes']['raw'];assert sha(out/f'{group}.pt')==s['checkpoint_sha256']
 first=np.flatnonzero(tr['groups']==min(tr['groups']));cut=int(.8*len(first));fit,held=first[:cut],first[cut:];np.testing.assert_array_equal(fit,s['fit_indices']);np.testing.assert_array_equal(held,s['held_indices']);assert s['fit_count']==len(fit) and s['held_count']==len(held)
 intervals=[tuple(map(int,str(x).split(':'))) for x in tr['ids'][first]];assert all(intervals[j][1]<=intervals[j+1][0] for j in range(len(intervals)-1))
 ck=torch.load(out/f'{group}.pt',map_location='cpu',weights_only=False);basis=ck['basis'];expected=StableSemanticBasis(basis.amplitude_dim,512).fit(tr['x'][fit])
 for key in ('mean','scale','amp_mean','amp_sd','semantic_projection','amp_projection'):np.testing.assert_array_equal(getattr(basis,key),getattr(expected,key))
 past=raw['x'][:,0,-1,None];sd=np.maximum(tr['y'][fit].std(0),1e-6)
 for index,name in enumerate(('none','append','residual')):
  b=ck['branches'][name];ref=b['reference'];m=b['learner'];target=tr['y'][fit]-ref.offset(past[fit]);mu=target.mean(0);target=(target-mu)/sd
  np.testing.assert_array_equal(ref.mean,past[fit].mean(0));np.testing.assert_array_equal(ref.scale,np.maximum(past[fit].std(0),1e-6));np.testing.assert_array_equal(b['target_mean'],mu);np.testing.assert_array_equal(b['target_sd'],sd)
  x=ref.features(basis.transform(tr['x'][fit]),past[fit]);z=augment(x);assert m.exact_memory.n==len(fit);np.testing.assert_allclose(m.exact_memory.gram,z.T@z,rtol=1e-12,atol=1e-12);np.testing.assert_allclose(m.exact_memory.cross,z.T@target,rtol=1e-12,atol=1e-12)
  vx=ref.features(basis.transform(tr['x'][held]),past[held])
  for r in RIDGES:
   m.readout=regularized_innovation_readout(m.g,m.k,m.dimension,r)[0]
   for method,learner in [('task',m),('ridge',b['baselines'][r])]:
    with np.load(out/f'{group}_{method}_r{r:g}.npz') as saved:
     np.testing.assert_array_equal(saved['fit_ids'],tr['ids'][fit]);np.testing.assert_array_equal(saved['ids'],tr['ids'][held]);np.testing.assert_array_equal(saved['y'],tr['y'][held]);np.testing.assert_array_equal(saved['predictions'][index],learner.predict(vx)*sd+mu+ref.offset(past[held]))
 evidence.append(s);print(group,'verified',flush=True)
(root/'first_calibration_v72_audit.json').write_text(json.dumps(dict(complete=True,groups=evidence,checks=['chronological disjoint TRAIN prefix/suffix','prefix-only basis and normalization','direct prefix training moments','checkpoint calibration predictions exact']),indent=2))
