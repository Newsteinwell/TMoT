import argparse,json,math
from pathlib import Path
import numpy as np
import torch
from tmot.stable_residual import StableSemanticBasis
from tmot.functional_basis import FunctionalBasis
from tmot.moments import ClassificationMoments,RegressionMoments,ridge_readout
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('out',type=Path);a=p.parse_args()
torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
cache={};checked=[];controls=0
for path in sorted((a.root/'runs/functional_history_v37').glob('*/summary.json')):
    s=json.loads(path.read_text());c=s['config'];ck=torch.load(path.parent/'final.pt',map_location='cpu',weights_only=False);f=ck['basis'];m=ck['learner']
    old=a.root/'runs/functional_v36'/f"{c['dataset']}_{c['encoder']}_trained_r{c['ridge']:g}_o17_s42"
    old_summary=json.loads((old/'summary.json').read_text());assert s['input_hashes']==old_summary['input_hashes'] and s['order']==old_summary['order']
    compare=range(1,len(s['stages'])+1) if c['function_scope']=='first' else [1]
    for stage in compare:
        with np.load(path.parent/f'predictions_stage{stage}.npz') as x,np.load(old/f'predictions_stage{stage}.npz') as y:
            for k in x.files:np.testing.assert_array_equal(x[k],y[k])
    if c['function_scope']=='first':controls+=1
    key=c['dataset'],c['encoder'],c['function_scope']
    if key not in cache:
        with np.load(a.root/'features'/c['encoder']/f"{c['dataset']}_train.npz") as d:
            x=d['x'];y=d['y'].copy();ids=d['ids'].copy()
            parts=[np.flatnonzero(np.isin(y,cl)) for cl in s['order']] if s['task']=='classification' else [np.flatnonzero(d['groups']==g) for g in s['order']]
            if s['task']=='regression':y=(y-ck['target_mean'])/ck['target_sd']
            meta=json.loads((a.root/'prepared'/f"{c['dataset']}_manifest.json").read_text())
            base=StableSemanticBasis(2*meta['shapes']['train'][1],512,seed=c['basis_seed']).fit(x[parts[0]])
            all_ids=np.concatenate(parts);fit_ids=all_ids if c['function_scope']=='arrived' else parts[0]
            fresh=FunctionalBasis(base,s['task'],mode='trained',seed=c['seed'],epochs=c['epochs'],device='cuda').fit(x[fit_ids],y[fit_ids],calibration_x=x[parts[0]])
            z=fresh.transform(x[all_ids]) if c['function_scope']=='arrived' else np.concatenate([fresh.transform(x[ii]) for ii in parts])
            direct=ClassificationMoments(640) if s['task']=='classification' else RegressionMoments(640,y.shape[1]);direct.update(z,y[all_ids])
            cache[key]=(fresh,direct,parts,ids,x.dtype.itemsize*x.shape[1]+y.dtype.itemsize*(1 if y.ndim==1 else y.shape[1]))
    fresh,direct,parts,ids,row_bytes=cache[key]
    for stage,record in enumerate(s['stages']):
        arrived=np.concatenate(parts[:stage+1]);expected=arrived if c['function_scope']=='arrived' else parts[0]
        with np.load(path.parent/f'function_fit_stage{stage+1}.npz') as d:np.testing.assert_array_equal(d['ids'],ids[expected])
        assert record['function_fit_samples']==len(expected)
        assert record['function_optimizer_steps']==c['epochs']*math.ceil(len(expected)/256)
        assert record['historical_feature_access_bytes']==(len(arrived)*row_bytes if c['function_scope']=='arrived' else 0)
    for k,v in f.net.state_dict().items():torch.testing.assert_close(v,fresh.net.state_dict()[k],rtol=0,atol=0)
    assert all(not p.requires_grad for p in f.net.parameters()) and m.buffer is None
    for name in ('mean','sd','hidden_mean','hidden_scale'):np.testing.assert_array_equal(getattr(f,name),getattr(fresh,name))
    g,k,v=m.memory.risk_moments(m.classes);dg,dk,dv=direct.risk_moments(m.classes)
    for left,right in ((g,dg),(k,dk),(v,dv)):np.testing.assert_allclose(left,right,rtol=1e-9,atol=1e-8)
    np.testing.assert_allclose(m.readout,ridge_readout(g,k,c['ridge']),rtol=1e-10,atol=1e-10)
    checked.append(path.parent.name)
assert len(checked)==63 and controls==9 and len(cache)==18
a.out.write_text(json.dumps(dict(complete=True,trajectories=63,exact_full_trajectory_controls=9,exact_initial_stage_controls=63,final_function_reconstructions=18,checks=['recorded function-training IDs equal first or arrived subset at each stage','first-stage-only calibration','function fit steps and required feature payload','final function retraining and moment/readout reconstruction','real-history access explicitly recorded'],checked=checked),indent=2))
print('63 history-scope audits passed, including 9 full-trajectory controls')
