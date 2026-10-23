import argparse,copy,json,math
from pathlib import Path
import numpy as np
import torch
from tmot.replay_functional import ReplayFunctionalLearner
from tmot.stable_residual import StableSemanticBasis
from tmot.gradient_diagnostic import empirical_sample
from tmot.moments import augment,ridge_readout
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('out',type=Path);a=p.parse_args()
torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
folder=a.root/'runs/replay_functional_v38';cache={};checked=[];pairs=0
for path in sorted(folder.glob('*/summary.json')):
    s=json.loads(path.read_text());c=s['config'];ck=torch.load(path.parent/'final.pt',map_location='cpu',weights_only=False);m=ck['learner'];key=c['dataset'],c['encoder'],c['method']
    if key not in cache:
        with np.load(a.root/'features'/c['encoder']/f"{c['dataset']}_train.npz") as d:
            x=d['x'];y=d['y'].copy()
            parts=[np.flatnonzero(np.isin(y,cl)) for cl in s['order']] if s['task']=='classification' else [np.flatnonzero(d['groups']==g) for g in s['order']]
            if s['task']=='regression':y=(y-ck['target_mean'])/ck['target_sd']
            meta=json.loads((a.root/'prepared'/f"{c['dataset']}_manifest.json").read_text())
            base=StableSemanticBasis(2*meta['shapes']['train'][1],512).fit(x[parts[0]])
            ref=ReplayFunctionalLearner(512,s['task'],c['seed'],ridge=c['ridge'],mode=c['method'],epochs=c['epochs'],device='cuda');records=[]
            for stage,ids in enumerate(parts):
                current=base.transform(x[ids]);target=y[ids];history=None
                if stage==len(parts)-1 and ref.seen:
                    rng=copy.deepcopy(ref.rng)
                    history=ref.local.sample(4096,ref.classes,rng) if ref.mode=='local' else empirical_sample(*ref.buffer,4096,rng,ref.classes)
                    tx=np.r_[current,history[0]];ty=np.r_[target,history[1]]
                    old_ids=np.concatenate(parts[:stage]);w=np.zeros(len(tx))
                    if s['task']=='regression':
                        w[:len(ids)]=1/(len(old_ids)+len(ids));w[len(ids):]=len(old_ids)/((len(old_ids)+len(ids))*4096)
                    else:
                        classes=np.unique(ty)
                        for label in classes:
                            nc=np.sum(target==label);no=np.sum(y[old_ids]==label);nh=np.sum(history[1]==label)
                            w[:len(ids)][target==label]=1/(len(classes)*(nc+no))
                            if nh:w[len(ids):][history[1]==label]=no/(len(classes)*(nc+no)*nh)
                ref.observe(current,target);records.append((ref.optimizer_steps,ref.historical_training_samples,ref.weight_summary.copy()))
            z=augment(ref.function.transform(tx));targets=np.eye(len(ref.classes))[[ref.classes.index(int(v)) for v in ty]] if s['task']=='classification' else ty
            np.testing.assert_allclose(ref.g,z.T@(w[:,None]*z),rtol=1e-10,atol=1e-10)
            np.testing.assert_allclose(ref.k,z.T@(w[:,None]*targets),rtol=1e-10,atol=1e-10)
            cache[key]=(ref,records,base,parts,y)
    ref,records,base,parts,y=cache[key]
    assert m.seen==len(y) and m.classes==ref.classes
    for record,(steps,history,mass) in zip(s['stages'],records):
        assert record['optimizer_steps']==steps and record['historical_training_samples']==history
        for k in mass:np.testing.assert_allclose(record['representative_mass'][k],mass[k],atol=1e-12)
    for field in ('g','k'):np.testing.assert_array_equal(getattr(m,field),getattr(ref,field))
    for k,v in m.function.net.state_dict().items():torch.testing.assert_close(v,ref.function.net.state_dict()[k],rtol=0,atol=0)
    for left,right in zip(m.input_stats,ref.input_stats):np.testing.assert_array_equal(left,right)
    np.testing.assert_allclose(m.readout,ridge_readout(m.g,m.k,c['ridge']),rtol=1e-10,atol=1e-10)
    if c['method']=='local':
        assert m.buffer is None and sum(cell.n for cells in m.local.cells.values() for cell in cells)==len(y)
        assert min(cell.n for cells in m.local.cells.values() for cell in cells)>=4
        other=folder/path.parent.name.replace('_local_','_er_')
        with np.load(path.parent/'predictions_stage1.npz') as x,np.load(other/'predictions_stage1.npz') as z:
            for field in x.files:np.testing.assert_array_equal(x[field],z[field])
        other_summary=json.loads((other/'summary.json').read_text())
        for x,z in zip(s['stages'],other_summary['stages']):
            assert x['optimizer_steps']==z['optimizer_steps'] and x['historical_training_samples']==z['historical_training_samples']
        pairs+=1
    else:
        assert sum(v.nbytes for v in m.buffer)+m.keys.nbytes<=4*1024**2
        for x,z in zip(m.buffer,ref.buffer):np.testing.assert_array_equal(x,z)
    checked.append(path.parent.name)
assert len(checked)==108 and len(cache)==18 and pairs==54
a.out.write_text(json.dumps(dict(complete=True,trajectories=108,reconstructed_streams=18,matched_pairs=54,checks=['first-stage exact paired predictions','equal update and draw budgets','stream reconstruction reproduces all ridge functions and risk moments','independent final representative weights reconstruct risk','local counts/minimum cells/no buffer; ER payload budget','analytic readouts and fixed input normalization'],checked=checked),indent=2))
print('108 trajectories, 18 rebuilt streams and 54 paired controls verified')
