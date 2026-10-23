import argparse,json
from pathlib import Path
import numpy as np
import torch
from tmot.moments import ClassificationMoments,RegressionMoments,ridge_readout
from tmot.exact_base_completion import complete_joint
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('out',type=Path);a=p.parse_args()
torch.set_num_threads(1);cache={};checked=[];controls=0;minimum=0.
folder=a.root/'runs/exact_base_v39'
for path in sorted(folder.glob('*/summary.json')):
    s=json.loads(path.read_text());c=s['config'];ck=torch.load(path.parent/'final.pt',map_location='cpu',weights_only=False);m=ck['learner'];key=c['dataset'],c['encoder'],c['method']
    olddir=a.root/'runs/replay_functional_v38'/f"{c['dataset']}_{c['encoder']}_{c['method']}_r0.1_o17_s42"
    old_summary=json.loads((olddir/'summary.json').read_text())
    assert s['input_hashes']==old_summary['input_hashes'] and s['order']==old_summary['order']
    if key not in cache:
        oldck=torch.load(olddir/'final.pt',map_location='cpu',weights_only=False);old=oldck['learner']
        with np.load(a.root/'features'/c['encoder']/f"{c['dataset']}_train.npz") as data:
            x=data['x'];y=data['y'].copy()
            parts=[np.flatnonzero(np.isin(y,cl)) for cl in s['order']] if s['task']=='classification' else [np.flatnonzero(data['groups']==g) for g in s['order']]
            if s['task']=='regression':y=(y-ck['target_mean'])/ck['target_sd']
            z=np.concatenate([oldck['basis'].transform(x[ids]) for ids in parts]);target=np.concatenate([y[ids] for ids in parts])
            direct=ClassificationMoments(512) if s['task']=='classification' else RegressionMoments(512,y.shape[1]);direct.update(z,target)
        cache[key]=(old,direct)
    old,direct=cache[key]
    for k,v in m.function.net.state_dict().items():torch.testing.assert_close(v,old.function.net.state_dict()[k],rtol=0,atol=0)
    for field in ('mean','sd','hidden_mean','hidden_scale'):np.testing.assert_array_equal(getattr(m.function,field),getattr(old.function,field))
    for row,prior in zip(s['stages'],old_summary['stages']):
        for field in ('optimizer_steps','historical_training_samples'):assert row[field]==prior[field]
        assert row['representative_mass']==prior['representative_mass']
    eg,ek,ev=m.exact_memory.risk_moments(m.classes);dg,dk,dv=direct.risk_moments(m.classes)
    for x,y in ((eg,dg),(ek,dk),(ev,dv)):np.testing.assert_allclose(x,y,rtol=1e-9,atol=1e-8)
    v=np.eye(len(m.classes))/len(m.classes) if s['task']=='classification' else m.exact_memory.target_gram/m.exact_memory.n
    if c['completion']:
        g,k,_=complete_joint(old.g,old.k,eg,ek,v,512)
        np.testing.assert_array_equal(m.g,g);np.testing.assert_array_equal(m.k,k)
        phi=np.r_[np.arange(512),len(m.g)-1]
        np.testing.assert_array_equal(m.g[np.ix_(phi,phi)],eg);np.testing.assert_array_equal(m.k[phi],ek)
        joint=np.block([[m.g,m.k],[m.k.T,v]]);eigen=np.linalg.eigvalsh((joint+joint.T)/2)
        ratio=float(eigen.min()/max(1.,eigen.max()));minimum=min(minimum,ratio);assert ratio>=-1e-7
        for row in s['stages']:
            d=row['completion'];assert d['input_block_error']==0 and d['target_cross_error']==0
            assert d['joint_min_eigenvalue']>=-1e-7*max(1,d['joint_max_eigenvalue'])
    else:
        controls+=1
        for stage in range(1,len(s['stages'])+1):
            with np.load(path.parent/f'predictions_stage{stage}.npz') as x,np.load(olddir/f'predictions_stage{stage}.npz') as y:
                for field in x.files:np.testing.assert_array_equal(x[field],y[field])
    np.testing.assert_allclose(m.readout,ridge_readout(m.g,m.k,c['ridge']),rtol=1e-10,atol=1e-10)
    checked.append(path.parent.name)
assert len(checked)==126 and controls==18
a.out.write_text(json.dumps(dict(complete=True,trajectories=126,exact_controls=18,min_relative_joint_eigenvalue=minimum,checks=['original arrived-data moments independently reconstructed','network functions and update budgets identical to v38','exact input/target blocks retained at all recorded stages','conditional completion reconstructed from v38 representative moments','joint PSD within declared tolerance','readout and all disabled control predictions'],checked=checked),indent=2))
print(f'126 audits and 18 exact controls passed; minimum relative eigenvalue {minimum:.3g}')
