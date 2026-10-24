"""Check normal-equation residuals, saved predictions and existing 512-D controls."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('out',type=Path);a=p.parse_args();torch.set_num_threads(1)
folder=a.root/'runs/full_feature_v46';terminal=json.loads((folder/'complete.json').read_text());assert len(terminal)==144 and all(r['status']=='completed' for r in terminal)
checked=[];controls=[];cache={};maximum=0.;max_control_error=0.
for path in sorted(folder.glob('*/summary.json')):
    s=json.loads(path.read_text());c=s['config'];ck=torch.load(path.parent/'final.pt',map_location='cpu',weights_only=False);b=ck['basis'];key=c['dataset'],c['encoder'],c['mode'];cl=s['task']=='classification'
    def transform(x):
        if c['mode']=='projected':return b.transform(x)
        d=b.semantic_dim
        z1=((x[:,:d]-b.mean)/b.scale)*np.sqrt((512-b.amp_width)/d)
        z2=((x[:,d:]-b.amp_mean)/b.amp_sd)*np.sqrt(b.amp_width/b.amplitude_dim)
        return np.c_[z1,z2].astype(np.float32)
    if key not in cache:
        data=np.load(a.root/'features'/c['encoder']/f"{c['dataset']}_train.npz");val=np.load(a.root/'features'/c['encoder']/f"{c['dataset']}_val.npz")
        parts=[np.flatnonzero(np.isin(data['y'],labels)) for labels in s['order']] if cl else [np.flatnonzero(data['groups']==g) for g in s['order']]
        z=np.concatenate([transform(data['x'][inds]) for inds in parts]).astype(float);y=np.concatenate([data['y'][inds] for inds in parts]);count=0
        for record,inds in zip(s['stages'],parts):count+=len(inds);assert record['arrived_samples']==count
        cache[key]=(z,y,val)
    z,y,val=cache[key]
    if cl:
        expected=[int(label) for labels in s['order'] for label in sorted(labels)];assert ck['classes']==expected
        target=np.eye(len(expected))[[expected.index(int(label)) for label in y]];w=np.array([1/(len(expected)*np.sum(y==label)) for label in y])
    else:target=(y-ck['target_mean'])/ck['target_sd'];w=np.full(len(y),1/len(y))
    coef=ck['coef'];bias=ck['bias'];res=z@coef+bias-target
    gradient=z.T@(w[:,None]*res)+c['ridge']*coef;bias_gradient=w@res
    relative=max(float(np.max(abs(gradient))),float(np.max(abs(bias_gradient))))/max(1.,float(np.max(abs(z.T@(w[:,None]*target)))))
    assert relative<1e-7,(path,relative);maximum=max(maximum,relative)
    with np.load(path.parent/f"predictions_stage{len(s['order'])}.npz") as saved:
        for k,group in enumerate(s['order']):
            inds=np.flatnonzero(np.isin(val['y'],group)) if cl else np.flatnonzero(val['groups']==group)
            score=transform(val['x'][inds]).astype(float)@coef+bias
            pred=np.asarray(ck['classes'])[score.argmax(1)] if cl else score*ck['target_sd']+ck['target_mean']
            np.testing.assert_allclose(saved[f'pred_{k}'],pred,rtol=1e-10,atol=1e-10)
            np.testing.assert_array_equal(saved[f'ids_{k}'],val['ids'][inds]);np.testing.assert_array_equal(saved[f'y_{k}'],val['y'][inds])
    old=None;r=c['ridge'];d=c['dataset'];e=c['encoder']
    if c['mode']=='projected':
        if d in ('phonemespectra','lsst','handwriting'):old=a.root/'runs/harder_validation_v45'/f'{d}_{e}_linear_r{r:g}_o17_s42'
        elif d in ('beijing_pm25','air_quality'):old=a.root/'runs/round3_validation_v9'/f'{d}_{e}_linear_r{r:g}_o17_s42'
        elif d=='energy' and e=='mantis':old=a.root/'runs/stable_development_v7'/f'energy_ridge_r{r:g}_o17_s42'
    if old:
        prior=json.loads((old/'summary.json').read_text());assert prior['input_hashes']==s['input_hashes'] and prior['order']==s['order']
        for stage in range(1,len(s['order'])+1):
            with np.load(path.parent/f'predictions_stage{stage}.npz') as now,np.load(old/f'predictions_stage{stage}.npz') as past:
                assert set(now.files)==set(past.files)
                for name in now.files:
                    if cl or not name.startswith('pred_'):np.testing.assert_array_equal(now[name],past[name])
                    else:
                        np.testing.assert_allclose(now[name],past[name],rtol=1e-7,atol=1e-7);max_control_error=max(max_control_error,float(np.max(abs(now[name]-past[name]))))
        controls.append(path.parent.name)
    checked.append(path.parent.name)
assert len(checked)==144 and len(controls)==66
report=dict(complete=True,checked=checked,controls=controls,max_relative_normal_residual=maximum,max_regression_control_prediction_error=max_control_error)
a.out.write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ('checked','controls')}))
