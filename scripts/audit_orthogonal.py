"""Independent final-state/readout reconstruction and all-stage control audit."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from tmot.moments import ridge_readout
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('out',type=Path);a=p.parse_args()
torch.set_num_threads(1);checked=[];controls=0;cache={};max_orth=0.;max_psd=0.
for path in sorted((a.root/'runs/orthogonal_v40').glob('*/summary.json')):
    s=json.loads(path.read_text());c=s['config'];ck=torch.load(path.parent/'final.pt',map_location='cpu',weights_only=False);m=ck['learner']
    key=c['dataset'],c['encoder'],c['method']
    olddir=a.root/'runs/exact_base_v39'/f"{c['dataset']}_{c['encoder']}_{c['method']}_exact1_r0.1_o17_s42"
    if key not in cache:cache[key]=(torch.load(olddir/'final.pt',map_location='cpu',weights_only=False)['learner'],json.loads((olddir/'summary.json').read_text()))
    old,prior=cache[key]
    assert s['input_hashes']==prior['input_hashes'] and s['order']==prior['order']
    for field in ('g','k'):np.testing.assert_array_equal(getattr(m,field),getattr(old,field))
    for name,v in m.function.net.state_dict().items():torch.testing.assert_close(v,old.function.net.state_dict()[name],rtol=0,atol=0)
    for field in ('mean','sd','hidden_mean','hidden_scale'):np.testing.assert_array_equal(getattr(m.function,field),getattr(old.function,field))
    for row,ref in zip(s['stages'],prior['stages']):
        for field in ('optimizer_steps','historical_training_samples','representative_mass','completion'):assert row[field]==ref[field]
        if c['orthogonal']:
            max_orth=max(max_orth,row['innovation']['orthogonality_error']);max_psd=max(max_psd,row['innovation']['residual_psd_adjustment'])
    if c['orthogonal']:
        phi=np.r_[np.arange(m.dimension),len(m.g)-1];hi=np.arange(m.dimension,len(m.g)-1)
        aa=m.g[np.ix_(phi,phi)];u=m.g[np.ix_(phi,hi)]
        values,vectors=np.linalg.eigh((aa+aa.T)/2);cut=max(np.max(abs(values))*1e-12,1e-14);keep=values>cut
        b=(vectors[:,keep]/values[keep])@(vectors[:,keep].T@u)
        transform=np.zeros_like(m.g);transform[phi,:len(phi)]=np.eye(len(phi));transform[hi,len(phi):]=np.eye(len(hi));transform[phi,len(phi):]=-b
        residual=(transform.T@m.g@transform)[len(phi):,len(phi):]
        val,vec=np.linalg.eigh((residual+residual.T)/2);cut=max(np.max(abs(val))*1e-12,1e-14)
        residual=(vec*np.where(val>cut,val,0))@vec.T
        gram=np.zeros_like(m.g);gram[:len(phi),:len(phi)]=aa;gram[len(phi):,len(phi):]=residual
        penalty=np.eye(len(m.g))*c['ridge'];penalty[len(phi)-1,len(phi)-1]=0
        expected=transform@np.linalg.solve((gram+gram.T)/2+penalty,transform.T@m.k)
        np.testing.assert_allclose(m.readout,expected,rtol=1e-6,atol=2e-7)
    else:
        controls+=1;np.testing.assert_array_equal(m.readout,old.readout)
        for stage in range(1,len(s['stages'])+1):
            with np.load(path.parent/f'predictions_stage{stage}.npz') as x,np.load(olddir/f'predictions_stage{stage}.npz') as y:
                for field in x.files:np.testing.assert_array_equal(x[field],y[field])
    checked.append(path.parent.name)
assert len(checked)==126 and controls==18
a.out.write_text(json.dumps(dict(complete=True,trajectories=len(checked),exact_controls=controls,max_orthogonality_error=max_orth,max_residual_psd_adjustment=max_psd,checks=['v39 moments/functions unchanged','paired stage training budgets and completed moments unchanged','independent transformed-coordinate dense solve','18 all-stage exact prediction controls'],checked=checked),indent=2))
print(f'{len(checked)} state audits, {controls} exact controls; orthogonality {max_orth:.3g}, PSD adjustment {max_psd:.3g}')
