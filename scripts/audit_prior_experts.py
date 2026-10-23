"""Pair v18 with v17 gates and independently executed cumulative-ridge controls."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('out',type=Path);a=p.parse_args()
references={'har':('stable_development_v7','har_ridge_r1_o17_s42'),
'uwave':('stable_development_v7','uwave_ridge_r0.1_o17_s42'),
'dailysports':('stable_development_v7','dailysports_ridge_r0.1_o17_s42'),
'energy':('stable_development_v7','energy_ridge_r1_o17_s42'),
'beijing_pm25':('round3_validation_v9','beijing_pm25_summary_linear_r0.1_o17_s42'),
'air_quality':('round3_validation_v9','air_quality_summary_linear_r1_o17_s42')}
cache={};checked=[]
for path in sorted((a.root/'runs'/'prior_expert_development_v18').glob('*/summary.json')):
    summary=json.loads(path.read_text());dataset=summary['config']['dataset']
    old=a.root/'runs'/'expert_development_v17'/path.parent.name
    reference=json.loads((old/'summary.json').read_text())
    assert summary['input_hashes']==reference['input_hashes'] and summary['order']==reference['order']
    new=torch.load(path.parent/'final.pt',map_location='cpu',weights_only=False)
    previous=torch.load(old/'final.pt',map_location='cpu',weights_only=False)
    if dataset not in cache:
        queue,name=references[dataset];cache[dataset]=torch.load(a.root/'runs'/queue/name/'final.pt',map_location='cpu',weights_only=False)
    control=cache[dataset]
    for key,value in vars(new['basis']).items():
        np.testing.assert_equal(value,vars(previous['basis'])[key]);np.testing.assert_equal(value,vars(control['basis'])[key])
    model=new['learner'];np.testing.assert_array_equal(model.global_readout,control['learner'].readout)
    np.testing.assert_array_equal(model.log_priors,previous['learner'].log_priors)
    assert model.labels==previous['learner'].labels
    for e,r in zip(model.experts,previous['learner'].experts,strict=True):
        for key in ('mean','invdiag','weighted','middle_inv','logdet'):np.testing.assert_equal(getattr(e,key),getattr(r,key))
    for stage in summary['stages']:
        filename=f"predictions_stage{stage['stage']}.npz"
        with np.load(path.parent/filename) as new_preds,np.load(old/filename) as old_preds:
            for key in old_preds.files:
                if not key.startswith('pred_'):np.testing.assert_array_equal(new_preds[key],old_preds[key])
    checked.append(path.parent.name)
assert len(checked)==144
a.out.write_text(json.dumps(dict(complete=True,paired=len(checked),checks=['identical local input densities and gate priors','identical frozen basis','shared ridge equals independently executed baseline','identical previously audited validation IDs and labels'],runs=checked),indent=2));print(json.dumps(dict(complete=True,paired=len(checked))))
