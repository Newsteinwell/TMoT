"""Audit target-protected replay against the same v14 configurations."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('out',type=Path);a=p.parse_args()
reports=[]
for path in sorted((a.root/'runs'/'calibrated_development_v16').glob('*/summary.json')):
    old=a.root/'runs'/'target_development_v15'/path.parent.name
    summary=json.loads(path.read_text());reference=json.loads((old/'summary.json').read_text())
    assert summary['input_hashes']==reference['input_hashes']
    assert summary['order']==reference['order']
    history_count=0
    for new_stage,old_stage in zip(summary['stages'],reference['stages'],strict=True):
        calibration=new_stage['calibration']
        if history_count:
            assert calibration['history_samples']==history_count
            assert calibration['after_cross_error']<1e-5
            assert calibration['after_target_cov_error']<1e-5
            assert calibration['after_input_variance_error']<1e-5
        else:assert calibration is None
        history_count+=new_stage['current_samples']
        assert new_stage['optimizer_steps']==old_stage['optimizer_steps']
        assert new_stage['historical_training_samples']==old_stage['historical_training_samples']
        with np.load(path.parent/f"predictions_stage{new_stage['stage']}.npz") as new, np.load(old/f"predictions_stage{old_stage['stage']}.npz") as ref:
            assert set(new.files)==set(ref.files)
            for key in new.files:
                if not key.startswith('pred_'):np.testing.assert_array_equal(new[key],ref[key])
    checkpoint=torch.load(path.parent/'final.pt',map_location='cpu',weights_only=False)
    previous=torch.load(old/'final.pt',map_location='cpu',weights_only=False)
    model=checkpoint['learner'];assert model.local.preserve_targets
    np.testing.assert_array_equal(model.base,previous['learner'].base)
    for key,value in vars(checkpoint['basis']).items():np.testing.assert_equal(value,vars(previous['basis'])[key])
    cells=model.local.cells[None];d=model.dimension;n=sum(c.n for c in cells)
    assert n==model.memory.n
    mean=sum(c.n*c.mean.astype(float) for c in cells)/n
    raw=sum(c.n*(np.outer(c.mean.astype(float),c.mean.astype(float))+c.factor.astype(float)@c.factor.astype(float).T+np.diag(c.residual_sd.astype(float)**2)) for c in cells)/n
    implied=raw-np.outer(mean,mean)
    g,k,_=model.memory.risk_moments();mx=g[:-1,-1];my=k[-1]
    cross=k[:-1]-np.outer(mx,my);yy=model.memory.target_gram/n-np.outer(my,my)
    cross_error=np.linalg.norm(implied[:d,d:]-cross)/max(np.linalg.norm(cross),1e-12)
    yy_error=np.linalg.norm(implied[d:,d:]-yy)/max(np.linalg.norm(yy),1e-12)
    assert cross_error<1e-5 and yy_error<1e-5
    reports.append(dict(run=path.parent.name,cross_relative_error=float(cross_error),target_covariance_relative_error=float(yy_error)))
assert len(reports)==36
a.out.write_text(json.dumps(dict(complete=True,paired=36,checks=['identical analytic shortcut and basis','equal optimizer/replay counts','identical validated sample IDs and labels','protected empirical XY and YY covariance within 1e-5 relative tolerance','finite replay XY/YY/diagonal XX errors below 1e-5 at every historical stage','calibration uses only preceding historical sample count'],runs=reports),indent=2))
print(json.dumps(dict(complete=True,paired=36,max_cross_error=max(r['cross_relative_error'] for r in reports),max_target_covariance_error=max(r['target_covariance_relative_error'] for r in reports))))
