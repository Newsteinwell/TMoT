"""Validation-only ridge sweep from fixed final training checkpoints.

No encoder/adapter retraining and no test reads. Select lambda by mean validation
RMSE across seeds; rerun the locked choice for stagewise/test evaluation.
"""
import argparse,json,copy
from pathlib import Path
import numpy as np
import torch
from tmot.moments import affine_matrix,transport_risk,ridge_readout
p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();torch.set_num_threads(2)
val=np.load(a.root/'features/mantis/energy_val.npz');train=np.load(a.root/'features/mantis/energy_train.npz')
rows=[]
for variant in ('full','no_hist','frozen'):
    for seed in (42,43):
        source=a.root/'runs/transport_v3'/f'energy_{variant}_o17_s{seed}'/'final.pt'
        if not source.exists():continue
        cp=torch.load(source,weights_only=False,map_location='cpu');b=cp['basis'];l=cp['learner']
        b.device=l.device='cpu';phi=b.transform(val['x']);g,k,_=l.memory.risk_moments(l.classes)
        matrix=affine_matrix(l.adapter.weight.detach().numpy(),l.adapter.bias.detach().numpy());g,k=transport_risk(g,k,matrix)
        for ridge in (.001,.01,.1,1.,10.,100.):
            l.linear_readout=ridge_readout(g,k,ridge);pred=l.predict(phi)*cp['target_sd']+cp['target_mean']
            rows.append(dict(variant=variant,seed=seed,ridge=ridge,rmse=float(np.sqrt(np.mean((pred-val['y'])**2)))))
mean=train['y'].mean(0);reference=float(np.sqrt(np.mean((val['y']-mean)**2)))
report=dict(evaluation='val',criterion='final validation RMSE',rows=rows,global_training_mean_rmse=reference)
out=a.root/'reports/regression_ridge_probe.json';out.write_text(json.dumps(report,indent=2));print(json.dumps(report))
