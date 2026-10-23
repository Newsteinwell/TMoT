"""Measure finite replay XY errors despite analytically protected distribution."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('out',type=Path);a=p.parse_args()
rows=[]
for d,e in [('energy','mantis'),('beijing_pm25','summary'),('air_quality','summary')]:
    for components in (1,4):
        name=f'{d}_{e}_local_k{components}_w1_o17_s42'
        model=torch.load(a.root/'runs'/'target_development_v15'/name/'final.pt',map_location='cpu',weights_only=False)['learner']
        g,k,_=model.memory.risk_moments();reference=k[:-1]-np.outer(g[:-1,-1],k[-1])
        errors=[]
        for seed in range(5):
            x,y=model.local.sample(4096,[],np.random.default_rng(1000+seed));x=x.astype(float);y=y.astype(float)
            covariance=(x-x.mean(0)).T@(y-y.mean(0))/len(x)
            errors.append(float(np.linalg.norm(covariance-reference)/max(np.linalg.norm(reference),1e-12)))
        rows.append(dict(dataset=d,components=components,draws=5,replay_count=4096,cross_relative_errors=errors,mean=float(np.mean(errors))))
a.out.write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
