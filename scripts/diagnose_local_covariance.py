"""Compare exact stored regression cross moments with implied replay moments."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('out',type=Path);a=p.parse_args()
records=[]
for dataset,encoder in [('energy','mantis'),('beijing_pm25','summary'),('air_quality','summary')]:
    for components in (1,4):
        name=f'{dataset}_{encoder}_local_k{components}_w1_o17_s42'
        path=a.root/'runs'/'local_development_v14'/name/'final.pt'
        model=torch.load(path,map_location='cpu',weights_only=False)['learner']
        cells=model.local.cells[None];d=model.dimension;n=sum(c.n for c in cells)
        mean=sum(c.n*c.mean.astype(float) for c in cells)/n
        raw=sum(c.n*(np.outer(c.mean,c.mean).astype(float)+c.factor.astype(float)@c.factor.astype(float).T+np.diag(c.residual_sd.astype(float)**2)) for c in cells)/n
        implied=raw-np.outer(mean,mean)
        g,k,_=model.memory.risk_moments();mx=g[:-1,-1];my=k[-1]
        cross=k[:-1]-np.outer(mx,my)
        variance_x=np.diag(g)[:-1]-mx**2
        variance_y=np.diag(model.memory.target_gram/model.memory.n)-my**2
        records.append(dict(dataset=dataset,components=components,effective_cells=len(cells),samples=n,
            cross_cov_relative_error=float(np.linalg.norm(implied[:d,d:]-cross)/max(np.linalg.norm(cross),1e-12)),
            input_variance_relative_error=float(np.linalg.norm(np.diag(implied)[:d]-variance_x)/max(np.linalg.norm(variance_x),1e-12)),
            target_variance_relative_error=float(np.linalg.norm(np.diag(implied)[d:]-variance_y)/max(np.linalg.norm(variance_y),1e-12))))
a.out.write_text(json.dumps(records,indent=2));print(json.dumps(records,indent=2))
