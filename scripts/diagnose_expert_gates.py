"""VAL-only diagnostic: known-domain routing, not deployable task-free scores."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('out',type=Path);a=p.parse_args()
records=[]
for folder in sorted((a.root/'runs'/'expert_development_v17').iterdir()):
    if not (folder/'summary.json').exists():continue
    summary=json.loads((folder/'summary.json').read_text())
    if summary['task']!='regression':continue
    ckpt=torch.load(folder/'final.pt',map_location='cpu',weights_only=False);model=ckpt['learner']
    with np.load(a.root/'features'/summary['config']['encoder']/f"{summary['config']['dataset']}_val.npz") as data:
        x=ckpt['basis'].transform(data['x']);y=data['y'];groups=data['groups']
    if y.ndim==1:y=y[:,None]
    weights=model.probabilities(x);expert_preds=np.stack([e.conditional_mean(x) for e in model.experts],axis=1)
    learned=np.einsum('nk,nkp->np',weights,expert_preds)
    oracle=np.zeros_like(learned);correct_mass=[];offset=0
    for stage,group in zip(summary['stages'],summary['order'],strict=True):
        count=0;end=offset
        while count<stage['current_samples']:
            count+=model.local.cells[None][end].n;end+=1
        assert count==stage['current_samples']
        ids=np.flatnonzero(groups==group)
        local_weights=weights[ids,offset:end]
        mass=local_weights.sum(1);correct_mass.extend(mass.tolist())
        # Recompute log weights to avoid underflow when the correct-domain mass is tiny.
        from scipy.special import logsumexp
        logs=np.column_stack([model.experts[j].log_density(x[ids],model.df)+model.log_priors[j] for j in range(offset,end)])
        normalized=np.exp(logs-logsumexp(logs,axis=1,keepdims=True))
        oracle[ids]=np.einsum('nk,nkp->np',normalized,expert_preds[ids,offset:end])
        offset=end
    assert offset==len(model.experts)
    learned=learned*ckpt['target_sd']+ckpt['target_mean'];oracle=oracle*ckpt['target_sd']+ckpt['target_mean']
    observed=float(np.sqrt(np.mean((learned-y)**2)))
    np.testing.assert_allclose(observed,summary['final_rmse'],rtol=1e-6)
    records.append(dict(run=folder.name,dataset=summary['config']['dataset'],actual_rmse=observed,
                        known_domain_rmse=float(np.sqrt(np.mean((oracle-y)**2))),mean_correct_domain_mass=float(np.mean(correct_mass))))
assert len(records)==72
result=dict(diagnostic_only=True,uses_validation_domain_id=True,records=records)
a.out.write_text(json.dumps(result,indent=2))
for dataset in ('energy','beijing_pm25','air_quality'):
    rows=[r for r in records if r['dataset']==dataset]
    best=min(rows,key=lambda r:r['actual_rmse'])
    print(json.dumps(dict(best_task_free=best,best_known_domain=min(rows,key=lambda r:r['known_domain_rmse']))))
