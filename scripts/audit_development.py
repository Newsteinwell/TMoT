"""Metric, input and expected-name audit for third-round development queues."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('queue');p.add_argument('out',type=Path);a=p.parse_args()
folder=a.root/'runs'/a.queue;manifest=json.loads((folder/'manifest.json').read_text());jobs=manifest['jobs']
pruned={};pruned_checked=[]
for filename in ('expert_checkpoint_pruning.json','projection_checkpoint_pruning.json','feature_checkpoint_pruning.json','function_checkpoint_pruning.json','replay_checkpoint_pruning.json','orthogonal_checkpoint_pruning.json','regularized_checkpoint_pruning.json','history_scale_checkpoint_pruning.json','fecam_checkpoint_pruning.json','historical_checkpoint_pruning.json'):
    retention=a.root/filename
    if retention.exists():
        record=json.loads(retention.read_text())
        if record['applied']:pruned.update({r['path']:r for r in record['removed_candidates']})
if a.queue in ('stable_development_v7','causal_development_v19_retry'):names={f'{d}_{v}_o17_s{seed}' for d,v,m,r,seed,args in jobs}
elif a.queue=='kernel_development_v8':names={f'{d}_{v}_r{r:g}_o17_s42' for d,v,r in jobs}
elif a.queue in ('round3_validation_v9','nonlinear_development_v10','nonlinear_matched_v11','nonlinear_validation_v12','fecam_validation_v13','local_development_v14','target_development_v15','calibrated_development_v16','expert_development_v17','prior_expert_development_v18','classification_robustness_v20','order_baselines_v21','projection_diagnostic_v22','projection_seed_v23','evidence_development_v24','evidence_refined_v25','task_loss_v26','temporal_joint_v27','raw_temporal_v28','raw_transport_v29','neighbors_v30','warm_transport_v31','local_warm_v32','local_support_v33','local_merge_v34','functional_v36','functional_history_v37','replay_functional_v38','exact_base_v39','orthogonal_v40','regularized_innovation_v41','history_scale_v42','history_robustness_v43','arrived_calibration_v44','harder_validation_v45','full_feature_v46'):names={job['name'] for job in jobs}
else:raise ValueError(a.queue)
hashes={};checked=[]
for path in sorted(folder.glob('*/summary.json')):
    s=json.loads(path.read_text());cfg=s['config'];assert cfg['eval_split']=='val'
    checkpoint=path.parent/'final.pt'
    if not checkpoint.exists():
        record=pruned[str(checkpoint)]
        assert record['config']==cfg and record['source']==s['git_commit']
        pruned_checked.append(str(checkpoint))
    for split,digest in s['input_hashes'].items():
        source=a.root/'features'/cfg['encoder']/f"{cfg['dataset']}_{split}.npz"
        if str(source) not in hashes:hashes[str(source)]=hashlib.sha256(source.read_bytes()).hexdigest()
        assert hashes[str(source)]==digest
    for stage in s['stages']:
        with np.load(path.parent/f"predictions_stage{stage['stage']}.npz") as d:
            keys=sorted(int(k.split('_')[1]) for k in d.files if k.startswith('y_'))
            y=np.concatenate([d[f'y_{k}'] for k in keys]);pred=np.concatenate([d[f'pred_{k}'] for k in keys])
        metric=(np.mean([np.mean(y[y==c]==pred[y==c]) for c in np.unique(y)]) if s['task']=='classification' else np.sqrt(np.mean((y-pred)**2)))
        np.testing.assert_allclose(metric,stage['macro_accuracy' if s['task']=='classification' else 'rmse'],rtol=1e-6)
    checked.append(path.parent.name)
statuses=[json.loads(line) for line in (folder/'status.jsonl').read_text().splitlines()]
failures=[r for r in statuses if r['status'] in ('failed','timeout')]
missing=sorted(names-set(checked));extra=sorted(set(checked)-names)
report=dict(queue=a.queue,expected=len(names),completed_metric_checked=len(checked),missing=missing,unexpected=extra,failures=failures,complete=not missing and not extra and not failures,checked=checked,input_hashes=hashes,pruned_checkpoint_count=len(pruned_checked),pruned_checkpoints=pruned_checked,checkpoint_note='Metrics and hashes verified; recorded pruned model states require regeneration and were not reloaded' if pruned_checked else 'All checkpoint paths present; this audit checks metrics and hashes, not model state')
a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ('checked','input_hashes','pruned_checkpoints')}))
